"""One self-contained Google Colab cell for the repaired Generation-0 run.

Paste this entire file into one Colab cell. By default it only prints preparation
instructions. Set EXECUTE=True and pin ICRL_G0_HARNESS_COMMIT to the full commit
containing the C15 implementation for a later manual run. Execution mounts the frozen Drive
corpus, fetches the public repository anonymously (or securely falls back
to a short-lived askpass helper when authentication is needed), creates fresh
planning/candidate worktrees at the pinned harness commit, and delegates all
execution to the fail-closed supervisor.  It never deletes an existing
campaign or silently retries a failed slot.
"""

# This is intentionally a plain Python cell: it has no dependency on variables
# or imports from an earlier notebook cell.
from __future__ import annotations

from datetime import datetime, timezone
from getpass import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import stat
import subprocess
import sys
import tempfile


REPOSITORY = "https://github.com/arahe-dev/iclr-autoresearch.git"
HARNESS_BRANCH = "codex/generation0/sm120-runner-20260909"
HARNESS_COMMIT = os.environ.get("ICRL_G0_HARNESS_COMMIT", "")
EXECUTE = False
CHAMPION_BRANCH = "codex/champion/0000-validated-bdh-baseline"
CHAMPION_COMMIT = "908b0b1438ba038d319adf97787aac08f213b590"
PLAN_ID = "G0-L8-SM120-EXACT-B32-C15"

DRIVE_ROOT = Path("/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh")
CORPUS_ROOT = DRIVE_ROOT / "corpus/stage2/frozen_5b_v1"
REPO_STORE = Path(f"/content/iclr-g0-repository-{HARNESS_COMMIT[:12]}")
PLANNING_ROOT = Path(f"/content/iclr-g0-planning-{HARNESS_COMMIT[:12]}")
CANDIDATE_ROOT = Path(f"/content/iclr-g0-candidate-{HARNESS_COMMIT[:12]}")
CANDIDATE_BRANCH = f"codex/generation0/colab-{HARNESS_COMMIT[:12]}"
CAMPAIGN_ROOT = DRIVE_ROOT / "experiments/generation0" / f"iclr-g0-c15-{HARNESS_COMMIT[:12]}"

# Keep this false for normal resumes. Set it to true only after inspecting a
# deliberate treatment repair; the supervisor still retries only the failed
# ordered slot and never skips it.
RETRY_FAILED = False
GPU_UUID: str | None = None
GPU_INDEX: int | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {command[0]} {detail}")
    return completed.stdout.strip()


def git(cwd: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    return run(["git", "-C", str(cwd), *arguments], env=env)


def ref_contains_commit(cwd: Path, ref: str, expected_commit: str) -> bool:
    """Accept a branch that contains the exact pin, even if it has later cell commits."""
    try:
        actual = git(cwd, "rev-parse", ref)
        if actual == expected_commit:
            return True
        git(cwd, "merge-base", "--is-ancestor", expected_commit, actual)
        return True
    except RuntimeError:
        return False


def assert_clean_checkout(path: Path, expected_commit: str) -> None:
    if not path.is_dir() or not (path / ".git").exists():
        raise RuntimeError(f"expected existing Git checkout is missing: {path}")
    top = Path(git(path, "rev-parse", "--show-toplevel")).resolve()
    if top != path.resolve():
        raise RuntimeError(f"checkout path is not the expected worktree: {path}")
    if git(path, "status", "--porcelain"):
        raise RuntimeError(f"refusing to reuse a dirty checkout: {path}")
    actual = git(path, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise RuntimeError(f"checkout {path} is at {actual}, expected pinned {expected_commit}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_canonical_checkout(path: Path, contract: dict[str, object]) -> None:
    canonical = contract.get("canonical_source")
    if not isinstance(canonical, dict):
        raise RuntimeError("pinned checkout has no canonical-source contract")
    relative = canonical.get("path")
    expected = canonical.get("sha256")
    source = path / str(relative)
    if not source.is_file() or sha256_file(source) != expected:
        raise RuntimeError(f"canonical source hash mismatch in checkout: {source}")


def assert_frozen_corpus(planning: Path) -> None:
    contract = json.loads((planning / "configs/frozen_contract.json").read_text(encoding="utf-8"))
    frozen_path = CORPUS_ROOT / "FROZEN.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    data = contract.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("pinned frozen contract has no data section")
    expected = {
        "status": "FROZEN",
        "corpus_id": data["corpus_id"],
        "context_length": data["context_length"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "logical_replay_sha256": data["logical_replay_sha256"],
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen corpus contract mismatch: {key}")
    required = (
        CORPUS_ROOT / "train" / "shard_000000.tokens.bin",
        CORPUS_ROOT / "train" / "shard_000000.valid_lengths.bin",
        CORPUS_ROOT / "train" / "shard_000000.provenance.parquet",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"frozen corpus artifacts are missing: {missing}")
    print("Frozen corpus contract passed.", flush=True)


def _is_authentication_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in (
        "could not read username", "authentication failed", "repository not found",
        "terminal prompts disabled", "http 401", "http 403", "403 forbidden",
    ))


def assert_full_sha(value: str, label: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise RuntimeError(f"{label} must be a full 40-character hexadecimal Git SHA")


def credentials_if_needed() -> tuple[dict[str, str], Path | None]:
    """Create a short-lived askpass helper; the PAT never enters argv/output."""
    username = os.environ.get("GITHUB_USER") or input("GitHub username: ").strip()
    token = os.environ.get("GITHUB_TOKEN") or getpass("GitHub PAT (input hidden): ")
    if not username or not token:
        raise RuntimeError("a GitHub username and PAT are required for the private repository")
    helper_fd, helper_name = tempfile.mkstemp(prefix="iclr-git-askpass-", suffix=".sh")
    os.close(helper_fd)
    helper = Path(helper_name)
    try:
        helper.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) printf '%s\\n' \"$GITHUB_USER\" ;;\n"
            "  *) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        helper.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except BaseException:
        if helper.exists():
            helper.unlink()
        raise
    auth_env = os.environ.copy()
    auth_env.update({
        "GIT_ASKPASS": str(helper),
        "GIT_TERMINAL_PROMPT": "0",
        "GITHUB_USER": username,
        "GITHUB_TOKEN": token,
    })
    return auth_env, helper


def clone_repository() -> None:
    """Clone anonymously when possible, with an ephemeral authenticated fallback."""
    if REPO_STORE.exists():
        raise RuntimeError(f"refusing to clone over an existing path: {REPO_STORE}")
    stage_parent = Path(tempfile.mkdtemp(prefix="iclr-g0-clone-", dir="/content"))
    anonymous = stage_parent / "anonymous"
    authenticated = stage_parent / "authenticated"
    helper: Path | None = None
    auth_env: dict[str, str] | None = None
    clone_args = lambda destination: [
        "git", "clone", "--no-checkout", "--no-tags", REPOSITORY, str(destination)
    ]
    try:
        anonymous_env = os.environ.copy()
        anonymous_env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            run(clone_args(anonymous), env=anonymous_env)
            source = anonymous
        except RuntimeError as anonymous_error:
            if not _is_authentication_error(anonymous_error):
                raise
            auth_env, helper = credentials_if_needed()
            run(clone_args(authenticated), env=auth_env)
            source = authenticated
        os.replace(source, REPO_STORE)
    finally:
        if helper is not None and helper.exists():
            helper.unlink()
        if auth_env is not None:
            auth_env.pop("GITHUB_TOKEN", None)
        if stage_parent.exists():
            shutil.rmtree(stage_parent)


def ensure_source_checkout() -> None:
    auth_env: dict[str, str] | None = None
    helper: Path | None = None
    try:
        if not REPO_STORE.exists():
            clone_repository()
        elif not (REPO_STORE / ".git").exists():
            raise RuntimeError(f"refusing to use a non-Git path: {REPO_STORE}")

        harness_ref = f"refs/remotes/origin/{HARNESS_BRANCH}"
        champion_ref = f"refs/remotes/origin/{CHAMPION_BRANCH}"
        needs_fetch = True
        try:
            needs_fetch = not ref_contains_commit(REPO_STORE, harness_ref, HARNESS_COMMIT)
            needs_fetch = needs_fetch or not ref_contains_commit(REPO_STORE, champion_ref, CHAMPION_COMMIT)
        except RuntimeError:
            needs_fetch = True
        if needs_fetch:
            fetch = [
                "git", "-C", str(REPO_STORE), "fetch", "--prune", "origin",
                f"refs/heads/{HARNESS_BRANCH}:refs/remotes/origin/{HARNESS_BRANCH}",
                f"refs/heads/{CHAMPION_BRANCH}:refs/remotes/origin/{CHAMPION_BRANCH}",
            ]
            anonymous_env = os.environ.copy()
            anonymous_env["GIT_TERMINAL_PROMPT"] = "0"
            try:
                run(fetch, env=anonymous_env)
            except RuntimeError as anonymous_error:
                if not _is_authentication_error(anonymous_error):
                    raise
                auth_env, helper = credentials_if_needed()
                run(fetch, env=auth_env)

        if not ref_contains_commit(REPO_STORE, harness_ref, HARNESS_COMMIT):
            raise RuntimeError("remote harness branch does not contain the pinned harness commit")
        if git(REPO_STORE, "rev-parse", champion_ref) != CHAMPION_COMMIT:
            raise RuntimeError("immutable champion ref is not the certified 40-character commit")
        try:
            local_champion = git(REPO_STORE, "rev-parse", f"refs/heads/{CHAMPION_BRANCH}")
        except RuntimeError:
            run(["git", "-C", str(REPO_STORE), "branch", CHAMPION_BRANCH, CHAMPION_COMMIT])
            local_champion = git(REPO_STORE, "rev-parse", f"refs/heads/{CHAMPION_BRANCH}")
        if local_champion != CHAMPION_COMMIT:
            raise RuntimeError("local champion ref is not the certified 40-character commit")

        if PLANNING_ROOT.exists():
            assert_clean_checkout(PLANNING_ROOT, HARNESS_COMMIT)
        else:
            run(["git", "-C", str(REPO_STORE), "worktree", "add", "--detach", str(PLANNING_ROOT), HARNESS_COMMIT])
        contract = json.loads((PLANNING_ROOT / "configs/frozen_contract.json").read_text(encoding="utf-8"))
        assert_canonical_checkout(PLANNING_ROOT, contract)

        if CANDIDATE_ROOT.exists():
            assert_clean_checkout(CANDIDATE_ROOT, HARNESS_COMMIT)
            if git(CANDIDATE_ROOT, "symbolic-ref", "--quiet", "--short", "HEAD") != CANDIDATE_BRANCH:
                raise RuntimeError(f"candidate worktree is on the wrong branch: {CANDIDATE_ROOT}")
        else:
            branch_exists = True
            try:
                git(REPO_STORE, "show-ref", "--verify", f"refs/heads/{CANDIDATE_BRANCH}")
            except RuntimeError:
                branch_exists = False
            if branch_exists:
                run(["git", "-C", str(REPO_STORE), "worktree", "add", str(CANDIDATE_ROOT), CANDIDATE_BRANCH])
            else:
                run(["git", "-C", str(REPO_STORE), "worktree", "add", "-b", CANDIDATE_BRANCH, str(CANDIDATE_ROOT), HARNESS_COMMIT])
        assert_clean_checkout(CANDIDATE_ROOT, HARNESS_COMMIT)
        assert_canonical_checkout(CANDIDATE_ROOT, contract)
    finally:
        if helper is not None and helper.exists():
            helper.unlink()
        if auth_env is not None:
            auth_env.pop("GITHUB_TOKEN", None)


def stop_process_group(process: subprocess.Popen[object]) -> None:
    """Stop only the supervisor tree started by this cell."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    else:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        process.wait(timeout=10)


def run_supervisor(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    """Expose child failures even before the harness opens its own console log."""
    options: dict[str, object] = {}
    if os.name == "posix":
        options["start_new_session"] = True
    else:
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as transcript:
        transcript.write(f"\n=== Colab supervisor {_utcnow()} ===\n")
        print(f"Supervisor transcript (append-only): {log_path}", flush=True)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **options,
            )
        except OSError as exc:
            message = f"Generation-0 supervisor could not start: {exc}"
            transcript.write(message + "\n")
            raise RuntimeError(f"{message}. Supervisor transcript: {log_path}") from exc
        output_tail: list[str] = []
        try:
            assert process.stdout is not None
            for line in process.stdout:
                transcript.write(line)
                output_tail.append(line)
                if len(output_tail) > 200:
                    del output_tail[0]
                print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            # A notebook output or Drive write failure must also stop execution.
            stop_process_group(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
        transcript.write(f"\n=== Supervisor exit code {returncode} at {_utcnow()} ===\n")
    if returncode:
        tail = "".join(output_tail).strip() or "(supervisor emitted no output)"
        raise RuntimeError(
            "Generation-0 supervisor stopped fail-closed with exit code "
            f"{returncode}. Full supervisor transcript: {log_path}\n{tail}"
        )


def main() -> None:
    if not EXECUTE:
        print(json.dumps({
            "status": "PREPARED_NO_EXECUTION", "plan_id": PLAN_ID,
            "treatment_arms": 15, "run_slots": 50, "original_control_slots": 20,
            "excluded_cell": "G0-F05 (structurally infeasible; no imputed result)",
            "design": "constrained rank-complete; missing-cell caveat; no exact orthogonality or de-aliasing",
            "harness_commit": HARNESS_COMMIT or "UNPINNED: parent must commit the C15 implementation first",
            "manual_run": "Set ICRL_G0_HARNESS_COMMIT to that full 40-character SHA before pasting, and set EXECUTE=True.",
            "preserved_campaign": "iclr-g0-0ce16e9922b2",
        }, indent=2, sort_keys=True))
        return
    assert_full_sha(HARNESS_COMMIT, "HARNESS_COMMIT")
    if HARNESS_COMMIT == "0ce16e9922b2766df24698c96bf5eac34d5eb2b4":
        raise RuntimeError("C15 requires a new implementation commit; the original evidence campaign is immutable")
    if "iclr-g0-0ce16e9922b2" in CAMPAIGN_ROOT.parts:
        raise RuntimeError("the original evidence campaign is immutable; select a new C15 campaign")
    from google.colab import drive

    if not (Path("/content/drive") / "MyDrive").is_dir():
        drive.mount("/content/drive", force_remount=False)
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"required shared Drive path is missing: {DRIVE_ROOT}")
    if not (CORPUS_ROOT / "FROZEN.json").is_file():
        raise RuntimeError(f"frozen corpus manifest is missing: {CORPUS_ROOT / 'FROZEN.json'}")
    assert_full_sha(HARNESS_COMMIT, "HARNESS_COMMIT")
    assert_full_sha(CHAMPION_COMMIT, "CHAMPION_COMMIT")

    ensure_source_checkout()
    preview = json.loads(run([sys.executable, str(PLANNING_ROOT / "scripts/run_generation0.py")], cwd=PLANNING_ROOT))
    if preview.get("plan_id") != PLAN_ID or preview.get("arms") != 15 or preview.get("run_slots") != 50:
        raise RuntimeError("pinned checkout does not implement the C15 plan")
    assert_frozen_corpus(PLANNING_ROOT)
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    # Use the notebook runtime, including its installed packages, for both children.
    command_template = shlex.join([sys.executable, "-u", "scripts/generation0_candidate_runner.py"])
    supervisor = [
        sys.executable,
        "-u",
        str(PLANNING_ROOT / "scripts/run_generation0.py"),
        "--execute",
        "--candidate-root", str(CANDIDATE_ROOT),
        "--command-template", command_template,
        "--ledger", str(CAMPAIGN_ROOT / "generation0.jsonl"),
        "--campaign-dir", str(CAMPAIGN_ROOT),
        "--console-log", str(CAMPAIGN_ROOT / "generation0_console.log"),
        "--artifact-root", str(CAMPAIGN_ROOT / "artifacts"),
        "--corpus-root", str(CORPUS_ROOT),
    ]
    if GPU_UUID:
        supervisor.extend(["--gpu-uuid", GPU_UUID])
    if GPU_INDEX is not None:
        supervisor.extend(["--gpu-index", str(GPU_INDEX)])
    if RETRY_FAILED:
        supervisor.append("--retry-failed")
    print(json.dumps({
        "plan_id": PLAN_ID,
        "harness_commit": HARNESS_COMMIT,
        "champion_commit": CHAMPION_COMMIT,
        "python_executable": sys.executable,
        "candidate_root": str(CANDIDATE_ROOT),
        "campaign_root": str(CAMPAIGN_ROOT),
        "retry_failed": RETRY_FAILED,
        "note": "No failed slot is retried automatically; the supervisor stops at the first violation.",
    }, indent=2, sort_keys=True))
    environment = os.environ.copy()
    environment.update({
        "ICRL_CORPUS_ROOT": str(CORPUS_ROOT),
        "PYTHONUNBUFFERED": "1",
    })
    run_supervisor(
        supervisor,
        cwd=PLANNING_ROOT,
        env=environment,
        log_path=CAMPAIGN_ROOT / "generation0_cell.log",
    )


main()
