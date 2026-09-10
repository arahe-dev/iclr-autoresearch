"""One self-contained Google Colab cell for the repaired Generation-0 run.

Paste this entire file into one Colab cell.  It mounts the frozen Drive
corpus, securely fetches the private repository when needed, creates fresh
planning/candidate worktrees at the pinned harness commit, and delegates all
execution to the fail-closed supervisor.  It never deletes an existing
campaign or silently retries a failed slot.
"""

# This is intentionally a plain Python cell: it has no dependency on variables
# or imports from an earlier notebook cell.
from __future__ import annotations

from getpass import getpass
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


REPOSITORY = "https://github.com/arahe-dev/iclr-autoresearch.git"
HARNESS_BRANCH = "codex/generation0/sm120-runner-20260909"
HARNESS_COMMIT = "0ce16e9922b2766df24698c96bf5eac34d5eb2b4"
CHAMPION_BRANCH = "codex/champion/0000-validated-bdh-baseline"
CHAMPION_COMMIT = "908b0b1438ba038d319adf97787aac08f213b590"
PLAN_ID = "G0-L8-SM120-EXACT-B32"

DRIVE_ROOT = Path("/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh")
CORPUS_ROOT = DRIVE_ROOT / "corpus/stage2/frozen_5b_v1"
REPO_STORE = Path(f"/content/iclr-g0-repository-{HARNESS_COMMIT[:12]}")
PLANNING_ROOT = Path(f"/content/iclr-g0-planning-{HARNESS_COMMIT[:12]}")
CANDIDATE_ROOT = Path(f"/content/iclr-g0-candidate-{HARNESS_COMMIT[:12]}")
CANDIDATE_BRANCH = f"codex/generation0/colab-{HARNESS_COMMIT[:12]}"
CAMPAIGN_ROOT = DRIVE_ROOT / "experiments/generation0" / f"iclr-g0-{HARNESS_COMMIT[:12]}"


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


def assert_clean_checkout(path: Path, expected_commit: str) -> None:
    if not path.is_dir() or not (path / ".git").exists():
        raise RuntimeError(f"expected existing Git checkout is missing: {path}")
    if git(path, "status", "--porcelain"):
        raise RuntimeError(f"refusing to reuse a dirty checkout: {path}")
    actual = git(path, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise RuntimeError(f"checkout {path} is at {actual}, expected pinned {expected_commit}")


def credentials_if_needed() -> tuple[dict[str, str], Path | None]:
    """Create a short-lived askpass helper; the PAT never enters argv/output."""
    username = os.environ.get("GITHUB_USER") or input("GitHub username: ").strip()
    token = os.environ.get("GITHUB_TOKEN") or getpass("GitHub PAT (input hidden): ")
    if not username or not token:
        raise RuntimeError("a GitHub username and PAT are required for the private repository")
    helper_fd, helper_name = tempfile.mkstemp(prefix="iclr-git-askpass-", suffix=".sh")
    os.close(helper_fd)
    helper = Path(helper_name)
    helper.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' \"$GITHUB_USER\" ;;\n"
        "  *) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    helper.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    auth_env = os.environ.copy()
    auth_env.update({
        "GIT_ASKPASS": str(helper),
        "GIT_TERMINAL_PROMPT": "0",
        "GITHUB_USER": username,
        "GITHUB_TOKEN": token,
    })
    return auth_env, helper


def ensure_source_checkout() -> None:
    auth_env: dict[str, str] | None = None
    helper: Path | None = None
    try:
        if not REPO_STORE.exists():
            auth_env, helper = credentials_if_needed()
            run(["git", "clone", "--no-checkout", REPOSITORY, str(REPO_STORE)], env=auth_env)
        elif not (REPO_STORE / ".git").exists():
            raise RuntimeError(f"refusing to use a non-Git path: {REPO_STORE}")

        harness_ref = f"refs/remotes/origin/{HARNESS_BRANCH}"
        champion_ref = f"refs/remotes/origin/{CHAMPION_BRANCH}"
        needs_fetch = True
        try:
            needs_fetch = git(REPO_STORE, "rev-parse", harness_ref) != HARNESS_COMMIT
            needs_fetch = needs_fetch or git(REPO_STORE, "rev-parse", champion_ref) != CHAMPION_COMMIT
        except RuntimeError:
            needs_fetch = True
        if needs_fetch:
            if auth_env is None:
                auth_env, helper = credentials_if_needed()
            run([
                "git", "-C", str(REPO_STORE), "fetch", "--prune", "origin",
                f"refs/heads/{HARNESS_BRANCH}:refs/remotes/origin/{HARNESS_BRANCH}",
                f"refs/heads/{CHAMPION_BRANCH}:refs/remotes/origin/{CHAMPION_BRANCH}",
            ], env=auth_env)

        if git(REPO_STORE, "rev-parse", harness_ref) != HARNESS_COMMIT:
            raise RuntimeError("remote harness branch is not the pinned commit")
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
    finally:
        if helper is not None and helper.exists():
            helper.unlink()
        if auth_env is not None:
            auth_env.pop("GITHUB_TOKEN", None)


def main() -> None:
    from google.colab import drive

    if not (Path("/content/drive") / "MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)
    if not DRIVE_ROOT.exists():
        raise RuntimeError(f"required shared Drive path is missing: {DRIVE_ROOT}")
    if not (CORPUS_ROOT / "FROZEN.json").exists():
        raise RuntimeError(f"frozen corpus manifest is missing: {CORPUS_ROOT / 'FROZEN.json'}")
    if len(HARNESS_COMMIT) != 40 or len(CHAMPION_COMMIT) != 40:
        raise RuntimeError("pinned Git commits must be full 40-character SHAs")

    ensure_source_checkout()
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    command_template = f"{shutil.which('python3') or '/usr/bin/python3'} -u scripts/generation0_candidate_runner.py"
    supervisor = [
        shutil.which("python3") or "/usr/bin/python3",
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
    print(json.dumps({
        "plan_id": PLAN_ID,
        "harness_commit": HARNESS_COMMIT,
        "champion_commit": CHAMPION_COMMIT,
        "candidate_root": str(CANDIDATE_ROOT),
        "campaign_root": str(CAMPAIGN_ROOT),
        "note": "No failed slot is retried automatically; the supervisor stops at the first violation.",
    }, indent=2, sort_keys=True))
    completed = subprocess.run(supervisor, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Generation-0 stopped fail-closed with exit code {completed.returncode}")


main()
