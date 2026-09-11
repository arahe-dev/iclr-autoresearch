"""Torch-free identities shared by the supervisor, workers, and ledger gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .contract import REPO_ROOT, canonical_json, contract_sha256, load_contract
from .generation0 import BASE_COMMIT, CHAMPION_BRANCH


CORPUS_FILES = (
    "train/shard_000000.tokens.bin",
    "train/shard_000000.valid_lengths.bin",
    "train/shard_000000.provenance.parquet",
)
TIMING_BOUNDARIES = {
    "clock": "cuda_events",
    "start": "after_input_transfer_valid_target_count_and_zero_grad_before_forward",
    "end": "after_loss_backward_before_zero_grad_and_cleanup",
    "synchronize_before_start": True,
    "synchronize_after_end": True,
    "includes": ["forward", "masked_cross_entropy", "backward"],
    "excludes": ["input_transfer", "optimizer_step", "zero_grad", "cleanup"],
    "peak_memory_scope": "max_allocated_across_warmup_and_timed_forward_backward",
}


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hex(value: Any, length: int, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{label} must be a full {length}-character lowercase hexadecimal identity")


def code_identity(root: Path) -> dict[str, Any]:
    """Bind a clean commit to the actual bytes of every tracked file loaded there.

    Hashing tracked files also binds schemas, plans, configuration and canonical
    evidence used by the static control gate. Paths are relative for portability.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    require_hex(commit, 40, "commit")
    if git("status", "--porcelain"):
        raise ValueError("execution provenance requires clean, committed harness and candidate checkouts")
    paths = git("ls-files", "-z").split("\0")
    files = {name: file_sha256(root / name) for name in paths if name}
    if not files:
        raise ValueError("execution provenance has no tracked source files")
    return {"commit": commit, "code_sha256": json_sha256(files), "files": files}


def frozen_identity(plan: dict[str, Any], root: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        "plan_id": plan["plan_id"],
        "plan_sha256": json_sha256(plan),
        "champion_commit": BASE_COMMIT,
        "champion_branch": CHAMPION_BRANCH,
        "contract_sha256": contract_sha256(load_contract(root / "configs/frozen_contract.json")),
        "config_sha256": file_sha256(root / "configs/generation_0_screening.json"),
        "canonical_source_sha256": file_sha256(root / "vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py"),
    }


def corpus_fingerprint(corpus_root: Path | None) -> dict[str, Any]:
    configured = corpus_root or os.environ.get("ICRL_CORPUS_ROOT")
    if configured is None:
        return {"status": "UNRESOLVED", "root": None}
    root = Path(configured).expanduser().resolve()
    frozen = root / "FROZEN.json"
    if not frozen.is_file():
        return {"status": "UNRESOLVED", "root": str(root)}
    result: dict[str, Any] = {
        "status": "RESOLVED", "root": str(root),
        "frozen_json_sha256": file_sha256(frozen),
        "frozen_json": json.loads(frozen.read_text(encoding="utf-8")),
        "required_files": [],
    }
    for name in CORPUS_FILES:
        path = root / name
        entry: dict[str, Any] = {"path": name, "exists": path.is_file()}
        if path.is_file():
            entry.update(size=path.stat().st_size, sha256=file_sha256(path))
        else:
            result["status"] = "INCOMPLETE"
        result["required_files"].append(entry)
    result["fingerprint"] = json_sha256(result)
    return result


def validate_corpus(corpus: Any) -> None:
    if not isinstance(corpus, dict) or corpus.get("status") != "RESOLVED" or not corpus.get("root"):
        raise ValueError("corpus must have a resolved fingerprint")
    require_hex(corpus.get("frozen_json_sha256"), 64, "FROZEN.json SHA256")
    require_hex(corpus.get("fingerprint"), 64, "corpus fingerprint")
    if corpus["fingerprint"] != json_sha256({k: v for k, v in corpus.items() if k != "fingerprint"}):
        raise ValueError("corpus fingerprint disagrees with its contents")
    data = load_contract()["data"]
    expected = {"status": "FROZEN", **{key: data[key] for key in (
        "corpus_id", "context_length", "tokenizer_sha256", "logical_replay_sha256")}}
    frozen = corpus.get("frozen_json")
    if not isinstance(frozen, dict) or any(frozen.get(k) != v for k, v in expected.items()):
        raise ValueError("corpus differs from the frozen science contract")
    files = corpus.get("required_files")
    if not isinstance(files, list) or [entry.get("path") for entry in files if isinstance(entry, dict)] != list(CORPUS_FILES):
        raise ValueError("corpus must identify all three benchmark files in order")
    for entry in files:
        if entry.get("exists") is not True or type(entry.get("size")) is not int or entry["size"] <= 0:
            raise ValueError("corpus file is missing or empty")
        require_hex(entry.get("sha256"), 64, "corpus file SHA256")


def validate_code_identity(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("source code identity must be an object")
    require_hex(value.get("commit"), 40, "code commit")
    require_hex(value.get("code_sha256"), 64, "code SHA256")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source code identity must include file hashes")
    for name, digest in files.items():
        if not isinstance(name, str) or not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise ValueError("source code hashes must use repository-relative paths")
        require_hex(digest, 64, "code file SHA256")
    if value["code_sha256"] != json_sha256(files):
        raise ValueError("code SHA256 disagrees with its file hashes")
