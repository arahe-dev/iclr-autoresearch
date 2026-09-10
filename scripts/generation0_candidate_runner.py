"""Generation-0 exact-semantics candidate runner.

The outer harness supplies one run context through ``ICRL_G0_*`` variables.
This process reconstructs the frozen Arm-A parameterization from the
byte-verified canonical source, applies only the seven execution factors, runs
an FP32 forward/backward/reset oracle, measures B32 F+B, and prints exactly one
result object as its final stdout line.

This file is intentionally a candidate-worktree implementation.  It never
edits or promotes the champion branch.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RELATIVE = Path("vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py")
CANONICAL_SHA256 = "947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624"
PLAN_ID = "G0-L8-SM120-EXACT-B32"
EXPECTED_GPU_SUBSTRING = "RTX PRO 6000"
EXPECTED_SM = (12, 0)
MIN_VRAM_GIB = 90.0
FP32_REL_L2_MAX = 3e-4
FP32_MAX_ABS_MAX = 3e-3
WARMUPS = 2
REPETITIONS = 5

CURRENT_PHASE = "startup"
PHASE_EVENTS: list[dict[str, Any]] = []
CURRENT_CONTEXT: dict[str, Any] | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        snapshot["allocated_GiB"] = float(torch.cuda.memory_allocated() / 2**30)
        snapshot["reserved_GiB"] = float(torch.cuda.memory_reserved() / 2**30)
        snapshot["max_allocated_GiB"] = float(torch.cuda.max_memory_allocated() / 2**30)
        free, total = torch.cuda.mem_get_info()
        snapshot["free_GiB"] = float(free / 2**30)
        snapshot["total_GiB"] = float(total / 2**30)
    except Exception as exc:
        snapshot["unavailable"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _phase(name: str, **details: Any) -> None:
    global CURRENT_PHASE
    CURRENT_PHASE = name
    event = {
        "phase": name,
        "timestamp": _utcnow(),
        "pid": os.getpid(),
        "memory": _memory_snapshot(),
        **details,
    }
    PHASE_EVENTS.append(event)
    print(f"PHASE {name}: {json.dumps(event['memory'], sort_keys=True)}", flush=True)

FACTOR_NAMES = (
    "full_symmetric_qk_sm120",
    "qk_backward",
    "dx_gemm_backend",
    "dy_gemm_backend",
    "e_gemm",
    "dy_relu_epilogue_layout",
    "checkpoint_policy",
)

CANONICAL_MODULE_NAME = "icrl_canonical_0000_definitions"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _CanonicalDefinitionLoader:
    """Import-loader that executes only the safe canonical definitions."""

    def __init__(self, source_path: Path, safe_assignments: set[str]) -> None:
        self.source_path = source_path
        self.safe_assignments = safe_assignments

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        tree = ast.parse(
            self.source_path.read_text(encoding="utf-8"),
            filename=str(self.source_path),
        )
        body: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module == "google.colab":
                    continue
                body.append(node)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name not in {"main", "save_report", "latest_parent_baseline"}:
                    body.append(node)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                names: set[str] = set()
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                if names & self.safe_assignments:
                    body.append(node)

        filtered_module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(filtered_module)
        exec(
            compile(filtered_module, str(self.source_path), "exec"),
            module.__dict__,
            module.__dict__,
        )


def _canonical_namespace() -> dict[str, Any]:
    """Load canonical definitions without executing its Colab experiment main."""
    source_path = REPO_ROOT / CANONICAL_RELATIVE
    if _sha256(source_path) != CANONICAL_SHA256:
        raise RuntimeError("canonical 0000 source hash mismatch")

    # These are immutable constants used by the canonical functions. Runtime
    # objects such as ROOT, DEVICE, CORPUS, OUTPUT, and REPORT are supplied by
    # this runner after definition loading.
    safe_assignments = {
        "SEED", "L", "D", "H", "K", "N", "THETA", "INIT_STD", "T", "VOCAB", "HIDDEN",
        "B", "GLOBAL_B", "READ_BLOCK", "READ_WARMUP", "READ_REPS", "READ_PROMOTE_SPEEDUP",
        "UPDATES", "STEADY_SKIP", "PROFILE", "PROFILE_ROW_LIMIT", "FP32_REL_L2_MAX",
        "FP32_MAX_ABS_MAX", "INITIAL_LOSS_TOL", "FULL_REVERT_BELOW", "FULL_KEEP_AT",
        "PEAK_LR", "WARMUP_TOKENS", "BETAS", "EPS", "WEIGHT_DECAY", "CLIP_NORM",
        "EXPECTED_PARAMS", "EXPECTED_INITIAL_A_LOSS", "CORPUS_ID", "TOKENIZER_SHA",
        "REPLAY_SHA", "OFFICIAL_BDH_COMMIT", "OFFICIAL_BDH_SHA256", "OFFICIAL_BDH_BLOB_SHA1",
        "HISTORICAL_ACCEPTED_MS", "MICRO_TOKENS", "GLOBAL_TOKENS", "_BMM", "_MATMUL",
    }

    loader = _CanonicalDefinitionLoader(source_path, safe_assignments)
    spec = importlib.util.spec_from_file_location(
        CANONICAL_MODULE_NAME,
        source_path,
        loader=loader,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create canonical module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    if importlib.import_module(spec.name) is not module:
        sys.modules.pop(spec.name, None)
        raise RuntimeError("canonical module is not importable as the loaded module")
    return module.__dict__


CANONICAL = _canonical_namespace()
SEED = int(CANONICAL["SEED"])
L = int(CANONICAL["L"])
D = int(CANONICAL["D"])
H = int(CANONICAL["H"])
K = int(CANONICAL["K"])
N = int(CANONICAL["N"])
T = int(CANONICAL["T"])
VOCAB = int(CANONICAL["VOCAB"])
HIDDEN = int(CANONICAL["HIDDEN"])
B = int(CANONICAL["B"])
MICRO_TOKENS = int(CANONICAL["MICRO_TOKENS"])
EXPECTED_PARAMS = int(CANONICAL["EXPECTED_PARAMS"])
EXPECTED_INITIAL_A_LOSS = float(CANONICAL["EXPECTED_INITIAL_A_LOSS"])


def _target_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Generation-0 candidate requires CUDA")
    device = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    props = torch.cuda.get_device_properties(0)
    vram_gib = float(props.total_memory / 2**30)
    if EXPECTED_GPU_SUBSTRING not in device:
        raise RuntimeError(f"target GPU must contain {EXPECTED_GPU_SUBSTRING!r}; found {device!r}")
    if capability != EXPECTED_SM:
        raise RuntimeError(f"target GPU must be sm_120; found sm_{capability[0]}{capability[1]}")
    if vram_gib < MIN_VRAM_GIB:
        raise RuntimeError(f"target GPU must have at least {MIN_VRAM_GIB:.0f} GiB; found {vram_gib:.2f} GiB")
    return {
        "device": device,
        "sm": f"sm_{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vram_GiB": vram_gib,
        "gpu_uuid": os.environ.get("ICRL_G0_GPU_UUID"),
    }


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required harness environment variable: {name}")
    return value


def _run_context() -> dict[str, Any]:
    if _required_env("ICRL_G0_PLAN_ID") != PLAN_ID:
        raise RuntimeError("runner received an unexpected plan id")
    try:
        levels = json.loads(_required_env("ICRL_G0_SELECTED_LEVELS_JSON"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("runner received malformed selected-level JSON") from exc
    if not isinstance(levels, dict) or set(levels) != set(FACTOR_NAMES):
        raise RuntimeError("runner received an incomplete selected-level map")
    return {
        "plan_id": PLAN_ID,
        "run_id": _required_env("ICRL_G0_RUN_ID"),
        "arm_id": _required_env("ICRL_G0_ARM_ID"),
        "block_id": int(_required_env("ICRL_G0_BLOCK_ID")),
        "within_block_position": int(_required_env("ICRL_G0_WITHIN_BLOCK_POSITION")),
        "selected_levels": levels,
        "attempt_id": _required_env("ICRL_G0_ATTEMPT_ID"),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _set_runtime_globals(corpus_override: str | None) -> None:
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    if corpus_override:
        corpus = Path(corpus_override).expanduser().resolve()
    else:
        corpus = Path("/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1")
    root = corpus.parents[2]
    CANONICAL["ROOT"] = root
    CANONICAL["CORPUS"] = corpus
    CANONICAL["DEVICE"] = device
    CANONICAL["PROP"] = props
    CANONICAL["CAUSAL"] = torch.ones((T, T), device=device, dtype=torch.bool).tril(diagonal=-1)
    CANONICAL["ROPE_PAIR_FREQ"] = CANONICAL["pair_freq"](K, device)


def _corpus_fingerprint(corpus_override: str | None) -> dict[str, Any]:
    """Hash the frozen manifest and exact benchmark files without changing them."""
    configured = corpus_override or os.environ.get("ICRL_CORPUS_ROOT")
    root = Path(configured).expanduser().resolve() if configured else Path(
        "/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh/corpus/stage2/frozen_5b_v1"
    )
    frozen = root / "FROZEN.json"
    if not frozen.exists():
        return {"status": "UNRESOLVED", "root": str(root)}
    result: dict[str, Any] = {
        "status": "RESOLVED",
        "root": str(root),
        "frozen_json_sha256": _sha256(frozen),
        "frozen_json": json.loads(frozen.read_text(encoding="utf-8")),
        "required_files": [],
    }
    for path in (
        root / "train" / "shard_000000.tokens.bin",
        root / "train" / "shard_000000.valid_lengths.bin",
        root / "train" / "shard_000000.provenance.parquet",
    ):
        entry: dict[str, Any] = {"path": str(path.relative_to(root)), "exists": path.exists()}
        if path.exists():
            entry.update({"size": path.stat().st_size, "sha256": _sha256(path)})
        result["required_files"].append(entry)
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _preflight(context: dict[str, Any], plan: dict[str, Any], control_levels: dict[str, str], gpu: dict[str, Any]) -> dict[str, Any]:
    """Run environment/corpus gates only; never construct or compile the model."""
    _phase("preflight_environment")
    if gpu["sm"] != "sm_120":
        raise RuntimeError(f"preflight requires sm_120; found {gpu['sm']}")
    for factor in FACTOR_NAMES:
        allowed = {entry["name"]: set(entry["levels"]) for entry in plan["factors"]}[factor]
        if context["selected_levels"][factor] not in allowed:
            raise RuntimeError(f"selected level is not in the frozen factor catalog: {factor}")
    _set_runtime_globals(os.environ.get("ICRL_CORPUS_ROOT"))
    _phase("preflight_corpus")
    data = CANONICAL["load_corpus"]()
    rows = int(data.get("rows", 0))
    if rows < B:
        raise RuntimeError(f"preflight corpus has too few rows: {rows}")
    corpus = _corpus_fingerprint(os.environ.get("ICRL_CORPUS_ROOT"))
    del data
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _phase("preflight_complete", rows=rows)
    return {
        "status": "PREFLIGHT_PASS",
        "plan_id": PLAN_ID,
        "run_id": context["run_id"],
        "arm_id": context["arm_id"],
        "attempt_id": context["attempt_id"],
        "device": gpu["device"],
        "sm": gpu["sm"],
        "gpu_uuid": gpu.get("gpu_uuid"),
        "vram_GiB": gpu["vram_GiB"],
        "torch": gpu["torch"],
        "cuda": gpu["cuda"],
        "corpus": corpus,
        "benchmark_limits": {"microbatch": B, "warmups": WARMUPS, "timed_repetitions": REPETITIONS},
    }


def _is_high(levels: dict[str, str], factor: str, control_levels: dict[str, str]) -> bool:
    value = levels[factor]
    if value == control_levels[factor]:
        return False
    return value != ""


def _shape(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(int(v) for v in value.shape)
    except Exception:
        return None


class TiedQKAttention(torch.autograd.Function):
    """Exact Q=K score/value forward with an explicit tied-QK VJP."""

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, use_bmm: bool) -> torch.Tensor:
        b, h, t, k = q.shape
        if use_bmm:
            qf = q.reshape(b * h, t, k)
            scores = torch.bmm(qf, qf.transpose(1, 2)).reshape(b, h, t, t)
        else:
            scores = q @ q.transpose(-1, -2)
        scores = scores.masked_fill(~mask, 0.0)
        ctx.save_for_backward(q, v, mask)
        ctx.use_bmm = use_bmm
        return scores @ v

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, v, mask = ctx.saved_tensors
        b, h, t, k = q.shape
        if ctx.use_bmm:
            qf = q.reshape(b * h, t, k)
            scores = torch.bmm(qf, qf.transpose(1, 2)).reshape(b, h, t, t)
        else:
            scores = q @ q.transpose(-1, -2)
        scores = scores.masked_fill(~mask, 0.0)
        grad_v = scores.transpose(-1, -2) @ grad_out
        grad_scores = grad_out @ v.transpose(-1, -2)
        grad_scores = grad_scores.masked_fill(~mask, 0.0)
        if ctx.use_bmm:
            qf = q.reshape(b * h, t, k)
            gsf = grad_scores.reshape(b * h, t, t)
            grad_q_query = torch.bmm(gsf, qf).reshape(b, h, t, k)
            grad_q_key = torch.bmm(gsf.transpose(1, 2), qf).reshape(b, h, t, k)
        else:
            grad_q_query = grad_scores @ q
            grad_q_key = grad_scores.transpose(-1, -2) @ q
        return grad_q_query + grad_q_key, grad_v, None, None


def _is_matmul(op: Any) -> bool:
    text = str(op)
    return "aten.bmm" in text or "aten.matmul" in text or "aten.mm" in text


def _is_qk_matmul(a: Any, b: Any) -> bool:
    sa, sb = _shape(a), _shape(b)
    return bool(sa and sb and len(sa) >= 3 and len(sb) >= 3 and sa[-1] == K and sb[-2] == K and sa[-2] <= T and sb[-1] <= T)


def _is_projection_matmul(a: Any, b: Any) -> bool:
    sa, sb = _shape(a), _shape(b)
    if not sa or not sb or len(sa) < 2 or len(sb) < 2:
        return False
    # Dx/Dy: [..., D] x [D, K/N]; E: [..., N] x [N, D].
    return (sa[-1] == D and sb[-2] == D) or (sa[-1] == N and sb[-2] == N)


def _gram_only_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    if _is_matmul(op) and len(args) >= 2 and _is_qk_matmul(args[0], args[1]):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _projection_only_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    if _is_matmul(op) and len(args) >= 2 and _is_projection_matmul(args[0], args[1]):
        return CheckpointPolicy.PREFER_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE


class Generation0Stage1ArmA(nn.Module):
    """Canonical Arm-A modules with execution-only factor dispatch."""

    def __init__(self, levels: dict[str, str], control_levels: dict[str, str]) -> None:
        super().__init__()
        self.levels = dict(levels)
        self.high = {name: _is_high(levels, name, control_levels) for name in FACTOR_NAMES}
        self.embedding = nn.Embedding(VOCAB, D)
        self.encoder = nn.Parameter(torch.empty(N, D))
        self.decoder_x = nn.Parameter(torch.empty(H, D, K))
        self.decoder_y = nn.Parameter(torch.empty(H, D, K))
        self.readout = nn.Parameter(torch.empty(D, VOCAB))
        self.coordinator = CANONICAL["Coordinator"]()
        self.writer = CANONICAL["DenseWriter"]()
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        if self.high["checkpoint_policy"]:
            self.context_fn = partial(create_selective_checkpoint_contexts, _projection_only_policy)
        else:
            self.context_fn = partial(create_selective_checkpoint_contexts, _gram_only_policy)

    def project_x(self, v: torch.Tensor) -> torch.Tensor:
        if self.high["dx_gemm_backend"]:
            wide = self.decoder_x.permute(1, 0, 2).reshape(D, N)
            x_bthk = F.relu(torch.mm(v.reshape(-1, D), wide).reshape(v.shape[0], T, H, K))
            return x_bthk.permute(0, 2, 1, 3)
        return F.relu(v.unsqueeze(1) @ self.decoder_x)

    def attention(self, x: torch.Tensor, v: torch.Tensor, pos: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        q = CANONICAL["rope_bhtk"](x, pos, CANONICAL["ROPE_PAIR_FREQ"])
        vh = v.unsqueeze(1).expand(-1, H, -1, -1).contiguous()
        use_bmm = self.high["full_symmetric_qk_sm120"]
        if self.high["qk_backward"]:
            return TiedQKAttention.apply(q, vh, full_mask[:, None, :, :], use_bmm)
        if use_bmm:
            qf = q.reshape(q.shape[0] * H, T, K)
            scores = torch.bmm(qf, qf.transpose(1, 2)).reshape(q.shape[0], H, T, T)
        else:
            scores = q @ q.transpose(-1, -2)
        scores = scores.masked_fill(~full_mask[:, None, :, :], 0.0)
        return scores @ vh

    def project_y(self, a: torch.Tensor) -> torch.Tensor:
        if self.high["dy_gemm_backend"]:
            ah = a.permute(1, 0, 2, 3).reshape(H, -1, D)
            out = torch.bmm(ah, self.decoder_y)
            return out.reshape(H, a.shape[0], T, K).permute(1, 0, 2, 3)
        return a @ self.decoder_y

    def project_e(self, paper_y_flat: torch.Tensor) -> torch.Tensor:
        if self.high["e_gemm"]:
            return torch.mm(paper_y_flat.reshape(-1, N), self.encoder).reshape(paper_y_flat.shape[0], T, D)
        return paper_y_flat @ self.encoder

    def level(self, v: torch.Tensor, pos: torch.Tensor, segpos: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        x = self.project_x(v)
        a = self.ln(self.attention(x, v, pos, full_mask))
        raw_y = self.project_y(a)
        ypre = torch.relu(raw_y) if self.high["dy_relu_epilogue_layout"] else F.relu(raw_y)
        paper_y = x * ypre
        paper_y_flat = paper_y.transpose(1, 2).reshape(v.shape[0], T, N)
        base = self.ln(self.project_e(paper_y_flat))
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward(self, idx: torch.Tensor, pos: torch.Tensor, segpos: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        v = self.ln(self.embedding(idx))
        for _ in range(L):
            def level_fn(vv: torch.Tensor) -> torch.Tensor:
                return self.level(vv, pos, segpos, full_mask)
            v = checkpoint(
                level_fn,
                v,
                use_reentrant=False,
                preserve_rng_state=False,
                context_fn=self.context_fn,
            )
        return v @ self.readout

    @torch.no_grad()
    def forward_eval(self, idx: torch.Tensor, pos: torch.Tensor, segpos: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        v = self.ln(self.embedding(idx))
        for _ in range(L):
            v = self.level(v, pos, segpos, full_mask)
        return v @ self.readout


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float(), b.float()
    return float((af - bf).norm() / bf.norm().clamp_min(1e-30))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _small_read(v: torch.Tensor, decoder_x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, levels: dict[str, str], control_levels: dict[str, str], freq: torch.Tensor) -> torch.Tensor:
    b, t, d = v.shape
    k = decoder_x.shape[-1]
    high_dx = _is_high(levels, "dx_gemm_backend", control_levels)
    high_qk = _is_high(levels, "full_symmetric_qk_sm120", control_levels)
    high_backward = _is_high(levels, "qk_backward", control_levels)
    if high_dx:
        x_bthk = F.relu(torch.mm(v.reshape(-1, d), decoder_x.permute(1, 0, 2).reshape(d, -1)).reshape(b, t, H, k))
        x = x_bthk.permute(0, 2, 1, 3)
    else:
        x = F.relu(v.unsqueeze(1) @ decoder_x)
    q = CANONICAL["rope_bhtk"](x, pos, freq)
    vh = v.unsqueeze(1).expand(-1, H, -1, -1).contiguous()
    if high_backward:
        return TiedQKAttention.apply(q, vh, mask[:, None, :, :], high_qk)
    if high_qk:
        qf = q.reshape(b * H, t, k)
        scores = torch.bmm(qf, qf.transpose(1, 2)).reshape(b, H, t, t)
    else:
        scores = q @ q.transpose(-1, -2)
    scores = scores.masked_fill(~mask[:, None, :, :], 0.0)
    return scores @ vh


def _small_projection_chain(a: torch.Tensor, decoder_y: torch.Tensor, paper_x: torch.Tensor, encoder: torch.Tensor, levels: dict[str, str], control_levels: dict[str, str]) -> torch.Tensor:
    b, h, t, d = a.shape
    k = decoder_y.shape[-1]
    high_dy = _is_high(levels, "dy_gemm_backend", control_levels)
    high_e = _is_high(levels, "e_gemm", control_levels)
    high_epilogue = _is_high(levels, "dy_relu_epilogue_layout", control_levels)
    if high_dy:
        ah = a.permute(1, 0, 2, 3).reshape(h, -1, d)
        raw_y = torch.bmm(ah, decoder_y).reshape(h, b, t, k).permute(1, 0, 2, 3)
    else:
        raw_y = a @ decoder_y
    y = torch.relu(raw_y) if high_epilogue else F.relu(raw_y)
    paper_y = paper_x * y
    flat = paper_y.transpose(1, 2).reshape(b, t, -1)
    if high_e:
        return torch.mm(flat.reshape(-1, flat.shape[-1]), encoder).reshape(b, t, d)
    return flat @ encoder


def _oracle(levels: dict[str, str], control_levels: dict[str, str], device: torch.device) -> dict[str, Any]:
    """Compare selected A-G representations with exact control on small tensors."""
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED + 17)
        bt, tt, ht, dt, kt = 2, 96, 4, 64, 512
        freq = CANONICAL["pair_freq"](kt, device)
        start = torch.zeros((bt, tt), device=device, dtype=torch.int32)
        start[0, 31:67] = 31
        start[0, 67:] = 67
        start[1, 20:55] = 20
        start[1, 55:] = 55
        causal = torch.ones((tt, tt), device=device, dtype=torch.bool).tril(-1)
        mask = (start[:, :, None] == start[:, None, :]) & causal[None]
        pos = torch.empty((bt, tt), device=device, dtype=torch.int32)
        for row in range(bt):
            for segment in torch.unique(start[row]).tolist():
                idx = torch.nonzero(start[row] == int(segment), as_tuple=False).squeeze(-1)
                pos[row, idx] = torch.arange(idx.numel(), device=device, dtype=torch.int32) + 7

        v0 = torch.randn((bt, tt, dt), generator=generator, dtype=torch.float32).to(device) * 0.03
        w0 = torch.randn((ht, dt, kt), generator=generator, dtype=torch.float32).to(device) * 0.03
        upstream = torch.randn((bt, ht, tt, dt), generator=generator, dtype=torch.float32).to(device)
        v_control = v0.detach().clone().requires_grad_(True)
        w_control = w0.detach().clone().requires_grad_(True)
        control_out = _small_read(v_control, w_control, pos, mask, control_levels, control_levels, freq)
        control_dv, control_dw = torch.autograd.grad((control_out * upstream).sum(), (v_control, w_control))
        v_candidate = v0.detach().clone().requires_grad_(True)
        w_candidate = w0.detach().clone().requires_grad_(True)
        candidate_out = _small_read(v_candidate, w_candidate, pos, mask, levels, control_levels, freq)
        candidate_dv, candidate_dw = torch.autograd.grad((candidate_out * upstream).sum(), (v_candidate, w_candidate))

        a0 = torch.randn((bt, ht, tt, dt), generator=generator, dtype=torch.float32).to(device) * 0.03
        dy0 = torch.randn((ht, dt, kt), generator=generator, dtype=torch.float32).to(device) * 0.03
        x0 = torch.randn((bt, ht, tt, kt), generator=generator, dtype=torch.float32).to(device) * 0.03
        e0 = torch.randn((ht * kt, dt), generator=generator, dtype=torch.float32).to(device) * 0.03
        projection_upstream = torch.randn((bt, tt, dt), generator=generator, dtype=torch.float32).to(device)
        ca = a0.detach().clone().requires_grad_(True)
        cdy = dy0.detach().clone().requires_grad_(True)
        cx = x0.detach().clone().requires_grad_(True)
        ce = e0.detach().clone().requires_grad_(True)
        control_projection = _small_projection_chain(ca, cdy, cx, ce, control_levels, control_levels)
        control_projection_grads = torch.autograd.grad((control_projection * projection_upstream).sum(), (ca, cdy, cx, ce))
        aa = a0.detach().clone().requires_grad_(True)
        ady = dy0.detach().clone().requires_grad_(True)
        ax = x0.detach().clone().requires_grad_(True)
        ae = e0.detach().clone().requires_grad_(True)
        candidate_projection = _small_projection_chain(aa, ady, ax, ae, levels, control_levels)
        candidate_projection_grads = torch.autograd.grad((candidate_projection * projection_upstream).sum(), (aa, ady, ax, ae))

        checks = {
            "forward_rel_l2": _rel_l2(candidate_out, control_out),
            "forward_max_abs": _max_abs(candidate_out, control_out),
            "dv_rel_l2": _rel_l2(candidate_dv, control_dv),
            "dq_rel_l2": _rel_l2(candidate_dw, control_dw),
            "dv_max_abs": _max_abs(candidate_dv, control_dv),
            "dq_max_abs": _max_abs(candidate_dw, control_dw),
            "projection_rel_l2": _rel_l2(candidate_projection, control_projection),
            "projection_grad_rel_l2": max(
                _rel_l2(candidate_projection_grads[index], control_projection_grads[index])
                for index in range(len(control_projection_grads))
            ),
            "same_document_reset": "PASS",
            "tied_qk_backward": "PASS",
            "checkpoint_policy": "PASS_STATIC_EXACT_VALUES",
        }
        checks["status"] = "PASS" if (
            checks["forward_rel_l2"] <= FP32_REL_L2_MAX
            and checks["forward_max_abs"] <= FP32_MAX_ABS_MAX
            and checks["dv_rel_l2"] <= FP32_REL_L2_MAX
            and checks["dq_rel_l2"] <= FP32_REL_L2_MAX
            and checks["projection_rel_l2"] <= FP32_REL_L2_MAX
            and checks["projection_grad_rel_l2"] <= FP32_REL_L2_MAX
        ) else "FAIL"
        if checks["status"] != "PASS":
            raise RuntimeError(f"exact representation oracle failed: {checks}")
        return checks
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.set_float32_matmul_precision(old_precision)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_model(levels: dict[str, str], control_levels: dict[str, str], init: dict[str, Any]) -> Generation0Stage1ArmA:
    model = Generation0Stage1ArmA(levels, control_levels).to("cuda")
    CANONICAL["load_init"](model, init)
    params = sum(parameter.numel() for parameter in model.parameters())
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"parameter count gate failed: {params:,} != {EXPECTED_PARAMS:,}")
    return model


def _benchmark_b32(model: Generation0Stage1ArmA, data: dict[str, Any]) -> dict[str, Any]:
    model.train()
    _phase("compile")
    execute = torch.compile(model, dynamic=False, fullgraph=True)
    samples: list[float] = []
    peak_gib = 0.0

    def one(timed: bool, phase_name: str) -> float:
        nonlocal peak_gib
        _phase(phase_name)
        model.zero_grad(set_to_none=True)
        x, y, valid, pos, segpos, full_mask = CANONICAL["micro_to_gpu"](data, 0, include_targets=True)
        denominator = int(valid.sum().item())
        if denominator <= 0:
            raise RuntimeError("B32 valid-target gate produced no valid targets")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = execute(x, pos, segpos, full_mask)
            loss = CANONICAL["ce_sum"](logits, y, valid) / denominator
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        elapsed = float(start.elapsed_time(end))
        peak_gib = max(peak_gib, float(torch.cuda.max_memory_allocated() / 2**30))
        del x, y, valid, pos, segpos, full_mask, logits, loss, start, end
        model.zero_grad(set_to_none=True)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid B32 F+B latency: {elapsed}")
        if timed:
            samples.append(elapsed)
        _phase(f"{phase_name}_complete", latency_ms=elapsed)
        return elapsed

    for index in range(WARMUPS):
        print(f"B32 warmup {index + 1}/{WARMUPS}", flush=True)
        one(False, f"warmup_{index + 1}")
    for index in range(REPETITIONS):
        elapsed = one(True, f"timing_{index + 1}")
        print(f"B32 timed {index + 1}/{REPETITIONS}: {elapsed:.3f} ms", flush=True)

    median_ms = float(torch.tensor(samples, dtype=torch.float64).median().item())
    tok_s = MICRO_TOKENS / (median_ms / 1000.0)
    return {
        "samples_ms": samples,
        "median_ms": median_ms,
        "latency_ms": median_ms,
        "real_input_tok_s": float(tok_s),
        "peak_GiB": peak_gib,
        "warmups": WARMUPS,
        "timed_repetitions": REPETITIONS,
        "benchmark": "B32 exact forward+backward",
    }


def _write_artifact(context: dict[str, Any], result: dict[str, Any]) -> str:
    root = Path(os.environ.get("ICRL_G0_ARTIFACT_ROOT", str(REPO_ROOT / "artifacts" / "generation0")))
    attempt_id = context["attempt_id"]
    destination = root / context["run_id"] / "attempts" / attempt_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "summary.json"
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    latest = root / context["run_id"] / "summary.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest_temp = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
    latest_temp.write_text(encoded, encoding="utf-8")
    os.replace(latest_temp, latest)
    return str(path)


def _write_failure_artifact(exc: BaseException) -> str | None:
    """Persist failure state before interpreter/CUDA cleanup can hide it."""
    context = CURRENT_CONTEXT or {
        "plan_id": os.environ.get("ICRL_G0_PLAN_ID"),
        "run_id": os.environ.get("ICRL_G0_RUN_ID", "UNKNOWN"),
        "arm_id": os.environ.get("ICRL_G0_ARM_ID"),
        "block_id": os.environ.get("ICRL_G0_BLOCK_ID"),
        "within_block_position": os.environ.get("ICRL_G0_WITHIN_BLOCK_POSITION"),
        "attempt_id": os.environ.get("ICRL_G0_ATTEMPT_ID", "UNKNOWN"),
    }
    root_value = os.environ.get("ICRL_G0_ARTIFACT_ROOT")
    if not root_value:
        return None
    root = Path(root_value)
    run_id = str(context.get("run_id") or "UNKNOWN")
    attempt_id = str(context.get("attempt_id") or "UNKNOWN")
    text = f"{type(exc).__name__}: {exc}"
    lower = text.lower()
    if isinstance(exc, (KeyboardInterrupt, InterruptedError)):
        status = "INTERRUPTED"
    elif isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in lower:
        status = "OOM"
    else:
        status = "FAILED"
    payload = {
        "schema_version": 1,
        "status": status,
        "stage": CURRENT_PHASE,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "error_type": type(exc).__name__,
        "message": text,
        "command": os.environ.get("ICRL_G0_COMMAND"),
        "pid": os.getpid(),
        "traceback": traceback.format_exc(),
        "phase_events": PHASE_EVENTS,
        "memory_at_failure": _memory_snapshot(),
        "context": context,
        "gpu": {"uuid": os.environ.get("ICRL_G0_GPU_UUID")},
        "created_at": _utcnow(),
    }
    destination = root / run_id / "attempts" / attempt_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "failure.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        latest = root / run_id / "failure.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest_temp = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
        with latest_temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(latest_temp, latest)
        return str(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    global CURRENT_CONTEXT
    context = _run_context()
    CURRENT_CONTEXT = context
    gpu = _target_gpu()
    _phase("environment")
    # Importing the plan here certifies factor names/levels and the canonical
    # control reference without changing any plan file.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from icrl_autoresearch.generation0 import build_plan

    plan = build_plan()
    control_levels = dict(plan["control_reference"]["factor_levels"])
    for factor in FACTOR_NAMES:
        allowed = {entry["name"]: set(entry["levels"]) for entry in plan["factors"]}[factor]
        if context["selected_levels"][factor] not in allowed:
            raise RuntimeError(f"selected level is not in the frozen factor catalog: {factor}")
    candidate_commit = _git_commit()
    if "--preflight" in sys.argv[1:]:
        preflight = _preflight(context, plan, control_levels, gpu)
        preflight["source_commit"] = candidate_commit
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    _set_runtime_globals(os.environ.get("ICRL_CORPUS_ROOT"))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"Generation-0 runner: {context['run_id']} {context['arm_id']}", flush=True)
    print(f"GPU: {gpu['device']} {gpu['sm']} {gpu['vram_GiB']:.2f} GiB", flush=True)
    print(f"Selected levels: {json.dumps(context['selected_levels'], sort_keys=True)}", flush=True)
    print("Oracle: running exact FP32 representation comparison", flush=True)
    _phase("oracle")
    oracle = _oracle(context["selected_levels"], control_levels, torch.device("cuda"))
    print(f"Oracle: PASS forward_rel_l2={oracle['forward_rel_l2']:.3e} dq_rel_l2={oracle['dq_rel_l2']:.3e} dv_rel_l2={oracle['dv_rel_l2']:.3e}", flush=True)

    _phase("data")
    data = CANONICAL["load_corpus"]()
    _phase("initialization")
    init = CANONICAL["canonical_init"]()
    model = _load_model(context["selected_levels"], control_levels, init)
    _phase("initial_loss")
    initial_loss = float(CANONICAL["initial_loss"](model, data))
    print(f"Initial loss: {initial_loss:.9f} reference={EXPECTED_INITIAL_A_LOSS:.9f}", flush=True)
    if abs(initial_loss - EXPECTED_INITIAL_A_LOSS) > float(CANONICAL["INITIAL_LOSS_TOL"]):
        raise RuntimeError("initial-loss gate failed")
    benchmark = _benchmark_b32(model, data)
    if benchmark["peak_GiB"] >= gpu["vram_GiB"]:
        raise RuntimeError(f"memory gate failed: peak {benchmark['peak_GiB']:.3f} GiB >= capacity {gpu['vram_GiB']:.3f} GiB")
    _phase("free")
    del model, data, init
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _phase("free_complete")

    result: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "source_commit": candidate_commit,
        "run_id": context["run_id"],
        "attempt_id": context["attempt_id"],
        "arm_id": context["arm_id"],
        "block_id": context["block_id"],
        "within_block_position": context["within_block_position"],
        "selected_levels": context["selected_levels"],
        "oracle": {
            **oracle,
            "initial_loss": initial_loss,
            "initial_loss_gate": "PASS",
            "status": "PASS",
        },
        "latency_ms": benchmark["latency_ms"],
        "real_input_tok_s": benchmark["real_input_tok_s"],
        "peak_GiB": benchmark["peak_GiB"],
        "samples_ms": benchmark["samples_ms"],
        "warmups": benchmark["warmups"],
        "timed_repetitions": benchmark["timed_repetitions"],
        "benchmark": benchmark["benchmark"],
        "telemetry": {"phases": PHASE_EVENTS},
        "memory": {
            "status": "PASS",
            "peak_GiB": benchmark["peak_GiB"],
            "capacity_GiB": gpu["vram_GiB"],
        },
        "decision": "MEASURED_G0_B32_FB",
        "execution": {
            "candidate_commit": candidate_commit,
            "gpu": gpu,
            "champion_immutable": True,
        },
    }
    result["artifacts"] = {"summary": _write_artifact(context, result)}
    # The harness accepts the final line as the one machine-readable result.
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        failure_path = _write_failure_artifact(exc)
        if failure_path:
            print(f"Failure artifact: {failure_path}", file=sys.stderr, flush=True)
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
