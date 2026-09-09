# ICRL TEMPORAL LOOM — PAPER-NATIVE READ PATH GATE v1
#
# One whole self-contained Colab cell/script.
#
# QUESTION
# --------
# Can the exact BDH read path become materially faster if we:
#   1) produce neuron activations in physical [B,T,H,K] layout with one wide Dx GEMM;
#   2) execute ONLY the strict-causal/direct prefix blocks that mathematically exist,
#      instead of a full T x T Gram followed by masking;
#   3) keep the frozen Stage-1 science and the rest of Arm-A unchanged?
#
# THIS IS AN EXECUTION-REPRESENTATION EXPERIMENT, NOT A MODEL CHANGE.
#
# Control read path:
#   x[B,H,T,K] = relu(v.unsqueeze(1) @ decoder_x)
#   q = RoPE(x)
#   S = q @ q^T                       # full T x T
#   S = strict_same_doc_lower(S)
#   a = S @ v
#
# Candidate read path:
#   x[B,T,H,K] = relu(v[BT,D] @ decoder_x[D,H*K])
#   q = RoPE(x)
#   for query block c:
#       S_c = q_c @ q_prefix^T        # prefix only; no future blocks computed
#       S_c = strict_same_doc_lower(S_c)
#       a_c = S_c @ v_prefix
#
# Important: v1 deliberately stays compiler-visible PyTorch. It does NOT yet fuse the
# score tile into the score@V consumer. It removes the FULL Gram materialization and
# future-triangle arithmetic, while allowing transient prefix score tiles. If this gate
# wins, the next causal step is to kernelize the same exact blocked dataflow so each
# score tile stays on chip.
#
# Frozen source/model invariants:
#   BDH paper arXiv:2509.26507 Appendix E / Eq. 4 / Eq. 8
#   pinned pathwaycom/bdh commit 2b0d7a45b058d4309c84a10e0768d541fe18bdc2
#   d=256, n=32768, h=4, K/head=8192, L=6, Q=K, strict lower, no softmax/scale
#   Stage-1 Arm-A coordinator + dense writer + exact document reset + doc-local RoPE
#
# Parent baseline:
#   experiments/bdh_reintegrated_baseline_v1/<latest>/report.json
#   fallback accepted SAC median: 3292.4 ms/B64
#
# Predeclared decisions:
#   read micro F+B speedup < 1.10x  -> REVERT; do not spend full-model minutes
#   full B64 speedup < 1.05x        -> REVERT
#   full B64 speedup 1.05-1.10x     -> MARGINAL; profile before more engineering
#   full B64 speedup >= 1.10x       -> KEEP READ REPRESENTATION; build fused WRITE next
#   full B64 <= baseline/2          -> 2x gate passed early

import gc
import json
import math
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

# ======================================================================================
# 0. FROZEN CONTRACT + EXPERIMENT GATES
# ======================================================================================

SEED = 1337

L = 6
D = 256
H = 4
K = 8192
N = H * K
THETA = 2**16
INIT_STD = 0.02

T = 2048
VOCAB = 8192
HIDDEN = 1040
B = 32
GLOBAL_B = 64

# One representation parameter only. This is NOT a sweep.
READ_BLOCK = 256
assert T % READ_BLOCK == 0

# Cheap read-path discovery first; full B64 only if it passes.
READ_WARMUP = 2
READ_REPS = 5
READ_PROMOTE_SPEEDUP = 1.10

# Full-model validation cadence. First two include compile / settling and are excluded.
UPDATES = 6
STEADY_SKIP = 2
PROFILE = True
PROFILE_ROW_LIMIT = 60

# Exact/algebraic correctness gates.
FP32_REL_L2_MAX = 3e-4
FP32_MAX_ABS_MAX = 3e-3
INITIAL_LOSS_TOL = 1e-2

# Full-model KEEP / REVERT gates.
FULL_REVERT_BELOW = 1.05
FULL_KEEP_AT = 1.10

PEAK_LR = 1e-3
WARMUP_TOKENS = 10_000_000
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0

EXPECTED_PARAMS = 29_958_401
EXPECTED_INITIAL_A_LOSS = 9.077229499816895

ROOT = Path('/content/drive/Shareddrives/ICLR PHASE BDH/phase_bdh')
CORPUS = ROOT / 'corpus/stage2/frozen_5b_v1'
CORPUS_ID = 'phase_bdh_stage2_5b_v1'
TOKENIZER_SHA = '9a05bca4c065d9952a01ee1e3f4b6e23e39200e1f33da65e5dad1819419f71c3'
REPLAY_SHA = 'c60bd7df4a8ef7329d20ec5b0f022f4579fca20c95657a90f79a2e9e67edb043'

OFFICIAL_BDH_COMMIT = '2b0d7a45b058d4309c84a10e0768d541fe18bdc2'
OFFICIAL_BDH_SHA256 = 'cfe24008f920965cc3c8236feff52c89ca794a31e52324acf9ddb4cd6fd50ac9'
OFFICIAL_BDH_BLOB_SHA1 = '4cfff79df97fa6faafb33bf20739a027edd8f360'

HISTORICAL_ACCEPTED_MS = 3292.4
MICRO_TOKENS = B * T
GLOBAL_TOKENS = GLOBAL_B * T
assert MICRO_TOKENS == 65_536
assert GLOBAL_TOKENS == 131_072
assert N == 32768

print('=' * 120)
print('ICRL TEMPORAL LOOM — PAPER-NATIVE READ PATH GATE v1')
print('ONE CAUSAL CHANGE: physical neuron/read representation; frozen model science.')
print(f'Candidate block: {READ_BLOCK} | real shape B{B} T{T} H{H} K{K} D{D}')
print('=' * 120)

# ======================================================================================
# 1. RUNTIME / DRIVE / PARENT BASELINE
# ======================================================================================

def reset_rng():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def gpu_clean(reset_compiler=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if reset_compiler:
        try:
            torch.compiler.reset()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if not torch.cuda.is_available():
    raise RuntimeError('A fresh Colab CUDA GPU runtime is required')

from google.colab import drive
print('\nRequesting Google Drive mount...')
drive.mount('/content/drive', force_remount=False)

if not CORPUS.exists():
    raise RuntimeError(f'Frozen corpus not found: {CORPUS}')

DEVICE = torch.device('cuda')
PROP = torch.cuda.get_device_properties(0)
if (PROP.major, PROP.minor) != (12, 0):
    raise RuntimeError(f'This gate was designed for sm_120; got sm_{PROP.major}{PROP.minor}')

reset_rng()
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

print(f'GPU      : {PROP.name}')
print(f'SM       : sm_{PROP.major}{PROP.minor}')
print(f'VRAM     : {PROP.total_memory / 2**30:.2f} GiB')
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.version.cuda}')


def latest_parent_baseline():
    base = ROOT / 'experiments/bdh_reintegrated_baseline_v1'
    if base.exists():
        reports = sorted(base.glob('*/report.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in reports:
            try:
                r = json.loads(p.read_text())
                b = r.get('baseline', {})
                ms = float(b.get('median_B64_ms'))
                if r.get('status') == 'BASELINE_REESTABLISHED' and math.isfinite(ms) and ms > 0:
                    return ms, str(p), r
            except Exception:
                pass
    return HISTORICAL_ACCEPTED_MS, 'historical accepted Gram-only SAC control', None


CONTROL_MS, CONTROL_SOURCE, PARENT_REPORT = latest_parent_baseline()
TARGET_2X_MS = CONTROL_MS / 2.0
TARGET_4X_MS = CONTROL_MS / 4.0

print(f'Parent B64 control : {CONTROL_MS:,.3f} ms')
print(f'Parent source      : {CONTROL_SOURCE}')
print(f'2x target          : {TARGET_2X_MS:,.3f} ms')
print(f'4x target          : {TARGET_4X_MS:,.3f} ms')

OUTPUT = (
    ROOT / 'experiments/bdh_native_read_v1'
    / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
)
OUTPUT.mkdir(parents=True, exist_ok=False)

REPORT = {
    'status': 'RUNNING',
    'question': 'Does exact [B,T,H,K] wide-Dx + direct causal prefix-block attention materially beat the current read representation?',
    'source': {
        'paper': 'arXiv:2509.26507 Appendix E / Eq. 4 / Eq. 8',
        'official_commit': OFFICIAL_BDH_COMMIT,
        'official_bdh_sha256': OFFICIAL_BDH_SHA256,
        'official_bdh_blob_sha1': OFFICIAL_BDH_BLOB_SHA1,
    },
    'science': {
        'model_changed': False,
        'stage1_arm': 'A',
        'Q_is_K': True,
        'strict_lower': True,
        'softmax': False,
        'attention_scale': False,
        'document_reset': 'same-doc exact mask from frozen provenance',
        'rope': 'doc-local; theta=2^16; q=2 pair frequencies',
    },
    'candidate': {
        'physical_x_layout': '[B,T,H,K]',
        'Dx': '[BT,D] @ [D,H*K] from an exact view/copy of the same decoder_x parameter',
        'attention': 'direct causal prefix blocks; transient score blocks only; no full Gram',
        'block': READ_BLOCK,
        'write_path': 'unchanged frozen baseline',
        'checkpoint': 'selective checkpointing saves only candidate QK block matmuls',
    },
    'gates': {
        'fp32_rel_l2_max': FP32_REL_L2_MAX,
        'fp32_max_abs_max': FP32_MAX_ABS_MAX,
        'read_promote_speedup': READ_PROMOTE_SPEEDUP,
        'full_revert_below': FULL_REVERT_BELOW,
        'full_keep_at': FULL_KEEP_AT,
    },
    'parent_control_ms': CONTROL_MS,
    'parent_control_source': CONTROL_SOURCE,
    'target_2x_ms': TARGET_2X_MS,
    'target_4x_ms': TARGET_4X_MS,
    'gpu': PROP.name,
    'sm': f'sm_{PROP.major}{PROP.minor}',
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'output': str(OUTPUT),
}


def save_report():
    (OUTPUT / 'report.json').write_text(json.dumps(REPORT, indent=2))

# ======================================================================================
# 2. FROZEN CORPUS WINDOW
# ======================================================================================

def load_corpus():
    print('\n[1/7] Loading frozen corpus window...')
    try:
        import pyarrow.parquet as pq
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyarrow'])
        import pyarrow.parquet as pq

    frozen = json.loads((CORPUS / 'FROZEN.json').read_text())
    assert frozen['status'] == 'FROZEN'
    assert frozen['corpus_id'] == CORPUS_ID
    assert frozen['context_length'] == T
    assert frozen['tokenizer_sha256'] == TOKENIZER_SHA
    assert frozen['logical_replay_sha256'] == REPLAY_SHA

    train = CORPUS / 'train'
    tokens_path = train / 'shard_000000.tokens.bin'
    lengths_path = train / 'shard_000000.valid_lengths.bin'
    prov_path = train / 'shard_000000.provenance.parquet'
    for p in (tokens_path, lengths_path, prov_path):
        if not p.exists():
            raise RuntimeError(f'Missing corpus artifact: {p}')

    input_rows = max(UPDATES * GLOBAL_B, B)
    needed_rows = input_rows + 1

    lens = np.memmap(lengths_path, mode='r', dtype=np.uint16)
    if len(lens) < needed_rows:
        raise RuntimeError('Not enough frozen rows for benchmark')
    if not np.all(np.asarray(lens[:needed_rows]) == T):
        raise RuntimeError('Unexpected partial row in benchmark window')

    tok = np.memmap(tokens_path, mode='r', dtype=np.uint16, shape=(len(lens), T))
    tokens = np.asarray(tok[:needed_rows], dtype=np.uint16).copy()
    if int(tokens.max()) >= VOCAB:
        raise RuntimeError(f'Token out of vocabulary: {int(tokens.max())}')

    table = pq.read_table(
        prov_path,
        columns=[
            'sequence_index', 'sequence_token_start', 'sequence_token_end',
            'selected_document_index', 'document_token_start', 'document_token_end',
        ],
        filters=[('sequence_index', '<', needed_rows)],
    )
    pd = table.to_pydict()

    docs = np.full((needed_rows, T), -1, dtype=np.int32)
    pos = np.full((needed_rows, T), -1, dtype=np.int32)
    for i in range(table.num_rows):
        s = int(pd['sequence_index'][i])
        a = int(pd['sequence_token_start'][i])
        e = int(pd['sequence_token_end'][i])
        doc = int(pd['selected_document_index'][i])
        da = int(pd['document_token_start'][i])
        de = int(pd['document_token_end'][i])
        if e - a != de - da:
            raise RuntimeError('Provenance span mismatch')
        docs[s, a:e] = doc
        pos[s, a:e] = np.arange(da, de, dtype=np.int32)

    if np.any(docs < 0) or np.any(pos < 0):
        raise RuntimeError('Provenance coverage failed')

    x = tokens[:input_rows]
    y = np.empty_like(x)
    y[:, :-1] = tokens[:input_rows, 1:]
    y[:, -1] = tokens[1:input_rows + 1, 0]

    target_docs = np.empty_like(docs[:input_rows])
    target_docs[:, :-1] = docs[:input_rows, 1:]
    target_docs[:, -1] = docs[1:input_rows + 1, 0]
    valid = docs[:input_rows] == target_docs

    start = np.empty((input_rows, T), dtype=np.int16)
    segpos = np.empty((input_rows, T), dtype=np.int16)
    for r in range(input_rows):
        dr = docs[r]
        starts = np.concatenate((np.array([0], dtype=np.int32), np.flatnonzero(dr[1:] != dr[:-1]).astype(np.int32) + 1))
        ends = np.concatenate((starts[1:], np.array([T], dtype=np.int32)))
        for a, e in zip(starts.tolist(), ends.tolist()):
            start[r, a:e] = a
            segpos[r, a:e] = np.arange(e - a, dtype=np.int16)

    if int(valid[:32].sum()) != 65_498:
        raise RuntimeError('micro0 valid-target regression failed')
    if int(valid[32:64].sum()) != 65_496:
        raise RuntimeError('micro1 valid-target regression failed')
    if int(valid[:64].sum()) != 130_994:
        raise RuntimeError('B64 valid-target regression failed')

    def pin(a):
        return torch.from_numpy(np.ascontiguousarray(a)).pin_memory()

    data = {
        'x': pin(x), 'y': pin(y), 'valid': pin(valid),
        'pos': pin(pos[:input_rows]), 'start': pin(start), 'segpos': pin(segpos),
        'rows': input_rows,
    }

    del lens, tok, tokens, table, pd, docs, pos, x, y, target_docs, valid, start, segpos
    gc.collect()
    print(f'Rows       : {input_rows:,} + 1 lookahead')
    print(f'Candidate  : {UPDATES * GLOBAL_TOKENS:,} real input tokens max')
    print('CORPUS     : PASS')
    return data

# ======================================================================================
# 3. CANONICAL INITIALIZATION
# ======================================================================================

def canonical_init():
    g = torch.Generator(device='cpu')
    g.manual_seed(SEED)
    def rnd(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32) * INIT_STD
    return {
        'embedding': rnd((VOCAB, D)),
        'encoder': rnd((N, D)),
        'decoder_x': rnd((H, D, K)),
        'decoder_y': rnd((H, D, K)),
        'readout': rnd((D, VOCAB)),
        'coord_Wc': rnd((D, D)),
        'coord_bc': torch.zeros(D, dtype=torch.float32),
        'coord_alpha': torch.zeros((), dtype=torch.float32),
        'writer_W1': rnd((D, HIDDEN)),
        'writer_W2': rnd((HIDDEN, D)),
    }

# ======================================================================================
# 4. EXACT READ REPRESENTATIONS — GENERIC ORACLE + REAL PRODUCTION PATH
# ======================================================================================

CAUSAL = torch.ones((T, T), device=DEVICE, dtype=torch.bool).tril(diagonal=-1)
ROPE_PAIR_FREQ = (
    1.0 / (THETA ** ((2.0 * torch.arange(K // 2, dtype=torch.float32, device=DEVICE)) / K))
    / (2.0 * math.pi)
)


def pair_freq(k, device):
    return (
        1.0 / (THETA ** ((2.0 * torch.arange(k // 2, dtype=torch.float32, device=device)) / k))
        / (2.0 * math.pi)
    )


def rope_bhtk(q, pos, freq):
    # q [B,H,T,K]
    b, h, t, k = q.shape
    qp = q.reshape(b, h, t, k // 2, 2)
    phase = pos.float().unsqueeze(1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    cs = torch.cos(phase).to(q.dtype)
    sn = torch.sin(phase).to(q.dtype)
    qe, qo = qp[..., 0], qp[..., 1]
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn), dim=-1).reshape_as(q)


def rope_bthk(q, pos, freq):
    # q [B,T,H,K] — same pairwise rotation, different physical layout.
    b, t, h, k = q.shape
    qp = q.reshape(b, t, h, k // 2, 2)
    phase = pos.float().unsqueeze(-1).unsqueeze(-1) * freq.view(1, 1, 1, -1)
    phase = torch.remainder(phase, 1.0) * (2.0 * math.pi)
    cs = torch.cos(phase).to(q.dtype)
    sn = torch.sin(phase).to(q.dtype)
    qe, qo = qp[..., 0], qp[..., 1]
    return torch.stack((qe * cs - qo * sn, qo * cs + qe * sn), dim=-1).reshape_as(q)


def reference_read(v, decoder_x, pos, full_mask, freq):
    # v [B,T,D], decoder_x [H,D,K]
    x = F.relu(v.unsqueeze(1) @ decoder_x)                 # [B,H,T,K]
    q = rope_bhtk(x, pos, freq)
    scores = q @ q.transpose(-1, -2)                      # full Gram control
    scores = scores.masked_fill(~full_mask.unsqueeze(1), 0.0)
    vh = v.unsqueeze(1).expand(-1, decoder_x.shape[0], -1, -1).contiguous()
    return scores @ vh


def native_read(v, decoder_x, pos, full_mask, freq, block):
    # Exact same decoder_x parameter, flattened only as an execution view.
    b, t, d = v.shape
    h, dw, k = decoder_x.shape
    assert d == dw
    assert t % block == 0

    # Column order is (h,k), therefore the contiguous output is exactly [B,T,H,K].
    w_wide = decoder_x.permute(1, 0, 2).reshape(d, h * k)
    x_bt = F.relu((v.reshape(b * t, d) @ w_wide).reshape(b, t, h, k))
    q_bt = rope_bthk(x_bt, pos, freq)

    # Logical [B,H,T,K] view. We intentionally do not call contiguous() here.
    qh = q_bt.permute(0, 2, 1, 3)
    vh = v.unsqueeze(1).expand(-1, h, -1, -1).contiguous()

    out = []
    for s0 in range(0, t, block):
        s1 = s0 + block
        qb = qh[:, :, s0:s1, :]
        kp = qh[:, :, :s1, :]
        scores = qb @ kp.transpose(-1, -2)
        mask = full_mask[:, None, s0:s1, :s1]
        scores = scores.masked_fill(~mask, 0.0)
        out.append(scores @ vh[:, :, :s1, :])

    return x_bt, torch.cat(out, dim=2)


def native_read_output(v, decoder_x, pos, full_mask, freq, block):
    return native_read(v, decoder_x, pos, full_mask, freq, block)[1]


def rel_l2(a, b):
    af, bf = a.float(), b.float()
    return float((af - bf).norm() / bf.norm().clamp_min(1e-30))


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max())


def fp32_oracle_gate():
    print('\n[2/7] FP32 algebra / gradient / document-boundary gate...')
    reset_rng()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    bt, tt, ht, dt, kt, block = 2, 96, 4, 64, 512, 24
    freq = pair_freq(kt, DEVICE)

    # Two/three segments per row so same-doc reset is part of the oracle.
    start = torch.zeros((bt, tt), device=DEVICE, dtype=torch.int32)
    start[0, 31:67] = 31
    start[0, 67:] = 67
    start[1, 20:55] = 20
    start[1, 55:] = 55
    causal = torch.ones((tt, tt), device=DEVICE, dtype=torch.bool).tril(-1)
    full_mask = (start[:, :, None] == start[:, None, :]) & causal[None]

    pos = torch.empty((bt, tt), device=DEVICE, dtype=torch.int32)
    for r in range(bt):
        for s0 in torch.unique(start[r]).tolist():
            idx = torch.nonzero(start[r] == int(s0), as_tuple=False).squeeze(-1)
            pos[r, idx] = torch.arange(idx.numel(), device=DEVICE, dtype=torch.int32) + 7

    v = torch.randn((bt, tt, dt), device=DEVICE, dtype=torch.float32, requires_grad=True) * 0.03
    w = torch.randn((ht, dt, kt), device=DEVICE, dtype=torch.float32, requires_grad=True) * 0.03
    g = torch.randn((bt, ht, tt, dt), device=DEVICE, dtype=torch.float32)

    ref = reference_read(v, w, pos, full_mask, freq)
    rv, rw = torch.autograd.grad(ref, (v, w), g, retain_graph=False)

    v2 = v.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    _, cand = native_read(v2, w2, pos, full_mask, freq, block)
    cv, cw = torch.autograd.grad(cand, (v2, w2), g, retain_graph=False)

    checks = {
        'forward_rel_l2': rel_l2(cand, ref),
        'forward_max_abs': max_abs(cand, ref),
        'dV_rel_l2': rel_l2(cv, rv),
        'dV_max_abs': max_abs(cv, rv),
        'dDx_rel_l2': rel_l2(cw, rw),
        'dDx_max_abs': max_abs(cw, rw),
    }
    checks['pass'] = (
        checks['forward_rel_l2'] <= FP32_REL_L2_MAX
        and checks['dV_rel_l2'] <= FP32_REL_L2_MAX
        and checks['dDx_rel_l2'] <= FP32_REL_L2_MAX
        and checks['forward_max_abs'] <= FP32_MAX_ABS_MAX
    )

    print(json.dumps(checks, indent=2))
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.set_float32_matmul_precision(old_prec)
    del v, w, g, ref, rv, rw, v2, w2, cand, cv, cw, pos, start, full_mask, causal, freq
    gpu_clean()

    if not checks['pass']:
        raise RuntimeError('FP32 oracle gate FAILED — candidate is not allowed to benchmark')
    return checks

# ======================================================================================
# 5. DATA HELPERS + REAL B32 READ MICROBENCH
# ======================================================================================

def micro_to_gpu(data, row0, include_targets=True):
    sl = slice(row0, row0 + B)
    x = data['x'][sl].to(DEVICE, dtype=torch.long, non_blocking=True)
    pos = data['pos'][sl].to(DEVICE, dtype=torch.int32, non_blocking=True)
    start = data['start'][sl].to(DEVICE, dtype=torch.int32, non_blocking=True)
    segpos = data['segpos'][sl].to(DEVICE, dtype=torch.int32, non_blocking=True)
    full_mask = (start[:, :, None] == start[:, None, :]) & CAUSAL.unsqueeze(0)

    if not include_targets:
        return x, pos, segpos, full_mask

    y = data['y'][sl].to(DEVICE, dtype=torch.long, non_blocking=True)
    valid = data['valid'][sl].to(DEVICE, non_blocking=True)
    return x, y, valid, pos, segpos, full_mask


def make_real_v_and_dx(data, init):
    idx, pos, segpos, full_mask = micro_to_gpu(data, 0, include_targets=False)
    emb = init['embedding'].to(DEVICE)
    with torch.no_grad():
        v = F.layer_norm(F.embedding(idx, emb), (D,))
    dx = init['decoder_x'].to(DEVICE)
    del idx, segpos, emb
    return v.contiguous(), dx.contiguous(), pos, full_mask


def _compile(fn):
    return torch.compile(fn, dynamic=False, fullgraph=True)


def bench_read(name, fn, v0, w0, pos, full_mask):
    print(f'\nREAD BENCH: {name}', flush=True)
    gpu_clean(reset_compiler=True)

    v = v0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    execute = _compile(fn)

    fwds, bwds, totals = [], [], []
    peak = 0.0

    def one(timed):
        nonlocal peak
        v.grad = None
        w.grad = None
        torch.cuda.reset_peak_memory_stats()
        e0, e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            out = execute(v, w, pos, full_mask)
        e1.record()
        loss = out.float().square().mean()
        loss.backward()
        e2.record()
        torch.cuda.synchronize()
        f = e0.elapsed_time(e1)
        b = e1.elapsed_time(e2)
        p = torch.cuda.max_memory_allocated() / 2**30
        peak = max(peak, p)
        del out, loss, e0, e1, e2
        if timed:
            fwds.append(float(f)); bwds.append(float(b)); totals.append(float(f + b))
            print(f'  rep {len(totals)}/{READ_REPS}: fwd={f:.2f} bwd={b:.2f} F+B={f+b:.2f} ms peak={p:.2f}GiB', flush=True)
        else:
            print(f'  warmup: {f+b:.2f} ms', flush=True)

    try:
        for _ in range(READ_WARMUP):
            one(False)
        for _ in range(READ_REPS):
            one(True)
        result = {
            'status': 'PASS',
            'fwd_ms': float(np.median(fwds)),
            'bwd_ms': float(np.median(bwds)),
            'fb_ms': float(np.median(totals)),
            'peak_GiB': peak,
        }
    except Exception as exc:
        result = {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        execute = None
        v = None
        w = None
        gpu_clean(reset_compiler=True)

    print(json.dumps(result, indent=2))
    return result


def real_read_gate(data, init):
    print('\n[3/7] Real B32 read-path F+B gate...')
    v0, w0, pos, full_mask = make_real_v_and_dx(data, init)

    def control(v, w, p, m):
        return reference_read(v, w, p, m, ROPE_PAIR_FREQ)

    def candidate(v, w, p, m):
        return native_read_output(v, w, p, m, ROPE_PAIR_FREQ, READ_BLOCK)

    control_r = bench_read('CONTROL full Gram + baseline projection', control, v0, w0, pos, full_mask)
    candidate_r = bench_read('CANDIDATE [B,T,H,K] wide Dx + causal prefix blocks', candidate, v0, w0, pos, full_mask)

    del v0, w0, pos, full_mask
    gpu_clean(reset_compiler=True)

    if control_r['status'] != 'PASS' or candidate_r['status'] != 'PASS':
        return {'status': 'FAIL', 'control': control_r, 'candidate': candidate_r, 'decision': 'REVERT_READ_REPRESENTATION'}

    speedup = control_r['fb_ms'] / candidate_r['fb_ms']
    decision = 'PROMOTE_TO_FULL_MODEL' if speedup >= READ_PROMOTE_SPEEDUP else 'REVERT_READ_REPRESENTATION'
    return {
        'status': 'PASS',
        'control': control_r,
        'candidate': candidate_r,
        'speedup': speedup,
        'decision': decision,
    }

# ======================================================================================
# 6. FULL FROZEN ARM-A WITH ONLY THE READ PATH REPLACED
# ======================================================================================

class DenseWriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(D, HIDDEN))
        self.W2 = nn.Parameter(torch.empty(HIDDEN, D))
    def forward(self, x):
        return F.relu(x @ self.W1) @ self.W2


class Coordinator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wc = nn.Parameter(torch.empty(D, D))
        self.bc = nn.Parameter(torch.zeros(D))
        self.alpha = nn.Parameter(torch.zeros(()))
    def forward(self, v, segpos, full_mask):
        z = v @ self.Wc + self.bc
        prev_sum = torch.bmm(full_mask.to(dtype=z.dtype), z)
        den = segpos.clamp_min(1).to(z.dtype).unsqueeze(-1)
        c = prev_sum / den - z
        rho = torch.sigmoid(self.alpha)
        return 1.0 + rho.to(c.dtype) * torch.tanh(c)


_BMM = torch.ops.aten.bmm.default
_MATMUL = torch.ops.aten.matmul.default


def native_read_sac_policy(ctx, op, *args, **kwargs):
    # Save ONLY QK score blocks, exactly analogous to the accepted full-Gram SAC policy.
    if op in (_BMM, _MATMUL) and len(args) >= 2:
        a, b = args[0], args[1]
        if hasattr(a, 'shape') and hasattr(b, 'shape') and len(a.shape) >= 3 and len(b.shape) >= 3:
            try:
                if (
                    int(a.shape[-1]) == K
                    and int(b.shape[-2]) == K
                    and int(a.shape[-2]) <= READ_BLOCK
                    and int(b.shape[-1]) <= T
                ):
                    return CheckpointPolicy.MUST_SAVE
            except Exception:
                pass
    return CheckpointPolicy.PREFER_RECOMPUTE


NATIVE_SAC_CONTEXT_FN = partial(create_selective_checkpoint_contexts, native_read_sac_policy)


class NativeReadStage1ArmA(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D)
        self.encoder = nn.Parameter(torch.empty(N, D))
        self.decoder_x = nn.Parameter(torch.empty(H, D, K))
        self.decoder_y = nn.Parameter(torch.empty(H, D, K))
        self.readout = nn.Parameter(torch.empty(D, VOCAB))
        self.coordinator = Coordinator()
        self.writer = DenseWriter()
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)

    def project_x_native(self, v):
        w_wide = self.decoder_x.permute(1, 0, 2).reshape(D, N)
        return F.relu((v.reshape(v.shape[0] * T, D) @ w_wide).reshape(v.shape[0], T, H, K))

    def attention_native(self, x_bt, v, pos, full_mask):
        q_bt = rope_bthk(x_bt, pos, ROPE_PAIR_FREQ)
        qh = q_bt.permute(0, 2, 1, 3)
        vh = v.unsqueeze(1).expand(-1, H, -1, -1).contiguous()
        outs = []
        for s0 in range(0, T, READ_BLOCK):
            s1 = s0 + READ_BLOCK
            qb = qh[:, :, s0:s1, :]
            kp = qh[:, :, :s1, :]
            scores = qb @ kp.transpose(-1, -2)
            scores = scores.masked_fill(~full_mask[:, None, s0:s1, :s1], 0.0)
            outs.append(scores @ vh[:, :, :s1, :])
        return torch.cat(outs, dim=2)

    def level(self, v, pos, segpos, full_mask):
        # ---------------- EXACT BDH PAPER CORE ----------------
        x_bt = self.project_x_native(v)                       # physical [B,T,H,K]
        a = self.ln(self.attention_native(x_bt, v, pos, full_mask))
        ypre = F.relu(a @ self.decoder_y)                     # unchanged [B,H,T,K]

        # Only a view back to the baseline logical orientation; write path remains unchanged.
        paper_y = x_bt.permute(0, 2, 1, 3) * ypre
        paper_y_flat = paper_y.transpose(1, 2).reshape(v.shape[0], T, N)
        base = self.ln(paper_y_flat @ self.encoder)

        # ---------------- FROZEN STAGE-1 EXTENSION ------------
        g = self.coordinator(v, segpos, full_mask)
        delta = self.writer(g * base)
        return self.ln(v + delta)

    def forward(self, idx, pos, segpos, full_mask):
        v = self.ln(self.embedding(idx))
        for _ in range(L):
            def level_fn(vv):
                return self.level(vv, pos, segpos, full_mask)
            v = checkpoint(
                level_fn, v,
                use_reentrant=False,
                preserve_rng_state=False,
                context_fn=NATIVE_SAC_CONTEXT_FN,
            )
        return v @ self.readout

    @torch.no_grad()
    def forward_eval(self, idx, pos, segpos, full_mask):
        v = self.ln(self.embedding(idx))
        for _ in range(L):
            v = self.level(v, pos, segpos, full_mask)
        return v @ self.readout


def load_init(model, init):
    with torch.no_grad():
        model.embedding.weight.copy_(init['embedding'].to(DEVICE))
        model.encoder.copy_(init['encoder'].to(DEVICE))
        model.decoder_x.copy_(init['decoder_x'].to(DEVICE))
        model.decoder_y.copy_(init['decoder_y'].to(DEVICE))
        model.readout.copy_(init['readout'].to(DEVICE))
        model.coordinator.Wc.copy_(init['coord_Wc'].to(DEVICE))
        model.coordinator.bc.copy_(init['coord_bc'].to(DEVICE))
        model.coordinator.alpha.copy_(init['coord_alpha'].to(DEVICE))
        model.writer.W1.copy_(init['writer_W1'].to(DEVICE))
        model.writer.W2.copy_(init['writer_W2'].to(DEVICE))


def ce_sum(logits, target, valid):
    per = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1), reduction='none')
    return per[valid.reshape(-1)].sum(dtype=torch.float32)


def valid_count_cpu(data, row0):
    return int(data['valid'][row0:row0 + B].sum().item())


def make_optimizer(model):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {'params': decay, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    try:
        return torch.optim.AdamW(groups, lr=PEAK_LR, betas=BETAS, eps=EPS, fused=True), 'AdamW(fused=True)'
    except Exception:
        return torch.optim.AdamW(groups, lr=PEAK_LR, betas=BETAS, eps=EPS), 'AdamW'


def lr_for_update(update_1based):
    return PEAK_LR * min((update_1based * GLOBAL_TOKENS) / WARMUP_TOKENS, 1.0)


def initial_loss(model, data):
    model.eval()
    x, y, valid, pos, segpos, full_mask = micro_to_gpu(data, 0, include_targets=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        logits = model.forward_eval(x, pos, segpos, full_mask)
        ls = ce_sum(logits, y, valid)
        nv = int(valid.sum().item())
        loss = float((ls / nv).item())
    model.train()
    del x, y, valid, pos, segpos, full_mask, logits, ls
    torch.cuda.synchronize()
    return loss


def profile_one_micro(execute, model, data):
    from torch.profiler import ProfilerActivity, profile
    print('\n[6/7] Profiling one separate candidate B32 F+B...')
    model.zero_grad(set_to_none=True)
    x, y, valid, pos, segpos, full_mask = micro_to_gpu(data, 0, include_targets=True)
    nv = int(valid.sum().item())
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=False,
    ) as prof:
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            logits = execute(x, pos, segpos, full_mask)
            loss = ce_sum(logits, y, valid) / nv
        loss.backward()
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=PROFILE_ROW_LIMIT)
    path = OUTPUT / 'candidate_profile.txt'
    path.write_text(table)
    model.zero_grad(set_to_none=True)
    del x, y, valid, pos, segpos, full_mask, logits, loss, prof
    gpu_clean()
    print(f'Profile: {path}')
    return str(path)


def benchmark_full_candidate(data, init):
    print('\n[5/7] Full frozen B64 Arm-A validation with native READ only...')
    reset_rng()
    gpu_clean(reset_compiler=True)

    model = NativeReadStage1ArmA().to(DEVICE)
    load_init(model, init)
    params = sum(p.numel() for p in model.parameters())
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f'parameter-count gate failed: {params:,} != {EXPECTED_PARAMS:,}')

    iloss = initial_loss(model, data)
    print(f'Parameters  : {params:,}')
    print(f'Initial loss: {iloss:.9f} (reference {EXPECTED_INITIAL_A_LOSS:.9f})')
    if abs(iloss - EXPECTED_INITIAL_A_LOSS) > INITIAL_LOSS_TOL:
        raise RuntimeError('initial-loss gate failed')

    execute = torch.compile(model)
    optimizer, optimizer_name = make_optimizer(model)
    print('Compile     : torch.compile() defaults')
    print('Checkpoint  : native QK-block SAC')
    print(f'Optimizer   : {optimizer_name}')

    step_ms, losses = [], []
    peak = 0.0

    for u in range(UPDATES):
        row0 = u * GLOBAL_B
        n0 = valid_count_cpu(data, row0)
        n1 = valid_count_cpu(data, row0 + B)
        denom = n0 + n1
        if denom <= 0:
            raise RuntimeError('empty valid-target update')

        lr = lr_for_update(u + 1)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_sums = []

        for rr in (row0, row0 + B):
            x, y, valid, pos, segpos, full_mask = micro_to_gpu(data, rr, include_targets=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                logits = execute(x, pos, segpos, full_mask)
                ls = ce_sum(logits, y, valid)
                loss = ls / denom
            loss.backward()
            loss_sums.append(ls.detach())
            del x, y, valid, pos, segpos, full_mask, logits, ls, loss

        grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        step_ms.append(ms)
        train_loss = sum(float(v.item()) for v in loss_sums) / denom
        losses.append(train_loss)

        steady = step_ms[STEADY_SKIP:]
        med = float(np.median(steady)) if steady else float('nan')
        tps = GLOBAL_TOKENS / (ms / 1000.0)
        print(
            f'upd={u+1:02d}/{UPDATES} loss={train_loss:.5f} lr={lr:.3e} '
            f'grad={float(grad_norm_t.item()):.3f} step={ms:,.1f}ms '
            f'steady_med={med:,.1f}ms tok/s={tps:,.0f} peak={peak:.1f}GiB',
            flush=True,
        )

    steady = step_ms[STEADY_SKIP:] if len(step_ms) > STEADY_SKIP else step_ms
    median_ms = float(np.median(steady))
    tps = GLOBAL_TOKENS / (median_ms / 1000.0)
    speedup = CONTROL_MS / median_ms

    if median_ms <= TARGET_2X_MS:
        decision = 'TWO_X_GATE_PASSED__KEEP'
    elif speedup >= FULL_KEEP_AT:
        decision = 'KEEP_NATIVE_READ__NEXT_BUILD_FUSED_WRITE'
    elif speedup >= FULL_REVERT_BELOW:
        decision = 'MARGINAL__PROFILE_BEFORE_MORE_ENGINEERING'
    else:
        decision = 'REVERT_NATIVE_READ'

    result = {
        'status': 'PASS',
        'initial_loss': iloss,
        'final_loss': float(losses[-1]),
        'median_B64_ms': median_ms,
        'tokens_per_second': tps,
        'peak_GiB': peak,
        'optimizer': optimizer_name,
        'speedup_vs_parent': speedup,
        'distance_to_2x_ms': median_ms - TARGET_2X_MS,
        'distance_to_4x_ms': median_ms - TARGET_4X_MS,
        'decision': decision,
    }

    if PROFILE:
        result['profile'] = profile_one_micro(execute, model, data)

    model.zero_grad(set_to_none=True)
    optimizer = None; execute = None; model = None
    gpu_clean(reset_compiler=True)
    return result

# ======================================================================================
# 7. MAIN
# ======================================================================================

def main():
    try:
        data = load_corpus()
        init = canonical_init()

        oracle = fp32_oracle_gate()
        REPORT['fp32_oracle'] = oracle
        save_report()

        read = real_read_gate(data, init)
        REPORT['read_micro'] = read
        save_report()

        print('\n[4/7] READ MICRO DECISION')
        print('=' * 120)
        if read.get('status') == 'PASS':
            print(f"Control F+B   : {read['control']['fb_ms']:.2f} ms")
            print(f"Candidate F+B : {read['candidate']['fb_ms']:.2f} ms")
            print(f"Read speedup  : {read['speedup']:.3f}x")
        print(f"Decision      : {read.get('decision')}")
        print('=' * 120)

        if read.get('decision') != 'PROMOTE_TO_FULL_MODEL':
            REPORT['status'] = 'REVERTED_AT_READ_GATE'
            REPORT['final_decision'] = 'REVERT_READ_REPRESENTATION'
            save_report()
            print('\nSTOP: candidate failed the predeclared 1.10x real B32 read F+B gate.')
            print('Do not spend full-model minutes on it.')
            print(f'Report: {OUTPUT / "report.json"}')
            return REPORT

        full = benchmark_full_candidate(data, init)
        REPORT['full_candidate'] = full
        REPORT['status'] = 'COMPLETED'
        REPORT['final_decision'] = full['decision']
        save_report()

        print('\n[7/7] FINAL RESULT')
        print('=' * 120)
        print(f'Parent B64 control : {CONTROL_MS:,.1f} ms')
        print(f'Candidate B64      : {full["median_B64_ms"]:,.1f} ms')
        print(f'Whole speedup      : {full["speedup_vs_parent"]:.3f}x')
        print(f'Throughput         : {full["tokens_per_second"]:,.0f} tok/s')
        print(f'Peak memory        : {full["peak_GiB"]:.1f} GiB')
        print(f'2x target          : {TARGET_2X_MS:,.1f} ms')
        print(f'4x target          : {TARGET_4X_MS:,.1f} ms')
        print('-' * 120)
        print(f'DECISION           : {full["decision"]}')
        print('-' * 120)
        if full['decision'] == 'KEEP_NATIVE_READ__NEXT_BUILD_FUSED_WRITE':
            print('Next causal experiment: keep this READ representation fixed; fuse Dy -> ReLU*x -> E write path.')
        elif full['decision'] == 'MARGINAL__PROFILE_BEFORE_MORE_ENGINEERING':
            print('Do not add another mechanism yet. Read candidate_profile.txt and identify the remaining paid boundary.')
        elif full['decision'] == 'REVERT_NATIVE_READ':
            print('Delete this branch. The high-level blocked representation did not pay enough; descend only if a custom on-chip score tile is justified.')
        print(f'Report : {OUTPUT / "report.json"}')
        if PROFILE:
            print(f'Profile: {full.get("profile")}')
        print('=' * 120)
        return REPORT

    except Exception as exc:
        REPORT['status'] = 'FAILED_DO_NOT_USE'
        REPORT['error'] = f'{type(exc).__name__}: {exc}'
        REPORT['traceback'] = traceback.format_exc()
        save_report()
        raise


NATIVE_READ_GATE_REPORT = main()
