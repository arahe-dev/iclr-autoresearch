"""Small, independent exact oracle for the frozen Q=K attention contract.

This is deliberately a reference implementation, not a performance backend. It
uses the same Q tensor as both query and key, applies strict-exclusive causal
and same-document masking, and does not apply softmax or scale.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError("The numerical oracle requires PyTorch when executed") from exc
    return torch


def exact_qk_attention(q: Any, v: Any, segment_id: Any, valid: Any) -> Any:
    """Compute exact strict-exclusive same-document Q=K attention.

    Shapes are ``q[B,H,T,N]``, ``v[B,T,D]``, ``segment_id[B,T]`` and
    ``valid[B,T]``. The result is ``[B,H,T,D]``. Invalid query positions and
    positions with no strict prefix return zeros.
    """
    torch = _torch()
    if q.ndim != 4 or v.ndim != 3 or segment_id.ndim != 2 or valid.ndim != 2:
        raise ValueError("expected q[B,H,T,N], v[B,T,D], segment_id[B,T], valid[B,T]")
    b, h, t_len, _ = q.shape
    if v.shape[0] != b or v.shape[1] != t_len or segment_id.shape != (b, t_len) or valid.shape != (b, t_len):
        raise ValueError("inconsistent oracle shapes")

    outputs = []
    for t in range(t_len):
        if t == 0:
            outputs.append(torch.zeros((b, h, v.shape[-1]), device=q.device, dtype=q.dtype))
            continue
        # The same q tensor is used for query and key. Autograd therefore
        # accumulates both roles into the tied-QK gradient automatically.
        scores = torch.einsum("bhn,bhtn->bht", q[:, :, t, :], q[:, :, :t, :])
        same_doc = segment_id[:, :t] == segment_id[:, t : t + 1]
        prefix_valid = valid[:, :t] & valid[:, t : t + 1]
        mask = (same_doc & prefix_valid).to(dtype=scores.dtype)
        out = torch.einsum("bht,btd->bhd", scores * mask[:, None, :], v[:, :t, :])
        outputs.append(out)
    return torch.stack(outputs, dim=2)


def exact_qk_vjp(q: Any, v: Any, segment_id: Any, valid: Any, dy: Any) -> tuple[Any, Any]:
    """Return the exact tied-QK and V vector-Jacobian products."""
    torch = _torch()
    q_req = q.detach().requires_grad_(True)
    v_req = v.detach().requires_grad_(True)
    y = exact_qk_attention(q_req, v_req, segment_id, valid)
    dq, dv = torch.autograd.grad((y * dy).sum(), (q_req, v_req))
    return dq, dv


def oracle_contract_summary() -> dict[str, Any]:
    return {
        "q_equals_k": True,
        "strict_exclusive": True,
        "same_document_only": True,
        "softmax": False,
        "attention_scale": False,
        "shared_v": True,
        "implementation_class": "reference_only",
    }

