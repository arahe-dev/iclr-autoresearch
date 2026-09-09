from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    batch: int
    sequence_length: int
    microbatches: int
    real_input_tokens: int
    role: str


B32 = BenchmarkSpec("B32", 32, 2048, 1, 32 * 2048, "production microbatch")
B64 = BenchmarkSpec("B64", 32, 2048, 2, 2 * 32 * 2048, "exact B32+B32 optimizer update")


def throughput_tok_s(real_input_tokens: int, wall_ms: float) -> float:
    if wall_ms <= 0:
        raise ValueError("wall_ms must be positive")
    return real_input_tokens / (wall_ms / 1000.0)


def summarize(samples_ms: Iterable[float], spec: BenchmarkSpec) -> dict[str, Any]:
    samples = [float(x) for x in samples_ms]
    if not samples or any(x <= 0 for x in samples):
        raise ValueError("benchmark samples must be non-empty and positive")
    med = statistics.median(samples)
    return {
        "benchmark": asdict(spec),
        "samples_ms": samples,
        "median_ms": med,
        "p90_ms": sorted(samples)[min(len(samples) - 1, int(0.9 * (len(samples) - 1)))],
        "real_input_tok_s": throughput_tok_s(spec.real_input_tokens, med),
    }


def measure_callable(
    fn: Callable[[], Any],
    spec: BenchmarkSpec,
    *,
    warmups: int = 2,
    repetitions: int = 5,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Measure an explicitly supplied callable; never selects or edits a model."""
    if repetitions < 1 or warmups < 0:
        raise ValueError("invalid warmups/repetitions")
    sync = synchronize or (lambda: None)
    for _ in range(warmups):
        fn()
    sync()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    result = summarize(samples, spec)
    result["warmups"] = warmups
    result["repetitions"] = repetitions
    result["measurement_class"] = "wall_clock_callable"
    return result

