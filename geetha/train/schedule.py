r"""Learning-rate schedule and progress for a run that is bought by the hour.

The pilot used a constant learning rate with no warm-up. That is fine for twenty
minutes and wrong for a real run: a LoRA initialised at zero takes its first steps at
full rate on whatever batch comes first, and a run that ends at full rate ends on a
noisy iterate rather than a settled one.

**Progress is measured against two budgets at once, and the larger wins.** A paid run
has a token target (so many epochs) *and* a wall-clock budget (so many rupees). Nobody
knows the pod's real throughput until it is running, so a schedule keyed to steps alone
would either finish its decay hours early or be cut off by the deadline at a high rate.
Taking ``max(tokens_fraction, time_fraction)`` means the cosine always lands at its
floor exactly when the run stops, whichever budget stops it.
"""

from __future__ import annotations

import math

__all__ = ["progress", "lr_factor"]


def progress(*, tokens_seen: int, target_tokens: int,
             elapsed_s: float, planned_s: float) -> float:
    """Fraction of the run that is done, in ``[0, 1]``."""
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if planned_s <= 0:
        raise ValueError(f"planned_s must be positive, got {planned_s}")
    fraction = max(tokens_seen / target_tokens, elapsed_s / planned_s)
    return min(1.0, max(0.0, fraction))


def lr_factor(*, optimiser_step: int, warmup_steps: int,
              progress_fraction: float, floor: float = 0.05) -> float:
    """Multiplier on the peak learning rate: linear warm-up, then cosine to ``floor``.

    ``optimiser_step`` counts steps **since this process started**, not since the run
    began. A resumed run rebuilds its optimiser from nothing, so it warms up again --
    briefly, and into whatever point of the cosine the run had already reached.
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be within [0, 1], got {floor}")
    fraction = min(1.0, max(0.0, progress_fraction))
    cosine = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * fraction))
    if warmup_steps > 0 and optimiser_step < warmup_steps:
        return cosine * (optimiser_step + 1) / warmup_steps
    return cosine
