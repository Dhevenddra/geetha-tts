r"""When a paid training run must stop.

the project rules: "If a job misbehaves (loss NaN, probe CER rising, idle GPU), stop it, pull
what exists, and tell the operator in plain words what happened and what you propose. A
failed run is fine; a silently burning pod is not."

These are those rules as code, because on a 25-hour Stage A run nobody is reading the
logs for most of it, and "I would have noticed" is not a safety layer.

Four stop reasons, in order of how hard they are to spot by eye:

``nan_loss``
    Obvious once seen, and it outranks everything else so the report names the real
    problem rather than the budget that happened to expire at the same moment.
``diverging``
    The one that actually costs money. A NaN announces itself; a loss that has quietly
    climbed to three times where it started burns the whole budget and ships a model
    worse than the base, with a log full of numbers each of which looks plausible alone.
    The baseline is the **opening window's mean**, never the last step -- otherwise one
    spike raises the bar for every step after it and the guard disables itself exactly
    when it is needed.
``max_steps`` / ``budget_exhausted``
    Ordinary completion.

Divergence is deliberately not judged before ``divergence_after`` steps: early loss is
noisy and judging it would stop healthy runs in their first minute.
"""

from __future__ import annotations

import math

__all__ = ["TrainingBudget"]


class TrainingBudget:
    """Stop rules for one training run. Stateful: it accumulates the baseline."""

    def __init__(
        self,
        *,
        max_seconds: float,
        max_steps: int,
        divergence_factor: float | None = 3.0,
        divergence_after: int = 50,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if divergence_factor is not None and divergence_factor <= 1.0:
            raise ValueError(
                f"divergence_factor must be above 1.0, got {divergence_factor}: at 1.0 the "
                "run would stop the first time the loss ticked up"
            )
        if divergence_after < 1:
            raise ValueError(f"divergence_after must be at least 1, got {divergence_after}")

        self.max_seconds = max_seconds
        self.max_steps = max_steps
        self.divergence_factor = divergence_factor
        self.divergence_after = divergence_after
        self._opening: list[float] = []
        self.baseline_loss: float | None = None

    def should_stop(self, *, step: int, elapsed: float, loss: float) -> str | None:
        """The reason to stop now, or ``None`` to keep training."""
        # A NaN first: it explains itself, and any other reason reported alongside it
        # would send whoever reads the report looking in the wrong place.
        if not math.isfinite(loss):
            return "nan_loss"

        if self.baseline_loss is None:
            self._opening.append(loss)
            if len(self._opening) >= self.divergence_after:
                self.baseline_loss = sum(self._opening) / len(self._opening)
        elif self.divergence_factor is not None:
            if loss > self.baseline_loss * self.divergence_factor:
                return "diverging"

        if elapsed >= self.max_seconds:
            return "budget_exhausted"
        if step >= self.max_steps:
            return "max_steps"
        return None
