"""What to do when an eval prompt's text also appears in training.

Runbook rule 3 names the remedy and it is not the one people reach for by instinct:

    any sentence (normalised hash, ``norm_text()``) in a held-out split is **removed
    from every training manifest**

So a collision is a *removal instruction for the ETL*, not a reason to rewrite the eval
set. Getting this backwards would quietly bias the eval set toward whatever text happens
to be rare, which is its own contamination.

This matters most for the Directability-Ta intent probes. A probe has to be a short,
common phrase — "சரி", "போ", "ஆமா" — or its intent cannot flip with delivery, so
collisions are inherent to the design rather than a mistake in it.

Measured 26 Oct 2026 across 909,265 training rows: honouring rule 3 for all 16 colliding
probes costs **2,044 clips, 0.22%**. Four probes account for most of it (சரி 1,341,
போ 371, அப்படியா 165, ஆமா 107). At that price the rule stands exactly as written and
nothing needs weakening.

The guard is therefore on *cost*, not on the existence of collisions: if removal would
cost a large share of the training data, the prompt set is badly designed and the build
should stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = ["LeakRemovalPlan", "MAX_REMOVAL_FRACTION", "plan_leak_removal"]

# Above this, stop: losing more than 1% of training data to protect a prompt set means
# the prompts are too generic to measure anything.
MAX_REMOVAL_FRACTION = 0.01


@dataclass(frozen=True)
class LeakRemovalPlan:
    """Texts the ETL must drop from every training manifest, and what that costs."""

    rows: list[tuple[str, str, int]]  # (prompt id, text, clips removed)
    clips_removed: int
    training_rows: int

    @property
    def fraction(self) -> float:
        if self.training_rows == 0:
            return 0.0
        return self.clips_removed / self.training_rows


def plan_leak_removal(
    *,
    leaks: Sequence[tuple[str, str]],
    clip_counts: Mapping[str, int],
    training_rows: int,
) -> LeakRemovalPlan:
    """Build the removal plan for ``leaks``, ordered most expensive first.

    Raises if a leak has no recorded clip count — writing a removal instruction without
    knowing its cost would hide the cost — or if the total cost exceeds
    ``MAX_REMOVAL_FRACTION``.
    """
    rows: list[tuple[str, str, int]] = []
    for prompt_id, text in leaks:
        if prompt_id not in clip_counts:
            raise ValueError(
                f"{prompt_id} collides with training text but has no clip count; count "
                "the cost before writing a removal instruction"
            )
        rows.append((prompt_id, text, clip_counts[prompt_id]))

    rows.sort(key=lambda row: (-row[2], row[0]))
    plan = LeakRemovalPlan(
        rows=rows,
        clips_removed=sum(row[2] for row in rows),
        training_rows=training_rows,
    )

    if plan.fraction > MAX_REMOVAL_FRACTION:
        raise ValueError(
            f"honouring rule 3 for these prompts would remove {plan.clips_removed:,} of "
            f"{training_rows:,} training clips ({plan.fraction:.1%}), over the "
            f"{MAX_REMOVAL_FRACTION:.0%} guard. The prompt set is too generic — rewrite "
            "the costliest prompts rather than weaken the rule."
        )
    return plan
