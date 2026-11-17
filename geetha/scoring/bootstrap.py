"""Bootstrap confidence intervals for error rates (runbook section 10).

Two functions:

* ``bootstrap_ci``     -- a 95% interval for one system's rate on one test set.
* ``paired_bootstrap`` -- the difference between two systems measured on the *same*
  utterances. Both systems are resampled on the same drawn utterances, which is the
  whole point: the comparison keeps the per-utterance pairing instead of treating
  the two runs as independent samples, and the interval comes out correspondingly
  tighter and honest.

Resampling is at the utterance level and the rate of each resample is recomputed as
summed errors over summed reference length -- the same aggregation rule the point
estimate uses (see ``metrics.aggregate``).

The seed is explicit and recorded, so a published interval can be reproduced
exactly. Default 1234, matching the CV sample seed in runbook section 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geetha.scoring.metrics import ErrorCounts

__all__ = ["BootstrapResult", "PairedResult", "bootstrap_ci", "paired_bootstrap", "two_sample_ci"]

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class PairedResult:
    """``delta`` is rate(a) - rate(b); negative means system a has fewer errors."""

    delta: float
    low: float
    high: float
    p_value: float
    confidence: float
    n_resamples: int
    seed: int


def _arrays(counts: Sequence[ErrorCounts]) -> tuple[np.ndarray, np.ndarray]:
    if not counts:
        raise ValueError("cannot bootstrap an empty corpus")
    errors = np.array([c.errors for c in counts], dtype=np.float64)
    lengths = np.array([c.reference_length for c in counts], dtype=np.float64)
    if lengths.sum() == 0:
        raise ValueError("cannot bootstrap a corpus whose references are all empty")
    return errors, lengths


def _resampled_rates(
    errors: np.ndarray, lengths: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Rate per resample: summed errors over summed reference length."""
    return errors[indices].sum(axis=1) / lengths[indices].sum(axis=1)


def _draw(rng: np.random.Generator, n_items: int, n_resamples: int) -> np.ndarray:
    return rng.integers(0, n_items, size=(n_resamples, n_items))


def bootstrap_ci(
    counts: Sequence[ErrorCounts],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap interval for the corpus error rate."""
    errors, lengths = _arrays(counts)
    rng = np.random.default_rng(seed)
    rates = _resampled_rates(errors, lengths, _draw(rng, len(counts), n_resamples))
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(rates, [tail, 100.0 - tail])
    return BootstrapResult(
        point=float(errors.sum() / lengths.sum()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_bootstrap(
    counts_a: Sequence[ErrorCounts],
    counts_b: Sequence[ErrorCounts],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> PairedResult:
    """Paired percentile bootstrap of rate(a) - rate(b) on shared utterances."""
    if len(counts_a) != len(counts_b):
        raise ValueError(
            f"paired bootstrap needs the same length on both sides, got {len(counts_a)} and {len(counts_b)}"
        )
    errors_a, lengths_a = _arrays(counts_a)
    errors_b, lengths_b = _arrays(counts_b)

    rng = np.random.default_rng(seed)
    indices = _draw(rng, len(counts_a), n_resamples)
    deltas = _resampled_rates(errors_a, lengths_a, indices) - _resampled_rates(
        errors_b, lengths_b, indices
    )

    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(deltas, [tail, 100.0 - tail])

    # Two-sided bootstrap test: how often the resampled difference lands on or past
    # zero from the other side than observed. Doubled for two-sidedness, capped at 1.
    not_below = float(np.mean(deltas >= 0.0))
    not_above = float(np.mean(deltas <= 0.0))
    p_value = min(1.0, 2.0 * min(not_below, not_above))

    delta = float(errors_a.sum() / lengths_a.sum() - errors_b.sum() / lengths_b.sum())
    return PairedResult(
        delta=delta,
        low=float(low),
        high=float(high),
        p_value=p_value,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )


def two_sample_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    statistic: str = "median",
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> PairedResult:
    """Percentile bootstrap of ``stat(a) - stat(b)`` for two *independent* samples.

    ``paired_bootstrap`` above is the right tool when both numbers come from the same
    utterances. It is the wrong tool when they do not: the contamination check (B4a)
    compares a Rasa *test* arm against a Rasa *train* arm, which are different clips by
    construction, and pairing them would invent a correspondence that does not exist and
    report an interval far too tight.

    Each side is resampled independently with replacement at its own size. ``statistic``
    is ``"median"`` (robust, and what the contamination verdict rule is written in terms
    of) or ``"mean"``.
    """
    if statistic not in {"median", "mean"}:
        raise ValueError(f"unknown statistic {statistic!r}")
    if len(sample_a) == 0 or len(sample_b) == 0:
        raise ValueError("cannot bootstrap an empty sample")

    values_a = np.asarray(sample_a, dtype=np.float64)
    values_b = np.asarray(sample_b, dtype=np.float64)
    reduce = np.median if statistic == "median" else np.mean

    rng = np.random.default_rng(seed)
    draws_a = reduce(values_a[_draw(rng, len(values_a), n_resamples)], axis=1)
    draws_b = reduce(values_b[_draw(rng, len(values_b), n_resamples)], axis=1)
    deltas = draws_a - draws_b

    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(deltas, [tail, 100.0 - tail])

    not_below = float(np.mean(deltas >= 0.0))
    not_above = float(np.mean(deltas <= 0.0))
    p_value = min(1.0, 2.0 * min(not_below, not_above))

    return PairedResult(
        delta=float(reduce(values_a) - reduce(values_b)),
        low=float(low),
        high=float(high),
        p_value=p_value,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
