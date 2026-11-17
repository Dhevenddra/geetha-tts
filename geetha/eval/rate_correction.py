r"""Recall-corrected event rates, with intervals (D-030, outstanding item 3).

Why a raw count may never be published
--------------------------------------
D-028 rejected the AudioSet breath tagger, and the reason is the whole justification for
this module: **a threshold can always be tuned until the rate matches.** That tagger was
dialled to produce 9.64 breaths/min against a ground truth of 9.13 -- and at the same
threshold it fired at 1.6x that rate on clips containing no breath at all. The rate
looked right and meant nothing.

So a detector's own count is not an event rate. Two corrections stand between them:

* the detector **misses** events, at rate ``1 - recall``;
* the detector **invents** events, at rate ``1 - precision``.

The corrected count is ``detections x precision / recall``: multiply by precision to
keep only the detections that are real, divide by recall to scale up to the events that
were there.

Why the interval is bootstrapped and not a formula
--------------------------------------------------
Precision and recall are themselves **estimates**, from a finite calibration set -- 407
annotated breaths over 1,714 located pauses, in our case. An interval that counted only
the Poisson noise in the detection count would report the correction as if it were
exact, which is the same error in a new costume. So the bootstrap resamples three things
together: the detection count, and the two confusion-matrix proportions behind precision
and recall. The width then reflects how well the detector is actually calibrated, and a
detector calibrated on a tenth of the data gets a visibly wider interval -- which is a
test in this module, not a hope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RateEstimate", "corrected_event_rate"]


@dataclass(frozen=True)
class RateEstimate:
    """A corrected event rate with everything needed to read it honestly."""

    rate_per_minute: float
    low: float
    high: float
    corrected_events: float
    detections: int
    minutes: float
    precision: float
    recall: float

    def summary(self) -> str:
        """One line that cannot be quoted without its interval."""
        return (
            f"{self.rate_per_minute:.2f} events per minute "
            f"[95% CI {self.low:.2f}, {self.high:.2f}] "
            f"— {self.detections} raw detections over {self.minutes:.1f} min, "
            f"corrected to {self.corrected_events:.1f} events using "
            f"precision {self.precision:.3f} and recall {self.recall:.3f}"
        )


def corrected_event_rate(
    *,
    detections: int,
    minutes: float,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    bootstrap: int = 4_000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> RateEstimate:
    """Correct a detector's raw count into an event rate, with a bootstrap interval.

    The three calibration counts come from the detector's own held-out confusion matrix
    at the operating point being published. They are required, not optional: a rate
    without them is exactly what D-030 forbids.
    """
    if detections < 0:
        raise ValueError(f"detections must not be negative, got {detections}")
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    for name, value in (("true_positives", true_positives),
                        ("false_positives", false_positives),
                        ("false_negatives", false_negatives)):
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")

    detected_total = true_positives + false_positives
    actual_total = true_positives + false_negatives
    if detected_total == 0:
        raise ValueError("precision is undefined: the calibration set has no detections")
    if actual_total == 0:
        raise ValueError("recall is undefined: the calibration set has no real events")

    precision = true_positives / detected_total
    recall = true_positives / actual_total
    if true_positives == 0:
        # recall == 0 and precision == 0 are the same condition, not two: both mean the
        # detector produced no true positive on the calibration set. There is then
        # nothing to scale up from and nothing to trust in what it did fire on.
        raise ValueError(
            "no true positives on the calibration set, so precision and recall are both "
            "0: this detector's count cannot be corrected into a rate at all"
        )

    corrected = detections * precision / recall

    # Resample the count and both calibration proportions together. Binomial draws on
    # the confusion matrix, Poisson on the count: the count is a number of events
    # observed over a window, not a proportion of a fixed sample.
    rng = np.random.default_rng(seed)
    drawn_detections = rng.poisson(detections, size=bootstrap).astype(float)
    drawn_tp_of_detected = rng.binomial(detected_total, precision, size=bootstrap) / detected_total
    drawn_tp_of_actual = rng.binomial(actual_total, recall, size=bootstrap) / actual_total

    usable = drawn_tp_of_actual > 0
    rates = np.full(bootstrap, np.nan)
    rates[usable] = (drawn_detections[usable] * drawn_tp_of_detected[usable]
                     / drawn_tp_of_actual[usable] / minutes)
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:  # pragma: no cover - needs a pathological calibration set
        raise ValueError("the bootstrap produced no usable draws; the calibration set is "
                         "too small to support an interval")

    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(rates, [100 * tail, 100 * (1.0 - tail)])

    return RateEstimate(
        rate_per_minute=corrected / minutes,
        low=float(low),
        high=float(high),
        corrected_events=corrected,
        detections=detections,
        minutes=minutes,
        precision=precision,
        recall=recall,
    )
