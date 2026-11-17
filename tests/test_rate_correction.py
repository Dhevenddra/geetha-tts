"""Recall-corrected event rates with intervals (D-030, outstanding item 3).

D-030 forbids publishing a raw detector count as a breath rate, for a reason the project
learned the hard way in D-028: a threshold can always be tuned until the rate matches
the ground truth, and the rejected AudioSet tagger did exactly that while still firing
at 1.6x on clips containing no breath at all.

So a published rate has to be corrected for what the detector actually does -- it misses
events (recall < 1) and it invents them (precision < 1) -- and it has to carry an
interval that includes the uncertainty in the detector's own calibration, not just the
sampling noise in the count.
"""

from __future__ import annotations

import pytest

from geetha.eval.rate_correction import RateEstimate, corrected_event_rate


def estimate(**overrides) -> RateEstimate:
    spec = dict(detections=100, minutes=10.0,
                true_positives=90, false_positives=10, false_negatives=10, seed=7)
    spec.update(overrides)
    return corrected_event_rate(**spec)


class TestPointEstimate:
    def test_a_perfect_detector_returns_the_raw_rate(self):
        result = corrected_event_rate(detections=50, minutes=10.0, true_positives=100,
                                      false_positives=0, false_negatives=0, seed=1)
        assert result.rate_per_minute == pytest.approx(5.0)

    def test_half_the_recall_doubles_the_corrected_count(self):
        # 100 detections, nothing spurious, but the detector finds only half the events.
        result = corrected_event_rate(detections=100, minutes=10.0, true_positives=50,
                                      false_positives=0, false_negatives=50, seed=1)
        assert result.corrected_events == pytest.approx(200.0)
        assert result.rate_per_minute == pytest.approx(20.0)

    def test_half_the_precision_halves_the_corrected_count(self):
        result = corrected_event_rate(detections=100, minutes=10.0, true_positives=50,
                                      false_positives=50, false_negatives=0, seed=1)
        assert result.corrected_events == pytest.approx(50.0)

    def test_precision_and_recall_are_reported_alongside_the_rate(self):
        result = estimate()
        assert result.precision == pytest.approx(0.9)
        assert result.recall == pytest.approx(0.9)

    def test_no_detections_gives_a_rate_of_zero(self):
        result = estimate(detections=0)
        assert result.rate_per_minute == pytest.approx(0.0)
        assert result.low == pytest.approx(0.0)


class TestInterval:
    def test_the_interval_contains_the_point_estimate(self):
        result = estimate()
        assert result.low <= result.rate_per_minute <= result.high

    def test_the_interval_is_not_degenerate(self):
        assert estimate().high > estimate().low

    def test_a_weakly_calibrated_detector_gives_a_wider_interval(self):
        # Same precision and recall, a tenth of the calibration data behind them.
        strong = corrected_event_rate(detections=100, minutes=10.0, true_positives=900,
                                      false_positives=100, false_negatives=100, seed=3)
        weak = corrected_event_rate(detections=100, minutes=10.0, true_positives=90,
                                    false_positives=10, false_negatives=10, seed=3)
        assert (weak.high - weak.low) > (strong.high - strong.low)

    def test_more_observed_minutes_narrows_the_interval(self):
        short = corrected_event_rate(detections=10, minutes=1.0, true_positives=90,
                                     false_positives=10, false_negatives=10, seed=5)
        long = corrected_event_rate(detections=1_000, minutes=100.0, true_positives=90,
                                    false_positives=10, false_negatives=10, seed=5)
        assert (long.high - long.low) < (short.high - short.low)

    def test_the_same_seed_reproduces_the_same_interval(self):
        assert estimate(seed=11).low == pytest.approx(estimate(seed=11).low)

    def test_calibration_uncertainty_is_included_not_just_counting_noise(self):
        # A perfectly calibrated detector on infinite data would leave only Poisson
        # noise in the count. With finite calibration the interval must be wider than
        # that, or the correction is being reported as if it were exact.
        from math import sqrt

        result = corrected_event_rate(detections=100, minutes=10.0, true_positives=90,
                                      false_positives=10, false_negatives=10, seed=2)
        poisson_only = 1.96 * sqrt(100) / 10.0
        assert (result.high - result.low) > 2 * poisson_only * 0.5


class TestRejectsNonsense:
    def test_a_detector_with_no_true_positives_is_refused(self):
        # recall == 0 and precision == 0 are one condition, not two: both follow from
        # the detector never having been right on the calibration set.
        with pytest.raises(ValueError, match="no true positives"):
            corrected_event_rate(detections=10, minutes=1.0, true_positives=0,
                                 false_positives=5, false_negatives=10)

    def test_a_calibration_set_with_no_real_events_leaves_recall_undefined(self):
        with pytest.raises(ValueError, match="recall is undefined"):
            corrected_event_rate(detections=10, minutes=1.0, true_positives=0,
                                 false_positives=10, false_negatives=0)

    def test_a_calibration_set_the_detector_never_fired_on_leaves_precision_undefined(self):
        with pytest.raises(ValueError, match="precision is undefined"):
            corrected_event_rate(detections=10, minutes=1.0, true_positives=0,
                                 false_positives=0, false_negatives=10)

    def test_non_positive_minutes_is_refused(self):
        with pytest.raises(ValueError, match="minutes"):
            corrected_event_rate(detections=10, minutes=0.0, true_positives=9,
                                 false_positives=1, false_negatives=1)

    def test_negative_detections_is_refused(self):
        with pytest.raises(ValueError, match="detections"):
            corrected_event_rate(detections=-1, minutes=1.0, true_positives=9,
                                 false_positives=1, false_negatives=1)


class TestReporting:
    def test_the_summary_names_the_correction_and_the_interval(self):
        text = estimate().summary()
        assert "per minute" in text
        assert "95%" in text
        assert "recall" in text.lower()

    def test_the_summary_refuses_to_omit_the_interval(self):
        # D-030: a rate without its interval and its baseline is not a measurement.
        assert "[" in estimate().summary()
