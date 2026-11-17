"""ETL step 5: the QC gates and loudness gain that decide what trains (D-036).

Each gate exists because step 1 measured something real, so each threshold is tested at
its boundary rather than somewhere comfortably past it: an off-by-one on a gate silently
changes the training set, and the loss would never show it.
"""

from __future__ import annotations

import pytest

from geetha.data.manifest import (
    EXPECTED_SAMPLE_RATE,
    MAX_DURATION_S,
    MIN_DURATION_S,
    TARGET_RMS_DBFS,
    gain_db,
    qc_verdict,
)


def qc(**overrides):
    """A clip that passes every gate, with the field under test overridden."""
    row = {
        "error": "", "is_silent": False, "duration_s": 8.0, "clipped_fraction": 0.0,
        "leading_silence_s": 0.05, "trailing_silence_s": 0.05, "rms_dbfs": -20.0,
        "sample_rate": EXPECTED_SAMPLE_RATE,
    }
    row.update(overrides)
    return row


class TestQcVerdict:
    def test_a_good_clip_passes_with_no_reason(self):
        assert qc_verdict(qc()) == ""

    def test_a_clip_that_failed_to_decode_is_rejected(self):
        assert qc_verdict(qc(error="RuntimeError: boom")) == "decode_error"

    def test_a_silent_clip_is_rejected(self):
        assert qc_verdict(qc(is_silent=True)) == "silent"

    def test_a_missing_measurement_is_rejected_rather_than_treated_as_zero(self):
        # A null here means step 1 could not measure the clip. Defaulting it to 0.0
        # would let it through every numeric gate.
        assert qc_verdict(qc(duration_s=None)) == "decode_error"
        assert qc_verdict(qc(rms_dbfs=None)) == "decode_error"


class TestClippingGate:
    def test_audible_clipping_is_rejected(self):
        assert qc_verdict(qc(clipped_fraction=0.002)) == "clipping"

    def test_a_lone_full_scale_sample_is_kept_because_it_is_a_rounding_artefact(self):
        assert qc_verdict(qc(clipped_fraction=0.0005)) == ""

    def test_the_threshold_itself_is_kept(self):
        assert qc_verdict(qc(clipped_fraction=0.001)) == ""


class TestEdgeSilenceGate:
    def test_a_clip_more_than_half_edge_silence_is_rejected(self):
        assert qc_verdict(qc(duration_s=2.0, leading_silence_s=1.1, trailing_silence_s=0.0)) \
            == "edge_silence"

    def test_edge_silence_counts_both_ends_together(self):
        assert qc_verdict(qc(duration_s=2.0, leading_silence_s=0.6, trailing_silence_s=0.6)) \
            == "edge_silence"

    def test_ordinary_studio_padding_is_kept(self):
        # Rasa's median is 9% edge silence; that is normal, not a defect.
        assert qc_verdict(qc(duration_s=4.68, leading_silence_s=0.20, trailing_silence_s=0.22)) == ""


class TestDurationGate:
    def test_a_clip_too_short_to_train_on_is_rejected(self):
        assert qc_verdict(qc(duration_s=MIN_DURATION_S - 0.01)) == "too_short"

    def test_a_clip_longer_than_the_cap_is_rejected(self):
        assert qc_verdict(qc(duration_s=MAX_DURATION_S + 0.01)) == "too_long"

    def test_both_bounds_are_inclusive(self):
        assert qc_verdict(qc(duration_s=MIN_DURATION_S)) == ""
        assert qc_verdict(qc(duration_s=MAX_DURATION_S)) == ""


class TestGainDb:
    def test_a_hot_clip_is_attenuated_to_the_target(self):
        assert gain_db(-16.6) == pytest.approx(TARGET_RMS_DBFS + 16.6)
        assert gain_db(-16.6) < 0

    def test_a_clip_already_at_the_target_is_left_alone(self):
        assert gain_db(TARGET_RMS_DBFS) == pytest.approx(0.0)

    def test_a_quiet_clip_is_never_boosted(self):
        # D-036: attenuation only. IV-R has no headroom and boosting would clip audio
        # that is currently clean.
        assert gain_db(-34.9) == pytest.approx(0.0)

    def test_the_target_sits_below_the_hot_source_so_it_only_ever_attenuates_it(self):
        assert gain_db(-13.0) < 0

    def test_a_missing_level_raises_rather_than_silently_applying_no_gain(self):
        with pytest.raises(ValueError, match="rms_dbfs"):
            gain_db(None)


class TestSampleRateGate:
    """Rasa holds 408 clips at 22,050 Hz; the encoder refuses a non-integer ratio."""

    def test_the_expected_rate_passes(self):
        assert qc_verdict(qc(sample_rate=48_000)) == ""

    def test_a_clip_at_22050_is_rejected(self):
        assert qc_verdict(qc(sample_rate=22_050)) == "sample_rate"

    def test_any_other_rate_is_rejected_rather_than_assumed_resamplable(self):
        assert qc_verdict(qc(sample_rate=16_000)) == "sample_rate"
        assert qc_verdict(qc(sample_rate=44_100)) == "sample_rate"
