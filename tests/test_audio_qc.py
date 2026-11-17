"""Step 1 of the ETL: per-clip audio QC.

These numbers decide what enters a training manifest, so each one is pinned to a
waveform whose answer is known by construction rather than to a real clip whose
answer is only whatever the code happens to print.
"""

from __future__ import annotations

import numpy as np
import pytest

from geetha.audio.qc import ClipQC, clip_qc, to_mono


def tone(seconds: float, sample_rate: int = 48_000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def silence(seconds: float, sample_rate: int = 48_000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


class TestToMono:
    def test_averages_the_two_channels_of_a_stereo_clip(self):
        stereo = np.array([[0.2, 0.4], [-0.6, -0.2]], dtype=np.float32)
        assert to_mono(stereo).tolist() == pytest.approx([0.3, -0.4])

    def test_leaves_a_mono_clip_untouched(self):
        mono = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        assert to_mono(mono).tolist() == pytest.approx([0.1, -0.2, 0.3])

    def test_a_single_column_counts_as_mono(self):
        assert to_mono(np.array([[0.1], [0.2]], dtype=np.float32)).shape == (2,)


class TestDurationAndLevel:
    def test_duration_is_samples_over_sample_rate(self):
        assert clip_qc(tone(1.5), 48_000).duration_s == pytest.approx(1.5)

    def test_peak_is_the_largest_absolute_sample(self):
        assert clip_qc(tone(0.2, amplitude=0.5), 48_000).peak == pytest.approx(0.5, abs=1e-3)

    def test_a_constant_tenth_scale_signal_measures_minus_20_dbfs(self):
        wave = np.full(4_800, 0.1, dtype=np.float32)
        assert clip_qc(wave, 48_000).rms_dbfs == pytest.approx(-20.0, abs=0.01)

    def test_digital_silence_reports_a_floor_rather_than_negative_infinity(self):
        qc = clip_qc(silence(0.5), 48_000)
        assert qc.rms_dbfs == ClipQC.SILENCE_DBFS
        assert np.isfinite(qc.rms_dbfs)


class TestClipping:
    def test_counts_samples_at_or_beyond_full_scale(self):
        wave = tone(0.1)
        wave[10:15] = 1.0
        assert clip_qc(wave, 48_000).clipped_samples == 5

    def test_a_quiet_clip_has_no_clipped_samples(self):
        assert clip_qc(tone(0.1, amplitude=0.3), 48_000).clipped_samples == 0

    def test_clipped_fraction_is_relative_to_clip_length(self):
        wave = np.zeros(1_000, dtype=np.float32)
        wave[:10] = -1.0
        assert clip_qc(wave, 48_000).clipped_fraction == pytest.approx(0.01)


class TestEdgeSilence:
    def test_measures_silence_at_each_end_separately(self):
        wave = np.concatenate([silence(0.5), tone(1.0), silence(0.25)])
        qc = clip_qc(wave, 48_000)
        assert qc.leading_silence_s == pytest.approx(0.5, abs=0.02)
        assert qc.trailing_silence_s == pytest.approx(0.25, abs=0.02)

    def test_a_clip_that_starts_and_ends_on_speech_has_no_edge_silence(self):
        qc = clip_qc(tone(1.0), 48_000)
        assert qc.leading_silence_s == pytest.approx(0.0, abs=0.02)
        assert qc.trailing_silence_s == pytest.approx(0.0, abs=0.02)

    def test_an_entirely_silent_clip_is_flagged_and_not_double_counted(self):
        qc = clip_qc(silence(0.75), 48_000)
        assert qc.is_silent
        assert qc.leading_silence_s == pytest.approx(0.75, abs=0.02)
        assert qc.trailing_silence_s == pytest.approx(0.0)

    def test_quiet_breath_above_the_floor_is_not_counted_as_silence(self):
        wave = np.concatenate([tone(0.3, amplitude=0.02), tone(0.5)])
        assert clip_qc(wave, 48_000).leading_silence_s == pytest.approx(0.0, abs=0.02)


class TestRejectsNonsense:
    def test_a_zero_length_clip_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            clip_qc(np.zeros(0, dtype=np.float32), 48_000)

    def test_a_non_positive_sample_rate_is_rejected(self):
        with pytest.raises(ValueError, match="sample_rate"):
            clip_qc(tone(0.1), 0)

    def test_more_than_two_dimensions_is_rejected(self):
        with pytest.raises(ValueError, match="dimension"):
            to_mono(np.zeros((2, 2, 2), dtype=np.float32))
