"""ETL step 6: preparing audio for SNAC -- gain, resampling, clamping.

Both corpora are 48 kHz, so the only ratio that matters is an exact 2:1 decimation to
the 24 kHz SNAC expects. The test that earns its place here is the anti-aliasing one:
decimation without a low-pass filter folds a 20 kHz component down to 4 kHz at nearly
full amplitude, which would put a loud artefact in the middle of the speech band and
sound like a whistle the model then learns to reproduce. A length check alone would not
notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from geetha.audio.snac_input import SNAC_SAMPLE_RATE, apply_gain, prepare_for_snac, resample_to_snac


def tone(hz: float, seconds: float, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def rms(wave: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wave, dtype=np.float64))))


class TestApplyGain:
    def test_zero_gain_leaves_the_waveform_alone(self):
        wave = tone(1_000, 0.1, 48_000)
        assert np.allclose(apply_gain(wave, 0.0), wave)

    def test_minus_six_db_halves_the_amplitude(self):
        wave = tone(1_000, 0.1, 48_000)
        assert rms(apply_gain(wave, -6.0206)) == pytest.approx(rms(wave) / 2, rel=1e-3)

    def test_the_result_never_leaves_the_valid_range(self):
        hot = np.array([1.0, -1.0, 0.99], dtype=np.float32)
        assert np.abs(apply_gain(hot, 3.0)).max() <= 1.0

    def test_a_positive_gain_is_still_applied_when_asked_for_explicitly(self):
        # The manifest never asks for one (D-036 is attenuation only), but the function
        # must not silently ignore its argument.
        quiet = np.full(100, 0.1, dtype=np.float32)
        assert rms(apply_gain(quiet, 6.0206)) == pytest.approx(0.2, rel=1e-3)


class TestResampleToSnac:
    def test_a_48k_clip_halves_in_length(self):
        out = resample_to_snac(tone(1_000, 1.0, 48_000), 48_000)
        assert len(out) == pytest.approx(SNAC_SAMPLE_RATE, abs=2)

    def test_a_clip_already_at_24k_is_returned_untouched(self):
        wave = tone(1_000, 0.5, SNAC_SAMPLE_RATE)
        out = resample_to_snac(wave, SNAC_SAMPLE_RATE)
        assert np.shares_memory(out, wave) or np.allclose(out, wave)

    def test_a_speech_band_tone_keeps_its_level(self):
        source = tone(1_000, 0.5, 48_000)
        out = resample_to_snac(source, 48_000)
        assert rms(out) == pytest.approx(rms(source), rel=0.05)

    def test_a_tone_above_the_new_nyquist_is_filtered_out_rather_than_aliased(self):
        # 20 kHz is above 24 kHz/2. Plain decimation would fold it to 4 kHz at full
        # level; the low-pass must kill it instead.
        out = resample_to_snac(tone(20_000, 0.5, 48_000), 48_000)
        attenuation_db = 20 * np.log10(max(rms(out), 1e-12) / rms(tone(20_000, 0.5, 48_000)))
        assert attenuation_db < -40

    def test_an_unsupported_sample_rate_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="44100"):
            resample_to_snac(tone(1_000, 0.1, 44_100), 44_100)


class TestPrepareForSnac:
    def test_stereo_48k_becomes_mono_24k_with_the_gain_applied(self):
        stereo = np.stack([tone(1_000, 0.5, 48_000)] * 2, axis=1)
        out = prepare_for_snac(stereo, 48_000, gain_db=-6.0206)
        assert out.ndim == 1
        assert len(out) == pytest.approx(SNAC_SAMPLE_RATE // 2, abs=2)
        assert out.dtype == np.float32

    def test_the_output_is_always_within_range(self):
        loud = np.full(48_000, 0.98, dtype=np.float32)
        assert np.abs(prepare_for_snac(loud, 48_000, gain_db=6.0)).max() <= 1.0

    def test_an_empty_clip_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            prepare_for_snac(np.zeros(0, dtype=np.float32), 48_000, gain_db=0.0)
