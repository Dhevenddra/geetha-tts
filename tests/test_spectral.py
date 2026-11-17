"""A machine-checkable round-trip test for the SNAC codec (ETL step 6).

The runbook's done-when for step 6 is "20 round-trip decodes sound right", which needs
a person. But the failure this guards against -- a wrong token layout -- does not
produce subtly worse audio, it produces noise, and that is measurable. So the ear check
stays for quality and this covers correctness.

SNAC is lossy, so the waveforms will not match sample for sample; the spectra will.
"""

from __future__ import annotations

import numpy as np
import pytest

from geetha.audio.spectral import spectral_similarity


def tone(hz: float, seconds: float, rate: int = 24_000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def speechlike(seconds: float = 0.5, rate: int = 24_000) -> np.ndarray:
    """A few harmonics with a slow amplitude envelope -- enough structure to score."""
    rng = np.random.default_rng(1234)
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    wave = sum(np.sin(2 * np.pi * f * t) / (index + 1)
               for index, f in enumerate((140, 280, 420, 560, 700)))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (0.3 * wave * envelope + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


class TestSpectralSimilarity:
    def test_a_signal_is_identical_to_itself(self):
        wave = speechlike()
        assert spectral_similarity(wave, wave, 24_000) == pytest.approx(1.0, abs=1e-6)

    def test_noise_scores_far_from_the_signal_it_replaced(self):
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(speechlike().size).astype(np.float32) * 0.3
        assert spectral_similarity(speechlike(), noise, 24_000) < 0.5

    def test_a_lossy_but_faithful_copy_still_scores_high(self):
        # Quantising to 8 bits is audible damage but keeps the spectrum.
        wave = speechlike()
        lossy = np.round(wave * 127).astype(np.float32) / 127
        assert spectral_similarity(wave, lossy, 24_000) > 0.9

    def test_a_different_pitch_scores_lower_than_the_same_pitch(self):
        same = spectral_similarity(tone(200, 0.5), tone(200, 0.5), 24_000)
        other = spectral_similarity(tone(200, 0.5), tone(900, 0.5), 24_000)
        assert other < same

    def test_lengths_that_differ_slightly_are_compared_over_the_shared_part(self):
        wave = speechlike()
        assert spectral_similarity(wave, wave[:-500], 24_000) > 0.95

    def test_an_empty_signal_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            spectral_similarity(np.zeros(0, dtype=np.float32), speechlike(), 24_000)

    def test_a_signal_shorter_than_one_analysis_frame_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            spectral_similarity(tone(200, 0.001), tone(200, 0.001), 24_000)

    def test_silence_against_silence_is_reported_as_undefined_rather_than_perfect(self):
        # Two flat spectra have no variance to correlate; claiming 1.0 would let a
        # codec that output silence pass the round-trip check.
        silence = np.zeros(24_000, dtype=np.float32)
        assert np.isnan(spectral_similarity(silence, silence, 24_000))
