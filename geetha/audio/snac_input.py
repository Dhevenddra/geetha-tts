r"""Prepare a clip for SNAC encoding -- ETL step 6 (D-036, D-037).

Three things happen to every clip before it reaches the codec, and the order matters:
channels are averaged to mono, the manifest's loudness gain is applied, and the result
is decimated from 48 kHz to the 24 kHz SNAC expects.

Why the resampler is written out here rather than delegated
-----------------------------------------------------------
Both TTS corpora are 48 kHz (measured, step 1), so the only ratio needed is an exact
2:1 decimation -- and that is the one case where getting it wrong is both easy and
inaudible at a glance. Dropping every second sample without filtering first **folds**
everything above 12 kHz back down into the speech band: a 20 kHz component lands at
4 kHz at nearly full amplitude. The result is a whistle sitting in the middle of the
voice, which the model would faithfully learn to reproduce.

So the low-pass is a windowed-sinc FIR applied before decimation, and
``test_a_tone_above_the_new_nyquist_is_filtered_out_rather_than_aliased`` measures the
attenuation rather than trusting it. Doing it in numpy also keeps the unit tests free
of ``torch``, which only the codec itself needs.
"""

from __future__ import annotations

import numpy as np

from geetha.audio.qc import to_mono

__all__ = [
    "SNAC_SAMPLE_RATE",
    "apply_gain",
    "prepare_for_snac",
    "resample_to_snac",
]

SNAC_SAMPLE_RATE = 24_000

# Transition band for the decimation filter. 0.95 of the new Nyquist keeps the speech
# band intact (a 1 kHz tone passes within 5%) while leaving room for the roll-off to
# reach the >40 dB stop-band the test asserts.
_CUTOFF_FRACTION = 0.95
_FILTER_TAPS = 127


def apply_gain(wave: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale ``wave`` by ``gain_db``, clamped to the valid range."""
    array = np.asarray(wave, dtype=np.float32)
    if gain_db:
        array = array * np.float32(10.0 ** (gain_db / 20.0))
    return np.clip(array, -1.0, 1.0).astype(np.float32, copy=False)


def _lowpass(cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """A Hamming-windowed sinc low-pass, normalised to unity DC gain."""
    n = np.arange(_FILTER_TAPS, dtype=np.float64) - (_FILTER_TAPS - 1) / 2.0
    # 2 * fc/fs * sinc(2 * fc/fs * n) is the ideal low-pass impulse response.
    ratio = 2.0 * cutoff_hz / sample_rate
    kernel = ratio * np.sinc(ratio * n) * np.hamming(_FILTER_TAPS)
    return (kernel / kernel.sum()).astype(np.float64)


def resample_to_snac(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Decimate ``wave`` to :data:`SNAC_SAMPLE_RATE`, anti-aliasing first.

    Only integer ratios are accepted. A corpus arriving at some other rate is a fact
    worth stopping on rather than resampling approximately and forgetting about.
    """
    array = np.asarray(wave, dtype=np.float32).reshape(-1)
    if sample_rate == SNAC_SAMPLE_RATE:
        return array
    if sample_rate <= 0 or sample_rate % SNAC_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate {sample_rate} is not an integer multiple of {SNAC_SAMPLE_RATE}; "
            "both TTS corpora measured 48000, so an unexpected rate is a finding, not a "
            "resampling problem"
        )

    factor = sample_rate // SNAC_SAMPLE_RATE
    kernel = _lowpass(_CUTOFF_FRACTION * SNAC_SAMPLE_RATE / 2.0, sample_rate)
    # "same" keeps the output length at len(array), so the decimated length stays
    # len(array)/factor and the clip's duration is preserved.
    filtered = np.convolve(array.astype(np.float64), kernel, mode="same")
    return filtered[::factor].astype(np.float32, copy=False)


def prepare_for_snac(wave: np.ndarray, sample_rate: int, *, gain_db: float) -> np.ndarray:
    """Mono, gain-corrected, 24 kHz float32 audio ready for the codec."""
    mono = to_mono(wave)
    if mono.size == 0:
        raise ValueError("prepare_for_snac got an empty clip")
    # Gain first, so the filter never sees samples outside the valid range; then clamp
    # again afterwards. The second clamp is not redundant: a windowed-sinc filter
    # overshoots (Gibbs ringing, and an edge transient from the zero-padded
    # convolution), so a clip already at full scale comes out slightly above it. Left
    # alone that is distortion handed straight to the codec.
    resampled = resample_to_snac(apply_gain(mono, gain_db), sample_rate)
    return np.clip(resampled, -1.0, 1.0).astype(np.float32, copy=False)
