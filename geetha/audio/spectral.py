r"""A machine-checkable similarity between two renderings of the same speech.

Written for ETL step 6's round-trip check. The runbook asks for 20 decodes that "sound
right", which needs a person -- but the failure worth guarding against there is a wrong
SNAC token layout, and that does not degrade audio subtly, it produces noise. Noise is
measurable, so the ear check can be about quality while this one is about correctness.

The measure is the Pearson correlation of the two log magnitude spectrograms. Log,
because speech energy spans orders of magnitude and a linear correlation would be
dominated by whichever frames are loudest; magnitude only, because a codec is free to
discard phase and SNAC does.

Deliberately NaN, not 1.0, for two silent inputs: a flat spectrum has no variance to
correlate, and reporting a perfect score there would let a codec that emitted silence
pass the round-trip check. The caller has to notice.

**Limitation, measured and binding (D-040).** This does NOT measure intelligibility or
quality, and must never be used as a proxy for either. Against the operator's verdicts on
20 round-tripped clips it correlates at **r = 0.013**: the clip he rated okay holds the
lowest score of all 20 (0.469) and one he called gibberish scores 0.840. Its single
licensed use is detecting a wrong token layout, which it does -- a deliberately
misaligned stream is refused on every clip. For intelligibility, use CER with the two
frozen ASR judges.

Written in numpy rather than with a DSP dependency: the project has no scipy or librosa,
and an FFT and a correlation do not justify adding one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["spectral_similarity", "FRAME_SIZE", "HOP_SIZE"]

FRAME_SIZE = 1_024
HOP_SIZE = 256
# Keeps log(0) finite without flattening the quiet frames that carry breath.
_FLOOR = 1e-10


def _log_spectrogram(wave: np.ndarray) -> np.ndarray:
    frames = 1 + (len(wave) - FRAME_SIZE) // HOP_SIZE
    window = np.hanning(FRAME_SIZE)
    # One strided view, so a long clip does not cost a copy per frame.
    indices = np.arange(FRAME_SIZE)[None, :] + HOP_SIZE * np.arange(frames)[:, None]
    magnitude = np.abs(np.fft.rfft(wave[indices] * window, axis=1))
    return np.log(magnitude + _FLOOR)


def spectral_similarity(first: np.ndarray, second: np.ndarray, sample_rate: int) -> float:
    """Correlation of the two log spectrograms, in ``[-1, 1]``; NaN if undefined.

    Compared over the shared leading portion when lengths differ, because a codec may
    pad or trim by a frame.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        raise ValueError("spectral_similarity got an empty signal")

    shared = min(a.size, b.size)
    if shared < FRAME_SIZE:
        raise ValueError(
            f"signals are too short to analyse: {shared} samples against a "
            f"{FRAME_SIZE}-sample frame"
        )

    first_spec = _log_spectrogram(a[:shared]).ravel()
    second_spec = _log_spectrogram(b[:shared]).ravel()
    if _is_flat(first_spec) or _is_flat(second_spec):
        return float("nan")
    return float(np.corrcoef(first_spec, second_spec)[0, 1])


def _is_flat(spectrogram: np.ndarray) -> bool:
    """Whether a spectrogram has no variance to correlate.

    Uses peak-to-peak rather than ``std() == 0``: on a large constant array the
    two-pass variance loses the cancellation and returns something like 3.6e-15, so the
    equality never fires and two silent clips score a perfect 1.0.
    """
    spread = float(np.ptp(spectrogram))
    return spread <= 1e-9 * max(1.0, abs(float(spectrogram.mean())))
