r"""Per-clip audio QC -- ETL step 1 (runbook section 5).

What this measures and why each number is here
----------------------------------------------
A training manifest is only as good as the clips in it, and the failures that matter
are the ones that do not announce themselves:

* **clipping** -- IV-R is *enhanced* audio and sits hot (RMS near -15 dBFS). An
  enhancement model that drove a clip into the rails leaves flat-topped peaks that a
  vocoder learns to reproduce as distortion.
* **edge silence** -- a clip that opens with half a second of nothing teaches the model
  to begin by saying nothing. Measured at each end separately because the two have
  different causes: leading silence comes from segmentation, trailing from the speaker
  trailing off.
* **level** -- RMS in dBFS, so quiet and hot sources can be spotted before they are
  mixed into one manifest.
* **duration** -- re-derived from the samples rather than trusted from the metadata
  column, because the two disagreeing is itself a finding.

Silence is judged on short frames against an absolute floor, not on the clip's own
peak. A relative floor would rescale with the clip and call the quiet half of a quiet
recording "silence". The floor is deliberately well below breath: a breath frame lands
around -37 dBFS, so -45 dBFS keeps breath as signal. That matters because the breath
instrument (D-030) depends on breath surviving QC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["ClipQC", "clip_qc", "to_mono", "SILENCE_FLOOR_DBFS", "FRAME_SECONDS"]

# A breath frame measures about -37 dBFS. Stay clear below it.
SILENCE_FLOOR_DBFS = -45.0
FRAME_SECONDS = 0.010
# PCM_16 full scale, with a hair of tolerance for the float conversion.
CLIP_LEVEL = 0.999


@dataclass(frozen=True)
class ClipQC:
    """QC for one clip. ``SILENCE_DBFS`` stands in for log(0)."""

    SILENCE_DBFS = -120.0

    duration_s: float
    peak: float
    rms_dbfs: float
    clipped_samples: int
    clipped_fraction: float
    leading_silence_s: float
    trailing_silence_s: float
    is_silent: bool

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def to_mono(wave: np.ndarray) -> np.ndarray:
    """Average any channels down to one.

    IV-R ships 48 kHz stereo whose channels correlate at 0.9999 -- duplicates with
    dither, not a stereo image -- so averaging cannot comb-filter anything.
    """
    array = np.asarray(wave, dtype=np.float32)
    if array.ndim > 2:
        raise ValueError(f"to_mono expects 1 or 2 dimensions, got {array.ndim}")
    if array.ndim == 1 or array.shape[1] == 1:
        return array.reshape(-1)
    return array.mean(axis=1, dtype=np.float32)


def _frame_dbfs(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS of each whole frame, in dBFS. A trailing partial frame is dropped."""
    size = max(1, int(round(FRAME_SECONDS * sample_rate)))
    count = len(mono) // size
    if count == 0:
        # Clip shorter than one frame: judge it whole rather than report nothing.
        frames = mono.reshape(1, -1)
    else:
        frames = mono[: count * size].reshape(count, size)
    power = np.mean(np.square(frames, dtype=np.float64), axis=1)
    with np.errstate(divide="ignore"):
        return np.where(power > 0.0, 10.0 * np.log10(power), ClipQC.SILENCE_DBFS)


def clip_qc(wave: np.ndarray, sample_rate: int) -> ClipQC:
    """Measure one clip. Raises on an empty clip or a non-positive sample rate."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    mono = to_mono(wave)
    if mono.size == 0:
        raise ValueError("clip_qc got an empty clip")

    peak = float(np.abs(mono).max())
    power = float(np.mean(np.square(mono, dtype=np.float64)))
    rms_dbfs = 10.0 * np.log10(power) if power > 0.0 else ClipQC.SILENCE_DBFS

    clipped = int(np.count_nonzero(np.abs(mono) >= CLIP_LEVEL))

    quiet = _frame_dbfs(mono, sample_rate) < SILENCE_FLOOR_DBFS
    frame_s = max(1, int(round(FRAME_SECONDS * sample_rate))) / sample_rate
    leading = int(np.argmin(quiet)) if not quiet.all() else len(quiet)
    # Count from the end only once the clip has some signal in it, so an entirely
    # silent clip is reported as leading silence and not as both ends at once.
    trailing = 0 if quiet.all() else int(np.argmin(quiet[::-1]))

    return ClipQC(
        duration_s=len(mono) / sample_rate,
        peak=peak,
        rms_dbfs=rms_dbfs,
        clipped_samples=clipped,
        clipped_fraction=clipped / len(mono),
        leading_silence_s=leading * frame_s,
        trailing_silence_s=trailing * frame_s,
        is_silent=bool(quiet.all()),
    )
