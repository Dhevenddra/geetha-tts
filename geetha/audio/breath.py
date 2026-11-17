r"""The breath instrument, as a module (D-030, outstanding item 1).

What it does
------------
Given a pause that has already been *located* -- by forced alignment, C4's job -- this
scores how likely it is to contain an inhalation. That ordering is the whole reason it
works. D-028 rejected four detectors that tried to count breaths in a whole clip; on the
800-clip calibration set, ``rho(clip duration, annotated breaths) = 0.633`` against the
best tagger's 0.595, so **a stopwatch beat every detector**. Classifying located pauses
instead reaches **AUC 0.881** on held-out speakers (0.8812 measured), against a
duration-only baseline of 0.824.

Why duration is structurally excluded
-------------------------------------
Pause duration alone scores AUC **0.824**, which is most of the headline -- and it is
exactly the feature we must not use. A TTS system *controls* how long its pauses are, so
a duration-aware detector would score a model that merely pauses longer as one that
breathes. D-030 excludes it, and this module makes that a construction error rather than
a matter of discipline: :class:`BreathDetector` refuses to be built, fitted, or loaded
with ``dur`` among its features, and ignores a ``dur`` key handed to ``score``.

``gap_features`` still *reports* duration, because a report needs it and because the
band it is checked against is published alongside any number from here.

Why this exists as a module and not the script it came from
-----------------------------------------------------------
The 28 Oct artefact (``work\breath_calibration\detectability.json``) recorded the eight
acoustic weights but **no bias and no normalisation statistics**, so the published model
could not actually be reproduced from it -- a result nobody could re-run is not a
result. :meth:`BreathDetector.to_dict` carries everything needed to score, and a test
asserts that it does.

Still owed before any breath rate is published (D-030): the false-positive rate in the
**synthetic** domain on Naturalness-Ta's 24 ``clean`` prompts, and recall correction with
a CI. This module is item 1 of three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "ACOUSTIC_FEATURES",
    "BAND",
    "BreathDetector",
    "DURATION_FEATURE",
    "HOP_SECONDS",
    "SAMPLE_RATE",
    "fit_detector",
    "frame_features",
    "gap_features",
]

SAMPLE_RATE = 16_000
WINDOW = 400          # 25 ms
HOP = 160             # 10 ms
HOP_SECONDS = HOP / SAMPLE_RATE

# A gap shorter than this cannot hold a breath; a gap longer than it is a turn boundary
# rather than an inhalation. Fixed before the measurement and published with any number.
BAND = (0.08, 1.50)

DURATION_FEATURE = "dur"
# The published feature set. Duration is absent by design, not by omission.
ACOUSTIC_FEATURES = ("e_max", "e_mean", "low", "mid", "high", "zcr", "flat", "cent")


def _reject_duration(names: Sequence[str]) -> None:
    if DURATION_FEATURE in tuple(names):
        raise ValueError(
            f"{DURATION_FEATURE!r} must not be a feature of a published breath detector "
            "(D-030): a TTS system controls its own pause lengths, so a duration-aware "
            "model would score a system that merely pauses longer as one that breathes. "
            f"Duration alone reaches AUC 0.824 on this data, which is why this is "
            "refused rather than discouraged."
        )


@dataclass(frozen=True)
class BreathDetector:
    """A fitted logistic model over pause acoustics.

    Carries its own normalisation, so a score is reproducible from the saved form alone.
    """

    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    deviation: np.ndarray

    def __post_init__(self) -> None:
        _reject_duration(self.feature_names)
        sizes = {len(self.feature_names), len(self.weights), len(self.mean),
                 len(self.deviation)}
        if len(sizes) != 1:
            raise ValueError(
                f"feature_names, weights, mean and deviation must agree in length, got "
                f"{len(self.feature_names)}, {len(self.weights)}, {len(self.mean)}, "
                f"{len(self.deviation)}"
            )

    def score(self, features: Mapping[str, float]) -> float:
        """Probability that this pause contains a breath.

        A ``dur`` key in ``features`` is ignored rather than rejected: callers naturally
        pass the whole dict from :func:`gap_features`, and the point is that duration
        cannot influence the answer.
        """
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise KeyError(f"breath score needs {missing}; got {sorted(features)}")
        vector = np.array([float(features[name]) for name in self.feature_names])
        normalised = (vector - self.mean) / self.deviation
        return float(1.0 / (1.0 + np.exp(-(normalised @ self.weights + self.bias))))

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "weights": [float(value) for value in self.weights],
            "bias": float(self.bias),
            "mean": [float(value) for value in self.mean],
            "deviation": [float(value) for value in self.deviation],
            "band": list(BAND),
            "excluded_features": [DURATION_FEATURE],
        }

    @classmethod
    def from_dict(cls, saved: Mapping[str, object]) -> "BreathDetector":
        names = tuple(saved["feature_names"])  # type: ignore[arg-type]
        _reject_duration(names)
        return cls(
            feature_names=names,
            weights=np.asarray(saved["weights"], dtype=float),
            bias=float(saved["bias"]),  # type: ignore[arg-type]
            mean=np.asarray(saved["mean"], dtype=float),
            deviation=np.asarray(saved["deviation"], dtype=float),
        )


def frame_features(wave: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame acoustics. Pure numpy, so the instrument has no model dependency."""
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    window = np.hanning(WINDOW).astype(np.float32)
    freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
    count = 1 + max(0, len(wave) - WINDOW) // HOP
    if len(wave) < WINDOW:
        return {}
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    frames = wave[index] * window
    magnitude = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) + 1e-10
    power = magnitude ** 2
    total = power.sum(1)

    def band(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        return power[:, mask].sum(1) / total

    raw = np.lib.stride_tricks.sliding_window_view(wave, WINDOW)[::HOP][:count]
    return {
        "energy_db": (10 * np.log10(total + 1e-12)).astype(np.float32),
        # geometric over arithmetic mean of the spectrum: high for noise, low for voicing
        "flat": (np.exp(np.log(power).mean(1)) / (power.mean(1) + 1e-20)).astype(np.float32),
        "centroid": ((power * freqs).sum(1) / total).astype(np.float32),
        "low": band(0, 300).astype(np.float32),        # f0 / voicing
        "mid": band(300, 1500).astype(np.float32),
        "high": band(1500, 6000).astype(np.float32),   # where breath turbulence shows
        "zcr": (np.diff(np.sign(raw), axis=1) != 0).mean(1).astype(np.float32),
    }


def gap_features(
    frames: Mapping[str, np.ndarray], start: float, end: float
) -> dict[str, float] | None:
    """Summarise one located pause, or ``None`` if it cannot be summarised.

    Energy is expressed relative to the clip's own 5th-percentile floor, so a loud
    recording and a quiet one are comparable.
    """
    if not frames:
        return None
    energy = frames["energy_db"]
    floor = float(np.percentile(energy, 5))
    first = max(0, int(start / HOP_SECONDS))
    last = min(len(energy), int(end / HOP_SECONDS) + 1)
    if last - first < 2:
        return None
    window = slice(first, last)
    return {
        DURATION_FEATURE: end - start,
        "e_max": float(energy[window].max() - floor),
        "e_mean": float(energy[window].mean() - floor),
        "low": float(frames["low"][window].mean()),
        "mid": float(frames["mid"][window].mean()),
        "high": float(frames["high"][window].mean()),
        "zcr": float(frames["zcr"][window].mean()),
        "flat": float(frames["flat"][window].mean()),
        "cent": float(frames["centroid"][window].mean()),
    }


def fit_detector(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    steps: int = 4_000,
    rate: float = 0.5,
    l2: float = 1e-3,
) -> BreathDetector:
    """Fit a logistic model, recording the normalisation it was fitted with.

    Plain gradient descent in numpy rather than a scikit-learn dependency: eight
    features and a few thousand rows do not justify one, and the project already keeps
    its scorers dependency-free.
    """
    _reject_duration(feature_names)
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"{features.shape[0]} rows of features against {labels.shape[0]} labels")
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"{features.shape[1]} feature columns against {len(feature_names)} names")

    mean = features.mean(axis=0)
    deviation = features.std(axis=0) + 1e-9
    normalised = (features - mean) / deviation

    weights = np.zeros(normalised.shape[1])
    bias = 0.0
    for _ in range(steps):
        prediction = 1.0 / (1.0 + np.exp(-(normalised @ weights + bias)))
        error = prediction - labels
        weights -= rate * (normalised.T @ error / len(labels) + l2 * weights)
        bias -= rate * error.mean()

    return BreathDetector(feature_names=tuple(feature_names), weights=weights, bias=bias,
                          mean=mean, deviation=deviation)
