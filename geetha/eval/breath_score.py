r"""Apply the breath instrument (D-030) to generated audio.

``geetha.audio.breath`` scores a pause that forced alignment has already located. A
generated clip has no alignment, so here pauses are located by energy: a run of frames
at least ``PAUSE_BELOW_SPEECH_DB`` under the clip's own speech level, inside the
instrument's duration band. Each is then scored by the saved detector.

Limits, to be printed beside any number from here:
  * an energy-located pause is not an aligned one: a loud breath can be missed as "not
    a pause", and a weak consonant counted as one;
  * the detector was fitted on real telephone speech, and its false-positive rate on
    synthetic audio is not established. Always compare against a no-breath arm of the
    same voice, never read a count on its own;
  * a pause quieter than ``AUDIBLE_ABOVE_SPEECH_DB`` is silence whatever its spectrum
    looks like - digital silence has a "breath-like" flat spectrum.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from geetha.audio.breath import (BAND, HOP_SECONDS, SAMPLE_RATE, BreathDetector,
                                 frame_features, gap_features)

__all__ = ["AUDIBLE_ABOVE_SPEECH_DB", "PAUSE_BELOW_SPEECH_DB", "locate_pauses",
           "resample_to_instrument", "score_wave"]

PAUSE_BELOW_SPEECH_DB = 15.0
AUDIBLE_ABOVE_SPEECH_DB = -50.0


def resample_to_instrument(wave: np.ndarray, rate: int) -> np.ndarray:
    """Band-limited resampling to the instrument's 16 kHz by truncating the spectrum.

    numpy only, exact for a whole clip; the instrument looks no higher than 6 kHz.
    """
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if rate == SAMPLE_RATE or not len(wave):
        return wave
    target = int(round(len(wave) * SAMPLE_RATE / rate))
    spectrum = np.fft.rfft(wave.astype(np.float64))
    if target // 2 + 1 > len(spectrum):
        spectrum = np.pad(spectrum, (0, target // 2 + 1 - len(spectrum)))
    return (np.fft.irfft(spectrum[: target // 2 + 1], n=target)
            * (target / len(wave))).astype(np.float32)


def locate_pauses(frames: Mapping[str, np.ndarray]) -> tuple[list[tuple[float, float, str]], float]:
    """In-band runs of non-speech frames as (start, end, lead|mid|tail), and the speech level."""
    energy = frames["energy_db"]
    speech = float(np.percentile(energy, 90))
    quiet = energy < speech - PAUSE_BELOW_SPEECH_DB
    pauses: list[tuple[float, float, str]] = []
    start = None
    for i, flag in enumerate(np.append(quiet, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            begin, end = start * HOP_SECONDS, i * HOP_SECONDS
            if BAND[0] <= end - begin <= BAND[1]:
                where = "lead" if start == 0 else ("tail" if i >= len(quiet) else "mid")
                pauses.append((begin, end, where))
            start = None
    return pauses, speech


def score_wave(wave: np.ndarray, rate: int, detector: BreathDetector,
               threshold: float) -> dict:
    """Every located pause of one clip with its breath score, and the count over threshold."""
    frames = frame_features(resample_to_instrument(wave, rate))
    if not frames:
        return {"seconds": 0.0, "pauses": [], "breaths": 0, "max_score": None}
    pauses, speech = locate_pauses(frames)
    energy = frames["energy_db"]
    out = []
    for begin, end, where in pauses:
        vector = gap_features(frames, begin, end)
        if vector is None:
            continue
        first, last = int(begin / HOP_SECONDS), int(end / HOP_SECONDS) + 1
        level = float(energy[first:last].mean() - speech)
        score = detector.score(vector)
        out.append({"at_s": round(begin, 2), "dur_s": round(end - begin, 2), "where": where,
                    "score": round(score, 3), "level_db_re_speech": round(level, 1),
                    "breath": bool(score >= threshold and level >= AUDIBLE_ABOVE_SPEECH_DB)})
    audible = [p["score"] for p in out if p["level_db_re_speech"] >= AUDIBLE_ABOVE_SPEECH_DB]
    return {"seconds": round(len(energy) * HOP_SECONDS, 2), "pauses": out,
            "breaths": sum(p["breath"] for p in out),
            "max_score": max(audible, default=None)}
