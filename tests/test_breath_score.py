"""The breath instrument applied to generated audio: pause location and the silence guard."""

from __future__ import annotations

import numpy as np

from geetha.audio.breath import ACOUSTIC_FEATURES, SAMPLE_RATE, BreathDetector
from geetha.eval.breath_score import (AUDIBLE_ABOVE_SPEECH_DB, locate_pauses,
                                      resample_to_instrument, score_wave)
from geetha.audio.breath import frame_features


def _always_yes() -> BreathDetector:
    size = len(ACOUSTIC_FEATURES)
    return BreathDetector(feature_names=ACOUSTIC_FEATURES, weights=np.zeros(size), bias=10.0,
                          mean=np.zeros(size), deviation=np.ones(size))


def _tone(seconds: float, rate: int = SAMPLE_RATE, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _noise(seconds: float, level: float, rate: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * rate))
            ).astype(np.float32)


def test_resampling_keeps_duration_and_a_low_tone():
    wave = _tone(1.0, rate=24_000)
    out = resample_to_instrument(wave, 24_000)
    assert len(out) == SAMPLE_RATE
    assert abs(float(np.sqrt(np.mean(out ** 2))) - float(np.sqrt(np.mean(wave ** 2)))) < 0.01


def test_a_pause_between_two_words_is_located_as_mid():
    wave = np.concatenate([_tone(0.6), _noise(0.3, 0.003), _tone(0.6)])
    pauses, _ = locate_pauses(frame_features(wave))
    assert [where for _, _, where in pauses] == ["mid"]
    begin, end, _ = pauses[0]
    assert 0.5 < begin < 0.7 and 0.8 < end < 1.0


def test_an_audible_noisy_pause_counts_and_digital_silence_never_does():
    detector = _always_yes()
    audible = np.concatenate([_tone(0.6), _noise(0.3, 0.003), _tone(0.6)])
    silent = np.concatenate([_tone(0.6), np.zeros(int(0.3 * SAMPLE_RATE), np.float32),
                             _tone(0.6)])
    assert score_wave(audible, SAMPLE_RATE, detector, 0.5)["breaths"] == 1
    scored = score_wave(silent, SAMPLE_RATE, detector, 0.5)
    assert scored["breaths"] == 0
    assert scored["pauses"][0]["level_db_re_speech"] < AUDIBLE_ABOVE_SPEECH_DB
    assert scored["max_score"] is None


def test_an_empty_clip_scores_nothing():
    assert score_wave(np.zeros(0, np.float32), 24_000, _always_yes(), 0.5)["breaths"] == 0
