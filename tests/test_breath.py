"""The breath instrument as a module (D-030, outstanding item 1).

D-029 and D-030 left three things owed before any breath rate may be published. This
covers the first: the instrument must be a tested module, not an analysis script.

The test that matters most is the one that refuses pause **duration**. D-030 excludes it
deliberately: a TTS system controls how long its pauses are, so a duration-aware
detector would score a model that merely pauses longer as one that breathes. Duration
alone reaches AUC 0.824 on this data, so it is genuinely tempting and must be
structurally impossible rather than left to discipline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geetha.audio.breath import (
    ACOUSTIC_FEATURES,
    BreathDetector,
    DURATION_FEATURE,
    fit_detector,
    frame_features,
    gap_features,
)


def noise(seconds: float = 2.0, rate: int = 16_000, amplitude: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (amplitude * rng.standard_normal(int(seconds * rate))).astype(np.float32)


def detector(**overrides) -> BreathDetector:
    names = tuple(ACOUSTIC_FEATURES)
    spec = dict(
        feature_names=names,
        weights=np.linspace(0.5, 1.0, len(names)),
        bias=0.0,
        mean=np.zeros(len(names)),
        deviation=np.ones(len(names)),
    )
    spec.update(overrides)
    return BreathDetector(**spec)


class TestDurationIsStructurallyExcluded:
    def test_a_detector_naming_duration_is_refused(self):
        names = (DURATION_FEATURE, *ACOUSTIC_FEATURES)
        with pytest.raises(ValueError, match=DURATION_FEATURE):
            detector(feature_names=names, weights=np.ones(len(names)),
                     mean=np.zeros(len(names)), deviation=np.ones(len(names)))

    def test_the_published_feature_set_does_not_contain_duration(self):
        assert DURATION_FEATURE not in ACOUSTIC_FEATURES

    def test_fitting_refuses_duration_too(self):
        names = (DURATION_FEATURE, "high")
        with pytest.raises(ValueError, match=DURATION_FEATURE):
            fit_detector(np.zeros((4, 2)), np.array([0, 1, 0, 1]), names, steps=5)

    def test_scoring_ignores_a_duration_key_handed_in_by_a_caller(self):
        # A caller passing the whole gap dict, duration included, must not change the
        # score -- otherwise duration leaks back in through the back door.
        values = {name: 0.3 for name in ACOUSTIC_FEATURES}
        plain = detector().score(values)
        with_duration = detector().score({**values, DURATION_FEATURE: 99.0})
        assert plain == pytest.approx(with_duration)


class TestScore:
    def test_a_score_is_a_probability(self):
        values = {name: 0.2 for name in ACOUSTIC_FEATURES}
        assert 0.0 <= detector().score(values) <= 1.0

    def test_a_zero_input_with_zero_bias_scores_one_half(self):
        assert detector().score({name: 0.0 for name in ACOUSTIC_FEATURES}) == pytest.approx(0.5)

    def test_raising_a_positively_weighted_feature_raises_the_score(self):
        low = {name: 0.0 for name in ACOUSTIC_FEATURES}
        high = {**low, ACOUSTIC_FEATURES[-1]: 3.0}
        assert detector().score(high) > detector().score(low)

    def test_normalisation_is_applied_before_the_weights(self):
        # With mean 1 and deviation 2, an input of 1.0 normalises to 0 and must give 0.5.
        model = detector(mean=np.ones(len(ACOUSTIC_FEATURES)),
                         deviation=2 * np.ones(len(ACOUSTIC_FEATURES)))
        assert model.score({name: 1.0 for name in ACOUSTIC_FEATURES}) == pytest.approx(0.5)

    def test_a_missing_feature_is_an_error_not_a_zero(self):
        values = {name: 0.1 for name in ACOUSTIC_FEATURES[:-1]}
        with pytest.raises(KeyError, match=ACOUSTIC_FEATURES[-1]):
            detector().score(values)


class TestSerialisation:
    def test_a_saved_detector_scores_identically_when_reloaded(self, tmp_path):
        model = detector(bias=-0.4, mean=np.linspace(0, 1, len(ACOUSTIC_FEATURES)),
                         deviation=np.linspace(1, 2, len(ACOUSTIC_FEATURES)))
        values = {name: 0.37 for name in ACOUSTIC_FEATURES}
        path = tmp_path / "breath.json"
        path.write_text(json.dumps(model.to_dict()), encoding="utf-8")

        reloaded = BreathDetector.from_dict(json.loads(path.read_text(encoding="utf-8")))

        assert reloaded.score(values) == pytest.approx(model.score(values))

    def test_the_saved_form_carries_everything_needed_to_score(self):
        # The 28 Oct artefact held weights but no bias and no normalisation, so the
        # published model could not be reproduced from it. That must not recur.
        saved = detector(bias=-1.0).to_dict()
        for key in ("feature_names", "weights", "bias", "mean", "deviation"):
            assert key in saved

    def test_loading_a_file_that_names_duration_is_refused(self):
        bad = detector().to_dict()
        bad["feature_names"] = [DURATION_FEATURE, *bad["feature_names"][1:]]
        with pytest.raises(ValueError, match=DURATION_FEATURE):
            BreathDetector.from_dict(bad)


class TestFeatures:
    def test_frame_features_returns_one_value_per_frame_for_every_band(self):
        # These are per-frame arrays; ACOUSTIC_FEATURES are the gap-level summaries
        # gap_features derives from them, which is a different vocabulary.
        frames = frame_features(noise(1.0))
        lengths = {name: len(values) for name, values in frames.items()}
        assert len(set(lengths.values())) == 1
        for name in ("energy_db", "flat", "centroid", "low", "mid", "high", "zcr"):
            assert name in frames

    def test_a_clip_shorter_than_one_frame_yields_nothing(self):
        assert frame_features(noise(0.001)) == {}

    def test_a_gap_inside_the_clip_yields_every_acoustic_feature(self):
        values = gap_features(frame_features(noise(2.0)), 0.5, 0.9)
        assert values is not None
        assert set(ACOUSTIC_FEATURES) <= set(values)

    def test_a_gap_carries_its_duration_for_reporting_but_not_for_scoring(self):
        values = gap_features(frame_features(noise(2.0)), 0.5, 0.9)
        assert values[DURATION_FEATURE] == pytest.approx(0.4, abs=0.02)

    def test_a_gap_beyond_the_clip_is_rejected(self):
        assert gap_features(frame_features(noise(1.0)), 5.0, 5.4) is None

    def test_a_gap_with_no_frames_in_it_is_rejected(self):
        frames = frame_features(noise(1.0))
        assert gap_features(frames, 0.5, 0.5) is None


class TestFitDetector:
    def test_fitting_separable_data_gives_a_detector_that_separates_it(self):
        rng = np.random.default_rng(3)
        names = ("high", "zcr")
        positive = rng.normal(2.0, 0.3, size=(60, 2))
        negative = rng.normal(-2.0, 0.3, size=(60, 2))
        features = np.vstack([positive, negative])
        labels = np.array([1] * 60 + [0] * 60)

        model = fit_detector(features, labels, names, steps=800)

        hits = [model.score(dict(zip(names, row))) for row in features]
        assert np.mean(np.array(hits)[labels == 1]) > np.mean(np.array(hits)[labels == 0])

    def test_the_fitted_detector_records_the_normalisation_it_was_fitted_with(self):
        features = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0], [4.0, 40.0]])
        model = fit_detector(features, np.array([0, 1, 0, 1]), ("high", "zcr"), steps=10)
        assert model.mean == pytest.approx(features.mean(axis=0))
        assert model.deviation[1] > model.deviation[0]
