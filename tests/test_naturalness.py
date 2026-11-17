"""Naturalness-Ta's speaker holdout — the one split that must never leak."""

from __future__ import annotations

import pytest

from geetha.eval.naturalness import (
    MIN_CLIPS_PER_SPEAKER,
    MIN_SECONDS_PER_SPEAKER,
    REALNESS_LABEL,
    SpeakerProfile,
    gender_balance,
    is_held_out,
    plan_holdout,
)


def _speaker(index: int, clips: int = 50, seconds: float = 600.0, gender: str = "Female"):
    return SpeakerProfile(f"S{index:016d}", clips=clips, seconds=seconds, gender=gender)


def _population(n: int = 1000):
    return [_speaker(i, gender="Female" if i % 2 else "Male") for i in range(n)]


class TestHoldoutRule:
    def test_the_same_speaker_always_lands_the_same_side(self):
        # ETL and eval both call this. If it drifted, training data would leak into
        # the reference set and nothing would report an error.
        for i in range(50):
            first = is_held_out(f"S{i}", 0.1)
            assert is_held_out(f"S{i}", 0.1) is first

    def test_it_does_not_depend_on_iteration_order_or_a_seed(self):
        assert is_held_out("S123", 0.1) == is_held_out("S123", 0.1)

    def test_a_larger_fraction_is_a_superset(self):
        # Growing the holdout must never release a speaker back into training.
        for i in range(200):
            if is_held_out(f"S{i}", 0.05):
                assert is_held_out(f"S{i}", 0.20)

    def test_the_fraction_is_roughly_honoured(self):
        held = sum(is_held_out(f"S{i}", 0.10) for i in range(5000))
        assert 0.085 < held / 5000 < 0.115

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_a_nonsense_fraction_is_refused(self, bad):
        with pytest.raises(ValueError):
            is_held_out("S1", bad)


class TestPlanHoldout:
    def test_every_usable_speaker_lands_on_exactly_one_side(self):
        plan = plan_holdout(_population(), fraction=0.08)
        held = {s.speaker_id for s in plan.held_out}
        trainable = {s.speaker_id for s in plan.trainable}
        assert not held & trainable
        assert len(held) + len(trainable) == 1000

    def test_small_speakers_are_excluded_from_both_sides(self):
        population = _population(100) + [
            _speaker(900, clips=MIN_CLIPS_PER_SPEAKER - 1),
            _speaker(901, seconds=MIN_SECONDS_PER_SPEAKER - 1),
        ]
        plan = plan_holdout(population, fraction=0.1)
        excluded = {s.speaker_id for s in plan.excluded_as_too_small}
        assert len(excluded) == 2
        assert not excluded & {s.speaker_id for s in plan.held_out}
        assert not excluded & {s.speaker_id for s in plan.trainable}

    def test_hours_are_reported_for_both_sides(self):
        plan = plan_holdout(_population(200), fraction=0.1)
        assert plan.held_out_hours > 0
        assert plan.trainable_hours > plan.held_out_hours
        assert "held out" in plan.summary()

    def test_an_empty_holdout_is_refused(self):
        # A silently empty holdout would mean evaluating on training data.
        tiny = [_speaker(0, clips=1, seconds=1.0)]
        with pytest.raises(ValueError, match="came out empty"):
            plan_holdout(tiny, fraction=0.1)

    def test_the_split_is_stable_when_the_population_grows(self):
        # Adding speakers later must not reshuffle who was already held out.
        first = plan_holdout(_population(500), fraction=0.08)
        second = plan_holdout(_population(1000), fraction=0.08)
        held_first = {s.speaker_id for s in first.held_out}
        held_second = {s.speaker_id for s in second.held_out}
        assert held_first <= held_second

    def test_both_genders_survive_the_split(self):
        plan = plan_holdout(_population(1000), fraction=0.08)
        balance = gender_balance(plan.held_out)
        assert balance.get("Female", 0) > 0
        assert balance.get("Male", 0) > 0


class TestRealnessLabel:
    def test_the_caveat_names_the_measurement_that_forced_it(self):
        # A caveat living only in prose is one edit from being dropped.
        assert "B4a" in REALNESS_LABEL
        assert "upper bound" in REALNESS_LABEL.lower()
        assert "indic-speak" in REALNESS_LABEL.lower()
