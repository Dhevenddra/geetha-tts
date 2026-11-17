"""The contamination check's selection and its pre-registered verdict rule."""

from __future__ import annotations

import pytest

from geetha.eval.contamination import (
    MIN_CER_GAP,
    MIN_DELTA_GAP,
    Clip,
    memorisation_margin,
    naturalness_source,
    select_matched_arms,
    style_group,
)


def _clip(name: str, gender: str = "Female", style: str = "CONV", duration: float = 5.0) -> Clip:
    return Clip(filename=name, text="ஒரு வாக்கியம்", gender=gender, style=style, duration=duration)


def _pool(prefix: str, n: int, **kwargs) -> list[Clip]:
    return [_clip(f"{prefix}_{i:05d}", **kwargs) for i in range(n)]


class TestStyleGroups:
    def test_the_six_emotions_group_together(self):
        for label in ("ANGER", "SAD", "HAPPY", "FEAR", "SURPRISE", "DISGUST"):
            assert style_group(label) == "EMOTION"

    def test_conversation_is_its_own_group(self):
        # Naturalness-Ta draws on CONV, so it must never be merged into read speech.
        assert style_group("CONV") == "CONV"
        assert style_group("WIKI") == "READ"

    def test_an_unknown_label_does_not_explode(self):
        assert style_group("SOMETHING_NEW") == "OTHER"


class TestSelection:
    def test_both_arms_have_the_requested_size(self):
        test_arm, train_arm = select_matched_arms(_pool("TEST", 400), _pool("TRAIN", 4000), 50)
        assert len(test_arm) == 50
        assert len(train_arm) == 50

    def test_the_arms_are_matched_stratum_for_stratum(self):
        test_pool = _pool("TESTF", 200, gender="Female", style="CONV", duration=4.0)
        test_pool += _pool("TESTM", 100, gender="Male", style="ANGER", duration=8.0)
        train_pool = _pool("TRF", 2000, gender="Female", style="CONV", duration=4.0)
        train_pool += _pool("TRM", 1000, gender="Male", style="ANGER", duration=8.0)

        test_arm, train_arm = select_matched_arms(test_pool, train_pool, 60)

        def histogram(clips):
            counts: dict = {}
            for clip in clips:
                counts[clip.stratum] = counts.get(clip.stratum, 0) + 1
            return counts

        assert histogram(test_arm) == histogram(train_arm)

    def test_selection_is_deterministic_for_a_seed(self):
        first = select_matched_arms(_pool("T", 300), _pool("R", 900), 40, seed=7)
        second = select_matched_arms(_pool("T", 300), _pool("R", 900), 40, seed=7)
        assert [c.filename for c in first[0]] == [c.filename for c in second[0]]
        assert [c.filename for c in first[1]] == [c.filename for c in second[1]]

    def test_a_different_seed_picks_differently(self):
        a, _ = select_matched_arms(_pool("T", 300), _pool("R", 900), 40, seed=7)
        b, _ = select_matched_arms(_pool("T", 300), _pool("R", 900), 40, seed=8)
        assert [c.filename for c in a] != [c.filename for c in b]

    def test_the_train_arm_never_repeats_a_clip(self):
        # The shortfall path can reach into neighbouring bins; it must not hand back a
        # clip it has already used, which would silently shrink the control arm.
        thin_train = _pool("TR", 12, duration=4.0) + _pool("TRB", 200, duration=8.0)
        _, train_arm = select_matched_arms(_pool("T", 300, duration=4.0), thin_train, 50)
        assert len({c.filename for c in train_arm}) == len(train_arm)

    def test_an_empty_pool_is_refused(self):
        with pytest.raises(ValueError):
            select_matched_arms([], _pool("R", 10), 5)


class TestMemorisationMargin:
    def test_an_exact_reproduction_approaches_one(self):
        assert memorisation_margin([0.0] * 10, [1.0] * 50) == pytest.approx(1.0)

    def test_chance_pairing_scores_zero(self):
        assert memorisation_margin([1.0] * 10, [1.0] * 50) == pytest.approx(0.0)

    def test_a_worse_than_chance_pairing_goes_negative(self):
        assert memorisation_margin([2.0] * 10, [1.0] * 50) == pytest.approx(-1.0)

    def test_an_empty_null_is_refused(self):
        with pytest.raises(ValueError):
            memorisation_margin([1.0], [])


class TestVerdictRule:
    """The rule is fixed before results exist; these pin it so it cannot drift."""

    def test_clean_needs_both_gaps_material_and_significant(self):
        from geetha.eval.contamination import verdict_from_gaps

        assert verdict_from_gaps(4.0, (1.5, 6.0), 0.20, (0.08, 0.31)) == "CLEAN"

    def test_significant_but_trivial_gaps_are_not_clean(self):
        from geetha.eval.contamination import verdict_from_gaps

        gap = MIN_CER_GAP / 4
        delta = MIN_DELTA_GAP / 4
        assert verdict_from_gaps(gap, (0.1, 1.0), delta, (0.001, 0.02)) == "AMBIGUOUS"

    def test_both_cis_spanning_zero_is_contamination(self):
        from geetha.eval.contamination import verdict_from_gaps

        assert verdict_from_gaps(0.3, (-1.8, 2.4), 0.01, (-0.06, 0.08)) == "CONTAMINATED"

    def test_one_signal_only_is_ambiguous(self):
        from geetha.eval.contamination import verdict_from_gaps

        assert verdict_from_gaps(5.0, (2.0, 8.0), 0.01, (-0.05, 0.07)) == "AMBIGUOUS"

    def test_ambiguous_is_treated_as_contaminated_for_the_source(self):
        assert naturalness_source("AMBIGUOUS") == naturalness_source("CONTAMINATED")
        assert naturalness_source("CLEAN") == "rasa_test"
        assert naturalness_source("CONTAMINATED") == "indicvoices_r_heldout"
