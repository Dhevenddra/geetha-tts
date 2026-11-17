"""Deciding which training rows ETL step 2 must drop (runbook section 10, rule 3).

The dangerous failure here is not a missed leak, it is an over-broad match: a single
blank or punctuation-only row in a held-out set normalises to the empty string, and an
index that treats that as a member would silently reject most of the corpus while
reporting that it was faithfully honouring rule 3.
"""

from __future__ import annotations

import pytest

from geetha.data.holdout import HoldoutIndex, decide


class TestHoldoutIndex:
    def test_matches_on_normalised_text_not_on_the_raw_string(self):
        index = HoldoutIndex([("rasa_test", "புத்தகம் மேஜையில் உள்ளது.")])
        assert index.source_of("புத்தகம்   மேஜையில்  உள்ளது") == "rasa_test"

    def test_names_which_held_out_set_a_collision_came_from(self):
        index = HoldoutIndex([("rasa_test", "ஒன்று"), ("ivr_test", "இரண்டு")])
        assert index.source_of("இரண்டு") == "ivr_test"

    def test_text_that_is_in_no_held_out_set_returns_none(self):
        assert HoldoutIndex([("rasa_test", "ஒன்று")]).source_of("மூன்று") is None

    def test_an_empty_held_out_row_does_not_become_a_wildcard(self):
        index = HoldoutIndex([("rasa_test", ""), ("rasa_test", "   "), ("rasa_test", "...")])
        assert len(index) == 0
        assert index.source_of("எந்த வாக்கியமும்") is None
        assert index.source_of("") is None

    def test_the_first_set_to_claim_a_text_keeps_it_so_reasons_are_stable(self):
        index = HoldoutIndex([("ivr_test", "சரி"), ("rasa_test", "சரி")])
        assert index.source_of("சரி") == "ivr_test"
        assert len(index) == 1

    def test_counts_distinct_normalised_texts(self):
        index = HoldoutIndex([("a", "ஒன்று"), ("b", "ஒன்று."), ("c", "இரண்டு")])
        assert len(index) == 2


class TestDecide:
    def setup_method(self):
        self.index = HoldoutIndex([("rasa_test", "இது கசிந்த வாக்கியம்")])
        self.kwargs = dict(index=self.index, held_out_speakers={"spk_held"},
                           small_speakers={"spk_small"})

    def test_a_clean_row_from_a_trainable_speaker_is_kept(self):
        decision = decide("ஒரு புதிய வாக்கியம்", "spk_ok", **self.kwargs)
        assert decision.keep
        assert decision.reason == ""

    def test_a_leaked_row_is_dropped_and_the_reason_names_the_set(self):
        decision = decide("இது கசிந்த வாக்கியம்", "spk_ok", **self.kwargs)
        assert not decision.keep
        assert decision.reason == "leak:rasa_test"

    def test_a_held_out_speaker_is_dropped(self):
        assert decide("ஒரு புதிய வாக்கியம்", "spk_held", **self.kwargs).reason == "speaker_held_out"

    def test_a_speaker_with_too_little_audio_is_dropped(self):
        assert decide("ஒரு புதிய வாக்கியம்", "spk_small", **self.kwargs).reason == "speaker_too_small"

    def test_a_held_out_speaker_is_reported_as_such_even_when_the_text_also_leaks(self):
        # Speaker holdout is a property of the split and outranks a text collision,
        # so the audited reason stays stable however the text happens to fall.
        decision = decide("இது கசிந்த வாக்கியம்", "spk_held", **self.kwargs)
        assert decision.reason == "speaker_held_out"

    def test_a_row_with_no_text_is_dropped_rather_than_trained_on(self):
        decision = decide("   ", "spk_ok", **self.kwargs)
        assert not decision.keep
        assert decision.reason == "empty_text"

    def test_a_missing_speaker_id_is_dropped_rather_than_assumed_trainable(self):
        assert decide("ஒரு புதிய வாக்கியம்", None, **self.kwargs).reason == "no_speaker_id"
