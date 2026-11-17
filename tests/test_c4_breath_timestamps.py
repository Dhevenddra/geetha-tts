"""Tests for C4's pure logic: where a breath mark sits, and how a word is timed.

These are the two places where a silent bug would corrupt the ground truth rather than
crash the run. ``tokenise`` decides which gap each breath belongs to, and ``word_spans``
decides where a word starts and ends; everything C4 reports is downstream of both.

The Kaggle script imports torch at module level inside ``main`` only, so the module is
importable here without the ML stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (Path(__file__).resolve().parents[1]
               / "kaggle" / "c4_breath_timestamps" / "c4_breath_timestamps.py")


def load_module():
    spec = importlib.util.spec_from_file_location("c4", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c4 = load_module()


class TestTokenise:
    def test_mark_between_words_names_the_gap_between_them(self):
        words, slots = c4.tokenise("alpha [breathing] beta")
        assert words == ["alpha", "beta"]
        assert slots == [(0, 1)]

    def test_non_breath_marks_are_dropped_without_claiming_a_gap(self):
        words, slots = c4.tokenise("alpha [uhh] beta [tsk] gamma")
        assert words == ["alpha", "beta", "gamma"]
        assert slots == []

    def test_a_sigh_counts_as_a_breath(self):
        _, slots = c4.tokenise("alpha [sigh] beta")
        assert slots == [(0, 1)]

    def test_case_is_ignored(self):
        _, slots = c4.tokenise("alpha [Breathing] beta")
        assert slots == [(0, 1)]

    def test_two_breaths_in_one_gap_are_both_recorded(self):
        words, slots = c4.tokenise("alpha [breathing] [breathing] beta")
        assert words == ["alpha", "beta"]
        assert slots == [(0, 1), (0, 1)]

    def test_breath_at_the_start_has_no_word_before_it(self):
        # -1 marks the missing side. main() drops these: an edge has no gap.
        words, slots = c4.tokenise("[breathing] alpha beta")
        assert words == ["alpha", "beta"]
        assert slots == [(-1, 0)]

    def test_breath_at_the_end_points_past_the_last_word(self):
        words, slots = c4.tokenise("alpha beta [breathing]")
        assert words == ["alpha", "beta"]
        assert slots == [(1, 2)]
        # index 2 is out of range for a 2-word utterance, which is how main() rejects it
        assert slots[0][1] == len(words)

    def test_noise_markers_around_the_utterance_do_not_shift_positions(self):
        words, slots = c4.tokenise(
            "[Persistent-noise-start] alpha [breathing] beta [Persistent-noise-end]")
        assert words == ["alpha", "beta"]
        assert slots == [(0, 1)]

    def test_a_breath_mark_interleaved_with_other_marks_keeps_its_slot(self):
        words, slots = c4.tokenise("alpha [uhh] [breathing] [tsk] beta")
        assert words == ["alpha", "beta"]
        assert slots == [(0, 1)]

    def test_real_tamil_line_from_the_corpus(self):
        # trimmed from raw\hf\indicvoices\tamil\train-00000-of-00118.parquet
        line = "வண்டில [tsk] இது [stammers] " \
               "சீதப் [breathing] அந்த"
        words, slots = c4.tokenise(line)
        assert len(words) == 4
        assert slots == [(2, 3)]


class TestWordSpans:
    def test_each_word_gets_the_frames_its_tokens_occupy(self):
        # two words of one token each, separated by a delimiter; blank is 0
        flat = [5, 9, 6]
        owner = [0, -1, 1]
        path = [0, 5, 5, 0, 9, 0, 6, 6, 0]
        spans = c4.word_spans(path, flat, owner, blank=0)
        assert spans == {0: [1, 2], 1: [6, 7]}

    def test_delimiter_frames_belong_to_no_word(self):
        flat = [5, 9, 6]
        owner = [0, -1, 1]
        path = [5, 9, 9, 9, 6]
        spans = c4.word_spans(path, flat, owner, blank=0)
        # frames 1-3 are the delimiter and are claimed by neither word, which is exactly
        # the gap C4 measures
        assert spans[0] == [0, 0]
        assert spans[1] == [4, 4]

    def test_a_multi_token_word_spans_all_of_its_tokens(self):
        flat = [5, 7]
        owner = [0, 0]
        path = [0, 5, 5, 7, 7, 0]
        spans = c4.word_spans(path, flat, owner, blank=0)
        assert spans == {0: [1, 4]}

    def test_blank_only_path_yields_no_words(self):
        assert c4.word_spans([0, 0, 0], [5], [0], blank=0) == {}

    def test_repeated_token_across_two_words_is_not_merged(self):
        # the same character ends one word and starts the next; walking `flat` in step is
        # what keeps them apart, so this is the regression that matters
        flat = [5, 9, 5]
        owner = [0, -1, 1]
        path = [5, 9, 5]
        spans = c4.word_spans(path, flat, owner, blank=0)
        assert spans == {0: [0, 0], 1: [2, 2]}


class TestThresholdsAreFixed:
    def test_pass_criteria_are_constants_not_computed(self):
        # the verdict must not be movable to fit the result it is judging
        assert c4.MIN_DURATION_LIFT == 1.25
        assert c4.MIN_IN_BAND == 0.50

    def test_breath_band_is_physiologically_plausible(self):
        assert 0.0 < c4.BREATH_GAP_MIN < 0.2
        assert 0.5 < c4.BREATH_GAP_MAX <= 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
