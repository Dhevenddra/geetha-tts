"""ETL step 3b: recovering IndicVoices event marks for IV-R rows (D-018).

IV-R is restored wideband audio whose own text columns carry no event marks; the marks
live in IndicVoices' ``unsanitized_verbatim`` for the same utterance. The join key is
``(speaker_id, norm_text(text))``, and the risk worth testing is a key that is not
unique: if two different utterances by one speaker normalise alike, an index that
silently keeps the last one would attach the wrong breath positions to real audio and
nothing downstream would notice.
"""

from __future__ import annotations

from geetha.data.mark_join import MarkIndex, count_marks, has_marks


class TestMarkIndex:
    def test_finds_the_marked_verbatim_for_a_speaker_and_sentence(self):
        index = MarkIndex([("spk1", "வணக்கம் நண்பரே", "வணக்கம் [breathing] நண்பரே")])
        assert index.lookup("spk1", "வணக்கம் நண்பரே") == "வணக்கம் [breathing] நண்பரே"

    def test_matches_across_punctuation_and_spacing_differences(self):
        index = MarkIndex([("spk1", "வணக்கம், நண்பரே.", "வணக்கம் [breathing] நண்பரே")])
        assert index.lookup("spk1", "வணக்கம்  நண்பரே") is not None

    def test_the_same_sentence_from_a_different_speaker_does_not_match(self):
        index = MarkIndex([("spk1", "வணக்கம்", "வணக்கம் [breathing]")])
        assert index.lookup("spk2", "வணக்கம்") is None

    def test_an_unknown_sentence_returns_none(self):
        index = MarkIndex([("spk1", "வணக்கம்", "வணக்கம் [breathing]")])
        assert index.lookup("spk1", "வேறு ஒரு வாக்கியம்") is None

    def test_an_empty_sentence_is_never_indexed_or_matched(self):
        index = MarkIndex([("spk1", "", "[breathing]"), ("spk1", "  ", "[breathing]")])
        assert len(index) == 0
        assert index.lookup("spk1", "") is None

    def test_a_row_with_no_verbatim_is_not_indexed(self):
        assert len(MarkIndex([("spk1", "வணக்கம்", None)])) == 0

    def test_a_key_claimed_twice_with_conflicting_marks_is_counted_as_ambiguous(self):
        index = MarkIndex([
            ("spk1", "வணக்கம்", "வணக்கம் [breathing]"),
            ("spk1", "வணக்கம்.", "[uhh] வணக்கம்"),
        ])
        assert index.ambiguous == 1
        assert len(index) == 1

    def test_a_key_repeated_with_identical_marks_is_not_ambiguous(self):
        index = MarkIndex([
            ("spk1", "வணக்கம்", "வணக்கம் [breathing]"),
            ("spk1", "வணக்கம்.", "வணக்கம் [breathing]"),
        ])
        assert index.ambiguous == 0

    def test_an_ambiguous_key_resolves_to_the_first_entry_so_runs_reproduce(self):
        index = MarkIndex([
            ("spk1", "வணக்கம்", "first [breathing]"),
            ("spk1", "வணக்கம்", "second [uhh]"),
        ])
        assert index.lookup("spk1", "வணக்கம்") == "first [breathing]"


class TestCountMarks:
    def test_counts_a_voiceable_event_under_its_model_token(self):
        assert count_marks("வணக்கம் [breathing] நண்பரே") == {"<|nv_breath|>": 1}

    def test_counts_both_bracket_shapes_the_corpus_uses(self):
        assert count_marks("[breathing] x <breathing> y") == {"<|nv_breath|>": 2}

    def test_is_case_insensitive_because_the_corpus_is_not_consistent(self):
        assert count_marks("[Breathing] x [BREATHING]") == {"<|nv_breath|>": 2}

    def test_fillers_are_not_counted_as_voiceable_events(self):
        # "uhh" is speech, not an event: the closed token set has no filler token.
        assert count_marks("[uhh] வணக்கம் [umm]") == {}

    def test_environment_noise_is_not_counted(self):
        assert count_marks("[Persistent-noise-start] x [bird_squawk] y [noise]") == {}

    def test_counts_several_distinct_events_separately(self):
        counts = count_marks("[breathing] a [laughter] b [breathing] c [tsk]")
        assert counts == {"<|nv_breath|>": 2, "<|nv_laugh|>": 1}

    def test_text_with_no_marks_counts_nothing(self):
        assert count_marks("வணக்கம் நண்பரே") == {}
        assert count_marks(None) == {}


class TestHasMarks:
    def test_true_only_when_a_voiceable_event_is_present(self):
        assert has_marks("x [breathing] y")
        assert not has_marks("x [uhh] y")
        assert not has_marks("x [noise] y")
        assert not has_marks("plain text")
        assert not has_marks(None)
