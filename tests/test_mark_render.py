"""ETL step 4: turning IndicVoices marked text into training text (D-035).

Each mark kind has a decided fate and they are not the same fate, so every kind gets a
test: voiceable events become model tokens, fillers become Tamil text, clicks and
environment noise vanish, and the three speech-bearing marks reject the whole row.

The property that matters most is the last one in this file: punctuation survives
untouched. `voices.md` says the model derives pauses and intonation from punctuation,
so a render that quietly tidied it would flatten the prosody of everything we train.
"""

from __future__ import annotations

import pytest

from geetha.data.mark_render import FILLER_TEXT, RejectedRow, render_training_text


class TestVoiceableEvents:
    def test_a_breath_mark_becomes_the_breath_token(self):
        assert render_training_text("வணக்கம் [breathing] நண்பரே") == "வணக்கம் <|nv_breath|> நண்பரே"

    def test_the_angle_bracket_spelling_is_treated_the_same(self):
        assert render_training_text("வணக்கம் <breathing> நண்பரே") == "வணக்கம் <|nv_breath|> நண்பரே"

    def test_mark_names_are_matched_case_insensitively(self):
        assert render_training_text("x [Breathing] y") == "x <|nv_breath|> y"

    def test_a_stammer_becomes_the_stammer_token_rather_than_a_literal_repeat(self):
        # D-035(b): the corpus writes no repeated syllable, so there is nothing to keep.
        assert render_training_text("எங்களுக்கு [stammers] வேல") == "எங்களுக்கு <|nv_stammer|> வேல"

    def test_laughter_and_throat_clearing_map_to_their_own_tokens(self):
        assert render_training_text("a [laughter] b [throat_clearing] c") == \
            "a <|nv_laugh|> b <|nv_throat|> c"


class TestFillers:
    def test_the_vowel_filler_becomes_its_attested_tamil_spelling(self):
        assert render_training_text("ஒரு நிமிடம் [uhh] நான்") == f"ஒரு நிமிடம் {FILLER_TEXT['uhh']} நான்"

    def test_the_nasal_filler_becomes_its_attested_tamil_spelling(self):
        assert render_training_text("சரி [umm] போ") == f"சரி {FILLER_TEXT['umm']} போ"

    def test_hmm_uses_the_same_nasal_spelling_as_umm(self):
        assert render_training_text("[hmm]") == FILLER_TEXT["umm"]

    def test_the_two_filler_classes_stay_distinct(self):
        # Székely et al. keep uh and um apart because a vowel and a nasal are not the
        # same sound; IndicVoices draws the same distinction and it is preserved.
        assert FILLER_TEXT["uhh"] != FILLER_TEXT["umm"]

    def test_no_bracketed_marker_survives_into_the_training_text(self):
        rendered = render_training_text("a [uhh] b [umm] c")
        assert "[" not in rendered and "]" not in rendered


class TestDiscardedMarks:
    def test_a_tsk_click_is_removed_and_the_row_survives(self):
        assert render_training_text("வண்டில [tsk] போறது") == "வண்டில போறது"

    def test_environment_noise_marks_are_removed(self):
        assert render_training_text("[Persistent-noise-start] a [noise] b [horn] c "
                                    "[Persistent-noise-end]") == "a b c"

    def test_removing_a_mark_does_not_leave_double_spaces(self):
        assert render_training_text("a [noise] b") == "a b"
        assert render_training_text("a  [noise]  b") == "a b"

    def test_an_unknown_bracketed_name_is_removed_rather_than_passed_through(self):
        assert render_training_text("a [some_new_mark_we_have_not_seen] b") == "a b"


class TestRowsThatMustBeRejected:
    @pytest.mark.parametrize("mark", ["unintelligible", "talking", "child_talking"])
    def test_a_speech_bearing_mark_rejects_the_whole_row(self, mark):
        # D-035(d): the audio holds words the text does not, so training on it teaches
        # the model to utter speech nobody asked for.
        with pytest.raises(RejectedRow, match=mark):
            render_training_text(f"a [{mark}] b")

    def test_rejection_beats_rendering_even_when_good_marks_are_present(self):
        with pytest.raises(RejectedRow):
            render_training_text("a [breathing] b [unintelligible] c")

    def test_a_row_left_with_no_words_is_rejected(self):
        with pytest.raises(RejectedRow, match="no_text"):
            render_training_text("[noise] [tsk]")

    def test_an_empty_input_is_rejected(self):
        with pytest.raises(RejectedRow, match="no_text"):
            render_training_text("")


class TestPunctuationIsLoadBearing:
    def test_commas_and_full_stops_are_preserved_exactly(self):
        source = "வணக்கம், நண்பரே. எப்படி இருக்கீங்க?"
        assert render_training_text(source) == source

    def test_punctuation_next_to_a_removed_mark_stays_where_it_was(self):
        assert render_training_text("வணக்கம், [noise] நண்பரே.") == "வணக்கம், நண்பரே."

    def test_a_token_does_not_swallow_the_comma_before_it(self):
        assert render_training_text("ஒரு நிமிடம், [breathing] நான்") == \
            "ஒரு நிமிடம், <|nv_breath|> நான்"


class TestMalformedCorpusBrackets:
    """IndicVoices contains nested and unclosed marks: `[breathing [noise]]`.

    Found on 12 of 31,033 joined rows, costing 6 real breath marks. Small, but an
    unrendered fragment puts literal ASCII into Tamil training text and throws away
    an event the breath headline is counting.
    """

    def test_an_unclosed_known_mark_still_becomes_its_token(self):
        assert render_training_text("a [breathing [noise]] b") == "a <|nv_breath|> b"

    def test_an_unclosed_filler_still_becomes_its_tamil_text(self):
        assert render_training_text("a [uhh [noise]] b") == f"a {FILLER_TEXT['uhh']} b"

    def test_an_unclosed_rejecting_mark_still_rejects_the_row(self):
        with pytest.raises(RejectedRow, match="talking"):
            render_training_text("a [talking [noise]] b")

    def test_a_leftover_bracket_character_is_removed(self):
        assert render_training_text("a ] b") == "a b"
        assert render_training_text("a [ b") == "a b"
        assert render_training_text("a > b") == "a b"

    def test_the_non_verbal_tokens_survive_the_bracket_cleanup(self):
        rendered = render_training_text("a [breathing] b [laughter] c")
        assert rendered == "a <|nv_breath|> b <|nv_laugh|> c"

    def test_the_real_corpus_row_that_exposed_this_renders_clean(self):
        source = ("[Persistent-noise-start] ஒரு [breathing] இரண்டு [stammers] மூன்று "
                  "[breathing [Persistent-noise-end]] ")
        rendered = render_training_text(source)
        assert rendered == "ஒரு <|nv_breath|> இரண்டு <|nv_stammer|> மூன்று <|nv_breath|>"
        assert "[" not in rendered and "]" not in rendered
