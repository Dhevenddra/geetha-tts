"""Tests for geetha.scoring.metrics — WER, CER and Tamil grapheme CER."""
import pytest

from geetha.scoring.metrics import ErrorCounts, aggregate, cer, graphemes, grapheme_cer, wer


def test_identical_strings_score_zero():
    assert wer("வணக்கம் நண்பா", "வணக்கம் நண்பா").rate == 0.0


def test_one_substitution_in_four_words_is_a_quarter():
    counts = wer("ஒன்று இரண்டு மூன்று ஐந்து", "ஒன்று இரண்டு மூன்று நான்கு")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)
    assert counts.rate == pytest.approx(0.25)


def test_a_missing_word_is_a_deletion():
    counts = wer("ஒன்று மூன்று", "ஒன்று இரண்டு மூன்று")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 1, 0)


def test_an_extra_word_is_an_insertion():
    counts = wer("ஒன்று இரண்டு மூன்று", "ஒன்று மூன்று")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)


def test_scoring_normalises_both_sides():
    # Punctuation and case differences must not count as errors.
    assert wer("Train number 12, late!", "train number 12 late").rate == 0.0


def test_an_empty_reference_does_not_divide_by_zero():
    counts = wer("ஏதோ", "")
    assert counts.reference_length == 0
    assert counts.insertions == 1
    assert counts.rate is None


def test_tamil_vowel_sign_is_part_of_its_grapheme():
    assert graphemes("கொ") == ["கொ"]


def test_pulli_closes_a_grapheme_and_the_next_consonant_starts_a_new_one():
    assert graphemes("ந்த") == ["ந்", "த"]


def test_tamil_word_splits_into_aksharas():
    assert graphemes("தமிழ்") == ["த", "மி", "ழ்"]


def test_latin_text_graphemes_are_single_characters():
    assert graphemes("bus") == ["b", "u", "s"]


def test_grapheme_cer_charges_a_wrong_vowel_sign_as_one_whole_akshara():
    # "தமிழ்" is 5 codepoints but 3 aksharas. Swapping the vowel sign on the middle
    # akshara is one error out of three, not one out of five.
    assert grapheme_cer("தமுழ்", "தமிழ்").rate == pytest.approx(1 / 3)
    assert cer("தமுழ்", "தமிழ்").rate == pytest.approx(1 / 5)


def test_aggregate_weights_by_reference_length_not_by_utterance():
    # A one-word utterance scored wrong must not outweigh a long correct one.
    short_bad = ErrorCounts(substitutions=1, deletions=0, insertions=0, reference_length=1)
    long_good = ErrorCounts(substitutions=0, deletions=0, insertions=0, reference_length=99)
    assert aggregate([short_bad, long_good]).rate == pytest.approx(0.01)


def test_aggregate_of_nothing_has_no_rate():
    assert aggregate([]).rate is None
