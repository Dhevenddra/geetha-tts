"""Tests for geetha.text.normalise — the single definition of norm_text()."""
import unicodedata

from geetha.text.normalise import norm_text


def test_composed_and_decomposed_tamil_vowel_signs_normalise_to_the_same_string():
    # U+0BCA (TAMIL VOWEL SIGN O) decomposes to U+0BC6 + U+0BBE.
    composed = "கொ"          # கொ written with the single codepoint
    decomposed = "கொ"  # கொ written as two marks
    assert composed != decomposed
    assert norm_text(composed) == norm_text(decomposed)


def test_norm_text_output_is_nfc():
    out = norm_text("கொ")
    assert out == unicodedata.normalize("NFC", out)


def test_zero_width_joiners_are_removed():
    assert norm_text("தமி‍ழ்‌") == norm_text("தமிழ்")


def test_byte_order_mark_and_nbsp_are_removed():
    assert norm_text("﻿தமிழ் நாடு") == "தமிழ் நாடு"


def test_tamil_digits_become_ascii_digits():
    # ௧௨௩ -> 123, so a reference written in Tamil numerals matches one in ASCII.
    assert norm_text("௧௨௩") == "123"


def test_punctuation_is_removed():
    assert norm_text("வணக்கம், தமிழ்நாடு!") == "வணக்கம் தமிழ்நாடு"


def test_curly_quotes_and_dashes_are_removed():
    assert norm_text("“தமிழ்” — நாடு") == "தமிழ் நாடு"


def test_latin_script_is_lowercased_for_code_mixed_text():
    assert norm_text("Train number 12345 late") == "train number 12345 late"


def test_tamil_letters_are_untouched_by_case_folding():
    tamil = "அஆஇஈஉஊஎஏஐஒஓஔஃ"
    assert norm_text(tamil) == tamil


def test_grantha_letters_and_aytham_are_preserved():
    # These are standard Tamil for loanwords; folding them away would destroy
    # distinctions the ASR is scored on.
    assert norm_text("ஜஸ்ரீ ஷஹ ஃ") == "ஜஸ்ரீ ஷஹ ஃ"


def test_removing_punctuation_does_not_glue_words_together():
    assert norm_text("ரயில்,பஸ்") == "ரயில் பஸ்"


def test_apostrophes_are_deleted_not_spaced():
    # "don't" is one spoken word; splitting it would inflate WER by a token.
    assert norm_text("don't") == "dont"
    assert norm_text("don’t") == "dont"


def test_hyphen_separates_words():
    # Guard on existing behaviour: hyphens fall under the P* rule above.
    assert norm_text("code-mixed") == "code mixed"


def test_a_decimal_point_keeps_the_number_as_one_token():
    # "25.5" must not become "25 5": that is two tokens where speech has one, and
    # it inflates WER on the Sarvam numerics prompts.
    assert norm_text("25.5 சதவீதம்") == "25.5 சதவீதம்"


def test_a_time_colon_keeps_the_clock_time_as_one_token():
    assert norm_text("3:30 மணி") == "3:30 மணி"


def test_thousands_separators_are_deleted_so_the_number_matches_either_spelling():
    # Indian grouping: "1,00,000" and "100000" are the same number and must hash
    # and score alike.
    assert norm_text("ரூ 1,00,000 மட்டும்") == "ரூ 100000 மட்டும்"
    assert norm_text("100000") == norm_text("1,00,000")


def test_a_comma_between_words_still_separates_them():
    # The digit rule must not leak into ordinary punctuation.
    assert norm_text("ரயில்,பஸ்") == "ரயில் பஸ்"


def test_a_sentence_final_period_after_a_digit_is_still_removed():
    assert norm_text("மொத்தம் 25.") == "மொத்தம் 25"


def test_a_comma_after_a_digit_before_a_word_still_separates():
    assert norm_text("25, பின்னர்") == "25 பின்னர்"
