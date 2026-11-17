"""Tests for geetha.text.lexicon — applying the harvested lexicon to bare-Latin refs.

SPRING-INX test references write English in Latin with no Tamil form ("okay ma'am").
DiarBench gives us 3,206 Latin-to-Tamil mappings for free, so those bare Latin tokens
become {latin|native} alternations and an ASR that outputs Tamil script is no longer
marked wrong for a script choice a Tamil transcriber would also have made.
"""
import pytest

from geetha.scoring.dual_script import score_dual
from geetha.text.lexicon import MAX_ALTERNATIVES, apply_lexicon, filter_lexicon

LEXICON = {
    "okay": ["ஓகே", "ஒகே", "ஓக்கே"],
    "time": ["டைம்"],
    "so": ["சோ", "ஸோ"],
}


def test_a_known_latin_word_gains_its_tamil_spellings():
    assert apply_lexicon("okay nanba", LEXICON) == "{okay|ஓகே|ஒகே|ஓக்கே} nanba"


def test_an_unknown_latin_word_is_left_alone():
    assert apply_lexicon("xyzzy nanba", LEXICON) == "xyzzy nanba"


def test_tamil_text_is_untouched():
    assert apply_lexicon("வணக்கம் நண்பா", LEXICON) == "வணக்கம் நண்பா"


def test_the_latin_form_comes_first_so_the_original_reading_is_preserved():
    assert apply_lexicon("time", LEXICON).startswith("{time|")


def test_matching_is_case_insensitive_but_keeps_the_original_casing():
    assert apply_lexicon("Okay", LEXICON) == "{Okay|ஓகே|ஒகே|ஓக்கே}"


def test_punctuation_attached_to_a_word_stays_outside_the_alternation():
    assert apply_lexicon("okay, nanba", LEXICON) == "{okay|ஓகே|ஒகே|ஓக்கே}, nanba"


def test_only_a_few_spellings_are_offered_per_word():
    # The tail of rare spellings is where the annotation errors live, and every extra
    # reading can only make a reference more lenient.
    many = {"x": [f"த{i}" for i in range(10)]}
    assert apply_lexicon("x", many).count("|") == MAX_ALTERNATIVES


def test_max_alternatives_is_adjustable():
    assert apply_lexicon("okay", LEXICON, max_alternatives=1) == "{okay|ஓகே}"


def test_a_crowded_reference_still_scores():
    # Twelve code-mixed tokens, each with three Tamil spellings. The cartesian product
    # is 4**12, which is exactly why score_dual scores the lattice instead.
    crowded = apply_lexicon(" ".join(["okay"] * 12), LEXICON)
    assert score_dual(" ".join(["ஓகே"] * 12), crowded).counts.rate == 0.0


def test_a_mixed_script_hypothesis_scores_zero_on_a_crowded_reference():
    crowded = apply_lexicon("okay so time okay so time okay so time okay", LEXICON)
    hypothesis = "ஓகே so டைம் okay சோ time ஓகே so டைம் okay"
    assert score_dual(hypothesis, crowded).counts.rate == 0.0


def test_filter_lexicon_drops_rare_spellings():
    # The corpus contains annotation errors, e.g. ஓகே mistakenly tagged as "so".
    raw = {"so": ["சோ", "ஸோ", "ஓகே"]}
    counts = {("so", "சோ"): 177, ("so", "ஸோ"): 4, ("so", "ஓகே"): 1}
    assert filter_lexicon(raw, counts, min_count=2) == {"so": ["சோ", "ஸோ"]}


def test_filter_lexicon_drops_a_word_whose_only_spelling_is_rare():
    assert filter_lexicon({"zzz": ["ஜ"]}, {("zzz", "ஜ"): 1}, min_count=2) == {}


def test_applying_an_empty_lexicon_changes_nothing():
    assert apply_lexicon("okay nanba", {}) == "okay nanba"


def test_a_reference_already_containing_an_alternation_is_refused():
    with pytest.raises(ValueError, match="alternation"):
        apply_lexicon("{okay|ஓகே}", LEXICON)
