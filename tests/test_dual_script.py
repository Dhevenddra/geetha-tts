"""Tests for geetha.scoring.dual_script.

Tamil ASR references for code-mixed speech can legitimately be written two ways:
English in Latin ("train number") or transliterated ("டிரெயின் நம்பர்"). A reference
carries both as ``{native|latin}`` alternations and the scorer takes whichever the
hypothesis matches, so a correct transcription is not punished for a script choice.
"""
import pytest

from geetha.scoring.dual_script import expand_variants, score_dual
from geetha.scoring.metrics import cer


def test_a_reference_without_alternations_has_one_variant():
    assert expand_variants("வணக்கம் நண்பா") == ["வணக்கம் நண்பா"]


def test_one_alternation_expands_to_two_variants():
    assert expand_variants("{டிரெயின்|train} நம்பர்") == ["டிரெயின் நம்பர்", "train நம்பர்"]


def test_two_alternations_expand_to_four_variants_in_a_stable_order():
    assert expand_variants("{a|b} x {c|d}") == ["a x c", "a x d", "b x c", "b x d"]


def test_an_empty_option_makes_a_word_optional():
    assert expand_variants("ரயில் {வந்தது|}") == ["ரயில் வந்தது", "ரயில்"]


def test_too_many_alternations_raises_instead_of_exploding():
    with pytest.raises(ValueError, match="variants"):
        expand_variants(" ".join("{a|b}" for _ in range(20)))


def test_a_latin_hypothesis_scores_zero_against_a_dual_script_reference():
    result = score_dual("train number 12", "{டிரெயின்|train} {நம்பர்|number} 12")
    assert result.counts.rate == 0.0
    assert result.variant == "train number 12"


def test_a_native_script_hypothesis_scores_zero_against_the_same_reference():
    result = score_dual("டிரெயின் நம்பர் 12", "{டிரெயின்|train} {நம்பர்|number} 12")
    assert result.counts.rate == 0.0
    assert result.variant == "டிரெயின் நம்பர் 12"


def test_a_genuinely_wrong_word_still_counts_as_an_error():
    result = score_dual("train number 99", "{டிரெயின்|train} {நம்பர்|number} 12")
    assert result.counts.substitutions == 1


def test_the_first_variant_wins_a_tie_so_scoring_is_deterministic():
    assert score_dual("x", "{a|b}").variant == "a"


def test_dual_script_scoring_works_with_any_metric():
    result = score_dual("train", "{டிரெயின்|train}", metric=cer)
    assert result.counts.rate == 0.0
