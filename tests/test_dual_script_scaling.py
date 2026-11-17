"""Scoring references with many alternation sites.

Real DiarBench segments carry 14+ code-mixed words, and every one becomes an
alternation site. Expanding the full cartesian product is 2**14 variants and simply
fails, so scoring resolves each site independently instead: pick the reading that best
matches the hypothesis, site by site, which is linear in sites rather than exponential.

Observed 26 Oct 2026: a single DiarBench segment with 14 sites refused to expand.
"""
import random

from geetha.scoring.dual_script import expand_variants, parse_alternations, score_dual

CROWDED = (
    "இங்க {அட்வான்ஸ்|advance} {யு-எஸ்|us}அ {கம்பேர்|compare} பண்றோம் "
    "{எக்ஸாம்பிள்|example} {ஃபாரின்|foreign} {கண்ட்ரீஸ்|countries}ஆ {லெவல்|level}ல "
    "{ஏ-ஐ|ai} {சோ|so} {ஃபார்வேர்ட்|forward}ஆ {ரியல்|real} {டைம்|time} {நவ்|now}"
)


def test_a_reference_with_fourteen_sites_scores_without_blowing_up():
    result = score_dual("இங்க advance usஅ compare பண்றோம்", CROWDED)
    assert result.counts.errors >= 0  # it scored at all


def test_parse_alternations_splits_literals_from_choices():
    assert parse_alternations("a {b|c} d") == ["a ", ["b", "c"], " d"]


def test_parse_alternations_handles_a_reference_with_no_choices():
    assert parse_alternations("plain text") == ["plain text"]


def test_each_site_is_resolved_independently_so_a_mixed_hypothesis_scores_zero():
    # The real case this exists for: a transcriber writes some loanwords in Tamil and
    # some in Latin within one sentence. Both readings are correct, and the reference
    # must accept the mixture, not force one script for the whole utterance.
    reference = "{ஓகே|okay} {மேம்|ma'am} {டூ|two} நிமிஷம்"
    result = score_dual("ஓகே ma'am டூ நிமிஷம்", reference)
    assert result.counts.rate == 0.0


def test_the_chosen_variant_is_reported_as_the_tokens_actually_scored():
    # Normalised, not as spelled in the source: this is what the comparison really
    # used, which is what belongs in the released raw outputs. norm_text deletes the
    # apostrophe, so "ma'am" is scored as "maam".
    result = score_dual("ஓகே ma'am", "{ஓகே|okay} {மேம்|ma'am}")
    assert result.variant == "ஓகே maam"


def test_scoring_is_deterministic():
    first = score_dual("something else", CROWDED)
    second = score_dual("something else", CROWDED)
    assert (first.counts.errors, first.variant) == (second.counts.errors, second.variant)


def test_site_resolution_agrees_with_exhaustive_search_on_small_references():
    # Property check: where the cartesian product is still affordable, the cheap
    # algorithm must find the same best score as brute force.
    from geetha.scoring.metrics import wer

    rng = random.Random(1234)
    words = ["alpha", "beta", "gamma", "அது", "இது", "வந்து"]
    for _ in range(60):
        sites = rng.randint(1, 3)
        parts = []
        for _ in range(sites):
            options = rng.sample(words, rng.randint(2, 3))
            parts.append("{" + "|".join(options) + "}")
            parts.append(rng.choice(words))
        reference = " ".join(parts)
        hypothesis = " ".join(rng.sample(words, rng.randint(1, 5)))

        best_exhaustive = min(
            wer(hypothesis, variant).errors for variant in expand_variants(reference)
        )
        assert score_dual(hypothesis, reference).counts.errors == best_exhaustive


# --- A suffix glued to an alternation is part of the same word ----------------
# "டோன்(Tone)ல" becomes "{டோன்|tone}ல": the locative suffix ல is Tamil in both
# readings, so it sits outside the braces. It is still the SAME WORD, and
# tokenising the parts separately turns one word into two -- which inflated WER on
# 2,394 DiarBench annotations before this was caught.


def test_a_suffix_after_an_alternation_stays_in_the_same_word():
    result = score_dual("டோன்ல", "{டோன்|tone}ல")
    assert result.counts.reference_length == 1
    assert result.counts.rate == 0.0


def test_the_latin_reading_also_carries_the_suffix():
    assert score_dual("toneல", "{டோன்|tone}ல").counts.rate == 0.0


def test_a_prefix_before_an_alternation_stays_in_the_same_word():
    assert score_dual("ஆங்tone", "ஆங்{டோன்|tone}").counts.rate == 0.0


def test_reference_length_counts_words_not_parts():
    # Three words, one of which happens to carry an alternation plus a suffix.
    result = score_dual("ஒரு டோன்ல வந்து", "ஒரு {டோன்|tone}ல வந்து")
    assert result.counts.reference_length == 3


def test_round_tripping_every_reading_of_a_real_reference_scores_zero():
    # The property that matters for the benchmark: feeding back any single reading
    # of the reference must score exactly zero, whichever reading it is.
    reference = "{ஓகே|okay} {மேம்|ma'am} {டூ|two}-ல {கரெக்ட்|correct}னு"
    for reading in ("ஓகே மேம் டூ-ல கரெக்ட்னு", "okay ma'am two-ல correctனு"):
        assert score_dual(reading, reference).counts.rate == 0.0, reading


def test_character_level_scoring_also_respects_word_boundaries():
    from geetha.scoring.metrics import cer

    assert score_dual("டோன்ல", "{டோன்|tone}ல", metric=cer).counts.rate == 0.0
