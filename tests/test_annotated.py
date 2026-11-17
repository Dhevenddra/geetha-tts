"""Tests for geetha.text.annotated — parsing DiarBench's inline dual-script annotation.

DiarBench Tamil writes every English loanword as the Tamil-script form it was actually
spoken as, followed by the English in parentheses, and sometimes a Tamil case suffix
after the bracket:

    "நேச்சுரல்(Natural) டோன்(Tone)ல நார்மல்(Normal)ஆ"

That is a dual-script reference already, 12,813 pairs of it, so B3 harvests it rather
than transliterating anything. It converts to the {native|latin} alternation convention
that geetha.scoring.dual_script understands.

Parenthetical content is not always Latin: 1,190 of the pairs hold a numeral
(ஒரு(1), ரெண்டு(2), ஃபைவ்(5)), and a few hold 10,000 / 7:00 / 1/2.
"""
import pytest

from geetha.text.annotated import (
    AnnotatedSegment,
    canonical_tag,
    harvest_lexicon,
    parse_annotated,
)


def ref(text: str) -> str:
    return parse_annotated(text).reference


def test_a_loanword_becomes_a_native_latin_alternation():
    assert ref("ஃபர்ஸ்ட்(First) வந்து") == "{ஃபர்ஸ்ட்|first} வந்து"


def test_a_tamil_case_suffix_after_the_bracket_is_kept_outside_the_alternation():
    # "டோன்(Tone)ல" is the word "tone" plus the locative suffix ல. The suffix is
    # Tamil in both renderings, so it must not go inside the braces.
    assert ref("டோன்(Tone)ல") == "{டோன்|tone}ல"


def test_latin_is_lowercased_so_the_reference_is_canonical():
    assert ref("ஓகே(Okay)") == "{ஓகே|okay}"


def test_an_apostrophe_inside_the_latin_form_survives():
    assert ref("தட்ஸ்(That's)") == "{தட்ஸ்|that's}"


def test_hyphenated_multiword_loanwords_are_one_alternation():
    assert ref("ஃபிரெஞ்ச்-ஃப்ரைஸ்(French-Fries)") == "{ஃபிரெஞ்ச்-ஃப்ரைஸ்|french-fries}"


def test_a_numeral_annotation_becomes_an_alternation_too():
    # The word spoken is Tamil; the bracket gives the numeral. An ASR may output
    # either, and neither is wrong.
    assert ref("ஃபைவ்(5) ஆகுது") == "{ஃபைவ்|5} ஆகுது"
    assert ref("ஒரு(1)") == "{ஒரு|1}"


def test_a_grouped_number_keeps_its_digits_together():
    assert ref("பத்தாயிரம்(10,000)") == "{பத்தாயிரம்|10,000}"


def test_a_space_before_the_bracket_is_tolerated():
    # Observed in the real data: "ம்ம்ம் (Mmm)".
    assert ref("ம்ம்ம் (Mmm)") == "{ம்ம்ம்|mmm}"


def test_plain_tamil_with_no_annotation_is_unchanged():
    assert ref("வணக்கம் நண்பா") == "வணக்கம் நண்பா"


def test_the_segment_reports_the_pairs_it_found():
    segment = parse_annotated("நேச்சுரல்(Natural) டோன்(Tone)ல")
    assert segment.pairs == [("நேச்சுரல்", "natural"), ("டோன்", "tone")]


def test_misspelled_tags_are_canonicalised():
    # DiarBench contains <unintelligble>, <unintellegible> and <unitelligible>.
    for raw in ("unintelligible", "unintelligble", "unintellegible", "unitelligible"):
        assert canonical_tag(raw) == "unintelligible"
    assert canonical_tag("background speech") == "background_speech"
    assert canonical_tag("background_speech") == "background_speech"


def test_tags_are_removed_from_the_reference_and_reported_separately():
    segment = parse_annotated("அது மெயின்(main) <laughter> கான்செப்ட்(concept)")
    assert segment.reference == "அது {மெயின்|main} {கான்செப்ட்|concept}"
    assert segment.tags == ["laughter"]


def test_a_segment_that_is_only_a_tag_is_not_scorable():
    segment = parse_annotated("<unintelligible>")
    assert segment.scorable is False
    assert segment.reference == ""
    assert segment.tags == ["unintelligible"]


def test_a_segment_with_words_is_scorable():
    assert parse_annotated("வணக்கம்").scorable is True


def test_an_empty_segment_is_not_scorable():
    assert parse_annotated("").scorable is False


def test_source_text_containing_the_alternation_syntax_is_refused():
    # A stray brace or pipe would silently corrupt the alternation grammar and make
    # the reference score against nonsense.
    with pytest.raises(ValueError, match="alternation"):
        parse_annotated("வணக்கம் {நண்பா}")
    with pytest.raises(ValueError, match="alternation"):
        parse_annotated("வணக்கம் | நண்பா")


def test_harvest_lexicon_maps_latin_to_the_native_forms_seen():
    segments = [
        parse_annotated("ஓகே(Okay) ப்ரோ(bro)"),
        parse_annotated("ஒகே(okay)"),
    ]
    lexicon = harvest_lexicon(segments)
    # Both Tamil spellings of "okay" are real and both are kept, commonest first.
    assert lexicon["okay"] == ["ஓகே", "ஒகே"]
    assert lexicon["bro"] == ["ப்ரோ"]


def test_harvest_lexicon_orders_native_forms_by_frequency():
    segments = [parse_annotated("ஒகே(okay)")] * 3 + [parse_annotated("ஓகே(okay)")]
    assert harvest_lexicon(segments)["okay"] == ["ஒகே", "ஓகே"]


def test_an_annotated_segment_is_a_plain_dataclass():
    segment = AnnotatedSegment(reference="x", pairs=[], tags=[], scorable=True)
    assert segment.reference == "x"


# --- Findings from the real corpus, 26 Oct 2026 -------------------------------
# DiarBench mixes its conventions. Both of these are in the same file:
#   * nonverbal tags as <laughter> (13x) AND as [Laughter] (448x)
#   * dual-script annotation as ஓகே(okay) AND as ஓகே[okay]  (~833x)
# The bracket shape therefore does not tell you which is which -- only the word
# inside does. And 363 annotations have a ZWNJ wedged before the bracket.


def test_a_square_bracket_nonverbal_is_a_tag_not_a_word():
    # [Laughter] appears 448 times. Left alone it becomes the literal word
    # "laughter" in the reference, i.e. a word the ASR is scored for not saying.
    segment = parse_annotated("அது நல்லது [Laughter]")
    assert segment.reference == "அது நல்லது"
    assert segment.tags == ["laughter"]


def test_square_and_angle_forms_of_the_same_tag_agree():
    assert parse_annotated("அது [Unintelligible]").tags == ["unintelligible"]
    assert parse_annotated("அது <unintelligible>").tags == ["unintelligible"]


def test_the_full_diarbench_nonverbal_inventory_is_recognised():
    for name in ("vocalization", "stutters", "gasps", "coughing", "noise", "laughter"):
        segment = parse_annotated(f"வணக்கம் [{name}]")
        assert segment.tags == [name], name
        assert segment.reference == "வணக்கம்"


def test_tag_names_are_case_insensitive():
    assert parse_annotated("வணக்கம் [VOCALIZATION]").tags == ["vocalization"]


def test_a_square_bracket_english_word_is_a_dual_script_annotation():
    # [so] occurs 51 times, [and] 22, [trump] 7 -- these are loanword annotations
    # using square brackets instead of parentheses.
    assert ref("சோ[so] அது") == "{சோ|so} அது"
    assert ref("அண்ட்[and]") == "{அண்ட்|and}"


def test_a_zwnj_before_the_bracket_is_tolerated():
    # 363 annotations carry a zero-width non-joiner between the word and the
    # bracket, which silently defeated the pattern and left "(correct)" in the text.
    assert ref("கரெக்ட்‌(correct)") == "{கரெக்ட்|correct}"
    assert ref("ரூம்‌(room)") == "{ரூம்|room}"


def test_a_non_breaking_space_is_treated_as_a_space():
    assert ref("வணக்கம் நண்பா") == "வணக்கம் நண்பா"


def test_a_bracketed_english_word_with_no_tamil_before_it_keeps_the_word():
    # The brackets are annotation syntax; the word inside was still spoken.
    assert ref("[Trump] சொன்னார்") == "trump சொன்னார்"


def test_an_unknown_bracketed_token_is_not_silently_dropped():
    segment = parse_annotated("வணக்கம் [mystery]")
    assert "mystery" in segment.reference
    assert segment.tags == []


def test_more_than_one_space_before_the_bracket_is_tolerated():
    # "டூ  (to)" -- two spaces. Parentheses are only ever annotation in this corpus,
    # so the gap does not change the meaning.
    assert ref("டூ  (to)") == "{டூ|to}"


def test_mismatched_brackets_from_a_transcription_typo_still_parse():
    # "மீ (me]" occurs in DiarBench: an opening round bracket closed with a square
    # one. Left unparsed it puts the literal "(me]" into a scored reference.
    assert ref("மீ (me]") == "{மீ|me}"
    assert ref("மீ [me)") == "{மீ|me}"


def test_punctuation_between_the_word_and_the_bracket_is_tolerated():
    # "டீச்சரா?(teacher)-ரா" and "காலேஜில்.(college)-இல்" both occur.
    assert ref("டீச்சரா?(teacher)") == "{டீச்சரா|teacher}"
    assert ref("ரொம்ப ஸ்வீட். (sweet)") == "ரொம்ப {ஸ்வீட்|sweet}"


def test_a_hyphenated_tamil_suffix_after_the_bracket_is_kept():
    assert ref("டீச்சரா?(teacher)-ரா") == "{டீச்சரா|teacher}-ரா"
    assert ref("ஃபிஃப்டீன்(fifteen)-ல") == "{ஃபிஃப்டீன்|fifteen}-ல"
