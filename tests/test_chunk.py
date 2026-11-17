"""Tests for geetha.text.chunk — splitting long prompts into renderable pieces.

This is a measuring instrument, not a convenience. B7 measured that 145 of 262 frozen
Sarvam prompts hit Indic-Speak's token cap and were cut off mid-sentence, which pushed
the reported CER from 36.64 to 61.57 and inverted the category ranking. Every published
TTS number depends on this module splitting the same way for every system we score, and
on it never losing a word.
"""
import pytest

from geetha.text.chunk import Chunk, split_for_synthesis


def _words(text: str) -> list[str]:
    return text.split()


def test_text_within_budget_is_returned_as_one_unforced_chunk():
    chunks = split_for_synthesis("வணக்கம் தமிழ்நாடு.", max_chars=100)
    assert chunks == [Chunk(text="வணக்கம் தமிழ்நாடு.", forced=False)]


def test_two_sentences_over_budget_split_at_the_sentence_boundary():
    text = "முதல் வாக்கியம். இரண்டாவது வாக்கியம்."
    chunks = split_for_synthesis(text, max_chars=20)
    assert [c.text for c in chunks] == ["முதல் வாக்கியம்.", "இரண்டாவது வாக்கியம்."]


def test_the_sentence_terminator_stays_with_its_sentence():
    # A chunk ending without its "?" would be read as a statement.
    # Budget fits the question (21 chars) but not the question plus the answer.
    chunks = split_for_synthesis("நீங்கள் வருகிறீர்களா? ஆம்.", max_chars=22)
    assert chunks[0].text.endswith("?")
    assert not chunks[0].forced


def test_a_split_at_a_sentence_boundary_is_not_marked_forced():
    chunks = split_for_synthesis("ஒன்று. இரண்டு.", max_chars=8)
    assert all(not c.forced for c in chunks)


def test_an_over_long_single_sentence_splits_at_a_comma_and_is_marked_forced():
    # No sentence boundary exists inside it, so the split is an imposed pause.
    text = "ரயில், பஸ், விமானம் ஆகியவை போக்குவரத்து சாதனங்கள்"
    chunks = split_for_synthesis(text, max_chars=20)
    assert len(chunks) > 1
    assert all(c.forced for c in chunks)
    # Fragments are re-packed up to the budget, so we impose as few pauses as the
    # budget allows rather than one per comma.
    assert chunks[0].text == "ரயில், பஸ்,"


def test_a_clause_longer_than_the_budget_splits_at_whitespace_never_mid_word():
    text = "ஒரு இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு"
    chunks = split_for_synthesis(text, max_chars=15)
    for chunk in chunks:
        for word in _words(chunk.text):
            assert word in text, f"{word!r} was invented by the splitter"


def test_a_single_token_longer_than_the_budget_is_emitted_whole_not_dropped():
    # A 60-character URL cannot be split without changing what is spoken; emitting it
    # over budget is honest, dropping it would silently delete reference text.
    url = "https://www.example.com/" + "a" * 60
    chunks = split_for_synthesis(url, max_chars=20)
    assert len(chunks) == 1
    assert chunks[0].text == url


def test_no_chunk_is_empty_or_whitespace_only():
    text = "ஒன்று.  \n\n  இரண்டு.   மூன்று."
    for chunk in split_for_synthesis(text, max_chars=10):
        assert chunk.text.strip() == chunk.text
        assert chunk.text


def test_every_word_survives_the_split_in_order():
    # The invariant that makes a CER number trustworthy: the splitter may move
    # whitespace, never a word.
    text = ("தமிழ்நாடு ஒரு மாநிலம், அதன் தலைநகரம் சென்னை. "
            "மக்கள் தொகை ஏழு கோடி; மொழி தமிழ். "
            "https://tn.gov.in என்ற முகவரியில் விவரங்கள் உள்ளன!")
    for budget in (10, 25, 40, 80, 500):
        chunks = split_for_synthesis(text, max_chars=budget)
        assert [w for c in chunks for w in _words(c.text)] == _words(text), budget


def test_chunks_respect_the_budget_whenever_a_break_point_allows_it():
    text = "ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது பத்து"
    for chunk in split_for_synthesis(text, max_chars=20):
        # Only an unbreakable token may exceed the budget, and this text has none.
        assert len(chunk.text) <= 20, chunk


def test_splitting_is_deterministic():
    text = "ஒன்று, இரண்டு, மூன்று. நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது."
    assert split_for_synthesis(text, max_chars=18) == split_for_synthesis(text, max_chars=18)


def test_an_ellipsis_is_not_split_into_separate_chunks():
    chunks = split_for_synthesis("அவர் வந்தார்... பிறகு சென்றார்.", max_chars=22)
    assert not any(c.text.startswith(".") for c in chunks)


def test_a_decimal_number_is_not_treated_as_a_sentence_boundary():
    # "25.5" must not become "25." + "5" — the normaliser already protects this form
    # for scoring, and the splitter must agree or the spoken text differs.
    chunks = split_for_synthesis("விலை 25.5 ரூபாய் ஆகும்.", max_chars=100)
    assert len(chunks) == 1


def test_empty_or_blank_text_yields_no_chunks():
    assert split_for_synthesis("", max_chars=50) == []
    assert split_for_synthesis("   \n ", max_chars=50) == []


def test_a_non_positive_budget_is_rejected():
    with pytest.raises(ValueError):
        split_for_synthesis("தமிழ்", max_chars=0)


def test_every_frozen_sarvam_prompt_splits_without_losing_a_word():
    # The real instrument against the real prompts: 262 frozen rows, up to 723 chars.
    import json
    from pathlib import Path

    prompts = Path(r"E:\Geetha\work\b7_prompts\prompts.json")
    if not prompts.exists():
        pytest.skip("frozen prompt set not on this machine")
    clips = json.loads(prompts.read_text(encoding="utf-8"))["clips"]
    assert clips
    for row in clips:
        chunks = split_for_synthesis(row["text"], max_chars=200)
        assert [w for c in chunks for w in _words(c.text)] == _words(row["text"]), row["id"]
