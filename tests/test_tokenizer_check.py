"""Guard the tokenizer before compiling data or starting a run (token_contract.md §6).

The contract is blunt about why: compile data with one tokenizer and train with another
and "training runs, loss descends, and the model learns the wrong symbol for every audio
frame". It also names why a range check is not enough -- sibling tokenizers
``llama-3-audio-tokenizer`` (156,938), ``-tok_trimmed`` (156,942) and ``-style``
(156,952) all share audio base 128,266, so every range check passes while ``<|style>``
means something different in the data than in the model.

So the check is on the ids themselves, token by token. Verified against the real
``bodhan-ai/indic-speak`` ``tokenizer.json`` on 3 Nov 2026: 28,960 added tokens, vocab
156,960, zero mismatches.
"""

from __future__ import annotations

import pytest

from geetha.text.contract import (
    CONDITIONING_TOKENS,
    CONTROL_TOKENS,
    NONVERBAL_TOKENS,
    VOCAB_SIZE,
)
from geetha.text.tokenizer_check import (
    EXPECTED_ADDED_TOKENS,
    TokenizerMismatch,
    assert_tokenizer,
    tokenizer_mismatches,
)


def good_added() -> dict[str, int]:
    """An added-token map that agrees with the contract everywhere."""
    added = {**CONTROL_TOKENS, **CONDITIONING_TOKENS, **NONVERBAL_TOKENS}
    # The 28,672 SNAC ids carry no content string of their own in our records; the
    # real file lists them, so stand them in by id.
    added.update({f"<|audio_{index}|>": 128_266 + index for index in range(28_672)})
    # The real tokenizer has 288 non-SNAC added tokens and `contract.py` names only 29
    # of them -- the rest are reserved slots the project never refers to. Stand them in
    # so the count check sees a realistic tokenizer.
    unnamed = EXPECTED_ADDED_TOKENS - 28_672 - 29
    added.update({f"<|reserved_{index}|>": 128_000 + index for index in range(unnamed)})
    return added


class TestMismatches:
    def test_a_matching_tokenizer_reports_nothing(self):
        assert tokenizer_mismatches(good_added(), VOCAB_SIZE) == []

    def test_a_control_token_at_the_wrong_id_is_reported_by_name(self):
        added = good_added()
        added["<|start_of_speech|>"] = 999
        problems = tokenizer_mismatches(added, VOCAB_SIZE)
        assert any("<|start_of_speech|>" in problem for problem in problems)

    def test_a_style_wrapper_at_the_wrong_id_is_reported(self):
        # The exact sibling-tokenizer failure the contract calls out.
        added = good_added()
        added["<|style>"] = 156_952
        assert any("<|style>" in problem for problem in tokenizer_mismatches(added, VOCAB_SIZE))

    def test_a_missing_token_is_reported_rather_than_skipped(self):
        added = good_added()
        del added["<|nv_breath|>"]
        assert any("<|nv_breath|>" in problem for problem in tokenizer_mismatches(added, VOCAB_SIZE))

    def test_the_wrong_vocab_size_is_reported(self):
        problems = tokenizer_mismatches(good_added(), 156_938)
        assert any("vocab" in problem.lower() for problem in problems)

    def test_a_short_audio_band_is_reported(self):
        added = good_added()
        for index in range(100):
            del added[f"<|audio_{index}|>"]
        assert any("audio" in problem.lower() for problem in tokenizer_mismatches(added, VOCAB_SIZE))

    def test_every_mismatch_is_reported_not_just_the_first(self):
        added = good_added()
        added["<|start_of_speech|>"] = 1
        added["<|nv_laugh|>"] = 2
        problems = tokenizer_mismatches(added, 1)
        assert len(problems) >= 3

    def test_the_expected_token_count_matches_the_contract(self):
        # 28,672 SNAC + 288 non-SNAC, per §3.
        assert EXPECTED_ADDED_TOKENS == 28_960


class TestAssertTokenizer:
    def test_a_matching_tokenizer_passes_quietly(self):
        assert_tokenizer(good_added(), VOCAB_SIZE)

    def test_a_mismatch_raises_and_the_message_names_the_token(self):
        added = good_added()
        added["<|end_of_speech|>"] = 42
        with pytest.raises(TokenizerMismatch, match="<\\|end_of_speech\\|>"):
            assert_tokenizer(added, VOCAB_SIZE)

    def test_the_error_is_not_a_warning(self):
        # "Mismatch is a hard error, never a warning. A silent wrong-tokenizer run
        # costs a full training cycle."
        with pytest.raises(TokenizerMismatch):
            assert_tokenizer({}, 0)
