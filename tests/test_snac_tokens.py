"""The SNAC 7-token frame flattening -- ETL step 6.

Pinned directly to `token_contract.md` §3, because this is the one place in the project
where being wrong is **silent**: the contract records that a base/layout mismatch is
"exactly how the gemma3 SNAC base mismatch produced undecodable checkpoints", and that
with the wrong layout "training runs, loss descends, and the model learns the wrong
symbol for every audio frame".

So the id range of every one of the seven frame positions is asserted against the
table in the contract, not against whatever this implementation happens to produce.
The subtlety the tests exist to lock down: the offset encodes **where in the frame** a
code sits, not which codebook it came from -- positions 1 and 4 are both `c1`, and
2, 3, 5, 6 are all `c2`.
"""

from __future__ import annotations

import pytest

from geetha.audio.snac_tokens import (
    TOKENS_PER_FRAME,
    drop_duplicate_frames,
    flatten_frames,
    unflatten_frames,
)
from geetha.text.contract import AUDIO_TOKEN_BASE_ID, CODEBOOK_SIZE

BASE = AUDIO_TOKEN_BASE_ID

# token_contract.md §3, verbatim: the inclusive id range of each frame position.
DOCUMENTED_RANGES = [
    (0, 128_266, 132_361),
    (1, 132_362, 136_457),
    (2, 136_458, 140_553),
    (3, 140_554, 144_649),
    (4, 144_650, 148_745),
    (5, 148_746, 152_841),
    (6, 152_842, 156_937),
]


class TestTheDocumentedLayout:
    def test_the_base_id_and_band_match_the_contract(self):
        assert BASE == 128_266
        assert BASE + TOKENS_PER_FRAME * CODEBOOK_SIZE - 1 == 156_937

    @pytest.mark.parametrize("position,low,high", DOCUMENTED_RANGES)
    def test_each_frame_position_lands_in_its_documented_id_range(self, position, low, high):
        # Code 0 sits at the bottom of the position's block, 4095 at the top.
        lowest = flatten_frames([0], [0, 0], [0, 0, 0, 0])[position]
        highest = flatten_frames([4095], [4095] * 2, [4095] * 4)[position]
        assert lowest == low
        assert highest == high


class TestFlattenFrames:
    def test_one_frame_becomes_seven_tokens(self):
        assert len(flatten_frames([1], [2, 3], [4, 5, 6, 7])) == TOKENS_PER_FRAME

    def test_the_interleave_order_is_the_documented_one(self):
        # frame i -> [ c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3] ]
        tokens = flatten_frames([10], [20, 21], [30, 31, 32, 33])
        assert tokens == [
            BASE + 0 * CODEBOOK_SIZE + 10,
            BASE + 1 * CODEBOOK_SIZE + 20,
            BASE + 2 * CODEBOOK_SIZE + 30,
            BASE + 3 * CODEBOOK_SIZE + 31,
            BASE + 4 * CODEBOOK_SIZE + 21,
            BASE + 5 * CODEBOOK_SIZE + 32,
            BASE + 6 * CODEBOOK_SIZE + 33,
        ]

    def test_two_frames_use_the_same_position_offsets_again(self):
        tokens = flatten_frames([1, 2], [0] * 4, [0] * 8)
        assert tokens[0] == BASE + 1
        assert tokens[TOKENS_PER_FRAME] == BASE + 2

    def test_a_c1_of_the_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="c1"):
            flatten_frames([1], [2], [4, 5, 6, 7])

    def test_a_c2_of_the_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="c2"):
            flatten_frames([1], [2, 3], [4, 5])

    def test_a_code_outside_the_codebook_is_rejected(self):
        with pytest.raises(ValueError, match="4096"):
            flatten_frames([CODEBOOK_SIZE], [0, 0], [0, 0, 0, 0])

    def test_a_negative_code_is_rejected(self):
        with pytest.raises(ValueError, match="range"):
            flatten_frames([-1], [0, 0], [0, 0, 0, 0])

    def test_an_empty_clip_produces_no_tokens(self):
        assert flatten_frames([], [], []) == []


class TestDropDuplicateFrames:
    def test_a_frame_repeating_the_previous_c0_is_dropped_whole(self):
        c0, c1, c2 = drop_duplicate_frames([5, 5], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8])
        assert c0 == [5]
        assert c1 == [1, 2]
        assert c2 == [1, 2, 3, 4]

    def test_distinct_consecutive_frames_are_both_kept(self):
        c0, _, _ = drop_duplicate_frames([5, 6], [0] * 4, [0] * 8)
        assert c0 == [5, 6]

    def test_only_consecutive_repeats_collapse(self):
        # 5, 6, 5 has no consecutive repeat, so nothing is dropped.
        c0, _, _ = drop_duplicate_frames([5, 6, 5], [0] * 6, [0] * 12)
        assert c0 == [5, 6, 5]

    def test_a_long_run_collapses_to_one_frame(self):
        c0, c1, c2 = drop_duplicate_frames([7] * 4, [0] * 8, [0] * 16)
        assert c0 == [7]
        assert len(c1) == 2 and len(c2) == 4

    def test_dedup_is_why_token_count_is_not_proportional_to_duration(self):
        # The contract says so explicitly; this records the consequence.
        c0, c1, c2 = drop_duplicate_frames([1, 1, 1], [0] * 6, [0] * 12)
        assert len(flatten_frames(c0, c1, c2)) == TOKENS_PER_FRAME


class TestRoundTrip:
    def test_flattening_and_unflattening_returns_the_original_codes(self):
        c0 = [1, 2, 3]
        c1 = [10, 11, 12, 13, 14, 15]
        c2 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        assert unflatten_frames(flatten_frames(c0, c1, c2)) == (c0, c1, c2)

    def test_a_token_outside_the_audio_band_fails_loudly(self):
        tokens = flatten_frames([1], [2, 3], [4, 5, 6, 7])
        tokens[0] = 127_000
        with pytest.raises(ValueError, match="audio band"):
            unflatten_frames(tokens)

    def test_a_token_count_that_is_not_a_whole_number_of_frames_is_rejected(self):
        with pytest.raises(ValueError, match="7"):
            unflatten_frames([BASE] * 8)

    def test_a_code_in_the_wrong_position_block_fails_loudly(self):
        # A position-0 slot holding a position-3 id means the stream is misaligned.
        tokens = flatten_frames([1], [2, 3], [4, 5, 6, 7])
        tokens[0] = BASE + 3 * CODEBOOK_SIZE
        with pytest.raises(ValueError, match="position 0"):
            unflatten_frames(tokens)
