"""Batching training sequences (the pod trainer's collator).

Padding bugs are the classic silent trainer failure: the loss descends, the samples
are wrong, and nothing in the logs says so. The load-bearing test here is
`test_padding_in_the_labels_is_masked_not_filled_with_the_pad_id` -- padding labels with
the pad token teaches the model to emit padding, which at inference shows up as
clips that trail off into nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from geetha.data.sequence import IGNORE_INDEX, TrainingSequence
from geetha.train.collate import collate


def sequence(prompt: int, audio: int) -> TrainingSequence:
    """A sequence with `prompt` masked positions followed by `audio` supervised ones."""
    ids = list(range(1, prompt + audio + 1))
    labels = [IGNORE_INDEX] * prompt + ids[prompt:]
    return TrainingSequence(input_ids=ids, labels=labels)


PAD = 128_263  # <|pad|> from the token contract


class TestShape:
    def test_a_single_sequence_is_returned_unchanged(self):
        batch = collate([sequence(2, 3)], pad_id=PAD)
        assert batch.input_ids.tolist() == [[1, 2, 3, 4, 5]]
        assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1]]

    def test_sequences_are_padded_to_the_longest_in_the_batch(self):
        batch = collate([sequence(1, 2), sequence(2, 6)], pad_id=PAD)
        assert batch.input_ids.shape == (2, 8)
        assert batch.labels.shape == (2, 8)
        assert batch.attention_mask.shape == (2, 8)

    def test_padding_goes_on_the_right(self):
        # Causal LM training: left padding would shift every position relative to the
        # positions the model was trained with.
        batch = collate([sequence(1, 1), sequence(1, 3)], pad_id=PAD)
        assert batch.input_ids[0].tolist() == [1, 2, PAD, PAD]

    def test_an_empty_batch_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            collate([], pad_id=PAD)


class TestMasking:
    def test_padding_in_the_labels_is_masked_not_filled_with_the_pad_id(self):
        # Filling labels with PAD trains the model to emit padding, which sounds like
        # a clip trailing off into nothing.
        batch = collate([sequence(1, 1), sequence(1, 3)], pad_id=PAD)
        assert batch.labels[0].tolist()[-2:] == [IGNORE_INDEX, IGNORE_INDEX]
        assert PAD not in batch.labels[0].tolist()

    def test_the_attention_mask_is_zero_exactly_on_the_padding(self):
        batch = collate([sequence(1, 1), sequence(1, 3)], pad_id=PAD)
        assert batch.attention_mask[0].tolist() == [1, 1, 0, 0]
        assert batch.attention_mask[1].tolist() == [1, 1, 1, 1]

    def test_no_supervised_label_ever_sits_on_a_padded_position(self):
        batch = collate([sequence(2, 1), sequence(1, 9)], pad_id=PAD)
        supervised = batch.labels != IGNORE_INDEX
        assert not np.any(supervised & (batch.attention_mask == 0))

    def test_the_prompt_stays_masked_through_collation(self):
        batch = collate([sequence(4, 2)], pad_id=PAD)
        assert batch.labels[0].tolist()[:4] == [IGNORE_INDEX] * 4

    def test_the_supervised_token_count_survives_collation(self):
        sequences = [sequence(2, 3), sequence(1, 5)]
        before = sum(s.supervised_tokens for s in sequences)
        batch = collate(sequences, pad_id=PAD)
        assert int((batch.labels != IGNORE_INDEX).sum()) == before


class TestCounts:
    def test_the_batch_reports_how_many_tokens_are_padding(self):
        batch = collate([sequence(1, 1), sequence(1, 5)], pad_id=PAD)
        assert batch.padding_tokens == 4

    def test_a_uniform_batch_wastes_nothing(self):
        batch = collate([sequence(1, 3), sequence(1, 3)], pad_id=PAD)
        assert batch.padding_tokens == 0
