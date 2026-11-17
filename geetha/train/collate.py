r"""Batch training sequences for the pod trainer.

Three choices here, each of which is a silent failure if made the other way:

* **Labels pad with ``IGNORE_INDEX``, never with the pad token.** Padding labels with
  the pad id teaches the model to *emit* padding, which at inference sounds like a clip
  trailing off into nothing. The loss would descend the whole time.
* **Padding goes on the right.** This is a causal LM; left padding shifts every
  position relative to the positions the base model was trained at.
* **The attention mask is zero exactly on the padding.** Not "roughly" -- a test asserts
  no supervised label ever lands on a masked position, because that combination trains
  the model on tokens it is simultaneously told not to look at.

Numpy rather than torch, so the batching logic is testable without a GPU or a 2.4 GB
wheel; the trainer converts to tensors at the point of use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geetha.data.sequence import IGNORE_INDEX, TrainingSequence

__all__ = ["Batch", "collate"]


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray

    @property
    def padding_tokens(self) -> int:
        """How much of this batch is padding -- the cost of mixing clip lengths."""
        return int((self.attention_mask == 0).sum())

    @property
    def supervised_tokens(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum())


def collate(sequences: Sequence[TrainingSequence], *, pad_id: int) -> Batch:
    """Pad ``sequences`` to the longest in the batch."""
    if not len(sequences):
        raise ValueError("collate got an empty batch")

    width = max(len(item.input_ids) for item in sequences)
    rows = len(sequences)

    input_ids = np.full((rows, width), pad_id, dtype=np.int64)
    labels = np.full((rows, width), IGNORE_INDEX, dtype=np.int64)
    attention_mask = np.zeros((rows, width), dtype=np.int64)

    for row, item in enumerate(sequences):
        length = len(item.input_ids)
        if length != len(item.labels):
            raise ValueError(
                f"sequence {row} has {length} input ids against {len(item.labels)} labels"
            )
        input_ids[row, :length] = item.input_ids
        labels[row, :length] = item.labels
        attention_mask[row, :length] = 1

    return Batch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
