r"""Length-bucketed batches under a padded-token budget.

The pilot trained one clip per forward pass. That is why it cost what it did: an A40
running a 3 B model on a single 800-token sequence is mostly idle, and every hour of it
is billed the same as a full one. Batching is the cheapest quality lever the project
has, because rupees buy gradient steps and this is how many steps a rupee buys.

Two things make batching audio-token sequences different from batching text:

* **Lengths span 30x** (154 to 2,554 tokens). A fixed batch *size* is therefore either
  wasteful on short clips or an out-of-memory error on long ones. The budget is on
  **padded tokens** -- ``rows * longest_row`` -- which is what the GPU actually holds.
* **Padding is paid for and learns nothing.** Sorting by length inside a shuffled pool
  keeps neighbours similar, so a batch pads by a few percent instead of by half.

Order is a pure function of ``(seed, epoch)``. That is what makes a run resumable from a
batch index: a pod that dies at batch 1,812 of epoch 1 restarts at batch 1,812 of the
same order, rather than re-reading clips it has seen and skipping ones it has not.

No clip is ever dropped here. One longer than the whole budget becomes a batch of one.
"""

from __future__ import annotations

import random
from typing import Sequence

__all__ = ["token_budget_batches", "padded_tokens"]


def padded_tokens(lengths: Sequence[int], batch: Sequence[int]) -> int:
    """What the GPU holds for ``batch``: rows times the longest row."""
    return len(batch) * max(lengths[i] for i in batch)


def token_budget_batches(
    lengths: Sequence[int],
    *,
    max_tokens: int,
    seed: int,
    epoch: int,
    max_rows: int = 64,
    pool: int = 4096,
    indices: Sequence[int] | None = None,
) -> list[list[int]]:
    """Every index exactly once, grouped so no batch pads past ``max_tokens``.

    ``indices`` selects (and may repeat) the examples for this epoch; by default it is
    every example once. Repeats are how Stage B upweights its emotional clips without
    storing them three times.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if pool <= 0:
        raise ValueError(f"pool must be positive, got {pool}")

    order = list(range(len(lengths))) if indices is None else list(indices)
    rng = random.Random(f"geetha-batches:{seed}:{epoch}")
    rng.shuffle(order)

    batches: list[list[int]] = []
    for start in range(0, len(order), pool):
        chunk = sorted(order[start:start + pool], key=lambda i: lengths[i])
        current: list[int] = []
        for index in chunk:
            # chunk is ascending, so the candidate is always the longest row so far.
            would_hold = (len(current) + 1) * lengths[index]
            if current and (would_hold > max_tokens or len(current) >= max_rows):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)

    # Shuffle whole batches, or every epoch would run short clips first and long last.
    rng.shuffle(batches)
    return batches
