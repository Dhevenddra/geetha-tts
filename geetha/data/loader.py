r"""Encoded shards -> training examples. The loader B10 exercises.

The tokenizer is passed in rather than imported, for two reasons: these tests run
offline, and the tokenizer is the thing that must be asserted separately
(:mod:`geetha.text.tokenizer_check`) rather than assumed by whatever happens to be on
the machine.

Three behaviours worth knowing about:

* **``repeat`` is applied here, not baked into the codes.** Stage B wants its CONV clips
  twice and its emotion clips three times. Storing them repeated would triple the shard
  and make the stored codes a function of the stage; applying it at load keeps one set
  of codes serving both stages.
* **A row that does not fit the context is skipped, never truncated.** Cutting an audio
  span mid-frame misaligns every frame after the cut.
* **A row whose audio span is malformed is skipped, and counted.** ``training_sequence``
  validates the band and the frame alignment; a loader that let those through would be
  feeding the model the wrong symbol for every frame of that clip.

The text reaches the tokenizer exactly as step 4 wrote it, non-verbal tokens and
punctuation included. Punctuation is load-bearing for the model's pauses, and the
``<|nv_breath|>`` tokens are the whole point of Stage A.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import pyarrow.parquet as pq

from geetha.data.sequence import TrainingSequence, training_sequence
from geetha.text.contract import SOURCE_LABELS, build_tts_prompt

__all__ = ["LoaderStats", "examples_from_shard", "prompt_text"]

EncodeText = Callable[[str], Sequence[int]]


@dataclass
class LoaderStats:
    rows: int = 0
    examples: int = 0
    skipped_too_long: int = 0
    skipped_malformed: int = 0


def prompt_text(*, speaker: str, style: str, text: str, env: str) -> str:
    """The conditioned prompt for one clip, ending at ``<|start_of_ai|>``.

    ``build_tts_prompt`` raises on an ``env`` outside :data:`SOURCE_LABELS`, which is
    deliberate: ``<|env>`` is open class in the tokenizer but the project keeps it to a
    fixed small set so the model can learn to separate the sources.
    """
    if env not in SOURCE_LABELS:
        raise ValueError(f"env {env!r} is not one of {SOURCE_LABELS!r}")
    return build_tts_prompt(speaker=speaker, style=style, text=text, source=env)


def examples_from_shard(
    path: Path | str,
    encode_text: EncodeText,
    *,
    max_length: int | None = None,
    honour_repeats: bool = True,
    stats: LoaderStats | None = None,
) -> Iterator[TrainingSequence]:
    """Yield one training example per row, repeated per the row's ``repeat``."""
    table = pq.read_table(path)
    for row in table.to_pylist():
        if stats is not None:
            stats.rows += 1
        prompt = prompt_text(speaker=row["speaker"], style=row["style"],
                             text=row["text"], env=row["env"])
        prompt_ids = list(encode_text(prompt))
        try:
            example = training_sequence(prompt_ids, row["tokens"], max_length=max_length)
        except ValueError as problem:
            # Skip rather than raise: one bad row must not stop a 204-shard run, and
            # training on it would be worse than losing it.
            if stats is not None:
                if "context" in str(problem):
                    stats.skipped_too_long += 1
                else:
                    stats.skipped_malformed += 1
            continue
        copies = max(1, row.get("repeat") or 1) if honour_repeats else 1
        for _ in range(copies):
            if stats is not None:
                stats.examples += 1
            yield example
