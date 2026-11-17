"""Split a prompt into pieces a TTS model can actually render in one pass.

Why this exists
---------------
Indic-Speak holds 2,048 tokens at ~82 tokens per second of audio, so a single pass is
about 20 usable seconds. The frozen Sarvam prompt set runs to 723 characters. B7 part 1
rendered all 262 prompts in one pass each and **145 of them were cut off at the cap**;
the reported CER was 61.57, the untruncated subset scored 36.64, and the category
ranking inverted -- ``high_quality`` looked like one of the worst categories when in
fact it was the best, because its prompts are long. The number measured our harness.

So chunking is not a convenience. It is part of the measuring instrument, and the rule
that follows is: **every system in a published table is chunked by this function with
the same budget**, including the paid APIs, which chunk internally whether we ask them
to or not.

What it does not fix
--------------------
A split inside a sentence inserts a pause that was not in the text. That costs nothing
for intelligibility (CER) but is a real penalty for naturalness, so prompts authored
for naturalness evaluation must fit a single pass and never need a forced split.
``Chunk.forced`` exists so a report can state how many prompts were affected instead of
leaving the reader to wonder.

A character budget is also only a proxy for duration. Measured over B7's 117
untruncated clips, Indic-Speak speaks Tamil at a median 11.7 characters per second but
as slowly as 2.7 when it spells an abbreviation out letter by letter. No single budget
covers both, which is why the caller must still check whether a rendered chunk hit the
cap and re-split it on a smaller budget when it did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Chunk", "split_for_synthesis"]

# A terminator counts as a sentence end only when whitespace follows it, so "25.5" and
# "tn.gov.in" stay intact. The same protection the normaliser gives the scored form.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")

# Clause separators keep company with the text on their left: a chunk must not open
# with a comma.
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """One renderable piece.

    ``forced`` is True when the piece is a fragment of a sentence rather than one or
    more whole sentences -- i.e. when we imposed a pause the writer did not write.
    """

    text: str
    forced: bool


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _by_words(text: str, max_chars: int) -> list[str]:
    """Greedy whitespace split. A word longer than the budget is emitted over budget.

    Dropping it would delete reference text and show up as a deletion the model never
    made; splitting it would change what is spoken. Over budget is the honest option.
    """
    pieces: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence at clauses first, then at whitespace."""
    pieces: list[str] = []
    for clause in _CLAUSE_BREAK.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            pieces.append(clause)
        else:
            pieces.extend(_by_words(clause, max_chars))
    return pieces


def split_for_synthesis(text: str, max_chars: int) -> list[Chunk]:
    """Split ``text`` into pieces of at most ``max_chars`` characters.

    Whole sentences are kept whole and packed together while they fit. A sentence that
    does not fit on its own is broken at clause boundaries, then at word boundaries, and
    every piece of it is marked ``forced``. Words are never split and never dropped:
    the words of the returned chunks are the words of the input, in order.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    text = _collapse(text)
    if not text:
        return []

    # (piece, is_fragment) in reading order.
    pieces: list[tuple[str, bool]] = []
    for sentence in _SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append((sentence, False))
            continue
        parts = _split_sentence(sentence, max_chars)
        # A sentence with no usable break point comes back whole; it was not fragmented.
        fragmented = len(parts) > 1
        pieces.extend((part, fragmented) for part in parts)

    chunks: list[Chunk] = []
    current = ""
    current_forced = False
    for piece, is_fragment in pieces:
        candidate = f"{current} {piece}" if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(Chunk(text=current, forced=current_forced))
            current, current_forced = piece, is_fragment
        else:
            current, current_forced = candidate, current_forced or is_fragment
    if current:
        chunks.append(Chunk(text=current, forced=current_forced))
    return chunks
