"""Error-rate metrics for Geetha v1.

Three rates, because Tamil needs them:

* ``wer``          -- whitespace tokens. The headline ASR number.
* ``cer``          -- Unicode codepoints. Comparable with other papers.
* ``grapheme_cer`` -- Tamil aksharas. A wrong vowel sign is one error against the
  akshara it sits on, not a fraction of a codepoint pair, so this is the rate that
  reflects what a Tamil listener would call a mistake.

Every rate is computed from summed edit counts over summed reference length
(runbook section 10). A mean of per-utterance rates is a different, worse number:
it lets a one-word utterance outweigh a hundred-word one.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

from geetha.text.normalise import norm_text

__all__ = ["ErrorCounts", "aggregate", "cer", "edit_counts", "graphemes", "grapheme_cer", "wer"]


@dataclass(frozen=True)
class ErrorCounts:
    """Edit counts for one or many utterances, plus the reference length they cover."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """Errors over reference length, or ``None`` when there is no reference.

        ``None`` rather than 0.0 or infinity: an empty reference has no defined
        rate, and silently returning a number would put a fake value in a table.
        """
        if self.reference_length == 0:
            return None
        return self.errors / self.reference_length


def aggregate(counts: Iterable[ErrorCounts]) -> ErrorCounts:
    """Sum counts across utterances so the rate is reference-length weighted."""
    total = ErrorCounts(0, 0, 0, 0)
    for c in counts:
        total = ErrorCounts(
            total.substitutions + c.substitutions,
            total.deletions + c.deletions,
            total.insertions + c.insertions,
            total.reference_length + c.reference_length,
        )
    return total


def edit_counts(hypothesis: Sequence[str], reference: Sequence[str]) -> ErrorCounts:
    """Levenshtein alignment of two token sequences, split into S/D/I.

    Deletion = a reference token the hypothesis is missing.
    Insertion = a hypothesis token with nothing behind it in the reference.
    """
    n, m = len(reference), len(hypothesis)
    # cell = (total_errors, subs, dels, ins); row index walks the reference.
    row: list[tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        previous, row = row, [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                row.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            dele = previous[j]
            ins = row[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                key=lambda cell: cell[0],
            )
            row.append(best)
    _, subs, dels, ins = row[m]
    return ErrorCounts(subs, dels, ins, n)


def graphemes(text: str) -> list[str]:
    """Split ``text`` into grapheme clusters (Tamil aksharas, Latin letters).

    A combining mark (Unicode category M*) attaches to the cluster before it. In
    Tamil that makes the pulli close its cluster -- ``ந்த`` is ``ந்`` + ``த`` --
    which matches how the syllables are spoken.
    """
    clusters: list[str] = []
    for char in unicodedata.normalize("NFC", text):
        if clusters and unicodedata.category(char).startswith("M"):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def wer(hypothesis: str, reference: str) -> ErrorCounts:
    """Word error counts over normalised whitespace tokens."""
    return edit_counts(norm_text(hypothesis).split(), norm_text(reference).split())


def cer(hypothesis: str, reference: str) -> ErrorCounts:
    """Character error counts over normalised codepoints, spaces included."""
    return edit_counts(list(norm_text(hypothesis)), list(norm_text(reference)))


def grapheme_cer(hypothesis: str, reference: str) -> ErrorCounts:
    """Character error counts over Tamil aksharas, spaces included."""
    return edit_counts(graphemes(norm_text(hypothesis)), graphemes(norm_text(reference)))
