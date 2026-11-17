r"""Recover IndicVoices event marks for IV-R rows -- ETL step 3b (D-018).

The problem
-----------
IV-R is the restored, wideband release of IndicVoices: good acoustics, and the audio we
actually want to train a TTS model on. But its text columns (``text``, ``verbatim``,
``normalized``) carry **no event marks** -- no ``[breathing]``, no ``[uhh]``. The marks
exist only in IndicVoices' ``unsanitized_verbatim`` for the same utterance, whose audio
is 8 kHz telephone.

So the breath headline needs both halves of the same recording: IV-R for the sound,
IndicVoices for where the breaths are. They are joined on
``(speaker_id, norm_text(text))`` -- speaker included, because a sentence alone is not
unique across 2,407 speakers and a cross-speaker match would attach one person's breath
positions to another's audio.

Why ambiguity is counted rather than ignored
--------------------------------------------
A speaker can say two different things that normalise alike, and prompted corpora make
that likelier than it sounds: short commands and keyword-spotting tasks repeat. When a
key is claimed twice with *different* marks there is no way to know which recording the
IV-R row is, so the collision is counted and reported. The first entry wins, so a run
reproduces, but the ambiguity count is the honest measure of how much of the join is
guesswork -- and it belongs in the step 3b table beside the join rate.

Only *voiceable* events are counted. ``map_corpus_nonverbal`` returns ``None`` for
fillers (speech, not events) and for environment noise (removed, not voiced), and that
``None`` is honoured here rather than smuggled into a count.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from geetha.text.contract import map_corpus_nonverbal
from geetha.text.normalise import norm_text

__all__ = ["MarkIndex", "count_marks", "has_marks"]

# Either bracket shape the corpus uses, with the name inside.
_MARK = re.compile(r"[\[<]([^\[\]<>]{1,40})[\]>]")


class MarkIndex:
    """IndicVoices marked text, keyed by ``(speaker_id, norm_text(text))``."""

    def __init__(self, entries: Iterable[tuple[str, str, str | None]]) -> None:
        self._by_key: dict[tuple[str, str], str] = {}
        self.ambiguous = 0

        for speaker, text, verbatim in entries:
            if not verbatim:
                continue
            key = norm_text(text or "")
            if not key or not speaker:
                continue
            full_key = (str(speaker), key)
            existing = self._by_key.get(full_key)
            if existing is None:
                self._by_key[full_key] = verbatim
            elif existing != verbatim:
                # Same speaker, same normalised sentence, different marks: which
                # recording the IV-R row is cannot be known. Keep the first and say so.
                self.ambiguous += 1

    def lookup(self, speaker_id: str | None, text: str) -> str | None:
        key = norm_text(text or "")
        if not key or not speaker_id:
            return None
        return self._by_key.get((str(speaker_id), key))

    def __len__(self) -> int:
        return len(self._by_key)


def count_marks(verbatim: str | None) -> dict[str, int]:
    """Voiceable non-verbal events in ``verbatim``, counted by model token."""
    if not verbatim:
        return {}
    counts: Counter[str] = Counter()
    for name in _MARK.findall(verbatim):
        token = map_corpus_nonverbal(name)
        if token is not None:
            counts[token] += 1
    return dict(counts)


def has_marks(verbatim: str | None) -> bool:
    """Whether ``verbatim`` carries at least one voiceable event."""
    return bool(count_marks(verbatim))
