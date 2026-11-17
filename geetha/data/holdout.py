r"""Which training rows must be dropped, and why -- ETL step 2 (runbook sections 5, 10).

Rule 3 of section 10 is a removal instruction for the ETL, not a licence to edit the
eval sets: any sentence appearing in a held-out split comes out of **every** training
manifest. On top of that, IV-R has a speaker-level holdout
(``eval_vault\naturalness_ta_v1\ivr_heldout_speakers.json``), and a speaker holdout is
stronger than a text holdout -- it protects the eval even for sentences the speaker
never happened to say twice.

Every drop carries a machine-readable reason, because "we removed 3,114 rows" is not an
auditable claim and "3,114 rows: 2,044 leak:directability, 812 speaker_held_out, ..." is.

The guard that matters
----------------------
``norm_text`` strips punctuation and folds whitespace, so a blank or punctuation-only
row in a held-out file normalises to the empty string. An index that stored that would
match the empty normalisation of *any* unusual row and quietly reject swathes of the
corpus while reporting perfect compliance. Empty normalisations are therefore never
stored and never matched -- and a training row that normalises to nothing is dropped on
its own ``empty_text`` reason rather than silently passing as clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable

from geetha.text.normalise import norm_text

__all__ = ["HoldoutIndex", "KeepDecision", "decide"]


class HoldoutIndex:
    """Normalised texts of every held-out sentence, and which set each came from."""

    def __init__(self, entries: Iterable[tuple[str, str]]) -> None:
        self._by_text: dict[str, str] = {}
        for source, text in entries:
            key = norm_text(text or "")
            if not key:
                continue  # never let an empty normalisation become a wildcard
            # First set to claim a text keeps it, so a reason string is reproducible
            # regardless of the order the files happen to be read in.
            self._by_text.setdefault(key, source)

    def source_of(self, text: str) -> str | None:
        key = norm_text(text or "")
        return self._by_text.get(key) if key else None

    def __len__(self) -> int:
        return len(self._by_text)


@dataclass(frozen=True)
class KeepDecision:
    keep: bool
    reason: str

    @classmethod
    def kept(cls) -> "KeepDecision":
        return cls(True, "")

    @classmethod
    def dropped(cls, reason: str) -> "KeepDecision":
        return cls(False, reason)


def decide(
    text: str,
    speaker_id: str | None,
    *,
    index: HoldoutIndex,
    held_out_speakers: Collection[str],
    small_speakers: Collection[str] = (),
) -> KeepDecision:
    """Keep or drop one training row, with the reason an audit will quote."""
    if speaker_id is None or not str(speaker_id).strip():
        return KeepDecision.dropped("no_speaker_id")
    speaker = str(speaker_id)
    # Speaker holdout first: it is a property of the split, so it must not be masked
    # by whatever the text happens to collide with.
    if speaker in held_out_speakers:
        return KeepDecision.dropped("speaker_held_out")
    if speaker in small_speakers:
        return KeepDecision.dropped("speaker_too_small")
    if not norm_text(text or ""):
        return KeepDecision.dropped("empty_text")
    source = index.source_of(text)
    if source is not None:
        return KeepDecision.dropped(f"leak:{source}")
    return KeepDecision.kept()
