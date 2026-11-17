r"""Naturalness-Ta, re-specified around D-020: what contamination cannot touch.

What changed and why
--------------------
The runbook's headline was a blind "which one is the real recording?" test against
held-out Rasa. B4a measured on 27 Oct that Rasa's test split is statistically
indistinguishable from its train split under the base model -- TEST's memorisation
margin came out *higher* than TRAIN's -- so those recordings cannot be assumed new to
the model. The contamination is in the **base** model, so holding sentences out of our
own fine-tune does not clean it, and IndicVoices-R (the runbook's fallback) gives
recordings of speakers we may neither render nor expose (D-008). There is no clean
version of that test available at any price.

So the headline moves to three claims that do not depend on a third party's data
hygiene, and the realness test survives as a *labelled secondary*.

The three components
--------------------
``CAPABILITY``
    Indic-Speak's card says: "No sound effects. No laughter, sighs, or breaths." If
    that holds (C2 is measuring it), Geetha breathing and hesitating where the base
    model cannot is a **binary** difference, audible in seconds, and immune to any
    argument about whose data was where. Measured as breath and filler events per
    minute, against the base model and the commercial APIs on the same prompts, plus
    a false-positive rate on prompts where no event is expected.

``DISTRIBUTION``
    TTSDS2 against real held-out Tamil. Distributional and **speaker-agnostic**, which
    is exactly why IV-R works here when it cannot work for the realness test: nobody
    has to render those voices, only match their statistics.

``REALNESS``
    The original test, kept, reported with its CI, and labelled: *the base model cannot
    be shown to have held out these recordings, so this is an upper bound.*

The speaker holdout
-------------------
TTSDS2's references must come from speakers that **no stage of training ever saw**.
IV-R has 1,052 Tamil speakers, so this is affordable in a way Rasa's two never were --
and it is the one place in this project where real speaker-disjointness is achievable
at all. Rasa cannot be speaker-disjoint with two speakers, and the model card has to
say so (R2).

Holdout is deterministic from the speaker id, so it can be recomputed on any machine
from the manifest rather than stored as a list that might drift from the ETL's idea of
it. The ETL reads :func:`is_held_out` and drops those speakers; the eval reads the same
function and keeps only those speakers. One rule, two consumers, no way for them to
disagree.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

__all__ = [
    "Component",
    "HOLDOUT_SALT",
    "HoldoutPlan",
    "MIN_CLIPS_PER_SPEAKER",
    "MIN_SECONDS_PER_SPEAKER",
    "REALNESS_LABEL",
    "SpeakerProfile",
    "is_held_out",
    "plan_holdout",
]

Component = Literal["CAPABILITY", "DISTRIBUTION", "REALNESS"]

# The wording that must accompany every published realness number. Kept in code rather
# than in a document, because a caveat that lives only in prose is one edit away from
# being dropped, and this one is the difference between an honest claim and a false one.
REALNESS_LABEL = (
    "Upper bound. The base model (bodhan-ai/indic-speak) cannot be shown to have held "
    "out these recordings: measured 27 Oct 2026, Rasa's test split is statistically "
    "indistinguishable from its train split under this model (B4a). Treat this as a "
    "ceiling on realism, not an estimate of it."
)

# Changing this re-draws the holdout and invalidates every number measured against the
# old one. It is a constant, not a parameter, for that reason.
HOLDOUT_SALT = "geetha-ivr-holdout-v1"

# A speaker with three clips cannot anchor a distributional comparison, and dropping
# them from training costs nothing. Both thresholds are deliberately modest: the point
# is to exclude the unusable, not to curate.
MIN_CLIPS_PER_SPEAKER = 20
MIN_SECONDS_PER_SPEAKER = 120.0


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    clips: int
    seconds: float
    gender: str | None = None
    district: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.clips >= MIN_CLIPS_PER_SPEAKER and self.seconds >= MIN_SECONDS_PER_SPEAKER


def is_held_out(speaker_id: str, fraction: float) -> bool:
    """Whether this speaker belongs to the evaluation holdout.

    Deterministic in the id alone, so ETL and eval cannot disagree about who is held
    out, and the split can be recomputed anywhere rather than shipped as a list.

    ``fraction`` is the share of speakers to hold out, in (0, 1).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
    digest = hashlib.sha256(f"{HOLDOUT_SALT}:{speaker_id}".encode("utf-8")).digest()
    # First eight bytes as a big-endian integer, scaled to [0, 1).
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < fraction


@dataclass(frozen=True)
class HoldoutPlan:
    held_out: tuple[SpeakerProfile, ...]
    trainable: tuple[SpeakerProfile, ...]
    excluded_as_too_small: tuple[SpeakerProfile, ...]

    @property
    def held_out_hours(self) -> float:
        return sum(s.seconds for s in self.held_out) / 3600.0

    @property
    def trainable_hours(self) -> float:
        return sum(s.seconds for s in self.trainable) / 3600.0

    def summary(self) -> str:
        return (
            f"{len(self.held_out)} speakers held out ({self.held_out_hours:.1f} h), "
            f"{len(self.trainable)} trainable ({self.trainable_hours:.1f} h), "
            f"{len(self.excluded_as_too_small)} too small to use"
        )


def plan_holdout(
    speakers: Iterable[SpeakerProfile], *, fraction: float = 0.08
) -> HoldoutPlan:
    """Split IV-R speakers into an evaluation holdout and a trainable remainder.

    Speakers below the usability thresholds are excluded from *both* sides. Leaving
    them in training would be harmless but leaving them in the holdout would not: a
    reference set built from speakers with a handful of clips each measures sampling
    noise as much as it measures anything about the voice.

    The default 8% of ~1,052 speakers is roughly 80 voices -- enough for a stable
    distributional comparison, cheap enough that training barely notices. It is a
    luxury IV-R affords and Rasa, with two speakers, never could.
    """
    held: list[SpeakerProfile] = []
    trainable: list[SpeakerProfile] = []
    too_small: list[SpeakerProfile] = []
    for speaker in sorted(speakers, key=lambda s: s.speaker_id):
        if not speaker.is_usable:
            too_small.append(speaker)
        elif is_held_out(speaker.speaker_id, fraction):
            held.append(speaker)
        else:
            trainable.append(speaker)
    if not held:
        raise ValueError(
            "the holdout came out empty — either the fraction is too small or no "
            "speaker clears the usability thresholds"
        )
    return HoldoutPlan(tuple(held), tuple(trainable), tuple(too_small))


def gender_balance(speakers: Sequence[SpeakerProfile]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for speaker in speakers:
        key = (speaker.gender or "unknown").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts
