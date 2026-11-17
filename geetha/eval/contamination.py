"""Did Indic-Speak train on the Rasa *test* split? — selection and the verdict rule.

Why this exists
---------------
Naturalness-Ta's headline is a blind "which one is the real recording?" test: our
rendering of a held-out Rasa sentence against the real Rasa recording of it. That claim
is only worth making if the base model has not already memorised those recordings.

What is already settled without a GPU (measured 27 Oct 2026, from the files):

* ``voices.md`` — the voice library is "the full Rasa studio roster", Tamil = Anitha /
  Arun, and the voices not covered by benchmarks are "measured from their studio source
  recordings";
* Rasa Tamil's ``style`` column holds exactly 16 labels (``CONV``, ``WIKI``, ``ANGER``,
  ``SAD``, ``HAPPY``, ``FEAR``, ``SURPRISE``, ``DISGUST``, ``ALEXA``, ``BOOK``, ``NEWS``,
  ``INDIC``, ``UMANG``, ``BB``, ``PROPER NOUN``, ``DIGI``) and Indic-Speak's README says
  its free-text ``style`` field takes "uppercase emotion labels (``ANGER``, ``FEAR``, …
  17 in total)". The model's style vocabulary *is* Rasa's style column;
* both come from AI4Bharat.

So "the model trained on Rasa" is not in doubt. The only open question is whether whoever
built it honoured Rasa's own train/test split. Nothing in the model card says.

The design
----------
Three arms, 50 clips each, and the one that makes this a measurement rather than a vibe
is the middle one:

``TRAIN``
    Rasa Tamil *train* — the **positive control**. Almost certainly trained on. This is
    what memorisation looks like on this model, this corpus, these two speakers.
``TEST``
    Rasa Tamil *test* — the arm under suspicion.
``UNSEEN``
    FLEURS Tamil test sentences — a loose anchor for "text from another corpus". Used for
    CER only: its speakers are not Anitha/Arun, so its recordings cannot anchor prosody.

TRAIN and TEST are matched arm-to-arm on speaker gender, style and clip duration, so the
*only* systematic difference between them is which side of Rasa's split they fall on.
That matched comparison is the whole test. A cross-corpus baseline could never be this
clean, because corpus, channel, speaker and text difficulty would all move at once.

The statistics
--------------
Per clip *i*, with real recording ``R_i`` and our rendering ``S_i`` of the same text,
speaker and style:

``cer_i``
    CER of ASR(``S_i``) against the reference text. Memorised text renders unusually
    cleanly.
``matched_i``
    Prosodic distance between ``S_i`` and ``R_i`` — the *true* take.
``mismatched_i``
    The same distance from ``S_i`` to each of K other real takes by the same speaker in
    the same style. A null distribution for "how close does this rendering land to a
    random take by this voice?".

From those, the **memorisation margin**::

    delta = (median(mismatched) - median(matched)) / median(mismatched)

the fraction by which the true pairing beats a random same-speaker pairing. Any
competent TTS scores delta > 0 — it is reading the same words, so the length and phrase
shape agree somewhat. The question is never "is delta positive" but "is TEST's delta as
large as TRAIN's".

The verdict rule — fixed here BEFORE any audio was generated
------------------------------------------------------------
Two paired-bootstrap contrasts, 95% CI (``geetha.scoring.bootstrap``):

1. ``cer_gap   = median(cer, TEST) - median(cer, TRAIN)``      — expect > 0 if clean
2. ``delta_gap = delta(TRAIN) - delta(TEST)``                  — expect > 0 if clean

``CLEAN``
    both gaps are material (``cer_gap >= MIN_CER_GAP``, ``delta_gap >= MIN_DELTA_GAP``)
    **and** both CIs exclude zero. The test split behaves like unseen data.
``CONTAMINATED``
    both CIs include zero. Test is indistinguishable from the positive control.
``AMBIGUOUS``
    anything else.

``AMBIGUOUS`` is treated exactly like ``CONTAMINATED`` by
:func:`naturalness_source`. A headline claim resting on a result we could not read is
the same mistake as one resting on a result we read wrong, and the project rules' honesty rule
settles which way to fail: change the claim, not the number.

If the source is not Rasa test, Naturalness-Ta is built from IndicVoices-R held-out
speakers instead (97.3 h, 1,052 speakers, already restored — D-018), and the eval report
says plainly why.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

__all__ = [
    "Arm",
    "ArmStats",
    "Clip",
    "MIN_CER_GAP",
    "MIN_DELTA_GAP",
    "Verdict",
    "memorisation_margin",
    "naturalness_source",
    "select_matched_arms",
    "verdict_from_gaps",
]

Arm = Literal["TRAIN", "TEST", "UNSEEN"]
Verdict = Literal["CLEAN", "CONTAMINATED", "AMBIGUOUS"]

# A difference smaller than this is real only in the statistical sense. Two CER points is
# roughly the gap the runbook already treats as meaningful for ASR parity, and 0.05 of
# margin is a twentieth of the distance between a true and a random pairing.
MIN_CER_GAP = 2.0
MIN_DELTA_GAP = 0.05

# Duration strata, in seconds. Matching on these stops the TRAIN arm from being quietly
# shorter than TEST, which would hand it a CER advantage that has nothing to do with
# memorisation.
_DURATION_EDGES = (0.0, 3.0, 6.0, 10.0, float("inf"))

# Rasa's 16 style labels collapse to four groups for stratification. Keeping emotions
# together matters because Naturalness-Ta is drawn from CONV + the emotions.
_STYLE_GROUPS: Mapping[str, str] = {
    "CONV": "CONV",
    "WIKI": "READ",
    "BOOK": "READ",
    "NEWS": "READ",
    "INDIC": "READ",
    "ANGER": "EMOTION",
    "SAD": "EMOTION",
    "HAPPY": "EMOTION",
    "FEAR": "EMOTION",
    "SURPRISE": "EMOTION",
    "DISGUST": "EMOTION",
    "ALEXA": "OTHER",
    "UMANG": "OTHER",
    "BB": "OTHER",
    "DIGI": "OTHER",
    "PROPER NOUN": "OTHER",
}


def style_group(style: str) -> str:
    """Which stratum a Rasa style label belongs to. Unknown labels go to ``OTHER``."""

    return _STYLE_GROUPS.get(style.strip().upper(), "OTHER")


def _duration_bin(seconds: float) -> int:
    for index in range(len(_DURATION_EDGES) - 1):
        if _DURATION_EDGES[index] <= seconds < _DURATION_EDGES[index + 1]:
            return index
    return len(_DURATION_EDGES) - 2


@dataclass(frozen=True)
class Clip:
    """One Rasa row, reduced to what selection and the verdict need."""

    filename: str
    text: str
    gender: str
    style: str
    duration: float

    @property
    def stratum(self) -> tuple[str, str, int]:
        return (self.gender, style_group(self.style), _duration_bin(self.duration))


def _rank(filename: str, seed: int) -> str:
    """A stable pseudo-random key, so a re-run picks the same clips on any machine."""

    return hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).hexdigest()


def select_matched_arms(
    test_clips: Sequence[Clip],
    train_clips: Sequence[Clip],
    n_per_arm: int = 50,
    seed: int = 20261027,
) -> tuple[list[Clip], list[Clip]]:
    """Pick ``n_per_arm`` TEST clips and a TRAIN arm matched stratum for stratum.

    The TEST arm is drawn first, proportionally across strata; the TRAIN arm then copies
    its stratum histogram exactly. Where TRAIN cannot fill a stratum the shortfall is
    taken from the nearest duration bin with the same gender and style group, and the
    TEST arm is *not* trimmed to compensate — a one- or two-clip imbalance costs less
    than silently shrinking the arm under suspicion.

    Selection is deterministic in ``seed`` and the filename, so the chosen clips can be
    listed in the freeze manifest and re-derived without storing them.
    """

    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    if not test_clips or not train_clips:
        raise ValueError("both arms need candidate clips")

    by_stratum: dict[tuple[str, str, int], list[Clip]] = {}
    for clip in test_clips:
        by_stratum.setdefault(clip.stratum, []).append(clip)

    # Largest-remainder apportionment, so the arm is exactly n_per_arm clips and the
    # rounding does not all land on one stratum.
    total = len(test_clips)
    quotas: dict[tuple[str, str, int], int] = {}
    remainders: list[tuple[float, tuple[str, str, int]]] = []
    for stratum, clips in by_stratum.items():
        exact = n_per_arm * len(clips) / total
        quotas[stratum] = min(int(exact), len(clips))
        remainders.append((exact - int(exact), stratum))
    for _, stratum in sorted(remainders, reverse=True):
        if sum(quotas.values()) >= n_per_arm:
            break
        if quotas[stratum] < len(by_stratum[stratum]):
            quotas[stratum] += 1

    test_arm: list[Clip] = []
    for stratum, quota in quotas.items():
        ordered = sorted(by_stratum[stratum], key=lambda c: _rank(c.filename, seed))
        test_arm.extend(ordered[:quota])

    train_pool: dict[tuple[str, str, int], list[Clip]] = {}
    for clip in train_clips:
        train_pool.setdefault(clip.stratum, []).append(clip)

    train_arm: list[Clip] = []
    for stratum, quota in quotas.items():
        if quota == 0:
            continue
        ordered = sorted(train_pool.get(stratum, []), key=lambda c: _rank(c.filename, seed))
        taken = ordered[:quota]
        train_arm.extend(taken)
        shortfall = quota - len(taken)
        if shortfall:
            train_arm.extend(_fill_shortfall(train_pool, stratum, shortfall, seed, train_arm))

    return sorted(test_arm, key=lambda c: c.filename), sorted(train_arm, key=lambda c: c.filename)


def _fill_shortfall(
    pool: Mapping[tuple[str, str, int], list[Clip]],
    stratum: tuple[str, str, int],
    shortfall: int,
    seed: int,
    already: Sequence[Clip],
) -> list[Clip]:
    """Nearest-duration-bin substitutes, keeping gender and style group fixed."""

    gender, group, wanted_bin = stratum
    chosen: list[Clip] = []
    used = {clip.filename for clip in already}
    neighbours = sorted(
        (key for key in pool if key[0] == gender and key[1] == group and key != stratum),
        key=lambda key: abs(key[2] - wanted_bin),
    )
    for key in neighbours:
        if len(chosen) >= shortfall:
            break
        ordered = sorted(pool[key], key=lambda c: _rank(c.filename, seed))
        for clip in ordered:
            if clip.filename not in used:
                chosen.append(clip)
                used.add(clip.filename)
                if len(chosen) >= shortfall:
                    break
    return chosen


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no values to take a median of")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def memorisation_margin(
    matched: Sequence[float], mismatched: Iterable[float]
) -> float:
    """How much closer the rendering lands to the true take than to a random one.

    ``0`` means the true recording is no closer than any other take by that voice; ``1``
    would mean an exact reproduction. Negative values are possible and mean the pairing
    is *worse* than chance, which in practice signals a broken alignment rather than
    anything about training data.
    """

    mismatched_values = list(mismatched)
    if not mismatched_values:
        raise ValueError("the null distribution is empty")
    null = _median(mismatched_values)
    if null <= 0:
        raise ValueError("the null distance must be positive")
    return (null - _median(matched)) / null


@dataclass(frozen=True)
class ArmStats:
    """What one arm contributed to the verdict."""

    arm: Arm
    n: int
    median_cer: float
    delta: float | None = None


@dataclass(frozen=True)
class ContaminationResult:
    verdict: Verdict
    cer_gap: float
    cer_gap_ci: tuple[float, float]
    delta_gap: float
    delta_gap_ci: tuple[float, float]
    arms: tuple[ArmStats, ...]

    @property
    def rasa_test_usable(self) -> bool:
        return self.verdict == "CLEAN"


def verdict_from_gaps(
    cer_gap: float,
    cer_gap_ci: tuple[float, float],
    delta_gap: float,
    delta_gap_ci: tuple[float, float],
) -> Verdict:
    """Apply the pre-registered rule. No thresholds are chosen after seeing results."""

    def excludes_zero(ci: tuple[float, float]) -> bool:
        low, high = ci
        return low > 0.0 or high < 0.0

    cer_clean = cer_gap >= MIN_CER_GAP and excludes_zero(cer_gap_ci)
    delta_clean = delta_gap >= MIN_DELTA_GAP and excludes_zero(delta_gap_ci)
    if cer_clean and delta_clean:
        return "CLEAN"
    if not excludes_zero(cer_gap_ci) and not excludes_zero(delta_gap_ci):
        return "CONTAMINATED"
    return "AMBIGUOUS"


def naturalness_source(verdict: Verdict) -> str:
    """Which corpus Naturalness-Ta is built from, given the verdict.

    ``AMBIGUOUS`` resolves to the same answer as ``CONTAMINATED`` on purpose.
    """

    return "rasa_test" if verdict == "CLEAN" else "indicvoices_r_heldout"
