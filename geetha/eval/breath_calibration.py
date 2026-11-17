r"""Choosing the real Tamil clips that calibrate the breath detector.

Why real clips are needed at all
--------------------------------
``geetha.audio.events`` turns detector scores into a count, but a count depends on a
threshold, and a threshold picked by taste is a number we would be unable to defend.
IndicVoices annotates ``[breathing]`` **inline at the position it occurs** in
``unsanitized_verbatim`` (2,845 of them in a single shard of 118), and IndicVoices-R
carries the restored wideband audio for the same utterances, joinable on
``(speaker_id, norm_text(text))`` at 97.4% (D-018). So we have real Tamil speech whose
breath count is known, and the threshold can be set where precision and recall are
actually measured.

The two selection mistakes this module exists to prevent
--------------------------------------------------------
**A noisy clip in either set** teaches the threshold to fire on noise. The corpus has
``[Persistent-noise-start]``/``[-end]`` markers from the annotators, and they catch
things the ``snr`` figure does not, so both filters apply.

**A clip with a filler but no breath in the *negative* set.** This is the subtle one. A
hesitation is breathy; if the detector fires on it we would record a false positive it
did not really make, the calibrated threshold would be pushed too high, and we would
then under-count our own model's breaths. Such clips are therefore ``AMBIGUOUS`` and go
in neither set. A negative has to be clean of **every** non-verbal mark, not merely of
breathing.

What this does not do
---------------------
The marks give the *position in the text*, not a timestamp, so this module calibrates a
**count per clip**, not an onset detector.

**Corrected 28 Oct (D-029): a count is NOT all these labels can honestly give.** This
docstring previously claimed it was, and that claim cost two rebuilds. A count per clip
turned out to be too weak to validate any detector - on these 800 clips, clip duration
predicts the annotated count better (rho 0.633) than the AudioSet tagger does (0.595), so
a count-labelled set can *reject* a detector but never validate one, which is why the
tagger and three hand-built replacements all failed the same way (D-028, D-029).

A text position is a timestamp once the words around it are aligned. 2,816 of 2,845 marks
in a single shard have words on **both sides**, so each one names the gap between two
specific words; C4 forced-aligns those words and 619 of 647 marks (95.7%) land in a
located pause, which run 4.09x longer than ordinary word gaps. Inside those pauses a
breath is then acoustically separable at **AUC 0.881** on held-out speakers against a
duration-only baseline of 0.824 (`scripts\measure_breath_detectability.py`).

So the honest limit of these labels is an **onset-localised, pause-level** ground truth,
not a per-clip count. Keep the count here - it is still what the published rate is built
from - but do not treat it as the ceiling.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from geetha.text.contract import map_corpus_nonverbal

__all__ = [
    "AMBIGUOUS",
    "BREATH_MARKS",
    "MAX_SECONDS",
    "MIN_SECONDS",
    "MIN_SNR_DB",
    "NEGATIVE",
    "POSITIVE",
    "count_marks",
    "label_for",
    "rejection_reason",
]

POSITIVE = "positive"
NEGATIVE = "negative"
AMBIGUOUS = "ambiguous"

# Which events count as "a breath" for the headline. A choice, not a fact, so it is
# named here and quoted in the report. Sighs are included: a sigh is an audible
# exhalation and the base model can produce neither.
#
# Measured 28 Oct across all 118 IndicVoices Tamil shards (422,963 rows), counting both
# the [square] and <angle> mark forms: breathing 274,823 but **sigh 0**. The sigh half of
# this set therefore never fires on Tamil, so every count it has ever produced is purely
# breathing. Kept rather than removed, because it costs nothing and the Rasa side may yet
# carry sighs - but the number must not be described as "breaths and sighs", and there is
# no real-Tamil sigh rate to calibrate against.
BREATH_MARKS = frozenset({"<|nv_breath|>", "<|nv_sigh|>"})

# Fillers are speech rather than acoustic events (contract.py), so they are counted
# under their own key and never into the breath count.
_FILLERS = frozenset({"uhh", "umm", "uh", "um", "tsk", "hmm", "mmm"})

# Anything the annotators marked that is neither a voiceable event nor a filler: room
# noise, a horn, a bird. Its presence disqualifies a clip from the clean negative set.
_NOISE_PATTERN = re.compile(r"noise|squawk|horn|talking|tapping|bell|animal|motorcycle"
                            r"|unintelligible|music|vehicle|traffic", re.IGNORECASE)

# Annotator-flagged noise spans. Present in clips whose snr looks fine.
_PERSISTENT_NOISE = re.compile(r"persistent-noise", re.IGNORECASE)

_MARK = re.compile(r"\[([^\[\]]{1,40})\]")

# IV-R reports snr per clip. 20 dB keeps ordinary room recordings and drops the ones
# where a breath would be indistinguishable from the floor.
MIN_SNR_DB = 20.0

# Long enough to contain an event with context; short enough that one clip does not
# dominate a per-clip rate.
MIN_SECONDS = 2.0
MAX_SECONDS = 20.0


def count_marks(text: str | None) -> dict[str, int]:
    """Annotation counts for one utterance, keyed by the model's token names.

    Non-verbal events come back under their ``<|nv_...|>`` token so one event cannot be
    counted under two names. Fillers come back under ``"filler"`` and environment noise
    under ``"noise"``; neither is a voiceable event.
    """
    if not text:
        return {}
    counts: Counter[str] = Counter()
    for raw in _MARK.findall(text):
        name = raw.strip().lower().replace(" ", "_")
        token = map_corpus_nonverbal(name)
        if token:
            counts[token] += 1
        elif name in _FILLERS:
            counts["filler"] += 1
        elif _PERSISTENT_NOISE.search(name) or _NOISE_PATTERN.search(name):
            counts["noise"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def label_for(marks: Mapping[str, int]) -> str:
    """``POSITIVE``, ``NEGATIVE`` or ``AMBIGUOUS`` for one utterance's marks."""
    if any(marks.get(mark, 0) for mark in BREATH_MARKS):
        return POSITIVE
    # No breath. A clean negative must carry no annotation at all: any other event, a
    # filler, or flagged noise could all produce a breath-like sound.
    return NEGATIVE if not marks else AMBIGUOUS


def rejection_reason(row: Mapping[str, Any]) -> str | None:
    """Why this clip cannot be used for calibration, or ``None`` if it can.

    Reasons are short and from a closed set, because they are tabulated in the report:
    a selection whose exclusions are unexplained is not a selection anyone can check.
    """
    # Either the raw annotation string or the counts ``count_marks`` produced from it.
    # The builder indexes 423k utterances into counts rather than holding every string
    # in memory, and one decision must not have two code paths.
    marks = row.get("marks")
    if marks is None:
        text = row.get("unsanitized_verbatim") or ""
        marks = count_marks(text)
        if _PERSISTENT_NOISE.search(text):
            return "annotator-flagged noise"
    # A [noise], [horn] or [bird_squawk] is a transient the detector could score as a
    # breath. Labelling would exclude such a clip anyway, but rejecting it here puts the
    # reason in the report's table instead of leaving it implicit.
    if marks.get("noise", 0):
        return "noise annotated"

    snr = row.get("snr")
    if snr is None:
        return "no snr reported"
    try:
        snr = float(snr)
    except (TypeError, ValueError):
        return "snr not a number"
    if snr < MIN_SNR_DB:
        return f"snr below {MIN_SNR_DB:.0f} dB"

    duration = row.get("duration")
    if duration is None:
        return "no duration reported"
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return "duration not a number"
    if duration < MIN_SECONDS:
        return f"too short (<{MIN_SECONDS:.0f}s)"
    if duration > MAX_SECONDS:
        return f"too long (>{MAX_SECONDS:.0f}s)"

    return None
