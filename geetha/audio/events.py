r"""Counting non-verbal events in audio, and the rate that goes in the release table.

What this is for
----------------
D-021 made the release headline a measured capability gap: Indic-Speak's sixteen
non-verbal tokens are **inert**, so a Geetha that breathes is doing something the base
model cannot do at all. The sentence we publish is of the form

    Geetha 14.2 breaths/min [12.1, 16.4]; Indic-Speak 0.0 [0.0, 0.0]

and this module is the arithmetic between a detector's scores and that sentence. A bug
here inflates the claim rather than breaking it, which is the worse direction, so every
decision is a named constant with a test rather than a literal in a loop.

The detector is ours after all, and that was forced (D-028, D-029)
-----------------------------------------------------------------
This module was written expecting scores from a public AudioSet-trained tagger, on the
principle that someone else's published classifier is what makes the number re-runnable.
That principle survives; the tagger did not.

* The AudioSet tagger **cannot hear breaths in continuous Tamil speech**: best F1 0.352,
  and at usable precision 3 breaths found out of 400 (D-028). Worse, on the same 800 clips
  clip *duration* predicts the annotated breath count better than it does (rho 0.633 vs
  0.595), and its apparently-correct 9.64/min rate was threshold-tuning - it fired at 1.6x
  that rate on clips with no breath at all.
* There is **nothing to borrow in its place**: the Hub holds no breath-event detector for
  speech (searched 28 Oct; the one related repo is a breath *removal* separator).

So the instrument is ours, and re-runnability is bought a different way: a documented,
deterministic algorithm with no learned weights, released with the eval and the 800-clip
calibration set, and checked against human counts. See
``scripts\measure_breath_detectability.py``.

What it does instead of tagging a clip
--------------------------------------
Detecting a breath inside continuous speech is the problem that defeated everything above.
The problem is re-posed: forced-align the words, which locates the pauses, then classify
only the pauses. On held-out speakers a breath is separable at **AUC 0.881** against a
duration-only baseline of 0.824; at precision 0.903 recall is 0.315, and at precision
0.802 recall is 0.567. For evaluating our own TTS this is always available, because we
know the prompt text we asked the model to say.

**Pause duration is excluded from the published model on purpose.** A TTS system controls
how long it pauses, so a duration-aware rule would score a model that merely pauses longer
as one that breathes. Excluding it costs nothing measurable (AUC 0.881 either way) and
removes the one way the headline could be gamed by silence.

What counts as one event
------------------------
The hard part is not thresholding, it is deciding what is *one* breath:

* a breath lasting 1.5 s spans three 0.5 s windows and is **one** breath, so runs of
  above-threshold windows are merged;
* a breath that dips under the threshold for a single window is still one breath, so
  runs separated by less than ``min_gap_seconds`` are merged too;
* a single stray window above threshold is detector jitter, not a breath, so a run
  shorter than ``min_event_seconds`` is discarded.

Without the last rule, jitter becomes breaths and our own number goes up.

The honest caveat
-----------------
The threshold is calibrated on *real* restored Tamil speech (IndicVoices-R audio joined
to IndicVoices' ``unsanitized_verbatim`` marks, which sit inline at the position of each
``[breathing]``). It is then applied to *synthetic* speech, which is a different
acoustic domain. That mismatch is why Naturalness-Ta carries 24 ``clean`` prompts: they
give a false-positive rate measured in the synthetic domain, on the systems actually
being compared. A capability number published without that rate is not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geetha.scoring.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED, BootstrapResult

__all__ = [
    "DEFAULT_MIN_EVENT_SECONDS",
    "DEFAULT_MIN_GAP_SECONDS",
    "EventSpan",
    "count_events",
    "events_per_minute",
    "rate_with_ci",
]

# A breath is audibly shorter than a word but longer than one analysis frame. 0.3 s is
# below the shortest breath we expect to keep and above the length of a single spurious
# window at the hops we use.
DEFAULT_MIN_EVENT_SECONDS = 0.3

# Two breaths closer together than this are one breath whose score dipped. Set from the
# physiology rather than the detector: a speaker does not inhale twice in a third of a
# second.
DEFAULT_MIN_GAP_SECONDS = 0.35


@dataclass(frozen=True)
class EventSpan:
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


def count_events(
    scores: Sequence[tuple[float, float]],
    *,
    threshold: float,
    hop_seconds: float,
    min_event_seconds: float = DEFAULT_MIN_EVENT_SECONDS,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
) -> list[EventSpan]:
    """Merge above-threshold windows into events.

    ``scores`` is ``(window_start_seconds, score)`` in time order. A score equal to the
    threshold counts as above it, so sweeping a threshold across the observed scores
    cannot step over one of them.
    """
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")
    if not scores:
        return []

    runs: list[list[float]] = []
    for start, score in scores:
        if score >= threshold:
            end = start + hop_seconds
            if runs and start - runs[-1][1] <= min_gap_seconds:
                runs[-1][1] = end
            else:
                runs.append([start, end])

    return [
        EventSpan(start=start, end=end)
        for start, end in runs
        # Floating point: a run built from exact hop multiples can land a hair under.
        if (end - start) >= min_event_seconds - 1e-9
    ]


def events_per_minute(events: int, seconds: float) -> float:
    """Events per minute of audio. Zero events is 0.0, never undefined."""
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    if events < 0:
        raise ValueError(f"events must not be negative, got {events}")
    return 60.0 * events / seconds


def rate_with_ci(
    clips: Sequence[tuple[int, float]],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Events per minute over a set of clips, with a percentile bootstrap interval.

    ``clips`` is ``(events, seconds)`` per clip. Resampling is **by clip**, matching
    ``geetha.scoring.bootstrap``: the unit of independence is an utterance, not a
    second of audio, so an interval drawn over seconds would be far too tight.
    """
    if not clips:
        raise ValueError("cannot bootstrap an empty set of clips")
    events = np.array([count for count, _ in clips], dtype=np.float64)
    seconds = np.array([length for _, length in clips], dtype=np.float64)
    if seconds.sum() <= 0:
        raise ValueError("cannot bootstrap clips with no duration")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clips), size=(n_resamples, len(clips)))
    rates = 60.0 * events[indices].sum(axis=1) / seconds[indices].sum(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(rates, [tail, 100.0 - tail])
    return BootstrapResult(
        point=float(60.0 * events.sum() / seconds.sum()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
