r"""Is a rendered clip speech at all? Decided before any judge hears it.

On 6 Nov the operator found by listening what every number had hidden: three of 53
pilot clips were silent (-58.3, -53.4 and -49.0 dBFS against -26 to -31 for every other
clip), all three pinned at the token cap. Whisper then *hallucinated* repeated Tamil
out of the silence, so those clips scored CER 167 and 224 -- the judge's imagination,
not the model's speech.

**CER does not merely miss this failure, it inverts it.** A mute model reads as a
babbling one, and a specific ~5% defect reads as diffuse poor quality. So the rule,
applied to every clip in every eval before it reaches a judge:

* a clip quieter than ``SILENCE_DBFS`` is **invalid**;
* a clip that ran into the token cap is **invalid** -- it never said it was finished,
  so its transcript is at best a truncation and at worst a hallucination;
* invalid clips are reported as **their own rate** and never enter the CER pool.

Both failure rates are published beside the CER they were excluded from. Dropping the
worst clips quietly would flatter the model; counting them as their own number does
not, and it is the number that says whether the model is safe to ship.

The threshold is not tuned: the pilot's silent clips sat at or below -49 dBFS and its
quietest real clip at -31, so -45 has 14 dB of clear air on the speech side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["SILENCE_DBFS", "ClipValidity", "ValiditySummary", "rms_dbfs",
           "check_clip", "summarise"]

SILENCE_DBFS = -45.0
# A clip within this many seconds of the cap is treated as having hit it: the decoder's
# output length is quantised to frames, so "equal to the cap" is not an exact test.
CAP_TOLERANCE_S = 0.25


def rms_dbfs(samples: Sequence[float]) -> float:
    """RMS level of ``samples`` (floats in -1..1) in dB re full scale. Empty = -inf."""
    count = len(samples)
    if count == 0:
        return -math.inf
    try:
        import numpy as np

        array = np.asarray(samples, dtype=np.float64)
        power = float(np.mean(array * array))
    except ImportError:  # pragma: no cover -- numpy is always present where audio is
        power = sum(float(s) * float(s) for s in samples) / count
    if power <= 0.0:
        return -math.inf
    return 10.0 * math.log10(power)


@dataclass(frozen=True)
class ClipValidity:
    valid: bool
    level_dbfs: float
    seconds: float
    silent: bool
    hit_cap: bool

    @property
    def reason(self) -> str:
        if self.valid:
            return ""
        parts = []
        if self.silent:
            parts.append(f"silent ({self.level_dbfs:.1f} dBFS)")
        if self.hit_cap:
            parts.append(f"ran to the token cap ({self.seconds:.2f} s)")
        return " and ".join(parts)


def check_clip(
    samples: Sequence[float],
    sample_rate: int,
    *,
    cap_seconds: float | None = None,
    ended_itself: bool | None = None,
    silence_dbfs: float = SILENCE_DBFS,
) -> ClipValidity:
    """Judge one rendered clip.

    ``ended_itself`` is the direct evidence -- whether the model emitted
    ``<|end_of_speech|>`` -- and is used when the caller has it. Otherwise the clip's
    length against ``cap_seconds`` stands in for it.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    seconds = len(samples) / sample_rate
    level = rms_dbfs(samples)
    silent = level < silence_dbfs
    if ended_itself is not None:
        hit_cap = not ended_itself
    elif cap_seconds is not None:
        hit_cap = seconds >= cap_seconds - CAP_TOLERANCE_S
    else:
        hit_cap = False
    return ClipValidity(valid=not (silent or hit_cap), level_dbfs=level,
                        seconds=seconds, silent=silent, hit_cap=hit_cap)


@dataclass(frozen=True)
class ValiditySummary:
    clips: int
    valid: int
    silent: int
    hit_cap: int

    @property
    def invalid(self) -> int:
        return self.clips - self.valid

    @property
    def invalid_rate(self) -> float:
        return self.invalid / self.clips if self.clips else 0.0

    def line(self) -> str:
        return (f"{self.valid}/{self.clips} clips valid; invalid {self.invalid} "
                f"({100 * self.invalid_rate:.1f}%): {self.silent} silent, "
                f"{self.hit_cap} ran to the token cap")


def summarise(checks: Iterable[ClipValidity]) -> ValiditySummary:
    checks = list(checks)
    return ValiditySummary(
        clips=len(checks),
        valid=sum(1 for c in checks if c.valid),
        silent=sum(1 for c in checks if c.silent),
        hit_cap=sum(1 for c in checks if c.hit_cap),
    )
