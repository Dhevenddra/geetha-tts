"""Reassemble chunked synthesis into one waveform.

A prompt too long for one pass is rendered as several chunks (see
``geetha.text.chunk``) and rejoined here. The gap between chunks is a judgement call
with a cost either way: too short and two sentences run together, too long and the
clip sounds padded and the naturalness distribution shifts. 120 ms is the default --
about a comfortable inter-sentence pause -- and it is a parameter so a report can say
which value produced its numbers.

No gap is added before the first chunk or after the last, so the joined duration stays
comparable with a single-pass render of the same text.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["join_clips"]

DEFAULT_GAP_SECONDS = 0.12


def join_clips(
    waves: Sequence[np.ndarray],
    sample_rate: int,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> np.ndarray:
    """Concatenate ``waves``, separated by ``gap_seconds`` of silence."""
    if not len(waves):
        raise ValueError("join_clips needs at least one waveform")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if gap_seconds < 0:
        raise ValueError(f"gap_seconds must not be negative, got {gap_seconds}")

    # reshape rather than squeeze: squeeze turns a one-sample clip into a 0-d array,
    # which cannot be concatenated.
    pieces = [np.asarray(wave, dtype=np.float32).reshape(-1) for wave in waves]
    if len(pieces) == 1:
        return pieces[0]

    gap = np.zeros(int(round(gap_seconds * sample_rate)), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, piece in enumerate(pieces):
        if index:
            joined.append(gap)
        joined.append(piece)
    return np.concatenate(joined).astype(np.float32, copy=False)
