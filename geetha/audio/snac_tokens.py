r"""SNAC's 7-token frame layout -- ETL step 6, from ``token_contract.md`` §3.

SNAC 24 kHz emits three codebooks at a 1:2:4 temporal ratio::

    c0: [seq_len]     12 Hz   coarsest
    c1: [2*seq_len]   23 Hz
    c2: [4*seq_len]   47 Hz   finest

flattened to seven tokens per frame in one fixed interleave::

    frame i -> [ c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3] ]

The part that is easy to get wrong: each raw code is offset by **its position in the
frame, not by which codebook it came from**. Positions 1 and 4 are both ``c1``;
positions 2, 3, 5 and 6 are all ``c2``. That is what makes the stream decodable
without a separate structure signal, and it is why a plain "offset by codebook index"
implementation produces a stream that looks right and decodes to noise.

Why this module is paranoid about validation
--------------------------------------------
``token_contract.md`` warns that getting the audio band wrong fails *silently* --
"training runs, loss descends, and the model learns the wrong symbol for every audio
frame" -- and names a real instance, the gemma3 SNAC base mismatch that produced
undecodable checkpoints. There is no cheap way to notice that from a loss curve, so
every boundary is checked here and every id range is asserted against the contract's
own table in the tests.

Consecutive frames sharing a ``c0`` are dropped at encode time, which is why token
count is not exactly proportional to duration (the project's rule of thumb is
82 tokens ≈ 1 s, against the 84 a frame rate of 12 Hz would imply).
"""

from __future__ import annotations

from typing import Sequence

from geetha.text.contract import AUDIO_TOKEN_BASE_ID, CODEBOOKS, CODEBOOK_SIZE

__all__ = [
    "TOKENS_PER_FRAME",
    "drop_duplicate_frames",
    "flatten_frames",
    "unflatten_frames",
]

TOKENS_PER_FRAME = CODEBOOKS  # 7

# (frame position, source codebook, index within that codebook for frame i).
_LAYOUT = (
    (0, "c0", 1, 0),
    (1, "c1", 2, 0),
    (2, "c2", 4, 0),
    (3, "c2", 4, 1),
    (4, "c1", 2, 1),
    (5, "c2", 4, 2),
    (6, "c2", 4, 3),
)


def _check_codes(name: str, codes: Sequence[int]) -> None:
    for code in codes:
        if not 0 <= code < CODEBOOK_SIZE:
            raise ValueError(
                f"{name} holds {code}, outside the codebook range 0..{CODEBOOK_SIZE - 1} "
                f"(codebook size {CODEBOOK_SIZE})"
            )


def _check_shape(c0: Sequence[int], c1: Sequence[int], c2: Sequence[int]) -> int:
    frames = len(c0)
    if len(c1) != 2 * frames:
        raise ValueError(f"c1 must hold 2 codes per frame: expected {2 * frames}, got {len(c1)}")
    if len(c2) != 4 * frames:
        raise ValueError(f"c2 must hold 4 codes per frame: expected {4 * frames}, got {len(c2)}")
    return frames


def flatten_frames(c0: Sequence[int], c1: Sequence[int], c2: Sequence[int]) -> list[int]:
    """Flatten SNAC's three codebooks into the model's 7-tokens-per-frame stream."""
    frames = _check_shape(c0, c1, c2)
    for name, codes in (("c0", c0), ("c1", c1), ("c2", c2)):
        _check_codes(name, codes)

    books = {"c0": c0, "c1": c1, "c2": c2}
    tokens: list[int] = []
    for frame in range(frames):
        for position, book, stride, offset in _LAYOUT:
            code = books[book][stride * frame + offset]
            tokens.append(AUDIO_TOKEN_BASE_ID + position * CODEBOOK_SIZE + code)
    return tokens


def drop_duplicate_frames(
    c0: Sequence[int], c1: Sequence[int], c2: Sequence[int]
) -> tuple[list[int], list[int], list[int]]:
    """Remove consecutive frames that repeat the previous frame's ``c0``.

    The whole frame goes, not just its ``c0``, so the 1:2:4 shape is preserved.
    """
    frames = _check_shape(c0, c1, c2)
    keep = [frame for frame in range(frames) if frame == 0 or c0[frame] != c0[frame - 1]]
    return (
        [c0[frame] for frame in keep],
        [c1[2 * frame + offset] for frame in keep for offset in (0, 1)],
        [c2[4 * frame + offset] for frame in keep for offset in (0, 1, 2, 3)],
    )


def unflatten_frames(tokens: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """Recover the three codebooks from a flattened stream.

    Fails loudly on anything the decoder could not make sense of: a token outside the
    audio band, a length that is not a whole number of frames, or a code sitting in
    the wrong position block (which means the stream is misaligned, the failure mode
    that otherwise only shows up as noise out of the vocoder).
    """
    if len(tokens) % TOKENS_PER_FRAME:
        raise ValueError(
            f"{len(tokens)} tokens is not a whole number of frames of {TOKENS_PER_FRAME}"
        )

    band_top = AUDIO_TOKEN_BASE_ID + CODEBOOKS * CODEBOOK_SIZE - 1
    out: dict[str, list[int]] = {"c0": [], "c1": [], "c2": []}
    for frame_start in range(0, len(tokens), TOKENS_PER_FRAME):
        frame = tokens[frame_start:frame_start + TOKENS_PER_FRAME]
        for position, book, _stride, _offset in _LAYOUT:
            token = frame[position]
            if not AUDIO_TOKEN_BASE_ID <= token <= band_top:
                raise ValueError(
                    f"token {token} is outside the audio band "
                    f"{AUDIO_TOKEN_BASE_ID}..{band_top}"
                )
            relative = token - AUDIO_TOKEN_BASE_ID
            if relative // CODEBOOK_SIZE != position:
                raise ValueError(
                    f"token {token} sits in frame position {position} but carries the "
                    f"offset of position {relative // CODEBOOK_SIZE}; the stream is misaligned"
                )
            out[book].append(relative % CODEBOOK_SIZE)
    return out["c0"], out["c1"], out["c2"]
