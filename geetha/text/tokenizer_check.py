r"""Assert the tokenizer is the one the data was compiled against (contract §6).

``token_contract.md``: compile data with one tokenizer and train with another and
"training runs, loss descends, and the model learns the wrong symbol for every audio
frame". Mismatch is "a hard error, never a warning. A silent wrong-tokenizer run costs
a full training cycle."

Why the ids and not the published hash
--------------------------------------
§6 defines a sha256 over the added-token map, the vocab size and the audio layout
constants, and publishes the value
``b645cf6612315393eea21d29e94304bdbcbfee131a25de5eecf6bae6ee9f39c5``. It does not define
the byte serialisation those three things are hashed in, and guessing it would produce a
check that fails for a reason that has nothing to do with the tokenizer. So this verifies
the **components** directly: every token the project names, at the id the project
recorded, plus the vocab size and the full 28,672-id audio band.

That is also strictly stronger than a hash for diagnosis -- a hash says "different",
this says *which token moved*. §6 explains why a mere range check is not enough: the
sibling tokenizers ``llama-3-audio-tokenizer`` (156,938), ``-tok_trimmed`` (156,942) and
``-style`` (156,952) all share audio base 128,266, so a range check passes on all three
while ``<|style>`` means something different in the data than in the model.

Verified against the real ``bodhan-ai/indic-speak`` ``tokenizer.json`` on 3 Nov 2026:
28,960 added tokens, vocab 156,960, **zero mismatches** against ``contract.py``.
"""

from __future__ import annotations

from typing import Mapping

from geetha.text.contract import (
    AUDIO_TOKEN_BASE_ID,
    CODEBOOKS,
    CODEBOOK_SIZE,
    CONDITIONING_TOKENS,
    CONTROL_TOKENS,
    NONVERBAL_TOKENS,
    TOKENIZER_FINGERPRINT,
    VOCAB_SIZE,
)

__all__ = [
    "EXPECTED_ADDED_TOKENS",
    "EXPECTED_AUDIO_TOKENS",
    "TokenizerMismatch",
    "assert_tokenizer",
    "tokenizer_mismatches",
]

EXPECTED_AUDIO_TOKENS = CODEBOOKS * CODEBOOK_SIZE  # 28,672
# §3: "28,960 added tokens in total (28,672 SNAC + 288 non-SNAC)".
EXPECTED_ADDED_TOKENS = 28_960

_NAMED = (
    ("control", CONTROL_TOKENS),
    ("conditioning", CONDITIONING_TOKENS),
    ("non-verbal", NONVERBAL_TOKENS),
)


class TokenizerMismatch(RuntimeError):
    """The tokenizer is not the one the project's ids were recorded from."""


def tokenizer_mismatches(added: Mapping[str, int], vocab_size: int) -> list[str]:
    """Every disagreement between ``added``/``vocab_size`` and the contract.

    Returns all of them rather than the first, because knowing the whole shape of the
    disagreement is what tells you whether you have the wrong checkpoint or a locally
    edited tokenizer.
    """
    problems: list[str] = []

    for group, tokens in _NAMED:
        for content, expected in tokens.items():
            actual = added.get(content)
            if actual is None:
                problems.append(f"{group} token {content} is absent from the tokenizer")
            elif actual != expected:
                problems.append(
                    f"{group} token {content} is id {actual}, but the project compiled "
                    f"its data against {expected}"
                )

    if vocab_size != VOCAB_SIZE:
        problems.append(f"vocab size is {vocab_size:,}, expected {VOCAB_SIZE:,}")

    ids = set(added.values())
    band = range(AUDIO_TOKEN_BASE_ID, AUDIO_TOKEN_BASE_ID + EXPECTED_AUDIO_TOKENS)
    present = sum(1 for token_id in band if token_id in ids)
    if present != EXPECTED_AUDIO_TOKENS:
        problems.append(
            f"the audio band holds {present:,} of the expected {EXPECTED_AUDIO_TOKENS:,} "
            f"ids ({AUDIO_TOKEN_BASE_ID}..{AUDIO_TOKEN_BASE_ID + EXPECTED_AUDIO_TOKENS - 1})"
        )

    if len(added) != EXPECTED_ADDED_TOKENS:
        problems.append(
            f"the tokenizer has {len(added):,} added tokens, expected "
            f"{EXPECTED_ADDED_TOKENS:,} ({EXPECTED_AUDIO_TOKENS:,} SNAC + 288 non-SNAC)"
        )
    return problems


def assert_tokenizer(added: Mapping[str, int], vocab_size: int) -> None:
    """Raise :class:`TokenizerMismatch` if the tokenizer is not the expected one."""
    problems = tokenizer_mismatches(added, vocab_size)
    if problems:
        listed = "\n  - ".join(problems)
        raise TokenizerMismatch(
            f"this tokenizer is not the one Geetha's data was compiled against "
            f"(fingerprint on record: {TOKENIZER_FINGERPRINT[:16]}...). "
            f"{len(problems)} problem(s):\n  - {listed}\n"
            "Training on mismatched ids does not fail loudly -- the loss descends and "
            "the model learns the wrong symbol for every audio frame. Fix the tokenizer."
        )
