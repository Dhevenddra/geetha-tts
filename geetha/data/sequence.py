r"""Assemble one training example -- prompt, audio span, loss mask.

The layout is ``token_contract.md`` §5::

    <|start_of_human|><|begin_of_text|> ...prompt... <|eot_id|><|end_of_human|>
    <|start_of_ai|><|start_of_speech|> ...audio... <|end_of_speech|><|end_of_ai|>

where the prompt (from :func:`geetha.text.contract.build_tts_prompt`) already ends at
``<|start_of_ai|>``.

Where the loss boundary goes, and why it is not obvious
------------------------------------------------------
``<|start_of_speech|>`` is **supervised**, even though it looks like part of the
scaffolding. §2 of the contract: it "opens the audio span; the model emits this itself
as its first generated token". Mask it and the model is never taught to open an audio
span, so at inference it has to be forced -- which is exactly the ambiguity
``contract.py`` records between ``inference.py`` and the contract about where a prompt
ends. Supervising it is what makes the two agree.

Everything before it is masked. The prompt is conditioning the model is always handed,
so training on it spends capacity predicting its own input.

Validation is deliberately strict. An audio token outside the band, a control token
smuggled into the span, or a span that is not a whole number of 7-token frames all
raise here, because each of them otherwise produces a run that trains happily and
learns the wrong symbol for every frame -- the silent failure the contract warns about.
And a sequence over the context limit raises rather than being truncated: cutting an
audio span mid-frame misaligns every frame after the cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from geetha.audio.snac_tokens import TOKENS_PER_FRAME
from geetha.text.contract import AUDIO_TOKEN_BASE_ID, CODEBOOKS, CODEBOOK_SIZE, CONTROL_TOKENS

__all__ = ["IGNORE_INDEX", "TrainingSequence", "training_sequence"]

# What torch's cross-entropy ignores.
IGNORE_INDEX = -100

_START_SPEECH = CONTROL_TOKENS["<|start_of_speech|>"]
_END_SPEECH = CONTROL_TOKENS["<|end_of_speech|>"]
_END_AI = CONTROL_TOKENS["<|end_of_ai|>"]
_BAND_TOP = AUDIO_TOKEN_BASE_ID + CODEBOOKS * CODEBOOK_SIZE - 1


@dataclass(frozen=True)
class TrainingSequence:
    input_ids: list[int]
    labels: list[int]

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


def training_sequence(
    prompt_ids: Sequence[int],
    audio_tokens: Sequence[int],
    *,
    max_length: int | None = None,
) -> TrainingSequence:
    """One training example, with the prompt masked out of the loss."""
    if not len(prompt_ids):
        raise ValueError("training_sequence needs a prompt")
    if not len(audio_tokens):
        raise ValueError("training_sequence got no audio tokens")

    if len(audio_tokens) % TOKENS_PER_FRAME:
        raise ValueError(
            f"the audio span is {len(audio_tokens)} tokens, not a whole number of "
            f"{TOKENS_PER_FRAME}-token frames"
        )
    for token in audio_tokens:
        if not AUDIO_TOKEN_BASE_ID <= token <= _BAND_TOP:
            raise ValueError(
                f"token {token} in the audio span is outside the audio band "
                f"{AUDIO_TOKEN_BASE_ID}..{_BAND_TOP}"
            )

    input_ids = [*prompt_ids, _START_SPEECH, *audio_tokens, _END_SPEECH, _END_AI]
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(
            f"sequence is {len(input_ids)} tokens, over the context limit of {max_length}. "
            "Drop the clip or raise the limit -- truncating an audio span mid-frame "
            "misaligns every frame after the cut."
        )

    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return TrainingSequence(input_ids=input_ids, labels=labels)
