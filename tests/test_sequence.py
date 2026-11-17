"""Training sequence assembly for Stage A/B (token_contract.md §5).

The layout:

    <|start_of_human|><|begin_of_text|> ...prompt... <|eot_id|><|end_of_human|>
    <|start_of_ai|><|start_of_speech|> ...audio... <|end_of_speech|><|end_of_ai|>

Two decisions these tests pin down, both of which are wrong by default:

* the prompt stops at ``<|start_of_ai|>`` and ``<|start_of_speech|>`` is **supervised**,
  because §2 says the model emits that token itself as its first generated token. Mask
  it and the model never learns to open an audio span.
* everything before it is masked. Train on the prompt and the model spends capacity
  learning to predict its own conditioning, which it is always given.
"""

from __future__ import annotations

import pytest

from geetha.data.sequence import IGNORE_INDEX, training_sequence
from geetha.text.contract import AUDIO_TOKEN_BASE_ID, CONTROL_TOKENS

START_SPEECH = CONTROL_TOKENS["<|start_of_speech|>"]
END_SPEECH = CONTROL_TOKENS["<|end_of_speech|>"]
END_AI = CONTROL_TOKENS["<|end_of_ai|>"]

PROMPT = [1, 2, 3, 4]
# One whole 7-token frame, each code in its own position block.
AUDIO = [AUDIO_TOKEN_BASE_ID + position * 4096 + position for position in range(7)]


class TestLayout:
    def test_the_audio_span_is_wrapped_in_its_control_tokens(self):
        sequence = training_sequence(PROMPT, AUDIO)
        assert sequence.input_ids == PROMPT + [START_SPEECH] + AUDIO + [END_SPEECH, END_AI]

    def test_the_prompt_is_reproduced_untouched(self):
        assert training_sequence(PROMPT, AUDIO).input_ids[:len(PROMPT)] == PROMPT

    def test_the_sequence_ends_with_end_of_ai(self):
        assert training_sequence(PROMPT, AUDIO).input_ids[-1] == END_AI

    def test_length_is_the_prompt_plus_the_audio_plus_three_control_tokens(self):
        sequence = training_sequence(PROMPT, AUDIO)
        assert len(sequence.input_ids) == len(PROMPT) + len(AUDIO) + 3


class TestLossMask:
    def test_the_prompt_is_masked_out_of_the_loss(self):
        labels = training_sequence(PROMPT, AUDIO).labels
        assert labels[:len(PROMPT)] == [IGNORE_INDEX] * len(PROMPT)

    def test_start_of_speech_is_supervised_because_the_model_must_emit_it(self):
        sequence = training_sequence(PROMPT, AUDIO)
        assert sequence.labels[len(PROMPT)] == START_SPEECH

    def test_every_audio_token_is_supervised(self):
        sequence = training_sequence(PROMPT, AUDIO)
        start = len(PROMPT) + 1
        assert sequence.labels[start:start + len(AUDIO)] == AUDIO

    def test_the_closing_control_tokens_are_supervised(self):
        assert training_sequence(PROMPT, AUDIO).labels[-2:] == [END_SPEECH, END_AI]

    def test_labels_and_input_ids_are_the_same_length(self):
        sequence = training_sequence(PROMPT, AUDIO)
        assert len(sequence.labels) == len(sequence.input_ids)

    def test_exactly_the_audio_span_and_its_markers_are_supervised(self):
        sequence = training_sequence(PROMPT, AUDIO)
        supervised = sum(1 for label in sequence.labels if label != IGNORE_INDEX)
        assert supervised == len(AUDIO) + 3


class TestValidation:
    def test_an_empty_audio_span_is_rejected(self):
        with pytest.raises(ValueError, match="no audio tokens"):
            training_sequence(PROMPT, [])

    def test_an_empty_prompt_is_rejected(self):
        with pytest.raises(ValueError, match="prompt"):
            training_sequence([], AUDIO)

    def test_an_audio_token_outside_the_band_is_rejected(self):
        # Catching this here is the difference between a loud failure and a training
        # run that quietly learns the wrong symbol for every frame.
        bad = [*AUDIO[:-1], AUDIO_TOKEN_BASE_ID - 1]
        with pytest.raises(ValueError, match="audio band"):
            training_sequence(PROMPT, bad)

    def test_a_control_token_smuggled_into_the_audio_span_is_rejected(self):
        with pytest.raises(ValueError, match="audio band"):
            training_sequence(PROMPT, [*AUDIO[:-1], END_SPEECH])

    def test_an_audio_span_that_is_not_a_whole_number_of_frames_is_rejected(self):
        with pytest.raises(ValueError, match="7"):
            training_sequence(PROMPT, [AUDIO_TOKEN_BASE_ID] * 8)


class TestTokenBudget:
    def test_a_sequence_over_the_context_limit_is_reported_not_silently_truncated(self):
        # Truncating an audio span mid-frame would teach the model to stop mid-word and
        # would misalign every frame after the cut.
        huge = [AUDIO_TOKEN_BASE_ID] * (7 * 4_000)
        with pytest.raises(ValueError, match="context"):
            training_sequence(PROMPT, huge, max_length=1_024)

    def test_a_sequence_inside_the_limit_is_fine(self):
        assert training_sequence(PROMPT, AUDIO, max_length=1_024).input_ids
