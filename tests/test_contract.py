"""Tests for geetha.text.contract — Indic-Speak's own token contract.

Read from `bodhan-ai/indic-speak` (`token_contract.md`, `config.json`, `voices.md`) on
26 Oct 2026, not from the runbook. The runbook invented a tag scheme; the model already
has one, and every deviation is a place for a silent tokenizer mismatch. The contract
file says so itself: compiling with one tokenizer and training with another is silent --
"training runs, loss descends, and the model learns the wrong symbol for every audio
frame".
"""
import pytest

from geetha.text.contract import (
    AUDIO_TOKEN_BASE_ID,
    CODEBOOKS,
    CODEBOOK_SIZE,
    EMOTION_STYLES,
    NONVERBAL_TOKENS,
    is_known_style,
    SOURCE_LABELS,
    STYLE_LABELS,
    TOKENIZER_FINGERPRINT,
    TOKENS_PER_SECOND,
    VOCAB_SIZE,
    build_tts_prompt,
    estimate_audio_tokens,
    map_corpus_nonverbal,
)


def test_the_fingerprint_is_the_one_the_contract_publishes():
    # Asserted at compile time and training start. A mismatch is a hard error.
    assert TOKENIZER_FINGERPRINT == (
        "b645cf6612315393eea21d29e94304bdbcbfee131a25de5eecf6bae6ee9f39c5"
    )


def test_the_vocabulary_size_matches_config_json():
    assert VOCAB_SIZE == 156_960


def test_there_are_exactly_sixteen_nonverbal_tokens_at_the_documented_ids():
    assert len(NONVERBAL_TOKENS) == 16
    assert NONVERBAL_TOKENS["<|nv_breath|>"] == 156_944
    assert NONVERBAL_TOKENS["<|nv_stammer|>"] == 156_945
    assert NONVERBAL_TOKENS["<|nv_laugh|>"] == 156_947
    assert NONVERBAL_TOKENS["<|nv_burp|>"] == 156_959
    assert sorted(NONVERBAL_TOKENS.values()) == list(range(156_944, 156_960))


def test_the_audio_band_constants_match_the_contract():
    assert (AUDIO_TOKEN_BASE_ID, CODEBOOKS, CODEBOOK_SIZE) == (128_266, 7, 4_096)
    # base .. base + 7*4096 - 1, immediately below the conditioning block.
    assert AUDIO_TOKEN_BASE_ID + CODEBOOKS * CODEBOOK_SIZE - 1 == 156_937


def test_the_emotion_styles_are_the_six_the_model_was_trained_on():
    # The runbook listed <happy> <sad> <angry> <fear> <surprise> <disgust> <neutral>.
    # The model's trained labels have "anger", not "angry", and no "neutral" at all.
    assert EMOTION_STYLES == ("HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST")
    assert "angry" not in EMOTION_STYLES
    assert "neutral" not in EMOTION_STYLES


def test_the_non_emotion_styles_are_included_too():
    assert "TV style news" in STYLE_LABELS
    assert "single person narration audiobook" in STYLE_LABELS
    assert "Customer Care" in STYLE_LABELS
    assert set(EMOTION_STYLES) <= set(STYLE_LABELS)


def test_the_source_labels_are_free_text_env_values():
    # <|env> is an open class: only the wrapper is a token, so a new source class
    # needs no tokenizer change.
    assert SOURCE_LABELS == ("studio", "enhanced", "phone_restored")


def test_a_tts_prompt_follows_the_documented_layout():
    prompt = build_tts_prompt(speaker="Anitha", style="HAPPY", text="வணக்கம்")
    assert prompt == (
        "<|start_of_human|><|begin_of_text|>"
        "<|speaker>Anitha<speaker|>\n"
        "<|style>HAPPY<style|>\n"
        "வணக்கம்"
        "<|eot_id|><|end_of_human|><|start_of_ai|>"
    )


def test_the_prompt_can_carry_an_environment_label():
    prompt = build_tts_prompt(speaker="Anitha", style="HAPPY", text="x", source="studio")
    assert "<|env>studio<env|>\n" in prompt
    # Metadata order is speaker -> style -> env.
    assert prompt.index("<|speaker>") < prompt.index("<|style>") < prompt.index("<|env>")


def test_an_empty_metadata_value_emits_nothing_at_all():
    # The contract: "An empty value emits nothing at all (no empty wrapper)."
    prompt = build_tts_prompt(speaker="Anitha", style="", text="x")
    assert "<|style>" not in prompt


def test_any_style_string_is_allowed_because_style_is_free_text():
    # inference.py's build_prompt does not validate style at all: it encodes
    # enc(style.strip()). The README confirms "`style` is free text. Two forms appear
    # in training: uppercase emotion labels (ANGER, FEAR, ... 17 in total) and
    # descriptive phrases such as 'happiness, speaking with a strong texan accent'."
    # Refusing an unlisted style would block a form the model was trained on.
    prompt = build_tts_prompt(speaker="Anitha", style="happiness, spoken softly", text="x")
    assert "<|style>happiness, spoken softly<style|>" in prompt


def test_the_trained_emotion_labels_are_uppercase_as_the_model_saw_them():
    # README and the CLI both use --style ANGER.
    assert EMOTION_STYLES == ("HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST")


def test_a_style_that_is_not_a_known_label_is_reported_by_the_checker():
    # Free text is allowed, but we still want to know when we are off the trained set.
    assert is_known_style("ANGER") is True
    assert is_known_style("anger") is True        # case-insensitive check
    assert is_known_style("angry") is False       # the runbook's spelling
    assert is_known_style("neutral") is False     # never trained


def test_the_inference_prompt_appends_start_of_speech():
    # inference.py ends the prompt with <|start_of_ai|>, <|start_of_speech|>, while
    # token_contract.md s5 says the prompt stops before <|start_of_speech|> and the
    # model emits it. The shipped code is what actually runs, so inference follows it;
    # training-data compilation follows the contract, because that is what defines the
    # loss boundary. Both forms are available and named.
    inference = build_tts_prompt(speaker="Anitha", style="ANGER", text="x", for_inference=True)
    training = build_tts_prompt(speaker="Anitha", style="ANGER", text="x")
    assert inference.endswith("<|start_of_ai|><|start_of_speech|>")
    assert training.endswith("<|start_of_ai|>")
    assert not training.endswith("<|start_of_speech|>")


def test_an_unknown_source_label_is_refused():
    with pytest.raises(ValueError, match="source"):
        build_tts_prompt(speaker="Anitha", style="HAPPY", text="x", source="telephone")


def test_corpus_nonverbals_map_onto_the_models_own_tokens():
    # IndicVoices unsanitized_verbatim and DiarBench both use these source forms.
    assert map_corpus_nonverbal("breathing") == "<|nv_breath|>"
    assert map_corpus_nonverbal("stammers") == "<|nv_stammer|>"
    assert map_corpus_nonverbal("laughter") == "<|nv_laugh|>"
    assert map_corpus_nonverbal("coughing") == "<|nv_cough|>"
    assert map_corpus_nonverbal("gasps") == "<|nv_gasp|>"
    assert map_corpus_nonverbal("swallowing") == "<|nv_swallow|>"
    assert map_corpus_nonverbal("throat_clearing") == "<|nv_throat|>"


def test_fillers_have_no_token_and_are_not_mapped():
    # [uhh] and [umm] are speech, not events, and the closed set has no filler token.
    # They become Tamil filler text instead, so mapping must refuse rather than guess.
    assert map_corpus_nonverbal("uhh") is None
    assert map_corpus_nonverbal("umm") is None


def test_noise_annotations_are_not_nonverbals():
    for name in ("noise", "unintelligible", "bird_squawk", "horn", "background_speech"):
        assert map_corpus_nonverbal(name) is None, name


def test_token_rate_follows_the_contracts_rule_of_thumb():
    assert TOKENS_PER_SECOND == 82


def test_twenty_seconds_of_audio_fits_inside_the_sequence_budget():
    # 2,048 tokens minus room for the text prompt. The contract's inference.py uses
    # max_new_tokens=2520 for "~30 s", which is the same 82 tok/s.
    assert estimate_audio_tokens(20.0) == 1640
    assert estimate_audio_tokens(20.0) + 150 < 2048


def test_the_estimate_is_documented_as_an_upper_bound():
    # Consecutive duplicate frames are dropped at encode time, so real token counts
    # run at or below the estimate -- never above it.
    assert estimate_audio_tokens(1.0) == 82
