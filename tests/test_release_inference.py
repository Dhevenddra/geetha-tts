"""The shipped inference file must send exactly the prompt the adapter was trained on."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from geetha.text.contract import build_tts_prompt

_PATH = Path(__file__).resolve().parents[1] / "release" / "inference.py"
_SPEC = importlib.util.spec_from_file_location("geetha_release_inference", _PATH)
release = importlib.util.module_from_spec(_SPEC)
import sys  # noqa: E402

sys.modules[_SPEC.name] = release       # dataclasses resolves the module by name
_SPEC.loader.exec_module(release)


@pytest.mark.parametrize("style", ["CONV", "HAPPY", ""])
def test_prompt_is_the_training_contract(style):
    text = "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு."
    expected = build_tts_prompt(speaker="anitha", style=style, text=text, source="studio",
                                for_inference=True)
    assert release.build_prompt(text, "anitha", style) == expected


def test_speaker_is_lower_cased_and_closed():
    assert release.normalise_speaker(" Anitha ") == "anitha"
    assert release.build_prompt("x", "Arun").startswith(
        "<|start_of_human|><|begin_of_text|><|speaker>arun<speaker|>\n")
    with pytest.raises(ValueError):
        release.normalise_speaker("ivr_f251dc9c")


def test_neutral_speech_is_not_guided_and_emotions_are():
    decoding = release.Decoding()
    assert decoding.guidance_for("CONV") == 0.0
    assert decoding.guidance_for("SAD") > 0
    assert decoding.guidance_for("an unknown style") == 0.0
    assert decoding.sampling_for("CONV") == (0.6, 1.0)
    assert decoding.sampling_for("SAD") == (0.9, 1.1)


def test_long_text_is_split_without_losing_or_cutting_words():
    sentence = "இது ஒரு நீளமான வாக்கியம், இதுல நிறைய வார்த்தைகள் இருக்கு."
    text = " ".join([sentence] * 12)
    chunks = release.split_text(text, max_chars=120)
    assert len(chunks) > 1 and all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    assert release.split_text("சரி.") == ["சரி."]
    assert release.split_text("   ") == []


def test_validity_gate():
    tone = (0.2 * np.sin(np.arange(24_000) * 0.05)).astype(np.float32)
    assert release.is_valid(tone, ended=True)
    assert not release.is_valid(tone, ended=False)            # ran to the cap
    assert not release.is_valid(np.zeros(24_000, np.float32), ended=True)   # mute
    assert not release.is_valid(np.zeros(0, np.float32), ended=True)


def test_numbers_are_read_by_the_shipped_normaliser():
    assert release.verbalise("₹500") == "ஐநூறு ரூபாய்"


def test_breath_plan_places_one_breath():
    tag = release.BREATH_TAG
    assert release.breath_plan("சரி நான் சொல்றேன்") == ("சரி நான் சொல்றேன்", None)
    assert release.breath_plan("[breath] சரி நான் சொல்றேன்") == (f"{tag} சரி நான் சொல்றேன்", "lead")
    assert release.breath_plan(f"சரி {tag} நான் சொல்றேன்")[1] == "mid"
    assert release.breath_plan("[Breath] சரி [breath] நான்")[1] == "lead"      # never both


def test_a_mid_tag_can_cut_the_sentence_and_lead_the_second_half():
    tag = release.BREATH_TAG
    units = release.split_at_breath(f"சரி நான் சொல்றேன், {tag} கவனமா கேளு {tag} இப்போ")
    assert units == [("சரி நான் சொல்றேன்,", None, False),
                     (f"{tag} கவனமா கேளு இப்போ", "lead", True)]      # one breath, at the tag
    assert release.breath_plan(units[1][0])[1] == "lead"
    assert release.split_at_breath(f"சரி {tag}") == [(f"சரி {tag}", None, False)]


def test_shipped_breaths_are_whole_frames_for_both_voices():
    breaths = release.load_breaths()
    assert set(breaths) == set(release.VOICES)
    for voice in release.VOICES:
        assert len(breaths[voice]) >= 5
        for tokens in breaths[voice]:
            assert len(tokens) % release.FRAME_TOKENS == 0 and 2 <= len(tokens) // 7 <= 8
    assert release.load_breaths("no_such_file.json") == {}
