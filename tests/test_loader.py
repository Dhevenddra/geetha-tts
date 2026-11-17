"""Turning encoded shards into training examples (what B10's loader does).

The tokenizer is injected rather than loaded, so these run offline and so the one
genuinely dangerous input -- a prompt whose text encoding is wrong -- is visible.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from geetha.data.loader import examples_from_shard, prompt_text
from geetha.data.sequence import IGNORE_INDEX
from geetha.text.contract import AUDIO_TOKEN_BASE_ID

FRAME = [AUDIO_TOKEN_BASE_ID + position * 4096 + position for position in range(7)]


def fake_encode(text: str) -> list[int]:
    """Stand-in tokenizer: one id per character, offset into the text band."""
    return [ord(character) % 128_000 for character in text]


def write_codes(path, rows):
    table = pa.table({
        "source": [row.get("source", "rasa") for row in rows],
        "shard": [row.get("shard", "train-00000.parquet") for row in rows],
        "row_index": pa.array([row.get("row_index", 0) for row in rows], pa.int32()),
        "speaker": [row.get("speaker", "anitha") for row in rows],
        "style": [row.get("style", "CONV") for row in rows],
        "env": [row.get("env", "studio") for row in rows],
        "text": [row.get("text", "வணக்கம்") for row in rows],
        "duration_s": [row.get("duration_s", 2.0) for row in rows],
        "gain_db": [row.get("gain_db", -1.0) for row in rows],
        "repeat": pa.array([row.get("repeat", 1) for row in rows], pa.int32()),
        "n_frames": pa.array([len(row.get("tokens", FRAME)) // 7 for row in rows], pa.int32()),
        "n_tokens": pa.array([len(row.get("tokens", FRAME)) for row in rows], pa.int32()),
        "tokens": pa.array([row.get("tokens", FRAME) for row in rows], pa.list_(pa.int32())),
    })
    pq.write_table(table, path)
    return path


class TestPromptText:
    def test_the_conditioning_blocks_appear_in_the_documented_order(self):
        text = prompt_text(speaker="anitha", style="CONV", text="வணக்கம்", env="studio")
        assert text.index("<|speaker>") < text.index("<|style>") < text.index("<|env>")

    def test_the_prompt_stops_at_start_of_ai_so_the_model_emits_the_speech_token(self):
        assert prompt_text(speaker="a", style="b", text="c", env="studio").endswith(
            "<|start_of_ai|>")
        assert "<|start_of_speech|>" not in prompt_text(
            speaker="a", style="b", text="c", env="studio")

    def test_an_unknown_env_label_is_refused(self):
        with pytest.raises(ValueError, match="telephone"):
            prompt_text(speaker="a", style="b", text="c", env="telephone")


class TestExamplesFromShard:
    def test_one_example_per_row(self, tmp_path):
        path = write_codes(tmp_path / "s.parquet", [{}, {}])
        assert len(list(examples_from_shard(path, fake_encode))) == 2

    def test_the_audio_span_is_the_shard_tokens(self, tmp_path):
        path = write_codes(tmp_path / "s.parquet", [{"tokens": FRAME + FRAME}])
        example = next(iter(examples_from_shard(path, fake_encode)))
        supervised = [label for label in example.labels if label != IGNORE_INDEX]
        # 14 audio tokens plus start_of_speech, end_of_speech, end_of_ai.
        assert len(supervised) == 17

    def test_the_prompt_is_masked(self, tmp_path):
        path = write_codes(tmp_path / "s.parquet", [{}])
        example = next(iter(examples_from_shard(path, fake_encode)))
        assert example.labels[0] == IGNORE_INDEX

    def test_a_repeat_count_yields_that_many_copies(self, tmp_path):
        # Stage B asks for a clip 2 or 3 times; the loader is where that happens, so a
        # repeat is never baked into the stored codes.
        path = write_codes(tmp_path / "s.parquet", [{"repeat": 3}])
        assert len(list(examples_from_shard(path, fake_encode))) == 3

    def test_repeats_can_be_ignored_for_stage_a(self, tmp_path):
        path = write_codes(tmp_path / "s.parquet", [{"repeat": 3}])
        assert len(list(examples_from_shard(path, fake_encode, honour_repeats=False))) == 1

    def test_a_row_too_long_for_the_context_is_skipped_not_truncated(self, tmp_path):
        path = write_codes(tmp_path / "s.parquet",
                           [{"tokens": FRAME * 500}, {"tokens": FRAME}])
        kept = list(examples_from_shard(path, fake_encode, max_length=256))
        assert len(kept) == 1

    def test_a_corrupt_audio_span_is_skipped_rather_than_trained_on(self, tmp_path):
        path = write_codes(tmp_path / "s.parquet",
                           [{"tokens": [1, 2, 3, 4, 5, 6, 7]}, {"tokens": FRAME}])
        assert len(list(examples_from_shard(path, fake_encode))) == 1

    def test_the_text_reaches_the_tokenizer_with_its_tags_intact(self, tmp_path):
        marked = "ஒரு <|nv_breath|> இரண்டு"
        path = write_codes(tmp_path / "s.parquet", [{"text": marked}])
        seen: list[str] = []

        def spy(text: str) -> list[int]:
            seen.append(text)
            return fake_encode(text)

        next(iter(examples_from_shard(path, spy)))
        assert "<|nv_breath|>" in seen[0]
