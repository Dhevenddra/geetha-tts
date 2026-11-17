"""The full-length trainer: batching, schedule, corpus, and the loop end to end.

The end-to-end tests run the real training loop on a two-layer model on the CPU. They
exist because every launch bug this project has had was found by launching, at pod
prices; a loop that has run to completion and resumed on this machine has one fewer
way to die on a paid one.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import pytest

from geetha.text.contract import AUDIO_TOKEN_BASE_ID
from geetha.train.batching import padded_tokens, token_budget_batches
from geetha.train.corpus import (FULL, Variant, epoch_plan, is_held_out, load_items,
                                 replay_pool, strip_nonverbal)
from geetha.train.schedule import lr_factor, progress
from tests.test_loader import FRAME, fake_encode, write_codes


# --------------------------------------------------------------------------- batching

class TestTokenBudgetBatches:
    LENGTHS = [154, 300, 2554, 800, 820, 790, 160, 150, 1200, 1250, 500, 505] * 20

    def test_every_example_appears_exactly_once(self):
        batches = token_budget_batches(self.LENGTHS, max_tokens=6000, seed=1, epoch=0)
        seen = Counter(i for batch in batches for i in batch)
        assert seen == Counter(range(len(self.LENGTHS)))

    def test_no_batch_pads_past_the_budget(self):
        batches = token_budget_batches(self.LENGTHS, max_tokens=6000, seed=1, epoch=0)
        assert max(padded_tokens(self.LENGTHS, b) for b in batches) <= 6000

    def test_a_clip_longer_than_the_budget_is_kept_as_a_batch_of_one(self):
        batches = token_budget_batches([100, 9000, 120], max_tokens=1000, seed=1, epoch=0)
        assert [1] in batches
        assert sum(len(b) for b in batches) == 3

    def test_bucketing_wastes_little_on_padding(self):
        batches = token_budget_batches(self.LENGTHS, max_tokens=6000, seed=1, epoch=0)
        real = sum(self.LENGTHS)
        held = sum(padded_tokens(self.LENGTHS, b) for b in batches)
        assert (held - real) / held < 0.10

    def test_the_order_is_a_pure_function_of_seed_and_epoch(self):
        first = token_budget_batches(self.LENGTHS, max_tokens=6000, seed=7, epoch=3)
        again = token_budget_batches(self.LENGTHS, max_tokens=6000, seed=7, epoch=3)
        other = token_budget_batches(self.LENGTHS, max_tokens=6000, seed=7, epoch=4)
        assert first == again
        assert first != other

    def test_repeated_indices_are_all_served(self):
        batches = token_budget_batches([100, 200], max_tokens=1000, seed=1, epoch=0,
                                       indices=[0, 0, 0, 1])
        assert Counter(i for b in batches for i in b) == Counter({0: 3, 1: 1})

    def test_the_row_cap_holds(self):
        batches = token_budget_batches([10] * 500, max_tokens=10 ** 6, seed=1, epoch=0,
                                       max_rows=16)
        assert max(len(b) for b in batches) == 16

    @pytest.mark.parametrize("bad", [dict(max_tokens=0), dict(max_tokens=10, max_rows=0),
                                     dict(max_tokens=10, pool=0)])
    def test_nonsense_limits_are_refused(self, bad):
        with pytest.raises(ValueError):
            token_budget_batches([1, 2], seed=1, epoch=0, **bad)


# --------------------------------------------------------------------------- schedule

class TestSchedule:
    def test_progress_takes_whichever_budget_is_further_along(self):
        assert progress(tokens_seen=10, target_tokens=100,
                        elapsed_s=50, planned_s=100) == 0.5
        assert progress(tokens_seen=80, target_tokens=100,
                        elapsed_s=50, planned_s=100) == 0.8

    def test_progress_is_capped_at_one(self):
        assert progress(tokens_seen=500, target_tokens=100, elapsed_s=0, planned_s=1) == 1.0

    def test_warmup_starts_small_and_reaches_the_peak(self):
        first = lr_factor(optimiser_step=0, warmup_steps=100, progress_fraction=0.0)
        last = lr_factor(optimiser_step=100, warmup_steps=100, progress_fraction=0.0)
        assert first == pytest.approx(0.01)
        assert last == pytest.approx(1.0)

    def test_the_cosine_lands_on_the_floor_when_the_run_ends(self):
        assert lr_factor(optimiser_step=10 ** 6, warmup_steps=100,
                         progress_fraction=1.0, floor=0.05) == pytest.approx(0.05)

    def test_the_rate_never_rises_after_warmup(self):
        values = [lr_factor(optimiser_step=10 ** 6, warmup_steps=0, progress_fraction=f / 50)
                  for f in range(51)]
        assert values == sorted(values, reverse=True)

    def test_a_resumed_run_warms_up_into_the_decayed_rate_not_the_peak(self):
        resumed = lr_factor(optimiser_step=10 ** 6, warmup_steps=100, progress_fraction=0.9)
        assert resumed < 0.1


# --------------------------------------------------------------------------- corpus

def audio(frames: int) -> list[int]:
    return FRAME * frames


def shard(tmp_path, name, rows):
    return write_codes(tmp_path / name, rows)


class TestStripNonverbal:
    def test_tokens_are_removed_and_the_gap_closed(self):
        assert strip_nonverbal("அது <|nv_breath|> சரி <|nv_laugh|>") == "அது சரி"

    def test_text_without_tokens_is_untouched(self):
        assert strip_nonverbal("வணக்கம், நலமா?") == "வணக்கம், நலமா?"

    def test_other_special_tokens_survive(self):
        assert "<|eot_id|>" in strip_nonverbal("a <|nv_breath|> b<|eot_id|>")


class TestLoadItems:
    def test_an_event_clip_gets_both_forms_and_a_plain_clip_only_one(self, tmp_path):
        path = shard(tmp_path, "a.parquet", [
            {"row_index": 0, "speaker": "ivr_1", "env": "enhanced",
             "text": "ஒன்று <|nv_breath|> இரண்டு"},
            {"row_index": 1, "text": "வணக்கம்"},
        ])
        items = load_items([path], fake_encode, max_length=2560).train
        assert items[0].has_events and not items[1].has_events
        untagged = items[0].sequence(Variant(untagged=True))
        assert len(untagged.input_ids) < len(items[0].sequence().input_ids)
        assert items[0].group == "ivr" and items[1].group == "anitha"
        # Asking a clip without events for its untagged form changes nothing.
        assert items[1].sequence(Variant(untagged=True)) == items[1].sequence()

    def test_the_untagged_form_keeps_the_same_audio(self, tmp_path):
        path = shard(tmp_path, "a.parquet", [
            {"speaker": "ivr_1", "env": "enhanced", "text": "a <|nv_breath|> b",
             "tokens": audio(5)}])
        item = load_items([path], fake_encode, max_length=2560).train[0]
        assert item.sequence(Variant(untagged=True)).input_ids[-37:] == \
            item.sequence().input_ids[-37:]

    def test_the_full_prompt_is_the_documented_layout_token_for_token(self, tmp_path):
        from geetha.data.loader import prompt_text

        path = shard(tmp_path, "a.parquet", [
            {"speaker": "ivr_9", "style": "extempore", "env": "enhanced",
             "text": "ஒன்று <|nv_breath|> இரண்டு"}])
        item = load_items([path], fake_encode, max_length=2560).train[0]
        whole = prompt_text(speaker="ivr_9", style="extempore", env="enhanced",
                            text="ஒன்று <|nv_breath|> இரண்டு")
        assert item.prompt_ids() == fake_encode(whole)

    def test_dropping_a_block_removes_exactly_that_block(self, tmp_path):
        from geetha.text.contract import build_tts_prompt

        path = shard(tmp_path, "a.parquet", [{"style": "HAPPY", "text": "வணக்கம்"}])
        item = load_items([path], fake_encode, max_length=2560).train[0]
        no_style = build_tts_prompt(speaker="anitha", style="", text="வணக்கம்",
                                    source="studio")
        no_env = build_tts_prompt(speaker="anitha", style="HAPPY", text="வணக்கம்")
        bare = build_tts_prompt(speaker="anitha", style="", text="வணக்கம்")
        assert item.prompt_ids(Variant(drop_style=True)) == fake_encode(no_style)
        assert item.prompt_ids(Variant(drop_env=True)) == fake_encode(no_env)
        assert item.prompt_ids(Variant(drop_style=True, drop_env=True)) == fake_encode(bare)

    def test_an_encoder_that_adds_a_bos_to_every_call_is_refused(self, tmp_path):
        path = shard(tmp_path, "a.parquet", [{}])
        with pytest.raises(ValueError, match="add_special_tokens=False"):
            load_items([path], lambda text: [128_000, *fake_encode(text)], max_length=2560)

    def test_the_held_out_slice_is_stable_and_disjoint(self, tmp_path):
        rows = [{"row_index": i} for i in range(400)]
        path = shard(tmp_path, "a.parquet", rows)
        first = load_items([path], fake_encode, max_length=2560, held_out_per_thousand=100)
        again = load_items([path], fake_encode, max_length=2560, held_out_per_thousand=100)
        assert [i.key for i in first.held_out] == [i.key for i in again.held_out]
        assert 15 < len(first.held_out) < 70
        assert not {i.key for i in first.train} & {i.key for i in first.held_out}
        assert len(first.train) + len(first.held_out) == 400

    def test_nothing_is_held_out_by_default(self):
        assert not is_held_out("anything", per_thousand=0)

    def test_a_clip_over_the_context_is_skipped_and_counted(self, tmp_path):
        from geetha.data.loader import LoaderStats

        path = shard(tmp_path, "a.parquet", [{"tokens": audio(400)}, {"row_index": 1}])
        stats = LoaderStats()
        corpus = load_items([path], fake_encode, max_length=2560, stats=stats)
        assert len(corpus.train) == 1 and stats.skipped_too_long == 1


class TestEpochPlan:
    def items(self, tmp_path):
        rows = [{"row_index": i, "repeat": 3, "style": "HAPPY"} for i in range(40)]
        rows += [{"row_index": 100 + i, "speaker": f"ivr_{i}", "env": "enhanced",
                  "text": "a <|nv_breath|> b"} for i in range(200)]
        return load_items([shard(tmp_path, "a.parquet", rows)], fake_encode,
                          max_length=2560).train

    def test_repeats_are_honoured_only_when_asked(self, tmp_path):
        items = self.items(tmp_path)
        with_repeats, _ = epoch_plan(items, seed=1, epoch=0, honour_repeats=True,
                                     nv_dropout=0.0)
        without, _ = epoch_plan(items, seed=1, epoch=0, honour_repeats=False, nv_dropout=0.0)
        assert len(with_repeats) == 40 * 3 + 200
        assert len(without) == 240

    def test_replay_makes_up_the_requested_share_and_only_from_the_pool(self, tmp_path):
        items = self.items(tmp_path)
        pool = replay_pool(items)
        assert len(pool) == 200
        indices, _ = epoch_plan(items, seed=1, epoch=0, honour_repeats=True, nv_dropout=0.0,
                                replay_from=pool, replay_fraction=0.2)
        replayed = [i for i in indices if i in set(pool)]
        assert len(indices) - len(replayed) == 120
        assert len(replayed) == 30                     # 30 / 150 = 0.2
        assert len(set(replayed)) == len(replayed)

    def test_replay_changes_between_epochs(self, tmp_path):
        items = self.items(tmp_path)
        pool = replay_pool(items)
        picks = [set(epoch_plan(items, seed=1, epoch=e, honour_repeats=True, nv_dropout=0.0,
                                replay_from=pool, replay_fraction=0.2)[0]) & set(pool)
                 for e in (0, 1)]
        assert picks[0] != picks[1]

    def test_dropout_only_ever_untags_clips_that_have_events(self, tmp_path):
        items = self.items(tmp_path)
        _, variants = epoch_plan(items, seed=1, epoch=0, honour_repeats=False,
                                 nv_dropout=0.3)
        untagged = [i for i, v in variants.items() if v.untagged]
        assert untagged and all(items[i].has_events for i in untagged)
        assert 0.18 < len(untagged) / 200 < 0.42

    def test_env_is_only_ever_dropped_from_studio_clips(self, tmp_path):
        items = self.items(tmp_path)
        _, variants = epoch_plan(items, seed=1, epoch=0, honour_repeats=False,
                                 nv_dropout=0.0, style_dropout=0.5, env_dropout=0.5)
        dropped_env = [i for i, v in variants.items() if v.drop_env]
        assert dropped_env and all(items[i].env == "studio" for i in dropped_env)
        assert any(v.drop_style for i, v in variants.items() if items[i].env == "enhanced")

    def test_no_dropout_means_every_clip_is_shown_in_full(self, tmp_path):
        items = self.items(tmp_path)
        assert epoch_plan(items, seed=1, epoch=0, honour_repeats=False,
                          nv_dropout=0.0)[1] == {}
        assert FULL == Variant()

    def test_the_plan_is_reproducible(self, tmp_path):
        items = self.items(tmp_path)
        kwargs = dict(seed=5, epoch=2, honour_repeats=True, nv_dropout=0.3,
                      replay_from=replay_pool(items), replay_fraction=0.2)
        assert epoch_plan(items, **kwargs) == epoch_plan(items, **kwargs)


# --------------------------------------------------------------------------- the loop

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

VOCAB = AUDIO_TOKEN_BASE_ID + 7 * 4096 + 64


class FakeTokenizer:
    IDS = {"<|pad|>": 128_263, "<|nv_breath|>": 156_944, "<|nv_burp|>": 156_959}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False, "the trainer must not let the tokenizer add a BOS"
        ids: list[int] = []
        for piece in text.replace("<|nv_breath|>", "\x00").split("\x00"):
            ids.extend(fake_encode(piece))
            ids.append(self.IDS["<|nv_breath|>"])
        return ids[:-1]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.IDS[token]


def tiny_model(args, token):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
        tie_word_embeddings=True)
    return transformers.LlamaForCausalLM(config), FakeTokenizer()


class StepClock:
    """Each look at the clock is one second later: wall-clock made deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def stage_args(tmp_path, **overrides):
    from geetha.train.stage import parse_args

    argv = ["--stage", "A", "--run-name", "t1", "--planned-seconds", "100000",
            "--max-epochs", "2", "--lora-rank", "4", "--batch-tokens", "400",
            "--grad-accum", "2", "--warmup-steps", "2", "--learning-rate", "0.01",
            "--held-out-per-thousand", "100", "--checkpoint-minutes", "100000",
            "--out", str(tmp_path / "out")]
    args = parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_shards(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows_a = [{"row_index": i, "speaker": f"ivr_{i % 7}", "env": "enhanced",
               "style": "extempore", "source": "indicvoices_r",
               "text": "ஒன்று <|nv_breath|> இரண்டு" if i % 2 else "மூன்று",
               "tokens": audio(3 + i % 5)} for i in range(60)]
    rows_b = [{"row_index": i, "speaker": "anitha", "style": "HAPPY", "repeat": 2,
               "tokens": audio(2 + i % 4)} for i in range(40)]
    a = shard(tmp_path, "indicvoices_r__a.parquet", rows_a)
    b = shard(tmp_path, "rasa__b.parquet", rows_b)
    return a, b


def run_stage(tmp_path, args, *, clock=None):
    from geetha.train.stage import LocalStore, run

    a, b = make_shards(tmp_path)

    def shards(arguments, token):
        if arguments.stage == "A":
            return [a, b], []
        return [b], ([a] if arguments.replay_fraction > 0 else [])

    store = LocalStore(tmp_path / "store")
    code = run(args, store=store, model_and_tokenizer=tiny_model, shards=shards,
               clock=clock or StepClock())
    return code, store


def read_result(tmp_path, stage="A"):
    import json

    return json.loads((tmp_path / "out" / f"stage{stage}_result.json").read_text("utf-8"))


class TestStageLoop:
    def test_a_stage_runs_to_completion_and_stores_a_final_adapter(self, tmp_path):
        code, store = run_stage(tmp_path, stage_args(tmp_path))
        result = read_result(tmp_path)
        assert code == 0 and result["finished"] and result["stop_reason"] == "complete"
        assert "t1/stageA/final/trainer_state.json" in store.files()
        assert "t1/stageA/final/adapter_config.json" in store.files()
        assert result["tokens_seen"] >= 2 * result["tokens_per_epoch"] * 0.95

    def test_training_actually_learns_and_only_the_breath_embedding_moves(self, tmp_path):
        run_stage(tmp_path, stage_args(tmp_path, max_epochs=6.0))
        result = read_result(tmp_path)
        first, last = result["held_out_loss_first"], result["held_out_loss_last"]
        assert set(first) == {"anitha", "ivr"}
        assert all(last[g] < first[g] for g in first)
        delta = result["embedding_delta_this_session"]
        assert delta["nv_breath"] > 0 and delta["nv_burp_control"] == 0.0

    def test_the_checkpoint_does_not_carry_the_base_embedding_table(self, tmp_path):
        _, store = run_stage(tmp_path, stage_args(tmp_path))
        weights = tmp_path / "store" / "t1" / "stageA" / "final" / "adapter_model.safetensors"
        # The tiny model's embedding table alone is VOCAB * 16 floats = ~10 MB.
        assert weights.stat().st_size < 4_000_000

    def test_a_finished_stage_is_not_trained_again(self, tmp_path):
        run_stage(tmp_path, stage_args(tmp_path))
        before = read_result(tmp_path)
        code, _ = run_stage(tmp_path, stage_args(tmp_path))
        assert code == 0
        assert read_result(tmp_path) == before          # no second result was written
        assert (tmp_path / "out" / "adapter_stageA_final" / "adapter_config.json").exists()

    def test_the_paid_wall_clock_ends_the_stage_even_if_the_epochs_are_not_done(self, tmp_path):
        run_stage(tmp_path / "whole", stage_args(tmp_path / "whole"))
        whole = read_result(tmp_path / "whole")

        # Same run, but the hours run out first. That is a *completed* stage -- the
        # cosine reached its floor on the clock -- with fewer tokens behind it.
        short = stage_args(tmp_path, planned_seconds=12.0, max_epochs=2.0)
        code, store = run_stage(tmp_path, short)
        cut = read_result(tmp_path)
        assert code == 0 and cut["stop_reason"] == "complete"
        assert cut["tokens_seen"] < whole["tokens_seen"]
        assert "t1/stageA/final/trainer_state.json" in store.files()

    def test_an_interrupted_run_continues_from_its_last_checkpoint(self, tmp_path):
        from geetha.train import stage as stage_module

        run_stage(tmp_path / "whole", stage_args(tmp_path / "whole"))
        whole = read_result(tmp_path / "whole")

        # Checkpoint constantly, then die after a few steps.
        class Died(RuntimeError):
            pass

        args = stage_args(tmp_path, checkpoint_minutes=0.0)
        original = stage_module.collate
        calls = {"n": 0}

        def dying_collate(*a, **k):
            calls["n"] += 1
            if calls["n"] > 9:
                raise Died()
            return original(*a, **k)

        stage_module.collate = dying_collate
        try:
            with pytest.raises(Died):
                run_stage(tmp_path, args)
        finally:
            stage_module.collate = original

        from geetha.train.stage import LocalStore, latest_checkpoint

        prefix, finished = latest_checkpoint(LocalStore(tmp_path / "store").files(),
                                             "t1/stageA")
        assert prefix and not finished

        code, store = run_stage(tmp_path, stage_args(tmp_path))
        resumed = read_result(tmp_path)
        assert code == 0 and resumed["finished"]
        # Same clips, same order, nothing read twice: the token count matches exactly.
        assert resumed["tokens_seen"] == whole["tokens_seen"]
        assert resumed["micro_steps"] == whole["micro_steps"]

    def test_stage_b_continues_stage_a_and_replays_event_clips(self, tmp_path):
        run_stage(tmp_path, stage_args(tmp_path))
        args_b = stage_args(tmp_path, stage="B", init_from_stage="A", replay_fraction=0.2,
                            max_epochs=1.0)
        code, store = run_stage(tmp_path, args_b)
        result = read_result(tmp_path, "B")
        assert code == 0 and result["finished"]
        assert result["replay_clips"] > 0
        assert "t1/stageB/final/adapter_config.json" in store.files()

    def test_a_nan_loss_fails_the_run_and_is_not_stored_as_final(self, tmp_path):
        from geetha.train import stage as stage_module

        class Exploding(stage_module.TrainingBudget):
            def should_stop(self, *, step, elapsed, loss):
                return super().should_stop(step=step, elapsed=elapsed,
                                           loss=math.nan if step >= 3 else loss)

        original = stage_module.TrainingBudget
        stage_module.TrainingBudget = Exploding
        try:
            code, store = run_stage(tmp_path, stage_args(tmp_path))
        finally:
            stage_module.TrainingBudget = original
        assert code == 1
        assert read_result(tmp_path)["stop_reason"] == "nan_loss"
        assert not any("/final/" in name for name in store.files())


class TestLatestCheckpoint:
    def test_the_highest_numbered_whole_checkpoint_wins(self):
        from geetha.train.stage import latest_checkpoint

        files = ["r/stageA/ckpt-0000010/trainer_state.json",
                 "r/stageA/ckpt-0000200/trainer_state.json",
                 "r/stageA/ckpt-0000300/adapter_model.safetensors"]   # no state: not whole
        assert latest_checkpoint(files, "r/stageA") == ("r/stageA/ckpt-0000200", False)

    def test_final_outranks_every_checkpoint(self):
        from geetha.train.stage import latest_checkpoint

        files = ["r/stageA/ckpt-0000010/trainer_state.json",
                 "r/stageA/final/trainer_state.json"]
        assert latest_checkpoint(files, "r/stageA") == ("r/stageA/final", True)

    def test_another_run_or_stage_is_not_mistaken_for_this_one(self):
        from geetha.train.stage import latest_checkpoint

        files = ["r/stageB/final/trainer_state.json", "r2/stageA/final/trainer_state.json"]
        assert latest_checkpoint(files, "r/stageA") == (None, False)
