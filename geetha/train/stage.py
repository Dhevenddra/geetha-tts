r"""The full-length trainer for Stage A and Stage B (gate M2 onward).

    python -m geetha.train.stage --stage A --run-name m2-main --planned-seconds 36000 \
        --max-epochs 2

``pilot.py`` proved the path: tokens in, adapter out, audible Tamil. It stays as the
record of what the pilot ran. This is what a real run needs on top of it, and each item
is here because the pilot's log or its audio asked for it (D-051):

* **Batches, not single clips** (:mod:`geetha.train.batching`). The pilot ran one clip
  per forward pass; the same pod-hour now carries several times the gradient signal.
* **Warm-up and a cosine that lands when the run stops** (:mod:`geetha.train.schedule`),
  whichever of the token target or the paid wall-clock stops it.
* **Non-verbal tag dropout and Stage B replay** (:mod:`geetha.train.corpus`) -- the
  recipe's answer to D-050.
* **A real resume.** The pilot's ``--resume-from`` restored weights and nothing else.
  Here every checkpoint carries the epoch, the batch index and the tokens seen, and the
  batch order is a pure function of ``(seed, epoch)``, so a relaunched pod continues the
  same run with the same command. A finished stage is detected and skipped.
* **A held-out loss per source**, so "the loss fell" can be told apart from "the model
  got better at the phone-derived clips and worse at Anitha".
* **Checkpoints without a copy of the base embedding table.** PEFT saves the whole
  embedding matrix whenever ``embed_tokens`` is a LoRA target; that was 0.96 GB of
  every 1.39 GB pilot checkpoint, identical to the base model's every time.

What it still deliberately does not do: choose its own hours or its own GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from geetha.train.batching import token_budget_batches
from geetha.train.budget import TrainingBudget
from geetha.train.collate import collate
from geetha.train.corpus import FULL, Corpus, Item, epoch_plan, load_items, replay_pool
from geetha.train.schedule import lr_factor, progress

CONTEXT = 2_560                 # D-041: keeps 100% of Stage A's clips
PAD_TOKEN = "<|pad|>"
BASE_MODEL = "bodhan-ai/indic-speak"
MODEL_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CODES_REPO = {"A": "Dhevenddra/geetha-tts-codes-stagea-private",
              "B": "Dhevenddra/geetha-tts-codes-stageb-private"}
REPLAY_SHARD_PREFIX = "indicvoices_r__"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "embed_tokens"]
STATE_FILE = "trainer_state.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- state

@dataclass
class TrainerState:
    """Everything needed to continue a run on a different pod."""
    stage: str
    epoch: int = 0
    batch_index: int = 0
    micro_steps: int = 0
    optimiser_steps: int = 0
    tokens_seen: int = 0
    elapsed_s: float = 0.0
    finished: bool = False
    stop_reason: str = ""

    def write(self, folder: Path) -> None:
        (folder / STATE_FILE).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def read(cls, folder: Path) -> "TrainerState":
        return cls(**json.loads((folder / STATE_FILE).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- stores

class LocalStore:
    """A checkpoint store on a directory. What the tests use instead of the Hub."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def files(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix()
                      for p in self.root.rglob("*") if p.is_file())

    def upload(self, folder: Path, prefix: str) -> None:
        target = self.root / prefix
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(folder, target)

    def download(self, prefix: str, target: Path) -> Path:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.root / prefix, target)
        return target


class HubStore:
    """The private checkpoint repo. One commit per checkpoint, so never half-written."""

    def __init__(self, repo: str, token: str) -> None:
        from huggingface_hub import HfApi

        self.repo = repo
        self.token = token
        self.api = HfApi(token=token)
        self.api.create_repo(repo, repo_type="model", private=True, exist_ok=True)

    def files(self) -> list[str]:
        return sorted(self.api.list_repo_files(self.repo, repo_type="model"))

    def upload(self, folder: Path, prefix: str) -> None:
        self.api.upload_folder(folder_path=str(folder), repo_id=self.repo,
                               repo_type="model", path_in_repo=prefix,
                               commit_message=f"checkpoint {prefix}")

    def download(self, prefix: str, target: Path) -> Path:
        from huggingface_hub import snapshot_download

        staging = target.parent / (target.name + "__dl")
        snapshot_download(self.repo, repo_type="model", token=self.token,
                          allow_patterns=[f"{prefix}/*"], local_dir=str(staging))
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging / prefix), str(target))
        shutil.rmtree(staging, ignore_errors=True)
        return target


def latest_checkpoint(files: Sequence[str], prefix: str) -> tuple[str | None, bool]:
    """``(checkpoint_prefix, stage_is_finished)`` for the run at ``prefix``.

    Only a checkpoint whose state file is present counts: the state file is written
    into the same commit as the weights, so its presence means the folder is whole.
    """
    if any(name == f"{prefix}/final/{STATE_FILE}" for name in files):
        return f"{prefix}/final", True
    best = -1
    for name in files:
        if not name.startswith(f"{prefix}/ckpt-") or not name.endswith(f"/{STATE_FILE}"):
            continue
        number = name[len(prefix) + len("/ckpt-"):].split("/")[0]
        if number.isdigit():
            best = max(best, int(number))
    if best < 0:
        return None, False
    return f"{prefix}/ckpt-{best:07d}", False


# --------------------------------------------------------------------------- args

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "C"], required=True,
                        help="C = self-augmentation fine-tune (D-057); its shards are "
                             "handed in by the caller, there is no codes repo for it")
    parser.add_argument("--run-name", required=True,
                        help="names the checkpoint folder; reuse it to resume the same run")
    parser.add_argument("--planned-seconds", type=float, required=True,
                        help="training wall-clock for this stage, summed over resumes; "
                             "comes from the approved gate")
    parser.add_argument("--max-epochs", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-tokens", type=int, default=6000,
                        help="padded tokens per forward pass")
    parser.add_argument("--grad-accum", type=int, default=3)
    parser.add_argument("--nv-dropout", type=float, default=0.3)
    parser.add_argument("--style-dropout", type=float, default=0.1)
    parser.add_argument("--env-dropout", type=float, default=0.15,
                        help="studio clips only, so an absent env always means studio")
    parser.add_argument("--replay-fraction", type=float, default=0.0,
                        help="Stage B: share of each epoch replayed from Stage A's "
                             "event-bearing clips (D-050)")
    parser.add_argument("--init-from-stage", default="",
                        help="start from <run-name>/stage<X>/final instead of a fresh LoRA")
    parser.add_argument("--held-out-per-thousand", type=int, default=5)
    parser.add_argument("--checkpoint-minutes", type=float, default=20.0)
    parser.add_argument("--shards", type=int, default=0, help="0 = every shard")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="trade ~30%% speed for memory; switched on automatically "
                             "after an out-of-memory error")
    parser.add_argument("--checkpoint-repo", default="Dhevenddra/geetha-ckpt-private")
    parser.add_argument("--out", default="/workspace/out")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- pieces

def _real_model_and_tokenizer(args: argparse.Namespace, token: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from geetha.text.tokenizer_check import assert_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, token=token)
    assert_tokenizer(dict(tokenizer.get_added_vocab()), len(tokenizer))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"loading {BASE_MODEL}@{MODEL_REVISION[:8]} in {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION, token=token, dtype=dtype, device_map={"": 0})
    return model, tokenizer


def _real_shards(args: argparse.Namespace, token: str) -> tuple[list[Path], list[Path]]:
    """``(stage_shards, replay_shards)`` downloaded from the private code repos."""
    from huggingface_hub import hf_hub_download, list_repo_files

    def fetch(repo: str, keep: Callable[[str], bool]) -> list[Path]:
        names = sorted(n for n in list_repo_files(repo, repo_type="dataset", token=token)
                       if n.endswith(".parquet") and keep(n))
        if args.shards:
            # Every k-th shard rather than the first k: the names sort by source, and
            # the pilot's "first 12" was twelve shards of one corpus.
            stride = max(1, len(names) // args.shards)
            names = names[::stride][: args.shards]
        return [Path(hf_hub_download(repo, n, repo_type="dataset", token=token))
                for n in names]

    stage_shards = fetch(CODES_REPO[args.stage], lambda n: True)
    replay_shards: list[Path] = []
    if args.stage == "B" and args.replay_fraction > 0:
        replay_shards = fetch(CODES_REPO["A"], lambda n: n.startswith(REPLAY_SHARD_PREFIX))
    return stage_shards, replay_shards


def held_out_loss(model: Any, items: Sequence[Item], *, pad_id: int, device: str,
                  batch_tokens: int, per_group: int = 96) -> dict[str, float]:
    """Mean per-token loss on the held-out slice, by source group."""
    import torch

    by_group: dict[str, list[Item]] = {}
    for item in items:
        bucket = by_group.setdefault(item.group, [])
        if len(bucket) < per_group:
            bucket.append(item)

    was_training = model.training
    model.eval()
    result: dict[str, float] = {}
    with torch.no_grad():
        for group, members in sorted(by_group.items()):
            lengths = [m.length for m in members]
            total, count = 0.0, 0
            for batch_ids in token_budget_batches(lengths, max_tokens=batch_tokens,
                                                  seed=0, epoch=0):
                batch = collate([members[i].sequence() for i in batch_ids], pad_id=pad_id)
                output = model(
                    input_ids=torch.as_tensor(batch.input_ids, device=device),
                    labels=torch.as_tensor(batch.labels, device=device),
                    attention_mask=torch.as_tensor(batch.attention_mask, device=device))
                total += float(output.loss) * batch.supervised_tokens
                count += batch.supervised_tokens
            result[group] = total / max(1, count)
    if was_training:
        model.train()
    return result


# --------------------------------------------------------------------------- main

def run(
    args: argparse.Namespace,
    *,
    store: Any = None,
    model_and_tokenizer: Callable[..., Any] | None = None,
    shards: Callable[..., tuple[list[Path], list[Path]]] | None = None,
    clock: Callable[[], float] = time.time,
) -> int:
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    from geetha.data.loader import LoaderStats

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    if store is None or model_and_tokenizer is None or shards is None:
        if not token:
            raise SystemExit("HF_TOKEN is not set; the pod cannot reach the private repos")
    store = store if store is not None else HubStore(args.checkpoint_repo, token)
    prefix = f"{args.run_name}/stage{args.stage}"
    final_local = out / f"adapter_stage{args.stage}_final"

    # ---- is there anything to do, and where from? -----------------------------
    resume_prefix, finished = latest_checkpoint(store.files(), prefix)
    if finished:
        log(f"{prefix} is already finished on the checkpoint store; nothing to train")
        store.download(resume_prefix, final_local)
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = (model_and_tokenizer or _real_model_and_tokenizer)(args, token)
    # Wherever the model actually is, not wherever a GPU happens to exist.
    device = next(model.parameters()).device.type
    pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    breath_id = tokenizer.convert_tokens_to_ids("<|nv_breath|>")
    control_id = tokenizer.convert_tokens_to_ids("<|nv_burp|>")

    # ---- the data -------------------------------------------------------------
    stage_shards, replay_shards = (shards or _real_shards)(args, token)
    stats = LoaderStats()

    def encode(text: str) -> list[int]:
        # No tokenizer-added specials: the prompt is laid out exactly as the base
        # model's inference.py lays it out, and that layout has one <|begin_of_text|>.
        return tokenizer.encode(text, add_special_tokens=False)

    corpus: Corpus = load_items(stage_shards, encode, max_length=CONTEXT, stats=stats,
                                held_out_per_thousand=args.held_out_per_thousand)
    items = list(corpus.train)
    held_out = list(corpus.held_out)
    replay_from: list[int] = []
    if replay_shards:
        replay_corpus = load_items(replay_shards, encode, max_length=CONTEXT, stats=stats,
                                   held_out_per_thousand=args.held_out_per_thousand)
        events = [replay_corpus.train[i] for i in replay_pool(replay_corpus.train)]
        replay_from = list(range(len(items), len(items) + len(events)))
        items.extend(events)
        held_out.extend(replay_corpus.held_out)
    if not items:
        raise SystemExit("the loader produced no usable examples")
    lengths = [item.length for item in items]
    honour_repeats = args.stage in ("B", "C")

    def plan(epoch: int):
        indices, variants = epoch_plan(
            items, seed=args.seed, epoch=epoch, honour_repeats=honour_repeats,
            nv_dropout=args.nv_dropout, style_dropout=args.style_dropout,
            env_dropout=args.env_dropout, replay_from=replay_from,
            replay_fraction=args.replay_fraction)
        batches = token_budget_batches(lengths, max_tokens=args.batch_tokens,
                                       seed=args.seed, epoch=epoch, indices=indices)
        return indices, variants, batches

    indices0, variants0, batches0 = plan(0)
    epoch_tokens = sum(lengths[i] for i in indices0)
    target_tokens = max(1, int(epoch_tokens * args.max_epochs))
    groups: dict[str, int] = {}
    for index in indices0:
        groups[items[index].group] = groups.get(items[index].group, 0) + 1
    log(f"stage {args.stage}: {len(items):,} clips ({len(replay_from):,} replay), "
        f"{len(held_out):,} held out; skipped {stats.skipped_too_long:,} too long, "
        f"{stats.skipped_malformed:,} malformed")
    log(f"one epoch = {len(indices0):,} examples, {epoch_tokens:,} tokens, "
        f"{len(batches0):,} batches; by group {groups}; "
        f"shown untagged {sum(v.untagged for v in variants0.values()):,}, "
        f"without style {sum(v.drop_style for v in variants0.values()):,}, "
        f"without env {sum(v.drop_env for v in variants0.values()):,}")

    # ---- the model ------------------------------------------------------------
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    grad_checkpointing = bool(args.grad_checkpointing)

    state = TrainerState(stage=args.stage)
    if resume_prefix:
        local = store.download(resume_prefix, out / "resume")
        state = TrainerState.read(local)
        log(f"resuming {resume_prefix}: epoch {state.epoch}, batch {state.batch_index}, "
            f"{state.tokens_seen:,} tokens, {state.elapsed_s / 3600:.2f} h already trained")
        peft_model = PeftModel.from_pretrained(model, str(local), is_trainable=True)
    elif args.init_from_stage:
        source = out / f"adapter_stage{args.init_from_stage}_final"
        if not (source / "adapter_config.json").exists():
            store.download(f"{args.run_name}/stage{args.init_from_stage}/final", source)
        log(f"continuing the adapter from {source}")
        peft_model = PeftModel.from_pretrained(model, str(source), is_trainable=True)
    else:
        peft_model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            # embed_tokens is non-negotiable (D-039): without it the inert non-verbal
            # tokens can never be learned, and nothing in the loss would say so.
            target_modules=LORA_TARGETS,
        ))
    parameters = [p for p in peft_model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in parameters)
    dtypes = sorted({str(p.dtype) for p in parameters})
    log(f"LoRA rank {args.lora_rank}: {trainable:,} trainable parameters, dtypes {dtypes}")
    if any("float32" not in d for d in dtypes):
        # AdamW on bf16 weights silently drops every update smaller than ~1/256 of the
        # weight, which at lr 1e-4 is most of them. PEFT casts adapters to fp32 by
        # default; if that ever stops being true the run must not start.
        raise SystemExit(f"trainable parameters are {dtypes}, not float32")

    def embed_of(token_id: int):
        with torch.no_grad():
            return peft_model.get_input_embeddings()(
                torch.tensor([token_id], device=device)).detach().float().cpu().clone()

    breath_before, control_before = embed_of(breath_id), embed_of(control_id)

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT", "geetha"),
                               name=f"{args.run_name}-stage{args.stage}",
                               resume="allow", config={**vars(args), "context": CONTEXT})

    optimiser = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    # A base model in fp16 (cards without bf16, e.g. Kaggle's T4) backpropagates through
    # fp16 activations, where small gradients underflow to zero. Scaling the loss keeps
    # them representable; the adapters themselves are fp32, so unscaling is exact.
    # Disabled -- a pass-through -- for bf16 and fp32, so pod runs are unchanged.
    fp16_base = next(model.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler(device, enabled=fp16_base)
    if fp16_base:
        log("base model is fp16: gradient scaling is on")
    budget = TrainingBudget(max_seconds=10 ** 9, max_steps=10 ** 12)

    def save(tag: str) -> None:
        local = out / f"ckpt_{tag}"
        if local.exists():
            shutil.rmtree(local)
        # save_embedding_layers=False: the base embedding matrix is untouched by a LoRA
        # and PEFT would otherwise copy all 0.96 GB of it into every checkpoint.
        peft_model.save_pretrained(str(local), save_embedding_layers=False)
        state.write(local)
        try:
            store.upload(local, f"{prefix}/{tag}")
            log(f"checkpoint {prefix}/{tag} stored")
        except Exception as problem:  # noqa: BLE001 -- an upload must never end a run
            log(f"!! checkpoint upload failed ({type(problem).__name__}: {problem}); "
                f"kept locally at {local}")
            return
        if tag == "final":
            if final_local.exists():
                shutil.rmtree(final_local)
            shutil.move(str(local), str(final_local))
        else:
            shutil.rmtree(local, ignore_errors=True)

    def evaluate() -> dict[str, float]:
        if not held_out:
            return {}
        scores = held_out_loss(peft_model, held_out, pad_id=pad_id, device=device,
                               batch_tokens=args.batch_tokens)
        log("held-out loss  " + "  ".join(f"{k} {v:.4f}" for k, v in scores.items()))
        if wandb_run is not None:
            wandb_run.log({f"heldout/{k}": v for k, v in scores.items()}
                          | {"tokens_seen": state.tokens_seen})
        return scores

    # ---- train ----------------------------------------------------------------
    peft_model.train()
    session_start = clock()
    elapsed_before = state.elapsed_s
    last_checkpoint = clock()
    session_optimiser_steps = 0
    session_tokens = 0
    smoothed: float | None = None
    oom_events = 0
    first_scores = evaluate() if not resume_prefix else {}
    stop_reason = ""
    accumulated = 0

    def to_device(batch):
        return {"input_ids": torch.as_tensor(batch.input_ids, device=device),
                "labels": torch.as_tensor(batch.labels, device=device),
                "attention_mask": torch.as_tensor(batch.attention_mask, device=device)}

    epoch = state.epoch
    start_batch = state.batch_index
    while not stop_reason:
        _, variants, batches = plan(epoch)
        for batch_index in range(start_batch, len(batches)):
            state.elapsed_s = elapsed_before + (clock() - session_start)
            fraction = progress(tokens_seen=state.tokens_seen, target_tokens=target_tokens,
                                elapsed_s=state.elapsed_s, planned_s=args.planned_seconds)
            if fraction >= 1.0:
                stop_reason = "complete"
                break

            sequences = [items[i].sequence(variants.get(i, FULL))
                         for i in batches[batch_index]]
            batch = collate(sequences, pad_id=pad_id)
            output = None
            try:
                output = peft_model(**to_device(batch))
                scaler.scale(output.loss / args.grad_accum).backward()
            except torch.OutOfMemoryError:
                # A backward that runs out of memory leaves ``output`` holding the whole
                # graph. Unless it is dropped here, empty_cache frees nothing and every
                # later batch fails too (15 Nov: 26 failures in a row on a T4).
                output = None
                oom_events += 1
                optimiser.zero_grad(set_to_none=True)
                accumulated = 0
                gc.collect()
                torch.cuda.empty_cache()
                log(f"!! out of memory on a batch of {batch.input_ids.shape} "
                    f"(event {oom_events}); still held "
                    f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")
                if not grad_checkpointing:
                    model.gradient_checkpointing_enable()
                    grad_checkpointing = True
                    log("   gradient checkpointing switched on for the rest of the run")
                if oom_events > 25:
                    stop_reason = "out_of_memory"
                    break
                continue

            loss = float(output.loss.detach())
            reason = budget.should_stop(step=state.micro_steps, elapsed=0.0, loss=loss)
            if reason:
                stop_reason = reason
                break
            smoothed = loss if smoothed is None else 0.98 * smoothed + 0.02 * loss
            accumulated += 1
            state.micro_steps += 1
            state.tokens_seen += int(batch.attention_mask.sum())
            session_tokens += int(batch.attention_mask.sum())
            state.epoch, state.batch_index = epoch, batch_index + 1

            rate = args.learning_rate * lr_factor(
                optimiser_step=session_optimiser_steps, warmup_steps=args.warmup_steps,
                progress_fraction=fraction)
            if accumulated >= args.grad_accum:
                for group in optimiser.param_groups:
                    group["lr"] = rate
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                accumulated = 0
                state.optimiser_steps += 1
                session_optimiser_steps += 1

            if state.micro_steps % 20 == 0:
                session_s = max(1e-6, clock() - session_start)
                memory = (torch.cuda.max_memory_allocated() / 2 ** 30
                          if device == "cuda" else 0.0)
                log(f"e{epoch} b{batch_index + 1}/{len(batches)}  loss {loss:.4f} "
                    f"(avg {smoothed:.4f})  lr {rate:.2e}  {fraction * 100:5.1f}%  "
                    f"{session_tokens / session_s:,.0f} tok/s  "
                    f"rows {batch.input_ids.shape[0]} pad {batch.padding_tokens}  "
                    f"mem {memory:.1f}G")
                if wandb_run is not None:
                    wandb_run.log({"loss": loss, "loss_avg": smoothed, "lr": rate,
                                   "progress": fraction, "tokens_seen": state.tokens_seen,
                                   "tokens_per_s": session_tokens / session_s,
                                   "elapsed_h": state.elapsed_s / 3600})

            if clock() - last_checkpoint >= args.checkpoint_minutes * 60:
                evaluate()
                save(f"ckpt-{state.optimiser_steps:07d}")
                last_checkpoint = clock()
        else:
            epoch += 1
            start_batch = 0
            state.epoch, state.batch_index = epoch, 0
            log(f"epoch {epoch} begins")
            continue
        break

    state.elapsed_s = elapsed_before + (clock() - session_start)
    state.stop_reason = stop_reason
    state.finished = stop_reason == "complete"
    log(f"stopped: {stop_reason} after {state.optimiser_steps:,} optimiser steps, "
        f"{state.tokens_seen:,} tokens, {state.elapsed_s / 3600:.2f} h")
    last_scores = evaluate()
    # A run that stopped for a bad reason still stores what it has -- but not as
    # "final", or the next launch would take a diverged adapter for a finished stage.
    save("final" if state.finished else f"ckpt-{state.optimiser_steps:07d}")

    breath_delta = float((embed_of(breath_id) - breath_before).norm())
    control_delta = float((embed_of(control_id) - control_before).norm())
    result = {
        **asdict(state),
        "run_name": args.run_name,
        "examples_per_epoch": len(indices0),
        "tokens_per_epoch": epoch_tokens,
        "replay_clips": len(replay_from),
        "held_out": len(held_out),
        "held_out_loss_first": first_scores,
        "held_out_loss_last": last_scores,
        "loss_avg_last": smoothed,
        "trainable_params": trainable,
        "tokens_per_s": session_tokens / max(1e-6, clock() - session_start),
        "oom_events": oom_events,
        "grad_checkpointing": grad_checkpointing,
        "embedding_delta_this_session": {"nv_breath": breath_delta,
                                         "nv_burp_control": control_delta},
        "args": {k: v for k, v in vars(args).items()},
    }
    (out / f"stage{args.stage}_result.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    log(f"breath embedding moved {breath_delta:.6f} vs control {control_delta:.6f}")
    if wandb_run is not None:
        wandb_run.finish()
    return 0 if state.finished else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
