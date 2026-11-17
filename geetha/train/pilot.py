r"""The pod-side trainer for Stage A and Stage B.

    python -m geetha.train.pilot --stage A --max-seconds 7200 --max-steps 4000

Runs on a Runpod pod. Everything it needs arrives over the network: the base model and
the Vocos decoder from ``bodhan-ai/indic-speak``, the audio tokens from our private code
repos. **It never touches E:** -- that is runbook step 7's done-when, and it is why the
codes were pushed at 118 MB instead of the audio at 88 GB (D-037).

What this trainer is careful about, and why
-------------------------------------------
**The tokenizer is asserted before the first gradient step.** ``token_contract.md`` §6:
compile data with one tokenizer and train with another and "training runs, loss
descends, and the model learns the wrong symbol for every audio frame". A mismatch here
costs the whole run, so it is a hard failure before anything is spent.

**``embed_tokens`` is in the LoRA target list (D-039).** D-021 measured the 16
non-verbal tokens inert in the base model. A projection-only LoRA never touches the
embedding matrix, so ``<|nv_breath|>`` would stay a meaningless vector for the whole run
and the breath headline would be structurally unlearnable -- with a perfectly healthy
loss curve throughout. B10 confirmed the fix works: the breath embedding moved 0.165
while an unused control token moved exactly 0.000.

**It stops itself.** ``TrainingBudget`` enforces the wall clock, the step cap, NaN loss
and divergence. A pod that trains past its budget is the failure mode that costs real
money, and the on-pod watchdog is only the second line of defence.

**It checkpoints to the Hub, not to the pod.** A pod can vanish; everything long-running
on this project must be resumable from a checkpoint on the private HF repo.

What it deliberately does not do: choose its own hours or its own GPU. Those come from
the approved gate, through the job spec, and are the operator's money.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

CONTEXT = 2_560                 # D-041: keeps 100% of Stage A's clips
PAD_TOKEN = "<|pad|>"
BASE_MODEL = "bodhan-ai/indic-speak"
MODEL_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CODES_REPO = {"A": "Dhevenddra/geetha-tts-codes-stagea-private",
              "B": "Dhevenddra/geetha-tts-codes-stageb-private"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--max-seconds", type=float, required=True,
                        help="wall-clock budget; comes from the approved gate")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--shards", type=int, default=0,
                        help="0 = every shard; a small number for a pilot")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--checkpoint-repo", default="Dhevenddra/geetha-ckpt-private")
    parser.add_argument("--resume-from", default="",
                        help="adapter path or repo revision to continue from")
    parser.add_argument("--out", default="/workspace/out")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from geetha.data.loader import LoaderStats, examples_from_shard
    from geetha.train.budget import TrainingBudget
    from geetha.train.collate import collate
    from geetha.text.tokenizer_check import assert_tokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; the pod cannot reach the private repos")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- the tokenizer, before anything is spent ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REVISION, token=token)
    assert_tokenizer(dict(tokenizer.get_added_vocab()), len(tokenizer))
    pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    log(f"tokenizer asserted: {len(tokenizer):,} vocab, pad id {pad_id}")

    # ---- the data -------------------------------------------------------------
    repo = CODES_REPO[args.stage]
    names = [n for n in list_repo_files(repo, repo_type="dataset", token=token)
             if n.endswith(".parquet")]
    names.sort()
    if args.shards:
        names = names[: args.shards]
    log(f"stage {args.stage}: {len(names)} code shards from {repo}")

    stats = LoaderStats()
    examples = []
    for name in names:
        path = Path(hf_hub_download(repo, name, repo_type="dataset", token=token))
        examples.extend(examples_from_shard(path, tokenizer.encode,
                                            max_length=CONTEXT, stats=stats))
    if not examples:
        raise SystemExit("the loader produced no usable examples")
    random.shuffle(examples)
    breath_id = tokenizer.convert_tokens_to_ids("<|nv_breath|>")
    with_breath = sum(1 for e in examples if breath_id in e.input_ids)
    log(f"{len(examples):,} examples ({with_breath:,} carry a breath token); "
        f"skipped {stats.skipped_too_long:,} too long, {stats.skipped_malformed:,} malformed")

    # ---- the model ------------------------------------------------------------
    # bf16 where the card supports it. The pilot targets a 48 GB A6000, so no 4-bit
    # quantisation: the pilot should resemble the main run, not the smoke test.
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    log(f"loading {BASE_MODEL} in {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=MODEL_REVISION, token=token,
        dtype=dtype, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.resume_from:
        log(f"resuming adapter from {args.resume_from}")
        peft_model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        peft_model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            # embed_tokens is non-negotiable (D-039): without it the inert non-verbal
            # tokens can never be learned, and nothing in the loss would say so.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", "embed_tokens"],
        ))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log(f"LoRA rank {args.lora_rank}: {trainable:,} trainable parameters")

    def embed_of(token_id: int):
        with torch.no_grad():
            return peft_model.get_input_embeddings()(
                torch.tensor([token_id], device="cuda")).detach().float().cpu().clone()

    control_id = tokenizer.convert_tokens_to_ids("<|nv_burp|>")
    breath_before, control_before = embed_of(breath_id), embed_of(control_id)

    run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "geetha"),
                         name=f"stage{args.stage}-{int(started)}",
                         config={**vars(args), "context": CONTEXT, "dtype": str(dtype)})

    optimiser = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=args.learning_rate)
    budget = TrainingBudget(max_seconds=args.max_seconds, max_steps=args.max_steps)
    api = HfApi(token=token)

    def save(tag: str) -> str:
        local = out / f"adapter_{tag}"
        peft_model.save_pretrained(str(local))
        try:
            api.create_repo(args.checkpoint_repo, repo_type="model",
                            private=True, exist_ok=True)
            api.upload_folder(folder_path=str(local), repo_id=args.checkpoint_repo,
                              repo_type="model", path_in_repo=f"stage{args.stage}/{tag}",
                              commit_message=f"stage {args.stage} {tag}")
            log(f"checkpoint {tag} pushed to {args.checkpoint_repo}")
            # Delete the local copy once the Hub has it. Without this the pod keeps
            # every checkpoint it has ever written and the container disk fills: on
            # m1-pilot-10 eleven adapters accumulated beside a 4.3 GB image, the base
            # model and the HF cache, and step 2750 died with
            #     SafetensorError: ... No space left on device (os error 28)
            # after 26 minutes of real training. The Hub copy is the one that matters --
            # it is what `--resume-from` reads and what survives the pod -- so keeping
            # the local one costs disk and buys nothing.
            #
            # "final" is the exception: jobs/m1_pilot.sh renames $OUT/adapter_final to
            # continue Stage B from it, so that one stays on disk.
            if tag != "final":
                shutil.rmtree(local, ignore_errors=True)
        except Exception as problem:
            # A failed upload must not end the run, and the local copy is then the ONLY
            # copy -- so it is deliberately not deleted on this path.
            log(f"!! checkpoint upload failed ({type(problem).__name__}: {problem}); "
                f"kept locally at {local}")
        return str(local)

    # ---- train ----------------------------------------------------------------
    peft_model.train()
    losses: list[float] = []
    stop_reason = "max_steps"
    step = 0
    position = 0

    while True:
        elapsed = time.time() - started
        window = losses[-1] if losses else float("nan")
        reason = budget.should_stop(step=step, elapsed=elapsed,
                                   loss=window if losses else 1.0)
        if reason:
            stop_reason = reason
            break

        batch_items = [examples[(position + i) % len(examples)] for i in range(args.batch_size)]
        position += args.batch_size
        batch = collate(batch_items, pad_id=pad_id)
        inputs = {k: torch.as_tensor(v, device="cuda") for k, v in
                  (("input_ids", batch.input_ids), ("labels", batch.labels),
                   ("attention_mask", batch.attention_mask))}

        output = peft_model(**inputs)
        (output.loss / args.grad_accum).backward()
        loss = float(output.loss.detach())
        losses.append(loss)

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

        if run is not None:
            run.log({"loss": loss, "step": step, "elapsed_s": elapsed,
                     "padding_tokens": batch.padding_tokens})
        if step % 25 == 0:
            log(f"step {step:5d}  loss {loss:.4f}  {elapsed / 60:.1f} min  "
                f"pad {batch.padding_tokens}")
        if args.checkpoint_every and step and step % args.checkpoint_every == 0:
            save(f"step{step}")
        step += 1

    log(f"stopped after {step} steps: {stop_reason}")
    final = save("final")

    breath_delta = float((embed_of(breath_id) - breath_before).norm())
    control_delta = float((embed_of(control_id) - control_before).norm())
    first = sum(losses[:25]) / max(1, len(losses[:25]))
    last = sum(losses[-25:]) / max(1, len(losses[-25:]))

    result = {
        "stage": args.stage,
        "stop_reason": stop_reason,
        "steps": step,
        "elapsed_s": time.time() - started,
        "loss_first_25": first,
        "loss_last_25": last,
        "loss_fell": last < first,
        "baseline_loss": budget.baseline_loss,
        "examples": len(examples),
        "examples_with_breath": with_breath,
        "skipped_too_long": stats.skipped_too_long,
        "skipped_malformed": stats.skipped_malformed,
        "trainable_params": trainable,
        "dtype": str(dtype),
        "context": CONTEXT,
        "embedding_delta": {"nv_breath": breath_delta, "nv_burp_control": control_delta},
        "final_adapter": final,
    }
    (out / f"stage{args.stage}_result.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    log(f"loss {first:.4f} -> {last:.4f}; breath embedding moved {breath_delta:.6f} "
        f"vs control {control_delta:.6f}")
    if run is not None:
        run.finish()
    # A NaN or a divergence is a failed run, and the exit code has to say so: the
    # launcher reads it to decide whether to keep the pod's output.
    return 0 if stop_reason in {"max_steps", "budget_exhausted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
