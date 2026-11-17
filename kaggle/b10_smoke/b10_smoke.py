r"""B10 - the Kaggle smoke test: does our data actually train this model?

What this run is for
--------------------
Everything upstream of here is verified *statically*: the leak table is zero, the token
layout round-trips exactly, the tokenizer's ids match the contract to the token. None of
that proves a gradient step works. B10 is the first time Geetha's own data is put through
Indic-Speak, and its job is to answer four questions before any money is spent at M1:

1. does the loader produce sequences the model accepts, and does **loss fall**?
2. do the **non-verbal tokens train**? The 16 ``<|nv_*|>`` tokens already exist in the
   vocabulary -- D-021 measured them **inert** in the base model, and the runbook's
   "add new tag tokens" step is superseded (no resize is needed, and resizing would
   break the tokenizer fingerprint). So the question is not whether they were added but
   whether their embeddings *move* when we train on 27,054 breath marks.
3. does a **checkpoint resume** rather than restart? Everything long-running on this
   project must survive a dropped pod, so this is load-bearing, not hygiene.
4. is a generated sample **audible** -- i.e. does the model emit well-formed audio
   tokens that SNAC can decode?

It is free: Kaggle GPU, no gate, no pod. A failure here costs nothing and saves the
Rs 300 pilot from being spent on a loader that does not work.

Why 4-bit
---------
The model is ~4.14 B parameters (3,072 hidden x 28 layers, plus a 156,960-token
embedding), about 8.3 GB in fp16. A Kaggle T4 has 16 GB and is Turing, so no bf16. 4-bit
QLoRA leaves comfortable headroom for the logits, which are the real cost here: a
156,960-wide vocabulary over a 1,024-token sequence is a large tensor. The M1 pilot runs
on a 4090 at higher precision -- this run is about the *pipeline*, not the final numbers,
and that distinction is stated in the report it writes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = Path("/kaggle/working")
STEPS = 50
RESUME_AT = 25
MAX_LENGTH = 1_024
LORA_RANK = 16
LEARNING_RATE = 1e-4
BASE_MODEL = "bodhan-ai/indic-speak"
CODES_REPO = "Dhevenddra/geetha-tts-codes-stagea-private"
SHARDS_TO_PULL = 6


def log(message: str) -> None:
    print(message, flush=True)


def find_input(name: str) -> Path:
    for base in sorted(Path("/kaggle/input").glob("*")):
        hit = base / name
        if hit.exists():
            return hit
        for deeper in base.rglob(name):
            return deeper
    raise SystemExit(f"could not find {name} under /kaggle/input")


def setup_token() -> str:
    """Read the read-only HF token from the private Kaggle secrets dataset.

    Kaggle secrets attached through the UI do not survive an API push (measured
    27 Oct), which is why the token travels as a dataset instead.
    """
    path = find_input("hf_read_token.txt")
    token = path.read_text(encoding="utf-8").strip()
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def import_geetha_package() -> Path:
    """Make ``import geetha`` work from the geetha-code dataset.

    ``push_code_dataset.py`` stages the *contents* of the package at the dataset root,
    so the mount looks like ``/kaggle/input/geetha-code/text/contract.py`` -- there is
    no importable ``geetha/`` directory, and ``geetha-code`` cannot be one (a hyphen is
    not legal in a module name). Version 1 of this kernel died on exactly that, in
    3.7 seconds. So the contents are copied into ``/kaggle/working/geetha`` and that
    parent goes on the path.
    """
    import shutil

    # Search the whole tree, not just the direct children of /kaggle/input: Kaggle does
    # not always mount a dataset at /kaggle/input/<slug>. Version 2 of this kernel found
    # only ['datasets'] there and died. Anchor on a file only this package has.
    for marker in sorted(Path("/kaggle/input").rglob("contract.py")):
        if marker.parent.name == "text":
            base = marker.parent.parent
            target = OUT / "geetha"
            if not target.exists():
                shutil.copytree(base, target)
                # A copied dataset root carries dataset-metadata.json; harmless, but
                # the package needs an __init__ if the staging dropped it.
                init = target / "__init__.py"
                if not init.exists():
                    init.write_text("", encoding="utf-8")
            return target
    listing = sorted(str(entry) for entry in Path("/kaggle/input").rglob("*"))[:60]
    for entry in listing:
        log(f"  input: {entry}")
    raise SystemExit("could not find the geetha package: no */text/contract.py anywhere "
                     "under /kaggle/input (tree logged above)")


def install_dependencies() -> None:
    """Kaggle's image has torch and transformers but not these two.

    Version 3 got all the way through the tokenizer check and the loader before dying
    on `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes`, so
    this is the last gap between a cold image and a gradient step.
    """
    import subprocess

    # torchao is in the image at 0.10.0, and peft's LoRA dispatcher calls
    # is_torchao_available() which *raises* on anything below 0.16.0 rather than
    # returning False -- so an unrelated stale package blocks adapter injection
    # entirely. v4 died there after getting all the way to get_peft_model.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "bitsandbytes>=0.46.1", "peft", "accelerate",
                    "torchao>=0.16.0", "snac"], check=True)
    log("installed bitsandbytes, peft, accelerate, torchao, snac")


def main() -> int:
    install_dependencies()
    package = import_geetha_package()
    sys.path.insert(0, str(package.parent))
    log(f"geetha package ready at {package}")
    token = setup_token()

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from geetha.data.loader import examples_from_shard, LoaderStats
    from geetha.text.tokenizer_check import assert_tokenizer

    results: dict[str, object] = {}

    # ---- 1. the tokenizer, before anything else --------------------------------
    # token_contract.md §6: assert at data-compile time and again at train start.
    # "Mismatch is a hard error, never a warning."
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    # get_added_vocab() is {token string: id}, which is what the check expects.
    added = dict(tokenizer.get_added_vocab())
    assert_tokenizer(added, len(tokenizer))
    log(f"tokenizer OK: {len(added):,} added tokens, vocab {len(tokenizer):,}")
    results["tokenizer_added_tokens"] = len(added)

    # ---- 2. the data -----------------------------------------------------------
    shard_names = [name for name in list_repo_files(CODES_REPO, repo_type="dataset", token=token)
                   if name.endswith(".parquet")][:SHARDS_TO_PULL]
    shards = [Path(hf_hub_download(CODES_REPO, name, repo_type="dataset", token=token))
              for name in shard_names]
    log(f"pulled {len(shards)} code shards from {CODES_REPO}")

    stats = LoaderStats()
    examples = []
    for shard in shards:
        for example in examples_from_shard(shard, tokenizer.encode,
                                           max_length=MAX_LENGTH, stats=stats):
            examples.append(example)
            if len(examples) >= STEPS * 4:
                break
        if len(examples) >= STEPS * 4:
            break
    log(f"loader: {stats.rows:,} rows -> {len(examples):,} usable examples "
        f"(skipped {stats.skipped_too_long:,} too long, {stats.skipped_malformed:,} malformed)")
    results["loader"] = dict(rows=stats.rows, examples=len(examples),
                             skipped_too_long=stats.skipped_too_long,
                             skipped_malformed=stats.skipped_malformed)
    if len(examples) < STEPS:
        raise SystemExit(f"only {len(examples)} examples for {STEPS} steps")

    # How many of these actually carry a breath token? If none do, question 2 is
    # unanswerable and the run must say so rather than report a meaningless number.
    breath_id = tokenizer.convert_tokens_to_ids("<|nv_breath|>")
    with_breath = sum(1 for ex in examples if breath_id in ex.input_ids)
    log(f"examples containing <|nv_breath|>: {with_breath:,} of {len(examples):,}")
    results["examples_with_breath"] = with_breath

    # ---- 3. the model ----------------------------------------------------------
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, token=token, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        # NOTE (v4 log): the model has `tie_word_embeddings=True`, so an adapter on
        # embed_tokens is shared with lm_head. For Geetha that is harmless and arguably
        # right -- the same learned representation is used to read a tag and to emit
        # audio -- but it must be remembered when merging the adapter or exporting to
        # anything other than safetensors, which is what peft warns about.
        #
        # embed_tokens is in here on purpose, and it is the whole point. D-021
        # measured the 16 non-verbal tokens INERT in the base model, so they have to be
        # *learned*. A projection-only LoRA cannot teach an inert token anything -- it
        # never touches the embedding matrix, so <|nv_breath|> would enter every forward
        # pass as the same meaningless vector it started as. Gradient does reach it even
        # though the prompt is masked from the loss, because the prompt's embeddings
        # determine the audio predictions that are supervised.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "embed_tokens"],
    ))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log(f"LoRA trainable parameters: {trainable:,}")
    results["trainable_params"] = trainable

    # Snapshot the breath embedding so question 2 can be answered by measurement.
    # Measured through a forward pass, not off .weight: with LoRA on embed_tokens the
    # base matrix stays frozen and reading it would report zero movement forever.
    control_id = tokenizer.convert_tokens_to_ids("<|nv_burp|>")

    def embed_of(token_id: int) -> "torch.Tensor":
        module = peft_model.get_input_embeddings()
        with torch.no_grad():
            return module(torch.tensor([token_id], device="cuda")).detach().float().cpu().clone()

    breath_before = embed_of(breath_id)
    control_before = embed_of(control_id)

    # ---- 4. train, checkpoint, resume ------------------------------------------
    optimiser = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    losses: list[float] = []

    def step(index: int) -> float:
        example = examples[index % len(examples)]
        ids = torch.tensor([example.input_ids], device="cuda")
        labels = torch.tensor([example.labels], device="cuda")
        out = peft_model(input_ids=ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in peft_model.parameters() if p.requires_grad], 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    peft_model.train()
    for index in range(RESUME_AT):
        losses.append(step(index))
        if index % 5 == 0:
            log(f"  step {index:3d} loss {losses[-1]:.4f}")

    checkpoint = OUT / "adapter_step25"
    peft_model.save_pretrained(str(checkpoint))
    log(f"saved checkpoint at step {RESUME_AT} -> {checkpoint}")

    # Reload the adapter into a fresh model and carry on. If resume were broken the
    # loss would jump back to roughly its starting value, which is exactly the check.
    from peft import PeftModel

    del peft_model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, token=token, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    peft_model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True)
    optimiser = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    resumed_first = step(RESUME_AT)
    losses.append(resumed_first)
    log(f"first loss after resume: {resumed_first:.4f} "
        f"(pre-save was {losses[RESUME_AT - 1]:.4f}, start was {losses[0]:.4f})")
    peft_model.train()
    for index in range(RESUME_AT + 1, STEPS):
        losses.append(step(index))
        if index % 5 == 0:
            log(f"  step {index:3d} loss {losses[-1]:.4f}")

    first_five = sum(losses[:5]) / 5
    last_five = sum(losses[-5:]) / 5
    results["loss"] = dict(first=losses[0], first_five=first_five, last_five=last_five,
                           resumed_first=resumed_first, all=losses)

    # ---- 5. did the non-verbal embeddings move? --------------------------------
    breath_delta = float((embed_of(breath_id) - breath_before).norm())
    control_delta = float((embed_of(control_id) - control_before).norm())
    results["embedding_delta"] = dict(nv_breath=breath_delta, nv_burp_control=control_delta)
    log(f"<|nv_breath|> embedding moved {breath_delta:.6f}; "
        f"<|nv_burp|> (absent from the data) moved {control_delta:.6f}")

    # ---- 6. is a sample audible? -----------------------------------------------
    sample_tokens = 0
    audio_path = ""
    try:
        from geetha.audio.snac_tokens import unflatten_frames
        from geetha.text.contract import AUDIO_TOKEN_BASE_ID, CODEBOOKS, CODEBOOK_SIZE, CONTROL_TOKENS

        probe = examples[0]
        cut = probe.labels.index(CONTROL_TOKENS["<|start_of_speech|>"])
        prompt_ids = torch.tensor([probe.input_ids[:cut + 1]], device="cuda")
        peft_model.eval()
        with torch.inference_mode():
            generated = peft_model.generate(
                input_ids=prompt_ids, max_new_tokens=7 * 40, do_sample=True,
                temperature=0.8, top_p=0.95)
        band_top = AUDIO_TOKEN_BASE_ID + CODEBOOKS * CODEBOOK_SIZE - 1
        emitted = [int(t) for t in generated[0][prompt_ids.shape[1]:].tolist()
                   if AUDIO_TOKEN_BASE_ID <= int(t) <= band_top]
        sample_tokens = len(emitted)
        whole = (len(emitted) // 7) * 7
        log(f"generated {len(emitted)} in-band audio tokens ({whole // 7} whole frames)")
        if whole >= 7:
            import soundfile as sf
            from snac import SNAC

            codec = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
            c0, c1, c2 = unflatten_frames(emitted[:whole])
            with torch.inference_mode():
                codes = [torch.tensor(b, dtype=torch.long, device="cuda").reshape(1, -1)
                         for b in (c0, c1, c2)]
                wave = codec.decode(codes).reshape(-1).cpu().numpy()
            audio_path = str(OUT / "b10_sample.wav")
            sf.write(audio_path, wave, 24_000)
            log(f"wrote {audio_path} ({len(wave) / 24_000:.2f} s)")
    except Exception as problem:  # a failed sample must not lose the training result
        log(f"sample generation failed: {type(problem).__name__}: {problem}")
    results["sample"] = dict(in_band_tokens=sample_tokens, wav=audio_path)

    # ---- 7. the verdict --------------------------------------------------------
    loss_fell = last_five < first_five
    resume_ok = resumed_first < first_five
    breath_moved = breath_delta > 0.0
    breath_moved_more = breath_delta > control_delta
    verdict = "PASS" if (loss_fell and resume_ok and breath_moved) else "FAIL"
    results["verdict"] = dict(verdict=verdict, loss_fell=loss_fell, resume_ok=resume_ok,
                              breath_moved=breath_moved, breath_moved_more=breath_moved_more)

    (OUT / "smoke.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (OUT / "smoke.md").write_text(f"""# B10 smoke test - {verdict}

{STEPS} LoRA steps on `{BASE_MODEL}` with Geetha's own Stage A codes, 4-bit, rank
{LORA_RANK}, lr {LEARNING_RATE}, max length {MAX_LENGTH}.

| question | answer |
| --- | --- |
| loss falls | **{loss_fell}** - first five {first_five:.4f} -> last five {last_five:.4f} |
| checkpoint resumes | **{resume_ok}** - first loss after reload {resumed_first:.4f}, against {first_five:.4f} at the start |
| `<\\|nv_breath\\|>` embedding moves | **{breath_moved}** - moved {breath_delta:.6f} |
| ...more than an unused control | **{breath_moved_more}** - `<\\|nv_burp\\|>` moved {control_delta:.6f} |
| sample decodes | {sample_tokens} in-band audio tokens, wav: `{audio_path or "none"}` |

- examples: {len(examples):,} from {len(shards)} shards; {with_breath:,} contain a breath token
- loader skipped {stats.skipped_too_long:,} as too long, {stats.skipped_malformed:,} as malformed
- LoRA trainable parameters: {trainable:,}
- tokenizer asserted against the contract before the first step: {len(added):,} added tokens

**What this does and does not establish.** It establishes that the pipeline trains: the
loader's sequences are accepted, the loss descends, a checkpoint resumes rather than
restarts, and the non-verbal tokens D-021 measured as inert do move once they appear in
the data. It establishes **nothing about quality** - 50 steps at 4-bit on a T4 is not the
M1 pilot, which runs at higher precision on a 4090. Any number here that looks like a
result is a smoke reading, not a measurement.
""", encoding="utf-8")
    log(f"\n{verdict}")
    log(f"loss {first_five:.4f} -> {last_five:.4f}; resume {resumed_first:.4f}; "
        f"breath delta {breath_delta:.6f} vs control {control_delta:.6f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
