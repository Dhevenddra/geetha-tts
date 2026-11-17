r"""D-049 - does merging the adapter change what the model emits? (free, Kaggle GPU)

The decision this settles
-------------------------
D-039 requires ``embed_tokens`` in every LoRA target list: the 16 non-verbal tokens are
inert in the base model, so a projection-only LoRA leaves them structurally unlearnable.
The base model also sets ``tie_word_embeddings=True``. PEFT warned about exactly this
combination during m1-pilot-11's Stage B resume:

    Model has tie_word_embeddings=True and a tied layer is part of the adapter, but
    ensure_weight_tying is not set to True. This can lead to complications, for example
    when merging the adapter.

The adapter was saved with ``ensure_weight_tying: false``.

The mechanism at stake, and why it is bigger than the non-verbal tokens
----------------------------------------------------------------------
With tied weights, ``lm_head.weight`` *is* ``model.embed_tokens.weight`` -- one tensor
read two ways. PEFT wraps the ``Embedding`` module, so the **input** lookup returns
base + delta. ``lm_head`` is a separate ``Linear`` that is not in ``target_modules``, so
nothing wraps it: while unmerged, the **output head reads the base tensor, without the
delta**. ``merge_and_unload`` writes the delta into that shared tensor in place, and the
output head then sees it too.

If that reading is right, then **training optimised a function the merged model does not
compute**. Every logit during training came from the un-delta'd head; merging changes the
head afterwards. That is not a non-verbal-token problem, it is a problem for the 28,672
SNAC audio tokens the model actually generates -- i.e. for the speech itself.

So this does not test a warning, it tests an identity:

    unmerged(x) == merged(x) ?

Measurements
------------
1. **Tying, at runtime.** ``lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()``
   on the loaded base model. Asserted, not assumed from the config.
2. **Does the head see the delta before merge?** Logits for a fixed input, three arms:
   base, unmerged adapter, merged adapter.
3. **Controls.** ``<|nv_burp|>``, ``<|nv_sneeze|>``, ``<|pad|>`` and token 1000 had an
   exactly-zero A column after training (results\d039_embedding_check.md). Their logits
   must be **identical** across merged and unmerged. If a control moves, the arms differ
   for some reason other than the merge and nothing else here can be read.
4. **Behaviour.** Greedy generation (``do_sample=False``, so no seed can explain a
   difference), same prompt, unmerged vs merged. First divergence index and the fraction
   of identical tokens. This is D-049's own acceptance test.

What a null result would and would not mean
-------------------------------------------
m1-pilot-11 trained 20 minutes per stage to a final loss of 3.838. If generation is
poor, that says nothing about the recipe. But arms 1-3 are **structural**: they hold or
fail regardless of how well the model was trained, which is why they are the primary
measurement and generation is the confirmation.

Spends no money: free Kaggle GPU, no gate, no pod.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

OUT = Path("/kaggle/working")
BASE_MODEL = "bodhan-ai/indic-speak"
# Pinned to the revision the project trained against; `main` moving would silently
# change what "the base model" means between this run and the pilot.
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTER_SUBDIR = "final/m1-pilot-11/adapter_final"

# From results\d039_embedding_check.md. The zero rows are the controls, and they are
# what make a difference in the moved rows meaningful.
MEASURED_TOKENS = {
    "<|nv_breath|>": (156_944, "moved in Stage A (0.5758)"),
    "<|nv_laugh|>": (156_947, "moved in Stage A (0.6472)"),
    "<|nv_burp|>": (156_959, "control: exactly zero"),
    "<|nv_sneeze|>": (156_955, "control: exactly zero"),
    "<|pad|>": (128_263, "control: exactly zero"),
    "token 1000": (1_000, "control: exactly zero"),
}
CONTROL_TOKENS = {"<|nv_burp|>", "<|nv_sneeze|>", "<|pad|>", "token 1000"}

NEW_TOKENS = 192

# A real Stage A training row (ivr speaker), rendered in a voice we are allowed to
# expose. The breath tag is the direction whose effect is in question.
PROMPT_TEXT = "இல்ல <|nv_breath|> ஆட்டாங்கல்லோ அந்த மாரி இருந்தா"


def log(message: str) -> None:
    print(message, flush=True)


def fix_environment() -> None:
    """Remove Kaggle's torchao, which makes PEFT unable to build any LoRA.

    Version 1 of this kernel died inside the toy check with::

        ImportError: Found an incompatible version of torchao. Found version 0.10.0,
        but only versions above 0.16.0 are supported

    PEFT's module dispatcher calls ``is_torchao_available()`` for every target module,
    and that helper **raises** on a present-but-too-old torchao instead of returning
    False. So an unrelated package pinned in the image blocks ``get_peft_model``
    entirely. Uninstalling is the right lever rather than upgrading: we never ask for
    quantised layers, and pulling torchao >= 0.16 would drag torch along with it on an
    image whose torch build everything else here depends on.
    """
    import subprocess

    before = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        capture_output=True, text=True,
    )
    log(f"  pip uninstall torchao -> {before.returncode}")

    import torch
    import peft
    import transformers

    log(f"  torch {torch.__version__}, transformers {transformers.__version__}, "
        f"peft {peft.__version__}")
    # The pilot trained under peft 0.21.2 (adapter_config.json). A different version
    # here would not invalidate the identity being tested, but it is the first thing
    # to suspect if the arms disagree with the pod's behaviour.
    log(f"  cuda: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")


def find_input(name: str) -> Path:
    for base in sorted(Path("/kaggle/input").glob("*")):
        hit = base / name
        if hit.exists():
            return hit
        for deeper in base.rglob(name):
            return deeper
    raise SystemExit(f"could not find {name} under /kaggle/input")


def import_geetha_package() -> Path:
    for marker in sorted(Path("/kaggle/input").rglob("contract.py")):
        if marker.parent.name == "text":
            target = OUT / "geetha"
            if not target.exists():
                shutil.copytree(marker.parent.parent, target)
                init = target / "__init__.py"
                if not init.exists():
                    init.write_text("", encoding="utf-8")
            return target
    for entry in sorted(str(e) for e in Path("/kaggle/input").rglob("*"))[:60]:
        log(f"  input: {entry}")
    raise SystemExit("could not find the geetha package under /kaggle/input")


def mechanism_check() -> dict:
    """Settle the tying question on a toy model, before spending minutes on 3B weights.

    A 2-layer Llama with tied embeddings reproduces the exact module structure that
    matters here, in megabytes. If the identity holds on the toy, the real-weights arms
    below are a confirmation rather than a discovery -- and if the toy disagrees with
    the real model, that is itself worth knowing before interpreting anything.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    tied_at_birth = (
        model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    )

    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=8, target_modules=["q_proj", "embed_tokens"],
        task_type="CAUSAL_LM",
    ))

    # Give the embedding LoRA a real delta. PEFT zero-initialises `lora_embedding_A`
    # for Embedding layers (the reverse of Linear), so without this the adapter is a
    # no-op and merged would trivially equal unmerged -- a passing test that proves
    # nothing. Same trap D-039's check documents.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_embedding_A" in name:
                param.normal_(0, 0.5)

    ids = torch.tensor([[1, 2, 3, 4]])
    peft_model.eval()
    with torch.no_grad():
        unmerged = peft_model(ids).logits.clone()

    merged_model = peft_model.merge_and_unload()
    with torch.no_grad():
        merged = merged_model(ids).logits.clone()

    difference = float((merged - unmerged).abs().max())
    return {
        "tied_at_birth": bool(tied_at_birth),
        "max_abs_logit_difference_merged_vs_unmerged": difference,
        "merged_equals_unmerged": difference == 0.0,
    }


def main() -> int:
    # Before importing torch, transformers or peft. `is_torchao_available` resolves the
    # module through `find_spec`, which consults `sys.modules` first -- so a torchao
    # already pulled in by another import would still be found after the uninstall, and
    # the ImportError would fire anyway.
    log("=== 0. environment ===")
    fix_environment()

    import torch

    token = find_input("hf_read_token.txt").read_text(encoding="utf-8").strip()
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report: dict = {}

    log("")
    log("=== 1. mechanism, on a toy tied model ===")
    report["mechanism"] = mechanism_check()
    for key, value in report["mechanism"].items():
        log(f"  {key}: {value}")

    log("")
    log("=== 2. the adapter ===")
    # Fail here, loudly, rather than after loading 6.6 GB of base weights: the read
    # token's access to a *private* repo is the one credential assumption in this run.
    adapter_root = snapshot_download(
        CKPT_REPO, allow_patterns=[f"{ADAPTER_SUBDIR}/*"], token=token
    )
    adapter = Path(adapter_root) / ADAPTER_SUBDIR
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    log(f"  target_modules: {sorted(config['target_modules'])}")
    log(f"  ensure_weight_tying: {config.get('ensure_weight_tying')}")
    if "embed_tokens" not in config["target_modules"]:
        raise SystemExit("adapter has no embed_tokens LoRA; D-049 does not apply to it")
    report["adapter_config"] = {
        "target_modules": sorted(config["target_modules"]),
        "ensure_weight_tying": config.get("ensure_weight_tying"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
    }

    log("")
    log("=== 3. base model ===")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, token=token
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, token=token,
        dtype=torch.bfloat16, device_map="cuda",
    )
    base.eval()
    tied = base.lm_head.weight.data_ptr() == base.model.embed_tokens.weight.data_ptr()
    log(f"  lm_head tied to embed_tokens: {tied}")
    log(f"  input embedding: {tuple(base.get_input_embeddings().weight.shape)}")
    report["real_model_tied"] = bool(tied)

    package = import_geetha_package()
    sys.path.insert(0, str(package.parent))
    from geetha.text.contract import build_tts_prompt

    prompt = build_tts_prompt(
        speaker="anitha", style="CONV", text=PROMPT_TEXT, for_inference=True
    )
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_len = int(ids["input_ids"].shape[-1])
    log(f"  prompt tokens: {prompt_len}")
    report["prompt"] = {
        "text": PROMPT_TEXT, "speaker": "anitha", "style": "CONV",
        "n_tokens": prompt_len,
    }

    def last_logits(model):
        with torch.no_grad():
            return model(**ids).logits[0, -1].float().cpu()

    base_logits = last_logits(base)

    log("")
    log("=== 4. unmerged adapter ===")
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    peft_model.eval()
    unmerged_logits = last_logits(peft_model)
    with torch.no_grad():
        unmerged_generated = peft_model.generate(
            **ids, max_new_tokens=NEW_TOKENS, do_sample=False
        )[0].cpu().tolist()

    log("")
    log("=== 5. merged adapter ===")
    merged = peft_model.merge_and_unload()
    merged.eval()
    merged_logits = last_logits(merged)
    merged_tied = (
        merged.lm_head.weight.data_ptr() == merged.model.embed_tokens.weight.data_ptr()
    )
    log(f"  still tied after merge: {merged_tied}")
    with torch.no_grad():
        merged_generated = merged.generate(
            **ids, max_new_tokens=NEW_TOKENS, do_sample=False
        )[0].cpu().tolist()
    report["tied_after_merge"] = bool(merged_tied)

    log("")
    log("=== 6. logits, per measured token ===")
    log(f"{'token':16s} {'base':>12s} {'unmerged':>12s} {'merged':>12s} {'m-u':>12s}")
    rows = []
    for name, (token_id, note) in MEASURED_TOKENS.items():
        b = float(base_logits[token_id])
        u = float(unmerged_logits[token_id])
        m = float(merged_logits[token_id])
        rows.append({
            "token": name, "token_id": token_id, "note": note,
            "base": b, "unmerged": u, "merged": m, "merged_minus_unmerged": m - u,
            "is_control": name in CONTROL_TOKENS,
        })
        log(f"{name:16s} {b:12.5f} {u:12.5f} {m:12.5f} {m - u:12.5f}")
    report["logits"] = rows

    dirty = [r for r in rows if r["is_control"] and r["merged_minus_unmerged"] != 0.0]
    report["controls_clean"] = not dirty

    log("")
    log("=== 7. generation, unmerged vs merged (greedy) ===")
    a = unmerged_generated[prompt_len:]
    b = merged_generated[prompt_len:]
    pairs = list(zip(a, b))
    identical = sum(1 for x, y in pairs if x == y)
    first_divergence = next((i for i, (x, y) in enumerate(pairs) if x != y), None)
    log(f"  new tokens compared: {len(pairs)}")
    log(f"  identical: {identical}/{len(pairs)}")
    log(f"  first divergence at: {first_divergence}")
    report["generation"] = {
        "compared": len(pairs), "identical": identical,
        "first_divergence": first_divergence,
        "unmerged_first_32": a[:32], "merged_first_32": b[:32],
    }

    log("")
    if not report["controls_clean"]:
        log("CONTROLS DIRTY -- a zero-delta token's logit moved. Read nothing else here.")
    elif first_divergence is None and all(
        r["merged_minus_unmerged"] == 0.0 for r in rows
    ):
        log("VERDICT: merging is a no-op on this adapter. D-049 closes benign.")
    else:
        log("VERDICT: merged != unmerged. Training optimised the unmerged function,")
        log("so a merged release build is NOT the model we trained. Ship unmerged, or")
        log("retrain with ensure_weight_tying=True.")

    (OUT / "d049_merge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {OUT / 'd049_merge.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
