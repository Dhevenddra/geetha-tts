"""Stage the public Hugging Face model repo in a local folder. Uploads nothing.

    python scripts\\make_release_hf.py [--out E:\\Geetha\\results\\release_hf] [--no-eval]

Folder layout (this is exactly what the public repo will hold):

    README.md                          release\\README.md with the Hub front matter on top
    adapter_config.json, adapter_model.safetensors     the final Stage B adapter
    inference.py, tamil_text.py, breaths.json           the recipe; not optional
    Bodhan_AI_Open_Model_License.md    the base model's licence text, unchanged
    eval\\                              from scripts\\make_eval_release.py
    samples\\                           chosen showcase clips, when they exist

The adapter and the licence come from the Hub (private checkpoint repo, gated base), so
HF_HOME must point at the project's token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
EVAL = Path(r"E:\Geetha\results\release_eval")
SAMPLES = Path(r"E:\Geetha\results\release_samples")
CKPT_REPO, ADAPTER = "Dhevenddra/geetha-ckpt-private", "m2-main/stageB/final"
BASE, BASE_REVISION = "bodhan-ai/indic-speak", "76e0814f189321efe550011850cd288dae366aad"
LICENCE = "Bodhan_AI_Open_Model_License.md"
FRONT_MATTER = """---
language:
- ta
license: other
license_name: indic-open-model-license-v1.0
license_link: https://huggingface.co/Dhevenddra/geetha-tts/blob/main/Bodhan_AI_Open_Model_License.md
base_model: bodhan-ai/indic-speak
library_name: peft
pipeline_tag: text-to-speech
tags:
- tamil
- tts
- lora
- snac
---

"""


HERO = ROOT / "site" / "assets" / "img" / "og.jpg"      # the website's link card, so the two match
RESOLVE = "https://huggingface.co/Dhevenddra/geetha-tts/resolve/main"
ANCHOR = "## What it is, and what it is not"
ORDER = ("neutral", "happy", "sad", "anger", "fear", "surprise", "disgust", "breath", "numbers", "long")
LABEL = {"neutral": "Neutral", "happy": "Happy", "sad": "Sad", "anger": "Angry", "fear": "Afraid",
         "surprise": "Surprised", "disgust": "Disgusted", "breath": "With a breath",
         "numbers": "Numbers and dates", "long": "A longer passage"}


def listen_section() -> str:
    """The samples table. They are chosen takes, and the card says so."""
    rows = json.loads((SAMPLES / "samples.json").read_text(encoding="utf-8"))
    kind = lambda row: row["file"].split("_")[0].rstrip("0123456789")       # noqa: E731
    rows.sort(key=lambda row: (ORDER.index(kind(row)), row["voice"], row["file"]))
    lines = ["## Listen", "",
             "These are selected samples: for each sentence we rendered one to three takes "
             "and kept the one we liked, and left out sentences where we liked none. The "
             "exact call for each is in `samples/samples.json`. For unselected output, every "
             "benchmark clip is in the companion dataset.", "",
             "| | Voice | Text | Audio |", "| --- | --- | --- | --- |"]
    for row in rows:
        text = row["text"].replace("|", "\\|")
        lines.append(f"| {LABEL[kind(row)]} | {row['voice']} | {text} | "
                     f'<audio controls preload="none" src="{RESOLVE}/samples/{row["file"]}"></audio> |')
    return "\n".join(lines) + "\n\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"E:\Geetha\results\release_hf"))
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    card = (RELEASE / "README.md").read_text(encoding="utf-8")
    draft = card.startswith("<!-- DRAFT")
    card = re.sub(r"\A<!--.*?-->\s*", "", card, flags=re.S)
    if HERO.exists():
        (out / "assets").mkdir()
        shutil.copy2(HERO, out / "assets" / "hero.jpg")
        card = card.replace("\n\nGeetha is a LoRA adapter", f'\n\n<img src="{RESOLVE}/assets/hero.jpg" '
                            'alt="Geetha, an open voice for Tamil" width="720">\n\nGeetha is a LoRA adapter', 1)
    if SAMPLES.exists():
        card = card.replace(ANCHOR, listen_section() + ANCHOR, 1)
    (out / "README.md").write_text(FRONT_MATTER + card, encoding="utf-8", newline="\n")
    for name in ("inference.py", "tamil_text.py", "breaths.json"):
        shutil.copy2(RELEASE / name, out / name)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(hf_hub_download(CKPT_REPO, f"{ADAPTER}/{name}"), out / name)
    shutil.copy2(hf_hub_download(BASE, LICENCE, revision=BASE_REVISION), out / LICENCE)
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    print(f"adapter: r={config.get('r')} base={config.get('base_model_name_or_path')}")
    if not args.no_eval:
        shutil.copytree(EVAL, out / "eval")
    if SAMPLES.exists():
        shutil.copytree(SAMPLES, out / "samples")
    else:
        print("no samples\\ yet (showcase clips not chosen)")
    weights = out / "adapter_model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"adapter sha256 {digest.hexdigest()}")
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB -> {out}")
    if draft:
        print("NOTE: the model card is still marked DRAFT (claim not yet decided).")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
