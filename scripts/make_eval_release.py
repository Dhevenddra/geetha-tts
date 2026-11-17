"""Assemble the eval/ folder that ships beside the model card: every raw output behind
the published numbers, and the code that produced them.

    python scripts\\make_eval_release.py [--out E:\\Geetha\\results\\release_eval]

Nothing is uploaded. The folder is rebuilt from scratch each time and checked: every
clip named in a transcript file must be present, and the key of a listening test is
only copied once its answers exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(r"E:\Geetha\runs")
BENCH = Path(r"E:\Geetha\work\bench_ta_v2\bench_ta_v2.json")
SCORES = Path(r"E:\Geetha\results\bench_v2")
SYSTEMS = {"geetha": "m3_bench_ours", "base": "m3_bench_ours",
           "rumik_oss_1": "m3_bench_rivals", "snortts_indic_v0": "m3_bench_rivals"}
DATASET_CARD = """---
language:
- ta
license: other
license_name: mixed-see-card
pretty_name: Geetha TTS - benchmark outputs and listening tests
tags:
- tamil
- tts
- evaluation
---

# Geetha TTS: benchmark outputs and listening tests

The raw material behind every number on the
[Geetha model card](https://huggingface.co/Dhevenddra/geetha-tts): what four Tamil
text-to-speech systems said for the same 304 sentences, what two speech recognisers
heard, and what one listener chose in four blind tests. Nothing here was selected:
the clips that went wrong are included.

> This is evidence, not a training set. Please do not train on it: the sentences are
> test sentences, and the audio is machine-made.

## What is here

| Path | What it is |
| --- | --- |
| `bench_ta_v2.json` | the 304 sentences, their source, and the voice each was given |
| `transcripts/bench_<system>.json` | per clip: usable or not and why, length, both judges' transcripts; plus that run's decoding settings, seeds and model revisions |
| `clips/<system>__<id>.wav` | every clip for Geetha, the base model and snorTTS-Indic-v0 |
| `bench_v2.md`, `bench_v2.json` | the scores, down to per-clip error rates |
| `code/` | the kernels that rendered and transcribed the clips, and the scorer |
| `listening/` | four blind listening tests: the page, the pairs, the key and the answers |
| `SHA256SUMS.json` | a checksum for every file |

Rumik-OSS-1's audio is not included because that model's licence is non-commercial.
Its transcripts and scores are, and `code/m3_bench_rivals.py` regenerates the audio.

## How to read it

- One take per sentence, seed 1234 + 100 x sentence index, each system at the settings
  its own model card recommends.
- A clip is "usable" unless it is silent or ran to the 22-second cap without ending.
- Character error rate compares a judge's transcript with the written sentence. It says
  whether the words can be understood. It says nothing about how human the voice sounds.
- The listening tests have one listener, the person who made the model. Two of the four
  were blind to the system as well as the order; the card says which.

## Licences

- Sentences from FLEURS are CC BY 4.0 (Google). The other sentences were written for
  this project and are released under CC BY 4.0.
- Audio made by Geetha and by the base model is shared here for evaluation and research;
  see the Indic Open Model License v1.0 for any terms it places on outputs.
  snorTTS-Indic-v0 is an Apache-2.0 model.
- Code: Apache-2.0.
"""

NO_AUDIO = {"rumik_oss_1"}      # CC BY-NC 4.0: transcripts and scores only, no audio
CODE = [ROOT / "scripts" / "score_bench_v2.py", ROOT / "scripts" / "make_bench_v2.py",
        ROOT / "scripts" / "make_blind_ab.py", ROOT / "scripts" / "make_blind_rivals.py",
        ROOT / "kaggle" / "m3_bench" / "worker.py", ROOT / "kaggle" / "m3_bench" / "template.py",
        ROOT / "kaggle" / "m3_bench_ours" / "m3_bench_ours.py",
        ROOT / "kaggle" / "m3_bench_rivals" / "m3_bench_rivals.py"]
# (name in eval/listening, folder with listen.html and the pair clips, key, answers)
LISTENS = [("geetha_vs_base", RUNS / "m3_base_vs_geetha" / "blind",
            RUNS / "m3_base_vs_geetha" / "blind_key.json",
            RUNS / "m3_base_vs_geetha" / "blind_answers.json"),
           ("geetha_vs_rivals", RUNS / "m3_bench_rivals" / "blind",
            RUNS / "m3_bench_rivals" / "blind_rivals_key.json",
            RUNS / "m3_bench_rivals" / "blind_rivals_answers.json"),
           ("mid_sentence_breath", RUNS / "m3_mid_split" / "blind",
            RUNS / "m3_mid_split" / "blind_breath_key.json",
            RUNS / "m3_mid_split" / "blind_breath_answers.json"),
           ("emotion_styled_vs_plain", RUNS / "m3_emotion_listen" / "blind",
            RUNS / "m3_emotion_listen" / "blind_emotion_key.json",
            RUNS / "m3_emotion_listen" / "blind_emotion_answers.json")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"E:\Geetha\results\release_eval"))
    out = parser.parse_args().out
    if out.exists():
        shutil.rmtree(out)
    for sub in ("transcripts", "clips", "code", "listening"):
        (out / sub).mkdir(parents=True)
    shutil.copy2(BENCH, out / BENCH.name)
    for name in ("bench_v2.md", "bench_v2.json"):
        shutil.copy2(SCORES / name, out / name)
    problems, copied = [], 0
    for system, run in SYSTEMS.items():
        source = RUNS / run / f"bench_{system}.json"
        report = json.loads(source.read_text(encoding="utf-8"))
        if not report.get("finished") or report.get("failure"):
            problems.append(f"{system}: finished={report.get('finished')} failure={report.get('failure')}")
        shutil.copy2(source, out / "transcripts" / source.name)
        for clip_id, clip in report["clips"].items():
            if not clip.get("file") or system in NO_AUDIO:
                continue
            wav = RUNS / run / "clips" / clip["file"]
            if not wav.exists():
                problems.append(f"{system}: clip missing for {clip_id}")
                continue
            shutil.copy2(wav, out / "clips" / clip["file"])
            copied += 1
    for path in CODE:
        shutil.copy2(path, out / "code" / path.name)
    for name, folder, key, answers in LISTENS:
        if not answers.exists():
            problems.append(f"listening/{name}: no answers yet, key not released")
            continue
        shutil.copytree(folder, out / "listening" / name)
        for pid, row in json.loads(key.read_text(encoding="utf-8"))["pairs"].items():
            for side in ("A", "B"):                  # drop audio we do not redistribute
                if row.get(side) in NO_AUDIO:
                    (out / "listening" / name / f"{pid}_{side}.wav").unlink()
        shutil.copy2(key, out / "listening" / name / key.name)
        shutil.copy2(answers, out / "listening" / name / answers.name)
    (out / "README.md").write_text(DATASET_CARD, encoding="utf-8", newline="\n")
    sums = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"{copied} clips, {len(sums)} files, {size:.0f} MB -> {out}")
    for line in problems:
        print("PROBLEM:", line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
