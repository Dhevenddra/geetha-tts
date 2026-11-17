r"""Which ASR judge can be trusted with Tamil? (D-053 item 2; free Kaggle GPU)

Every TTS CER in this project was judged by ``openai/whisper-large-v3``. C5 measured that
judge at 32.9% CER on *real* Tamil audio, and a competitor's model card reports about 2%
Tamil CER for five systems under IndicConformer RNNT. Before any CER of ours is published,
three judges read the same three sets:

    real    40 real Rasa test recordings (Naturalness-Ta)   -> each judge's own floor
    base    41 archived B7 clips of the base model
    geetha  40 valid clips of the M2 Stage B adapter (arm B_plain of m2_main_eval)

    judges  whisper-large-v3 | IndicConformer-600M (RNNT and CTC) | Indic-Transcribe-core

The base and geetha sets share prompts, so each judge also gives its own paired delta.
A judge whose floor on real studio Tamil is tens of percent cannot rank two TTS systems.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
WHISPER_ID = "openai/whisper-large-v3"
CONFORMER_ID = "ai4bharat/indic-conformer-600m-multilingual"
TRANSCRIBE_ID = "bodhan-ai/indic-transcribe-core"
# Both repos ship Python that this kernel imports, so both are pinned to the commits
# that produced the 11 Nov numbers (neither had changed since: 8 Oct and 17 Sep 2026).
CONFORMER_REVISION = "2a77d305038dfc502fd9464e7d028b90381976cb"
TRANSCRIBE_REVISION = "e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537"
JUDGE_RATE = 16_000
_MAX_MOUNT_DEPTH = 4


def log(message: str) -> None:
    print(message, flush=True)


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"could not find a mount containing {markers}")


def main() -> int:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime==1.20.1",
                    "soundfile==0.14.0", "jiwer==4.0.0", "sentencepiece"], check=True)
    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download

    token = (find_dataset("hf_read_token.txt") / "hf_read_token.txt").read_text(
        encoding="utf-8").strip()
    source = find_dataset("scoring", "text", "eval")
    package = Path("/kaggle/tmp/pkg/geetha")
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(package.parent))
    from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap
    from geetha.scoring.metrics import cer

    data = find_dataset("manifest.json")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    log(f"{len(manifest)} clips; gpu: {torch.cuda.get_device_name(0)}")

    # Everything gated is fetched here, with the token passed explicitly. The two judge
    # repos ship Python that runs in this process, so the token never enters the
    # environment and is not passed to their code.
    conformer_dir = snapshot_download(CONFORMER_ID, revision=CONFORMER_REVISION,
                                     token=token)
    try:
        transcribe_dir = snapshot_download(TRANSCRIBE_ID, revision=TRANSCRIBE_REVISION,
                                           token=token)
    except Exception as problem:  # noqa: BLE001 -- an optional third judge
        transcribe_dir = None
        log(f"Indic-Transcribe-core not downloadable: {type(problem).__name__}: {problem}")

    # Whisper is fetched now as well, so the token is not used again once any
    # third-party judge code has been imported. (The secrets mount itself stays
    # readable by everything in this kernel; the token on it is read-only, and the two
    # repos are pinned to reviewed commits. That is the limit of what a kernel can do.)
    whisper_dir = snapshot_download(WHISPER_ID, token=token)
    del token

    def load(entry: dict) -> torch.Tensor:
        samples, rate = sf.read(str(data / entry["file"]), dtype="float32", always_2d=True)
        wav = torch.from_numpy(samples).mean(dim=1)
        if rate != JUDGE_RATE:
            wav = torchaudio.functional.resample(wav, rate, JUDGE_RATE)
        return wav

    waves = {entry["file"]: load(entry) for entry in manifest}
    hyps: dict[str, dict[str, str]] = {}

    # ---------------------------------------------------------------- IndicConformer
    log("")
    log("=== IndicConformer-600M ===")
    sys.path.insert(0, conformer_dir)
    from model_onnx import IndicASRConfig, IndicASRModel  # type: ignore

    settings = json.loads((Path(conformer_dir) / "config.json").read_text())
    settings.pop("auto_map", None)
    conformer = IndicASRModel(IndicASRConfig(ts_folder=conformer_dir, **settings))
    started = time.time()
    for mode in ("rnnt", "ctc"):
        out = {}
        for entry in manifest:
            try:
                out[entry["file"]] = conformer(waves[entry["file"]].unsqueeze(0), "ta", mode)
            except Exception as problem:  # noqa: BLE001
                out[entry["file"]] = ""
                log(f"  {mode} failed on {entry['file']}: {type(problem).__name__}: {problem}")
        hyps[f"indicconformer_{mode}"] = out
        log(f"  {mode}: {len(out)} clips, {time.time() - started:.0f} s elapsed")
    del conformer

    # ---------------------------------------------------------------- Indic-Transcribe
    if transcribe_dir is not None:
        log("")
        log("=== Indic-Transcribe-core (the runbook's named judge) ===")
        try:
            sys.path.insert(0, transcribe_dir)
            from indic_transcribe import IndicTranscribe  # type: ignore

            judge = IndicTranscribe.from_pretrained(transcribe_dir, device="cuda")
            out = {}
            for entry in manifest:
                text = judge.transcribe(str(data / entry["file"]), lang="ta")
                out[entry["file"]] = (text[0] if isinstance(text, tuple) else text).strip()
            hyps["indic_transcribe_core"] = out
            del judge
            torch.cuda.empty_cache()
            log(f"  {len(out)} clips")
        except Exception as problem:  # noqa: BLE001
            log(f"  SKIPPED: {type(problem).__name__}: {problem}")

    # ---------------------------------------------------------------- Whisper
    log("")
    log("=== whisper-large-v3 (the judge of every number so far) ===")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(whisper_dir)
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_dir, dtype=torch.float16).to("cuda").eval()
    out = {}
    for entry in manifest:
        features = processor(waves[entry["file"]].numpy(), sampling_rate=JUDGE_RATE,
                             return_tensors="pt").input_features.to("cuda").half()
        with torch.inference_mode():
            ids = whisper.generate(features, language="ta", task="transcribe",
                                   max_new_tokens=220)
        out[entry["file"]] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    hyps["whisper_large_v3"] = out
    # How well this run's Whisper agrees with the hypotheses stored by the earlier runs:
    stored = [e for e in manifest if e.get("whisper_hyp") is not None]
    same = sum(1 for e in stored if out[e["file"]] == e["whisper_hyp"].strip())
    log(f"  {len(out)} clips; identical to the stored hypothesis on {same}/{len(stored)}")

    # ---------------------------------------------------------------- scoring
    log("")
    log("=== CER by judge and set (95% bootstrap interval) ===")
    report: dict = {"judges": {}, "clips": manifest, "hypotheses": hyps}
    for judge_name, out in hyps.items():
        row: dict = {}
        counts: dict[str, dict[str, object]] = {}
        for set_name in ("real", "base", "geetha"):
            entries = [e for e in manifest if e["set"] == set_name]
            scored = {e["id"]: cer(out[e["file"]], e["reference"]) for e in entries}
            counts[set_name] = scored
            interval = bootstrap_ci(list(scored.values()))
            per_clip = [100 * (c.rate or 0.0) for c in scored.values()]
            row[set_name] = {
                "n": len(entries), "cer": round(100 * interval.point, 2),
                "ci": [round(100 * interval.low, 2), round(100 * interval.high, 2)],
                "median_clip_cer": round(float(np.median(per_clip)), 2),
                "clips_under_10": sum(1 for v in per_clip if v < 10)}
        common = [i for i in counts["geetha"] if i in counts["base"]]
        paired = paired_bootstrap([counts["geetha"][i] for i in common],
                                  [counts["base"][i] for i in common])
        row["geetha_minus_base"] = {
            "n": len(common), "delta": round(100 * paired.delta, 2),
            "ci": [round(100 * paired.low, 2), round(100 * paired.high, 2)],
            "p_value": paired.p_value}
        report["judges"][judge_name] = row
        log(f"  {judge_name}")
        for set_name in ("real", "base", "geetha"):
            r = row[set_name]
            log(f"    {set_name:6s} CER {r['cer']:6.2f} {r['ci']}  median clip "
                f"{r['median_clip_cer']:.1f}  under 10%: {r['clips_under_10']}/{r['n']}")
        d = row["geetha_minus_base"]
        log(f"    geetha - base on {d['n']} shared prompts: {d['delta']:+.2f} {d['ci']}, "
            f"p = {d['p_value']:.4f}")
    (OUT_DIR / "judge_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log("")
    log("wrote judge_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
