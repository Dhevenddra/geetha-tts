r"""M2 Stage B at the default decoding (S3), judged by the two Indic judges. (free Kaggle GPU)

Every number before this was decoded at S0 (temperature 0.6, no penalty, no guidance) and
judged by Whisper. D-055 made S3 the default decoding and D-053 dropped Whisper, so every
publication-candidate number is re-run here on the same 41 paired prompts.

    systems   geetha_S3   Stage B adapter, speaker + CONV + studio, temperature 0.9,
                          repetition 1.1, style guidance 3.5        (rendered here)
              base_S1     the base model, its own prompt layout, temperature 0.9,
                          repetition 1.1                             (rendered here)
              base_S0     the base model as archived by B7           (archived clips)
              geetha_S0   Stage B as heard on 11 Nov                 (archived clips)
    judges    Indic-Transcribe-core (primary), IndicConformer-600M RNNT (second)
    refs      written  the prompt as written: Latin script and digits, so only the
                       paired difference means anything
              spoken   spoken-form Tamil script (spoken_refs_v1.json), closest of the
                       accepted readings: an absolute CER, with spelling noise that is
                       the same for every system

A clip that is mute or never ended is invalid: its own rate, never in a CER pool. Paired
differences use the prompts valid in both systems. Every hypothesis is written out.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "bodhan-ai/indic-speak"
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTER = "m2-main/stageB/final"
CONFORMER_ID = "ai4bharat/indic-conformer-600m-multilingual"
TRANSCRIBE_ID = "bodhan-ai/indic-transcribe-core"
CONFORMER_REVISION = "2a77d305038dfc502fd9464e7d028b90381976cb"
TRANSCRIBE_REVISION = "e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537"
SAMPLE_RATE, JUDGE_RATE = 24_000, 16_000
CAP_SECONDS = 22.0
MAX_NEW_TOKENS = int(CAP_SECONDS * 82)
HEAD = "<|start_of_human|><|begin_of_text|>"
TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"
SEED = 1234
TEMPERATURE, REPETITION, GUIDANCE = 0.9, 1.1, 3.5
_MAX_MOUNT_DEPTH = 4
SYSTEMS = ("geetha_S3", "base_S1", "base_S0", "geetha_S0")
PAIRS = (("geetha_S3", "base_S0"), ("geetha_S3", "base_S1"), ("geetha_S3", "geetha_S0"),
         ("base_S1", "base_S0"))


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

    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            capture_output=True, text=True)
    log(f"pip uninstall torchao -> {result.returncode}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snac==1.2.1",
                    "onnxruntime==1.20.1", "soundfile==0.14.0", "jiwer==4.0.0",
                    "sentencepiece"], check=True)
    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import LogitsProcessor, LogitsProcessorList

    token = (find_dataset("hf_read_token.txt") / "hf_read_token.txt").read_text(
        encoding="utf-8").strip()
    source = find_dataset("scoring", "text", "eval")
    package = Path("/kaggle/tmp/pkg/geetha")
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(package.parent))
    from geetha.eval.validity import check_clip
    from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap
    from geetha.scoring.metrics import cer

    data = find_dataset("manifest.json", "spoken_refs_v1.json")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    spoken = json.loads((data / "spoken_refs_v1.json").read_text(encoding="utf-8"))["refs"]
    subset = json.loads((find_dataset("m2pre_subset.json") / "m2pre_subset.json")
                        .read_text(encoding="utf-8"))["clips"]
    subset.sort(key=lambda row: row["b7_index"])
    written = {e["id"]: e["reference"] for e in manifest if e["set"] == "base"}
    archived = {"base_S0": {e["id"]: e["file"] for e in manifest if e["set"] == "base"},
                "geetha_S0": {e["id"]: e["file"] for e in manifest if e["set"] == "geetha"}}
    missing = [r["id"] for r in subset if r["id"] not in written or r["id"] not in spoken]
    if missing:
        raise SystemExit(f"prompts without a reference: {missing}")
    pure_tamil = {i for i, text in written.items() if not re.search(r"[A-Za-z0-9]", text)}
    log(f"{len(subset)} prompts ({len(pure_tamil)} pure Tamil as written); "
        f"gpu: {torch.cuda.get_device_name(0)}")

    # Everything gated is fetched now, with the token passed explicitly; it never enters
    # the environment and is deleted before any third-party code is imported.
    adapter_root = Path(snapshot_download(CKPT_REPO, allow_patterns=[f"{ADAPTER}/*"],
                                          token=token)) / ADAPTER
    if not (adapter_root / "adapter_config.json").exists():
        adapter_root = sorted(adapter_root.rglob("adapter_config.json"))[0].parent
    local = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)
    conformer_dir = snapshot_download(CONFORMER_ID, revision=CONFORMER_REVISION, token=token)
    transcribe_dir = snapshot_download(TRANSCRIBE_ID, revision=TRANSCRIBE_REVISION,
                                       token=token)
    del token

    # ------------------------------------------------------------------ rendering
    sys.path.insert(0, local)
    from inference import TTS, build_prompt, ids_to_codes  # type: ignore

    tts = TTS(local)
    tts.lm = PeftModel.from_pretrained(tts.lm, str(adapter_root), adapter_name="stageB")
    tts.lm.eval()  # NOT merged (D-049)
    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
    tok = tts.tok
    start_speech = tok.convert_tokens_to_ids("<|start_of_speech|>")
    stop_ids = {tts._eos, tok.eos_token_id}

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def geetha_prompt(text: str, speaker: str, style: str) -> list[int]:
        return (enc(HEAD) + enc(f"<|speaker>{speaker}<speaker|>\n")
                + (enc(f"<|style>{style}<style|>\n") if style else [])
                + enc("<|env>studio<env|>\n") + enc(text) + enc(TAIL) + [start_speech])

    class Guidance(LogitsProcessor):
        def __init__(self, other: list[int], scale: float):
            self.other = torch.tensor([other], device=tts.device)
            self.scale, self.past, self.started = scale, None, False

        def __call__(self, input_ids, scores):
            feed = input_ids[:, -1:] if self.started else self.other
            out = tts.lm(input_ids=feed, past_key_values=self.past, use_cache=True)
            self.past, self.started = out.past_key_values, True
            cond = torch.log_softmax(scores.float(), dim=-1)
            unc = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
            return (unc + self.scale * (cond - unc)).to(scores.dtype)

    def generate(ids: list[int], other: list[int] | None, seed: int):
        torch.manual_seed(seed)
        x = torch.tensor([ids], device=tts.device)
        processors = LogitsProcessorList([Guidance(other, GUIDANCE)] if other else [])
        with torch.no_grad():
            gen = tts.lm.generate(
                input_ids=x, attention_mask=torch.ones_like(x),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                top_p=0.9, top_k=50, repetition_penalty=REPETITION,
                eos_token_id=[tts._eos, tok.eos_token_id],
                pad_token_id=tok.eos_token_id, logits_processor=processors)
            new = gen[0].tolist()[len(ids):]
            ended = bool(new) and new[-1] in stop_ids
            try:
                codes = ids_to_codes(new, tts._base, tts.device)
                wav = tts.vocos(tts.snac.quantizer.from_codes(codes).float())
                wav = wav[0, 0].clamp(-1, 1).float().cpu().numpy()
            except ValueError:
                wav = np.zeros(0, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32).reshape(-1), ended

    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    clips: dict[str, dict[str, dict]] = {name: {} for name in SYSTEMS}
    started = time.time()

    def keep(system: str, clip_id: str, wav: np.ndarray, check) -> None:
        file = f"{system}__{clip_id}.wav"
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        wave16 = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), SAMPLE_RATE, JUDGE_RATE)
        clips[system][clip_id] = {"file": file, "valid": check.valid, "reason": check.reason,
                                  "seconds": round(check.seconds, 2), "wave": wave16}

    for position, row in enumerate(subset, 1):
        text, seed = written[row["id"]], SEED + row["b7_index"] * 100
        speaker = row["speaker"].lower()
        wav, ended = generate(geetha_prompt(text, speaker, "CONV"),
                              geetha_prompt(text, speaker, ""), seed)
        keep("geetha_S3", row["id"], wav, check_clip(wav, SAMPLE_RATE, ended_itself=ended))
        if position % 5 == 0 or position == len(subset):
            log(f"  geetha_S3 {position}/{len(subset)}  {time.time() - started:.0f} s")
    with tts.lm.disable_adapter():
        for position, row in enumerate(subset, 1):
            text, seed = written[row["id"]], SEED + row["b7_index"] * 100
            wav, ended = generate(build_prompt(tok, text, row["speaker"], ""), None, seed)
            keep("base_S1", row["id"], wav, check_clip(wav, SAMPLE_RATE, ended_itself=ended))
            if position % 5 == 0 or position == len(subset):
                log(f"  base_S1 {position}/{len(subset)}  {time.time() - started:.0f} s")
    for system, files in archived.items():
        for clip_id, file in files.items():
            samples, rate = sf.read(str(data / file), dtype="float32", always_2d=True)
            wav = samples.mean(axis=1)
            # No end-of-speech flag was archived: length against the cap stands in.
            check = check_clip(wav, rate, cap_seconds=CAP_SECONDS)
            wave16 = torch.from_numpy(np.ascontiguousarray(wav))
            if rate != JUDGE_RATE:
                wave16 = torchaudio.functional.resample(wave16, rate, JUDGE_RATE)
            clips[system][clip_id] = {"file": file, "valid": check.valid,
                                      "reason": check.reason,
                                      "seconds": round(check.seconds, 2), "wave": wave16}
    for system in SYSTEMS:
        good = sum(1 for c in clips[system].values() if c["valid"])
        log(f"  {system}: {good}/{len(clips[system])} clips valid")
    del tts
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ judging
    hyps: dict[str, dict[str, dict[str, str]]] = {}
    log("")
    log("=== Indic-Transcribe-core (primary judge) ===")
    sys.path.insert(0, transcribe_dir)
    from indic_transcribe import IndicTranscribe  # type: ignore

    judge = IndicTranscribe.from_pretrained(transcribe_dir, device="cuda")
    scratch = Path("/kaggle/tmp/judge.wav")
    out: dict[str, dict[str, str]] = {name: {} for name in SYSTEMS}
    for system in SYSTEMS:
        for clip_id, clip in clips[system].items():
            if not clip["valid"]:
                continue
            sf.write(scratch, clip["wave"].numpy(), JUDGE_RATE)
            text = judge.transcribe(str(scratch), lang="ta")
            out[system][clip_id] = (text[0] if isinstance(text, tuple) else text).strip()
    hyps["indic_transcribe_core"] = out
    del judge
    torch.cuda.empty_cache()

    log("=== IndicConformer-600M RNNT (second judge) ===")
    sys.path.insert(0, conformer_dir)
    from model_onnx import IndicASRConfig, IndicASRModel  # type: ignore

    settings = json.loads((Path(conformer_dir) / "config.json").read_text())
    settings.pop("auto_map", None)
    conformer = IndicASRModel(IndicASRConfig(ts_folder=conformer_dir, **settings))
    out = {name: {} for name in SYSTEMS}
    for system in SYSTEMS:
        for clip_id, clip in clips[system].items():
            if not clip["valid"]:
                continue
            try:
                out[system][clip_id] = conformer(clip["wave"].unsqueeze(0), "ta", "rnnt")
            except Exception as problem:  # noqa: BLE001
                out[system][clip_id] = ""
                log(f"  rnnt failed on {system}/{clip_id}: {type(problem).__name__}")
    hyps["indicconformer_rnnt"] = out

    # ------------------------------------------------------------------ scoring
    def spoken_cer(hypothesis: str, clip_id: str):
        return min((cer(hypothesis, ref) for ref in spoken[clip_id]),
                   key=lambda c: c.rate if c.rate is not None else 9.9)

    def summary(counts: list) -> dict:
        if len(counts) < 2:
            return {"n": len(counts)}
        ci = bootstrap_ci(counts)
        return {"n": len(counts), "cer": round(100 * ci.point, 2),
                "ci": [round(100 * ci.low, 2), round(100 * ci.high, 2)]}

    report: dict = {
        "decoding": {"geetha_S3": {"temperature": TEMPERATURE, "repetition": REPETITION,
                                   "style": "CONV", "style_guidance": GUIDANCE},
                     "base_S1": {"temperature": TEMPERATURE, "repetition": REPETITION},
                     "base_S0": "archived B7 clips", "geetha_S0": "archived 11 Nov clips",
                     "top_p": 0.9, "top_k": 50, "seed_rule": "1234 + b7_index*100"},
        "references": "spoken_refs_v1.json: drafted 14 Nov, not yet checked by a reader",
        "validity": {s: {"valid": sum(1 for c in clips[s].values() if c["valid"]),
                         "n": len(clips[s]),
                         "invalid": {i: c["reason"] for i, c in clips[s].items()
                                     if not c["valid"]}} for s in SYSTEMS},
        "judges": {}, "hypotheses": hyps,
        "clips": {s: {i: {k: v for k, v in c.items() if k != "wave"}
                      for i, c in clips[s].items()} for s in SYSTEMS}}
    log("")
    for judge_name, by_system in hyps.items():
        scored = {s: {i: {"written": cer(h, written[i]), "spoken": spoken_cer(h, i)}
                      for i, h in by_system[s].items()} for s in SYSTEMS}
        block: dict = {"systems": {}, "paired": {}}
        log(f"=== {judge_name} ===")
        for s in SYSTEMS:
            ids = list(scored[s])
            block["systems"][s] = {
                "spoken": summary([scored[s][i]["spoken"] for i in ids]),
                "written": summary([scored[s][i]["written"] for i in ids]),
                "spoken_pure_tamil": summary([scored[s][i]["spoken"] for i in ids
                                              if i in pure_tamil])}
            r = block["systems"][s]
            log(f"  {s:10s} spoken-ref CER {r['spoken'].get('cer')} {r['spoken'].get('ci')}"
                f"  written-ref {r['written'].get('cer')}  pure-Tamil "
                f"{r['spoken_pure_tamil'].get('cer')} (n={r['spoken_pure_tamil']['n']})"
                f"  on {len(ids)} valid clips")
        for a, b in PAIRS:
            common = [i for i in scored[a] if i in scored[b]]
            entry = {"n": len(common)}
            for kind in ("spoken", "written"):
                p = paired_bootstrap([scored[a][i][kind] for i in common],
                                     [scored[b][i][kind] for i in common])
                entry[kind] = {"delta": round(100 * p.delta, 2),
                               "ci": [round(100 * p.low, 2), round(100 * p.high, 2)],
                               "p_value": p.p_value}
            block["paired"][f"{a}_minus_{b}"] = entry
            log(f"  {a} - {b} on {len(common)} shared valid prompts: spoken "
                f"{entry['spoken']['delta']:+.2f} {entry['spoken']['ci']} "
                f"p={entry['spoken']['p_value']:.4f}; written "
                f"{entry['written']['delta']:+.2f} {entry['written']['ci']}")
        block["per_clip"] = {s: {i: {"spoken": round(100 * (c["spoken"].rate or 0), 2),
                                     "written": round(100 * (c["written"].rate or 0), 2)}
                                 for i, c in scored[s].items()} for s in SYSTEMS}
        report["judges"][judge_name] = block
    (OUT_DIR / "m2_eval_s3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log("")
    log(f"wrote m2_eval_s3.json and {len(list(audio.iterdir()))} wavs "
        f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
