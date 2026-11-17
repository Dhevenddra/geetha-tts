r"""Open Tamil baselines on Geetha's 41 prompts, same judges, same references. (free Kaggle GPU)

D-052 and D-053 make two open models mandatory baselines on every Tamil set we report:

    rumik_oss_1        rumik-ai/rumik-oss-1 (CC BY-NC 4.0; benchmarking is a permitted use).
                       Voice Ira, no description, the card's sampling: temperature 0.8,
                       top-k 30. Mimi codec from the same repo.
    snortts_indic_v0   snorbyte/snorTTS-Indic-v0 (Apache-2.0). Speaker tamil188, the card's
                       sampling: temperature 0.4, top-p 0.9, repetition penalty 1.05. Stock
                       SNAC decoder; the card's optional 1.1x speed-up is NOT applied.
    base_S0, geetha_S0 archived clips, re-judged here so every row has the same judge run.

Each system is run the way its own card says, single shot, one fixed seed per prompt. The
prompts are Geetha's B7 subset as written (Latin script and digits included), so a system
that reads English badly pays for it here exactly as Geetha does. A clip that is mute or
never ended is invalid: its own rate, never in a CER pool. A system that fails to load is
reported as failed, with the error, and the others still run. Every hypothesis is written.
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
# Pinned to commits read on 14 Nov 2026. Rumik ships two Python files that run in this
# process (trust_remote_code); both were read at this commit and are checked by hash
# before the model is loaded. snorTTS is a plain Llama checkpoint: no remote code.
RUMIK_ID, RUMIK_REVISION = "rumik-ai/rumik-oss-1", "0ed3c98684e14350c910129c0efa179841c41ad2"
SNOR_ID, SNOR_REVISION = "snorbyte/snorTTS-Indic-v0", "c771394e1a427a6d2256ff881c8ed9f3cb6821c9"
SNAC_ID, SNAC_REVISION = "hubertsiuzdak/snac_24khz", "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
RUMIK_CODE_SHA256 = {
    "modeling_rumik_oss.py": "9b2c05014fa825cc5db030229a8e6707df4a80ebcd2c34b44dc9cbc294d0b8a6",
    "configuration_rumik_oss.py": "f7da28962c6314d52515f9ca4cf8c6ebbce80e08c928474e9dd17c3944c692cc",
}
RUMIK = dict(speaker="Ira", temperature=0.8, top_k=30, max_new_tokens=2048)
SNOR = dict(speaker="tamil188", temperature=0.4, top_p=0.9, repetition_penalty=1.05,
            max_new_tokens=int(CAP_SECONDS * 82))
_MAX_MOUNT_DEPTH = 4
SYSTEMS = ("rumik_oss_1", "snortts_indic_v0", "base_S0", "geetha_S0")
PAIRS = (("geetha_S0", "rumik_oss_1"), ("geetha_S0", "snortts_indic_v0"),
         ("base_S0", "rumik_oss_1"), ("base_S0", "snortts_indic_v0"))


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

    # Everything is fetched now, with the token passed explicitly; it never enters the
    # environment and is deleted before any third-party code is imported or run.
    conformer_dir = snapshot_download(CONFORMER_ID, revision=CONFORMER_REVISION, token=token)
    transcribe_dir = snapshot_download(TRANSCRIBE_ID, revision=TRANSCRIBE_REVISION,
                                       token=token)
    del token
    # The baselines are public: fetched without the token, at pinned commits, and only
    # the files inference needs (no server, training or demo scripts).
    rumik_dir = snapshot_download(
        RUMIK_ID, revision=RUMIK_REVISION, token=False,
        allow_patterns=["*.json", "*.safetensors", "codec/*", *RUMIK_CODE_SHA256])
    snor_dir = snapshot_download(SNOR_ID, revision=SNOR_REVISION, token=False,
                                 allow_patterns=["*.json", "*.safetensors", "*.jinja"])
    import hashlib
    for name, expected in RUMIK_CODE_SHA256.items():
        found = hashlib.sha256((Path(rumik_dir) / name).read_bytes()).hexdigest()
        if found != expected:
            raise SystemExit(f"{name} is not the file that was reviewed: {found}")
    extra = sorted(p.name for p in Path(rumik_dir).rglob("*.py")
                   if p.name not in RUMIK_CODE_SHA256)
    if extra:
        raise SystemExit(f"unreviewed Python in the Rumik snapshot: {extra}")
    revisions = {RUMIK_ID: RUMIK_REVISION, SNOR_ID: SNOR_REVISION, SNAC_ID: SNAC_REVISION}
    log(f"baseline revisions (pinned, code hashes verified): {revisions}")

    import traceback

    half = torch.cuda.get_device_capability(0)[0] < 8
    dtype = torch.float16 if half else torch.bfloat16
    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    clips: dict[str, dict[str, dict]] = {name: {} for name in SYSTEMS}
    failures: dict[str, str] = {}
    started = time.time()

    def keep(system: str, clip_id: str, wav: np.ndarray, check) -> None:
        file = f"{system}__{clip_id}.wav"
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        wave16 = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), SAMPLE_RATE, JUDGE_RATE)
        clips[system][clip_id] = {"file": file, "valid": check.valid, "reason": check.reason,
                                  "seconds": round(check.seconds, 2), "wave": wave16}

    def run_system(system: str, render) -> None:
        """Render every prompt; one clip failing is that clip's row, not the run's end."""
        errors = 0
        for position, row in enumerate(subset, 1):
            text, seed = written[row["id"]], SEED + row["b7_index"] * 100
            torch.manual_seed(seed)
            try:
                wav, ended = render(text)
            except Exception as problem:  # noqa: BLE001
                errors += 1
                if errors <= 2:
                    log(f"  {system}/{row['id']}: {type(problem).__name__}: {problem}")
                wav, ended = np.zeros(0, dtype=np.float32), False
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            keep(system, row["id"], wav, check_clip(wav, SAMPLE_RATE, ended_itself=ended))
            if position % 5 == 0 or position == len(subset):
                log(f"  {system} {position}/{len(subset)}  {time.time() - started:.0f} s")
        if errors:
            failures[system] = f"{errors} of {len(subset)} prompts raised"

    # ---------------------------------------------------------------- rumik-oss 1
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

        r_tok = AutoTokenizer.from_pretrained(rumik_dir)   # a stock tokenizer class
        r_lm = AutoModelForCausalLM.from_pretrained(
            rumik_dir, trust_remote_code=True, dtype=dtype).eval().cuda()
        mimi = MimiModel.from_pretrained(rumik_dir, subfolder="codec").eval().cuda()

        def render_rumik(text: str):
            inputs = r_tok(f'<text>{RUMIK["speaker"]}: {text}<audio>',
                           return_tensors="pt").to(r_lm.device)
            with torch.inference_mode():
                ids = r_lm.generate_audio(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=RUMIK["max_new_tokens"],
                    temperature=RUMIK["temperature"], top_k=RUMIK["top_k"], do_sample=True)
                tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]
                # the model appends its end token when it stops by itself
                ended = bool(tokens) and tokens[-1] == int(r_lm.config.audio_end_token_id)
                codes = r_lm.audio_tokens_to_codes(tokens)
                wav = mimi.decode(codes.to(mimi.device)).audio_values[0, 0]
            return wav.float().clamp(-1, 1).cpu().numpy(), ended

        run_system("rumik_oss_1", render_rumik)
        del r_lm, mimi
    except Exception as problem:  # noqa: BLE001
        failures["rumik_oss_1"] = f"{type(problem).__name__}: {problem}"
        log("rumik_oss_1 FAILED:\n" + traceback.format_exc())
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ snorTTS-Indic-v0
    try:
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s_tok = AutoTokenizer.from_pretrained(snor_dir)
        s_lm = AutoModelForCausalLM.from_pretrained(snor_dir, dtype=dtype).eval().cuda()
        snac = SNAC.from_pretrained(SNAC_ID, revision=SNAC_REVISION).eval().cuda()
        end_of_speech, audio_start = 128256 + 2, 128256 + 10

        def render_snor(text: str):
            prompt = (f"<custom_token_3><|begin_of_text|>{SNOR['speaker']}: {text}<|eot_id|>"
                      "<custom_token_4><custom_token_5><custom_token_1>")
            inputs = s_tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = s_lm.generate(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=SNOR["max_new_tokens"], do_sample=True,
                    temperature=SNOR["temperature"], top_p=SNOR["top_p"],
                    repetition_penalty=SNOR["repetition_penalty"],
                    eos_token_id=end_of_speech, pad_token_id=end_of_speech)
                new = out[0].tolist()[inputs.input_ids.shape[1]:]
                ended = bool(new) and new[-1] == end_of_speech
                codes = [t - audio_start for t in new if t >= audio_start]
                frames = len(codes) // 7
                if not frames:
                    return np.zeros(0, dtype=np.float32), ended
                grid = np.array(codes[: frames * 7], np.int64).reshape(frames, 7)
                grid -= np.arange(7, dtype=np.int64) * 4096
                grid = grid[np.all((grid >= 0) & (grid < 4096), axis=1)]
                if not len(grid):
                    return np.zeros(0, dtype=np.float32), ended
                layer1 = np.stack([grid[:, 1], grid[:, 4]], axis=1).reshape(-1)
                layer2 = np.stack([grid[:, 2], grid[:, 3], grid[:, 5], grid[:, 6]],
                                  axis=1).reshape(-1)
                tensors = [torch.from_numpy(x.copy()).unsqueeze(0).cuda()
                           for x in (grid[:, 0], layer1, layer2)]
                wav = snac.decode(tensors)[0, 0]
            return wav.float().clamp(-1, 1).cpu().numpy(), ended

        run_system("snortts_indic_v0", render_snor)
        del s_lm, snac
    except Exception as problem:  # noqa: BLE001
        failures["snortts_indic_v0"] = f"{type(problem).__name__}: {problem}"
        log("snortts_indic_v0 FAILED:\n" + traceback.format_exc())
    torch.cuda.empty_cache()

    for system in ("rumik_oss_1", "snortts_indic_v0"):
        for row in subset:   # a system that never loaded still gets an (invalid) row each
            if row["id"] not in clips[system]:
                empty = np.zeros(0, dtype=np.float32)
                keep(system, row["id"], empty, check_clip(empty, SAMPLE_RATE, ended_itself=False))

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
        "decoding": {"rumik_oss_1": RUMIK, "snortts_indic_v0": SNOR,
                     "base_S0": "archived B7 clips", "geetha_S0": "archived 11 Nov clips",
                     "seed_rule": "1234 + b7_index*100", "single_shot": True},
        "revisions": revisions, "failures": failures,
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
            if len(common) < 2:
                block["paired"][f"{a}_minus_{b}"] = entry
                continue
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
    (OUT_DIR / "m2_baselines.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log("")
    log(f"wrote m2_baselines.json and {len(list(audio.iterdir()))} wavs "
        f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
