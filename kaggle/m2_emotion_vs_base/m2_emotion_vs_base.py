r"""M2: is the emotion Geetha's, or the decoder's? Geetha against the base, same decoding.

On 12 Nov the operator heard that style guidance makes Geetha's emotions "sound better",
and S3 became the default. That job never ran the BASE model through the same guided
decoding. The base already takes a style block and already speaks in these two voices, so
until it is run the same way, "Geetha is more expressive" is not a finding.

  systems    geetha   Stage B adapter, speaker + style + env=studio (its own layout)
             base     bodhan-ai/indic-speak, its own layout (speaker, style)
  decoding   G0    temperature 0.9, repetition 1.1, no guidance        (both)
             G3.5  the same with style guidance 3.5                     (both)
             G2    style guidance 2.0                                   (geetha only)
  texts      four neutral carriers x HAPPY / SAD / ANGER, and three sentences whose words
             carry the emotion, in both voices. Each text also gets one neutral render per
             system at G0 (CONV for geetha, no style for the base): the reference the
             emotion render is measured against.
  measures   per emotion render, against its own system's neutral render of the same
             text: median-pitch shift (signed, %), pitch range (semitones), energy
             spread (dB), duration ratio. Paired geetha - base at the same guidance,
             bootstrap 95% CI over the 30 items.
             CER by both Indic judges against the text itself (pure Tamil script, so the
             figure is absolute): what the guidance costs in intelligibility.

A pitch statistic says the prosody MOVED, not that it sounds like the emotion asked for;
that is the operator's ear. The signed shifts are printed by emotion so a wrong direction
(a SAD that goes up and speeds up) is visible.
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
MODEL_ID = "bodhan-ai/indic-speak"
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTERS = {"stageB": "m2-main/stageB/final"}
CONFORMER_ID = "ai4bharat/indic-conformer-600m-multilingual"
TRANSCRIBE_ID = "bodhan-ai/indic-transcribe-core"
CONFORMER_REVISION = "2a77d305038dfc502fd9464e7d028b90381976cb"
TRANSCRIBE_REVISION = "e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537"
SAMPLE_RATE, JUDGE_RATE = 24_000, 16_000
MAX_NEW_TOKENS = int(22.0 * 82)
HEAD = "<|start_of_human|><|begin_of_text|>"
TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"
SEED = 1416
TEMPERATURE, REPETITION = 0.9, 1.1
_MAX_MOUNT_DEPTH = 4
EMOTIONS = ("HAPPY", "SAD", "ANGER")
# Written for this run; not eval-vault items. Carriers say nothing emotional.
CARRIERS = {
    "c1": "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.",
    "c2": "அந்த புத்தகம் மேசை மேல தான் இருக்கு.",
    "c3": "இன்னைக்கு சாயங்காலம் அவங்க வீட்டுக்கு வர்றாங்க.",
    "c4": "பஸ் இன்னும் பத்து நிமிஷத்துல கிளம்பிடும்.",
}
WORDED = {
    "HAPPY": "ஐயோ, நிஜமாவா! எனக்கு ரொம்ப சந்தோஷமா இருக்கு!",
    "SAD": "அவர் இனிமே வரமாட்டார்னு நினைச்சா மனசு ரொம்ப கஷ்டமா இருக்கு.",
    "ANGER": "எத்தனை தடவை சொல்லணும்? இனிமே இப்படி பண்ணாத!",
}
# (system, guidance) rendered for the emotion items; neutral references are G0 only.
CONDITIONS = (("geetha", 0.0), ("base", 0.0), ("geetha", 3.5), ("base", 3.5), ("geetha", 2.0))


def log(message: str) -> None:
    print(message, flush=True)


def fix_environment() -> None:
    """Kaggle's torchao 0.10.0 makes PEFT unable to build any LoRA; remove it first."""
    import subprocess

    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            capture_output=True, text=True)
    log(f"  pip uninstall torchao -> {result.returncode}")


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    """Address inputs by their contents, never by a vendor's path."""
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"could not find a mount containing {markers}")


def require_token() -> str:
    path = find_dataset("hf_read_token.txt") / "hf_read_token.txt"
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("hf_read_token.txt is empty")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def ensure_dependencies() -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "snac==1.2.1", "soundfile==0.14.0", "librosa==1.0.0",
                    "jiwer==4.0.0", "onnxruntime==1.20.1", "sentencepiece"], check=True)
    log("installed snac 1.2.1, soundfile 0.14.0, librosa 1.0.0, jiwer 4.0.0 (pinned)")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    if not (source / "eval" / "validity.py").exists():
        raise SystemExit("geetha-code has no eval/validity.py: re-push the code dataset")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def fetch_adapters(token: str) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    found = {}
    for name, subdir in ADAPTERS.items():
        root = Path(snapshot_download(CKPT_REPO, allow_patterns=[f"{subdir}/*"],
                                      token=token))
        folder = root / subdir
        if not (folder / "adapter_config.json").exists():
            nested = sorted(folder.rglob("adapter_config.json"))
            if not nested:
                raise SystemExit(f"no adapter_config.json under {subdir}")
            folder = nested[0].parent
        log(f"  {name}: {folder}  ({sorted(p.name for p in folder.iterdir())})")
        found[name] = folder
    return found


def build_tts(token: str, adapters: dict[str, Path]):
    """The base repo's TTS wrapper with both adapters attached, NOT merged (D-049)."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    local = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)
    # inference.py is third-party code run in this process: everything private is on
    # disk by now, so the token leaves the environment before it is imported.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    sys.path.insert(0, local)
    from inference import TTS  # type: ignore

    tts = TTS(local)
    before = sum(p.numel() for p in tts.lm.parameters())
    names = list(adapters)
    tts.lm = PeftModel.from_pretrained(tts.lm, str(adapters[names[0]]),
                                       adapter_name=names[0])
    for name in names[1:]:
        tts.lm.load_adapter(str(adapters[name]), adapter_name=name)
    tts.lm.eval()
    after = sum(p.numel() for p in tts.lm.parameters())
    log(f"  adapters attached (unmerged): {before:,} -> {after:,} parameters")
    if after <= before:
        raise SystemExit("attaching the adapters added no parameters")
    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        log("  LM in fp16 (no native bf16 on this card)")
    return tts



def prosody(wav: np.ndarray) -> dict:
    """Pitch and energy statistics: crude, but the same for every clip."""
    import librosa

    if len(wav) < SAMPLE_RATE // 4:
        return {}
    f0, _, _ = librosa.pyin(wav.astype(np.float32), fmin=70, fmax=500, sr=SAMPLE_RATE)
    f0 = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=wav.astype(np.float32))[0]
    if len(f0) < 5:
        return {}
    semitones = 12 * np.log2(f0 / np.median(f0))
    return {"f0_median_hz": round(float(np.median(f0)), 1),
            "f0_range_semitones": round(float(np.percentile(semitones, 95)
                                              - np.percentile(semitones, 5)), 2),
            "energy_db_std": round(float(np.std(20 * np.log10(rms + 1e-6))), 2),
            "seconds": round(len(wav) / SAMPLE_RATE, 2)}


def main() -> int:
    fix_environment()
    import soundfile as sf
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download
    from transformers import LogitsProcessor, LogitsProcessorList

    token = require_token()
    ensure_dependencies()
    import_geetha()
    from geetha.eval.validity import check_clip
    from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap
    from geetha.scoring.metrics import cer

    adapters = fetch_adapters(token)
    conformer_dir = snapshot_download(CONFORMER_ID, revision=CONFORMER_REVISION, token=token)
    transcribe_dir = snapshot_download(TRANSCRIBE_ID, revision=TRANSCRIBE_REVISION,
                                       token=token)
    tts = build_tts(token, adapters)     # drops the token from the environment
    del token
    from inference import build_prompt, ids_to_codes  # type: ignore

    tok = tts.tok
    start_speech = tok.convert_tokens_to_ids("<|start_of_speech|>")
    stop_ids = {tts._eos, tok.eos_token_id}
    log(f"  GPU: {torch.cuda.get_device_name(0)}")

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def layout(system: str, text: str, voice: str, style: str) -> list[int]:
        """Each system in its own prompt layout. style '' = the unconditional branch."""
        if system == "base":
            return build_prompt(tok, text, voice.capitalize(), style)
        return (enc(HEAD) + enc(f"<|speaker>{voice}<speaker|>\n")
                + (enc(f"<|style>{style}<style|>\n") if style else [])
                + enc("<|env>studio<env|>\n") + enc(text) + enc(TAIL) + [start_speech])

    class Guidance(LogitsProcessor):
        """Classifier-free guidance against a second prompt, with its own cache."""

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

    def generate(ids: list[int], other: list[int] | None, scale: float, seed: int):
        torch.manual_seed(seed)
        x = torch.tensor([ids], device=tts.device)
        processors = LogitsProcessorList()
        if other is not None and scale > 0:
            processors.append(Guidance(other, scale))
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

    items = []   # (item id, text, emotion)
    for emotion in EMOTIONS:
        for name, text in CARRIERS.items():
            items.append((f"{name}_{emotion}", name, text, emotion))
        items.append((f"w_{emotion}", f"w_{emotion}", WORDED[emotion], emotion))
    texts = {**CARRIERS, **{f"w_{e}": t for e, t in WORDED.items()}}

    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    waves: dict[str, "torch.Tensor"] = {}
    started = time.time()

    def flush(extra: dict | None = None) -> None:
        (OUT_DIR / "m2_emotion_vs_base.json").write_text(
            json.dumps({"decoding": {"temperature": TEMPERATURE, "repetition": REPETITION,
                                     "top_p": 0.9, "top_k": 50, "seed": SEED},
                        "conditions": [list(c) for c in CONDITIONS],
                        "rows": rows, **(extra or {})}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    def render(system, voice, text_id, text, style, scale, kind, seed) -> None:
        ids = layout(system, text, voice, style)
        other = layout(system, text, voice, "") if scale > 0 else None
        if system == "base":
            with tts.lm.disable_adapter():
                wav, ended = generate(ids, other, scale, seed)
        else:
            wav, ended = generate(ids, other, scale, seed)
        check = check_clip(wav, SAMPLE_RATE, ended_itself=ended)
        file = f"{system}_G{scale:g}_{voice}_{text_id}_{kind}.wav"
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        if check.valid:
            waves[file] = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(wav)), SAMPLE_RATE, JUDGE_RATE)
        rows.append({"file": file, "system": system, "guidance": scale, "voice": voice,
                     "text_id": text_id, "text": text, "kind": kind,
                     "valid": check.valid, "reason": check.reason,
                     **(prosody(wav) if check.valid else {})})

    log("=== 1. neutral references (G0), each system in its own layout ===")
    for system in ("geetha", "base"):
        for v_index, voice in enumerate(("anitha", "arun")):
            for t_index, (text_id, text) in enumerate(texts.items()):
                render(system, voice, text_id, text, "CONV" if system == "geetha" else "",
                       0.0, "NEUTRAL", SEED + 100 * v_index + t_index)
        log(f"  {system}: {len(rows)} clips, {time.time() - started:.0f} s")
        flush()

    log("=== 2. emotion renders ===")
    for system, scale in CONDITIONS:
        for v_index, voice in enumerate(("anitha", "arun")):
            for t_index, (_, text_id, text, emotion) in enumerate(items):
                render(system, voice, text_id, text, emotion, scale, emotion,
                       SEED + 1000 + 100 * v_index + t_index)
        log(f"  {system} G{scale:g}: {len(rows)} clips, {time.time() - started:.0f} s")
        flush()
    del tts
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ prosody
    neutral = {(r["system"], r["voice"], r["text_id"]): r for r in rows if r["kind"] == "NEUTRAL"}
    effect: dict[tuple, dict] = {}
    for r in rows:
        ref = neutral.get((r["system"], r["voice"], r["text_id"]))
        if r["kind"] == "NEUTRAL" or not r.get("f0_median_hz") or not ref \
                or not ref.get("f0_median_hz"):
            continue
        r["vs_neutral"] = {
            "pitch_shift_pct": round(100 * (r["f0_median_hz"] / ref["f0_median_hz"] - 1), 1),
            "pitch_range_st": round(r["f0_range_semitones"] - ref["f0_range_semitones"], 2),
            "energy_spread_db": round(r["energy_db_std"] - ref["energy_db_std"], 2),
            "duration_ratio": round(r["seconds"] / ref["seconds"], 3)}
        effect[(r["system"], r["guidance"], r["voice"], r["text_id"], r["kind"])] = r["vs_neutral"]

    summary: dict = {"by_condition": {}, "by_emotion": {}, "paired": {}}
    log("")
    log("=== did the style move the prosody? (emotion render minus the same system's "
        "neutral render) ===")
    for system, scale in CONDITIONS:
        mine = [(k, v) for k, v in effect.items() if k[0] == system and k[1] == scale]
        if not mine:
            continue
        absolute = float(np.mean([abs(v["pitch_shift_pct"]) for _, v in mine]))
        block = {"n": len(mine), "abs_pitch_shift_pct": round(absolute, 1),
                 "pitch_range_st": round(float(np.mean([v["pitch_range_st"] for _, v in mine])), 2),
                 "energy_spread_db": round(float(np.mean([v["energy_spread_db"] for _, v in mine])), 2)}
        summary["by_condition"][f"{system}_G{scale:g}"] = block
        log(f"  {system:6s} G{scale:<3g} n={block['n']:2d}  |pitch shift| "
            f"{block['abs_pitch_shift_pct']:5.1f}%  range {block['pitch_range_st']:+.2f} st  "
            f"energy spread {block['energy_spread_db']:+.2f} dB")
        for emotion in EMOTIONS:
            part = [v for k, v in mine if k[4] == emotion]
            if part:
                signed = {"n": len(part),
                          "pitch_shift_pct": round(float(np.mean([v["pitch_shift_pct"] for v in part])), 1),
                          "duration_ratio": round(float(np.mean([v["duration_ratio"] for v in part])), 3),
                          "energy_spread_db": round(float(np.mean([v["energy_spread_db"] for v in part])), 2)}
                summary["by_emotion"][f"{system}_G{scale:g}_{emotion}"] = signed
                log(f"      {emotion:6s} pitch {signed['pitch_shift_pct']:+6.1f}%  duration x"
                    f"{signed['duration_ratio']:.2f}  energy spread {signed['energy_spread_db']:+.2f} dB")

    def mean_diff(a: list[float], b: list[float]) -> dict:
        """Mean paired difference with a percentile bootstrap 95% CI (10,000 resamples)."""
        d = np.array(a) - np.array(b)
        rng = np.random.default_rng(1234)
        boots = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
        return {"n": len(d), "delta": round(float(d.mean()), 2),
                "ci": [round(float(np.percentile(boots, 2.5)), 2),
                       round(float(np.percentile(boots, 97.5)), 2)]}

    log("")
    log("=== geetha minus base at the same guidance, paired over items ===")
    for a, b in ((("geetha", 0.0), ("base", 0.0)), (("geetha", 3.5), ("base", 3.5)),
                 (("geetha", 3.5), ("geetha", 0.0)), (("base", 3.5), ("base", 0.0)),
                 (("geetha", 2.0), ("geetha", 0.0)), (("geetha", 2.0), ("base", 3.5))):
        keys = [k[2:] for k in effect if k[:2] == a and (b + k[2:]) in effect]
        if len(keys) < 3:
            continue
        entry = {}
        for field, absolute in (("pitch_shift_pct", True), ("pitch_range_st", False),
                                ("energy_spread_db", False)):
            left = [effect[a + k][field] for k in keys]
            right = [effect[b + k][field] for k in keys]
            if absolute:
                left, right = [abs(x) for x in left], [abs(x) for x in right]
            entry[("abs_" if absolute else "") + field] = mean_diff(left, right)
        name = f"{a[0]}_G{a[1]:g}_minus_{b[0]}_G{b[1]:g}"
        summary["paired"][name] = entry
        log(f"  {name}: |pitch shift| {entry['abs_pitch_shift_pct']['delta']:+.1f} "
            f"{entry['abs_pitch_shift_pct']['ci']} pts; range "
            f"{entry['pitch_range_st']['delta']:+.2f} {entry['pitch_range_st']['ci']} st; "
            f"energy {entry['energy_spread_db']['delta']:+.2f} {entry['energy_spread_db']['ci']} dB"
            f"  (n={len(keys)})")
    flush({"prosody": summary})

    # ------------------------------------------------------------------ judging
    hyps: dict[str, dict[str, str]] = {}
    log("")
    log("=== Indic-Transcribe-core (primary judge) ===")
    sys.path.insert(0, transcribe_dir)
    from indic_transcribe import IndicTranscribe  # type: ignore

    judge = IndicTranscribe.from_pretrained(transcribe_dir, device="cuda")
    scratch = Path("/kaggle/tmp/judge.wav")
    scratch.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for file, wave in waves.items():
        sf.write(scratch, wave.numpy(), JUDGE_RATE)
        text = judge.transcribe(str(scratch), lang="ta")
        out[file] = (text[0] if isinstance(text, tuple) else text).strip()
    hyps["indic_transcribe_core"] = out
    del judge
    torch.cuda.empty_cache()

    log("=== IndicConformer-600M RNNT (second judge) ===")
    sys.path.insert(0, conformer_dir)
    from model_onnx import IndicASRConfig, IndicASRModel  # type: ignore

    settings = json.loads((Path(conformer_dir) / "config.json").read_text())
    settings.pop("auto_map", None)
    conformer = IndicASRModel(IndicASRConfig(ts_folder=conformer_dir, **settings))
    out = {}
    for file, wave in waves.items():
        try:
            out[file] = conformer(wave.unsqueeze(0), "ta", "rnnt")
        except Exception as problem:  # noqa: BLE001
            out[file] = ""
            log(f"  rnnt failed on {file}: {type(problem).__name__}")
    hyps["indicconformer_rnnt"] = out

    by_file = {r["file"]: r for r in rows}
    cer_report: dict = {}
    log("")
    for judge_name, by_clip in hyps.items():
        counts = {file: cer(h, by_file[file]["text"]) for file, h in by_clip.items()}
        for file, c in counts.items():
            by_file[file].setdefault("cer", {})[judge_name] = round(100 * (c.rate or 0), 2)
        block: dict = {"systems": {}, "paired": {}}
        log(f"=== {judge_name}: CER against the text itself ===")
        groups: dict[str, dict] = {}
        for file, c in counts.items():
            r = by_file[file]
            key = (f"{r['system']}_neutral" if r["kind"] == "NEUTRAL"
                   else f"{r['system']}_G{r['guidance']:g}")
            groups.setdefault(key, {})[(r["voice"], r["text_id"], r["kind"])] = c
        for key, members in sorted(groups.items()):
            ci = bootstrap_ci(list(members.values()))
            total = sum(1 for r in rows if (f"{r['system']}_neutral" if r["kind"] == "NEUTRAL"
                                            else f"{r['system']}_G{r['guidance']:g}") == key)
            block["systems"][key] = {"n": len(members), "rendered": total,
                                     "cer": round(100 * ci.point, 2),
                                     "ci": [round(100 * ci.low, 2), round(100 * ci.high, 2)]}
            log(f"  {key:16s} CER {block['systems'][key]['cer']:6.2f} "
                f"{block['systems'][key]['ci']}  on {len(members)}/{total} valid clips")
        for a, b in (("geetha_G0", "base_G0"), ("geetha_G3.5", "base_G3.5"),
                     ("geetha_G3.5", "geetha_G0"), ("geetha_G2", "geetha_G0"),
                     ("base_G3.5", "base_G0")):
            if a not in groups or b not in groups:
                continue
            common = [k for k in groups[a] if k in groups[b]]
            if len(common) < 2:
                continue
            p = paired_bootstrap([groups[a][k] for k in common], [groups[b][k] for k in common])
            block["paired"][f"{a}_minus_{b}"] = {
                "n": len(common), "delta": round(100 * p.delta, 2),
                "ci": [round(100 * p.low, 2), round(100 * p.high, 2)], "p_value": p.p_value}
            log(f"  {a} - {b}: {100 * p.delta:+.2f} [{100 * p.low:.2f}, {100 * p.high:.2f}] "
                f"p={p.p_value:.4f}  (n={len(common)})")
        cer_report[judge_name] = block
    flush({"prosody": summary, "cer": cer_report, "hypotheses": hyps})
    bad = [r["file"] for r in rows if not r["valid"]]
    log("")
    log(f"invalid clips: {len(bad)}/{len(rows)} {bad}")
    log(f"wrote m2_emotion_vs_base.json and {len(list(audio.iterdir()))} wavs "
        f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
