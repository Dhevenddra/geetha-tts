r"""M2: can a breathing training voice generate ONLY the pauses inside anitha's speech?

What 14 Nov's control showed (runs\m2_breath_control, scored with the D-030 instrument):
  * the ivr training voices make breath-like sound in their pauses at S3 - with OR
    without <|nv_breath|> (21 / 10 / 23 pauses over threshold in 9 clips per arm);
  * anitha never does (0 in 18 clips, max score 0.31), under every decode-time arm.
So the sound is in the model and it rides on the SPEAKER label, not on the tag. The
arms that failed borrowed the tag direction, which is near zero.

This run hands the pen over. Two prompts are run beside each other on ONE audio
history: MAIN (anitha / arun) and DONOR (an ivr speaker, breath tag in the text). MAIN
writes the speech. When the frame MAIN has just written is silent - measured, by
decoding it - the DONOR writes the next frames; it hands back when its frame is silent
again (it had nothing to add), when it gets loud (it started to talk), or after
MAX_DONOR frames. One token stream, one codec decode: no waveform is pasted.

  P0   main only, no tag                       (reference)
  P1   main only, tag                          (reference: the tag alone)
  H1   hand-over, donor = Stage B adapter, ivr_d821efa9
  H2   hand-over, donor = Stage A adapter, ivr_d821efa9   (Stage A last saw breaths)
  H3   H2, and the donor also opens the clip   (before the first word)
  H4   H2 with donor ivr_f251dc9c
  CTL  Stage A adapter alone, two ivr voices, tag / no tag: did Stage B erase a tag
       that Stage A obeyed?

Every clip is scored in the kernel by the breath instrument (detector weights are the
ones saved in work\breath_calibration\detector_v1.json; held-out AUC 0.8812). Limits:
pauses are located by energy, and the detector's false-positive rate on synthetic audio
is not established - read H against P0/P1 of the same voice, never alone. Whether a
flagged pause SOUNDS like a breath, and whether the voice is still hers, is the
operator's ear. Audio is private to this kernel and never released (D-007).
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
ADAPTERS = {"stageB": "m2-main/stageB/final", "stageA": "m2-main/stageA/final"}
SAMPLE_RATE = 24_000
FRAME_TOKENS = 7
CODEBOOK = 4096
MAX_NEW_TOKENS = int(20.0 * 82)
HEAD = "<|start_of_human|><|begin_of_text|>"
TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"
SEED = 1415
TEMPERATURE, REPETITION, TOP_K, TOP_P = 0.9, 1.1, 50, 0.9
_MAX_MOUNT_DEPTH = 4
BREATH_TAG = "<|nv_breath|>"
IVR = dict(style="extempore", env="enhanced")

# Hand-over rule, fixed before the run. Levels are dBFS of one decoded frame (85 ms);
# anitha's speech sits near -19 dBFS and her pauses near -60.
SILENT_DB = -42.0     # MAIN's frame at or under this: the pause has begun, hand over
LOUD_DB = -30.0       # DONOR's frame over this: it is talking, hand back at once
REARM_DB = -35.0      # MAIN must speak this loudly again before the next hand-over
MIN_DONOR, MAX_DONOR = 2, 8   # frames
CONTEXT_FRAMES = 8    # frames decoded to measure the newest one

DETECTOR = {
    "feature_names": ["e_max", "e_mean", "low", "mid", "high", "zcr", "flat", "cent"],
    "weights": [0.2788811410630346, -0.985537779138968, -0.9842160547894164,
                -0.8744290396775748, 1.3349101853970333, 0.4548834928326557,
                0.1032392455902468, -1.7393024084874105],
    "bias": -1.721060729028366,
    "mean": [43.47422071475888, 27.99607551836256, 0.40890352063203017,
             0.4224787964451993, 0.14780841055672947, 0.13127883246066557,
             0.0326270385950298, 943.4974767125068],
    "deviation": [22.4972705189227, 24.863487319396114, 0.2519033539223044,
                  0.24849028904344778, 0.1910851476316463, 0.10888335598702682,
                  0.05424661213770078, 831.0223924990869],
}
THRESHOLD = 0.4313014771032603   # precision 0.80 / recall 0.57 on held-out speakers

# Written for these runs; not training text, not eval-vault items.
TEXTS = [
    "நேத்து ராத்திரி ரொம்ப நேரம் வேலை பாத்துட்டு இருந்தேன் <|nv_breath|> அதனால காலையில எந்திரிக்கவே முடியல <|nv_breath|> ஆபீஸுக்கு லேட்டா தான் போனேன்",
    "எங்க ஊர்ல மழை பெய்ய ஆரம்பிச்சா <|nv_breath|> ரெண்டு மூணு நாளைக்கு நிக்கவே நிக்காது <|nv_breath|> அப்போ எல்லாரும் வீட்டுக்குள்ளயே இருப்பாங்க",
    "<|nv_breath|> சரி நான் ஒண்ணு சொல்றேன் கவனமா கேளு <|nv_breath|> இந்த விஷயத்த இப்போதைக்கு யார்கிட்டயும் சொல்ல வேண்டாம்",
    "இன்னைக்கு காலையில பஸ்ல வரும்போது ஒரு பழைய நண்பன பாத்தேன் <|nv_breath|> ரொம்ப வருஷம் கழிச்சு பேசினோம் <|nv_breath|> மனசுக்கு ரொம்ப சந்தோஷமா இருந்துச்சு",
]
DONORS = ("ivr_d821efa9", "ivr_f251dc9c")


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
                    "snac==1.2.1", "soundfile==0.14.0"], check=True)
    log("installed snac 1.2.1, soundfile 0.14.0 (pinned)")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    for needed in ("eval/validity.py", "eval/breath_score.py", "audio/breath.py"):
        if not (source / needed).exists():
            raise SystemExit(f"geetha-code has no {needed}: re-push the code dataset")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def fetch_adapters(token: str) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(CKPT_REPO, token=token,
                                  allow_patterns=[f"{sub}/*" for sub in ADAPTERS.values()]))
    found = {}
    for name, sub in ADAPTERS.items():
        folder = root / sub
        if not (folder / "adapter_config.json").exists():
            nested = sorted(folder.rglob("adapter_config.json"))
            if not nested:
                raise SystemExit(f"no adapter_config.json under {sub}")
            folder = nested[0].parent
        found[name] = folder
        log(f"  adapter {name}: {folder}")
    return found


def build_tts(token: str, adapters: dict[str, Path]):
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
    tts.lm = PeftModel.from_pretrained(tts.lm, str(adapters[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        tts.lm.load_adapter(str(adapters[name]), adapter_name=name)
    after = sum(p.numel() for p in tts.lm.parameters())
    if after <= before:
        raise SystemExit("attaching the adapters added no parameters")
    tts.lm.eval()  # NOT merged: merging changes the function on this tied model (D-049)
    log(f"  adapters attached, unmerged ({before:,} -> {after:,} parameters)")
    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        log("  LM in fp16 (no native bf16 on this card)")
    return tts


def main() -> int:
    fix_environment()
    import soundfile as sf
    import torch

    token = require_token()
    ensure_dependencies()
    import_geetha()
    from geetha.audio.breath import BreathDetector
    from geetha.eval.breath_score import score_wave
    from geetha.eval.validity import check_clip

    detector = BreathDetector.from_dict(DETECTOR)
    tts = build_tts(token, fetch_adapters(token))
    from inference import ids_to_codes  # type: ignore

    tok = tts.tok
    base = tts._base
    start_speech = tok.convert_tokens_to_ids("<|start_of_speech|>")
    stop_ids = sorted({tts._eos, tok.eos_token_id})
    active = {"adapter": None}
    log(f"  GPU: {torch.cuda.get_device_name(0)}")

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def prompt(text: str, speaker: str, style: str = "CONV", env: str = "studio") -> list[int]:
        return (enc(HEAD) + enc(f"<|speaker>{speaker}<speaker|>\n")
                + enc(f"<|style>{style}<style|>\n") + enc(f"<|env>{env}<env|>\n")
                + enc(text) + enc(TAIL) + [start_speech])

    def strip(text: str) -> str:
        return " ".join(text.replace(BREATH_TAG, " ").split())

    def decode(tokens: list[int]) -> np.ndarray:
        try:
            with torch.no_grad():
                codes = ids_to_codes(tokens, base, tts.device)
                wav = tts.vocos(tts.snac.quantizer.from_codes(codes).float())
            return wav[0, 0].clamp(-1, 1).float().cpu().numpy().astype(np.float32).reshape(-1)
        except ValueError:
            return np.zeros(0, dtype=np.float32)

    def newest_frame_db(tokens: list[int]) -> float:
        """dBFS of the frame just completed, decoded with a little left context."""
        frames = min(CONTEXT_FRAMES, len(tokens) // FRAME_TOKENS)
        wav = decode(tokens[-frames * FRAME_TOKENS:])
        if not len(wav):
            return 0.0
        tail = wav[-(len(wav) // frames):]
        return float(20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-7))

    class Branch:
        """One prompt with its own cache; every branch is fed the same audio tokens."""

        def __init__(self, ids: list[int], adapter: str):
            self.ids, self.adapter = list(ids), adapter
            self.past = None

        def step(self, new_token: int | None) -> "torch.Tensor":
            if active["adapter"] != self.adapter:   # switching walks every LoRA layer
                tts.lm.set_adapter(self.adapter)
                active["adapter"] = self.adapter
            if new_token is None:
                feed = torch.tensor([self.ids], device=tts.device)
            else:
                self.ids.append(new_token)
                feed = torch.tensor([[new_token]], device=tts.device)
            out = tts.lm(input_ids=feed, past_key_values=self.past, use_cache=True)
            self.past = out.past_key_values
            return out.logits[0, -1].float()

    def sample(logits, seen: list[int], position: int, may_stop: bool) -> int:
        """Repetition penalty, the frame grammar, temperature, top-k, top-p, one draw."""
        ids = torch.tensor(sorted(set(seen)), device=logits.device)
        picked = logits[ids]
        logits = logits.clone()
        logits[ids] = torch.where(picked < 0, picked * REPETITION, picked / REPETITION)
        allowed = torch.full_like(logits, float("-inf"))
        low = base + position * CODEBOOK
        allowed[low:low + CODEBOOK] = 0.0
        if may_stop:
            allowed[stop_ids] = 0.0
        logits = (logits + allowed) / TEMPERATURE
        top = torch.topk(logits, TOP_K)
        probs = torch.softmax(top.values, dim=-1)
        keep = torch.cumsum(probs, dim=-1) - probs < TOP_P
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        return int(top.indices[torch.multinomial(probs / probs.sum(), 1)])

    def generate(main_ids, main_adapter, seed, donor_ids=None, donor_adapter=None,
                 lead=False):
        """Returns (tokens, ended, per-frame record). No donor: plain sampling."""
        torch.manual_seed(seed)
        main = Branch(main_ids, main_adapter)
        donor = Branch(donor_ids, donor_adapter) if donor_ids is not None else None
        tokens: list[int] = []
        frames: list[dict] = []          # one row per completed frame
        mode = "donor" if (donor is not None and lead) else "main"
        donor_run, armed, ended = 0, True, False
        with torch.no_grad():
            logits = {"main": main.step(None)}
            if donor is not None:
                logits["donor"] = donor.step(None)
            while len(tokens) < MAX_NEW_TOKENS:
                position = len(tokens) % FRAME_TOKENS
                writer = main if mode == "main" else donor
                token = sample(logits[mode], writer.ids, position,
                               may_stop=(mode == "main" and position == 0 and len(tokens) > 0))
                if token in stop_ids:
                    ended = True
                    break
                tokens.append(token)
                logits["main"] = main.step(token)
                if donor is not None:
                    logits["donor"] = donor.step(token)
                if len(tokens) % FRAME_TOKENS:
                    continue
                if donor is None:
                    frames.append({"by": "main"})
                    continue
                level = newest_frame_db(tokens)
                frames.append({"by": mode, "db": round(level, 1)})
                if mode == "main":
                    if level > REARM_DB:
                        armed = True
                    elif level <= SILENT_DB and armed and len(frames) >= 4:
                        mode, donor_run, armed = "donor", 0, False
                else:
                    donor_run += 1
                    if (level > LOUD_DB or donor_run >= MAX_DONOR
                            or (level <= SILENT_DB and donor_run >= MIN_DONOR)):
                        mode, armed = "main", False
        return tokens, ended, frames

    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()

    def flush() -> None:
        (OUT_DIR / "m2_breath_handover.json").write_text(
            json.dumps({"decoding": {"temperature": TEMPERATURE, "repetition": REPETITION,
                                     "top_k": TOP_K, "top_p": TOP_P, "seed": SEED},
                        "rule": {"silent_db": SILENT_DB, "loud_db": LOUD_DB,
                                 "rearm_db": REARM_DB, "min_donor": MIN_DONOR,
                                 "max_donor": MAX_DONOR},
                        "detector_threshold": THRESHOLD, "rows": rows},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    def render(file: str, seed: int, main_ids, main_adapter="stageB", **kw) -> None:
        meta = {k: kw.pop(k) for k in ("arm", "voice", "text")}
        tokens, ended, frames = generate(main_ids, main_adapter, seed, **kw)
        wav = decode(tokens)
        check = check_clip(wav, SAMPLE_RATE, ended_itself=ended)
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        row = {"file": file, **meta, "valid": check.valid, "reason": check.reason,
               "seconds": round(len(wav) / SAMPLE_RATE, 2)}
        frame_s = (len(wav) / SAMPLE_RATE / len(frames)) if frames else 0.0
        spans, run = [], None
        for index, frame in enumerate(frames + [{"by": "main"}]):
            if frame["by"] == "donor" and run is None:
                run = index
            elif frame["by"] != "donor" and run is not None:
                levels = [f["db"] for f in frames[run:index]]
                spans.append({"from_s": round(run * frame_s, 2), "to_s": round(index * frame_s, 2),
                              "frames": index - run, "max_db": max(levels), "min_db": min(levels)})
                run = None
        row["donor_spans"] = spans
        row["donor_frames"] = sum(s["frames"] for s in spans)
        if check.valid:
            scored = score_wave(wav, SAMPLE_RATE, detector, THRESHOLD)
            hits = [p for p in scored["pauses"] if p["breath"]]
            in_span = sum(any(s["from_s"] - 0.1 <= p["at_s"] + p["dur_s"] / 2 <= s["to_s"] + 0.1
                              for s in spans) for p in hits)
            row.update(n_pauses=len(scored["pauses"]), breaths=len(hits),
                       breaths_in_donor_spans=in_span, max_score=scored["max_score"],
                       pauses=scored["pauses"])
        rows.append(row)
        log(f"  {file}: valid={check.valid} {row['seconds']} s  breaths={row.get('breaths')} "
            f"(in donor spans {row.get('breaths_in_donor_spans')})  max score "
            f"{row.get('max_score')}  donor spans "
            f"{[(s['from_s'], s['frames'], s['max_db']) for s in spans]}"
            f"   [{time.time() - started:.0f} s]")
        flush()

    log("=== 1. CTL: Stage A alone in two ivr voices - does IT obey the tag? ===")
    for d_index, speaker in enumerate(DONORS):
        for t_index in (0, 1):
            seed = SEED + 10 * d_index + t_index
            meta = dict(arm="CTL", voice=speaker, text=f"t{t_index}")
            render(f"ctlA_{speaker}_t{t_index}_notag.wav", seed,
                   prompt(strip(TEXTS[t_index]), speaker, **IVR), "stageA", **meta)
            meta = dict(arm="CTLtag", voice=speaker, text=f"t{t_index}")
            render(f"ctlA_{speaker}_t{t_index}_tag.wav", seed,
                   prompt(TEXTS[t_index], speaker, **IVR), "stageA", **meta)

    log("")
    log("=== 2. anitha and arun: references, then the hand-over ===")
    for v_index, voice in enumerate(("anitha", "arun")):
        for t_index, marked in enumerate(TEXTS):
            seed = SEED + 100 + 10 * v_index + t_index
            with_tag, without = prompt(marked, voice), prompt(strip(marked), voice)
            name = f"{voice}_t{t_index}"
            meta = dict(voice=voice, text=f"t{t_index}")
            render(f"{name}_P0_notag.wav", seed, without, arm="P0", **meta)
            render(f"{name}_P1_tag.wav", seed, with_tag, arm="P1", **meta)
            donor_d = prompt(marked, DONORS[0], **IVR)
            donor_f = prompt(marked, DONORS[1], **IVR)
            render(f"{name}_H1_donorB.wav", seed, with_tag, arm="H1", donor_ids=donor_d,
                   donor_adapter="stageB", **meta)
            render(f"{name}_H2_donorA.wav", seed, with_tag, arm="H2", donor_ids=donor_d,
                   donor_adapter="stageA", **meta)
            lead_text = marked if marked.startswith(BREATH_TAG) else f"{BREATH_TAG} {marked}"
            render(f"{name}_H3_donorA_lead.wav", seed, with_tag, arm="H3",
                   donor_ids=prompt(lead_text, DONORS[0], **IVR), donor_adapter="stageA",
                   lead=True, **meta)
            render(f"{name}_H4_donorA_f251.wav", seed, with_tag, arm="H4", donor_ids=donor_f,
                   donor_adapter="stageA", **meta)

    log("")
    log("=== summary: pauses over the breath threshold, by the D-030 instrument ===")
    for voice in sorted({r["voice"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["voice"] == voice}):
            group = [r for r in rows if r["voice"] == voice and r["arm"] == arm]
            good = [r for r in group if r["valid"]]
            log(f"  {voice:14s} {arm:7s} valid {len(good)}/{len(group)}  "
                f"breaths {sum(r['breaths'] for r in good)} "
                f"(in donor spans {sum(r['breaths_in_donor_spans'] for r in good)})  "
                f"clips with one {sum(r['breaths'] > 0 for r in good)}  "
                f"donor frames {sum(r['donor_frames'] for r in good)}  "
                f"mean seconds {np.mean([r['seconds'] for r in good]) if good else 0:.1f}")
    bad = [r["file"] for r in rows if not r["valid"]]
    log(f"invalid clips: {len(bad)}/{len(rows)} {bad}")
    log(f"wrote m2_breath_handover.json and {len(list(audio.iterdir()))} wavs "
        f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
