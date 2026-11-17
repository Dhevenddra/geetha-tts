r"""M2 Stage B: does the model breathe at all, and can decoding bind it to our voices?

The 11 Nov positive control was invalid on 8 of 9 pairs. Cause, read off the Stage A
manifest on 14 Nov: the control speaker's training rows are 10-42 words (median 21, 8.5 s)
and the control asked her for 3-6 word sentences at temperature 0.6 with no repetition
penalty, so the model never found an end and ran to the token cap. The speaker label,
style and env were right.

This run asks the question properly.

  CONTROL  three IndicVoices training speakers (the three with the most breath marks),
           two of their own training sentences and one new one, all of training length:
             without the tag / with the tag / with the tag and guidance 3.5 on it
  BIND     anitha and arun, three new sentences of the same length:
             P0 no tag                      (reference)
             P1 tag                         (what 11 Nov heard as a pause)
             P2 tag + guidance 3.5 on it
             P3 tag, style extempore + env enhanced (the register the breaths live in)
             P4 tag + the breath direction borrowed from an IndicVoices speaker, w=2
             P5 the same, w=4
           P4/P5 add  w * (log p[ivr speaker, tag] - log p[ivr speaker, no tag])  to
           anitha's own logits at every step. All branches see the same audio history,
           so the borrowed term is near zero while she is speaking and only pushes where
           the tag would make the ivr voice breathe. One token stream, one decode.

Control audio IS written this time: the operator has to hear whether the ivr voice
breathes. It is synthetic, private to this kernel, for his ears only, and never released
(D-007). A pause-noise statistic is written beside every clip; it is a ranker, not a
breath detector.
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
ADAPTER = "m2-main/stageB/final"
SAMPLE_RATE = 24_000
MAX_NEW_TOKENS = int(20.0 * 82)
HEAD = "<|start_of_human|><|begin_of_text|>"
TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"
SEED = 1411
TEMPERATURE, REPETITION = 0.9, 1.1
TAG_GUIDANCE = 3.5
BORROW_WEIGHTS = {"P4": 2.0, "P5": 4.0}
_MAX_MOUNT_DEPTH = 4
BREATH_TAG = "<|nv_breath|>"
IVR = dict(style="extempore", env="enhanced")
BORROW_FROM = "ivr_f251dc9c"

# Real Stage A training text of each speaker (so the control is in distribution).
CONTROL_OWN = {
    "ivr_f251dc9c": [
        "ஸோ சுற்றுலாப் பயணிகள் மூலியமா எப்பிடி வேலை வாய்ப்பு இருக்கு அப்டின்னா <|nv_breath|> ஆ இப்போ சுற்றுலாத்தலம் அப்டின்னா எங்க ஊர்ல பெருசா இல்ல",
        "அது சந்துல மட்டும் தான் விளையாடணும் அதுனால நெறைய ஏரியா கிடைக்காது <|nv_breath|> ஏரியா இருக்குனா அந்த வெளியில விளையாடுற விளையாட்டுக தான் <|nv_breath|> அது போக என்னென்ன பண்ணுவாங்க அப்டினா",
    ],
    "ivr_3b4fec4b": [
        "ரொம்ப ரொம்ப ரொம்ப ரொம்ப இம்பார்டன்ஸ் வந்து நர்ஸ்தான் <|nv_breath|> டாக்டர்ஸ் சொலியூஷன் சொன்னதும் நர்ஸ்தான் அத வந்து ஆக்ட்டிவேட் பண்வாங்க",
        "அவனையும் நல்லா படிக்க வைப்பேன் பொண்ணு வந்தா <|nv_breath|> அவளுக்கு எவ்ளோ படிக்க தோணுதோ அவ்ளவும் படிக்க வைப்பேன் <|nv_breath|> பெண்கல்வி வே பெண்கல்வி வந்து யாரும் வேண்டா சொன்னாலும்",
    ],
    "ivr_d821efa9": [
        "தன் <|nv_breath|> நியாயமான வழியில வந்து போராடி <|nv_breath|> ஆ சுதந்திரம் வாங்கி குடுத்தாரு அந்த சுதந்திரம் வாங்கி குடுக்குறதுக்கு முன்னாடி அவரு பட்ட கஷ்டங்கள்",
        "ஆ ஆமா நீங்க ஃபோன் பண்ணிட்டு என்னனு பாத்துட்டு <|nv_breath|> ஏன் ஃபோன் அட்டன் பண்ணலங்கிறதையும் கேட்டுட்டு எனக்கு நீங்க <|nv_breath|> கால் பண்ணுனீங்கனா கொஞ்சோம் வசதியா இருக்கும்",
    ],
}
# Written for this run; not training text, not eval-vault items.
NEW = [
    "நேத்து ராத்திரி ரொம்ப நேரம் வேலை பாத்துட்டு இருந்தேன் <|nv_breath|> அதனால காலையில எந்திரிக்கவே முடியல <|nv_breath|> ஆபீஸுக்கு லேட்டா தான் போனேன்",
    "எங்க ஊர்ல மழை பெய்ய ஆரம்பிச்சா <|nv_breath|> ரெண்டு மூணு நாளைக்கு நிக்கவே நிக்காது <|nv_breath|> அப்போ எல்லாரும் வீட்டுக்குள்ளயே இருப்பாங்க",
    "<|nv_breath|> சரி நான் ஒண்ணு சொல்றேன் கவனமா கேளு <|nv_breath|> இந்த விஷயத்த இப்போதைக்கு யார்கிட்டயும் சொல்ல வேண்டாம்",
]


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
    if not (source / "eval" / "validity.py").exists():
        raise SystemExit("geetha-code has no eval/validity.py: re-push the code dataset")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def fetch_adapter(token: str) -> Path:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(CKPT_REPO, allow_patterns=[f"{ADAPTER}/*"], token=token))
    folder = root / ADAPTER
    if not (folder / "adapter_config.json").exists():
        nested = sorted(folder.rglob("adapter_config.json"))
        if not nested:
            raise SystemExit(f"no adapter_config.json under {ADAPTER}")
        folder = nested[0].parent
    log(f"  adapter: {folder}")
    return folder


def build_tts(token: str, adapter: Path):
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
    tts.lm = PeftModel.from_pretrained(tts.lm, str(adapter), adapter_name="stageB")
    after = sum(p.numel() for p in tts.lm.parameters())
    if after <= before:
        raise SystemExit("attaching the adapter added no parameters")
    tts.lm.eval()  # NOT merged: merging changes the function on this tied model (D-049)
    log(f"  adapter attached, unmerged ({before:,} -> {after:,} parameters)")
    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        log("  LM in fp16 (no native bf16 on this card)")
    return tts


def pause_noise(wav: np.ndarray) -> dict:
    """How loud are the pauses, against the quietest the clip gets? A ranker only.

    A pause filled with silence sits on the clip's floor; one filled with a breath sits
    well above it. Fricatives and room tone also lift it, so compare within a pair.
    """
    frame, hop = int(0.020 * SAMPLE_RATE), int(0.010 * SAMPLE_RATE)
    if len(wav) < 10 * frame:
        return {}
    count = 1 + (len(wav) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    level = 20 * np.log10(np.sqrt(np.mean(wav[index] ** 2, axis=1)) + 1e-7)
    speech, floor = np.percentile(level, 90), np.percentile(level, 5)
    quiet = level < speech - 18
    gaps, start = [], None
    for i, flag in enumerate(list(quiet) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 12:  # 120 ms
                gaps.append({"at_s": round(start * 0.01, 2),
                             "dur_s": round((i - start) * 0.01, 2),
                             "lift_db": round(float(np.median(level[start:i]) - floor), 1)})
            start = None
    return {"speech_db": round(float(speech), 1), "floor_db": round(float(floor), 1),
            "gaps": gaps, "n_gaps": len(gaps),
            "gap_seconds": round(sum(g["dur_s"] for g in gaps), 2),
            "max_lift_db": max((g["lift_db"] for g in gaps), default=None)}


def main() -> int:
    fix_environment()
    import soundfile as sf
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    token = require_token()
    ensure_dependencies()
    import_geetha()
    from geetha.eval.validity import check_clip

    tts = build_tts(token, fetch_adapter(token))
    from inference import ids_to_codes  # type: ignore

    tok = tts.tok
    start_speech = tok.convert_tokens_to_ids("<|start_of_speech|>")
    stop_ids = {tts._eos, tok.eos_token_id}
    log(f"  GPU: {torch.cuda.get_device_name(0)}")

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def prompt(text: str, speaker: str, style: str = "CONV", env: str = "studio") -> list[int]:
        return (enc(HEAD) + enc(f"<|speaker>{speaker}<speaker|>\n")
                + enc(f"<|style>{style}<style|>\n") + enc(f"<|env>{env}<env|>\n")
                + enc(text) + enc(TAIL) + [start_speech])

    def strip(text: str) -> str:
        return " ".join(text.replace(BREATH_TAG, " ").split())

    class Branch:
        """A second prompt run beside the main one on the same audio, with its own cache."""

        def __init__(self, ids: list[int]):
            self.ids = torch.tensor([ids], device=tts.device)
            self.past, self.started = None, False

        def step(self, input_ids):
            feed = input_ids[:, -1:] if self.started else self.ids
            out = tts.lm(input_ids=feed, past_key_values=self.past, use_cache=True)
            self.past, self.started = out.past_key_values, True
            return torch.log_softmax(out.logits[:, -1].float(), dim=-1)

    class Steer(LogitsProcessor):
        """main + sum of w * (log p[positive] - log p[negative]).

        positive=None means the main prompt itself, which makes one term with
        w = scale - 1 exactly classifier-free guidance at ``scale``.
        """

        def __init__(self, terms):
            self.terms = [(Branch(p) if p is not None else None, Branch(n), w)
                          for p, n, w in terms]

        def __call__(self, input_ids, scores):
            main = torch.log_softmax(scores.float(), dim=-1)
            out = main.clone()
            for positive, negative, weight in self.terms:
                pos = main if positive is None else positive.step(input_ids)
                out = out + weight * (pos - negative.step(input_ids))
            return out.to(scores.dtype)

    def generate(ids: list[int], terms, seed: int):
        torch.manual_seed(seed)
        x = torch.tensor([ids], device=tts.device)
        processors = LogitsProcessorList([Steer(terms)] if terms else [])
        with torch.no_grad():
            gen = tts.lm.generate(
                input_ids=x, attention_mask=torch.ones_like(x),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=TEMPERATURE, top_p=0.9, top_k=50,
                repetition_penalty=REPETITION,
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
    rows: list[dict] = []
    started = time.time()

    def flush() -> None:
        (OUT_DIR / "m2_breath_control.json").write_text(
            json.dumps({"decoding": {"temperature": TEMPERATURE, "repetition": REPETITION,
                                     "tag_guidance": TAG_GUIDANCE,
                                     "borrow_weights": BORROW_WEIGHTS,
                                     "borrow_from": BORROW_FROM},
                        "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    def render(file: str, ids: list[int], terms, seed: int, **meta) -> None:
        wav, ended = generate(ids, terms, seed)
        check = check_clip(wav, SAMPLE_RATE, ended_itself=ended)
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        row = {"file": file, **meta, "valid": check.valid, "reason": check.reason,
               "seconds": round(len(wav) / SAMPLE_RATE, 2),
               **(pause_noise(wav) if check.valid else {})}
        rows.append(row)
        log(f"  {file}: valid={check.valid} {row['seconds']} s  gaps={row.get('n_gaps')} "
            f"max lift={row.get('max_lift_db')} dB   [{time.time() - started:.0f} s]")
        flush()

    log("=== 1. CONTROL: do the IndicVoices training voices breathe? ===")
    for s_index, (speaker, own) in enumerate(CONTROL_OWN.items()):
        for t_index, marked in enumerate(own + [NEW[0]]):
            kind = f"own{t_index}" if t_index < len(own) else "new0"
            seed = SEED + 10 * s_index + t_index
            with_tag = prompt(marked, speaker, **IVR)
            without = prompt(strip(marked), speaker, **IVR)
            base = f"ctl_{speaker}_{kind}"
            meta = dict(arm="control", voice=speaker, text=kind)
            render(f"{base}_notag.wav", without, None, seed, variant="notag", **meta)
            render(f"{base}_tag.wav", with_tag, None, seed, variant="tag", **meta)
            render(f"{base}_tagG.wav", with_tag, [(None, without, TAG_GUIDANCE - 1)], seed,
                   variant="tag_guided", **meta)

    log("")
    log("=== 2. BIND: the same tag on anitha and arun, six ways ===")
    for v_index, voice in enumerate(("anitha", "arun")):
        for t_index, marked in enumerate(NEW):
            seed = SEED + 100 + 10 * v_index + t_index
            plain = strip(marked)
            with_tag, without = prompt(marked, voice), prompt(plain, voice)
            donor_pos = prompt(marked, BORROW_FROM, **IVR)
            donor_neg = prompt(plain, BORROW_FROM, **IVR)
            base = f"bind_{voice}_new{t_index}"
            meta = dict(arm="bind", voice=voice, text=f"new{t_index}")
            render(f"{base}_P0_notag.wav", without, None, seed, variant="P0", **meta)
            render(f"{base}_P1_tag.wav", with_tag, None, seed, variant="P1", **meta)
            render(f"{base}_P2_tagG.wav", with_tag, [(None, without, TAG_GUIDANCE - 1)],
                   seed, variant="P2", **meta)
            render(f"{base}_P3_register.wav", prompt(marked, voice, **IVR), None, seed,
                   variant="P3", **meta)
            for name, weight in BORROW_WEIGHTS.items():
                render(f"{base}_{name}_borrow{weight:g}.wav", with_tag,
                       [(donor_pos, donor_neg, weight)], seed, variant=name, **meta)

    log("")
    log("=== summary: loudest pause above the clip floor (dB), mean over valid clips ===")
    for arm in ("control", "bind"):
        variants = sorted({r["variant"] for r in rows if r["arm"] == arm})
        for variant in variants:
            group = [r for r in rows if r["arm"] == arm and r["variant"] == variant]
            good = [r for r in group if r["valid"]]
            lifts = [r["max_lift_db"] for r in good if r.get("max_lift_db") is not None]
            log(f"  {arm:8s} {variant:11s} valid {len(good)}/{len(group)}  "
                f"mean seconds {np.mean([r['seconds'] for r in good]) if good else 0:.1f}  "
                f"mean max lift {np.mean(lifts) if lifts else float('nan'):.1f} dB  "
                f"gap seconds {np.mean([r['gap_seconds'] for r in good]) if good else 0:.2f}")
    bad = [r["file"] for r in rows if not r["valid"]]
    log(f"invalid clips: {len(bad)}/{len(rows)} {bad}")
    log(f"wrote m2_breath_control.json and {len(list(audio.iterdir()))} wavs "
        f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
