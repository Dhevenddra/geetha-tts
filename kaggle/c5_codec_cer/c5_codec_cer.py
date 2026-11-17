r"""C5 - how much intelligibility does SNAC itself cost? (D-040)

Why this run exists
-------------------
The operator listened to 20 round-tripped clips and judged **8 of 20 unintelligible**:
"gibberish", "sounds like a drunkard", "warble". These are not generations -- the model
has not trained -- they are our codec's reconstruction of real recordings. So they are
the **ceiling** on how good Geetha can ever sound.

Two hypotheses were on the table and one is already dead. Clip length does not predict
his verdicts (r = 0.012 against duration; the best threshold still misses 7 of 20), and
neither does the spectral measure I built (r = 0.013). What survives is his own
phonological reading: `மாறிக்கிட்டு` came back as "mahikitu" with the `ற` gone, and he
points at the Vallinam / Mellinam / Idaiyinam distinctions Tamil is built on. A codec
that smears retroflex and dental consonants would do exactly that, and no broad spectral
statistic would notice.

**CER is the instrument that can see it.** Not because it is clever, but because a lost
consonant is a character error and nothing else here counts characters.

What is measured
----------------
The same clips, twice, through the same ASR judge:

    A = CER(judge(original 24 kHz audio),      reference transcript)
    B = CER(judge(SNAC round-tripped audio),   reference transcript)

**B - A is the codec's own cost**, and it is the number that matters. A on its own is
the judge's error on real speech and says nothing about SNAC; reporting B alone would
blame the codec for whatever the judge cannot transcribe anyway. The delta cancels that.

Per-clip deltas are kept, not just the mean, because the operator's verdicts were not
uniform -- 12 of 20 were fine. If the damage is concentrated in a minority of clips, the
fix is a data filter; if it is spread evenly, it is a hard ceiling on the release and
every published CER has to be reported against it.

This run spends no money: free Kaggle GPU, no gate, no pod.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

OUT = Path("/kaggle/working")
CLIPS = 300
SNAC_REPO = "hubertsiuzdak/snac_24khz"
CODES_REPO = "Dhevenddra/geetha-tts-codes-stagea-private"
# Pinned: the Vocos decoder ships as importable Python in the model repo, so the
# revision is fixed rather than tracking `main`.
MODEL_REVISION = "76e0814f189321efe550011850cd288dae366aad"


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


def import_geetha_package() -> Path:
    import shutil

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


def install_dependencies() -> None:
    """Kaggle's image has torch, transformers and datasets but not snac.

    Carried over from B10, which needed the same treatment. Version 1 of this kernel
    died on `ModuleNotFoundError: No module named 'snac'` after loading nothing -- a
    miss I should have anticipated from B10's own failures.
    """
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "snac", "soundfile", "datasets"], check=True)
    log("installed snac, soundfile, datasets")


def main() -> int:
    install_dependencies()
    package = import_geetha_package()
    sys.path.insert(0, str(package.parent))
    token = find_input("hf_read_token.txt").read_text(encoding="utf-8").strip()
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import load_dataset
    from snac import SNAC
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    from geetha.audio.snac_input import SNAC_SAMPLE_RATE, prepare_for_snac
    from geetha.audio.snac_tokens import drop_duplicate_frames, flatten_frames, unflatten_frames
    from geetha.scoring.metrics import cer  # the project's own scorer
    from geetha.text.normalise import norm_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = SNAC.from_pretrained(SNAC_REPO).eval().to(device)

    # Vocos is the production decoder (inference.py: `dec = self.snac.decoder if stock
    # else self.vocos`, stock defaulting to False). v3 of this run measured the stock
    # decoder only, which is the A/B arm -- see D-043. Both are measured now.
    # This imports Python written by the model's authors, which is a step beyond
    # loading their weights, so it is pinned to one immutable commit rather than to
    # whatever `main` holds when the job runs. The revision is the snapshot every other
    # read of this repo in the project already resolved to.
    from huggingface_hub import hf_hub_download as _hf

    checkpoint = Path(_hf("bodhan-ai/indic-speak", "vocos/best.pt",
                          revision=MODEL_REVISION, token=token))
    for extra in ("vocos/load.py", "vocos/model.py", "vocos/__init__.py", "vocos/config.yaml"):
        _hf("bodhan-ai/indic-speak", extra, revision=MODEL_REVISION, token=token)
    # Appended, not prepended: a downloaded directory at sys.path[0] would shadow the
    # standard library and our own modules by name.
    sys.path.append(str(checkpoint.parent.parent))
    from vocos.load import load_vocos

    vocos = load_vocos(str(checkpoint), device)

    # IV-R is public; pulling it here avoids shipping audio to Kaggle ourselves.
    stream = load_dataset("ai4bharat/indicvoices_r", "Tamil", split="train",
                          streaming=True, token=token)

    judge_id = "openai/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(judge_id)
    judge = AutoModelForSpeechSeq2Seq.from_pretrained(
        judge_id, torch_dtype=torch.float16).to(device).eval()

    JUDGE_RATE = 16_000

    def transcribe(wave: np.ndarray) -> str:
        # Whisper refuses anything but 16 kHz, and our audio is 24 kHz (v2 died here).
        # torchaudio handles the 3:2 ratio properly; both arms get the identical
        # treatment, so this cannot bias the delta.
        import torchaudio

        resampled = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wave)), SNAC_SAMPLE_RATE, JUDGE_RATE
        ).numpy()
        features = processor(resampled, sampling_rate=JUDGE_RATE,
                             return_tensors="pt").input_features.to(device, torch.float16)
        with torch.inference_mode():
            ids = judge.generate(features, language="ta", task="transcribe", max_new_tokens=200)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    rows = []
    for index, record in enumerate(stream):
        if len(rows) >= CLIPS:
            break
        reference = record.get("text") or ""
        if not norm_text(reference):
            continue
        audio = record["audio"]
        wave = np.asarray(audio["array"], dtype=np.float32)
        rate = int(audio["sampling_rate"])
        if rate % SNAC_SAMPLE_RATE:
            continue  # D-038: only integer ratios; 22,050 Hz clips are excluded
        original = prepare_for_snac(wave, rate, gain_db=0.0)
        if len(original) < SNAC_SAMPLE_RATE // 2:
            continue

        with torch.inference_mode():
            codes = codec.encode(torch.from_numpy(original).to(device).reshape(1, 1, -1))
        books = [code.reshape(-1).tolist() for code in codes]
        c0, c1, c2 = drop_duplicate_frames(*books)
        # Round-trip through our own flatten/unflatten, so this measures the codec as
        # the pipeline actually uses it, not the codec in isolation.
        recovered = unflatten_frames(flatten_frames(c0, c1, c2))
        with torch.inference_mode():
            tensors = [torch.tensor(b, dtype=torch.long, device=device).reshape(1, -1)
                       for b in recovered]
            z_q = codec.quantizer.from_codes(tensors)
            through = codec.decoder(z_q).reshape(-1).cpu().numpy()
            through_vocos = vocos(z_q.float()).reshape(-1).cpu().numpy()

        # cer(hypothesis, reference) -> ErrorCounts; .rate is None for an empty
        # reference, which is why that case is skipped rather than scored as 0.0.
        target = norm_text(reference)
        a = cer(norm_text(transcribe(original)), target).rate
        b = cer(norm_text(transcribe(through)), target).rate
        v = cer(norm_text(transcribe(through_vocos)), target).rate
        if a is None or b is None or v is None:
            continue
        rows.append(dict(index=index, seconds=len(original) / SNAC_SAMPLE_RATE,
                         reference=reference, cer_original=a, cer_roundtrip=b,
                         cer_vocos=v, delta=b - a, delta_vocos=v - a))
        if len(rows) % 25 == 0:
            deltas = np.array([r["delta"] for r in rows])
            log(f"  {len(rows)} clips | mean delta {deltas.mean():+.4f}")

    deltas = np.array([r["delta"] for r in rows])
    deltas_v = np.array([r["delta_vocos"] for r in rows])
    a_all = np.array([r["cer_original"] for r in rows])
    b_all = np.array([r["cer_roundtrip"] for r in rows])
    v_all = np.array([r["cer_vocos"] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])

    # Bootstrap the mean delta, because a single mean over 300 clips with a long tail is
    # not a result anyone should act on without an interval.
    rng = np.random.default_rng(1234)
    def interval(values):
        draws = np.array([rng.choice(values, values.size, replace=True).mean()
                          for _ in range(2_000)])
        return np.percentile(draws, [2.5, 97.5])

    low, high = interval(deltas)
    low_v, high_v = interval(deltas_v)
    # Paired: the same clip under both decoders, so this is the decoder difference
    # with every other source of variation held fixed.
    paired = deltas - deltas_v
    low_p, high_p = interval(paired)

    badly_hurt = int((deltas_v > 0.10).sum())
    summary = dict(
        clips=len(rows),
        cer_original_mean=float(a_all.mean()), cer_roundtrip_mean=float(b_all.mean()),
        delta_mean=float(deltas.mean()), delta_ci=[float(low), float(high)],
        cer_vocos_mean=float(v_all.mean()),
        delta_vocos_mean=float(deltas_v.mean()), delta_vocos_ci=[float(low_v), float(high_v)],
        paired_stock_minus_vocos=float(paired.mean()),
        paired_ci=[float(low_p), float(high_p)],
        delta_median=float(np.median(deltas_v)),
        clips_over_10pt=badly_hurt,
        delta_vs_duration_r=float(np.corrcoef(seconds, deltas_v)[0, 1]),
    )
    (OUT / "codec_cer.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=1, ensure_ascii=False),
        encoding="utf-8")

    concentrated = badly_hurt / max(len(rows), 1) < 0.25
    (OUT / "codec_cer.md").write_text(f"""# C5 - the codec's own intelligibility cost

{len(rows)} IV-R clips, each transcribed twice by `{judge_id}` and scored with the
project's CER against the reference.

| | CER |
| --- | ---: |
| original audio (judge's own error) | **{a_all.mean():.4f}** |
| after round-trip, **SNAC stock decoder** (A/B arm) | **{b_all.mean():.4f}** |
| after round-trip, **Vocos** (the production decoder) | **{v_all.mean():.4f}** |
| stock decoder cost | **{deltas.mean():+.4f}**, 95% CI [{low:+.4f}, {high:+.4f}] |
| **Vocos cost — the real ceiling** | **{deltas_v.mean():+.4f}**, 95% CI [{low_v:+.4f}, {high_v:+.4f}] |
| paired difference (stock − Vocos) | **{paired.mean():+.4f}**, 95% CI [{low_p:+.4f}, {high_p:+.4f}] |
| median Vocos delta | {np.median(deltas_v):+.4f} |
| clips losing more than 10 CER points | **{badly_hurt} of {len(rows)}** ({badly_hurt / max(len(rows), 1):.1%}) |
| delta vs clip duration | r = {summary['delta_vs_duration_r']:+.3f} |

**Read the delta, not the second row.** The judge makes its own errors on real Tamil
speech; the first row is that error and has nothing to do with SNAC. Only the difference
is the codec's cost.

**Shape of the damage: {"concentrated in a minority of clips" if concentrated else "spread across the corpus"}.**
{"Fewer than a quarter of clips take a large hit, so a data filter is worth considering." if concentrated else "The loss is broad, so it is a ceiling on the release rather than a tail to filter: every published Geetha CER must be reported against it."}

The duration correlation is reported because the operator's first hypothesis was that
short clips fare worse. On his 20 clips that was r = 0.012 against his own verdicts;
this is the same question asked of 300 clips with an objective measure.

Judge caveat: `{judge_id}` is **not** one of the two frozen eval judges, and its Tamil is
mediocre. That is acceptable *here and only here*, because both arms use the same judge
on the same clips, so its weakness cancels in the delta. It must not be reused for any
published Geetha number.
""", encoding="utf-8")

    log(f"\noriginal {a_all.mean():.4f} -> round-trip {b_all.mean():.4f}; "
        f"delta {deltas.mean():+.4f} [{low:+.4f}, {high:+.4f}]")
    log(f"clips losing >10 CER points: {badly_hurt}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
