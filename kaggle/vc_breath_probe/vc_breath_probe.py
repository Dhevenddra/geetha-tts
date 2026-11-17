r"""Does a breath survive voice conversion into Anitha's and Arun's voices? (free Kaggle GPU)

Measured 12 Nov: the Rasa studio recordings of the two voices we ship contain no real
breaths (operator's ears on the 20 strongest candidates out of 32,506 clips). Every breath
we own belongs to an IndicVoices speaker. This probe asks the one question that decides
whether voice conversion can move those breaths into the shipped voices:

    convert 12 breath-bearing conversational clips into each voice with Seed-VC
    (zero-shot, GPL-3.0 code, pinned commit) and check - by ear first, by number second -
    that (a) it sounds like Anitha / Arun and (b) the breath is still a breath.

Two arms per clip and voice:
    A  seed-uvit-whisper-small-wavenet, 22.05 kHz   the repo's offline default
    B  seed-uvit-whisper-base, 44.1 kHz, f0-conditioned, pitch auto-shifted to the target

The number: energy over the clip floor inside C4's breath gaps against the unmarked gaps
of the same clips, before and after conversion. Length is preserved (length-adjust 1.0),
so C4's timestamps still apply. It cannot tell a breath from noise; the ears can.

A listening test only. Nothing here is training data and nothing is published.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
CLIPS = OUT_DIR / "clips"
REPO = "https://github.com/Plachtaa/seed-vc.git"
COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
WORK = Path("/kaggle/temp/seed-vc")
DIFFUSION_STEPS = 30
ARMS = {"A": ["--f0-condition", "False"],
        "B": ["--f0-condition", "True", "--auto-f0-adjust", "True"]}
VOICES = ("anitha", "arun")
_MAX_MOUNT_DEPTH = 4
# pip names for modules whose import name differs
PIP_NAME = {"dac": "descript-audio-codec", "yaml": "pyyaml", "hydra": "hydra-core",
            "dotenv": "python-dotenv", "sklearn": "scikit-learn", "cv2": "opencv-python"}


def log(message: str) -> None:
    print(message, flush=True)


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def show_inputs() -> None:
    for mount in _directories(Path("/kaggle/input")):
        log(f"  input: {mount}")


def find_dataset(*markers: str) -> Path:
    """Address inputs by their contents, never by a vendor's path."""
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"could not find a mount containing {markers}")


def setup() -> None:
    WORK.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", REPO, str(WORK)], check=True)
    subprocess.run(["git", "-C", str(WORK), "checkout", "-q", COMMIT], check=True)
    # The repo pins torch 2.4.0 and numpy 1.26.4; reinstalling those in Kaggle's image is
    # the fragile path. Install only what is missing, and let convert() add the rest.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "munch", "einops",
                    "descript-audio-codec", "hydra-core", "pydub", "soundfile"], check=False)
    # Version 1 died on every clip: the vendored BigVGAN loader still declares and forwards
    # `proxies` and `resume_download`, which current huggingface_hub no longer passes or takes.
    patched = 0
    for source in WORK.rglob("*.py"):
        text = source.read_text(encoding="utf-8", errors="ignore")
        if "resume_download" not in text and "proxies" not in text:
            continue
        fixed = re.sub(r"^(\s*)proxies: Optional\[Dict\],\s*$", r"\1proxies: Optional[Dict] = None,",
                       text, flags=re.M)
        fixed = re.sub(r"^(\s*)resume_download: bool,\s*$", r"\1resume_download: bool = False,",
                       fixed, flags=re.M)
        fixed = re.sub(r"^\s*(proxies=proxies|resume_download=resume_download),\s*\n", "",
                       fixed, flags=re.M)
        if fixed != text:
            source.write_text(fixed, encoding="utf-8")
            patched += 1
            log(f"  patched {source.relative_to(WORK)}")
    log(f"seed-vc at {COMMIT[:10]}, {patched} file(s) patched for huggingface_hub")


def convert(source: Path, target: Path, arm: str, out_dir: Path) -> Path | None:
    """One conversion. A missing module is installed and the run retried."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "inference.py", "--source", str(source), "--target", str(target),
               "--output", str(out_dir), "--diffusion-steps", str(DIFFUSION_STEPS),
               "--length-adjust", "1.0", "--inference-cfg-rate", "0.7", *ARMS[arm]]
    for _ in range(10):
        result = subprocess.run(command, cwd=str(WORK), capture_output=True, text=True)
        if result.returncode == 0:
            made = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            return made[-1] if made else None
        missing = re.search(r"No module named '([\w\.]+)'", result.stderr)
        if not missing:
            log(f"  FAILED {source.name} -> {target.stem} arm {arm}\n{result.stderr[-1500:]}")
            return None
        module = missing.group(1).split(".")[0]
        log(f"  installing missing module {module}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        PIP_NAME.get(module, module)], check=False)
    return None


def gap_levels(wave: np.ndarray, rate: int, gaps: list[list[float]]) -> list[float]:
    """Mean frame energy inside each gap, in dB over the clip's 2nd-percentile floor."""
    hop, window = int(0.010 * rate), int(0.025 * rate)
    if len(wave) < window:
        return []
    count = 1 + (len(wave) - window) // hop
    index = np.arange(window)[None, :] + hop * np.arange(count)[:, None]
    energy = 10 * np.log10((wave[index] ** 2).mean(1) + 1e-12)
    floor = float(np.percentile(energy, 2))
    levels = []
    for start, end in gaps:
        first, last = int(start / 0.010), min(count, int(end / 0.010) + 1)
        if last - first >= 2:
            levels.append(float(energy[first:last].mean() - floor))
    return levels


def main() -> int:
    import soundfile as sf

    started = time.time()
    show_inputs()
    inputs = find_dataset("picks.json", "refs")
    calibration = find_dataset("clips", "manifest.json")
    picks = json.loads((inputs / "picks.json").read_text(encoding="utf-8"))["picks"]
    log(f"{len(picks)} source clips; references in {inputs / 'refs'}")
    setup()
    CLIPS.mkdir(parents=True, exist_ok=True)
    for voice in VOICES:
        shutil.copy(inputs / "refs" / f"{voice}.wav", CLIPS / f"00_reference_{voice}.wav")

    rows, sheet = [], []
    for number, pick in enumerate(picks, 1):
        source_flac = calibration / "clips" / f"{pick['id']}.flac"
        wave, rate = sf.read(str(source_flac), dtype="float32")
        source = CLIPS / f"{number:02d}_source.wav"
        sf.write(str(source), wave, rate)
        row = {"n": number, "id": pick["id"], "breath_gaps": pick["breath_gaps"],
               "source": {"breath": gap_levels(wave, rate, pick["breath_gaps"]),
                          "control": gap_levels(wave, rate, pick["control_gaps"])}}
        marks = ", ".join(f"{start:.1f}s" for start, _ in pick["breath_gaps"])
        sheet.append(f"{source.name}\tbreath at {marks}\toriginal speaker (do not share)")
        for voice in VOICES:
            for arm in ARMS:
                made = convert(source, inputs / "refs" / f"{voice}.wav", arm,
                               Path(f"/kaggle/temp/out/{number}_{voice}_{arm}"))
                if made is None:
                    row[f"{voice}_{arm}"] = None
                    if number == 1 and not any(row.get(f"{v}_{a}") for v in VOICES for a in ARMS) \
                            and (voice, arm) == (VOICES[-1], list(ARMS)[-1]):
                        raise SystemExit("every conversion of clip 1 failed; stopping early")
                    continue
                final = CLIPS / f"{number:02d}_{voice}_{arm}.wav"
                shutil.copy(made, final)
                converted, converted_rate = sf.read(str(final), dtype="float32")
                if converted.ndim > 1:
                    converted = converted.mean(axis=1)
                row[f"{voice}_{arm}"] = {
                    "seconds": round(len(converted) / converted_rate, 3),
                    "source_seconds": round(len(wave) / rate, 3),
                    "breath": gap_levels(converted, converted_rate, pick["breath_gaps"]),
                    "control": gap_levels(converted, converted_rate, pick["control_gaps"])}
                sheet.append(f"{final.name}\tbreath at {marks}\t{voice}, arm {arm}")
        rows.append(row)
        log(f"clip {number}/{len(picks)} done at {time.time() - started:.0f}s")

    summary = {}
    for key in ["source"] + [f"{voice}_{arm}" for voice in VOICES for arm in ARMS]:
        breath = [x for row in rows if row.get(key) for x in row[key]["breath"]]
        control = [x for row in rows if row.get(key) for x in row[key]["control"]]
        done = sum(1 for row in rows if row.get(key))
        summary[key] = {"clips": done,
                        "breath_gap_db_over_floor": round(float(np.median(breath)), 1) if breath else None,
                        "control_gap_db_over_floor": round(float(np.median(control)), 1) if control else None}
        log(f"{key:10s} clips {done:2d}  breath gaps {summary[key]['breath_gap_db_over_floor']} dB  "
            f"control gaps {summary[key]['control_gap_db_over_floor']} dB over floor (medians)")
    (OUT_DIR / "vc_breath_probe.json").write_text(json.dumps(
        {"commit": COMMIT, "diffusion_steps": DIFFUSION_STEPS, "arms": ARMS,
         "summary": summary, "clips": rows}, indent=1), encoding="utf-8")
    (CLIPS / "LISTEN.tsv").write_text("\n".join(sheet), encoding="utf-8")
    log(f"done in {time.time() - started:.0f}s")
    return 0 if any(row.get(f"{v}_{a}") for row in rows for v in VOICES for a in ARMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
