r"""Build the three arms of the Rasa contamination check and package them for Kaggle.

    py -3.11 scripts\make_contamination_arms.py

Writes ``E:\Geetha\work\c1_contamination\``:

    manifest.json          every selected clip, its arm, text, speaker, style, duration
    audio\<filename>.wav   the real recordings: both arms plus the null pool
    README.md              what the folder is, for whoever opens it later

The heavy work (generating speech, ASR, F0, speaker embeddings) happens on Kaggle —
see ``kaggle\c1_contamination\``. This script only decides *which* clips, so that the
choice is reproducible, auditable and fixed before any audio is generated.

Nothing here is an eval-vault artefact. These clips measure the base model, not ours.
"""

from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.eval.contamination import Clip, select_matched_arms, style_group  # noqa: E402

RASA = Path(r"E:\Geetha\raw\hf\rasa\Tamil")
OUT = Path(r"E:\Geetha\work\c1_contamination")

N_PER_ARM = 50
SEED = 20261027

# The null distribution: real takes the rendering is *not* of. Twelve per
# (gender, style group) is enough for a stable median without bloating the upload.
NULL_PER_CELL = 12

# Long clips dominate the upload and say little extra; very short ones make F0 contours
# noisy. Both arms and the null pool are drawn from the same window so the restriction
# cannot favour one arm.
MIN_SECONDS = 3.0
MAX_SECONDS = 12.0

SPEAKER = {"Female": "Anitha", "Male": "Arun"}

META_COLUMNS = ["filename", "text", "gender", "style", "duration"]


def _load(split: str) -> list[Clip]:
    clips: list[Clip] = []
    for path in sorted(glob.glob(str(RASA / f"{split}-*.parquet"))):
        table = pq.read_table(path, columns=META_COLUMNS).to_pydict()
        for name, text, gender, style, duration in zip(
            table["filename"], table["text"], table["gender"], table["style"], table["duration"]
        ):
            seconds = float(duration)
            if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
                continue
            clips.append(
                Clip(
                    filename=name,
                    text=text.strip(),
                    gender=gender,
                    style=style,
                    duration=seconds,
                )
            )
    return clips


def _pick_null_pool(test_clips: list[Clip], chosen: set[str]) -> list[Clip]:
    """Other real takes by the same voices, spread across style groups."""
    cells: dict[tuple[str, str], list[Clip]] = {}
    for clip in test_clips:
        if clip.filename in chosen:
            continue
        cells.setdefault((clip.gender, style_group(clip.style)), []).append(clip)

    pool: list[Clip] = []
    for cell, members in sorted(cells.items()):
        members.sort(key=lambda c: c.filename)
        step = max(1, len(members) // NULL_PER_CELL)
        pool.extend(members[::step][:NULL_PER_CELL])
    return pool


def _audio_index(split: str, wanted: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(glob.glob(str(RASA / f"{split}-*.parquet"))):
        table = pq.read_table(path, columns=["filename", "audio"]).to_pydict()
        for name, audio in zip(table["filename"], table["audio"]):
            if name in wanted:
                found[name] = audio["bytes"]
        if len(found) == len(wanted):
            break
    return found


def main() -> int:
    test_clips = _load("test")
    train_clips = _load("train")
    print(f"candidates after the {MIN_SECONDS}-{MAX_SECONDS}s window: "
          f"test {len(test_clips)}, train {len(train_clips)}")

    test_arm, train_arm = select_matched_arms(test_clips, train_clips, N_PER_ARM, SEED)
    null_pool = _pick_null_pool(test_clips, {c.filename for c in test_arm})
    print(f"arms: TEST {len(test_arm)}, TRAIN {len(train_arm)}, null pool {len(null_pool)}")

    rows = []
    for arm, clips in (("TEST", test_arm), ("TRAIN", train_arm), ("NULL", null_pool)):
        for clip in clips:
            rows.append(
                {
                    "arm": arm,
                    "filename": clip.filename,
                    "text": clip.text,
                    "gender": clip.gender,
                    "speaker": SPEAKER[clip.gender],
                    "style": clip.style,
                    "style_group": style_group(clip.style),
                    "duration": round(clip.duration, 3),
                    "split": "train" if arm == "TRAIN" else "test",
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)

    for split in ("test", "train"):
        wanted = {r["filename"] for r in rows if r["split"] == split}
        blobs = _audio_index(split, wanted)
        missing = wanted - set(blobs)
        if missing:
            raise SystemExit(f"{len(missing)} {split} clips had no audio, e.g. {sorted(missing)[:3]}")
        for name, data in blobs.items():
            (OUT / "audio" / f"{name}.wav").write_bytes(data)
        print(f"wrote {len(blobs)} {split} recordings")

    manifest = {
        "created": "2026-10-27",
        "purpose": "Rasa train/test contamination check for Indic-Speak (B4a)",
        "seed": SEED,
        "n_per_arm": N_PER_ARM,
        "duration_window_s": [MIN_SECONDS, MAX_SECONDS],
        "speaker_map": SPEAKER,
        "clips": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    megabytes = sum(p.stat().st_size for p in (OUT / "audio").glob("*.wav")) / 1e6
    print(f"manifest.json + {len(rows)} clips, audio {megabytes:.0f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
