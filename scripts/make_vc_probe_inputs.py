r"""Inputs for the voice-conversion breath probe (kaggle\vc_breath_probe).

    py -3.11 scripts\make_vc_probe_inputs.py

Picks 12 IndicVoices calibration clips (already on Kaggle as geetha-breath-calibration)
whose C4-located breath gap carries the most energy over the clip's floor - audible
breaths, from 12 different speakers - and one clean 9-14 s Rasa reference per voice.
Writes a small Kaggle dataset folder: refs\anitha.wav, refs\arun.wav, picks.json.

Rasa train shards only. Nothing here is training data: the probe is a listening test.
"""

from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geetha.audio.breath import HOP_SECONDS, frame_features  # noqa: E402

CALIBRATION = Path(r"E:\Geetha\work\breath_calibration")
ALIGNMENTS = Path(r"E:\Geetha\runs\c4_breath_timestamps\c4_alignments.json")
RASA = r"E:\Geetha\raw\hf\rasa\Tamil\train-*.parquet"
OUT = Path(r"E:\Geetha\work\vc_breath_probe\dataset")
N_PICKS = 12


def pick_sources() -> list[dict]:
    import soundfile as sf
    manifest = {c["id"]: c for c in
                json.loads((CALIBRATION / "manifest.json").read_text(encoding="utf-8"))["clips"]}
    scored = []
    for record in json.loads(ALIGNMENTS.read_text(encoding="utf-8"))["clips"]:
        if record["label"] != "positive" or not 6.0 <= record["seconds"] <= 14.0:
            continue
        breaths = [g for g in record["gaps"] if g["has_breath"] and 0.3 <= g["seconds"] <= 1.2]
        if not breaths:
            continue
        wave, _ = sf.read(str(CALIBRATION / "clips" / f"{record['id']}.flac"), dtype="float32")
        frames = frame_features(wave)
        energy = frames["energy_db"]
        floor = float(np.percentile(energy, 2))
        best = max(float(energy[int(g["start"] / HOP_SECONDS):int(g["end"] / HOP_SECONDS) + 1].mean())
                   - floor for g in breaths)
        controls = [g for g in record["gaps"] if not g["has_breath"] and g["seconds"] >= 0.15]
        scored.append({"id": record["id"], "speaker": manifest[record["id"]]["speaker_id"],
                       "seconds": record["seconds"], "breath_db_over_floor": round(best, 1),
                       "breath_gaps": [[g["start"], g["end"]] for g in record["gaps"] if g["has_breath"]],
                       "control_gaps": [[g["start"], g["end"]] for g in controls]})
    scored.sort(key=lambda row: -row["breath_db_over_floor"])
    picks, seen = [], set()
    for row in scored:
        if row["speaker"] in seen:
            continue
        seen.add(row["speaker"])
        del row["speaker"]          # anonymous training signal only; the id stays off Kaggle
        picks.append(row)
        if len(picks) == N_PICKS:
            break
    return picks


def pick_references() -> dict[str, str]:
    import pyarrow.parquet as pq
    import soundfile as sf
    want = {"Female": "anitha", "Male": "arun"}
    found: dict[str, str] = {}
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    for shard in sorted(glob.glob(RASA)):
        table = pq.ParquetFile(shard).read(columns=["filename", "gender", "style", "duration"])
        for index, row in enumerate(table.to_pylist()):
            voice = want.get(row["gender"])
            if voice is None or voice in found or row["style"] not in ("CONV", "BOOK"):
                continue
            if not 9.0 <= float(row["duration"]) <= 14.0:
                continue
            audio = pq.ParquetFile(shard).read(columns=["audio"]).slice(index, 1).to_pylist()[0]["audio"]
            wave, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            sf.write(str(OUT / "refs" / f"{voice}.wav"), wave, rate)
            found[voice] = f"{row['filename']} ({row['style']}, {row['duration']} s, {rate} Hz)"
        if len(found) == 2:
            break
    return found


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    picks = pick_sources()
    references = pick_references()
    (OUT / "picks.json").write_text(json.dumps({"picks": picks, "references": references},
                                               indent=1), encoding="utf-8")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "geetha-vc-probe-inputs", "id": "dhevenddra/geetha-vc-probe-inputs",
        "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"{len(picks)} source clips; breath dB over floor "
          f"{picks[-1]['breath_db_over_floor']} .. {picks[0]['breath_db_over_floor']}")
    print("references:", references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
