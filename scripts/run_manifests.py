r"""ETL step 5: build the Stage A and Stage B training manifests.

    py -3.11 scripts\run_manifests.py

Joins the three things the earlier steps produced, on ``(source, shard, row_index)``:

    work\audio_qc.parquet              step 1 -- level, clipping, edge silence
    processed\text_norm\<source>\      step 2 -- keep/drop and the raw text
    processed\ivr_marks\               step 3b -- IndicVoices marks for IV-R

and applies step 4's render (D-035) and step 5's gates and gain (D-036).

Outputs
    processed\tts_stageA\manifest.parquet
    processed\tts_stageB\manifest.parquet
    processed\manifest_report.md        hours per source and per speaker

Stage A v0 is IV-R + Rasa, with no restored conversation: D-034 moved Sidon behind M2.
Stage B is Rasa's conversational and emotional styles, repeated, which is where
directability is taught.

Two things this deliberately does **not** do. It never uses ``norm_text`` for training
text -- punctuation is load-bearing for the model's pauses (D-035(f)) -- and it never
writes a real IV-R speaker id: those speakers are anonymous training signal only
(the project rules), so they enter the manifest as salted pseudonyms and the mapping stays
on this machine.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.data.manifest import TARGET_RMS_DBFS, gain_db, qc_verdict  # noqa: E402
from geetha.data.mark_render import RejectedRow, render_training_text  # noqa: E402

ROOT = Path(r"E:\Geetha")
QC = ROOT / "work" / "audio_qc.parquet"
TEXT_NORM = ROOT / "processed" / "text_norm"
IVR_MARKS = ROOT / "processed" / "ivr_marks"
STAGE_A = ROOT / "processed" / "tts_stageA"
STAGE_B = ROOT / "processed" / "tts_stageB"
REPORT = ROOT / "processed" / "manifest_report.md"

# The salt that already anonymises the IV-R holdout, reused so one speaker has one
# pseudonym everywhere in the project.
SALT = "geetha-ivr-holdout-v1"

# Rasa's two voices are the only speakers v1 ever exposes.
RASA_VOICES = {"TAM_F": "anitha", "TAM_M": "arun"}

EMOTION_STYLES = {"ANGER", "HAPPY", "SAD", "FEAR", "SURPRISE", "DISGUST"}
# Stage B repeats: emotions get one pass more than CONV, because emotion adherence is
# the direction headline and these styles are the rarer half of Rasa.
STAGE_B_REPEATS = {"CONV": 2}
STAGE_B_EMOTION_REPEATS = 3

SCHEMA = pa.schema([
    pa.field("source", pa.string()),
    pa.field("shard", pa.string()),
    pa.field("row_index", pa.int32()),
    pa.field("speaker", pa.string()),
    pa.field("style", pa.string()),
    pa.field("env", pa.string()),
    pa.field("text", pa.string()),
    pa.field("duration_s", pa.float64()),
    pa.field("gain_db", pa.float64()),
    pa.field("rms_dbfs", pa.float64()),
    pa.field("n_breath", pa.int32()),
    pa.field("n_events", pa.int32()),
    pa.field("repeat", pa.int32()),
])


def pseudonym(speaker_id: str) -> str:
    digest = hashlib.sha256(f"{SALT}:{speaker_id}".encode()).hexdigest()
    return f"ivr_{digest[:8]}"


def load_qc() -> dict[tuple[str, str, int], dict]:
    table = pq.read_table(QC)
    rows = table.to_pylist()
    return {(row["source"], row["shard"], row["row_index"]): row for row in rows}


def load_text(source: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for path in sorted((TEXT_NORM / source).glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            out[(row["shard"], row["row_index"])] = row
    return out


def load_marks() -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for path in sorted(IVR_MARKS.glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            out[(row["shard"], row["row_index"])] = row
    return out


def main() -> int:
    print("loading step 1 / step 2 / step 3b outputs")
    qc = load_qc()
    rejected: Counter[str] = Counter()
    rows_a: list[dict] = []
    rows_b: list[dict] = []

    # ---- IV-R -------------------------------------------------------------
    text_ivr = load_text("indicvoices_r")
    marks = load_marks()
    print(f"IV-R: {len(text_ivr):,} step-2 rows, {len(marks):,} step-3b rows")
    for (shard, row_index), decision in text_ivr.items():
        if not decision["keep"]:
            rejected[f"indicvoices_r/step2:{decision['reason']}"] += 1
            continue
        qc_row = qc.get(("indicvoices_r", shard, row_index))
        if qc_row is None:
            rejected["indicvoices_r/no_qc_row"] += 1
            continue
        verdict = qc_verdict(qc_row)
        if verdict:
            rejected[f"indicvoices_r/qc:{verdict}"] += 1
            continue

        mark_row = marks.get((shard, row_index))
        # Prefer the marked text when a twin was found; fall back to the plain
        # transcript, which is correct text with no event annotation.
        marked = mark_row["marked_text"] if mark_row and mark_row["joined"] else None
        try:
            text = render_training_text(marked or decision["text"])
        except RejectedRow as problem:
            rejected[f"indicvoices_r/render:{str(problem).split(':')[0]}"] += 1
            continue

        rows_a.append({
            "source": "indicvoices_r", "shard": shard, "row_index": row_index,
            "speaker": pseudonym(decision["speaker_id"]),
            "style": (qc_row.get("scenario") or "extempore").lower(),
            "env": "enhanced",
            "text": text,
            "duration_s": qc_row["duration_s"],
            "gain_db": gain_db(qc_row["rms_dbfs"]),
            "rms_dbfs": qc_row["rms_dbfs"],
            "n_breath": (mark_row or {}).get("n_breath") or 0,
            "n_events": (mark_row or {}).get("n_events") or 0,
            "repeat": 1,
        })

    # ---- Rasa -------------------------------------------------------------
    text_rasa = load_text("rasa")
    print(f"Rasa: {len(text_rasa):,} step-2 rows")
    for (shard, row_index), decision in text_rasa.items():
        if not decision["keep"]:
            rejected[f"rasa/step2:{decision['reason']}"] += 1
            continue
        qc_row = qc.get(("rasa", shard, row_index))
        if qc_row is None:
            rejected["rasa/no_qc_row"] += 1
            continue
        verdict = qc_verdict(qc_row)
        if verdict:
            rejected[f"rasa/qc:{verdict}"] += 1
            continue

        style = (qc_row.get("style") or "").strip().upper()
        speaker = RASA_VOICES.get(decision["speaker_id"] or "")
        if speaker is None:
            rejected["rasa/unknown_voice"] += 1
            continue
        # Rasa transcripts carry no event marks, so the text passes through the render
        # only to be rejected if it is empty.
        try:
            text = render_training_text(decision["text"])
        except RejectedRow:
            rejected["rasa/render:no_text"] += 1
            continue

        row = {
            "source": "rasa", "shard": shard, "row_index": row_index,
            "speaker": speaker, "style": style, "env": "studio",
            "text": text,
            "duration_s": qc_row["duration_s"],
            "gain_db": gain_db(qc_row["rms_dbfs"]),
            "rms_dbfs": qc_row["rms_dbfs"],
            "n_breath": 0, "n_events": 0, "repeat": 1,
        }
        rows_a.append(row)

        repeats = (STAGE_B_EMOTION_REPEATS if style in EMOTION_STYLES
                   else STAGE_B_REPEATS.get(style))
        if repeats:
            rows_b.append({**row, "repeat": repeats})

    STAGE_A.mkdir(parents=True, exist_ok=True)
    STAGE_B.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows_a, schema=SCHEMA), STAGE_A / "manifest.parquet")
    pq.write_table(pa.Table.from_pylist(rows_b, schema=SCHEMA), STAGE_B / "manifest.parquet")

    # Keep the pseudonym map local; it is the only thing that could re-identify a
    # speaker, and it never leaves this machine.
    mapping = {pseudonym(row["speaker_id"]): row["speaker_id"]
               for row in text_ivr.values() if row["keep"] and row["speaker_id"]}
    (ROOT / "work" / "ivr_speaker_pseudonyms.json").write_text(
        json.dumps(mapping, indent=1), encoding="utf-8")

    report = build_report(rows_a, rows_b, rejected, len(mapping))
    REPORT.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"wrote {STAGE_A / 'manifest.parquet'}, {STAGE_B / 'manifest.parquet'}, {REPORT}")
    return 0


def build_report(rows_a, rows_b, rejected, n_speakers) -> str:
    def hours(rows, weighted=False):
        return sum((row["duration_s"] or 0.0) * (row["repeat"] if weighted else 1)
                   for row in rows) / 3600.0

    by_source: dict[str, float] = defaultdict(float)
    by_speaker: dict[str, float] = defaultdict(float)
    by_style: dict[str, float] = defaultdict(float)
    breaths = events = 0
    for row in rows_a:
        by_source[row["source"]] += (row["duration_s"] or 0.0) / 3600.0
        by_speaker[row["speaker"]] += (row["duration_s"] or 0.0) / 3600.0
        by_style[f"{row['source']}/{row['style']}"] += (row["duration_s"] or 0.0) / 3600.0
        breaths += row["n_breath"] or 0
        events += row["n_events"] or 0

    b_style: dict[str, tuple[int, float, float]] = {}
    for row in rows_b:
        clips, raw, weighted = b_style.get(row["style"], (0, 0.0, 0.0))
        seconds = (row["duration_s"] or 0.0) / 3600.0
        b_style[row["style"]] = (clips + 1, raw + seconds, weighted + seconds * row["repeat"])

    lines = [
        "# ETL step 5: Stage A and Stage B manifests",
        "",
        f"Built 3 Nov 2026. Loudness target {TARGET_RMS_DBFS} dBFS RMS, attenuation only (D-036).",
        "Stage A v0 carries no restored conversation: D-034 moved Sidon behind M2.",
        "",
        "## Stage A",
        "",
        f"- clips: **{len(rows_a):,}**",
        f"- hours: **{hours(rows_a):.1f} h** (runbook target 120-150 h)",
        f"- IV-R speakers (pseudonymous): {n_speakers:,}; plus Rasa's 2 named voices",
        f"- breath marks carried into training text: **{breaths:,}**; all voiceable events {events:,}",
        "",
        "| source | hours |",
        "| --- | ---: |",
    ]
    for name, value in sorted(by_source.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {value:.1f} |")

    lines += ["", "| exposed speaker | hours |", "| --- | ---: |"]
    for name, value in sorted(by_speaker.items(), key=lambda kv: -kv[1]):
        if not name.startswith("ivr_"):
            lines.append(f"| {name} | {value:.1f} |")
    lines.append(f"| {n_speakers:,} IV-R pseudonyms (not exposed) | "
                 f"{sum(v for k, v in by_speaker.items() if k.startswith('ivr_')):.1f} |")

    lines += ["", "| source / style | hours |", "| --- | ---: |"]
    for name, value in sorted(by_style.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {value:.1f} |")

    lines += [
        "",
        "## Stage B",
        "",
        f"- clips: **{len(rows_b):,}** distinct, **{hours(rows_b):.1f} h** of distinct audio",
        f"- with repeats: **{hours(rows_b, weighted=True):.1f} h** of training signal",
        "",
        "| style | clips | distinct h | with repeats |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, (clips, raw, weighted) in sorted(b_style.items(), key=lambda kv: -kv[1][2]):
        lines.append(f"| {name} | {clips:,} | {raw:.1f} | {weighted:.1f} |")

    lines += [
        "",
        "**Still outstanding for Stage B:** the ~2 h of base-model Anitha/Arun anchor clips",
        "the runbook asks for. Those have to be generated by the base model, so they arrive",
        "with the first GPU run rather than from this script.",
        "",
        "## Rows excluded, and why",
        "",
        "| reason | rows |",
        "| --- | ---: |",
    ]
    for name, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count:,} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
