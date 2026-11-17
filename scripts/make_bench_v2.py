"""Build the 300+ sentence pure-Tamil benchmark and the two Kaggle kernels that run it.

    python scripts\\make_bench_v2.py            reuse work\\bench_ta_v2\\bench_ta_v2.json if present
    python scripts\\make_bench_v2.py --rebuild  rebuild the prompt list (re-runs the leak check)

Prompts are sentences written only in Tamil script, so the written text is the reference:

    fleurs  FLEURS ta_in TEST raw transcriptions, 40-150 characters, no digits or Latin
    dir     Directability-Ta v1: the 40 carriers and the 60 untagged emotional sentences
    cap     Capability-Ta v1: the 84 sentences without marks (spoken_text)

Every prompt is checked by normalised text against the training corpora; a collision is
dropped and counted. Voices alternate anitha / arun in list order. The kernels embed the
prompt list, worker.py and the three release files byte for byte.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = Path(r"E:\Geetha\raw")
VAULT = Path(r"E:\Geetha\eval_vault")
OUT = Path(r"E:\Geetha\work\bench_ta_v2\bench_ta_v2.json")
RELEASE = ROOT / "release"
SOURCE = ROOT / "kaggle" / "m3_bench"
SEED = 1234
FLEURS_WANTED = 120
VOICES = ("anitha", "arun")
# The same list the B4 leak check used (scripts\make_directability_ta.py).
TRAINING_SOURCES = [
    ("indicvoices", r"hf\indicvoices\tamil\train-*.parquet", "text"),
    ("indicvoices_r", r"hf\indicvoices_r\Tamil\train-*.parquet", "text"),
    ("rasa", r"hf\rasa\Tamil\train-*.parquet", "text"),
    ("spring_r1", r"hf\spring_inx_tamil_r1\data\train-*.parquet", "text"),
    ("spring_r2", r"hf\spring_inx_tamil_r2\data\train-*.parquet", "text"),
    ("fleurs", r"hf\fleurs\parquet-data\ta_in\train-*.parquet", "raw_transcription"),
    ("shrutilipi", r"hf\shrutilipi\**\train-*.parquet", "text"),
]
NOT_TAMIL_ONLY = re.compile(r"[A-Za-z0-9]")
TAMIL = re.compile(r"[\u0B80-\u0BFF]")


def clean(text: str) -> str:
    return " ".join(text.replace('"', " ").replace("\u201c", " ").replace("\u201d", " ").split())


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def candidates() -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    table = pq.read_table(RAW / r"hf\fleurs\parquet-data\ta_in\test-00000-of-00001.parquet",
                          columns=["raw_transcription"])
    fleurs = sorted({clean(t) for t in table.column(0).to_pylist() if t})
    fleurs = [t for t in fleurs if 40 <= len(t) <= 150]
    rows += [{"source": "fleurs", "text": t} for t in fleurs]
    seen: set[str] = set()
    for item in jsonl(VAULT / "directability_ta_v1.jsonl"):
        if item["kind"] == "tagged" and item["carrier_id"] not in seen:
            seen.add(item["carrier_id"])
            rows.append({"source": "dir", "origin": item["carrier_id"],
                         "text": clean(item["text"])})
        elif item["kind"] == "untagged_emotional":
            rows.append({"source": "dir", "origin": item["id"], "text": clean(item["text"])})
    for item in jsonl(VAULT / "naturalness_ta_v1" / "capability_ta_v1.jsonl"):
        if item["for_baselines"]:
            rows.append({"source": "cap", "origin": item["item_id"],
                         "text": clean(item["spoken_text"])})
    return rows


def build_prompts() -> dict:
    import pyarrow.parquet as pq

    from geetha.text.normalise import norm_text

    rows = candidates()
    dropped = {"not_tamil_only": 0, "duplicate": 0, "in_training": 0}
    kept: list[dict] = []
    keys: dict[str, dict] = {}
    for row in rows:
        if NOT_TAMIL_ONLY.search(row["text"]) or not TAMIL.search(row["text"]):
            dropped["not_tamil_only"] += 1
            continue
        key = norm_text(row["text"])
        if key in keys:
            dropped["duplicate"] += 1
            continue
        keys[key] = row
        kept.append(row)
    coverage: dict[str, int] = {}
    leaked: set[str] = set()
    for name, pattern, column in TRAINING_SOURCES:
        files = sorted(glob.glob(str(RAW / pattern), recursive=True))
        count = 0
        for path in files:
            try:
                values = pq.read_table(path, columns=[column]).column(column).to_pylist()
            except Exception as error:  # noqa: BLE001
                print(f"   {name}: {Path(path).name} unreadable ({error}); skipped")
                continue
            for value in values:
                if value:
                    count += 1
                    key = norm_text(value)
                    if key in keys:
                        leaked.add(key)
        coverage[name] = count if files else -1
        print(f"   leak check, {name}: {count:,} training sentences, "
              f"{len(leaked)} collisions so far")
    for key in leaked:
        print(f"   dropped (in training): [{keys[key]['source']}] {keys[key]['text'][:70]}")
    kept = [row for row in kept if norm_text(row["text"]) not in leaked]
    dropped["in_training"] = len(leaked)

    fleurs = [row for row in kept if row["source"] == "fleurs"]
    random.Random(SEED).shuffle(fleurs)
    fleurs = sorted(fleurs[:FLEURS_WANTED], key=lambda row: row["text"])
    ordered = fleurs + [r for r in kept if r["source"] == "dir"] + \
        [r for r in kept if r["source"] == "cap"]
    counters: dict[str, int] = {}
    prompts = []
    for index, row in enumerate(ordered):
        counters[row["source"]] = counters.get(row["source"], 0) + 1
        prompts.append({"id": f"{row['source']}{counters[row['source']]:03d}",
                        "source": row["source"], "origin": row.get("origin", ""),
                        "voice": VOICES[index % 2], "text": row["text"]})
    return {"name": "bench_ta_v2", "seed": SEED, "prompts": prompts, "dropped": dropped,
            "by_source": counters, "leak_index": coverage,
            "note": "Tamil script only; the written text is the reference."}


def write_kernels(bench: dict) -> None:
    payload_names = {"ours": ("worker.py", "inference.py", "tamil_text.py", "breaths.json"),
                     "rivals": ("worker.py",)}
    template = (SOURCE / "template.py").read_text(encoding="utf-8")
    prompts = json.dumps(bench["prompts"], ensure_ascii=False)
    if "'''" in prompts or "\\" in prompts:
        raise SystemExit("a prompt holds a quote or backslash the kernel cannot embed")
    for group, names in payload_names.items():
        payload = {}
        for name in names:
            data = ((SOURCE if name == "worker.py" else RELEASE) / name).read_bytes()
            payload[name] = base64.b64encode(data).decode("ascii")
            print(f"{group}: {name}: {len(data):,} bytes  "
                  f"sha256 {hashlib.sha256(data).hexdigest()[:16]}")
        script = (template.replace("__GROUP__", group)
                  .replace("__N__", str(len(bench["prompts"])))
                  .replace("__PROMPTS__", prompts)
                  .replace("__PAYLOAD__", json.dumps(payload, indent=1)))
        compile(script, f"m3_bench_{group}.py", "exec")
        kernel = ROOT / "kaggle" / f"m3_bench_{group}"
        kernel.mkdir(parents=True, exist_ok=True)
        (kernel / f"m3_bench_{group}.py").write_text(script, encoding="utf-8", newline="\n")
        slug = f"geetha-m3-bench-{group}"
        (kernel / "kernel-metadata.json").write_text(json.dumps({
            "id": f"dhevenddra/{slug}", "title": slug,
            "code_file": f"m3_bench_{group}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True,
            "dataset_sources": ["dhevenddra/geetha-code", "dhevenddra/geetha-secrets"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []},
            indent=2), encoding="utf-8")
        print(f"wrote {kernel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.rebuild:
        bench = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"reusing {OUT}")
    else:
        bench = build_prompts()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(bench, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {OUT}")
    lengths = [len(p["text"]) for p in bench["prompts"]]
    print(f"{len(bench['prompts'])} prompts {bench['by_source']}  dropped {bench['dropped']}  "
          f"characters: mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")
    write_kernels(bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
