r"""What actually predicts codec damage? (follows D-040, D-043)

    py -3.11 scripts\analyse_codec_damage.py

Runs once C5 has reported. Joins its per-clip Vocos CER delta back to the metadata
IndicVoices-R already ships -- speaking rate, SNR, C50 reverberation, pitch, duration --
and reports the correlation of each against the damage.

Why this exists
---------------
Three hypotheses for why some clips come back unintelligible have now been tested on
the operator's 20 listened clips, and all three failed:

* **clip duration** (his first guess): r = 0.012 against his verdicts, and clips 12 and
  13 are identical in length with opposite verdicts;
* **my spectral similarity**: r = 0.013 -- the clip he rated okay scores lowest of all 20;
* **speaking rate** (his second guess, from "too fast warbled"): r = -0.213, in the
  *wrong direction*, and his slowest clip of all (8.12) is one he called warble.

Twenty clips with four bad ones cannot separate these. Guessing again on the same twenty
would be the error the project already named: a hypothesis that fits a small sample is
not a finding. C5 gives 300 clips with an objective per-clip number, which is enough to
ask all the questions at once instead of one at a time.

What this deliberately does not do: pick the winner. It reports every correlation with
its interval, including the ones near zero, so that "nothing here explains it" stays a
possible and publishable answer. If that is the result, the next instrument is
phoneme-level -- the operator's own observation that `ழ` and `ற` lose their distinctness
points there, and that needs forced alignment rather than clip-level statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(r"E:\Geetha")
RESULT = ROOT / "results" / "codec_cer.json"
IVR = ROOT / "raw" / "hf" / "indicvoices_r" / "Tamil"
REPORT = ROOT / "results" / "codec_damage_analysis.md"

# Everything IV-R gives us per clip without any extra computation.
METADATA = ("speaking_rate", "snr", "c50", "utterance_pitch_mean",
            "utterance_pitch_std", "duration")


def bootstrap_r(x: np.ndarray, y: np.ndarray, draws: int = 4_000, seed: int = 7):
    """Correlation with a bootstrap interval, so a near-zero r can be read as near-zero."""
    rng = np.random.default_rng(seed)
    point = float(np.corrcoef(x, y)[0, 1])
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(x), len(x))
        if np.std(x[pick]) == 0 or np.std(y[pick]) == 0:
            continue
        values.append(np.corrcoef(x[pick], y[pick])[0, 1])
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def main() -> int:
    if not RESULT.exists():
        print(f"!! {RESULT} not found -- pull C5 first", file=sys.stderr)
        return 1
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = payload["rows"]
    key = "delta_vocos" if "delta_vocos" in rows[0] else "delta"
    if key == "delta":
        print("!! this C5 output predates the Vocos A/B (D-043); the damage column is "
              "the stock decoder's, which is not the production path", file=sys.stderr)

    # C5 streams IV-R in order and records the stream index, so the metadata is
    # recoverable by walking the same shards in the same order.
    wanted = {row["index"]: row for row in rows}
    collected: dict[int, dict] = {}
    position = 0
    for shard in sorted(IVR.glob("train-*.parquet")):
        table = pq.read_table(shard, columns=[c for c in METADATA])
        block = table.to_pylist()
        for record in block:
            if position in wanted:
                collected[position] = record
            position += 1
        if len(collected) == len(wanted):
            break

    matched = [(wanted[i], collected[i]) for i in sorted(collected)]
    if not matched:
        print("!! no clips matched; C5's stream indices do not line up with the shards",
              file=sys.stderr)
        return 1
    damage = np.array([float(r[key]) for r, _ in matched])

    lines = [
        "# What predicts codec damage?",
        "",
        f"{len(matched)} clips, damage = per-clip CER delta "
        f"({'Vocos, the production decoder' if key == 'delta_vocos' else 'STOCK decoder — not the production path'}).",
        f"Mean damage {damage.mean():+.4f}, median {np.median(damage):+.4f}.",
        "",
        "| factor | r | 95% CI | reading |",
        "| --- | ---: | :---: | --- |",
    ]
    for name in METADATA:
        values = np.array([float(meta.get(name) or np.nan) for _, meta in matched])
        usable = np.isfinite(values) & np.isfinite(damage)
        if usable.sum() < 10 or np.std(values[usable]) == 0:
            lines.append(f"| {name} | - | - | too few usable values |")
            continue
        point, low, high = bootstrap_r(values[usable], damage[usable])
        crosses_zero = low <= 0 <= high
        reading = "no relationship" if crosses_zero else (
            "worse when higher" if point > 0 else "worse when lower")
        lines.append(f"| {name} | {point:+.3f} | [{low:+.3f}, {high:+.3f}] | {reading} |")

    lines += [
        "",
        "## How to read this",
        "",
        "An interval straddling zero means the factor does **not** explain the damage,",
        "and that is a real answer rather than a failed search. Three clip-level",
        "hypotheses have already failed on the operator's 20 listened clips: duration",
        "(r = 0.012), my spectral similarity (r = 0.013) and speaking rate (r = -0.213,",
        "in the wrong direction). If they all fail here too, on 300 clips with an",
        "objective measure, then codec damage is not a clip-level property and the next",
        "instrument has to be phoneme-level.",
        "",
        "The operator's own observation points that way: `ழ` losing its emphasis and",
        "`மாறிக்கிட்டு` arriving as \"mahikitu\" are specific consonants failing, not whole",
        "clips degrading. Testing that needs forced alignment -- which this project",
        "already built for C4 -- not another clip-level statistic.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
