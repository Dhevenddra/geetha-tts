"""Score benchmark v2 on the PC from the transcripts the two Kaggle kernels wrote.

    python scripts\\score_bench_v2.py [--out E:\\Geetha\\results\\bench_v2]

Reads runs\\m3_bench_ours\\bench_{geetha,base}.json and
runs\\m3_bench_rivals\\bench_{rumik_oss_1,snortts_indic_v0}.json. The prompts are Tamil
script only, so the written text is the reference. For each judge and system it reports:

  * validity: valid / invalid (silent, or ran to the cap) / not run, as their own rates;
  * CER on that system's valid clips with a 95% bootstrap interval, overall and by
    source (fleurs / dir / cap) and, where the system has them, by voice;
  * CER on the prompts where EVERY system has a valid clip (the like-for-like row);
  * paired differences between systems on their shared valid prompts.

Nothing is dropped quietly: a system that is missing or unfinished is printed as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap  # noqa: E402
from geetha.scoring.metrics import cer  # noqa: E402

RUNS = Path(r"E:\Geetha\runs")
BENCH = Path(r"E:\Geetha\work\bench_ta_v2\bench_ta_v2.json")
FILES = {"geetha": RUNS / "m3_bench_ours" / "bench_geetha.json",
         "base": RUNS / "m3_bench_ours" / "bench_base.json",
         "rumik_oss_1": RUNS / "m3_bench_rivals" / "bench_rumik_oss_1.json",
         "snortts_indic_v0": RUNS / "m3_bench_rivals" / "bench_snortts_indic_v0.json"}
PAIRS = (("geetha", "base"), ("geetha", "rumik_oss_1"), ("geetha", "snortts_indic_v0"),
         ("base", "rumik_oss_1"), ("base", "snortts_indic_v0"))


def summary(counts: list) -> dict:
    if len(counts) < 2:
        return {"n": len(counts)}
    ci = bootstrap_ci(counts)
    return {"n": len(counts), "cer": round(100 * ci.point, 2),
            "ci": [round(100 * ci.low, 2), round(100 * ci.high, 2)]}


def cell(block: dict) -> str:
    if "cer" not in block:
        return f"- (n={block['n']})"
    return f"{block['cer']:.2f} [{block['ci'][0]:.2f}, {block['ci'][1]:.2f}] (n={block['n']})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"E:\Geetha\results\bench_v2"))
    args = parser.parse_args()
    prompts = {p["id"]: p for p in json.loads(BENCH.read_text(encoding="utf-8"))["prompts"]}
    reports = {}
    for system, path in FILES.items():
        if path.exists():
            reports[system] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"MISSING: {system} ({path})")
    if not reports:
        return 1
    lines = [f"# Benchmark v2: {len(prompts)} pure-Tamil sentences", "",
             "Single shot, one fixed seed per prompt, each system at its own card's "
             "settings. Reference = the written text. CER in %, 95% bootstrap interval.", "",
             "## Validity", "",
             "| System | Valid | Silent or ran to the cap | Not run | Finished | Failure |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    result: dict = {"n_prompts": len(prompts), "validity": {}, "judges": {}}
    for system, report in reports.items():
        clips = report["clips"]
        good = sum(1 for c in clips.values() if c["valid"])
        skipped = sum(1 for c in clips.values() if c["reason"] == "not_run")
        missing = len(prompts) - len(clips)
        bad = len(clips) - good - skipped
        result["validity"][system] = {"valid": good, "invalid": bad,
                                      "not_run": skipped + missing,
                                      "finished": bool(report.get("finished")),
                                      "failure": report.get("failure"),
                                      "invalid_reasons": {i: c["reason"] for i, c in clips.items()
                                                          if not c["valid"]
                                                          and c["reason"] != "not_run"}}
        lines.append(f"| {system} | {good}/{len(prompts)} | {bad} | {skipped + missing} | "
                     f"{bool(report.get('finished'))} | {report.get('failure') or ''} |")

    judges = sorted({j for r in reports.values() for j in r["hypotheses"]})
    for judge in judges:
        scored = {s: {i: cer(h, prompts[i]["text"]) for i, h in r["hypotheses"].get(judge, {}).items()
                      if i in prompts} for s, r in reports.items()}
        everyone = set(prompts)
        for system in scored:
            everyone &= set(scored[system])
        block: dict = {"systems": {}, "paired": {}, "shared_by_all": len(everyone)}
        lines += ["", f"## Judge: {judge}", "",
                  f"| System | All valid clips | The {len(everyone)} prompts valid for every system "
                  "| fleurs | dir | cap | anitha | arun |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for system, per_clip in scored.items():
            ids = list(per_clip)
            entry = {"all": summary([per_clip[i] for i in ids]),
                     "shared_by_all": summary([per_clip[i] for i in sorted(everyone)])}
            for source in ("fleurs", "dir", "cap"):
                entry[source] = summary([per_clip[i] for i in ids
                                         if prompts[i]["source"] == source])
            for voice in ("anitha", "arun"):
                entry[voice] = summary([per_clip[i] for i in ids if prompts[i]["voice"] == voice])
            block["systems"][system] = entry
            own_voices = system in ("geetha", "base")
            lines.append(f"| {system} | " + " | ".join(
                cell(entry[k]) if (k not in ("anitha", "arun") or own_voices) else "n/a"
                for k in ("all", "shared_by_all", "fleurs", "dir", "cap", "anitha", "arun"))
                + " |")
        lines += ["", "| Paired difference (first minus second; negative = first is better) "
                      "| Delta CER | 95% interval | p | n |", "| --- | ---: | --- | ---: | ---: |"]
        for a, b in PAIRS:
            if a not in scored or b not in scored:
                continue
            common = sorted(set(scored[a]) & set(scored[b]))
            if len(common) < 2:
                continue
            p = paired_bootstrap([scored[a][i] for i in common], [scored[b][i] for i in common])
            block["paired"][f"{a}_minus_{b}"] = {
                "n": len(common), "delta": round(100 * p.delta, 2),
                "ci": [round(100 * p.low, 2), round(100 * p.high, 2)], "p_value": p.p_value}
            lines.append(f"| {a} - {b} | {100 * p.delta:+.2f} | [{100 * p.low:+.2f}, "
                         f"{100 * p.high:+.2f}] | {p.p_value:.4f} | {len(common)} |")
        block["per_clip"] = {s: {i: round(100 * (c.rate or 0), 2) for i, c in per_clip.items()}
                             for s, per_clip in scored.items()}
        result["judges"][judge] = block

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "bench_v2.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
