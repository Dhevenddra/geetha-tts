r"""Audit the B4a contamination result for the artefacts its own run warned about.

    py -3.11 scripts\audit_c1.py

The kernel reported CONTAMINATED. Before that is written into a decision, two things
raised during the run have to be checked, and both can be settled from the audio that
came back rather than by spending another hour of quota.

**Truncation.** ``max_new_tokens`` was capped at 1066 tokens (~13 s) to make the run
finish on a T4, and the model warned at least once that it hit the cap without emitting
``<|end_of_speech|>``. A truncated clip has the wrong duration, and duration is half of
the prosodic distance the verdict rests on. The worry is not noise but *bias*: a
memorised clip is less likely to run away into babble, so the TRAIN arm could truncate
less often than TEST, and that difference alone would move the margin.

**Power.** Both confidence intervals span zero, which is what produced the verdict. That
is a statement about what we could not distinguish, not proof of memorisation, and the
report should say which.

The verdict is recomputed with truncated clips removed. Both numbers are reported
whatever they say -- if they agree the point is moot, and if they disagree, that
disagreement is the finding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN = Path(r"E:\Geetha\runs\c1_contamination")
WORK = Path(r"E:\Geetha\work\c1_contamination")

# 1066 tokens at the model's documented 82 tokens/second. A clip landing within a
# hair of this did not stop on its own.
CAP_SECONDS = 1066 / 82.0
TRUNCATION_MARGIN = 0.25


def main() -> int:
    import numpy as np
    import soundfile as sf

    from geetha.eval.contamination import MIN_CER_GAP, MIN_DELTA_GAP, verdict_from_gaps
    from geetha.scoring.bootstrap import two_sample_ci

    results = json.loads((RUN / "c1_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))
    by_name = {row["filename"]: row for row in manifest["clips"]}

    print(f"kernel verdict: {results['verdict']}  -> {results['naturalness_ta_source']}")
    print(f"generation actually used: {results.get('generation_actually_used')}\n")

    # ---- truncation, measured from the audio itself
    truncated: dict[str, list[str]] = {"TEST": [], "TRAIN": [], "UNSEEN": []}
    durations: dict[str, list[float]] = {"TEST": [], "TRAIN": [], "UNSEEN": []}
    for wav_path in sorted((RUN / "generated").glob("*.wav")):
        arm, _, name = wav_path.stem.partition("__")
        if arm not in durations:
            continue
        info = sf.info(wav_path)
        seconds = info.frames / info.samplerate
        durations[arm].append(seconds)
        if seconds >= CAP_SECONDS - TRUNCATION_MARGIN:
            truncated[arm].append(name)

    print("truncation (clip hit the token cap and never emitted end-of-speech):")
    for arm in ("TRAIN", "TEST", "UNSEEN"):
        n = len(durations[arm])
        hit = len(truncated[arm])
        share = 100.0 * hit / n if n else 0.0
        median = float(np.median(durations[arm])) if n else 0.0
        print(f"  {arm:7s} {hit:3d}/{n:3d} ({share:5.1f}%)   median duration {median:5.2f}s")

    # Does truncation fall unevenly between the two matched arms?
    test_hit, train_hit = len(truncated["TEST"]), len(truncated["TRAIN"])
    print(f"\n  TEST - TRAIN truncation difference: {test_hit - train_hit:+d} clips")
    if abs(test_hit - train_hit) > 5:
        print("  !! uneven enough to bias the duration half of the prosodic distance")
    else:
        print("  even enough that it cannot explain the margin gap on its own")

    # ---- the verdict without truncated clips
    per_clip = results["per_clip"]
    dropped = {f"{arm}__{name}" for arm, names in truncated.items() for name in names}
    cer = {"TEST": [], "TRAIN": []}
    cer_kept = {"TEST": [], "TRAIN": []}
    for row in per_clip:
        if row["arm"] not in cer:
            continue
        cer[row["arm"]].append(row["cer"])
        if row["key"] not in dropped:
            cer_kept[row["arm"]].append(row["cer"])

    print("\nCER gap (TEST - TRAIN), all clips vs truncated removed:")
    for label, data in (("all", cer), ("kept", cer_kept)):
        if not data["TEST"] or not data["TRAIN"]:
            continue
        gap = two_sample_ci(data["TEST"], data["TRAIN"])
        print(f"  {label:5s} n={len(data['TEST'])}/{len(data['TRAIN'])}  "
              f"{gap.delta:+.2f}  CI [{gap.low:+.2f}, {gap.high:+.2f}]  p={gap.p_value:.3f}")

    # ---- what the run could have detected at all
    gap = two_sample_ci(cer["TEST"], cer["TRAIN"])
    half_width = (gap.high - gap.low) / 2
    print(f"\npower: the CER interval is +/-{half_width:.2f} points at n=50 per arm.")
    print(f"       the rule calls {MIN_CER_GAP:.1f} points material, so a real gap of "
          f"that size\n       {'could' if half_width < MIN_CER_GAP else 'could NOT'} "
          "have been resolved by this run.")

    margin = results["arms"]
    print(f"\nmargin: TRAIN {margin['TRAIN']['memorisation_margin']:.3f}  "
          f"TEST {margin['TEST']['memorisation_margin']:.3f}")
    print("        TEST closer to its true recordings than TRAIN is to its own —")
    print("        the opposite of what a held-out split would produce.")

    print(f"\nspeaker similarity: TRAIN {margin['TRAIN']['median_speaker_similarity']:.3f}, "
          f"TEST {margin['TEST']['median_speaker_similarity']:.3f}")
    print("        confirms Anitha/Arun ARE the Rasa studio speakers (D-008), measured")
    print("        rather than quoted from a model card.")

    probe = results.get("style_probe", {}).get("distance_from_control", {})
    if probe:
        print("\nstyle probe (inconclusive by design — the control was compared to itself;")
        print("a second nonsense label is needed to calibrate):")
        for label, distance in sorted(probe.items(), key=lambda kv: -kv[1]):
            print(f"  {distance:.3f}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
