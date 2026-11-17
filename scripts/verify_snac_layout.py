r"""Prove the SNAC token layout against the codec itself -- ETL step 6's real check.

    py -3.11 scripts\verify_snac_layout.py [--clips 20]

Why a plain round-trip score is not the test
--------------------------------------------
The obvious check -- decode our tokens and compare with the original -- conflates three
different things:

1. whether our 7-token flattening is right (what we actually want to know);
2. how lossy SNAC is at ~1 kbps (nothing to do with us);
3. that dropping duplicate frames makes the decode *shorter* than the original, so a
   frame-aligned comparison is penalised for time compression even when every token is
   perfect.

Measured on 20 real clips, (2) and (3) together put the naive score around 0.68, which
is neither obviously right nor obviously wrong. A number you cannot interpret is not a
measurement.

So this compares against the codec's **own ceiling** instead, which removes all three
confounds:

    A = similarity(original, decode(encode(x)))                       codec alone
    B = similarity(original, decode(unflatten(flatten(encode(x)))))   through our layout

Both carry SNAC's loss and neither is deduped, so **A and B must be equal**. If the
flattening, the position offsets or the 1:2:4 interleave were wrong, B would collapse
while A stayed put. On top of that the codes themselves are compared exactly: a correct
flatten/unflatten pair is lossless, so any difference at all is a bug.

Dedup is then measured separately, as a duration ratio, because it is a deliberate part
of the contract rather than an error.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.audio.snac_input import SNAC_SAMPLE_RATE, prepare_for_snac  # noqa: E402
from geetha.audio.snac_tokens import drop_duplicate_frames, flatten_frames, unflatten_frames  # noqa: E402
from geetha.audio.spectral import spectral_similarity  # noqa: E402

ROOT = Path(r"E:\Geetha")
RAW = {
    "indicvoices_r": ROOT / "raw" / "hf" / "indicvoices_r" / "Tamil",
    "rasa": ROOT / "raw" / "hf" / "rasa" / "Tamil",
}
MANIFEST = ROOT / "processed" / "tts_stageA" / "manifest.parquet"
REPORT = ROOT / "work" / "snac_layout_verification.md"


def main() -> int:
    import torch
    from snac import SNAC

    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    rows = pq.read_table(MANIFEST).to_pylist()
    # Spread the sample across both sources and a range of durations rather than taking
    # the first N, which would all be one speaker in one shard.
    rows.sort(key=lambda row: (row["source"], row["duration_s"]))
    picked = rows[:: max(1, len(rows) // args.clips)][: args.clips]

    codec_only: list[float] = []
    through_layout: list[float] = []
    nondeterminism: list[float] = []
    control: list[float] = []
    duration_ratio: list[float] = []
    code_mismatches = 0

    for row in picked:
        blob = pq.read_table(RAW[row["source"]] / row["shard"],
                             columns=["audio"])["audio"][row["row_index"]].as_py()
        wave, rate = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=True)
        audio = prepare_for_snac(wave, rate, gain_db=row["gain_db"])

        with torch.inference_mode():
            tensor = torch.from_numpy(audio).to(device).reshape(1, 1, -1)
            codes = model.encode(tensor)
            plain = model.decode(codes).reshape(-1).cpu().numpy()
            # The same codes decoded a second time. SNAC's decode is not
            # bit-reproducible on CUDA (measured: the same tensors twice differ by up
            # to 0.08 in sample value), so this is the noise floor any A-vs-B
            # comparison has to be judged against rather than against zero.
            plain_again = model.decode(codes).reshape(-1).cpu().numpy()

        books = [code.reshape(-1).tolist() for code in codes]
        # Exact check first: flatten then unflatten must be the identity.
        recovered = unflatten_frames(flatten_frames(*books))
        if list(recovered) != [list(book) for book in books]:
            code_mismatches += 1

        with torch.inference_mode():
            rebuilt = [torch.tensor(book, dtype=torch.long, device=device).reshape(1, -1)
                       for book in recovered]
            ours = model.decode(rebuilt).reshape(-1).cpu().numpy()

        first = spectral_similarity(audio, plain, SNAC_SAMPLE_RATE)
        codec_only.append(first)
        through_layout.append(spectral_similarity(audio, ours, SNAC_SAMPLE_RATE))
        nondeterminism.append(
            abs(first - spectral_similarity(audio, plain_again, SNAC_SAMPLE_RATE)))

        # Negative control. Rotate the seven frame positions by one before decoding:
        # a stream that is misaligned in exactly the way a layout bug would misalign it.
        # If this does NOT collapse, the comparison above has no power to detect a
        # layout error and must not be reported as evidence of anything.
        tokens = flatten_frames(*books)
        rotated = [tokens[(index // 7) * 7 + (index % 7 + 1) % 7] for index in range(len(tokens))]
        try:
            wrong_books = unflatten_frames(rotated)
            with torch.inference_mode():
                wrong_tensors = [torch.tensor(book, dtype=torch.long, device=device).reshape(1, -1)
                                 for book in wrong_books]
                wrong_audio = model.decode(wrong_tensors).reshape(-1).cpu().numpy()
            control.append(spectral_similarity(audio, wrong_audio, SNAC_SAMPLE_RATE))
        except ValueError:
            # unflatten refused the rotated stream outright, which is also a detection.
            control.append(float("nan"))

        deduped = drop_duplicate_frames(*books)
        with torch.inference_mode():
            shrunk = [torch.tensor(book, dtype=torch.long, device=device).reshape(1, -1)
                      for book in deduped]
            shortened = model.decode(shrunk).reshape(-1).cpu().numpy()
        duration_ratio.append(len(shortened) / max(len(plain), 1))

    a = np.array(codec_only)
    b = np.array(through_layout)
    ratio = np.array(duration_ratio)
    gap = float(np.abs(a - b).max())
    floor = float(np.max(nondeterminism))

    # The decisive test is the exact one: a correct flatten/unflatten pair is lossless,
    # so the codes must come back identical. The spectral comparison corroborates it,
    # and is judged against the decoder's own run-to-run noise -- requiring better than
    # that would be requiring the impossible.
    control_array = np.array(control)
    rejected = int(np.isnan(control_array).sum())
    control_scores = control_array[~np.isnan(control_array)]
    # The check has power if a deliberately wrong layout is either refused outright or
    # scores clearly worse than the correct one.
    has_power = rejected == len(control_array) or (
        control_scores.size > 0 and control_scores.max() < b.min() - 0.05)

    # The verdict rests on the exact comparison, because that is the actual claim: a
    # correct flatten/unflatten pair is lossless, which is binary and admits no
    # tolerance. The spectral numbers corroborate it and are reported with the
    # decoder's own noise floor beside them -- gating on them would be gating on a
    # quantity whose resolution (9.5e-04 here) is no better than the difference being
    # measured, which is how an earlier version of this script produced a meaningless
    # FAIL on a layout that was provably exact.
    verdict = "PASS" if code_mismatches == 0 and has_power else "FAIL"
    lines = [
        "# ETL step 6: SNAC token layout verification",
        "",
        f"**{verdict}** on {len(picked)} clips sampled across both sources and the duration range.",
        "",
        f"- flatten -> unflatten returned the codes exactly, clip for clip: "
        f"**{len(picked) - code_mismatches}/{len(picked)}**",
        f"- codec alone, similarity to the original: median **{np.median(a):.4f}** "
        f"(min {a.min():.4f}, max {a.max():.4f})",
        f"- through our 7-token layout: median **{np.median(b):.4f}** "
        f"(min {b.min():.4f}, max {b.max():.4f})",
        f"- largest difference between the two, any clip: **{gap:.2e}**",
        f"- the decoder's own run-to-run noise on the same codes: **{floor:.2e}**, the "
        f"same order as the line above -- which is why the verdict does not rest on it",
        "",
        "## Negative control: does this check have any power?",
        "",
        f"- a deliberately misaligned stream (frame positions rotated by one) scored "
        + (f"**refused outright on all {rejected} clips**" if rejected == len(control_array)
           else f"median **{np.median(control_scores):.4f}**, max **{control_scores.max():.4f}**"
                + (f", and was refused outright on {rejected} clips" if rejected else "")),
        f"- the correct layout's worst clip scored **{b.min():.4f}**",
        f"- so a layout error of this kind {'IS' if has_power else 'IS NOT'} detectable here",
        "",
        "SNAC's decode is not bit-reproducible on CUDA: decoding one set of codes twice",
        "gives audio differing by up to 0.08 in sample value. So A and B cannot be",
        "compared against zero, and the exact code comparison above is what actually",
        "settles the question -- a lossless flatten/unflatten is an exact claim.",
        "",
        "The two medians being equal is the result that matters. Both carry SNAC's own",
        "loss at ~1 kbps, so neither is close to 1.0 and neither should be; what a wrong",
        "interleave or a wrong position offset would do is collapse the second while",
        "leaving the first untouched.",
        "",
        "## What dropping duplicate frames costs",
        "",
        f"- decoded duration after dedup, as a fraction of before: median "
        f"**{np.median(ratio):.3f}** (min {ratio.min():.3f}, max {ratio.max():.3f})",
        "",
        "Dedup is in `token_contract.md` as something the base model's own training data",
        "went through, so our streams have to match it. It is also why a naive",
        "original-vs-decode score reads low: the decode is genuinely shorter, and a",
        "frame-aligned comparison charges that as infidelity. Measured here rather than",
        "folded into the layout check.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
