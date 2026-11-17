r"""ETL step 6: SNAC-encode the training manifests (D-036, D-037).

    py -3.11 scripts\run_snac_encode.py --stage A [--limit N] [--force]
    py -3.11 scripts\run_snac_encode.py --roundtrip 20

Runs on this PC, not Kaggle: the codes are a couple of hundred MB where the audio is
22-88 GB, so it is cheaper by every measure to encode where the audio already lives and
ship only codes (D-037).

Each clip goes: raw parquet -> decode -> mono -> manifest gain (-23 dBFS target,
attenuation only) -> anti-aliased 2:1 decimation to 24 kHz -> SNAC -> drop consecutive
duplicate frames -> flatten to 7 tokens per frame with position offsets.

Output
    processed\tts_codes\stage<A|B>\<shard>.parquet    tokens per clip, plus conditioning
    processed\tts_codes\stage<A|B>\checksums.csv      sha256 per shard
    work\snac_roundtrip\                              --roundtrip decodes, for listening

Resumable per shard, because E: has disconnected twice.

The guard that matters: the flattening is validated on every clip by
``flatten_frames``, which raises if SNAC's three codebooks are not in the 1:2:4 shape
the layout assumes, and the tokens-per-second is reported against the contract's
"82 tokens ~= 1 second" rule of thumb. A silent layout mismatch is the failure mode
``token_contract.md`` warns about -- it trains happily and decodes to noise.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.audio.snac_input import SNAC_SAMPLE_RATE, prepare_for_snac  # noqa: E402
from geetha.audio.spectral import spectral_similarity  # noqa: E402
from geetha.audio.snac_tokens import (  # noqa: E402
    TOKENS_PER_FRAME,
    drop_duplicate_frames,
    flatten_frames,
    unflatten_frames,
)

ROOT = Path(r"E:\Geetha")
RAW = {
    "indicvoices_r": (ROOT / "raw" / "hf" / "indicvoices_r" / "Tamil", "audio"),
    "rasa": (ROOT / "raw" / "hf" / "rasa" / "Tamil", "audio"),
}
CODES = ROOT / "processed" / "tts_codes"
ROUNDTRIP = ROOT / "work" / "snac_roundtrip"
SNAC_REPO = "hubertsiuzdak/snac_24khz"

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
    pa.field("repeat", pa.int32()),
    pa.field("n_frames", pa.int32()),
    pa.field("n_tokens", pa.int32()),
    pa.field("tokens", pa.list_(pa.int32())),
])


def load_model():
    import torch
    from snac import SNAC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained(SNAC_REPO).eval().to(device)
    print(f"SNAC {SNAC_REPO} on {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # Self-check the checkpoint against token_contract.md §3 before encoding 127 h with
    # it. flatten_frames validates the 1:2:4 *ratio* on every clip, but a checkpoint
    # with different encoder rates could satisfy that ratio at the wrong frame rate and
    # produce a stream the model was never trained to read -- the silent failure the
    # contract warns about. One second of audio must give ~12 c0 frames.
    with torch.inference_mode():
        probe = model.encode(torch.zeros(1, 1, SNAC_SAMPLE_RATE, device=device))
    frames = [code.reshape(-1).numel() for code in probe]
    if len(frames) != 3 or frames[1] != 2 * frames[0] or frames[2] != 4 * frames[0]:
        raise SystemExit(f"{SNAC_REPO} gave codebook lengths {frames}; the layout needs "
                         "three codebooks at a 1:2:4 ratio")
    if not 11 <= frames[0] <= 13:
        raise SystemExit(
            f"{SNAC_REPO} gave {frames[0]} coarse frames for 1 s of audio; the contract's "
            "layout assumes ~12 Hz (2,048 samples per c0 frame). This is the wrong "
            "checkpoint -- encoding with it would train the model on a stream it cannot read."
        )
    print(f"  checkpoint self-check ok: {frames[0]}/{frames[1]}/{frames[2]} codes per second "
          f"= {sum(frames)} tokens/s before dedup")
    return model, device


def encode_clip(model, device, audio: np.ndarray) -> tuple[list[int], int, int]:
    """SNAC-encode one 24 kHz mono clip into the model's flattened token stream."""
    import torch

    with torch.inference_mode():
        tensor = torch.from_numpy(audio).to(device).reshape(1, 1, -1)
        codes = model.encode(tensor)
    books = [code.reshape(-1).tolist() for code in codes]
    if len(books) != 3:
        raise ValueError(f"expected 3 SNAC codebooks, got {len(books)}; the 7-token "
                         "layout in token_contract.md assumes three at a 1:2:4 ratio")
    raw_frames = len(books[0])
    c0, c1, c2 = drop_duplicate_frames(*books)
    return flatten_frames(c0, c1, c2), len(c0), raw_frames


def manifest_rows(stage: str) -> dict[tuple[str, str], list[dict]]:
    path = ROOT / "processed" / f"tts_stage{stage}" / "manifest.parquet"
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        grouped[(row["source"], row["shard"])].append(row)
    return grouped


def read_audio(blob: dict) -> tuple[np.ndarray, int]:
    wave, rate = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=True)
    return wave, rate


def encode_stage(stage: str, limit: int | None, force: bool) -> int:
    model, device = load_model()
    out_dir = CODES / f"stage{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = manifest_rows(stage)
    keys = sorted(grouped)
    if limit:
        keys = keys[:limit]
    print(f"stage {stage}: {sum(len(grouped[k]) for k in keys):,} clips over {len(keys)} shards")

    total_tokens = total_frames = total_raw_frames = 0
    total_seconds = 0.0
    done = 0

    for position, (source, shard) in enumerate(keys, start=1):
        target = out_dir / f"{source}__{shard}"
        if target.exists() and not force:
            continue
        directory, audio_column = RAW[source]
        table = pq.read_table(directory / shard, columns=[audio_column])
        audio_column_values = table[audio_column].to_pylist()

        rows = []
        for row in grouped[(source, shard)]:
            wave, rate = read_audio(audio_column_values[row["row_index"]])
            audio = prepare_for_snac(wave, rate, gain_db=row["gain_db"])
            tokens, frames, raw_frames = encode_clip(model, device, audio)
            rows.append({
                **{key: row[key] for key in
                   ("source", "shard", "row_index", "speaker", "style", "env", "text",
                    "duration_s", "gain_db", "repeat")},
                "n_frames": frames,
                "n_tokens": len(tokens),
                "tokens": tokens,
            })
            total_tokens += len(tokens)
            total_frames += frames
            total_raw_frames += raw_frames
            total_seconds += row["duration_s"] or 0.0

        staging = target.with_suffix(".parquet.part")
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), staging)
        staging.replace(target)
        done += 1
        if position % 10 == 0 or position == len(keys):
            rate_hint = total_tokens / max(total_seconds, 1e-9)
            print(f"  [{position}/{len(keys)}] {source}/{shard}: "
                  f"{total_tokens:,} tokens, {rate_hint:.1f} tok/s", flush=True)

    if total_seconds:
        print(f"\nencoded {done} shards, {total_frames:,} frames, {total_tokens:,} tokens "
              f"over {total_seconds / 3600:.1f} h")
        print(f"tokens per second: {total_tokens / total_seconds:.2f} "
              f"(contract rule of thumb: 82)")
        print(f"before dedup: {total_raw_frames:,} frames = "
              f"{TOKENS_PER_FRAME * total_raw_frames / total_seconds:.2f} tok/s; dedup removed "
              f"{1 - total_frames / max(total_raw_frames, 1):.1%} of frames")

    write_checksums(out_dir)
    return 0


def write_checksums(out_dir: Path) -> None:
    lines = ["shard,sha256,bytes"]
    for path in sorted(out_dir.glob("*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{path.name},{digest},{path.stat().st_size}")
    (out_dir / "checksums.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checksums: {len(lines) - 1} shards -> {out_dir / 'checksums.csv'}")


def roundtrip(count: int, stage: str) -> int:
    """Decode a sample of encoded clips back to wav, for the step 6 listening check."""
    import torch

    model, device = load_model()
    ROUNDTRIP.mkdir(parents=True, exist_ok=True)
    shards = sorted((CODES / f"stage{stage}").glob("*.parquet"))
    if not shards:
        print(f"!! nothing encoded yet in {CODES / f'stage{stage}'}", file=sys.stderr)
        return 1

    taken = 0
    scores: list[float] = []
    lines = ["file,source,shard,row_index,speaker,style,n_tokens,decoded_s,manifest_s,"
             "spectral_similarity"]
    for path in shards:
        if taken >= count:
            break
        for row in pq.read_table(path).to_pylist():
            if taken >= count:
                break
            # Round-tripping through unflatten is the point: it re-checks the band and
            # the position blocks, so a layout error cannot pass this step quietly.
            c0, c1, c2 = unflatten_frames(row["tokens"])
            with torch.inference_mode():
                codes = [torch.tensor(book, dtype=torch.long, device=device).reshape(1, -1)
                         for book in (c0, c1, c2)]
                audio = model.decode(codes).reshape(-1).cpu().numpy()
            # Score the decode against the original clip, prepared exactly as the
            # encoder saw it. A wrong token layout decodes to noise, which scores near
            # zero -- so this catches the silent failure without needing ears.
            directory, audio_column = RAW[row["source"]]
            blob = pq.read_table(directory / row["shard"],
                                 columns=[audio_column])[audio_column][row["row_index"]].as_py()
            original_wave, original_rate = read_audio(blob)
            original = prepare_for_snac(original_wave, original_rate, gain_db=row["gain_db"])
            score = spectral_similarity(original, audio, SNAC_SAMPLE_RATE)
            scores.append(score)

            name = f"{taken:02d}_{row['source']}_{row['row_index']}_{row['style']}.wav"
            sf.write(ROUNDTRIP / name, audio, SNAC_SAMPLE_RATE)
            sf.write(ROUNDTRIP / name.replace(".wav", "_original.wav"), original, SNAC_SAMPLE_RATE)
            (ROUNDTRIP / name.replace(".wav", ".txt")).write_text(row["text"], encoding="utf-8")
            lines.append(f"{name},{row['source']},{row['shard']},{row['row_index']},"
                         f"{row['speaker']},{row['style']},{row['n_tokens']},"
                         f"{len(audio) / SNAC_SAMPLE_RATE:.2f},{row['duration_s']:.2f},"
                         f"{score:.4f}")
            taken += 1
    (ROUNDTRIP / "index.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {taken} round-trip decodes (+ originals and text) to {ROUNDTRIP}")

    finite = [score for score in scores if not np.isnan(score)]
    if finite:
        array = np.array(finite)
        print(f"spectral similarity to the original: median {np.median(array):.3f}, "
              f"min {array.min():.3f}, max {array.max():.3f} over {array.size} clips")
        if array.min() < 0.5:
            print("!! at least one decode is unlike its original. A wrong token layout "
                  "decodes to noise -- check the flattening before trusting these codes.",
                  file=sys.stderr)
            return 1
    if len(finite) < len(scores):
        print(f"!! {len(scores) - len(finite)} clips scored NaN (a flat spectrum, i.e. "
              "silence in or out)", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], default="A")
    parser.add_argument("--limit", type=int, help="first N shards only")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--roundtrip", type=int, metavar="N",
                        help="decode N already-encoded clips back to wav instead of encoding")
    args = parser.parse_args()

    if args.roundtrip:
        return roundtrip(args.roundtrip, args.stage)
    return encode_stage(args.stage, args.limit, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
