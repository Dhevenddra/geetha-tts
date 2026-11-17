r"""Split a two-person recording into each person's turns, keeping pauses and breaths.

    py -3.11 scripts\diarize_two_speakers.py <mono16k.wav> <full_quality.wav> <out_dir>

Energy VAD -> windows of at most 3 s -> WeSpeaker embedding per window -> 2-way clustering ->
smoothing over time -> turns. A turn is a stretch where only one speaker is active; it is
exported from the full-quality file with its internal pauses intact, because the pauses are
where the breaths are. Nothing is denoised or trimmed inside a turn.

Writes turns.json, six sample clips per speaker (for a human to say which is which), and
one folder of turns per speaker. Local only; nothing is uploaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geetha.audio.breath import HOP_SECONDS, frame_features  # noqa: E402

MODEL = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"
WINDOW_S, MIN_WINDOW_S = 3.0, 0.8
MERGE_GAP_S = 0.35          # speech chunks closer than this are one chunk
TURN_GAP_S = 2.0            # same speaker across a silence up to this long is one turn
MIN_TURN_S = 2.0


def speech_chunks(wave: np.ndarray) -> list[tuple[float, float]]:
    energy = frame_features(wave)["energy_db"]
    speech = energy > float(np.percentile(energy, 95)) - 30.0
    padded = np.concatenate([[False], speech, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    chunks: list[list[float]] = []
    for start, end in zip(edges[::2] * HOP_SECONDS, edges[1::2] * HOP_SECONDS):
        if chunks and start - chunks[-1][1] < MERGE_GAP_S:
            chunks[-1][1] = end
        else:
            chunks.append([start, end])
    return [(a, b) for a, b in chunks if b - a >= 0.3]


def windows(chunks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for start, end in chunks:
        count = max(1, int(np.ceil((end - start) / WINDOW_S)))
        step = (end - start) / count
        for i in range(count):
            a, b = start + i * step, start + (i + 1) * step
            if b - a >= MIN_WINDOW_S:
                out.append((a, b))
    return out


def embed(wave: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """WeSpeaker ResNet34 (VoxCeleb, CC BY 4.0) through ONNX: no pickled weights involved.
    The WavLM x-vector checkpoints exist only as .bin, which transformers refuses to load on
    torch < 2.6 (CVE-2025-32434); that guard is respected, not bypassed."""
    import onnxruntime as ort
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL, MODEL_FILE, token=False)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    vectors = []
    for i, (a, b) in enumerate(spans):
        piece = torch.from_numpy(wave[int(a * 16000): int(b * 16000)]).unsqueeze(0) * (1 << 15)
        feats = kaldi.fbank(piece, num_mel_bins=80, frame_length=25, frame_shift=10,
                            dither=0.0, energy_floor=0.0, sample_frequency=16000)
        feats = (feats - feats.mean(0, keepdim=True)).unsqueeze(0).numpy()
        vector = session.run(None, {name: feats})[0][0]
        vectors.append(vector / (np.linalg.norm(vector) + 1e-9))
        if i % 100 == 0:
            print(f"  embedded {i}/{len(spans)}", flush=True)
    return np.stack(vectors)


def two_means(vectors: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Cosine 2-means, best of 8 starts. Returns labels and the mean silhouette-like margin."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(8):
        centres = vectors[rng.choice(len(vectors), 2, replace=False)]
        for _ in range(50):
            labels = (vectors @ centres.T).argmax(1)
            if len(set(labels)) < 2:
                break
            new = np.stack([vectors[labels == k].mean(0) for k in (0, 1)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, centres):
                break
            centres = new
        sims = vectors @ centres.T
        score = float(np.abs(sims[:, 0] - sims[:, 1]).mean())
        if best is None or score > best[1]:
            best = (sims.argmax(1), score, centres)
    return best[0], best[1]


def main(argv: list[str]) -> int:
    import soundfile as sf

    analysis, full, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    os.environ.setdefault("HF_HOME", r"E:\Geetha\.hf_home")
    wave, rate = sf.read(str(analysis), dtype="float32")
    assert rate == 16000 and wave.ndim == 1
    chunks = speech_chunks(wave)
    spans = windows(chunks)
    print(f"{len(chunks)} speech chunks, {len(spans)} windows, "
          f"{sum(b - a for a, b in chunks) / 60:.1f} min of speech in {len(wave) / rate / 60:.1f} min")
    vectors = embed(wave, spans)
    labels, margin = two_means(vectors)
    # smooth: a single window that disagrees with both neighbours is flipped
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i] and spans[i][0] - spans[i - 1][1] < 1.0 \
                and spans[i + 1][0] - spans[i][1] < 1.0:
            labels[i] = labels[i - 1]
    print(f"cluster margin {margin:.3f} (near 0 = the two clusters are not distinct)")

    turns: list[dict] = []
    for (a, b), label in zip(spans, labels):
        if turns and turns[-1]["speaker"] == int(label) and a - turns[-1]["end"] <= TURN_GAP_S:
            turns[-1]["end"] = b
        else:
            turns.append({"speaker": int(label), "start": a, "end": b})
    totals = {k: sum(t["end"] - t["start"] for t in turns if t["speaker"] == k) for k in (0, 1)}
    names = {max(totals, key=totals.get): "A_talks_more", min(totals, key=totals.get): "B_talks_less"}
    for k in (0, 1):
        mine = [t for t in turns if t["speaker"] == k]
        print(f"{names[k]}: {totals[k] / 60:.1f} min in {len(mine)} turns, "
              f"{sum(1 for t in mine if t['end'] - t['start'] >= MIN_TURN_S)} of them >= {MIN_TURN_S:.0f} s")

    source, source_rate = sf.read(str(full), dtype="float32")
    if source.ndim > 1:
        source = source.mean(axis=1)
    out.mkdir(parents=True, exist_ok=True)
    record = []
    for k in (0, 1):
        folder = out / names[k]
        folder.mkdir(exist_ok=True)
        mine = [t for t in turns if t["speaker"] == k and t["end"] - t["start"] >= MIN_TURN_S]
        for n, turn in enumerate(mine, 1):
            previous = max([t["end"] for t in turns if t["end"] <= turn["start"] and t is not turn], default=0.0)
            following = min([t["start"] for t in turns if t["start"] >= turn["end"] and t is not turn],
                            default=len(source) / source_rate)
            # take the silence around the turn too, up to 0.6 s, but never into the other speaker
            a = max(previous + 0.05, turn["start"] - 0.6)
            b = min(following - 0.05, turn["end"] + 0.3)
            name = f"{names[k][0]}_{n:03d}_{int(turn['start'] // 60):02d}m{int(turn['start'] % 60):02d}s.wav"
            sf.write(str(folder / name), source[int(a * source_rate): int(b * source_rate)], source_rate)
            record.append({"speaker": names[k], "file": name, "start": round(a, 2), "end": round(b, 2)})
        longest = sorted(mine, key=lambda t: t["start"])
        pick = [longest[i] for i in np.linspace(0, len(longest) - 1, min(6, len(longest))).astype(int)]
        for n, turn in enumerate(pick, 1):
            a, b = turn["start"], min(turn["end"], turn["start"] + 12.0)
            sf.write(str(out / f"sample_{names[k]}_{n}.wav"),
                     source[int(a * source_rate): int(b * source_rate)], source_rate)
    (out / "turns.json").write_text(json.dumps({"margin": margin, "turns": record}, indent=1),
                                    encoding="utf-8")
    print(f"wrote {len(record)} turns to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
