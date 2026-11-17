r"""Is a generated clip still in the voice it was asked for?

Cosine similarity between a clip's speaker embedding and the centroid of a set of
reference clips, with WeSpeaker ResNet34 (VoxCeleb, CC BY 4.0) through ONNX - the model
already used by ``diarize_two_speakers.py``. A reference clip is scored against the
centroid of the OTHER references (leave-one-out), so the references' own spread is the
noise floor a test clip is read against.

A similarity is only meaningful beside two anchors, and both are printed:
  * same voice, another clip      (the ceiling: how alike two takes of one voice are)
  * a different speaker           (the floor: pass ``--other`` clips of another voice)

Usage:
  python scripts\speaker_similarity.py <folder> --ref "anitha_*_P0_*.wav" ^
         --test "anitha_*_H*.wav" --other "ctlA_*.wav" [--out sim.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geetha.eval.breath_score import resample_to_instrument  # noqa: E402

MODEL = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"
_SESSION = None


def embed(path: Path) -> np.ndarray:
    """One unit-length embedding for the whole clip."""
    global _SESSION
    import onnxruntime as ort
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from huggingface_hub import hf_hub_download

    if _SESSION is None:
        _SESSION = ort.InferenceSession(hf_hub_download(MODEL, MODEL_FILE, token=False),
                                        providers=["CPUExecutionProvider"])
    wave, rate = sf.read(str(path), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    wave = resample_to_instrument(wave, rate)       # 16 kHz, which this model expects
    piece = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0) * (1 << 15)
    feats = kaldi.fbank(piece, num_mel_bins=80, frame_length=25, frame_shift=10,
                        dither=0.0, energy_floor=0.0, sample_frequency=16000)
    feats = (feats - feats.mean(0, keepdim=True)).unsqueeze(0).numpy()
    vector = _SESSION.run(None, {_SESSION.get_inputs()[0].name: feats})[0][0]
    return vector / (np.linalg.norm(vector) + 1e-9)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-9)


def compare(refs: dict[str, np.ndarray], tests: dict[str, np.ndarray],
            others: dict[str, np.ndarray]) -> dict:
    """Leave-one-out similarity for the references; similarity to the full centroid otherwise."""
    names = list(refs)
    total = np.sum([refs[n] for n in names], axis=0)
    centroid = _unit(total)
    own = {n: float(refs[n] @ _unit(total - refs[n])) for n in names} if len(names) > 1 else {}
    return {"reference": own,
            "test": {n: float(v @ centroid) for n, v in tests.items()},
            "other": {n: float(v @ centroid) for n, v in others.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="speaker_similarity")
    parser.add_argument("folder")
    parser.add_argument("--ref", required=True, help="glob of clips that define the voice")
    parser.add_argument("--test", required=True, action="append",
                        help="glob of clips to check; may be given more than once")
    parser.add_argument("--other", help="glob of clips in a different voice (the floor)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    folder = Path(args.folder)

    def load(pattern: str | None) -> dict[str, np.ndarray]:
        return {p.name: embed(p) for p in sorted(folder.glob(pattern))} if pattern else {}

    refs = load(args.ref)
    if len(refs) < 2:
        raise SystemExit(f"need at least two reference clips, found {len(refs)} for {args.ref}")
    result = {"model": MODEL, "ref": args.ref, "groups": {}}
    others = load(args.other)
    base = compare(refs, {}, others)

    def line(label: str, values: dict[str, float]) -> None:
        if values:
            v = np.array(list(values.values()))
            print(f"  {label:<34} n={len(v):<3} mean {v.mean():.3f}  min {v.min():.3f}  "
                  f"max {v.max():.3f}")

    print(f"voice defined by {len(refs)} clips matching {args.ref}")
    line("same voice, another clip (ceiling)", base["reference"])
    line("a different speaker (floor)", base["other"])
    result["ceiling"], result["floor"] = base["reference"], base["other"]
    for pattern in args.test:
        scores = compare(refs, load(pattern), {})["test"]
        line(pattern, scores)
        result["groups"][pattern] = scores
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
