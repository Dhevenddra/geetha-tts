r"""B7 (part 1, revision 2) — the Indic-Speak TTS baseline on Tamil, by category.

262 frozen Tamil prompts: Sarvam TTS general (165) and robustness (97), both hashed
into ``manifests\eval_freeze.csv`` part 1 on 27 Oct 2026, so this run is reproducible
against exactly the bytes it was measured on.

Why there is a revision 2
-------------------------
Revision 1 ran every prompt in a single pass with a 16-second cap. **145 of the 262
prompts hit that cap and were cut off mid-sentence.** The reported CER was 61.57; the
117 prompts that finished scored 36.64. Worse, the category ranking inverted: the
``high_quality`` set looked like the second-worst category at 61.85 when, among
prompts that were not truncated, it was the *best* at 29.64. Its prompts are simply
longer. Abbreviations and URLs looked excellent because they are short.

So revision 1 measured the harness, not Tamil. It has been kept in
``runs\b7_baseline\`` as the evidence for that statement and is not a baseline.

What changed
------------
* **Chunked synthesis.** ``geetha.text.chunk.split_for_synthesis`` splits a prompt at
  sentence boundaries, then clauses, then words, and ``geetha.audio.join.join_clips``
  rejoins the pieces with 120 ms of silence. No word is ever dropped (an invariant
  under test against all 262 of these prompts).
* **A budget derived from measurement, not taste.** Over revision 1's 117 untruncated
  clips, Indic-Speak speaks Tamil at a median 11.7 characters per second, but as slowly
  as 2.7 when it spells an abbreviation out letter by letter. The slow cases are all
  *short* prompts: for prompts of 120 characters or more the slowest rate observed was
  8.58 ch/s. A 180-character chunk therefore needs at most ~21 s, inside a 22 s cap.
* **Adaptive re-split.** A character budget is still only a proxy for duration, so any
  chunk that reaches the cap anyway is re-split at half the budget and re-rendered.
  The count is reported. After this, a truncated clip is a bug, not a footnote.
* **The judge is told the audio is long.** A joined clip runs to about 65 s; Whisper's
  default window is 30 s, so ``chunk_length_s`` is set. Otherwise the same truncation
  bug reappears one level down, in the scorer.
* **Partial results are written as they are produced**, so a kernel that dies at 80%
  still yields 80% of the data instead of nothing.

Design choices carried over from revision 1
-------------------------------------------
* **Both voices, alternating.** Anitha on even prompts, Arun on odd.
* **No style tag.** The baseline is the model's default behaviour. Conditioning it
  would measure our prompt engineering. (C2 measured that Rasa's terse style labels do
  nothing to this model anyway.)
* **Whisper-large-v3 as CER judge**, an internal contrast only: the release reports CER
  with Indic-Transcribe-core and SraVaani (R4).
* **Every clip is kept**, so the operator can listen to a bad category rather than
  taking a number on trust.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "bodhan-ai/indic-speak"
ASR_ID = "openai/whisper-large-v3"
SAMPLE_RATE = 24_000

# See the module docstring: 180 characters is ~21 s at the slowest rate ever measured
# for a prompt of that length, and the cap is 22 s.
MAX_CHARS_PER_CHUNK = 180
MAX_SECONDS_PER_CHUNK = 22.0
TOKENS_PER_SECOND = 82
CONTEXT_TOKENS = 2_048
GAP_SECONDS = 0.12
RETRY_DIVISOR = 2

GEN_KWARGS = dict(
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    max_new_tokens=int(MAX_SECONDS_PER_CHUNK * TOKENS_PER_SECOND),
)
SEED = 20261027

_MAX_MOUNT_DEPTH = 4


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"no mounted dataset contains {list(markers)}")


def show_inputs() -> None:
    root = Path("/kaggle/input")
    print(f"/kaggle/input: {[str(p.relative_to(root)) for p in _directories(root)]}", flush=True)


def show_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        print("NO GPU — this will take hours. Stopping.", flush=True)
        raise SystemExit("no CUDA device")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU {name}, capability {major}.{minor}, {total:.1f} GB, "
          f"bf16={'yes' if major >= 8 else 'no'}", flush=True)


def require_token() -> str:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        candidate = mount / "hf_read_token.txt"
        if candidate.exists():
            token = candidate.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                print(f"HF token from private dataset, length {len(token)}", flush=True)
                return token
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("NO HF TOKEN — attach dhevenddra/geetha-secrets.")
    return token


def ensure_dependencies() -> None:
    import subprocess

    try:
        import snac  # noqa: F401
    except ImportError:
        print("installing snac", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snac"], check=False)


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def build_tts(token: str):
    import torch
    from huggingface_hub import snapshot_download

    local = snapshot_download(MODEL_ID, token=token)
    sys.path.insert(0, local)
    from inference import TTS  # type: ignore

    tts = TTS(local)
    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        print("LM in fp16 (no native bf16 on this card)", flush=True)
    return tts


def check_token_budget(tts, chunks: list[str]) -> None:
    """Refuse to run if a text prompt plus its audio cannot fit the context window.

    The generation cap is set in seconds of audio; the context also has to hold the
    text. If the longest chunk's text leaves less room than the cap asks for, every
    long chunk silently truncates and we would be measuring revision 1 again.
    """
    tokenizer = getattr(tts, "tokenizer", None)
    if tokenizer is None:
        print("WARNING: no tokenizer on the TTS object; token budget unchecked", flush=True)
        return
    longest = max(chunks, key=len)
    text_tokens = len(tokenizer(longest)["input_ids"])
    audio_tokens = GEN_KWARGS["max_new_tokens"]
    print(f"token budget: longest chunk {len(longest)} chars = {text_tokens} text tokens "
          f"+ {audio_tokens} audio tokens = {text_tokens + audio_tokens} of {CONTEXT_TOKENS}",
          flush=True)
    if text_tokens + audio_tokens > CONTEXT_TOKENS:
        raise SystemExit(
            f"context overflow: {text_tokens} + {audio_tokens} > {CONTEXT_TOKENS}. "
            f"Lower MAX_CHARS_PER_CHUNK or MAX_SECONDS_PER_CHUNK."
        )


def main() -> int:
    import soundfile as sf
    import torch

    show_inputs()
    show_gpu()
    token = require_token()
    ensure_dependencies()
    import_geetha()

    from geetha.audio.join import join_clips
    from geetha.scoring.bootstrap import bootstrap_ci
    from geetha.scoring.metrics import ErrorCounts, cer, grapheme_cer
    from geetha.text.chunk import split_for_synthesis
    from geetha.text.normalise import norm_text

    prompts_dir = find_dataset("prompts.json")
    clips = json.loads((prompts_dir / "prompts.json").read_text(encoding="utf-8"))["clips"]

    plan = {row["id"]: split_for_synthesis(row["text"], MAX_CHARS_PER_CHUNK) for row in clips}
    total_chunks = sum(len(v) for v in plan.values())
    forced_prompts = [k for k, v in plan.items() if any(c.forced for c in v)]
    print(f"{len(clips)} Tamil prompts -> {total_chunks} chunks at "
          f"{MAX_CHARS_PER_CHUNK} chars; {sum(1 for v in plan.values() if len(v) > 1)} prompts "
          f"split, {len(forced_prompts)} of them mid-sentence", flush=True)

    tts = build_tts(token)
    check_token_budget(tts, [c.text for v in plan.values() for c in v])

    audio_dir = OUT_DIR / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)

    cap_samples = int((MAX_SECONDS_PER_CHUNK - 0.3) * SAMPLE_RATE)
    truncated: list[str] = []
    resplit: list[str] = []
    durations: dict[str, float] = {}
    started = time.time()
    done_chunks = 0

    def render(text: str, speaker: str, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        wav = tts(text, speaker=speaker, style="", seed=seed, **GEN_KWARGS)
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    for index, row in enumerate(clips, 1):
        pieces: list[np.ndarray] = []
        for position, chunk in enumerate(plan[row["id"]]):
            seed = SEED + index * 100 + position
            wav = render(chunk.text, row["speaker"], seed)
            done_chunks += 1
            if len(wav) >= cap_samples:
                # The budget mis-estimated this chunk's duration. Split it finer and
                # re-render rather than keep a cut-off clip.
                finer = split_for_synthesis(chunk.text, MAX_CHARS_PER_CHUNK // RETRY_DIVISOR)
                if len(finer) > 1:
                    resplit.append(row["id"])
                    parts = []
                    for sub_position, sub in enumerate(finer):
                        sub_wav = render(sub.text, row["speaker"], seed * 10 + sub_position)
                        done_chunks += 1
                        parts.append(sub_wav)
                        if len(sub_wav) >= cap_samples:
                            truncated.append(row["id"])
                    wav = join_clips(parts, SAMPLE_RATE, GAP_SECONDS)
                else:
                    # An unbreakable token at the cap: nothing more we can do.
                    truncated.append(row["id"])
            pieces.append(wav)

        joined = join_clips(pieces, SAMPLE_RATE, GAP_SECONDS)
        sf.write(audio_dir / f"{row['id']}.wav", joined, SAMPLE_RATE)
        durations[row["id"]] = round(len(joined) / SAMPLE_RATE, 2)

        if index <= 3 or index % 20 == 0 or index == len(clips):
            per_prompt = (time.time() - started) / index
            print(f"  {index}/{len(clips)}  {done_chunks} chunks  {per_prompt:.1f}s/prompt, "
                  f"~{per_prompt * (len(clips) - index) / 60:.0f} min left, "
                  f"{len(resplit)} re-split, {len(truncated)} still truncated", flush=True)

    print(f"synthesis done in {(time.time() - started) / 60:.1f} min: {done_chunks} chunks, "
          f"{len(set(resplit))} prompts re-split, {len(set(truncated))} still truncated",
          flush=True)

    del tts
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print("transcribing", flush=True)
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=ASR_ID,
                   chunk_length_s=30, stride_length_s=5,
                   device=0 if torch.cuda.is_available() else -1,
                   dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    keys = [row["id"] for row in clips]
    hypotheses: dict[str, str] = {}
    for start in range(0, len(keys), 4):
        batch = keys[start:start + 4]
        out = asr([str(audio_dir / f"{k}.wav") for k in batch],
                  generate_kwargs={"language": "ta", "task": "transcribe"})
        for key, result in zip(batch, out):
            hypotheses[key] = result["text"].strip()
        (OUT_DIR / "b7_hypotheses_partial.json").write_text(
            json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {min(start + 4, len(keys))}/{len(keys)}", flush=True)

    by_category: dict[str, list[ErrorCounts]] = defaultdict(list)
    by_set: dict[str, list[ErrorCounts]] = defaultdict(list)
    by_speaker: dict[str, list[ErrorCounts]] = defaultdict(list)
    overall: list[ErrorCounts] = []
    per_clip = []
    for row in clips:
        reference = norm_text(row["text"])
        hypothesis = norm_text(hypotheses[row["id"]])
        counts = cer(hypothesis, reference)
        g_counts = grapheme_cer(hypothesis, reference)
        overall.append(counts)
        by_category[row["category"]].append(counts)
        by_set[row["set"]].append(counts)
        by_speaker[row["speaker"]].append(counts)
        per_clip.append({
            "id": row["id"], "set": row["set"], "category": row["category"],
            "speaker": row["speaker"],
            "chars": len(row["text"]),
            "chunks": len(plan[row["id"]]),
            "forced_split": any(c.forced for c in plan[row["id"]]),
            "seconds": durations[row["id"]],
            "cer": round(100.0 * counts.errors / max(1, counts.reference_length), 2),
            "grapheme_cer": round(100.0 * g_counts.errors / max(1, g_counts.reference_length), 2),
            "resplit": row["id"] in resplit,
            "truncated": row["id"] in truncated,
            "hypothesis": hypotheses[row["id"]],
        })

    def summarise(groups: dict[str, list[ErrorCounts]]) -> dict:
        out = {}
        for name, counts in sorted(groups.items()):
            result = bootstrap_ci(counts)
            out[name] = {"n": len(counts), "cer": round(100 * result.point, 2),
                         "ci": [round(100 * result.low, 2), round(100 * result.high, 2)]}
        return out

    total = bootstrap_ci(overall)
    results = {
        "model": MODEL_ID,
        "revision": 2,
        "n": len(clips),
        "cer": round(100 * total.point, 2),
        "ci": [round(100 * total.low, 2), round(100 * total.high, 2)],
        "by_set": summarise(by_set),
        "by_category": summarise(by_category),
        "by_speaker": summarise(by_speaker),
        "chunking": {
            "max_chars": MAX_CHARS_PER_CHUNK,
            "max_seconds": MAX_SECONDS_PER_CHUNK,
            "gap_seconds": GAP_SECONDS,
            "chunks": done_chunks,
            "prompts_split": sum(1 for v in plan.values() if len(v) > 1),
            "prompts_forced_mid_sentence": len(forced_prompts),
            "prompts_resplit": sorted(set(resplit)),
        },
        "truncated": sorted(set(truncated)),
        "judge": ASR_ID,
        "judge_window_seconds": 30,
        "generation": GEN_KWARGS,
        "seed": SEED,
        "per_clip": per_clip,
    }
    (OUT_DIR / "b7_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    chunking = results["chunking"]
    lines = [
        "# B7 part 1 (rev 2) — Indic-Speak baseline on Tamil (262 frozen prompts)",
        "",
        f"**CER {results['cer']:.2f}** [{results['ci'][0]:.2f}, {results['ci'][1]:.2f}] "
        f"over {len(clips)} prompts, judge `{ASR_ID}`.",
        "",
        f"Chunked synthesis at {MAX_CHARS_PER_CHUNK} chars / {MAX_SECONDS_PER_CHUNK:.0f} s, "
        f"rejoined with {GAP_SECONDS * 1000:.0f} ms gaps: {chunking['chunks']} chunks for "
        f"{len(clips)} prompts. {chunking['prompts_split']} prompts needed more than one pass, "
        f"**{chunking['prompts_forced_mid_sentence']} of them split mid-sentence** — those carry "
        f"a pause the writer did not write, which costs nothing for CER but would for "
        f"naturalness.",
        f"{len(chunking['prompts_resplit'])} prompts hit the cap and were re-split finer; "
        f"**{len(results['truncated'])} clips are still truncated** "
        f"({'none — the number below is intelligibility, not truncation' if not results['truncated'] else 'treat their rows with suspicion'}).",
        "",
        "| category | n | CER | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in sorted(results["by_category"].items(), key=lambda kv: -kv[1]["cer"]):
        lines.append(f"| {name} | {stats['n']} | {stats['cer']:.2f} | "
                     f"[{stats['ci'][0]:.2f}, {stats['ci'][1]:.2f}] |")
    lines += ["", "| voice | n | CER |", "| --- | --- | --- |"]
    for name, stats in results["by_speaker"].items():
        lines.append(f"| {name} | {stats['n']} | {stats['cer']:.2f} |")
    lines += [
        "",
        "Revision 1 of this run capped generation at 16 s with no chunking; 145 of the 262",
        "prompts were cut off and its CER of 61.57 was mostly truncation. Its category",
        "ranking was an artefact of prompt length and must not be quoted. This revision",
        "replaces it.",
        "",
        "This is an internal contrast, not a published baseline: the release reports CER",
        "with Indic-Transcribe-core and SraVaani (R4). What it is for is deciding where",
        "Stage A has work to do.",
    ]
    (OUT_DIR / "b7_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
