r"""C4 - turn IndicVoices' breath *text* marks into breath *timestamps*.

Why this run exists
-------------------
D-028 rejected the AudioSet tagger: it cannot hear breaths in Tamil speech. Two further
measurements (28 Oct, local) say the problem is deeper than the choice of tagger:

* On the 800-clip calibration set, ``rho(clip duration, annotated breaths) = 0.633``
  while the tagger's best is ``0.595`` - **a stopwatch predicts breath count better than
  the detector**. At a threshold tuned to make the rate come out at 9.64/min against a
  truth of 9.13/min, it still fires at 1.6x the rate on clips with no breath at all. The
  rate can always be dialled to the right answer, so the rate alone proves nothing.
* No replacement exists to borrow: the Hub has no breath-event detector for speech.

The blocker is not the detector, it is the **labels**. The calibration set carries a
breath *count per clip*. With no onsets there is no frame-level precision or recall, so
every candidate detector can only be fitted to a weak per-clip number and judged against
a baseline that clip length already beats. A count-labelled set can reject a detector;
it cannot validate one.

What unlocks it
---------------
IndicVoices puts each ``[breathing]`` mark **inline at the position it occurs**, and
**2,816 of 2,845 marks in a single shard (99%) have words on both sides**. So the mark
names the gap between two specific words. Align those words to the audio and the breath
has a time window. 648 weak count labels become 648 located pauses.

A crude local proxy already supports this, with a control: assuming uniform word timing,
the audio at a predicted breath position is non-speech 0.327 of the time against 0.183
at random times in the same clip (1.79x at +/-0.25 s), and the lift *falls* as the window
widens (1.56x at 0.50 s, 1.35x at 1.00 s) - the signature of a real localised effect
rather than an artefact. Proper alignment should sharpen it.

This run's first duty is to try to falsify itself
-------------------------------------------------
An aligner is an instrument, and D-028 (and D-022 before it) are both cases of a borrowed
instrument that was trusted before it was tested. So this job does not report breath
windows and stop. It states a prediction that must hold if alignment worked at all:

    a gap carrying a breath mark is LONGER than an ordinary word gap

and it computes that against every unmarked inter-word gap in the same clips as control.
If the separation is absent, the report says **FAIL** and the alignment is not to be built
on. Negatives (clips with no annotation at all) are aligned too, and their gaps give the
false-positive opportunity rate: how often an unmarked gap looks exactly like a breath.

Nothing here chooses a detector or a threshold. It produces the ground truth that would
let a detector be measured honestly, and the evidence that the ground truth is sound.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
SAMPLE_RATE = 16_000

# A CTC model with a Tamil-script vocabulary, so the transcript aligns directly and no
# romanisation step can introduce its own errors. Vakyansh is the most-downloaded Tamil
# wav2vec2 CTC checkpoint; MMS is the fallback if its vocab cannot cover the text.
MODEL_IDS = (
    "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
    "facebook/mms-1b-all",
)

BREATH_MARK = re.compile(r"^\[(breathing|sigh)\]$", re.IGNORECASE)
ANY_MARK = re.compile(r"^\[[^\]]*\]$")

# A breath is audibly shorter than a word and longer than a frame; gaps outside this are
# not candidate breaths. Stated here rather than tuned, so the control is not gamed.
BREATH_GAP_MIN = 0.08
BREATH_GAP_MAX = 1.50

# PASS needs both: breath gaps clearly longer than ordinary gaps, and most of them inside
# the plausible band. Fixed before the run so the verdict cannot be moved to fit.
MIN_DURATION_LIFT = 1.25
MIN_IN_BAND = 0.50


def log(message: str) -> None:
    print(message, flush=True)


def find_input(name: str) -> Path:
    for base in sorted(Path("/kaggle/input").glob("*")):
        hit = base / name
        if hit.exists():
            return hit
        for deeper in base.rglob(name):
            return deeper
    raise SystemExit(f"could not find {name} under /kaggle/input")


def tokenise(verbatim: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return ``(words, breath_slots)``.

    ``breath_slots`` holds ``(index of word before, index of word after)`` for every
    breath mark, as indices into ``words``. A mark at an utterance edge yields ``-1``
    for the missing side and is dropped later, because an edge has no gap.
    """
    words: list[str] = []
    slots: list[tuple[int, int]] = []
    pending = 0
    for token in verbatim.split():
        if ANY_MARK.match(token):
            if BREATH_MARK.match(token):
                pending += 1
            continue
        if pending:
            for _ in range(pending):
                slots.append((len(words) - 1, len(words)))
            pending = 0
        words.append(token)
    if pending:
        for _ in range(pending):
            slots.append((len(words) - 1, len(words)))
    return words, slots


def load_aligner():
    """First usable CTC model, or ``(None, None, None)``."""
    from transformers import AutoModelForCTC, AutoProcessor
    import torch

    for model_id in MODEL_IDS:
        try:
            if "mms" in model_id:
                processor = AutoProcessor.from_pretrained(model_id, target_lang="tam")
                model = AutoModelForCTC.from_pretrained(
                    model_id, target_lang="tam", ignore_mismatched_sizes=True)
            else:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForCTC.from_pretrained(model_id)
            return processor, model.cuda().eval(), model_id
        except Exception as exc:
            log(f"  {model_id} unusable: {type(exc).__name__}: {exc}")
    return None, None, None


def word_spans(path, flat, owner, blank):
    """Frame extent per word index, walking the aligned path alongside ``flat``.

    CTC emits the *same* target token on consecutive frames while it is still inside that
    token, so the pointer into ``flat`` may only advance on a transition to a new token
    occurrence - never once per non-blank frame. Advancing per frame collapses every word
    to its first frame, which would shrink every word and inflate every gap between them:
    exactly the direction that would manufacture a result here.
    """
    spans: dict[int, list[int]] = {}
    position = -1
    previous = blank
    for frame, token in enumerate(path):
        if token == blank:
            previous = blank
            continue
        if token != previous:
            position += 1
            while position < len(flat) and flat[position] != token:
                position += 1
            if position >= len(flat):
                break
        previous = token
        who = owner[position] if position < len(owner) else -1
        if who >= 0:
            if who in spans:
                spans[who][1] = frame
            else:
                spans[who] = [frame, frame]
    return spans


def main() -> int:
    started = time.time()
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except Exception as exc:
        log(f"imports failed: {exc}")
        return 1

    log(f"torch {torch.__version__}  torchaudio {torchaudio.__version__}")
    if not hasattr(torchaudio.functional, "forced_align"):
        log("torchaudio has no forced_align - need torchaudio >= 2.1. Stopping.")
        return 1
    if not torch.cuda.is_available():
        log("no GPU - alignment on CPU would be slow. Stopping.")
        return 1

    manifest_path = find_input("manifest.json")
    clips_dir = manifest_path.parent / "clips"
    transcripts = json.loads(find_input("transcripts.json").read_text(encoding="utf-8"))
    clips = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"]
    log(f"manifest {manifest_path}")
    log(f"clips dir {clips_dir}  transcripts {len(transcripts)}  clips {len(clips)}")

    processor, model, chosen = load_aligner()
    if model is None:
        log("no CTC model loaded. Stopping.")
        return 1
    log(f"aligner: {chosen}")

    vocab = processor.tokenizer.get_vocab()
    blank = processor.tokenizer.pad_token_id
    if blank is None:
        blank = 0
    delimiter = processor.tokenizer.word_delimiter_token
    delim = vocab.get("|", vocab.get(delimiter))
    log(f"vocab {len(vocab)}  blank id {blank}  word delimiter id {delim}")

    strip_punct = re.compile(r"[^\w஀-௿]+", re.UNICODE)

    records = []
    aligned = skipped = 0
    for n, clip in enumerate(clips, 1):
        cid = clip["id"]
        record = transcripts.get(cid)
        if record is None:
            skipped += 1
            continue
        words, slots = tokenise(record["verbatim"])
        if len(words) < 2:
            skipped += 1
            continue

        # Map each word to vocab ids. A word with no usable character is dropped, and its
        # disappearance would shift every slot index, so keep the old->new mapping.
        kept_words: list[str] = []
        kept_tokens: list[list[int]] = []
        old_to_new: dict[int, int] = {}
        for i, word in enumerate(words):
            clean = strip_punct.sub("", word)
            ids = [vocab[ch] for ch in clean if ch in vocab]
            if not ids:
                continue
            old_to_new[i] = len(kept_words)
            kept_words.append(clean)
            kept_tokens.append(ids)
        if len(kept_words) < 2:
            skipped += 1
            continue

        wave, rate = sf.read(str(clips_dir / f"{cid}.flac"), dtype="float32")
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{cid} is {rate} Hz, expected {SAMPLE_RATE}")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)

        with torch.inference_mode():
            inputs = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            logits = model(inputs.input_values.cuda()).logits
            emission = torch.log_softmax(logits, dim=-1)

        # A flat token sequence with the word delimiter between words, which is what
        # gives the aligner somewhere to put a pause.
        flat: list[int] = []
        owner: list[int] = []
        for index, ids in enumerate(kept_tokens):
            if index and delim is not None:
                flat.append(delim)
                owner.append(-1)
            for token in ids:
                flat.append(token)
                owner.append(index)

        targets = torch.tensor([flat], dtype=torch.int32, device=emission.device)
        try:
            path, scores = torchaudio.functional.forced_align(emission, targets, blank=blank)
        except Exception as exc:
            log(f"  {cid}: forced_align failed ({type(exc).__name__}: {exc})")
            skipped += 1
            continue

        frames = emission.shape[1]
        seconds_per_frame = len(wave) / SAMPLE_RATE / frames
        confidence = float(scores[0].exp().mean())
        spans = word_spans(path[0].cpu().numpy(), flat, owner, blank)
        if len(spans) < 2:
            skipped += 1
            continue
        aligned += 1

        times = {w: (a * seconds_per_frame, (b + 1) * seconds_per_frame)
                 for w, (a, b) in spans.items()}

        breath_pairs = set()
        for before, after in slots:
            if before < 0 or after >= len(words):
                continue
            if before in old_to_new and after in old_to_new:
                breath_pairs.add((old_to_new[before], old_to_new[after]))

        gaps = []
        order = sorted(times)
        for a, b in zip(order, order[1:]):
            start = times[a][1]
            end = times[b][0]
            gaps.append({"after_word": int(a),
                         "start": round(float(start), 4),
                         "end": round(float(end), 4),
                         "seconds": round(float(max(0.0, end - start)), 4),
                         "has_breath": (a, b) in breath_pairs})

        records.append({"id": cid, "label": clip["label"], "seconds": clip["seconds"],
                        "breaths_annotated": clip["breaths"],
                        "confidence": round(confidence, 4),
                        "n_words": len(kept_words), "gaps": gaps,
                        "breath_gaps_found": sum(1 for g in gaps if g["has_breath"])})

        if n <= 3 or n % 100 == 0 or n == len(clips):
            per = (time.time() - started) / n
            log(f"  {n}/{len(clips)}  aligned {aligned} skipped {skipped}  "
                f"{per:.2f}s/clip  ~{per * (len(clips) - n) / 60:.0f} min left")
            (OUT_DIR / "c4_partial.json").write_text(json.dumps(records), encoding="utf-8")

    (OUT_DIR / "c4_alignments.json").write_text(
        json.dumps({"model": chosen, "clips": records}, ensure_ascii=False),
        encoding="utf-8")

    # ---- the falsification test ------------------------------------------------
    breath = np.array([g["seconds"] for r in records for g in r["gaps"] if g["has_breath"]])
    other = np.array([g["seconds"] for r in records if r["label"] == "positive"
                      for g in r["gaps"] if not g["has_breath"]])
    negative = np.array([g["seconds"] for r in records if r["label"] == "negative"
                         for g in r["gaps"]])
    confidences = np.array([r["confidence"] for r in records])
    marks_total = sum(r["breaths_annotated"] for r in records if r["label"] == "positive")
    located = int(sum(r["breath_gaps_found"] for r in records))

    def pct(values, q):
        return round(float(np.percentile(values, q)), 4) if len(values) else None

    def mean(values):
        return round(float(values.mean()), 4) if len(values) else None

    lift = None
    if len(breath) and len(other) and other.mean() > 0:
        lift = round(float(breath.mean() / other.mean()), 4)
    in_band = None
    if len(breath):
        in_band = round(float(((breath >= BREATH_GAP_MIN) & (breath <= BREATH_GAP_MAX)).mean()), 4)
    other_in_band = None
    if len(other):
        other_in_band = round(float(((other >= BREATH_GAP_MIN) & (other <= BREATH_GAP_MAX)).mean()), 4)

    verdict = "PASS" if (lift is not None and in_band is not None
                         and lift >= MIN_DURATION_LIFT and in_band >= MIN_IN_BAND) else "FAIL"

    summary = {
        "model": chosen,
        "aligned": aligned,
        "skipped": skipped,
        "mean_confidence": mean(confidences),
        "breath_marks_in_positive_clips": marks_total,
        "breath_marks_located_in_a_gap": located,
        "located_fraction": round(located / marks_total, 4) if marks_total else None,
        "gap_seconds": {
            "breath": {"n": int(len(breath)), "mean": mean(breath),
                       "p25": pct(breath, 25), "p50": pct(breath, 50), "p75": pct(breath, 75)},
            "unmarked_in_positive": {"n": int(len(other)), "mean": mean(other),
                                     "p50": pct(other, 50)},
            "all_in_negative": {"n": int(len(negative)), "mean": mean(negative),
                                "p50": pct(negative, 50)},
        },
        "duration_lift_breath_over_unmarked": lift,
        "breath_gaps_in_band": in_band,
        "unmarked_gaps_in_band": other_in_band,
        "band": [BREATH_GAP_MIN, BREATH_GAP_MAX],
        "pass_requires": {"duration_lift": MIN_DURATION_LIFT, "in_band": MIN_IN_BAND},
        "verdict": verdict,
    }
    (OUT_DIR / "c4_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    gap = summary["gap_seconds"]
    lines = [
        "# C4 - breath timestamps by forced alignment",
        "",
        f"Aligner: `{chosen}`. Aligned {aligned} of {len(clips)} clips "
        f"(skipped {skipped}), mean CTC confidence {summary['mean_confidence']}.",
        "",
        "## Did alignment actually locate the breaths?",
        "",
        f"- {located} of {marks_total} breath marks in the positive clips landed in an "
        f"inter-word gap ({summary['located_fraction']}).",
        f"- A breath gap runs **{gap['breath']['mean']} s** on average "
        f"(median {gap['breath']['p50']}, IQR {gap['breath']['p25']}-{gap['breath']['p75']}), "
        f"against **{gap['unmarked_in_positive']['mean']} s** for an unmarked gap in the "
        f"same clips (median {gap['unmarked_in_positive']['p50']}).",
        f"- Duration lift, breath over unmarked: **{lift}x** "
        f"(PASS needs >= {MIN_DURATION_LIFT}).",
        f"- Breath gaps inside the plausible band {BREATH_GAP_MIN}-{BREATH_GAP_MAX} s: "
        f"**{in_band}** (PASS needs >= {MIN_IN_BAND}); unmarked gaps in the same band: "
        f"{other_in_band}. That second figure is the false-positive opportunity rate - how "
        f"often an ordinary pause is already breath-shaped on duration alone, which is the "
        f"ceiling any duration-only rule would hit.",
        f"- Gaps in the negative clips: n={gap['all_in_negative']['n']}, "
        f"mean {gap['all_in_negative']['mean']} s, median {gap['all_in_negative']['p50']} s.",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "The prediction was that a gap carrying a breath mark is longer than an ordinary word",
        "gap. PASS means alignment located something real and these windows can serve as",
        "ground truth for a detector. FAIL means the aligner did not place the words reliably",
        "enough and nothing should be built on this output - the same mistake as D-028 and",
        "D-022, caught before the pipeline instead of after it.",
        "",
        "## What this does not yet show",
        "",
        "Duration separation says the marks landed in real pauses. It does **not** say a breath",
        "is acoustically detectable inside those pauses; that is the next measurement, and it",
        "is now possible because each breath has a window instead of only a count.",
    ]
    (OUT_DIR / "c4_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))
    log(f"\ndone in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
