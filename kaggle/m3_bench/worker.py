r"""One system, one GPU: render every benchmark prompt, then transcribe with both judges.

Run by the m3_bench kernels as ``python worker.py spec.json`` with CUDA_VISIBLE_DEVICES
set to one card, and with no Hugging Face token in the environment: everything it
loads was downloaded by the parent and is passed as a local folder.

Each system is run the way its own card says, single shot, one fixed seed per prompt
(``seed + 100 * index``, the same for every system):

    geetha            release\inference.py exactly as shipped: speak(text, speaker,
                      retries=0), neutral style, no breath tag.
    base              bodhan-ai/indic-speak through its own inference.py:
                      tts(text, speaker="Anitha"|"Arun"), temperature 0.6, top-p 0.9,
                      top-k 50 (its defaults), same 22 s cap as Geetha.
    rumik_oss_1       rumik-ai/rumik-oss-1, voice Ira, temperature 0.8, top-k 30.
    snortts_indic_v0  snorbyte/snorTTS-Indic-v0, speaker tamil188, temperature 0.4,
                      top-p 0.9, repetition penalty 1.05, stock SNAC decoder.

A clip that is mute or never ended is invalid: its own rate, never sent to a judge. A
prompt the time budget did not reach is "not_run", which is neither. The result file is
rewritten after every clip, so a kernel that dies still leaves what it made.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

SAMPLE_RATE, JUDGE_RATE = 24_000, 16_000
CAP_SECONDS = 22.0
TOKENS_PER_SECOND = 82
RUMIK = dict(speaker="Ira", temperature=0.8, top_k=30, max_new_tokens=2048)
SNOR = dict(speaker="tamil188", temperature=0.4, top_p=0.9, repetition_penalty=1.05,
            max_new_tokens=int(CAP_SECONDS * TOKENS_PER_SECOND))
BASE = dict(temperature=0.6, top_p=0.9, top_k=50,
            max_new_tokens=int(CAP_SECONDS * TOKENS_PER_SECOND))


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    system: str = spec["system"]
    prompts: list[dict] = spec["prompts"]
    out_dir = Path(spec["out_dir"])
    audio = out_dir / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"bench_{system}.json"
    started = time.time()
    render_deadline = started + float(spec["render_budget_s"])

    def log(message: str) -> None:
        print(f"[{system}] {message}", flush=True)

    import soundfile as sf
    import torch
    import torchaudio

    sys.path.insert(0, spec["package_parent"])
    from geetha.eval.validity import check_clip

    half = torch.cuda.get_device_capability(0)[0] < 8
    dtype = torch.float16 if half else torch.bfloat16
    log(f"gpu: {torch.cuda.get_device_name(0)}  half precision: {half}  "
        f"{len(prompts)} prompts")

    rows: dict[str, dict] = {}
    hyps: dict[str, dict[str, str]] = {}
    report = {"system": system, "seed": spec["seed"], "seed_rule": "seed + 100 * index",
              "single_shot": True, "cap_seconds": CAP_SECONDS, "half_precision": half,
              "decoding": {"geetha": "release/inference.py defaults (CONV: temperature 0.6, "
                                     "top-p 0.9, top-k 50, no guidance), retries=0",
                           "base": BASE, "rumik_oss_1": RUMIK,
                           "snortts_indic_v0": SNOR}[system],
              "revisions": spec.get("revisions", {}), "failure": None,
              "clips": rows, "hypotheses": hyps}

    def flush() -> None:
        report["elapsed_s"] = round(time.time() - started)
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")

    # ------------------------------------------------------------------ renderers
    def load_geetha():
        sys.path.insert(0, spec["release_dir"])
        import inference                       # the shipped file, as the README says

        tts = inference.Geetha(spec["adapter_dir"], base=spec["base_dir"], device="cuda")

        def render(text: str, voice: str, seed: int):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wav = tts.speak(text, speaker=voice, seed=seed, retries=0)
            stuck = any("stayed invalid" in str(w.message) for w in caught)
            # speak() folds "silent" and "never ended" into one warning; the level check
            # below separates them, so only a loud clip with the warning is a cap hit.
            return wav, not stuck or not check_clip(wav, SAMPLE_RATE).valid
        return render

    def load_base():
        import importlib.util

        sys.path.insert(0, spec["base_dir"])
        module_spec = importlib.util.spec_from_file_location(
            "indic_speak_inference", str(Path(spec["base_dir"]) / "inference.py"))
        loader = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = loader
        module_spec.loader.exec_module(loader)
        tts = loader.TTS(spec["base_dir"], device="cuda")
        if half:
            tts.lm = tts.lm.half()             # the same cast the Geetha release makes

        def render(text: str, voice: str, seed: int):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wav = tts(text, speaker=voice.capitalize(), seed=seed, **BASE)
            capped = any("hit max_new_tokens" in str(w.message) for w in caught)
            return wav, not capped
        return render

    def load_rumik():
        from transformers import AutoModelForCausalLM, AutoTokenizer, MimiModel

        folder = spec["rumik_dir"]
        tok = AutoTokenizer.from_pretrained(folder)
        lm = AutoModelForCausalLM.from_pretrained(
            folder, trust_remote_code=True, dtype=dtype).eval().cuda()
        mimi = MimiModel.from_pretrained(folder, subfolder="codec").eval().cuda()

        def render(text: str, voice: str, seed: int):
            torch.manual_seed(seed)
            inputs = tok(f'<text>{RUMIK["speaker"]}: {text}<audio>',
                         return_tensors="pt").to(lm.device)
            with torch.inference_mode():
                ids = lm.generate_audio(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=RUMIK["max_new_tokens"],
                    temperature=RUMIK["temperature"], top_k=RUMIK["top_k"], do_sample=True)
                tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]
                ended = bool(tokens) and tokens[-1] == int(lm.config.audio_end_token_id)
                codes = lm.audio_tokens_to_codes(tokens)
                wav = mimi.decode(codes.to(mimi.device)).audio_values[0, 0]
            return wav.float().clamp(-1, 1).cpu().numpy(), ended
        return render

    def load_snor():
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        folder = spec["snor_dir"]
        tok = AutoTokenizer.from_pretrained(folder)
        lm = AutoModelForCausalLM.from_pretrained(folder, dtype=dtype).eval().cuda()
        snac = SNAC.from_pretrained(spec["snac_id"],
                                    revision=spec["snac_revision"]).eval().cuda()
        end_of_speech, audio_start = 128256 + 2, 128256 + 10

        def render(text: str, voice: str, seed: int):
            torch.manual_seed(seed)
            prompt = (f"<custom_token_3><|begin_of_text|>{SNOR['speaker']}: {text}<|eot_id|>"
                      "<custom_token_4><custom_token_5><custom_token_1>")
            inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            empty = np.zeros(0, dtype=np.float32)
            with torch.inference_mode():
                out = lm.generate(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=SNOR["max_new_tokens"], do_sample=True,
                    temperature=SNOR["temperature"], top_p=SNOR["top_p"],
                    repetition_penalty=SNOR["repetition_penalty"],
                    eos_token_id=end_of_speech, pad_token_id=end_of_speech)
                new = out[0].tolist()[inputs.input_ids.shape[1]:]
                ended = bool(new) and new[-1] == end_of_speech
                codes = [t - audio_start for t in new if t >= audio_start]
                frames = len(codes) // 7
                if not frames:
                    return empty, ended
                grid = np.array(codes[: frames * 7], np.int64).reshape(frames, 7)
                grid -= np.arange(7, dtype=np.int64) * 4096
                grid = grid[np.all((grid >= 0) & (grid < 4096), axis=1)]
                if not len(grid):
                    return empty, ended
                layer1 = np.stack([grid[:, 1], grid[:, 4]], axis=1).reshape(-1)
                layer2 = np.stack([grid[:, 2], grid[:, 3], grid[:, 5], grid[:, 6]],
                                  axis=1).reshape(-1)
                tensors = [torch.from_numpy(x.copy()).unsqueeze(0).cuda()
                           for x in (grid[:, 0], layer1, layer2)]
                wav = snac.decode(tensors)[0, 0]
            return wav.float().clamp(-1, 1).cpu().numpy(), ended
        return render

    loaders = {"geetha": load_geetha, "base": load_base, "rumik_oss_1": load_rumik,
               "snortts_indic_v0": load_snor}

    # ------------------------------------------------------------------ rendering
    try:
        render = loaders[system]()
        log(f"model loaded in {time.time() - started:.0f} s")
    except Exception as problem:  # noqa: BLE001
        report["failure"] = f"load: {type(problem).__name__}: {problem}"
        log("FAILED to load:\n" + traceback.format_exc())
        flush()
        return 1

    errors = 0
    audio_seconds = 0.0
    for index, prompt in enumerate(prompts):
        clip_id, file = prompt["id"], f"{system}__{prompt['id']}.wav"
        if time.time() > render_deadline:
            rows[clip_id] = {"file": None, "valid": False, "reason": "not_run"}
            continue
        ended = False
        try:
            wav, ended = render(prompt["text"], prompt["voice"], spec["seed"] + 100 * index)
        except Exception as problem:  # noqa: BLE001
            errors += 1
            if errors <= 3:
                log(f"{clip_id}: {type(problem).__name__}: {problem}")
            wav = np.zeros(0, dtype=np.float32)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        check = check_clip(wav, SAMPLE_RATE, ended_itself=bool(ended))
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        rows[clip_id] = {"file": file if len(wav) else None, "valid": check.valid,
                         "reason": check.reason, "seconds": round(check.seconds, 2),
                         "voice": prompt["voice"], "source": prompt["source"]}
        audio_seconds += check.seconds
        flush()
        if (index + 1) % 10 == 0 or index + 1 == len(prompts):
            good = sum(1 for r in rows.values() if r["valid"])
            log(f"rendered {index + 1}/{len(prompts)}  valid {good}  "
                f"{audio_seconds:.0f} s of audio  {time.time() - started:.0f} s elapsed")
    if errors:
        report["render_errors"] = errors
    skipped = sum(1 for r in rows.values() if r["reason"] == "not_run")
    if skipped:
        log(f"render budget ran out: {skipped} prompts not run")
    flush()
    del render
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ judging
    def judge_wave(file: str):
        samples, rate = sf.read(str(audio / file), dtype="float32", always_2d=True)
        wave = torch.from_numpy(np.ascontiguousarray(samples.mean(axis=1)))
        return torchaudio.functional.resample(wave, rate, JUDGE_RATE)

    valid_ids = [i for i, r in rows.items() if r["valid"]]
    try:
        log(f"judge 1: Indic-Transcribe-core on {len(valid_ids)} valid clips")
        sys.path.insert(0, spec["transcribe_dir"])
        from indic_transcribe import IndicTranscribe  # type: ignore

        judge = IndicTranscribe.from_pretrained(spec["transcribe_dir"], device="cuda")
        scratch = out_dir / f"_judge_{system}.wav"
        out: dict[str, str] = {}
        hyps["indic_transcribe_core"] = out
        for count, clip_id in enumerate(valid_ids, 1):
            sf.write(scratch, judge_wave(rows[clip_id]["file"]).numpy(), JUDGE_RATE)
            text = judge.transcribe(str(scratch), lang="ta")
            out[clip_id] = (text[0] if isinstance(text, tuple) else text).strip()
            if count % 50 == 0:
                flush()
                log(f"judge 1: {count}/{len(valid_ids)}  {time.time() - started:.0f} s")
        scratch.unlink(missing_ok=True)
        flush()
        del judge
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as problem:  # noqa: BLE001
        report["failure"] = f"judge 1: {type(problem).__name__}: {problem}"
        log("judge 1 FAILED:\n" + traceback.format_exc())
        flush()

    try:
        log("judge 2: IndicConformer-600M RNNT")
        sys.path.insert(0, spec["conformer_dir"])
        from model_onnx import IndicASRConfig, IndicASRModel  # type: ignore

        settings = json.loads((Path(spec["conformer_dir"]) / "config.json").read_text())
        settings.pop("auto_map", None)
        conformer = IndicASRModel(IndicASRConfig(ts_folder=spec["conformer_dir"], **settings))
        out = {}
        hyps["indicconformer_rnnt"] = out
        failed = 0
        for count, clip_id in enumerate(valid_ids, 1):
            try:
                out[clip_id] = conformer(judge_wave(rows[clip_id]["file"]).unsqueeze(0),
                                         "ta", "rnnt")
            except Exception as problem:  # noqa: BLE001
                out[clip_id] = ""
                failed += 1
                if failed <= 3:
                    log(f"judge 2 failed on {clip_id}: {type(problem).__name__}: {problem}")
            if count % 50 == 0:
                flush()
                log(f"judge 2: {count}/{len(valid_ids)}  {time.time() - started:.0f} s")
        if failed:
            report["judge2_failures"] = failed
    except Exception as problem:  # noqa: BLE001
        report["failure"] = f"judge 2: {type(problem).__name__}: {problem}"
        log("judge 2 FAILED:\n" + traceback.format_exc())
    report["finished"] = True
    flush()
    good = sum(1 for r in rows.values() if r["valid"])
    log(f"done: {good}/{len(rows)} valid, {len(hyps.get('indic_transcribe_core', {}))} and "
        f"{len(hyps.get('indicconformer_rnnt', {}))} transcripts, "
        f"{time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
