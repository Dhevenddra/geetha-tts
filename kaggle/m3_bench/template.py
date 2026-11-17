r"""M3 benchmark v2: __GROUP__ on 300+ pure-Tamil sentences, both judges. (free Kaggle GPU)

The 41-prompt set was too small and its references were drafted by hand. This set is
__N__ sentences written only in Tamil script (no digits, no Latin), so the written text
IS the reference and no reader has to draft a spoken form:

    fleurs   FLEURS ta_in TEST sentences (formal, read register; public)
    dir      Directability-Ta v1 carriers and untagged emotional sentences (colloquial)
    cap      Capability-Ta v1 sentences without marks

None of them is in a training manifest (checked by normalised text). Voices alternate
anitha / arun for the two systems that have them. What it does NOT cover: digits,
English words, long text. The 41-prompt set stays as the mixed-text slice.

This kernel only renders and transcribes: one worker process per system, one GPU each
(worker.py, embedded below by scripts\make_bench_v2.py). CER, confidence intervals and
paired differences are computed afterwards on the PC from the released transcripts
(scripts\score_bench_v2.py), so the scoring can be re-run without a GPU.
Audio is private to this kernel (D-007).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

OUT_DIR = Path("/kaggle/working")
GROUP = "__GROUP__"
SYSTEMS = {"ours": ("geetha", "base"), "rivals": ("rumik_oss_1", "snortts_indic_v0")}[GROUP]
MODEL_ID = "bodhan-ai/indic-speak"
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTER = "m2-main/stageB/final"
CONFORMER_ID = "ai4bharat/indic-conformer-600m-multilingual"
TRANSCRIBE_ID = "bodhan-ai/indic-transcribe-core"
CONFORMER_REVISION = "2a77d305038dfc502fd9464e7d028b90381976cb"
TRANSCRIBE_REVISION = "e083ab7dfeeddcb5c0b4a2d7df53f53e7b0f1537"
# Pinned to commits read on 14 Nov 2026. Rumik ships two Python files that run in the
# worker (trust_remote_code); both were read at this commit and are checked by hash
# before any worker starts. snorTTS is a plain Llama checkpoint: no remote code.
RUMIK_ID, RUMIK_REVISION = "rumik-ai/rumik-oss-1", "0ed3c98684e14350c910129c0efa179841c41ad2"
SNOR_ID, SNOR_REVISION = "snorbyte/snorTTS-Indic-v0", "c771394e1a427a6d2256ff881c8ed9f3cb6821c9"
SNAC_ID, SNAC_REVISION = "hubertsiuzdak/snac_24khz", "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
RUMIK_CODE_SHA256 = {
    "modeling_rumik_oss.py": "9b2c05014fa825cc5db030229a8e6707df4a80ebcd2c34b44dc9cbc294d0b8a6",
    "configuration_rumik_oss.py": "f7da28962c6314d52515f9ca4cf8c6ebbce80e08c928474e9dd17c3944c692cc",
}
SEED = 1234
# A Kaggle session ends at 12 h. With a card each, a system gets 8 h to render and the
# rest to transcribe; sharing one card, they run one after the other with 4.5 h each.
RENDER_BUDGET_PARALLEL_S, RENDER_BUDGET_SERIAL_S = 8 * 3600, 4.5 * 3600
_MAX_MOUNT_DEPTH = 4

PROMPTS = json.loads(r'''__PROMPTS__''')

PAYLOAD = __PAYLOAD__


def log(message: str) -> None:
    print(message, flush=True)


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
    raise SystemExit(f"could not find a mount containing {markers}")


def main() -> int:
    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            capture_output=True, text=True)
    log(f"pip uninstall torchao -> {result.returncode}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snac==1.2.1",
                    "onnxruntime==1.20.1", "soundfile==0.14.0", "sentencepiece"], check=True)
    import torch
    from huggingface_hub import snapshot_download

    gpus = torch.cuda.device_count()
    log(f"group {GROUP}: {SYSTEMS}; {len(PROMPTS)} prompts; {gpus} GPU(s): "
        f"{[torch.cuda.get_device_name(i) for i in range(gpus)]}")
    if not gpus:
        raise SystemExit("no GPU in this session")

    token = (find_dataset("hf_read_token.txt") / "hf_read_token.txt").read_text(
        encoding="utf-8").strip()
    source = find_dataset("scoring", "text", "eval")
    package = Path("/kaggle/tmp/pkg/geetha")
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)

    spec_common = {
        "prompts": PROMPTS, "seed": SEED, "out_dir": str(OUT_DIR),
        "package_parent": str(package.parent),
        "conformer_dir": snapshot_download(CONFORMER_ID, revision=CONFORMER_REVISION,
                                           token=token),
        "transcribe_dir": snapshot_download(TRANSCRIBE_ID, revision=TRANSCRIBE_REVISION,
                                            token=token),
        "snac_id": SNAC_ID, "snac_revision": SNAC_REVISION,
        "revisions": {CONFORMER_ID: CONFORMER_REVISION, TRANSCRIBE_ID: TRANSCRIBE_REVISION},
    }
    work = Path("/kaggle/tmp/bench")
    work.mkdir(parents=True, exist_ok=True)
    release = work / "release"
    release.mkdir(exist_ok=True)
    for name, encoded in PAYLOAD.items():
        target = work / name if name == "worker.py" else release / name
        target.write_bytes(base64.b64decode(encoded))
        log(f"  unpacked {name}: {target.stat().st_size:,} bytes  "
            f"sha256 {hashlib.sha256(target.read_bytes()).hexdigest()[:16]}")

    if GROUP == "ours":
        base_dir = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)
        ckpt = Path(snapshot_download(CKPT_REPO, token=token,
                                      allow_patterns=[f"{ADAPTER}/*"]))
        adapter_dir = ckpt / ADAPTER
        if not (adapter_dir / "adapter_config.json").exists():
            adapter_dir = sorted(adapter_dir.rglob("adapter_config.json"))[0].parent
        spec_common.update(base_dir=base_dir, adapter_dir=str(adapter_dir),
                           release_dir=str(release))
        spec_common["revisions"].update({MODEL_ID: BASE_REVISION, "geetha_adapter": ADAPTER})
    token = ""
    del token
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    if GROUP == "rivals":
        # Public models: fetched without the token, at pinned commits, and only the
        # files inference needs (no server, training or demo scripts).
        rumik_dir = snapshot_download(
            RUMIK_ID, revision=RUMIK_REVISION, token=False,
            allow_patterns=["*.json", "*.safetensors", "codec/*", *RUMIK_CODE_SHA256])
        snor_dir = snapshot_download(SNOR_ID, revision=SNOR_REVISION, token=False,
                                     allow_patterns=["*.json", "*.safetensors", "*.jinja"])
        for name, expected in RUMIK_CODE_SHA256.items():
            found = hashlib.sha256((Path(rumik_dir) / name).read_bytes()).hexdigest()
            if found != expected:
                raise SystemExit(f"{name} is not the file that was reviewed: {found}")
        extra = sorted(p.name for p in Path(rumik_dir).rglob("*.py")
                       if p.name not in RUMIK_CODE_SHA256)
        if extra:
            raise SystemExit(f"unreviewed Python in the Rumik snapshot: {extra}")
        spec_common.update(rumik_dir=rumik_dir, snor_dir=snor_dir)
        spec_common["revisions"].update({RUMIK_ID: RUMIK_REVISION, SNOR_ID: SNOR_REVISION,
                                         SNAC_ID: SNAC_REVISION})
    snapshot_download(SNAC_ID, revision=SNAC_REVISION, token=False)
    log(f"revisions: {spec_common['revisions']}")

    logs = OUT_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    parallel = gpus >= len(SYSTEMS)
    budget = RENDER_BUDGET_PARALLEL_S if parallel else RENDER_BUDGET_SERIAL_S
    started = time.time()

    def launch(system: str, gpu: int) -> subprocess.Popen:
        spec_path = work / f"spec_{system}.json"
        spec_path.write_text(json.dumps({**spec_common, "system": system,
                                         "render_budget_s": budget}, ensure_ascii=False),
                             encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
        process = subprocess.Popen(
            [sys.executable, str(work / "worker.py"), str(spec_path)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace")

        def relay() -> None:
            with open(logs / f"{system}.log", "w", encoding="utf-8") as sink:
                for line in process.stdout:
                    sink.write(line)
                    sink.flush()
                    if line.startswith(f"[{system}]"):
                        log(line.rstrip())

        thread = threading.Thread(target=relay, daemon=True)
        thread.start()
        process.relay = thread            # type: ignore[attr-defined]
        log(f"started {system} on GPU {gpu} (render budget {budget / 3600:.1f} h)")
        return process

    def finish(system: str, process: subprocess.Popen) -> None:
        code = process.wait()
        process.relay.join(timeout=30)    # type: ignore[attr-defined]
        log(f"{system} exited {code} after {time.time() - started:.0f} s")
        if code:
            tail = (logs / f"{system}.log").read_text(encoding="utf-8").splitlines()[-40:]
            log("\n".join(f"    {line}" for line in tail))

    if parallel:
        running = [(system, launch(system, gpu)) for gpu, system in enumerate(SYSTEMS)]
        for system, process in running:
            finish(system, process)
    else:
        for system in SYSTEMS:
            finish(system, launch(system, 0))

    log("")
    log("=== summary (CER is scored on the PC from these transcripts) ===")
    for system in SYSTEMS:
        path = OUT_DIR / f"bench_{system}.json"
        if not path.exists():
            log(f"  {system}: NO RESULT FILE")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        clips = report["clips"]
        good = sum(1 for c in clips.values() if c["valid"])
        skipped = sum(1 for c in clips.values() if c["reason"] == "not_run")
        log(f"  {system}: valid {good}/{len(clips)}  not run {skipped}  transcripts "
            f"{ {j: len(h) for j, h in report['hypotheses'].items()} }  "
            f"finished {report.get('finished', False)}  failure {report.get('failure')}")
    log(f"wrote results in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
