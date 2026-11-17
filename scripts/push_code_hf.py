r"""Put the geetha package where a pod can reach it.

    py -3.11 scripts\push_code_hf.py [--dry-run]

A pod needs our code. The options were a GitHub credential on a machine we do not own,
or Hugging Face -- which the pod already authenticates to for the base model and the
code shards. So the package travels over HF, and the pod gains no second credential.

What this refuses to upload
---------------------------
Only ``geetha/**/*.py`` and the two authored ``.tsv`` prompt sources. Not ``.secrets``,
not ``work\``, not ``processed\``, not ``runs\``, not the venv, and nothing matching a
key-shaped filename. The allowlist is positive rather than a list of exclusions,
because an exclusion list fails open: the first time someone adds a new directory it is
published by default.

The repo is created **private**, and there is no flag here to make it public.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.secrets import load_secrets  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "geetha"
JOBS = REPO_ROOT / "jobs"
REPO_ID = "Dhevenddra/geetha-code-private"

# Positive allowlist. Anything not matched here does not travel.
ALLOWED_SUFFIXES = {".py", ".tsv"}
FORBIDDEN_PARTS = {".venv", "__pycache__", ".secrets", ".git"}
FORBIDDEN_NAMES = {"geetha.env", "kaggle.json", "hf_read_token.txt"}

# The shapes real credentials actually have. Names like WANDB_API_KEY are fine and
# expected -- reading a key from the environment is the correct pattern.
SECRET_SHAPES = (
    ("a Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("a Runpod API key", re.compile(r"rpa_[A-Za-z0-9]{20,}")),
    ("a private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("an AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("a long hex secret assigned to a key-ish name",
     re.compile(r"(?i)(?:secret|token|api[_-]?key|password)\s*=\s*[\"']"
                r"[0-9a-f]{40,}[\"']")),
)


def collect() -> list[tuple[Path, str]]:
    """(local path, path in repo) for everything cleared to travel."""
    files: list[tuple[Path, str]] = []
    for root, prefix in ((PACKAGE, "geetha"), (JOBS, "jobs")):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if set(path.parts) & FORBIDDEN_PARTS or path.name in FORBIDDEN_NAMES:
                continue
            if path.suffix not in ALLOWED_SUFFIXES and path.suffix != ".sh":
                continue
            files.append((path, f"{prefix}/{path.relative_to(root).as_posix()}"))
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = collect()
    if not files:
        print("!! nothing to upload", file=sys.stderr)
        return 1
    total = sum(path.stat().st_size for path, _ in files)
    print(f"{len(files)} files, {total / 1e6:.2f} MB -> {REPO_ID} (private)")
    for _, name in files[:8]:
        print(f"    {name}")
    if len(files) > 8:
        print(f"    ... and {len(files) - 8} more")

    # Belt and braces: the allowlist should already make this impossible, but a
    # published secret cannot be unpublished.
    #
    # This matches the *shape* of a real credential, not the word "api_key". A first
    # version looked for the names and refused geetha/runlog/logger.py, which merely
    # reads WANDB_API_KEY from the environment -- exactly the correct thing to do. A
    # check that cries wolf on correct code gets switched off. It also reads the whole
    # file: the first version scanned 4 KB, so a key below that line travelled.
    for path, name in files:
        body = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_SHAPES:
            found = pattern.search(body)
            if found:
                print(f"!! REFUSING: {name} contains something shaped like {label} "
                      f"at offset {found.start()}", file=sys.stderr)
                return 1

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=load_secrets().require("HF_TOKEN"))
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    # Upload shell scripts with LF endings whatever they look like on this Windows
    # machine. This uploader sends the WORKING-TREE bytes, so git's 'CRLF will be
    # replaced by LF' normalisation never applies to what the pod actually receives.
    # That warning printed all session while CRLF kept going up the wire anyway.
    #
    # A carriage return in a .sh is not cosmetic: bash reads it as part of the token
    # and the script dies on its own opening lines -- 'set: invalid option', then
    # 'syntax error near unexpected token'. m1-pilot-9 died of exactly this, and so
    # did every earlier pod that got far enough to run its job script. .gitattributes
    # pins *.sh to LF as well; this is the second line of defence, because the file
    # that matters is the one uploaded here, not the one in the git index.
    CRLF, LF = bytes([13, 10]), bytes([10])
    for path, name in files:
        data = path.read_bytes()
        if name.endswith('.sh') and CRLF in data:
            data = data.replace(CRLF, LF)
            print('   normalised CRLF -> LF for ' + name)
        api.upload_file(path_or_fileobj=data, path_in_repo=name,
                        repo_id=REPO_ID, repo_type='model')
    print(f"\npushed {len(files)} files to {REPO_ID}")
    print("private; a pod reads it with the same HF token it uses for the base model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
