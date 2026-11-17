r"""Put the read-only HF token into a private Kaggle dataset, so jobs can run unattended.

    py -3.11 scripts\push_token_dataset.py

Reads ``E:\Geetha\.secrets\hf_read_token.txt`` and publishes it as the single file of
the **private** Kaggle dataset ``dhevenddra/geetha-secrets``. The token is never
printed, logged, or written into the repo.

Why this exists
---------------
Measured 27 Oct 2026: a kernel pushed through the Kaggle API does not keep a secret
attached through the UI. Version 4 of ``geetha-c1-check`` died in 1.7 seconds with
"Kaggle secret unavailable (ConnectionError)" after the operator had attached
``HF_TOKEN`` by hand. Every remaining Kaggle job (B6-B10) needs the same gated model, so
without a credential path that survives an API push, each one needs the operator sitting
at the keyboard.

What this deliberately refuses to do
------------------------------------
It will not upload a token that can write. ``.secrets\geetha.env`` holds a Hugging Face
**write** token -- the one that can push model weights to our private repos -- and
copying that onto a third-party store to save a few minutes would be trading a real risk
for a small convenience. The check below asks the Hub what the token can actually do and
stops if the answer is anything but read-only.

That check is not a formality. The failure it prevents is silent: a write token works
perfectly for every job, so nothing would ever go wrong visibly, and the mistake would
only surface if the dataset were exposed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_FILE = Path(r"E:\Geetha\.secrets\geetha.env")
ENV_KEY = "HF_TOKEN_READ"
TOKEN_FILE = Path(r"E:\Geetha\.secrets\hf_read_token.txt")


def read_token() -> str:
    """Prefer HF_TOKEN_READ from geetha.env; fall back to the standalone file.

    The operator created HF_TOKEN_READ as a fine-grained, read-only token on
    27 Oct 2026. HF_TOKEN in the same file is a *write* token and must never be read
    here -- the key is named explicitly rather than taken from the environment, so a
    stray export cannot substitute one for the other.
    """
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == ENV_KEY:
                token = value.strip().strip('"').strip("'")
                if token:
                    print(f"using {ENV_KEY} from geetha.env")
                    return token
    if TOKEN_FILE.exists():
        print(f"using {TOKEN_FILE.name}")
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return ""
DATASET_ID = "dhevenddra/geetha-secrets"
FILENAME = "hf_read_token.txt"

# Roles the Hub reports for a token. Only these may leave this machine.
ALLOWED_ROLES = {"read", "fineGrained"}
GATED_REPOS = ["bodhan-ai/indic-speak"]

# A fine-grained token reports role "fineGrained" whatever is ticked inside it, so the
# role alone is not a safety check -- the operator's first fine-grained token carried
# repo.write, inference.endpoints.write, job.write and user.billing.read, and the role
# string said nothing about any of it. The permissions have to be read directly.
#
# Anything ending in ".write" is disqualifying. So is reading billing. The test is a
# denylist on purpose: a new permission Hugging Face invents later should not become
# automatically acceptable because nobody updated an allowlist.
FORBIDDEN_PERMISSION_SUFFIXES = (".write",)
FORBIDDEN_PERMISSIONS = {"user.billing.read"}


def write_permissions(whoami: dict) -> list[str]:
    """Every permission on this token that is not plain reading of repo content."""
    auth = (whoami.get("auth") or {}).get("accessToken") or {}
    fine = auth.get("fineGrained") or {}
    found: set[str] = set()
    scopes = list(fine.get("scoped") or [])
    for scope in scopes:
        for permission in scope.get("permissions") or []:
            if permission.endswith(FORBIDDEN_PERMISSION_SUFFIXES) or permission in FORBIDDEN_PERMISSIONS:
                found.add(permission)
    for permission in fine.get("global") or []:
        if permission.endswith(FORBIDDEN_PERMISSION_SUFFIXES) or permission in FORBIDDEN_PERMISSIONS:
            found.add(permission)
    return sorted(found)


# Hugging Face tokens are "hf_" plus 34 alphanumerics. Checking the shape before
# calling the Hub turns "Invalid user token" plus a stack trace into a sentence saying
# what is actually wrong -- which matters, because the operator cannot paste the value
# into a chat for me to look at, so the error message is the only diagnostic either of
# us gets.
TOKEN_PATTERN = r"hf_[A-Za-z0-9]{34}"


def describe_token(token: str) -> tuple[str, str, dict] | None:
    """Ask the Hub who this token is and what it may do, or None if it is not valid."""
    import re

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    if not re.fullmatch(TOKEN_PATTERN, token):
        shape = token[:3] + "..." if len(token) > 3 else "(too short)"
        print(f"that is not a Hugging Face token: {len(token)} chars starting {shape!r}")
        print("HF tokens are 'hf_' followed by 34 alphanumerics.")
        print("Get one at https://huggingface.co/settings/tokens -> Create new token -> Read")
        return None

    try:
        info = HfApi(token=token).whoami()
    except HfHubHTTPError as error:
        print(f"the Hub rejected this token: {str(error).splitlines()[0]}")
        print("It may have been revoked, or copied from a different account.")
        return None

    access = (info.get("auth") or {}).get("accessToken") or {}
    return info.get("name", "?"), access.get("role", "unknown"), info


def check_it_can_read_what_we_need(token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for repo in GATED_REPOS:
        api.hf_hub_download(repo, "config.json")
        print(f"  can read {repo}")


def main() -> int:
    token = read_token()
    if not token:
        print(f"no {ENV_KEY} in {ENV_FILE} and no {TOKEN_FILE}.")
        return 1

    described = describe_token(token)
    if described is None:
        return 1
    user, role, whoami = described
    print(f"token belongs to {user}, role {role!r}, length {len(token)}")

    writes = write_permissions(whoami)
    if writes:
        print()
        print("REFUSING to upload: this token can do more than read.")
        for permission in writes:
            print(f"  {permission}")
        print("Make a read-only token instead: huggingface.co/settings/tokens ->")
        print("Create new token -> the 'Read' tab (not Fine-grained).")
        return 2

    if role not in ALLOWED_ROLES:
        print()
        print(f"REFUSING to upload: this token's role is {role!r}, not read-only.")
        print("A write token on a third-party store is a real risk traded for a small")
        print("convenience. Make a fine-grained read-only token instead.")
        return 2

    if role == "fineGrained":
        # Fine-grained tokens report one role regardless of what was ticked, so the
        # only honest check is whether it can actually do the one thing we need.
        print("  fine-grained token — verifying it can read the gated model:")
        check_it_can_read_what_we_need(token)

    staging = Path(tempfile.mkdtemp(prefix="geetha-secrets-"))
    try:
        (staging / FILENAME).write_text(token, encoding="utf-8")
        (staging / "dataset-metadata.json").write_text(
            json.dumps({"title": "geetha-secrets", "id": DATASET_ID,
                        "licenses": [{"name": "other"}]}, indent=2),
            encoding="utf-8",
        )
        existing = subprocess.run(
            [sys.executable, "-m", "kaggle.cli", "datasets", "status", DATASET_ID],
            capture_output=True, text=True,
        )
        if existing.returncode == 0 and "ready" in existing.stdout.lower():
            command = ["datasets", "version", "-p", str(staging), "-m", "rotate token", "-q"]
        else:
            command = ["datasets", "create", "-p", str(staging), "-q"]
        result = subprocess.run(
            [sys.executable, "-m", "kaggle.cli", *command], capture_output=True, text=True
        )
        output = (result.stdout + result.stderr).strip()
        # Never echo the staging contents; only Kaggle's own response.
        print(output)
        if result.returncode != 0:
            return result.returncode
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print()
    print(f"{DATASET_ID} is private and holds {FILENAME}.")
    print("Kernels listing it in dataset_sources can now run unattended.")
    print("To rotate: replace the token file and run this again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
