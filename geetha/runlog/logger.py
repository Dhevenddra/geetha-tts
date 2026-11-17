"""Provenance logging for every ETL and evaluation output.

Runbook rule 9 requires that each output record the hash of its inputs and the seed
it ran with, so any number we publish can be traced to the exact bytes and code
that produced it.

Secrets are actively refused rather than merely "not added". A provenance JSON is
the sort of file that ends up in an issue or a model card, and the project rules is
unambiguous: keys are never printed, logged or committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geetha.text.normalise import norm_text

__all__ = ["RunLog", "file_sha256", "text_hash"]

# Whole-word matches only: "keep_top_fraction" is fine, "hf_token" is not.
_SECRET_NAME = re.compile(
    r"(?:^|[^a-z0-9])(token|key|secret|password|passwd|pwd|credential|cookie|auth)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)

_CHUNK = 1 << 20


def text_hash(text: str) -> str:
    """The leak hash of runbook rule 3: sha256 of the normalised sentence."""
    return hashlib.sha256(norm_text(text).encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    """sha256 of a file, read in chunks so manifests of any size are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


class RunLog:
    """Collects provenance for one run and writes it as JSON on ``finish``."""

    def __init__(self, run_id: str, seed: int, out_dir: str | os.PathLike[str]):
        self.run_id = run_id
        self.seed = seed
        self.out_dir = Path(out_dir)
        self.inputs: dict[str, str] = {}
        self.outputs: dict[str, str] = {}
        self.params: dict[str, Any] = {}
        self.git_commit = _git_commit()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._started_monotonic = time.monotonic()

    @staticmethod
    def _reject_secrets(name: str) -> None:
        if _SECRET_NAME.search(name):
            raise ValueError(
                f"refusing to log {name!r}: the name looks like a secret, and secrets are "
                "never written to a provenance file"
            )

    def add_param(self, name: str, value: Any) -> None:
        """Record a run parameter. Raises if the name looks like a credential."""
        self._reject_secrets(name)
        self.params[name] = value

    def add_input(self, path: str | os.PathLike[str]) -> str:
        """Record an input file and its sha256."""
        digest = file_sha256(path)
        self.inputs[str(path)] = digest
        return digest

    def add_output(self, path: str | os.PathLike[str]) -> str:
        """Record an output file and its sha256."""
        digest = file_sha256(path)
        self.outputs[str(path)] = digest
        return digest

    def to_dict(self, status: str, note: str | None = None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": status,
            "note": note,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - self._started_monotonic, 3),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "params": self.params,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    def finish(self, status: str = "ok", note: str | None = None) -> Path:
        """Write ``<out_dir>/<run_id>.runlog.json`` and return its path."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}.runlog.json"
        payload = json.dumps(self.to_dict(status, note), ensure_ascii=False, indent=2)
        path.write_text(payload + "\n", encoding="utf-8")
        return path
