r"""A register of Kaggle jobs, so no run is ever lost to a dormant session.

The problem this solves
-----------------------
Kaggle kernels run **server-side**. Once pushed, a job keeps running whether or not
anyone is watching: it survives this machine sleeping, the network dropping, and the
assistant running out of budget mid-sprint. Its outputs sit in Kaggle's storage until
someone fetches them.

What does *not* survive is the knowledge that the job exists. A session that ends
without writing that down means the next session either re-runs the work -- burning a
slice of a 30 h weekly quota we cannot buy more of -- or never collects a result that
was finished and waiting.

So every pushed job gets a row here *before* it is pushed, and ``/session-start`` reads
this file. The register is the memory; the kernel is the worker.

    py -3.11 -m ops.kaggle_jobs register <slug> <purpose> <result_path>
    py -3.11 -m ops.kaggle_jobs status            # every unfinished job
    py -3.11 -m ops.kaggle_jobs pull <slug>       # fetch outputs, mark collected

Kaggle quota is free but finite (30 GPU-h/week, read off the site 27 Oct 2026). This
file is also the record of what that quota was spent on.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

__all__ = ["KaggleJob", "REGISTER", "pending_jobs", "read_register", "register_job", "write_register"]

REGISTER = Path(r"E:\Geetha\manifests\kaggle_jobs.csv")

# Statuses a row can hold. "pushed" and "running" both mean "go and look"; the
# distinction is only whether anyone has checked since.
OPEN_STATUSES = frozenset({"pushed", "running"})


@dataclass(frozen=True)
class KaggleJob:
    pushed_at: str
    slug: str
    purpose: str
    status: str
    gpu: str
    result_path: str
    collected_at: str
    notes: str

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES


FIELDS = [f.name for f in fields(KaggleJob)]


def read_register(path: Path = REGISTER) -> list[KaggleJob]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [KaggleJob(**{key: row.get(key, "") or "" for key in FIELDS}) for row in csv.DictReader(handle)]


def write_register(jobs: Sequence[KaggleJob], path: Path = REGISTER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for job in jobs:
            writer.writerow({key: getattr(job, key) for key in FIELDS})


def register_job(
    slug: str,
    purpose: str,
    result_path: str,
    *,
    gpu: str = "T4x2",
    notes: str = "",
    path: Path = REGISTER,
    now: str | None = None,
) -> KaggleJob:
    """Add a row, or refresh an existing one for the same slug.

    Re-pushing a slug replaces its row rather than appending a second: Kaggle keeps one
    kernel per slug, so two open rows for it could only ever disagree.
    """
    job = KaggleJob(
        pushed_at=now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        slug=slug,
        purpose=purpose,
        status="pushed",
        gpu=gpu,
        result_path=result_path,
        collected_at="",
        notes=notes,
    )
    jobs = [existing for existing in read_register(path) if existing.slug != slug]
    jobs.append(job)
    write_register(jobs, path)
    return job


def mark(slug: str, status: str, *, notes: str | None = None, path: Path = REGISTER) -> None:
    jobs = read_register(path)
    updated: list[KaggleJob] = []
    for job in jobs:
        if job.slug == slug:
            collected = (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if status == "collected"
                else job.collected_at
            )
            job = KaggleJob(
                pushed_at=job.pushed_at,
                slug=job.slug,
                purpose=job.purpose,
                status=status,
                gpu=job.gpu,
                result_path=job.result_path,
                collected_at=collected,
                notes=job.notes if notes is None else notes,
            )
        updated.append(job)
    write_register(updated, path)


def pending_jobs(path: Path = REGISTER) -> list[KaggleJob]:
    """Everything a new session must go and look at. This is what /session-start reads."""
    return [job for job in read_register(path) if job.is_open]


def _kaggle(*args: str) -> tuple[int, str]:
    process = subprocess.run(
        [sys.executable, "-m", "kaggle.cli", *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    # Kaggle's log can hold bytes the console code page cannot decode; under the default
    # encoding the reader thread dies and stdout comes back None after the download.
    return process.returncode, ((process.stdout or "") + (process.stderr or "")).strip()


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    command, rest = argv[0], argv[1:]

    if command == "register":
        if len(rest) < 3:
            print("usage: register <slug> <purpose> <result_path>")
            return 2
        job = register_job(rest[0], rest[1], rest[2])
        print(f"registered {job.slug} -> {job.result_path}")
        return 0

    if command == "status":
        open_jobs = pending_jobs()
        if not open_jobs:
            print("no Kaggle jobs outstanding.")
            return 0
        for job in open_jobs:
            code, out = _kaggle("kernels", "status", job.slug)
            live = out.splitlines()[-1] if out else "unknown"
            print(f"{job.slug}  pushed {job.pushed_at}  [{live}]  {job.purpose}")
            if "complete" in live.lower():
                print(f"    -> ready to pull into {job.result_path}")
        return 0

    if command == "pull":
        if not rest:
            print("usage: pull <slug>")
            return 2
        slug = rest[0]
        match = [job for job in read_register() if job.slug == slug]
        if not match:
            print(f"{slug} is not in the register")
            return 1
        destination = Path(match[0].result_path)
        destination.mkdir(parents=True, exist_ok=True)
        code, out = _kaggle("kernels", "output", slug, "-p", str(destination))
        print(out)
        if code == 0:
            mark(slug, "collected")
            print(f"collected into {destination}")
        return code

    print(f"unknown command {command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
