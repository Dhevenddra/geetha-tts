r"""Freezing the eval vault: what was measured on, fixed by hash, on an honest date.

Why a freeze at all
-------------------
Every number we publish is our own re-run. That is only worth anything if the thing we
re-ran on cannot have moved in between. A SHA-256 per file turns "we evaluated on
DiarBench Tamil" into a claim anyone can check, including us in three weeks when a
result looks too good.

Why it is in two parts (R13)
----------------------------
The benchmark corpora and ``refs_v1`` are ready now. The authored prompt sets --
Directability-Ta, Naturalness-Ta -- are not: they are waiting on the operator's reading
of the Tamil, and Naturalness-Ta is waiting on the contamination check that decides
which corpus it is even drawn from.

The tempting move is to freeze everything on the runbook's date and fix the prompt sets
up quietly afterwards. That is backdating, and it would make the freeze a decoration.
So the vault freezes in parts, each carrying **the date it actually happened**, and the
only rule that matters is the one both parts share: everything is frozen before the
first training step, so nothing in the vault can have been influenced by a model of
ours.

``PARTS`` records what each part covers. A part is written once; re-freezing an asset
that is already recorded is refused, because the second row could only disagree with the
first and there would be no way to tell which run a published number came from. Changing
a frozen asset means a new asset name and a new row, visibly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "FrozenFile",
    "PARTS",
    "already_frozen",
    "freeze_rows",
    "hash_file",
    "merge_rows",
]

PARTS: dict[int, str] = {
    1: "benchmark corpora and refs_v1 — the sets every published number is measured on",
    2: "authored prompt sets — Directability-Ta and Naturalness-Ta, once the Tamil is "
    "approved and the contamination check has settled Naturalness-Ta's source",
}

# 1 MiB. Large enough that hashing is I/O-bound rather than call-bound, small enough
# that a 10 GB tar does not have to fit in memory.
_CHUNK = 1 << 20


@dataclass(frozen=True)
class FrozenFile:
    frozen_on: str
    part: int
    asset: str
    kind: str
    path: str
    sha256: str
    bytes: int

    def key(self) -> tuple[str, str]:
        return (self.asset, self.path)


FIELDS = [field.name for field in fields(FrozenFile)]


def hash_file(path: Path) -> tuple[str, int]:
    """SHA-256 and byte count, streamed."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def already_frozen(existing: Sequence[FrozenFile]) -> set[tuple[str, str]]:
    return {row.key() for row in existing}


def freeze_rows(
    asset: str,
    kind: str,
    paths: Iterable[Path],
    *,
    part: int,
    frozen_on: str,
    root: Path,
    existing: Sequence[FrozenFile] = (),
) -> list[FrozenFile]:
    """Hash ``paths`` into rows, skipping nothing and refusing a silent re-freeze.

    Paths are recorded relative to ``root`` so the manifest stays meaningful if the
    project ever moves off ``E:``.
    """
    if part not in PARTS:
        raise ValueError(f"unknown freeze part {part}; known parts are {sorted(PARTS)}")

    seen = already_frozen(existing)
    rows: list[FrozenFile] = []
    for path in sorted(paths):
        relative = str(path.relative_to(root)).replace("\\", "/")
        if (asset, relative) in seen:
            raise ValueError(
                f"{asset}:{relative} is already frozen. A second row could only "
                "disagree with the first — give the changed set a new asset name."
            )
        digest, size = hash_file(path)
        rows.append(
            FrozenFile(
                frozen_on=frozen_on,
                part=part,
                asset=asset,
                kind=kind,
                path=relative,
                sha256=digest,
                bytes=size,
            )
        )
    if not rows:
        raise ValueError(f"{asset} matched no files — a freeze row with nothing in it is a lie")
    return rows


def merge_rows(existing: Sequence[FrozenFile], new: Sequence[FrozenFile]) -> list[FrozenFile]:
    """Append-only: earlier parts keep their dates and their order."""
    return [*existing, *new]
