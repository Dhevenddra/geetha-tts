r"""Publish the ``geetha`` package as the private Kaggle dataset ``geetha-code``.

    py -3.11 scripts\push_code_dataset.py

Every Kaggle job copies the package out of this dataset and imports it, so the dataset
has to be re-pushed whenever a job needs a module that was written since the last push.
B7 revision 2 is the first job to import ``geetha.audio``, which is why this script
exists instead of the manual upload that got the first version there.

What it ships, and what it refuses to
-------------------------------------
Python sources under ``geetha\``, nothing else. Not the tests, not the ops tooling, and
emphatically nothing from outside the package: ``.secrets\`` is two directories up from
here and a glob with one wrong character would put a Hugging Face write token on a
third-party store. The file list is therefore built by one function with tests on it
(``tests\test_code_dataset.py``) rather than inline in a subprocess call.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "geetha"
DATASET_ID = "dhevenddra/geetha-code"


def files_to_ship(package_root: Path) -> list[Path]:
    """Python sources under ``package_root``, as paths relative to it, sorted.

    Relative on purpose: the caller stages them under a fresh temporary directory, and
    a relative list cannot reach anything the package does not contain.
    """
    package_root = Path(package_root)
    if not package_root.is_dir():
        raise FileNotFoundError(f"no package at {package_root}")
    return sorted(
        path.relative_to(package_root)
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def stage(package_root: Path, destination: Path) -> list[Path]:
    shipped = files_to_ship(package_root)
    for relative in shipped:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_root / relative, target)
    (destination / "dataset-metadata.json").write_text(
        json.dumps({"title": "geetha-code", "id": DATASET_ID,
                    "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8",
    )
    return shipped


def _kaggle(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kaggle.cli", *args], capture_output=True, text=True
    )


def main(argv: list[str] | None = None) -> int:
    message = " ".join(argv or sys.argv[1:]) or "sync geetha package"

    staging = Path(tempfile.mkdtemp(prefix="geetha-code-"))
    try:
        shipped = stage(PACKAGE_ROOT, staging)
        print(f"{len(shipped)} modules from {PACKAGE_ROOT}:")
        for relative in shipped:
            print(f"  {relative.as_posix()}")

        status = _kaggle("datasets", "status", DATASET_ID)
        if status.returncode == 0 and "ready" in status.stdout.lower():
            command = ["datasets", "version", "-p", str(staging), "-m", message, "-q", "-r", "zip"]
        else:
            command = ["datasets", "create", "-p", str(staging), "-q", "-r", "zip"]
        result = _kaggle(*command)
        print((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            return result.returncode
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print()
    print(f"{DATASET_ID} updated. A kernel pushed from now on sees these modules;")
    print("one already queued does not — re-push the kernel after this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
