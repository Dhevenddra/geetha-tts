"""Tests for scripts.push_code_dataset — what gets shipped to Kaggle as geetha-code.

A Kaggle job imports ``geetha`` from this dataset. Two things can go wrong silently:
a module the job needs is missing (the job dies minutes in, after the model download),
or something that is not source code rides along (there is a secrets directory one
level up). Both are cheap to test and expensive to discover on the cluster.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.push_code_dataset import PACKAGE_ROOT, files_to_ship  # noqa: E402


def test_every_module_the_kaggle_jobs_import_is_shipped():
    shipped = {p.as_posix() for p in files_to_ship(PACKAGE_ROOT)}
    # Imported by name inside kaggle/*.py; a missing one is a dead job.
    for needed in [
        "text/normalise.py",
        "text/chunk.py",
        "text/contract.py",
        "text/tags.py",
        "audio/join.py",
        "scoring/metrics.py",
        "scoring/bootstrap.py",
        "eval/naturalness.py",
        "eval/contamination.py",
    ]:
        assert needed in shipped, needed


def test_every_package_ships_its_init():
    shipped = {p.as_posix() for p in files_to_ship(PACKAGE_ROOT)}
    packages = {p.rsplit("/", 1)[0] for p in shipped if "/" in p}
    for package in packages:
        assert f"{package}/__init__.py" in shipped, package


def test_only_python_sources_are_shipped():
    for path in files_to_ship(PACKAGE_ROOT):
        assert path.suffix == ".py", path


def test_compiled_caches_are_not_shipped():
    for path in files_to_ship(PACKAGE_ROOT):
        assert "__pycache__" not in path.parts, path


def test_nothing_outside_the_package_is_shipped():
    # Relative paths only, and no escape upwards: the secrets directory is a sibling
    # of the repo and must stay unreachable from this list.
    for path in files_to_ship(PACKAGE_ROOT):
        assert not path.is_absolute(), path
        assert ".." not in path.parts, path


def test_a_missing_package_root_is_an_error_not_an_empty_upload():
    with pytest.raises(FileNotFoundError):
        files_to_ship(Path("E:/Geetha/geetha/geetha-does-not-exist"))
