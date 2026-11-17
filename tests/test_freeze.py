"""The eval-vault freeze: hashes, parts, and the refusal to re-freeze."""

from __future__ import annotations

import hashlib

import pytest

from geetha.eval.freeze import (
    PARTS,
    FrozenFile,
    freeze_rows,
    hash_file,
    merge_rows,
)


def _write(tmp_path, name: str, content: bytes):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestHashing:
    def test_the_digest_matches_hashlib(self, tmp_path):
        payload = b"tamil references" * 1000
        path = _write(tmp_path, "refs.jsonl", payload)
        digest, size = hash_file(path)
        assert digest == hashlib.sha256(payload).hexdigest()
        assert size == len(payload)

    def test_a_file_larger_than_one_chunk_hashes_correctly(self, tmp_path):
        payload = bytes(3_000_000)
        path = _write(tmp_path, "big.tar", payload)
        digest, size = hash_file(path)
        assert digest == hashlib.sha256(payload).hexdigest()
        assert size == 3_000_000

    def test_an_empty_file_still_hashes(self, tmp_path):
        path = _write(tmp_path, "empty.txt", b"")
        digest, size = hash_file(path)
        assert size == 0
        assert digest == hashlib.sha256(b"").hexdigest()


class TestFreezeRows:
    def test_rows_carry_relative_paths(self, tmp_path):
        _write(tmp_path, "sets/a.parquet", b"a")
        rows = freeze_rows(
            "fleurs_test",
            "BENCH",
            (tmp_path / "sets").glob("*.parquet"),
            part=1,
            frozen_on="2026-10-27",
            root=tmp_path,
        )
        assert [row.path for row in rows] == ["sets/a.parquet"]
        assert rows[0].asset == "fleurs_test"
        assert rows[0].part == 1

    def test_paths_use_forward_slashes_on_every_platform(self, tmp_path):
        _write(tmp_path, "deep/nested/b.parquet", b"b")
        rows = freeze_rows(
            "x", "BENCH", (tmp_path / "deep" / "nested").glob("*"),
            part=1, frozen_on="2026-10-27", root=tmp_path,
        )
        assert "\\" not in rows[0].path

    def test_an_asset_matching_nothing_is_refused(self, tmp_path):
        # A row for a set that is not there would read as "frozen" forever.
        with pytest.raises(ValueError, match="matched no files"):
            freeze_rows("ghost", "BENCH", [], part=1, frozen_on="2026-10-27", root=tmp_path)

    def test_an_unknown_part_is_refused(self, tmp_path):
        _write(tmp_path, "a.txt", b"a")
        with pytest.raises(ValueError, match="unknown freeze part"):
            freeze_rows("x", "BENCH", tmp_path.glob("*.txt"), part=9, frozen_on="d", root=tmp_path)

    def test_refreezing_the_same_file_is_refused(self, tmp_path):
        _write(tmp_path, "a.parquet", b"a")
        first = freeze_rows(
            "set", "BENCH", tmp_path.glob("*.parquet"), part=1, frozen_on="2026-10-27", root=tmp_path
        )
        with pytest.raises(ValueError, match="already frozen"):
            freeze_rows(
                "set", "BENCH", tmp_path.glob("*.parquet"),
                part=2, frozen_on="2026-11-02", root=tmp_path, existing=first,
            )

    def test_a_different_asset_name_may_cover_the_same_file(self, tmp_path):
        # The escape hatch is deliberate: a changed set gets a new name and a new row,
        # visibly, rather than a silent overwrite.
        _write(tmp_path, "a.parquet", b"a")
        first = freeze_rows(
            "set_v1", "BENCH", tmp_path.glob("*.parquet"), part=1, frozen_on="2026-10-27", root=tmp_path
        )
        second = freeze_rows(
            "set_v2", "BENCH", tmp_path.glob("*.parquet"),
            part=2, frozen_on="2026-11-02", root=tmp_path, existing=first,
        )
        assert second[0].sha256 == first[0].sha256


class TestParts:
    def test_both_parts_are_described(self):
        assert set(PARTS) == {1, 2}
        assert all(text for text in PARTS.values())

    def test_merge_is_append_only_and_keeps_earlier_dates(self):
        old = [FrozenFile("2026-10-27", 1, "a", "BENCH", "a", "h", 1)]
        new = [FrozenFile("2026-11-02", 2, "b", "PROMPTS", "b", "h2", 2)]
        merged = merge_rows(old, new)
        assert [row.frozen_on for row in merged] == ["2026-10-27", "2026-11-02"]
        assert merged[0] is old[0]
