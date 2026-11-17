"""The shard scanner behind ETL step 1.

Built against real parquet files written in the test, not mocks: the failures worth
catching here are parquet-shaped (a struct column, a corrupt blob, a half-finished
run) and a mock would reproduce none of them.
"""

from __future__ import annotations

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from geetha.audio.qc_scan import ShardSpec, qc_shard, scan_shards

IVR = ShardSpec(source="indicvoices_r", audio_column="audio", carry=("speaker_id", "scenario"))


def wav_bytes(seconds: float, sample_rate: int = 48_000, amplitude: float = 0.5) -> bytes:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    buffer = io.BytesIO()
    sf.write(buffer, amplitude * np.sin(2 * np.pi * 440 * t), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_shard(path, clips, carry=None, duration_column="duration"):
    """``clips`` is a list of (name, bytes). ``carry`` a dict of column -> values."""
    columns = {
        "audio": [{"bytes": blob, "path": name} for name, blob in clips],
        duration_column: [1.0] * len(clips),
    }
    columns.update(carry or {"speaker_id": ["s1"] * len(clips), "scenario": ["Extempore"] * len(clips)})
    pq.write_table(pa.table(columns), path)
    return path


class TestQcShard:
    def test_one_row_per_clip_with_the_measured_columns(self, tmp_path):
        shard = write_shard(tmp_path / "train-00000.parquet",
                            [("a.wav", wav_bytes(1.0)), ("b.wav", wav_bytes(0.5))])
        table = qc_shard(shard, IVR)
        assert table.num_rows == 2
        for column in ("duration_s", "peak", "rms_dbfs", "clipped_samples", "leading_silence_s", "is_silent"):
            assert column in table.column_names
        assert table["duration_s"].to_pylist() == pytest.approx([1.0, 0.5])

    def test_carries_the_requested_metadata_and_stamps_provenance(self, tmp_path):
        shard = write_shard(tmp_path / "train-00007.parquet", [("a.wav", wav_bytes(0.3))],
                            carry={"speaker_id": ["spk42"], "scenario": ["Read"]})
        row = qc_shard(shard, IVR).to_pylist()[0]
        assert row["speaker_id"] == "spk42"
        assert row["scenario"] == "Read"
        assert row["source"] == "indicvoices_r"
        assert row["shard"] == "train-00007.parquet"
        assert row["row_index"] == 0
        assert row["clip_path"] == "a.wav"

    def test_records_the_sample_rate_actually_decoded(self, tmp_path):
        shard = write_shard(tmp_path / "s.parquet", [("a.wav", wav_bytes(0.2, sample_rate=24_000))])
        assert qc_shard(shard, IVR)["sample_rate"].to_pylist() == [24_000]

    def test_keeps_both_the_stored_and_the_decoded_duration_so_they_can_be_compared(self, tmp_path):
        # The stored column says 1.0; the audio is really 2.0 s.
        shard = write_shard(tmp_path / "s.parquet", [("a.wav", wav_bytes(2.0))])
        row = qc_shard(shard, IVR).to_pylist()[0]
        assert row["duration_meta_s"] == pytest.approx(1.0)
        assert row["duration_s"] == pytest.approx(2.0)

    def test_a_corrupt_clip_is_recorded_as_an_error_and_the_shard_still_finishes(self, tmp_path):
        shard = write_shard(tmp_path / "s.parquet",
                            [("good.wav", wav_bytes(0.4)), ("bad.wav", b"not audio at all"),
                             ("also_good.wav", wav_bytes(0.4))])
        rows = qc_shard(shard, IVR).to_pylist()
        assert len(rows) == 3
        assert rows[0]["error"] == ""
        assert rows[1]["error"] != ""
        assert rows[1]["duration_s"] is None
        assert rows[2]["error"] == ""

    def test_a_missing_carry_column_is_reported_by_name(self, tmp_path):
        shard = write_shard(tmp_path / "s.parquet", [("a.wav", wav_bytes(0.2))])
        with pytest.raises(KeyError, match="utterance_pitch_mean"):
            qc_shard(shard, ShardSpec("ivr", "audio", carry=("utterance_pitch_mean",)))


class TestScanShards:
    def test_writes_one_output_per_shard(self, tmp_path):
        shards = [write_shard(tmp_path / f"train-0000{i}.parquet", [("a.wav", wav_bytes(0.2))])
                  for i in range(3)]
        out = tmp_path / "out"
        written = scan_shards(shards, IVR, out)
        assert len(written) == 3
        assert sorted(p.name for p in out.glob("*.parquet")) == [
            "train-00000.parquet", "train-00001.parquet", "train-00002.parquet"]

    def test_skips_a_shard_already_done_so_a_crashed_run_resumes(self, tmp_path):
        shard = write_shard(tmp_path / "train-00000.parquet", [("a.wav", wav_bytes(0.2))])
        out = tmp_path / "out"
        scan_shards([shard], IVR, out)
        stamp = (out / "train-00000.parquet").stat().st_mtime_ns

        result = scan_shards([shard], IVR, out)

        assert result == []
        assert (out / "train-00000.parquet").stat().st_mtime_ns == stamp

    def test_force_redoes_a_shard_that_was_already_done(self, tmp_path):
        shard = write_shard(tmp_path / "train-00000.parquet", [("a.wav", wav_bytes(0.2))])
        out = tmp_path / "out"
        scan_shards([shard], IVR, out)
        assert len(scan_shards([shard], IVR, out, force=True)) == 1

    def test_an_unreadable_shard_does_not_stop_the_remaining_shards(self, tmp_path):
        good = write_shard(tmp_path / "train-00000.parquet", [("a.wav", wav_bytes(0.2))])
        broken = tmp_path / "train-00001.parquet"
        broken.write_bytes(b"this is not parquet")
        later = write_shard(tmp_path / "train-00002.parquet", [("a.wav", wav_bytes(0.2))])

        written = scan_shards([good, broken, later], IVR, tmp_path / "out")

        assert [p.name for p in written] == ["train-00000.parquet", "train-00002.parquet"]
        assert not (tmp_path / "out" / "train-00001.parquet").exists()
