r"""QC gates and loudness gain for the training manifests -- ETL step 5 (D-036).

Every threshold here came out of step 1's measurements over all 72,297 TTS clips
(``work\audio_qc_report.md``), not out of a textbook:

* **clipping** -- 405 Rasa clips hold at least one full-scale sample and IV-R holds
  none. One sample at full scale is a rounding artefact; the 64 clips with more than
  0.1% of their samples there are audible distortion a vocoder would learn. Hence the
  gate at 0.1% rather than at "any".
* **edge silence** -- Rasa's median clip is 9% silence at the ends and IV-R's is 0.7%.
  Ordinary padding is normal and kept; the 533 clips that are more than *half* silence
  are not clips of speech. They are excluded rather than trimmed, because trimming
  changes a clip's relationship to its own transcript.
* **duration** -- below half a second there is not enough signal to learn from; above
  30 s a clip costs ~2,460 audio tokens and the long tail is 42 s of Rasa.
* **loudness** -- the two sources sit 10 dB apart, so level would otherwise encode
  *source*, which is confounded with the style conditioning. Gain is **attenuation
  only**: IV-R's RMS p95 is -13.0 dBFS with zero clipped samples, so there is no
  headroom to boost into.

A missing measurement is never treated as a passing one. Step 1 writes nulls when a
clip could not be decoded, and ``0.0`` would sail through every numeric gate here.
"""

from __future__ import annotations

from typing import Mapping

__all__ = [
    "MAX_CLIPPED_FRACTION",
    "MAX_DURATION_S",
    "EXPECTED_SAMPLE_RATE",
    "MAX_EDGE_SILENCE_SHARE",
    "MIN_DURATION_S",
    "TARGET_RMS_DBFS",
    "gain_db",
    "qc_verdict",
]

# Both corpora are overwhelmingly 48 kHz, but Rasa holds 408 clips at 22,050 Hz
# (1.23 h). They are excluded rather than resampled: 22,050 -> 24,000 needs a 160/147
# rational resampler, and the result would carry no energy above ~11 kHz while every
# other clip reaches 12 kHz -- a band-limit tied to nothing the model can condition on,
# which is the same kind of hidden inconsistency as the 10 dB loudness gap (D-036). At
# 1% of Stage A's hours that is not worth either the code or the confound.
EXPECTED_SAMPLE_RATE = 48_000

MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0
MAX_CLIPPED_FRACTION = 0.001
MAX_EDGE_SILENCE_SHARE = 0.5
TARGET_RMS_DBFS = -23.0

_REQUIRED = ("duration_s", "rms_dbfs", "clipped_fraction",
             "leading_silence_s", "trailing_silence_s")


def qc_verdict(row: Mapping[str, object]) -> str:
    """``""`` if the clip may train, else the reason it may not."""
    if row.get("error"):
        return "decode_error"
    if any(row.get(name) is None for name in _REQUIRED):
        return "decode_error"
    if row.get("is_silent"):
        return "silent"
    if row.get("sample_rate") not in (None, EXPECTED_SAMPLE_RATE):
        return "sample_rate"

    duration = float(row["duration_s"])  # type: ignore[arg-type]
    if duration < MIN_DURATION_S:
        return "too_short"
    if duration > MAX_DURATION_S:
        return "too_long"
    if float(row["clipped_fraction"]) > MAX_CLIPPED_FRACTION:  # type: ignore[arg-type]
        return "clipping"

    edge = float(row["leading_silence_s"]) + float(row["trailing_silence_s"])  # type: ignore[arg-type]
    if duration > 0 and edge / duration > MAX_EDGE_SILENCE_SHARE:
        return "edge_silence"
    return ""


def gain_db(rms_dbfs: float | None, target: float = TARGET_RMS_DBFS) -> float:
    """Attenuation in dB to bring ``rms_dbfs`` to ``target``. Never positive.

    Raises on a missing level rather than returning 0.0, which would read as "this clip
    needs no gain" when it actually means "this clip was never measured".
    """
    if rms_dbfs is None:
        raise ValueError("gain_db needs rms_dbfs; a clip with no measured level cannot be normalised")
    return min(0.0, target - float(rms_dbfs))
