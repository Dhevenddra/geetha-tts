"""Write kaggle\\m3_release_smoke\\m3_release_smoke.py with the release files embedded.

The smoke test must run the bytes we ship, not a copy that drifted. The three release
files are embedded base64 in the kernel; the breath detector constants are lifted from
the kernel that last used them.

Usage:  python scripts\\make_release_smoke.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
KERNEL = ROOT / "kaggle" / "m3_release_smoke"
FILES = ("inference.py", "tamil_text.py", "breaths.json")
DETECTOR_SOURCE = ROOT / "kaggle" / "m3_base_vs_geetha" / "m3_base_vs_geetha.py"


def main() -> int:
    source = DETECTOR_SOURCE.read_text(encoding="utf-8")
    block = re.search(r"^DETECTOR = \{.*?^\}\n^THRESHOLD = [^\n]*\n", source, re.S | re.M)
    if not block:
        raise SystemExit("could not find the DETECTOR block")
    payload = {}
    for name in FILES:
        data = (RELEASE / name).read_bytes()
        payload[name] = base64.b64encode(data).decode("ascii")
        print(f"{name}: {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    template = (KERNEL / "template.py").read_text(encoding="utf-8")
    script = (template.replace("__DETECTOR__", block.group(0).rstrip())
              .replace("__PAYLOAD__", json.dumps(payload, indent=1)))
    compile(script, "m3_release_smoke.py", "exec")
    (KERNEL / "m3_release_smoke.py").write_text(script, encoding="utf-8", newline="\n")
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": "dhevenddra/geetha-m3-release-smoke", "title": "geetha-m3-release-smoke",
        "code_file": "m3_release_smoke.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["dhevenddra/geetha-code", "dhevenddra/geetha-secrets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2),
        encoding="utf-8")
    print(f"wrote {KERNEL / 'm3_release_smoke.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
