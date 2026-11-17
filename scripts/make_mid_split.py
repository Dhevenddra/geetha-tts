"""Write kaggle\\m3_mid_split\\m3_mid_split.py with the release files embedded.

Usage:  python scripts\\make_mid_split.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
KERNEL = ROOT / "kaggle" / "m3_mid_split"
FILES = ("inference.py", "tamil_text.py", "breaths.json")
SLUG = "geetha-m3-mid-split"


def main() -> int:
    payload = {}
    for name in FILES:
        data = (RELEASE / name).read_bytes()
        payload[name] = base64.b64encode(data).decode("ascii")
        print(f"{name}: {len(data):,} bytes  sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    script = (KERNEL / "template.py").read_text(encoding="utf-8").replace(
        "__PAYLOAD__", json.dumps(payload, indent=1))
    compile(script, "m3_mid_split.py", "exec")
    (KERNEL / "m3_mid_split.py").write_text(script, encoding="utf-8", newline="\n")
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": f"dhevenddra/{SLUG}", "title": SLUG,
        "code_file": "m3_mid_split.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["dhevenddra/geetha-secrets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2),
        encoding="utf-8")
    print(f"wrote {KERNEL / 'm3_mid_split.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
