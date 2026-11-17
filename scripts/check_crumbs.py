r"""How far has the running pod got? Prints the number of breadcrumbs it has posted.

    py -3.11 scripts\check_crumbs.py

Four pods died taking their logs with them, so jobs/m1_pilot.sh uploads a tiny file to
the checkpoint repo as it passes each stage. 0 means the container has not started our
job yet (usually still pulling the 8 GB image); 4 means it reached Stage A training.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", r"E:\Geetha\.hf_home")
from huggingface_hub import list_repo_files
from ops.secrets import load_secrets

try:
    files = list_repo_files("Dhevenddra/geetha-ckpt-private", repo_type="model",
                            token=load_secrets().require("HF_TOKEN"))
    crumbs = sorted(f for f in files if f.startswith("breadcrumbs/"))
    print(len(crumbs))
    for c in crumbs:
        print("   ", c, file=sys.stderr)
except Exception:
    print(0)
