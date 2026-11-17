"""Create the two release repos on Hugging Face as PRIVATE and upload the staged folders.

    python scripts\\push_release_hf.py model|eval|collection

Never changes visibility: making them public is the operator's click. The write token is
read from .secrets\\geetha.env into this process only.
"""
from __future__ import annotations

import sys
from pathlib import Path

MODEL, EVAL = "Dhevenddra/geetha-tts", "Dhevenddra/geetha-tts-eval"
STAGED = {"model": (MODEL, "model", Path(r"E:\Geetha\results\release_hf")),
          "eval": (EVAL, "dataset", Path(r"E:\Geetha\results\release_eval"))}


def token() -> str:
    for line in Path(r"E:\Geetha\.secrets\geetha.env").read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no HF_TOKEN in the secrets file")


def main() -> int:
    from huggingface_hub import HfApi

    what = sys.argv[1]
    api = HfApi(token=token())
    if what == "collection":
        made = api.create_collection("Geetha", namespace="Dhevenddra", private=True, exists_ok=True,
                                     description="Geetha: open Tamil speech models and their evidence.")
        have = {item.item_id for item in made.items}
        for repo, kind in ((MODEL, "model"), (EVAL, "dataset")):
            if repo not in have:
                api.add_collection_item(made.slug, repo, item_type=kind)
        print(f"collection {made.slug} private={made.private}")
        return 0
    repo, kind, folder = STAGED[what]
    api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(folder),
                      # a replaced header image must not linger on the Hub under its old name
                      delete_patterns=["assets/*"] if what == "model" else None,
                      commit_message="Geetha TTS v1" if what == "model"
                      else "Benchmark v2 outputs and listening tests")
    info = api.repo_info(repo, repo_type=kind)
    print(f"{repo}: private={info.private}, {len(info.siblings)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
