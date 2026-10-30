"""push_data_v6.py: back up the training data to a PRIVATE HF dataset repo, once, from any machine that has it.
After this, the v6 notebooks download it automatically on any new instance.

Run from the repo root on the machine that has data/en-{hi,zh,ar}/{train,val}.jsonl:
  python3 push_data_v6.py
"""
import os
from huggingface_hub import HfApi
api = HfApi()
repo = api.whoami()["name"] + "/cml-lang-self-data"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for l in ["hi", "zh", "ar"]:
    for s in ["train", "val"]:
        p = f"data/en-{l}/{s}.jsonl"
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"en-{l}/{s}.jsonl", repo_id=repo, repo_type="dataset")
            print("uploaded", p, f"{os.path.getsize(p) / 1e6:.0f} MB")
        else:
            print("not here:", p)
print("done ->", repo)
