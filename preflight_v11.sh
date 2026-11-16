#!/bin/bash
# 10-minute check on THIS machine before the overnight queue: runs every part of core_v11.py for real, but tiny.
# Usage:   cd /workspace/shreshth && bash preflight_v11.sh
# Every line must end in OK. If one says FAILED, its last lines are shown: send them to Claude before starting the queue.
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs; export HF_HUB_DISABLE_XET=1
FAIL=0
step() {   # $1 = label, rest = command
  local label="$1"; shift
  local log="logs/preflight_$(echo "$label" | tr ' /' '__').log"
  if "$@" > "$log" 2>&1; then echo "OK      $label"; else echo "FAILED  $label   (last lines of $log:)"; tail -n 15 "$log"; FAIL=1; fi
}
echo "--- machine"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)  (the queue needs 2)"
df -h /workspace | tail -1 | awk '{print "disk free on /workspace: " $4}'
step "python packages" python -c "import torch, sentencepiece, sacrebleu, scipy, pandas, matplotlib, datasets, huggingface_hub, nbconvert; print(torch.__version__, pandas.__version__)"
step "jupyter nbconvert" jupyter nbconvert --version
step "hugging face login" python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
step "git push works" git push --dry-run
echo "--- files"
step "tokenizers + data files" python - <<'PY'
import os, sys
need = ["tokenizer_v4/spm.model", "tokenizer_v8/spm.model"]
need += [f"data/flores/{s}.{l}.jsonl" for s in ["dev", "devtest"] for l in ["en", "hi", "zh", "ar", "gu", "pa"]]
need += [f"data/en-{l}/{s}.jsonl" for l in ["hi", "zh", "ar", "gu", "pa"] for s in ["train", "val"]]
need += [f"analysis/v10/interp/neurons_{m}.csv" for m in ["hindi45", "cont40", "cont10", "random"]]
miss = [f for f in need if not os.path.exists(f)]
print("missing:", miss); sys.exit(1 if miss else 0)
PY
step "download 3 checkpoints" python - <<'PY'
import glob, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); repo = api.whoami()["name"] + "/cml-lang-self-checkpoints"; files = api.list_repo_files(repo)
bad = 0
for run, ph in [("v4b_continual_s0", 40), ("v4b_continual_s0", 41), ("v8_indic_continual_s0", 13)]:
    if not glob.glob(f"runs/{run}/ckpt/p{ph:03d}_*.pt"):
        name = [f for f in files if f.startswith(f"{run}/ckpt/p{ph:03d}_")]
        if name: hf_hub_download(repo, name[0], local_dir="runs")
    ok = bool(glob.glob(f"runs/{run}/ckpt/p{ph:03d}_*.pt")); print(run, ph, "ok" if ok else "MISSING"); bad += (not ok)
for run in ["v4b_continual_s0", "v4b_only_hi_s0", "v6b_mixed_s0", "v8_indic_continual_s0"]:
    phs = sorted(int(f.split("/")[-1][1:4]) for f in files if f.startswith(run + "/ckpt/p") and f.endswith(".pt"))
    print(run, "phases on HF:", len(phs), "from", phs[:1], "to", phs[-1:])
sys.exit(1 if bad else 0)
PY
step "Indonesian data (prep_newtasks.py)" python -u prep_newtasks.py
echo "--- core_v11.py, every part, tiny (results go to analysis/v11/_pre_* and are deleted at the end)"
rm -rf analysis/v11/_pre_*
D="v4b_continual_s0"; R="v8_indic_continual_s0"
export CML_LANGSET=distant
step "sizes (distant 40)"            python -u core_v11.py --gpu 0 --part sizes  --out _pre_sizes  --family distant --model distant=$D:40
step "matrix (distant 40 -> 41)"     python -u core_v11.py --gpu 0 --part matrix --out _pre_matrix --family distant --model distant=$D:40
step "static (distant 40, tiny)"     python -u core_v11.py --gpu 0 --part static --out _pre_static --family distant --model distant=$D:40 --draws 0 --n_val 30 --n_chrf 8
step "train next: lesion"            python -u core_v11.py --gpu 0 --part train --task next --mode lesion  --out _pre_train --family distant --model distant40=$D:40 --seeds 0 --steps 20
step "train next: freeze"            python -u core_v11.py --gpu 0 --part train --task next --mode freeze  --out _pre_train --family distant --model distant40=$D:40 --seeds 0 --steps 20 --conds none,self@1.0,task@1.0
step "train next: silence"           python -u core_v11.py --gpu 0 --part train --task next --mode silence --out _pre_train --family distant --model distant40=$D:40 --seeds 0 --steps 20 --conds none,self@1.0,task@1.0
step "train indonesian: lesion"      python -u core_v11.py --gpu 0 --part train --task indonesian --mode lesion --out _pre_indo --family distant --model distant40=$D:40 --seeds 0 --steps 20 --lr 1e-4 --conds none,self@1.0,task@0.5,look@0.5
step "train indonesian: fresh model" python -u core_v11.py --gpu 0 --part train --task indonesian --out _pre_indo --family fresh --model fresh --seeds 0 --steps 20 --lr 1e-4
export CML_LANGSET=indic
step "train next: related 13 (GPU 1)" python -u core_v11.py --gpu 1 --part train --task next --mode lesion --out _pre_train --family related --model related13=$R:13 --seeds 0 --steps 20 --conds none,self@1.0,task@1.0
step "summary sizes"  python core_v11.py --summarize --part sizes  --out _pre_sizes
step "summary static" python core_v11.py --summarize --part static --out _pre_static
step "summary train"  python core_v11.py --summarize --part train  --out _pre_train
step "summary indonesian" python core_v11.py --summarize --part train --out _pre_indo
rm -rf analysis/v11/_pre_*
echo
if [ $FAIL -eq 0 ]; then echo "ALL OK -> start the queue:   mkdir -p logs && nohup bash queue_v11.sh > logs/queue_v11.log 2>&1 &"; else echo "SOMETHING FAILED: send the lines above to Claude before starting the queue."; fi
