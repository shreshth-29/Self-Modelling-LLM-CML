#!/bin/bash
# 10-minute check on THIS machine before tonight's queue: runs every NEW piece of code for real, but tiny.
# Usage:   cd /workspace/shreshth && bash preflight_v12.sh
# Every line must end in OK. If one says FAILED, its last lines are shown: send them to Claude before starting the queue.
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs; export HF_HUB_DISABLE_XET=1
FAIL=0
step() {   # $1 = label, rest = command
  local label="$1"; shift
  local log="logs/preflight12_$(echo "$label" | tr ' /:(),>' '_______').log"
  if "$@" > "$log" 2>&1; then echo "OK      $label"; else echo "FAILED  $label   (last lines of $log:)"; tail -n 15 "$log"; FAIL=1; fi
}
echo "--- machine"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)  (1 is enough; with 2 the queue is faster)"
df -h /workspace | tail -1 | awk '{print "disk free on /workspace: " $4 "  (about 15 GB needed)"}'
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
miss = [f for f in need if not os.path.exists(f)]
print("missing:", miss); sys.exit(1 if miss else 0)
PY
step "download 4 checkpoints + list what is on HF" python - <<'PY'
import glob, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); repo = api.whoami()["name"] + "/cml-lang-self-checkpoints"; files = api.list_repo_files(repo)
bad = 0
for run, ph in [("v4b_continual_s0", 40), ("v4b_only_hi_s0", 45), ("v8_indic_continual_s0", 13), ("v8_indic_continual_s1", 13)]:
    if not glob.glob(f"runs/{run}/ckpt/p{ph:03d}_*.pt"):
        name = [f for f in files if f.startswith(f"{run}/ckpt/p{ph:03d}_")]
        if name: hf_hub_download(repo, name[0], local_dir="runs")
    ok = bool(glob.glob(f"runs/{run}/ckpt/p{ph:03d}_*.pt")); print(run, ph, "ok" if ok else "MISSING"); bad += (not ok)
want = {"v4b_continual_s0": [40, 41, 42], "v8_indic_continual_s0": [13, 14, 15], "v4b_only_hi_s0": [10, 30, 45, 60], "v6b_mixed_s0": [45],
        "v8_indic_continual_s1": list(range(5, 17))}
for run, phs in want.items():
    there = {int(f.split("/")[-1][1:4]) for f in files if f.startswith(run + "/ckpt/p") and f.endswith(".pt")}
    miss = [p for p in phs if p not in there]
    print(run, "all needed phases on HF" if not miss else f"NOT on HF: {miss} (these will be skipped tonight)")
sys.exit(1 if bad else 0)
PY
echo "--- new code, every part, tiny (results go to analysis/*/_pre_* and are deleted at the end)"
rm -rf analysis/v11/_pre_* analysis/v12/_pre_*
D="v4b_continual_s0"; H="v4b_only_hi_s0"; R="v8_indic_continual_s0"; R1="v8_indic_continual_s1"
TINY="--draws 0 --n_val 40 --n_chrf 8 --gains 0,2 --alphas -2,2"
export CML_LANGSET=distant
step "dial (distant 40)"                 python -u mind_v12.py --gpu 0 --part dial --out _pre_dial --family distant --model distant=$D:40 $TINY
step "dial (Hindi-only 45, self on hi)"  python -u mind_v12.py --gpu 0 --part dial --out _pre_dial --family hindi --model hindi=$H:45 --self_langs hi $TINY
step "broadcast (distant 40)"            python -u mind_v12.py --gpu 0 --part broadcast --out _pre_broadcast --family distant --model distant=$D:40 --n_dirs 3 --n_sent 8
step "overlap (distant 40)"              python -u mind_v12.py --gpu 0 --part overlap --out _pre_overlap --family distant --model distant=$D:40
step "Hindi-only fix (static, tiny)"     python -u core_v11.py --gpu 0 --part static --out _pre_hfix --family hindi --self_langs hi --eval_langs hi --model hindi=$H:45 --draws 0 --n_val 30 --n_chrf 8
export CML_LANGSET=indic
step "dial (related 13)"                 python -u mind_v12.py --gpu 0 --part dial --out _pre_dial --family related --model related=$R:13 $TINY
step "broadcast (related 13)"            python -u mind_v12.py --gpu 0 --part broadcast --out _pre_broadcast --family related --model related=$R:13 --n_dirs 3 --n_sent 8
step "overlap (related 13)"              python -u mind_v12.py --gpu 0 --part overlap --out _pre_overlap --family related --model related=$R:13
step "seed 1: sizes"                     python -u core_v11.py --gpu 0 --part sizes --out _pre_sizes --family related_s1 --model related_s1=$R1:13
step "seed 1: lesion + train 20 steps"   python -u core_v11.py --gpu 0 --part train --task next --mode lesion --out _pre_train --family related_s1 --model related_s1_13=$R1:13 --seeds 0 --steps 20 --conds none,self@1.0,task@1.0
step "summary dial"       python mind_v12.py --summarize --part dial --out _pre_dial
step "summary broadcast"  python mind_v12.py --summarize --part broadcast --out _pre_broadcast
step "summary overlap"    python mind_v12.py --summarize --part overlap --out _pre_overlap
step "summary Hindi fix"  python core_v11.py --summarize --part static --out _pre_hfix
step "summary seed 1 sizes" python core_v11.py --summarize --part sizes --out _pre_sizes
step "summary seed 1 train" python core_v11.py --summarize --part train --out _pre_train
echo "--- a look at the tiny dial result (numbers are noisy at this size; this only shows the table has the right shape)"
head -c 1800 analysis/v12/_pre_dial/report.txt 2>/dev/null
rm -rf analysis/v11/_pre_* analysis/v12/_pre_*
echo
if [ $FAIL -eq 0 ]; then echo "ALL OK -> start the queue:   mkdir -p logs && nohup bash queue_v12.sh > logs/queue_v12.log 2>&1 &"; else echo "SOMETHING FAILED: send the lines above to Claude before starting the queue."; fi
