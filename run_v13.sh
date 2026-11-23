#!/bin/bash
# Definition checks on existing checkpoints (no training). One GPU is enough. About 1 hour.
#   1. downloads the checkpoints it needs     2. a 2-minute tiny test of the new code (stops here if it fails)
#   3. the full run                             4. the summary report     5. commits and pushes the results
# Start:   cd /workspace/shreshth && mkdir -p logs && nohup bash run_v13.sh > logs/run_v13.log 2>&1 &
# Watch:   tail -f logs/run_v13.log
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs; export HF_HUB_DISABLE_XET=1
D="v4b_continual_s0"; H="v4b_only_hi_s0"; M="v6b_mixed_s0"; R="v8_indic_continual_s0"; R1="v8_indic_continual_s1"
say() { echo "$(date '+%a %H:%M') $*"; }

say "downloading checkpoints"
python - <<'PY'
import glob
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); repo = api.whoami()["name"] + "/cml-lang-self-checkpoints"; files = api.list_repo_files(repo)
want = {"v4b_continual_s0": [0, 1, 10, 40, 41, 42], "v4b_only_hi_s0": [0, 40, 45], "v6b_mixed_s0": [40, 45],
        "v8_indic_continual_s0": [0, 13, 14, 15], "v8_indic_continual_s1": [13]}
for run, phs in want.items():
    for ph in phs:
        if not glob.glob(f"runs/{run}/ckpt/p{ph:03d}_*.pt"):
            name = [f for f in files if f.startswith(f"{run}/ckpt/p{ph:03d}_")]
            if name:
                try: hf_hub_download(repo, name[0], local_dir="runs")
                except Exception as e: print("download failed", run, ph, e)
        print(run, ph, "ok" if glob.glob(f"runs/{run}/ckpt/p{ph:03d}_*.pt") else "NOT AVAILABLE (will be skipped)")
PY

say "tiny test of the new code"
rm -rf analysis/v13/_pre
if CML_LANGSET=distant python -u defn_v13.py --gpu 0 --out _pre --family distant --model distant=$D:40 --taus 0.20,0.50 > logs/v13_pre.log 2>&1 \
   && CML_LANGSET=distant python -u defn_v13.py --gpu 0 --out _pre --family untrained --fresh_seeds 0 --taus 0.20,0.50 >> logs/v13_pre.log 2>&1 \
   && CML_LANGSET=indic python -u defn_v13.py --gpu 0 --out _pre --family related --model related=$R:13 --taus 0.20,0.50 >> logs/v13_pre.log 2>&1 \
   && python defn_v13.py --summarize --out _pre --taus 0.20,0.50 >> logs/v13_pre.log 2>&1; then
  say "tiny test OK"; rm -rf analysis/v13/_pre
else
  say "TINY TEST FAILED. Nothing else was run. Send these lines to Claude:"; tail -n 25 logs/v13_pre.log; exit 1
fi

say "full run (progress: tail -f logs/v13.log)"
run() { local ls="$1"; shift; CML_LANGSET=$ls python -u defn_v13.py --gpu 0 "$@" >> logs/v13.log 2>&1 || say "ERROR in: $* (see logs/v13.log); continuing"; }
run distant --family distant --model distant=$D:0,40,41,42,1,10
run distant --family untrained --fresh_seeds 0,1
run indic   --family related --model related=$R:0,13,14,15
run indic   --family untrained --fresh_seeds 0,1
run distant --family hindi --self_langs hi --model hindi=$H:0,40,45
run distant --family mixed --model mixed=$M:40,45
run indic   --family related_s1 --model related_s1=$R1:13
say "summary"
python defn_v13.py --summarize > logs/v13_summary.log 2>&1 || say "ERROR in the summary (see logs/v13_summary.log)"
git add -A; git commit -q -m "v13: definition checks (threshold sweep, untrained nulls, shared signals, split-half, gap by threshold)" || true
if git push -q; then say "pushed"; else say "PUSH FAILED (results are on this machine; run: git push)"; fi
say "ALL DONE. No checkpoints were created, so the instance can be destroyed once the line above says pushed."
