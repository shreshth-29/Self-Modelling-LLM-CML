#!/bin/bash
# Status watcher for the v14 run. Every 15 minutes in the first hour, then every 30 minutes, it writes analysis/v14_status.txt
# (main log, phases finished, checkpoints on this machine vs on Hugging Face, speed, problems, GPUs, disk) and pushes that one file
# to GitHub, so the run can be checked from outside the machine.
# It stops by itself when training is over (from then on run_v14.sh does the pushing) or when the run stopped with an error.
# Start:   cd /workspace/shreshth && nohup bash watch_v14.sh > logs/watch_v14.log 2>&1 &
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
export HF_HUB_DISABLE_XET=1
OUT=analysis/v14_status.txt; mkdir -p analysis
n=0
while true; do
  {
    echo "checked: $(date -u '+%a %Y-%m-%d %H:%M UTC')"
    echo; echo "--- main log, last 15 lines"; tail -n 15 logs/run_v14.log 2>/dev/null | cut -c1-220
    echo; echo "--- last phase finished, per run"
    for f in runs/v14_*/phases.jsonl; do
      [ -f "$f" ] && echo "$(basename "$(dirname "$f")"): $(tail -n 1 "$f" | python -c 'import sys, json; d = json.loads(sys.stdin.read()); print("phase", d["phase"], "(" + str(d["lang"]) + ") | hours so far", round(d.get("total_min", 0) / 60, 1))' 2>/dev/null)"
    done
    echo; echo "--- checkpoints: on this machine | on Hugging Face with the right size"
    python -W ignore -c "
import v14_hf as H, v14_config as C
for m in C.models(include_old=False):
    r, l = H.remote_ckpts(m['run']), H.local_ckpts(m['run'])
    ok = sum(r.get(n) == s for n, s in l.items())
    print(m['run'], '|', len(l), '|', ok, '' if ok >= len(l) - 1 else '  <-- uploads are behind')
" 2>&1 | grep -v Warning
    echo; echo "--- speed: the newest progress line of each run"
    for f in logs/v14_train_*.log; do [ -f "$f" ] && grep "tok/s" "$f" | tail -n 1 | cut -c1-175; done
    echo; echo "--- problems (empty = none)"
    grep -h "HF: .*failed\|MISSING\|PROBLEM\|TEST FAILED\|STOP:\|Traceback\|out of memory\|DIVERGED\|NOT FINISHED" logs/run_v14.log logs/v14_train_*.log 2>/dev/null | tail -n 12 | cut -c1-220
    echo; echo "--- GPUs (index, % busy, memory used)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo; echo "--- disk"; df -h /workspace | tail -n 1
  } > "$OUT" 2>&1
  if grep -q "STAGE verify" logs/run_v14.log 2>/dev/null; then
    echo "$(date '+%a %H:%M') training is over: run_v14.sh pushes from here on; watcher stops"; exit 0
  fi
  git add "$OUT" && git commit -q -m "v14 status" "$OUT"
  git pull --rebase -q || git rebase --abort
  git push -q && echo "$(date '+%a %H:%M') status pushed" || echo "$(date '+%a %H:%M') status push failed"
  if grep -q "TEST FAILED\|STOP:\|FINISHED WITH\|ALL DONE" logs/run_v14.log 2>/dev/null; then
    echo "$(date '+%a %H:%M') the run has stopped: watcher stops"; exit 0
  fi
  n=$((n + 1)); if [ $n -le 4 ]; then sleep 900; else sleep 1800; fi
done
