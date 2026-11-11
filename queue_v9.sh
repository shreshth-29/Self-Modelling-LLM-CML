#!/bin/bash
# Starts relearn_v9.ipynb on GPU 0 and patch_v9.ipynb on GPU 1, each as soon as that GPU's current job is finished.
# "finished" = no newtask_v9.py process on that GPU for 3 checks in a row, 1 minute apart.
# Start it once:   cd /workspace/shreshth && nohup bash queue_v9.sh > logs/queue.log 2>&1 &
# Watch:           tail -f logs/queue.log      (and logs/relearn_v9.log, logs/patch_v9.log once they start)
cd /workspace/shreshth && source /venv/main/bin/activate && mkdir -p logs
wait_gpu() {
  n=0
  while [ $n -lt 3 ]; do
    if pgrep -f "newtask_v9.py --gpu $1" > /dev/null; then n=0; else n=$((n+1)); fi
    sleep 60
  done
}
run_nb() {   # $1 = GPU, $2 = notebook
  wait_gpu $1
  echo "$(date '+%H:%M') GPU $1 free -> starting $2"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$2" \
    && echo "$(date '+%H:%M') $2 FINISHED (open it to see the outputs)" || echo "$(date '+%H:%M') $2 STOPPED WITH AN ERROR"
}
echo "$(date '+%H:%M') queue started: waiting for GPU 0 and GPU 1"
run_nb 0 relearn_v9.ipynb &
run_nb 1 patch_v9.ipynb &
wait
echo "$(date '+%H:%M') queue done"
