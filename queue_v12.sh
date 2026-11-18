#!/bin/bash
# Tonight's queue (v12). Two notebooks; after each one, everything new is committed and pushed.
#   M  v12_M_mind_gpu0.ipynb      dial + broadcast + overlap (no training)                    about 1.5 hours
#   R  v12_R_seed1_hindifix.ipynb Hindi-only fix, then the lesion tests on related seed 1     about 2.3 hours
# With 2 GPUs the two run side by side (about 2.3 hours in total). With 1 GPU they run one after the other (about 3.8 hours).
#
# Start once:   cd /workspace/shreshth && mkdir -p logs && nohup bash queue_v12.sh > logs/queue_v12.log 2>&1 &
# Watch:        tail -f logs/queue_v12.log
# Do NOT open and save these notebooks in Jupyter while the queue runs.
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs
export HF_HUB_DISABLE_XET=1

push() {   # $1 = what just finished. One push at a time (lock), never pulls.
  (
    flock 9
    git add -A
    git commit -q -m "v12 run: after $1" || true
    if git push -q; then echo "$(date '+%a %H:%M') pushed (after $1)"; else echo "$(date '+%a %H:%M') PUSH FAILED after $1 (results are still on this machine; run: git push)"; fi
  ) 9>/tmp/cml_push.lock
}

run_nb() {   # $1 = notebook, $2 = GPU number
  echo "$(date '+%a %H:%M') START    $1 on GPU $2"
  if CML_GPU="$2" jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$1" > "logs/nbconvert_$1.log" 2>&1; then
    echo "$(date '+%a %H:%M') FINISHED $1"
  else
    echo "$(date '+%a %H:%M') ERROR in $1 (see logs/nbconvert_$1.log)"
  fi
  push "$1"
}

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "$(date '+%a %H:%M') queue started, GPUs: $NGPU"
if [ "$NGPU" -ge 2 ]; then
  run_nb v12_M_mind_gpu0.ipynb 0 &
  run_nb v12_R_seed1_hindifix.ipynb 1 &
  wait
else
  run_nb v12_M_mind_gpu0.ipynb 0
  run_nb v12_R_seed1_hindifix.ipynb 0
fi
echo "$(date '+%a %H:%M') ALL DONE. Everything is on GitHub if the lines above say 'pushed'. No checkpoints were created, so the instance can be destroyed."
