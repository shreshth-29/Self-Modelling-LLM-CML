#!/bin/bash
# Overnight queue. Two chains run side by side, one per GPU; in each chain the notebooks run one after another.
# After EVERY notebook, everything new is committed and pushed to GitHub (results so far are never left only on the machine).
#
#   GPU 0: interp3a  ->  A static lesions  ->  B lesion + train next phase  ->  C freeze + silence
#   GPU 1: interp3b  ->  S sizes + matrix  ->  D Indonesian (5 seeds)       ->  E second seed of the related run (more than a day)
#
# Start once:   cd /workspace/shreshth && mkdir -p logs && nohup bash queue_v11.sh > logs/queue_v11.log 2>&1 &
# Watch:        tail -f logs/queue_v11.log        (each notebook also has its own log: logs/v11_*.log)
# Do NOT open and save these notebooks in Jupyter while the queue runs: saving would overwrite the outputs the queue writes into them.
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs
export HF_HUB_DISABLE_XET=1

push() {   # $1 = what just finished. One push at a time (lock), never pulls (nothing else pushes during the night).
  (
    flock 9
    git add -A
    git commit -q -m "v11 overnight: after $1" || true
    if git push -q; then echo "$(date '+%a %H:%M') pushed (after $1)"; else echo "$(date '+%a %H:%M') PUSH FAILED after $1 (results are still on this machine; run: git push)"; fi
  ) 9>/tmp/cml_push.lock
}

run_nb() {   # $1 = notebook
  echo "$(date '+%a %H:%M') START    $1"
  if jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$1" > "logs/nbconvert_$1.log" 2>&1; then
    echo "$(date '+%a %H:%M') FINISHED $1"
  else
    echo "$(date '+%a %H:%M') ERROR in $1 (see logs/nbconvert_$1.log); the queue continues with the next notebook"
  fi
  push "$1"
}

chain0() { for nb in interp3a_v10.ipynb v11_A_static_gpu0.ipynb v11_B_lesion_next_gpu0.ipynb v11_C_freeze_silence_next_gpu0.ipynb; do run_nb "$nb"; done; echo "$(date '+%a %H:%M') GPU 0 chain done"; }
chain1() { for nb in interp3b_v10.ipynb v11_S_sizes_matrix_gpu1.ipynb v11_D_indonesian_gpu1.ipynb v11_E_train_related_seed1_gpu1.ipynb; do run_nb "$nb"; done; echo "$(date '+%a %H:%M') GPU 1 chain done"; }

echo "$(date '+%a %H:%M') queue started"
chain0 &
chain1 &
wait
echo "$(date '+%a %H:%M') queue done"
