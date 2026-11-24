#!/bin/bash
# Multi-seed pilot (v14): 3 new seeds each of distant (hi/zh/ar), related (hi/gu/pa) and Hindi-only, 30 phases (10 cycles), same recipe as before.
# Then self size, persistence and the count-matched lesion tests at every checkpoint from phase 4 until the model loses plasticity (lesion without training,
# lesion + the next language, lesion + the held-out language Indonesian), on every model plus the old seed-0 runs, and one summary.
#   1. checks the machine and downloads what is missing     2. tiny test of everything (stops here if it fails)
#   3. trains (as many runs at once as fit)                  4. verifies every checkpoint on HF
#   5. lesion tests     6. summary     7. commits and pushes (after steps 4, 5 and 6)
# Start:   cd /workspace/shreshth && mkdir -p logs && nohup bash run_v14.sh > logs/run_v14.log 2>&1 &
# Watch:   tail -f logs/run_v14.log            one training run:  tail -f logs/v14_train_v14_distant_s1.log
# Stopped or crashed? Start it again with the same command: finished phases, runs and tests are skipped.
# Only part of it:   bash run_v14.sh --stage analyze,summary,push       Other seeds:   bash run_v14.sh --seeds 4,5
# Extend the same runs to 20 cycles later (resumes after phase 30, skips finished tests):   bash run_v14.sh --n_phases 60
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs; export HF_HUB_DISABLE_XET=1
python -u exp_v14.py "$@"
