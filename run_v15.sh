#!/bin/bash
# Adi's trials on the finished v14 checkpoints (no new training runs). Lesion or freeze, strong (full count, about half a layer)
# and weak (half of that), self against the same number of random task neurons, then train:
#   trial A / alpha  on a language outside every set (Indonesian, 1,500 steps)
#   trial B / beta   on another language of the run's own set (the next one, 1,000 steps)
#   trial C / gamma  on the language the checkpoint was just trained on (1,000 steps)
# A, B, C = lesion (re-initialised neurons). alpha, beta, gamma = freeze (incoming weights + bias cannot change while training).
# What v14 already measured (lesion A and B: strong self and task, weak self) is not run again; the summary joins both.
#   1. checks the machine, downloads the start checkpoints (phases 4, 6, ... 30 of every run) from HF
#   2. tiny test of every trial (stops if it fails)   3. all trials   4. summary   5. commit + push (after 3 and after 4)
# Start:   cd /workspace/shreshth && mkdir -p logs && nohup bash run_v15.sh > logs/run_v15.log 2>&1 &
# Watch:   tail -f logs/run_v15.log        Stopped? Start it again with the same command: finished jobs are skipped.
cd /workspace/shreshth || exit 1
source /venv/main/bin/activate
mkdir -p logs; export HF_HUB_DISABLE_XET=1
python -u exp_v14.py --stage prep,trials,trials_summary,push "$@"
