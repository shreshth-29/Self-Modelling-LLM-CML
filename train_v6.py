"""train_v6.py: launch one v6 training run from the command line (the notebooks start two of these per GPU).

Examples (from the repo root):
  # FIX: continual, branch from phase 60, per-step noise
  python3 -u train_v6.py --gpu 0 --run v6_cont_noise2e-4_from60 --mode continual \
      --init runs/v4b_continual_s0/ckpt/p060_ar.pt --start_phase 60 --n_phases 90 --noise_per_step 2e-4
  # Q2: Hindi only, branch from phase 30, push at every phase boundary
  python3 -u train_v6.py --gpu 1 --run v6_hi_push0.5_from30 --mode only_hi \
      --init runs/v4b_only_hi_s0/ckpt/p030_hi.pt --start_phase 30 --n_phases 90 --boundary_push 0.5

Rerunning the same command resumes from the last saved phase.
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--run", required=True)
ap.add_argument("--mode", default="continual", choices=["continual", "only_hi"])
ap.add_argument("--init", required=True, help="checkpoint to branch from (path inside the repo)")
ap.add_argument("--start_phase", type=int, required=True)
ap.add_argument("--n_phases", type=int, default=90, help="last phase to train")
ap.add_argument("--noise_per_step", type=float, default=0.0)
ap.add_argument("--boundary_push", type=float, default=0.0)
ap.add_argument("--time_cap_h", type=float, default=120.0)
ap.add_argument("--no_hf", action="store_true", help="do not back up to HF")
ap.add_argument("--n_chrf", type=int, default=200)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)
from common_v6 import *

hf_repo = None
if not args.no_hf:
    from huggingface_hub import HfApi
    hf_repo = HfApi().whoami()["name"] + "/cml-lang-self-checkpoints"

C = RunConfig(run_name=args.run, mode=args.mode, order=("hi", "zh", "ar"), init_ckpt=args.init,
              start_phase=args.start_phase, n_phases=args.n_phases, seed=0, time_cap_h=args.time_cap_h,
              hf_repo=hf_repo, noise_per_step=args.noise_per_step, boundary_push=args.boundary_push)
print(C, flush=True)
D = load_data(train_langs=["hi", "zh", "ar"] if args.mode == "continual" else ["hi"], n_chrf=args.n_chrf)
train_run(C, D)
