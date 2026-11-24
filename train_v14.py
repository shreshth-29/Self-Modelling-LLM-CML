"""train_v14.py: ONE training run of the multi-seed pilot (exp_v14.py starts one of these per free GPU slot).

Same model, data and training recipe as the earlier runs: this file only picks the settings and calls common_v8.train_run.
  python -u train_v14.py --gpu 0 --family distant --seed 1
Rerunning the same command resumes after the last finished phase (also on a new machine: the newest checkpoint is fetched from HF).
Every finished checkpoint is uploaded to HF while training goes on, and at the end the upload is verified by file size.

--tiny = the 5-minute test: 20,000 training pairs, phases of a few hundred steps, nothing uploaded.
"""
import os, sys, json, glob, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--family", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n_phases", type=int, default=0, help="default: N_PHASES in v14_config.py")
ap.add_argument("--run", default="", help="run name (default: v14_<family>_s<seed>)")
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--no_hf", action="store_true")
args = ap.parse_args()

import v14_config as CFG
FAM = CFG.FAMILIES[args.family]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["CML_LANGSET"] = FAM["langset"]          # must be set before common_v8 is imported (it picks the tokenizer and languages)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
REPO = CFG.REPO; os.chdir(REPO); sys.path.insert(0, REPO)
import common_v8
from common_v8 import *
import v14_hf as HF

RUN = args.run or CFG.run_name(args.family, args.seed)
N_PHASES = args.n_phases or CFG.N_PHASES
USE_HF = not (args.no_hf or args.tiny)
train_langs = ["hi"] if FAM["mode"] == "only_hi" else list(FAM["order"])
assert all(l in LANGS for l in train_langs), f"language set {LANGSET} has {LANGS}, the run needs {train_langs}"

C = RunConfig(run_name=RUN, mode=FAM["mode"], order=tuple(FAM["order"]), n_phases=N_PHASES, seed=args.seed, time_cap_h=240.0,
              hf_repo=None)                           # hf_repo=None: the old silent background backup is off; the Uploader below replaces it
if args.tiny:
    C.eval_every, C.warmup_steps, C.min_steps, C.cooldown_steps, C.max_stable_steps = 50, 50, 100, 100, 300
print(C, flush=True)
print(f"family {args.family} | languages {LANGS} | trains {train_langs} | tokenizer {os.path.basename(os.path.dirname(common_v8._SPM))} | device {DEVICE}", flush=True)

def load_tiny(n_train=20000):
    """the test's data: validation + FLORES as usual (small), and only the first n_train training pairs per language"""
    D = load_data(train_langs=[], n_val=200, n_chrf=20)
    os.makedirs(P("data/_tiny"), exist_ok=True)
    for l in train_langs:
        tmp = P(f"data/_tiny/{LANGSET}-{l}.jsonl")
        with open(P(f"data/en-{l}/train.jsonl"), encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
            for _, line in zip(range(n_train), f):
                g.write(line)
        D["train"][l] = load_pairs(tmp, l)
    return D

# ---- resume on a new machine: bring back the newest checkpoint from HF
if USE_HF:
    ph = HF.restore_latest(RUN)
    if ph is not None:
        f = glob.glob(P(f"runs/{RUN}/ckpt/p{ph:03d}_*.pt"))[0]
        if not os.path.exists(P(f"runs/{RUN}/phases.jsonl")):          # train_run needs this file to resume: rebuild its last row from the checkpoint
            ck = torch.load(f, map_location="cpu")
            with open(P(f"runs/{RUN}/phases.jsonl"), "w") as g:
                g.write(json.dumps({"phase": ck["phase"], "lang": ck["lang"], **ck["info"], **ck["eval"], "total_min": 0.0}) + "\n")
            del ck
        print(f"restored phase {ph} of {RUN} from HF; training resumes after it", flush=True)

D = load_tiny() if args.tiny else load_data(train_langs=train_langs)
up = None
if USE_HF:
    print(f"checkpoints go to HF repo {HF.ckpt_repo()} under {RUN}/", flush=True)
    up = HF.Uploader(RUN); up.start()
try:
    train_run(C, D)
finally:
    if up is not None:
        ok, n, missing = up.stop_and_finish()
        print(f"HF CHECK {RUN}: {ok} of {n} checkpoints on HF with the right size" + (f" | MISSING: {missing}" if missing else " | all there"), flush=True)

n_ck = len(glob.glob(P(f"runs/{RUN}/ckpt/p*.pt")))
if n_ck < N_PHASES + 1:
    print(f"NOT FINISHED: {RUN} has {n_ck} checkpoints, expected {N_PHASES + 1} (rerun the same command to continue)", flush=True)
    sys.exit(2)
if up is not None and missing:
    sys.exit(3)
print(f"RUN COMPLETE: {RUN}", flush=True)
