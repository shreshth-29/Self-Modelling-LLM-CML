"""v14_config.py: WHAT the multi-seed pilot runs. The one place to change the experiment.

The pilot: the same training recipe as before (common_v8.train_run, nothing changed), repeated over seeds, for three kinds of run:
  distant  Hindi -> Chinese -> Arabic, repeating      (tokenizer_v4, as v4b_continual_s0)
  related  Hindi -> Gujarati -> Punjabi, repeating    (tokenizer_v8, as v8_indic_continual_s0)
  hindi    Hindi in every phase                        (tokenizer_v4, as v4b_only_hi_s0)
Then the count-matched lesion tests of core_v11.py on every model, and one summary across seeds.

No torch imports here: train_v14.py, exp_v14.py and summary_v14.py all read this file.
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
OUTROOT = "v14"                                   # results go to analysis/v14/

# ---------------------------------------------------------------- runs
FAMILIES = {
    "distant": dict(langset="distant", mode="continual", order=("hi", "zh", "ar"), only="", label="Distant\n(Hindi, Chinese,\nArabic)"),
    "related": dict(langset="indic", mode="continual", order=("hi", "gu", "pa"), only="", label="Related\n(Hindi, Gujarati,\nPunjabi)"),
    # only="hi": a Hindi-only model knows one language, so its self is found on Hindi, lesions are scored on Hindi,
    # and "the next phase" is Hindi again (the tokenizer's other two languages were never trained)
    "hindi": dict(langset="distant", mode="only_hi", order=("hi", "zh", "ar"), only="hi", label="Single task\n(Hindi only)"),
}
FAMILY_ORDER = ["hindi", "related", "distant"]   # left to right on the trend figure: least to most different languages
SEEDS = [1, 2, 3]                                 # new seeds (seed 0 = the old runs below)
N_PHASES = 60                                     # 20 full cycles of 3 languages; phase 0 = the untrained model

# Old single-seed runs, already on HF. Same model and training recipe, so they are analysed with the same code as extra models.
# They are marked "old" in every table and drawn hollow on the figure.
OLD_RUNS = [
    dict(family="distant", tag="old0", run="v4b_continual_s0"),
    dict(family="related", tag="old0", run="v8_indic_continual_s0"),
    dict(family="related", tag="old1", run="v8_indic_continual_s1"),
    dict(family="hindi", tag="old0", run="v4b_only_hi_s0"),
]

def run_name(family, seed):
    return f"v14_{family}_s{seed}"

def trained_lang(family, phase):
    """language trained in this phase (phase 1 = first language)"""
    f = FAMILIES[family]
    return "hi" if f["mode"] == "only_hi" else f["order"][(phase - 1) % len(f["order"])]

def models(seeds=None, families=None, include_old=True):
    """every model of the pilot: dicts with family, tag (s1 / old0), label (family_tag), run, new (trained by this pilot?)"""
    seeds = SEEDS if seeds is None else seeds
    families = list(FAMILIES) if families is None else families
    out = [dict(family=f, tag=f"s{s}", label=f"{f}_s{s}", run=run_name(f, s), seed=s, new=True) for f in families for s in seeds]
    if include_old:
        out += [dict(o, label=f"{o['family']}_{o['tag']}", seed=None, new=False) for o in OLD_RUNS if o["family"] in families]
    return out

# ---------------------------------------------------------------- analysis (all through core_v11.py, tau 0.2, FLORES dev)
# MANY checkpoints per model, not a hand-picked few: every phase after the first cycle is used, UNTIL THE MODEL LOSES PLASTICITY.
# With 60 phases that is up to 57 checkpoints per model = 19 per language per seed (phases 4 .. 60).
SIZE_PHASES = list(range(0, N_PHASES + 1))        # self size per layer at every checkpoint
PERSIST_FROM = list(range(3, N_PHASES))           # persistence between phase p and p+1, for p = 3 .. 59
HEADLINE_FROM = list(range(6, N_PHASES))          # p = 6 on: used for the headline persistence numbers
LESION_PHASES = list(range(4, N_PHASES + 1))      # lesion without training: at EVERY checkpoint from phase 4 on
RETRAIN_EVERY = 2                                 # the two lesion-then-train tests: every 2nd checkpoint (4, 6, 8, ...). With 3 languages
                                                  # this still visits every language equally often. 1 = every checkpoint (twice the time).

# Loss of plasticity, found per run from its own training log. After each phase we look at the language just trained:
# it "did not come back" if its validation loss is more than PLASTICITY_TOL above the best that language ever reached.
# Onset = the first phase where that happens and it happens again at the language's next visit (one bad visit alone is noise).
# Only checkpoints BEFORE the onset are analysed. (Old distant run: onset at phase 64. Old Hindi-only run: none in 90 phases.)
PLASTICITY_TOL = 0.03

def plasticity_onset(rows, tol=None):
    """rows = the lines of a run's phases.jsonl (dicts). Returns the onset phase, or None if the run never lost plasticity."""
    tol = PLASTICITY_TOL if tol is None else tol
    visits = {}
    for r in sorted({r["phase"]: r for r in rows}.values(), key=lambda r: r["phase"]):
        v = (r.get("val_masked") or {}).get(r.get("lang"))
        if r["phase"] == 0 or v is None:
            continue
        if r.get("reason") == "failed":                 # a phase that diverged even at a quarter of the learning rate
            return r["phase"]
        visits.setdefault(r["lang"], []).append((r["phase"], v))
    onset = None
    for seq in visits.values():
        best, first_bad = float("inf"), None
        for ph, v in seq:
            bad = v > best * (1 + tol)
            if bad and first_bad is not None:
                onset = first_bad if onset is None else min(onset, first_bad)
                break
            first_bad = ph if bad else None
            best = min(best, v)
    return onset
STATIC_DRAWS = "0,1"                              # lesion without training: two random draws of the lesioned neurons, three doses
# lesion, then training. One repeat per checkpoint and three conditions (untouched, self, task): the many checkpoints are the repeats.
TRAIN_CONDS = "none,self@1.0,task@1.0"
TRAIN_SEEDS = "0"
TRAIN_STEPS = 1000                                # ... on the run's next language
HELDOUT = "indonesian"                            # ... and on a language no model has seen (same for every kind of run)
HELDOUT_STEPS = 1500                              # as in the earlier Indonesian test (v11)
HELDOUT_LR = "1e-4"
HELDOUT_REACH = 300                               # "time to catch up": steps a lesioned model needs to reach the loss the untouched one has at this step

# ---------------------------------------------------------------- GPU memory each kind of job needs (GB); the scheduler packs jobs by this
NEED_GB = {"train": float(os.environ.get("V14_TRAIN_GB", 22.5)),      # one training run (16,000 tokens per batch)
           "retrain": float(os.environ.get("V14_RETRAIN_GB", 13.0)),  # lesion + 1,000 training steps (8,000 tokens per batch)
           "light": float(os.environ.get("V14_LIGHT_GB", 6.0))}       # sizes, persistence, static lesions (no training)
# ordinary memory (RAM, GB) per job: the training pairs are held as Python lists, about 3.5 GB per language
NEED_RAM = {"train": 14.0, "train_hi": 7.5, "retrain": 6.0, "light": 4.0, "tiny": 4.0}
