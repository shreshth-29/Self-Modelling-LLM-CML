"""trials_v15.py: Adi's trials, one table and two figures. CPU only; reads analysis/v14/ (lesion A and B, already measured) and analysis/v15/.

  python trials_v15.py

Operation   lesion = the neurons are re-initialised          freeze = their incoming weights + bias cannot change while training
Trial       A / alpha = then learn Indonesian (outside every set)      B / beta = then learn another language of the run's own set
            C / gamma = then train again on the language the checkpoint was just trained on
Size        strong = the full count (the smaller of self and task in each layer, about half a layer)      weak = half of that
Always self against the SAME NUMBER of random task neurons, all 8 layers, and against the untouched checkpoint trained the same way.
Number reported: extra held-out loss. Trial A: average of the evaluations from step 100 on. Trials B and C: at the last step.
Writes analysis/v15/summary/: trials.csv (the table), trials_per_checkpoint.csv, trials_strong.png, trials_weak.png, report.txt
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import v14_config as CFG

A14 = os.path.join(CFG.REPO, "analysis", "v14"); A15 = os.path.join(CFG.REPO, "analysis", "v15")
OUT = os.path.join(A15, "summary"); os.makedirs(OUT, exist_ok=True)
COL = {"hindi": "#2A6F97", "related": "#00918F", "distant": "#C8102E"}
NAME = {"hindi": "Hindi only", "related": "Related (Hindi, Gujarati, Punjabi)", "distant": "Distant (Hindi, Chinese, Arabic)"}
FAMS = ["hindi", "related", "distant"]
TRIAL = {("lesion", "A"): "Trial A", ("lesion", "B"): "Trial B", ("lesion", "C"): "Trial C",
         ("freeze", "A"): "Trial alpha", ("freeze", "B"): "Trial beta", ("freeze", "C"): "Trial gamma"}
WHAT = {"A": "then learn Indonesian (outside every set)", "B": "then learn another language of its own set (the next one)",
        "C": "then train again on the language just trained"}
fam_of = lambda label: label.rsplit("_", 1)[0]

def load(pattern, op, trial, keep):
    """one row per (run, start checkpoint, condition): extra loss against the untouched checkpoint of the same file"""
    rows = []
    for f in sorted(glob.glob(pattern)):
        label = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if label.startswith("fresh"):
            continue
        c = pd.read_csv(f)
        if not (c["cond"] == "none").any():
            continue
        last = c["step"].max(); none = c[c["cond"] == "none"].groupby("step")["val_loss"].mean()
        ph = int(str(c["start"].iloc[0]).rsplit("_p", 1)[1])
        for cond, q in c[c["cond"] != "none"].groupby("cond"):
            if cond not in keep:
                continue
            ex = q.groupby("step")["val_loss"].mean() - none
            late = ex[ex.index >= 100] if (ex.index >= 100).any() else ex
            grp, dose = cond.split("@")
            rows.append(dict(op=op, trial=trial, label=label, family=fam_of(label), phase=ph, group=grp, size="strong" if dose == "1.0" else "weak",
                             avg=float(late.mean()), end=float(ex.get(last, np.nan))))
    return rows

rows = []
rows += load(f"{A14}/train_heldout/*/p*/curves.csv", "lesion", "A", {"self@1.0", "task@1.0", "self@0.5"})
rows += load(f"{A15}/lesion_A_weak/*/p*/curves.csv", "lesion", "A", {"task@0.5"})
b14 = load(f"{A14}/train_next/*/p*/curves.csv", "lesion", "B", {"self@1.0", "task@1.0", "self@0.5"})
rows += [r for r in b14 if not CFG.FAMILIES[r["family"]]["only"]]          # for a Hindi-only run "the next language" is Hindi again: that is trial C, measured anew below
rows += load(f"{A15}/lesion_B_weak/*/p*/curves.csv", "lesion", "B", {"task@0.5"})
ALLC = {"self@1.0", "task@1.0", "self@0.5", "task@0.5"}
rows += load(f"{A15}/lesion_C/*/p*/curves.csv", "lesion", "C", ALLC)
for t in "ABC":
    rows += load(f"{A15}/freeze_{t}/*/p*/curves.csv", "freeze", t, ALLC)
D = pd.DataFrame(rows)
if not len(D):
    print("no results found"); raise SystemExit(0)
D["value"] = np.where(D["trial"] == "A", D["avg"], D["end"])              # the number each trial is reported with
W = D.pivot_table(index=["op", "trial", "size", "family", "label", "phase"], columns="group", values="value").reset_index()
for g in ("self", "task"):
    if g not in W:
        W[g] = np.nan
W["new"] = ~W["label"].str.contains("_old")
W.to_csv(f"{OUT}/trials_per_checkpoint.csv", index=False)

# ------------------------------------------------------------------ the table
from scipy import stats
out = []
for op in ("lesion", "freeze"):
    for trial in "ABC":
        for size in ("strong", "weak"):
            for fam in FAMS:
                q = W[(W["op"] == op) & (W["trial"] == trial) & (W["size"] == size) & (W["family"] == fam)].dropna(subset=["self", "task"])
                if not len(q):
                    continue
                r = q.groupby("label")[["self", "task"]].mean()
                p = stats.ttest_rel(r["self"], r["task"]).pvalue if len(r) >= 3 else np.nan
                out.append({"trial": TRIAL[(op, trial)], "what": f"{op}, {WHAT[trial]}", "size": size, "language set": fam,
                            "self": q["self"].mean(), "task": q["task"].mean(), "self minus task": (q["self"] - q["task"]).mean(),
                            "checkpoints where self costs more": f"{int((q['self'] > q['task']).sum())} of {len(q)}",
                            "runs where self costs more": f"{int((r['self'] > r['task']).sum())} of {len(r)}", "p (paired, over runs)": p})
T = pd.DataFrame(out); T.to_csv(f"{OUT}/trials.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
lines = ["ADI'S TRIALS on the v14 checkpoints. Extra held-out loss against the untouched checkpoint (trial A / alpha: average from step 100 on; B, C, beta, gamma: at the last step).",
         "strong = full count (about half a layer), weak = half of that. Self against the same number of random task neurons. 'costs more' = higher extra loss.", ""]
for trial in T["trial"].unique():
    q = T[T["trial"] == trial]
    lines += [f"=== {trial}: {q['what'].iloc[0]} ===", q.drop(columns=["trial", "what"]).round(4).to_string(index=False), ""]
print("\n".join(lines)); open(f"{OUT}/report.txt", "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ figures: every checkpoint its own dot
def figure(size):
    rows_def = [(op, t) for op in ("lesion", "freeze") for t in "ABC" if len(W[(W["op"] == op) & (W["trial"] == t) & (W["size"] == size)].dropna(subset=["self", "task"]))]
    if not rows_def:
        return
    fig, axs = plt.subplots(len(rows_def), 3, figsize=(13.5, 4.0 * len(rows_def)), squeeze=False)
    for i, (op, t) in enumerate(rows_def):
        qr = W[(W["op"] == op) & (W["trial"] == t) & (W["size"] == size)].dropna(subset=["self", "task"])
        lo = min(0, qr["self"].min(), qr["task"].min()); hi = max(qr["self"].max(), qr["task"].max()) * 1.08
        for j, fam in enumerate(FAMS):
            ax = axs[i, j]; q = qr[qr["family"] == fam]
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if not len(q):
                ax.text(0.5, 0.5, "does not exist:\na Hindi-only run knows one language" if fam == "hindi" else "no results", ha="center", va="center", fontsize=9.5, color="#777777", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ("left", "bottom"):
                    ax.spines[s_].set_visible(False)
                continue
            ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.9, ls=":")
            for new, fc in ((True, COL[fam]), (False, "white")):
                z = q[q["new"] == new]
                ax.scatter(z["task"], z["self"], s=18, facecolors=fc, edgecolors=COL[fam], linewidths=0.9, alpha=0.8 if new else 1.0, zorder=3)
            k, n = int((q["self"] > q["task"]).sum()), len(q)
            ax.set_title(NAME[fam], fontsize=9.5, loc="left")
            ax.text(0.04, 0.97, f"self costs more in {k} of {n} checkpoints\naverage: self {q['self'].mean():.3g}, task {q['task'].mean():.3g}", transform=ax.transAxes, va="top", fontsize=8.8, color="#222222")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f"{op} of TASK neurons: extra loss", fontsize=9)
            ax.grid(color="#F0F0F0", lw=0.7); ax.set_axisbelow(True)
        axs[i, 0].set_ylabel(f"{TRIAL[(op, t)].upper()}\n{op}, {WHAT[t]}\n\n{op} of SELF neurons: extra loss", fontsize=8.8)
    fig.text(0.01, 0.006, f"{size.upper()} size ({'the full count, about half a layer' if size == 'strong' else 'half of the full count, about a quarter of a layer'}). "
             "One dot = one start checkpoint of one run. Filled = the 3 new seeds, hollow = the earlier seed-0 runs. Dots ABOVE the dotted line = touching the self cost more than touching the task part.\n"
             "Extra loss = held-out loss minus that of the untouched checkpoint trained the same way (trial A / alpha: average from step 100 on; the others: at the last step). "
             "Lesion = re-initialised. Freeze = incoming weights cannot change during the training.", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{OUT}/trials_{size}.png", dpi=140); plt.close(fig)

figure("strong"); figure("weak")
print("written:", OUT)
