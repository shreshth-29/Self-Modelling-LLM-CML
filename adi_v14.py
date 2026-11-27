"""adi_v14.py: answers to Adi's comments on the v14 pilot, from the tables already in analysis/v14/ (CPU only, no new runs).

  python adi_v14.py

1. training_curve.png      chrF of every language after every phase, average of the 3 new seeds (his "reward across phases" curve)
2. per_checkpoint.png      every checkpoint as its own dot: loss rise after a SELF lesion (up) against the same after a TASK lesion (across),
                           split his way: language just trained / another language of the model's set / a language outside the set
3. by_test_language.csv    the same split as a table (strong = full count, about half a layer; weak = half of that)
All in analysis/v14/summary/.
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import v14_config as CFG

ROOT = os.path.join(CFG.REPO, "analysis", CFG.OUTROOT); OUT = os.path.join(ROOT, "summary")
COL = {"hindi": "#2A6F97", "related": "#00918F", "distant": "#C8102E"}
NAME = {"hindi": "Hindi only", "related": "Related (Hindi, Gujarati, Punjabi)", "distant": "Distant (Hindi, Chinese, Arabic)"}
LANG = {"hi": "Hindi", "zh": "Chinese", "ar": "Arabic", "gu": "Gujarati", "pa": "Punjabi"}
fam_of = lambda label: label.rsplit("_", 1)[0]

# ============================================================== 1. training curve
def training_curve():
    fams = [f for f in ["hindi", "related", "distant"]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    shades = ["#1b1b1b", "#7a7a7a", "#b9b9b9"]
    for ax, fam in zip(axs, fams):
        F = CFG.FAMILIES[fam]; langs = ["hi"] if F["only"] else list(F["order"])
        runs = sorted(glob.glob(f"{ROOT}/training/v14_{fam}_s*_phases.jsonl"))
        data = {l: {} for l in langs}
        for f in runs:
            for line in open(f):
                if line.strip():
                    d = json.loads(line)
                    for l in langs:
                        if l in d.get("chrf_masked", {}):
                            data[l].setdefault(d["phase"], []).append(d["chrf_masked"][l])
        for l, sh in zip(langs, shades):
            ph = sorted(data[l]); m = np.array([np.mean(data[l][p]) for p in ph])
            lo = np.array([np.min(data[l][p]) for p in ph]); hi = np.array([np.max(data[l][p]) for p in ph])
            c = COL[fam] if l == "hi" else sh
            ax.fill_between(ph, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(ph, m, color=c, lw=1.8, marker="o", ms=3, label=LANG[l])
            ax.annotate(LANG[l], (ph[-1] + 0.4, m[-1]), fontsize=8.5, color=c, va="center")
        ax.set_title(f"{NAME[fam]}  ({len(runs)} seeds)", fontsize=10.5, loc="left")
        ax.set_xlabel("phase (one phase = training on one language)"); ax.set_xlim(0, 34)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    axs[0].set_ylabel("chrF on FLORES devtest (200 sentences)\nmeasured after every phase")
    fig.text(0.01, 0.01, "Line = average of the 3 new seeds, band = lowest to highest seed. In the continual runs a language is trained every 3rd phase: it drops while the "
             "other two are trained and comes back when it is trained again. Nothing gets worse over the 30 phases.", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{OUT}/training_curve.png", dpi=170); plt.close(fig)

# ============================================================== 2. one row per checkpoint, split by which language is tested
def split_cond(df):
    p = df["cond"].astype(str).str.split("@", expand=True); df = df.copy(); df["group"] = p[0]; df["dose"] = p[1] if p.shape[1] > 1 else ""
    return df

def collect():
    rows = []
    # ---- lesion without training: loss rise on each language the model knows
    for f in sorted(glob.glob(f"{ROOT}/static/*/static.csv")):
        label = os.path.basename(os.path.dirname(f)); fam = fam_of(label)
        s = split_cond(pd.read_csv(f)); s = s[(s["scope"].astype(str) == "all") & (s["mode"] == "lesion") & s["group"].isin(["self", "task"])]
        g = s.groupby(["phase", "lang", "dose", "group"])["d_loss"].mean().unstack("group").reset_index()
        for _, r in g.iterrows():
            just = r["lang"] == CFG.trained_lang(fam, int(r["phase"]))
            rows.append(dict(label=label, family=fam, phase=int(r["phase"]), lang=r["lang"], dose=r["dose"], self=r.get("self"), task=r.get("task"),
                             test="A. lesion, no training", which="1. language just trained" if just else "2. another language of its set"))
    # ---- lesion, then 1,000 steps on the run's next language (Hindi-only: Hindi again)
    for f in sorted(glob.glob(f"{ROOT}/train_next/*/p*/curves.csv")):
        label = os.path.basename(os.path.dirname(os.path.dirname(f))); fam = fam_of(label)
        c = pd.read_csv(f); last = c["step"].max(); c = c[c["step"] == last]
        none = c[c["cond"] == "none"]["val_loss"].mean(); x = split_cond(c[c["cond"] != "none"])
        ph = int(str(c["start"].iloc[0]).rsplit("_p", 1)[1]); nxt = CFG.trained_lang(fam, ph + 1)
        for dose in ("1.0", "0.5"):
            v = x[x["dose"] == dose].groupby("group")["val_loss"].mean() - none
            if "self" in v:
                rows.append(dict(label=label, family=fam, phase=ph, lang=nxt, dose=dose, self=v.get("self"), task=v.get("task"), look=v.get("look"),
                                 test="B. lesion, then 1,000 steps", which="1. the language just trained (trained again)" if CFG.FAMILIES[fam]["only"] else "2. another language of its set (the next one)"))
    # ---- lesion, then 1,500 steps on Indonesian (no model has seen it)
    for f in sorted(glob.glob(f"{ROOT}/train_heldout/*/p*/curves.csv")):
        label = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if label.startswith("fresh"):
            continue
        fam = fam_of(label); c = pd.read_csv(f); c = c[c["step"] >= 100]
        none = c[c["cond"] == "none"].set_index("step")["val_loss"]; x = split_cond(c[c["cond"] != "none"])
        x["extra"] = x["val_loss"] - x["step"].map(none); ph = int(str(c["start"].iloc[0]).rsplit("_p", 1)[1])
        for dose in ("1.0", "0.5"):
            v = x[x["dose"] == dose].groupby("group")["extra"].mean()
            if "self" in v:
                rows.append(dict(label=label, family=fam, phase=ph, lang="id", dose=dose, self=v.get("self"), task=v.get("task"), look=v.get("look"),
                                 test="B. lesion, then 1,500 steps", which="3. a language outside its set (Indonesian)"))
    D = pd.DataFrame(rows); D["new"] = ~D["label"].str.contains("_old")
    return D

def table(D):
    out = []
    for (test, which, fam, dose), q in D.groupby(["test", "which", "family", "dose"]):
        other = "task" if q["task"].notna().any() else "look"
        q = q.dropna(subset=["self", other])
        if not len(q):
            continue
        per_run = q.groupby("label")[["self", other]].mean()
        out.append({"test": test, "language tested": which, "language set": fam, "lesion size": "strong (full count, about half a layer)" if dose == "1.0" else "weak (half of that)",
                    "compared with": "task neurons, random" if other == "task" else "task neurons that look like the self", "self": q["self"].mean(), "other": q[other].mean(),
                    "self / other": q["self"].mean() / q[other].mean() if q[other].mean() > 0 else np.nan,
                    "checkpoints where self is worse": f"{int((q['self'] > q[other]).sum())} of {len(q)}",
                    "runs where self is worse": f"{int((per_run['self'] > per_run[other]).sum())} of {len(per_run)}"})
    T = pd.DataFrame(out)
    T["_o"] = T["language set"].map({"hindi": 0, "related": 1, "distant": 2})
    T = T.sort_values(["test", "language tested", "lesion size", "_o"]).drop(columns="_o")
    T.to_csv(f"{OUT}/by_test_language.csv", index=False)
    return T

def per_checkpoint(D):
    rows_def = [("A. lesion, no training", "1.", "Lesion, no training\nLANGUAGE JUST TRAINED", "loss rise"),
                ("A. lesion, no training", "2.", "Lesion, no training\nANOTHER LANGUAGE OF ITS SET", "loss rise"),
                ("B. lesion, then 1,000 steps", "", "Lesion, then 1,000 training steps\n(language named in each panel)", "extra loss left at the end"),
                ("B. lesion, then 1,500 steps", "3.", "Lesion, then learn INDONESIAN\n(outside every set)", "extra loss while learning")]
    fams = ["hindi", "related", "distant"]
    fig, axs = plt.subplots(len(rows_def), 3, figsize=(13.5, 4.1 * len(rows_def)))
    S = D[D["dose"] == "1.0"]
    for i, (test, pre, rname, unit) in enumerate(rows_def):
        q_row = S[(S["test"] == test) & (S["which"].str.startswith(pre))].dropna(subset=["self", "task"])
        lim = max(q_row["self"].max(), q_row["task"].max()) * 1.06 if len(q_row) else 1
        for j, fam in enumerate(fams):
            ax = axs[i, j]; q = q_row[q_row["family"] == fam]
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if not len(q):
                ax.text(0.5, 0.5, "does not exist:\na Hindi-only model knows one language", ha="center", va="center", fontsize=9.5, color="#777777", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ("left", "bottom"):
                    ax.spines[s_].set_visible(False)
                continue
            ax.plot([0, lim], [0, lim], color="#888888", lw=0.9, ls=":")
            for new, fc in ((True, COL[fam]), (False, "white")):
                z = q[q["new"] == new]
                ax.scatter(z["task"], z["self"], s=16, facecolors=fc, edgecolors=COL[fam], linewidths=0.9, alpha=0.75 if new else 1.0, zorder=3)
            k, n = int((q["self"] > q["task"]).sum()), len(q)
            sub = q["which"].iloc[0].split(". ", 1)[1] if pre == "" else ""
            ax.set_title(f"{NAME[fam]}" + (f"\n{sub}" if sub else ""), fontsize=9.5, loc="left")
            ax.text(0.04, 0.96, f"self lesion is worse in {k} of {n} checkpoints\naverage: self {q['self'].mean():.3g}, task {q['task'].mean():.3g}",
                    transform=ax.transAxes, va="top", fontsize=8.8, color="#222222")
            ax.set_xlim(min(0, q_row["task"].min()), lim); ax.set_ylim(min(0, q_row["self"].min()), lim)
            ax.set_xlabel(f"after a TASK lesion: {unit}", fontsize=9)
            ax.grid(color="#F0F0F0", lw=0.7); ax.set_axisbelow(True)
        axs[i, 0].set_ylabel(f"{rname}\n\nafter a SELF lesion: {unit}", fontsize=9)
    fig.text(0.01, 0.006, "One dot = one checkpoint of one run (in the 'no training' rows: one checkpoint and one language). Filled = the 3 new seeds, hollow = the earlier seed-0 runs. "
             "Dots ABOVE the dotted line = the self lesion did more damage than the task lesion.\nSame number of self and task neurons lesioned in every layer (the size of the smaller group, "
             "about half a layer), all 8 layers, self found with |correlation| >= 0.2. Loss is on held-out sentences.", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(f"{OUT}/per_checkpoint.png", dpi=150); plt.close(fig)

training_curve()
D = collect(); D.to_csv(f"{OUT}/per_checkpoint.csv", index=False)
T = table(D)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
print(T.round(3).to_string(index=False))
per_checkpoint(D)
print("\nwritten:", OUT)
