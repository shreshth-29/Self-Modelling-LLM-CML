"""summary_v14.py: one summary across seeds of the multi-seed pilot. CPU only; reads the tables in analysis/v14/.

  python summary_v14.py            (run it anywhere after  git pull)

One row per MODEL (a kind of run x a seed). Three questions, each asked for Hindi only, related and distant languages:
  1. persistence   do self neurons change less than task neurons from one phase to the next?
  2. lesion        does re-initialising self neurons hurt more than the same number of task neurons? (no training after)
  3. lasting cost  after the lesion and 1,000 training steps on the next language, is the self lesion still worse?
Writes analysis/v14/summary/: per_model.csv, by_family.csv, tests.csv, persistence_by_layer.csv, report.txt, trend.png, gap_by_phase.png
"""
import os, glob, json, argparse
import numpy as np, pandas as pd
import v14_config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=CFG.OUTROOT)
ap.add_argument("--tiny", action="store_true", help="the test: use whatever phases are there")
args = ap.parse_args()
ROOT = os.path.join(CFG.REPO, "analysis", args.root)
OUT = os.path.join(ROOT, "summary"); os.makedirs(OUT, exist_ok=True)
LAYERS = [f"enc{j}" for j in range(4)] + [f"dec{j}" for j in range(4)]
REPORT = []

def say(s=""):
    print(s); REPORT.append(str(s))

def family_of(label):
    return label.rsplit("_", 1)[0]

def pick(phases_wanted, phases_there):
    """the wanted phases that exist; in the tiny test (or if none exist) all that are there"""
    there = sorted(set(int(p) for p in phases_there))
    got = [p for p in phases_wanted if p in there]
    return there if (args.tiny or not got) else got

def split_cond(df):
    df = df.copy()
    parts = df["cond"].astype(str).str.split("@", expand=True)
    df["group"] = parts[0]; df["dose"] = parts[1] if parts.shape[1] > 1 else ""
    return df

M = {}                                              # label -> dict of numbers
def put(label, **kv):
    M.setdefault(label, {}).update(kv)

# =============================================================================================== 0. training
def part_training():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/training/*_phases.jsonl")):
        run = os.path.basename(f)[:-len("_phases.jsonl")]
        r = {}
        for line in open(f):
            if line.strip():
                d = json.loads(line); r[d["phase"]] = d
        if not r:
            continue
        last = r[max(r)]
        row = {"run": run, "phases": max(r), "total_steps": sum(x.get("steps", 0) for x in r.values()), "hours": round(last.get("total_min", 0) / 60, 1)}
        best = {}
        for x in r.values():                         # chrF of each language right after its own most recent phase
            if x["lang"] in x.get("chrf_masked", {}):
                best[x["lang"]] = x["chrf_masked"][x["lang"]]
        row.update({f"chrF_{l}": v for l, v in best.items()})
        rows.append(row)
    if rows:
        t = pd.DataFrame(rows)
        say("=== 0. Training: one row per new run. chrF_xx = chrF on language xx right after its last own phase (FLORES devtest, 200 sentences) ===")
        say(t.to_string(index=False)); say()
        t.to_csv(f"{OUT}/training.csv", index=False)

# =============================================================================================== 1. self size
def part_sizes():
    for f in sorted(glob.glob(f"{ROOT}/sizes/*/sizes.csv")):
        label = os.path.basename(os.path.dirname(f)); fam = CFG.FAMILIES[family_of(label)]
        s = pd.read_csv(f)
        want = "Hindi only" if fam["only"] else "all the model's languages"
        s = s[s["self_def"] == want]
        s = s[s["phase"].isin(pick(CFG.LESION_PHASES, s["phase"]))]
        if len(s):
            pct = 100 * s["n_self"] / s["n_alive"].clip(lower=1)
            put(label, self_pct=float(pct.mean()), self_pct_enc=float(pct[s["layer"].str.startswith("enc")].mean()),
                self_pct_dec=float(pct[s["layer"].str.startswith("dec")].mean()))

# =============================================================================================== 2. persistence
def matched_gap(x):
    """self minus task persistence inside 10 bins of equal average activity (same method as defn_v13.py)"""
    m = x["in_self"].values == 1
    bins = pd.qcut(x["mean"].rank(method="first"), 10, labels=False).values
    diffs, w = [], []
    for b in range(10):
        i = bins == b
        if (m & i).sum() >= 3 and ((~m) & i).sum() >= 3:
            diffs.append(x["persistence"].values[m & i].mean() - x["persistence"].values[(~m) & i].mean()); w.append(min((m & i).sum(), ((~m) & i).sum()))
    return float(np.average(diffs, weights=w)) if len(diffs) >= 3 else np.nan

def part_persistence():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/persistence/*/pers_p*.csv.gz")):
        label = os.path.basename(os.path.dirname(f))
        d = pd.read_csv(f); d = d[d["valid"] == 1]
        for L, x in d.groupby("layer"):
            m = x["in_self"].values == 1
            if m.sum() < 20 or (~m).sum() < 20:
                continue
            ps, pt = x["persistence"].values[m].mean(), x["persistence"].values[~m].mean()
            rows.append({"label": label, "family": family_of(label), "phase_a": int(x["phase_a"].iloc[0]), "layer": L, "side": L[:3],
                         "pers_self": ps, "pers_task": pt, "gap": ps - pt, "gap_matched": matched_gap(x), "self_pct": 100 * m.mean()})
    if not rows:
        return None
    P = pd.DataFrame(rows); P.to_csv(f"{OUT}/persistence_by_layer.csv", index=False)
    for label, q in P.groupby("label"):
        h = q[q["phase_a"].isin(pick(CFG.HEADLINE_FROM, q["phase_a"]))]
        put(label, pers_self=h["pers_self"].mean(), pers_task=h["pers_task"].mean(), gap=h["gap"].mean(), gap_matched=h["gap_matched"].mean(),
            gap_matched_enc=h[h["side"] == "enc"]["gap_matched"].mean(), gap_matched_dec=h[h["side"] == "dec"]["gap_matched"].mean(),
            n_transitions=int(h["phase_a"].nunique()))
    return P

# =============================================================================================== 3. lesion, no training
def part_static():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/static/*/static.csv")):
        label = os.path.basename(os.path.dirname(f)); fam = family_of(label)
        s = split_cond(pd.read_csv(f))
        s = s[s["scope"].astype(str) == "all"]
        s = s[s["phase"].isin(pick(CFG.LESION_PHASES, s["phase"]))]
        if not CFG.FAMILIES[fam]["only"]:
            # PREVIOUS language = the one trained one phase earlier (a Hindi-only model has no previous language)
            pv = s[[int(p) >= 2 and l == CFG.trained_lang(fam, int(p) - 1) for l, p in zip(s["lang"], s["phase"])]]
            pv = pv[(pv["mode"] == "lesion") & (pv["dose"] == "1.0")]
            g = pv.groupby(["phase", "draw", "group"])[["d_loss", "d_chrf"]].mean().unstack("group")
            if len(g) and ("d_loss", "self") in g.columns and ("d_loss", "task") in g.columns:
                put(label, prev_les_self=g[("d_loss", "self")].mean(), prev_les_task=g[("d_loss", "task")].mean(),
                    prev_les_diff=(g[("d_loss", "self")] - g[("d_loss", "task")]).mean(),
                    prev_chrf_diff=(g[("d_chrf", "self")] - g[("d_chrf", "task")]).mean(),
                    prev_les_k=int((g[("d_loss", "self")] > g[("d_loss", "task")]).sum()), prev_les_n=len(g))
            gh = pv[pv["lang"] == "hi"].groupby(["phase", "draw", "group"])[["d_loss", "d_chrf"]].mean().unstack("group")
            if len(gh) and ("d_loss", "self") in gh.columns and ("d_loss", "task") in gh.columns:      # Hindi, one phase after Hindi was trained
                put(label, hi_prev_les_self=gh[("d_loss", "self")].mean(), hi_prev_les_task=gh[("d_loss", "task")].mean(),
                    hi_prev_les_diff=(gh[("d_loss", "self")] - gh[("d_loss", "task")]).mean(),
                    hi_prev_chrf_diff=(gh[("d_chrf", "self")] - gh[("d_chrf", "task")]).mean())
        s = s[[l == CFG.trained_lang(fam, int(p)) for l, p in zip(s["lang"], s["phase"])]]      # the language just trained
        if not len(s):
            continue
        s = s.assign(label=label, family=fam); rows.append(s)
        for mode, tag in [("lesion", "les"), ("silence", "sil")]:
            g = s[s["mode"] == mode].groupby(["phase", "draw", "dose", "group"])["d_loss"].mean().unstack("group").reset_index()
            one = g[g["dose"] == "1.0"]
            if {"self", "task"} <= set(one.columns) and len(one.dropna(subset=["self", "task"])):
                o = one.dropna(subset=["self", "task"])
                put(label, **{f"{tag}_self": o["self"].mean(), f"{tag}_task": o["task"].mean(), f"{tag}_diff": (o["self"] - o["task"]).mean(),
                              f"{tag}_ratio": o["self"].mean() / max(o["task"].mean(), 1e-9), f"{tag}_k": int((o["self"] > o["task"]).sum()), f"{tag}_n": len(o)})
            half = g[g["dose"] == "0.5"]
            if mode == "lesion" and {"self", "look"} <= set(half.columns) and len(half.dropna(subset=["self", "look"])):
                h = half.dropna(subset=["self", "look"])
                put(label, les_self05=h["self"].mean(), les_look05=h["look"].mean())
        c = s[(s["mode"] == "lesion") & (s["dose"] == "1.0")].dropna(subset=["d_chrf"]).groupby("group")["d_chrf"].mean()
        if "self" in c and "task" in c:
            put(label, chrf_drop_self=c["self"], chrf_drop_task=c["task"])
        # the SAME language in every kind of run: Hindi, at the checkpoints where Hindi was just trained
        h = s[(s["mode"] == "lesion") & (s["dose"] == "1.0") & (s["lang"] == "hi")]
        g = h.groupby(["phase", "draw", "group"])[["d_loss", "d_chrf"]].mean().unstack("group")
        if len(g) and ("d_loss", "self") in g.columns and ("d_loss", "task") in g.columns:
            put(label, hi_les_self=g[("d_loss", "self")].mean(), hi_les_task=g[("d_loss", "task")].mean(),
                hi_les_diff=(g[("d_loss", "self")] - g[("d_loss", "task")]).mean(),
                hi_chrf_self=g[("d_chrf", "self")].mean(), hi_chrf_task=g[("d_chrf", "task")].mean(),
                hi_chrf_diff=(g[("d_chrf", "self")] - g[("d_chrf", "task")]).mean(), hi_les_n=len(g))
    if rows:
        pd.concat(rows).to_csv(f"{OUT}/static_trained_language.csv", index=False)

# =============================================================================================== 4. lesion, then training
def part_train():
    for label in sorted({os.path.basename(os.path.dirname(os.path.dirname(f))) for f in glob.glob(f"{ROOT}/train_next/*/p*/curves.csv")}):
        cur = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/train_next/{label}/p*/curves.csv"))])
        last = int(cur["step"].max())
        none = cur[cur["cond"] == "none"].groupby(["start", "seed", "step"]).agg(none_loss=("val_loss", "mean"), none_score=("score", "mean")).reset_index()
        x = split_cond(cur[cur["cond"] != "none"]).merge(none, on=["start", "seed", "step"], how="left")
        x["extra"] = x["val_loss"] - x["none_loss"]; x["d_score"] = x["score"] - x["none_score"]
        run = x.groupby(["start", "seed", "dose", "group", "step"])[["extra", "d_score"]].mean().reset_index()
        def at(step, dose, group, col="extra"):
            q = run[(run["step"] == step) & (run["dose"] == dose) & (run["group"] == group)][col]
            return float(q.mean()) if len(q) else np.nan
        e = run[(run["step"] == last) & (run["dose"] == "1.0")].pivot_table(index=["start", "seed"], columns="group", values="extra")
        k = n = 0
        if {"self", "task"} <= set(e.columns):
            e = e.dropna(subset=["self", "task"]); k, n = int((e["self"] > e["task"]).sum()), len(e)
        fam = family_of(label)
        hi_starts = [st for st in run["start"].unique() if CFG.trained_lang(fam, int(str(st).rsplit("_p", 1)[1]) + 1) == "hi"]
        rh = run[run["start"].isin(hi_starts)]
        def at_hi(step, group, col="extra"):
            q = rh[(rh["step"] == step) & (rh["dose"] == "1.0") & (rh["group"] == group)][col]
            return float(q.mean()) if len(q) else np.nan
        put(label, hi_tr0_diff=at_hi(0, "self") - at_hi(0, "task"), hi_trN_self=at_hi(last, "self"), hi_trN_task=at_hi(last, "task"),
            hi_trN_diff=at_hi(last, "self") - at_hi(last, "task"), hi_chrf_end_diff=at_hi(last, "task", "d_score") - at_hi(last, "self", "d_score"))
        of = [f for f in sorted(glob.glob(f"{ROOT}/train_next/{label}/p*/others.csv"))]
        if of and not CFG.FAMILIES[fam]["only"]:
            o = pd.concat([pd.read_csv(f) for f in of])
            o = o[[l == CFG.trained_lang(fam, int(str(st).rsplit("_p", 1)[1])) for l, st in zip(o["lang"], o["start"])]]      # previous language only
            base = o[o["cond"] == "none"].groupby(["start", "seed"])["after"].mean().rename("none_after").reset_index()
            o = split_cond(o[o["cond"] != "none"]).merge(base, on=["start", "seed"], how="left")
            o["lost_now"] = o["before"] - o["intact"]                 # right after the lesion, before any training
            o["extra_forget"] = o["after"] - o["none_after"]          # at the end: forgotten MORE than the untouched model forgot
            o = o[o["dose"] == "1.0"]
            def stats_for(q, pre):
                w = q.pivot_table(index=["start", "seed"], columns="group", values=["lost_now", "extra_forget"])
                if ("extra_forget", "self") in w.columns and ("extra_forget", "task") in w.columns and len(w):
                    put(label, **{f"{pre}_now_self": w[("lost_now", "self")].mean(), f"{pre}_now_task": w[("lost_now", "task")].mean(),
                                  f"{pre}_end_self": w[("extra_forget", "self")].mean(), f"{pre}_end_task": w[("extra_forget", "task")].mean(),
                                  f"{pre}_end_diff": (w[("extra_forget", "self")] - w[("extra_forget", "task")]).mean(),
                                  f"{pre}_k": int((w[("extra_forget", "self")] > w[("extra_forget", "task")]).sum()), f"{pre}_n": len(w)})
            stats_for(o, "prev_tr")
            stats_for(o[o["lang"] == "hi"], "hi_prev_tr")             # Hindi as the previous language (starts where Hindi was just trained)
        put(label, tr0_self=at(0, "1.0", "self"), tr0_task=at(0, "1.0", "task"), trN_self=at(last, "1.0", "self"), trN_task=at(last, "1.0", "task"),
            trN_diff=at(last, "1.0", "self") - at(last, "1.0", "task"), trN_k=k, trN_n=n, trN_step=last,
            trN_self05=at(last, "0.5", "self"), trN_look05=at(last, "0.5", "look"),
            chrf_end_self=at(last, "1.0", "self", "d_score"), chrf_end_task=at(last, "1.0", "task", "d_score"))

# =============================================================================================== 5. lesion, then a held-out language
def steps_to(steps, loss, target):
    """first training step at which the loss is at or below target (straight line between evaluation points). nan if never."""
    steps, loss = np.asarray(steps, float), np.asarray(loss, float)
    o = np.argsort(steps); steps, loss = steps[o], loss[o]
    for i in range(len(steps)):
        if loss[i] <= target:
            if i == 0 or loss[i - 1] <= loss[i]:
                return float(steps[i])
            return float(steps[i - 1] + (steps[i] - steps[i - 1]) * (loss[i - 1] - target) / (loss[i - 1] - loss[i]))
    return np.nan

FRESH = []
def part_heldout():
    for label in sorted({os.path.basename(os.path.dirname(os.path.dirname(f))) for f in glob.glob(f"{ROOT}/train_heldout/*/p*/curves.csv")}):
        cur = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/train_heldout/{label}/p*/curves.csv"))])
        last = int(cur["step"].max())
        if label.startswith("fresh"):
            e = cur[cur["step"] == last]
            FRESH.append({"untrained model": label, "loss at the end": e["val_loss"].mean(), "chrF at the end": e["score"].mean(), "repeats": e["seed"].nunique(), "steps": last})
            continue
        none = cur[cur["cond"] == "none"][["start", "seed", "step", "val_loss", "score"]].rename(columns={"val_loss": "none_loss", "score": "none_score"})
        x = split_cond(cur[cur["cond"] != "none"]).merge(none, on=["start", "seed", "step"], how="left")
        x = x[x["dose"] == "1.0"]
        x["extra"] = x["val_loss"] - x["none_loss"]
        rows = []
        for (st, sd, grp), q in x.groupby(["start", "seed", "group"]):
            n0 = none[(none["start"] == st) & (none["seed"] == sd)]
            tgt = n0[n0["step"] == CFG.HELDOUT_REACH]["none_loss"]
            reach = steps_to(q["step"], q["val_loss"], float(tgt.iloc[0])) if len(tgt) else np.nan
            late = q[q["step"] >= 100] if (q["step"] >= 100).any() else q
            end = q[q["step"] == last]
            rows.append({"start": st, "seed": sd, "group": grp, "avg": late["extra"].mean(), "end": end["extra"].mean(),
                         "steps": min(reach, last) if np.isfinite(reach) else (float(last) if len(tgt) else np.nan),
                         "chrf": (end["score"] - end["none_score"]).mean()})
        w = pd.DataFrame(rows).pivot_table(index=["start", "seed"], columns="group", values=["avg", "end", "steps", "chrf"])
        if not (("avg", "self") in w.columns and ("avg", "task") in w.columns):
            continue
        g = lambda a, b: w[(a, b)].mean() if (a, b) in w.columns else np.nan
        put(label, ho_avg_self=g("avg", "self"), ho_avg_task=g("avg", "task"), ho_avg_diff=g("avg", "self") - g("avg", "task"),
            ho_end_self=g("end", "self"), ho_end_task=g("end", "task"), ho_end_diff=g("end", "self") - g("end", "task"),
            ho_steps_self=g("steps", "self"), ho_steps_task=g("steps", "task"), ho_steps_diff=g("steps", "self") - g("steps", "task"),
            ho_chrf_diff=g("chrf", "task") - g("chrf", "self"),
            ho_k=int((w[("avg", "self")] > w[("avg", "task")]).sum()), ho_n=len(w), ho_last_step=last)

# =============================================================================================== tables, tests, figures
KEY = [("ho_avg_diff", "IMPORTANCE for a NEW language (Indonesian), lesion + training: extra loss while learning it, self minus task"),
       ("ho_steps_diff", "IMPORTANCE for a NEW language (Indonesian): extra training steps needed to catch up, self minus task"),
       ("ho_end_diff", "IMPORTANCE for a NEW language (Indonesian): extra loss at the end of training, self minus task"),
       ("hi_les_diff", "IMPORTANCE on Hindi, lesion without training: loss rise, self minus task"),
       ("hi_chrf_diff", "IMPORTANCE on Hindi, lesion without training: chrF points lost, self minus task"),
       ("hi_trN_diff", "IMPORTANCE on Hindi, lesion + 1,000 steps of Hindi: extra loss at the end, self minus task"),
       ("hi_prev_les_diff", "PREVIOUS language = Hindi, lesion without training: loss rise, self minus task (continual runs only)"),
       ("hi_prev_tr_end_diff", "PREVIOUS language = Hindi, lesion + 1,000 steps of the next language: extra forgetting, self minus task (continual runs only)"),
       ("gap_matched", "persistence gap, self minus task (task neurons of equal activity)"),
       ("les_diff", "lesion, no training, language just trained (a different language per kind of run): self minus task"),
       ("trN_diff", "lesion + training on the next language (differs per kind of run): self minus task")]
PAIRED = [("pers_self", "pers_task", "persistence: self vs task"), ("les_self", "les_task", "lesion, no training: self vs task"),
          ("trN_self", "trN_task", "lesion + training, at the end: self vs task")]

def tests(T):
    from scipy import stats
    rows = []
    for fam in CFG.FAMILY_ORDER:
        q = T[T["family"] == fam]
        for a, b, name in PAIRED:
            if a in q and b in q:
                d = q[[a, b]].dropna()
                p = stats.ttest_rel(d[a], d[b]).pvalue if len(d) >= 3 else np.nan
                rows.append({"test": name, "groups": fam, "n_models": len(d), "mean_1": d[a].mean() if len(d) else np.nan, "mean_2": d[b].mean() if len(d) else np.nan,
                             "models_where_self_is_higher": f"{int((d[a] > d[b]).sum())}/{len(d)}", "p": p, "kind": "paired t-test over models"})
    for col, name in KEY:
        if col not in T:
            continue
        for i, fa in enumerate(CFG.FAMILY_ORDER):
            for fb in CFG.FAMILY_ORDER[i + 1:]:
                a, b = T[T["family"] == fa][col].dropna(), T[T["family"] == fb][col].dropna()
                if not len(a) or not len(b):
                    continue
                p = stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) >= 2 and len(b) >= 2 else np.nan
                rows.append({"test": name, "groups": f"{fa} vs {fb}", "n_models": f"{len(a)} vs {len(b)}", "mean_1": a.mean() if len(a) else np.nan,
                             "mean_2": b.mean() if len(b) else np.nan, "models_where_self_is_higher": "", "p": p, "kind": "Welch t-test between kinds of run"})
    order = {f: i for i, f in enumerate(CFG.FAMILY_ORDER)}
    for col, name in KEY:                                # does the number RISE from Hindi only to related to distant?
        if col in T:
            q = T[[col, "family"]].dropna()
            if q["family"].nunique() >= 3 and len(q) >= 5:
                r = stats.spearmanr([order[f] for f in q["family"]], q[col])
                rows.append({"test": name, "groups": "trend: " + " < ".join(CFG.FAMILY_ORDER), "n_models": len(q), "mean_1": r.statistic, "mean_2": np.nan,
                             "models_where_self_is_higher": "", "p": r.pvalue, "kind": "rank correlation with language distance (mean_1 = correlation; +1 = always rising)"})
    return pd.DataFrame(rows)

COL = {"hindi": "#2A6F97", "related": "#00918F", "distant": "#C8102E"}      # the colours these runs have on the slides

def fig_trend(T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fams = [f for f in CFG.FAMILY_ORDER if f in set(T["family"])]
    panels = [("ho_avg_diff", "New language (Indonesian): lesion, then learn it", "self minus task\n(extra Indonesian loss while learning)"),
              ("ho_steps_diff", "New language (Indonesian): time to catch up", "self minus task\n(extra training steps to reach the untouched model)"),
              ("hi_les_diff", "Hindi: lesion, no training", "self minus task\n(loss rise on Hindi)"),
              ("hi_chrf_diff", "Hindi: lesion, no training", "self minus task\n(chrF points lost on Hindi)"),
              ("hi_trN_diff", "Hindi: lesion, then 1,000 steps of Hindi", "self minus task\n(extra Hindi loss still there at the end)"),
              ("hi_prev_les_diff", "Previous language (Hindi): lesion, no training", "self minus task\n(loss rise on Hindi, one phase after Hindi)"),
              ("hi_prev_tr_end_diff", "Previous language (Hindi): lesion, then the next language", "self minus task\n(extra Hindi forgetting at the end)"),
              ("gap_matched", "Persistence (existence, not importance)", "self minus task\n(persistence from one phase to the next)")]
    panels = [p for p in panels if p[0] in T and T[p[0]].notna().any()]
    if not panels or not fams:
        return
    ncol = min(3, len(panels)); nrow = -(-len(panels) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.3 * nrow + 0.5), squeeze=False)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (col, title, yl) in zip(axs.ravel(), panels):
        for k, fam in enumerate(fams):
            q = T[T["family"] == fam].dropna(subset=[col])
            if not len(q):
                continue
            jit = np.linspace(-0.12, 0.12, len(q)) if len(q) > 1 else [0.0]
            for (_, r), dx in zip(q.iterrows(), jit):
                ax.scatter(k + dx, r[col], s=46, zorder=3, linewidths=1.6, edgecolors=COL[fam], facecolors=COL[fam] if r["new"] else "white")
            mu = q[col].mean()
            ax.plot([k - 0.24, k + 0.24], [mu, mu], color=COL[fam], lw=2, zorder=2)
            ax.annotate(f"{mu:.3f}", (k + 0.27, mu), va="center", fontsize=9, color="#333333")
        ax.axhline(0, color="#888888", lw=0.8, ls=":")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels([CFG.FAMILIES[f]["label"] for f in fams], fontsize=8.5)
        ax.set_xlim(-0.5, len(fams) - 0.3)
        ax.set_title(title, fontsize=10.5, loc="left"); ax.set_ylabel(yl, fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "One dot = one model (one seed). Filled = new seeds, hollow = the old seed-0 runs. Line and number = average of the models. "
             "Above the dotted zero line = higher for self neurons than for task neurons.\n"
             "Every importance panel uses ONE language for all three kinds of run (Indonesian, which no model has seen, or Hindi, which all have), "
             "the same number of self and task neurons lesioned, all 8 layers, and every checkpoint from phase 4 on.\n"
             "Previous-language panels: a Hindi-only model has no previous language, so they show the two continual kinds of run.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/trend.png", dpi=170); plt.close(fig)

def fig_gap_by_phase(P):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = P.groupby(["family", "label", "phase_a"])["gap_matched"].mean().reset_index()
    fams = [f for f in CFG.FAMILY_ORDER if f in set(g["family"])]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for fam in fams:
        q = g[g["family"] == fam]
        for _, x in q.groupby("label"):
            ax.plot(x["phase_a"], x["gap_matched"], color=COL[fam], lw=0.8, alpha=0.35)
        m = q.groupby("phase_a")["gap_matched"].mean()
        ax.plot(m.index, m.values, color=COL[fam], lw=2, marker="o", ms=4, label=CFG.FAMILIES[fam]["label"].replace("\n", " "))
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("phase p (persistence is measured from phase p to phase p + 1)"); ax.set_ylabel("persistence gap: self minus task\n(average of the 8 layers)")
    ax.set_title("Persistence gap at every phase change. Thin lines = single models, thick = their average", fontsize=10, loc="left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/gap_by_phase.png", dpi=170); plt.close(fig)

# =============================================================================================== main
def guarded(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        say(f"({name}: left out, {type(e).__name__}: {e})")
        return None

say(f"MULTI-SEED PILOT (v14). Tables in analysis/{args.root}/. Self = largest co-firing group per layer, |correlation| >= 0.2, FLORES dev.")
say(f"Lesion tests are count-matched (same number of self and task neurons per layer) and use phases {CFG.LESION_PHASES}; "
    f"persistence uses the phase changes {CFG.HEADLINE_FROM[0]}->{CFG.HEADLINE_FROM[0] + 1} to {CFG.HEADLINE_FROM[-1]}->{CFG.HEADLINE_FROM[-1] + 1}.")
say("s1, s2, s3 = new seeds. old0, old1 = the earlier single runs (same recipe), analysed with the same code.\n")
guarded("training table", part_training)
guarded("self size", part_sizes)
P = guarded("persistence", part_persistence)
guarded("lesion without training", part_static)
guarded("lesion with training", part_train)
guarded("lesion with a held-out language", part_heldout)

T = pd.DataFrame([dict(label=k, **v) for k, v in M.items()]) if M else pd.DataFrame(columns=["label"])
T["family"] = [family_of(l) for l in T["label"]]; T["tag"] = [l.rsplit("_", 1)[1] for l in T["label"]]
T["new"] = [not t.startswith("old") for t in T["tag"]]
T["_o"] = [CFG.FAMILY_ORDER.index(f) if f in CFG.FAMILY_ORDER else 9 for f in T["family"]]
T = T.sort_values(["_o", "tag"]).drop(columns="_o").reset_index(drop=True)
T.to_csv(f"{OUT}/per_model.csv", index=False)

def show(title, cols, note=""):
    cols = [c for c in cols if c in T.columns and T[c].notna().any()]
    if not cols or not len(T):
        return
    say(f"=== {title} ===")
    if note:
        say(note)
    say(T[["family", "tag"] + cols].round(4).to_string(index=False))
    say("average over models (mean, standard deviation, number of models):")
    say(T.groupby("family", sort=False)[cols].agg(["mean", "std", "count"]).round(4).to_string()); say()

show("1. Self size (% of a layer's working neurons, average of the 8 layers)", ["self_pct", "self_pct_enc", "self_pct_dec"])
show("2. Persistence from one phase to the next (1 = unchanged)", ["pers_self", "pers_task", "gap", "gap_matched", "gap_matched_enc", "gap_matched_dec", "n_transitions"],
     "gap = self minus task. gap_matched = the same, comparing only neurons of equal average activity. enc / dec = the 4 encoder / 4 decoder layers.")
show("3. Lesion without training: loss rise on the language just trained, all 8 layers", ["les_self", "les_task", "les_diff", "les_ratio", "les_k", "les_n", "chrf_drop_self", "chrf_drop_task", "les_self05", "les_look05"],
     "les_k of les_n = in how many (checkpoint x draw) comparisons the self lesion was worse. chrf_drop = chrF points lost. "
     "les_self05 / les_look05 = half the count: self vs task neurons that look like the self (same activity, firing rate, weight size).")
show("3b. The same with 'silence' (each neuron held at its own average) in place of re-initialising", ["sil_self", "sil_task", "sil_diff", "sil_ratio", "sil_k", "sil_n"])
show("4. Lesion, then training on the next language: held-out loss minus the untouched model's", ["tr0_self", "tr0_task", "trN_self", "trN_task", "trN_diff", "trN_k", "trN_n", "trN_self05", "trN_look05", "chrf_end_self", "chrf_end_task", "trN_step"],
     "tr0 = right after the lesion. trN = at the last training step (trN_step). trN_k of trN_n = in how many (start x repeat) runs the self lesion was still worse. chrf_end = chrF vs the untouched model.")

show("5a. THE TREND QUESTION: importance of the self for learning a NEW language (Indonesian), which no model has seen",
     ["ho_avg_self", "ho_avg_task", "ho_avg_diff", "ho_steps_self", "ho_steps_task", "ho_steps_diff", "ho_end_self", "ho_end_task", "ho_end_diff", "ho_chrf_diff", "ho_k", "ho_n", "ho_last_step"],
     f"Lesion, then {CFG.HELDOUT_STEPS:,} training steps on Indonesian, from every checkpoint. All numbers are relative to the untouched model from the same checkpoint. "
     f"ho_avg = extra Indonesian loss, average of the evaluations from step 100 on. ho_steps = training steps needed to reach the loss the untouched model has at step {CFG.HELDOUT_REACH} "
     "(so the untouched model's own value is that step count; capped at the last step). ho_end = extra loss at the last step. ho_chrf_diff = chrF of the task-lesioned model "
     "minus the self-lesioned one at the end. ho_k of ho_n = checkpoints where the self lesion was worse (ho_avg). diff = self minus task.")
if FRESH:
    say("for reference, an UNTRAINED model learning Indonesian with the same settings:"); say(pd.DataFrame(FRESH).round(3).to_string(index=False)); say()
show("5. THE TREND QUESTION: importance of the self on HINDI, the one language all three kinds of run share",
     ["hi_les_self", "hi_les_task", "hi_les_diff", "hi_chrf_self", "hi_chrf_task", "hi_chrf_diff", "hi_les_n", "hi_trN_self", "hi_trN_task", "hi_trN_diff", "hi_chrf_end_diff"],
     "Same language and same test sentences everywhere, so the three kinds of run can be compared directly. hi_les = loss rise on Hindi after the lesion "
     "(checkpoints where Hindi was just trained). hi_chrf = chrF points lost. hi_trN = extra Hindi loss after the lesion and 1,000 steps of Hindi. diff = self minus task.")
show("5b. The PREVIOUS language (the one trained one phase earlier). Continual runs only: a Hindi-only model has no previous language",
     ["prev_les_self", "prev_les_task", "prev_les_diff", "prev_chrf_diff", "prev_les_k", "prev_les_n", "hi_prev_les_self", "hi_prev_les_task", "hi_prev_les_diff", "hi_prev_chrf_diff"],
     "Lesion without training. prev_les = loss rise on the previous language, all 8 layers; diff = self minus task; prev_chrf_diff = chrF points lost, self minus task. "
     "hi_prev = the same when the previous language is Hindi (phase 14), so related and distant can be compared on one language.")
show("5c. The PREVIOUS language while the next one is learned (lesion, then 1,000 steps on the next language)",
     ["prev_tr_now_self", "prev_tr_now_task", "prev_tr_end_self", "prev_tr_end_task", "prev_tr_end_diff", "prev_tr_k", "prev_tr_n",
      "hi_prev_tr_end_self", "hi_prev_tr_end_task", "hi_prev_tr_end_diff"],
     "now = loss rise on the previous language right after the lesion. end = how much MORE of it is forgotten after the 1,000 steps than the untouched model forgets. "
     "prev_tr_k of prev_tr_n = runs where the self lesion forgot more. hi_prev = the starts where the previous language is Hindi (phase 13).")
by = None
if len(T):
    num = [c for c in T.columns if c not in ("label", "family", "tag", "new") and pd.api.types.is_numeric_dtype(T[c])]
    by = T.groupby("family", sort=False)[num].agg(["mean", "std", "count"]); by.columns = [f"{a}_{b}" for a, b in by.columns]
    by.reset_index().to_csv(f"{OUT}/by_family.csv", index=False)
    tt = guarded("tests", tests, T)
    if tt is not None and len(tt):
        tt.to_csv(f"{OUT}/tests.csv", index=False)
        say("=== 6. Tests. One data point = one MODEL (not one neuron, not one checkpoint), so n is small and p-values are rough ===")
        say(tt.round(4).to_string(index=False)); say()
    say("=== 7. The trend (average over models; left to right = more different languages) ===")
    for col, name in KEY:
        if col in T:
            m = T.groupby("family")[col].agg(["mean", "std", "count"])
            say(f"{name}:  " + "  |  ".join(f"{f}: {m.loc[f, 'mean']:.3f} (sd {m.loc[f, 'std']:.3f}, {int(m.loc[f, 'count'])} models)" for f in CFG.FAMILY_ORDER if f in m.index))
    guarded("trend figure", fig_trend, T)
if P is not None:
    guarded("gap-by-phase figure", fig_gap_by_phase, P)
with open(f"{OUT}/report.txt", "w") as f:
    f.write("\n".join(REPORT) + "\n")
print("report in", OUT)
