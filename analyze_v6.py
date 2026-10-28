"""analyze_v6.py: CPU analysis of all runs (references + v6 fix runs + v6 Q2 push runs), from analysis/v5/<run>/pXXX.npz.

Run AFTER analysis_v5_compute.py has produced the npz files (the reference runs' npz are already in the repo).
Missing runs/phases are skipped, so it works on partial runs.

IMPROVED versions of the earlier numbers (same definitions as the deck, plus the fixes that the quick checks showed we need):
  1. fixed tau = 0.2 for every MLP layer                         (what we used in v5 at the end)
  2. SIZE-MATCHED self: per checkpoint and layer, the tau at which the biggest group covers ~40% of the layer.
     Removes "a bigger self scores better" (H3) from every comparison.
  3. EXCESS relative gap = relative gap minus what pure noise of the SAME size gives on the SAME kind of network
     (noise curves from quick checks). > 0: the self is protected more than by chance; < 0: hit harder than by chance.
  4. 95% bootstrap intervals over switches for every window mean.
  5. encoder vs decoder split, and membership (share of the self that stays, IoU).

Outputs in analysis/v6/:
  switches.csv, health_from_npz.csv, summary.txt (all tables), and figures
  A_q2_movement_vs_gap.png   relative gap vs movement: switches, drift, push runs, per-network noise curves
  E_excess_gap.png           excess relative gap per run and window (with 95% intervals), fixed tau and size-matched
  F_enc_vs_dec.png           relative gap and stays, encoder vs decoder
  B_fix_training.png, C_health.png, D_q2_training.png   training curves and health

Usage: python3 analyze_v6.py     (env TAU=0.2, SIZE=0.4 to change)
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
TAU = float(os.environ.get("TAU", 0.2))
SIZE = float(os.environ.get("SIZE", 0.4))
BASE = "analysis/v5"; OUT = "analysis/v6"; os.makedirs(OUT, exist_ok=True)
MLP = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- runs
REF = {"continual (ref)": "v4b_continual_s0", "Hindi only (ref)": "v4b_only_hi_s0"}
FIX = sorted(os.path.basename(p) for p in glob.glob(f"{BASE}/v6_cont_*"))
Q2 = sorted(os.path.basename(p) for p in glob.glob(f"{BASE}/v6_hi_*"))
RUNS = {**REF, **{r: r for r in FIX + Q2}}
RUNS = {k: v for k, v in RUNS.items() if os.path.isdir(f"{BASE}/{v}")}
print("runs found:", list(RUNS))

def group_of(name):
    return {"continual (ref)": "continual", "Hindi only (ref)": "hindi"}.get(name, "fix" if name.startswith("v6_cont") else "push")
NET = {"continual": "continual", "fix": "continual", "hindi": "hindi", "push": "hindi"}   # which noise curve applies

# ---------------------------------------------------------------- the self, two ways
def largest(n, i, j, alive):
    _, lab = connected_components(csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    lab = np.where(alive, lab, -1)
    if (lab >= 0).sum() == 0:
        return np.zeros(n, bool)
    return lab == np.bincount(lab[lab >= 0]).argmax()

def self_fixed(d, pt, tau=TAU):
    alive = d[f"{pt}/alive"]; keep = d[f"{pt}/mst_w"].astype(np.float32) >= tau
    return largest(len(alive), d[f"{pt}/mst_i"].astype(np.int64)[keep], d[f"{pt}/mst_j"].astype(np.int64)[keep], alive)

def self_sized(d, pt, frac=SIZE):
    """Add spanning-tree edges strongest first (union-find); stop when the biggest group reaches frac of the alive neurons.
    Returns the mask and the tau where that happened."""
    alive = d[f"{pt}/alive"]; n = len(alive); target = frac * alive.sum()
    w = d[f"{pt}/mst_w"].astype(np.float32); order = np.argsort(-w)
    I, J = d[f"{pt}/mst_i"].astype(np.int64)[order], d[f"{pt}/mst_j"].astype(np.int64)[order]
    parent, size = np.arange(n), np.ones(n, int)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    k_stop = len(I)
    for k in range(len(I)):
        a, b = find(I[k]), find(J[k])
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a; size[a] += size[b]
            if size[a] >= target:
                k_stop = k + 1; break
    tau = float(w[order][k_stop - 1]) if k_stop else 1.0
    return largest(n, I[:k_stop], J[:k_stop], alive), tau

# ---------------------------------------------------------------- per-switch rows
rows, health = [], []
for name, run in RUNS.items():
    prev = None
    for f in sorted(glob.glob(f"{BASE}/{run}/p*.npz")):
        d = dict(np.load(f)); ph = int(d["phase"])
        health.append({"run": name, "group": group_of(name), "phase": ph,
                       "rarely_active": np.mean([float((d[f"{pt}/frac_pos"].astype(np.float32) < 0.01).mean()) for pt in MLP]),
                       "dead": np.mean([float((~d[f"{pt}/alive"]).mean()) for pt in MLP]),
                       "rank_enc1": float(d["enc1.mlp/eff_rank"]), "rank_dec1": float(d["dec1.mlp/eff_rank"])})
        if prev is not None and prev[0] == ph - 1 and ph - 1 >= 1:
            dp = prev[1]
            for pt in MLP:
                if f"{pt}/perm" not in d:
                    continue
                perm = d[f"{pt}/perm"].astype(np.int64); valid = d[f"{pt}/valid"]
                change = 100 * (1 - (d[f"{pt}/act"].astype(np.float32) + d[f"{pt}/conn"].astype(np.float32)) / 2)
                for mode in ["fixed", "sized"]:
                    if mode == "fixed":
                        sa, sb, tau_b = self_fixed(dp, pt), self_fixed(d, pt), TAU
                    else:
                        (sa, _), (sb, tau_b) = self_sized(dp, pt), self_sized(d, pt)
                    sb = sb[perm]                                   # partner's membership after (source index)
                    ins, intask = valid & sb, valid & ~sb
                    tc = change[intask].mean() if intask.any() else np.nan
                    sc = change[ins].mean() if ins.any() else np.nan
                    union = (sa | sb).sum()
                    rows.append({"run": name, "group": group_of(name), "mode": mode, "from_phase": ph - 1, "to_phase": ph,
                                 "from_lang": str(dp["lang"]), "to_lang": str(d["lang"]), "point": pt,
                                 "side": pt[:3], "tau": tau_b, "task_change": tc, "self_change": sc, "gap": tc - sc,
                                 "self_frac": sb[valid].mean() if valid.any() else np.nan,
                                 "stays": (sa & sb).sum() / max(sa.sum(), 1), "iou": (sa & sb).sum() / max(union, 1)})
        prev = (ph, d)
    print(f"  {name}: done", flush=True)
S = pd.DataFrame(rows); H = pd.DataFrame(health)
S.to_csv(f"{OUT}/switches.csv", index=False); H.to_csv(f"{OUT}/health_from_npz.csv", index=False)
if S.empty:
    sys.exit("no switches found yet: run analysis_v5_compute.py on the runs first")

# ---------------------------------------------------------------- noise baselines (quick checks, tau 0.2, fixed self)
NOISE = {}
qc = "analysis/v6/quick_checks/results.csv"
Q = pd.read_csv(qc) if os.path.exists(qc) else pd.DataFrame()
if len(Q):
    for net, pref in [("continual", "continual"), ("hindi", "only")]:
        q = Q[(Q.check == "noise") & Q.label.str.startswith(pref)]
        m = q.groupby("label").agg(task=("task_change", "mean"), gap=("gap", "mean")).reset_index().sort_values("task")
        NOISE[net] = (m.task.values, 100 * m.gap.values / m.task.values)
def baseline(net, task):
    if net not in NOISE:
        return np.nan
    x, y = NOISE[net]
    return float(np.interp(task, x, y))                      # flat outside the measured range

# per switch = mean over layers (one point per switch); per side too
def per_switch(df, by=()):
    g = df.groupby(["run", "group", "mode", "to_phase", *by]).agg(task=("task_change", "mean"), self_ch=("self_change", "mean"),
            gap=("gap", "mean"), size=("self_frac", "mean"), stays=("stays", "mean"), iou=("iou", "mean"),
            tau=("tau", "mean")).reset_index()
    g["rel_gap"] = 100 * g.gap / g.task
    g["excess"] = [r.rel_gap - baseline(NET[r.group], r.task) for r in g.itertuples()]
    return g
P_ = per_switch(S); PS = per_switch(S, ("side",))

def boot(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))

WINDOWS = {"2-60": (2, 60), "28-60 (cycles 10-20)": (28, 60), "31-60": (31, 60), "61-70": (61, 70), "61-90": (61, 90)}
lines = [f"self = biggest group of MLP neurons. fixed: tau {TAU} for every layer. sized: per checkpoint, tau where the biggest group = {SIZE:.0%} of the layer.",
         "One number per switch = mean over the 8 MLP layers; table = mean over switches in the window (switch counted by its to_phase).",
         "rel gap = (task - self)/task, from the window means. excess = rel gap minus pure noise of the same size on the same kind of network",
         "(continual curve for continual + fix runs, Hindi curve for Hindi + push runs). [..] = 95% bootstrap interval over switches.\n"]
summ = []
for mode in ["fixed", "sized"]:
    lines.append(f"===== self: {mode} =====")
    lines.append(f"{'run':<30}{'window':>22}{'n':>4}{'task':>7}{'self':>7}{'rel gap':>9}{'[95%]':>15}{'excess':>8}{'[95%]':>15}{'size':>6}{'stays':>7}{'IoU':>6}")
    for name in RUNS:
        for w, (a, b) in WINDOWS.items():
            q = P_[(P_.run == name) & (P_["mode"] == mode) & P_.to_phase.between(a, b)]
            if len(q) < 2:
                continue
            rg = 100 * q.gap.mean() / q.task.mean(); ci = boot(q.rel_gap); ex = q.excess.mean(); ce = boot(q.excess)
            summ.append({"run": name, "group": q.group.iloc[0], "mode": mode, "window": w, "n": len(q), "task": q.task.mean(),
                         "rel_gap": rg, "lo": ci[0], "hi": ci[1], "excess": ex, "ex_lo": ce[0], "ex_hi": ce[1],
                         "size": q["size"].mean(), "stays": q.stays.mean(), "iou": q.iou.mean()})
            lines.append(f"{name:<30}{w:>22}{len(q):>4}{q.task.mean():7.1f}{q.self_ch.mean():7.1f}{rg:8.1f}%"
                         f"   [{ci[0]:5.1f},{ci[1]:5.1f}]{ex:7.1f} [{ce[0]:5.1f},{ce[1]:5.1f}]"
                         f"{100 * q['size'].mean():5.0f}%{100 * q.stays.mean():6.0f}%{100 * q.iou.mean():5.0f}%")
    lines.append("")
lines.append("===== encoder vs decoder (fixed tau, window 2-60 for references, all phases for v6 runs) =====")
for name in RUNS:
    for side in ["enc", "dec"]:
        q = PS[(PS.run == name) & (PS["mode"] == "fixed") & (PS.side == side) & ((PS.to_phase <= 60) | (PS.group.isin(["fix", "push"])))]
        if len(q) >= 2:
            lines.append(f"{name:<30}{side:>5}  rel gap {100 * q.gap.mean() / q.task.mean():5.1f}%  task {q.task.mean():5.1f}  "
                         f"stays {100 * q.stays.mean():4.0f}%  size {100 * q['size'].mean():4.0f}%")
lines.append("\nsized mode: mean tau used per run = " + ", ".join(
    f"{n}: {P_[(P_.run == n) & (P_['mode'] == 'sized')].tau.mean():.2f}" for n in RUNS))
SUM = pd.DataFrame(summ); SUM.to_csv(f"{OUT}/summary_table.csv", index=False)
open(f"{OUT}/summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))

COL = {"continual": "#c0392b", "hindi": "#2c7fb8", "fix": "#e67e22", "push": "#2ca25f"}
LAB = {"hindi": "Hindi only, phase to phase", "continual": "continual, at switches",
       "push": "Hindi only + push each boundary (Q2)", "fix": "continual + per-step noise (fix)"}

# ---------------------------------------------------------------- A: the Q2 figure (fixed tau, like the quick checks)
fig, ax = plt.subplots(figsize=(8.5, 5.8))
for g in ["hindi", "continual", "push", "fix"]:
    q = P_[(P_.group == g) & (P_["mode"] == "fixed") & (P_.to_phase.between(2, 60) if g in ("hindi", "continual") else P_.to_phase > 0)]
    for k, (run, qq) in enumerate(q.groupby("run")):
        ax.scatter(qq.task, qq.rel_gap, s=12, alpha=.35, color=COL[g], label=LAB[g] if k == 0 else None)
        mx, my = qq.task.mean(), 100 * qq.gap.mean() / qq.task.mean(); lo, hi = boot(qq.rel_gap)
        ax.errorbar(mx, my, yerr=[[my - lo], [hi - my]], fmt="*", ms=15, color=COL[g], mec="k", zorder=5, capsize=3)
        if g in ("push", "fix"):
            ax.annotate(run.replace("v6_", ""), (mx, my), fontsize=7, xytext=(6, 6), textcoords="offset points")
if len(Q):
    for sel, mk, ls, c, lab in [((Q.check == "noise") & Q.label.str.startswith("continual"), "D", "--", COL["continual"], "pure noise on continual p45"),
                                ((Q.check == "noise") & Q.label.str.startswith("only"), "D", "--", COL["hindi"], "pure noise on Hindi-only p45"),
                                (Q.check == "drift_hi", "s", ":", COL["hindi"], "Hindi only, far-apart phases"),
                                (Q.check == "drift_cont", "^", ":", COL["continual"], "continual, same language far apart")]:
        q = Q[sel]
        if len(q):
            m = q.groupby("label").agg(task=("task_change", "mean"), gap=("gap", "mean")).reset_index().sort_values("task")
            ax.plot(m.task, 100 * m.gap / m.task, marker=mk, color=c, mfc="none", lw=1, ms=6, label=lab, ls=ls)
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel("task-neuron change at the transition (how much the network moved)")
ax.set_ylabel("relative gap = (task - self) / task   [%]")
ax.set_title(f"Q2: at the same movement, is the self protected MORE under switching? (tau {TAU})\n"
             "stars = run means with 95% interval; dashed = what pure noise gives on that network", fontsize=10)
ax.legend(fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(f"{OUT}/A_q2_movement_vs_gap.png", dpi=150)

# ---------------------------------------------------------------- E: excess gap per run and window
if len(SUM):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, mode in zip(axs, ["fixed", "sized"]):
        q = SUM[(SUM["mode"] == mode) & SUM.window.isin(["2-60", "31-60", "61-90"])].reset_index(drop=True)
        q = q[~((q.group.isin(["continual", "hindi"])) & (q.window == "61-90") & (q.group == "continual"))]  # ref continual breaks after 60
        x = np.arange(len(q))
        ax.bar(x, q.excess, color=[COL[g] for g in q.group], alpha=.85)
        ax.errorbar(x, q.excess, yerr=[q.excess - q.ex_lo, q.ex_hi - q.excess], fmt="none", color="k", capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([f"{r.run.replace('v6_', '')}\n{r.window}" for r in q.itertuples()], fontsize=7, rotation=30, ha="right")
        ax.axhline(0, color="k", lw=.8); ax.grid(axis="y", alpha=.3)
        ax.set_title(f"excess relative gap over pure noise of the same size ({'tau ' + str(TAU) if mode == 'fixed' else 'size-matched self, ' + format(SIZE, '.0%')})", fontsize=9)
        ax.set_ylabel("points of relative gap above noise")
    fig.suptitle("Above 0: the self is protected more than random disturbance would give. Below 0: hit harder than chance.", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/E_excess_gap.png", dpi=150)

# ---------------------------------------------------------------- F: encoder vs decoder
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
names = list(RUNS); x = np.arange(len(names))
for k, side in enumerate(["enc", "dec"]):
    rg, st = [], []
    for n in names:
        q = PS[(PS.run == n) & (PS["mode"] == "fixed") & (PS.side == side) & ((PS.to_phase <= 60) | (PS.group.isin(["fix", "push"])))]
        rg.append(100 * q.gap.mean() / q.task.mean() if len(q) else np.nan); st.append(100 * q.stays.mean() if len(q) else np.nan)
    axs[0].bar(x + (k - .5) * .38, rg, .38, label={"enc": "encoder", "dec": "decoder"}[side])
    axs[1].bar(x + (k - .5) * .38, st, .38, label={"enc": "encoder", "dec": "decoder"}[side])
for ax, t in zip(axs, ["relative gap (%)", "share of the self that stays (%)"]):
    ax.set_xticks(x); ax.set_xticklabels([n.replace("v6_", "") for n in names], fontsize=7, rotation=25, ha="right")
    ax.set_title(t, fontsize=10); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
fig.suptitle(f"Encoder vs decoder (tau {TAU}; references phases 2-60)", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/F_enc_vs_dec.png", dpi=150)

# ---------------------------------------------------------------- training curves from runs/<run>/phases.jsonl (or the npz copy)
def phases(run):
    for f in [f"runs/{run}/phases.jsonl", f"{BASE}/{run}/phases.jsonl"]:
        if os.path.exists(f):
            break
    else:
        return pd.DataFrame()
    out = []
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        if "chrf_masked" not in r and "eval" in r:
            r = {**r, **r["eval"]}
        if "chrf_masked" in r:
            out.append({"phase": r["phase"], "lang": r.get("lang"), **{f"chrf_{l}": r["chrf_masked"].get(l) for l in ["hi", "zh", "ar"]},
                        **{f"val_{l}": r["val_masked"].get(l) for l in ["hi", "zh", "ar"]}})
    return pd.DataFrame(out)

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for name in [n for n in ["continual (ref)"] + FIX if n in RUNS]:
    T = phases(RUNS[name])
    if T.empty:
        continue
    T = T[T.phase >= 50]
    for ax, l in zip(axs, ["hi", "zh", "ar"]):
        tr = T[T.lang == l]
        ax.plot(tr.phase, tr[f"chrf_{l}"], marker="o", ms=3, label=name, lw=2 if "ref" in name else 1.2)
for ax, l in zip(axs, ["hi", "zh", "ar"]):
    ax.set_title(f"chrF {l}, right after training {l}"); ax.set_xlabel("phase"); ax.grid(alpha=.3)
axs[0].set_ylabel("chrF (script-restricted)"); axs[0].legend(fontsize=7)
fig.suptitle("Fix runs vs original continual: does per-step noise keep the model learning after phase 60?", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/B_fix_training.png", dpi=150)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for name in RUNS:
    h = H[H.run == name].sort_values("phase")
    if h.empty:
        continue
    kw = dict(lw=2 if "ref" in name else 1.2, label=name)
    axs[0].plot(h.phase, 100 * h.rarely_active, **kw); axs[1].plot(h.phase, h.rank_enc1, **kw); axs[2].plot(h.phase, h.rank_dec1, **kw)
for ax, t in zip(axs, ["rarely-active MLP neurons (% of layer, mean of 8)", "effective rank enc1 MLP", "effective rank dec1 MLP"]):
    ax.set_title(t, fontsize=9); ax.set_xlabel("phase"); ax.grid(alpha=.3)
axs[0].legend(fontsize=6); fig.suptitle("Health from the analysis pass (FLORES dev): same measure for every run", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/C_health.png", dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for name in [n for n in ["Hindi only (ref)"] + Q2 if n in RUNS]:
    T = phases(RUNS[name])
    if T.empty:
        continue
    T = T[T.phase >= 25]
    kw = dict(marker="o", ms=2.5, lw=2 if "ref" in name else 1.2, label=name)
    axs[0].plot(T.phase, T.chrf_hi, **kw); axs[1].plot(T.phase, T.val_hi, **kw)
axs[0].set_title("Hindi chrF"); axs[1].set_title("Hindi val loss")
for ax in axs:
    ax.set_xlabel("phase"); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); fig.suptitle("Q2 push runs: does the push cost skill once the phase has retrained?", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/D_q2_training.png", dpi=150)
print("figures ->", OUT)
