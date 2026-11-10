"""Slide-57 breakdown by model + self size related vs distant. CPU only, reads existing CSVs."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
D = "/home/claude/self-modelling-llm-cml/analysis/v8/"; OUT = D + "breakdown/"
rng = np.random.default_rng(0)
MODELS = [("continual", "Continual (hi→zh→ar, distant)", "*", "#4B1D6B"),
          ("hindi", "Hindi only", "x", "#B35806"),
          ("mixed", "Mixed (all 3 at once)", "_", "#2D7F5E"),
          ("related", "Continual (hi→gu→pa, related)", "D", "#1F5FA8")]
per = pd.concat([pd.read_csv(D + d + "/lesion_per_layer.csv") for d in ["lesion_distant", "lesion_distant_more", "lesion_indic"]])
per = per[(per.model != "push") & (per.ablation == "mean")]
allL = pd.concat([pd.read_csv(D + d + "/lesion_all_layers.csv") for d in ["lesion_distant", "lesion_distant_more", "lesion_indic"]])
allL = allL[allL.model != "push"]

def hboot(df, col, B=4000):
    """hierarchical bootstrap: resample checkpoints, then layers inside each checkpoint"""
    groups = [g[col].values for _, g in df.groupby("phase")]
    m = np.empty(B)
    for b in range(B):
        pick = rng.integers(len(groups), size=len(groups))
        m[b] = np.mean(np.concatenate([rng.choice(groups[i], len(groups[i])) for i in pick]))
    return np.percentile(m, [2.5, 97.5])

# one comparison = one checkpoint x one layer: average over 3 repeats and the eval languages
rows, summ = [], []
for dose in [0.05, 0.10, 0.20]:
    x = per[per.dose == dose].groupby(["model", "phase", "point", "group"]).rise.mean().unstack("group").reset_index()
    x["extra"] = x["self"] - x["matched"]; x["dose"] = dose; rows.append(x)
cmp_ = pd.concat(rows); cmp_.to_csv(OUT + "comparisons_per_ckpt_layer.csv", index=False)
for m, lab, *_ in MODELS:
    for dose in [0.05, 0.10, 0.20]:
        c = cmp_[(cmp_.model == m) & (cmp_.dose == dose)].dropna(subset=["extra"])
        lo, hi = hboot(c, "extra"); k = int((c.extra > 0).sum()); n = len(c)
        summ.append(dict(model=m, dose=dose, ckpts=c.phase.nunique(), comparisons=n, self_worse=k,
                         sign_p=binomtest(k, n, 0.5, alternative="greater").pvalue,
                         self=c["self"].mean(), lookalike=c["matched"].mean(), random=c["random"].mean(),
                         extra=c.extra.mean(), lo=lo, hi=hi))
S = pd.DataFrame(summ); S.to_csv(OUT + "summary_by_model.csv", index=False)

# chrF test: 10% of all 8 layers, cases with intact chrF > 20, drop = intact - after, averaged over 3 repeats
q = allL[(allL.dose == 0.10) & allL.chrf.notna() & (allL.chrf_intact > 20)].copy()
q["drop"] = q.chrf_intact - q.chrf
cq = q.groupby(["model", "phase", "eval_lang", "group"])["drop"].mean().unstack("group").reset_index()
cq["extra"] = cq["self"] - cq["matched"]; cq.to_csv(OUT + "chrf_cases.csv", index=False)
chs = []
for m, *_ in MODELS:
    c = cq[cq.model == m]; k = int((c.extra > 0).sum())
    lo, hi = hboot(c.rename(columns={}), "extra")
    chs.append(dict(model=m, cases=len(c), self_worse=k, sign_p=binomtest(k, len(c), .5, alternative="greater").pvalue,
                    self_drop=c["self"].mean(), lookalike_drop=c["matched"].mean(), random_drop=c["random"].mean(),
                    extra=c.extra.mean(), lo=lo, hi=hi))
C = pd.DataFrame(chs); C.to_csv(OUT + "chrf_by_model.csv", index=False)

# ---------- FIGURE 1: extra surprise (self - look-alike) vs dose, one line per model
fig, ax = plt.subplots(figsize=(7.2, 4.6))
off = {"continual": -1.2, "hindi": -0.4, "mixed": 0.4, "related": 1.2}
for m, lab, mk, col in MODELS:
    s = S[S.model == m]; xs = 100 * s.dose + off[m]
    ax.errorbar(xs, s.extra, yerr=[s.extra - s.lo, s.hi - s.extra], marker=mk, ms=11 if mk != "_" else 16, mew=2.2,
                color=col, capsize=4, lw=1.4, label=f"{lab}  ({s.ckpts.iloc[0]} checkpoints)")
ax.axhline(0, color="gray", lw=1, ls=":"); ax.set_xticks([5, 10, 20]); ax.set_xticklabels(["5%", "10%", "20%"])
ax.set_xlabel("% of ONE layer switched off (each neuron always outputs its own average)")
ax.set_ylabel("extra Δ surprise\n(self switched off − look-alike task switched off)")
ax.set_title("Above 0 = switching off self neurons hurts more than switching off look-alike task neurons\n"
             "dot = mean over checkpoints × 8 layers; bar = 95% range (bootstrap over checkpoints, then layers)", fontsize=9.5)
ax.legend(fontsize=8.5, frameon=False, loc="upper left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT + "fig1_extra_surprise_by_model.png", dpi=200); plt.close(fig)

# ---------- FIGURE 2: 10% dose, raw Δ surprise for self / look-alike / random, one column per model
fig, axs = plt.subplots(1, 4, figsize=(12, 3.9))
for ax, (m, lab, mk, col) in zip(axs, MODELS):
    c = cmp_[(cmp_.model == m) & (cmp_.dose == 0.10)]
    for i, (g, gl, gc) in enumerate([("self", "self", col), ("matched", "look-alike\ntask", "#888"), ("random", "random\ntask", "#BBB")]):
        lo, hi = hboot(c.dropna(subset=[g]), g); mu = c[g].mean()
        ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], marker=mk, ms=12 if mk != "_" else 18, mew=2.2, color=gc, capsize=5)
    s = S[(S.model == m) & (S.dose == 0.10)].iloc[0]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["self", "look-alike\ntask", "random\ntask"], fontsize=8.5); ax.set_xlim(-.6, 2.6)
    ax.set_title(f"{lab}\nself worse in {s.self_worse}/{s.comparisons}, p = {s.sign_p:.2g}", fontsize=8.8); ax.grid(alpha=.25, axis="y")
    ax.set_ylim(bottom=0)
axs[0].set_ylabel("Δ surprise when 10% of\nONE layer is switched off")
fig.tight_layout(); fig.savefig(OUT + "fig2_raw_10pct_by_model.png", dpi=200); plt.close(fig)

# ---------- FIGURE 3: chrF drop, 10% of all 8 layers
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, (m, lab, mk, col) in enumerate(MODELS):
    c = cq[cq.model == m]
    for j, (g, gc) in enumerate([("self", col), ("matched", "#888"), ("random", "#BBB")]):
        lo, hi = hboot(c, g); mu = c[g].mean()
        ax.errorbar(i + (j - 1) * .22, mu, yerr=[[mu - lo], [hi - mu]], marker=mk, ms=11 if mk != "_" else 16, mew=2.2, color=gc, capsize=4)
    r = C[C.model == m].iloc[0]
    ax.text(i, ax.get_ylim()[1] if False else -1.4, f"self worse\n{r.self_worse}/{r.cases}", ha="center", fontsize=8)
ax.axhline(0, color="gray", lw=1, ls=":")
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, *_ in MODELS], fontsize=8)
ax.set_ylabel("chrF drop (points)"); ax.set_ylim(-2.2, None)
ax.set_title("10% of ALL 8 layers switched off → translation quality drop\n"
             "in each group: left = self (colour), middle = look-alike task (dark grey), right = random task (light grey)", fontsize=9.5)
ax.grid(alpha=.25, axis="y"); fig.tight_layout(); fig.savefig(OUT + "fig3_chrf_by_model.png", dpi=200); plt.close(fig)

# ---------- FIGURE 4: self size, related vs distant
rd = pd.read_csv(D + "related_distant/pairs.csv"); ri = pd.read_csv(D + "related_indic/pairs.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
for df, lab, col, mk in [(rd, "distant (hi → zh → ar)", "#4B1D6B", "*"), (ri, "related (hi → gu → pa)", "#1F5FA8", "D")]:
    ax.plot(df["to"], 100 * df["size"], color=col, lw=1.2, alpha=.6)
    for lang, lm in [("hi", "o"), ("zh", "s"), ("ar", "^"), ("gu", "s"), ("pa", "^")]:
        d = df[df.lang_to == lang]
        if len(d): ax.scatter(d["to"], 100 * d["size"], color=col, marker=lm, s=40, zorder=3)
    ax.plot([], [], color=col, label=lab)
for lang, lm in [("Hindi phase", "o"), ("2nd language (zh / gu)", "s"), ("3rd language (ar / pa)", "^")]:
    ax.scatter([], [], color="k", marker=lm, label=lang)
ax.set_xlabel("phase"); ax.set_ylabel("self size\n(% of a layer's neurons, mean of 8 layers)")
ax.set_title("Self size at each phase", fontsize=10); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25)
ax = axs[1]
both = sorted(set(ri["to"]) & set(rd["to"])); a = rd.set_index("to").loc[both]; b = ri.set_index("to").loc[both]
xs = np.arange(len(both))
ax.bar(xs - .2, 100 * a["size"], .4, color="#4B1D6B", label="distant"); ax.bar(xs + .2, 100 * b["size"], .4, color="#1F5FA8", label="related")
ax.set_xticks(xs); ax.set_xticklabels([f"{p}\n{a.loc[p,'lang_to']} | {b.loc[p,'lang_to']}" for p in both], fontsize=8)
ax.set_ylim(40, 60); ax.set_xlabel("phase  (language: distant | related)"); ax.set_ylabel("self size (%)")
ax.set_title(f"Same phases side by side: related bigger in {(b['size'] > a['size']).sum()} of {len(both)}", fontsize=10)
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(OUT + "fig4_self_size_related_vs_distant.png", dpi=200); plt.close(fig)
sz = pd.DataFrame({"phase": both, "lang_distant": a.lang_to.values, "size_distant": a["size"].values,
                   "lang_related": b.lang_to.values, "size_related": b["size"].values,
                   "gap_distant": a.rel_gap.values, "gap_related": b.rel_gap.values})
sz.to_csv(OUT + "self_size.csv", index=False)
print(S.round(4).to_string()); print(C.round(3).to_string()); print(sz.round(3).to_string())
print("size related>distant sign p", binomtest(int((sz.size_related > sz.size_distant).sum()), len(sz), .5).pvalue)
