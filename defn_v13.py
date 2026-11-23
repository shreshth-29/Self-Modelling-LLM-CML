"""defn_v13.py: how solid is the definition of the self? No training. Runs on existing checkpoints and on untrained models.

Self so far = largest connected group of a layer's MLP neurons when two neurons are linked if |correlation| >= 0.2
(FLORES dev, decoder language means removed). Problem found earlier: an untrained model also gives a huge group at 0.2.

For every model and layer this script measures:
  sweep      the largest group (and the second largest, and how many groups of 5+) at every threshold from 0.05 to 0.95,
             for 5 ways of scoring "fire together":
               corr   correlation (what we use)
               cos    cosine of the raw activity (the robot paper's measure)
               pc1    correlation after removing the layer's strongest shared signal (first principal component)
               pc3    ... after removing the 3 strongest          pc10   ... after removing the 10 strongest
             plus what a random graph with the same number of links would give (er_giant).
  pcs        how much of the layer's activity the strongest shared signals explain
  split      split-half check: the self found on one half of the sentences vs on the other half
  neurons    one row per neuron: in the group or not at every threshold and measure, loading on the 3 strongest signals,
             co-firing strength (average |correlation| with all other neurons), average activity, firing rate, outgoing weight size

Models: trained checkpoints (--model name=run:phases), a run's own starting point (phase 0), and fresh untrained models (--fresh_seeds).
The summary (--summarize, no GPU) then finds the NULL-CALIBRATED threshold per layer (lowest threshold at which the untrained
models' largest group is under 5% of the layer) and re-computes the self-task persistence gap at every threshold from the
persistence values already saved in analysis/v11/matrix.

  CML_LANGSET=distant python -u defn_v13.py --gpu 0 --family distant --model distant=v4b_continual_s0:0,40
  CML_LANGSET=distant python -u defn_v13.py --gpu 0 --family distant --fresh_seeds 0,1
  python defn_v13.py --summarize
"""
import os, sys, time, argparse, glob, math
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--model", default="", help="name=run:phase[,phase,...]   (phase 0 = the run's own untrained start)")
ap.add_argument("--fresh_seeds", default="", help="comma list: fresh untrained models with these seeds")
ap.add_argument("--family", default="")
ap.add_argument("--self_langs", default="", help="languages whose text is used (decoder); default = all of the language set")
ap.add_argument("--taus", default=",".join(f"{t / 100:.2f}" for t in range(5, 100, 5)))
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--out", default="defn")
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "v13", args.out); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
LAYERS = [f"enc{j}" for j in range(4)] + [f"dec{j}" for j in range(4)]
TAUS = [float(t) for t in args.taus.split(",")]
MEAS = ["corr", "cos", "pc1", "pc3", "pc10"]
col = lambda meas, tau: f"m_{meas}_{int(round(tau * 100)):02d}"

def append_csv(path, rows):
    if len(rows):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# =============================================================================================== SUMMARY (CPU)
def auc(score, label):
    """chance that a random member has a higher score than a random non-member"""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.sum() == 0 or (~label).sum() == 0:
        return np.nan
    r = pd.Series(score).rank().values
    return (r[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum())

def summarize():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
    R = []
    def say(x=""):
        print(x); R.append(str(x))
    s = pd.read_csv(f"{OUT}/sweep.csv").drop_duplicates(["langset", "run", "phase", "kind", "seed", "layer", "measure", "tau"], keep="last")
    s["pct"] = 100 * s["largest"] / s["n_alive"].clip(lower=1); s["pct2"] = 100 * s["second"] / s["n_alive"].clip(lower=1)
    s["er_pct"] = 100 * s["er_giant"]
    null = s[s["kind"] != "trained"]; tr = s[s["kind"] == "trained"]
    say("=== 1. Largest group (% of the layer) by threshold: untrained models vs trained models, measure = correlation ===")
    say("untrained = the runs' own starting points and fresh random models; average over those models and the 8 layers")
    a = null[null["measure"] == "corr"].groupby(["langset", "tau"])["pct"].mean().unstack("tau")
    say("untrained:"); say(a.round(1).to_string())
    b = tr[tr["measure"] == "corr"].groupby(["family", "tau"])["pct"].mean().unstack("tau")
    say("trained:"); say(b.round(1).to_string())

    say("\n=== 2. Null-calibrated threshold per layer: lowest threshold where the untrained models' largest group is under 5% of the layer ===")
    cal = {}
    rows = []
    for (ls, meas, L), q in null.groupby(["langset", "measure", "layer"]):
        m = q.groupby("tau")["pct"].mean().sort_index()
        ok = m[m < 5.0]
        t = float(ok.index.min()) if len(ok) else np.nan
        cal[(ls, meas, L)] = t; rows.append({"langset": ls, "measure": meas, "layer": L, "tau_star": t})
    c = pd.DataFrame(rows)
    say(c.pivot_table(index=["langset", "measure"], columns="layer", values="tau_star").reindex(columns=LAYERS).to_string())
    c.to_csv(f"{OUT}/calibrated_tau.csv", index=False)

    say("\n=== 3. The trained self at the calibrated threshold (measure = correlation): size, second largest group, groups of 5+ ===")
    rows = []
    for (fam, ls, L), q in tr[tr["measure"] == "corr"].groupby(["family", "langset", "layer"]):
        t = cal.get((ls, "corr", L), np.nan)
        for tag, tt in [("0.20", 0.2), ("calibrated", t)]:
            x = q[np.isclose(q["tau"], tt)] if tt == tt else q.iloc[0:0]
            if len(x):
                rows.append({"family": fam, "layer": L, "threshold": tag, "tau": tt, "self_pct": x["pct"].mean(), "second_pct": x["pct2"].mean(),
                             "groups_5plus": x["n_comp5"].mean(), "random_graph_pct": x["er_pct"].mean(), "checkpoints": x[["run", "phase"]].drop_duplicates().shape[0]})
    z = pd.DataFrame(rows)
    if len(z):
        say(z.pivot_table(index=["family", "layer"], columns="threshold", values=["tau", "self_pct", "second_pct", "groups_5plus", "random_graph_pct"]).round(2).to_string())
        say("\naverage of the 8 layers:")
        say(z.groupby(["family", "threshold"])[["self_pct", "second_pct", "groups_5plus", "random_graph_pct"]].mean().round(2).to_string())

    say("\n=== 4. Other measures: largest group (% of layer) at threshold 0.20 and 0.50, average of the 8 layers ===")
    q = s[np.isclose(s["tau"], 0.2) | np.isclose(s["tau"], 0.5)].copy(); q["who"] = q["family"].astype(str).where(q["kind"] == "trained", "UNTRAINED " + q["langset"].astype(str))
    say(q.pivot_table(index="who", columns=["tau", "measure"], values="pct").round(1).to_string())

    if os.path.exists(f"{OUT}/pcs.csv"):
        p = pd.read_csv(f"{OUT}/pcs.csv").drop_duplicates(["langset", "run", "phase", "kind", "seed", "layer"], keep="last")
        p["who"] = p["family"].astype(str).where(p["kind"] == "trained", "UNTRAINED " + p["langset"].astype(str))
        say("\n=== 5. Strongest shared signals: % of the layer's activity explained by the top 1 / 3 / 10 (average of layers) ===")
        say(p.groupby("who")[["var1", "var3", "var10"]].mean().mul(100).round(1).to_string())
        say("by layer, top 1:"); say(p.pivot_table(index="who", columns="layer", values="var1").reindex(columns=LAYERS).mul(100).round(1).to_string())

    files = sorted(glob.glob(f"{OUT}/neurons_*.csv.gz"))
    N = {os.path.basename(f)[8:-7]: pd.read_csv(f) for f in files}
    if N:
        say("\n=== 6. Is the self (threshold 0.20) just the neurons that load on the strongest shared signal? ===")
        say("AUC: 0.5 = no relation, 1.0 = the self is exactly the highest-loading neurons. strength = average |correlation| with all other neurons")
        rows = []
        for name, d in N.items():
            for L, q in d.groupby("layer"):
                q = q[q["alive"] == 1]; lab = q[col("corr", 0.2)].values == 1
                rows.append({"model": name, "layer": L, "self_pct": 100 * lab.mean(), "auc_pc1": auc(q["pc1"].abs(), lab), "auc_strength": auc(q["strength"], lab),
                             "auc_activity": auc(q["mean"], lab), "auc_fire": auc(q["fire"], lab), "auc_wout": auc(q["wout"], lab)})
        a6 = pd.DataFrame(rows)
        say(a6.groupby("model")[["self_pct", "auc_pc1", "auc_strength", "auc_activity", "auc_fire", "auc_wout"]].mean().round(3).to_string())

    if os.path.exists(f"{OUT}/split.csv"):
        sp_ = pd.read_csv(f"{OUT}/split.csv").drop_duplicates(["langset", "run", "phase", "kind", "seed", "layer", "tau"], keep="last")
        sp_ = sp_[sp_["kind"] == "trained"]
        say("\n=== 7. Split-half: the self found on one half of the sentences vs the other half (shared / in either; 1 = identical) ===")
        say(sp_.pivot_table(index="family", columns="tau", values="jaccard").round(3).to_string())
        say("size of the self on half the data (% of layer):"); say((100 * sp_.assign(f=sp_["n_a"] / sp_["n_alive"]).pivot_table(index="family", columns="tau", values="f")).round(1).to_string())

    pf = os.path.join(REPO, "analysis", "v11", "matrix", "persistence_per_neuron.csv")
    if N and os.path.exists(pf):
        pp = pd.read_csv(pf); pp = pp[pp["valid"] == True]
        say("\n=== 8. Self-task persistence gap at EVERY threshold (persistence values from analysis/v11/matrix; one phase transition per model) ===")
        say("gap = average persistence of neurons in the group MINUS average of the alive neurons outside it; average of the 8 layers")
        say("matched = same, but compared inside 10 bins of equal average activity (so the gap is not just 'active neurons persist more')")
        rows = []
        for fam, pa in [("distant", 40), ("related", 13), ("hindi", 40), ("mixed", 40)]:
            key = [k for k in N if k.startswith(fam + "_") and k.endswith(f"_{pa}") and "s1" not in k]
            q = pp[(pp["model"] == fam) & (pp["phase_a"] == pa)]
            if not key or not len(q):
                continue
            d = N[key[0]].merge(q[["layer", "neuron", "persistence"]], on=["layer", "neuron"])
            d = d[d["alive"] == 1]
            for meas in MEAS:
                for tau in TAUS:
                    c_ = col(meas, tau)
                    if c_ not in d.columns:
                        continue
                    for L, x in d.groupby("layer"):
                        m = x[c_].values == 1
                        if m.sum() < 20 or (~m).sum() < 20:
                            continue
                        g = x["persistence"].values[m].mean() - x["persistence"].values[~m].mean()
                        bins = pd.qcut(x["mean"].rank(method="first"), 10, labels=False).values; diffs = []; w = []
                        for b_ in range(10):
                            i = bins == b_
                            if (m & i).sum() >= 3 and ((~m) & i).sum() >= 3:
                                diffs.append(x["persistence"].values[m & i].mean() - x["persistence"].values[(~m) & i].mean()); w.append(min((m & i).sum(), ((~m) & i).sum()))
                        gm = float(np.average(diffs, weights=w)) if len(diffs) >= 3 else np.nan
                        rows.append({"family": fam, "measure": meas, "tau": tau, "layer": L, "side": L[:3], "gap": g, "gap_matched": gm, "self_pct": 100 * m.mean()})
        g8 = pd.DataFrame(rows)
        if len(g8):
            g8.to_csv(f"{OUT}/gap_by_tau.csv", index=False)
            for meas in ["corr", "pc1", "cos"]:
                x = g8[g8["measure"] == meas]
                if not len(x):
                    continue
                say(f"\n--- measure = {meas}: gap"); say(x.pivot_table(index="family", columns="tau", values="gap").round(3).to_string())
                say("gap, activity-matched:"); say(x.pivot_table(index="family", columns="tau", values="gap_matched").round(3).to_string())
                say("layers (of 8) where a group of 20+ and a rest of 20+ exist:"); say(x.pivot_table(index="family", columns="tau", values="gap", aggfunc="count").to_string())
                say("self size (% of layer):"); say(x.pivot_table(index="family", columns="tau", values="self_pct").round(1).to_string())
            x = g8[g8["measure"] == "corr"]
            say("\n--- measure = corr, encoder and decoder separately: activity-matched gap")
            say(x.pivot_table(index=["family", "side"], columns="tau", values="gap_matched").round(3).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fams = sorted(tr["family"].unique()); fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
        for ax, L in zip(axes.ravel(), LAYERS):
            q = null[(null["measure"] == "corr") & (null["layer"] == L)].groupby("tau")["pct"].mean()
            ax.plot(q.index, q.values, "k--", lw=2, label="untrained")
            for fam in fams:
                q = tr[(tr["measure"] == "corr") & (tr["layer"] == L) & (tr["family"] == fam)].groupby("tau")["pct"].mean()
                ax.plot(q.index, q.values, "o-", ms=3, label=fam)
            ax.axvline(0.2, color="#999", lw=0.8, ls=":"); ax.set_title(L); ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=8)
        for ax in axes[1]: ax.set_xlabel("threshold on |correlation|")
        for ax in axes[:, 0]: ax.set_ylabel("largest group (% of layer)")
        fig.tight_layout(); fig.savefig(f"{OUT}/sweep_corr.png", dpi=150); plt.close(fig)
    except Exception as e:
        say(f"(figure skipped: {e})")
    with open(f"{OUT}/report.txt", "w") as f:
        f.write("\n".join(R))
    print("report in", OUT)

if args.summarize:
    summarize(); sys.exit()

# =============================================================================================== GPU
from common_v8 import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
FAMILY = args.family or (args.model.split("=")[0] if "=" in args.model else "untrained")
SELF_LANGS = [l for l in (args.self_langs.split(",") if args.self_langs else LANGS) if l in LANGS]

def mlp_mod(model, L):
    return (model.enc_blocks if L.startswith("enc") else model.dec_blocks)[int(L[3])].mlp

DEV_EN = read_flores("dev", "en")
SRC = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}

@torch.no_grad()
def record(model, langs, bs=64):
    """post-GELU activity at every real token (CPU, fp16) + which language each decoder row belongs to (index into langs)"""
    model.eval(); cur = {}
    hooks = [mlp_mod(model, L).act.register_forward_hook(lambda mm, i, o, k=L: cur.__setitem__(k, o)) for L in LAYERS]
    acts = {L: [] for L in LAYERS}; dlang = []
    for s in range(0, len(SRC), bs):
        S_ = pad_batch(SRC[s:s + bs]); mem, sm = model.encode(S_)
        sk = (S_ != PAD) & (S_ != EOS)
        for L in LAYERS[:4]:
            acts[L].append(cur[L][sk].half().cpu())
        for li, l in enumerate(langs):
            T_ = pad_batch(TGT[l][s:s + bs]); model.decode(T_, mem, sm)
            tk = T_ != PAD; tk[:, 0] = False
            for L in LAYERS[4:]:
                acts[L].append(cur[L][tk].half().cpu())
            dlang.append(torch.full((int(tk.sum()),), li))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(dlang).numpy()

def sample_rows(acts, dlang, n_langs):
    """same token sample as core_v11.groups; also marks which sampled rows fall in half A (the earlier sentences)"""
    g0 = np.random.default_rng(0)
    n_src = acts["enc0"].shape[0]
    i_src = np.sort(g0.choice(n_src, min(n_src, args.max_t), replace=False))
    half_src = np.arange(len(i_src)) < len(i_src) // 2
    g1, parts, halves = np.random.default_rng(1), [], []
    for li in range(n_langs):
        pos = np.nonzero(dlang == li)[0]
        p = np.sort(g1.choice(pos, min(len(pos), args.max_t // n_langs), replace=False))
        parts.append(p); halves.append(np.arange(len(p)) < len(p) // 2)
    return i_src, half_src, np.concatenate(parts), np.concatenate(halves)

def corr_of(X):
    """X: tokens x neurons (float, on the GPU). Returns the correlation matrix and the alive mask."""
    mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
    alive = sd[0] > 1e-6
    Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd)); Z[:, ~alive] = 0
    Rm = Z.T @ Z / Z.shape[0]
    return Rm, alive

def components(Rabs, alive, tau):
    """largest group, second largest, groups of 5+, average number of links per alive neuron, membership of the largest"""
    A = ((Rabs >= tau) & alive[:, None] & alive[None, :])
    A.fill_diagonal_(False)
    deg = float(A.sum().item()) / max(int(alive.sum().item()), 1)
    An = A.cpu().numpy(); alive_np = alive.cpu().numpy()
    _, lab = connected_components(csr_matrix(An), directed=False)
    lab = np.where(alive_np, lab, -1)
    if (lab >= 0).any():
        cnt = np.sort(np.bincount(lab[lab >= 0]))[::-1]
        big = int(np.bincount(lab[lab >= 0]).argmax()); mem = lab == big
        largest, second, n5 = int(cnt[0]), (int(cnt[1]) if len(cnt) > 1 else 0), int((cnt >= 5).sum())
    else:
        mem = np.zeros(len(lab), bool); largest = second = n5 = 0
    if largest <= 1:
        mem = np.zeros(len(lab), bool); largest = 0
    return largest, second, n5, deg, mem

def er_giant(c):
    """share of nodes in the giant group of a random graph with average c links per node"""
    if not (c > 1.0):
        return 0.0
    S = 0.5
    for _ in range(200):
        S = 1.0 - math.exp(-c * S)
    return S

def analyse(model, tag):
    """tag: dict(langset, family, run, phase, kind, seed). Writes sweep, pcs, split rows and the per-neuron table."""
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time(); acts, dlang = record(model, SELF_LANGS)
    i_src, half_src, i_tgt, half_tgt = sample_rows(acts, dlang, len(SELF_LANGS))
    sweep_rows, pc_rows, split_rows, neuron_frames = [], [], [], []
    for L in LAYERS:
        enc = L.startswith("enc"); idx = i_src if enc else i_tgt; half = half_src if enc else half_tgt
        Xraw = acts[L][torch.as_tensor(idx)].float().to(DEVICE)
        X = Xraw.clone()
        if not enc:
            g = torch.as_tensor(dlang[i_tgt]).to(DEVICE)
            for v in g.unique():
                m = g == v; X[m] -= X[m].mean(0, keepdim=True)
        Rm, alive = corr_of(X); n_alive = int(alive.sum().item()); Nn = Rm.shape[0]
        mats = {"corr": Rm.abs()}
        nrm = Xraw.norm(dim=0); nrm = torch.where(nrm > 0, nrm, torch.ones_like(nrm))
        mats["cos"] = ((Xraw.T @ Xraw) / (nrm[:, None] * nrm[None, :])).abs()
        # strongest shared signals: eigen-decomposition of the correlation matrix
        ev, V = torch.linalg.eigh(Rm.double()); ev, V = ev.flip(0), V.flip(1)
        tot = float(ev.clamp(min=0).sum().item())
        pc_rows.append(dict(tag, layer=L, n_alive=n_alive, var1=float(ev[:1].sum()) / tot, var3=float(ev[:3].sum()) / tot, var10=float(ev[:10].sum()) / tot))
        load = (V[:, :3] * ev[:3].clamp(min=0).sqrt()).float().cpu().numpy()            # correlation of each neuron with signals 1..3
        for k in (1, 3, 10):
            Rk = Rm.double() - (V[:, :k] * ev[:k]) @ V[:, :k].T
            d = Rk.diagonal().clamp(min=1e-8).sqrt()
            mats[f"pc{k}"] = (Rk / (d[:, None] * d[None, :])).float().abs()
            del Rk
        off = ~torch.eye(Nn, dtype=torch.bool, device=DEVICE)
        strength = ((Rm.abs() * off).sum(1) / max(n_alive - 1, 1)).cpu().numpy()
        A_ = acts[L].float(); mean = A_.mean(0).numpy(); fire = (A_ > 0).float().mean(0).numpy(); del A_
        wout = mlp_mod(model, L).proj.weight.detach().float().norm(dim=0).cpu().numpy()
        nd = {"layer": L, "neuron": np.arange(Nn), "alive": alive.cpu().numpy().astype(np.uint8), "mean": mean, "fire": fire, "wout": wout,
              "pc1": load[:, 0], "pc2": load[:, 1], "pc3": load[:, 2], "strength": strength}
        for meas in MEAS:
            for tau in TAUS:
                largest, second, n5, deg, mem = components(mats[meas], alive, tau)
                sweep_rows.append(dict(tag, layer=L, measure=meas, tau=tau, n_alive=n_alive, largest=largest, second=second, n_comp5=n5,
                                       mean_deg=deg, er_giant=er_giant(deg)))
                nd[col(meas, tau)] = mem.astype(np.uint8)
        neuron_frames.append(pd.DataFrame(nd))
        # split-half: correlation matrix on each half of the sentences
        hm = torch.as_tensor(half).to(DEVICE)
        Ra, al_a = corr_of(X[hm]); Rb, al_b = corr_of(X[~hm])
        gp = np.random.default_rng(0); pi = torch.as_tensor(gp.integers(0, Nn, 200000), device=DEVICE); pj = torch.as_tensor(gp.integers(0, Nn, 200000), device=DEVICE)
        keep = pi != pj; va, vb = Ra[pi[keep], pj[keep]], Rb[pi[keep], pj[keep]]
        r_ab = float(torch.corrcoef(torch.stack([va, vb]))[0, 1].item())
        for tau in TAUS:
            _, _, _, _, ma = components(Ra.abs(), al_a, tau); _, _, _, _, mb = components(Rb.abs(), al_b, tau)
            both, either = int((ma & mb).sum()), int((ma | mb).sum())
            split_rows.append(dict(tag, layer=L, tau=tau, n_alive=n_alive, n_a=int(ma.sum()), n_b=int(mb.sum()), n_both=both,
                                   jaccard=(both / either if either else np.nan), r_halves=r_ab))
        del X, Xraw, Rm, mats, V, ev, Ra, Rb
        torch.cuda.empty_cache()
    append_csv(f"{OUT}/sweep.csv", sweep_rows); append_csv(f"{OUT}/pcs.csv", pc_rows); append_csv(f"{OUT}/split.csv", split_rows)
    name = f"{tag['family']}_{tag['run']}_{tag['kind']}{tag['seed']}_{tag['phase']}"
    pd.concat(neuron_frames).to_csv(f"{OUT}/neurons_{name}.csv.gz", index=False, compression="gzip")
    q = pd.DataFrame(sweep_rows); q = q[q["measure"] == "corr"]; q = q.assign(pct=100 * q["largest"] / q["n_alive"].clip(lower=1))
    show = q[q["tau"].isin([t for t in TAUS if int(round(t * 100)) in (10, 20, 30, 40, 50, 60, 70)])].pivot_table(index="layer", columns="tau", values="pct").reindex(LAYERS)
    print(f"{name} (text: {','.join(SELF_LANGS)}): largest group, % of layer, by threshold on |correlation|"); print(show.round(1).to_string())
    print(f"  {time.time() - t0:.0f}s", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True

def done_before(run, phase, kind, seed):
    f = f"{OUT}/sweep.csv"
    if not os.path.exists(f):
        return False
    d = pd.read_csv(f, usecols=["langset", "run", "phase", "kind", "seed"])
    return bool(((d["langset"] == LANGSET) & (d["run"].astype(str) == str(run)) & (d["phase"] == phase) & (d["kind"] == kind) & (d["seed"] == seed)).any())

if args.model:
    name, rest = args.model.split("="); run, phs = rest.split(":")
    for ph in [int(p) for p in phs.split(",")]:
        f = glob.glob(P(f"runs/{run}/ckpt/p{ph:03d}_*.pt"))
        kind = "init" if ph == 0 else "trained"
        if not f:
            print(f"  {run} phase {ph}: checkpoint not on this machine, skipped", flush=True); continue
        if done_before(run, ph, kind, 0):
            print(f"  {run} phase {ph}: already done, skipped", flush=True); continue
        model, ck = load_model(f[0])
        analyse(model, dict(langset=LANGSET, family=FAMILY, run=run, phase=ph, kind=kind, seed=0))
        del model
        torch.cuda.empty_cache()
for sd in [int(x) for x in args.fresh_seeds.split(",") if x != ""]:
    if done_before("fresh", 0, "fresh", sd):
        print(f"  fresh untrained model, seed {sd}: already done, skipped", flush=True); continue
    torch.manual_seed(1000 + sd); model = Seq2Seq(Config()).to(DEVICE)
    analyse(model, dict(langset=LANGSET, family=FAMILY, run="fresh", phase=0, kind="fresh", seed=sd))
    del model
    torch.cuda.empty_cache()
print("done")
