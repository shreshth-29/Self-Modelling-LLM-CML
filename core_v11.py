"""core_v11.py: importance of the self, redone COUNT-MATCHED (Adi, Nov 11). Language tasks only.

Self = largest co-firing group of a layer's MLP neurons at tau 0.2 (same definition as analysis_v8.py / newtask_v9.py:
FLORES dev, all of the model's languages, decoder language means removed). Task = all other working neurons.

COUNT-MATCHED: in every layer, n_max = size of the SMALLER of the two groups (self, task).
  self@1.0 = n_max random self neurons        task@1.0 = n_max random task neurons
  self@0.5 / task@0.5 = half of n_max         look@0.5 = half of n_max task neurons that LOOK LIKE the self@0.5 pick
                                               (same average activity, firing rate, outgoing weight size)
Three interventions (--mode), named as in the meeting:
  lesion   the neurons' incoming weights (fc row + bias) and outgoing weights (proj column) are re-drawn as at
           initialisation. They CAN learn again. (This is the paper's lesion.)
  silence  each neuron always outputs its own average (measured on the start model). Held for the whole run.
  freeze   the neurons keep working, but their incoming weights (fc row + bias) cannot change during training.

Parts (--part):
  sizes   module sizes per layer for many checkpoints of one run (self found on all the model's languages, and on Hindi only)
  matrix  the paper's figure: co-activation matrix reordered into groups + per-neuron persistence below it
          (persistence between phase p and p+1: Hungarian matching, (activation similarity + connectivity similarity) / 2)
  static  no training: lesion / silence on a finished model, per layer and all 8 layers; loss rise per language (+ chrF for all layers)
  train   lesion / silence / freeze, then train: task "next" (the run's next language, normal masked loss)
          or "indonesian" (new language; needs prep_newtasks.py)

  CML_LANGSET=distant python -u core_v11.py --gpu 0 --part static --model distant=v4b_continual_s0:10,11,12
  CML_LANGSET=distant python -u core_v11.py --gpu 0 --part train --task next --mode lesion --model distant12=v4b_continual_s0:12 --seeds 0,1,2
  python -u core_v11.py --summarize --part train --out train_next
"""
import os, sys, json, time, argparse, glob, random, math
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--part", default="", choices=["", "sizes", "matrix", "static", "train"])
ap.add_argument("--model", default="", help="name=run:phase[,phase,...]   (train: exactly one phase)   or fresh")
ap.add_argument("--family", default="", help="label used to group results: distant / related / hindi / mixed")
ap.add_argument("--eval_langs", default="", help="static: comma list; default = all languages of the language set")
ap.add_argument("--self_langs", default="", help="static / matrix / train: comma list of languages the self is found on; default = all languages of the language set")
ap.add_argument("--task", default="next", choices=["next", "indonesian"])
ap.add_argument("--mode", default="lesion", choices=["lesion", "silence", "freeze"])
ap.add_argument("--conds", default="none,self@1.0,task@1.0,self@0.5,task@0.5,look@0.5")
ap.add_argument("--seeds", default="0")
ap.add_argument("--draws", default="0,1")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max_tokens", type=int, default=8000)
ap.add_argument("--n_val", type=int, default=200)
ap.add_argument("--n_chrf", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--out", default="")
ap.add_argument("--outroot", default="v11", help="results go to analysis/<outroot>/<out>  (v14 uses its own folder)")
ap.add_argument("--next_lang", default="", help="train, task next: train this language (default: the next one in the run's order). Hindi-only runs use hi")
ap.add_argument("--compact", action="store_true", help="matrix: no figures; one small per-neuron file per transition (persistence + average activity + firing rate)")
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
DEFAULT_OUT = {"sizes": "sizes", "matrix": "matrix", "static": "static", "train": f"train_{args.task}"}
OUT = os.path.join(REPO, "analysis", args.outroot, args.out or DEFAULT_OUT.get(args.part, "misc")); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
LAYERS = [f"enc{j}" for j in range(4)] + [f"dec{j}" for j in range(4)]
EVAL_AT = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]

def append_csv(path, rows):
    if len(rows):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def split_cond(df):
    """adds columns group (self / task / look) and dose (text, e.g. '1.0') from cond like 'self@1.0'"""
    df = df.copy()
    parts = df["cond"].astype(str).str.split("@", expand=True)
    df["group"] = parts[0]; df["dose"] = parts[1] if parts.shape[1] > 1 else ""
    return df

# =============================================================================================== SUMMARY (CPU)
def sign_p(k, n):
    from scipy.stats import binomtest
    return binomtest(int(k), int(n), 0.5, alternative="greater").pvalue if n > 0 else float("nan")

def summarize():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = []
    def say(s):
        print(s); R.append(str(s))
    COL = {"self": "#7A3FB0", "task": "#D96A12", "look": "#9CA3AF"}

    if args.part == "sizes":
        s = pd.read_csv(f"{OUT}/sizes.csv")
        s["self_pct"] = 100 * s.n_self / s.n_alive.clip(lower=1); s["task_pct"] = 100 * s.n_task / s.n_alive.clip(lower=1)
        say("=== module sizes per layer, averaged over the checkpoints listed (phases <= 60) ===")
        say("self_pct = % of the layer's working neurons in the self; n_max = size of the smaller group (the count-matched lesion size)")
        s["n_max"] = s[["n_self", "n_task"]].min(axis=1)
        for d, q in s.groupby("self_def"):
            say(f"\n--- self found on: {d}")
            g = q.groupby(["family", "layer"]).agg(self_pct=("self_pct", "mean"), sd=("self_pct", "std"), n_self=("n_self", "mean"),
                                                   n_task=("n_task", "mean"), n_max=("n_max", "mean"), working=("n_alive", "mean"), checkpoints=("phase", "nunique"))
            say(g.round(1).to_string())
            say("\naverage of the 8 layers:")
            say(q.groupby(["family", "phase"]).self_pct.mean().groupby("family").agg(["mean", "std", "count"]).round(1).to_string())
        try:
            fams = [f for f in ["hindi", "mixed", "distant", "related"] if f in set(s.family)]
            FC = {"hindi": "#2A6F97", "mixed": "#6B7280", "distant": "#C8102E", "related": "#00918F"}
            for d, q in s.groupby("self_def"):
                fig, ax = plt.subplots(figsize=(10, 4.6))
                for k, f in enumerate(fams):
                    g = q[q.family == f].groupby("layer").self_pct.agg(["mean", "std"]).reindex(LAYERS)
                    x = np.arange(8) + (k - (len(fams) - 1) / 2) * 0.2
                    ax.bar(x, g["mean"], 0.19, yerr=g["std"], color=FC[f], capsize=2, label=f)
                ax.set_xticks(range(8)); ax.set_xticklabels(LAYERS); ax.set_ylabel("self size (% of the layer's working neurons)")
                ax.set_title(f"Self size per layer, mean and spread over checkpoints (self found on: {d})", fontsize=10); ax.legend(frameon=False)
                fig.tight_layout(); fig.savefig(f"{OUT}/sizes_{str(d).replace(' ', '_')}.png", dpi=160); plt.close(fig)
        except Exception as e:
            say(f"(figure skipped: {e})")

    if args.part == "static":
        s = split_cond(pd.read_csv(f"{OUT}/static.csv"))
        s["kind"] = np.where(s["scope"].astype(str) == "all", "all 8 layers", "one layer")
        key = ["family", "phase", "scope", "kind", "mode", "dose", "draw"]
        say("=== static (no training): loss rise on the model's languages; one comparison = one checkpoint x one scope x one draw (average of the languages) ===")
        g = s.groupby(key + ["group"]).d_loss.mean().unstack("group").reset_index()
        for comp in ["task", "look"]:
            if comp not in g.columns:
                continue
            say(f"\n--- self vs {comp} (count-matched)")
            rows = []
            for (fam, kind, mode, dose), q in g.dropna(subset=["self", comp]).groupby(["family", "kind", "mode", "dose"]):
                k, n = int((q["self"] > q[comp]).sum()), len(q)
                rows.append({"family": fam, "scope": kind, "mode": mode, "dose": dose, "self": q["self"].mean(), comp: q[comp].mean(),
                             "ratio": q["self"].mean() / max(q[comp].mean(), 1e-9), "self_worse": f"{k}/{n}", "sign_p": sign_p(k, n)})
            say(pd.DataFrame(rows).round(4).to_string(index=False))
        c = s[s["scope"].astype(str) == "all"].dropna(subset=["d_chrf"])
        if len(c):
            say("\n--- chrF drop (points), all 8 layers")
            say(c.groupby(["family", "mode", "dose", "group"]).d_chrf.mean().unstack("group").round(2).to_string())
        say("\n--- per layer (dose 1.0): loss rise self / task")
        pl = s[(s.kind == "one layer") & (s.dose == "1.0")].groupby(["family", "mode", "scope", "group"]).d_loss.mean().unstack("group")
        say(pl.round(4).to_string())

    if args.part == "train":
        cur = pd.read_csv(f"{OUT}/curves.csv")
        last = int(cur.step.max())
        none = cur[cur.cond == "none"].groupby(["start", "seed", "step"]).agg(none_loss=("val_loss", "mean"), none_score=("score", "mean")).reset_index()
        x = split_cond(cur[cur.cond != "none"]).merge(none, on=["start", "seed", "step"], how="left")
        x["extra"] = x.val_loss - x.none_loss
        say(f"=== train ({os.path.basename(OUT)}): held-out loss on the trained language MINUS the untouched model's, same start and seed ===")
        say("average over starts and seeds; step 0 = right after the intervention, before any training")
        steps = [st for st in [0, 100, 300, last] if st in set(cur.step)]
        t = x[x.step.isin(steps)].groupby(["family", "mode", "dose", "group", "step"]).extra.mean().unstack("step")
        say(t.round(4).to_string())
        piv = x.pivot_table(index=["family", "start", "seed", "mode", "dose", "step"], columns="group", values="val_loss").reset_index()
        for comp in ["task", "look"]:
            if comp not in piv.columns:
                continue
            say(f"\n--- self vs {comp}: in how many runs is the self version worse (one run = one start x one seed)")
            rows = []
            for (fam, mode, dose, st), q in piv[piv.step.isin(steps)].dropna(subset=["self", comp]).groupby(["family", "mode", "dose", "step"]):
                d = q["self"] - q[comp]; k, n = int((d > 0).sum()), len(d)
                rows.append({"family": fam, "mode": mode, "dose": dose, "step": st, "mean_gap": d.mean(), "self_worse": f"{k}/{n}", "sign_p": sign_p(k, n)})
            say(pd.DataFrame(rows).round(4).to_string(index=False))
        un = cur[(cur.cond == "none") & (cur.step == last)]
        say(f"\n--- untouched model (and the fresh untrained model, if run) at step {last}: loss and chrF, average over seeds")
        say(un.groupby("start").agg(loss=("val_loss", "mean"), chrf=("score", "mean"), seeds=("seed", "nunique")).round(3).to_string())
        sc = x[(x.step == last)].dropna(subset=["score"])
        if len(sc):
            say(f"\n--- final score (chrF) at step {last}: average, and change vs the untouched model")
            sc = sc.assign(d_score=sc.score - sc.none_score)
            say(sc.groupby(["family", "mode", "dose", "group"]).agg(chrf=("score", "mean"), vs_untouched=("d_score", "mean"), runs=("score", "size")).round(2).to_string())
            ps = sc.pivot_table(index=["family", "start", "seed", "mode", "dose"], columns="group", values="score").reset_index()
            rows = []
            for comp in ["task", "look"]:
                if comp not in ps.columns:
                    continue
                for (fam, mode, dose), q in ps.dropna(subset=["self", comp]).groupby(["family", "mode", "dose"]):
                    d = q[comp] - q["self"]; k, n = int((d > 0).sum()), len(d)
                    rows.append({"family": fam, "mode": mode, "dose": dose, "vs": comp, "chrf_lower_by": d.mean(), "self_lower": f"{k}/{n}", "sign_p": sign_p(k, n)})
            if rows:
                say("\n--- final chrF: self version lower than the comparison?")
                say(pd.DataFrame(rows).round(3).to_string(index=False))
        if os.path.exists(f"{OUT}/others.csv"):
            o = split_cond(pd.read_csv(f"{OUT}/others.csv"))
            o["lost"] = o.before - o.intact; o["after_vs_intact"] = o.after - o.intact
            say("\n--- the model's ORIGINAL languages (300 val pairs each): loss rise right after the intervention (lost), and at the end vs the intact model")
            say(o.groupby(["family", "mode", "dose", "group"])[["lost", "after_vs_intact"]].mean().round(3).to_string())
        try:
            fams = list(x.family.unique()); modes = list(x["mode"].unique())
            fig, axs = plt.subplots(len(modes), len(fams), figsize=(5.2 * len(fams), 3.6 * len(modes)), squeeze=False)
            for i, mo in enumerate(modes):
                for j, fa in enumerate(fams):
                    ax = axs[i, j]
                    for (grp, dose), q in x[(x["mode"] == mo) & (x.family == fa)].groupby(["group", "dose"]):
                        m = q.groupby("step").extra.mean()
                        ax.plot(m.index, m.values, color=COL.get(grp, "k"), ls="-" if dose == "1.0" else "--", marker="o", ms=3, label=f"{grp} @ {dose}")
                    ax.axhline(0, color="k", lw=0.7, ls=":"); ax.set_title(f"{fa} | {mo}", fontsize=10); ax.set_xlabel("training step")
                    if j == 0:
                        ax.set_ylabel("extra loss vs the untouched model")
            axs[0, 0].legend(frameon=False, fontsize=8)
            fig.tight_layout(); fig.savefig(f"{OUT}/curves.png", dpi=160); plt.close(fig)
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
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee
from scipy.optimize import linear_sum_assignment
FAMILY = args.family or (args.model.split("=")[0] if "=" in args.model else args.model)
SELF_LANGS = [l for l in (args.self_langs.split(",") if args.self_langs else LANGS) if l in LANGS]      # languages the self is found on

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

def groups(acts, dlang, n_langs, keep=False):
    """per layer: alive mask, self mask (largest co-firing group at tau), size of the 2nd largest group.
    keep=True also returns the z-scored traces Z and the correlation matrix R (CPU) for the matrix part."""
    g0 = np.random.default_rng(0)
    n_src = acts["enc0"].shape[0]
    i_src = torch.tensor(np.sort(g0.choice(n_src, min(n_src, args.max_t), replace=False)))
    g1, parts = np.random.default_rng(1), []
    for li in range(n_langs):
        pos = np.nonzero(dlang == li)[0]
        parts.append(np.sort(g1.choice(pos, min(len(pos), args.max_t // n_langs), replace=False)))
    i_tgt = torch.tensor(np.concatenate(parts)); out = {}
    for L in LAYERS:
        enc = L.startswith("enc")
        X = acts[L][i_src if enc else i_tgt].float().to(DEVICE)
        if not enc:
            g = torch.as_tensor(dlang)[i_tgt].to(DEVICE)
            for v in g.unique():
                m = g == v; X[m] -= X[m].mean(0, keepdim=True)
        mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
        alive = sd[0] > 1e-6
        Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd)); Z[:, ~alive] = 0
        Rm = Z.T @ Z / Z.shape[0]
        A = ((Rm.abs() >= args.tau) & alive[:, None] & alive[None, :]).cpu().numpy(); np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        alive_np = alive.cpu().numpy(); lab = np.where(alive_np, lab, -1)
        if (lab >= 0).any():
            cnt = np.bincount(lab[lab >= 0]); smask = lab == int(cnt.argmax())
            second = int(np.sort(cnt)[-2]) if len(cnt) > 1 else 0
        else:
            smask = np.zeros(len(lab), bool); second = 0
        out[L] = {"alive": alive_np, "self": smask, "second": second}
        if keep:
            out[L].update({"Z": Z.half().cpu(), "R": Rm.cpu(), "A": A, "lab": lab})
        del X, Z, Rm
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out

def neuron_info(model, langs):
    torch.backends.cuda.matmul.allow_tf32 = False
    acts, dlang = record(model, langs); G = groups(acts, dlang, len(langs)); info = {}
    for L in LAYERS:
        A_ = acts[L].float(); mean = A_.mean(0).numpy(); fire = (A_ > 0).float().mean(0).numpy()
        outn = mlp_mod(model, L).proj.weight.detach().float().norm(dim=0).cpu().numpy()
        alive, smask = G[L]["alive"], G[L]["self"]
        info[L] = dict(mean=mean, feat=np.stack([mean, fire, outn], 1), sid=np.nonzero(alive & smask)[0],
                       tid=np.nonzero(alive & ~smask)[0], nalive=int(alive.sum()), second=G[L]["second"])
        del A_
    del acts
    torch.backends.cuda.matmul.allow_tf32 = True
    return info

def matched_pick(feat, self_ids, task_ids, g):
    F_ = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
    free = np.ones(len(task_ids), bool); out = []
    for s in g.permutation(self_ids):
        d = np.linalg.norm(F_[task_ids] - F_[s], axis=1); d[~free] = np.inf
        cand = np.argsort(d)[:3]; cand = cand[np.isfinite(d[cand])]
        c = g.choice(cand); free[c] = False; out.append(task_ids[c])
    return np.array(out)

def pick_sets(info, seed, conds):
    """count-matched picks. Each (seed, layer, dose) has its own random stream, so a pick never depends on which
    other conditions were asked for. Returns {cond: [(layer, sorted neuron ids)]} and a table of what was picked."""
    sets = {c: [] for c in conds if c != "none"}; rows = []
    doses = sorted({c.split("@")[1] for c in conds if "@" in c}, key=float, reverse=True)
    for li, L in enumerate(LAYERS):
        d = info[L]; n_max = min(len(d["sid"]), len(d["tid"]))
        if n_max < 10:
            rows.append({"layer": L, "n_self": len(d["sid"]), "n_task": len(d["tid"]), "n_max": n_max, "dose": "", "n": 0, "note": "skipped: a group has fewer than 10 neurons"})
            continue
        for ds in doses:
            g = np.random.default_rng([1000 + seed, li, int(round(float(ds) * 1000))])
            n = max(1, int(round(float(ds) * n_max)))
            ss = g.choice(d["sid"], n, replace=False); tt = g.choice(d["tid"], n, replace=False)
            cand = {f"self@{ds}": ss, f"task@{ds}": tt}
            if f"look@{ds}" in sets:
                cand[f"look@{ds}"] = matched_pick(d["feat"], ss, d["tid"], g)
            for c, ids in cand.items():
                if c in sets:
                    sets[c].append((L, np.sort(np.asarray(ids))))
            rows.append({"layer": L, "n_self": len(d["sid"]), "n_task": len(d["tid"]), "n_max": n_max, "dose": ds, "n": n, "note": ""})
    return sets, rows

# ---------------- the three interventions
def silence_hooks(model, S, info):
    hooks = []
    for L, ids in S:
        ii = torch.as_tensor(ids, device=DEVICE, dtype=torch.long)
        vv = torch.as_tensor(info[L]["mean"][ids], device=DEVICE, dtype=torch.float32)
        def hook(mod, inp, out, ii=ii, vv=vv):
            out = out.clone(); out[..., ii] = vv.to(out.dtype); return out
        hooks.append(mlp_mod(model, L).act.register_forward_hook(hook))
    return hooks

@torch.no_grad()
def reinit(model, S, seed):
    """re-draw the chosen neurons' weights exactly as at initialisation. Returns what is needed to undo it."""
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed); saved = []
    c = model.cfg; sd_proj = 0.02 / math.sqrt(2 * (c.n_enc + c.n_dec))
    for L, ids in S:
        m = mlp_mod(model, L); ii = torch.as_tensor(ids, device=DEVICE, dtype=torch.long)
        saved.append((m, ii, m.fc.weight[ii].clone(), m.fc.bias[ii].clone(), m.proj.weight[:, ii].clone()))
        m.fc.weight[ii] = (torch.randn(len(ids), m.fc.weight.shape[1], generator=g) * 0.02).to(m.fc.weight)
        m.fc.bias[ii] = 0
        m.proj.weight[:, ii] = (torch.randn(m.proj.weight.shape[0], len(ids), generator=g) * sd_proj).to(m.proj.weight)
    return saved

@torch.no_grad()
def undo_reinit(saved):
    for m, ii, w, b, pw in saved:
        m.fc.weight[ii] = w; m.fc.bias[ii] = b; m.proj.weight[:, ii] = pw

def parse_model(spec):
    name, rest = spec.split("="); run, phs = rest.split(":")
    return name, run, [int(p) for p in phs.split(",")]

def ckpt_path(run, ph):
    f = glob.glob(P(f"runs/{run}/ckpt/p{int(ph):03d}_*.pt"))
    return f[0] if f else None

# =============================================================================================== PART sizes
if args.part == "sizes":
    name, run, phases = parse_model(args.model)
    defs = [("all the model's languages", list(LANGS))] + ([("Hindi only", ["hi"])] if "hi" in LANGS else [])
    for ph in phases:
        path = ckpt_path(run, ph)
        if path is None:
            print(f"  {run} phase {ph}: checkpoint not on this machine, skipped", flush=True); continue
        t0 = time.time(); model, ck = load_model(path); model.eval(); rows = []
        torch.backends.cuda.matmul.allow_tf32 = False
        for dname, langs in defs:
            acts, dlang = record(model, langs); G = groups(acts, dlang, len(langs)); del acts
            for L in LAYERS:
                a, s_ = G[L]["alive"], G[L]["self"]
                rows.append({"family": FAMILY, "run": run, "phase": ph, "lang": str(ck.get("lang")), "self_def": dname, "layer": L,
                             "n_alive": int(a.sum()), "n_self": int((a & s_).sum()), "n_task": int((a & ~s_).sum()), "second": G[L]["second"]})
        append_csv(f"{OUT}/sizes.csv", rows)
        q = [r for r in rows if r["self_def"] == defs[0][0]]
        print(f"  {FAMILY} phase {ph} ({ck.get('lang')}): self % per layer " + " ".join(f"{100 * r['n_self'] / max(r['n_alive'], 1):.0f}" for r in q)
              + f" | {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done"); sys.exit()

# =============================================================================================== PART matrix
if args.part == "matrix":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    name, run, phases = parse_model(args.model)
    pa, pb = phases[0], phases[0] + 1
    fa, fb = ckpt_path(run, pa), ckpt_path(run, pb)
    if fa is None or fb is None:
        print(f"  {run}: need phases {pa} and {pb} on this machine; missing, skipped"); sys.exit()
    torch.backends.cuda.matmul.allow_tf32 = False
    G = {}
    for tag, f in [("a", fa), ("b", fb)]:
        model, ck = load_model(f); model.eval()
        acts, dlang = record(model, SELF_LANGS); G[tag] = groups(acts, dlang, len(SELF_LANGS), keep=True); G[tag + "_lang"] = str(ck.get("lang"))
        if tag == "a":                                                       # average activity and firing rate at the first checkpoint (for activity-matched comparisons)
            STAT = {L: (acts[L].float().mean(0).numpy(), (acts[L] > 0).float().mean(0).numpy()) for L in LAYERS}
        del acts, model
        torch.cuda.empty_cache()
    rows = []
    for L in LAYERS:
        a, b = G["a"][L], G["b"][L]
        Za, Zb = a["Z"].float().to(DEVICE), b["Z"].float().to(DEVICE); Ra, Rb = a["R"].to(DEVICE), b["R"].to(DEVICE)
        S = (Za.T @ Zb) / Za.shape[0]
        _, perm = linear_sum_assignment(-S.double().cpu().numpy())
        pt_ = torch.tensor(perm, device=DEVICE)
        act = S[torch.arange(len(perm), device=DEVICE), pt_]
        A_ = Ra.clone(); A_.fill_diagonal_(0); B_ = Rb[pt_][:, pt_].clone(); B_.fill_diagonal_(0)
        conn = (A_ * B_).sum(1) / (A_.norm(dim=1) * B_.norm(dim=1) + 1e-12)
        pers = ((act + conn) / 2).cpu().numpy()
        valid = a["alive"] & b["alive"][perm]
        # order: reverse Cuthill-McKee on the thresholded graph (groups become blocks), self block first, dead neurons last
        rcm = np.asarray(reverse_cuthill_mckee(csr_matrix(a["A"].astype(np.int8)), symmetric_mode=True)).astype(np.int64)
        rank = np.where(~a["alive"][rcm], 2, np.where(a["self"][rcm], 0, 1))
        order = rcm[np.argsort(rank, kind="stable")]
        n_self, n_alive = int(a["self"].sum()), int(a["alive"].sum())
        if args.compact:
            for n_ in range(len(pers)):
                rows.append({"model": name, "family": FAMILY, "phase_a": pa, "phase_b": pb, "layer": L, "neuron": n_, "in_self": int(a["self"][n_]),
                             "alive": int(a["alive"][n_]), "valid": int(valid[n_]), "persistence": round(float(pers[n_]), 4),
                             "mean": round(float(STAT[L][0][n_]), 5), "fire": round(float(STAT[L][1][n_]), 4)})
            sv, tv = pers[valid & a["self"]], pers[valid & ~a["self"]]
            print(f"  {name} {pa}->{pb} {L}: self {n_self}/{n_alive} | persistence self {sv.mean() if len(sv) else float('nan'):.3f} vs task {tv.mean() if len(tv) else float('nan'):.3f}", flush=True)
            del Za, Zb, Ra, Rb, S, A_, B_
            torch.cuda.empty_cache()
            continue
        Rabs = Ra.abs().cpu().numpy()[np.ix_(order, order)]
        k = 8; nb = Rabs.shape[0] // k; small = Rabs[:nb * k, :nb * k].reshape(nb, k, nb, k).mean((1, 3))
        np.save(f"{OUT}/R_{name}_{L}.npy", small.astype(np.float16))
        for pos_, n_ in enumerate(order):
            rows.append({"model": name, "family": FAMILY, "phase_a": pa, "phase_b": pb, "layer": L, "position": pos_, "neuron": int(n_),
                         "in_self": bool(a["self"][n_]), "alive": bool(a["alive"][n_]), "valid": bool(valid[n_]), "persistence": float(pers[n_])})
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 8.2), gridspec_kw={"height_ratios": [4, 1]}, sharex=True)
            ax1.imshow(Rabs, cmap="Purples", vmin=0, vmax=0.6, aspect="auto", interpolation="nearest")
            ax1.plot([-0.5, n_self - 0.5, n_self - 0.5, -0.5, -0.5], [-0.5, -0.5, n_self - 0.5, n_self - 0.5, -0.5], color="#111111", lw=1.2)
            ax1.axvline(n_alive - 0.5, color="#888888", lw=0.6); ax1.axhline(n_alive - 0.5, color="#888888", lw=0.6)
            ax1.set_ylabel("neuron (reordered)")
            ax1.set_title(f"{name} | {L} | phase {pa} ({G['a_lang']}) | co-firing |correlation|; box = self ({n_self} of {n_alive} working neurons)", fontsize=8.5)
            xs = np.arange(len(order)); ok = valid[order]; isf = a["self"][order]
            ax2.scatter(xs[ok & isf], pers[order][ok & isf], s=1.5, color="#7A3FB0", label="self")
            ax2.scatter(xs[ok & ~isf], pers[order][ok & ~isf], s=1.5, color="#D96A12", label="task")
            ax2.set_ylabel(f"persistence\nphase {pa} -> {pb} ({G['b_lang']})", fontsize=8); ax2.set_xlabel("neuron (same order as above)")
            ax2.set_ylim(-0.1, 1.05); ax2.legend(frameon=False, fontsize=7, markerscale=5, loc="lower left")
            fig.tight_layout(); fig.savefig(f"{OUT}/matrix_{name}_{L}.png", dpi=150); plt.close(fig)
        except Exception as e:
            print(f"  (figure for {L} skipped: {e})")
        sv, tv = pers[valid & a["self"]], pers[valid & ~a["self"]]
        print(f"  {name} {L}: self {n_self}/{n_alive} | persistence self {sv.mean() if len(sv) else float('nan'):.3f} vs task {tv.mean() if len(tv) else float('nan'):.3f}", flush=True)
        del Za, Zb, Ra, Rb, S, A_, B_
        torch.cuda.empty_cache()
    if args.compact:
        pd.DataFrame(rows).to_csv(f"{OUT}/pers_p{pa:03d}.csv.gz", index=False)       # one file per transition (overwritten on a rerun)
    else:
        append_csv(f"{OUT}/persistence_per_neuron.csv", rows)
    print("done"); sys.exit()

# =============================================================================================== PART static
if args.part == "static":
    name, run, phases = parse_model(args.model)
    EV = [l for l in (args.eval_langs.split(",") if args.eval_langs else LANGS) if l in LANGS]
    VAL = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:args.n_val] for l in EV}
    FT_EN = read_flores("devtest", "en")[:args.n_chrf]; FT = {l: read_flores("devtest", l)[:args.n_chrf] for l in EV}
    conds = [c for c in args.conds.split(",") if c != "none"]
    def ev_loss(model):
        return {l: val_loss(model, VAL[l], l, masked=True) for l in EV}
    def ev_chrf(model):
        return {l: chrf(model, FT_EN, FT[l], l, masked=True)[0] for l in EV}
    for ph in phases:
        path = ckpt_path(run, ph)
        if path is None:
            print(f"  {run} phase {ph}: checkpoint not on this machine, skipped", flush=True); continue
        t0 = time.time(); model, ck = load_model(path); model.eval()
        info = neuron_info(model, SELF_LANGS)
        base, base_c = ev_loss(model), ev_chrf(model)
        print(f"{FAMILY} phase {ph} ({ck.get('lang')}): intact loss " + ", ".join(f"{l} {base[l]:.3f}" for l in EV)
              + " | chrF " + ", ".join(f"{l} {base_c[l]:.1f}" for l in EV), flush=True)
        print("  count-matched n per layer (n_max = smaller group): " + " ".join(
            f"{L}:{min(len(info[L]['sid']), len(info[L]['tid']))}" for L in LAYERS), flush=True)
        rows = []
        for draw in [int(d) for d in args.draws.split(",")]:
            sets, _ = pick_sets(info, draw, conds)
            for scope in LAYERS + ["all"]:
                for cond in conds:
                    S = [(L, ids) for L, ids in sets[cond] if scope == "all" or L == scope]
                    if not S:
                        continue
                    for mode in ["silence", "lesion"]:
                        if mode == "silence":
                            hooks = silence_hooks(model, S, info)
                        else:
                            saved = reinit(model, S, draw)
                        los = ev_loss(model); ch = ev_chrf(model) if scope == "all" else None
                        if mode == "silence":
                            for h in hooks:
                                h.remove()
                        else:
                            undo_reinit(saved)
                        for l in EV:
                            rows.append({"family": FAMILY, "run": run, "phase": ph, "scope": scope, "mode": mode, "cond": cond, "draw": draw, "lang": l,
                                         "n_neurons": sum(len(i) for _, i in S), "d_loss": los[l] - base[l],
                                         "d_chrf": (base_c[l] - ch[l]) if ch is not None else np.nan})
        check = ev_loss(model)                                               # must equal the intact loss again
        drift = max(abs(check[l] - base[l]) for l in EV)
        append_csv(f"{OUT}/static.csv", rows)
        try:
            r = pd.DataFrame(rows); r = r[r["scope"] == "all"].groupby(["mode", "cond"]).d_loss.mean().unstack("cond")
            print("  all 8 layers, loss rise (average of languages and draws):"); print(r.round(3).to_string())
        except Exception as e:
            print(f"  (short table skipped: {e})")
        print(f"  restored model check: loss differs from intact by {drift:.5f} (should be ~0) | {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done"); sys.exit()

# =============================================================================================== PART train
NEXT = {}
def load_task(task):
    if task == "next":
        l = NEXT["lang"]; rows = []
        with open(P(f"data/en-{l}/train.jsonl"), encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= 300_000:
                    break
        S = sp.encode([r["en"] for r in rows]); T = sp.encode([r["tgt"] for r in rows])
        tr = [(s_[:MAX_TOK - 1] + [EOS], [LANG_TAG[l]] + t[:MAX_TOK - 1], t[:MAX_TOK - 1] + [EOS]) for s_, t in zip(S, T)]
        test = [{"src": a, "tgt": b} for a, b in zip(read_flores("devtest", "en")[:200], read_flores("devtest", l)[:200])]
        return {"train": tr, "val": load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:1000], "test": test}
    def rd(split):
        with open(P(f"data/newtask/{task}/{split}.jsonl"), encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    def tok(rows):
        S = sp.encode([r["src"] for r in rows]); T = sp.encode([r["tgt"] for r in rows])
        return [(s[:MAX_TOK - 1] + [EOS], [BOS] + t[:MAX_TOK - 1], t[:MAX_TOK - 1] + [EOS]) for s, t in zip(S, T)]
    return {"train": tok(rd("train")[:300_000]), "val": tok(rd("val")), "test": rd("test")}

@torch.no_grad()
def greedy(model, srcs, max_new, bs=128):
    model.eval(); out = []
    with amp():
        for i in range(0, len(srcs), bs):
            src = pad_batch([sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in srcs[i:i + bs]])
            mem, sm = model.encode(src)
            ys = torch.full((src.size(0), 1), BOS, device=DEVICE); done = torch.zeros(src.size(0), dtype=torch.bool, device=DEVICE)
            for _ in range(max_new):
                nxt = model.decode(ys, mem, sm)[:, -1].float().argmax(-1); nxt[done] = PAD
                ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == EOS
                if done.all():
                    break
            for row in ys[:, 1:].tolist():
                row = row[:row.index(EOS)] if EOS in row else row
                out.append(sp.decode([t for t in row if t != PAD]))
    return out

def score(model, task, test):
    if task == "next":
        return chrf(model, [r["src"] for r in test], [r["tgt"] for r in test], NEXT["lang"], masked=True)[0]
    hyp = greedy(model, [r["src"] for r in test], MAX_TOK)
    return CHRF_METRIC.corpus_score(hyp, [[r["tgt"] for r in test]]).score

def train_one(start_name, start_sd, cfg, task, data, mode, cond, seed, S, info, ORIG, INTACT):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Seq2Seq(Config(**cfg)).to(DEVICE)
    if start_sd is not None:
        model.load_state_dict(start_sd)
    hooks, frozen = [], []
    if S:
        if mode == "silence":
            hooks = silence_hooks(model, S, info)
        elif mode == "lesion":
            reinit(model, S, seed)
        elif mode == "freeze":
            for L, ids in S:
                m = mlp_mod(model, L); ii = torch.as_tensor(ids, device=DEVICE, dtype=torch.long)
                frozen.append((m.fc, ii, m.fc.weight.data[ii].clone(), m.fc.bias.data[ii].clone()))
    opt = make_optimizer(model, args.lr, 0.1)
    evals = [s for s in EVAL_AT if s <= args.steps]
    if args.steps not in evals:
        evals.append(args.steps)
    chrf_at = {0, args.steps}
    nl = NEXT.get("lang") if task == "next" else None
    n_neu = sum(len(i) for _, i in S)
    rows, orows = [], []
    def evaluate(step):
        vl = val_loss(model, data["val"], nl, masked=nl is not None)
        sc = score(model, task, data["test"]) if step in chrf_at else np.nan
        rows.append({"family": FAMILY, "start": start_name, "task": task, "mode": mode, "cond": cond, "seed": seed, "step": step,
                     "val_loss": vl, "score": sc, "n_neurons": n_neu})
        model.train()
    before = {l: val_loss(model, ORIG[l], l, masked=True) for l in ORIG}
    evaluate(0)
    step, ep = 0, 0
    model.train()
    while step < args.steps:
        for bidx in make_batches(data["train"], args.max_tokens, seed=seed * 1000 + ep):
            src, tin, tout = collate(data["train"], bidx)
            lr = args.lr * min(1.0, (step + 1) / 100)                    # 100 warmup steps, then constant
            for gp in opt.param_groups:
                gp["lr"] = lr
            with amp():
                if nl is None:
                    _, loss = model(src, tin, tout)
                else:                                                    # same script-masked loss as normal training
                    logits, _ = model(src, tin)
                    loss = masked_ce(logits.view(-1, logits.size(-1)), tout.view(-1), ALLOWED[nl])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for fc, ii, w, b in frozen:                                  # freeze: put the incoming weights back after every step
                fc.weight.data[ii] = w; fc.bias.data[ii] = b
            step += 1
            if step in evals:
                evaluate(step)
            if step >= args.steps:
                break
        ep += 1
    after = {l: val_loss(model, ORIG[l], l, masked=True) for l in ORIG}
    for l in ORIG:
        orows.append({"family": FAMILY, "start": start_name, "task": task, "mode": mode, "cond": cond, "seed": seed, "lang": l,
                      "intact": INTACT[l], "before": before[l], "after": after[l]})
    for h in hooks:
        h.remove()
    del model, opt
    torch.cuda.empty_cache()
    return rows, orows

if args.part == "train":
    conds = args.conds.split(",")
    if args.model == "fresh":
        start_name, start_sd, info = "fresh", None, None
        cfg = asdict(Config()); conds = ["none"]; ORIG, INTACT = {}, {}
    else:
        start_name, run, phases = parse_model(args.model)
        path = ckpt_path(run, phases[0])
        if path is None:
            print(f"{run} phase {phases[0]}: checkpoint not on this machine, nothing to do"); sys.exit()
        m0, ck0 = load_model(path); cfg = ck0["cfg"]; m0.eval()
        start_sd = {k: v.detach().clone() for k, v in m0.state_dict().items()}
        print(f"start model: {start_name} = {run} phase {phases[0]} (last trained on {ck0.get('lang')}) | languages {', '.join(LANGS)}", flush=True)
        t0 = time.time(); info = neuron_info(m0, SELF_LANGS)
        for L in LAYERS:
            d = info[L]
            print(f"  {L}: working {d['nalive']}, self {len(d['sid'])}, task {len(d['tid'])} -> count-matched n_max {min(len(d['sid']), len(d['tid']))}")
        print(f"  self found in {time.time() - t0:.0f}s", flush=True)
        ORIG = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:300] for l in LANGS}
        INTACT = {l: val_loss(m0, ORIG[l], l, masked=True) for l in ORIG}
        if args.task == "next":
            if ck0.get("lang") not in LANGS:
                print(f"this checkpoint's language is {ck0.get('lang')}: no 'next' language, nothing to do"); sys.exit()
            NEXT["lang"] = args.next_lang if args.next_lang in LANGS else LANGS[(LANGS.index(ck0["lang"]) + 1) % len(LANGS)]
            print(f"  next language = {NEXT['lang']} (phase {phases[0] + 1} of the normal schedule)", flush=True)
        del m0
        torch.cuda.empty_cache()
    mode_of = lambda c: "none" if c == "none" else args.mode
    done = set()
    if os.path.exists(f"{OUT}/curves.csv"):
        c = pd.read_csv(f"{OUT}/curves.csv")
        c = c[c.step == args.steps]                                       # only runs that reached the last step count as done
        done = set(zip(c["start"].astype(str), c["mode"].astype(str), c["cond"].astype(str), c["seed"].astype(int)))
    seeds = [int(s) for s in args.seeds.split(",")]
    todo = [(c, s) for s in seeds for c in conds if (start_name, mode_of(c), c, s) not in done]
    print(f"runs to do in this call: {len(todo)} (mode {args.mode}; conditions {conds}; seeds {seeds}; finished runs are skipped)", flush=True)
    data = load_task(args.task) if todo else None
    t_all, n_done = time.time(), 0
    for seed in seeds:
        sets = None
        for cond in conds:
            if (start_name, mode_of(cond), cond, seed) in done:
                continue
            if sets is None and info is not None:
                sets, prow = pick_sets(info, seed, conds)
                append_csv(f"{OUT}/picks.csv", [dict(r, family=FAMILY, start=start_name, seed=seed) for r in prow])
            S = [] if cond == "none" else sets[cond]
            if cond != "none" and not S:
                print(f"  {cond}: no layer had both groups large enough, skipped"); continue
            t0 = time.time()
            rows, orows = train_one(start_name, start_sd, cfg, args.task, data, mode_of(cond), cond, seed, S, info, ORIG, INTACT)
            append_csv(f"{OUT}/curves.csv", rows); append_csv(f"{OUT}/others.csv", orows)
            n_done += 1; per = (time.time() - t_all) / n_done
            r0, r1 = rows[0], rows[-1]
            print(f"  {start_name} | {args.task} | {mode_of(cond):7s} | {cond:9s} | seed {seed} | {r0['n_neurons']} neurons | loss {r0['val_loss']:.3f} -> {r1['val_loss']:.3f} | "
                  f"chrF {r0['score']:.1f} -> {r1['score']:.1f} | {time.time() - t0:.0f}s | left in this call: {len(todo) - n_done} runs, about {per * (len(todo) - n_done) / 60:.0f} min", flush=True)
    print("done")
