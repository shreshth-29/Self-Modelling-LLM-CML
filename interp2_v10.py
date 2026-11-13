"""interp2_v10.py: what does the self DO? English -> Hindi only, no training. Uses the self found by interp_v10.py
(analysis/v10/interp/neurons_<model>.csv: in_self, alive, act_mean, fire_rate, w_out) and its token tables.

  h2  WHEN DOES IT FIRE?   every neuron z-scored (its own mean 0, spread 1). Average z-activity of self vs task neurons
                           by token kind (punctuation, number, the 30 most common Hindi words, other Hindi word start,
                           Hindi word-piece, Latin), position, how common the token is, and (decoder) how confident the
                           model is. + share of each neuron's activity explained by each of those (eta squared).
                           + top-activating tokens (with context) for sample self and task neurons.
  h3  WHAT DOES IT WRITE?  decoder neurons: output weights -> change in every Hindi word's score. Top promoted /
                           suppressed tokens per neuron; which KINDS of tokens the self vs task neurons push; how alike
                           the self neurons' outputs are (do they push one shared direction?).
  h4  WHAT BREAKS WITHOUT IT?  switch off 10% of the self (each neuron = its own average) vs the same number of
                           look-alike task neurons vs random task neurons, 3 draws, in 3 places: all 8 layers / the 4
                           encoder layers only / the 4 decoder layers only. Per token: loss change, confidence (entropy)
                           change, did the top choice flip, KL. Split by token kind, position, frequency, confidence.
  h5  POPULATION PICTURE   how many dimensions the self's activity really uses (participation ratio) vs task neurons,
                           and what its top 3 directions track (position, confidence, loss, frequency, token kind).

  CML_LANGSET=distant python -u interp2_v10.py --gpu 0 --model cont40=v4b_continual_s0:40 --parts h2,h3,h5
  python -u interp2_v10.py --summarize
"""
import os, sys, json, time, argparse, glob
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--model", default="")
ap.add_argument("--parts", default="h2,h3,h4,h5")
ap.add_argument("--n_sent", type=int, default=997)
ap.add_argument("--dose", type=float, default=0.10)
ap.add_argument("--draws", default="0,1,2")
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
IN = os.path.join(REPO, "analysis", "v10", "interp"); OUT = os.path.join(REPO, "analysis", "v10", "interp2"); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
LAYERS = [f"enc{j}" for j in range(4)] + [f"dec{j}" for j in range(4)]
KINDS = ["punct", "number", "common word", "Hindi word start", "Hindi word-piece", "Latin", "other"]
EN_KINDS = ["punct", "number", "common word", "English word start", "English word-piece", "other"]
POSB = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 999, "21+")]

def ttype(i):
    p = sp.id_to_piece(int(i)).replace("\u2581", "")
    if sp.is_control(int(i)) or p == "": return "other"
    if all(not ch.isalnum() for ch in p): return "punct"
    if any(ch.isdigit() for ch in p): return "digit"
    if any("\u0900" <= ch <= "\u097F" for ch in p): return "hindi"
    if any(ch.isascii() and ch.isalpha() for ch in p): return "latin"
    return "other"

def add_kinds(df, side, common):
    p = df.piece.astype(str)
    ws = p.str.startswith("▁")
    k = np.full(len(df), "other", dtype=object)
    t = df["type"].values
    if side == "dec":
        k[(t == "hindi") & ws] = "Hindi word start"; k[(t == "hindi") & ~ws] = "Hindi word-piece"; k[t == "latin"] = "Latin"
    else:
        k[(t == "latin") & ws] = "English word start"; k[(t == "latin") & ~ws] = "English word-piece"
    k[t == "punct"] = "punct"; k[t == "digit"] = "number"
    k[df.token.isin(common).values & ws.values] = "common word"
    df["kind"] = k
    df["pos_b"] = pd.cut(df.pos, [-1] + [b for _, b, _ in POSB], labels=[l for _, _, l in POSB]).astype(str)
    df["freq_b"] = pd.qcut(df.log_freq.rank(method="first"), 5, labels=["rarest 20%", "rare", "middle", "common", "commonest 20%"]).astype(str)
    if "entropy" in df:
        df["conf_b"] = pd.qcut(df.entropy.rank(method="first"), 3, labels=["confident", "medium", "unsure"]).astype(str)
    return df

# =============================================================================== SUMMARY
def summarize():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = []; say = lambda s: (print(s), R.append(s))
    if os.path.exists(f"{OUT}/h2_by_kind.csv"):
        h = pd.read_csv(f"{OUT}/h2_by_kind.csv")
        say("=== H2: when does the self fire? average z-activity, self minus task neurons (positive = self fires MORE on this kind) ===")
        for fac in ["kind", "pos_b", "freq_b", "conf_b"]:
            q = h[h.factor == fac]
            if len(q):
                say(f"\n-- by {fac} (rows: model, side; columns: category)")
                say(q.pivot_table(index=["model", "side"], columns="category", values="self_minus_task").round(3).to_string())
        e = pd.read_csv(f"{OUT}/h2_eta.csv")
        say("\n-- share of each neuron's activity explained by each fact (eta squared, average), self vs task")
        say(e.pivot_table(index=["model", "side", "fact"], columns="group", values="eta").round(3).to_string())
        ms = list(h.model.unique())
        for side, kinds in [("dec", KINDS), ("enc", EN_KINDS)]:
            fig, axs = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.8), squeeze=False, sharey=True)
            for ax, m in zip(axs[0], ms):
                q = h[(h.model == m) & (h.side == side) & (h.factor == "kind")].pivot_table(index="category", columns="layer", values="self_minus_task")
                q = q.reindex([k for k in kinds if k in q.index])
                im = ax.imshow(q.values, cmap="PuOr", vmin=-.4, vmax=.4, aspect="auto")
                ax.set_xticks(range(q.shape[1])); ax.set_xticklabels(q.columns, fontsize=8); ax.set_yticks(range(len(q))); ax.set_yticklabels(q.index, fontsize=8)
                ax.set_title(f"{m} ({'Hindi output' if side == 'dec' else 'English input'})", fontsize=9)
            fig.colorbar(im, ax=axs[0].tolist(), label="self minus task\n(average z-activity)", shrink=.8)
            fig.savefig(f"{OUT}/h2_kind_{side}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    if os.path.exists(f"{OUT}/h3_summary.csv"):
        s3 = pd.read_csv(f"{OUT}/h3_summary.csv")
        say("\n=== H3: what do decoder neurons write? share of each neuron's top-20 promoted tokens by kind (self vs task) ===")
        say(s3.pivot_table(index=["model", "layer", "group"], columns="kind", values="share").round(3).to_string())
        a3 = pd.read_csv(f"{OUT}/h3_alike.csv")
        say("\n-- how alike are the output directions? average |cosine| between neurons' output weights (higher = push the same way)")
        say(a3.pivot_table(index=["model"], columns=["layer", "group"], values="mean_abs_cos").round(3).to_string())
    if os.path.exists(f"{OUT}/h4_tokens.csv"):
        t4 = pd.read_csv(f"{OUT}/h4_tokens.csv")
        def wmean(df, by, cols):
            d = df.copy()
            for c in cols: d[c] = d[c] * d.n
            g = d.groupby(by)[cols + ["n"]].sum()
            for c in cols: g[c] = g[c] / g.n
            return g[cols]
        say("\n=== H4: what breaks when 10% is switched off? average over tokens and 3 draws ===")
        say(wmean(t4, ["model", "scope", "group"], ["d_loss", "d_entropy", "flip", "kl"]).round(4).to_string())
        for fac in ["kind", "pos_b", "freq_b", "conf_b"]:
            say(f"\n-- loss rise by {fac}: self / look-alike (ratio > 1 = the self matters more for this kind), scope = all 8 layers")
            q = wmean(t4[t4.scope == "all"], ["model", fac, "group"], ["d_loss"]).d_loss.unstack("group")
            q["self / look-alike"] = q["self"] / q["lookalike"]
            say(q.round(4).to_string())
        ms = list(t4.model.unique())
        fig, axs = plt.subplots(1, len(ms), figsize=(5 * len(ms), 4.2), squeeze=False, sharey=True)
        COL = {"self": "#7A3FB0", "lookalike": "#D96A12", "random": "#9CA3AF"}
        for ax, m in zip(axs[0], ms):
            q = wmean(t4[(t4.model == m) & (t4.scope == "all")], ["kind", "group"], ["d_loss"]).d_loss.unstack("group").reindex([k for k in KINDS if k in set(t4.kind)])
            for j, g in enumerate(["self", "lookalike", "random"]):
                ax.bar(np.arange(len(q)) + (j - 1) * .27, q[g], .26, color=COL[g], label=g if m == ms[0] else None)
            ax.set_xticks(range(len(q))); ax.set_xticklabels(q.index, rotation=35, ha="right", fontsize=8); ax.set_title(m, fontsize=10)
        axs[0, 0].set_ylabel("loss rise on this kind of Hindi token\n(10% of all 8 layers switched off)"); axs[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/h4_by_kind.png", dpi=150); plt.close(fig)
    if os.path.exists(f"{OUT}/h5_dims.csv"):
        d5 = pd.read_csv(f"{OUT}/h5_dims.csv")
        say("\n=== H5: how many dimensions does the activity really use? (participation ratio; per 100 neurons so self and task compare fairly) ===")
        say(d5.pivot_table(index=["model", "group"], columns="layer", values="pr_per_100").round(2).to_string())
        say("\n-- share of variance in the top 1 / top 3 directions")
        say(d5.pivot_table(index=["model", "group"], columns="layer", values="top1").round(3).to_string())
        c5 = pd.read_csv(f"{OUT}/h5_pc_corr.csv")
        say("\n-- what the self's top direction tracks: |correlation| of PC1 with each fact (self vs task)")
        say(c5[c5.pc == 1].pivot_table(index=["model", "layer"], columns=["fact", "group"], values="abs_corr").round(2).to_string())
    open(f"{OUT}/report.txt", "w").write("\n".join(R))
    print("figures + report in", OUT)

if args.summarize:
    summarize(); sys.exit()

# =============================================================================== GPU
from common_v8 import *
from scipy.stats import skew
name, spec = args.model.split("=")
if spec == "random":
    torch.manual_seed(0); model = Seq2Seq(Config()).to(DEVICE)
else:
    run, ph = spec.split(":"); model, _ = load_model(glob.glob(P(f"runs/{run}/ckpt/p{int(ph):03d}_*.pt"))[0])
model.eval()
mlp = lambda L: (model.enc_blocks if L.startswith("enc") else model.dec_blocks)[int(L[3])].mlp
NEU = pd.read_csv(f"{IN}/neurons_{name}.csv")
SELFM = {L: (q.sort_values("neuron").alive.values, q.sort_values("neuron").in_self.values) for L, q in NEU.groupby("layer")}
TOK = {"enc": pd.read_csv(f"{IN}/tokens_enc_{name}.csv"), "dec": pd.read_csv(f"{IN}/tokens_dec_{name}.csv")}
# 30 most common Hindi / English word-start tokens in the training data
cnt = {"hi": np.zeros(V), "en": np.zeros(V)}
with open(P("data/en-hi/train.jsonl"), encoding="utf-8") as f:
    E_, H_ = [], []
    for i, line in enumerate(f):
        r = json.loads(line); E_.append(r["en"]); H_.append(r["tgt"])
        if i >= 100_000: break
for ids in sp.encode(H_): np.add.at(cnt["hi"], ids, 1)
for ids in sp.encode(E_): np.add.at(cnt["en"], ids, 1)
def common_words(c):
    ws = [i for i in np.argsort(-c) if sp.id_to_piece(int(i)).startswith("▁") and any(ch.isalpha() for ch in sp.id_to_piece(int(i)))]
    return set(int(i) for i in ws[:30])
COMMON = {"dec": common_words(cnt["hi"]), "enc": common_words(cnt["en"])}
print("30 most common Hindi words:", " ".join(sp.id_to_piece(i).replace("▁", "") for i in sorted(COMMON["dec"], key=lambda i: -cnt["hi"][i])), flush=True)
for s in TOK:
    TOK[s] = add_kinds(TOK[s], s, COMMON[s])

@torch.no_grad()
def record():
    en = read_flores("dev", "en")[:args.n_sent]; tg = read_flores("dev", "hi")[:args.n_sent]
    S_all = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in en]; T_all = [sp.encode(s)[:MAX_TOK - 1] for s in tg]
    cur = {}; hooks = [mlp(L).act.register_forward_hook(lambda m, i, o, k=L: cur.__setitem__(k, o)) for L in LAYERS]
    post = {L: [] for L in LAYERS}
    for s in range(0, len(S_all), 32):
        S_ = pad_batch(S_all[s:s + 32]); mem, sm = model.encode(S_); sk = (S_ != PAD) & (S_ != EOS)
        for L in LAYERS[:4]: post[L].append(cur[L][sk].half().cpu())
        tin = pad_batch([[LANG_TAG["hi"]] + t for t in T_all[s:s + 32]]); model.decode(tin, mem, sm); tk = tin != PAD
        for L in LAYERS[4:]: post[L].append(cur[L][tk].half().cpu())
    for h in hooks: h.remove()
    return {L: torch.cat(v) for L, v in post.items()}, S_all, T_all

t0 = time.time()
POST, S_ALL, T_ALL = record()
assert POST["enc0"].shape[0] == len(TOK["enc"]) and POST["dec0"].shape[0] == len(TOK["dec"]), "token tables do not match: rerun interp_v10.py first"
print(f"recorded in {time.time() - t0:.0f}s", flush=True)
side = lambda L: "enc" if L.startswith("enc") else "dec"

def zscore(X):
    X = X.float(); return (X - X.mean(0)) / (X.std(0) + 1e-6)

def eta2(x, groups, min_n=5):
    tot = x.var(0, unbiased=False) + 1e-12
    gs, inv, cnt_ = np.unique(groups, return_inverse=True, return_counts=True)
    keep = cnt_[inv] >= min_n; xg = x[torch.tensor(keep)]; inv = torch.tensor(inv[keep])
    sums = torch.zeros(len(gs), x.shape[1]).index_add_(0, inv, xg); n = torch.zeros(len(gs)).index_add_(0, inv, torch.ones(len(inv)))
    means = sums / n.clamp(min=1)[:, None]; mu = xg.mean(0)
    return ((((means - mu) ** 2) * n[:, None]).sum(0) / len(inv) / tot).numpy()

def app(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# ---------------------------------------------------------------- H2
if "h2" in args.parts:
    rows, erows, ex = [], [], []
    for L in LAYERS:
        alive, sm = SELFM[L]; T = TOK[side(L)]; Z = zscore(POST[L])
        sidx, tidx = torch.tensor(np.nonzero(alive & sm)[0]), torch.tensor(np.nonzero(alive & ~sm)[0])
        zs, zt = Z[:, sidx].mean(1).numpy(), Z[:, tidx].mean(1).numpy()
        for fac in ["kind", "pos_b", "freq_b"] + (["conf_b"] if side(L) == "dec" else []):
            for cat, q in T.groupby(fac):
                ii = q.index.values
                rows.append({"model": name, "layer": L, "side": side(L), "factor": fac, "category": cat, "n_tokens": len(ii),
                             "self": zs[ii].mean(), "task": zt[ii].mean(), "self_minus_task": zs[ii].mean() - zt[ii].mean()})
        for fact in ["token", "kind", "pos_b", "freq_b"] + (["conf_b"] if side(L) == "dec" else []):
            et = eta2(Z, T[fact].astype(str).values, min_n=20 if fact == "token" else 5)
            for g, idx in [("self", sidx), ("task", tidx)]:
                erows.append({"model": name, "layer": L, "side": side(L), "fact": fact, "group": g, "eta": float(np.nanmean(et[idx.numpy()]))})
        if L in ("enc2", "dec2"):                     # top-activating examples
            g = np.random.default_rng(0)
            for grp, idx in [("self", sidx), ("task", tidx)]:
                for n_ in g.choice(idx.numpy(), 6, replace=False):
                    top = torch.topk(POST[L][:, n_].float(), 8).indices.numpy()
                    for t in top:
                        r = T.iloc[t]; toks = (S_ALL[r.sent][:-1] if side(L) == "enc" else T_ALL[r.sent])
                        lo = max(0, r.pos - 6); ctx = sp.decode(toks[lo:r.pos]) + " [[" + sp.id_to_piece(int(r.token)).replace("▁", " ") + "]] " + sp.decode(toks[r.pos + 1:r.pos + 5])
                        ex.append({"model": name, "layer": L, "group": grp, "neuron": int(n_), "activity": float(POST[L][t, n_]), "kind": r.kind, "context": ctx})
        print(f"  H2 {L}", flush=True)
    app(f"{OUT}/h2_by_kind.csv", rows); app(f"{OUT}/h2_eta.csv", erows); app(f"{OUT}/h2_examples.csv", ex)

# ---------------------------------------------------------------- H3
if "h3" in args.parts:
    E = model.tok_emb.weight.detach().float(); gamma = model.dec_ln.weight.detach().float()
    hi_ok = np.nonzero(cnt["hi"] > 0)[0]
    tk = pd.DataFrame({"token": hi_ok}); tk["piece"] = [sp.id_to_piece(int(i)) for i in hi_ok]
    tk["type"] = [ttype(i) for i in hi_ok]; tk["log_freq"] = np.log(cnt["hi"][hi_ok] + 1); tk["pos"] = 0
    tk = add_kinds(tk, "dec", COMMON["dec"])
    srows, arows, toprows = [], [], []
    for L in LAYERS[4:]:
        alive, sm = SELFM[L]
        W = mlp(L).proj.weight.detach().float() * gamma[:, None]       # d x 2560
        le = (E[torch.tensor(hi_ok, device=E.device)] @ W).T.cpu().numpy()   # 2560 x hindi vocab
        top20 = np.argsort(-le, axis=1)[:, :20]
        Wn = (W / (W.norm(dim=0, keepdim=True) + 1e-9)).cpu()
        for g, msk in [("self", alive & sm), ("task", alive & ~sm)]:
            idx = np.nonzero(msk)[0]
            kinds = tk.kind.values[top20[idx]].ravel()
            for k in KINDS:
                srows.append({"model": name, "layer": L, "group": g, "kind": k, "share": float((kinds == k).mean()), "base": float((tk.kind == k).mean())})
            sub = Wn[:, torch.tensor(np.random.default_rng(0).choice(idx, min(400, len(idx)), replace=False))]
            C = (sub.T @ sub).abs(); n_ = C.shape[0]
            arows.append({"model": name, "layer": L, "group": g, "mean_abs_cos": float((C.sum() - n_) / (n_ * (n_ - 1)))})
            for n_i in np.random.default_rng(1).choice(idx, 5, replace=False):
                o = np.argsort(-le[n_i]); toprows.append({"model": name, "layer": L, "group": g, "neuron": int(n_i),
                    "promotes": " ".join(tk.piece.values[o[:10]]), "suppresses": " ".join(tk.piece.values[o[-10:]])})
        print(f"  H3 {L}", flush=True)
    app(f"{OUT}/h3_summary.csv", srows); app(f"{OUT}/h3_alike.csv", arows); app(f"{OUT}/h3_top_tokens.csv", toprows)

# ---------------------------------------------------------------- H4
if "h4" in args.parts and spec != "random":
    def matched_pick(feat, self_ids, task_ids, g):
        F_ = (feat - feat.mean(0)) / (feat.std(0) + 1e-9); free = np.ones(len(task_ids), bool); out = []
        for s in g.permutation(self_ids):
            d = np.linalg.norm(F_[task_ids] - F_[s], axis=1); d[~free] = np.inf
            cand = np.argsort(d)[:3]; cand = cand[np.isfinite(d[cand])]; c = g.choice(cand); free[c] = False; out.append(task_ids[c])
        return np.array(out)
    def per_token(hooks_spec):
        hooks = []
        for L, ids, vals in hooks_spec:
            ii = torch.as_tensor(ids, device=DEVICE); vv = torch.as_tensor(vals, device=DEVICE, dtype=torch.float32)
            def hk(m, i, o, ii=ii, vv=vv):
                o = o.clone(); o[..., ii] = vv.to(o.dtype); return o
            hooks.append(mlp(L).act.register_forward_hook(hk))
        out = []
        with torch.no_grad():
            for s in range(0, len(S_ALL), 32):
                S_ = pad_batch(S_ALL[s:s + 32]); mem, sm_ = model.encode(S_)
                tin = pad_batch([[LANG_TAG["hi"]] + t for t in T_ALL[s:s + 32]]); tout = pad_batch([t + [EOS] for t in T_ALL[s:s + 32]])
                lp = (model.decode(tin, mem, sm_).float() + NEG["hi"]).log_softmax(-1)
                tk_ = tin != PAD
                out.append(lp[tk_].cpu())
        for h in hooks: h.remove()
        return torch.cat(out)
    T = TOK["dec"]; tgt = torch.tensor(T.next_token.values)
    base = per_token([]); bl = -base.gather(1, tgt[:, None])[:, 0]; bp = base.exp()
    bent = -(bp * base.masked_fill(bp == 0, 0)).sum(1); btop = base.argmax(1)
    MEAN = {L: POST[L].float().mean(0).numpy() for L in LAYERS}
    rows = []
    for draw in [int(d) for d in args.draws.split(",")]:
        g = np.random.default_rng(3000 + draw); sets = {"self": [], "lookalike": [], "random": []}
        for L in LAYERS:
            q = NEU[NEU.layer == L].sort_values("neuron"); alive, sm = SELFM[L]
            sid, tid = np.nonzero(alive & sm)[0], np.nonzero(alive & ~sm)[0]; n = int(round(args.dose * alive.sum()))
            if n < 5 or n > len(sid) or n > len(tid): continue
            ss = g.choice(sid, n, replace=False); feat = q[["act_mean", "fire_rate", "w_out"]].values
            for grp, ids in [("self", ss), ("lookalike", matched_pick(feat, ss, tid, g)), ("random", g.choice(tid, n, replace=False))]:
                sets[grp].append((L, ids, MEAN[L][ids]))
        for scope in ["all", "enc", "dec"]:
            for grp, S in sets.items():
                spec_ = [x for x in S if scope == "all" or x[0].startswith(scope)]
                lp = per_token(spec_); l = -lp.gather(1, tgt[:, None])[:, 0]; p_ = lp.exp()
                ent = -(p_ * lp.masked_fill(p_ == 0, 0)).sum(1)
                kl = (bp * (base - lp).masked_fill(bp == 0, 0)).sum(1)
                d = T[["kind", "pos_b", "freq_b", "conf_b"]].copy()
                d["d_loss"] = (l - bl).numpy(); d["d_entropy"] = (ent - bent).numpy(); d["flip"] = (lp.argmax(1) != btop).numpy(); d["kl"] = kl.numpy()
                d["model"] = name; d["draw"] = draw; d["scope"] = scope; d["group"] = grp
                rows.append(d.groupby(["model", "draw", "scope", "group", "kind", "pos_b", "freq_b", "conf_b"]).agg(
                    d_loss=("d_loss", "mean"), d_entropy=("d_entropy", "mean"), flip=("flip", "mean"), kl=("kl", "mean"), n=("d_loss", "size")).reset_index())
                print(f"  H4 draw {draw} | {scope:3s} | {grp:9s}: loss {d.d_loss.mean():+.4f}, entropy {d.d_entropy.mean():+.4f}, top-1 flipped {100 * d.flip.mean():.2f}%", flush=True)
    R4 = pd.concat(rows)                                   # grouped rows; column n = number of tokens in the group
    R4.to_csv(f"{OUT}/h4_tokens.csv", mode="a", header=not os.path.exists(f"{OUT}/h4_tokens.csv"), index=False)

# ---------------------------------------------------------------- H5
if "h5" in args.parts:
    drows, crows = [], []
    for L in LAYERS:
        alive, sm = SELFM[L]; T = TOK[side(L)]
        g = np.random.default_rng(0); ii = torch.tensor(np.sort(g.choice(POST[L].shape[0], min(20000, POST[L].shape[0]), replace=False)))
        Z = zscore(POST[L][ii]); Tt = T.iloc[ii.numpy()]
        facts = {"position": Tt.pos.values, "relative position": Tt.rel_pos.values, "how common": Tt.log_freq.values,
                 "punct": (Tt.kind == "punct").values, "common word": (Tt.kind == "common word").values, "word start": Tt.word_start.values}
        if side(L) == "dec":
            facts.update({"confidence (entropy)": Tt.entropy.values, "loss": Tt.loss.values})
        for grp, msk in [("self", alive & sm), ("task", alive & ~sm)]:
            idx = torch.tensor(np.nonzero(msk)[0]); X = Z[:, idx].to(DEVICE)
            C = (X.T @ X) / X.shape[0]; ev = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)
            pr = float(ev.sum() ** 2 / (ev ** 2).sum())
            drows.append({"model": name, "layer": L, "group": grp, "n_neurons": len(idx), "pr": pr, "pr_per_100": 100 * pr / len(idx),
                          "top1": float(ev[0] / ev.sum()), "top3": float(ev[:3].sum() / ev.sum())})
            evals, evecs = torch.linalg.eigh(C); pcs = (X @ evecs[:, -3:].flip(1)).cpu().numpy()
            for k in range(3):
                for f_, v in facts.items():
                    v = v.astype(float); r = np.corrcoef(pcs[:, k], v)[0, 1] if v.std() > 0 else np.nan
                    crows.append({"model": name, "layer": L, "group": grp, "pc": k + 1, "fact": f_, "abs_corr": abs(r)})
        print(f"  H5 {L}", flush=True)
    app(f"{OUT}/h5_dims.csv", drows); app(f"{OUT}/h5_pc_corr.csv", crows)
print("done")
