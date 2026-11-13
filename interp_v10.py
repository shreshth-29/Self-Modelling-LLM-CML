"""interp_v10.py: deep interpretability of the self, ONE language (English -> Hindi), no training.

Parts (run all with --parts h0,h6,h1):
  h0  RECORD   FLORES dev, English -> Hindi, teacher forcing. For every token: activity (after GELU) and pre-activation
               (before GELU) of all 2,560 neurons in all 8 MLP layers, plus token facts (token, type, word start, how
               common, position, the model's loss / confidence / top-1 correct). Saves tokens_<model>.csv.
  h6  IS IT REAL?  Size of the largest co-firing group (the "self") as the threshold tau goes 0.05 -> 0.5, for
               the real activity and 3 nulls:
                 rotated  : same activity seen through a random rotation of the 2,560 neurons (keeps all overall
                            correlation, removes the meaning of individual neurons). Self here => it is not about neurons.
                 shuffled : each neuron's activity shuffled over tokens (keeps each neuron, breaks who fires with whom)
                 same-density random graph: a random graph with exactly as many links (pure chance).
               Run on trained models AND on an untrained (random weights) model.
  h1  IS IT SOMETHING KNOWN?  Per-neuron scores for known neuron families, then: are self neurons over-represented?
               scores: dead, massive activation, weight sizes, input bias, pre-activation skew/kurtosis (universal-
               neuron signature), position-driven, token-driven, and for DECODER neurons what they write to the output:
               entropy-neuron score (share of output weight in the directions that map to no word), token-frequency
               score, prediction/suppression (skew of the output effect). Continual models also: how language-specific
               each decoder neuron is (hi vs zh vs ar), to compare with the "all-shared neurons" papers.

The self here = largest co-firing group at tau 0.2 on English->Hindi only (Adi: stick to one language).

  CML_LANGSET=distant python -u interp_v10.py --gpu 0 --model hindi45=v4b_only_hi_s0:45
  CML_LANGSET=distant python -u interp_v10.py --gpu 0 --model random=random
"""
import os, sys, json, time, argparse, glob
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--model", required=False, default="", help="name=run:phase  or  random=random")
ap.add_argument("--parts", default="h0,h6,h1")
ap.add_argument("--n_sent", type=int, default=997)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "v10", "interp"); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
TAUS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
LAYERS = [f"enc{j}" for j in range(4)] + [f"dec{j}" for j in range(4)]

# =============================================================================== SUMMARY (CPU)
SCORES = [  # column, plain name, which neurons
    ("fire_rate", "how often it fires (> 0)", "all"), ("act_mean", "average activity", "all"),
    ("w_in", "input weight size", "all"), ("w_out", "output weight size", "all"), ("b_in", "input bias (negative = harder to fire)", "all"),
    ("pre_kurt", "pre-activation kurtosis (universal-neuron sign)", "all"), ("pre_skew", "pre-activation skew", "all"),
    ("massive", "peak / typical activity (massive activation)", "all"), ("pos_eta", "share of activity explained by position", "all"),
    ("tok_eta", "share of activity explained by the current token", "all"),
    ("null_frac", "entropy-neuron score: output weight in no-word directions", "dec"),
    ("freq_corr", "token-frequency score: output effect vs log frequency", "dec"),
    ("out_skew", "output effect skew (+ predicts / - suppresses words)", "dec"), ("out_kurt", "output effect kurtosis", "dec"),
    ("lang_eta", "share of activity explained by language (hi/zh/ar)", "dec")]

def summarize():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = []; say = lambda s: (print(s), R.append(s))
    # ---- h6
    if os.path.exists(f"{OUT}/h6_giant.csv"):
        g = pd.read_csv(f"{OUT}/h6_giant.csv")
        say("=== H6: is the self real? size of the largest co-firing group (% of working neurons), average of the 8 layers ===")
        say(g.groupby(["model", "condition", "tau"]).giant_pct.mean().unstack("tau").round(1).to_string())
        say("\nat tau 0.2, per layer:")
        say(g[g.tau == 0.2].pivot_table(index=["model", "condition"], columns="layer", values="giant_pct").round(1).to_string())
        models = list(g.model.unique())
        fig, axs = plt.subplots(len(models), 8, figsize=(22, 2.9 * len(models)), squeeze=False, sharey=True)
        COLS = {"real": "#7A3FB0", "rotated": "#D96A12", "shuffled": "#9CA3AF", "same-density random graph": "#1B263B"}
        for i, m in enumerate(models):
            for j, L in enumerate(LAYERS):
                ax = axs[i, j]
                for cond, col in COLS.items():
                    q = g[(g.model == m) & (g.layer == L) & (g.condition == cond)].sort_values("tau")
                    ax.plot(q.tau, q.giant_pct, marker="o", ms=3, color=col, label=cond if (i == 0 and j == 0) else None)
                ax.axvline(0.2, color="k", lw=.6, ls=":")
                ax.set_title(f"{m} · {L}", fontsize=9)
                if j == 0: ax.set_ylabel("largest group\n(% of working neurons)", fontsize=8)
                if i == len(models) - 1: ax.set_xlabel("threshold tau", fontsize=8)
        fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(f"{OUT}/h6_giant_vs_tau.png", dpi=140); plt.close(fig)
    # ---- h1
    files = sorted(glob.glob(f"{OUT}/neurons_*.csv"))
    if files:
        N = pd.concat([pd.read_csv(f) for f in files])
        N = N[N.alive]
        say("\n=== H1: is the self made of known neuron types? ===")
        say("For each score: average in self vs task neurons, and of the 5% most extreme neurons per layer, how many are self")
        say("(compare with the self's share of all neurons: > 1 = self over-represented, < 1 = under-represented)\n")
        rows = []
        for m, q in N.groupby("model"):
            for col, name, which in SCORES:
                qq = q[q.layer.str.startswith("dec")] if which == "dec" else q
                qq = qq.dropna(subset=[col])
                if not len(qq):
                    continue
                share = qq.in_self.mean()
                top = qq.groupby("layer", group_keys=False).apply(lambda d: d[d[col] >= d[col].quantile(.95)])
                bot = qq.groupby("layer", group_keys=False).apply(lambda d: d[d[col] <= d[col].quantile(.05)])
                rows.append({"model": m, "score": col, "meaning": name, "self_mean": qq[qq.in_self][col].mean(), "task_mean": qq[~qq.in_self][col].mean(),
                             "self_share": share, "top5_self_share": top.in_self.mean(), "top5_enrichment": top.in_self.mean() / max(share, 1e-9),
                             "bottom5_enrichment": bot.in_self.mean() / max(share, 1e-9)})
        E = pd.DataFrame(rows); E.to_csv(f"{OUT}/h1_enrichment.csv", index=False)
        for m, q in E.groupby("model"):
            say(f"--- {m} (self = {100 * q.self_share.iloc[0]:.0f}% of working neurons)")
            say(q[["score", "self_mean", "task_mean", "top5_enrichment", "bottom5_enrichment"]].round(3).to_string(index=False))
        fam = N.assign(dead=N.fire_rate < 0.001)
        say("\nfamily counts (thresholds: massive = peak > 20x the layer's typical peak; entropy-like = top 1% null_frac per layer)")
        for m, q in fam.groupby("model"):
            ent = q[q.layer.str.startswith("dec")].groupby("layer", group_keys=False).apply(lambda d: d[d.null_frac >= d.null_frac.quantile(.99)])
            say(f"  {m}: massive {int(q.is_massive.sum())} (self {int(q[q.is_massive].in_self.sum())}) | entropy-like {len(ent)} (self {int(ent.in_self.sum())})")
        # plot enrichment
        ms = list(E.model.unique())
        fig, axs = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 6), squeeze=False, sharey=True)
        for ax, m in zip(axs[0], ms):
            q = E[E.model == m]
            ax.barh(range(len(q)), q.top5_enrichment, color="#7A3FB0")
            ax.axvline(1, color="k", lw=1, ls=":")
            ax.set_yticks(range(len(q))); ax.set_yticklabels(q.meaning, fontsize=8); ax.set_title(m, fontsize=10)
            ax.set_xlabel("self over-representation among the 5% most extreme\n(1 = as expected by chance)", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/h1_enrichment.png", dpi=140); plt.close(fig)
    open(f"{OUT}/report.txt", "w").write("\n".join(R))
    print("figures + report in", OUT)

if args.summarize:
    summarize(); sys.exit()

# =============================================================================== GPU parts
from common_v8 import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import skew, kurtosis
torch.backends.cuda.matmul.allow_tf32 = False
name, spec = args.model.split("=")
if spec == "random":
    torch.manual_seed(0); model = Seq2Seq(Config()).to(DEVICE)
else:
    run, ph = spec.split(":")
    model, ck = load_model(glob.glob(P(f"runs/{run}/ckpt/p{int(ph):03d}_*.pt"))[0])
model.eval()
print(f"model {name} = {spec} | {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", flush=True)
mlp = lambda L: (model.enc_blocks if L.startswith("enc") else model.dec_blocks)[int(L[3])].mlp

# ---------------------------------------------------------------- token frequencies (from training data, first 200k pairs)
t0 = time.time()
cnt_hi = np.zeros(V); cnt_en = np.zeros(V)
with open(P("data/en-hi/train.jsonl"), encoding="utf-8") as f:
    batch_en, batch_hi = [], []
    for i, line in enumerate(f):
        r = json.loads(line); batch_en.append(r["en"]); batch_hi.append(r["tgt"])
        if i >= 200_000:
            break
for ids in sp.encode(batch_hi): np.add.at(cnt_hi, ids, 1)
for ids in sp.encode(batch_en): np.add.at(cnt_en, ids, 1)
logf_hi = np.log(cnt_hi + 1); logf_en = np.log(cnt_en + 1)
print(f"token counts from 200k training pairs in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- H0: record
def ttype(i):
    p = sp.id_to_piece(int(i)).replace("▁", "")
    if sp.is_control(int(i)) or p == "": return "other"
    if all(not ch.isalnum() for ch in p): return "punct"
    if any(ch.isdigit() for ch in p): return "digit"
    if any("ऀ" <= ch <= "ॿ" for ch in p): return "hindi"
    if any(ch.isascii() and ch.isalpha() for ch in p): return "latin"
    return "other"

@torch.no_grad()
def record(lang="hi", n_sent=None, keep_pre=True):
    en = read_flores("dev", "en")[:n_sent or args.n_sent]; tg = read_flores("dev", lang)[:n_sent or args.n_sent]
    S_all = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in en]; T_all = [sp.encode(s)[:MAX_TOK - 1] for s in tg]
    cur = {}
    hooks = []
    for L in LAYERS:
        hooks.append(mlp(L).act.register_forward_hook(lambda m, i, o, k=L: cur.__setitem__(("post", k), o)))
        if keep_pre:
            hooks.append(mlp(L).fc.register_forward_hook(lambda m, i, o, k=L: cur.__setitem__(("pre", k), o)))
    post = {L: [] for L in LAYERS}; pre = {L: [] for L in LAYERS}; enc_rows, dec_rows = [], []
    for s in range(0, len(S_all), 32):
        S_ = pad_batch(S_all[s:s + 32]); mem, sm = model.encode(S_)
        sk = (S_ != PAD) & (S_ != EOS)
        for L in LAYERS[:4]:
            post[L].append(cur[("post", L)][sk].half().cpu())
            if keep_pre: pre[L].append(cur[("pre", L)][sk].half().cpu())
        tin = pad_batch([[LANG_TAG[lang]] + t for t in T_all[s:s + 32]]); tout = pad_batch([t + [EOS] for t in T_all[s:s + 32]])
        logits = model.decode(tin, mem, sm).float() + NEG[lang]
        lp = logits.log_softmax(-1); p_ = lp.exp()
        ent = -(p_ * lp.masked_fill(p_ == 0, 0)).sum(-1)
        nll = -lp.gather(-1, tout.clamp(min=0)[..., None])[..., 0]
        top1 = lp.argmax(-1) == tout
        tk = tin != PAD
        for L in LAYERS[4:]:
            post[L].append(cur[("post", L)][tk].half().cpu())
            if keep_pre: pre[L].append(cur[("pre", L)][tk].half().cpu())
        for b in range(S_.size(0)):
            src = S_all[s + b]
            for j, t in enumerate(src[:-1]):
                enc_rows.append((s + b, j, t, len(src) - 1))
            tl = len(T_all[s + b])
            for j in range(tl + 1):
                dec_rows.append((s + b, j, int(tin[b, j]), int(tout[b, j]), tl + 1, float(nll[b, j]), float(ent[b, j]), bool(top1[b, j])))
    for h in hooks: h.remove()
    post = {L: torch.cat(v) for L, v in post.items()}; pre = {L: torch.cat(v) for L, v in pre.items()} if keep_pre else None
    enc = pd.DataFrame(enc_rows, columns=["sent", "pos", "token", "sent_len"])
    dec = pd.DataFrame(dec_rows, columns=["sent", "pos", "token", "next_token", "sent_len", "loss", "entropy", "top1"])
    return post, pre, enc, dec

t0 = time.time()
POST, PRE, ENC, DEC = record("hi")
for df, logf in [(ENC, logf_en), (DEC, logf_hi)]:
    df["piece"] = [sp.id_to_piece(int(t)) for t in df.token]; df["type"] = [ttype(t) for t in df.token]
    df["word_start"] = df.piece.str.startswith("▁"); df["log_freq"] = logf[df.token.values]; df["rel_pos"] = df.pos / df.sent_len
ENC.assign(side="enc").to_csv(f"{OUT}/tokens_enc_{name}.csv", index=False); DEC.assign(side="dec").to_csv(f"{OUT}/tokens_dec_{name}.csv", index=False)
print(f"H0 recorded {len(ENC):,} encoder (English) and {len(DEC):,} decoder (Hindi) tokens in {time.time() - t0:.0f}s | "
      f"Hindi loss {DEC.loss.mean():.3f}, top-1 correct {100 * DEC.top1.mean():.1f}%", flush=True)

# ---------------------------------------------------------------- self + graphs
rng = np.random.default_rng(0)
def sample_idx(n):
    return torch.tensor(np.sort(rng.choice(n, min(n, args.max_t), replace=False)))
IDX = {L: sample_idx(POST[L].shape[0]) for L in LAYERS}

def corr(X):
    X = X.float().to(DEVICE); mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
    alive = sd[0] > 1e-6
    Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd)); Z[:, ~alive] = 0
    return (Z.T @ Z / Z.shape[0]), alive

def giant(R, alive, tau):
    A = ((R.abs() >= tau) & alive[:, None] & alive[None, :]).cpu().numpy(); np.fill_diagonal(A, False)
    n_alive = int(alive.sum()); n_edges = int(A.sum() // 2)
    _, lab = connected_components(csr_matrix(A), directed=False)
    lab = np.where(alive.cpu().numpy(), lab, -1); sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([0])
    order = np.sort(sizes)[::-1]
    return lab, order, n_alive, n_edges

def er_giant(n, m, reps=3):
    if m == 0: return 0.0
    out = []
    for r in range(reps):
        g = np.random.default_rng(100 + r); e = g.integers(0, n, size=(m, 2)); e = e[e[:, 0] != e[:, 1]]
        A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(A, directed=False); out.append(np.bincount(lab).max() / n)
    return 100 * float(np.mean(out))

SELF = {}
if "h6" in args.parts or "h1" in args.parts:
    rows = []
    t0 = time.time()
    for L in LAYERS:
        X = POST[L][IDX[L]]
        R, alive = corr(X)
        for tau in TAUS:
            lab, order, n_alive, n_edges = giant(R, alive, tau)
            rows.append({"model": name, "layer": L, "condition": "real", "tau": tau, "giant_pct": 100 * order[0] / max(n_alive, 1),
                         "second_pct": 100 * (order[1] if len(order) > 1 else 0) / max(n_alive, 1), "edge_density": n_edges / max(n_alive * (n_alive - 1) / 2, 1)})
            rows.append({"model": name, "layer": L, "condition": "same-density random graph", "tau": tau, "giant_pct": er_giant(n_alive, n_edges)})
            if abs(tau - args.tau) < 1e-9:
                SELF[L] = (alive.cpu().numpy(), lab == np.bincount(lab[lab >= 0]).argmax() if (lab >= 0).any() else np.zeros(len(lab), bool))
        if "h6" in args.parts:
            g = torch.Generator(device="cpu").manual_seed(1)
            Q, _ = torch.linalg.qr(torch.randn(X.shape[1], X.shape[1], generator=g))
            Xr = (X.float() @ Q).half()
            Rr, ar = corr(Xr)
            perm = torch.rand(X.shape[0], X.shape[1], generator=g).argsort(0)
            Xs = torch.gather(X, 0, perm); del perm
            Rs, as_ = corr(Xs)
            for tau in TAUS:
                for cond, RR, aa in [("rotated", Rr, ar), ("shuffled", Rs, as_)]:
                    _, order, n_alive, n_edges = giant(RR, aa, tau)
                    rows.append({"model": name, "layer": L, "condition": cond, "tau": tau, "giant_pct": 100 * order[0] / max(n_alive, 1),
                                 "second_pct": 100 * (order[1] if len(order) > 1 else 0) / max(n_alive, 1), "edge_density": n_edges / max(n_alive * (n_alive - 1) / 2, 1)})
            del Rr, Rs, Xr, Xs
        del R
        print(f"  {L}: self at tau {args.tau} = {int(SELF[L][1].sum())} of {int(SELF[L][0].sum())} working neurons", flush=True)
    G = pd.DataFrame(rows)
    if "h6" in args.parts:
        G.to_csv(f"{OUT}/h6_giant.csv", mode="a", header=not os.path.exists(f"{OUT}/h6_giant.csv"), index=False)
        q = G[G.tau == 0.2].groupby("condition").giant_pct.mean()
        print(f"H6 done in {time.time() - t0:.0f}s | largest group at tau 0.2 (avg of 8 layers): " + " | ".join(f"{k} {v:.1f}%" for k, v in q.items()), flush=True)

# ---------------------------------------------------------------- H1: per-neuron scores
def eta2(x, groups, min_n=5):
    """share of the neuron's variance explained by the group means (x: tokens x neurons; groups: int array)"""
    x = x.float(); tot = x.var(0, unbiased=False) + 1e-12
    gs, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_n
    xg = x[torch.tensor(keep)]; inv = torch.tensor(inv[keep]); k = len(gs)
    sums = torch.zeros(k, x.shape[1]).index_add_(0, inv, xg); n = torch.zeros(k).index_add_(0, inv, torch.ones(len(inv)))
    means = sums / n.clamp(min=1)[:, None]; mu = xg.mean(0)
    between = ((means - mu) ** 2 * n[:, None]).sum(0) / len(inv)
    return (between / tot).numpy()

if "h1" in args.parts:
    t0 = time.time()
    E = model.tok_emb.weight.detach().float()                            # vocab x d (shared input / output table)
    gamma = model.dec_ln.weight.detach().float()
    U, Sv, Vh = torch.linalg.svd(E, full_matrices=False)                # Vh rows: directions in the model's space
    k_null = max(1, int(round(0.02 * E.shape[1])))                       # bottom 2% of directions = barely map to any word
    NULL = Vh[-k_null:]                                                  # k x d
    hi_ok = torch.tensor(cnt_hi > 0, device=E.device)
    lang_post = {}
    if spec != "random" and LANGSET == "distant" and "v4b_only_hi" not in spec:
        lp_ = []; lab_ = []
        for li, l in enumerate(["hi", "zh", "ar"]):
            po, _, _, dd = record(l, n_sent=400, keep_pre=False)
            lp_.append(po); lab_.append(np.full(po["dec0"].shape[0], li))
        lang_post = {L: torch.cat([p[L] for p in lp_]) for L in LAYERS[4:]}; lang_lab = np.concatenate(lab_)
    rows = []
    for L in LAYERS:
        X = POST[L]; Xp = PRE[L]; m_ = mlp(L)
        alive, smask = SELF[L]
        side_df = ENC if L.startswith("enc") else DEC
        pos = np.minimum(side_df.pos.values, 60); tok = side_df.token.values
        Xs_ = X[IDX[L]].float(); Xps = Xp[IDX[L]].float().numpy()
        act_mean = X.float().mean(0).numpy(); fire = (X > 0).float().mean(0).numpy(); peak = X.float().abs().max(0).values.numpy()
        w_in = m_.fc.weight.detach().float().norm(dim=1).cpu().numpy(); b_in = m_.fc.bias.detach().float().cpu().numpy()
        w_out_m = m_.proj.weight.detach().float()                       # d x 2560
        w_out = w_out_m.norm(dim=0).cpu().numpy()
        pos_eta = eta2(X, pos); tok_eta = eta2(X, tok, min_n=20)
        r = {"model": name, "layer": L, "neuron": np.arange(X.shape[1]), "alive": alive, "in_self": smask & alive,
             "act_mean": act_mean, "fire_rate": fire, "peak": peak, "massive": peak / (np.median(peak) + 1e-9),
             "w_in": w_in, "b_in": b_in, "w_out": w_out, "pre_skew": skew(Xps, axis=0), "pre_kurt": kurtosis(Xps, axis=0),
             "pos_eta": pos_eta, "tok_eta": tok_eta}
        if L.startswith("dec"):
            W = (w_out_m * gamma[:, None])                                # what the neuron adds, after the final layer-norm scale
            logit_eff = (E @ W).T                                         # 2560 x vocab: change of every word's score
            le = logit_eff[:, hi_ok].cpu().numpy()
            r["null_frac"] = ((NULL @ W) ** 2).sum(0).cpu().numpy() / ((W ** 2).sum(0).cpu().numpy() + 1e-12)
            lf = logf_hi[cnt_hi > 0]; lfz = (lf - lf.mean()) / lf.std()
            lez = (le - le.mean(1, keepdims=True)) / (le.std(1, keepdims=True) + 1e-9)
            r["freq_corr"] = (lez * lfz[None, :]).mean(1)
            r["out_skew"] = skew(le, axis=1); r["out_kurt"] = kurtosis(le, axis=1)
            r["lang_eta"] = eta2(lang_post[L], lang_lab) if lang_post else np.full(X.shape[1], np.nan)
        df = pd.DataFrame(r); df["is_massive"] = df.massive > 20
        rows.append(df)
        print(f"  {L} scored", flush=True)
    N = pd.concat(rows)
    N.to_csv(f"{OUT}/neurons_{name}.csv", index=False)
    print(f"H1 done in {time.time() - t0:.0f}s (null directions = bottom {k_null} of {E.shape[1]})", flush=True)
print("done")
