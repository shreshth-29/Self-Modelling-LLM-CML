"""analysis_v6_deep.py: the new analyses (GPU), all from saved checkpoints. No training run needed.

Parts (choose with --parts; default: all):
  relearn    LESION + RELEARN (robot paper, done properly): reset 20% of each MLP layer's neurons to fresh random
             weights, chosen from the self / from task neurons / at random, then train the same language for a few
             hundred steps. Does losing self neurons hurt more, and is it harder to recover from?
  crosscoder MODEL DIFFING (sparse crosscoder, Lindsey et al. 2024): learn ONE set of features that reconstructs two
             checkpoints at once. Each feature gets a decoder in A and in B: present in both = shared, in one only =
             exclusive. Pairs: continual across a switch, Hindi-only across a phase, continual vs Hindi-only.
             Link to the self: are self neurons written mostly by SHARED features (= the stable part)?
  ends       GOODFIRE 1, both ends of a neuron: the tokens that push its input (pre-GELU) most positive and most negative.
  widen      GOODFIRE 2, does a neuron's job widen or jump? Same neuron followed over many checkpoints: overlap of its top
             tokens, and whether NEW top tokens are related (embedding similarity) to the old ones.
  steer      GOODFIRE 3, force a neuron on while translating: does the output change in the direction of its top tokens?
  sentences  SELF-PRESERVATION 1 (Cho et al.): which test sentences are forgotten at a switch, and do they rely more on
             task neurons than on self neurons (per-sentence lesion)?

Self = biggest group of MLP neurons with |r| >= tau (0.2) on FLORES dev, computed here from the checkpoint itself
(no npz needed). Output: analysis/v6/deep_<tag>/ (csv + png + report.txt).

Usage (repo root):
  python3 -u analysis_v6_deep.py --gpu 0                                   # references at phase 45, all parts
  python3 -u analysis_v6_deep.py --gpu 1 --parts relearn steer --cont v6_cont_noise0.0007_from60 --phase 69 --tag fix69
"""
import os, sys, argparse, json, glob, time, math
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--parts", nargs="+", default=["relearn", "crosscoder", "ends", "widen", "steer", "sentences"])
ap.add_argument("--cont", default="v4b_continual_s0")
ap.add_argument("--hi", default="v4b_only_hi_s0")
ap.add_argument("--phase", type=int, default=45, help="main phase (a switch INTO this phase for the continual run)")
ap.add_argument("--widen_phases", default="30,33,36,39,42,45,48,51,54,57,60", help="same-language checkpoints for 'widen'")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--lesion_frac", type=float, default=0.2)
ap.add_argument("--relearn_steps", type=int, default=400)
ap.add_argument("--cc_steps", type=int, default=4000)
ap.add_argument("--cc_dict", type=int, default=8192)
ap.add_argument("--cc_k", type=int, default=32)
ap.add_argument("--cc_layers", default="enc1.mlp,dec2.mlp")
ap.add_argument("--n_steer", type=int, default=100)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--tag", default=None)
ap.add_argument("--labels", default="continual,hindi", help="display names for --cont and --hi")
ap.add_argument("--small", action="store_true", help="tiny settings for a smoke test")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
from common_v6 import *
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sacrebleu.metrics import CHRF
torch.backends.cuda.matmul.allow_tf32 = False

TAG = args.tag or f"p{args.phase}"
OUT = P("analysis", "v6", f"deep_{TAG}"); os.makedirs(OUT, exist_ok=True)
MLP = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]
LAB = args.labels.split(",")
RUNS = {LAB[0]: args.cont, LAB[1]: args.hi}
C_RUN, H_RUN = args.cont, args.hi
rng = np.random.default_rng(0); torch.manual_seed(0)
REPORT = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REPORT.append(s)
    open(os.path.join(OUT, "report.txt"), "w").write("\n".join(REPORT))
if args.small:
    args.relearn_steps, args.cc_steps, args.cc_dict, args.n_steer, args.max_t = 6, 30, 256, 4, 2000

def ck(run, ph):
    f = glob.glob(P("runs", run, "ckpt", f"p{ph:03d}_*.pt"))
    if not f:
        raise FileNotFoundError(f"runs/{run}/ckpt/p{ph:03d}_*.pt (download it first)")
    return f[0]

# ------------------------------------------------------------------ inputs
DEV_EN = read_flores("dev", "en"); TEST_EN = read_flores("devtest", "en")
SRC = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}
piece = lambda t: sp.id_to_piece(int(t)).replace("▁", "_")

def mlp_mod(model, pt):
    side, j = pt[:3], int(pt[3])
    return (model.enc_blocks if side == "enc" else model.dec_blocks)[j].mlp

@torch.no_grad()
def record(model, pts=MLP, pre=False, langs=LANGS, src=SRC, tgt=TGT, bs=64):
    """Activations (post-GELU, or pre-GELU with pre=True) at every real token, kept on CPU in fp16.
    enc: English source tokens; dec: target tokens of `langs` (teacher-forced). Also token ids and language per row."""
    model.eval(); cur = {}
    mods = {pt: (mlp_mod(model, pt).fc if pre else mlp_mod(model, pt).act) for pt in pts}
    hooks = [m.register_forward_hook(lambda mm, i, o, k=pt: cur.__setitem__(k, o)) for pt, m in mods.items()]
    acts = {pt: [] for pt in pts}; stok, dtok, dlang = [], [], []
    for s in range(0, len(src), bs):
        S_ = pad_batch(src[s:s + bs]); mem, sm = model.encode(S_)
        sk = (S_ != PAD) & (S_ != EOS)
        for pt in pts:
            if pt.startswith("enc"):
                acts[pt].append(cur[pt][sk].half().cpu())
        stok.append(S_[sk].cpu())
        for li, l in enumerate(langs):
            T_ = pad_batch(tgt[l][s:s + bs]); model.decode(T_, mem, sm)
            tk = T_ != PAD; tk[:, 0] = False
            for pt in pts:
                if pt.startswith("dec"):
                    acts[pt].append(cur[pt][tk].half().cpu())
            dtok.append(T_[tk].cpu()); dlang.append(torch.full((int(tk.sum()),), LANGS.index(l)))
    for h in hooks:
        h.remove()
    return ({k: torch.cat(v) for k, v in acts.items()}, torch.cat(stok).numpy(), torch.cat(dtok).numpy(), torch.cat(dlang).numpy())

def self_mask(X, groups=None, tau=None):
    """Biggest connected group of |r| >= tau among alive neurons (same definition as the deck)."""
    tau = tau or args.tau
    n = X.shape[0]
    idx = torch.randperm(n)[:args.max_t] if n > args.max_t else torch.arange(n)
    Xs = X[idx].float().to(DEVICE)
    if groups is not None:
        g = torch.as_tensor(groups)[idx].to(DEVICE)
        for v in g.unique():
            m = g == v; Xs[m] -= Xs[m].mean(0, keepdim=True)
    sd = Xs.std(0, unbiased=False); alive = sd > 1e-6
    Z = (Xs - Xs.mean(0)) / torch.where(alive, sd, torch.ones_like(sd))
    R = (Z.T @ Z / Z.shape[0]).abs()
    R.fill_diagonal_(0); R[~alive] = 0; R[:, ~alive] = 0
    i, j = torch.nonzero(torch.triu(R >= tau), as_tuple=True)
    nn_ = X.shape[1]
    _, lab = connected_components(csr_matrix((np.ones(len(i)), (i.cpu().numpy(), j.cpu().numpy())), shape=(nn_, nn_)), directed=False)
    alive = alive.cpu().numpy(); lab = np.where(alive, lab, -1)
    del Xs, Z, R
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return (lab == np.bincount(lab[lab >= 0]).argmax()) if (lab >= 0).any() else np.zeros(nn_, bool), alive

def selves(model):
    acts, stok, dtok, dlang = record(model)
    out = {pt: self_mask(acts[pt], None if pt.startswith("enc") else dlang) for pt in MLP}
    return out, acts, stok, dtok, dlang

def load_train(l, n):
    rows = []
    with open(P(f"data/en-{l}/train.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    tmp = os.path.join(OUT, f"_tmp_{l}.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data = load_pairs(tmp, l); os.remove(tmp); return data
VAL = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:(50 if args.small else 500)] for l in LANGS}

# ================================================================== 1. LESION + RELEARN
def part_relearn():
    say("\n=== 1. LESION + RELEARN: reset", f"{args.lesion_frac:.0%}", "of each MLP layer, then train the same language ===")
    rows = []
    for name, run in RUNS.items():
        path = ck(run, args.phase)
        model, c = load_model(path); L = c["lang"]
        S, *_ = selves(model); del model
        train = load_train(L, 2000 if args.small else 40000)
        for gname in ["self", "task", "random"]:
            for seed in [0, 1]:
                model, _ = load_model(path)
                g = torch.Generator(device="cpu").manual_seed(seed)
                with torch.no_grad():
                    for pt in MLP:
                        smask, alive = S[pt]; k = int(args.lesion_frac * alive.sum())
                        pool = {"self": np.nonzero(smask)[0], "task": np.nonzero(alive & ~smask)[0], "random": np.nonzero(alive)[0]}[gname]
                        idx = torch.tensor(np.random.default_rng(seed).choice(pool, min(k, len(pool)), replace=False))
                        m = mlp_mod(model, pt)
                        m.fc.weight[idx] = (torch.randn(len(idx), m.fc.weight.shape[1], generator=g) * 0.02).to(DEVICE)
                        m.fc.bias[idx] = 0
                        m.proj.weight[:, idx] = (torch.randn(m.proj.weight.shape[0], len(idx), generator=g) * 0.02 / math.sqrt(16)).to(DEVICE)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
                batches = make_batches(train, 8000, seed=seed)
                evals = sorted(set([0, 25, 50, 100, 200, args.relearn_steps]) & set(range(args.relearn_steps + 1)))
                for step in range(args.relearn_steps + 1):
                    if step in evals:
                        r = {"run": name, "lang": L, "group": gname, "seed": seed, "step": step}
                        r.update({f"val_{l}": val_loss(model, VAL[l], l, masked=True) for l in LANGS})
                        rows.append(r)
                    if step == args.relearn_steps:
                        break
                    model.train()
                    s_, ti, to = collate(train, batches[step % len(batches)])
                    with amp():
                        logits, _ = model(s_, ti)
                    loss = masked_ce(logits.view(-1, logits.size(-1)), to.view(-1), ALLOWED[L])
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                del model, opt
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        base, _ = load_model(path)
        b = {l: val_loss(base, VAL[l], l, masked=True) for l in LANGS}; del base
        for r in rows:
            if r["run"] == name:
                r.update({f"intact_{l}": b[l] for l in LANGS})
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "relearn.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axs, RUNS):
        q = D[D.run == name]; L = q.lang.iloc[0]
        for gname, c in [("self", "#4B1D6B"), ("task", "#E07A1F"), ("random", "0.5")]:
            m = q[q.group == gname].groupby("step")[f"val_{L}"].mean()
            ax.plot(m.index, m.values - q[f"intact_{L}"].iloc[0], marker="o", color=c, label=f"reset {args.lesion_frac:.0%} {gname} neurons")
        ax.set_title(f"{name}: {L} val loss above the intact model"); ax.set_xlabel("relearning steps"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "relearn.png"), dpi=140)
    for name in RUNS:
        q = D[D.run == name]; L = q.lang.iloc[0]
        for gname in ["self", "task", "random"]:
            qq = q[q.group == gname].groupby("step")[f"val_{L}"].mean() - q[f"intact_{L}"].iloc[0]
            say(f"  {name:<10} reset {gname:<6}: loss rise right after = {qq.iloc[0]:.3f}, after {qq.index[-1]} steps = {qq.iloc[-1]:.3f}")

# ================================================================== 2. CROSSCODER
def crosscoder(XA, XB, steps, dict_size, k):
    """BatchTopK-style sparse crosscoder: one encoder reads both, separate decoders write each. Returns decoders etc."""
    if XA.shape[0] > 120_000:                                      # cap tokens (memory); same rows for A and B
        keep = torch.randperm(XA.shape[0], generator=torch.Generator().manual_seed(0))[:120_000]
        XA, XB = XA[keep], XB[keep]
    XA = XA.float().to(DEVICE); XB = XB.float().to(DEVICE)
    sA, sB = XA.norm(dim=1).mean(), XB.norm(dim=1).mean()          # put both models on the same scale
    XA, XB = XA / sA * math.sqrt(XA.shape[1]) / 10, XB / sB * math.sqrt(XB.shape[1]) / 10
    n, d = XA.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    W_dec_A = torch.randn(dict_size, d, generator=g).to(DEVICE); W_dec_A /= W_dec_A.norm(dim=1, keepdim=True)
    W_dec_B = W_dec_A.clone()
    W_enc = torch.cat([W_dec_A.T, W_dec_B.T], 0).clone() * 0.5
    b_enc = torch.zeros(dict_size, device=DEVICE); b_A = XA.mean(0).clone(); b_B = XB.mean(0).clone()
    params = [W_enc, b_enc, W_dec_A, W_dec_B, b_A, b_B]
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=3e-4)
    bs = min(2048, n)
    for step in range(steps):
        i = torch.randint(0, n, (bs,), device=DEVICE)
        xa, xb = XA[i], XB[i]
        pre = torch.relu(torch.cat([xa - b_A, xb - b_B], 1) @ W_enc + b_enc)
        thr = torch.topk(pre.flatten(), k * bs).values[-1]           # batch top-k: k active features per token on average
        f = pre * (pre >= thr)
        ra, rb = f @ W_dec_A + b_A, f @ W_dec_B + b_B
        loss = ((ra - xa) ** 2).sum(1).mean() + ((rb - xb) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fvu, cnt, tot = [], torch.zeros(dict_size, device=DEVICE), torch.zeros(dict_size, device=DEVICE)
        for s in range(0, n, 4096):
            xa, xb = XA[s:s + 4096], XB[s:s + 4096]
            pre = torch.relu(torch.cat([xa - b_A, xb - b_B], 1) @ W_enc + b_enc)
            f = pre * (pre >= thr); cnt += (f > 0).sum(0); tot += f.sum(0)
            fvu.append(((((f @ W_dec_A + b_A) - xa) ** 2).sum() / ((xa - XA.mean(0)) ** 2).sum()).item())
    return dict(dec_A=W_dec_A.detach().cpu(), dec_B=W_dec_B.detach().cpu(), freq=(cnt / n).cpu(), mean_act=(tot / n).cpu(),
                fvu=float(np.mean(fvu)))

def part_crosscoder():
    say("\n=== 2. CROSSCODER (model diffing) ===")
    ph = args.phase
    pairs = [("continual switch", C_RUN, ph - 1, C_RUN, ph, LANGS),
             ("Hindi-only step", H_RUN, ph - 1, H_RUN, ph, LANGS),
             ("continual vs Hindi-only (Hindi text)", C_RUN, ph, H_RUN, ph, ["hi"])]
    layers = args.cc_layers.split(",")
    src = SRC + [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in TEST_EN]      # dev + devtest: more tokens
    tgt = {l: TGT[l] + [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("devtest", l)] for l in LANGS}
    rows = []
    fig, axs = plt.subplots(len(layers), len(pairs), figsize=(5 * len(pairs), 3.4 * len(layers)), squeeze=False)
    for pj, (pname, ra, pa, rb, pb, langs) in enumerate(pairs):
        mA, cA = load_model(ck(ra, pa)); mB, cB = load_model(ck(rb, pb))
        A, *_ , dlA = record(mA, layers, langs=langs, src=src, tgt=tgt)
        B, *_ , dlB = record(mB, layers, langs=langs, src=src, tgt=tgt)
        SA = {pt: self_mask(A[pt], None if pt.startswith("enc") else dlA) for pt in layers}
        SB = {pt: self_mask(B[pt], None if pt.startswith("enc") else dlB) for pt in layers}
        del mA, mB
        for li, pt in enumerate(layers):
            cc = crosscoder(A[pt], B[pt], args.cc_steps, args.cc_dict, args.cc_k)
            nA, nB = cc["dec_A"].norm(dim=1), cc["dec_B"].norm(dim=1)
            live = cc["freq"] > 1e-4
            r = (nB / (nA + nB + 1e-9))[live].numpy()
            shared = (r > 0.3) & (r < 0.7); a_only = r <= 0.1; b_only = r >= 0.9
            # how much of each neuron's input from the crosscoder comes from shared features (weighted by use)
            wA = (cc["dec_A"][live].abs() * cc["mean_act"][live, None]); wB = (cc["dec_B"][live].abs() * cc["mean_act"][live, None])
            shA = (wA[torch.tensor(shared)].sum(0) / (wA.sum(0) + 1e-9)).numpy()
            shB = (wB[torch.tensor(shared)].sum(0) / (wB.sum(0) + 1e-9)).numpy()
            sa, ala = SA[pt]; sb, alb = SB[pt]
            row = {"pair": pname, "layer": pt, "A": f"{ra}:p{pa}", "B": f"{rb}:p{pb}", "fvu_A": cc["fvu"], "live": int(live.sum()),
                   "shared_frac": shared.mean(), "A_only_frac": a_only.mean(), "B_only_frac": b_only.mean(),
                   "shared_input_self_A": shA[sa].mean(), "shared_input_task_A": shA[ala & ~sa].mean(),
                   "shared_input_self_B": shB[sb].mean(), "shared_input_task_B": shB[alb & ~sb].mean()}
            rows.append(row)
            ax = axs[li, pj]; ax.hist(r, bins=40, color="#2c7fb8")
            ax.set_title(f"{pname}\n{pt}: shared {100 * row['shared_frac']:.0f}% | only A {100 * row['A_only_frac']:.0f}% | only B {100 * row['B_only_frac']:.0f}%", fontsize=8)
            ax.set_xlabel("relative decoder norm: 0 = only in A, 1 = only in B")
            say(f"  {pname:<38} {pt}: shared {100 * row['shared_frac']:4.0f}%  A-only {100 * row['A_only_frac']:3.0f}%  "
                f"B-only {100 * row['B_only_frac']:3.0f}%  | input from shared features: self {row['shared_input_self_A']:.2f} vs task "
                f"{row['shared_input_task_A']:.2f} (A), self {row['shared_input_self_B']:.2f} vs task {row['shared_input_task_B']:.2f} (B)"
                f"  | unexplained variance {row['fvu_A']:.2f}")
            del cc
        del A, B
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "crosscoder_hist.png"), dpi=140)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "crosscoder.csv"), index=False)

# ================================================================== 3-5. GOODFIRE
def token_means(X, toks, min_count=5):
    """Mean activation per token TYPE (only types seen >= min_count times). Returns (type ids, [types x neurons])."""
    toks = torch.as_tensor(toks); uniq, inv, cnt = torch.unique(toks, return_inverse=True, return_counts=True)
    S = torch.zeros(len(uniq), X.shape[1]).index_add_(0, inv, X.float())
    keep = cnt >= min_count
    return uniq[keep].numpy(), (S[keep] / cnt[keep, None]).numpy()

def pick(smask, alive, n=12):
    s, t = np.nonzero(smask)[0], np.nonzero(alive & ~smask)[0]
    return {"self": rng.choice(s, min(n, len(s)), replace=False) if len(s) else [], "task": rng.choice(t, min(n, len(t)), replace=False) if len(t) else []}

def part_ends():
    say("\n=== 3. GOODFIRE: both ends of a neuron (pre-GELU input, mean per token type) ===")
    rows = []
    for name, run in RUNS.items():
        model, c = load_model(ck(run, args.phase))
        S, *_ = selves(model)
        pre, stok, dtok, dlang = record(model, ["enc1.mlp", "dec2.mlp"], pre=True)
        L = c["lang"]
        for pt in ["enc1.mlp", "dec2.mlp"]:
            if pt.startswith("enc"):
                ids, M = token_means(pre[pt], stok)
            else:
                m = dlang == LANGS.index(L); ids, M = token_means(pre[pt][torch.as_tensor(m)], dtok[m])
            for gname, ns in pick(*S[pt]).items():
                for nid in ns:
                    o = np.argsort(M[:, nid])
                    rows.append({"run": name, "phase": args.phase, "layer": pt, "lang": "en" if pt.startswith("enc") else L, "group": gname,
                                 "neuron": int(nid), "positive_end": " ".join(piece(ids[k]) for k in o[::-1][:10]),
                                 "negative_end": " ".join(piece(ids[k]) for k in o[:10])})
        del model
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "ends.csv"), index=False)
    for r in D.head(8).itertuples():
        say(f"  {r.run} {r.layer} {r.group} #{r.neuron}:  + {r.positive_end}   |   - {r.negative_end}")
    say("  (full table: ends.csv; read it by eye: do the two ends form a contrast?)")

def part_widen():
    say("\n=== 4. GOODFIRE: does a neuron's job widen (related new tokens) or jump (unrelated)? ===")
    phs = [int(x) for x in args.widen_phases.split(",")]
    rows = []
    for name, run in RUNS.items():
        avail = [p for p in phs if glob.glob(P("runs", run, "ckpt", f"p{p:03d}_*.pt"))]
        if len(avail) < 2:
            say(f"  {name}: need >= 2 of phases {phs}"); continue
        m0, c0 = load_model(ck(run, avail[0])); S0, *_ = selves(m0)
        E = F.normalize(m0.tok_emb.weight.detach().float(), dim=1).cpu()        # meaning space for "related"
        L = c0["lang"]; del m0
        groups = {pt: pick(*S0[pt], n=40) for pt in ["enc1.mlp", "dec2.mlp"]}
        prev = {}
        for ph in avail:
            m, c = load_model(ck(run, ph))
            acts, stok, dtok, dlang = record(m, ["enc1.mlp", "dec2.mlp"]); del m
            for pt in ["enc1.mlp", "dec2.mlp"]:
                if pt.startswith("enc"):
                    ids, M = token_means(acts[pt], stok)
                else:
                    mm = dlang == LANGS.index(L); ids, M = token_means(acts[pt][torch.as_tensor(mm)], dtok[mm])
                for gname, ns in groups[pt].items():
                    for nid in ns:
                        top = set(ids[np.argsort(-M[:, nid])[:30]].tolist())
                        breadth = int((M[:, nid] > 0.5 * M[:, nid].max()).sum()) if M[:, nid].max() > 0 else 0
                        key = (pt, gname, int(nid))
                        if key in prev:
                            old = prev[key]; new = top - old
                            jac = len(top & old) / len(top | old)
                            Eo = E[list(old)]
                            rel_new = float((E[list(new)] @ Eo.T).mean()) if new else np.nan
                            rnd = torch.as_tensor(rng.choice(len(E), 200, replace=False))
                            rel_rand = float((E[rnd] @ Eo.T).mean())
                            rows.append({"run": name, "layer": pt, "group": gname, "neuron": int(nid), "from": prev[("ph",)], "to": ph,
                                         "top30_jaccard": jac, "n_new": len(new), "relatedness_new": rel_new,
                                         "relatedness_random": rel_rand, "breadth": breadth})
                        prev[key] = top
            prev[("ph",)] = ph
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "widen.csv"), index=False)
    if len(D):
        g = D.groupby(["run", "layer", "group"]).agg(jaccard=("top30_jaccard", "mean"), rel_new=("relatedness_new", "mean"),
                                                     rel_random=("relatedness_random", "mean"), breadth=("breadth", "mean")).round(3)
        say(g.to_string())
        say("  widening = high jaccard + new tokens MORE related than random (rel_new > rel_random); jumping = the opposite.")
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (name, q) in zip(axs, D.groupby("run")):
            for (pt, gname), qq in q.groupby(["layer", "group"]):
                m = qq.groupby("to").top30_jaccard.mean()
                ax.plot(m.index, m.values, marker="o", ls="-" if gname == "self" else "--", label=f"{pt} {gname}")
            ax.set_title(f"{name}: top-30 token overlap with the previous checkpoint"); ax.set_xlabel("phase"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "widen.png"), dpi=140)

def part_steer():
    say("\n=== 5. GOODFIRE: force a neuron on while translating ===")
    rows = []; metric = CHRF()
    src = TEST_EN[:args.n_steer]
    for name, run in RUNS.items():
        model, c = load_model(ck(run, args.phase)); L = c["lang"]
        S, *_ = selves(model)
        pt = "dec2.mlp"
        acts, stok, dtok, dlang = record(model, [pt])
        mm = dlang == LANGS.index(L); X = acts[pt][torch.as_tensor(mm)].float(); toks = dtok[mm]
        ids, M = token_means(X, toks)
        refs = read_flores("devtest", L)[:args.n_steer]
        base = translate(model, src, L, masked=True)
        base_tok = sp.encode(base); base_chrf = metric.corpus_score(base, [refs]).score
        for gname, ns in pick(*S[pt], n=8).items():
            for nid in ns:
                hi_val = float(torch.quantile(X[:, nid], 0.999)) * 1.5
                top5 = set(ids[np.argsort(-M[:, nid])[:5]].tolist())
                mod = mlp_mod(model, pt).act
                h = mod.register_forward_hook(lambda m_, i_, o, n_=int(nid), v=hi_val: o.index_fill(-1, torch.tensor([n_], device=o.device), v))
                out = translate(model, src, L, masked=True); h.remove()
                out_tok = sp.encode(out)
                cnt = lambda TT: sum(t in top5 for s_ in TT for t in s_) / max(sum(len(s_) for s_ in TT), 1)
                rows.append({"run": name, "layer": pt, "group": gname, "neuron": int(nid), "top5": " ".join(piece(t) for t in top5),
                             "share_top5_base": cnt(base_tok), "share_top5_forced": cnt(out_tok),
                             "chrf_base": base_chrf, "chrf_forced": metric.corpus_score(out, [refs]).score,
                             "example_base": base[0][:120], "example_forced": out[0][:120]})
        del model
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "steer.csv"), index=False)
    if len(D):
        D["top5_ratio"] = (D.share_top5_forced + 1e-4) / (D.share_top5_base + 1e-4)
        D["chrf_drop"] = D.chrf_base - D.chrf_forced
        say(D.groupby(["run", "group"]).agg(top5_ratio_median=("top5_ratio", "median"), chrf_drop=("chrf_drop", "mean")).round(2).to_string())
        say("  top5_ratio > 1: forcing the neuron makes the model write more of the tokens that neuron fires on (it is causal for them).")

# ================================================================== 6. SELF-PRESERVATION: which sentences are forgotten
@torch.no_grad()
def sent_losses(model, data, lang, hook=None):
    model.eval(); out = np.zeros(len(data))
    for s in range(0, len(data), 64):
        b = list(range(s, min(s + 64, len(data)))); src, ti, to = collate(data, b)
        with amp():
            lg, _ = model(src, ti)
        lp = F.log_softmax(lg.float().masked_fill(~ALLOWED[lang], float("-inf")), -1)
        nll = -lp.gather(2, to[..., None]).squeeze(2)
        keep = (to != PAD) & ALLOWED[lang][to] & torch.isfinite(nll)
        out[s:s + len(b)] = (torch.where(keep, nll, torch.zeros_like(nll)).sum(1) / keep.sum(1).clamp(min=1)).cpu().numpy()   # where, not *: inf x 0 = nan
    return out

def part_sentences():
    say("\n=== 6. SELF-PRESERVATION: which sentences are forgotten at a switch, and what do they rely on? ===")
    run = C_RUN; ph = args.phase
    tests = {l: [(sp.encode(e)[:MAX_TOK - 1] + [EOS], [LANG_TAG[l]] + sp.encode(t)[:MAX_TOK - 1], sp.encode(t)[:MAX_TOK - 1] + [EOS])
                 for e, t in zip(TEST_EN, read_flores("devtest", l))] for l in LANGS}
    mA, cA = load_model(ck(run, ph - 1)); mB, cB = load_model(ck(run, ph))
    old = [l for l in LANGS if l != cB["lang"]]
    S, acts, *_ = selves(mA)
    rows = []
    for l in old:
        before, after = sent_losses(mA, tests[l], l), sent_losses(mB, tests[l], l)
        rel = {}
        for gname in ["self", "task"]:
            hooks = []
            for pt in MLP[4:]:
                smask, alive = S[pt]; k = int(args.lesion_frac * alive.sum())
                pool = np.nonzero(smask if gname == "self" else (alive & ~smask))[0]
                idx = torch.tensor(rng.choice(pool, min(k, len(pool)), replace=False), device=DEVICE)
                mean = acts[pt].float().mean(0).to(DEVICE)[idx]
                hooks.append(mlp_mod(mA, pt).act.register_forward_hook(
                    lambda m_, i_, o, ii=idx, vv=mean: o.index_copy(-1, ii, vv.to(o.dtype).expand(*o.shape[:-1], len(ii)).contiguous())))
            rel[gname] = sent_losses(mA, tests[l], l) - before
            for h in hooks:
                h.remove()
        forget = after - before
        r_s, _ = spearmanr(forget, rel["self"]); r_t, _ = spearmanr(forget, rel["task"])
        r_len, _ = spearmanr(forget, [len(t[2]) for t in tests[l]])
        say(f"  switch {ph - 1}->{ph} ({cA['lang']}->{cB['lang']}), forgetting of {l}: mean loss rise {forget.mean():.3f}. "
            f"Spearman with reliance on self {r_s:+.2f}, on task neurons {r_t:+.2f}, with sentence length {r_len:+.2f}")
        for i in range(len(forget)):
            rows.append({"lang": l, "sent": i, "forget": forget[i], "rely_self": rel["self"][i], "rely_task": rel["task"][i],
                         "len": len(tests[l][i][2]), "loss_before": before[i]})
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "sentences.csv"), index=False)
    fig, axs = plt.subplots(1, len(old), figsize=(5.5 * len(old), 4), squeeze=False)
    for ax, l in zip(axs[0], old):
        q = D[D.lang == l]
        ax.scatter(q.rely_task, q.forget, s=6, alpha=.4, label="reliance on task neurons", color="#E07A1F")
        ax.scatter(q.rely_self, q.forget, s=6, alpha=.4, label="reliance on self neurons", color="#4B1D6B")
        ax.set_xlabel("loss rise when the group is mean-ablated (before the switch)"); ax.set_ylabel("loss rise across the switch")
        ax.set_title(f"{l}: which sentences are forgotten"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sentences.png"), dpi=140)

say(f"analysis_v6_deep | runs {RUNS} | phase {args.phase} | tau {args.tau} | parts {args.parts} | out {OUT}")
for part in args.parts:
    t0 = time.time()
    globals()[f"part_{part}"]()
    say(f"  [{part} took {(time.time() - t0) / 60:.1f} min]")
say("finished")
