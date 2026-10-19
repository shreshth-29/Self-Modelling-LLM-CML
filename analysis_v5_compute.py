"""analysis_v5_compute.py: the heavy (GPU) pass of the v5 analysis. Threshold-free.

For every checkpoint of one run, and for all 20 probe points (8 MLP layers + 12 attention value-vector points):
  * record activations on fixed FLORES dev inputs (same inputs for every checkpoint)
  * z-scored traces -> neuron x neuron correlation R
  * the maximum spanning tree of |R| (2,559 edges). The groups you get by thresholding |R| at ANY tau are
    exactly the groups of this tree's edges >= tau, so every tau can be analysed later without the GPU.
  * health: per-neuron std / mean / fraction of positions where the unit is > 0, effective rank of R, weight norms
  * at every switch (previous checkpoint -> this one): Hungarian matching and per-neuron
    activation similarity, connectivity similarity and change = 100 x (1 - mean of the two)  (as in the robot paper)

Output: analysis/v5/<run>/pXXX.npz (one small file per checkpoint), weights.jsonl, phases.jsonl, log.jsonl.
Rerunning skips checkpoints that are already done.

Usage (one run per GPU, in tmux):
  python -u analysis_v5_compute.py --run v4b_continual_s0 --gpu 0
  python -u analysis_v5_compute.py --run v4b_only_hi_s0  --gpu 1
"""
import os, sys, argparse, time, json, glob, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--max_t", type=int, default=24000, help="token positions per trace")
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--last", type=int, default=10_000)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)
from common_v5 import *
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree

torch.backends.cuda.matmul.allow_tf32 = False          # full precision for correlations
OUT = P("analysis", "v5", args.run)
os.makedirs(OUT, exist_ok=True)
for f in ["phases.jsonl", "log.jsonl"]:
    if os.path.exists(P("runs", args.run, f)):
        shutil.copyfile(P("runs", args.run, f), os.path.join(OUT, f))

# ---------------------------------------------------------------- fixed inputs (FLORES dev)
DEV_EN = read_flores("dev", "en")
SRC_IDS = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT_IN = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}

def on_source(pt):
    """Points whose positions are English source tokens (encoder, and cross-attention values)."""
    return pt.startswith("enc") or pt.endswith("cross_v")

@torch.no_grad()
def record(model, bs=64):
    model.eval()
    pts = probe_points(model)
    cur = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, k=k: cur.__setitem__(k, o)) for k, m in pts.items()]
    acts, dec_lang = {k: [] for k in pts}, []
    for s in range(0, len(SRC_IDS), bs):
        src = pad_batch(SRC_IDS[s:s + bs])
        mem, src_mask = model.encode(src)
        skeep = (src != PAD) & (src != EOS)
        for k in pts:
            if k.startswith("enc"):
                acts[k].append(cur[k][skeep].half())
        for li, l in enumerate(LANGS):
            tgt = pad_batch(TGT_IN[l][s:s + bs])
            model.decode(tgt, mem, src_mask)
            tkeep = tgt != PAD
            tkeep[:, 0] = False                              # skip the language-tag position
            for k in pts:
                if k.startswith("dec"):
                    if k.endswith("cross_v"):
                        if li == 0:                          # same English source for every language
                            acts[k].append(cur[k][skeep].half())
                    else:
                        acts[k].append(cur[k][tkeep].half())
            dec_lang.append(torch.full((int(tkeep.sum()),), li, device=DEVICE))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(dec_lang)

def subsample_idx(n, k, seed=0):
    if n <= k:
        return torch.arange(n, device=DEVICE)
    return torch.tensor(np.sort(np.random.default_rng(seed).choice(n, k, replace=False)), device=DEVICE)

def dec_subsample(dec_lang, k, seed=1):
    g, lang_np, out = np.random.default_rng(seed), dec_lang.cpu().numpy(), []
    for li in range(len(LANGS)):
        pos = np.nonzero(lang_np == li)[0]
        out.append(np.sort(g.choice(pos, min(len(pos), k // len(LANGS)), replace=False)))
    return torch.tensor(np.concatenate(out), device=DEVICE)

def traces(X, groups=None):
    """Remove each language's mean per neuron (target-side points), then z-score each neuron."""
    X = X.clone()
    if groups is not None:
        for g in groups.unique():
            m = groups == g
            X[m] -= X[m].mean(0, keepdim=True)
    mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
    alive = sd[0] > 1e-6
    Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd))
    Z[:, ~alive] = 0
    return Z, alive

def spanning_tree(R, alive):
    """Maximum spanning tree of |R| over alive neurons. Returns (i, j, |r|) in full-layer indices."""
    idx = torch.nonzero(alive).squeeze(1)
    A = R[idx][:, idx].abs().double().cpu().numpy()
    W = 1.0 - A + 1e-6                                   # min spanning tree of (1 - |r|) = max spanning tree of |r|
    np.fill_diagonal(W, 0.0)
    T = minimum_spanning_tree(W).tocoo()
    ix = idx.cpu().numpy()
    return ix[T.row].astype(np.int16), ix[T.col].astype(np.int16), (1.0 - T.data + 1e-6).astype(np.float16)

def transition(Za, Ra, alive_a, Zb, Rb, alive_b):
    """Hungarian matching a -> b, then per-neuron activation and connectivity similarity (robot paper)."""
    S = (Za.T @ Zb) / Za.shape[0]
    _, perm = linear_sum_assignment(-S.double().cpu().numpy())
    perm = torch.tensor(perm, device=DEVICE)
    act = S[torch.arange(len(perm), device=DEVICE), perm]
    A_ = Ra.clone(); A_.fill_diagonal_(0)
    B_ = Rb[perm][:, perm].clone(); B_.fill_diagonal_(0)
    conn = (A_ * B_).sum(1) / (A_.norm(dim=1) * B_.norm(dim=1) + 1e-12)
    valid = alive_a & alive_b[perm]
    return perm, act, conn, valid

def eff_rank(R, alive):
    """Participation ratio of the correlation spectrum: how many dimensions the layer really uses."""
    idx = torch.nonzero(alive).squeeze(1)
    ev = torch.linalg.eigvalsh(R[idx][:, idx].double()).clamp(min=0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())

def analyse(path, idx):
    model, ck = load_model(path)
    acts, dec_lang = record(model)
    if not idx:
        first_src = next(k for k in acts if on_source(k))
        idx["src"] = subsample_idx(acts[first_src].shape[0], args.max_t)
        idx["tgt"] = dec_subsample(dec_lang, args.max_t)
    layers = {}
    for pt, X in acts.items():
        Xs = X[idx["src"] if on_source(pt) else idx["tgt"]].float()
        Z, alive = traces(Xs, None if on_source(pt) else dec_lang[idx["tgt"]])
        R = Z.T @ Z / Z.shape[0]
        layers[pt] = (Z, R, alive, Xs.std(0, unbiased=False), Xs.mean(0), (Xs > 0).float().mean(0))
    wn = {k: round(float(v.float().norm()), 4) for k, v in model.state_dict().items() if v.dim() >= 1}
    del model, acts
    return ck, layers, wn

ckpts = sorted(glob.glob(P("runs", args.run, "ckpt", "p*.pt")))
ckpts = [c for c in ckpts if args.first <= int(os.path.basename(c)[1:4]) <= args.last]
print(f"{args.run}: {len(ckpts)} checkpoints -> {OUT}", flush=True)
idx, prev, t_run = {}, None, time.time()
for n, path in enumerate(ckpts):
    phase = int(os.path.basename(path)[1:4])
    out = os.path.join(OUT, f"p{phase:03d}.npz")
    if os.path.exists(out):
        prev = None                                        # recomputed if the next checkpoint needs it
        continue
    t0 = time.time()
    if prev is None and n > 0:                             # resuming: need the previous checkpoint for the switch
        _, prev, _ = analyse(ckpts[n - 1], idx)
    ck, layers, wn = analyse(path, idx)
    save = {"phase": np.array(phase), "lang": np.array(ck["lang"])}
    for pt, (Z, R, alive, sd, mu, fpos) in layers.items():
        i, j, w = spanning_tree(R, alive)
        save.update({f"{pt}/mst_i": i, f"{pt}/mst_j": j, f"{pt}/mst_w": w,
                     f"{pt}/alive": alive.cpu().numpy(), f"{pt}/std": sd.half().cpu().numpy(),
                     f"{pt}/mean": mu.half().cpu().numpy(), f"{pt}/frac_pos": fpos.half().cpu().numpy(),
                     f"{pt}/eff_rank": np.array(eff_rank(R, alive))})
        if prev is not None:
            Za, Ra, alive_a = prev[pt][:3]
            perm, act, conn, valid = transition(Za, Ra, alive_a, Z, R, alive)
            save.update({f"{pt}/perm": perm.cpu().numpy().astype(np.int16), f"{pt}/act": act.half().cpu().numpy(),
                         f"{pt}/conn": conn.half().cpu().numpy(), f"{pt}/valid": valid.cpu().numpy()})
    np.savez_compressed(out + ".tmp.npz", **save)
    os.replace(out + ".tmp.npz", out)
    with open(os.path.join(OUT, "weights.jsonl"), "a") as f:
        f.write(json.dumps({"phase": phase, "norms": wn}) + "\n")
    prev = {pt: v[:3] for pt, v in layers.items()}
    del layers
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    done = n + 1
    eta = (time.time() - t_run) / done * (len(ckpts) - done)
    print(f"  phase {phase:3d} ({ck['lang']}) done in {time.time() - t0:.0f}s | ETA {eta / 60:.0f} min", flush=True)
print("finished:", OUT, flush=True)
