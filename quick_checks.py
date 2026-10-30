"""quick_checks.py: the fast checks before training (no training, GPU minutes).

CHECK 1, pure noise (H3, "does our measure favour the self mechanically?").
    Take a checkpoint, add random Gaussian noise to every weight matrix (size = s x that matrix's own std),
    NO retraining, and measure self vs task vs random change exactly as in the main analysis.
    Noise has no task and no learning signal. If the self still "changes less", part of our gap is mechanical.
    It also gives the calibration curve  noise size s -> how much neurons change  (used to size the Q2 push).

CHECK 2, far-apart checkpoints (natural drift, "free" matched-movement check for Q2).
    Hindi only: compare phase p with p+k for growing k. Does task change ever reach ~45-50% (the continual level),
    and is the self still protected there?  Continual: same-language pairs far apart, and a few ordinary switches
    as the reference point.

Everything uses: MLP layers only, fixed tau = 0.2 (self = largest connected group of |r| >= 0.2),
FLORES dev inputs (same tokens for every checkpoint), Hungarian matching, change = 100 x (1 - (act + conn) / 2),
membership on the "after" side (as in the paper and our pipeline). Phases <= 60 only.

Run from the repo root (needs runs/v4b_continual_s0/ckpt and runs/v4b_only_hi_s0/ckpt):
    python -u quick_checks.py --gpu 0
Outputs in analysis/v6/quick_checks/: results.csv, summary.txt, q2_movement_vs_gap.png
About 20-30 minutes on one 4090.
"""
import os, sys, argparse, time, json, glob
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--noise", type=float, nargs="+", default=[0.03, 0.1, 0.2, 0.4, 0.5, 0.6])
ap.add_argument("--checks", nargs="+", default=["noise", "drift"], help="which checks to run: noise drift")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)
from common_v5 import *
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
torch.backends.cuda.matmul.allow_tf32 = False

OUT = P("analysis", "v6", "quick_checks"); os.makedirs(OUT, exist_ok=True)
RUN_C, RUN_H = "v4b_continual_s0", "v4b_only_hi_s0"
def ck(run, ph):
    """Checkpoints are saved as p{phase:03d}_{lang}.pt."""
    hits = sorted(glob.glob(P("runs", run, "ckpt", f"p{ph:03d}_*.pt")))
    if not hits:
        raise FileNotFoundError(f"no checkpoint for {run} phase {ph} in runs/{run}/ckpt/")
    return hits[0]
MLP = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]

# ---------------------------------------------------------------- fixed inputs (identical for every checkpoint)
DEV_EN = read_flores("dev", "en")
SRC_IDS = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT_IN = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}

@torch.no_grad()
def record(model, bs=64):
    """MLP activations on FLORES dev (encoder: English source tokens; decoder: hi/zh/ar teacher-forced)
    plus the teacher-forced masked loss per language (to see how much damage noise does)."""
    model.eval()
    pts = {k: m for k, m in probe_points(model).items() if k in MLP}
    cur = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, k=k: cur.__setitem__(k, o)) for k, m in pts.items()]
    acts, dec_lang = {k: [] for k in pts}, []
    loss_sum, loss_n = {l: 0.0 for l in LANGS}, {l: 0 for l in LANGS}
    for s in range(0, len(SRC_IDS), bs):
        src = pad_batch(SRC_IDS[s:s + bs])
        mem, src_mask = model.encode(src)
        skeep = (src != PAD) & (src != EOS)
        for k in pts:
            if k.startswith("enc"):
                acts[k].append(cur[k][skeep].half())
        for li, l in enumerate(LANGS):
            tgt = pad_batch(TGT_IN[l][s:s + bs])
            logits = model.decode(tgt, mem, src_mask).float()
            # teacher-forced loss: predict token t+1 from tokens <= t, script-masked like training
            lg = logits[:, :-1].masked_fill(~ALLOWED[l], float("-inf"))
            y = tgt[:, 1:]
            m = y != PAD
            lp = torch.log_softmax(lg, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
            m = m & torch.isfinite(lp)                      # skip the rare target token outside the allowed script
            loss_sum[l] += float(-(lp[m]).sum()); loss_n[l] += int(m.sum())
            tkeep = tgt != PAD
            tkeep[:, 0] = False
            for k in pts:
                if k.startswith("dec"):
                    acts[k].append(cur[k][tkeep].half())
            dec_lang.append(torch.full((int(tkeep.sum()),), li, device=DEVICE))
    for h in hooks:
        h.remove()
    loss = {l: loss_sum[l] / max(loss_n[l], 1) for l in LANGS}
    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(dec_lang), loss

IDX = {}
def fixed_idx(acts, dec_lang):
    if not IDX:
        g = np.random.default_rng(0)
        n_src = acts["enc0.mlp"].shape[0]
        IDX["src"] = torch.tensor(np.sort(g.choice(n_src, min(n_src, args.max_t), replace=False)), device=DEVICE)
        g, ln, out = np.random.default_rng(1), dec_lang.cpu().numpy(), []
        for li in range(len(LANGS)):
            pos = np.nonzero(ln == li)[0]
            out.append(np.sort(g.choice(pos, min(len(pos), args.max_t // len(LANGS)), replace=False)))
        IDX["tgt"] = torch.tensor(np.concatenate(out), device=DEVICE)
    return IDX

def traces(X, groups=None):
    X = X.float().clone()
    if groups is not None:
        for g in groups.unique():
            m = groups == g
            X[m] -= X[m].mean(0, keepdim=True)
    mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
    alive = sd[0] > 1e-6
    Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd))
    Z[:, ~alive] = 0
    return Z, alive

def layer_stats(model):
    """Per MLP layer: z-scored traces, correlation matrix, alive mask, self mask at tau."""
    acts, dec_lang, loss = record(model)
    idx = fixed_idx(acts, dec_lang)
    out = {}
    for pt in MLP:
        on_src = pt.startswith("enc")
        X = acts[pt][idx["src"] if on_src else idx["tgt"]]
        Z, alive = traces(X, None if on_src else dec_lang[idx["tgt"]])
        R = Z.T @ Z / Z.shape[0]
        A = ((R.abs() >= args.tau) & alive[:, None] & alive[None, :]).cpu().numpy()
        np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        lab = np.where(alive.cpu().numpy(), lab, -1)
        self_mask = lab == np.bincount(lab[lab >= 0]).argmax() if (lab >= 0).any() else np.zeros(len(lab), bool)
        out[pt] = (Z.half().cpu(), R.cpu(), alive.cpu(), self_mask)       # keep cached states off the GPU
    return out, loss

def compare(a, b, rng):
    """Robot-paper persistence between two states a -> b, per layer: self / task / random change (after-side membership)."""
    rows = []
    for pt in MLP:
        Za, Ra, alive_a, self_a = [x.to(DEVICE).float() if torch.is_tensor(x) and x.dtype != torch.bool else (x.to(DEVICE) if torch.is_tensor(x) else x) for x in a[pt]]
        Zb, Rb, alive_b, self_b = [x.to(DEVICE).float() if torch.is_tensor(x) and x.dtype != torch.bool else (x.to(DEVICE) if torch.is_tensor(x) else x) for x in b[pt]]
        S = (Za.T @ Zb) / Za.shape[0]
        _, perm = linear_sum_assignment(-S.double().cpu().numpy())
        perm_t = torch.tensor(perm, device=DEVICE)
        act = S[torch.arange(len(perm), device=DEVICE), perm_t]
        A_ = Ra.clone(); A_.fill_diagonal_(0)
        B_ = Rb[perm_t][:, perm_t].clone(); B_.fill_diagonal_(0)
        conn = (A_ * B_).sum(1) / (A_.norm(dim=1) * B_.norm(dim=1) + 1e-12)
        valid = (alive_a & alive_b[perm_t]).cpu().numpy()
        change = 100 * (1 - ((act + conn) / 2).cpu().numpy())
        sb = self_b[perm]                              # partner's membership after
        ins, intask = valid & sb, valid & ~sb
        vidx = np.nonzero(valid)[0]; ns = int(ins.sum())
        rc = float(np.mean([change[rng.choice(vidx, ns, replace=False)].mean() for _ in range(20)])) if 0 < ns < len(vidx) else np.nan
        union = (self_a | sb).sum()
        rows.append({"point": pt, "self_change": change[ins].mean() if ns else np.nan,
                     "task_change": change[intask].mean() if intask.any() else np.nan, "random_change": rc,
                     "self_frac_before": self_a[valid].mean(), "self_frac_after": sb[valid].mean(),
                     "self_stays": (self_a & sb).sum() / max(self_a.sum(), 1), "self_iou": (self_a & sb).sum() / max(union, 1)})
    del S, A_, B_
    torch.cuda.empty_cache()
    return rows

CACHE = {}
def stats_of(run, ph):
    key = (run, ph)
    if key not in CACHE:
        model, _ = load_model(ck(run, ph))
        CACHE[key] = layer_stats(model)
        del model; torch.cuda.empty_cache()
    return CACHE[key]

ROWS = []
# keep earlier results of checks we are not re-running (so "--checks drift" does not wipe the noise results)
_prev = os.path.join(OUT, "results.csv")
if os.path.exists(_prev):
    _keep = {"noise": ["noise"], "drift": ["drift_hi", "drift_cont", "switch_cont"]}
    _drop = sum((_keep[c] for c in args.checks), [])
    _old = pd.read_csv(_prev)
    ROWS = _old[~_old.check.isin(_drop)].to_dict("records")
def add(rows, **meta):
    for r in rows:
        r.update(meta)
        r["gap"] = r["task_change"] - r["self_change"]
        r["rel_gap_pct"] = 100 * r["gap"] / r["task_change"] if r["task_change"] else np.nan
        ROWS.append(r)
    d = pd.DataFrame(rows)
    print(f"  {meta['check']:<12} {meta['label']:<34} task {d.task_change.mean():5.1f}  self {d.self_change.mean():5.1f}  "
          f"random {d.random_change.mean():5.1f}  gap {(d.task_change - d.self_change).mean():5.1f}  "
          f"rel {100 * (d.task_change - d.self_change).mean() / d.task_change.mean():5.1f}%  "
          f"self size {100 * d.self_frac_after.mean():4.0f}%  {meta.get('extra', '')}", flush=True)
    pd.DataFrame(ROWS).to_csv(os.path.join(OUT, "results.csv"), index=False)

t0 = time.time()
rng = np.random.default_rng(0)
print(f"tau = {args.tau} | MLP layers | inputs: FLORES dev | output -> {OUT}\n", flush=True)

# ---------------------------------------------------------------- CHECK 1: pure noise, no retraining
print("CHECK 1: pure noise on the weights, no retraining (does the measure favour the self by itself?)", flush=True)
for run, ph in ([(RUN_C, 45), (RUN_H, 45)] if "noise" in args.checks else []):
    base, base_loss = stats_of(run, ph)
    for s in args.noise:
        model, _ = load_model(ck(run, ph))
        g = torch.Generator(device=DEVICE).manual_seed(123)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2:                                   # weight matrices and embeddings, not norms/biases
                    p.add_(torch.randn(p.shape, generator=g, device=DEVICE) * s * p.std())
        noisy, loss = layer_stats(model)
        del model; torch.cuda.empty_cache()
        dl = {l: round(loss[l] - base_loss[l], 3) for l in LANGS}
        add(compare(base, noisy, rng), check="noise", run=run, phase_a=ph, phase_b=ph, noise=s,
            label=f"{run.split('_')[1]} p{ph} + noise {s}", extra=f"loss +{dl}")

# ---------------------------------------------------------------- CHECK 2: far-apart checkpoints (natural drift)
if "drift" in args.checks:
    print("\nCHECK 2: far-apart checkpoints, no switching in between for Hindi only (natural drift)", flush=True)
    for a, b in [(30, 31), (30, 35), (30, 40), (30, 50), (30, 60), (10, 60), (2, 60)]:
        add(compare(stats_of(RUN_H, a)[0], stats_of(RUN_H, b)[0], rng), check="drift_hi", run=RUN_H, phase_a=a, phase_b=b,
            noise=0.0, label=f"Hindi only p{a} -> p{b}")
    print("\n  continual: same language far apart (phase 30, 33, 45, 60 are all Arabic) and ordinary switches as reference", flush=True)
    for a, b in [(30, 33), (30, 45), (30, 60)]:
        add(compare(stats_of(RUN_C, a)[0], stats_of(RUN_C, b)[0], rng), check="drift_cont", run=RUN_C, phase_a=a, phase_b=b,
            noise=0.0, label=f"continual p{a} -> p{b} (same language)")
    for a, b in [(44, 45), (45, 46), (46, 47), (59, 60)]:
        add(compare(stats_of(RUN_C, a)[0], stats_of(RUN_C, b)[0], rng), check="switch_cont", run=RUN_C, phase_a=a, phase_b=b,
            noise=0.0, label=f"continual switch p{a} -> p{b}")

# ---------------------------------------------------------------- summary + the Q2 figure
D = pd.DataFrame(ROWS)
G = D.groupby(["check", "label"], sort=False).agg(task=("task_change", "mean"), self_ch=("self_change", "mean"),
                                                   random=("random_change", "mean"), gap=("gap", "mean"),
                                                   self_size=("self_frac_after", "mean"), stays=("self_stays", "mean")).reset_index()
G["rel_gap_pct"] = 100 * G.gap / G.task
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write(G.round(2).to_string(index=False))
print("\n" + G.round(1).to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
style = {"noise": ("o", "#C8102E", "pure noise (no task, no training)"), "drift_hi": ("s", "#555555", "Hindi only, far apart"),
         "drift_cont": ("^", "#2E8B57", "continual, same language far apart"), "switch_cont": ("D", "#E9B949", "continual, language switch")}
for chk, (mk, col, lab) in style.items():
    d = G[G.check == chk]
    ax[0].scatter(d.task, d.gap, marker=mk, color=col, s=60, label=lab)
    ax[1].scatter(d.task, d.rel_gap_pct, marker=mk, color=col, s=60, label=lab)
ax[0].set_ylabel("gap = task change - self change (points)"); ax[1].set_ylabel("relative gap = gap / task change (%)")
for a in ax:
    a.set_xlabel("how much neurons move: task change (%)"); a.axhline(0, color="#999", lw=0.8)
ax[0].legend(fontsize=8.5)
fig.suptitle(f"Is the self more protected under switching than under any disturbance of the same size? (MLP mean, tau {args.tau})", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "q2_movement_vs_gap.png"), dpi=150)
print(f"\ndone in {(time.time() - t0) / 60:.0f} min -> {OUT}", flush=True)