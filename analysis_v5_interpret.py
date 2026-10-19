"""analysis_v5_interpret.py: second GPU pass (run AFTER analysis_v5_compute.py has finished for BOTH runs).

At a few chosen phases, for the 8 MLP layers, both runs:
  1. LESION TEST (the robot paper's intervention, inference-only version): replace a group of neurons by their mean
     activation and measure how much the validation loss rises in each language. Groups: the self, the core (neurons
     that stayed in the self across the last switch), and size-matched random task neurons (3 draws) as the control.
  2. TOP TOKENS (Goodfire "already clean" idea): the FLORES tokens that 15 core neurons and 15 random task neurons fire
     on most (per language for decoder layers), and MEANING STABILITY: how much a neuron's top-50 positions stay the
     same from one chosen phase to the next (Jaccard), for core vs self vs task neurons.
  3. PAPER-STYLE FIGURES (robot paper Fig. 3 / 19-21): reordered co-activation map + per-neuron persistence,
     Hindi only vs continual side by side, one image per layer and phase.

tau per layer = tau50: smallest tau where the median self covers <= 50% of the layer, pooled over phases 43-60 of both runs.
Output: analysis/v5/interpret/   Usage: python -u analysis_v5_interpret.py --gpu 0
"""
import os, sys, argparse, glob, json, time

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--phases", default="18,45,60,72,84")
ap.add_argument("--n_val", type=int, default=500)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)
from common_v5 import *
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as patches
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee

torch.backends.cuda.matmul.allow_tf32 = False
RUNS = {"only_hi": "v4b_only_hi_s0", "continual": "v4b_continual_s0"}
MLP = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]
BASE = P("analysis", "v5")
OUT = os.path.join(BASE, "interpret"); os.makedirs(os.path.join(OUT, "paper_style"), exist_ok=True)
PHASES = [int(x) for x in args.phases.split(",")]
TAU_GRID = np.round(np.arange(0.05, 0.6001, 0.01), 2)
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ groups from the stored spanning trees
def npz(run, ph):
    f = os.path.join(BASE, run, f"p{ph:03d}.npz")
    return np.load(f) if os.path.exists(f) else None

def groups_at(d, pt, tau):
    alive = d[f"{pt}/alive"]; n = len(alive)
    w = d[f"{pt}/mst_w"].astype(np.float32); keep = w >= tau
    i, j = d[f"{pt}/mst_i"].astype(np.int64)[keep], d[f"{pt}/mst_j"].astype(np.int64)[keep]
    _, lab = connected_components(csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    lab = np.where(alive, lab, -1)
    return lab == np.bincount(lab[lab >= 0]).argmax(), lab

def tau50():
    out = {}
    for pt in MLP:
        fr = {t: [] for t in TAU_GRID}
        for run in RUNS.values():
            for ph in range(43, 61):
                d = npz(run, ph)
                if d is None:
                    continue
                na = d[f"{pt}/alive"].sum()
                for t in TAU_GRID:
                    fr[t].append(groups_at(d, pt, t)[0].sum() / max(na, 1))
        med = {t: np.median(v) for t, v in fr.items() if v}
        below = [t for t, m in med.items() if m <= 0.5]
        out[pt] = round(float(min(below)), 2) if below else 0.3
    return out
TAU = tau50()
json.dump(TAU, open(os.path.join(OUT, "tau50_used.json"), "w"), indent=2)
print("tau50 per layer:", TAU, flush=True)

# ------------------------------------------------------------------ data
VAL = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:args.n_val] for l in LANGS}
DEV_EN = read_flores("dev", "en")
SRC_IDS = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT_IN = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}

def act_module(model, pt):
    side, j = pt[:3], int(pt[3])
    return (model.enc_blocks if side == "enc" else model.dec_blocks)[j].mlp.act

@torch.no_grad()
def losses(model, pt=None, idx=None, value=None):
    """Masked val loss per language; optionally neurons `idx` of layer `pt` replaced by `value` (mean ablation)."""
    h = None
    if pt is not None:
        ii = torch.tensor(np.asarray(idx), device=DEVICE, dtype=torch.long)
        vv = torch.tensor(np.asarray(value), device=DEVICE, dtype=torch.float32)
        def hook(mod, inp, out):
            out = out.clone(); out[..., ii] = vv.to(out.dtype); return out
        h = act_module(model, pt).register_forward_hook(hook)
    res = {l: val_loss(model, VAL[l], l, masked=True) for l in LANGS}
    if h:
        h.remove()
    return res

@torch.no_grad()
def record_mlp(model, bs=64):
    """MLP activations at every position + the input token there (rows aligned).
    Encoder layers: English source tokens. Decoder layers: target tokens; dlang = language of each row."""
    model.eval(); cur = {}
    hooks = [act_module(model, pt).register_forward_hook(lambda m, i, o, k=pt: cur.__setitem__(k, o)) for pt in MLP]
    acts = {pt: [] for pt in MLP}; stok, dtok, dlang = [], [], []
    for s in range(0, len(SRC_IDS), bs):
        src = pad_batch(SRC_IDS[s:s + bs]); mem, src_mask = model.encode(src)
        sk = (src != PAD) & (src != EOS)
        for pt in MLP[:4]:
            acts[pt].append(cur[pt][sk].half())
        stok.append(src[sk])
        for li, l in enumerate(LANGS):
            tgt = pad_batch(TGT_IN[l][s:s + bs]); model.decode(tgt, mem, src_mask)
            tk = tgt != PAD; tk[:, 0] = False
            for pt in MLP[4:]:
                acts[pt].append(cur[pt][tk].half())
            dtok.append(tgt[tk]); dlang.append(torch.full((int(tk.sum()),), li, device=DEVICE))
    for h in hooks:
        h.remove()
    return ({k: torch.cat(v).float() for k, v in acts.items()}, torch.cat(stok).cpu().numpy(),
            torch.cat(dtok).cpu().numpy(), torch.cat(dlang).cpu().numpy())

def corr(X, groups=None):
    X = X.clone()
    if groups is not None:
        g = torch.tensor(groups, device=X.device)
        for v in g.unique():
            m = g == v; X[m] -= X[m].mean(0, keepdim=True)
    sd = X.std(0, unbiased=False); Z = (X - X.mean(0)) / torch.where(sd > 1e-6, sd, torch.ones_like(sd))
    return (Z.T @ Z / Z.shape[0]).cpu().numpy()

piece = lambda t: sp.id_to_piece(int(t)).replace("▁", "_")

def paper_panel(R, d, pt, tau):
    """Order: groups largest first (RCM inside each), then the rest. Persistence at the switch into this phase."""
    smask, lab = groups_at(d, pt, tau)
    n = len(lab)
    A = (np.abs(R) >= tau) & (lab[:, None] >= 0) & (lab[None, :] >= 0); np.fill_diagonal(A, False)
    sizes = np.bincount(lab[lab >= 0]); order, boxes = [], []
    for k, g in enumerate([g for g in np.argsort(-sizes) if sizes[g] >= 2]):
        idx = np.nonzero(lab == g)[0]
        sub = reverse_cuthill_mckee(csr_matrix(A[np.ix_(idx, idx)]), symmetric_mode=True)
        boxes.append((len(order), len(idx), k == 0)); order.extend(idx[sub])
    used = set(order); order = np.array(order + [i for i in range(n) if i not in used])
    pers = np.full(n, np.nan)
    if f"{pt}/perm" in d.files:
        perm = d[f"{pt}/perm"].astype(np.int64); valid = d[f"{pt}/valid"]
        p = (d[f"{pt}/act"].astype(np.float32) + d[f"{pt}/conn"].astype(np.float32)) / 2
        pers[perm[valid]] = p[valid]
    return np.abs(R[np.ix_(order, order)]), boxes, pers[order], smask[order]

def paper_figure(panels, pt, ph, tau, langs):
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[1, 0.28], hspace=0.08, wspace=0.12)
    for col, (name, (M, boxes, pers, is_self)) in enumerate(panels.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(M, cmap="Greys", vmin=0, vmax=0.6, interpolation="nearest")
        for start, size, sb in boxes:
            if size >= 10 or sb:
                ax.add_patch(patches.Rectangle((start - .5, start - .5), size, size, fill=False,
                                               edgecolor="#4B1D6B" if sb else "#E07A1F", lw=2.2 if sb else 0.8))
        ax.set_title(f"{name}: phase {ph - 1} -> {ph} ({langs[name]})\nself {int(is_self.sum())} of {len(is_self)} "
                     f"({100 * is_self.mean():.0f}%)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax2 = fig.add_subplot(gs[1, col])
        ax2.bar(np.arange(len(pers)), np.nan_to_num(pers), width=1.0, color=np.where(is_self, "#4B1D6B", "#B8B8B8"))
        ms, mt = np.nanmean(pers[is_self]), np.nanmean(pers[~is_self])
        ax2.axhline(ms, color="#4B1D6B", ls="--", lw=1); ax2.axhline(mt, color="#555555", ls="--", lw=1)
        ax2.text(len(pers) * .99, .04, f"mean persistence: self {ms:.2f} · task {mt:.2f}", ha="right", fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
        ax2.set_xlim(-.5, len(pers) - .5); ax2.set_ylim(0, 1); ax2.set_xticks([])
        if col == 0:
            ax2.set_ylabel("persistence", fontsize=9)
        else:
            ax2.set_yticks([])
    fig.suptitle(f"Layer {pt}: co-activation |r| (groups at |r| >= {tau}, self in purple) and per-neuron persistence", fontsize=11)
    return fig

lesion_rows, token_rows, stab_rows = [], [], []
prev_top = {}                                    # (run, pt) -> (phase, topk positions [n, 50], groups dict)
for ph in PHASES:
    panels, langs = {pt: {} for pt in MLP}, {}
    for name, run in RUNS.items():
        cks = glob.glob(P("runs", run, "ckpt", f"p{ph:03d}_*.pt"))
        d, dprev = npz(run, ph), npz(run, ph - 1)
        if not cks or d is None:
            print(f"skip {name} phase {ph}"); continue
        t0 = time.time()
        model, ck = load_model(cks[0])
        langs[name] = f"{dprev['lang']} -> {ck['lang']}" if dprev is not None and str(dprev['lang']) != ck['lang'] else ck["lang"]
        base = losses(model)
        acts, stok, dtok, dlang = record_mlp(model)
        for pt in MLP:
            tau = TAU[pt]
            smask, _ = groups_at(d, pt, tau)
            alive = d[f"{pt}/alive"]; mean = d[f"{pt}/mean"].astype(np.float32)
            core = np.zeros_like(smask)
            if dprev is not None and f"{pt}/perm" in d.files:
                sa, _ = groups_at(dprev, pt, tau)
                perm = d[f"{pt}/perm"].astype(np.int64)
                stay = sa & smask[perm]
                core[perm[stay]] = True
            task = np.nonzero(alive & ~smask)[0]
            G = {"self": np.nonzero(smask)[0], "core": np.nonzero(core)[0], "task": task}
            # 1. lesion
            for gname in ["self", "core"]:
                gidx = G[gname]
                if len(gidx) == 0 or len(gidx) > len(task):
                    continue
                r = losses(model, pt, gidx, mean[gidx])
                ctrl = []
                for _ in range(3):
                    sel = rng.choice(task, len(gidx), replace=False)
                    ctrl.append(losses(model, pt, sel, mean[sel]))
                for l in LANGS:
                    lesion_rows.append({"run": name, "phase": ph, "lang_trained": ck["lang"], "point": pt, "tau": tau,
                                        "group": gname, "n": len(gidx), "frac_of_layer": len(gidx) / max(alive.sum(), 1),
                                        "eval_lang": l, "base_loss": base[l], "delta_group": r[l] - base[l],
                                        "delta_random_task": float(np.mean([c[l] - base[l] for c in ctrl]))})
            # 2. top tokens + meaning stability
            A = acts[pt]
            top50 = torch.topk(A, 50, dim=0).indices.T.cpu().numpy()          # [neurons, 50] positions
            key = (name, pt)
            if key in prev_top:
                pph, ptop, pG = prev_top[key]
                for gname in ["core", "self", "task"]:
                    ids = pG[gname]
                    if len(ids):
                        jac = [len(set(ptop[i]) & set(top50[i])) / len(set(ptop[i]) | set(top50[i])) for i in ids]
                        stab_rows.append({"run": name, "point": pt, "from_phase": pph, "to_phase": ph, "group": gname,
                                          "n": len(ids), "top50_jaccard": float(np.mean(jac))})
            prev_top[key] = (ph, top50, G)
            enc = pt.startswith("enc")
            pool = G["core"] if len(G["core"]) else G["self"]
            for kind, ids in [("core", rng.choice(pool, min(15, len(pool)), replace=False) if len(pool) else []),
                              ("task", rng.choice(task, min(15, len(task)), replace=False) if len(task) else [])]:
                for nid in ids:
                    a = A[:, nid].cpu().numpy()
                    if enc:
                        token_rows.append({"run": name, "phase": ph, "point": pt, "group": kind, "neuron": int(nid),
                                           "lang": "en (source)", "top_tokens": " ".join(piece(stok[k]) for k in np.argsort(-a)[:12])})
                    else:
                        for li, l in enumerate(LANGS):
                            pos = np.nonzero(dlang == li)[0]
                            top = pos[np.argsort(-a[pos])[:12]]
                            token_rows.append({"run": name, "phase": ph, "point": pt, "group": kind, "neuron": int(nid),
                                               "lang": l, "top_tokens": " ".join(piece(dtok[k]) for k in top)})
            # 3. panel for the paper-style figure
            R = corr(A, None if enc else dlang)
            panels[pt][name] = paper_panel(R, d, pt, tau)
            del R
        del model, acts
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"{name} phase {ph}: done in {time.time() - t0:.0f}s", flush=True)
    for pt in MLP:
        if len(panels[pt]) == 2:
            fig = paper_figure(panels[pt], pt, ph, TAU[pt], langs)
            fig.savefig(os.path.join(OUT, "paper_style", f"{pt}_p{ph:03d}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)
    pd.DataFrame(lesion_rows).to_csv(os.path.join(OUT, "lesion.csv"), index=False)
    pd.DataFrame(token_rows).to_csv(os.path.join(OUT, "top_tokens.csv"), index=False)
    pd.DataFrame(stab_rows).to_csv(os.path.join(OUT, "meaning_stability.csv"), index=False)
print("finished:", OUT, flush=True)
