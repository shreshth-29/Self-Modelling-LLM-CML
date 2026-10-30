"""analysis_v6_fix.py: last attempt at the analyses that were uninformative or flawed in analysis_v6_deep.py.

Parts (--parts, default all):
  q2pairs     Q2 WITHOUT new training. The push run moved the network only 16 (not 47), so compare at MATCHED movement
              using checkpoints further apart. Most important: inside ONE network (continual), pairs at the same
              movement that end in the SAME language vs a DIFFERENT language. If the self is hit harder only when the
              language differs, the switch itself (not the amount of movement) is what disrupts it.
              Also: Hindi-only and push run, far-apart pairs (big movement, never a switch).
  crosscoder  fixed crosscoder: the two decoders start independent, sparsity penalty weighted by decoder norms (so a
              feature can drop out of one model), self-tuning penalty (about 40 active features per token), 15k steps,
              dead features reported, plus two CONTROLS: a model against itself (must be ~all shared) and a model that
              knows Arabic against one that does not, on Arabic text (must show exclusive features).
  steer       fixed "force a neuron on": instead of counting output words of greedy translations (too insensitive),
              measure how much the next-token predictions change (KL divergence), for self vs task neurons, decoder
              layers 2 and 3, at 3x and 10x the neuron's 99.9th-percentile value, and which tokens gain.
  relearn     fixed lesion + relearn: gentle learning rate (warm-up 50 steps to 3e-5), so relearning no longer knocks
              the model around (before: loss went back up after 100 steps).

Usage: python3 -u analysis_v6_fix.py --gpu 1          (needs the checkpoints listed by --list)
"""
import os, sys, argparse, json, glob, time, math
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--parts", nargs="+", default=["training", "pairs", "noise", "selfreturn", "crosscoder", "steer", "relearn"])
ap.add_argument("--cont", default="v4b_continual_s0")
ap.add_argument("--hi", default="v4b_only_hi_s0")
ap.add_argument("--push", default="v6_hi_push0.6_from30")
ap.add_argument("--mixed", default="v6b_mixed_s0")
ap.add_argument("--redo", default="v6b_cont_redo_from60")
ap.add_argument("--fix", default="v6_cont_noise0.0007_from60")
ap.add_argument("--n_steer", type=int, default=100)
ap.add_argument("--phase", type=int, default=45)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--cc_steps", type=int, default=8000)
ap.add_argument("--cc_dict", type=int, default=8192)
ap.add_argument("--cc_l0", type=float, default=40)
ap.add_argument("--relearn_steps", type=int, default=400)
ap.add_argument("--lesion_frac", type=float, default=0.2)
ap.add_argument("--tag", default="c")
ap.add_argument("--list", action="store_true", help="print the checkpoints needed and exit")
ap.add_argument("--small", action="store_true")
args = ap.parse_args()

PAIRS = {  # name: (run arg, anchor phase, target phases)
    "continual": ("cont", 30, [31, 32, 33, 39, 40, 44, 45, 52, 60]),
    "hindi": ("hi", 30, [31, 35, 40, 45, 50, 58]),
    "push": ("push", 30, [31, 33, 35, 40, 45, 50, 58]),
    "mixed": ("mixed", 30, [31, 33, 35, 40, 45, 50, 58]),
}
NOISE_CK = [("continual", "cont", p) for p in (20, 30, 45, 60)] + [("hindi", "hi", p) for p in (20, 30, 45, 60)] + \
           [("mixed", "mixed", p) for p in (20, 30, 45, 60)] + [("push", "push", 40), ("redo", "redo", 75)]
RETURN = {"continual": ("cont", list(range(30, 61))), "hindi": ("hi", list(range(30, 61, 3))), "mixed": ("mixed", list(range(30, 61, 3)))}
def needed():
    need = {}
    add = lambda key, phs: need.setdefault(getattr(args, key), set()).update(phs)
    for name, (key, a, ts) in PAIRS.items():
        add(key, [a] + ts)
    for _, key, p in NOISE_CK:
        add(key, [p])
    for name, (key, phs) in RETURN.items():
        add(key, phs)
    for key in ("cont", "hi", "mixed"):
        add(key, [44, 45, 46, 47])
    return {k: sorted(v) for k, v in need.items()}
if args.list:
    print(json.dumps(needed())); sys.exit()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
from common_v6 import *
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
torch.backends.cuda.matmul.allow_tf32 = False
if args.small:
    args.cc_steps, args.cc_dict, args.relearn_steps, args.max_t = 60, 256, 6, 2000

OUT = P("analysis", "v6", f"v6c_{args.tag}"); os.makedirs(OUT, exist_ok=True)
MLP = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]
rng = np.random.default_rng(0); torch.manual_seed(0)
REPORT = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REPORT.append(s)
    open(os.path.join(OUT, "report.txt"), "w").write("\n".join(REPORT))

def ck(run, ph):
    f = glob.glob(P("runs", run, "ckpt", f"p{ph:03d}_*.pt"))
    if not f:
        raise FileNotFoundError(f"runs/{run}/ckpt/p{ph:03d}_*.pt (download it first)")
    return f[0]

DEV_EN = read_flores("dev", "en")
SRC = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}
piece = lambda t: sp.id_to_piece(int(t)).replace("▁", "_")

def mlp_mod(model, pt):
    return (model.enc_blocks if pt.startswith("enc") else model.dec_blocks)[int(pt[3])].mlp

@torch.no_grad()
def record(model, pts=MLP, langs=LANGS, bs=64):
    """Post-GELU activations at every real token (CPU, fp16); token ids and language per decoder row."""
    model.eval(); cur = {}
    hooks = [mlp_mod(model, pt).act.register_forward_hook(lambda mm, i, o, k=pt: cur.__setitem__(k, o)) for pt in pts]
    acts = {pt: [] for pt in pts}; dtok, dlang = [], []
    for s in range(0, len(SRC), bs):
        S_ = pad_batch(SRC[s:s + bs]); mem, sm = model.encode(S_)
        sk = (S_ != PAD) & (S_ != EOS)
        for pt in pts:
            if pt.startswith("enc"):
                acts[pt].append(cur[pt][sk].half().cpu())
        for l in langs:
            T_ = pad_batch(TGT[l][s:s + bs]); model.decode(T_, mem, sm)
            tk = T_ != PAD; tk[:, 0] = False
            for pt in pts:
                if pt.startswith("dec"):
                    acts[pt].append(cur[pt][tk].half().cpu())
            dtok.append(T_[tk].cpu()); dlang.append(torch.full((int(tk.sum()),), LANGS.index(l)))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(dtok).numpy(), torch.cat(dlang).numpy()

# ------------------------------------------------------------------ the deck's measure (same as quick checks)
IDX = {}
def fixed_idx(n_src, dlang):
    if not IDX:
        g = np.random.default_rng(0)
        IDX["src"] = torch.tensor(np.sort(g.choice(n_src, min(n_src, args.max_t), replace=False)))
        g, out = np.random.default_rng(1), []
        for li in range(len(LANGS)):
            pos = np.nonzero(dlang == li)[0]
            out.append(np.sort(g.choice(pos, min(len(pos), args.max_t // len(LANGS)), replace=False)))
        IDX["tgt"] = torch.tensor(np.concatenate(out))
    return IDX

def stats(model):
    """Per MLP layer: z-scored traces, correlation, alive, self mask at tau (decoder: each language's mean removed)."""
    acts, dtok, dlang = record(model)
    idx = fixed_idx(acts["enc0.mlp"].shape[0], dlang)
    out = {}
    for pt in MLP:
        enc = pt.startswith("enc")
        X = acts[pt][idx["src"] if enc else idx["tgt"]].float().to(DEVICE)
        if not enc:
            g = torch.as_tensor(dlang)[idx["tgt"]].to(DEVICE)
            for v in g.unique():
                m = g == v; X[m] -= X[m].mean(0, keepdim=True)
        mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
        alive = sd[0] > 1e-6
        Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd)); Z[:, ~alive] = 0
        R = Z.T @ Z / Z.shape[0]
        A = ((R.abs() >= args.tau) & alive[:, None] & alive[None, :]).cpu().numpy(); np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        lab = np.where(alive.cpu().numpy(), lab, -1)
        smask = lab == np.bincount(lab[lab >= 0]).argmax() if (lab >= 0).any() else np.zeros(len(lab), bool)
        out[pt] = (Z.half().cpu(), R.cpu(), alive.cpu(), smask)
        del X, Z, R
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out

def compare(a, b):
    """Robot-paper change a -> b per layer; returns mean over layers + encoder/decoder means."""
    rows = []
    for pt in MLP:
        Za, Ra, al_a, sa = a[pt]; Zb, Rb, al_b, sb = b[pt]
        Za, Zb, Ra, Rb = Za.float().to(DEVICE), Zb.float().to(DEVICE), Ra.to(DEVICE), Rb.to(DEVICE)
        S = (Za.T @ Zb) / Za.shape[0]
        _, perm = linear_sum_assignment(-S.double().cpu().numpy())
        pt_ = torch.tensor(perm, device=DEVICE)
        act = S[torch.arange(len(perm), device=DEVICE), pt_]
        A_ = Ra.clone(); A_.fill_diagonal_(0); B_ = Rb[pt_][:, pt_].clone(); B_.fill_diagonal_(0)
        conn = (A_ * B_).sum(1) / (A_.norm(dim=1) * B_.norm(dim=1) + 1e-12)
        valid = (al_a.numpy() & al_b.numpy()[perm])
        change = 100 * (1 - ((act + conn) / 2).cpu().numpy())
        sbp = sb[perm]; ins, intask = valid & sbp, valid & ~sbp
        rows.append({"point": pt, "task": change[intask].mean() if intask.any() else np.nan,
                     "self": change[ins].mean() if ins.any() else np.nan, "size": sbp[valid].mean(),
                     "stays": (sa & sbp).sum() / max(sa.sum(), 1)})
        del S, A_, B_, Za, Zb, Ra, Rb
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run_of(key):
    return getattr(args, key)

def exists(run, ph):
    return bool(glob.glob(P("runs", run, "ckpt", f"p{ph:03d}_*.pt")))

# ================================================================== 0. training curves (CPU)
def phases_of(run):
    f = P("runs", run, "phases.jsonl")
    if not os.path.exists(f):
        return pd.DataFrame()
    rows = []
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            if "chrf_masked" in r:
                rows.append({"phase": r["phase"], "lang": r["lang"], **{f"chrf_{x}": r["chrf_masked"][x] for x in LANGS},
                             **{f"val_{x}": r["val_masked"][x] for x in LANGS},
                             "rarely_active": (r.get("health") or {}).get("rarely_active"), "rank_enc1": (r.get("health") or {}).get("rank_enc1"),
                             "steps": r.get("steps")})
    return pd.DataFrame(rows).drop_duplicates("phase", keep="last").sort_values("phase")

def part_training():
    say("\n=== TRAINING: mixed model and ReDo run vs the references ===")
    T = {k: phases_of(run_of(k)) for k in ["cont", "hi", "mixed", "redo", "fix"]}
    for k, d in T.items():
        say(f"  {run_of(k):<30} phases {int(d.phase.min()) if len(d) else '-'}..{int(d.phase.max()) if len(d) else '-'}")
    # A: mixed vs continual vs Hindi only: chrF of each language at the end of every phase
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), sharey=False)
    for ax, l in zip(axs, LANGS):
        for k, lab, c in [("mixed", "mixed (all 3 at once)", "#2E8B57"), ("cont", "continual (one at a time)", "#C0392B"), ("hi", "Hindi only", "#2A6F97")]:
            d = T[k]
            if len(d) and not (k == "hi" and l != "hi"):
                d = d[d.phase.between(1, 60)]
                ax.plot(d.phase, d[f"chrf_{l}"], lw=1.5 if k != "mixed" else 2.5, color=c, label=lab, alpha=.85)
        ax.set_title(f"chrF {l} after every phase"); ax.set_xlabel("phase"); ax.grid(alpha=.3)
    axs[0].legend(fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_mixed.png"), dpi=140)
    # B: ReDo vs fix vs original after phase 60: chrF on the language just trained + dead neurons
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    for k, lab, c, ls in [("cont", "original continual", "#C0392B", "-"), ("fix", "+ per-step noise", "#E07A1F", "--"), ("redo", "+ ReDo (reset dead neurons)", "#4B1D6B", "-")]:
        d = T[k]
        if len(d):
            d = d[d.phase.between(58, 90)]
            own = [r[f"chrf_{r['lang']}"] if r["lang"] in LANGS else np.nan for _, r in d.iterrows()]
            axs[0].plot(d.phase, own, ls, color=c, marker="o", ms=3, label=lab)
            if d.rarely_active.notna().any():
                axs[1].plot(d.phase, 100 * d.rarely_active, ls, color=c, marker="o", ms=3, label=lab)
    axs[0].set_title("chrF on the language just trained"); axs[1].set_title("neurons that almost never fire (%)")
    for a in axs:
        a.set_xlabel("phase"); a.grid(alpha=.3); a.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_redo.png"), dpi=140)
    for k in ["cont", "fix", "redo"]:
        d = T[k]
        if d.empty:
            continue
        for ph in (61, 70, 80, 86, 90):
            q = d[d.phase == ph]
            if len(q):
                r = q.iloc[0]; say(f"  {run_of(k):<30} phase {ph} ({r['lang']}): chrF on it {r['chrf_' + r['lang']]:.1f} | rarely active {r['rarely_active']}")
    d = T["mixed"]
    if not d.empty:
        for ph in (10, 20, 30, 45, 60):
            q = d[d.phase == ph]
            if len(q):
                r = q.iloc[0]; say(f"  mixed phase {ph}: chrF hi {r.chrf_hi:.1f} zh {r.chrf_zh:.1f} ar {r.chrf_ar:.1f} | steps {r.steps}")

# ================================================================== 1. matched-movement pairs (now incl. the mixed model)
def part_pairs():
    say("\n=== PAIRS at matched movement (tau", args.tau, ", MLP mean), now with the mixed model ===")
    rows = []
    for name, (key, a, ts) in PAIRS.items():
        run = run_of(key)
        if not exists(run, a):
            say("  skip", name, "(no anchor checkpoint)"); continue
        m, ca = load_model(ck(run, a)); A = stats(m); del m
        for t in ts:
            if not exists(run, t):
                continue
            m, cb = load_model(ck(run, t)); B = stats(m); del m
            d = compare(A, B)
            kind = {"hindi": "Hindi only", "push": "Hindi + push", "mixed": "mixed (all 3 at once)"}.get(name)
            if name == "continual":
                kind = "continual, SAME language" if ca["lang"] == cb["lang"] else "continual, DIFFERENT language"
            r = {"series": kind, "from": a, "to": t, "langs": f"{ca['lang']}->{cb['lang']}", "task": d.task.mean(), "self": d.self.mean(),
                 "rel_gap": 100 * (d.task.mean() - d.self.mean()) / d.task.mean(), "size": d["size"].mean(), "stays": d.stays.mean()}
            rows.append(r)
            say(f"  {kind:<32} p{a}->p{t} ({r['langs']}): other neurons {r['task']:5.1f}  self {r['self']:5.1f}  rel gap {r['rel_gap']:5.1f}%  self size {100 * r['size']:3.0f}%  stays {100 * r['stays']:3.0f}%")
            del B
        del A
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    if D.empty:
        return
    xs = [10, 20, 30, 40, 47]
    say("\n  relative gap read off at the SAME movement ('-' = outside that series' range):")
    say("  " + f"{'series':<34}" + "".join(f"{'moved ' + str(x):>11}" for x in xs))
    for s, q in D.groupby("series"):
        q = q.sort_values("task")
        say("  " + f"{s:<34}" + "".join(f"{np.interp(x, q.task, q.rel_gap):10.1f}%" if q.task.min() - 2 <= x <= q.task.max() + 2 else f"{'-':>11}" for x in xs))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    sty = {"continual, SAME language": ("#C0392B", "^", "-"), "continual, DIFFERENT language": ("#E07A1F", "x", "--"),
           "Hindi only": ("#2A6F97", "s", "-"), "Hindi + push": ("#2ca25f", "o", "-"), "mixed (all 3 at once)": ("#4B1D6B", "D", "-")}
    for s, q in D.groupby("series"):
        q = q.sort_values("task"); c, mk, ls = sty.get(s, ("k", "o", "-"))
        ax.plot(q.task, q.rel_gap, marker=mk, color=c, ls=ls, label=s, ms=7)
    ax.set_xlabel("how much the other neurons changed (change score, 0 to 100)"); ax.set_ylabel("how much LESS the self changed (%)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "pairs.png"), dpi=150)

# ================================================================== 2. chance level on many checkpoints
def part_noise():
    say("\n=== CHANCE LEVEL: random noise (no training) on many checkpoints ===")
    rows = []
    for label, key, ph in NOISE_CK:
        run = run_of(key)
        if not exists(run, ph):
            say("  skip", label, ph); continue
        m, c = load_model(ck(run, ph)); A = stats(m)
        base = {k: v.detach().clone() for k, v in m.state_dict().items()}
        for scale in (0.3, 0.45, 0.6):
            m.load_state_dict(base)
            g = torch.Generator(device=DEVICE).manual_seed(0)
            with torch.no_grad():
                for name, p in m.named_parameters():
                    if p.dim() >= 2:
                        p.add_(torch.randn(p.shape, generator=g, device=DEVICE) * scale * p.std())
            d = compare(A, stats(m))
            r = {"model": label, "run": run, "phase": ph, "scale": scale, "task": d.task.mean(), "self": d.self.mean(),
                 "rel_gap": 100 * (d.task.mean() - d.self.mean()) / d.task.mean(), "size_after": d["size"].mean()}
            rows.append(r)
            say(f"  {label:<10} phase {ph:>2} noise {scale}: other neurons {r['task']:5.1f}  rel gap {r['rel_gap']:5.1f}%  self size after {100 * r['size_after']:3.0f}%")
        del m, A, base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "noise.csv"), index=False)
    if D.empty:
        return
    say("\n  chance level at movement 47 (interpolated), per checkpoint:")
    for (lab, ph), q in D.groupby(["model", "phase"]):
        q = q.sort_values("task"); say(f"  {lab:<10} phase {ph:>2}: {np.interp(47, q.task, q.rel_gap):5.1f}%")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    col = {"continual": "#C0392B", "hindi": "#2A6F97", "mixed": "#4B1D6B", "push": "#2ca25f", "redo": "#E07A1F"}
    for (lab, ph), q in D.groupby(["model", "phase"]):
        q = q.sort_values("task"); ax.plot(q.task, q.rel_gap, "--o", color=col.get(lab, "k"), alpha=.4 + .15 * (ph >= 45), label=f"{lab} p{ph}")
    ax.set_xlabel("how much the other neurons changed (change score)"); ax.set_ylabel("how much LESS the self changed (%)")
    ax.set_title("random noise, no training: the chance level of every checkpoint"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "noise.png"), dpi=150)

# ================================================================== 3. does the same self come back?
def part_selfreturn():
    say("\n=== DOES THE SAME SELF COME BACK? share of phase-30 self neurons that are in the self again (same neuron numbers) ===")
    rows = []
    for name, (key, phs) in RETURN.items():
        run = run_of(key); phs = [p for p in phs if exists(run, p)]
        if len(phs) < 2:
            continue
        masks, langs = {}, {}
        for p in phs:
            m, c = load_model(ck(run, p)); S = stats(m); del m
            masks[p] = {pt: S[pt][3] for pt in MLP}; langs[p] = c["lang"]; del S
        a = phs[0]
        for p in phs[1:]:
            for side, pts in [("encoder", MLP[:4]), ("decoder", MLP[4:])]:
                kept = np.mean([(masks[a][pt] & masks[p][pt]).sum() / max(masks[a][pt].sum(), 1) for pt in pts])
                rows.append({"model": name, "from": a, "to": p, "lang_from": langs[a], "lang_to": langs[p], "side": side, "kept": kept,
                             "same_lang": langs[a] == langs[p]})
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "selfreturn.csv"), index=False)
    if D.empty:
        return
    for (name, side), q in D.groupby(["model", "side"]):
        s, d_ = q[q.same_lang].kept.mean(), q[~q.same_lang].kept.mean()
        say(f"  {name:<10} {side}: phase-30 self neurons still in the self, later phase in the SAME language {100 * s:4.0f}% | DIFFERENT language {100 * d_:4.0f}%")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, side in zip(axs, ["encoder", "decoder"]):
        for name, c in [("continual", "#C0392B"), ("hindi", "#2A6F97"), ("mixed", "#4B1D6B")]:
            q = D[(D.model == name) & (D.side == side)].sort_values("to")
            if len(q):
                ax.plot(q.to, 100 * q.kept, "-", color=c, alpha=.5, label=name)
                qs = q[q.same_lang]; ax.scatter(qs.to, 100 * qs.kept, color=c, s=30, zorder=3)
        ax.set_title(f"{side}: share of the phase-30 self still in the self (dots = same language as phase 30)", fontsize=10)
        ax.set_xlabel("phase"); ax.set_ylabel("%"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "selfreturn.png"), dpi=150)

# ================================================================== 4. crosscoder: switches, return, self vs task features
def crosscoder(XA, XB, steps, m, target_l0):
    if XA.shape[0] > 150_000:
        keep = torch.randperm(XA.shape[0], generator=torch.Generator().manual_seed(0))[:150_000]; XA, XB = XA[keep], XB[keep]
    XA, XB = XA.float().to(DEVICE), XB.float().to(DEVICE)
    XA = XA / XA.norm(dim=1).mean(); XB = XB / XB.norm(dim=1).mean()         # both models on the same scale
    n, d = XA.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    def rand_dirs():
        W = torch.randn(m, d, generator=g); return (W / W.norm(dim=1, keepdim=True) * 0.1).to(DEVICE)
    dA, dB = rand_dirs(), rand_dirs()                                        # INDEPENDENT starts
    W_enc = torch.cat([dA.T, dB.T], 0).clone()
    b_enc = torch.zeros(m, device=DEVICE); bA, bB = XA.mean(0).clone(), XB.mean(0).clone()
    P_ = [W_enc, b_enc, dA, dB, bA, bB]
    for p_ in P_:
        p_.requires_grad_(True)
    opt = torch.optim.Adam(P_, lr=2e-4); lam = 0.01; bs = min(2048, n); l0_hist = []
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = 2e-4 * min(1.0, (steps - step) / (0.2 * steps))         # decay over the last 20%
        i = torch.randint(0, n, (bs,), device=DEVICE); xa, xb = XA[i], XB[i]
        f = torch.relu(torch.cat([xa - bA, xb - bB], 1) @ W_enc + b_enc)
        mse = ((f @ dA + bA - xa) ** 2).sum(1).mean() + ((f @ dB + bB - xb) ** 2).sum(1).mean()
        norms = dA.norm(dim=1) + dB.norm(dim=1)
        loss = mse + lam * (f * norms).sum(1).mean()                         # sparsity weighted by BOTH decoder norms
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P_, 1.0); opt.step()
        if step % 50 == 0:
            l0 = float((f > 0).sum(1).float().mean()); l0_hist.append(l0)
            lam *= 1.15 if l0 > 1.2 * target_l0 else (1 / 1.15 if l0 < target_l0 / 1.2 else 1.0)
    with torch.no_grad():
        cnt = torch.zeros(m, device=DEVICE); tot = torch.zeros(m, device=DEVICE); ea = eb = va = vb = 0.0; l0s = []
        for s in range(0, n, 4096):
            xa, xb = XA[s:s + 4096], XB[s:s + 4096]
            f = torch.relu(torch.cat([xa - bA, xb - bB], 1) @ W_enc + b_enc)
            cnt += (f > 0).sum(0); tot += f.sum(0); l0s.append(float((f > 0).sum(1).float().mean()))
            ea += float(((f @ dA + bA - xa) ** 2).sum()); va += float(((xa - XA.mean(0)) ** 2).sum())
            eb += float(((f @ dB + bB - xb) ** 2).sum()); vb += float(((xb - XB.mean(0)) ** 2).sum())
    return dict(dA=dA.detach().cpu(), dB=dB.detach().cpu(), freq=(cnt / n).cpu(), mean_act=(tot / n).cpu(),
                fvu_A=ea / va, fvu_B=eb / vb, l0=float(np.mean(l0s)), lam=lam)


def part_crosscoder():
    say("\n=== CROSSCODER: which decoder features a switch replaces, do they come back, are self features kept more? ===")
    C_, H_, M_ = args.cont, args.hi, args.mixed
    jobs = [("continual 44 (zh) vs 45 (ar): switch", C_, 44, C_, 45),
            ("continual 44 (zh) vs 46 (hi): two switches", C_, 44, C_, 46),
            ("continual 44 (zh) vs 47 (zh): language back", C_, 44, C_, 47),
            ("Hindi only 44 vs 45", H_, 44, H_, 45),
            ("mixed 44 vs 45", M_, 44, M_, 45)]
    pt = "dec2.mlp"; rows = []; cache = {}
    def acts_of(run, p):
        if (run, p) not in cache:
            mm, _ = load_model(ck(run, p)); A, dtok, dlang = record(mm, [pt]); S = stats(mm); del mm
            cache[(run, p)] = (A[pt], S[pt][3], S[pt][2].numpy())
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return cache[(run, p)]
    fig, axs = plt.subplots(1, len(jobs), figsize=(4 * len(jobs), 3.6))
    for j, (name, ra, pa, rb, pb) in enumerate(jobs):
        if not (exists(ra, pa) and exists(rb, pb)):
            say("  skip", name); continue
        XA, sa, ala = acts_of(ra, pa); XB, sb, alb = acts_of(rb, pb)
        t0 = time.time(); cc = crosscoder(XA, XB, args.cc_steps, args.cc_dict, args.cc_l0)
        live = cc["freq"] > 1e-5
        nA, nB = cc["dA"].norm(dim=1), cc["dB"].norm(dim=1); r = (nB / (nA + nB + 1e-12))
        shared, a_only, b_only = (r > 0.3) & (r < 0.7) & live, (r <= 0.1) & live, (r >= 0.9) & live
        # a feature is a "self feature" if most of its write-weight (in A) lands on self neurons
        W = cc["dA"].abs(); self_share = (W[:, torch.tensor(sa)].sum(1) / (W.sum(1) + 1e-12))
        frac_self = float(sa.sum() / max(ala.sum(), 1))
        is_self_f = live & (self_share > 0.5); is_task_f = live & (self_share <= 0.5)
        kept_self = float((shared & is_self_f).sum() / max(is_self_f.sum(), 1)); kept_task = float((shared & is_task_f).sum() / max(is_task_f.sum(), 1))
        row = {"pair": name, "live": float(live.float().mean()), "l0": cc["l0"], "fvu_A": cc["fvu_A"], "fvu_B": cc["fvu_B"],
               "kept": float(shared.sum() / live.sum()), "lost": float(a_only.sum() / live.sum()), "new": float(b_only.sum() / live.sum()),
               "self_features": int(is_self_f.sum()), "task_features": int(is_task_f.sum()), "self_layer_share": frac_self,
               "kept_self_features": kept_self, "kept_task_features": kept_task}
        rows.append(row)
        say(f"  {name:<46} kept {100 * row['kept']:3.0f}%  lost {100 * row['lost']:3.0f}%  new {100 * row['new']:3.0f}% | "
            f"self features kept {100 * kept_self:3.0f}% vs task features kept {100 * kept_task:3.0f}% "
            f"({row['self_features']} vs {row['task_features']} features) | unexplained {row['fvu_A']:.2f}/{row['fvu_B']:.2f} [{(time.time() - t0) / 60:.1f} min]")
        axs[j].hist(r[live].numpy(), bins=40, color="#2c7fb8"); axs[j].set_title(name, fontsize=8); axs[j].set_xlabel("0 = only before, 1 = only after", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "crosscoder.png"), dpi=140)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "crosscoder.csv"), index=False)
    say("  (earlier controls: same model twice -> 100% kept; Arabic-knowing vs not, on Arabic -> 41% only in the Arabic model)")

# ================================================================== 5. force one neuron on: 100 neurons per group, every language
def part_steer():
    say(f"\n=== FORCE ONE NEURON ON: {args.n_steer} self vs {args.n_steer} task neurons, decoder layer 2, 10x its peak, every language it knows ===")
    rows = []; pt = "dec2.mlp"
    for name, key in [("continual", "cont"), ("hindi", "hi"), ("mixed", "mixed")]:
        run = run_of(key)
        if not exists(run, args.phase):
            continue
        model, c = load_model(ck(run, args.phase)); model.eval()
        S = stats(model); _, _, alive, sm = S[pt]; alive = alive.numpy(); del S
        langs = ["hi"] if key == "hi" else LANGS
        A, dtok, dlang = record(model, [pt], langs=langs)
        q999 = torch.quantile(A[pt][:200_000].float(), 0.999, dim=0).numpy()
        pools = {"self": np.nonzero(sm & (q999 > 0))[0], "task": np.nonzero(alive & ~sm & (q999 > 0))[0]}
        groups = {k: rng.choice(v, min(args.n_steer, len(v)), replace=False) for k, v in pools.items()}
        for L in langs:
            allowed = torch.nonzero(ALLOWED[L]).squeeze(1)
            data = TGT[L][:(20 if args.small else 200)]; srcs = SRC[:len(data)]
            base = []
            with torch.no_grad():
                for s in range(0, len(data), 50):
                    S_ = pad_batch(srcs[s:s + 50]); T_ = pad_batch(data[s:s + 50]); mem, smk = model.encode(S_)
                    base.append((S_, T_, mem, smk, F.log_softmax(model.decode(T_, mem, smk).float()[..., allowed], -1)))
            for gname, ns in groups.items():
                for nid in ns:
                    val = float(q999[nid]) * 10; kl_sum, npos = 0.0, 0
                    h = mlp_mod(model, pt).act.register_forward_hook(
                        lambda m_, i_, o, n_=int(nid), v=val: o.index_fill(-1, torch.tensor([n_], device=o.device), v))
                    with torch.no_grad():
                        for S_, T_, mem, smk, lb in base:
                            lf = F.log_softmax(model.decode(T_, mem, smk).float()[..., allowed], -1); keep = T_ != PAD
                            kl_sum += float((lb.exp() * (lb - lf)).sum(-1)[keep].sum()); npos += int(keep.sum())
                    h.remove()
                    rows.append({"model": name, "lang": L, "group": gname, "neuron": int(nid), "kl": kl_sum / max(npos, 1)})
            del base
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "steer.csv"), index=False)
    if D.empty:
        return
    for (name, L), q in D.groupby(["model", "lang"]):
        s, t = q[q.group == "self"].kl, q[q.group == "task"].kl
        bs = [rng.choice(s, len(s)).mean() / rng.choice(t, len(t)).mean() for _ in range(2000)]
        say(f"  {name:<10} {L}: self {s.mean():.3f} (median {s.median():.3f}) vs task {t.mean():.3f} (median {t.median():.3f}) | "
            f"ratio {s.mean() / t.mean():.2f} [95%: {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}]  n = {len(s)} + {len(t)}")

# ================================================================== 6. damage (+ relearn), every language, three models
def part_relearn():
    say("\n=== DAMAGE + RELEARN: reset 20% of each layer (self / task / random); loss rise on EVERY language right after ===")
    VAL = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:(50 if args.small else 500)] for l in LANGS}
    rows = []
    for name, key in [("continual", "cont"), ("hindi", "hi"), ("mixed", "mixed")]:
        run = run_of(key)
        if not exists(run, args.phase):
            continue
        path = ck(run, args.phase); model, c = load_model(path); L = c["lang"]
        S = stats(model); del model
        tl = LANGS if L == "mix" else [L]
        train = {}
        for l in tl:
            rr = []
            with open(P(f"data/en-{l}/train.jsonl"), encoding="utf-8") as f:
                for line in f:
                    rr.append(json.loads(line))
                    if len(rr) >= (2000 if args.small else 40000 // len(tl)):
                        break
            tmp = os.path.join(OUT, "_tmp.jsonl")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in rr:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            train[l] = load_pairs(tmp, l); os.remove(tmp)
        evl = ["hi"] if key == "hi" else LANGS
        base, _ = load_model(path); intact = {l: val_loss(base, VAL[l], l, masked=True) for l in evl}; del base
        for gname in ["self", "task", "random"]:
            for seed in [0, 1, 2]:
                model, _ = load_model(path); g = torch.Generator().manual_seed(seed)
                with torch.no_grad():
                    for pt in MLP:
                        _, _, alive, sm = S[pt]; alive = alive.numpy(); k = int(args.lesion_frac * alive.sum())
                        pool = {"self": np.nonzero(sm)[0], "task": np.nonzero(alive & ~sm)[0], "random": np.nonzero(alive)[0]}[gname]
                        idx = torch.tensor(np.random.default_rng(seed).choice(pool, min(k, len(pool)), replace=False))
                        mm = mlp_mod(model, pt)
                        mm.fc.weight[idx] = (torch.randn(len(idx), mm.fc.weight.shape[1], generator=g) * 0.02).to(DEVICE); mm.fc.bias[idx] = 0
                        mm.proj.weight[:, idx] = (torch.randn(mm.proj.weight.shape[0], len(idx), generator=g) * 0.02 / 4).to(DEVICE)
                for l in evl:
                    rows.append({"model": name, "trained": L, "eval_lang": l, "group": gname, "seed": seed, "step": 0,
                                 "rise": val_loss(model, VAL[l], l, masked=True) - intact[l]})
                opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
                bl = {l: make_batches(train[l], 8000, seed=seed) for l in tl}
                for step in range(1, args.relearn_steps + 1):
                    for gp in opt.param_groups:
                        gp["lr"] = 3e-5 * min(1.0, step / 50)
                    l = tl[step % len(tl)]
                    model.train(); s_, ti, to = collate(train[l], bl[l][step % len(bl[l])])
                    with amp():
                        logits, _ = model(s_, ti)
                    loss = masked_ce(logits.view(-1, logits.size(-1)), to.view(-1), ALLOWED[l])
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    if step in (50, 100, 200, 400, args.relearn_steps):
                        for le in tl:
                            rows.append({"model": name, "trained": L, "eval_lang": le, "group": gname, "seed": seed, "step": step,
                                         "rise": val_loss(model, VAL[le], le, masked=True) - intact.get(le, 0)})
                del model, opt
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    D = pd.DataFrame(rows).drop_duplicates(["model", "eval_lang", "group", "seed", "step"]); D.to_csv(os.path.join(OUT, "relearn.csv"), index=False)
    say("  loss rise RIGHT AFTER the reset (mean of 3 repeats), per language:")
    for (name, l), q in D[D.step == 0].groupby(["model", "eval_lang"]):
        m = q.groupby("group").rise.agg(["mean", "std"])
        say(f"  {name:<10} on {l}: self {m.loc['self', 'mean']:.3f} (±{m.loc['self', 'std']:.3f}) | task {m.loc['task', 'mean']:.3f} | random {m.loc['random', 'mean']:.3f}"
            f" | self/task {m.loc['self', 'mean'] / max(m.loc['task', 'mean'], 1e-6):.2f}")
    last = D.step.max()
    say(f"  after {last} relearning steps:")
    for (name, l), q in D[D.step == last].groupby(["model", "eval_lang"]):
        m = q.groupby("group").rise.mean(); say(f"  {name:<10} on {l}: self {m['self']:.3f} | task {m['task']:.3f} | random {m['random']:.3f}")

say(f"analysis_v6c | cont {args.cont} | hi {args.hi} | mixed {args.mixed} | redo {args.redo} | push {args.push} | phase {args.phase} | tau {args.tau} | parts {args.parts}")
for part in args.parts:
    t0 = time.time(); globals()[f"part_{part}"](); say(f"  [{part} took {(time.time() - t0) / 60:.1f} min]")
say("finished")
