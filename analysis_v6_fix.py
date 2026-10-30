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
ap.add_argument("--parts", nargs="+", default=["q2pairs", "crosscoder", "steer", "relearn"])
ap.add_argument("--cont", default="v4b_continual_s0")
ap.add_argument("--hi", default="v4b_only_hi_s0")
ap.add_argument("--push", default="v6_hi_push0.6_from30")
ap.add_argument("--phase", type=int, default=45)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--cc_steps", type=int, default=15000)
ap.add_argument("--cc_dict", type=int, default=8192)
ap.add_argument("--cc_l0", type=float, default=40)
ap.add_argument("--relearn_steps", type=int, default=400)
ap.add_argument("--lesion_frac", type=float, default=0.2)
ap.add_argument("--tag", default="ref")
ap.add_argument("--list", action="store_true", help="print the checkpoints needed and exit")
ap.add_argument("--small", action="store_true")
args = ap.parse_args()

PAIRS = {  # name: (run arg, anchor phase, target phases)
    "continual": ("cont", 30, [31, 32, 33, 39, 40, 44, 45, 52, 60]),
    "hindi": ("hi", 30, [31, 35, 40, 45, 50, 58]),
    "push": ("push", 30, [31, 33, 35, 40, 45, 50, 58]),
}
def needed():
    need = {}
    for name, (key, a, ts) in PAIRS.items():
        need.setdefault(getattr(args, key), set()).update([a] + ts)
    need.setdefault(args.cont, set()).update([args.phase - 1, args.phase])
    need.setdefault(args.hi, set()).update([args.phase - 1, args.phase])
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

OUT = P("analysis", "v6", f"fix_{args.tag}"); os.makedirs(OUT, exist_ok=True)
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

# ================================================================== Q2 at matched movement
def part_q2pairs():
    say("\n=== Q2 at MATCHED movement, from existing checkpoints (tau", args.tau, ", MLP mean) ===")
    rows = []
    for name, (key, a, ts) in PAIRS.items():
        run = getattr(args, key)
        try:
            m, ca = load_model(ck(run, a))
        except FileNotFoundError as e:
            say("  skip", name, e); continue
        A = stats(m); del m
        for t in ts:
            try:
                m, cb = load_model(ck(run, t))
            except FileNotFoundError as e:
                say("  skip", e); continue
            B = stats(m); del m
            d = compare(A, B)
            kind = {"hindi": "Hindi only, far apart", "push": "Hindi + push, far apart"}.get(name)
            if name == "continual":
                kind = "continual, SAME language" if ca["lang"] == cb["lang"] else "continual, DIFFERENT language"
            r = {"series": kind, "from": a, "to": t, "langs": f"{ca['lang']}->{cb['lang']}", "task": d.task.mean(),
                 "self": d.self.mean(), "rel_gap": 100 * (d.task.mean() - d.self.mean()) / d.task.mean(),
                 "rel_gap_enc": 100 * (1 - d.self[:4].mean() / d.task[:4].mean()),
                 "rel_gap_dec": 100 * (1 - d.self[4:].mean() / d.task[4:].mean()),
                 "size": d["size"].mean(), "stays": d.stays.mean()}
            rows.append(r)
            say(f"  {kind:<32} p{a}->p{t} ({r['langs']}): task {r['task']:5.1f}  self {r['self']:5.1f}  rel gap {r['rel_gap']:5.1f}%"
                f"  (enc {r['rel_gap_enc']:4.1f}%, dec {r['rel_gap_dec']:4.1f}%)  stays {100 * r['stays']:3.0f}%")
            del B
        del A
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "q2pairs.csv"), index=False)
    if D.empty:
        return
    # matched comparison: interpolate every series at the same movement
    say("\n  relative gap read off at the SAME movement (linear interpolation; '-' = outside that series' range):")
    xs = [20, 30, 40, 47]
    say("  " + f"{'series':<34}" + "".join(f"{'task=' + str(x):>10}" for x in xs))
    for s, q in D.groupby("series"):
        q = q.sort_values("task")
        vals = [f"{np.interp(x, q.task, q.rel_gap):9.1f}%" if q.task.min() - 2 <= x <= q.task.max() + 2 else f"{'-':>10}" for x in xs]
        say("  " + f"{s:<34}" + "".join(vals))
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    sty = {"continual, SAME language": ("#c0392b", "^", "-"), "continual, DIFFERENT language": ("#c0392b", "x", "--"),
           "Hindi only, far apart": ("#2c7fb8", "s", "-"), "Hindi + push, far apart": ("#2ca25f", "o", "-")}
    for s, q in D.groupby("series"):
        q = q.sort_values("task"); c, mk, ls = sty.get(s, ("k", "o", "-"))
        ax.plot(q.task, q.rel_gap, marker=mk, color=c, ls=ls, label=s, ms=7)
        for r in q.itertuples():
            ax.annotate(f"{r._2}-{r.to}", (r.task, r.rel_gap), fontsize=6, xytext=(3, 3), textcoords="offset points")
    sw = P("analysis", "v6", "switches.csv")
    if os.path.exists(sw):
        Sw = pd.read_csv(sw); Sw = Sw[(Sw.run == "continual (ref)") & ((Sw["mode"] == "fixed") if "mode" in Sw else True) & Sw.to_phase.between(2, 60)]
        g = Sw.groupby("to_phase").agg(task=("task_change", "mean"), gap=("gap", "mean"))
        ax.scatter(g.task.mean(), 100 * g.gap.mean() / g.task.mean(), marker="*", s=250, color="#c0392b", edgecolor="k",
                   zorder=5, label="continual, one switch (mean of 59)")
    qc = P("analysis", "v6", "quick_checks", "results.csv")
    if os.path.exists(qc):
        Q = pd.read_csv(qc)
        for pref, c, lab in [("continual", "#c0392b", "pure noise, continual net"), ("only", "#2c7fb8", "pure noise, Hindi net")]:
            q = Q[(Q.check == "noise") & Q.label.str.startswith(pref)].sort_values("task_change")
            ax.plot(q.task_change, 100 * q.gap / q.task_change, color=c, ls=":", lw=1, marker="D", mfc="none", ms=4, label=lab)
    ax.set_xlabel("task-neuron change between the two checkpoints (how much the network moved)")
    ax.set_ylabel("relative gap = (task - self) / task  [%]")
    ax.set_title("Q2 at matched movement: does ending in a DIFFERENT language hit the self harder?", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, "q2pairs.png"), dpi=150)

# ================================================================== crosscoder (fixed)
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
    say("\n=== CROSSCODER (fixed) ===")
    ph = args.phase
    jobs = [("CONTROL same model twice (expect ~all shared)", args.hi, ph, args.hi, ph, "dec2.mlp", LANGS, None),
            ("CONTROL knows Arabic vs not, Arabic text (expect exclusive)", args.cont, ph, args.hi, ph, "dec2.mlp", ["ar"], None),
            ("continual switch 44->45", args.cont, ph - 1, args.cont, ph, "dec2.mlp", LANGS, None),
            ("Hindi-only step 44->45", args.hi, ph - 1, args.hi, ph, "dec2.mlp", LANGS, None),
            ("continual vs Hindi-only, Hindi text", args.cont, ph, args.hi, ph, "dec2.mlp", ["hi"], None),
            ("continual switch 44->45, encoder", args.cont, ph - 1, args.cont, ph, "enc1.mlp", LANGS, None)]
    rows = []; cache = {}
    def acts_of(run, p, pt, langs):
        key = (run, p, pt, tuple(langs))
        if key not in cache:
            mm, _ = load_model(ck(run, p))
            A, dtok, dlang = record(mm, [pt], langs=langs)
            sm = None
            # self mask on the same activations (decoder: each language's mean removed)
            X = A[pt][:args.max_t].float().to(DEVICE) if A[pt].shape[0] <= args.max_t else A[pt][torch.randperm(A[pt].shape[0])[:args.max_t]].float().to(DEVICE)
            sd = X.std(0); alive = (sd > 1e-6).cpu().numpy()
            Z = (X - X.mean(0)) / torch.where(sd > 1e-6, sd, torch.ones_like(sd)); R = (Z.T @ Z / Z.shape[0]).abs()
            R.fill_diagonal_(0); Rm = (R >= args.tau).cpu().numpy() & alive[:, None] & alive[None, :]
            _, lab = connected_components(csr_matrix(Rm), directed=False); lab = np.where(alive, lab, -1)
            sm = lab == np.bincount(lab[lab >= 0]).argmax()
            cache[key] = (A[pt], sm, alive); del mm, X, Z, R
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        return cache[key]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5)); axs = axs.ravel()
    for j, (name, ra, pa, rb, pb, pt, langs, _) in enumerate(jobs):
        XA, sa, ala = acts_of(ra, pa, pt, langs); XB, sb, alb = acts_of(rb, pb, pt, langs)
        t0 = time.time(); cc = crosscoder(XA, XB, args.cc_steps, args.cc_dict, args.cc_l0)
        live = cc["freq"] > 1e-5
        nA, nB = cc["dA"].norm(dim=1), cc["dB"].norm(dim=1)
        r = (nB / (nA + nB + 1e-12))[live].numpy()
        shared, a_only, b_only = (r > 0.3) & (r < 0.7), r <= 0.1, r >= 0.9
        wA = cc["dA"][live].abs() * cc["mean_act"][live, None]; wB = cc["dB"][live].abs() * cc["mean_act"][live, None]
        sh = torch.tensor(shared)
        shA = (wA[sh].sum(0) / (wA.sum(0) + 1e-12)).numpy(); shB = (wB[sh].sum(0) / (wB.sum(0) + 1e-12)).numpy()
        row = {"pair": name, "layer": pt, "live_frac": float(live.float().mean()), "l0": cc["l0"], "fvu_A": cc["fvu_A"], "fvu_B": cc["fvu_B"],
               "shared": shared.mean(), "A_only": a_only.mean(), "B_only": b_only.mean(),
               "shared_in_self_A": shA[sa].mean(), "shared_in_task_A": shA[ala & ~sa].mean(),
               "shared_in_self_B": shB[sb].mean(), "shared_in_task_B": shB[alb & ~sb].mean()}
        rows.append(row)
        say(f"  {name:<58} {pt}: live {100 * row['live_frac']:3.0f}%  active/token {row['l0']:4.0f}  unexplained A {row['fvu_A']:.2f} B {row['fvu_B']:.2f}"
            f" | shared {100 * row['shared']:3.0f}%  only-A {100 * row['A_only']:3.0f}%  only-B {100 * row['B_only']:3.0f}%"
            f" | input from shared: self {row['shared_in_self_A']:.2f} vs task {row['shared_in_task_A']:.2f} (A), "
            f"self {row['shared_in_self_B']:.2f} vs task {row['shared_in_task_B']:.2f} (B)  [{(time.time() - t0) / 60:.1f} min]")
        ax = axs[j]; ax.hist(r, bins=40, color="#2c7fb8")
        ax.set_title(f"{name}\n{pt}: shared {100 * row['shared']:.0f}% | only A {100 * row['A_only']:.0f}% | only B {100 * row['B_only']:.0f}%", fontsize=7.5)
        ax.set_xlabel("relative decoder norm (0 = only in A, 1 = only in B)", fontsize=7)
    say("  read: the two CONTROLS must behave as expected (first ~all shared, second clearly exclusive); only then trust the rest.")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "crosscoder.png"), dpi=140)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "crosscoder.csv"), index=False)

# ================================================================== steer (fixed): change in predictions
def part_steer():
    say("\n=== FORCE A NEURON ON (fixed): how much do the next-token predictions change? ===")
    rows = []; E_cache = {}
    for name, run in [("continual", args.cont), ("hindi", args.hi)]:
        model, c = load_model(ck(run, args.phase)); L = c["lang"]; model.eval()
        A, dtok, dlang = record(model, ["dec2.mlp", "dec3.mlp"], langs=[L])
        E = F.normalize(model.tok_emb.weight.detach().float(), dim=1)
        allowed = torch.nonzero(ALLOWED[L]).squeeze(1)
        data = TGT[L][:(20 if args.small else 200)]; srcs = SRC[:len(data)]
        for pt in ["dec2.mlp", "dec3.mlp"]:
            X = A[pt].float()
            # self / task at tau on this language's tokens
            Xs = X[torch.randperm(X.shape[0])[:args.max_t]].to(DEVICE); sd = Xs.std(0); alive = (sd > 1e-6).cpu().numpy()
            Z = (Xs - Xs.mean(0)) / torch.where(sd > 1e-6, sd, torch.ones_like(sd)); R = (Z.T @ Z / Z.shape[0]).abs(); R.fill_diagonal_(0)
            _, lab = connected_components(csr_matrix((R >= args.tau).cpu().numpy() & alive[:, None] & alive[None, :]), directed=False)
            lab = np.where(alive, lab, -1); sm = lab == np.bincount(lab[lab >= 0]).argmax(); del Xs, Z, R
            q999 = torch.quantile(X[:200_000], 0.999, dim=0)
            # each neuron's job = its top input tokens (mean activation per token type)
            toks = torch.as_tensor(dtok); uniq, inv, cnt = torch.unique(toks, return_inverse=True, return_counts=True)
            Ms = torch.zeros(len(uniq), X.shape[1]).index_add_(0, inv, X); Ms = Ms / cnt[:, None]; okt = cnt >= 5
            pools = {"self": np.nonzero(sm & (q999.numpy() > 0))[0], "task": np.nonzero(alive & ~sm & (q999.numpy() > 0))[0]}
            groups = {k: rng.choice(v, min(8, len(v)), replace=False) for k, v in pools.items()}
            for gname, ns in groups.items():
                for nid in ns:
                    top_in = uniq[okt][torch.argsort(-Ms[okt, nid])[:10]].to(DEVICE)
                    for mult in [3, 10]:
                        val = float(q999[nid]) * mult; kl_sum = 0.0; npos = 0; dsum = torch.zeros(len(allowed), device=DEVICE)
                        for s in range(0, len(data), 50):
                            S_ = pad_batch(srcs[s:s + 50]); T_ = pad_batch(data[s:s + 50]); keep = T_ != PAD
                            with torch.no_grad():
                                mem, smk = model.encode(S_)
                                lb = F.log_softmax(model.decode(T_, mem, smk).float()[..., allowed], -1)
                                h = mlp_mod(model, pt).act.register_forward_hook(
                                    lambda m_, i_, o, n_=int(nid), v=val: o.index_fill(-1, torch.tensor([n_], device=o.device), v))
                                lf = F.log_softmax(model.decode(T_, mem, smk).float()[..., allowed], -1); h.remove()
                            kl = (lb.exp() * (lb - lf)).sum(-1)
                            kl_sum += float(kl[keep].sum()); npos += int(keep.sum()); dsum += (lf - lb)[keep].sum(0)
                        gained = allowed[torch.argsort(-dsum)[:10]]
                        rel_g = float((E[gained] @ E[top_in].T).mean()); rnd = allowed[torch.randint(0, len(allowed), (200,), device=DEVICE)]
                        rel_r = float((E[rnd] @ E[top_in].T).mean())
                        rows.append({"run": name, "layer": pt, "group": gname, "neuron": int(nid), "mult": mult, "kl": kl_sum / max(npos, 1),
                                     "rel_gained_to_job": rel_g, "rel_random_to_job": rel_r,
                                     "job_tokens": " ".join(piece(t) for t in top_in[:6].tolist()),
                                     "gained_tokens": " ".join(piece(t) for t in gained[:6].tolist())})
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "steer.csv"), index=False)
    g = D.groupby(["run", "layer", "mult", "group"]).agg(kl_mean=("kl", "mean"), kl_median=("kl", "median"),
                                                         gained_related=("rel_gained_to_job", "mean"), random_related=("rel_random_to_job", "mean"))
    say(g.round(4).to_string())
    say("  kl = how much forcing ONE neuron changes the next-token predictions (bigger = more causal power).")
    say("  gained_related > random_related: the tokens it pushes up resemble the tokens it fires on (its job is causal).")
    for r in D[(D.mult == 10)].sort_values("kl", ascending=False).head(6).itertuples():
        say(f"  e.g. {r.run} {r.layer} {r.group} #{r.neuron}: kl {r.kl:.3f} | fires on: {r.job_tokens} | pushes up: {r.gained_tokens}")

# ================================================================== relearn (fixed learning rate)
def part_relearn():
    say("\n=== LESION + RELEARN (fixed: warm-up to lr 3e-5) ===")
    VAL = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:(50 if args.small else 500)] for l in LANGS}
    rows = []
    for name, run in [("continual", args.cont), ("hindi", args.hi)]:
        path = ck(run, args.phase); model, c = load_model(path); L = c["lang"]
        S = stats(model); del model
        rows_tr = []
        with open(P(f"data/en-{L}/train.jsonl"), encoding="utf-8") as f:
            for line in f:
                rows_tr.append(json.loads(line))
                if len(rows_tr) >= (2000 if args.small else 40000):
                    break
        tmp = os.path.join(OUT, "_tmp.jsonl")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows_tr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        train = load_pairs(tmp, L); os.remove(tmp)
        base, _ = load_model(path); intact = val_loss(base, VAL[L], L, masked=True); del base
        for gname in ["self", "task", "random"]:
            for seed in [0, 1]:
                model, _ = load_model(path); g = torch.Generator().manual_seed(seed)
                with torch.no_grad():
                    for pt in MLP:
                        _, _, alive, sm = S[pt]; alive = alive.numpy(); k = int(args.lesion_frac * alive.sum())
                        pool = {"self": np.nonzero(sm)[0], "task": np.nonzero(alive & ~sm)[0], "random": np.nonzero(alive)[0]}[gname]
                        idx = torch.tensor(np.random.default_rng(seed).choice(pool, min(k, len(pool)), replace=False))
                        mm = mlp_mod(model, pt)
                        mm.fc.weight[idx] = (torch.randn(len(idx), mm.fc.weight.shape[1], generator=g) * 0.02).to(DEVICE); mm.fc.bias[idx] = 0
                        mm.proj.weight[:, idx] = (torch.randn(mm.proj.weight.shape[0], len(idx), generator=g) * 0.02 / 4).to(DEVICE)
                opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
                batches = make_batches(train, 8000, seed=seed)
                for step in range(args.relearn_steps + 1):
                    if step in (0, 25, 50, 100, 200, 400, args.relearn_steps):
                        rows.append({"run": name, "lang": L, "group": gname, "seed": seed, "step": step,
                                     "rise": val_loss(model, VAL[L], L, masked=True) - intact})
                    if step == args.relearn_steps:
                        break
                    for gp in opt.param_groups:
                        gp["lr"] = 3e-5 * min(1.0, (step + 1) / 50)
                    model.train(); s_, ti, to = collate(train, batches[step % len(batches)])
                    with amp():
                        logits, _ = model(s_, ti)
                    loss = masked_ce(logits.view(-1, logits.size(-1)), to.view(-1), ALLOWED[L])
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                del model, opt
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    D = pd.DataFrame(rows).drop_duplicates(["run", "group", "seed", "step"]); D.to_csv(os.path.join(OUT, "relearn.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, q) in zip(axs, D.groupby("run")):
        for gname, c in [("self", "#4B1D6B"), ("task", "#E07A1F"), ("random", "0.5")]:
            m = q[q.group == gname].groupby("step").rise.mean()
            ax.plot(m.index, m.values, marker="o", color=c, label=f"reset {args.lesion_frac:.0%} {gname}")
            say(f"  {name:<10} reset {gname:<6}: rise right after {m.iloc[0]:.3f} | after 50 {m.get(50, np.nan):.3f} | after {m.index[-1]} {m.iloc[-1]:.3f}")
        ax.set_title(f"{name}: {q.lang.iloc[0]} val loss above the intact model"); ax.set_xlabel("relearning steps"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "relearn.png"), dpi=140)

say(f"analysis_v6_fix | cont {args.cont} | hi {args.hi} | push {args.push} | phase {args.phase} | tau {args.tau} | parts {args.parts}")
for part in args.parts:
    t0 = time.time(); globals()[f"part_{part}"](); say(f"  [{part} took {(time.time() - t0) / 60:.1f} min]")
say("finished")
