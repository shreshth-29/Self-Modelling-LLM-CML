"""analysis_v8.py: tomorrow's two analyses. Run one process per language set (CML_LANGSET must be set BEFORE start).

  --parts related   RELATED vs DISTANT languages, inside ONE continual network, same phase numbers for both runs:
                    pairs (anchor phase -> every later phase): change of the self vs the other neurons, SAME vs DIFFERENT
                    language; chance level (random noise at the anchor); does the same self come back; self size.
                    -> analysis/v8/related_<tag>/
  --parts lesion    HARDCORE LESION. At each checkpoint and each of the 8 MLP layers, switch off (replace by its average)
                    a set of self neurons and compare with
                      MATCHED task neurons: same average activity, same firing rate, same outgoing weight size
                      RANDOM task neurons
                    doses 5 / 10 / 20 % of the layer, 3 repeats each; mean AND zero ablation; loss on every language the
                    model knows; then all 8 layers at once with chrF (real translations). -> analysis/v8/lesion_<tag>/
  --parts compare   CPU only: reads related_indic + related_distant and writes the side-by-side summary.

Usage:  CML_LANGSET=indic   python -u analysis_v8.py --gpu 1 --parts related --run v8_indic_continual_s0 --tag indic
        CML_LANGSET=distant python -u analysis_v8.py --gpu 1 --parts related --run v4b_continual_s0 --tag distant
        CML_LANGSET=distant python -u analysis_v8.py --gpu 1 --parts lesion --models "continual=v4b_continual_s0:10,30,45,60:all;..." --tag distant
"""
import os, sys, argparse, json, glob, time
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--parts", nargs="+", default=["related"])
ap.add_argument("--run", default="v8_indic_continual_s0")
ap.add_argument("--anchor", type=int, default=10)
ap.add_argument("--upto", type=int, default=30)
ap.add_argument("--models", default="", help="name=run:phases:langs;...  langs = all or e.g. hi")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--n_val", type=int, default=300)
ap.add_argument("--n_chrf", type=int, default=150)
ap.add_argument("--tag", default="indic")
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

def lesion_specs():
    out = []
    for item in [x for x in args.models.split(";") if x.strip()]:
        name, rest = item.split("="); run, phs, langs = rest.split(":")
        out.append((name, run, [int(p) for p in phs.split(",")], langs))
    return out
if args.list:
    need = {}
    if "related" in args.parts:
        need[args.run] = list(range(args.anchor, args.upto + 1))
    for name, run, phs, _ in lesion_specs():
        need.setdefault(run, []).extend(phs)
    print(json.dumps({k: sorted(set(v)) for k, v in need.items()})); sys.exit()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

if set(args.parts) <= {"compare", "pool"}:
    import torch                                           # CPU-only parts: no model, no tokenizer needed
    LANGSET = None
else:
    from common_v8 import *
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import linear_sum_assignment
    torch.backends.cuda.matmul.allow_tf32 = False
    MLP = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]
    DEV_EN = read_flores("dev", "en")
    SRC = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
    TGT = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}
rng = np.random.default_rng(0)
REPORT, OUT = [], None
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REPORT.append(s)
    if OUT:
        open(os.path.join(OUT, "report.txt"), "w").write("\n".join(REPORT))
def outdir(name):
    global OUT, REPORT
    OUT = os.path.join(REPO, "analysis", "v8", f"{name}_{args.tag}"); os.makedirs(OUT, exist_ok=True); REPORT = []
    return OUT

def ck(run, ph):
    f = glob.glob(os.path.join(REPO, "runs", run, "ckpt", f"p{ph:03d}_*.pt"))
    return f[0] if f else None

def mlp_mod(model, pt):
    return (model.enc_blocks if pt.startswith("enc") else model.dec_blocks)[int(pt[3])].mlp

@torch.no_grad()
def record(model, bs=64):
    """Post-GELU activations at every real token (CPU, fp16) + language of each decoder row (same as the deck)."""
    model.eval(); cur = {}
    hooks = [mlp_mod(model, pt).act.register_forward_hook(lambda mm, i, o, k=pt: cur.__setitem__(k, o)) for pt in MLP]
    acts = {pt: [] for pt in MLP}; dlang = []
    for s in range(0, len(SRC), bs):
        S_ = pad_batch(SRC[s:s + bs]); mem, sm = model.encode(S_)
        sk = (S_ != PAD) & (S_ != EOS)
        for pt in MLP[:4]:
            acts[pt].append(cur[pt][sk].half().cpu())
        for l in LANGS:
            T_ = pad_batch(TGT[l][s:s + bs]); model.decode(T_, mem, sm)
            tk = T_ != PAD; tk[:, 0] = False
            for pt in MLP[4:]:
                acts[pt].append(cur[pt][tk].half().cpu())
            dlang.append(torch.full((int(tk.sum()),), LANGS.index(l)))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(dlang).numpy()

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

def stats_from(acts, dlang):
    """Per MLP layer: z-scored traces, correlation, alive, self mask at tau (decoder: each language's mean removed)."""
    idx = fixed_idx(acts["enc0.mlp"].shape[0], dlang); out = {}
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

def stats(model):
    acts, dlang = record(model); S = stats_from(acts, dlang); del acts
    return S

def compare(a, b):
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

# ================================================================== RELATED vs DISTANT
def part_related():
    outdir("related")
    run, a = args.run, args.anchor
    targets = [p for p in range(a + 1, args.upto + 1) if ck(run, p)]
    say(f"=== {run} ({LANGSET}: {', '.join(LANGS)}) | anchor phase {a} | later phases {targets[0] if targets else '-'}..{targets[-1] if targets else '-'} | tau {args.tau} ===")
    if not ck(run, a) or not targets:
        say("  missing checkpoints"); return
    m, ca = load_model(ck(run, a)); A = stats(m)
    say(f"  anchor phase {a} was trained on {ca['lang']}")
    # chance level: random noise at the anchor, no training
    base = {k: v.detach().clone() for k, v in m.state_dict().items()}; noise = []
    for scale in (0.15, 0.3, 0.45, 0.6, 0.8, 1.0):
        m.load_state_dict(base); g = torch.Generator(device=DEVICE).manual_seed(0)
        with torch.no_grad():
            for n_, p in m.named_parameters():
                if p.dim() >= 2:
                    p.add_(torch.randn(p.shape, generator=g, device=DEVICE) * scale * p.std())
        d = compare(A, stats(m))
        noise.append({"scale": scale, "task": d.task.mean(), "self": d.self.mean(), "rel_gap": 100 * (d.task.mean() - d.self.mean()) / d.task.mean()})
        say(f"  chance (noise {scale}): other neurons change {noise[-1]['task']:5.1f} | self changes {noise[-1]['rel_gap']:5.1f}% less")
    del m, base
    N = pd.DataFrame(noise).sort_values("task"); N.to_csv(os.path.join(OUT, "noise.csv"), index=False)
    chance = lambda x: float(np.interp(x, N.task, N.rel_gap))
    rows = []
    for t in targets:
        m, cb = load_model(ck(run, t)); B = stats(m); del m
        d = compare(A, B)
        r = {"from": a, "to": t, "lang_from": ca["lang"], "lang_to": cb["lang"], "same_lang": ca["lang"] == cb["lang"],
             "task": d.task.mean(), "self": d.self.mean(), "rel_gap": 100 * (d.task.mean() - d.self.mean()) / d.task.mean(),
             "task_enc": d.task[:4].mean(), "task_dec": d.task[4:].mean(),
             "size": d["size"].mean(), "stays": d.stays.mean(),
             "kept_enc": np.mean([(A[pt][3] & B[pt][3]).sum() / max(A[pt][3].sum(), 1) for pt in MLP[:4]]),
             "kept_dec": np.mean([(A[pt][3] & B[pt][3]).sum() / max(A[pt][3].sum(), 1) for pt in MLP[4:]])}
        r["chance"] = chance(r["task"]); r["excess"] = r["rel_gap"] - r["chance"]
        rows.append(r); del B
        say(f"  p{a}->p{t} ({r['lang_from']}->{r['lang_to']}, {'SAME' if r['same_lang'] else 'DIFF'}): task change {r['task']:5.1f} | "
            f"self changes {r['rel_gap']:5.1f}% less (chance {r['chance']:4.1f}%) | self = {100 * r['size']:3.0f}% of layer | "
            f"phase-{a} self neurons back in the self: enc {100 * r['kept_enc']:3.0f}% dec {100 * r['kept_dec']:3.0f}%")
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    say("\n  SUMMARY (" + LANGSET + ")")
    for same, q in D.groupby("same_lang"):
        say(f"  {'SAME' if same else 'DIFFERENT'} language ({len(q)} pairs): task change {q.task.mean():5.1f} | self changes {q.rel_gap.mean():5.1f}% less "
            f"| chance {q.chance.mean():4.1f}% | above chance {q.excess.mean():+5.1f} points | self back: enc {100 * q.kept_enc.mean():3.0f}% dec {100 * q.kept_dec.mean():3.0f}%")

# ================================================================== LESION
def lesion_hook(model, pt, idx, value):
    ii = torch.as_tensor(np.asarray(idx), device=DEVICE, dtype=torch.long)
    vv = torch.as_tensor(np.asarray(value), device=DEVICE, dtype=torch.float32)
    def hook(mod, inp, out):
        out = out.clone(); out[..., ii] = vv.to(out.dtype); return out
    return mlp_mod(model, pt).act.register_forward_hook(hook)

def eval_loss(model, langs, VAL, lesions=()):
    hs = [lesion_hook(model, pt, i, v) for pt, i, v in lesions]
    try:
        return {l: val_loss(model, VAL[l], l, masked=True) for l in langs}
    finally:
        for h in hs:
            h.remove()

def eval_chrf(model, langs, FL, lesions=()):
    hs = [lesion_hook(model, pt, i, v) for pt, i, v in lesions]
    try:
        return {l: chrf(model, FL["src"], FL[l], l, masked=True)[0] for l in langs}
    finally:
        for h in hs:
            h.remove()

def matched_pick(feat, self_ids, task_ids, g, k=5):
    """For each self neuron (random order) take one of its k nearest unused task neurons (z-scored features)."""
    F_ = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
    free = np.ones(len(task_ids), bool); out = []
    for s in g.permutation(self_ids):
        d = np.linalg.norm(F_[task_ids] - F_[s], axis=1); d[~free] = np.inf
        cand = np.argsort(d)[:3]; cand = cand[np.isfinite(d[cand])]
        c = g.choice(cand); free[c] = False; out.append(task_ids[c])
    return np.array(out)

DOSES = (0.05, 0.10, 0.20)
def part_lesion():
    outdir("lesion")
    VAL = {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:args.n_val] for l in LANGS}
    FL = {"src": read_flores("devtest", "en")[:args.n_chrf], **{l: read_flores("devtest", l)[:args.n_chrf] for l in LANGS}}
    say(f"=== HARDCORE LESION ({LANGSET}: {', '.join(LANGS)}) | tau {args.tau} | doses {DOSES} of each layer | 3 repeats | "
        f"loss on {args.n_val} val pairs/lang, chrF on {args.n_chrf} FLORES sentences/lang ===")
    say("  'matched' = task neurons with the same average activity, firing rate and outgoing weight size as the self neurons")
    rows, allrows, balance = [], [], []
    for name, run, phs, langs_s in lesion_specs():
        langs = LANGS if langs_s == "all" else langs_s.split(",")
        for ph in phs:
            path = ck(run, ph)
            if not path:
                say(f"  skip {name} p{ph} (no checkpoint)"); continue
            t0 = time.time(); model, c = load_model(path)
            acts, dlang = record(model); S = stats_from(acts, dlang)
            base = eval_loss(model, langs, VAL)
            info = {}
            for pt in MLP:
                A = acts[pt].float()
                mean = A.mean(0).numpy(); fire = (A > 0).float().mean(0).numpy()
                outn = mlp_mod(model, pt).proj.weight.detach().norm(dim=0).cpu().numpy()
                alive, smask = S[pt][2].numpy(), S[pt][3]
                info[pt] = (mean, np.stack([mean, fire, outn], 1), np.nonzero(smask)[0], np.nonzero(alive & ~smask)[0], int(alive.sum()))
            del acts, S
            g = np.random.default_rng(ph)
            # per layer, dose, repeat, ablation type
            for pt in MLP:
                mean, feat, sid, tid, nalive = info[pt]
                for dose in DOSES:
                    n = int(round(dose * nalive))
                    if n < 5 or n > len(sid) or n > len(tid):
                        continue
                    for rep in range(3):
                        ss = g.choice(sid, n, replace=False); mm = matched_pick(feat, ss, tid, g); rr = g.choice(tid, n, replace=False)
                        if rep == 0 and dose == 0.10:
                            for gname, ids in [("self", ss), ("matched", mm), ("random", rr)]:
                                balance.append({"model": name, "phase": ph, "point": pt, "group": gname, "avg_activity": feat[ids, 0].mean(),
                                                "firing_rate": feat[ids, 1].mean(), "out_weight": feat[ids, 2].mean()})
                        for abl in (["mean", "zero"] if dose == 0.10 else ["mean"]):
                            for gname, ids in [("self", ss), ("matched", mm), ("random", rr)]:
                                val = mean[ids] if abl == "mean" else np.zeros(len(ids))
                                r = eval_loss(model, langs, VAL, [(pt, ids, val)])
                                for l in langs:
                                    rows.append({"model": name, "phase": ph, "trained": c["lang"], "point": pt, "dose": dose, "n": n,
                                                 "self_frac": len(sid) / nalive, "rep": rep, "ablation": abl, "group": gname,
                                                 "eval_lang": l, "rise": r[l] - base[l]})
            # all 8 layers at once: loss (3 doses) + chrF (10%)
            chrf0 = eval_chrf(model, langs, FL)
            for dose in DOSES:
                for rep in range(2):
                    sets = {"self": [], "matched": [], "random": []}; ok = True
                    for pt in MLP:
                        mean, feat, sid, tid, nalive = info[pt]; n = int(round(dose * nalive))
                        if n < 5 or n > len(sid) or n > len(tid):
                            continue                               # this layer's self (or task group) is too small: leave it intact
                        ss = g.choice(sid, n, replace=False)
                        for gname, ids in [("self", ss), ("matched", matched_pick(feat, ss, tid, g)), ("random", g.choice(tid, n, replace=False))]:
                            sets[gname].append((pt, ids, mean[ids]))
                    if not ok:
                        continue
                    for gname, les in sets.items():
                        r = eval_loss(model, langs, VAL, les)
                        cf = eval_chrf(model, langs, FL, les) if dose == 0.10 else {}
                        for l in langs:
                            allrows.append({"model": name, "phase": ph, "dose": dose, "rep": rep, "group": gname, "eval_lang": l,
                                            "rise": r[l] - base[l], "chrf_intact": chrf0[l], "chrf": cf.get(l, np.nan)})
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            if not rows:
                say(f"  {name} p{ph}: nothing to lesion (self or task group too small)"); continue
            R = pd.DataFrame(rows); q = R[(R.model == name) & (R.phase == ph) & (R.ablation == "mean") & (R.dose == 0.10)]
            m_ = q.groupby("group").rise.mean()
            AL = pd.DataFrame(allrows); qa = AL[(AL.model == name) & (AL.phase == ph) & (AL.dose == 0.10)]
            cf = qa.groupby("group").chrf.mean(); c0 = qa.chrf_intact.mean()
            say(f"  {name} p{ph} ({c['lang']}, {time.time() - t0:.0f}s) | ONE layer, 10%: loss rise self {m_.get('self', np.nan):.4f} "
                f"matched {m_.get('matched', np.nan):.4f} random {m_.get('random', np.nan):.4f} | ALL layers 10%: chrF {c0:.1f} -> "
                f"self {cf.get('self', np.nan):.1f} matched {cf.get('matched', np.nan):.1f} random {cf.get('random', np.nan):.1f}")
            R.to_csv(os.path.join(OUT, "lesion_per_layer.csv"), index=False); AL.to_csv(os.path.join(OUT, "lesion_all_layers.csv"), index=False)
            pd.DataFrame(balance).to_csv(os.path.join(OUT, "matching_check.csv"), index=False)
    if rows:
        summarize_lesion()

MIN_RISE = 0.002     # below this the matched neurons did (almost) no damage and a ratio means nothing
def boot_ratio(q, g, n=2000):
    """self rise / matched rise and self - matched; bootstrap over (layer, repeat, language) cells, paired."""
    w = q.pivot_table(index=["point", "rep", "eval_lang"], columns="group", values="rise").dropna()
    s, m_ = w["self"].values, w["matched"].values
    ok = m_.mean() >= MIN_RISE
    est = s.mean() / m_.mean() if ok else np.nan
    br, bd = [], []
    for _ in range(n):
        i = g.integers(0, len(s), len(s)); bd.append(s[i].mean() - m_[i].mean())
        if ok:
            br.append(s[i].mean() / max(m_[i].mean(), 1e-6))
    lo, hi = (np.percentile(br, 2.5), np.percentile(br, 97.5)) if ok else (np.nan, np.nan)
    return est, lo, hi, (s > m_).mean(), len(s), s.mean() - m_.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5)

def summarize_lesion():
    R = pd.read_csv(os.path.join(OUT, "lesion_per_layer.csv")); AL = pd.read_csv(os.path.join(OUT, "lesion_all_layers.csv"))
    B = pd.read_csv(os.path.join(OUT, "matching_check.csv"))
    g = np.random.default_rng(0)
    say("\n  MATCHING CHECK (10% dose, averages over layers): the matched set should look like the self set")
    for (mname, ph), q in B.groupby(["model", "phase"], sort=False):
        t = q.groupby("group")[["avg_activity", "firing_rate", "out_weight"]].mean()
        say(f"  {mname} p{ph}: " + " | ".join(f"{gname}: activity {t.loc[gname, 'avg_activity']:.3f} firing {t.loc[gname, 'firing_rate']:.2f} weight {t.loc[gname, 'out_weight']:.2f}" for gname in ["self", "matched", "random"] if gname in t.index))
    say("\n  RESULT, ONE layer at a time (mean ablation): how many times more damage the self does than MATCHED task neurons")
    say(f"  (ratio shown as '-' when the matched neurons raised the loss by less than {MIN_RISE}: then compare the extra damage instead)")
    say(f"  {'model':<12}{'phase':>6}{'dose':>6}{'self / matched':>16}{'95% range':>16}{'extra loss (self - matched)':>30}{'self worse in':>15}{'self / random':>15}")
    summ = []
    for (mname, ph), q0 in R[R.ablation == "mean"].groupby(["model", "phase"], sort=False):
        for dose, q in q0.groupby("dose"):
            est, lo, hi, frac, n, dd, dlo, dhi = boot_ratio(q, g)
            rm = q[q.group == "random"].rise.mean()
            rr = q[q.group == "self"].rise.mean() / rm if rm >= MIN_RISE else np.nan
            summ.append({"model": mname, "phase": ph, "dose": dose, "ratio_matched": est, "lo": lo, "hi": hi, "extra": dd, "extra_lo": dlo,
                         "extra_hi": dhi, "frac_self_worse": frac, "ratio_random": rr, "cells": n})
            fx = lambda v: f"{v:.2f}x" if np.isfinite(v) else "-"
            say(f"  {mname:<12}{ph:>6}{int(100 * dose):>5}%{fx(est):>16}{(f'{lo:.2f}-{hi:.2f}x' if np.isfinite(lo) else '-'):>16}"
                f"{f'{dd:+.4f} ({dlo:+.4f} to {dhi:+.4f})':>30}{f'{100 * frac:.0f}% of {n}':>15}{fx(rr):>15}")
    Sm = pd.DataFrame(summ); Sm.to_csv(os.path.join(OUT, "summary_ratio.csv"), index=False)
    say("\n  zero ablation (10%) instead of mean ablation:")
    for (mname, ph), q in R[(R.ablation == "zero")].groupby(["model", "phase"], sort=False):
        est, lo, hi, frac, n, dd, dlo, dhi = boot_ratio(q, g)
        say(f"  {mname:<12}{ph:>6}  self / matched {est:.2f}x ({lo:.2f}-{hi:.2f}x) | extra loss {dd:+.4f} ({dlo:+.4f} to {dhi:+.4f})")
    say("\n  per layer (10%, mean ablation, all models pooled): self / matched")
    q = R[(R.ablation == "mean") & (R.dose == 0.10)]
    say("  " + " | ".join(f"{pt} {boot_ratio(q[q.point == pt], g, 300)[0]:.2f}x" for pt in MLP if len(q[q.point == pt])))
    say("\n  ALL 8 layers at once, 10% of each: real translation quality (chrF, higher = better)")
    for (mname, ph), q in AL[AL.dose == 0.10].groupby(["model", "phase"], sort=False):
        t = q.groupby(["eval_lang", "group"]).chrf.mean().unstack(); c0 = q.groupby("eval_lang").chrf_intact.first()
        if t.empty or not {"self", "matched", "random"} <= set(t.columns):
            continue
        say(f"  {mname} p{ph}: " + " | ".join(f"{l}: intact {c0[l]:.1f}, -self {t.loc[l, 'self']:.1f}, -matched {t.loc[l, 'matched']:.1f}, -random {t.loc[l, 'random']:.1f}" for l in t.index))
    # figure 1: dose-response per checkpoint
    keys = list(R[R.ablation == "mean"].groupby(["model", "phase"], sort=False).groups.keys())
    nc = min(4, len(keys)); nr = int(np.ceil(len(keys) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for ax, (mname, ph) in zip(axs.flat, keys):
        q = R[(R.model == mname) & (R.phase == ph) & (R.ablation == "mean")]
        for gname, col in [("self", "#4B1D6B"), ("matched", "#E07A1F"), ("random", "#9A9A9A")]:
            t = q[q.group == gname].groupby("dose").rise.agg(["mean", "sem"])
            ax.errorbar(100 * t.index, t["mean"], yerr=1.96 * t["sem"], marker="o", color=col, label=gname, capsize=3)
        ax.set_title(f"{mname}, phase {ph}", fontsize=10); ax.set_xlabel("% of the layer switched off"); ax.grid(alpha=.3)
    for ax in axs.flat[len(keys):]:
        ax.axis("off")
    axs[0, 0].set_ylabel("loss rise (average over\n8 layers x 3 repeats x languages)"); axs[0, 0].legend(fontsize=8)
    fig.suptitle("Switching off self neurons vs matched / random task neurons (one layer at a time)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dose_response.png"), dpi=140); plt.close(fig)
    # figure 2: ratio with 95% range, 10%
    s = Sm[(Sm.dose == 0.10) & Sm.ratio_matched.notna()].reset_index(drop=True)
    if len(s):
        fig, ax = plt.subplots(figsize=(1.0 * len(s) + 3, 4))
        ax.bar(range(len(s)), s.ratio_matched, color="#4B1D6B", yerr=[s.ratio_matched - s.lo, s.hi - s.ratio_matched], capsize=4)
        ax.axhline(1, color="k", lw=1, ls="--"); ax.set_xticks(range(len(s))); ax.set_xticklabels([f"{a}\np{b}" for a, b in zip(s.model, s.phase)], fontsize=8)
        ax.set_ylabel("damage: self / matched task neurons"); ax.set_title("10% of one layer switched off (1 = no difference)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "ratio_matched.png"), dpi=140); plt.close(fig)

# ================================================================== COMPARE related vs distant (CPU)
def part_compare():
    outdir("compare")
    base = os.path.join(REPO, "analysis", "v8")
    D = {k: pd.read_csv(os.path.join(base, f"related_{k}", "pairs.csv")) for k in ["indic", "distant"]
         if os.path.exists(os.path.join(base, f"related_{k}", "pairs.csv"))}
    if len(D) < 2:
        say("need both related_indic and related_distant first"); return
    upto = min(d.to.max() for d in D.values())                 # same phases for both
    say(f"=== RELATED (Hindi, Gujarati, Punjabi) vs DISTANT (Hindi, Chinese, Arabic), same anchor, phases up to {upto} ===")
    say(f"  {'':<10}{'language':<11}{'task change':>12}{'self less':>11}{'chance':>8}{'above chance':>14}{'self back enc':>15}{'self back dec':>15}")
    for k, d in D.items():
        d = d[d.to <= upto]
        for same, q in d.groupby("same_lang"):
            say(f"  {k:<10}{'SAME' if same else 'DIFFERENT':<11}{q.task.mean():>12.1f}{q.rel_gap.mean():>10.1f}%{q.chance.mean():>7.1f}%"
                f"{q.excess.mean():>+13.1f}{100 * q.kept_enc.mean():>14.0f}%{100 * q.kept_dec.mean():>14.0f}%")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, col in [("indic", "#2E8B57"), ("distant", "#C0392B")]:
        if k not in D:
            continue
        d = D[k][D[k].to <= upto]
        for same, mk in [(True, "o"), (False, "x")]:
            q = d[d.same_lang == same]
            lab = f"{'related' if k == 'indic' else 'distant'}, {'same' if same else 'different'} language"
            axs[0].scatter(q.task, q.excess, color=col, marker=mk, s=40, label=lab)
            axs[1].scatter(q.to, 100 * q.kept_dec, color=col, marker=mk, s=40, label=lab)
    axs[0].axhline(0, color="k", lw=1); axs[0].set_xlabel("how much the other neurons changed (task change score)")
    axs[0].set_ylabel("self protected, above chance (points)"); axs[0].set_title("Is the self protected?", fontsize=10)
    axs[1].set_xlabel("later phase"); axs[1].set_ylabel("% of the anchor-phase self back in the self (decoder)")
    axs[1].set_title("Does the same self come back?", fontsize=10)
    for a in axs:
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "related_vs_distant.png"), dpi=150); plt.close(fig)
    # forgetting: chrF of every language after each phase, both runs
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (k, run) in zip(axs, [("indic", "v8_indic_continual_s0"), ("distant", "v4b_continual_s0")]):
        f = os.path.join(REPO, "runs", run, "phases.jsonl")
        if not os.path.exists(f):
            continue
        rows = {}
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "chrf_masked" in r and 1 <= r["phase"] <= upto:
                    rows[r["phase"]] = r
        ph = sorted(rows)
        for l in rows[ph[0]]["chrf_masked"]:
            ax.plot(ph, [rows[p]["chrf_masked"][l] for p in ph], marker=".", label=l)
        ax.set_title(f"{'related' if k == 'indic' else 'distant'}: chrF of each language after every phase", fontsize=10)
        ax.set_xlabel("phase"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        forg = []
        for p in ph:
            for l, v in rows[p]["chrf_masked"].items():
                if l != rows[p]["lang"] and p - 1 in rows:
                    forg.append(rows[p - 1]["chrf_masked"][l] - v)
        say(f"  {k}: average chrF lost by a language when ANOTHER language is trained for one phase: {np.mean(forg):.1f} points")
    axs[0].set_ylabel("chrF"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "forgetting.png"), dpi=150); plt.close(fig)

# ================================================================== POOL every lesion run into one verdict (CPU)
def part_pool():
    from scipy.stats import binomtest
    outdir("pool")
    fs = sorted(glob.glob(os.path.join(REPO, "analysis", "v8", "lesion_*", "lesion_per_layer.csv")))
    R = pd.concat([pd.read_csv(f).assign(langset="related" if "indic" in f else "distant") for f in fs]).drop_duplicates(
        ["model", "phase", "point", "dose", "rep", "ablation", "group", "eval_lang"])
    R["ckpt"] = R.model + " p" + R.phase.astype(str)
    say(f"=== LESION, ALL RESULTS POOLED: {R.ckpt.nunique()} checkpoints x {R.point.nunique()} layers ===")
    say("  one 'cell' = one checkpoint + one layer (averaged over its 3 repeats and its languages)")
    g = np.random.default_rng(0)
    def cells(q):
        return q.groupby(["ckpt", "point", "group"]).rise.mean().unstack().dropna(subset=["self", "matched"])
    def verdict(q, label):
        if q.empty or not {"self", "matched"} <= set(q.group):
            return
        c = cells(q)
        if c.empty:
            return
        win = int((c["self"] > c["matched"]).sum()); n = len(c)
        p = binomtest(win, n, 0.5, alternative="greater").pvalue
        ck_ = c.index.get_level_values(0).unique(); bs = []
        for _ in range(2000):                                   # resample checkpoints, then layers inside them
            pick = g.choice(ck_, len(ck_)); sub = pd.concat([c.loc[k].sample(frac=1, replace=True, random_state=int(g.integers(1e9))) for k in pick])
            bs.append(sub["self"].mean() / max(sub["matched"].mean(), 1e-6))
        ok = c["matched"].mean() >= MIN_RISE
        est = c["self"].mean() / c["matched"].mean() if ok else np.nan
        rng_ = f"{est:.2f}x (95% range {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}x)" if ok else \
            f"- (matched barely hurt: extra loss {c['self'].mean() - c['matched'].mean():+.4f})"
        say(f"  {label:<34} self did more damage in {win} of {n} cells ({100 * win / n:.0f}%, chance would be 50%, p = {p:.1e}) | self / matched {rng_}")
        return c
    say("\n  10% of one layer, mean ablation:")
    q10 = R[(R.ablation == "mean") & (R.dose == 0.10)]
    verdict(q10, "EVERYTHING")
    for m in q10.model.unique():
        verdict(q10[q10.model == m], f"model: {m}")
    for side, pts in [("encoder", MLP_[:4]), ("decoder", MLP_[4:])]:
        verdict(q10[q10.point.isin(pts)], f"{side} layers only")
    say("\n  per layer, over all checkpoints:")
    for pt in MLP_:
        verdict(q10[q10.point == pt], f"layer {pt}")
    say("\n  other doses and zero ablation (everything pooled):")
    for dose in (0.05, 0.20):
        verdict(R[(R.ablation == "mean") & (R.dose == dose)], f"{int(100 * dose)}% of one layer")
    verdict(R[(R.ablation == "zero") & (R.dose == 0.10)], "10%, ZERO instead of average")
    # heat map: every checkpoint x every layer
    c = cells(q10); ratio = np.log2((c["self"].clip(lower=1e-4)) / (c["matched"].clip(lower=1e-4))).unstack()
    ratio = ratio.reindex(columns=[p for p in MLP_ if p in ratio.columns])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(ratio) + 1.8))
    im = ax.imshow(ratio.values, cmap="PuOr_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(ratio.shape[1])); ax.set_xticklabels([p.replace(".mlp", "") for p in ratio.columns])
    ax.set_yticks(range(len(ratio))); ax.set_yticklabels(ratio.index, fontsize=8)
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            v = ratio.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{2 ** v:.1f}x", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax); cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
    ax.set_title("Damage from switching off 10% of a layer: self / matched task neurons\n(purple = self hurts more; every cell = one checkpoint, one layer)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "every_checkpoint_every_layer.png"), dpi=150); plt.close(fig)

MLP_ = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]
for part in args.parts:
    {"related": part_related, "lesion": part_lesion, "compare": part_compare, "pool": part_pool}[part]()
say("finished")
