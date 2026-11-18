"""mind_v12.py: three interpretability tests of the self, taken from the ideas in "Reading Model Minds" (steering dials,
same-size baselines, broadcast, one copy for many uses). No training. Count-matched like core_v11.py.

Self = largest co-firing group of a layer's MLP neurons at tau 0.2 (FLORES dev, decoder language means removed), found on
--self_langs (default: all languages of the language set). Task = all other working neurons.
Picks: in every layer, half of n_max neurons (n_max = size of the smaller group) from self, from task, and task
look-alikes of the self pick (same average activity, firing rate, outgoing weight size). Same picks as core_v11.py.

Parts (--part):
  dial       turn the picked neurons up or down and measure loss, uncertainty (entropy), top-1 accuracy, chrF
               gain k   every picked neuron's activity is multiplied by k (0 = off, 1 = untouched, 2 = doubled); all 8 layers
               clamp    every picked neuron is held at its own average (this is "silence" in core_v11.py); all 8 layers
               steer    decoder layers only: activity is pushed along the "confident minus unsure" direction
                          own:  each neuron by its own natural gap times a   (the old H7 test; groups get DIFFERENT push sizes)
                          same: the same direction, but rescaled so every group's push has the SAME total size as
                                the self group's natural gap times a          (the fair version)
  broadcast  nudge the model along ONE neuron's outgoing direction (same nudge size for every neuron) at its layer and
             measure how much the final hidden state moves per unit of nudge. Self neurons vs task neurons vs random directions.
  overlap    find the self separately on each language's text; how much do the sets overlap, compared with chance?

  CML_LANGSET=distant python -u mind_v12.py --gpu 0 --part dial --family distant --model distant=v4b_continual_s0:40,41,42
  python mind_v12.py --summarize --part dial
"""
import os, sys, time, argparse, glob, math
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--part", default="", choices=["", "dial", "broadcast", "overlap"])
ap.add_argument("--model", default="", help="name=run:phase[,phase,...]")
ap.add_argument("--family", default="")
ap.add_argument("--self_langs", default="", help="comma list of languages the self is found on; default = all of the language set")
ap.add_argument("--lang", default="", help="language used for the measurements; default = the language the checkpoint was last trained on")
ap.add_argument("--draws", default="0,1")
ap.add_argument("--gains", default="0,0.5,1.5,2")
ap.add_argument("--alphas", default="-2,-1,1,2")
ap.add_argument("--n_val", type=int, default=200)
ap.add_argument("--n_chrf", type=int, default=100)
ap.add_argument("--n_dirs", type=int, default=48, help="broadcast: neurons per group per layer")
ap.add_argument("--n_sent", type=int, default=64, help="broadcast: sentences")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--out", default="")
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "v12", args.out or args.part or "misc"); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
LAYERS = [f"enc{j}" for j in range(4)] + [f"dec{j}" for j in range(4)]

def append_csv(path, rows):
    if len(rows):
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def sign_p(k, n):
    """one-sided sign test: chance of k or more out of n by coin flips"""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0

# =============================================================================================== SUMMARY (CPU)
def summarize():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    R = []
    def say(x=""):
        print(x); R.append(str(x))
    if args.part == "dial":
        d = pd.read_csv(f"{OUT}/dial.csv")
        d = d.drop_duplicates(["family", "run", "phase", "draw", "group", "kind", "value"], keep="last")
        say("=== dial: change vs the untouched model (average over checkpoints and draws). loss, entropy: higher = worse / more unsure ===")
        say("one comparison = one checkpoint x one draw; self, task and look-alike picks all have the same number of neurons")
        for kind, title in [("gain", "gain k: picked neurons' activity multiplied by k, all 8 layers (1 = untouched)"),
                            ("clamp", "clamp: picked neurons held at their own average, all 8 layers"),
                            ("steer_same", "steer, SAME push size for every group (fair), decoder layers; a = push size in units of the self group's natural gap"),
                            ("steer_own", "steer, each neuron by its OWN natural gap (old H7; push sizes differ between groups), decoder layers")]:
            q = d[d["kind"] == kind]
            if not len(q):
                continue
            say(f"\n--- {title}")
            for col in ["d_loss", "d_entropy", "d_top1", "d_chrf"]:
                if q[col].notna().any():
                    t = q.pivot_table(index=["family", "group"], columns="value", values=col)
                    say(f"{col}:"); say(t.round(3).to_string())
            for other in ["task", "look"]:
                rows = []
                for (fam, val), x in q.groupby(["family", "value"]):
                    p = x.pivot_table(index=["run", "phase", "draw"], columns="group", values=["d_loss", "d_entropy"])
                    if ("d_loss", "self") not in p.columns or ("d_loss", other) not in p.columns:
                        continue
                    a_, b_ = p[("d_loss", "self")].abs(), p[("d_loss", other)].abs()
                    e_, f_ = p[("d_entropy", "self")].abs(), p[("d_entropy", other)].abs()
                    ok = (a_.notna() & b_.notna()); n = int(ok.sum()); k = int((a_[ok] > b_[ok]).sum()); k2 = int((e_[ok] > f_[ok]).sum())
                    rows.append({"family": fam, "value": val, "n": n, "|loss change| bigger for self": f"{k}/{n}", "sign_p": round(sign_p(k, n), 4),
                                 "|entropy change| bigger for self": f"{k2}/{n}", "sign_p ": round(sign_p(k2, n), 4)})
                if rows:
                    say(f"self vs {other}: in how many comparisons does the self pick move the model MORE (size of the change, either sign)")
                    say(pd.DataFrame(rows).to_string(index=False))
        if "push_norm" in d.columns:
            q = d[d["kind"].isin(["steer_same", "steer_own"]) & (d["value"] == 1)]
            if len(q):
                say("\n--- steer: total push size actually applied at a = 1 (sum over decoder layers of the push vector's length)")
                say(q.pivot_table(index=["family", "kind"], columns="group", values="push_norm").round(3).to_string())
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            COL = {"self": "#6b3fa0", "task": "#d9730d", "look": "#2a9d8f"}
            for kind, xl in [("gain", "gain k (1 = untouched)"), ("steer_same", "push size a (same for every group)")]:
                q = d[d["kind"] == kind]; fams = sorted(q["family"].unique())
                if not len(q):
                    continue
                for col, yl in [("d_loss", "loss change"), ("d_entropy", "entropy change")]:
                    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.6), squeeze=False)
                    for ax, fam in zip(axes[0], fams):
                        for grp in ["self", "task", "look"]:
                            x = q[(q["family"] == fam) & (q["group"] == grp)].groupby("value")[col].mean()
                            if kind == "gain":
                                x.loc[1.0] = 0.0
                            x = x.sort_index(); ax.plot(x.index, x.values, "o-", color=COL[grp], label=grp, lw=2, ms=5)
                        ax.axhline(0, color="#999", lw=0.8); ax.set_title(fam, fontsize=11); ax.set_xlabel(xl); ax.set_ylabel(yl)
                        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9)
                    fig.tight_layout(); fig.savefig(f"{OUT}/dial_{kind}_{col}.png", dpi=160); plt.close(fig)
        except Exception as e:
            say(f"(figure skipped: {e})")
    if args.part == "broadcast":
        d = pd.read_csv(f"{OUT}/broadcast.csv").drop_duplicates(["family", "run", "phase", "layer", "group", "idx"], keep="last")
        say("=== broadcast: how much the FINAL hidden state moves per unit of nudge along one direction at a layer (gain) ===")
        say("same nudge size for every direction; self / task = one neuron's outgoing weight direction; random = a random direction")
        t = d.pivot_table(index=["family", "layer"], columns="group", values="gain")
        t["self/task"] = t["self"] / t["task"]; t["self/random"] = t["self"] / t["random"]
        say(t.round(3).to_string())
        say("\n--- effect on the output words (KL between untouched and nudged predictions, x1000)")
        say((1000 * d.pivot_table(index=["family", "layer"], columns="group", values="kl")).round(3).to_string())
        rows = []
        for fam, x in d.groupby("family"):
            p = x.pivot_table(index=["run", "phase", "layer"], columns="group", values="gain")
            for other in ["task", "random"]:
                n = len(p); k = int((p["self"] > p[other]).sum())
                rows.append({"family": fam, "self vs": other, "self gain / other gain": round(float(p["self"].mean() / p[other].mean()), 3),
                             "self bigger in (checkpoint x layer)": f"{k}/{n}", "sign_p": round(sign_p(k, n), 4)})
        say("\n--- one comparison = one checkpoint x one layer (average of the sampled neurons)"); say(pd.DataFrame(rows).to_string(index=False))
    if args.part == "overlap":
        d = pd.read_csv(f"{OUT}/overlap.csv").drop_duplicates(["family", "run", "phase", "layer", "a", "b"], keep="last")
        say("=== overlap: the self found separately on each language's text (decoder layers; the encoder only ever reads English) ===")
        say("jaccard = shared / in either. x_chance = shared neurons / the number expected if the two sets were unrelated (1 = chance)")
        pr = d[d["b"] != "ALL3"]
        t = pr.groupby(["family", "layer"]).agg(self_a=("n_a", "mean"), self_b=("n_b", "mean"), shared=("n_both", "mean"),
                                                 jaccard=("jaccard", "mean"), x_chance=("x_chance", "mean"), pairs=("jaccard", "size"))
        say(t.round(3).to_string())
        c3 = d[d["b"] == "ALL3"]
        if len(c3):
            say("\n--- in the self for ALL the languages at once (n_both), vs expected by chance")
            say(c3.groupby(["family", "layer"]).agg(in_all=("n_both", "mean"), expected=("expected", "mean"), x_chance=("x_chance", "mean"),
                                                     share_of_layer=("share", "mean")).round(3).to_string())
    with open(f"{OUT}/report.txt", "w") as f:
        f.write("\n".join(R))
    print("report in", OUT)

if args.summarize:
    summarize(); sys.exit()

# =============================================================================================== GPU (helpers copied from core_v11.py)
from common_v8 import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
FAMILY = args.family or (args.model.split("=")[0] if "=" in args.model else args.model)
SELF_LANGS = [l for l in (args.self_langs.split(",") if args.self_langs else LANGS) if l in LANGS]

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

def groups(acts, dlang, n_langs):
    """per layer: alive mask, self mask (largest co-firing group at tau)"""
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
        else:
            smask = np.zeros(len(lab), bool)
        out[L] = {"alive": alive_np, "self": smask}
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
        info[L] = dict(mean=mean, feat=np.stack([mean, fire, outn], 1), sid=np.nonzero(alive & smask)[0], tid=np.nonzero(alive & ~smask)[0])
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

def pick_half(info, seed):
    """count-matched picks at dose 0.5, same random streams as core_v11.pick_sets. Returns {group: [(layer, sorted ids)]}"""
    sets = {"self": [], "task": [], "look": []}
    for li, L in enumerate(LAYERS):
        d = info[L]; n_max = min(len(d["sid"]), len(d["tid"]))
        if n_max < 10:
            continue
        g = np.random.default_rng([1000 + seed, li, 500])
        n = max(1, int(round(0.5 * n_max)))
        ss = g.choice(d["sid"], n, replace=False); tt = g.choice(d["tid"], n, replace=False)
        lk = matched_pick(d["feat"], ss, d["tid"], g)
        for k, ids in [("self", ss), ("task", tt), ("look", lk)]:
            sets[k].append((L, np.sort(np.asarray(ids))))
    return sets

def parse_model(spec):
    name, rest = spec.split("="); run, phs = rest.split(":")
    return name, run, [int(p) for p in phs.split(",")]

def ckpt_path(run, ph):
    f = glob.glob(P(f"runs/{run}/ckpt/p{int(ph):03d}_*.pt"))
    return f[0] if f else None

def measure_lang(ck):
    l = args.lang or str(ck.get("lang"))
    return l if l in LANGS else ("hi" if "hi" in LANGS else LANGS[0])

def done_before(path, run, ph):
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path, usecols=["run", "phase"])
    return bool(((d["run"].astype(str) == run) & (d["phase"] == ph)).any())

def edit_hooks(model, spec):
    """spec: list of (layer, ids, kind, value). kind 'gain': activity * value (number); 'set': activity = value (one number per id);
    'add': activity + value (one number per id). Applied at every position."""
    hooks = []
    for L, ids, kind, val in spec:
        ii = torch.as_tensor(np.asarray(ids), device=DEVICE, dtype=torch.long)
        vv = None if kind == "gain" else torch.as_tensor(np.asarray(val), device=DEVICE, dtype=torch.float32)
        def hook(mod, inp, out, ii=ii, vv=vv, kind=kind, val=val):
            out = out.clone()
            if kind == "gain":
                out[..., ii] = out[..., ii] * float(val)
            elif kind == "set":
                out[..., ii] = vv.to(out.dtype)
            else:
                out[..., ii] = out[..., ii] + vv.to(out.dtype)
            return out
        hooks.append(mlp_mod(model, L).act.register_forward_hook(hook))
    return hooks

def batches(data, bs=50):
    for s in range(0, len(data), bs):
        ch = data[s:s + bs]
        yield pad_batch([d[0] for d in ch]), pad_batch([d[1] for d in ch]), pad_batch([d[2] for d in ch])

def masked_logp(logits, lang):
    return F.log_softmax(logits.float().masked_fill(~ALLOWED[lang], float("-inf")), -1)

def entropy_of(lp):
    p = lp.exp()
    return -(p * lp.masked_fill(p == 0, 0)).sum(-1)

@torch.no_grad()
def tok_stats(model, data, lang):
    """average per target token (only tokens of the language's own script, so the loss is always finite): loss, entropy, top-1 %"""
    model.eval(); tl = te = tc = 0.0; n = 0
    for src, tin, tout in batches(data):
        with amp():
            logits, _ = model(src, tin)
        lp = masked_logp(logits, lang); keep = (tout != PAD) & ALLOWED[lang][tout]
        lpk, t = lp[keep], tout[keep]
        tl += -lpk.gather(1, t[:, None]).sum().item(); te += entropy_of(lpk).sum().item()
        tc += (lpk.argmax(1) == t).float().sum().item(); n += int(keep.sum())
    n = max(n, 1)
    return tl / n, te / n, 100 * tc / n

@torch.no_grad()
def conf_direction(model, data, lang):
    """per decoder layer, per neuron: average activity on the model's most confident third of tokens minus on its least confident third"""
    model.eval(); cur = {}
    hooks = [mlp_mod(model, L).act.register_forward_hook(lambda mm, i, o, k=L: cur.__setitem__(k, o)) for L in LAYERS[4:]]
    A = {L: [] for L in LAYERS[4:]}; E = []
    for src, tin, tout in batches(data):
        with amp():
            logits, _ = model(src, tin)
        ent = entropy_of(masked_logp(logits, lang)); keep = (tout != PAD) & ALLOWED[lang][tout]
        E.append(ent[keep].float().cpu())
        for L in LAYERS[4:]:
            A[L].append(cur[L][keep].float().cpu())
    for h in hooks:
        h.remove()
    E = torch.cat(E); lo, hi = torch.quantile(E, 1 / 3).item(), torch.quantile(E, 2 / 3).item()
    conf, uns = E <= lo, E >= hi
    out = {}
    for L in LAYERS[4:]:
        X = torch.cat(A[L]); out[L] = (X[conf].mean(0) - X[uns].mean(0)).numpy()
    return out

# =============================================================================================== PART dial
if args.part == "dial":
    name, run, phases = parse_model(args.model)
    gains = [float(x) for x in args.gains.split(",")]; alphas = [float(x) for x in args.alphas.split(",")]
    FT_EN = read_flores("devtest", "en")[:args.n_chrf]
    for ph in phases:
        path = ckpt_path(run, ph)
        if path is None:
            print(f"  {run} phase {ph}: checkpoint not on this machine, skipped", flush=True); continue
        if done_before(f"{OUT}/dial.csv", run, ph):
            print(f"  {run} phase {ph}: already in dial.csv, skipped", flush=True); continue
        t0 = time.time(); model, ck = load_model(path); model.eval(); lang = measure_lang(ck)
        pairs = load_pairs(P(f"data/en-{lang}/val.jsonl"), lang)
        EVAL, DIRSET = pairs[:args.n_val], pairs[args.n_val:2 * args.n_val]          # the direction is found on sentences NOT used for measuring
        FT = read_flores("devtest", lang)[:args.n_chrf]
        info = neuron_info(model, SELF_LANGS)
        DIR = conf_direction(model, DIRSET, lang)
        b_loss, b_ent, b_top = tok_stats(model, EVAL, lang); b_chrf = chrf(model, FT_EN, FT, lang, masked=True)[0]
        print(f"{FAMILY} phase {ph} (last trained {ck.get('lang')}; measured on {lang}; self found on {','.join(SELF_LANGS)}): "
              f"untouched loss {b_loss:.3f}, entropy {b_ent:.3f}, top-1 {b_top:.1f}%, chrF {b_chrf:.1f}", flush=True)
        rows = []
        def run_one(draw, grp, kind, value, spec, with_chrf, push_norm=np.nan):
            hooks = edit_hooks(model, spec)
            l, e, t1 = tok_stats(model, EVAL, lang)
            c = chrf(model, FT_EN, FT, lang, masked=True)[0] if with_chrf else np.nan
            for h in hooks:
                h.remove()
            rows.append({"family": FAMILY, "run": run, "phase": ph, "lang": lang, "draw": draw, "group": grp, "kind": kind, "value": value,
                         "n_neurons": sum(len(s[1]) for s in spec), "push_norm": push_norm,
                         "d_loss": l - b_loss, "d_entropy": e - b_ent, "d_top1": t1 - b_top, "d_chrf": (c - b_chrf) if c == c else np.nan})
        for draw in [int(x) for x in args.draws.split(",")]:
            sets = pick_half(info, draw)
            ref = {L: float(np.linalg.norm(DIR[L][ids])) for L, ids in sets["self"] if L in DIR}     # self group's natural gap size per decoder layer
            for grp in ["self", "task", "look"]:
                S = sets[grp]
                for k in gains:
                    run_one(draw, grp, "gain", k, [(L, ids, "gain", k) for L, ids in S], with_chrf=(k in (0.0, 2.0)))
                run_one(draw, grp, "clamp", 1.0, [(L, ids, "set", info[L]["mean"][ids]) for L, ids in S], with_chrf=True)
                SD = [(L, ids) for L, ids in S if L in DIR and L in ref]
                for a in alphas:
                    own = [(L, ids, "add", a * DIR[L][ids]) for L, ids in SD]
                    same = [(L, ids, "add", a * ref[L] * DIR[L][ids] / (np.linalg.norm(DIR[L][ids]) + 1e-12)) for L, ids in SD]
                    for kind, spec in [("steer_own", own), ("steer_same", same)]:
                        pn = float(sum(np.linalg.norm(s[3]) for s in spec))
                        run_one(draw, grp, kind, a, spec, with_chrf=(kind == "steer_same" and abs(a) == 2), push_norm=pn)
        c_loss, _, _ = tok_stats(model, EVAL, lang)
        append_csv(f"{OUT}/dial.csv", rows)
        try:
            r = pd.DataFrame(rows); r = r[r["kind"].isin(["gain", "steer_same"])].pivot_table(index=["kind", "group"], columns="value", values="d_entropy")
            print("  entropy change (average of draws):"); print(r.round(3).to_string())
        except Exception as e:
            print(f"  (short table skipped: {e})")
        print(f"  hooks removed check: loss differs from untouched by {abs(c_loss - b_loss):.5f} (should be 0) | {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done"); sys.exit()

# =============================================================================================== PART broadcast
if args.part == "broadcast":
    name, run, phases = parse_model(args.model)
    for ph in phases:
        path = ckpt_path(run, ph)
        if path is None:
            print(f"  {run} phase {ph}: checkpoint not on this machine, skipped", flush=True); continue
        if done_before(f"{OUT}/broadcast.csv", run, ph):
            print(f"  {run} phase {ph}: already in broadcast.csv, skipped", flush=True); continue
        t0 = time.time(); model, ck = load_model(path); model.eval(); lang = measure_lang(ck)
        info = neuron_info(model, SELF_LANGS)
        S_ = pad_batch(SRC[:args.n_sent]); T_ = pad_batch(TGT[lang][:args.n_sent])
        tk = T_ != PAD                                                                 # real target positions (each predicts the next word)
        fin = {}; hf = model.dec_ln.register_forward_hook(lambda mm, i, o: fin.__setitem__("h", o))
        mo = {}; hm = [mlp_mod(model, L).register_forward_hook(lambda mm, i, o, k=L: mo.__setitem__(k, o)) for L in LAYERS]
        with torch.no_grad():                                                           # full precision (no amp): the nudge is small
            mem, sm = model.encode(S_); lg0 = model.decode(T_, mem, sm)
            h0 = fin["h"].float().clone(); lp0 = masked_logp(lg0, lang)
            sk = S_ != PAD
            eps = {L: 0.1 * float(mo[L].float()[sk if L.startswith("enc") else tk].norm(dim=-1).mean()) for L in LAYERS}
        for h in hm:
            h.remove()
        rows = []; D_ = h0.shape[-1]
        for li, L in enumerate(LAYERS):
            g = np.random.default_rng([7, ph, li]); W = mlp_mod(model, L).proj.weight.detach().float()      # (d_model, n_neurons)
            cands = [("self", g.choice(info[L]["sid"], min(args.n_dirs, len(info[L]["sid"])), replace=False)),
                     ("task", g.choice(info[L]["tid"], min(args.n_dirs, len(info[L]["tid"])), replace=False)),
                     ("random", np.arange(args.n_dirs))]
            tg = torch.Generator(device="cpu").manual_seed(100_000 + 1000 * ph + li)
            for grp, ids in cands:
                for j, nid in enumerate(ids):
                    v = W[:, int(nid)].clone() if grp != "random" else torch.randn(D_, generator=tg).to(DEVICE)
                    v = eps[L] * v / (v.norm() + 1e-12)
                    hk = mlp_mod(model, L).register_forward_hook(lambda mm, i, o, v=v: o + v.to(o.dtype))
                    with torch.no_grad():
                        mem1, sm1 = model.encode(S_); lg1 = model.decode(T_, mem1, sm1)
                    hk.remove()
                    dh = (fin["h"].float() - h0)[tk].norm(dim=-1).mean().item()
                    lp1 = masked_logp(lg1, lang); p0 = lp0[tk].exp()
                    kl = (p0 * (lp0[tk].masked_fill(p0 == 0, 0) - lp1[tk].masked_fill(p0 == 0, 0))).sum(-1).mean().item()
                    rows.append({"family": FAMILY, "run": run, "phase": ph, "lang": lang, "layer": L, "group": grp, "idx": j, "neuron": int(nid),
                                 "eps": eps[L], "gain": dh / eps[L], "kl": kl})
        hf.remove()
        append_csv(f"{OUT}/broadcast.csv", rows)
        r = pd.DataFrame(rows).pivot_table(index="layer", columns="group", values="gain").reindex(LAYERS)
        print(f"{FAMILY} phase {ph} (measured on {lang}): final-state movement per unit nudge"); print(r.round(3).to_string())
        print(f"  {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done"); sys.exit()

# =============================================================================================== PART overlap
if args.part == "overlap":
    name, run, phases = parse_model(args.model)
    for ph in phases:
        path = ckpt_path(run, ph)
        if path is None:
            print(f"  {run} phase {ph}: checkpoint not on this machine, skipped", flush=True); continue
        if done_before(f"{OUT}/overlap.csv", run, ph):
            print(f"  {run} phase {ph}: already in overlap.csv, skipped", flush=True); continue
        t0 = time.time(); model, ck = load_model(path); model.eval(); M = {}
        torch.backends.cuda.matmul.allow_tf32 = False
        for l in LANGS:
            acts, dlang = record(model, [l]); G = groups(acts, dlang, 1); del acts
            M[l] = {L: (G[L]["alive"] & G[L]["self"], G[L]["alive"]) for L in LAYERS[4:]}
        torch.backends.cuda.matmul.allow_tf32 = True
        rows = []; LL = list(LANGS)
        for L in LAYERS[4:]:
            N = int(np.all([M[l][L][1] for l in LL], axis=0).sum())
            for i in range(len(LL)):
                for j in range(i + 1, len(LL)):
                    a, b = M[LL[i]][L][0], M[LL[j]][L][0]
                    na, nb, both = int(a.sum()), int(b.sum()), int((a & b).sum()); exp = na * nb / max(N, 1)
                    rows.append({"family": FAMILY, "run": run, "phase": ph, "layer": L, "a": LL[i], "b": LL[j], "n_layer": N, "n_a": na, "n_b": nb,
                                 "n_both": both, "jaccard": both / max(int((a | b).sum()), 1), "expected": exp, "x_chance": both / max(exp, 1e-9), "share": both / max(N, 1)})
            allm = np.all([M[l][L][0] for l in LL], axis=0); exp = N * float(np.prod([M[l][L][0].sum() / max(N, 1) for l in LL]))
            rows.append({"family": FAMILY, "run": run, "phase": ph, "layer": L, "a": "+".join(LL), "b": "ALL3", "n_layer": N, "n_a": np.nan, "n_b": np.nan,
                         "n_both": int(allm.sum()), "jaccard": np.nan, "expected": exp, "x_chance": int(allm.sum()) / max(exp, 1e-9), "share": int(allm.sum()) / max(N, 1)})
        append_csv(f"{OUT}/overlap.csv", rows)
        r = pd.DataFrame(rows); r = r[r["b"] != "ALL3"].groupby("layer")[["jaccard", "x_chance"]].mean()
        print(f"{FAMILY} phase {ph} ({ck.get('lang')}): overlap of the selfs found on each language"); print(r.round(3).to_string())
        print(f"  {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done"); sys.exit()
print("nothing to do: give --part or --summarize")
