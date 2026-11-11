"""patch_v9.py: does the SELF carry WHICH LANGUAGE the model writes? (patching + steering, no training)

Model: one trained checkpoint. Neurons: decoder MLP layers dec0-dec3 (the encoder only sees English, so its activity
is the same whatever the output language and there is nothing to patch there).
Groups (30% of each decoder layer's working neurons, 3 different draws):
  self / lookalike (task neurons matched on average activity, firing rate, outgoing weight size) / random task neurons.
For every ordered language pair A -> B (e.g. asked for Hindi, try to push it to Chinese), on 100 FLORES devtest sentences:
  PATCH  : the model is asked for A (A's language tag). The group's neurons are REPLACED by the activity they had when the
           same sentence was translated into B (each neuron: its average over that sentence's B tokens).
           patch_all = all 4 decoder layers at once; patch_dec0..3 = one layer only.
  STEER  : the group's neurons get   alpha x (average activity in B - average activity in A)   added (alpha = 1, 2, 4).
           Averages come from the FLORES dev set (different sentences).
Measures (teacher forcing, no generation):
  shift_to_B = loss of the B reference before - after   (positive = the B translation became MORE likely)
  damage_A   = loss of the A reference after - before   (positive = the requested translation got worse)
  (both under A's tag, unmasked: every token can be predicted)
Generation (patch_all and steer alpha=2): greedy, unmasked. share_B = % of the output's letters in B's script;
chrF against the A reference.
Question: if the self is shared machinery, patching it should barely move the output to B, while task neurons should.

  CML_LANGSET=distant python -u patch_v9.py --gpu 0 --start mixed45=v6b_mixed_s0:45
  python -u patch_v9.py --summarize
"""
import os, sys, json, time, argparse, glob, re
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--start", default="")
ap.add_argument("--dose", type=float, default=0.30)
ap.add_argument("--draws", default="0,1,2")
ap.add_argument("--n_sent", type=int, default=100)
ap.add_argument("--alphas", default="1,2,4")
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--out", default="patch")
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "v9", args.out); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
SCRIPT_RE = {"hi": "ऀ-ॿ", "zh": "一-鿿", "ar": "؀-ۿ", "gu": "઀-૿", "pa": "਀-੿"}

def summarize():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import binomtest
    tf = pd.read_csv(f"{OUT}/teacher_forced.csv"); R = []
    say = lambda s: (print(s), R.append(s))
    say("=== v9 patching + steering: does the self carry WHICH LANGUAGE is written? ===")
    say("shift_to_B > 0: the B translation became more likely while the model was asked for A. damage_A > 0: the A translation got worse.")
    say("average over language pairs and draws:\n")
    t = tf.groupby(["start", "intervention", "group"])[["shift_to_B", "damage_A"]].mean().unstack("group")
    say(t.round(3).to_string() + "\n")
    say("COMPARISON (one = one start model x one language pair x one draw): look-alike task neurons vs self")
    for iv in sorted(tf.intervention.unique()):
        q = tf[tf.intervention == iv].pivot_table(index=["start", "pair", "draw"], columns="group", values="shift_to_B").dropna()
        if not len(q) or "self" not in q or "lookalike" not in q:
            continue
        k = int((q.lookalike > q.self).sum()); n = len(q)
        say(f"  {iv:10s}: look-alike moves the output to B MORE than self in {k}/{n} (sign test two-sided p = {binomtest(k, n, .5).pvalue:.3g}); "
            f"mean shift self {q.self.mean():+.3f} | look-alike {q.lookalike.mean():+.3f} | random {q.get('random', pd.Series([np.nan])).mean():+.3f}")
    if os.path.exists(f"{OUT}/generation.csv"):
        g = pd.read_csv(f"{OUT}/generation.csv")
        say("\nGENERATION (greedy, unmasked): % of output letters in B's script | chrF vs the A reference\n"
            + g.groupby(["start", "intervention", "group"])[["share_B", "chrf_A"]].mean().unstack("group").round(1).to_string())
    open(f"{OUT}/report.txt", "w").write("\n".join(R))
    # figure: shift_to_B per intervention, one panel per start model
    starts = list(tf.start.unique()); ivs = ["patch_dec0", "patch_dec1", "patch_dec2", "patch_dec3", "patch_all"] + sorted(i for i in tf.intervention.unique() if i.startswith("steer"))
    ivs = [i for i in ivs if i in set(tf.intervention)]
    COL = {"self": "#4B1D6B", "lookalike": "#888888", "random": "#C8C8C8"}
    fig, axs = plt.subplots(len(starts), 1, figsize=(10, 3.4 * len(starts)), squeeze=False)
    for ax, st in zip(axs[:, 0], starts):
        for j, grp in enumerate(["self", "lookalike", "random"]):
            q = tf[(tf.start == st) & (tf.group == grp)].groupby("intervention").shift_to_B
            m, lo, hi = q.mean(), q.quantile(.1), q.quantile(.9)
            xs = np.arange(len(ivs)) + (j - 1) * .26
            ax.bar(xs, [m.get(i, np.nan) for i in ivs], .26, color=COL[grp], label={"self": "self", "lookalike": "look-alike task", "random": "random task"}[grp])
        ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(ivs))); ax.set_xticklabels([i.replace("_", "\n") for i in ivs], fontsize=8)
        ax.set_ylabel("shift to B\n(B reference loss before - after)", fontsize=8.5); ax.set_title(f"{st}: replace / push 30% of a group, asked for A", fontsize=9.5)
        ax.grid(alpha=.25, axis="y")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_shift_to_B.png", dpi=170); plt.close(fig)
    print("figures + report in", OUT)

if args.summarize:
    summarize(); sys.exit()

from common_v8 import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
MLP_ = [f"enc{j}.mlp" for j in range(4)] + [f"dec{j}.mlp" for j in range(4)]

def mlp_mod(model, pt):
    return (model.enc_blocks if pt.startswith("enc") else model.dec_blocks)[int(pt[3])].mlp

# ---------------- the self (same definition as analysis_v8.py: largest co-firing group at tau 0.2, decoder language means removed)
DEV_EN = read_flores("dev", "en")
SRC = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in DEV_EN]
TGT = {l: [[LANG_TAG[l]] + sp.encode(r)[:MAX_TOK - 1] for r in read_flores("dev", l)] for l in LANGS}

@torch.no_grad()
def record(model, bs=64):
    model.eval(); cur = {}
    hooks = [mlp_mod(model, pt).act.register_forward_hook(lambda mm, i, o, k=pt: cur.__setitem__(k, o)) for pt in MLP_]
    acts = {pt: [] for pt in MLP_}; dlang = []
    for s in range(0, len(SRC), bs):
        S_ = pad_batch(SRC[s:s + bs]); mem, sm = model.encode(S_)
        sk = (S_ != PAD) & (S_ != EOS)
        for pt in MLP_[:4]:
            acts[pt].append(cur[pt][sk].half().cpu())
        for l in LANGS:
            T_ = pad_batch(TGT[l][s:s + bs]); model.decode(T_, mem, sm)
            tk = T_ != PAD; tk[:, 0] = False
            for pt in MLP_[4:]:
                acts[pt].append(cur[pt][tk].half().cpu())
            dlang.append(torch.full((int(tk.sum()),), LANGS.index(l)))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(dlang).numpy()

def self_masks(acts, dlang):
    g0 = np.random.default_rng(0)
    i_src = torch.tensor(np.sort(g0.choice(acts["enc0.mlp"].shape[0], min(acts["enc0.mlp"].shape[0], args.max_t), replace=False)))
    g1, parts = np.random.default_rng(1), []
    for li in range(len(LANGS)):
        pos = np.nonzero(dlang == li)[0]
        parts.append(np.sort(g1.choice(pos, min(len(pos), args.max_t // len(LANGS)), replace=False)))
    i_tgt = torch.tensor(np.concatenate(parts)); out = {}
    for pt in MLP_:
        enc = pt.startswith("enc")
        X = acts[pt][i_src if enc else i_tgt].float().to(DEVICE)
        if not enc:
            g = torch.as_tensor(dlang)[i_tgt].to(DEVICE)
            for v in g.unique():
                m = g == v; X[m] -= X[m].mean(0, keepdim=True)
        mu, sd = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
        alive = sd[0] > 1e-6
        Z = (X - mu) / torch.where(sd > 0, sd, torch.ones_like(sd)); Z[:, ~alive] = 0
        R = Z.T @ Z / Z.shape[0]
        A = ((R.abs() >= args.tau) & alive[:, None] & alive[None, :]).cpu().numpy(); np.fill_diagonal(A, False)
        _, lab = connected_components(csr_matrix(A), directed=False)
        alive = alive.cpu().numpy(); lab = np.where(alive, lab, -1)
        smask = lab == np.bincount(lab[lab >= 0]).argmax() if (lab >= 0).any() else np.zeros(len(lab), bool)
        out[pt] = (alive, smask)
        del X, Z, R
    return out

def matched_pick(feat, self_ids, task_ids, g):
    F_ = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
    free = np.ones(len(task_ids), bool); out = []
    for s in g.permutation(self_ids):
        d = np.linalg.norm(F_[task_ids] - F_[s], axis=1); d[~free] = np.inf
        cand = np.argsort(d)[:3]; cand = cand[np.isfinite(d[cand])]
        c = g.choice(cand); free[c] = False; out.append(task_ids[c])
    return np.array(out)

def lesion_hook(model, pt, idx, value):
    ii = torch.as_tensor(np.asarray(idx), device=DEVICE, dtype=torch.long)
    vv = torch.as_tensor(np.asarray(value), device=DEVICE, dtype=torch.float32)
    def hook(mod, inp, out):
        out = out.clone(); out[..., ii] = vv.to(out.dtype); return out
    return mlp_mod(model, pt).act.register_forward_hook(hook)

def neuron_info(model):
    torch.backends.cuda.matmul.allow_tf32 = False
    acts, dlang = record(model); M = self_masks(acts, dlang); info = {}
    for pt in MLP_:
        A = acts[pt].float(); mean = A.mean(0).numpy(); fire = (A > 0).float().mean(0).numpy()
        outn = mlp_mod(model, pt).proj.weight.detach().norm(dim=0).cpu().numpy()
        alive, smask = M[pt]
        info[pt] = dict(mean=mean, feat=np.stack([mean, fire, outn], 1), sid=np.nonzero(smask)[0],
                        tid=np.nonzero(alive & ~smask)[0], nalive=int(alive.sum()))
    del acts
    torch.backends.cuda.matmul.allow_tf32 = True
    return info

def pick_sets(info, seed):
    """10% of each layer's working neurons: from the self / look-alike task / random task. Different draw per seed."""
    g = np.random.default_rng(1000 + seed); sets = {"none": [], "self": [], "lookalike": [], "random": []}; bal = []
    for pt in MLP_:
        d = info[pt]; n = int(round(args.dose * d["nalive"]))
        if n < 5 or n > len(d["sid"]) or n > len(d["tid"]):
            print(f"    {pt}: self or task group too small, layer left intact"); continue
        ss = g.choice(d["sid"], n, replace=False); mm = matched_pick(d["feat"], ss, d["tid"], g); rr = g.choice(d["tid"], n, replace=False)
        for name, ids in [("self", ss), ("lookalike", mm), ("random", rr)]:
            sets[name].append((pt, ids, d["mean"][ids]))
            bal.append({"seed": seed, "point": pt, "set": name, "avg_activity": d["feat"][ids, 0].mean(),
                        "firing_rate": d["feat"][ids, 1].mean(), "out_weight": d["feat"][ids, 2].mean()})
    return sets, bal


DEC = MLP_[4:]
start_name, rest = args.start.split("="); run, ph = rest.split(":")
path = glob.glob(P(f"runs/{run}/ckpt/p{int(ph):03d}_*.pt"))[0]
model, ck = load_model(path); model.eval()
print(f"start model: {start_name} = {run} phase {ph} | languages {LANGS}")
t0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = False
acts, dlang = record(model)
GMEAN = {l: {pt: acts[pt][torch.as_tensor(dlang == i)].float().mean(0).to(DEVICE) for pt in DEC} for i, l in enumerate(LANGS)}
M = self_masks(acts, dlang); info = {}
for pt in MLP_:
    A_ = acts[pt].float(); mean = A_.mean(0).numpy(); fire = (A_ > 0).float().mean(0).numpy()
    outn = mlp_mod(model, pt).proj.weight.detach().norm(dim=0).cpu().numpy(); alive, smask = M[pt]
    info[pt] = dict(mean=mean, feat=np.stack([mean, fire, outn], 1), sid=np.nonzero(smask)[0], tid=np.nonzero(alive & ~smask)[0], nalive=int(alive.sum()))
del acts
torch.backends.cuda.matmul.allow_tf32 = True
for pt in DEC:
    d = info[pt]; print(f"  {pt}: self {len(d['sid'])} of {d['nalive']} working neurons; groups of {int(round(args.dose * d['nalive']))}")
print(f"  self found in {time.time() - t0:.0f}s")

EN = read_flores("devtest", "en")[:args.n_sent]
SRC = [sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in EN]
REF = {l: read_flores("devtest", l)[:args.n_sent] for l in LANGS}
RTOK = {l: [sp.encode(r)[:MAX_TOK - 1] for r in REF[l]] for l in LANGS}
BS = 50
CUR = {}
cur_mod = {}
def make_hook(pt, ii):
    def hook(mod, inp, out):
        spec = CUR.get(pt)
        if spec is None:
            return None
        kind, val = spec; out = out.clone()
        if kind == "set":
            out[..., ii] = val[:, None, :].to(out.dtype)
        else:
            out[..., ii] = out[..., ii] + val.to(out.dtype)
        return out
    return hook

@torch.no_grad()
def sentence_means(lang):
    """per sentence: each decoder neuron's average activity over that sentence's tokens, translating into `lang`"""
    out = {pt: [] for pt in DEC}; cur = {}
    hs = [mlp_mod(model, pt).act.register_forward_hook(lambda m, i, o, k=pt: cur.__setitem__(k, o)) for pt in DEC]
    with amp():
        for s in range(0, len(SRC), BS):
            S_ = pad_batch(SRC[s:s + BS]); mem, sm = model.encode(S_)
            T_ = pad_batch([[LANG_TAG[lang]] + t for t in RTOK[lang][s:s + BS]]); model.decode(T_, mem, sm)
            m = (T_ != PAD).float()[..., None]
            for pt in DEC:
                out[pt].append(((cur[pt].float() * m).sum(1) / m.sum(1)))
    for h in hs:
        h.remove()
    return {pt: torch.cat(v) for pt, v in out.items()}

@torch.no_grad()
def tf_loss(ask, ref_lang, setter):
    """mean token loss of ref_lang's reference when the model is asked for `ask` (unmasked)"""
    tot, cnt = 0.0, 0
    with amp():
        for s in range(0, len(SRC), BS):
            setter(s, s + BS)
            S_ = pad_batch(SRC[s:s + BS]); mem, sm = model.encode(S_)
            tin = pad_batch([[LANG_TAG[ask]] + t for t in RTOK[ref_lang][s:s + BS]])
            tout = pad_batch([t + [EOS] for t in RTOK[ref_lang][s:s + BS]])
            lg = model.decode(tin, mem, sm).float()
            l = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tout.reshape(-1), ignore_index=PAD, reduction="sum")
            tot += l.item(); cnt += int((tout != PAD).sum())
    CUR.clear()
    return tot / cnt

@torch.no_grad()
def generate(ask, setter, max_new=MAX_TOK):
    hyps = []
    with amp():
        for s in range(0, len(SRC), BS):
            setter(s, s + BS)
            S_ = pad_batch(SRC[s:s + BS]); mem, sm = model.encode(S_)
            ys = torch.full((S_.size(0), 1), LANG_TAG[ask], device=DEVICE); done = torch.zeros(S_.size(0), dtype=torch.bool, device=DEVICE)
            for _ in range(max_new):
                nxt = model.decode(ys, mem, sm)[:, -1].float().argmax(-1); nxt[done] = PAD
                ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == EOS
                if done.all():
                    break
            for row in ys[:, 1:].tolist():
                row = row[:row.index(EOS)] if EOS in row else row
                hyps.append(sp.decode([t for t in row if t != PAD]))
    CUR.clear()
    return hyps

def share(hyps, lang):
    tot = sum(len(re.findall("[" + "".join(SCRIPT_RE[l] for l in LANGS) + "]", h)) for h in hyps)
    return 100 * sum(len(re.findall(f"[{SCRIPT_RE[lang]}]", h)) for h in hyps) / max(tot, 1)

SMEAN = {l: sentence_means(l) for l in LANGS}
no_set = lambda a, b: CUR.clear()
BASE = {(A, B): tf_loss(A, B, no_set) for A in LANGS for B in LANGS}
print("baseline loss of the B reference when asked for A (rows A, columns B):")
print(pd.DataFrame({B: {A: BASE[(A, B)] for A in LANGS} for B in LANGS}).round(3).to_string())
GBASE = {}
for A in LANGS:
    h = generate(A, no_set)
    GBASE[A] = h
    print(f"  asked {A}: script share " + ", ".join(f"{l} {share(h, l):.0f}%" for l in LANGS)
          + f" | chrF {CHRF_METRIC.corpus_score(h, [REF[A]]).score:.1f}", flush=True)
alphas = [float(a) for a in args.alphas.split(",")]
TF, GEN = [], []
for draw in [int(d) for d in args.draws.split(",")]:
    g = np.random.default_rng(2000 + draw); sets = {"self": {}, "lookalike": {}, "random": {}}
    for pt in DEC:
        d = info[pt]; n = int(round(args.dose * d["nalive"]))
        if n < 5 or n > len(d["sid"]) or n > len(d["tid"]):
            print(f"    {pt}: group too small, skipped"); continue
        ss = g.choice(d["sid"], n, replace=False)
        sets["self"][pt] = ss; sets["lookalike"][pt] = matched_pick(d["feat"], ss, d["tid"], g); sets["random"][pt] = g.choice(d["tid"], n, replace=False)
    for grp, S in sets.items():
        hooks = [mlp_mod(model, pt).act.register_forward_hook(make_hook(pt, torch.as_tensor(ids, device=DEVICE))) for pt, ids in S.items()]
        idx = {pt: torch.as_tensor(ids, device=DEVICE) for pt, ids in S.items()}
        for A in LANGS:
            for B in LANGS:
                if A == B:
                    continue
                def patcher(layers):
                    def f(a, b):
                        CUR.clear()
                        for pt in layers:
                            if pt in idx:
                                CUR[pt] = ("set", SMEAN[B][pt][a:b][:, idx[pt]])
                    return f
                def steerer(alpha):
                    def f(a, b):
                        CUR.clear()
                        for pt in idx:
                            CUR[pt] = ("add", alpha * (GMEAN[B][pt] - GMEAN[A][pt])[idx[pt]])
                    return f
                ivs = {"patch_all": patcher(DEC), **{f"patch_{pt[:4]}": patcher([pt]) for pt in DEC},
                       **{f"steer_a{a:g}": steerer(a) for a in alphas}}
                for name, setter in ivs.items():
                    lb = tf_loss(A, B, setter); la = tf_loss(A, A, setter)
                    TF.append({"start": start_name, "draw": draw, "group": grp, "pair": f"{A}->{B}", "A": A, "B": B, "intervention": name,
                               "shift_to_B": BASE[(A, B)] - lb, "damage_A": la - BASE[(A, A)], "n_neurons": sum(len(v) for v in idx.values())})
                for name in ["patch_all", "steer_a2"]:
                    if name not in ivs:
                        continue
                    h = generate(A, ivs[name])
                    GEN.append({"start": start_name, "draw": draw, "group": grp, "pair": f"{A}->{B}", "intervention": name,
                                "share_B": share(h, B), "share_B_base": share(GBASE[A], B),
                                "chrf_A": CHRF_METRIC.corpus_score(h, [REF[A]]).score, "chrf_A_base": CHRF_METRIC.corpus_score(GBASE[A], [REF[A]]).score,
                                "example": h[0][:120]})
                r = [x for x in TF if x["draw"] == draw and x["group"] == grp and x["pair"] == f"{A}->{B}"]
                ga = [x for x in GEN if x["draw"] == draw and x["group"] == grp and x["pair"] == f"{A}->{B}"]
                print(f"  draw {draw} | {grp:9s} | asked {A}, pushed to {B} | shift to B: patch_all {r[0]['shift_to_B']:+.3f}, "
                      f"steer a2 {next((x['shift_to_B'] for x in r if x['intervention'] == 'steer_a2'), float('nan')):+.3f} | "
                      f"generated {B} script: patch {ga[0]['share_B']:.0f}% steer {ga[-1]['share_B']:.0f}% (normally {ga[0]['share_B_base']:.0f}%)", flush=True)
        for h in hooks:
            h.remove()
    pd.DataFrame(TF).to_csv(f"{OUT}/teacher_forced.csv", mode="a", header=not os.path.exists(f"{OUT}/teacher_forced.csv"), index=False)
    pd.DataFrame(GEN).to_csv(f"{OUT}/generation.csv", mode="a", header=not os.path.exists(f"{OUT}/generation.csv"), index=False)
    TF, GEN = [], []
print("done")
