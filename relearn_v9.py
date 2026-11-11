"""relearn_v9.py: wipe part of the self, then let the model RELEARN its own language. Does it come back slower?

Start: a trained translator checkpoint. Right before training, the chosen neurons are WIPED: their incoming weights
(fc row + bias) and outgoing weights (proj column) are re-drawn exactly as at initialisation. Unlike the lesion
experiments, the neurons are NOT held off: they can learn again.
Conditions (30% of each MLP layer's working neurons by default):
  none       nothing wiped (just keeps training: the baseline)
  self       neurons taken from the self
  lookalike  task neurons with the same average activity, firing rate and outgoing weight size
  random     random task neurons
Training: the language the checkpoint was LAST trained on (the one it knows best), normal script-masked loss,
learning rate 3e-5, 1,000 steps. Learning curve: held-out loss (1,000 val pairs) at fixed steps + chrF (200 FLORES
devtest sentences). "Forgetting" columns = the other languages before (just after the wipe) vs after relearning.

  CML_LANGSET=distant python -u relearn_v9.py --gpu 0 --start distant30=v4b_continual_s0:30 --seeds 0
  python -u relearn_v9.py --summarize
"""
import os, sys, json, time, argparse, glob, random
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--start", default="", help="name=run:phase, or 'fresh' (random model, condition 'none' only)")
ap.add_argument("--tasks", default="same")
ap.add_argument("--conds", default="none,self,lookalike,random")
ap.add_argument("--seeds", default="0")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--dose", type=float, default=0.30)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--max_tokens", type=int, default=8000)
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--max_t", type=int, default=24000)
ap.add_argument("--summarize", action="store_true")
ap.add_argument("--out", default="relearn", help="results folder under analysis/v9/")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "analysis", "v9", args.out); os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
EVAL_AT = [0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
TASK_ORDER = ["same", "next", "indonesian", "sentiment", "topic", "addition"]          # near -> far from translation
TASK_LABEL = {"same": "Own language\n(relearn after wipe)", "next": "Next phase\n(its next language)", "indonesian": "Indonesian\n(new language)", "sentiment": "Sentiment\n(2 labels)",
              "topic": "Topic\n(4 labels)", "addition": "Addition\n(not language)"}

# =============================================================================================== SUMMARY (CPU)
def summarize():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import binomtest
    cur = pd.read_csv(f"{OUT}/curves.csv"); fg = pd.read_csv(f"{OUT}/forgetting.csv") if os.path.exists(f"{OUT}/forgetting.csv") else None
    R = []
    say = lambda s: (print(s), R.append(s))
    say("=== v9: WIPE part of the self, then RELEARN the language the model knows best ===")
    say("one comparison = one start model x one seed: self wiped vs look-alike task neurons wiped (same number)")
    say("learning slowness = average held-out loss over the learning curve (steps 50 ... end); lower = learned faster\n")
    # per run summary
    last = cur[cur.step > 0].groupby(["start", "task", "cond", "seed"]).agg(curve_loss=("val_loss", "mean")).reset_index()
    fin = cur.loc[cur.groupby(["start", "task", "cond", "seed"]).step.idxmax(), ["start", "task", "cond", "seed", "score", "val_loss", "step"]]
    run = last.merge(fin, on=["start", "task", "cond", "seed"]).rename(columns={"score": "final_score", "val_loss": "final_loss"})
    run.to_csv(f"{OUT}/runs_summary.csv", index=False)
    say("AVERAGE OVER SEEDS (curve loss | final score; score = accuracy % or chrF for Indonesian)")
    piv = run.groupby(["task", "start", "cond"]).agg(curve_loss=("curve_loss", "mean"), final_score=("final_score", "mean"), seeds=("seed", "nunique"))
    say(piv.round(3).to_string() + "\n")
    rows = []
    for (st, task, seed), q in run[run.start != "fresh"].groupby(["start", "task", "seed"]):
        q = q.set_index("cond")
        if {"self", "lookalike"} <= set(q.index):
            rows.append({"start": st, "task": task, "seed": seed,
                         "extra_loss": q.loc["self", "curve_loss"] - q.loc["lookalike", "curve_loss"],
                         "score_drop": q.loc["lookalike", "final_score"] - q.loc["self", "final_score"],
                         "extra_loss_vs_random": q.loc["self", "curve_loss"] - q.loc["random", "curve_loss"] if "random" in q.index else np.nan})
    C = pd.DataFrame(rows); C.to_csv(f"{OUT}/comparisons.csv", index=False)
    say("RESULT: self wiped vs look-alike wiped (positive = after wiping the self the model is WORSE while relearning)")
    for task in TASK_ORDER + ["ALL"]:
        q = C if task == "ALL" else C[C.task == task]
        if not len(q):
            continue
        k, n = int((q.extra_loss > 0).sum()), len(q)
        k2 = int((q.score_drop > 0).sum())
        say(f"  {task:10s}: curve loss higher in {k}/{n} (sign test p = {binomtest(k, n, .5, alternative='greater').pvalue:.3g}), "
            f"mean extra loss {q.extra_loss.mean():+.4f} | final score lower in {k2}/{n}, mean drop {q.score_drop.mean():+.2f}")
    # step 0 = the cheap check (switch off, no training yet)
    z = cur[(cur.step == 0) & (cur.start != "fresh")].groupby(["task", "start", "cond"]).val_loss.mean().unstack("cond")
    say("\nSTEP 0 (right after the wipe, no relearning yet): held-out loss with each set wiped\n" + z.round(4).to_string())
    if fg is not None and len(fg):
        f = fg.assign(rise=fg.after - fg.before).groupby(["task", "start", "cond"]).rise.mean().unstack("cond")
        say("\nOTHER LANGUAGES: loss after relearning minus loss right after the wipe (negative = they came back too)\n" + f.round(4).to_string())
    open(f"{OUT}/report.txt", "w").write("\n".join(R))

    # ---------- figure 1: learning curves
    COL = {"none": "#222222", "self": "#4B1D6B", "lookalike": "#888888", "random": "#C8C8C8", "fresh": "#B35806"}
    NAME = {"none": "nothing off", "self": "10% self off", "lookalike": "10% look-alike task off", "random": "10% random task off",
            "fresh": "fresh random model"}
    starts = [s for s in run.start.unique() if s != "fresh"]
    tasks = [t for t in TASK_ORDER if t in run.task.unique()]
    for metric, ylab, fname in [("val_loss", "held-out loss (lower = better)", "fig1_learning_curves_loss.png"),
                                ("score", "score (accuracy %, chrF for Indonesian)", "fig2_learning_curves_score.png")]:
        fig, axs = plt.subplots(len(starts), len(tasks), figsize=(4 * len(tasks), 3.3 * len(starts)), squeeze=False)
        for i, st in enumerate(starts):
            for j, task in enumerate(tasks):
                ax = axs[i, j]
                for cond in ["fresh", "none", "random", "lookalike", "self"]:
                    q = cur[(cur.task == task) & (cur.cond == cond) & (cur.start == ("fresh" if cond == "fresh" else st))].dropna(subset=[metric])
                    if not len(q):
                        continue
                    m = q.groupby("step")[metric].agg(["mean", "min", "max"])
                    ax.plot(m.index, m["mean"], color=COL[cond], lw=2 if cond == "self" else 1.4, marker="o", ms=3, label=NAME[cond])
                    ax.fill_between(m.index, m["min"], m["max"], color=COL[cond], alpha=.12)
                ax.set_title(f"{st} model | {TASK_LABEL[task].replace(chr(10), ' ')}", fontsize=9); ax.grid(alpha=.25)
                ax.set_xlabel("training step")
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8.5)
                if metric == "val_loss":
                    q = cur[(cur.task == task) & (cur.start == st) & (cur.step >= 200)].val_loss
                    if len(q):
                        ax.set_ylim(q.min() * .95, q.max() * 1.15)
        axs[0, 0].legend(fontsize=7.5, frameon=False)
        fig.suptitle("Relearning after a 30% wipe. Line = mean over seeds, band = lowest to highest seed", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=170); plt.close(fig)

    # ---------- figure 3: near -> far
    if len(C):
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        MK = {s: m for s, m in zip(starts, ["*", "D", "o", "s"])}
        for k, st in enumerate(starts):
            for j, task in enumerate(tasks):
                q = C[(C.start == st) & (C.task == task)]
                if not len(q):
                    continue
                x = j + (k - (len(starts) - 1) / 2) * .18
                ax.scatter([x] * len(q), q.extra_loss, color="#4B1D6B", alpha=.35, s=18, marker=MK[st])
                ax.scatter(x, q.extra_loss.mean(), color="#4B1D6B", s=130 if MK[st] == "*" else 70, marker=MK[st],
                           label=f"{st} start model" if j == 0 else None)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=8.5)
        ax.set_xlabel("near translation  ->  far from translation")
        ax.set_ylabel("extra learning-curve loss\n(self off - look-alike task off)")
        ax.set_title("Above 0 = after wiping part of the self, the model is worse while relearning\n"
                     "big marker = mean over seeds; small = one seed", fontsize=9.5)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25, axis="y")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig3_near_to_far.png", dpi=170); plt.close(fig)
    print("\nfigures + report in", OUT)

if args.summarize:
    summarize(); sys.exit()

# =============================================================================================== TRAINING (GPU)
from common_v8 import *
import math
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

# ---------------- tasks
NEXT = {}
def load_task(task):
    if task in ("next", "same"):
        l = NEXT["lang"]; rows = []
        with open(P(f"data/en-{l}/train.jsonl"), encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= 300_000:
                    break
        S = sp.encode([r["en"] for r in rows]); T = sp.encode([r["tgt"] for r in rows])
        tr = [(s_[:MAX_TOK - 1] + [EOS], [LANG_TAG[l]] + t[:MAX_TOK - 1], t[:MAX_TOK - 1] + [EOS]) for s_, t in zip(S, T)]
        test = [{"src": a, "tgt": b} for a, b in zip(read_flores("devtest", "en")[:200], read_flores("devtest", l)[:200])]
        return {"train": tr, "val": load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:1000], "test": test}
    def rd(split):
        return [json.loads(l) for l in open(P(f"data/newtask/{task}/{split}.jsonl"), encoding="utf-8")]
    def tok(rows):
        S = sp.encode([r["src"] for r in rows]); T = sp.encode([r["tgt"] for r in rows])
        return [(s[:MAX_TOK - 1] + [EOS], [BOS] + t[:MAX_TOK - 1], t[:MAX_TOK - 1] + [EOS]) for s, t in zip(S, T)]
    tr = rd("train")[:300_000]
    return {"train": tok(tr), "val": tok(rd("val")), "test": rd("test")}

@torch.no_grad()
def greedy(model, srcs, max_new, bs=128):
    model.eval(); out = []
    with amp():
        for i in range(0, len(srcs), bs):
            src = pad_batch([sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in srcs[i:i + bs]])
            mem, sm = model.encode(src)
            ys = torch.full((src.size(0), 1), BOS, device=DEVICE); done = torch.zeros(src.size(0), dtype=torch.bool, device=DEVICE)
            for _ in range(max_new):
                nxt = model.decode(ys, mem, sm)[:, -1].float().argmax(-1); nxt[done] = PAD
                ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == EOS
                if done.all():
                    break
            for row in ys[:, 1:].tolist():
                row = row[:row.index(EOS)] if EOS in row else row
                out.append(sp.decode([t for t in row if t != PAD]))
    return out

def score(model, task, test):
    if task in ("next", "same"):
        return chrf(model, [r["src"] for r in test], [r["tgt"] for r in test], NEXT["lang"], masked=True)[0]
    if task == "indonesian":
        hyp = greedy(model, [r["src"] for r in test], MAX_TOK)
        return CHRF_METRIC.corpus_score(hyp, [[r["tgt"] for r in test]]).score
    hyp = greedy(model, [r["src"] for r in test], 12)
    return 100 * float(np.mean([h.strip() == r["tgt"] for h, r in zip(hyp, test)]))

# ---------------- one training run
def train_one(start_name, start_sd, cfg, task, data, cond, seed, lesions, ORIG):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Seq2Seq(Config(**cfg)).to(DEVICE)
    if start_sd is not None:
        model.load_state_dict(start_sd)
    g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    with torch.no_grad():                                                  # WIPE: re-draw weights as at initialisation
        for pt, ids, _ in lesions:
            m = mlp_mod(model, pt); ii = torch.as_tensor(np.asarray(ids), dtype=torch.long)
            m.fc.weight[ii] = (torch.randn(len(ii), m.fc.weight.shape[1], generator=g) * 0.02).to(m.fc.weight)
            m.fc.bias[ii] = 0
            sd_proj = 0.02 / math.sqrt(2 * (cfg["n_enc"] + cfg["n_dec"]))
            m.proj.weight[:, ii] = (torch.randn(m.proj.weight.shape[0], len(ii), generator=g) * sd_proj).to(m.proj.weight)
    hooks = []
    opt = make_optimizer(model, args.lr, 0.1)
    evals = [s for s in EVAL_AT if s <= args.steps]
    if args.steps not in evals:
        evals.append(args.steps)
    chrf_at = {0, 100, 300, args.steps}
    nl = NEXT.get("lang") if task in ("next", "same") else None
    rows, frows = [], []
    def evaluate(step):
        vl = val_loss(model, data["val"], nl, masked=nl is not None)
        sc = score(model, task, data["test"]) if (task not in ("indonesian", "next", "same") or step in chrf_at) else np.nan
        rows.append({"start": start_name, "task": task, "cond": cond, "seed": seed, "step": step, "val_loss": vl, "score": sc,
                     "n_off": sum(len(i) for _, i, _ in lesions)})
        model.train()
    before = {l: val_loss(model, ORIG[l], l, masked=True) for l in ORIG}
    evaluate(0)
    step, ep = 0, 0
    model.train()
    while step < args.steps:
        for bidx in make_batches(data["train"], args.max_tokens, seed=seed * 1000 + ep):
            src, tin, tout = collate(data["train"], bidx)
            lr = args.lr * min(1.0, (step + 1) / 100)                    # 100 warmup steps, then constant
            for gp in opt.param_groups:
                gp["lr"] = lr
            with amp():
                if nl is None:
                    _, loss = model(src, tin, tout)
                else:                                                        # same script-masked loss as normal training
                    logits, _ = model(src, tin)
                    loss = masked_ce(logits.view(-1, logits.size(-1)), tout.view(-1), ALLOWED[nl])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step in evals:
                evaluate(step)
            if step >= args.steps:
                break
        ep += 1
    after = {l: val_loss(model, ORIG[l], l, masked=True) for l in ORIG}
    for l in ORIG:
        frows.append({"start": start_name, "task": task, "cond": cond, "seed": seed, "lang": l, "before": before[l], "after": after[l]})
    for h in hooks:
        h.remove()
    del model, opt
    torch.cuda.empty_cache()
    return rows, frows

def append_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# ---------------- main
if args.start == "fresh":
    sys.exit("relearn_v9.py needs a trained start model")
if False:
    start_name, start_sd, cfg, info = "fresh", None, None, None
    conds = ["none"]
else:
    start_name, rest = args.start.split("="); run, ph = rest.split(":")
    path = glob.glob(P(f"runs/{run}/ckpt/p{int(ph):03d}_*.pt"))[0]
    m0, ck0 = load_model(path); cfg = ck0["cfg"]
    start_sd = {k: v.detach().clone() for k, v in m0.state_dict().items()}
    print(f"start model: {start_name} = {run} phase {ph} (last trained on {ck0.get('lang')}) | language set {LANGSET}: {', '.join(LANGS)}")
    t0 = time.time(); info = neuron_info(m0); del m0
    for pt in MLP_:
        d = info[pt]
        print(f"  {pt}: {d['nalive']} working neurons, self = {len(d['sid'])} ({100 * len(d['sid']) / max(d['nalive'], 1):.0f}%), "
              f"switching off {int(round(args.dose * d['nalive']))}")
    print(f"  self found in {time.time() - t0:.0f}s")
    conds = args.conds.split(",")
    if ck0.get("lang") in LANGS:
        NEXT["lang"] = ck0["lang"]
        print(f"  relearning {NEXT['lang']}: the language this checkpoint was last trained on")
if cfg is None:
    cfg = asdict(Config())
ORIG = {} if start_name == "fresh" else {l: load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:300] for l in LANGS}

done = set()
if os.path.exists(f"{OUT}/curves.csv"):
    c = pd.read_csv(f"{OUT}/curves.csv"); done = set(map(tuple, c[["start", "task", "cond", "seed"]].drop_duplicates().astype(str).values))
seeds = [int(s) for s in args.seeds.split(",")]
tasks = args.tasks.split(",")
todo = [(t, c, s) for s in seeds for t in tasks for c in conds if (start_name, t, c, str(s)) not in done]
print(f"runs to do in this call: {len(todo)} ({len(tasks)} tasks x {len(conds)} conditions x seeds {seeds}; already done are skipped)")
DATA, t_all, n_done = {}, time.time(), 0
for seed in seeds:
    sets = None
    for task in tasks:
        for cond in conds:
            if (start_name, task, cond, str(seed)) in done:
                continue
            if sets is None and info is not None:
                sets, bal = pick_sets(info, seed); append_csv(f"{OUT}/matching_check.csv", [dict(b, start=start_name) for b in bal])
            if task not in DATA:
                DATA[task] = load_task(task); print(f"  loaded {task}: {len(DATA[task]['train']):,} train examples")
            t0 = time.time()
            rows, frows = train_one(start_name, start_sd, cfg, task, DATA[task], cond, seed, sets[cond] if sets else [], ORIG)
            append_csv(f"{OUT}/curves.csv", rows); append_csv(f"{OUT}/forgetting.csv", frows)
            n_done += 1; per = (time.time() - t_all) / n_done
            r0, r1 = rows[0], rows[-1]
            sc = lambda r: "" if pd.isna(r["score"]) else f" score {r['score']:.1f}"
            print(f"  {start_name} | {task:10s} | {cond:9s} | seed {seed} | loss {r0['val_loss']:.3f} -> {r1['val_loss']:.3f} |"
                  f"{sc(r0)} ->{sc(r1)} | {time.time() - t0:.0f}s | left in this call: {len(todo) - n_done} runs, about {per * (len(todo) - n_done) / 60:.0f} min",
                  flush=True)
print("done")
