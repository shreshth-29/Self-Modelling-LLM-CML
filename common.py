"""common.py: shared code for the three notebooks (training on GPU 0, controls on GPU 1, analysis).

Contains: tokenizer, model, script masks, data loading, evaluation, and the phase-based
training engine (warmup -> stable -> cooldown on plateau). Import with:  from common import *
Keeping this in one file guarantees all three notebooks use exactly the same model and code.
"""
import os, json, math, glob, random, re, time, contextlib
from dataclasses import dataclass, asdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm
from sacrebleu.metrics import CHRF

ROOT = os.path.dirname(os.path.abspath(__file__))
P = lambda *parts: os.path.join(ROOT, *parts)            # path inside the repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

def amp():
    """bf16 autocast on GPU, nothing on CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else contextlib.nullcontext()

# ============================================================ tokenizer (locked, never retrained)
_SPM = P("tokenizer/spm.model") if os.path.exists(P("tokenizer/spm.model")) else P("data/spm.model")
sp = spm.SentencePieceProcessor(model_file=_SPM)
PAD, UNK, BOS, EOS = 0, 1, 2, 3
LANGS = ["hi", "zh", "yo"]
LANG_TAG = {l: sp.piece_to_id(f"<2{l}>") for l in LANGS}
MAX_TOK = 128
V = sp.get_piece_size()

# ============================================================ model
@dataclass
class Config:
    vocab_size: int = V
    d_model: int = 512
    n_head: int = 8
    d_mlp: int = 2048
    n_enc: int = 3
    n_dec: int = 3
    max_len: int = 256
    dropout: float = 0.1
    label_smoothing: float = 0.1

class Attention(nn.Module):
    """Self-attention if mem is None, cross-attention otherwise."""
    def __init__(self, c):
        super().__init__()
        self.h, self.p = c.n_head, c.dropout
        self.q = nn.Linear(c.d_model, c.d_model)
        self.kv = nn.Linear(c.d_model, 2 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
    def forward(self, x, mem=None, mask=None, causal=False):
        B, T, C = x.shape
        src = x if mem is None else mem
        S = src.size(1)
        q = self.q(x).view(B, T, self.h, C // self.h).transpose(1, 2)
        k, v = self.kv(src).split(C, dim=2)
        k = k.view(B, S, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, S, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c.d_model, c.d_mlp)
        self.act = nn.GELU()                          # <- the "neurons" we analyse
        self.proj = nn.Linear(c.d_mlp, c.d_model)
        self.drop = nn.Dropout(c.dropout)
    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))

class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.d_model), Attention(c)
        self.ln2, self.mlp = nn.LayerNorm(c.d_model), MLP(c)
        self.drop = nn.Dropout(c.dropout)
    def forward(self, x, src_mask):
        x = x + self.drop(self.attn(self.ln1(x), mask=src_mask))
        return x + self.mlp(self.ln2(x))

class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.self_attn = nn.LayerNorm(c.d_model), Attention(c)
        self.ln2, self.cross_attn = nn.LayerNorm(c.d_model), Attention(c)
        self.ln3, self.mlp = nn.LayerNorm(c.d_model), MLP(c)
        self.drop = nn.Dropout(c.dropout)
    def forward(self, x, mem, src_mask):
        x = x + self.drop(self.self_attn(self.ln1(x), causal=True))
        x = x + self.drop(self.cross_attn(self.ln2(x), mem=mem, mask=src_mask))
        return x + self.mlp(self.ln3(x))

class Seq2Seq(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.enc_pos = nn.Embedding(c.max_len, c.d_model)
        self.dec_pos = nn.Embedding(c.max_len, c.d_model)
        self.enc_blocks = nn.ModuleList([EncoderBlock(c) for _ in range(c.n_enc)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(c) for _ in range(c.n_dec)])
        self.enc_ln, self.dec_ln = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                # one shared table
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * (c.n_enc + c.n_dec)))
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def encode(self, src):
        S = src.size(1)
        src_mask = (src != PAD)[:, None, None, :]
        x = self.drop(self.tok_emb(src) + self.enc_pos(torch.arange(S, device=src.device)))
        for b in self.enc_blocks:
            x = b(x, src_mask)
        return self.enc_ln(x), src_mask
    def decode(self, tgt_in, mem, src_mask):
        T = tgt_in.size(1)
        x = self.drop(self.tok_emb(tgt_in) + self.dec_pos(torch.arange(T, device=tgt_in.device)))
        for b in self.dec_blocks:
            x = b(x, mem, src_mask)
        return self.lm_head(self.dec_ln(x))
    def forward(self, src, tgt_in, tgt_out=None):
        mem, src_mask = self.encode(src)
        logits = self.decode(tgt_in, mem, src_mask)
        loss = None
        if tgt_out is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1),
                                   ignore_index=PAD, label_smoothing=self.cfg.label_smoothing)
        return logits, loss

def load_model(path):
    """Load any saved checkpoint into a fresh model (architecture taken from the checkpoint)."""
    ck = torch.load(path, map_location=DEVICE)
    model = Seq2Seq(Config(**ck["cfg"])).to(DEVICE)
    model.load_state_dict(ck["model"])
    return model, ck

# ============================================================ script masks (task-ID output restriction)
def piece_class(i):
    """Which script a vocabulary piece belongs to. 'common' = usable by every language."""
    if sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i):
        return "common"
    p = sp.id_to_piece(i)
    if p.startswith("<2"):
        return "common"
    if re.search(r"[\u0900-\u097F]", p):
        return "hi"
    if re.search(r"[\u3000-\u303F\u4E00-\u9FFF\uFF00-\uFFEF]", p):
        return "zh"
    if re.search(r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]", p):
        return "latin"
    return "common"                                   # punctuation, digits, spaces

CLS = [piece_class(i) for i in range(V)]
SCRIPT = {"hi": "hi", "zh": "zh", "yo": "latin"}
ALLOWED = {l: torch.tensor([c in ("common", SCRIPT[l]) for c in CLS], device=DEVICE) for l in LANGS}
NEG = {l: torch.zeros(V, device=DEVICE).masked_fill(~ALLOWED[l], float("-inf")) for l in LANGS}

def masked_ce(logits, tgt, allowed, eps=0.1):
    """Cross-entropy where only `allowed` tokens can be predicted.
    Blocked tokens get probability 0 and NO gradient (they are not pushed down).
    Label smoothing is spread over allowed tokens only.
    Positions whose correct token is outside the allowed set are skipped."""
    lp = F.log_softmax(logits.float().masked_fill(~allowed, float("-inf")), dim=-1)
    keep = (tgt != PAD) & allowed[tgt]
    lp, tgt = lp[keep], tgt[keep]
    nll = -lp.gather(1, tgt[:, None]).squeeze(1)
    smooth = -lp.masked_fill(~allowed, 0.0).sum(1) / allowed.sum()
    return ((1 - eps) * nll + eps * smooth).mean()

# ============================================================ data
def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs], device=DEVICE)

def load_pairs(path, lang):
    """jsonl of {"en","tgt"} -> list of (src, tgt_in, tgt_out) token-id lists.
    src = English + </s> ; tgt_in = <2xx> + target ; tgt_out = target + </s>"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    src_ids = sp.encode([r["en"] for r in rows])
    tgt_ids = sp.encode([r["tgt"] for r in rows])
    data = []
    for s, t in zip(src_ids, tgt_ids):
        s, t = s[:MAX_TOK - 1] + [EOS], t[:MAX_TOK - 1]
        data.append((s, [LANG_TAG[lang]] + t, t + [EOS]))
    return data

def make_batches(data, max_tokens, seed=0, shuffle=True):
    """Batches of similar-length examples, at most max_tokens (with padding) per side."""
    rng = random.Random(seed)
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    length = lambda i: max(len(data[i][0]), len(data[i][1]))
    batches = []
    for c in range(0, len(idx), 100_000):
        cur, cur_max = [], 0
        for i in sorted(idx[c:c + 100_000], key=length):
            L = length(i)
            if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
                batches.append(cur); cur, cur_max = [], 0
            cur.append(i); cur_max = max(cur_max, L)
        if cur:
            batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches

def collate(data, bidx):
    return (pad_batch([data[i][0] for i in bidx]),
            pad_batch([data[i][1] for i in bidx]),
            pad_batch([data[i][2] for i in bidx]))

def read_flores(split, lang):
    with open(P(f"data/flores/{split}.{lang}.jsonl"), encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]

def load_data(train_langs=LANGS, n_val=2000, n_chrf=200):
    """Tokenized train data for `train_langs`; val + FLORES devtest subset for ALL languages
    (needed to measure forgetting)."""
    t0 = time.time()
    D = {"train": {}, "val": {}, "flores_ref": {}}
    for l in LANGS:
        if l in train_langs:
            D["train"][l] = load_pairs(P(f"data/en-{l}/train.jsonl"), l)
        D["val"][l] = load_pairs(P(f"data/en-{l}/val.jsonl"), l)[:n_val]
        D["flores_ref"][l] = read_flores("devtest", l)[:n_chrf]
        n_tr = f"{len(D['train'][l]):,} train / " if l in D["train"] else ""
        print(f"  {l}: {n_tr}{len(D['val'][l]):,} val")
    D["flores_src"] = read_flores("devtest", "en")[:n_chrf]
    print(f"data loaded in {time.time() - t0:.0f}s")
    return D

# ============================================================ evaluation
CHRF_METRIC = CHRF()

@torch.no_grad()
def val_loss(model, data, lang, max_tokens=16000, masked=False):
    """Mean cross-entropy per target token (no label smoothing).
    masked=True: only the target language's script may be predicted."""
    model.eval()
    total, count = 0.0, 0
    for bidx in make_batches(data, max_tokens, shuffle=False):
        src, tgt_in, tgt_out = collate(data, bidx)
        with amp():
            logits, _ = model(src, tgt_in)
        logits = logits.float().view(-1, logits.size(-1))
        tgt = tgt_out.view(-1)
        keep = tgt != PAD
        if masked:
            logits = logits.masked_fill(~ALLOWED[lang], float("-inf"))
            keep &= ALLOWED[lang][tgt]
        total += F.cross_entropy(logits[keep], tgt[keep], reduction="sum").item()
        count += int(keep.sum())
    return total / max(count, 1)

@torch.no_grad()
def translate(model, sents, lang, masked=False, bs=64):
    """Greedy decoding. masked=True: only the target language's script can be generated."""
    model.eval()
    out = []
    with amp():
        for i in range(0, len(sents), bs):
            src = pad_batch([sp.encode(s)[:MAX_TOK - 1] + [EOS] for s in sents[i:i + bs]])
            mem, src_mask = model.encode(src)
            ys = torch.full((src.size(0), 1), LANG_TAG[lang], device=DEVICE)
            done = torch.zeros(src.size(0), dtype=torch.bool, device=DEVICE)
            for _ in range(MAX_TOK):
                logits = model.decode(ys, mem, src_mask)[:, -1].float()
                if masked:
                    logits = logits + NEG[lang]
                nxt = logits.argmax(-1)
                nxt[done] = PAD
                ys = torch.cat([ys, nxt[:, None]], dim=1)
                done |= nxt == EOS
                if done.all():
                    break
            for row in ys[:, 1:].tolist():
                row = row[:row.index(EOS)] if EOS in row else row
                out.append(sp.decode([t for t in row if t != PAD]))
    return out

def chrf(model, src, ref, lang, masked=False):
    hyps = translate(model, src, lang, masked=masked)
    return CHRF_METRIC.corpus_score(hyps, [ref]).score, hyps

def evaluate_all(model, D, max_tokens=16000):
    """Val loss and chrF on ALL languages, both unrestricted ('full') and script-restricted ('masked').
    Learning shows on the trained language; forgetting on the others."""
    ev = {"val_full": {}, "val_masked": {}, "chrf_full": {}, "chrf_masked": {}}
    for l in LANGS:
        ev["val_full"][l] = round(val_loss(model, D["val"][l], l, max_tokens, masked=False), 4)
        ev["val_masked"][l] = round(val_loss(model, D["val"][l], l, max_tokens, masked=True), 4)
        ev["chrf_full"][l] = round(chrf(model, D["flores_src"], D["flores_ref"][l], l, masked=False)[0], 2)
        ev["chrf_masked"][l] = round(chrf(model, D["flores_src"], D["flores_ref"][l], l, masked=True)[0], 2)
    return ev

# ============================================================ training engine
@dataclass
class RunConfig:
    run_name: str
    mode: str = "continual"            # "continual", "only_hi", "only_yo", "only_zh"
    order: tuple = ("hi", "yo", "zh")  # continual: phase p trains order[p % 3]; phase 0 = loaded Hindi model
    init_ckpt: str = "runs/en-hi_s0/best.pt"
    n_phases: int = 24                 # phases after phase 0 (24 = 8 full cycles)
    seed: int = 0
    max_tokens: int = 16000
    # learning-rate schedule per phase: warmup -> stable -> cooldown (after plateau) -> switch
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 500
    cooldown_steps: int = 2000
    min_steps: int = 2000              # no plateau check before this
    max_stable_steps: int = 20000      # start cooldown by here even without a plateau
    eval_every: int = 500
    plateau_window: int = 4            # mean of last 4 evals vs the 4 before
    plateau_tol: float = 0.005         # improvement below 0.5% -> converged -> start cooldown
    weight_decay: float = 0.1          # weight matrices only (not LayerNorm / biases)
    grad_clip: float = 1.0
    masked: bool = True                # script-masked loss and decoding
    diverge_factor: float = 1.15       # val > 1.15 x phase best, twice in a row -> diverged
    max_retries: int = 2               # diverged phase restarts from its start at half the lr
    time_cap_h: float = 30.0           # safety cap; no new phase starts after this

def phase_lang(C, phase):
    return C.order[phase % len(C.order)] if C.mode == "continual" else C.mode.split("_")[1]

def make_optimizer(model, lr, weight_decay):
    params = list(model.named_parameters())
    decay = [p for n, p in params if p.dim() >= 2]          # weight matrices, embeddings
    no_decay = [p for n, p in params if p.dim() < 2]        # LayerNorm weights, biases
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.98), eps=1e-9)

def lr_at(step, C, peak, cool_start):
    """Warmup to `peak`, flat until cooldown starts, then linear decay to the floor."""
    if step < C.warmup_steps:
        return peak * (step + 1) / C.warmup_steps
    if cool_start is None:
        return peak
    floor = peak * C.min_lr / C.lr                          # same ratio if peak was halved on a retry
    return peak - (peak - floor) * min(1.0, (step - cool_start) / C.cooldown_steps)

def window_improvement(hist, w):
    if len(hist) < 2 * w:
        return None
    prev, recent = sum(hist[-2 * w:-w]) / w, sum(hist[-w:]) / w
    return (prev - recent) / prev

def fmt(sec):
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

def _print_phase_table(phase, lang, info, ev, prev_ev, C):
    head = "chrf_masked" if C.masked else "chrf_full"
    vhead = "val_masked" if C.masked else "val_full"
    print("\n" + "=" * 86)
    print(f" PHASE {phase} DONE | trained: {lang} | {info}")
    print("-" * 86)
    print(f"   {'lang':<5} {'val':>7} {'Δ':>7} | {'chrF':>6} {'Δ':>6} | {'chrF (unrestricted)':>20} {'val (unrestr.)':>15}")
    for l in LANGS:
        dv = f"{ev[vhead][l] - prev_ev[vhead][l]:+.3f}" if prev_ev else ""
        dc = f"{ev[head][l] - prev_ev[head][l]:+.1f}" if prev_ev else ""
        mark = "  ◀ trained" if l == lang else ""
        print(f"   {l:<5} {ev[vhead][l]:>7.3f} {dv:>7} | {ev[head][l]:>6.1f} {dc:>6} | "
              f"{ev['chrf_full'][l]:>20.1f} {ev['val_full'][l]:>15.3f}{mark}")
    print("=" * 86 + "\n")

def train_run(C, D):
    """Run (or resume) a phase-based run. Saves one checkpoint per phase to runs/<run_name>/ckpt/."""
    RUN = P("runs", C.run_name)
    CK = os.path.join(RUN, "ckpt")
    os.makedirs(CK, exist_ok=True)
    with open(os.path.join(RUN, "config.json"), "w") as f:
        json.dump(asdict(C), f, indent=2)
    random.seed(C.seed); torch.manual_seed(C.seed)
    t_start = time.time()

    print("=" * 86)
    print(f" RUN {C.run_name} | mode {C.mode} | {C.n_phases} phases after phase 0 | device {DEVICE}")
    langs_seq = [phase_lang(C, p) for p in range(1, min(C.n_phases, 7) + 1)]
    print(f" phase order: hi (loaded) → {' → '.join(langs_seq)}{' → ...' if C.n_phases > 7 else ''}")
    print(f" each phase: fresh AdamW (wd {C.weight_decay} on matrices), warmup {C.warmup_steps} → lr {C.lr} "
          f"→ plateau (<{C.plateau_tol:.1%} over {C.plateau_window}-eval windows, after {C.min_steps:,}, "
          f"cap {C.max_stable_steps:,}) → cooldown {C.cooldown_steps:,} steps to {C.min_lr}")
    print(f" masked (script-restricted) loss & decoding: {C.masked} | divergence guard: "
          f">{C.diverge_factor}x best twice → retry at half lr (max {C.max_retries})")
    print("=" * 86)

    # ---- start from the trained Hindi model, or resume after the last saved phase
    saved = sorted(glob.glob(os.path.join(CK, "p*.pt")))
    if saved:
        model, ck = load_model(saved[-1])
        phase, prev_ev = ck["phase"] + 1, ck["eval"]
        print(f"RESUMING after phase {ck['phase']} ({ck['lang']})\n")
    else:
        model, _ = load_model(P(C.init_ckpt))
        print("phase 0: evaluating the loaded Hindi model on all languages...")
        ev = evaluate_all(model, D, C.max_tokens)
        _save_phase(CK, RUN, model, 0, "hi", {"steps": 0, "reason": "init"}, ev, t_start)
        _print_phase_table(0, "hi", "loaded model, no training", ev, None, C)
        phase, prev_ev = 1, ev

    log = open(os.path.join(RUN, "log.jsonl"), "a")
    phase_times = []
    while phase <= C.n_phases:
        if (time.time() - t_start) / 3600 > C.time_cap_h:
            print(f"time cap of {C.time_cap_h} h reached; stopping before phase {phase}")
            break
        lang = phase_lang(C, phase)
        data = D["train"][lang]
        print(f"▶ PHASE {phase}/{C.n_phases} | training {lang} ({len(data):,} pairs)")
        t_phase = time.time()
        start_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        peak = C.lr

        for attempt in range(C.max_retries + 1):
            opt = make_optimizer(model, peak, C.weight_decay)      # the switch: fresh optimizer
            model.train()
            step, epoch, hist, bad = 0, 0, [], 0
            cool_start, cool_reason, reason = None, None, None
            loss_sum, n_loss, tok_count, t_last = 0.0, 0, 0, time.time()

            while reason is None:
                seed = C.seed * 1_000_003 + phase * 1000 + attempt * 100 + epoch
                for bidx in make_batches(data, C.max_tokens, seed=seed):
                    lr = lr_at(step, C, peak, cool_start)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    src, tgt_in, tgt_out = collate(data, bidx)
                    with amp():
                        logits, _ = model(src, tgt_in)
                    if C.masked:
                        loss = masked_ce(logits, tgt_out, ALLOWED[lang], model.cfg.label_smoothing)
                    else:
                        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt_out.view(-1),
                                               ignore_index=PAD, label_smoothing=model.cfg.label_smoothing)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
                    opt.step()
                    step += 1
                    loss_sum += loss.item(); n_loss += 1
                    tok_count += int((tgt_out != PAD).sum())

                    if cool_start is not None and step >= cool_start + C.cooldown_steps:
                        reason = cool_reason                         # cooldown finished -> phase done
                        break

                    if step % C.eval_every == 0:
                        vl = val_loss(model, D["val"][lang], lang, C.max_tokens, masked=C.masked)
                        model.train()
                        best_before = min(hist) if hist else float("inf")
                        hist.append(vl)
                        imp = window_improvement(hist, C.plateau_window)
                        bad = bad + 1 if (step > C.warmup_steps and vl > best_before * C.diverge_factor) else 0
                        if bad >= 2:
                            reason = "diverged"
                        elif cool_start is None:
                            if step >= C.min_steps and imp is not None and imp < C.plateau_tol:
                                cool_start = step
                                cool_reason = "plateau" if imp >= -C.plateau_tol else "worsening"
                            elif step >= C.max_stable_steps:
                                cool_start, cool_reason = step, "capped"

                        now = time.time()
                        stage = ("warmup" if step < C.warmup_steps else
                                 "stable" if cool_start is None else
                                 f"cooldown ({cool_reason}, ends at {cool_start + C.cooldown_steps:,})")
                        plat = "n/a" if imp is None else f"{imp:+.2%}"
                        done_ph = len(phase_times)
                        eta = (f"run ETA ~{fmt(sum(phase_times) / done_ph * (C.n_phases - phase + 1))}"
                               if done_ph else "run ETA: after first phase")
                        print(f"  [p{phase:02d} {lang}{' retry' + str(attempt) if attempt else ''}] "
                              f"step {step:>6,} | lr {lr:.1e} | train {loss_sum / n_loss:.3f} | "
                              f"val {vl:.3f} (best {min(hist):.3f}) | window Δ {plat} | {stage} | "
                              f"{tok_count / (now - t_last) / 1000:.0f}k tok/s | phase {fmt(now - t_phase)} | {eta}")
                        log.write(json.dumps({"phase": phase, "lang": lang, "attempt": attempt, "step": step,
                                              "lr": lr, "train_loss": loss_sum / n_loss, "val_loss": vl,
                                              "window_improvement": imp, "stage": stage.split(" ")[0],
                                              "minutes": (now - t_start) / 60}) + "\n")
                        log.flush()
                        loss_sum, n_loss, tok_count, t_last = 0.0, 0, 0, now
                        if reason:
                            break
                epoch += 1

            if reason != "diverged":
                break
            print(f"  ⚠ phase {phase} DIVERGED (val {hist[-1]:.3f} vs best {min(hist):.3f}); "
                  f"restoring the phase's starting weights")
            model.load_state_dict(start_state)
            if attempt < C.max_retries:
                peak /= 2
                print(f"  ↻ retrying phase {phase} with peak lr {peak:.1e}")
            else:
                reason = "failed"
                print(f"  ✗ phase {phase} failed after {C.max_retries} retries; keeping its starting weights")

        print(f"  phase {phase} finished ({reason}); evaluating all languages...")
        ev = evaluate_all(model, D, C.max_tokens)
        phase_sec = time.time() - t_phase
        phase_times.append(phase_sec)
        info = {"steps": step, "reason": reason, "cooldown_start": cool_start, "peak_lr": peak,
                "attempts": attempt + 1, "phase_min": round(phase_sec / 60, 1)}
        _save_phase(CK, RUN, model, phase, lang, info, ev, t_start)
        _print_phase_table(phase, lang, f"{step:,} steps | stop: {reason} | peak lr {peak:.1e} | "
                                        f"took {fmt(phase_sec)} | run time {fmt(time.time() - t_start)}",
                           ev, prev_ev, C)
        prev_ev, phase = ev, phase + 1
        del start_state

    log.close()
    print_run_summary(C.run_name)
    return model

def _save_phase(CK, RUN, model, phase, lang, info, ev, t_start):
    path = os.path.join(CK, f"p{phase:03d}_{lang}.pt")
    torch.save({"model": model.state_dict(), "cfg": asdict(model.cfg), "phase": phase, "lang": lang,
                "info": info, "eval": ev}, path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(os.path.join(RUN, "phases.jsonl"), "a") as f:
        f.write(json.dumps({"phase": phase, "lang": lang, **info, **ev,
                            "total_min": round((time.time() - t_start) / 60, 1)}) + "\n")

def print_run_summary(run_name):
    path = P("runs", run_name, "phases.jsonl")
    rows = {}
    for l in open(path):
        r = json.loads(l)
        rows[r["phase"]] = r                                  # keep the latest row per phase
    rows = [rows[k] for k in sorted(rows)]
    print("=" * 86)
    print(f" SUMMARY {run_name}: {len(rows)} phases (incl. phase 0)")
    print("-" * 86)
    print(f" {'ph':>3} {'lang':>4} {'steps':>7} {'stop':>9} | " + " ".join(f"chrF {l}" for l in LANGS)
          + " | " + " ".join(f"val {l}" for l in LANGS) + "   (script-restricted)")
    for r in rows:
        print(f" {r['phase']:>3} {r['lang']:>4} {r['steps']:>7,} {r['reason']:>9} | "
              + " ".join(f"{r['chrf_masked'][l]:>7.1f}" for l in LANGS) + " | "
              + " ".join(f"{r['val_masked'][l]:>6.3f}" for l in LANGS))
    print("=" * 86)
