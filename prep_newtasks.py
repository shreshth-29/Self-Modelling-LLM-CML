"""prep_newtasks.py: data for the v9 "learn something new" experiment. Run once, from the repo root. Skips what exists.

Writes plain text (no tokenizer needed), one json per line: {"src": ..., "tgt": ...}
  data/newtask/indonesian/  English -> Indonesian (Helsinki-NLP/opus-100 "en-id"; test = FLORES+ devtest, 200 sentences)
  data/newtask/sentiment/   English movie review -> "positive" / "negative"           (stanfordnlp/sst2; test = its validation split)
  data/newtask/topic/       English news text -> "world" / "sports" / "business" / "science"   (fancyzhx/ag_news)
  data/newtask/addition/    "4 7 + 8 5" -> "1 3 2"   (generated; numbers with 1-4 digits; digits written with spaces)
Each task: train.jsonl, val.jsonl (1,000: loss curve), test.jsonl (score: accuracy / chrF).
"""
import os, json, random, re, unicodedata
from datasets import load_dataset

OUT = "data/newtask"
norm = lambda x: unicodedata.normalize("NFC", x or "").strip()

def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, t in rows:
            f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")
    print("wrote", path, f"{len(rows):,}")

def done(task):
    return all(os.path.exists(f"{OUT}/{task}/{s}.jsonl") for s in ["train", "val", "test"])

# ---------------------------------------------------------------- Indonesian (unrelated new language, Latin script)
if not done("indonesian"):
    def clean(pairs):
        out, seen = [], set()
        for en, tgt in pairs:
            en, tgt = norm(en), norm(tgt)
            if not (3 <= len(en) <= 250 and 3 <= len(tgt) <= 250) or not (0.33 <= len(tgt) / len(en) <= 3.0):
                continue
            if not re.search("[A-Za-z]", tgt) or en == tgt or (en, tgt) in seen:
                continue
            seen.add((en, tgt)); out.append((en, tgt))
        return out
    tr = load_dataset("Helsinki-NLP/opus-100", "en-id", split="train")
    rows = clean((r["translation"]["en"], r["translation"]["id"]) for r in tr)
    random.Random(0).shuffle(rows)
    print(f"indonesian: {len(tr):,} raw -> {len(rows):,} clean")
    write(f"{OUT}/indonesian/val.jsonl", rows[:1000]); write(f"{OUT}/indonesian/train.jsonl", rows[1000:301000])
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")["text"][:200]
    idn = load_dataset("openlanguagedata/flores_plus", "ind_Latn", split="devtest")["text"][:200]
    write(f"{OUT}/indonesian/test.jsonl", list(zip(en, idn)))

# ---------------------------------------------------------------- Sentiment (SST-2)
if not done("sentiment"):
    lab = {0: "negative", 1: "positive"}
    tr = load_dataset("stanfordnlp/sst2", split="train"); te = load_dataset("stanfordnlp/sst2", split="validation")
    rows = [(norm(r["sentence"]), lab[r["label"]]) for r in tr if len(norm(r["sentence"])) >= 3]
    random.Random(0).shuffle(rows)
    write(f"{OUT}/sentiment/val.jsonl", rows[:1000]); write(f"{OUT}/sentiment/train.jsonl", rows[1000:])
    write(f"{OUT}/sentiment/test.jsonl", [(norm(r["sentence"]), lab[r["label"]]) for r in te])

# ---------------------------------------------------------------- Topic (AG News)
if not done("topic"):
    lab = {0: "world", 1: "sports", 2: "business", 3: "science"}
    tr = load_dataset("fancyzhx/ag_news", split="train"); te = load_dataset("fancyzhx/ag_news", split="test")
    cut = lambda t: norm(t)[:400]
    rows = [(cut(r["text"]), lab[r["label"]]) for r in tr]
    random.Random(0).shuffle(rows)
    test = [(cut(r["text"]), lab[r["label"]]) for r in te]; random.Random(1).shuffle(test)
    write(f"{OUT}/topic/val.jsonl", rows[:1000]); write(f"{OUT}/topic/train.jsonl", rows[1000:])
    write(f"{OUT}/topic/test.jsonl", test[:1000])

# ---------------------------------------------------------------- Addition (generated)
if not done("addition"):
    def make(n, seed):
        g = random.Random(seed); out = []
        for _ in range(n):
            a = g.randrange(0, 10 ** g.randint(1, 4))          # first pick how many digits (1-4), then the number
            b = g.randrange(0, 10 ** g.randint(1, 4))
            sp_ = lambda x: " ".join(str(x))
            out.append((f"{sp_(a)} + {sp_(b)}", sp_(a + b)))
        return out
    write(f"{OUT}/addition/train.jsonl", make(300_000, 0))
    write(f"{OUT}/addition/val.jsonl", make(1000, 1)); write(f"{OUT}/addition/test.jsonl", make(500, 2))

for t in ["indonesian", "sentiment", "topic", "addition"]:
    ex = json.loads(open(f"{OUT}/{t}/test.jsonl", encoding="utf-8").readline())
    print(f"{t:11s} example: {ex['src'][:70]!r} -> {ex['tgt'][:40]!r}")
