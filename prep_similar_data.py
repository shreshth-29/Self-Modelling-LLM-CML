"""prep_similar_data.py: training data for the similar-language runs (Hindi, Marathi, Nepali). Run once, from the repo root.

  Marathi: ai4bharat/samanantar, config "mr" (3.6M pairs; 1M random pairs used for training, 2,000 for validation)
  Nepali:  Helsinki-NLP/opus-100, config "en-ne" (about 400k training pairs; its validation split)
  Hindi:   the existing data/en-hi (unchanged)
  FLORES+: dev + devtest for Marathi (mar_Deva) and Nepali (npi_Deva)
  Tokenizer: tokenizer_v7/spm.model, trained exactly like tokenizer_v4 (unigram, 32,000 pieces, 20M characters per language,
             half from the translation data and half from Wikipedia) but on English + Hindi + Marathi + Nepali,
             with tags <2hi>, <2mr>, <2ne>.
Writes data/en-mr/{train,val}.jsonl and data/en-ne/{train,val}.jsonl as {"en", "tgt"} lines. Skips what already exists.
Cleaning = the same as the Hindi data (tokenizer_train.ipynb): NFC, 3-250 characters, length ratio 0.33-3, Devanagari target, no duplicates.
"""
import os, json, re, random, unicodedata
from datasets import load_dataset

DEVA = re.compile("[ऀ-ॿ]")
norm = lambda x: unicodedata.normalize("NFC", x or "").strip()
_seen = set()
def ok(en, tgt):
    en, tgt = norm(en), norm(tgt)
    if not en or not tgt or len(en) > 250 or len(tgt) > 250 or len(en) < 3 or len(tgt) < 3:
        return False
    if not DEVA.search(tgt) or not (0.33 <= len(tgt) / len(en) <= 3.0) or (en, tgt) in _seen:
        return False
    _seen.add((en, tgt)); return True

def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for en, tgt in rows:
            f.write(json.dumps({"en": norm(en), "tgt": norm(tgt)}, ensure_ascii=False) + "\n")
    print("wrote", path, len(rows))

# Marathi
if not os.path.exists("data/en-mr/train.jsonl"):
    ds = load_dataset("ai4bharat/samanantar", "mr", split="train")
    rows = [(r["src"], r["tgt"]) for r in ds if ok(r["src"], r["tgt"])]
    random.Random(0).shuffle(rows)
    write("data/en-mr/val.jsonl", rows[:2000]); write("data/en-mr/train.jsonl", rows[2000:1_002_000])

# Nepali
if not os.path.exists("data/en-ne/train.jsonl"):
    tr = load_dataset("Helsinki-NLP/opus-100", "en-ne", split="train")
    va = load_dataset("Helsinki-NLP/opus-100", "en-ne", split="validation")
    rows = [(r["translation"]["en"], r["translation"]["ne"]) for r in tr if ok(r["translation"]["en"], r["translation"]["ne"])]
    random.Random(0).shuffle(rows)
    write("data/en-ne/train.jsonl", rows)
    write("data/en-ne/val.jsonl", [(r["translation"]["en"], r["translation"]["ne"]) for r in va if ok(r["translation"]["en"], r["translation"]["ne"])])

# FLORES+ (gated: needs your HF login)
for l, code in {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva", "ne": "npi_Deva"}.items():
    for split in ["dev", "devtest"]:
        p = f"data/flores/{split}.{l}.jsonl"
        if not os.path.exists(p):
            ds = load_dataset("openlanguagedata/flores_plus", code, split=split)
            os.makedirs("data/flores", exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                for t in ds["text"]:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            print("wrote", p, len(ds))
# tokenizer_v7: same recipe as tokenizer_v4, languages en / hi / mr / ne
if not os.path.exists("tokenizer_v7/spm.model"):
    import sentencepiece as spm
    CHARS, half = 20_000_000, 10_000_000
    def take_until(lines, budget):
        out, total = [], 0
        for line in lines:
            if total >= budget:
                break
            out.append(line); total += len(line)
        return out
    def field(path, k):
        with open(path, encoding="utf-8") as f:
            for l in f:
                yield json.loads(l)[k]
    def wiki(config):
        seen = set()
        for art in load_dataset("wikimedia/wikipedia", config, split="train", streaming=True):
            for line in art["text"].split("\n"):
                line = norm(line)
                if 20 <= len(line) <= 1000 and line not in seen:
                    seen.add(line); yield line
    text = {"en": take_until(field("data/en-hi/train.jsonl", "en"), half) + take_until(wiki("20231101.en"), half),
            "hi": take_until(field("data/en-hi/train.jsonl", "tgt"), half) + take_until(wiki("20231101.hi"), half),
            "mr": take_until(field("data/en-mr/train.jsonl", "tgt"), half) + take_until(wiki("20231101.mr"), half),
            "ne": take_until(field("data/en-ne/train.jsonl", "tgt"), half) + take_until(wiki("20231101.ne"), half)}
    lines = []
    for l, t in text.items():
        print(f"tokenizer text {l}: {len(t):,} lines, {sum(map(len, t)):,} characters"); lines += t
    random.Random(42).shuffle(lines)
    os.makedirs("tokenizer_v7", exist_ok=True)
    with open("data/tok_train_v7.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    del lines, text
    spm.SentencePieceTrainer.train(
        input="data/tok_train_v7.txt", model_prefix="tokenizer_v7/spm", model_type="unigram",
        vocab_size=32000, character_coverage=0.9995, byte_fallback=True, normalization_rule_name="identity",
        user_defined_symbols=["<2hi>", "<2mr>", "<2ne>"], pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        input_sentence_size=4_000_000, shuffle_input_sentence=True, max_sentence_length=4096)
    sp = spm.SentencePieceProcessor(model_file="tokenizer_v7/spm.model")
    for l, s_ in [("hi", "सरकार ने किसानों के लिए नई योजना की घोषणा की।"), ("mr", "त्याच्या घराजवळ एक मोठे तळे आहे आणि तिथे पक्षी येतात."),
                  ("ne", "सरकारले किसानहरूका लागि नयाँ योजनाको घोषणा गर्‍यो।")]:
        ids = sp.encode(s_); print(f"  {l}: {len(ids) / len(s_.split()):.2f} pieces per word |", " ".join(sp.id_to_piece(t) for t in ids))
print("data ready:", {l: sum(1 for _ in open(f"data/en-{l}/train.jsonl")) for l in ["hi", "mr", "ne"] if os.path.exists(f"data/en-{l}/train.jsonl")})