"""prep_indic_data.py: data + tokenizer for the v8 "indic" runs (Hindi, Gujarati, Punjabi). Run once, from the repo root.

  Hindi:    the existing data/en-hi (IITB, 993,016 training pairs; unchanged)
  Gujarati: ai4bharat/samanantar "gu"  -> data/en-gu
  Punjabi:  ai4bharat/samanantar "pa"  -> data/en-pa
  BALANCED: Gujarati and Punjabi get exactly as many training pairs as Hindi (2,000 validation pairs each).
            Stops with an error if a language has fewer clean pairs than that.
  Cleaning: the same as the Hindi data: NFC, 3-250 characters, length ratio 0.33-3, target in its own script, no duplicates.
  FLORES+:  dev + devtest for gu (guj_Gujr) and pa (pan_Guru) (plus en/hi if missing)
  Tokenizer: tokenizer_v8/spm.model, the same recipe as tokenizer_v4 (unigram, 32,000 pieces, 20M characters per language,
             half translation data, half Wikipedia) on English + Hindi + Gujarati + Punjabi, tags <2hi> <2gu> <2pa>.
Skips whatever already exists.
"""
import os, json, re, random, unicodedata
from datasets import load_dataset

SCRIPT_RE = {"gu": re.compile("[઀-૿]"), "pa": re.compile("[਀-੿]")}
norm = lambda x: unicodedata.normalize("NFC", x or "").strip()

def clean(pairs, lang):
    out, seen = [], set()
    for en, tgt in pairs:
        en, tgt = norm(en), norm(tgt)
        if not en or not tgt or len(en) > 250 or len(tgt) > 250 or len(en) < 3 or len(tgt) < 3:
            continue
        if not SCRIPT_RE[lang].search(tgt) or not (0.33 <= len(tgt) / len(en) <= 3.0) or (en, tgt) in seen:
            continue
        seen.add((en, tgt)); out.append((en, tgt))
    return out

def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for en, tgt in rows:
            f.write(json.dumps({"en": en, "tgt": tgt}, ensure_ascii=False) + "\n")
    print("wrote", path, f"{len(rows):,}")

N_TRAIN = sum(1 for _ in open("data/en-hi/train.jsonl", encoding="utf-8"))      # Hindi's size: everyone gets this many
print(f"Hindi training pairs: {N_TRAIN:,} -> Gujarati and Punjabi get exactly the same number")

for lang in ["gu", "pa"]:
    if os.path.exists(f"data/en-{lang}/train.jsonl"):
        continue
    ds = load_dataset("ai4bharat/samanantar", lang, split="train")
    rows = clean(((r["src"], r["tgt"]) for r in ds), lang)
    print(f"{lang}: {len(ds):,} raw -> {len(rows):,} clean pairs")
    if len(rows) < N_TRAIN + 2000:
        raise SystemExit(f"STOP: {lang} has only {len(rows):,} clean pairs, need {N_TRAIN + 2000:,} for a balanced run")
    random.Random(0).shuffle(rows)
    write(f"data/en-{lang}/val.jsonl", rows[:2000]); write(f"data/en-{lang}/train.jsonl", rows[2000:2000 + N_TRAIN])

codes = {"en": "eng_Latn", "hi": "hin_Deva", "gu": "guj_Gujr", "pa": "pan_Guru"}
os.makedirs("data/flores", exist_ok=True)
for l, code in codes.items():
    for split in ["dev", "devtest"]:
        p = f"data/flores/{split}.{l}.jsonl"
        if not os.path.exists(p):
            ds = load_dataset("openlanguagedata/flores_plus", code, split=split)
            with open(p, "w", encoding="utf-8") as f:
                for t in ds["text"]:
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            print("wrote", p, len(ds))

if not os.path.exists("tokenizer_v8/spm.model"):
    import sentencepiece as spm
    half = 10_000_000
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
            "gu": take_until(field("data/en-gu/train.jsonl", "tgt"), half) + take_until(wiki("20231101.gu"), half),
            "pa": take_until(field("data/en-pa/train.jsonl", "tgt"), half) + take_until(wiki("20231101.pa"), half)}
    lines = []
    for l, t in text.items():
        print(f"tokenizer text {l}: {len(t):,} lines, {sum(map(len, t)):,} characters"); lines += t
    random.Random(42).shuffle(lines)
    os.makedirs("tokenizer_v8", exist_ok=True)
    with open("data/tok_train_v8.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    del lines, text
    spm.SentencePieceTrainer.train(
        input="data/tok_train_v8.txt", model_prefix="tokenizer_v8/spm", model_type="unigram",
        vocab_size=32000, character_coverage=0.9995, byte_fallback=True, normalization_rule_name="identity",
        user_defined_symbols=["<2hi>", "<2gu>", "<2pa>"], pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        input_sentence_size=4_000_000, shuffle_input_sentence=True, max_sentence_length=4096)

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer_v8/spm.model")
for l in ["hi", "gu", "pa"]:
    sents = [json.loads(x)["tgt"] for _, x in zip(range(2000), open(f"data/en-{l}/val.jsonl", encoding="utf-8"))]
    w = sum(len(s.split()) for s in sents); t = sum(len(sp.encode(s)) for s in sents)
    print(f"  {l}: {t / w:.2f} pieces per word (2,000 validation sentences)")
print("data ready:", {l: sum(1 for _ in open(f"data/en-{l}/train.jsonl", encoding="utf-8")) for l in ["hi", "gu", "pa"]})
