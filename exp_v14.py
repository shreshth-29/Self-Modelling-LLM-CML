"""exp_v14.py: the multi-seed pilot from start to finish. One command (see run_v14.sh).

Stages, in this order (--stage all, or a comma list):
  prep     checks the machine, the logins, the data and the tokenizers; downloads what is missing
  test     a tiny version of EVERYTHING (training, upload to HF, every lesion test, the summary). Stops here if anything fails.
  train    every (kind of run, seed) of v14_config.py, as many at once as the GPUs and the memory allow; failed runs are retried
  verify   lists HF and compares every checkpoint by size; uploads what is missing
  analyze  self size, persistence and the count-matched lesion tests (core_v11.py) on every model, incl. the old seed-0 runs
  summary  one report + figures across seeds (summary_v14.py)
  push     commit + push (done after verify, after analyze and after summary)

What is run is decided in v14_config.py. This file only schedules it. Everything can be re-started: finished work is skipped.
Several machines: give each one  --shard 0/3, 1/3, 2/3  (the runs are split between them; shard 0 also analyses the old runs).
"""
import os, sys, json, glob, time, shutil, subprocess, argparse, collections
import v14_config as CFG

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all")
ap.add_argument("--seeds", default=",".join(str(s) for s in CFG.SEEDS))
ap.add_argument("--families", default=",".join(CFG.FAMILIES))
ap.add_argument("--n_phases", type=int, default=CFG.N_PHASES)
ap.add_argument("--old", type=int, default=1, help="1 = also analyse the old seed-0 runs from HF")
ap.add_argument("--shard", default="0/1", help="k/n: this machine takes every n-th new run, starting at k")
args = ap.parse_args()

REPO = CFG.REPO; os.chdir(REPO); sys.path.insert(0, REPO)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
PY = sys.executable
LOGS = os.path.join(REPO, "logs"); os.makedirs(LOGS, exist_ok=True)
ALL = ["prep", "test", "train", "verify", "analyze", "summary", "push"]
STAGES = ALL if args.stage == "all" else [s for s in args.stage.split(",") if s]
assert all(s in ALL for s in STAGES), f"stages are: {ALL}"
SEEDS = [int(s) for s in args.seeds.split(",")]
FAMS = [f for f in args.families.split(",") if f]
assert all(f in CFG.FAMILIES for f in FAMS), f"families are: {list(CFG.FAMILIES)}"
SHARD_K, SHARD_N = [int(x) for x in args.shard.split("/")]
N = args.n_phases
_all_new = CFG.models(SEEDS, FAMS, include_old=False)
NEW = [m for i, m in enumerate(_all_new) if i % SHARD_N == SHARD_K]
OLD = [m for m in CFG.models(SEEDS, FAMS, include_old=True) if not m["new"]] if (args.old and SHARD_K == 0) else []
STATE = {"verified": None, "pushed": None, "problems": []}

def say(*a):
    print(time.strftime("%a %H:%M"), *a, flush=True)

def problem(s):
    STATE["problems"].append(s); say("PROBLEM:", s)

def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)

# =============================================================================================== the machine
def gpus():
    """[(index, memory in GB)]"""
    r = sh(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"])
    out = []
    for line in r.stdout.strip().splitlines():
        i, mb = [x.strip() for x in line.split(",")]
        out.append((int(i), float(mb) / 1024))
    return out

def ram_gb():
    """memory this container may use (the smaller of the machine's memory and the container's limit)"""
    total = float("inf")
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                total = float(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = open(p).read().strip()
            if v.isdigit():
                total = min(total, int(v) / 2 ** 30)
        except Exception:
            pass
    return total if total != float("inf") else 64.0

def cpus():
    n = os.cpu_count() or 4
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            n = min(n, max(1, int(int(q) / int(per))))
    except Exception:
        pass
    return n

# =============================================================================================== the scheduler
def run_jobs(jobs, retries=0, status=None, status_every=1800):
    """Run jobs side by side. A job = dict(name, cmd, env, log, need (GPU GB), ram (GB), done (marker file or None), ok (exit codes that count as fine),
    clean (a folder to empty before every start, or None)).
    A job starts on the GPU with the most free memory as soon as its GPU memory and ordinary memory fit. "--gpu <index>" is added to cmd.
    Jobs whose marker file exists are skipped. Returns the names of the jobs that failed."""
    G = gpus()
    if not G:
        problem("no GPU found (nvidia-smi gave nothing)"); return [j["name"] for j in jobs]
    free = {i: gb - 1.0 for i, gb in G}; biggest = max(free.values())
    ram_total = ram_gb() - 4.0; ram_free = ram_total
    max_par = max(1, cpus() - 1)
    pending = collections.deque(j for j in jobs if not (j.get("done") and os.path.exists(j["done"])))
    if len(pending) < len(jobs):
        say(f"  {len(jobs) - len(pending)} of {len(jobs)} jobs were already finished earlier: skipped")
    total, fin, running, failed, tries = len(pending), 0, [], [], collections.Counter()
    t_status = time.time()
    while pending or running:
        started = True
        while pending and len(running) < max_par and started:
            started = False
            for k, j in enumerate(pending):
                need = min(j["need"], biggest); ram = min(j.get("ram", 4.0), ram_total)
                g = next((i for i in sorted(free, key=lambda i: -free[i]) if free[i] >= need - 1e-6), None)
                if g is None or (ram > ram_free + 1e-6 and running):
                    continue
                del pending[k]
                free[g] -= need; ram_free -= ram
                if j.get("clean"):
                    shutil.rmtree(j["clean"], ignore_errors=True)
                lf = open(j["log"], "a")
                lf.write(f"\n===== {time.strftime('%a %H:%M')} | GPU {g} | {' '.join(j['cmd'])}\n"); lf.flush()
                p = subprocess.Popen(j["cmd"] + ["--gpu", str(g)], cwd=REPO, env={**os.environ, **j.get("env", {})}, stdout=lf, stderr=subprocess.STDOUT)
                running.append((p, j, g, need, ram, lf)); started = True
                say(f"  started  {j['name']}  (GPU {g})")
                break
        time.sleep(5)
        for item in list(running):
            p, j, g, need, ram, lf = item
            rc = p.poll()
            if rc is None:
                continue
            running.remove(item); free[g] += need; ram_free += ram; lf.close()
            if rc in j.get("ok", (0,)):
                fin += 1
                if j.get("done"):
                    os.makedirs(os.path.dirname(j["done"]), exist_ok=True); open(j["done"], "w").write(time.strftime("%Y-%m-%d %H:%M") + "\n")
                say(f"  finished {j['name']}  ({fin} of {total} done)")
            elif tries[j["name"]] < retries:
                tries[j["name"]] += 1; pending.append(j)
                say(f"  FAILED   {j['name']} (exit code {rc}); it will be started again (try {tries[j['name']] + 1})")
            else:
                failed.append(j["name"])
                problem(f"{j['name']} failed (exit code {rc}); see {os.path.relpath(j['log'], REPO)}. The other jobs continue.")
        if status and time.time() - t_status > status_every:
            t_status = time.time(); status()
    return failed

def tail(path, n=25):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except Exception:
        return "(no log)"

# =============================================================================================== push
def push(msg):
    sh(["git", "add", "-A"])
    sh(["git", "commit", "-q", "-m", msg])
    r = sh(["git", "pull", "--rebase", "-q"])                # other machines of the same pilot may have pushed (their files have other names)
    if r.returncode != 0:
        sh(["git", "rebase", "--abort"])
    r = sh(["git", "push", "-q"])
    STATE["pushed"] = r.returncode == 0
    say("pushed" if STATE["pushed"] else f"PUSH FAILED (results are on this machine; run: git push) {r.stderr.strip()[:200]}")

# =============================================================================================== stage: prep
def needed_files():
    langs = set()
    toks = set()
    for f in FAMS:
        langs |= set(CFG.FAMILIES[f]["order"]); toks.add("tokenizer_v8" if CFG.FAMILIES[f]["langset"] == "indic" else "tokenizer_v4")
    need = [f"{t}/spm.model" for t in sorted(toks)]
    need += [f"data/en-{l}/{s}.jsonl" for l in sorted(langs) for s in ("train", "val")]
    need += [f"data/flores/{s}.{l}.jsonl" for s in ("dev", "devtest") for l in ["en"] + sorted(langs)]
    return need, sorted(langs), sorted(toks)

FLORES_CODE = {"en": "eng_Latn", "hi": "hin_Deva", "zh": "cmn_Hans", "ar": "arb_Arab", "gu": "guj_Gujr", "pa": "pan_Guru"}

def stage_prep():
    say("STAGE prep: machine, logins, data")
    G = gpus()
    say(f"  GPUs: {len(G)} (" + ", ".join(f"{gb:.0f} GB" for _, gb in G) + f") | memory {ram_gb():.0f} GB | CPU cores {cpus()}")
    par = sum(max(1, int((gb - 1.0) // CFG.NEED_GB["train"])) for _, gb in G)
    par = max(1, min(par, int((ram_gb() - 4) // CFG.NEED_RAM["train"])))
    say(f"  training runs that fit at the same time: about {par}  (limited by GPU memory and by ordinary memory, 14 GB per run)")
    free = shutil.disk_usage(REPO).free / 2 ** 30
    want = len(NEW) * (N + 1) * 0.27 + (5 * len(OLD) if OLD else 0) + 8
    say(f"  disk free: {free:.0f} GB | needed: about {want:.0f} GB")
    if free < want and not os.environ.get("V14_SKIP_DISK"):
        say("STOP: not enough disk. Rent more disk, or set V14_SKIP_DISK=1 to go on anyway."); sys.exit(1)
    import v14_hf as HF
    try:
        say(f"  Hugging Face login OK; checkpoints go to {HF.ckpt_repo()}")
    except Exception as e:
        say(f"STOP: Hugging Face login does not work ({type(e).__name__}). Run the standard setup block first."); sys.exit(1)
    if "push" in STAGES and sh(["git", "push", "--dry-run"]).returncode != 0:
        say("STOP: git push does not work on this machine. Run the standard setup block first."); sys.exit(1)

    need, langs, toks = needed_files()
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        say(f"  {len(miss)} files missing; trying the HF data repo")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(HF.api().whoami()["name"] + "/cml-lang-self-data", repo_type="dataset", local_dir="data",
                              allow_patterns=[f"en-{l}/*" for l in langs] + ["tokenizer_v*/*", "flores/*"])
        except Exception as e:
            say(f"  (HF data repo: {type(e).__name__}: {str(e)[:120]})")
        for t in toks:                                           # the data repo keeps a copy of the tokenizers
            if not os.path.exists(f"{t}/spm.model") and os.path.exists(f"data/{t}/spm.model"):
                os.makedirs(t, exist_ok=True); shutil.copyfile(f"data/{t}/spm.model", f"{t}/spm.model")
        for l in ["en"] + langs:
            for split in ("dev", "devtest"):
                p = f"data/flores/{split}.{l}.jsonl"
                if not os.path.exists(p):
                    try:
                        from datasets import load_dataset
                        ds = load_dataset("openlanguagedata/flores_plus", FLORES_CODE[l], split=split)
                        os.makedirs("data/flores", exist_ok=True)
                        with open(p, "w", encoding="utf-8") as f:
                            for t_ in ds["text"]:
                                f.write(json.dumps({"text": t_}, ensure_ascii=False) + "\n")
                    except Exception as e:
                        say(f"  (FLORES {l} {split}: {type(e).__name__})")
        miss = [f for f in need if not os.path.exists(f)]
    if miss:
        say("STOP: these files are still missing. Run the data step of the standard setup block, then start again:")
        for f in miss:
            print("   ", f)
        sys.exit(1)
    # the tokenizers must be exactly the ones the old runs used: compare with the piece lists that are in git
    import sentencepiece as spm
    for t in toks:
        sp = spm.SentencePieceProcessor(model_file=f"{t}/spm.model")
        if os.path.exists(f"{t}/spm.vocab"):
            ref = [line.rstrip("\n").split("\t")[0] for line in open(f"{t}/spm.vocab", encoding="utf-8")]
            if ref != [sp.id_to_piece(i) for i in range(sp.get_piece_size())]:
                say(f"STOP: {t}/spm.model is NOT the tokenizer of the earlier runs (its pieces differ from {t}/spm.vocab in git)."); sys.exit(1)
    say("  data, FLORES and tokenizers are here; tokenizers match the earlier runs")

# =============================================================================================== the analysis jobs (used by test and analyze)
def here(run):
    return {int(os.path.basename(f)[1:4]) for f in glob.glob(os.path.join(REPO, "runs", run, "ckpt", "p*.pt"))}

def analysis_jobs(mods, outroot, tiny=False):
    size_ph = [0, 1, 2, 3] if tiny else CFG.SIZE_PHASES
    pers_ph = [2] if tiny else CFG.PERSIST_FROM
    les_ph = [3] if tiny else CFG.LESION_PHASES
    static_x = ["--draws", "0", "--n_val", "30", "--n_chrf", "8"] if tiny else ["--draws", CFG.STATIC_DRAWS]
    train_x = (["--seeds", "0", "--steps", "20", "--conds", "none,self@1.0,task@1.0"] if tiny
               else ["--seeds", CFG.TRAIN_SEEDS, "--steps", str(CFG.TRAIN_STEPS)])
    jobs = []
    for m in mods:
        fam = CFG.FAMILIES[m["family"]]; lab, run = m["label"], m["run"]; have = here(run)
        base = [PY, "-u", "core_v11.py", "--outroot", outroot, "--family", m["family"]] + (["--self_langs", fam["only"]] if fam["only"] else [])
        log = os.path.join(LOGS, f"{outroot}_analysis_{lab}.log")
        def add(name, out, extra, kind, marker=".done", clean=True):
            d = os.path.join(REPO, "analysis", outroot, out); done = os.path.join(d, marker)
            jobs.append(dict(name=f"{name} {lab}", cmd=base + ["--out", out] + extra, env={"CML_LANGSET": fam["langset"]}, log=log,
                             need=CFG.NEED_GB[kind], ram=CFG.NEED_RAM[kind], done=done,
                             clean=d if clean else None))               # these parts append to their table: every (re)start begins with an empty folder
        ph = [p for p in size_ph if p in have]
        if ph:
            add("self size", f"sizes/{lab}", ["--part", "sizes", "--model", f"{m['family']}={run}:" + ",".join(map(str, ph))], "light")
        for p in pers_ph:
            if p in have and p + 1 in have:
                add(f"persistence {p}->{p + 1}", f"persistence/{lab}", ["--part", "matrix", "--compact", "--model", f"{lab}={run}:{p}"], "light",
                    marker=f".done_p{p:03d}", clean=False)
        ph = [p for p in les_ph if p in have]
        if ph:
            add("lesion (no training)", f"static/{lab}", ["--part", "static", "--model", f"{m['family']}={run}:" + ",".join(map(str, ph))] + static_x
                + (["--eval_langs", fam["only"]] if fam["only"] else []), "light")
        for p in ph:
            add(f"lesion + train, from phase {p}", f"train_next/{lab}/p{p:03d}",
                ["--part", "train", "--task", "next", "--mode", "lesion", "--model", f"{lab}_p{p}={run}:{p}"] + train_x
                + (["--next_lang", fam["only"]] if fam["only"] else []), "retrain", clean=False)
        missing = sorted((set(size_ph) | set(pers_ph) | {p + 1 for p in pers_ph} | set(les_ph)) - have)
        if missing:
            say(f"  {lab} ({run}): checkpoints of phases {missing} are not available; the tests that need them are left out")
    return jobs

# =============================================================================================== stage: test
def stage_test():
    say("STAGE test: a tiny version of everything (about 15 to 25 minutes). Nothing real starts until this passes.")
    root = CFG.OUTROOT + "_test"
    def clean():
        for d in glob.glob("runs/v14test_*") + [f"analysis/{root}", "data/_tiny"] + glob.glob(f"logs/{root}_*") + glob.glob("logs/v14test_*"):
            shutil.rmtree(d, ignore_errors=True) if os.path.isdir(d) else (os.path.exists(d) and os.remove(d))
    clean()
    def stop(what, log=None):
        say(f"TEST FAILED at: {what}. Nothing else was run. Send these lines to Claude:")
        if log:
            print(tail(log))
        sys.exit(1)
    mods = [dict(family=f, tag="t", label=f"{f}_t", run=f"v14test_{f}", new=True) for f in FAMS]
    jobs = [dict(name=f"tiny training {m['family']}", cmd=[PY, "-u", "train_v14.py", "--family", m["family"], "--seed", "0", "--tiny", "--n_phases", "3", "--run", m["run"]],
                 log=os.path.join(LOGS, f"v14test_train_{m['family']}.log"), need=CFG.NEED_GB["train"], ram=CFG.NEED_RAM["tiny"]) for m in mods]
    bad = run_jobs(jobs)
    if bad:
        stop(bad[0], os.path.join(LOGS, f"v14test_train_{bad[0].split()[-1]}.log"))
    try:
        import v14_hf as HF
        HF.selftest(sorted(glob.glob(f"runs/{mods[0]['run']}/ckpt/p*.pt"))[0])
    except Exception as e:
        say(f"  {type(e).__name__}: {e}"); stop("upload of a checkpoint to HF")
    jobs = analysis_jobs(mods, root, tiny=True)
    if len(jobs) != 4 * len(mods):                                # per model: self size, 1 persistence, static lesion, 1 lesion + train
        stop(f"building the analysis jobs (got {len(jobs)}, expected {4 * len(mods)})")
    bad = run_jobs(jobs)
    if bad:
        stop(bad[0], os.path.join(LOGS, f"{root}_analysis_{bad[0].split()[-1]}.log"))
    r = sh([PY, "summary_v14.py", "--root", root, "--tiny"])
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); stop("the summary")
    if not os.path.exists(f"analysis/{root}/summary/per_model.csv"):
        print(r.stdout[-3000:]); stop("the summary (it made no table)")
    clean()
    say("TEST OK")

# =============================================================================================== stage: train
def train_status():
    for m in NEW:
        try:
            rows = [json.loads(l) for l in open(f"runs/{m['run']}/phases.jsonl")]
            last = max(rows, key=lambda r: r["phase"])
            say(f"    {m['run']}: phase {last['phase']} of {N} finished ({last['lang']})")
        except Exception:
            say(f"    {m['run']}: no phase finished yet")

def save_training_logs():
    os.makedirs(f"analysis/{CFG.OUTROOT}/training", exist_ok=True)
    for m in NEW:
        for n in ("phases.jsonl", "log.jsonl"):
            src = f"runs/{m['run']}/{n}"
            if os.path.exists(src):
                shutil.copyfile(src, f"analysis/{CFG.OUTROOT}/training/{m['run']}_{n}")

def stage_train():
    import v14_hf as HF
    say(f"STAGE train: {len(NEW)} runs, {N} phases each ({', '.join(m['run'] for m in NEW)})")
    jobs = []
    for m in sorted(NEW, key=lambda m: ["distant", "related", "hindi"].index(m["family"]) if m["family"] in ("distant", "related", "hindi") else 9):
        if len(here(m["run"])) >= N + 1:
            say(f"  {m['run']}: all {N + 1} checkpoints are already on this machine"); continue
        if not here(m["run"]) and len([p for p in HF.phases_on_hf(m["run"]) if p <= N]) >= N + 1:
            say(f"  {m['run']}: already complete on HF; not trained again"); continue
        jobs.append(dict(name=f"training {m['run']}", cmd=[PY, "-u", "train_v14.py", "--family", m["family"], "--seed", str(m["seed"]), "--n_phases", str(N)],
                         log=os.path.join(LOGS, f"v14_train_{m['run']}.log"), need=CFG.NEED_GB["train"],
                         ram=CFG.NEED_RAM["train_hi" if CFG.FAMILIES[m["family"]]["only"] else "train"], ok=(0, 3)))
    say("  progress of one run:  tail -f logs/v14_train_<run>.log   (a status line per run is printed here every 30 minutes)")
    run_jobs(jobs, retries=2, status=train_status)
    save_training_logs()

# =============================================================================================== stage: verify
def stage_verify():
    import v14_hf as HF
    say("STAGE verify: is every checkpoint on HF, with the right size?")
    good = True
    for m in NEW:
        ok, n, missing = HF.verify(m["run"], fix=True)
        on_hf = [p for p in HF.phases_on_hf(m["run"]) if p <= N]
        full = len(on_hf) >= N + 1 and not missing
        good &= full
        say(f"  {m['run']}: {len(on_hf)} of {N + 1} checkpoints on HF" + ("" if full else f"   <-- NOT COMPLETE (on this machine: {n}; not uploaded: {missing})"))
    STATE["verified"] = good
    save_training_logs()

# =============================================================================================== stage: analyze
def stage_analyze():
    import v14_hf as HF
    mods = NEW + OLD
    say(f"STAGE analyze: {len(mods)} models ({len(NEW)} new, {len(OLD)} old)")
    need = sorted(set(CFG.SIZE_PHASES) | set(CFG.PERSIST_FROM) | {p + 1 for p in CFG.PERSIST_FROM} | set(CFG.LESION_PHASES))
    for m in mods:
        if not set(need) <= here(m["run"]):
            got = HF.download(m["run"], need)
            say(f"  {m['run']}: {len(got)} of {len(need)} needed checkpoints are on this machine")
    jobs = analysis_jobs(mods, CFG.OUTROOT)
    say(f"  {len(jobs)} jobs. Progress per model: logs/{CFG.OUTROOT}_analysis_<model>.log")
    run_jobs(jobs, retries=1)

# =============================================================================================== stage: summary
def stage_summary():
    if SHARD_N > 1:
        say("STAGE summary: left out on a shard (the results of the other machines are not here). When all machines have pushed:  "
            "git pull && python summary_v14.py"); return
    say("STAGE summary")
    r = sh([PY, "summary_v14.py", "--root", CFG.OUTROOT])
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], flush=True); problem("the summary stopped with an error (the tables of the single tests are still in analysis/v14/)")

# =============================================================================================== main
say(f"exp_v14 | stages: {', '.join(STAGES)} | families: {', '.join(FAMS)} | seeds: {SEEDS} | phases: {N} | shard {SHARD_K}/{SHARD_N} | old runs: {len(OLD)}")
RUNNERS = {"prep": stage_prep, "test": stage_test, "train": stage_train, "verify": stage_verify, "analyze": stage_analyze, "summary": stage_summary}
for s in ALL:
    if s not in STAGES or s == "push":
        continue
    RUNNERS[s]()
    if "push" in STAGES and s in ("verify", "analyze", "summary"):
        push(f"v14 multi-seed pilot: after {s}" + (f" (shard {SHARD_K}/{SHARD_N})" if SHARD_N > 1 else ""))

say("=" * 70)
if STATE["problems"]:
    say(f"FINISHED WITH {len(STATE['problems'])} PROBLEM(S):")
    for p in STATE["problems"]:
        print("   -", p, flush=True)
else:
    say("ALL DONE, no problems.")
if "verify" in STAGES:
    say("Checkpoints on HF: " + ("ALL VERIFIED (name and size)." if STATE["verified"] else "NOT ALL VERIFIED. Do not destroy this machine; send the lines above to Claude."))
if "push" in STAGES:
    say("Results on GitHub: " + ("pushed." if STATE["pushed"] else "NOT pushed. Run: git push"))
if STATE["verified"] and STATE["pushed"] and not STATE["problems"]:
    say("The instance can be destroyed.")
