"""v14_hf.py: everything that talks to the Hugging Face checkpoint repo. No torch imports.

Why this file exists: the old backup uploaded each checkpoint in a background thread that was allowed to fail silently,
and one checkpoint was lost that way. Here every upload is retried, and "is it really on HF?" is answered by listing
the repo and comparing file sizes, never by assuming an upload worked.
"""
import os, glob, time, threading

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
_API = {}

def api():
    if "api" not in _API:
        from huggingface_hub import HfApi
        _API["api"] = HfApi()
    return _API["api"]

def ckpt_repo():
    """<your HF user>/cml-lang-self-checkpoints (the repo all earlier runs are in)"""
    if "repo" not in _API:
        _API["repo"] = os.environ.get("V14_HF_REPO") or (api().whoami()["name"] + "/cml-lang-self-checkpoints")
    return _API["repo"]

def retry(fn, what, tries=5, wait=20):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"  HF: {what} failed ({type(e).__name__}: {str(e)[:150]}); try {k + 1} of {tries}", flush=True)
            time.sleep(wait * (k + 1))
    return None

def local_ckpts(run):
    """{file name: size in bytes} of the finished checkpoints of a run on this machine"""
    return {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(os.path.join(RUNS, run, "ckpt", "p*.pt")))}

def remote_ckpts(run):
    """{file name: size in bytes} of the run's checkpoints on HF. {} if the run is not there (or HF could not be reached)."""
    def work():
        out = {}
        for f in api().list_repo_tree(ckpt_repo(), path_in_repo=f"{run}/ckpt", repo_type="model"):
            size = getattr(f, "size", None)
            lfs = getattr(f, "lfs", None)
            if lfs is not None:
                size = (lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)) or size
            if size is not None and f.path.endswith(".pt"):
                out[os.path.basename(f.path)] = int(size)
        return out
    try:
        return work()
    except Exception as e:
        if type(e).__name__ not in ("EntryNotFoundError", "RemoteEntryNotFoundError"):
            print(f"  HF: could not list {run}/ckpt ({type(e).__name__}); treated as not uploaded", flush=True)
        return {}

def upload(path, tries=5):
    """upload one file under runs/ to the same relative path in the HF repo. True if it worked."""
    rel = os.path.relpath(path, RUNS)
    ok = retry(lambda: api().upload_file(path_or_fileobj=path, repo_id=ckpt_repo(), repo_type="model", path_in_repo=rel) or True,
               f"upload of {rel}", tries=tries)
    return bool(ok)

def small_files(run):
    return [f for f in [os.path.join(RUNS, run, n) for n in ("phases.jsonl", "log.jsonl", "config.json")] if os.path.exists(f)]

def verify(run, fix=True):
    """Compare this machine's checkpoints of `run` with HF (name and size). fix=True uploads whatever is missing or has
    the wrong size, then checks again. Returns (number verified on HF, number on this machine, list of missing names)."""
    loc = local_ckpts(run)
    rem = remote_ckpts(run)
    missing = [n for n, s in loc.items() if rem.get(n) != s]
    if missing and fix:
        for n in missing:
            upload(os.path.join(RUNS, run, "ckpt", n))
        for f in small_files(run):
            upload(f, tries=2)
        rem = remote_ckpts(run)
        missing = [n for n, s in loc.items() if rem.get(n) != s]
    return len(loc) - len(missing), len(loc), missing

def phases_on_hf(run):
    """sorted phase numbers of the run's checkpoints on HF"""
    return sorted(int(n[1:4]) for n in remote_ckpts(run))

def download(run, phases):
    """make sure these checkpoints of `run` are on this machine (downloads the missing ones). Returns the phases that are here."""
    from huggingface_hub import hf_hub_download
    have = lambda ph: bool(glob.glob(os.path.join(RUNS, run, "ckpt", f"p{ph:03d}_*.pt")))
    need = [ph for ph in phases if not have(ph)]
    if need:
        names = {int(n[1:4]): n for n in remote_ckpts(run)}
        for ph in need:
            if ph in names:
                retry(lambda: hf_hub_download(ckpt_repo(), f"{run}/ckpt/{names[ph]}", local_dir=RUNS), f"download of {run} phase {ph}", tries=3)
    return [ph for ph in phases if have(ph)]

def restore_latest(run):
    """A run that was interrupted on another machine: fetch its newest checkpoint and logs from HF so training resumes
    after that phase. Does nothing if the run already has checkpoints here, or has none on HF. Returns the phase or None."""
    from huggingface_hub import hf_hub_download
    if local_ckpts(run):
        return None
    rem = remote_ckpts(run)
    if not rem:
        return None
    last = max(rem, key=lambda n: int(n[1:4]))
    if retry(lambda: hf_hub_download(ckpt_repo(), f"{run}/ckpt/{last}", local_dir=RUNS), f"download of {run}/{last}", tries=3) is None:
        return None
    for n in ("phases.jsonl", "log.jsonl"):
        try:
            hf_hub_download(ckpt_repo(), f"{run}/{n}", local_dir=RUNS)
        except Exception:
            pass
    return int(last[1:4])

class Uploader(threading.Thread):
    """Watches runs/<run>/ckpt/ while training runs and uploads every new checkpoint, with retries.
    Training never waits for it. stop_and_finish() uploads what is left, so nothing depends on a thread dying quietly."""
    def __init__(self, run, every=60):
        super().__init__(daemon=True)
        self.run_name, self.every = run, every
        self.halt = threading.Event()
        rem = remote_ckpts(run)
        self.done = {n for n, s in local_ckpts(run).items() if rem.get(n) == s}      # already safely on HF (a resumed run)
    def sweep(self):
        new = False
        for n in local_ckpts(self.run_name):
            if n not in self.done and upload(os.path.join(RUNS, self.run_name, "ckpt", n)):
                self.done.add(n); new = True
                print(f"  HF: uploaded {self.run_name}/ckpt/{n}", flush=True)
        if new:
            for f in small_files(self.run_name):
                upload(f, tries=2)
    def run(self):
        while not self.halt.is_set():
            try:
                self.sweep()
            except Exception as e:
                print(f"  HF: uploader error ({type(e).__name__}: {e}); will try again", flush=True)
            self.halt.wait(self.every)
    def stop_and_finish(self, wait_s=3600):
        """stop watching, wait for an upload that is in progress (at most wait_s), then verify everything against HF"""
        self.halt.set()
        self.join(timeout=wait_s)
        return verify(self.run_name, fix=True)

def selftest(path):
    """Upload one real checkpoint to a throwaway folder, check its size on HF, delete the folder. Raises if anything is off."""
    run = "_v14_upload_test"
    dst = os.path.join(RUNS, run, "ckpt", os.path.basename(path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import shutil
    shutil.copyfile(path, dst)
    try:
        ok, n, missing = verify(run, fix=True)
        if ok != 1 or missing:
            raise RuntimeError(f"upload test failed: {ok}/{n} verified, missing {missing}")
        print(f"  HF upload test OK: {os.path.getsize(dst) / 1e6:.0f} MB uploaded to {ckpt_repo()} and size confirmed", flush=True)
    finally:
        shutil.rmtree(os.path.join(RUNS, run), ignore_errors=True)
        try:
            api().delete_folder(run, repo_id=ckpt_repo(), repo_type="model", commit_message="remove v14 upload test")
        except Exception as e:
            print(f"  (could not delete the test folder {run} on HF: {type(e).__name__}; harmless, delete it by hand)", flush=True)
