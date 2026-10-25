"""A stepwise, branchable store for per-clip analysis of Cantonese speech (W44, 2026-10-10).

The owner's terms: keep to raw as much as possible, add to the data step by step, and be able to branch and go back.
So:

* RAW IS NEVER WRITTEN. A collection's layer 0 (`ingest`) records each clip's absolute path and sha256, and every
  later step re-hashes the file before reading it: a changed source is refused, not silently re-measured.
* EVERY STEP IS A NEW, IMMUTABLE LAYER. Its directory is keyed by a hash of the step's name, version and parameters
  and of its parents' keys. A finished layer (`COMPLETE`) is never written again, so a result can always be traced
  to exactly what made it.
* A BRANCH IS A NEW KEY, GOING BACK IS MOVING A REF. `refs.json` names the current layer of each step;
  `refs.log.jsonl` keeps every move. Changing a parameter makes a sibling layer beside the old one, and both stay.
* RECOMPUTE ONLY WHAT CHANGED. A record carries a `deps` hash (its input's sha256 plus its parent records). A new
  layer of the same recipe (step, version, params and parent recipes) reuses any record whose deps match, so adding
  draws to a collection re-measures only the new clips.

Layout under $CANTO_STORE (default ~/canto-store), never inside the repository:

    <collection>/refs.json, refs.log.jsonl
    <collection>/<step>/<key>/step.json        what made it: recipe, parents, code sha256, times, counts
    <collection>/<step>/<key>/records.jsonl    one row per clip: clip_id, deps, outputs
    <collection>/<step>/<key>/arrays/<id>.npz  per-clip arrays, when a step has them
    <collection>/<step>/<key>/summary.json     collection-level steps' output
    <collection>/<step>/<key>/COMPLETE
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import time
from pathlib import Path
from typing import Callable

ROOT = Path(os.environ.get("CANTO_STORE", Path.home() / "canto-store"))


def h(obj, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:n]


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Step:
    """A registered analysis step. `fn(clip, parents, params) -> (record, arrays | None)` for per-clip steps;
    `fn(records_by_parent, params) -> (summary, per_clip_records | None)` for collection steps."""

    def __init__(self, name: str, version: str, fn: Callable, parents: tuple[str, ...] = (),
                 params: dict | None = None, kind: str = "clip", device: str = "cpu", reads_audio: bool = True,
                 doc: str = "", code_deps: Callable | None = None):
        if kind == "clip" and "ingest" not in parents:
            parents = ("ingest",) + tuple(parents)      # a clip step's key always carries the clip set's key
        self.name, self.version, self.fn, self.parents = name, version, fn, tuple(parents)
        self.params, self.kind, self.device, self.reads_audio = dict(params or {}), kind, device, reads_audio
        self.doc = doc or (fn.__doc__ or "").strip().split("\n")[0]
        self.code_deps = code_deps

    def code_sha(self) -> str:
        """The step's source, every function of its own module it reaches by name (transitively, inner functions
        and lambdas included), and its declared dependencies (other modules' functions or constants). It is part of
        the recipe, so an edit makes a new layer instead of being silently mixed into an old one's reuse."""
        seen, parts = set(), []

        def names(code):
            out = set(code.co_names)
            for c in code.co_consts:
                if hasattr(c, "co_names"):
                    out |= names(c)
            return out

        def source(f) -> str:
            try:
                return inspect.getsource(f).replace("\r\n", "\n")       # one key on every host's line endings
            except (OSError, TypeError):                                 # no source (python -c, .pyc only)
                c = f.__code__
                return repr((c.co_code, c.co_consts, c.co_names))

        def visit(f):
            if f in seen:
                return
            seen.add(f)
            parts.append(source(f))
            for n in sorted(names(f.__code__)):
                g = f.__globals__.get(n)
                if inspect.isfunction(g) and g.__module__ == f.__module__:
                    visit(g)
                elif isinstance(g, (int, float, str, bytes, tuple, frozenset)) or (
                        isinstance(g, (dict, list)) and len(repr(g)) < 4000):
                    parts.append(f"{n}={g!r}")                           # a module constant the step reads
        visit(self.fn)
        for d in (self.code_deps() if self.code_deps else []):
            parts.append(source(d) if callable(d) else repr(d))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


REGISTRY: dict[str, Step] = {}


def register(step: Step) -> Step:
    REGISTRY[step.name] = step
    return step


class Store:
    def __init__(self, collection: str, root: Path | None = None):
        self.root = Path(root) if root else ROOT
        self.c, self.dir = collection, self.root / collection
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ refs: the branch pointers
    def refs(self) -> dict:
        p = self.dir / "refs.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def set_ref(self, step: str, key: str, why: str) -> None:
        if not (self.dir / step / key / "COMPLETE").exists():
            raise SystemExit(f"{self.c}/{step}/{key} is not a complete layer")
        with _lock(self.dir / ".refs.lock"):
            refs = self.refs()
            old = refs.get(step)
            refs[step] = key
            tmp = self.dir / f"refs.json.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(refs, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(self.dir / "refs.json")
            with open(self.dir / "refs.log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "step": step, "from": old, "to": key,
                                    "why": why}) + "\n")

    def layers(self, step: str) -> list[dict]:
        out = []
        for d in sorted((self.dir / step).glob("*")) if (self.dir / step).exists() else []:
            if (d / "step.json").exists():
                meta = json.loads((d / "step.json").read_text(encoding="utf-8"))
                out.append({**meta, "complete": (d / "COMPLETE").exists()})
        return out

    # ------------------------------------------------------------------ reading
    def layer_dir(self, step: str, key: str | None = None) -> Path:
        key = key or self.refs().get(step)
        if not key:
            raise SystemExit(f"{self.c}: no ref for {step}; run it first")
        d = self.dir / step / key
        if not (d / "COMPLETE").exists():
            raise SystemExit(f"{self.c}/{step}/{key} is not complete")
        return d

    def records(self, step: str, key: str | None = None, errors: str = "refuse") -> dict[str, dict]:
        """A layer's rows by clip. An errored row is a clip the step could not measure: by default the read REFUSES
        (a caller treating {"error": ...} as an empty measurement reads a failure as a pass, review 2026-10-10);
        errors="drop" leaves those clips out, errors="keep" returns them."""
        d = self.layer_dir(step, key)
        rows = {r["clip_id"]: r for r in _jl(d / "records.jsonl")}
        bad = [c for c, r in rows.items() if "error" in r]
        if bad and errors == "refuse":
            raise SystemExit(f"{self.c}/{step}/{d.name}: {len(bad)} errored row(s), first {bad[0]}: "
                             f"{rows[bad[0]]['error'][:120]}. Read with errors='drop' or 'keep' to accept that")
        if errors == "drop":
            rows = {c: r for c, r in rows.items() if "error" not in r}
        return rows

    def arrays(self, step: str, clip_id: str, key: str | None = None):
        import numpy as np
        p = self.layer_dir(step, key) / "arrays" / f"{_safe(clip_id)}.npz"
        return dict(np.load(p)) if p.exists() else None

    def summary(self, step: str, key: str | None = None) -> dict:
        return json.loads((self.layer_dir(step, key) / "summary.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ running a step
    def recipe(self, step: Step, params: dict) -> str:
        """The data-free identity of a step: its name, version and params, and its parents' recipes as refs name them."""
        parent_recipes = {p: self.layer_meta(p)["recipe"] for p in step.parents}
        return h({"step": step.name, "version": step.version, "params": params, "parents": parent_recipes})

    def layer_meta(self, step: str, key: str | None = None) -> dict:
        return json.loads((self.layer_dir(step, key) / "step.json").read_text(encoding="utf-8"))

    def plan(self, step: Step, params: dict | None = None, parent_keys: dict | None = None) -> dict:
        params = {**step.params, **(params or {})}
        refs = self.refs()
        pk = {p: (parent_keys or {}).get(p) or refs.get(p) for p in step.parents}
        missing = [p for p, k in pk.items() if not k]
        if missing:
            raise SystemExit(f"{self.c}/{step.name}: parent step(s) {missing} have no layer; run them first")
        for p, k in pk.items():
            self.layer_dir(p, k)
        recipe = h({"step": step.name, "version": step.version, "params": params, "code": step.code_sha(),
                    "parents": {p: self.layer_meta(p, k)["recipe"] for p, k in pk.items()}})
        key = h({"recipe": recipe, "parent_keys": pk})
        return {"params": params, "parent_keys": pk, "recipe": recipe, "key": key}

    def run(self, step: Step, params: dict | None = None, parent_keys: dict | None = None, jobs: int = 4,
            set_ref: bool = True, dry: bool = False, clips: dict | None = None, log=print) -> dict:
        pl = self.plan(step, params, parent_keys) if step.name != "ingest" else None
        if step.name == "ingest":
            raise SystemExit("use Store.ingest for layer 0")
        d = self.dir / step.name / pl["key"]
        if (d / "COMPLETE").exists():
            log(f"{self.c}/{step.name}/{pl['key']}: complete already (immutable)")
            if set_ref and self.refs().get(step.name) != pl["key"]:
                self.set_ref(step.name, pl["key"], "re-selected an existing layer")
            return {"key": pl["key"], "computed": 0, "reused": 0, "existing": True}
        base = self.records("ingest", pl["parent_keys"]["ingest"]) if "ingest" in pl["parent_keys"] else {}
        parents = {p: self.records(p, k, errors="keep") for p, k in pl["parent_keys"].items()}
        if dry:
            log(f"would run {self.c}/{step.name} -> {pl['key']} over {len(base)} clips, parents {pl['parent_keys']}")
            return {"key": pl["key"], "dry": True}
        d.mkdir(parents=True, exist_ok=True)
        meta = {"collection": self.c, "step": step.name, "version": step.version, "params": pl["params"],
                "parents": pl["parent_keys"], "recipe": pl["recipe"], "key": pl["key"], "code_sha": step.code_sha(),
                "kind": step.kind, "device": step.device, "doc": step.doc, "host": socket.gethostname(),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (d / "step.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if step.kind == "collection":
            summary, recs = step.fn(parents, pl["params"])
            (d / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            if recs:
                _write_jl(d / "records.jsonl", [{"clip_id": c, **r} for c, r in sorted(recs.items())])
            else:
                (d / "records.jsonl").write_text("", encoding="utf-8")
            res = {"key": pl["key"], "computed": len(recs or {}), "reused": 0}
        else:
            with _lock(d / ".lock"):
                res = self._run_clips(step, pl, d, base, parents, jobs, log)
        if res.get("raw_changed"):
            raise SystemExit(f"{self.c}/{step.name}/{pl['key']}: {res['raw_changed']} raw file(s) changed since ingest, "
                             f"first {res['raw_changed_first']}. The layer is left PARTIAL: re-ingest, then run again")
        meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **{k: v for k, v in res.items() if k != "key"})
        (d / "step.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (d / "COMPLETE").write_text(meta["finished"] + "\n", encoding="utf-8")
        if set_ref:
            self.set_ref(step.name, pl["key"], f"ran {step.name} {step.version} {pl['params']}")
        log(f"{self.c}/{step.name}/{pl['key']}: {res}")
        return res

    def _run_clips(self, step, pl, d, base, parents, jobs, log) -> dict:
        rec_path = d / "records.jsonl"
        if rec_path.exists():
            raw = rec_path.read_bytes()
            if raw and not raw.endswith(b"\n"):           # a crash mid-write: drop the partial last row
                rec_path.write_bytes(raw[: raw.rfind(b"\n") + 1])
                log(f"{self.c}/{step.name}/{pl['key']}: dropped a truncated last row")
        done = {r["clip_id"]: r for r in _jl(rec_path)} if rec_path.exists() else {}
        done = {c: r for c, r in done.items() if not str(r.get("error", "")).startswith("RAW CHANGED")}
        # reuse: records of a sibling layer with the same recipe and the same deps
        sib = {}
        for other in (self.dir / step.name).glob("*"):
            if other != d and (other / "COMPLETE").exists():
                m = json.loads((other / "step.json").read_text(encoding="utf-8"))
                if m.get("recipe") == pl["recipe"]:
                    for r in _jl(other / "records.jsonl"):
                        if "error" not in r:                    # an error is recomputed, never inherited
                            sib[(r["clip_id"], r["deps"])] = (other, r)
        todo, reused = [], 0
        with open(d / "records.jsonl", "a", encoding="utf-8") as out:
            for cid, clip in sorted(base.items()):
                prec = {p: parents[p].get(cid) for p in step.parents if p != "ingest"}
                if any(v is None or "error" in v for v in prec.values()):
                    continue                                   # a parent did not measure this clip
                meta = {k: v for k, v in clip.items() if k != "deps"}
                deps = h({"clip": h(meta), "parents": {p: h(v) for p, v in prec.items()}}, 16)
                if cid in done and done[cid]["deps"] == deps:
                    continue
                if (cid, deps) in sib:
                    other, r = sib[(cid, deps)]
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    src = other / "arrays" / f"{_safe(cid)}.npz"
                    if src.exists():
                        (d / "arrays").mkdir(exist_ok=True)
                        (d / "arrays" / src.name).write_bytes(src.read_bytes())
                    reused += 1
                    continue
                todo.append((cid, clip, prec, deps))
        log(f"{self.c}/{step.name}/{pl['key']}: {len(todo)} to compute, {reused} reused, {len(done)} already here")
        computed, errors, raw_changed, raw_first = 0, 0, 0, None
        ctx = {"_collection": self.c, "_parent_keys": pl["parent_keys"], "_root": str(self.root), "_layer_dir": str(d)}
        work = [(step.name, cid, {**clip, **ctx}, prec, pl["params"], step.reads_audio) for cid, clip, prec, _ in todo]
        deps_of = {cid: deps for cid, _, _, deps in todo}
        t0 = time.time()
        with open(d / "records.jsonl", "a", encoding="utf-8") as out:
            for i, (cid, rec, arrs, err) in enumerate(_map(_one, work, jobs if step.device == "cpu" else 1), 1):
                if err:
                    errors += 1
                    rec = {"error": err}
                    if err.startswith("RAW CHANGED"):
                        raw_changed += 1
                        raw_first = raw_first or cid
                if arrs:
                    import numpy as np
                    (d / "arrays").mkdir(exist_ok=True)
                    np.savez_compressed(d / "arrays" / f"{_safe(cid)}.npz", **arrs)
                out.write(json.dumps({"clip_id": cid, "deps": deps_of[cid], **rec}, ensure_ascii=False) + "\n")
                out.flush()
                computed += 1
                if i % 200 == 0:
                    log(f"  {i}/{len(work)} ({(time.time() - t0) / i:.2f} s/clip)")
        out = {"key": pl["key"], "computed": computed, "reused": reused, "errors": errors}
        if raw_changed:
            out.update(raw_changed=raw_changed, raw_changed_first=raw_first)
        return out

    # ------------------------------------------------------------------ layer 0
    def ingest(self, rows: list[dict], source: dict, set_ref: bool = True, log=print) -> dict:
        """Layer 0: each row needs clip_id and path (absolute); anything else (text, draw, speaker...) is kept as
        metadata. The audio is hashed, never copied."""
        recs = {}
        for r in rows:
            p = Path(r["path"])
            if r["clip_id"] in recs:
                raise SystemExit(f"duplicate clip_id {r['clip_id']}")
            recs[r["clip_id"]] = {**r, "path": str(p), "sha256": file_sha(p)}
        key = h({"source": source, "clips": sorted((c, h(r)) for c, r in recs.items())})
        d = self.dir / "ingest" / key
        if not (d / "COMPLETE").exists():
            d.mkdir(parents=True, exist_ok=True)
            _write_jl(d / "records.jsonl", [{"clip_id": c, "deps": r["sha256"][:16], **r} for c, r in sorted(recs.items())])
            meta = {"collection": self.c, "step": "ingest", "version": "1", "params": {"source": source},
                    "parents": {}, "recipe": h({"step": "ingest", "source": source}), "key": key,
                    "clips": len(recs), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            (d / "step.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            (d / "COMPLETE").write_text(meta["finished"] + "\n", encoding="utf-8")
        if set_ref and self.refs().get("ingest") != key:
            self.set_ref("ingest", key, f"ingest {len(recs)} clips from {source}")
        log(f"{self.c}/ingest/{key}: {len(recs)} clips")
        return {"key": key, "clips": len(recs)}


# ---------------------------------------------------------------------- workers
def _one(job):
    name, cid, clip, prec, params, reads_audio = job
    step = REGISTRY[name]
    try:
        if reads_audio:
            p = Path(clip["path"])
            if file_sha(p) != clip["sha256"]:
                return cid, None, None, f"RAW CHANGED: {p} no longer has the sha256 ingested"
        rec, arrs = step.fn(clip, prec, params)
        return cid, rec, arrs, None
    except Exception as exc:                                        # one bad clip must not stop a layer
        return cid, None, None, f"{type(exc).__name__}: {exc}"


def _map(fn, work, jobs):
    if jobs <= 1 or len(work) < 2:
        for w in work:
            yield fn(w)
        return
    from multiprocessing import Pool
    with Pool(jobs) as pool:
        yield from pool.imap(fn, work, chunksize=4)


class _lock:
    """An exclusive lock on a file (fcntl, WSL/Linux), so two runs of one layer or two ref moves cannot interleave."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            raise SystemExit(f"{self.path.parent} is being written by another run")
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def _jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _safe(clip_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in clip_id)
