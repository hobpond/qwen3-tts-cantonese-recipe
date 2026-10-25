#!/usr/bin/env python3
"""The recipe, as one pipeline: Common Voice in, a gated, heard Hong Kong Cantonese voice out.

    python recipe.py check                       what is set, built and importable; runs nothing heavy
    python recipe.py select                      1. yue, zh-HK and the donor's clips; decode to 24 kHz FLAC
    python recipe.py cleanup                     2. Sidon > dynamics > energy, edges > trim [> pad]; audio + encode
    python recipe.py sets                        3. the trainer's jsonl: yue + zh-HK (language), donor (speaker)
    python recipe.py train-lang                  4a. language phase from the public base
    python recipe.py train-spk                   4b. speaker phase, one voice
    python recipe.py bake                        5. the chosen voice rows into the export
    python recipe.py convert                     6. GGUF q8_0 for Qwen-TTS Studio
    python recipe.py render                      7a. first draws: held-out short lines and long prose, candidate and v0.0.1
    python recipe.py gate                        7b. the release gate's Part A, per voice
    python recipe.py card                        7c. the checkpoint card, per voice
    python recipe.py pages                       8. the listening page (skips redrawn, by paragraph, blind, rated)
    python recipe.py status                      the store's layers

Every stage takes --dry-run (print what it would run and exit) and --config (default recipe.toml beside this file).
All hyperparameters, thresholds and paths are in recipe.toml; nothing here is a number of the recipe.

WHERE THINGS GO. Everything the recipe writes lands under [paths].work, which must be outside this repository: audio,
codes, transcripts, weights and listening pages may never enter it (scripts/check_clean_room.py). Every stage writes
a record to <work>/records/<stage>.json (commands, inputs' sha256, this repository's commit) and logs to
<work>/logs/<stage>.log, created before anything runs so a stage that cannot log refuses instead of running blind.

WHAT THE STAGES NEED. select: CV_ROOT or an MDC archive, and DONOR_IDS. cleanup: SIDON_MODELS and a GPU. train-*: the
base model and a GPU with [train].min_free_mib free. bake: VOICE_SOURCE. convert: the GGUF runtime
(scripts/build_gguf_runtime.sh). render/gate/card/pages: BASELINE_GGUF, EVAL_LINES, EVAL_PROSE and the evaluation
models (README). Stages are not chained: run them in order, read each one's record, then the next.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TRAINER = ROOT / "trainer" / "sft_12hz_lora_mlflow.py"
SETS = ("language", "zhhk", "speaker")          # the data sets, each selected and cleaned on its own
CARD_LAYERS = (("energy", {}), ("f0", {}), ("g2p", {}), ("align", {"aligner": "sensevoice-ctc"}),
               ("syllables", {"contour": "voiced"}))

try:
    import tomllib
except ModuleNotFoundError:                     # Python 3.10
    import tomli as tomllib


# ---------------------------------------------------------------------------------------------------- configuration
def expand(value: str) -> str:
    """${VAR} from the environment and ~. A path naming an unset variable is unset as a whole (""), so a stage that
    needs it refuses rather than writing to a half-expanded path like /canto-store."""
    names = re.findall(r"\$\{(\w+)\}", value)
    if any(not os.environ.get(n) for n in names):
        return ""
    return os.path.expanduser(re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m.group(1)], value))


class Recipe:
    def __init__(self, config: Path, dry: bool):
        self.cfg = tomllib.loads(config.read_text(encoding="utf-8"))
        self.config_path = config
        self.dry = dry
        self.paths = {k: expand(v) for k, v in self.cfg["paths"].items()}
        self.py = self.paths.get("python") or sys.executable

    def path(self, name: str, env_hint: str | None = None) -> Path:
        """A configured path, or a refusal naming what to set. In a dry run an unset path prints as <name>."""
        v = self.paths.get(name, "")
        if v:
            p = Path(v)
            if name in ("work", "store") and str(p.resolve()).startswith(str(ROOT)):
                raise SystemExit(f"[paths].{name} = {p} is inside this repository; the recipe writes data there")
            return p
        hint = env_hint or re.findall(r"\$\{(\w+)\}", self.cfg["paths"].get(name, "")) or ["it in recipe.toml"]
        if self.dry:
            return Path(f"<{name}: set {hint[0] if isinstance(hint, list) else hint}>")
        raise SystemExit(f"[paths].{name} is not set: set {hint[0] if isinstance(hint, list) else hint}")

    @property
    def work(self) -> Path:
        return self.path("work")

    def env(self, **extra) -> dict:
        e = dict(os.environ)
        e["CANTO_STORE"] = str(self.path("store"))
        if self.paths.get("base_model"):
            e.setdefault("TTS_BASE_MODEL", self.paths["base_model"])
        e.update({k: str(v) for k, v in extra.items()})
        return e

    # ------------------------------------------------------------------------------------------------ running
    def log(self, stage: str) -> Path:
        p = self.work / "logs" / f"{stage}.log"
        if not self.dry:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            if not os.access(p, os.W_OK):
                raise SystemExit(f"{p} is not writable; refusing to run a stage that cannot log")
        return p

    def run(self, stage: str, cmd: list, cwd: Path | None = None, env: dict | None = None) -> None:
        cmd = [str(c) for c in cmd]
        print("$ " + shlex.join(cmd), flush=True)
        if self.dry:
            return
        log = self.log(stage)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] $ {shlex.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env or self.env(), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                sys.stdout.write(line)
                fh.write(line)
            rc = p.wait()
            fh.write(f"[rc {rc}]\n")
        if rc:
            raise SystemExit(f"{stage}: exit {rc} from {cmd[1] if len(cmd) > 1 else cmd[0]} (log {log})")

    def record(self, stage: str, rec: dict) -> None:
        rec = {"stage": stage, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "recipe_commit": code_commit(),
               "config_sha256": sha256(self.config_path), **rec}
        if self.dry:
            return
        print(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
        p = self.work / "records" / f"{stage}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    def store(self):
        """The store modules, with CANTO_STORE set before store.py reads it."""
        os.environ["CANTO_STORE"] = str(self.path("store"))
        sys.path.insert(0, str(SRC))
        import store as st
        import steps  # noqa: F401  (registers the steps)
        import cli
        return st, cli


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_commit() -> str:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "") if head else "unknown"
    except OSError:
        return "unknown"


def free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
        total, used = (int(x) for x in out.split(","))
        return total - used
    except Exception:
        return None


def flag(name: str, on: bool) -> str:
    return f"--{name}" if on else f"--no-{name}"


# ---------------------------------------------------------------------------------------------------- 1. select
def stage_select(r: Recipe) -> None:
    out = {}
    for which in SETS:
        d = r.cfg["data"][which]
        sel, data = r.work / "data" / which / "selected", r.work / "data" / which
        pre = r.paths.get(d.get("selected_dir", ""), "") if d.get("selected_dir") else ""
        # The speaker set may come from another release (v0.0.1's 343 clips did): speaker_* paths, else the language's.
        # A set naming its own `archive` (zh-HK: Common Voice 27.0, MDC only) is always read from that archive.
        pick = lambda k: (r.paths.get(f"speaker_{k}") if which == "speaker" else "") or r.paths.get(k)
        cv, mdc = pick("cv_root"), pick("mdc_archive")
        if d.get("archive"):
            cv, mdc = "", str(r.path(d["archive"]))
        locale = d.get("locale", r.cfg["data"]["language"]["locale"])
        common = ["--splits", *d["splits"], "--min-duration", d["min_duration"], "--max-duration", d["max_duration"],
                  "--gender", d["gender"], flag("require-colloquial", d["require_colloquial"]),
                  flag("drop-down-voted", d["drop_down_voted"]), "--seed", d["seed"], "--out-dir", sel]
        if pre:                                    # already selected: audio/ + transcripts.tsv, ingest only
            sel = Path(pre)
            print(f"{which}: pre-selected at {sel} ([paths].{d['selected_dir']}); its selector is not run")
            if not r.dry and not ((sel / "audio").is_dir() and (sel / "transcripts.tsv").is_file()):
                raise SystemExit(f"{sel} is not a selector output (audio/ and transcripts.tsv)")
        elif which == "speaker":
            common += ["--client-id-file", r.path("donor_ids")]
        if pre:
            pass
        elif mdc and not cv:                         # a Mozilla Data Collective archive
            r.run("select", [r.py, SRC / "select_mdc.py", "--archive", mdc, "--locale", locale, *common])
        else:                                      # the Hugging Face mirror layout; refuses naming CV_ROOT if unset
            cv = cv or str(r.path("cv_root"))
            if r.dry:
                print(f"  (with CV_ROOT={cv})")
            r.run("select", [r.py, SRC / "select_common_voice.py", *common], env=r.env(CV_ROOT=cv))
        r.run("select", [r.py, SRC / "ingest_recordings.py", "--audio-dir", sel / "audio", "--transcripts",
                         sel / "transcripts.tsv", "--speaker-id", d["speaker_id"], "--out-dir", data, "--subdir", "audio",
                         "--out-jsonl", "raw.jsonl", "--min-duration", d["ingest_min_duration"],
                         "--max-duration", d["ingest_max_duration"],
                         *(["--trim-silence"] if d.get("ingest_trim_silence") else [])])
        manifest = data / "raw.jsonl"
        if r.dry:
            continue
        n = sum(1 for _ in manifest.open(encoding="utf-8"))
        out[which] = {"pre_selected": bool(pre), "selected_dir": str(sel),
                      "transcripts_sha256": sha256(sel / "transcripts.tsv"),
                      "selection_ran": "no: given pre-selected" if pre else "yes",
                      "manifest": str(manifest), "manifest_sha256": sha256(manifest), "clips": n,
                      "expect_clips": d.get("expect_clips"), "matches_expectation": n == d.get("expect_clips")}
        if n != d.get("expect_clips"):
            print(f"NOTE {which}: {n} clips, the lab's set had {d.get('expect_clips')} (reported, not refused)")
    r.record("select", {"sets": out})


# ---------------------------------------------------------------------------------------------------- 2. cleanup
def coll_names(r: Recipe, which: str) -> dict:
    raw = r.cfg["data"][which]["collection"]
    final = r.cfg["sets"]["parts"][which]["from_step"]
    return {"raw": raw, "sidon": f"{raw}-sidon1", "dyn": f"{raw}-sidon1-dyn1", "final": f"{raw}-sidon1-dyn1-{final}1",
            "final_step": final}


def cleanup_plan(r: Recipe, which: str) -> list[tuple[str, str, dict]]:
    c, cu = coll_names(r, which), r.cfg["cleanup"]
    sidon = {**cu["sidon"], "models": str(r.path("sidon_models"))}
    steps = ["energy", "edges", "trim"] + (["pad"] if which in cu["pad_sets"] else [])
    if c["final_step"] not in steps:
        raise SystemExit(f"[sets.parts].{which}.from_step = {c['final_step']} is not run on the {which} set")
    plan = [(c["raw"], "ingest", {"manifest": str(r.work / "data" / which / "raw.jsonl")}),
            (c["raw"], "sidon", sidon), (c["sidon"], "derive from sidon", {}), (c["sidon"], "dynamics", cu["dynamics"]),
            (c["dyn"], "derive from dynamics (raw_path carried)", {})]
    plan += [(c["dyn"], s, cu["trim"] if s == "trim" else {}) for s in steps]
    plan += [(c["final"], f"derive from {c['final_step']}", {}), (c["final"], "audio", {}),
             (c["final"], "encode", cu["encode"])]
    return plan


def stage_cleanup(r: Recipe) -> None:
    if r.dry:
        for which in SETS:
            for coll, step, params in cleanup_plan(r, which):
                print(f"would run  {coll:28s} {step:40s} {json.dumps(params)}")
        return
    st, cli = r.store()
    jobs = r.cfg["cleanup"]["jobs"]
    out = {}

    def derived(of: str, step: str, name: str, extra: dict | None = None):
        """exp099._derived: the clips of `of` at the files `step` made, errored clips left out on purpose."""
        bad = frozenset(c for c, x in st.Store(of).records(step, errors="keep").items() if "error" in x)
        rows, meta = cli.repaired_rows(of=of, step=step, exclude=bad)
        if extra:
            rows = [{**x, **extra.get(x["clip_id"], {})} for x in rows]
        d = st.Store(name)
        d.ingest(rows, meta)
        return d, len(bad)

    for which in SETS:
        c, cu = coll_names(r, which), r.cfg["cleanup"]
        manifest = r.work / "data" / which / "raw.jsonl"
        rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
        raw = st.Store(c["raw"])
        raw.ingest([{"clip_id": f"{c['raw']}-{Path(x['audio']).stem}", "path": x["audio"], "text": x["text"],
                     "kind": "raw", "speaker": x.get("speaker_id")} for x in rows],
                   {"what": f"the {which} set, {manifest.name}", "manifest_sha256": sha256(manifest),
                    "licence": "CC-0", "kind": "raw"})
        raw.run(st.REGISTRY["sidon"], params={**cu["sidon"], "models": str(r.path("sidon_models"))}, jobs=1)
        s1, e1 = derived(c["raw"], "sidon", c["sidon"])
        s1.run(st.REGISTRY["dynamics"], params=cu["dynamics"], jobs=jobs)
        raw_path = {cid: x["raw_path"] for cid, x in s1.records("ingest").items()}
        dyn, e2 = derived(c["sidon"], "dynamics", c["dyn"], {cid: {"raw_path": p} for cid, p in raw_path.items()})
        for step in ["energy", "edges", "trim"] + (["pad"] if which in cu["pad_sets"] else []):
            dyn.run(st.REGISTRY[step], params=cu["trim"] if step == "trim" else None, jobs=jobs)
        fin, e3 = derived(c["dyn"], c["final_step"], c["final"])
        fin.run(st.REGISTRY["audio"], jobs=2)
        fin.run(st.REGISTRY["encode"], params=cu["encode"], jobs=1)
        sid = raw.records("sidon", errors="keep")
        ok = [x for x in sid.values() if "error" not in x]
        out[which] = {"collections": c, "refs": {k: st.Store(v).refs() for k, v in c.items() if k != "final_step"},
                      "clips": len(sid), "sidon_errors": len(sid) - len(ok), "errors_dropped_at": [e1, e2, e3],
                      "env_corr_under": sum(x["env_corr"] < cu["env_corr_min"] for x in ok),
                      "length_changed": sum(not x["same_length"] for x in ok)}
    r.record("cleanup", out)


# ---------------------------------------------------------------------------------------------------- 3. sets
def stage_sets(r: Recipe) -> None:
    """One part per data set (cli.training_set, the Sidon exclusions on each), then each training set as its parts
    concatenated in the order [sets.train] names them: the language set is the yue rows, then the zh-HK rows, as the
    lab's exp100.stage_sets wrote it."""
    out = {"parts": {}, "train": {}}
    parts = r.cfg["sets"]["parts"]
    for which in SETS:
        c, d = coll_names(r, which), r.cfg["data"][which]
        dest = r.work / "sets" / parts[which]["file"]
        if r.dry:
            print(f"would write {dest} from {c['final']} (ingest, audio, encode), speaker_id {d['speaker_id']}, "
                  f"excluding {c['raw']}'s sidon errors and env_corr < {r.cfg['cleanup']['env_corr_min']}")
            continue
        st, cli = r.store()
        rec = st.Store(c["raw"]).records("sidon", errors="keep")
        bad = frozenset(cid for cid, x in rec.items() if "error" in x or x["env_corr"] < r.cfg["cleanup"]["env_corr_min"])
        have = set(st.Store(c["final"]).records("ingest"))
        n = cli.training_set(c["final"], dest, speaker_id=d["speaker_id"], exclude=frozenset(bad & have))
        out["parts"][which] = {"file": str(dest), "rows": n, "excluded": len(bad & have), "sha256": sha256(dest),
                               "collection": c["final"], "expect_rows": d.get("expect_rows")}
        if d.get("expect_rows") and n != d["expect_rows"]:
            print(f"NOTE {which}: {n} rows, the lab's set had {d['expect_rows']} (reported, not refused)")
    for name, t in r.cfg["sets"]["train"].items():
        dest = r.work / "sets" / t["file"]
        srcs = [r.work / "sets" / parts[w]["file"] for w in t["parts"]]
        if r.dry:
            print(f"would write {dest} = " + " + ".join(str(x) for x in srcs) + " (concatenated, in this order)")
            continue
        with dest.open("w", encoding="utf-8", newline="\n") as w:
            for f in srcs:
                w.write(f.read_text(encoding="utf-8"))
        out["train"][name] = {"file": str(dest), "parts": t["parts"], "sha256": sha256(dest),
                              "rows": sum(out["parts"][w]["rows"] for w in t["parts"])}
    r.record("sets", out)


# ---------------------------------------------------------------------------------------------------- 4. training
def recipe_flags(t: dict) -> list:
    keys = ("batch_size", "gradient_accumulation_steps", "lr", "num_epochs", "warmup_ratio", "lora_rank", "lora_alpha",
            "checkpoint_every_epochs", "checkpoint_every_steps", "sub_talker_weight", "attn_implementation", "seed")
    if "seed" not in t:
        raise SystemExit("[train].seed is not set: an unseeded build cannot be reproduced (the trainer shuffles and "
                         "initialises LoRA at random without it)")
    flags = [x for k in keys for x in (f"--{k}", t[k])]
    return flags + (["--fast_kernels"] if t.get("fast_kernels") else [])   # exploration only


def mlflow_run_id(db: Path) -> str | None:
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            row = con.execute("select run_uuid from runs order by start_time desc limit 1").fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def train(r: Recipe, phase: str, init: Path, data: Path, extra: list) -> dict:
    t = r.cfg["train"]
    out = r.work / "train" / phase
    db = r.work / "train" / "mlflow.db"
    cmd = [r.py, TRAINER, "--init_model_path", init, "--output_model_path", out, "--train_jsonl", data,
           *recipe_flags(t), "--mlflow_tracking_uri", f"sqlite:///{db}", *extra]
    if not r.dry:
        if (out / "final_merged").exists():
            raise SystemExit(f"{out}/final_merged exists: phase {phase} is trained already (move it to retrain)")
        if not data.exists():
            raise SystemExit(f"{data} is missing: run `recipe.py sets`")
        free = free_mib()
        if free is None or free < t["min_free_mib"]:
            raise SystemExit(f"REFUSING: {free} MiB free on the card, [train].min_free_mib {t['min_free_mib']}")
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r.run(f"train-{phase}", cmd, cwd=TRAINER.parent)
    if r.dry:
        return {}
    cfg = json.loads((out / "final_merged" / "config.json").read_text(encoding="utf-8"))
    rec = {"command": [str(c) for c in cmd], "wall_s": round(time.time() - t0), "data_sha256": sha256(data),
           "mlflow_run_id": mlflow_run_id(db), "tts_model_type": cfg.get("tts_model_type"),
           "tts_model_size": cfg.get("tts_model_size"), "spk_id": cfg.get("talker_config", {}).get("spk_id")}
    if rec["tts_model_size"] != "1b7":
        raise SystemExit(f"{phase}: tts_model_size {rec['tts_model_size']!r}, not '1b7'")
    return rec


def stage_train_lang(r: Recipe) -> None:
    rec = train(r, "lang", r.path("base_model"), r.work / "sets" / r.cfg["sets"]["train"]["language"]["file"],
                r.cfg["train"]["language_extra"])
    r.record("train-lang", rec)


def stage_train_spk(r: Recipe) -> None:
    name = r.cfg["train"]["speaker_name"]
    rec = train(r, "spk", r.work / "train" / "lang" / "final_merged",
                r.work / "sets" / r.cfg["sets"]["train"]["speaker"]["file"], ["--speaker_name", name])
    if not r.dry and rec["spk_id"] != {name: 3000}:
        raise SystemExit(f"the speaker phase exported spk_id {rec['spk_id']}, not {{{name!r}: 3000}}")
    r.record("train-spk", rec)


# ---------------------------------------------------------------------------------------------------- 5-6. voices, GGUF
def export_dir(r: Recipe) -> Path:
    return r.work / "train" / "spk" / "final_merged"


def gguf_dir(r: Recipe) -> Path:
    cache = Path(os.environ.get("QWEN_TTS_GGUF_CACHE", os.path.expanduser("~/.qwen-tts-gguf")))
    return cache / r.cfg["convert"]["key"]


def stage_bake(r: Recipe) -> None:
    voices = [f"--voice={n}={row}" for n, row in r.cfg["bake"]["voices"].items()]
    r.run("bake", [r.py, SRC / "bake_voices.py", "--target", export_dir(r), "--source", r.path("voice_source"),
                   *voices, "--record", r.work / "records" / "bake-rows.json"])
    r.record("bake", {"target": str(export_dir(r)), "source": str(r.path("voice_source")), "voices": r.cfg["bake"]["voices"]})


def stage_convert(r: Recipe) -> None:
    if not r.dry:
        cfg = json.loads((export_dir(r) / "config.json").read_text(encoding="utf-8"))
        if cfg.get("tts_model_size") != "1b7":       # Studio's converter defaults a missing size to 0b6
            raise SystemExit(f"{export_dir(r)}: tts_model_size {cfg.get('tts_model_size')!r}, not '1b7'")
        want = {n: int(v) for n, v in r.cfg["bake"]["voices"].items()}
        if cfg["talker_config"].get("spk_id") != want:
            raise SystemExit(f"spk_id {cfg['talker_config'].get('spk_id')} is not the baked {want}: run `recipe.py bake`")
    c = r.cfg["convert"]
    r.run("convert", [r.py, SRC / "gguf_backend.py", "--convert", export_dir(r), "--key", c["key"], "--quant", c["quant"]])
    rec = {"gguf_dir": str(gguf_dir(r)), "key": c["key"], "quant": c["quant"]}
    if not r.dry:
        talkers = sorted(gguf_dir(r).glob("qwen-talker-*.gguf"))
        rec["talker"] = {t.name: hashlib.md5(t.read_bytes()).hexdigest() for t in talkers}
    r.record("convert", rec)


# ---------------------------------------------------------------------------------------------------- 7. evaluation
def arms(r: Recipe) -> list[tuple[str, Path, str]]:
    """(arm, gguf dir, speaker): every candidate voice, then the baseline."""
    e = r.cfg["eval"]
    return [(v, gguf_dir(r), v) for v in e["voices"]] + [(e["baseline_name"], r.path("baseline_gguf"), e["baseline_speaker"])]


def stage_render(r: Recipe) -> None:
    e = r.cfg["eval"]
    short, prose = r.work / "eval" / "short", r.work / "eval" / "prose"
    for arm, g, spk in arms(r):
        r.run("render", [r.py, SRC / "render_short.py", "--arm", arm, "--gguf", g, "--speaker", spk,
                         "--lines", r.path("eval_lines"), "--out", short, "--draws", e["draws"], "--expect", e["expect_lines"]])
    if (prose / "manifest.json").exists() and not r.dry:
        print(f"{prose}/manifest.json exists: long prose rendered already (move it to re-render)")
    else:
        cmd = [r.py, SRC / "render_story.py", "--texts", r.path("eval_prose"), "--out", prose, "--hi", e["chunk_hi"]]
        cmd += [x for arm, g, spk in arms(r) for x in ("--arm", f"{arm}={g}:{spk}")]
        if e["seed_salt"]:
            cmd += ["--seed-salt", e["seed_salt"]]
        r.run("render", cmd)
    r.record("render", {"short": str(short), "prose": str(prose), "arms": [a for a, _, _ in arms(r)],
                        "lines_sha256": None if r.dry else sha256(r.path("eval_lines"))})


def ingest_short(r: Recipe) -> dict:
    """Each arm's short-line renders as a store collection short-<arm>; the donor's raw clips through the tone
    template layers (energy, f0, g2p, align sensevoice-ctc, syllables voiced, speaker)."""
    names = {arm: f"short-{arm}" for arm, _, _ in arms(r)}
    tpl = r.cfg["eval"]["templates"]
    if r.dry:
        for arm, coll in names.items():
            print(f"would ingest {r.work / 'eval/short' / arm / 'manifest.jsonl'} as {coll}")
        print(f"would run on {tpl}: " + ", ".join(f"{n}{p or ''}" for n, p in CARD_LAYERS) + ", speaker")
        return names
    st, _ = r.store()
    for arm, coll in names.items():
        d = r.work / "eval" / "short" / arm
        rows = [json.loads(x) for x in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        st.Store(coll).ingest([{"clip_id": x["clip_id"], "path": str(d / x["flac"]), "text": x["text"],
                                "kind": "generated"} for x in rows], {"what": f"{arm} short-line renders", "kind": "generated"})
    s = st.Store(tpl)
    for name, params in CARD_LAYERS:
        s.run(st.REGISTRY[name], params=params, jobs=1 if name == "align" else 4)
    s.run(st.REGISTRY["speaker"], jobs=1)
    return names


def stage_gate(r: Recipe) -> None:
    names = ingest_short(r)
    rules = {k: v for k, v in r.cfg["gate"].items()}
    base = r.cfg["eval"]["baseline_name"]
    out = {}
    for v in r.cfg["eval"]["voices"]:
        dest = r.work / "eval" / f"gate-{v}.json"
        r.run("gate", [r.py, SRC / "release_gate.py", "--candidate", names[v], "--baseline", names[base],
                       "--candidate-manifest", r.work / "eval/short" / v / "manifest.jsonl",
                       "--baseline-manifest", r.work / "eval/short" / base / "manifest.jsonl",
                       "--templates", r.cfg["eval"]["templates"], "--measure", "--rules", json.dumps(rules), "--out", dest])
        if not r.dry:
            out[v] = json.loads(dest.read_text(encoding="utf-8"))["verdict"]
    r.record("gate", {"verdicts": out, "waived_by_owner": r.cfg["gate"].get("waived", []), "part_a_complete": False,
                      "missing": "A2 length bars; Part B (long prose)"})


def stage_card(r: Recipe) -> None:
    names = ingest_short(r)
    e = r.cfg["eval"]
    prose = r.work / "eval" / "prose"
    for v in e["voices"]:
        cmd = [r.py, SRC / "checkpoint_card.py", "--name", f"{r.cfg['convert']['key']}-{v}", "--short", names[v],
               "--baseline-short", names[e["baseline_name"]], "--prose", f"{prose}:{v}",
               "--baseline-prose", f"{prose}:{e['baseline_name']}", "--donor", e["templates"], "--templates", e["templates"],
               "--out", r.work / "cards"]
        gate = r.work / "eval" / f"gate-{v}.json"
        if r.dry or gate.exists():
            cmd += ["--gate", gate]
        r.run("card", cmd)
    r.record("card", {"cards": str(r.work / "cards")})


# ---------------------------------------------------------------------------------------------------- 8. pages
def stage_pages(r: Recipe) -> None:
    e, p = r.cfg["eval"], r.cfg["pages"]
    prose, final = r.work / "eval" / "prose", r.work / "eval" / "prose-final"
    voices = e["voices"]
    if len(voices) > 2:
        raise SystemExit("a page holds at most three voices: two candidates and the baseline")
    if (final / "manifest.json").exists() and not r.dry:
        print(f"{final}/manifest.json exists: skips redrawn already")
    else:
        cmd = [r.py, SRC / "redraw_skips.py", "--renders", prose, "--out", final]
        cmd += [x for arm, g, spk in arms(r) for x in ("--arm", f"{arm}={g}:{spk}")]
        r.run("pages", cmd + (["--two-recognisers"] if p["two_recognisers"] else []))
    texts = r.path("eval_prose")
    stories = p["stories"] or (sorted(x.stem for x in texts.glob("*.txt")) if texts.is_dir() else ["<every .txt in eval_prose>"])
    labels = "ABC"
    slots = [f"{labels[i]}={v}:{v}" for i, v in enumerate(voices)]
    slots.append(f"{labels[len(voices)]}={e['baseline_name']}:v0.0.1")
    cmd = [r.py, SRC / "build_story_page.py", "--renders", final, "--texts", texts, "--stories", ",".join(stories),
           "--bm", p["bm"], "--page", "recipe-p1", "--out", r.work / "pages" / p["bm"], "--by-paragraph",
           "--max-chars", p["max_chars"], *[x for s in slots for x in ("--slot", s)]]
    r.run("pages", cmd + (["--blind"] if p["blind"] else []))
    r.record("pages", {"page": str(r.work / "pages" / p["bm"] / "page.html"), "bm": p["bm"], "slots": slots,
                       "note": "register the listen before it is heard; harvest every verdict"})


# ---------------------------------------------------------------------------------------------------- check, status
def stage_check(r: Recipe) -> None:
    print("paths (recipe.toml [paths], ${VAR} from the environment):")
    for k, raw in r.cfg["paths"].items():
        v = r.paths.get(k, "")
        state = "unset" if not v else ("ok" if Path(v).exists() else "missing")
        print(f"  {state:8s} {k:20s} {v or raw}")
    print(f"python: {r.py}")
    mods = "torch qwen_tts peft mlflow accelerate transformers safetensors librosa soundfile scipy pycantonese funasr " \
           "qwen_asr opencc speechbrain torchaudio"
    probe = ("import importlib, sys\nbad = []\nfor m in sys.argv[1:]:\n    try:\n        importlib.import_module(m)\n"
             "    except Exception as e:\n        bad.append(f'{m} ({type(e).__name__})')\nprint('missing: ' + ', '.join(bad) if bad else 'all importable')")
    if r.dry:
        print(f"would probe imports: {mods}")
    else:
        res = subprocess.run([r.py, "-c", probe, *mods.split()], capture_output=True, text=True,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        print("modules: " + (res.stdout.strip() or res.stderr.strip()[-300:]))
    sys.path.insert(0, str(SRC))
    import gguf_backend as gb
    a = gb.availability()
    print("gguf runtime: " + ("usable" if a["usable"] else "NOT usable") + f" (cli {a['cli_present']}, converter "
          f"{a['converter_present']}, codec {a['codec_present']}); build: bash scripts/build_gguf_runtime.sh")
    print(f"card free: {free_mib()} MiB")


def stage_status(r: Recipe) -> None:
    names = [coll_names(r, w)[k] for w in SETS for k in ("raw", "sidon", "dyn", "final")]
    names += [f"short-{a}" for a in r.cfg["eval"]["voices"] + [r.cfg["eval"]["baseline_name"]]]
    if r.dry:
        print("would show: " + ", ".join(names))
        return
    st, cli = r.store()
    cli.cmd_status([n for n in names if (st.ROOT / n).exists()])


STAGES = {"check": stage_check, "select": stage_select, "cleanup": stage_cleanup, "sets": stage_sets,
          "train-lang": stage_train_lang, "train-spk": stage_train_spk, "bake": stage_bake, "convert": stage_convert,
          "render": stage_render, "gate": stage_gate, "card": stage_card, "pages": stage_pages, "status": stage_status}


def main() -> int:
    ap = argparse.ArgumentParser(prog="recipe.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES), help="the stage to run")
    ap.add_argument("--dry-run", action="store_true", help="print what the stage would run, and run nothing")
    ap.add_argument("--config", type=Path, default=ROOT / "recipe.toml")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    r = Recipe(a.config, a.dry_run)
    if a.dry_run:
        print(f"[dry run] {a.stage}: nothing is executed or written")
    STAGES[a.stage](r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
