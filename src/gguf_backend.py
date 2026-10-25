"""Render through qwen3-tts-cpp (GGUF) instead of PyTorch, when it is worth it.

WHAT THIS BUYS. Measured 2026-09-05 on EXP-000's held-out passage, 10 chunks, the
speaker-phase arm, same texts both ways:

    PyTorch (stage 6 of the lane)      219 s wall
    GGUF q8_0, one process per chunk    50 s wall
    GGUF q8_0, ONE RESIDENT process     28 s wall

The middle row pays a fresh model load on every one of the ten invocations, which is
what the third row removes: see `--serve` below. The PyTorch number includes its retry
budget. Conversion costs 41 s per checkpoint, once, and is cached -- so it repays
itself inside the first passage and is free on every render after.

THE RESIDENT ROW CHANGES NOTHING BUT THE CLOCK, and that was checked rather than
assumed: re-rendering EXP-000's passage through the resident path reproduced all ten
committed chunks BYTE-IDENTICALLY (md5 per chunk, 10/10). It has to be that way or the
optimisation is a silent re-render of every result that used it.

Those are end-to-end wall-clock, which is the right number for "what does the lane
cost" and the wrong one for "how fast is the kernel" -- it charges GGUF a model load
per utterance that a resident process would not pay. For the decomposition, the
repository's own `scripts/experiments/throughput/bench_gguf_speed.py` uses the CLI's
`--repeat` to separate load from generation, and on this same checkpoint reports:

    median generate    0.100 s/char   against PyTorch's measured 0.330  -> 3.30x
    median RTF         3.76 x realtime
    median model load  2.6 s          paid once per process, not per utterance

So the ~3x that EXP-006 recorded survives, and it survives the conversion cost at
this passage size several times over.

WHAT IT COSTS, MEASURED RATHER THAN ASSUMED. `asr_gate` over the same ten chunks:

    median CER   PyTorch 0.211   GGUF 0.211      (6 of 10 clips transcribed IDENTICALLY)
    mean  CER    PyTorch 0.256   GGUF 0.279
    paired delta median +0.000, and the clips that moved went 2 better / 2 worse

Symmetric, median zero. At n=10 that is sampling noise between two stochastic draws,
not a quantisation effect. Note what it does NOT establish: it cannot separate "q8_0
changed nothing" from "q8_0 changed something smaller than draw-to-draw variance",
and `asr_gate` is an intactness instrument that its own docstring forbids pointing at
a quality question. So: q8_0 is not BROKEN here. It is not certified equivalent, and
a listening result must not mix arms rendered through different backends.

Speech PACING is preserved -- 0.241 s/char against PyTorch's 0.236, ratio 1.02x over
the same ten chunks -- but only after trimming. See the trailing-pad note below.

WHICH PATH IS ACTUALLY THE MATURE ONE FOR VOLUME. Left unsaid for too long: for
generating a lot of text, this is the better path, and not only because it is faster.
The PyTorch render path carries a retry budget and an ESCALATION that re-chunks a
passage when a chunk keeps coming back defective. That machinery exists because it is
needed, and it has a cost that only shows up at volume -- re-chunking changes the
chunk boundaries, so two arms of the same passage stop lining up:

    EXP-000, same passage, same 10 chunks planned
      PyTorch  base 10 -> 9 chunks (escalated), trained 10 -> 10   paired 8 of 10
      GGUF     base 10 -> 10,      trained 10 -> 10                paired 10 of 10

Both rendered clean in the end. But the torch base arm needed the escalation, and the
listening page then lost two clips because the arms no longer shared a chunking. The
GGUF path produced 0/10 defective on both arms with no retries and no escalation, on
two separate hosts -- this workstation and a GCP L4.

So for a long read, a batch of passages, or anything where the count of clips matters,
prefer GGUF: it is ~3.3x on generation, it holds the model resident, and it has not
needed the retry machinery. The caveats above still hold -- q8_0 is tested for
intactness and NOT certified equivalent, so never mix backends across the arms of one
comparison -- but "which one do I reach for at volume" has an answer, and it is this
one.

FOUR THINGS THAT WILL WASTE AN AFTERNOON IF THEY ARE NOT WRITTEN DOWN.

1. THE TALKER IS FOUND BY FILENAME, NOT BY BEING THE ONLY FILE THERE. The CLI scores
   the candidates in `-m <dir>` and awards +40 to a filename containing
   `qwen-talker` (pipeline_models.cpp). Name the converted file anything else --
   `talker-q8_0.gguf`, say -- and it scores too low, so the CLI falls through to a
   HARDCODED default name and reports that it cannot open
   `qwen-talker-0.6b-base-Q8_0.gguf`, a file nobody asked for and which was never
   going to exist. The error names a path you never chose, which is what makes it
   confusing. The converter's own usage example uses the `qwen-talker` prefix;
   `talker_gguf_name()` below keeps to it, and keeps the arm identity in the name so
   a converted checkpoint is still identifiable on disk.

2. THE CLI PADS, AND THE DEFAULT TRIM IS TOO LOOSE TO FIX IT. Measured on EXP-000
   chunk 0, both arms, same text:

       GGUF     6.80 s raw   lead 0.78 s   tail 2.10 s   (pad is ~-70 dBFS, silent)
       PyTorch  4.65 s raw   lead 0.12 s   tail 0.12 s

   `trim_edges` CAPS edge silence at `max_edge_s` rather than removing it -- which is
   deliberate on its part, a little air at each end is wanted. But at the usual cap of
   1.0 s a GGUF clip still carries ~1.8 s of edge silence where a PyTorch clip carries
   0.24 s, and in a blind A/B page that is a CUE: the listener hears one arm start
   late and end late and has learned which is which without hearing a phoneme.
   `audio_trim`'s own docstring names this hazard. So MAX_EDGE_S below defaults to
   0.15 s for this backend -- not a taste in silence, but the value that makes GGUF
   edges indistinguishable from the other backend's. Raise it only for a page where
   every arm is rendered through GGUF.

   The pad also matters for pacing: this project's ear tests ask the listener to judge
   PAUSES, and untrimmed dead air is scored as the model's own. Speech duration itself
   is unaffected -- 0.241 s/char against PyTorch's 0.236 once the pad is off. Per
   `audio_trim`, the raw duration is returned alongside so a metric computed on
   untrimmed audio stays comparable.

3. ONE PROCESS PER UTTERANCE IS MOSTLY NOT SYNTHESIS. Measured on one short line:
   1205 ms inside the binary against 6.13 s of wall. The rest is process start, 2.3 GB
   of model file, and CUDA init -- so a ten-clip passage spent longer loading than
   speaking. `--serve` (this project's patch, external/qwen3-tts-cpp/patches/) loads once and
   takes one request per line on stdin; `GGUFModel` keeps one such process per
   (checkpoint, startup flags) and feeds it the batch `_Speech.create` already has.

   The seed travels PER REQUEST, not at startup. One seed for a whole resident session
   would make every clip depend on how many came before it, so the same text would
   render differently as the second request than as the first and a single clip could
   not be re-rendered on its own. With a per-request seed, resident output equals
   one-shot output clip for clip -- which is what the 10/10 above measures.

   A binary without `--serve` -- anyone on stock upstream -- falls back to one process
   per clip and is merely slower. That path is exercised, not hoped for.

4. IT WRITES WAV. That is the CLI's only output format, so the wav is a scratch file:
   decoded, trimmed, returned as samples, deleted. Nothing here persists a `.wav` --
   `audio_uri.write_flac` owns what reaches disk.

WHAT IS NOT VENDORED. qwen3-tts-cpp is a C++ project that lives outside this
repository and has to be built once per machine (scripts/build_gguf_runtime.sh);
`python recipe.py check` reports whether it is present. Build it natively for the platform you render on -- a Windows .exe
driven from WSL needs its DLLs on PATH and buys nothing over the native Linux binary.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# siblings in src/ (audio_trim, audio_uri) are imported by bare name, as the
# rest of src/ does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# WHERE THE TOOLCHAIN IS. These defaults are what `build_gguf_runtime.sh` produces,
# and they are deliberately not anyone's personal checkout: this repository is public
# and must not point at a private one. The build script clones UPSTREAM
# (Danmoreng/qwen3-tts.cpp, MIT) at a pinned commit and applies the patch in
# external/qwen3-tts-cpp/patches/, so every path in the dependency chain is public and
# reproducible. Override either with the environment variable if you build elsewhere.
CPP_ROOT = Path(os.environ.get(
    "QWEN_TTS_CPP", os.path.expanduser("~/.qwen-tts-cpp/src")))
CLI_BIN = Path(os.environ.get(
    "QWEN_TTS_CLI", os.path.expanduser("~/.qwen-tts-cpp/build/qwen3-tts-cli")))
# Converted checkpoints are big (~2 GB each) and belong on a native disk, never
# under /mnt (on WSL, DrvFs is ~8x slower sequentially and ~100x on small files).
GGUF_CACHE = Path(os.environ.get(
    "QWEN_TTS_GGUF_CACHE", os.path.expanduser("~/.qwen-tts-gguf")))

CODEC_NAME = "qwen-tokenizer-12hz-q8_0.gguf"
# Match the PyTorch backend's natural edge silence (0.12 s), so that trailing pad
# cannot act as an arm cue in a blind page. See note 2.
MAX_EDGE_S = 0.15
# What `do_sample=False` means on this backend: sampling stays ON (turning it off
# does not terminate -- see GGUFModel._flags) and the seed is pinned instead, which
# gives byte-identical repeats. Any fixed value does; this one is arbitrary and
# recorded so a rendered set can be reproduced.
DEFAULT_SEED = int(os.environ.get("QWEN_TTS_SEED", "20260905"))
# OVERRIDABLE SINCE 2026-09-12, and EXP-050 could not run without it. Its primary
# metric is ASR character error rate at THREE seeds per arm per passage, so that
# within-arm variance is measured rather than assumed -- and with the seed pinned to a
# module constant there was no way to render a second one. The default is unchanged,
# so every existing rendered set and every recorded `seeds: [20260905]` still
# reproduces exactly; only an explicit QWEN_TTS_SEED moves it.
#
# WHAT A SECOND SEED DOES AND DOES NOT BUY. It varies the SAMPLER, not the training.
# Three seeds estimate how much one checkpoint's output wanders between generations;
# they say nothing about how much two training runs of the same recipe would differ.
# See docs/okf/operations/a-seed-spread-is-not-a-threshold.md before using the spread
# as a decision threshold.
# The substring the CLI's scorer requires. Changing this breaks discovery silently.
TALKER_PREFIX = "qwen-talker"


_SERVE_SUPPORTED: bool | None = None


def serve_supported() -> bool:
    """Does the built CLI have the resident `--serve` mode?

    A machine on stock upstream has a binary without it, and that must degrade to
    one process per clip rather than failing with something obscure about a pipe.
    Probed once from `--help` and cached, because the answer is a property of the
    binary and cannot change while we are running.
    """
    global _SERVE_SUPPORTED
    if _SERVE_SUPPORTED is None:
        try:
            p = subprocess.run([str(CLI_BIN), "--help"], capture_output=True,
                               text=True, timeout=30)
            _SERVE_SUPPORTED = "--serve" in (p.stdout + p.stderr)
        except Exception:
            _SERVE_SUPPORTED = False
    return _SERVE_SUPPORTED


class GGUFUnavailable(RuntimeError):
    """The toolchain is not built on this machine. The recipe has no other render path."""


def _converter_python() -> str:
    """The interpreter to run the upstream converters with.

    NOT a bare "python3". The converters import numpy, torch and gguf, and the
    system python3 has none of them -- a fresh checkout hits
    `ModuleNotFoundError: No module named 'numpy'` from inside a subprocess, which
    reads as the converter being broken rather than as the wrong interpreter.
    `sys.executable` is whatever is already running this code, and by construction
    that is the environment with the model stack in it.
    """
    return os.environ.get("QWEN_VENV_PY") or sys.executable or "python3"


def talker_gguf_name(key: str, quant: str = "q8_0") -> str:
    """`qwen-talker-<key>-<quant>.gguf` -- the prefix is load-bearing, see note 1."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-") or "checkpoint"
    return f"{TALKER_PREFIX}-{safe}-{quant}.gguf"


def _tokenizer_source() -> Path | None:
    """Where the 12 Hz tokenizer is, for converting the shared codec.

    THE BASE CHECKPOINT ALREADY CARRIES IT. `Qwen3-TTS-Tokenizer-12Hz` is a separate
    HF repo, and a fresh machine has no reason to have fetched it -- so before this
    fallback existed, `build_gguf_runtime.sh` succeeded and the bootstrap gate still
    said the codec was missing, with the fix being an undocumented download. But
    every base checkpoint ships `speech_tokenizer/`, which the recipe already
    requires, and converting from it produces a BYTE-IDENTICAL codec: md5
    39819bc6f94b35837b94b22cdd0ce95e either way, 398 tensors, 291.1 MB. Measured, not
    assumed -- the two files were converted separately and compared.

    Order: an explicit override, then the standalone repo if someone has it, then the
    base checkpoint.
    """
    env = os.environ.get("QWEN_TTS_TOKENIZER")
    if env:
        return Path(env)
    standalone = CPP_ROOT / "models" / "Qwen3-TTS-Tokenizer-12Hz"
    if standalone.exists():
        return standalone
    base = os.environ.get("TTS_BASE_MODEL")
    if base and (Path(base) / "speech_tokenizer").is_dir():
        return Path(base) / "speech_tokenizer"
    return None


def availability() -> dict:
    """What is present, as data. `recipe.py check` renders this; nothing raises."""
    conv = CPP_ROOT / "scripts" / "convert_tts_to_gguf.py"
    tok_conv = CPP_ROOT / "scripts" / "convert_tokenizer_to_gguf.py"
    codec = _find_codec()
    tok_src = _tokenizer_source()
    # The codec need not exist yet -- it is buildable from the 12 Hz tokenizer.
    codec_ok = bool(codec) or (tok_conv.exists() and bool(tok_src)
                               and tok_src.exists())
    if codec:
        codec_says = str(codec)
    elif tok_src:
        codec_says = f"(buildable from {tok_src})"
    else:
        codec_says = "(no 12 Hz tokenizer found -- check TTS_BASE_MODEL)"
    return {
        "cli": str(CLI_BIN), "cli_present": CLI_BIN.exists(),
        "converter": str(conv), "converter_present": conv.exists(),
        "codec": codec_says,
        "codec_present": codec_ok,
        "codec_built": bool(codec),
        "cache": str(GGUF_CACHE),
        "usable": CLI_BIN.exists() and conv.exists() and codec_ok,
    }


def _find_codec() -> Path | None:
    """The shared 12 Hz codec. One copy serves every converted talker.

    It is a converted artefact like the talker, not something that ships: a clean
    machine makes it from Qwen3-TTS-Tokenizer-12Hz with the sibling
    `convert_tokenizer_to_gguf.py`. `ensure_codec()` does that; this only looks.
    """
    env = os.environ.get("QWEN_TTS_CODEC")
    if env and Path(env).exists():
        return Path(env)
    cached = GGUF_CACHE / CODEC_NAME
    if cached.exists():
        return cached
    for base in (CPP_ROOT / "models", CPP_ROOT):
        if not base.exists():
            continue
        # case-insensitively: upstream's example writes ...-Q8_0.gguf
        hit = next((p for p in sorted(base.rglob("*.gguf"))
                    if p.name.lower() == CODEC_NAME), None)
        if hit:
            return hit
    return None


def ensure_codec(tokenizer_dir: str | Path | None = None,
                 quant: str = "q8_0") -> Path:
    """Return the codec GGUF, converting Qwen3-TTS-Tokenizer-12Hz if need be.

    Kept separate from the talker conversion because it is done ONCE per machine
    and shared by every checkpoint -- converting it per arm would be ~290 MB and a
    minute of nothing, repeated.
    """
    found = _find_codec()
    if found:
        return found
    src = Path(tokenizer_dir) if tokenizer_dir else _tokenizer_source()
    conv = CPP_ROOT / "scripts" / "convert_tokenizer_to_gguf.py"
    if not conv.exists():
        raise GGUFUnavailable(f"no tokenizer converter at {conv}")
    if src is None or not src.exists():
        raise GGUFUnavailable(
            "the 12 Hz tokenizer is not on this machine. It normally comes from "
            "the base checkpoint's speech_tokenizer/, so check TTS_BASE_MODEL; "
            "otherwise fetch Qwen3-TTS-Tokenizer-12Hz, or point QWEN_TTS_CODEC at "
            f"an existing {CODEC_NAME}.")
    GGUF_CACHE.mkdir(parents=True, exist_ok=True)
    dst = GGUF_CACHE / CODEC_NAME
    print(f"[gguf] converting the shared 12 Hz codec -> {dst.name}", flush=True)
    p = subprocess.run([_converter_python(), str(conv), "--input", str(src),
                        "--output", str(dst), "--type", quant],
                       capture_output=True, text=True)
    if p.returncode != 0 or not dst.exists():
        raise RuntimeError("codec conversion failed:\n"
                           + (p.stderr or p.stdout)[-1200:])
    return dst


def require() -> None:
    a = availability()
    if a["usable"]:
        return
    missing = [k for k in ("cli", "converter", "codec") if not a[f"{k}_present"]]
    raise GGUFUnavailable(
        "qwen3-tts-cpp is not usable on this machine: missing "
        + ", ".join(missing)
        + ".\nBuild it with `bash scripts/build_gguf_runtime.sh`, or "
          "point QWEN_TTS_CPP / QWEN_TTS_CLI at an existing build. It is "
          "the recipe's only render path: nothing renders until it exists.")


FP_VERSION = 2


def _digest_one(f: Path) -> str:
    """Content digest of one file. Read in large blocks; this runs on 3.9 GB files.

    Never raises. A verifier that takes the render down with it is worse than the
    stale-cache risk it exists to remove, so a failure degrades to an honest
    "unknown", which compares unequal to everything and forces a re-convert.
    """
    try:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(16 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return "unknown"


def source_fingerprint(checkpoint: Path, content: bool = True) -> dict:
    """Content identity for a checkpoint's weights, plus where they came from.

    WHAT THE PREVIOUS VERSION COULD NOT SEE, and why it was replaced. It was
    {n_files, bytes, mtime_ns} over a NON-recursive glob, which left three holes:

      * SIZE IS NOT IDENTITY. Within EXP-021 alone, `exp021-b4-spk-1ep-voiced` and its
        own `-lr0-` control both export `model.safetensors` at 3,833,402,520 bytes and
        differ only in content. A corrupt download measured on 2026-09-07 had that same
        byte count too. Size cannot separate an arm from its control, let alone from a
        broken copy.
      * THE KEY IS A BASENAME, SO TWO ROOTS COLLIDE. The same checkpoint name existed
        under a weights directory on /mnt (trained locally) and one under $HOME (fetched
        from GCS). One cache entry, two different models, and `source` was recorded but
        never compared. The path is now part of the identity.
      * ONLY THE TOP LEVEL WAS COUNTED. `glob` missed `lora_adapter/` and
        `speech_tokenizer/`, so a changed tokenizer or adapter was invisible.

    mtime is gone entirely, not merely demoted: it is not derived from content, so it is
    equal for things that differ and differs for things that are equal. See the note
    above `_digest_one` for why it could not survive even as a cache key.

    Pass content=False for the cheap shape only -- used by the self-test to show what
    the old rule would have concluded.
    """
    files = sorted(checkpoint.rglob("*.safetensors"))
    entries = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        e = {"rel": f.relative_to(checkpoint).as_posix(), "bytes": st.st_size}
        if content:
            e["sha256"] = _digest_one(f)
        entries.append(e)
    fp = {"version": FP_VERSION,
          "source": str(Path(checkpoint).resolve()),
          "n_files": len(entries),
          "bytes": sum(e["bytes"] for e in entries),
          "files": entries}
    return fp


def _digest_unknown(fp: dict) -> bool:
    """Whether any file in this fingerprint failed to hash.

    An unknown digest must never satisfy a cache hit. Two fingerprints that both
    failed to read would otherwise compare equal and serve a model nobody checked.
    """
    return any(f.get("sha256") == "unknown" for f in fp.get("files", []))


def _why_miss(recorded: dict | None, fp: dict) -> str:
    """Name WHICH difference forced the re-conversion. A cache miss the operator
    cannot explain is one they will disable."""
    if recorded is None:
        return "no fingerprint was recorded, so it cannot be shown to match"
    if recorded.get("version") != FP_VERSION:
        return ("its fingerprint is v%s, which was size and mtime only and could not "
                "tell an arm from its own control -- re-verifying by content"
                % recorded.get("version", "1"))
    if _digest_unknown(fp):
        return "a source file could not be hashed, so the cache cannot be trusted"
    if recorded.get("source") != fp.get("source"):
        return ("it was converted from %s and this is %s -- same name, different "
                "checkpoint" % (recorded.get("source"), fp.get("source")))
    was = {f["rel"]: f.get("sha256") for f in recorded.get("files", [])}
    now = {f["rel"]: f.get("sha256") for f in fp.get("files", [])}
    if set(was) != set(now):
        return "the set of weight files changed (%d -> %d)" % (len(was), len(now))
    changed = sorted(r for r in now if was.get(r) != now[r])
    if changed:
        return "the weights changed since it was converted: " + ", ".join(changed)
    return "the source no longer matches what was converted"


def ensure_converted(checkpoint: str | Path, key: str | None = None,
                     quant: str = "q8_0", force: bool = False) -> Path:
    """Convert `checkpoint` to GGUF if it is not already cached. Returns the DIR.

    The directory is what the CLI wants (`-m`), and it holds exactly one talker so
    the scorer has nothing to choose between. Conversion is ~41 s; the cache is keyed
    by `key`, so re-rendering an already-converted arm costs nothing.
    """
    require()
    checkpoint = Path(checkpoint)
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"no config.json under {checkpoint}")
    if not key:
        # `.../exp000-spk1ep-arm-voiced/final_merged` should key on the arm, not on the
        # word "final_merged", which every checkpoint shares.
        key = (checkpoint.parent.name if checkpoint.name == "final_merged"
               else checkpoint.name)

    out_dir = GGUF_CACHE / key
    talker = out_dir / talker_gguf_name(key, quant)
    stamp = out_dir / "converted.json"

    # A CACHE HIT IS NOT ENOUGH; THE SOURCE HAS TO BE THE SAME SOURCE. The key is the
    # checkpoint's directory name, and a lane that is resumable invites exactly the
    # collision: "delete the export to redo a phase" retrains IN PLACE under the same
    # name, and a cache keyed on existence alone then serves the PREVIOUS model's
    # weights to the next render. Nothing in the audio says so, and the result gets
    # recorded against the new checkpoint's name.
    fp = source_fingerprint(checkpoint)
    if talker.exists() and not force:
        recorded = None
        if stamp.exists():
            try:
                recorded = json.loads(stamp.read_text(encoding="utf-8")).get("source_fingerprint")
            except json.JSONDecodeError:
                recorded = None
        if recorded == fp and not _digest_unknown(fp):
            return out_dir
        print(f"[gguf] re-converting {key}: {_why_miss(recorded, fp)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    conv = CPP_ROOT / "scripts" / "convert_tts_to_gguf.py"
    t0 = time.time()
    print(f"[gguf] converting {key} -> {talker.name} ({quant}), ~40 s", flush=True)
    p = subprocess.run(
        [_converter_python(), str(conv), "--input", str(checkpoint),
         "--output", str(talker), "--type", quant],
        capture_output=True, text=True)
    if p.returncode != 0 or not talker.exists():
        raise RuntimeError(
            f"GGUF conversion of {key} failed (rc={p.returncode}):\n"
            + (p.stderr or p.stdout)[-1200:])
    el = time.time() - t0

    # The codec is shared and identical for every talker; keep one copy beside them.
    codec = ensure_codec()
    if codec.parent != GGUF_CACHE:
        GGUF_CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(codec, GGUF_CACHE / CODEC_NAME)

    stamp.write_text(json.dumps({
        "key": key, "source": str(checkpoint), "quant": quant,
        "talker": talker.name, "convert_seconds": round(el, 1),
        "bytes": talker.stat().st_size,
        # What the next call compares against to decide the cache is still valid.
        "source_fingerprint": fp,
    }, indent=1), encoding="utf-8")
    print(f"[gguf] converted {key} in {el:.0f} s "
          f"({talker.stat().st_size / 1e9:.2f} GB)", flush=True)
    return out_dir


def render(gguf_dir: str | Path, text: str, *, speaker: str | None = None,
           ref_audio: str | Path | None = None, ref_text: str | None = None,
           max_edge_silence: float | None = MAX_EDGE_S, seed: int | None = None,
           temperature: float | None = None, top_k: int | None = None,
           top_p: float | None = None, max_tokens: int | None = None,
           language: str | None = None, timeout: float = 300.0,
           _extra: list[str] | None = None):
    """Synthesise one line. Returns (samples, sample_rate, info).

    `info` carries `raw_seconds` and `trimmed_seconds` because `audio_trim` insists
    a metric computed on untrimmed audio stay comparable -- see note 2 above.
    """
    require()
    gguf_dir = Path(gguf_dir)
    codec = ensure_codec()
    cmd = [str(CLI_BIN), "-m", str(gguf_dir), "--codec-model", str(codec),
           "-t", text]
    if speaker:
        cmd += ["--speaker", speaker]
    if ref_audio:
        cmd += ["-r", str(ref_audio)]
        if ref_text:
            cmd += ["--reference-text", ref_text]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if top_k is not None:
        cmd += ["--top-k", str(top_k)]
    if top_p is not None:
        cmd += ["--top-p", str(top_p)]
    if max_tokens is not None:
        cmd += ["--max-tokens", str(max_tokens)]
    if language:
        cmd += ["-l", language]
    # flags assembled by GGUFModel (speaker / reference / sampler)
    cmd += list(_extra or [])

    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix="gguf-render-") as td:
        # The CLI emits WAV and nothing else. This one is decoded and dropped inside
        # this block; only FLAC ever reaches the caller's disk.
        scratch = Path(td) / "out.wav"          # audio-format: scratch
        t0 = time.time()
        p = subprocess.run(cmd + ["-o", str(scratch)], capture_output=True,
                           text=True, timeout=timeout)
        el = time.time() - t0
        if p.returncode != 0 or not scratch.exists():
            raise RuntimeError(
                f"qwen3-tts-cli failed (rc={p.returncode}) on {text[:24]!r}:\n"
                + (p.stderr or p.stdout)[-800:])
        y, sr = sf.read(str(scratch))

    if y.ndim > 1:
        y = y.mean(axis=1)
    raw_s = len(y) / sr
    info = {"raw_seconds": raw_s, "wall_seconds": el, "backend": "gguf"}
    if max_edge_silence is not None:
        from audio_trim import trim_edges
        y, rep = trim_edges(np.asarray(y), sr, max_edge_s=max_edge_silence)
        info["trim"] = rep
    info["trimmed_seconds"] = len(y) / sr
    return y, sr, info


def _self_test() -> int:
    bad = 0

    def check(name, cond, detail=""):
        nonlocal bad
        bad += not cond
        print(f"  {'PASS' if cond else '*** FAIL ***'}  {name}"
              + (f"   {detail}" if detail and not cond else ""))

    n = talker_gguf_name("exp000-spk1ep-arm-voiced")
    check("converted talker carries the substring the CLI scores on",
          TALKER_PREFIX in n, n)
    check("and keeps the checkpoint identity in the name", "exp000-spk1ep-arm-voiced" in n, n)
    # POLICY-CKPT-NAMING-001: the suffix IS the warning. If sanitisation mangled
    # `-FAILED` or `-voiced` the artefact would lose the thing that says do not ship it.
    for label in ("release-cc0-v1-1donor-baked", "blend3-voicedbase-FAILED",
                  "cp-cc0-voiced", "night-cc0v22-lang-1ep-keepenc"):
        check(f"the checkpoint name survives verbatim in the gguf: {label}",
              label in talker_gguf_name(label), talker_gguf_name(label))
    check("a key with awkward characters is still a legal filename",
          talker_gguf_name("a b/c:d").endswith(".gguf")
          and "/" not in talker_gguf_name("a b/c:d"),
          talker_gguf_name("a b/c:d"))

    m = GGUFModel(Path("/nonexistent"), "probe")
    f = m._flags({"do_sample": False, "max_new_tokens": 1200})
    check("do_sample=False pins a seed", "--seed" in f, str(f))
    check("do_sample=False never goes greedy -- greedy does not terminate",
          "--temperature" not in f, str(f))
    check("max_new_tokens reaches the CLI as --max-tokens",
          "--max-tokens" in f and "1200" in f, str(f))

    f2 = GGUFModel(Path("/nonexistent"), "probe")._flags(
        {"temperature": 0.9, "top_k": 50, "seed": 7})
    check("sampler settings are passed through, not dropped",
          f2.count("--temperature") == 1 and "--top-k" in f2 and "--seed" in f2,
          str(f2))

    f3 = GGUFModel(Path("/nonexistent"), "probe")._flags({"do_sample": False,
                                                          "seed": 99})
    check("an explicit seed wins over the default",
          f3.count("--seed") == 1 and "99" in f3, str(f3))

    # An unmapped kwarg must be REPORTED. A sampler sweep that silently does nothing
    # is a day of this project's history.
    import io as _io
    import contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        GGUFModel(Path("/nonexistent"), "probe")._flags({"nonsense_kwarg": 1})
    check("an unmapped generate kwarg is reported, not dropped in silence",
          "nonsense_kwarg" in buf.getvalue(), repr(buf.getvalue()))

    check("the edge-silence default matches the torch path's natural edges",
          MAX_EDGE_S <= 0.2, str(MAX_EDGE_S))

    print("  NOT TESTED here (needs a converted checkpoint and a GPU): seeded output is byte-identical,")
    print("  greedy runs to the token cap, resident output equals one-shot. The lab's --with-render ran them.")
    # THE CACHE-IDENTITY CONTROLS. These exist because the cache could serve the wrong
    # model: the key is a bare checkpoint basename and the fingerprint behind it was
    # size + mtime. Every check asserts BOTH directions -- a rule that is always
    # unequal would pass "detects a difference" on its own and tell you nothing -- and
    # the content checks hold the PATH FIXED, so that a path difference cannot be what
    # makes them pass. (An earlier draft of these controls got exactly that wrong and
    # a mutation run caught it.)
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        root = Path(_td)
        a = root / "armA" / "final_merged"
        (a / "speech_tokenizer").mkdir(parents=True)
        (a / "config.json").write_text("{}", encoding="utf-8")
        (a / "model.safetensors").write_bytes(bytes([1]) * 4096)
        (a / "speech_tokenizer" / "model.safetensors").write_bytes(bytes([170]) * 512)

        def _v1(ck):
            """What the replaced rule saw: top level only, size only."""
            files = sorted(ck.glob("*.safetensors"))
            return {"n_files": len(files),
                    "bytes": sum(f.stat().st_size for f in files)}

        fa, v1_before = source_fingerprint(a), _v1(a)
        check("a checkpoint matches itself -- the check is not simply always red",
              source_fingerprint(a) == fa)
        check("nested weight files are part of the identity",
              fa["n_files"] == 2 and any(
                  f["rel"].startswith("speech_tokenizer/") for f in fa["files"]),
              str([f["rel"] for f in fa["files"]]))

        # The real shape: EXP-021's arm and its own lr=0 control export
        # model.safetensors at the SAME byte count, differing only in content. Same
        # directory, same size -- content is the only thing that can tell them apart.
        (a / "model.safetensors").write_bytes(bytes([2]) * 4096)
        fa2, v1_after = source_fingerprint(a), _v1(a)
        check("SAME PATH, SAME SIZE, DIFFERENT WEIGHTS is a cache MISS "
              "(an arm against its own lr=0 control)", fa != fa2,
              "identical byte counts at one path must not compare equal")
        check("...and the miss names the weights, not something vague",
              "weights changed" in _why_miss(fa, fa2), _why_miss(fa, fa2))
        # THE REGRESSION CONTROL FOR THE DIGEST CACHE THAT USED TO LIVE HERE. It keyed
        # on (path, size, mtime_ns) on the premise that a write always moves mtime.
        # Filesystems stamp mtime at a timestamp tick, so two writes in one tick carry
        # the same mtime_ns and a same-size rewrite went unseen.
        #
        # os.utime PINS mtime rather than racing the clock. The original bug only
        # appeared when two writes happened to land in one tick, which made it
        # intermittent and machine-dependent -- it passed here and failed on a peer's
        # box. Forcing the timestamp makes the control fire on every machine, every
        # run. A control whose outcome depends on wall-clock granularity is not a
        # control.
        _keep = a / "model.safetensors"
        _t = _keep.stat().st_mtime_ns
        _keep.write_bytes(bytes([3]) * 4096)
        os.utime(_keep, ns=(_t, _t))
        check("SAME PATH, SAME SIZE, AND MTIME PINNED IDENTICAL is still a miss",
              _keep.stat().st_mtime_ns == _t
              and source_fingerprint(a) != fa2,
              "content must be read, never inferred from size+mtime")
        _keep.write_bytes(bytes([2]) * 4096)   # restore what fa2 describes

        check("RED TEST: the replaced size-only rule called that rewrite IDENTICAL",
              v1_before == v1_after,
              "if this fails the fixture no longer reproduces the defect")

        # A changed tokenizer with the top-level weights untouched -- invisible to the
        # non-recursive glob the old rule used.
        (a / "speech_tokenizer" / "model.safetensors").write_bytes(bytes([187]) * 512)
        check("a changed tokenizer alone is a miss, top-level weights equal",
              source_fingerprint(a) != fa2 and _v1(a) == v1_after)

        # Two roots, one basename -- /mnt/d/... and /home/... both held
        # `exp021-b4-spk-1ep-voiced`. `source` was recorded but never compared.
        c = root / "otherRoot" / "armA" / "final_merged"
        (c / "speech_tokenizer").mkdir(parents=True)
        for rel in ("model.safetensors", "speech_tokenizer/model.safetensors"):
            (c / rel).write_bytes((a / rel).read_bytes())
        fc, fa3 = source_fingerprint(c), source_fingerprint(a)
        check("IDENTICAL CONTENT AT A DIFFERENT PATH is still a miss "
              "(same name, two weight roots)", fa3 != fc)
        check("...and that miss names the path, having nothing else to name",
              "different checkpoint" in _why_miss(fa3, fc), _why_miss(fa3, fc))
        check("the two roots really did hash the same -- the miss is the path, "
              "not the bytes",
              [f["sha256"] for f in fa3["files"]] == [f["sha256"] for f in fc["files"]])

        check("an unreadable file never satisfies a cache hit",
              _digest_unknown({"files": [{"rel": "x", "sha256": "unknown"}]})
              and not _digest_unknown(fa3))
        check("a v1 fingerprint is refused rather than trusted",
              "v1" in _why_miss({"version": 1}, fa3), _why_miss({"version": 1}, fa3))

    print("  all controls behaved" if not bad
          else f"  {bad} control(s) misbehaved -- do not trust this backend")
    return 1 if bad else 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report what the toolchain has, and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the controls fire, without needing a GPU")
    ap.add_argument("--convert", type=Path, help="checkpoint dir to convert")
    ap.add_argument("--key", help="cache key for --convert")
    ap.add_argument("--quant", default="q8_0",
                    choices=["f16", "f32", "q8_0", "q4_k"])
    ap.add_argument("--text", help="render this line through --key")
    ap.add_argument("--speaker")
    ap.add_argument("--out", type=Path, help="FLAC to write for --text")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.check:
        a = availability()
        for k in ("cli", "converter", "codec"):
            print(f"  {'ok  ' if a[f'{k}_present'] else 'MISSING'}  {k:<10} "
                  f"{a[k]}")
        print(f"  cache {a['cache']}")
        print("GGUF backend usable" if a["usable"]
              else "GGUF backend NOT usable -- nothing renders until it is built")
        return 0 if a["usable"] else 1

    if args.convert:
        d = ensure_converted(args.convert, key=args.key, quant=args.quant)
        print(d)
        return 0

    if args.text:
        if not args.key:
            raise SystemExit("--text needs --key naming a converted checkpoint")
        y, sr, info = render(GGUF_CACHE / args.key, args.text,
                             speaker=args.speaker)
        print(f"  {info['trimmed_seconds']:.2f} s speech "
              f"({info['raw_seconds']:.2f} s raw) in {info['wall_seconds']:.2f} s")
        if args.out:
            from audio_uri import write_flac
            print("  ->", write_flac(args.out, y, sr))
        return 0

    ap.print_help()
    return 1


# torch sampler kwargs -> CLI flags. Anything absent from here is REPORTED, never
# dropped in silence: this project has already spent a day on a sampler sweep that
# turned out to be a no-op, and a quietly ignored temperature would do it again.
_SAMPLER_MAP = {
    "temperature": "--temperature",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "repetition_penalty": "--repetition-penalty",
    "max_new_tokens": "--max-tokens",
    "seed": "--seed",
}
# `-l` takes an ISO-ish code; the PyTorch path is called with "chinese". Cantonese
# has no code of its own here -- it comes from the baked speaker, not the flag.
_LANG_MAP = {"chinese": "zh", "cantonese": "zh", "english": "en", "auto": "auto"}


class GGUFModel:
    """A converted checkpoint, rendered through qwen3-tts-cli, one text per call.

    EDGE SILENCE IS NORMALISED HERE, not left to the caller. The CLI pads every clip
    with ~0.8 s lead and ~2.1 s tail (note 2 in this module's docstring) where the
    PyTorch path emits 0.12 s. That pad is an artefact of the CLI, not of the model,
    so it is removed at the backend boundary -- which means `raw_duration_s` upstream
    is "as the model produced it, with the CLI's fixed pad taken off", and a clip from
    either backend sounds like it starts and ends the same way. Leaving it to a caller
    flag was the alternative, and a caller that forgets it silently ships an A/B page
    whose arms are distinguishable by dead air alone.
    """

    def __init__(self, gguf_dir: Path, key: str, max_edge_s: float = MAX_EDGE_S,
                 seed: int = DEFAULT_SEED):
        self.dir = Path(gguf_dir)
        self.key = key
        self.max_edge_s = max_edge_s
        self.seed = seed
        self._warned: set[str] = set()
        self._proc = None            # resident qwen3-tts-cli, started on demand
        self._proc_key = None
        self._log = None
        self._no_serve = False       # set after a resident failure, see _render_many

    def _flags(self, kw: dict) -> list[str]:
        out: list[str] = []
        for k, v in kw.items():
            if v is None:
                continue
            if k in _SAMPLER_MAP:
                out += [_SAMPLER_MAP[k], str(v)]
            elif k in ("do_sample", "subtalker_dosample"):
                # NEVER map do_sample=False to `--temperature 0`. Greedy decoding on
                # this CLI does not terminate, exactly as OKF E3 records for the torch
                # stack: measured 2026-09-05, one 20-character line ran to the full
                # 1200-token cap -- 96.0 s of near-silence, 23.8 s of compute -- and
                # reproduced identically on all three attempts of the retry budget,
                # which is how it was caught. The retries cannot help, because it is
                # not a bad draw.
                #
                # AND IT IS TEXT-DEPENDENT, which is what makes it dangerous rather
                # than merely bad. Discovered by a typo in this module's own probe
                # text: one character changed (嗰 -> 唷, U+55F0 -> U+5537) and the
                # same greedy settings returned a normal 7.8 s clip instead of the
                # 96 s runaway. So a greedy render that comes back fine is NOT
                # evidence that greedy is safe here -- it is evidence about that one
                # line. Hence a blanket refusal rather than a warning.
                #
                # Repeatability comes from `--seed` instead, and on this backend that
                # is strictly better than the torch path manages: the same seed gives
                # BYTE-IDENTICAL output (verified, matching md5 over two runs) and
                # still terminates normally (85 frames, 2.3 s). tts_api's docstring
                # says determinism is unavailable without breaking termination -- true
                # of the torch stack, and not a limitation this backend inherits.
                if v is False and k == "do_sample":
                    # Repeatability is what do_sample=False is asking for, and a
                    # pinned seed delivers it. If the caller already supplied one,
                    # it is already satisfied -- say nothing.
                    if "seed" not in kw:
                        out += ["--seed", str(self.seed)]
                elif k not in self._warned:
                    self._warned.add(k)
                    print(f"[gguf] {k}={v} has no qwen3-tts-cli equivalent and is "
                          f"NOT being applied -- this render is not what the torch "
                          f"backend would produce with it", flush=True)
            elif k not in self._warned:
                self._warned.add(k)
                print(f"[gguf] ignoring unmapped generate kwarg {k}={v!r}",
                      flush=True)
        return out

    # ---- resident process ------------------------------------------------
    # WHY. Synthesis is not what this binary spends its time on. Measured on one
    # short line: 1205 ms inside, 6.13 s of wall -- the rest is process start, 2.3 GB
    # of model file, and CUDA context init. One process per utterance pays that
    # every single time, so a ten-clip passage spends longer loading than speaking.
    # `--serve` (this project's patch in external/qwen3-tts-cpp/patches/) loads once and
    # takes one request per line, and the batch already arrives here as a list.
    #
    # THE FALLBACK IS NOT DECORATION. A machine built from stock upstream has no
    # `--serve`, and the honest behaviour there is to be slower, not to be broken.

    def _startup_flags(self, kw: dict) -> list[str]:
        """Sampler flags fixed for the process. The seed is NOT one of them.

        `do_sample` is dropped too, not just `seed`: `_flags` expresses
        do_sample=False by EMITTING a seed, and in resident mode the seed travels
        per request instead. Filtering the emitted list would have to remove a flag
        and its value as a pair -- an earlier version removed only the `--seed`
        token and left `20260905` behind as a stray positional argument, at which
        point the CLI prints its usage and exits, and the only visible symptom is
        the fallback engaging with a help screen quoted back as the error.
        """
        return self._flags({k: v for k, v in kw.items()
                            if k not in ("seed", "do_sample")})

    def _seed_for(self, kw: dict) -> int:
        if kw.get("seed") is not None:
            return int(kw["seed"])
        # do_sample=False asks for repeatability; anything else still gets a fixed
        # seed here, because a resident session must not make a clip depend on how
        # many clips preceded it.
        return int(self.seed)

    def _ensure_proc(self, startup: list[str]):
        key = tuple(startup)
        if (self._proc is not None and self._proc_key == key
                and self._proc.poll() is None):
            return self._proc
        self.close()
        codec = ensure_codec()
        cmd = [str(CLI_BIN), "-m", str(self.dir), "--codec-model", str(codec),
               "--serve"] + startup
        # stderr to a FILE, never a pipe nobody drains: the CLI is chatty per
        # request and a full pipe buffer would deadlock the render, which is a
        # hang rather than an error and the worst way for this to fail.
        self._log = tempfile.NamedTemporaryFile(prefix="gguf-serve-", suffix=".log",
                                                delete=False, mode="w")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self._log,
                                      text=True, encoding="utf-8", bufsize=1)
        self._proc_key = key
        return self._proc

    def close(self) -> None:
        """Stop the resident process. Safe to call repeatedly."""
        if getattr(self, "_proc", None) is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=20)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            self._proc_key = None
        if getattr(self, "_log", None) is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _serve_log_tail(self, n: int = 600) -> str:
        try:
            return Path(self._log.name).read_text(errors="replace")[-n:]
        except Exception:
            return "(no log)"

    def _render_serve(self, texts, startup: list[str], seed: int):
        """One process, one line per text. Returns (list_of_arrays, sample_rate)."""
        import numpy as np
        import soundfile as sf

        proc = self._ensure_proc(startup)
        ws, sr = [], None
        with tempfile.TemporaryDirectory(prefix="gguf-serve-out-") as td:
            for i, t in enumerate(texts):
                # The framing is line-oriented, so a newline inside a request would
                # desynchronise every later reply. Texts are single lines here; this
                # is the guard that keeps that true rather than assumed.
                clean = " ".join(str(t).splitlines()).replace("\t", " ")
                dst = Path(td) / f"r{i:04d}.wav"      # audio-format: scratch
                proc.stdin.write(f"{dst}\t{seed}\t{clean}\n")
                proc.stdin.flush()
                reply = proc.stdout.readline()
                if not reply:
                    raise RuntimeError(
                        "qwen3-tts-cli --serve exited early (rc="
                        f"{proc.poll()}). Last log:\n{self._serve_log_tail()}")
                parts = reply.rstrip("\n").split("\t")
                if parts[0] != "OK":
                    raise RuntimeError(
                        f"qwen3-tts-cli --serve refused {clean[:24]!r}: "
                        f"{reply.strip()}\n{self._serve_log_tail(300)}")
                y, s = sf.read(str(dst))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if self.max_edge_s is not None:
                    from audio_trim import trim_edges
                    y, _rep = trim_edges(np.asarray(y), s,
                                         max_edge_s=self.max_edge_s)
                ws.append(np.asarray(y))
                sr = s
        return ws, sr

    def _render_many(self, texts, language, extra: list[str], kw: dict):
        import numpy as np
        lang = _LANG_MAP.get(str(language).lower(), None)
        startup = list(extra) + self._startup_flags(kw)
        if lang:
            startup += ["-l", lang]

        if serve_supported() and not self._no_serve:
            try:
                return self._render_serve(texts, startup, self._seed_for(kw))
            except Exception as e:
                # Fall back for the REST OF THIS OBJECT'S LIFE, not just this call:
                # whatever broke the resident process (a binary without the flag, a
                # crash on load) will break it again on the next batch, and retrying
                # per batch would pay the failure repeatedly.
                self._no_serve = True
                self.close()
                print(f"[gguf] resident mode failed ({type(e).__name__}: "
                      f"{str(e)[:200]}); falling back to one process per clip",
                      flush=True)

        ws, sr = [], None
        for t in texts:
            y, s, _info = render(self.dir, t, max_edge_silence=self.max_edge_s,
                                 language=lang, _extra=extra + self._flags(kw))
            ws.append(np.asarray(y))
            sr = s
        return ws, sr

    def generate_custom_voice(self, *, text, speaker, language, **kw):
        spk = speaker[0] if isinstance(speaker, (list, tuple)) else speaker
        lang = language[0] if isinstance(language, (list, tuple)) else language
        return self._render_many(text, lang, ["--speaker", str(spk)], kw)

    def generate_voice_design(self, *, text, instruct, language, **kw):
        """A VoiceDesign checkpoint: the voice comes from a text description (qwen3-tts-cli --instruction)."""
        ins = instruct[0] if isinstance(instruct, (list, tuple)) else instruct
        lang = language[0] if isinstance(language, (list, tuple)) else language
        return self._render_many(text, lang, ["--instruction", str(ins)], kw)

    def generate_voice_clone(self, *, text, language, ref_audio, ref_text, **kw):
        ra = ref_audio[0] if isinstance(ref_audio, (list, tuple)) else ref_audio
        rt = ref_text[0] if isinstance(ref_text, (list, tuple)) else ref_text
        lang = language[0] if isinstance(language, (list, tuple)) else language
        extra = ["-r", str(ra)] + (["--reference-text", str(rt)] if rt else [])
        return self._render_many(text, lang, extra, kw)


if __name__ == "__main__":
    raise SystemExit(main())
