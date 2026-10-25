#!/usr/bin/env python3
"""Render the held-out short lines on one GGUF voice, the release gate's Part A material and the checkpoint card's
short-line set: every line x draws 0..N-1, at a per-clip seed fixed by the line and the draw alone.

From the lab's render_ship_ab.py (the `render` half; its `measure` half is the lab's exp059 machinery and is not
here). Every arm, candidate and baseline, goes through this ONE code path with identical flags: the baked speaker,
the backend's seeded talker sampling, the sub-talker sampling left on, and NO edge trimming (the edge and length
measures are calibrated on untrimmed audio). A comparison whose arms took different render paths would be measuring
the path.

THE MATERIAL IS THE USER'S. The lab's set is EXP-059's held-out 200 sentences (pool-H.jsonl); it is not in this
repository, which holds no text. --lines is a JSONL of {"digest": <any stable id>, "text": <the line>}, one row per
sentence; the digest seeds the clip (seed = int(sha256(f"{digest}:{draw}")[:8], 16)), so the same file renders the
same seeds on every arm. Candidate and baseline must be rendered from the same file.

The output is <out>/<ARM>/manifest.jsonl and clips/<digest>-d<draw>.flac; the manifest's rows carry the text, seed,
raw duration and the speech span (speech_s, from audio_trim at max_edge_s 0) the gate's runaway test reads.

    python src/render_short.py --arm D1 --gguf GGUF_DIR --speaker hk_voice_343 --lines LINES.jsonl --out OUT
    python src/render_short.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SEED_RULE = "int(sha256(f'{digest}:{draw}')[:8], 16), passed per request as the GGUF seed"
MIN_FREE_MIB = int(os.environ.get("CARD_FLOOR_MIB", 8500))


def seed_for(digest: str, draw: int) -> int:
    """A clip's seed: fixed by its sentence and draw alone, never by render order."""
    return int(hashlib.sha256(f"{digest}:{draw}".encode()).hexdigest()[:8], 16)


def free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
        total, used = (int(x) for x in out.split(","))
        return total - used
    except Exception:
        return None


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def talker(gguf_dir: Path) -> Path:
    hits = sorted(gguf_dir.glob("qwen-talker-*.gguf"))
    if len(hits) != 1:
        raise SystemExit(f"{gguf_dir}: {len(hits)} talker GGUFs, exactly one expected")
    return hits[0]


def plan(rows: list[dict], draws: int, done: set[str]) -> list[tuple[dict, int]]:
    """(sentence, draw) pairs still to render, in a fixed order, skipping clips already in the manifest."""
    return [(r, d) for r in sorted(rows, key=lambda r: r["digest"]) for d in range(draws)
            if f"{r['digest']}-d{d}" not in done]


def render(arm: str, gguf_dir: Path, lines: Path, out: Path, draws: int, speaker: str, expect: int | None) -> int:
    import numpy as np
    import gguf_backend as gb
    from audio_trim import trim_edges, silence_report
    from audio_uri import write_flac
    rows = [json.loads(l) for l in lines.read_text(encoding="utf-8").splitlines() if l.strip()]
    if expect and len(rows) != expect:
        raise SystemExit(f"{lines}: {len(rows)} sentences, {expect} expected")
    if len({r["digest"] for r in rows}) != len(rows):
        raise SystemExit(f"{lines}: digests are not unique")
    d = out / arm
    (d / "clips").mkdir(parents=True, exist_ok=True)
    man = d / "manifest.jsonl"
    done = {json.loads(l)["clip_id"] for l in man.read_text(encoding="utf-8").splitlines() if l.strip()} if man.exists() else set()
    todo = plan(rows, draws, done)
    tg = talker(gguf_dir)
    ident = {"arm": arm, "gguf_dir": str(gguf_dir), "talker": tg.name, "talker_md5": md5(tg), "speaker": speaker,
             "seed_rule": SEED_RULE, "max_edge_s": None, "lines": str(lines),
             "lines_sha256": hashlib.sha256(lines.read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    runf = d / "run.json"
    if runf.exists():
        prev = json.loads(runf.read_text(encoding="utf-8"))
        diff = [k for k in ("gguf_dir", "talker_md5", "speaker", "max_edge_s", "lines_sha256") if prev.get(k) != ident[k]]
        if diff:
            raise SystemExit(f"{d} was rendered with different {diff}; render into a new --out")
    runf.write_text(json.dumps({**ident, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1) + "\n",
                    encoding="utf-8")
    print(f"{arm}: {len(todo)} to render, {len(done)} done", flush=True)
    if not todo:
        return 0
    free = free_mib()
    if free is None or free < MIN_FREE_MIB:
        raise SystemExit(f"REFUSING: {free} MiB free on the card, floor {MIN_FREE_MIB} (CARD_FLOOR_MIB)")
    m = gb.GGUFModel(gguf_dir, gguf_dir.name, max_edge_s=None)
    t0 = time.time()
    try:
        with open(man, "a", encoding="utf-8") as mf:
            for i, (r, k) in enumerate(todo, 1):
                seed = seed_for(r["digest"], k)
                ws, sr = m.generate_custom_voice(text=[r["text"]], speaker=speaker, language="chinese",
                                                 do_sample=False, seed=seed)
                y = np.asarray(ws[0], dtype="float32")
                flac = d / "clips" / f"{r['digest']}-d{k}.flac"
                write_flac(flac, y, sr)
                _, edge = trim_edges(y, sr, max_edge_s=0.0)
                gap = silence_report(y, sr)
                speech = y[edge["start_sample"]:edge["end_sample"]].astype("float64")
                mf.write(json.dumps({
                    "clip_id": f"{r['digest']}-d{k}", "digest": r["digest"], "draw": k, "seed": seed, "text": r["text"],
                    "flac": flac.relative_to(d).as_posix(), "raw_duration_s": round(len(y) / sr, 3),
                    "lead_silence_s": round(edge["lead_silence_s"], 3), "tail_silence_s": round(edge["tail_silence_s"], 3),
                    "speech_s": round(len(speech) / sr, 3), "max_gap_s": round(gap["max_gap_s"], 3),
                    "speech_rms": round(float(np.sqrt(np.mean(speech ** 2))) if len(speech) else 0.0, 5),
                    "backend": "gguf", "model": gguf_dir.name, "talker_md5": ident["talker_md5"]},
                    ensure_ascii=False) + "\n")
                mf.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"[{time.strftime('%H:%M:%S')}] {arm} {i}/{len(todo)}, {(time.time() - t0) / i:.2f} s/clip",
                          flush=True)
    finally:
        m.close()
    return 0


def self_test() -> int:
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        fails += not cond
    rows = [{"digest": d} for d in ("c", "a", "b")]
    p = plan(rows, 2, set())
    check("the plan is every sentence x draw, in digest order",
          [(r["digest"], k) for r, k in p] == [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("c", 0), ("c", 1)])
    check("RED: clips already in the manifest are not rendered again", len(plan(rows, 2, {"a-d0", "c-d1"})) == 4)
    check("both arms get the same seed for a sentence and draw (a function of digest and draw only)",
          seed_for("abc", 1) == seed_for("abc", 1) and seed_for("abc", 1) != seed_for("abc", 0))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        g = Path(td)
        try:
            talker(g)
            check("RED: a GGUF directory with no talker refuses", False)
        except SystemExit:
            check("RED: a GGUF directory with no talker refuses", True)
        (g / "qwen-talker-x-q8_0.gguf").write_bytes(b"x")
        check("one talker is found", talker(g).name == "qwen-talker-x-q8_0.gguf")
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--speaker", required=True, help="the baked voice to render")
    ap.add_argument("--lines", type=Path, required=True, help='JSONL of {"digest", "text"}')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--expect", type=int, help="refuse unless the file holds exactly this many sentences")
    a = ap.parse_args()
    return render(a.arm, a.gguf.expanduser(), a.lines.expanduser(), a.out.expanduser(), a.draws, a.speaker, a.expect)


if __name__ == "__main__":
    sys.exit(main())
