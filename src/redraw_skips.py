#!/usr/bin/env python3
"""Re-draw the chunks of a long-prose read that skip text, until the ASR hears the whole script: for FINAL listening
pages, never for measurement.

The owner, 2026-10-19: first-draw reads, skips and all, are "right as references, not as the final listen tests". A
final listen judges reading quality, and a clause the model never said is not a reading to judge. So this takes a
render_story.py output (which records every chunk's span), runs every chunk through Qwen3-ASR, and for each chunk with
a lost run (skipped_text.py: 4+ script characters not heard) renders it again at render_story's own
seed rule with k = 1, 2, ... (the same seeds render_story --redraw would use) until a draw loses nothing, at most
MAX_TRIES times; if none does, the draw that lost least is kept and the chunk is marked. Every other chunk, and the
silence between chunks, is copied sample for sample. A redrawn chunk is trimmed as render_story trims (EDGE_S) and
scaled to the read's median chunk loudness.

The skip counts of the FIRST draws stay the measurement (release gate A8, B4); this file only builds what the owner
listens to, and its manifest says which chunks it changed:

    python src/redraw_skips.py --renders DIR --out DIR --arm NAME=GGUF_DIR:SPEAKER [--arm ...] [--two-recognisers]

An arm not named with --arm is checked but copied unchanged (it cannot be redrawn without its model).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/: render_story, skipped_text, steps, gguf_backend
MAX_TRIES = 4


def active_rms(y: np.ndarray) -> float:
    a = y[np.abs(y) > 1e-3]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0


def main() -> int:
    import soundfile as sf
    import render_story as rs
    import skipped_text as sk
    import steps
    from audio_trim import trim_edges
    from audio_uri import write_flac
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[], help="NAME=GGUF_DIR:SPEAKER, an arm that may be redrawn")
    ap.add_argument("--two-recognisers", action="store_true",
                    help="a chunk loses text when Qwen3-ASR AND SenseVoice both miss a character at the same position, "
                         "no length floor (2026-10-21: planted cuts of one syllable are caught 37%% of the time this way "
                         "and 0%% by runs of 4; a no-op reads 2 of 210)")
    a = ap.parse_args()
    if (a.out / "manifest.json").exists():
        raise SystemExit(f"{a.out} already holds a manifest")
    man = json.loads((a.renders / "manifest.json").read_text(encoding="utf-8"))
    salt = man["rule"].get("seed_salt") or ""
    models = {}
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        gguf, speaker = rest.rsplit(":", 1)
        models[name] = (Path(gguf).expanduser(), speaker)
    asr = steps._asr("cuda:0")

    sv = steps._sensevoice("cuda:0") if a.two_recognisers else None

    def heard(y, sr) -> str:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as t:
            write_flac(Path(t.name), y, sr)
            out = asr.transcribe(t.name, language="Cantonese")[0].text
            if sv is not None:
                import re
                out = (out, re.sub(r"<\|[^|]*\|>", "", sv.generate(input=t.name, language="yue", use_itn=False)[0]["text"]))
        Path(t.name).unlink(missing_ok=True)
        return out

    def missing(ref, hyp):
        import difflib
        got = set()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
            if op == "delete":
                got.update(range(i1, i2))
            elif op == "replace" and (i2 - i1) > (j2 - j1):
                got.update(range(i2 - ((i2 - i1) - (j2 - j1)), i2))
        return got

    def losses(text, h):
        """[(start, length), ...] of what the chunk lost: runs of 4+ by default, or each character both recognisers
        missed with --two-recognisers."""
        if sv is None:
            return sk.lost_runs(text, h)
        ref = sk.cjk(text)
        return [(i, 1) for i in sorted(missing(ref, sk.cjk(h[0])) & missing(ref, sk.cjk(h[1])))]

    out_man = {**man, "rule": {**man["rule"], "skip_redrawn": True, "skip_redraw_max_tries": MAX_TRIES,
                               "skip_min_run": None if a.two_recognisers else sk.MIN_RUN,
                               "skip_detector": "two-recognisers" if a.two_recognisers else "qwen3-asr-runs",
                               "from_renders": str(a.renders)}, "arms": {}}
    for arm, am in man["arms"].items():
        gm = None
        if arm in models:
            import gguf_backend as gb
            gm = gb.GGUFModel(models[arm][0], models[arm][0].name, max_edge_s=None)
        new_arm = {**am, "stories": {}}
        try:
            for sid, st in am["stories"].items():
                pcm, sr = sf.read(str(a.renders / st["flac"]), dtype="float32")
                rows = st["chunks"]
                pieces = [pcm[r["span"][0]:r["span"][1]].copy() for r in rows]
                gaps = [rows[i + 1]["span"][0] - rows[i]["span"][1] for i in range(len(rows) - 1)]
                target = float(np.median([active_rms(p) for p in pieces if active_rms(p) > 0]))
                new_rows = []
                for i, r in enumerate(rows):
                    lost = losses(r["text"], heard(pieces[i], sr))
                    rec = {**r, "first_draw_lost_runs": [k for _, k in lost], "skip_redraws": 0}
                    if lost and gm is not None:
                        best = (sum(k for _, k in lost), pieces[i], r["seed"], 0)
                        for k in range(1, MAX_TRIES + 1):
                            seed = rs.seed_of(sid, r["i"], r["text"], k, salt)
                            ws, sr2 = gm.generate_custom_voice(text=[r["text"]], speaker=models[arm][1], language="chinese",
                                                               do_sample=False, seed=seed)
                            y, _ = trim_edges(np.asarray(ws[0], dtype="float32"), sr2, max_edge_s=rs.EDGE_S)
                            y = np.asarray(y, dtype="float32")
                            if active_rms(y) > 0:
                                y = np.clip(y * target / active_rms(y), -0.999, 0.999).astype("float32")
                            got = sum(n for _, n in losses(r["text"], heard(y, sr2)))
                            if got < best[0]:
                                best = (got, y, seed, k)
                            if got == 0:
                                break
                        pieces[i] = best[1]
                        rec.update({"seed": best[2], "skip_redraws": best[3], "lost_after": best[0]})
                    elif lost:
                        rec["lost_after"] = sum(k for _, k in lost)
                        rec["not_redrawn"] = "no model given for this arm"
                    new_rows.append(rec)
                parts, at = [], 0
                for i, (p, rec) in enumerate(zip(pieces, new_rows)):
                    rec["span"] = [at, at + len(p)]
                    parts.append(p)
                    at += len(p)
                    if i < len(gaps):
                        parts.append(np.zeros(gaps[i], dtype="float32"))
                        at += gaps[i]
                read = np.concatenate(parts)
                f = a.out / arm / f"{sid}.flac"
                f.parent.mkdir(parents=True, exist_ok=True)
                write_flac(f, read, sr)
                n_red = sum(1 for x in new_rows if x["skip_redraws"])
                n_left = sum(1 for x in new_rows if x.get("lost_after"))
                new_arm["stories"][sid] = {**st, "chunks": new_rows, "flac": str(f.relative_to(a.out)),
                                           "seconds": round(len(read) / sr, 2), "skip_redrawn_chunks": n_red,
                                           "skip_left_chunks": n_left}
                print(f"[{time.strftime('%H:%M:%S')}] {arm} {sid}: {len(rows)} chunks, first-draw skips "
                      f"{sum(1 for x in new_rows if x['first_draw_lost_runs'])}, redrawn {n_red}, still skipping {n_left}",
                      flush=True)
        finally:
            if gm is not None:
                gm.close()
        out_man["arms"][arm] = new_arm
    (a.out / "manifest.json").write_text(json.dumps(out_man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
