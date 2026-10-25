#!/usr/bin/env python3
"""The checkpoint card: every checkpoint this lab produces is measured on the same dimensions against the baseline
(v0.0.1), and the numbers travel with it (owner, 2026-10-21: "a checkpoint benchmark protocol. Each checkpoint we
generate should have these as part of the card").

Inputs are what every candidate already has: its 400 held-out short lines as a store collection (render_ship_ab.py,
EXP-059's ids and seeds) and, optionally, long-prose reads (render_story.py at chunk ceiling 45, first draws). Every
dimension is reported as checkpoint, baseline, the paired difference with a bootstrap 95% interval, and a STATUS:
`validated` (an instrument shown to move on a planted defect and to read no change on a no-op) or `exploratory` (not
yet so shown: report it, never decide on it). The card never decides: the release gate's tests may veto, the owner's
ear decides (the lab's release-gate and checkpoint-card concepts; README.md here states both).

    dropped      long prose: characters both Qwen3-ASR and SenseVoice miss at the same position, no length floor
                 (validated 2026-10-21: one-syllable cuts 37%, two-syllable 79%, no-op 1%)
    ends         short lines: loud end (p90) and soft end (p10) of voiced 10 ms levels and pitch p95 / p5, each from the
                 clip's median, inside the edges core (sidon-compresses-dynamics.md; checked here on the liveliness
                 plants live-plant-donor343-c80/c100, the A7 control)
    particles    short lines whose last character is a sentence-final particle: final F0 from the clip median, by the
                 particle's dictionary tone (high 1/2/5, low 3/4/6) (exploratory)
    tones        tonefit's nearest template per syllable: match rate, per tone, and the Hong Kong mergers 2<->5, 3<->6,
                 4<->6 (the overall match is release gate A5's measure, validated on the tone plants)
    voice        cepstral peak prominence, median over voiced frames inside the core (exploratory)
    speaker      ECAPA-TDNN cosine of each render to the donor's raw-clip centroid (checked here: held-out donor clips
                 against another voice)

    python src/checkpoint_card.py --name D1 --short short-D1 --baseline-short short-V1 --prose DIR:ARM \\
        --baseline-prose DIR:ARM --donor donor --templates donor --out DIR
    python src/checkpoint_card.py --self-test      (the card's arithmetic; the instrument controls are the lab's)
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/: store, steps, cli, skipped_text, audio_uri
SEED = 20261018
SFP = set("啦喇囉咩喎啊呀㗎嘅噃啩吖嘛咋咯嘞呢嗎吓噉添囖咧")
LAYERS = (("energy", {}), ("f0", {}), ("edges", {}), ("g2p", {}), ("align", {"aligner": "sensevoice-ctc"}),
          ("syllables", {"contour": "voiced"}), ("tails", {}), ("final", {}), ("cpp", {}))
STATUS = {"intelligibility": "validated (folded CER: the planted script switch reads no change)", "dropped": "validated", "ends": "validated", "particles": "exploratory", "tones": "validated (overall), "
          "exploratory (per tone)", "voice": "exploratory", "speaker": "validated"}


def boot(d: np.ndarray, stat=np.median, n: int = 2000) -> list[float]:
    """95% interval of the statistic of paired differences."""
    if len(d) < 5:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    m = stat(d[rng.integers(0, len(d), (n, len(d)))], axis=1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def paired(x: dict, y: dict, stat=np.median) -> dict:
    """stat=np.mean for coarse per-clip values (a tone match of 3 of 7): their median difference is almost always 0,
    which read as 'no change' on every arm on the first cards (2026-10-21)."""
    ids = sorted(set(x) & set(y))
    d = np.array([x[i] - y[i] for i in ids])
    return {"n": len(ids), "stat": stat.__name__,
            "checkpoint": round(float(stat(np.array([x[i] for i in ids]))), 3) if ids else None,
            "baseline": round(float(stat(np.array([y[i] for i in ids]))), 3) if ids else None,
            "diff": round(float(stat(d)), 3) if ids else None, "ci95": boot(d, stat)}


def ensure_layers(coll: str, templates: str) -> None:
    import store as st
    import steps  # noqa: F401
    from cli import resolve_params
    s = st.Store(coll)
    for name, params in LAYERS:
        s.run(st.REGISTRY[name], params=params, jobs=1 if name == "align" else 4)
    s.run(st.REGISTRY["tonefit"], params=resolve_params(coll, "tonefit", {"templates_from": templates}), jobs=4)


def ends(coll: str) -> dict:
    import store as st
    s = st.Store(coll)
    out = {}
    for cid, r in s.records("edges", errors="drop").items():
        if not r.get("speech"):
            continue
        f0, en = s.arrays("f0", cid), s.arrays("energy", cid)
        if f0 is None or en is None:
            continue
        a, b = int(r["core_start_s"] / 0.01), int(r["core_end_s"] / 0.01)
        f, db = f0["f0"][a:b], en["db"][a:b]
        n = min(len(f), len(db))
        f, db = f[:n], db[:n]
        v = ~np.isnan(f)
        if v.sum() < 20:
            continue
        d = db[v] - np.median(db[v])
        stv = 12 * np.log2(f[v] / np.median(f[v]))
        out[cid] = {"loud_p90": np.percentile(d, 90), "loud_p10": np.percentile(d, 10),
                    "pitch_p95": np.percentile(stv, 95), "pitch_p5": np.percentile(stv, 5)}
    return out


def particles(coll: str) -> dict:
    import store as st
    s = st.Store(coll)
    g2p, fin = s.records("g2p", errors="drop"), s.records("final", errors="drop")
    out = {}
    for cid, g in g2p.items():
        if g["chars"] and g["chars"][-1] in SFP and cid in fin and fin[cid].get("end_st") is not None:
            out[cid] = (g["tones"][-1], fin[cid]["end_st"])
    return out


def tones(coll: str) -> dict:
    import store as st
    out = {}
    for cid, r in st.Store(coll).records("tonefit", errors="drop").items():
        fits = [(f["target"], f["nearest"]) for f in r.get("fits", []) if f]
        if fits:
            out[cid] = fits
    return out


def tone_table(t: dict) -> dict:
    fits = [f for v in t.values() for f in v]
    res = {"syllables": len(fits), "match": round(float(np.mean([a == b for a, b in fits])), 4)}
    for k in range(1, 7):
        res[f"T{k}"] = round(float(np.mean([b == k for a, b in fits if a == k])), 3)
    for x, y in ((2, 5), (5, 2), (3, 6), (6, 3), (4, 6), (6, 4)):
        res[f"{x}->{y}"] = round(float(np.mean([b == y for a, b in fits if a == x])), 3)
    return res


def _lev(a, b) -> int:
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def intelligibility(coll: str, rewrite=None) -> dict:
    """Per clip: Qwen3-ASR's CER as stored (the gate's A4), and the CER with script and transcript folded to simplified
    (OpenCC t2s), so the recogniser writing Sidon audio in simplified characters (2026-10-22) stops counting as error.
    `rewrite` transforms the transcript first (the self-test's planted script switch)."""
    import opencc
    import store as st
    import skipped_text as sk
    t2s = opencc.OpenCC("t2s")
    s = st.Store(coll)
    asr, ing = s.records("asr", errors="drop"), s.records("ingest")
    out = {}
    for cid, r in asr.items():
        if cid not in ing or "hyp" not in r:
            continue
        hyp = rewrite(r["hyp"]) if rewrite else r["hyp"]
        ref = sk.cjk(ing[cid]["text"])
        raw = _lev(ref, sk.cjk(hyp)) / max(1, len(ref))
        fold = _lev(sk.cjk(t2s.convert(ing[cid]["text"])), sk.cjk(t2s.convert(hyp))) / max(1, len(ref))
        out[cid] = (raw, fold)
    return out


def voice(coll: str) -> dict:
    import store as st
    s = st.Store(coll)
    out = {}
    for cid, r in s.records("edges", errors="drop").items():
        c, f0 = s.arrays("cpp", cid), s.arrays("f0", cid)
        if not r.get("speech") or c is None or f0 is None:
            continue
        key = next(k for k in c if k != "prob") if c else None
        a, b = int(r["core_start_s"] / 0.01), int(r["core_end_s"] / 0.01)
        cc, f = c[key][a:b], f0["f0"][a:b]
        n = min(len(cc), len(f))
        v = ~np.isnan(f[:n]) & ~np.isnan(cc[:n])
        if v.sum() >= 20:
            out[cid] = float(np.median(cc[:n][v]))
    return out


_ECAPA = {}


def embed(paths: list[str]) -> np.ndarray:
    import librosa
    import torch
    if "enc" not in _ECAPA:
        from speechbrain.inference.speaker import EncoderClassifier
        _ECAPA["enc"] = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                        savedir=str(Path.home() / ".cache/speechbrain-ecapa"),
                                                        run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"})
    out = []
    for p in paths:
        y, _ = librosa.load(p, sr=16000, mono=True)
        with torch.no_grad():
            e = _ECAPA["enc"].encode_batch(torch.from_numpy(y).unsqueeze(0)).squeeze().cpu().numpy()
        out.append(e / np.linalg.norm(e))
    return np.stack(out)


def donor_centroid(donor: str, held_out_every: int = 5) -> tuple[np.ndarray, list[str]]:
    """The donor's raw clips: every fifth held out (for the control), the rest averaged."""
    import store as st
    ing = st.Store(donor).records("ingest")
    ids = sorted(ing)
    held = [ing[c]["path"] for i, c in enumerate(ids) if i % held_out_every == 0]
    rest = [ing[c]["path"] for i, c in enumerate(ids) if i % held_out_every]
    c = embed(rest).mean(0)
    return c / np.linalg.norm(c), held


def speaker(coll: str, centroid: np.ndarray) -> dict:
    import store as st
    ing = st.Store(coll).records("ingest")
    ids = sorted(ing)
    e = embed([ing[c]["path"] for c in ids])
    return {c: float(v @ centroid) for c, v in zip(ids, e)}


def _missing(ref, hyp) -> set:
    out = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if op == "delete":
            out.update(range(i1, i2))
        elif op == "replace" and (i2 - i1) > (j2 - j1):
            out.update(range(i2 - ((i2 - i1) - (j2 - j1)), i2))
    return out


_ASR = {}


def dropped(renders: Path, arm: str) -> dict:
    """Per chunk of a render_story read: CJK characters, and those both recognisers miss."""
    import soundfile as sf
    import skipped_text as sk
    import steps
    from audio_uri import write_flac
    if "q" not in _ASR:
        _ASR["q"], _ASR["s"] = steps._asr("cuda:0"), steps._sensevoice("cuda:0")
    man = json.loads((renders / "manifest.json").read_text(encoding="utf-8"))
    if man["rule"].get("max_redraw") not in (0, None) or man["rule"].get("skip_redrawn"):
        raise SystemExit(f"{renders}: not first draws ({man['rule']})")
    out = {}
    for sid, s in man["arms"][arm]["stories"].items():
        pcm, sr = sf.read(str(renders / s["flac"]), dtype="float32")
        for c in s["chunks"]:
            with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as t:
                write_flac(Path(t.name), pcm[c["span"][0]:c["span"][1]], sr)
                hq = _ASR["q"].transcribe(t.name, language="Cantonese")[0].text
                hs = re.sub(r"<\|[^|]*\|>", "", _ASR["s"].generate(input=t.name, language="yue", use_itn=False)[0]["text"])
            Path(t.name).unlink(missing_ok=True)
            ref = sk.cjk(c["text"])
            out[f"{sid}-{c['i']:03d}"] = (len(ref), len(_missing(ref, sk.cjk(hq)) & _missing(ref, sk.cjk(hs))))
    return {"rule": {k: man["rule"].get(k) for k in ("chunk_hi", "seed_salt")}, "chunks": out}


def drop_summary(d: dict) -> dict:
    ch = d["chunks"]
    chars, lost = sum(v[0] for v in ch.values()), sum(v[1] for v in ch.values())
    return {"chunks": len(ch), "chunks_with_drop": sum(v[1] > 0 for v in ch.values()), "chars": chars,
            "dropped": lost, "dropped_share": round(lost / max(chars, 1), 4)}


def card(a) -> dict:
    ensure_layers(a.short, a.templates)
    ensure_layers(a.baseline_short, a.templates)
    res = {"name": a.name, "made": time.strftime("%Y-%m-%d %H:%M"), "baseline": a.baseline_short, "status": STATUS,
           "short_lines": a.short, "dimensions": {}}
    D = res["dimensions"]
    e, eb = ends(a.short), ends(a.baseline_short)
    D["ends"] = {k: paired({i: v[k] for i, v in e.items()}, {i: v[k] for i, v in eb.items()})
                 for k in ("loud_p90", "loud_p10", "pitch_p95", "pitch_p5")}
    p, pb = particles(a.short), particles(a.baseline_short)
    D["particles"] = {g: paired({i: v[1] for i, v in p.items() if v[0] in ts}, {i: v[1] for i, v in pb.items() if v[0] in ts})
                      for g, ts in (("high_1_2_5", (1, 2, 5)), ("low_3_4_6", (3, 4, 6)))}
    t, tb = tones(a.short), tones(a.baseline_short)
    D["tones"] = {"checkpoint": tone_table(t), "baseline": tone_table(tb),
                  "match_per_clip": paired({i: np.mean([x == y for x, y in v]) for i, v in t.items()},
                                           {i: np.mean([x == y for x, y in v]) for i, v in tb.items()}, stat=np.mean)}
    it, ib = intelligibility(a.short), intelligibility(a.baseline_short)
    D["intelligibility"] = {"cer": paired({k: v[0] for k, v in it.items()}, {k: v[0] for k, v in ib.items()}, stat=np.mean),
                            "cer_folded": paired({k: v[1] for k, v in it.items()}, {k: v[1] for k, v in ib.items()}, stat=np.mean)}
    D["voice"] = {"cpp_db": paired(voice(a.short), voice(a.baseline_short))}
    if a.donor:
        cen, _ = donor_centroid(a.donor)
        D["speaker"] = {"ecapa_cosine_to_donor": paired(speaker(a.short, cen), speaker(a.baseline_short, cen))}
    if a.prose:
        D["dropped"] = {}
        cache = Path.home() / ".cache/checkpoint-card-dropped.json"           # outside the repo: per-chunk counts
        memo = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}

        def cached(d, arm):
            k = f"{Path(d).expanduser().resolve()}:{arm}"
            if k not in memo:
                memo[k] = dropped(Path(d).expanduser(), arm)
                cache.write_text(json.dumps(memo, ensure_ascii=False) + "\n", encoding="utf-8")
            return memo[k]
        for spec in a.prose:
            d, arm = spec.rsplit(":", 1)
            x = cached(d, arm)
            row = {"checkpoint": drop_summary(x), "rule": x["rule"]}
            if a.baseline_prose:
                bd, barm = a.baseline_prose.rsplit(":", 1)
                y = cached(bd, barm)
                if y["rule"] != x["rule"]:
                    raise SystemExit(f"baseline prose rendered at {y['rule']}, checkpoint at {x['rule']}")
                row["baseline"] = drop_summary(y)
            D["dropped"][Path(d).name] = row
    if a.gate:
        g = json.loads(Path(a.gate).expanduser().read_text(encoding="utf-8"))
        res["release_gate"] = {k: g[k] for k in ("release_verdict", "verdicts") if k in g}
    return res


def markdown(c: dict) -> str:
    D = c["dimensions"]
    L = [f"### Checkpoint card: {c['name']} (against {c['baseline']}, {c['made']})", "",
         "| Dimension | Measure | Checkpoint | Baseline | Difference [95% CI] | Status |", "|---|---|---|---|---|---|"]

    def row(dim, meas, r, unit=""):
        L.append(f"| {dim} | {meas} | {r['checkpoint']}{unit} | {r['baseline']}{unit} | {r['diff']:+}{unit} "
                 f"[{r['ci95'][0]}, {r['ci95'][1]}] (n {r['n']}) | {c['status'][dim]} |")
    for k, lab in (("loud_p90", "loud end (dB from median)"), ("loud_p10", "soft end (dB)"),
                   ("pitch_p95", "pitch high end (st)"), ("pitch_p5", "pitch low end (st)")):
        row("ends", lab, D["ends"][k])
    for k, lab in (("high_1_2_5", "high particles, final F0 (st)"), ("low_3_4_6", "low particles, final F0 (st)")):
        row("particles", lab, D["particles"][k])
    row("tones", "tone match per clip (mean)", D["tones"]["match_per_clip"])
    tc, tb = D["tones"]["checkpoint"], D["tones"]["baseline"]
    L.append(f"| tones | per tone T1-T6 | {' '.join(str(tc[f'T{k}']) for k in range(1, 7))} | "
             f"{' '.join(str(tb[f'T{k}']) for k in range(1, 7))} | | {c['status']['tones']} |")
    L.append(f"| tones | mergers 2>5 5>2 3>6 6>3 4>6 6>4 | {' '.join(str(tc[m]) for m in ('2->5', '5->2', '3->6', '6->3', '4->6', '6->4'))} | "
             f"{' '.join(str(tb[m]) for m in ('2->5', '5->2', '3->6', '6->3', '4->6', '6->4'))} | | {c['status']['tones']} |")
    row("intelligibility", "CER as gated (mean)", D["intelligibility"]["cer"])
    row("intelligibility", "CER, script folded (mean)", D["intelligibility"]["cer_folded"])
    row("voice", "CPP (dB)", D["voice"]["cpp_db"])
    if "speaker" in D:
        row("speaker", "ECAPA cosine to donor", D["speaker"]["ecapa_cosine_to_donor"])
    for name, r in D.get("dropped", {}).items():
        b = r.get("baseline", {})
        L.append(f"| dropped | {name}: characters both recognisers miss | {r['checkpoint']['dropped_share']:.2%} "
                 f"({r['checkpoint']['dropped']}/{r['checkpoint']['chars']}) | "
                 f"{(format(b['dropped_share'], '.2%') + ' (' + str(b['dropped']) + '/' + str(b['chars']) + ')') if b else '-'} | | validated |")
    if "release_gate" in c:
        L += ["", f"Release gate: {json.dumps(c['release_gate'], ensure_ascii=False)}"]
    L += ["", "The card reports; it never decides. Tests may veto (release gate), the owner's ear decides."]
    return "\n".join(L) + "\n"


def self_test() -> int:
    """The card's arithmetic on synthetic numbers, CPU only, seconds: a no-op reads zero, a planted shift is seen with
    an interval excluding zero, the tone table counts mergers, and the dropped-character alignment finds a deleted
    run and nothing in a faithful read.

    NOT HERE, and what the lab's --self-test ran instead: the instrument controls on stored collections (the A7
    liveliness plants live-plant-donor343-c80/c100, the tone plants tone-plant-donor343-k0/k6, the planted script
    switch on v0.0.1's renders, held-out donor clips against another voice). They need those collections, built by the
    lab's live_plant.py and tone_plant.py, which depend on its experiment chain and are not part of the recipe. Until
    they are rebuilt here, the card's `validated` statuses are the lab's, quoted, not re-shown by this copy."""
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
        fails += not cond
    rng = np.random.default_rng(0)
    x = {f"c{i}": float(v) for i, v in enumerate(rng.normal(0, 1, 200))}
    z = paired(x, x)
    check("no-op: a collection against itself reads zero", z["diff"] == 0.0 and z["ci95"] == [0.0, 0.0], str(z))
    y = {k: v - 1.0 for k, v in x.items()}
    r = paired(y, x)
    check("a planted -1.0 shift reads -1.0 with an interval below zero", r["diff"] == -1.0 and r["ci95"][1] < 0, str(r))
    check("RED: fewer than 5 pairs gives no interval", all(np.isnan(v) for v in paired({"a": 1.0}, {"a": 0.0})["ci95"]))
    t = {"c": [(2, 5), (2, 2), (3, 6), (4, 4)]}
    tt = tone_table(t)
    check("tone table: match rate and the 2->5 merger counted", tt["match"] == 0.5 and tt["2->5"] == 0.5, str(tt))
    ref = list("今日天氣好好所以我哋一齊去公園")
    check("dropped: a faithful read loses nothing", _missing(ref, ref) == set())
    check("RED: dropped: a deleted run of 3 is found where it was", _missing(ref, ref[:4] + ref[7:]) == {4, 5, 6})
    ds = drop_summary({"chunks": {"a": (10, 0), "b": (10, 2)}})
    check("drop summary counts chunks and characters", ds["chunks_with_drop"] == 1 and ds["dropped_share"] == 0.1, str(ds))
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--short", help="the checkpoint's 400 held-out short lines, a store collection")
    ap.add_argument("--baseline-short", required=False, help="the baseline's (v0.0.1's) short lines, a store collection")
    ap.add_argument("--prose", action="append", default=[], help="RENDERS_DIR:ARM, first-draw long prose")
    ap.add_argument("--baseline-prose", help="RENDERS_DIR:ARM, the baseline at the same ceiling and seeds")
    ap.add_argument("--donor", help="the voice's raw-clip collection, for speaker similarity")
    ap.add_argument("--templates", help="tone templates (tonefit templates_from): the donor's raw collection")
    ap.add_argument("--gate", help="the release gate's read.json, quoted on the card")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.name and a.short and a.baseline_short and a.templates and a.out):
        ap.error("--name, --short, --baseline-short, --templates and --out are required")
    a.out.mkdir(parents=True, exist_ok=True)
    c = card(a)
    (a.out / f"card-{a.name}.json").write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (a.out / f"card-{a.name}.md").write_text(markdown(c), encoding="utf-8")
    print(markdown(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
