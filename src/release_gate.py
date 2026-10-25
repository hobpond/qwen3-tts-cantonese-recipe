#!/usr/bin/env python3
"""The release gate's Part A (short lines), self-contained: one candidate voice against the baseline (v0.0.1) on the
same held-out lines, the same seeds and the same backend. The tests may VETO; they never close. Passing makes a
candidate eligible for the owner's ear, not better.

Codified by the owner on 2026-10-19 (the lab's release-gate concept). This file states the tests and thresholds of
that concept as code, reading the store layers the recipe already makes. It is NOT the lab's reference
implementation (exp093 -> exp071/exp073/exp089/exp097, an experiment chain), and where the two differ that is said
below rather than hidden.

    A1  edges        edges v2 peak_db_rel_speech, lead and trail, paired by clip; Wilcoxon one-sided that the
                     candidate is LOWER, Bonferroni over 4 tests (2 voices x 2 ends) at 0.05      PRIMARY
    A2  length bars  NOT RUN HERE. The lab's bars are exp059's calibrated duration bars (render_ship_ab.py measure),
                     part of the experiment chain. Reported as not run; the verdict says Part A is incomplete.
    A3  last char    Qwen3-ASR last_char_match lost, exact McNemar one-sided, worse at p < 0.05     VETO
    A4  CER          Qwen3-ASR CER per clip: Wilcoxon one-sided higher at p < 0.05 AND median more than 0.03 above
                     the baseline's (the concept's rule). The lab's exp071 vetoed on the Wilcoxon alone, so EXP-099
                     D1's "CER veto" (median 0.100 against 0.083, +0.017) does not fire under the codified rule.
                     The script-folded CER (OpenCC t2s, 2026-10-22) is REPORTED beside it; whether A4 vetoes on the
                     folded CER is the owner's decision, open as of 2026-10-22                     VETO
    A5  tone         tonefit nearest-template match (align sensevoice-ctc, syllables contour=voiced, the donor's
                     own templates), last syllable excluded; paired bootstrap, 2000 resamples, seed 20261018; 95%
                     interval of (candidate - baseline) wholly below zero                          VETO
    A6  runaways     ARTICULATED time (the speech span less its internal pauses of 0.25 s or more, audio_trim's
                     silence_report on the clip's FLAC) longer than twice the baseline's on the same clip: more than
                     the baseline's count the other way + 5. Until 2026-10-24 A6 read the whole speech span, so one
                     pause in a 1-second line doubled it: EXP-108 seed 1's 343 vetoed on 11 such clips, every one
                     carrying a 0.5-1.9 s pause, 4 left without its longest pause. The span count is REPORTED  VETO
    A9  pauses       clips with an internal pause of 1 s or more (rules pause_s), exact McNemar one-sided, more than
                     the baseline's at p < 0.05: a stall mid-line is its own failure, so it is counted on its own
                     rather than folded into the runaway count or dropped from it                   VETO
    A7  liveliness   loudness range (p10-p90 dB over voiced frames inside the edges core): median paired difference
                     at least 0.5 dB narrower at two-sided Wilcoxon p < 0.0125; pitch range and rate REPORTED  VETO
    A8  skipped text skipped_text.lost_runs (4+ script characters not heard) per clip: more clips with a lost run
                     than the baseline, exact McNemar one-sided p < 0.05                            VETO

    DAMAGE_VETO          any VETO fires that the owner has not waived
                         (rules["waived"]: a waived test is still measured, and its firing is reported as
                         waived_vetoes, never dropped)
    OBJECTIVELY_BETTER   no veto, and A1 shows the candidate quieter at either end
    NOT_SHOWN            no veto, A1 does not

Inputs are two store collections of the same clip ids (render_short.py's renders, ingested by recipe.py) and their
render manifests (speech_s for A6). `--measure` first runs any missing layer (asr and align need the GPU); without
it the gate only reads.

    python src/release_gate.py --candidate short-D1 --baseline short-V1 --candidate-manifest DIR/D1/manifest.jsonl \\
        --baseline-manifest DIR/V1/manifest.jsonl --templates donor --out gate.json [--measure]
    python src/release_gate.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
RULES = {"primary_alpha": 0.05, "primary_tests": 4, "veto_alpha": 0.05, "cer_median_margin": 0.03,
         "tone": {"resamples": 2000, "seed": 20261018, "ci": 0.95}, "runaway_factor": 2.0, "runaway_slack": 5,
         "loud_veto_db": -0.5, "loud_alpha": 0.0125, "skip_alpha": 0.05, "pause_s": 1.0, "pause_alpha": 0.05,
         "waived": []}
LAYERS = (("energy", {}), ("edges", {}), ("f0", {}), ("g2p", {}), ("align", {"aligner": "sensevoice-ctc"}),
          ("syllables", {"contour": "voiced"}))


# ---------------------------------------------------------------------------------------------- statistics
def wilcoxon_one_sided(x: list[float]) -> float:
    """One-sided signed-rank p for median(x) > 0, normal approximation with ties averaged (the lab's exp059)."""
    x = [v for v in x if v != 0]
    n = len(x)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(x[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(x[order[j + 1]]) == abs(x[order[i]]):
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, v in zip(ranks, x) if v > 0)
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return 0.5 * math.erfc((w - mu) / sd / math.sqrt(2))


def mcnemar_worse(x_bad: dict, y_bad: dict) -> dict:
    """Exact McNemar, one-sided: is the candidate (x) bad on more clips than the baseline (y)?"""
    ids = sorted(set(x_bad) & set(y_bad))
    b = sum(1 for c in ids if x_bad[c] and not y_bad[c])
    c_ = sum(1 for c in ids if y_bad[c] and not x_bad[c])
    n = b + c_
    p = sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0
    return {"x": sum(bool(x_bad[c]) for c in ids), "y": sum(bool(y_bad[c]) for c in ids), "only_x": b, "only_y": c_,
            "p_one_sided": round(p, 5)}


def tone_bootstrap(x: dict, y: dict, rules: dict = RULES["tone"]) -> dict:
    """x, y: clip -> [hits, syllables]. The pooled match rate's difference and its paired bootstrap interval."""
    ids = sorted(c for c in x if c in y and x[c][1] and y[c][1])
    hx, nx = np.array([x[c] for c in ids]).T
    hy, ny = np.array([y[c] for c in ids]).T
    rng = np.random.default_rng(rules["seed"])
    d = hx.sum() / nx.sum() - hy.sum() / ny.sum()
    bs = []
    for _ in range(rules["resamples"]):
        i = rng.integers(0, len(ids), len(ids))
        bs.append(hx[i].sum() / nx[i].sum() - hy[i].sum() / ny[i].sum())
    lo, hi = np.percentile(bs, [50 * (1 - rules["ci"]), 100 - 50 * (1 - rules["ci"])])
    return {"clips": len(ids), "x": round(float(hx.sum() / nx.sum()), 4), "y": round(float(hy.sum() / ny.sum()), 4),
            "diff_x_minus_y": round(float(d), 4), "ci": [round(float(lo), 4), round(float(hi), 4)], "vetoed": bool(hi < 0)}


def live_pair(x: dict, y: dict) -> dict:
    from scipy.stats import wilcoxon
    ids = sorted(set(x) & set(y))
    res = {"clips": len(ids)}
    for k in ("pitch_range_st", "loud_range_db", "rate_cps"):
        d = np.array([x[i][k] - y[i][k] for i in ids])
        res[k] = {"median_x": round(float(np.median([x[i][k] for i in ids])), 3),
                  "median_y": round(float(np.median([y[i][k] for i in ids])), 3),
                  "median_diff_x_minus_y": round(float(np.median(d)), 3),
                  "p_two_sided": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}
    return res


# ---------------------------------------------------------------------------------------------- per-clip reads
def _store():
    import store as st
    import steps  # noqa: F401
    return st


def ensure_layers(coll: str, templates: str) -> None:
    st = _store()
    from cli import resolve_params
    s = st.Store(coll)
    s.run(st.REGISTRY["asr"], params={"language": "Cantonese", "device": "cuda:0"}, jobs=1)
    for name, params in LAYERS:
        s.run(st.REGISTRY[name], params=params, jobs=1 if name == "align" else 4)
    s.run(st.REGISTRY["tonefit"], params=resolve_params(coll, "tonefit", {"templates_from": templates}), jobs=4)


def liveliness(coll: str) -> dict:
    """Per clip, inside the edges core: pitch range p5-p95 of voiced F0 (st), loudness range p10-p90 of the 10 ms
    level over voiced frames (dB), characters per second (the lab's exp074.liveliness)."""
    st = _store()
    import steps
    s = st.Store(coll)
    ed, ing = s.records("edges", errors="drop"), s.records("ingest")
    out = {}
    for cid, r in ed.items():
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
        stv = 12 * np.log2(f[v] / np.median(f[v]))
        out[cid] = {"pitch_range_st": float(np.percentile(stv, 95) - np.percentile(stv, 5)),
                    "loud_range_db": float(np.percentile(db[v], 90) - np.percentile(db[v], 10)),
                    "rate_cps": len(steps.cjk_only(ing[cid].get("text", ""))) / max(0.1, r["core_end_s"] - r["core_start_s"])}
    return out


def tone_counts(coll: str, templates: str) -> dict:
    st = _store()
    c = st.Store(coll)
    m = c.layer_meta("tonefit")
    want = st.Store(templates).refs().get("speaker")
    sy = c.layer_meta("syllables", m["parents"]["syllables"])
    al = c.layer_meta("align", sy["parents"]["align"])
    if (m["params"].get("templates_from"), m["params"].get("templates_key")) != (templates, want) \
            or sy["params"].get("contour") != "voiced" or al["params"].get("aligner") != "sensevoice-ctc":
        raise SystemExit(f"{coll}: tonefit is not on {templates}'s templates {want}, contour voiced, sensevoice-ctc")
    return {cid: [sum(f["target"] == f["nearest"] for f in r["fits"][:-1] if f and f["target"]),
                  sum(1 for f in r["fits"][:-1] if f and f["target"])] for cid, r in c.records("tonefit").items()}


def folded_cer(coll: str) -> dict:
    import opencc
    import skipped_text as sk
    from checkpoint_card import _lev
    st = _store()
    t2s = opencc.OpenCC("t2s")
    s = st.Store(coll)
    asr, ing = s.records("asr", errors="drop"), s.records("ingest")
    out = {}
    for cid, r in asr.items():
        if cid in ing and "hyp" in r:
            ref = sk.cjk(ing[cid]["text"])
            out[cid] = _lev(sk.cjk(t2s.convert(ing[cid]["text"])), sk.cjk(t2s.convert(r["hyp"]))) / max(1, len(ref))
    return out


# ---------------------------------------------------------------------------------------------- A6 and A9
def pauses(row: dict) -> dict:
    """A render's internal pauses, from its FLAC (manifest rows carry `_dir`, the manifest's folder): the longest, and
    the total of those of 0.25 s or more, inside the speech span (audio_trim.silence_report)."""
    import soundfile as sf
    from audio_trim import silence_report
    y, sr = sf.read(str(Path(row["_dir"]) / row["flac"]), dtype="float32")
    g = silence_report(y, sr)
    return {"max_gap_s": g["max_gap_s"], "gaps_total_s": g["gaps_total_s"]}


def runaways_and_pauses(cand_rows: list[dict], base_rows: list[dict], rules: dict = RULES, measure=pauses) -> tuple:
    """A6 on articulated time and A9 on long pauses, paired by clip id (`measure` is replaceable for the self-test)."""
    ids = sorted(r["clip_id"] for r in cand_rows)
    rx, ry = {r["clip_id"]: r for r in cand_rows}, {r["clip_id"]: r for r in base_rows}
    px, py = {c: measure(rx[c]) for c in ids}, {c: measure(ry[c]) for c in ids}
    art = lambda r, p: max(r["speech_s"] - p["gaps_total_s"], 0.0)
    f = rules["runaway_factor"]
    over = sum(art(rx[c], px[c]) > f * art(ry[c], py[c]) for c in ids)
    under = sum(art(ry[c], py[c]) > f * art(rx[c], px[c]) for c in ids)
    s_over = sum(rx[c]["speech_s"] > f * ry[c]["speech_s"] for c in ids)
    s_under = sum(ry[c]["speech_s"] > f * rx[c]["speech_s"] for c in ids)
    a6 = {"measure": "articulated seconds (speech span less internal pauses of 0.25 s or more)",
          "candidate_over_2x": over, "baseline_over_2x": under, "vetoed": over > under + rules["runaway_slack"],
          "span_REPORTED": {"candidate_over_2x": s_over, "baseline_over_2x": s_under,
                            "would_veto": s_over > s_under + rules["runaway_slack"]}}
    long_ = lambda p: p["max_gap_s"] >= rules["pause_s"]
    m9 = mcnemar_worse({c: long_(px[c]) for c in ids}, {c: long_(py[c]) for c in ids})
    a9 = {"pause_s": rules["pause_s"], **m9, "vetoed": m9["p_one_sided"] < rules["pause_alpha"],
          "pause_total_s_median_candidate": round(float(np.median([px[c]["gaps_total_s"] for c in ids])), 3),
          "pause_total_s_median_baseline": round(float(np.median([py[c]["gaps_total_s"] for c in ids])), 3)}
    return a6, a9


# ---------------------------------------------------------------------------------------------- the gate
def gate(cand: str, base: str, cand_rows: list[dict], base_rows: list[dict], templates: str, rules: dict = RULES) -> dict:
    import skipped_text as sk
    st = _store()
    ids = sorted(r["clip_id"] for r in cand_rows)
    if ids != sorted(r["clip_id"] for r in base_rows):
        raise SystemExit("the candidate and the baseline are not the same clip ids")
    sx, sy = st.Store(cand), st.Store(base)
    ex, ey = sx.records("edges", errors="drop"), sy.records("edges", errors="drop")
    ax, ay = sx.records("asr"), sy.records("asr")
    res = {"clips": len(ids), "candidate": cand, "baseline": base, "rules": rules, "tests": {}}
    T = res["tests"]
    alpha = rules["primary_alpha"] / rules["primary_tests"]
    T["A1_edges"] = {}
    for end in ("lead", "trail"):
        pr = [(ey[c][f"{end}_level"]["peak_db_rel_speech"], ex[c][f"{end}_level"]["peak_db_rel_speech"]) for c in ids
              if c in ex and c in ey and ex[c].get("speech") and ey[c].get("speech")
              and ex[c][f"{end}_level"]["peak_db_rel_speech"] is not None
              and ey[c][f"{end}_level"]["peak_db_rel_speech"] is not None]
        d = [b - a for b, a in pr]
        med = lambda v: round(sorted(v)[len(v) // 2], 2) if v else None
        p = wilcoxon_one_sided(d)
        T["A1_edges"][end] = {"n": len(d), "median_baseline": med([b for b, _ in pr]), "median_candidate": med([a for _, a in pr]),
                              "median_diff_baseline_minus_candidate": med(d), "p_candidate_lower": round(p, 6),
                              "alpha": alpha, "candidate_lower": p < alpha}
    T["A2_length_bars"] = {"run": False, "why": "the lab's exp059 duration bars are not part of the recipe (TODO)"}
    lost = lambda a, c: not a[c]["last_char_match"]
    m3 = mcnemar_worse({c: lost(ax, c) for c in ids}, {c: lost(ay, c) for c in ids})
    T["A3_last_char"] = {**m3, "vetoed": m3["p_one_sided"] < rules["veto_alpha"]}
    dc = [ax[c]["cer"] - ay[c]["cer"] for c in ids]
    mx, my = float(np.median([ax[c]["cer"] for c in ids])), float(np.median([ay[c]["cer"] for c in ids]))
    p4 = wilcoxon_one_sided(dc)
    fx, fy = folded_cer(cand), folded_cer(base)
    T["A4_cer"] = {"median_candidate": round(mx, 4), "median_baseline": round(my, 4), "p_candidate_higher": round(p4, 6),
                   "median_diff_candidate_minus_baseline": round(sorted(dc)[len(dc) // 2], 4),
                   "wilcoxon_fires": p4 < rules["veto_alpha"], "median_margin_fires": mx - my > rules["cer_median_margin"],
                   "vetoed": p4 < rules["veto_alpha"] and mx - my > rules["cer_median_margin"],
                   "folded_REPORTED": {"median_candidate": round(float(np.median([fx[c] for c in ids if c in fx])), 4),
                                       "median_baseline": round(float(np.median([fy[c] for c in ids if c in fy])), 4),
                                       "p_candidate_higher": round(wilcoxon_one_sided(
                                           [fx[c] - fy[c] for c in ids if c in fx and c in fy]), 6)}}
    T["A5_tone"] = tone_bootstrap(tone_counts(cand, templates), tone_counts(base, templates), rules["tone"])
    T["A6_runaways"], T["A9_pauses"] = runaways_and_pauses(cand_rows, base_rows, rules)
    live = live_pair(liveliness(cand), liveliness(base))
    lr = live["loud_range_db"]
    T["A7_liveliness"] = {**live, "vetoed": lr["median_diff_x_minus_y"] <= rules["loud_veto_db"]
                          and lr["p_two_sided"] < rules["loud_alpha"]}
    tx = {c: r["text"] for c, r in sx.records("ingest").items()}
    lx = {c: bool(sk.lost_runs(tx[c], ax[c].get("hyp", ""))) for c in ids}
    ly = {c: bool(sk.lost_runs(tx[c], ay[c].get("hyp", ""))) for c in ids}
    m8 = mcnemar_worse(lx, ly)
    T["A8_skipped"] = {**m8, "vetoed": m8["p_one_sided"] < rules["skip_alpha"]}
    res["verdict"] = verdict(T, rules.get("waived", ()))
    res["waived_vetoes"] = [k for k in rules.get("waived", ()) if T.get(k, {}).get("vetoed")]
    res["part_a_complete"] = False
    return res


def verdict(T: dict, waived=()) -> str:
    vetoes = [k for k in ("A3_last_char", "A4_cer", "A5_tone", "A6_runaways", "A7_liveliness", "A8_skipped", "A9_pauses")
              if T[k]["vetoed"] and k not in waived]
    if vetoes:
        return "DAMAGE_VETO:" + ",".join(vetoes)
    return "OBJECTIVELY_BETTER" if any(T["A1_edges"][e]["candidate_lower"] for e in ("lead", "trail")) else "NOT_SHOWN"


def self_test() -> int:
    """The statistics and the verdict on synthetic inputs, both ways (CPU, seconds)."""
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
        fails += not cond
    rng = np.random.default_rng(1)
    d = list(rng.normal(0.5, 1.0, 400))
    check("Wilcoxon: a positive shift is significant", wilcoxon_one_sided(d) < 0.001)
    check("RED: Wilcoxon: the same shift negated is not", wilcoxon_one_sided([-v for v in d]) > 0.999)
    check("Wilcoxon: a no-op (all zeros) reads p = 1", wilcoxon_one_sided([0.0] * 50) == 1.0)
    xb = {f"c{i}": i < 30 for i in range(400)}
    yb = {f"c{i}": i < 10 for i in range(400)}
    check("McNemar: 20 clips bad only on the candidate is worse", mcnemar_worse(xb, yb)["p_one_sided"] < 0.001)
    check("RED: McNemar: equal counts are not", mcnemar_worse(yb, yb)["p_one_sided"] == 1.0)
    x = {f"c{i}": [3, 7] for i in range(200)}
    y = {f"c{i}": [5, 7] for i in range(200)}
    check("tone: a lower match rate vetoes", tone_bootstrap(x, y)["vetoed"])
    check("RED: tone: the same rate does not", not tone_bootstrap(y, y)["vetoed"])
    ok = {"vetoed": False}
    base = {k: ok for k in ("A3_last_char", "A4_cer", "A5_tone", "A6_runaways", "A7_liveliness", "A8_skipped", "A9_pauses")}
    lower = {"lead": {"candidate_lower": True}, "trail": {"candidate_lower": False}}
    none = {"lead": {"candidate_lower": False}, "trail": {"candidate_lower": False}}
    check("verdict: no veto and quieter at an end -> OBJECTIVELY_BETTER", verdict({**base, "A1_edges": lower}) == "OBJECTIVELY_BETTER")
    check("verdict: no veto, not quieter -> NOT_SHOWN", verdict({**base, "A1_edges": none}) == "NOT_SHOWN")
    check("RED: verdict: one veto beats a primary win",
          verdict({**base, "A7_liveliness": {"vetoed": True}, "A1_edges": lower}) == "DAMAGE_VETO:A7_liveliness")
    a7 = {**base, "A7_liveliness": {"vetoed": True}, "A1_edges": none}
    check("a waived veto does not decide the verdict", verdict(a7, ("A7_liveliness",)) == "NOT_SHOWN")
    check("RED: a waiver names one test; another veto still decides",
          verdict({**a7, "A3_last_char": {"vetoed": True}}, ("A7_liveliness",)) == "DAMAGE_VETO:A3_last_char")
    # A6 and A9 on synthetic speech: a pause added to a clip is not a runaway but is a long pause; speech said three
    # times is a runaway and no pause; the same audio both sides is neither.
    import audio_trim
    sr = 24000
    tone = lambda s: (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(s * sr)) / sr)).astype("float32")
    gap = lambda s: np.zeros(int(s * sr), dtype="float32")
    clip = np.concatenate([gap(0.3), tone(0.4), gap(0.15), tone(0.5), gap(0.3)])
    g0 = audio_trim.silence_report(clip, sr)
    check("silence_report: a 0.15 s phrase break is not a pause", g0["gaps_total_s"] == 0.0, str(g0))
    pz = np.concatenate([clip[: int(0.5 * sr)], gap(1.5), clip[int(0.5 * sr):]])   # inside the first tone
    g1 = audio_trim.silence_report(pz, sr)
    check("silence_report: a 1.5 s pause is measured", abs(g1["gaps_total_s"] - 1.5) < 0.05 and g1["max_gap_s"] >= 1.0, str(g1))
    waves = {"plain": clip, "paused": pz, "thrice": np.concatenate([clip[: int(1.35 * sr)]] * 3)}

    def meas(r):
        g = audio_trim.silence_report(waves[r["w"]], sr)
        return {"max_gap_s": g["max_gap_s"], "gaps_total_s": g["gaps_total_s"]}

    def rows(w, n=40):
        _, e = audio_trim.trim_edges(waves[w], sr, max_edge_s=0.0)
        return [{"clip_id": f"c{i}", "w": w, "speech_s": (e["end_sample"] - e["start_sample"]) / sr} for i in range(n)]
    a6, a9 = runaways_and_pauses(rows("paused"), rows("plain"), measure=meas)
    check("A6: a pause in every clip is no runaway", not a6["vetoed"] and a6["candidate_over_2x"] == 0, str(a6))
    check("RED: A6 span reading would have vetoed it", a6["span_REPORTED"]["would_veto"], str(a6["span_REPORTED"]))
    check("A9: a pause in every clip vetoes", a9["vetoed"], str(a9))
    a6, a9 = runaways_and_pauses(rows("thrice"), rows("plain"), measure=meas)
    check("A6: speech said three times vetoes", a6["vetoed"], str(a6))
    check("RED: A9: speech said three times is no long pause", not a9["vetoed"], str(a9))
    a6, a9 = runaways_and_pauses(rows("plain"), rows("plain"), measure=meas)
    check("A6 and A9: the same audio both sides is neither", not a6["vetoed"] and not a9["vetoed"]
          and a6["candidate_over_2x"] == 0 and a9["p_one_sided"] == 1.0)
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--candidate", help="store collection of the candidate's short-line renders")
    ap.add_argument("--baseline", help="store collection of the baseline's (v0.0.1's) renders")
    ap.add_argument("--candidate-manifest", type=Path)
    ap.add_argument("--baseline-manifest", type=Path)
    ap.add_argument("--templates", help="the donor's raw collection (tonefit templates_from)")
    ap.add_argument("--measure", action="store_true", help="run missing layers first (asr and align need the GPU)")
    ap.add_argument("--rules", type=json.loads, default={}, help="JSON overriding RULES (recipe.py passes [gate])")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not all((a.candidate, a.baseline, a.candidate_manifest, a.baseline_manifest, a.templates)):
        ap.error("--candidate, --baseline, both manifests and --templates are required")
    if a.measure:
        for c in (a.candidate, a.baseline):
            ensure_layers(c, a.templates)
    rows = lambda p: [{**json.loads(l), "_dir": str(p.parent)} for l in p.read_text(encoding="utf-8").splitlines()
                      if l.strip()]
    rules = {**RULES, **a.rules, "tone": {**RULES["tone"], **a.rules.get("tone", {})}}
    res = gate(a.candidate, a.baseline, rows(a.candidate_manifest), rows(a.baseline_manifest), a.templates, rules)
    text = json.dumps(res, indent=1, ensure_ascii=False)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
