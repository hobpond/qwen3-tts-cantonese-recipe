"""The detached-tail detector (EXP-059's registered rule, "union-v1"): the `tails` store step, and through it the
`final` step the checkpoint card's particle row reads.

Extracted from the lab's build_pairs.py (the DPO pair builder), whose pair-building code the recipe does not use.
The functions and constants below are unchanged, so steps.tails' code hash (store.Step.code_deps -> tail_code)
names the same detector.
"""
from __future__ import annotations

# A DETACHED TAIL (owner's screenshot, 2026-10-10): the sentence ends, the model waits (median 1.24 s), emits a
# short sound (median 0.08 s), then stops. 57 of 3,432 pool renders (1.7%). The render's speech span runs to the
# stray sound, so the length bars caught 24 of them only by accident, as runaway or stalled, and one reached the
# PoC's pairs as a CHOSEN clip. A functional failure, no judgement needed, so a metric may veto it.
TAIL_GAP_S = 0.4          # a pause at least this long before the last sound
TAIL_MAX_S = 0.6          # and the last sound no longer than this
TAIL_FLOOR_DB = -40.0     # active = within 40 dB of the clip's loudest 20 ms
TAIL_HOP_S = 0.02
TAIL_MERGE_S = 0.15       # gaps shorter than this are inside speech (stops, short pauses)


def active_segments(y, sr: int, floor_db: float = TAIL_FLOOR_DB) -> list[tuple[float, float]]:
    """(start_s, end_s) of the clip's active stretches, gaps under TAIL_MERGE_S merged, blips under 40 ms dropped."""
    import numpy as np
    hop = int(TAIL_HOP_S * sr)
    n = len(y) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(np.asarray(y[: n * hop], dtype=np.float64).reshape(n, hop) ** 2, axis=1) + 1e-12))
    act = db > db.max() + floor_db
    segs, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        if not a and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, n])
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) * TAIL_HOP_S < TAIL_MERGE_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s * TAIL_HOP_S, e * TAIL_HOP_S) for s, e in merged if (e - s) * TAIL_HOP_S >= 0.04]


def detached_tail_of(y, sr: int, floor_db: float = TAIL_FLOOR_DB, gap_s: float = TAIL_GAP_S,
                     max_s: float = TAIL_MAX_S) -> dict | None:
    """The energy rule alone. The panel (build_pool_page.py, v4) still marks tails with it."""
    segs = active_segments(y, sr, floor_db)
    if len(segs) < 2:
        return None
    (_, pe), (ts, te) = segs[-2], segs[-1]
    if ts - pe >= gap_s and te - ts <= max_s:
        return {"main_end_s": round(pe, 3), "tail_start_s": round(ts, 3), "tail_end_s": round(te, 3),
                "gap_s": round(ts - pe, 3), "dur_s": round(te - ts, 3)}
    return None


# THE VOICED RULE (owner's second screenshot, 2026-10-10, clip 3de003d9c14cf3c5-d3): a quiet voiced murmur after the
# pause, peaking at the energy rule's -40 dB, which the energy rule missed. 60 pool renders by this rule, 57 by the
# energy rule, 22 by both: a tail is either a loud stretch or a voiced one, so the veto is their UNION (95, 2.8%).
# EXP-059 registered this as its detector D (2026-10-10, rule hash c5ec140628ff2bf4); change nothing without a new
# registration. Ported from the session's analysis script unchanged, so the pool's counts reproduce exactly.
TAIL_VOICED_DB = 6.0      # the last voiced run peaks at least this far above the median level of the gap before it
TAIL_PYIN = {"fmin": 65, "fmax": 500, "frame_length": 2048}
# EXP-059's D_loose, for HIDING only (a tail made quieter or closer, so D no longer sees it); never for pairs.
TAIL_LOOSE = {"floor_db": -50.0, "gap_s": 0.25, "max_s": 0.8, "voiced_db": 3.0}
TAIL_RULE = "union-v1"    # the cache key: bump it when any TAIL_ constant changes


def voiced_tail_of(y, sr: int, voiced=None, gap_s: float = TAIL_GAP_S, max_s: float = TAIL_MAX_S,
                   above_floor_db: float = TAIL_VOICED_DB) -> dict | None:
    """The voiced rule: the same merge, minimum, gap and length rules as the energy rule, on pYIN's voiced frames,
    and the last run must stand above the gap's floor. `voiced` reuses a pYIN pass (it is the slow part)."""
    import numpy as np
    hop = int(TAIL_HOP_S * sr)
    n = len(y) // hop
    if n == 0:
        return None
    db = 20 * np.log10(np.sqrt(np.mean(np.asarray(y[: n * hop]).reshape(n, hop) ** 2, axis=1)) + 1e-12)
    if voiced is None:
        voiced = pyin_voiced(y, sr)
    v = voiced[:n]
    runs, start = [], None
    for i, a in enumerate(v):
        if a and start is None:
            start = i
        if not a and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, n])
    merged = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) * TAIL_HOP_S < TAIL_MERGE_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [r for r in merged if (r[1] - r[0]) * TAIL_HOP_S >= 0.04]
    if len(merged) < 2:
        return None
    (_, pe), (ts, te) = merged[-2], merged[-1]
    gap, dur = (ts - pe) * TAIL_HOP_S, (te - ts) * TAIL_HOP_S
    if gap < gap_s or dur > max_s:
        return None
    floor = float(np.median(db[pe:ts]))
    if float(db[ts:te].max()) - floor < above_floor_db:
        return None
    return {"main_end_s": round(pe * TAIL_HOP_S, 3), "tail_start_s": round(ts * TAIL_HOP_S, 3),
            "tail_end_s": round(te * TAIL_HOP_S, 3), "gap_s": round(gap, 2), "dur_s": round(dur, 2),
            "above_floor_db": round(float(db[ts:te].max()) - floor, 1)}


def pyin_voiced(y, sr: int):
    import librosa
    _, voiced, _ = librosa.pyin(y, sr=sr, hop_length=int(TAIL_HOP_S * sr), **TAIL_PYIN)
    return voiced


def tails_of(y, sr: int) -> dict:
    """EXP-059's D (strict) and D_loose on one clip, one pYIN pass. Each is None or the firing rule's span, with
    "rule" naming energy, voiced or both; where both fire, the energy rule's span is reported."""
    voiced = pyin_voiced(y, sr)
    out = {}
    for name, kw in (("strict", {}), ("loose", TAIL_LOOSE)):
        e = (detached_tail_of(y, sr, floor_db=kw["floor_db"], gap_s=kw["gap_s"], max_s=kw["max_s"]) if kw
             else detached_tail_of(y, sr))
        v = (voiced_tail_of(y, sr, voiced, gap_s=kw["gap_s"], max_s=kw["max_s"], above_floor_db=kw["voiced_db"])
             if kw else voiced_tail_of(y, sr, voiced))
        out[name] = None if not (e or v) else {**(e or v), "rule": "both" if e and v else ("energy" if e else "voiced")}
    return out


def tail_code() -> list:
    """THE list of what the detector's output depends on: its functions and constants. The tail cache and the
    analysis store's steps both hash this one list, so they cannot drift apart (review, 2026-10-10)."""
    return [tails_of, voiced_tail_of, detached_tail_of, active_segments, pyin_voiced,
            (TAIL_RULE, TAIL_GAP_S, TAIL_MAX_S, TAIL_FLOOR_DB, TAIL_HOP_S, TAIL_MERGE_S, TAIL_VOICED_DB, TAIL_PYIN,
             TAIL_LOOSE)]
