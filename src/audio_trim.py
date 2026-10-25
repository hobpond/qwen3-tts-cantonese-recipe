"""Cap leading and trailing silence on generated audio.

WHY IT IS A SHARED UTILITY AND NOT A FLAG IN ONE SCRIPT. Every render path on this
project writes wavs straight from `generate_*` and several of them feed listening
pages, where a long silent lead-in is both irritating and a cue: a listener who hears
one clip start late has learned something about that arm that has nothing to do with
how it reads.

THE MEASUREMENT HAZARD, which is the reason the raw duration is still returned.
`codepred_probe.py` calls a clip a runaway when its seconds-per-character exceeds
twice a measured control. That statistic is computed from total duration. If a clip's
excess length is trailing silence rather than speech, then trimming first changes
what "runaway" means, and every stored runaway rate was measured on untrimmed audio.

So this function never silently rewrites a measurement. It returns the trimmed audio
AND both durations, and it is the caller's job to decide which one a metric uses.
Trim for listening; keep the raw duration for anything that has to compare against a
number recorded before today.
"""
from __future__ import annotations

import numpy as np


def trim_edges(y: np.ndarray, sr: int, max_edge_s: float = 1.0,
               top_db: float = 40.0) -> tuple[np.ndarray, dict]:
    """Cap silence at each end to `max_edge_s` seconds.

    Silence longer than the cap is cut back TO the cap, not removed outright -- a
    little air at each end is wanted, and a hard cut to the first sample of speech
    clips plosives and breath onsets.

    Returns (audio, info) where info carries raw_s, trimmed_s, and how much was
    removed at each end, so a caller can log or refuse the change.

    `start_sample` / `end_sample` / `raw_samples` are the EXACT integer bounds: the
    returned audio is `y[start_sample:end_sample]` of the `raw_samples`-long input.
    The rounded seconds beside them cannot tie a judged clip back to the codec
    frames it was decoded from (TASK-PARTICLE-CODES-001, R2 of the DPO task); these
    can. Present on every return path, including the ones that cut nothing.
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    raw_s = len(y) / sr
    if len(y) == 0:
        return y, {"raw_s": 0.0, "trimmed_s": 0.0, "lead_cut_s": 0.0, "tail_cut_s": 0.0,
                   "lead_silence_s": 0.0, "tail_silence_s": 0.0,
                   "start_sample": 0, "end_sample": 0, "raw_samples": 0}

    # frame-wise energy relative to the loudest frame, the same convention librosa's
    # trim uses; implemented here so this module has no import beyond numpy
    hop = max(1, sr // 100)
    frames = [y[i:i + hop] for i in range(0, len(y), hop)]
    rms = np.array([float(np.sqrt(np.mean(f.astype(np.float64) ** 2)) + 1e-12)
                    for f in frames])
    ref = rms.max()
    loud = 20 * np.log10(rms / ref) > -top_db
    if not loud.any():
        return y, {"raw_s": raw_s, "trimmed_s": raw_s, "lead_cut_s": 0.0,
                   "tail_cut_s": 0.0, "lead_silence_s": raw_s, "tail_silence_s": 0.0,
                   "start_sample": 0, "end_sample": len(y), "raw_samples": len(y)}

    first, last = int(np.argmax(loud)), int(len(loud) - 1 - np.argmax(loud[::-1]))
    lead_sil = first * hop / sr
    tail_sil = (len(loud) - 1 - last) * hop / sr
    keep = int(round(max_edge_s * sr))
    start = max(0, first * hop - keep)
    end = min(len(y), (last + 1) * hop + keep)
    out = y[start:end]
    return out, {
        "raw_s": round(raw_s, 3),
        "trimmed_s": round(len(out) / sr, 3),
        "lead_cut_s": round(start / sr, 3),
        "tail_cut_s": round((len(y) - end) / sr, 3),
        "lead_silence_s": round(lead_sil, 3),
        "tail_silence_s": round(tail_sil, 3),
        "start_sample": int(start),
        "end_sample": int(end),
        "raw_samples": int(len(y)),
    }

def silence_report(y, sr: int, top_db: float = 40.0,
                   gap_min_s: float = 0.25) -> dict:
    """Where the silence is: at the edges, and INSIDE the speech.

    `trim_edges` deals with the edges by cutting them, which hides a signal rather
    than reporting it: a clip needing 9.33 s of lead removed is a bad draw even
    after the lead is gone. And nothing has ever looked for silence in the MIDDLE.
    A long gap mid-utterance is a distinct failure -- the model stalling partway --
    that trimming cannot touch and that a duration check can miss entirely, because
    a stalled clip and a slow clip have the same length.

    Returns lead_s, tail_s, longest internal gap, their count, and the total
    silent fraction. Gaps shorter than `gap_min_s` are normal phrase breaks and
    are not counted.
    """
    import numpy as np
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = len(y)
    if n == 0:
        return {"lead_s": 0.0, "tail_s": 0.0, "max_gap_s": 0.0,
                "n_gaps": 0, "gaps_total_s": 0.0, "silent_fraction": 0.0}
    hop = max(1, sr // 100)
    frames = [y[i:i + hop] for i in range(0, n, hop)]
    rms = np.array([float(np.sqrt(np.mean(f.astype(np.float64) ** 2)) + 1e-12)
                    for f in frames])
    loud = 20 * np.log10(rms / rms.max()) > -top_db
    if not loud.any():
        return {"lead_s": round(n / sr, 3), "tail_s": 0.0, "max_gap_s": 0.0,
                "n_gaps": 0, "gaps_total_s": 0.0, "silent_fraction": 1.0}
    first = int(np.argmax(loud))
    last = int(len(loud) - 1 - np.argmax(loud[::-1]))
    gaps, run = [], 0
    for v in loud[first:last + 1]:
        if v:
            if run:
                gaps.append(run * hop / sr)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run * hop / sr)
    gaps = [g for g in gaps if g >= gap_min_s]
    return {"lead_s": round(first * hop / sr, 3),
            "tail_s": round((len(loud) - 1 - last) * hop / sr, 3),
            "max_gap_s": round(max(gaps), 3) if gaps else 0.0,
            "n_gaps": len(gaps),
            "gaps_total_s": round(sum(gaps), 3),
            "silent_fraction": round(1.0 - float(loud.mean()), 3)}
