#!/usr/bin/env python3
"""Skipped text: runs of the script that a render never says, found by aligning the script with what the ASR heard.

Found 2026-10-19 by the owner's ear ("Lots of missing characters", on every voice of BM-DEMO-091 at chunk ceiling 70)
and confirmed by EXP-090's own ASR: at ceiling 70 the model skipped whole clauses (16-20 characters) from the middle of
50-63-character chunks, 1.5-2.0% of all characters, against 0.1-0.3% at 26, 45 and 120. No test in the release gate
saw it: a skipped clause makes a chunk SHORTER, so it is never a runaway, and it barely moves a median CER.

    lost_runs(script, heard, min_run)  the script's CJK runs of >= min_run characters the ASR did not hear (deleted, or
                                       replaced by fewer characters), each as (start index, length)
    summarize(pairs)                   chunks or clips with a lost run, characters lost, and the share of all characters

The unit is CJK characters by code point, as the listening pages count them. A run, not a character, because the ASR
itself drops or merges single characters at a low rate on any render; MIN_RUN = 4 is above that floor (EXP-090 at 26:
3 and 6 characters lost in 3,751, none in a run of 4).

    python src/skipped_text.py --self-test
"""
from __future__ import annotations

import difflib
import sys

MIN_RUN = 4


def cjk(s: str) -> list[str]:
    return [c for c in s if "㐀" <= c <= "鿿" or "\U00020000" <= c <= "\U0002ffff"]


def lost_runs(script: str, heard: str, min_run: int = MIN_RUN) -> list[tuple[int, int]]:
    ref, hyp = cjk(script), cjk(heard or "")
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        k = (i2 - i1) - (j2 - j1) if op in ("delete", "replace") else 0
        if k >= min_run:
            out.append((i2 - k, k))
    return out


def summarize(pairs: list[tuple[str, str]], min_run: int = MIN_RUN) -> dict:
    """pairs: (script, heard) per chunk or clip."""
    n = chars = lost = 0
    with_run = 0
    for script, heard in pairs:
        runs = lost_runs(script, heard, min_run)
        n += 1
        chars += len(cjk(script))
        lost += sum(k for _, k in runs)
        with_run += bool(runs)
    return {"units": n, "chars": chars, "units_with_lost_run": with_run, "chars_lost_in_runs": lost,
            "lost_share": round(lost / chars, 5) if chars else 0.0, "min_run": min_run}


def self_test() -> int:
    # An invented sentence (written for this test, 34 characters), not corpus or story text.
    s = "今日天氣好好所以我哋一齊去公園行吓然後返屋企食飯飲茶睇電視傾偈到夜晚"
    cases = [
        ("a faithful read loses nothing", s, s, 0),
        ("one character the ASR misheard is not a skip", s, s[:5] + "晏" + s[6:], 0),
        ("two characters dropped are under the run floor", s, s[:5] + s[7:], 0),
        ("RED: a skipped 12-character clause is found", s, s[:20] + s[32:], 1),
        ("RED: a skip replaced by a shorter wrong run is found", s, s[:8] + "嗯" + s[16:], 1),
        ("RED: a lost ending is found", s, s[:25], 1),
        ("punctuation and Latin are not counted", s + "，OK！", s, 0),
    ]
    ok = True
    for name, a, b, want in cases:
        got = len(lost_runs(a, b))
        good = got == want
        ok &= good
        print(("  ok   " if good else "  FAIL ") + f"{name}: {got} run(s)")
    sm = summarize([(s, s[:20] + s[32:]), (s, s)])
    good = sm["units_with_lost_run"] == 1 and sm["chars_lost_in_runs"] == 12
    ok &= good
    print(("  ok   " if good else "  FAIL ") + f"summarize counts the unit and the 12 characters: {sm}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else 0)
