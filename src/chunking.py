"""Clause chunking for long prose: the one function render_story.py uses from the lab's synthesize_long.py.

A paragraph is cut at clause punctuation into chunks of at most `hi` characters (render_story's --hi, 45 since
2026-10-19: longer chunks read better, but at 70 the model skips whole clauses from 50-63-character chunks; none at
26 or 45 -- the lab's chunk-ceiling concept). Each chunk keeps its own closing punctuation, so the text the model
reads is the text that was written. Copied unchanged.
"""
from __future__ import annotations

SPLIT = "，。、；：！？"


def chunk(text: str, lo: int = 14, hi: int = 26) -> list[tuple[str, str]]:
    """Split into (phrase, trailing punctuation) pairs inside the length band."""
    parts, buf, out = [], "", []
    for ch in text:
        if ch in SPLIT:
            parts.append((buf.strip(), ch))
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append((buf.strip(), ""))
    cur, curp = "", ""
    for p, punct in parts:
        if not p:
            continue
        if not cur:
            cur, curp = p, punct
        elif len(cur) + len(p) + 1 <= hi:
            # KEEP THE FIRST PART'S OWN PUNCTUATION WHEN MERGING. This used to join
            # with a comma and throw `curp` away, which rewrote the text the model was
            # asked to read. On a composed story of short sentences that is not a
            # detail: 52 sentences merged into 27 chunks flattened roughly 25 sentence
            # boundaries into comma splices and turned every 。 and ？ inside a chunk
            # into ，, so a question was read with statement intonation and the
            # marking page displayed text that was not the text that had been
            # composed. Caught by a listener -- "text seems wrong on this one. That
            # seems to be the pattern" -- which is exactly the kind of defect the
            # render is silent about.
            #
            # The joiner is one character either way, and the band test above already
            # counts it, so chunk boundaries are unchanged by this. Only what the
            # boundary CARRIES changes.
            cur, curp = cur + (curp or "，") + p, punct
        else:
            out.append((cur, curp))
            cur, curp = p, punct
        while len(cur) > hi:                 # a single clause over the band
            out.append((cur[:hi], ""))
            cur = cur[hi:]
    if cur:
        out.append((cur, curp))
    return [(c, p) for c, p in out if len(c) >= 4]
