#!/usr/bin/env python3
"""A LABELLED long-prose page on the standard page: one trial per text, up to three voices (A, B, C) reading it,
per-character marks on every voice, one question, and an About dialog per text (who wrote it, what they meant it
to do, how it was judged).

Input is render_story.py's output (manifest.json, one FLAC per arm per text) and the text directory it read, plus
an about.json there: {"_rubric": str, "_process": str, "<text id>": {"author", "intention", "grade"}}.

    python build_story_page.py --renders DIR --texts DIR --stories 1,2,3 --bm BM-ID --out DIR \
        --slot "A=V4-343:v0.0.4 hk_voice_343" --slot "B=V4-777:v0.0.4 hk_voice_777" --slot "C=V1:v0.0.1"

Pages and keys are written under --out (the recipe's work directory), never into the repository: a page carries
its audio inline, and .gitignore and scripts/check_clean_room.py both refuse one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                  # src/: build_ab_page, render_story
import build_ab_page as bab  # noqa: E402

SENTENCE_END = "。！？"


def chunk_spans(pcm: np.ndarray, sr: int, chunks: list[dict], rows: list[dict]) -> list[tuple[int, int]]:
    """Where each chunk sits in a read: the span render_story.py recorded, or, for a read made before it recorded
    spans, recovered from the read itself. The assembler joins chunks with digital zeros of a length set by each
    chunk's punctuation, and FLAC is lossless, so every join is a run of exact zero samples at least that long (longer
    by whatever edge silence rounded to zero, at most two trimmed edges). Refused unless there is exactly one such run
    per join and each fits its own expected length: a cut that cannot be proven is not made."""
    import render_story as rs
    if all("span" in r for r in rows):
        return [tuple(r["span"]) for r in rows]
    want = [int((rs.PAUSE.get(c["punct"], rs.PAUSE[""]) + (rs.PARA_EXTRA if c["last_in_para"] else 0.0)) * sr)
            for c in chunks[:-1]]
    z = np.concatenate([[False], pcm == 0, [False]])
    d = np.flatnonzero(np.diff(z.astype(np.int8)))
    runs = [(a, b) for a, b in zip(d[::2], d[1::2]) if b - a >= int(0.14 * sr)]
    if len(runs) != len(want):
        raise SystemExit(f"{len(runs)} zero runs for {len(want)} joins: the read cannot be cut provably")
    slack = int((2 * rs.EDGE_S + 0.02) * sr)
    for j, ((a, b), w) in enumerate(zip(runs, want)):
        if not w <= b - a <= w + slack:
            raise SystemExit(f"join {j}: {b - a} zero samples, expected {w} to {w + slack}")
    starts = [0] + [b for _, b in runs]
    ends = [a for a, _ in runs] + [len(pcm)]
    return list(zip(starts, ends))


def segments(chunks: list[dict], max_chars: int) -> list[list[int]]:
    """Chunk indices per segment: one paragraph each, or with max_chars, a paragraph split at sentence ends once a
    segment holds at least max_chars characters."""
    import render_story as rs
    out, cur, n = [], [], 0
    for i, c in enumerate(chunks):
        cur.append(i)
        n += rs.cjk_len(c["text"])
        if c["last_in_para"] or (max_chars and n >= max_chars and c["punct"] in SENTENCE_END):
            out.append(cur)
            cur, n = [], 0
    return out


def main() -> int:
    import render_story as rs
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--stories", required=True, help="comma-separated text ids, in page order")
    ap.add_argument("--bm", required=True, help="the registered listen")
    ap.add_argument("--page", required=True, help="this page's tag, e.g. v1-s6: the page id is BM-PAGE, so every "
                    "page keeps its own answers in the browser and its own code prefix")
    ap.add_argument("--title", default="Long prose, labelled")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slot", action="append", required=True, help="LABEL=ARM:display name, labels A, B[, C]")
    ap.add_argument("--no-viz", action="store_true", help="skip pitch contours (very long reads)")
    ap.add_argument("--preset-code", help="a code from this page (any page id prefix): the page opens showing its "
                    "answers and marks, for a published demo; a browser's own saved answers still win")
    ap.add_argument("--preset-label", default="a native listener's answers and marks",
                    help="who the preset is, said in the intro")
    ap.add_argument("--by-paragraph", action="store_true", help="cut every read at its paragraph ends after "
                    "generation and make one trial per paragraph (owner 2026-10-19: a whole read is too long to "
                    "remember which voice was best)")
    ap.add_argument("--max-chars", type=int, default=0, help="with --by-paragraph, also split a paragraph at a "
                    "sentence end once a piece holds this many characters (0: whole paragraphs)")
    ap.add_argument("--no-notes", action="store_true", help="leave out the free-form notes boxes")
    ap.add_argument("--no-ratings", action="store_true", help="leave out the 1-5 rating per voice (the answer is then "
                    "pressed, not derived)")
    ap.add_argument("--blind", action="store_true", help="with --by-paragraph: the voices are shuffled across A, B, C "
                    "on every part (seeded by the page id and part, recorded in the key) and never named on the page")
    a = ap.parse_args()
    if (a.out / "page.html").exists() or (a.out / "key.json").exists():
        raise SystemExit(f"{a.out} already holds a page or key; a built page is not overwritten")
    man = json.loads((a.renders / "manifest.json").read_text(encoding="utf-8"))
    about = json.loads((a.texts / "about.json").read_text(encoding="utf-8"))
    slots = []
    for s in a.slot:
        lab, rest = s.split("=", 1)
        arm, disp = rest.split(":", 1)
        slots.append((lab, arm, disp))
    labels = [l for l, _, _ in slots]
    if labels not in (["A", "B"], ["A", "B", "C"]):
        raise SystemExit(f"slots must be A,B or A,B,C, not {labels}")
    sids = a.stories.split(",")
    flat, texts, key_trials, about_rows = [], [], [], []
    for sid in sids:
        title, paras = rs.read_text(a.texts / f"{sid}.txt")
        if man["texts"][sid]["sha256"] != __import__("hashlib").sha256((a.texts / f"{sid}.txt").read_bytes()).hexdigest():
            raise SystemExit(f"text {sid} changed since it was rendered")
        if a.by_paragraph:
            import random
            import soundfile as sf
            # Each voice's read was planned at its own chunk ceiling (an arm may carry its own chunk_hi), so each is
            # cut on its own plan; a paragraph is the same text whatever the ceiling, which is what lines them up.
            his = {arm: man["arms"][arm].get("chunk_hi", man["rule"].get("chunk_hi", 26)) for _, arm, _ in slots}
            if len(set(his.values())) > 1 and a.max_chars:
                raise SystemExit("voices read at different chunk ceilings can only be cut at paragraph ends")
            cut = {}
            for lab, arm, _ in slots:
                st = man["arms"][arm]["stories"][sid]
                ch = rs.plan(paras, his[arm])
                pcm, sr = sf.read(str(a.renders / st["flac"]), dtype="int16")
                cut[arm] = (pcm, sr, chunk_spans(pcm, sr, ch, st["chunks"]), st, ch, segments(ch, a.max_chars))
            first = slots[0][1]
            chunks, segs = cut[first][4], cut[first][5]
            for arm, (_, _, _, _, ch, sg) in cut.items():
                if ["".join(ch[i]["text"] for i in g) for g in sg] != ["".join(chunks[i]["text"] for i in g) for g in segs]:
                    raise SystemExit(f"{arm}'s parts do not carry the same text as {first}'s")
            ab = about.get(sid, {})
            for k, seg in enumerate(segs):
                body = "".join(chunks[i]["text"] for i in seg)
                t = {"pos": len(key_trials), "story": sid, "title": title, "segment": k, "of": len(segs),
                     "para": chunks[seg[0]]["para"], "chunks": {arm: [c[5][k][0], c[5][k][-1]] for arm, c in cut.items()},
                     "text": body, "cjk_len": len(bab.cjk_chars(body)), "sha256": {}}
                order = [arm for _, arm, _ in slots]
                if a.blind:
                    random.Random(f"{a.bm}-{a.page}-{len(key_trials)}").shuffle(order)
                for (lab, _, _), arm in zip(slots, order):
                    pcm, sr, sp, st, _, sg = cut[arm]
                    seg = sg[k]
                    y = pcm[sp[seg[0]][0]:sp[seg[-1]][1]].astype(np.float32) / 32768.0
                    b = bab.flac_bytes(y, sr)
                    flat.append(b)
                    t[lab] = arm
                    t["sha256"][lab] = bab.sha(b)
                    t.setdefault("defects", {})[arm] = {
                        "first_draw": sum(st["chunks"][i]["first_draw_defect"] is not None for i in seg)}
                key_trials.append(t)
                texts.append(body)
                about_rows.append({"title": f"{title}, part {k + 1} of {len(segs)}", "sections": [
                    ("Who wrote it", ab.get("author", "")), ("What it set out to do", ab.get("intention", "")),
                    ("How it was judged", ab.get("grade", "") + "\n\n" + about.get("_rubric", "")),
                    ("How these texts were written", about.get("_process", "")),
                    ("How this page was cut", "Each voice read the whole text as one continuous read. The reads "
                     "were cut afterwards at paragraph ends" + (" and, in a long paragraph, at a sentence end"
                     if a.max_chars else "") + ", at the same text on every voice, so each part here is a piece of "
                     "that read, not a separate render."),
                    ("Voices on this page", "Blind: the voices are shuffled on every part, and named only in the "
                     "key." if a.blind else "\n\n".join(f"{l}: {d}" for l, _, d in slots))]})
            continue
        body = "".join(paras)
        t = {"pos": len(key_trials), "story": sid, "title": title, "cjk_len": len(bab.cjk_chars(body)), "sha256": {}}
        for lab, arm, _ in slots:
            st = man["arms"][arm]["stories"][sid]
            b = (a.renders / st["flac"]).read_bytes()
            flat.append(b)
            t[lab] = arm
            t["sha256"][lab] = bab.sha(b)
            t.setdefault("defects", {})[arm] = {"first_draw": st["first_draw_defects"], "left": st["final_defects"]}
        key_trials.append(t)
        texts.append(body)
        ab = about.get(sid, {})
        about_rows.append({"title": title, "sections": [
            ("Who wrote it", ab.get("author", "")), ("What it set out to do", ab.get("intention", "")),
            ("How it was judged", ab.get("grade", "") + "\n\n" + about.get("_rubric", "")),
            ("How these texts were written", about.get("_process", "")),
            ("Voices on this page", "\n\n".join(f"{l}: {d}" for l, _, d in slots))]})
    if a.no_viz:
        contours = [{"t": [], "f": []} for _ in flat]
    else:
        contours = [bab.contour_of(b) for b in flat]
    cal = [bab.scale(contours) if not a.no_viz else [-12.0, 12.0]] * len(flat)
    n = len(labels)
    trials = [{"c": list(range(i * n, i * n + n)), "l": labels} for i in range(len(key_trials))]
    options = [("a", "A reads it better"), ("b", "B reads it better")] + ([("c", "C reads it better")] if n == 3 else [])         + [("s", "Same"), ("?", "Can't tell")]
    sides = {"a": ["A"], "b": ["B"], "s": [], "?": None, **({"c": ["C"]} if n == 3 else {})}
    session = f"{a.bm}-{a.page}"
    who = "; ".join(f"<b>{l}</b> {d}" for l, _, d in slots)
    if a.blind and not a.by_paragraph:
        raise SystemExit("--blind needs --by-paragraph")
    intro = (("BLIND: " + str(len(slots)) + " voices, shuffled across " + ", ".join(labels) + " on every part. "
              if a.blind else "LABELLED, not blind: " + who + ". ") + "Each text is read once by each voice as one continuous read, "
             "assembled from short chunks at clause breaks the way the model card says long text should be read"
             + (", then cut at paragraph ends so each part can be compared while it is fresh. " if a.by_paragraph
                else ". ")
             + ("<b>Rate every voice 1-5 with the number keys while it plays</b> (1 not Cantonese-like, 3 "
                "acceptable, 5 native): the next voice starts, and when all are rated the best is picked for you "
                "and the next part begins. A borderline word or an odd sentence is a lower number, not a mark; mark "
                "only clear mispronunciations. The answer buttons still override the pick. " if not a.no_ratings else "")
             + ("A notes box under each part, and one for the whole page, takes anything the buttons and marks "
                "cannot say. " if not a.no_notes else "")
             + (("<b>Skipped text re-drawn</b>: chunks where the speech recogniser found part of the script missing "
                 f"were rendered again (up to {man['rule'].get('skip_redraw_max_tries')} times) until it heard the whole "
                 f"text; {sum(c.get('skip_redraws', 0) > 0 for arm in man['arms'].values() for st in arm['stories'].values() for c in st['chunks'])} "
                 "chunk(s) were re-drawn and the key lists each. Every other chunk is the model's first draw. ")
                if man["rule"].get("skip_redrawn") else
                "<b>Every chunk is the model's first draw</b>: nothing was re-drawn, so a runaway or a silence is "
                "heard as the model made it. " if man["rule"].get("max_redraw", 4) == 0 else
                f"Runaway or silent chunks were RE-DRAWN (up to {man['rule'].get('max_redraw', 4)} times), so this "
                "page under-shows the model's failures; the key lists each one. ") +
             "<b>About this text</b> (key <kbd>I</kbd>) says who wrote it, what it set out to do and how it was judged. "
             f"Answer <b>1-{len(options)}</b> which voice reads it best"
             + ". <b>Space</b> plays all in order; <b>a</b>, <b>b</b>" + (", <b>c</b>" if n == 3 else "")
             + " play one. Answers save in this browser; copy the code at the bottom when done." + bab.SHIP_MARK_INTRO)
    preset = None
    if a.preset_code:
        head, _, rest = a.preset_code.partition("|")
        ans = head.split(":", 1)[1]
        fields = dict(s.split("=", 1) for s in rest.split("|") if s)
        toks = fields.get("M", "").split(",")
        if len(ans) != len(key_trials) or len(toks) != len(key_trials):
            raise SystemExit("the preset code does not have one answer and one mark token per text on this page")
        preset = {"v": {str(i): c for i, c in enumerate(ans) if c != "-"},
                  "mk": {str(i): m for i, tok in enumerate(toks)
                         if any((m := bab.parse_mark_token(tok, key_trials[i]["cjk_len"], i, "".join(labels))).values())},
                  "t": a.preset_label}
        intro = ("DEMO: this page opens with " + a.preset_label + ", shown in red under each voice and in the answer "
                 "buttons. You can change them; your own choices are kept in your browser. " + intro)
    page = bab.build_html(session, a.title, a.title, intro, "Which voice reads this best?", options, trials,
                          [bab.uri(b) for b in flat], [0.0] * len(flat), contours, cal, True, "Text",
                          "every clip on this page", texts=texts, marks=True, about=about_rows, show_text=False,
                          preset=preset, notes=not a.no_notes, ratings=not a.no_ratings)
    if [bab.sha(b) for b in bab.page_audio(page)] != [t["sha256"][l] for t in key_trials for l in labels]:
        raise SystemExit("the page's audio is not the key's, in the key's order")
    key = {"preset": preset, "bm": session, "registration": a.bm, "page": a.page, "version": 2, "labelled": None if a.blind else who, "blind": a.blind, "slots": {l: {"arm": arm, "name": d} for l, arm, d in slots},
           "marks": True, "marks_grammar": bab.MARK_GRAMMAR, "renders_manifest_sha256": bab.sha(
               (a.renders / "manifest.json").read_bytes()),
           "groups": [{"name": "", "question": "Which voice reads this best?", "answers": [c for c, _ in options],
                       "labels": dict(options), "sides": sides}],
           "code_grammar": bab.CODE_GRAMMAR, "trials": key_trials, "max_redraw": man["rule"].get("max_redraw", 4), "skip_redrawn": {arm: {sid: [c["i"] for c in st["chunks"] if c.get("skip_redraws")]
                                for sid, st in man["arms"][arm]["stories"].items()} for arm in man["arms"]}
                if man["rule"].get("skip_redrawn") else None,
           **({"notes": True, "notes_grammar": bab.NOTES_GRAMMAR} if not a.no_notes else {}),
           **({"ratings": True, "ratings_grammar": bab.RATINGS_GRAMMAR} if not a.no_ratings else {}),
           **({"cut": {"by": "paragraph", "max_chars": a.max_chars}} if a.by_paragraph else {})}
    nt = len(key_trials)
    code = session + ":" + ("c" if n == 3 else "a") * nt + "|M=" + ",".join(["A0" + ("C0" if n == 3 else "")] * nt)
    if not a.no_notes:
        import urllib.parse
        code += "|N=" + urllib.parse.quote(json.dumps({"0": "a, b|c", "all": "全頁"}, ensure_ascii=False), safe="")
    if not a.no_ratings:
        code += "|R=" + ",".join(["".join(f"{l}{j + 3}" for j, l in enumerate(labels))] * nt)
    r = bab.parse_code(code, key)
    assert a.no_ratings or r["rows"][0]["ratings"] == {l: j + 3 for j, l in enumerate(labels)}, "ratings do not round-trip"
    assert a.no_notes or (r["rows"][0]["note"] == "a, b|c" and r["note_all"] == "全頁"), "notes do not round-trip"
    assert r["answered"] == nt and all(row["marks_by_arm"][key_trials[i]["A"]] == [0]
                                             and row["arms"] == [key_trials[i]["C" if n == 3 else "A"]]
                                             for i, row in enumerate(r["rows"]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "page.html").write_text(page, encoding="utf-8")
    (a.out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"bm": a.bm, "texts": len(sids), "trials": nt, "voices": n, "mb": round(len(page.encode()) / 1e6, 2),
                      "page": str(a.out / "page.html")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
