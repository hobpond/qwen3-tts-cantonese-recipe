#!/usr/bin/env python3
"""The Common Voice selector's filters, applied to a Mozilla Data Collective (MDC) archive.

`select_common_voice.py` reads the Hugging Face mirror's layout (tar shards under audio/<locale>, TSVs under
transcript/<locale>). Common Voice 23.0 and later come only from MDC, as ONE .tar.gz whose members are
`cv-corpus-<ver>/<locale>/clips/<file>.mp3` beside the TSVs. This script applies the same filters, imported from
that script so the two cannot drift: split, duration from clip_durations.tsv, contributor, gender, accent,
down-votes and the colloquial-marker register filter. It writes the same out-dir: audio/*.mp3, transcripts.tsv and
selection.json, in the same seeded order.

    python src/select_mdc.py \\
        --archive ~/datasets/mdc/common-voice-scripted-speech-27-0-chines-26ba81db.tar.gz --locale zh-HK \\
        --splits validated --gender "" --min-duration 3.0 --max-duration 30.0 --out-dir ~/datasets/cv-zhhk-27

The archive is read twice as a stream (once for the TSVs, once for the selected clips). Nothing else is extracted,
so a 3.4 GB archive costs only the selected clips on disk. MDC's terms forbid re-hosting: the output is for
training here, never for upload.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_common_voice import CANTO  # noqa: E402  the register filter, shared


def tsvs_from(archive: Path, locale: str) -> tuple[str, dict[str, bytes]]:
    out, root = {}, None
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            parts = Path(m.name).parts
            if len(parts) == 3 and parts[1] == locale and parts[2].endswith(".tsv") and m.isfile():
                root = parts[0]
                out[parts[2]] = tf.extractfile(m).read()
    if not out:
        raise SystemExit(f"{archive} holds no {locale} TSVs")
    return root, out


def read_tsv_bytes(b: bytes) -> list[dict]:
    import csv
    return list(csv.DictReader(io.StringIO(b.decode("utf-8", errors="replace")), delimiter="\t"))


def select(rows: list[dict], durs: dict, a) -> tuple[list[dict], dict]:
    stage, n = [], {"contributor": 0, "duration": 0, "gender": 0, "accent": 0, "down_votes": 0, "transcript": 0}
    want_ids = set(a.client_id or [])
    for r in rows:
        if want_ids and (r.get("client_id") or "") not in want_ids:
            n["contributor"] += 1
            continue
        d = durs.get(r["path"])
        if d is None or not (a.min_duration <= d <= a.max_duration):
            n["duration"] += 1
            continue
        if a.gender and (r.get("gender") or "").strip() != a.gender:
            n["gender"] += 1
            continue
        if a.accent_contains and not any(k in (r.get("accents") or "") for k in a.accent_contains):
            n["accent"] += 1
            continue
        if a.drop_down_voted:
            try:
                if int(r.get("down_votes") or 0) > 0:
                    n["down_votes"] += 1
                    continue
            except ValueError:
                pass
        sent = (r.get("sentence") or "").strip()
        if not sent or (a.require_colloquial and not any(c in CANTO for c in sent)):
            n["transcript"] += 1
            continue
        stage.append({"path": r["path"], "sentence": sent, "client_id": r.get("client_id", ""), "dur": d})
    return stage, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--locale")
    ap.add_argument("--splits", nargs="+", default=["validated"])
    ap.add_argument("--min-duration", type=float, default=5.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--gender", default="female_feminine")
    ap.add_argument("--accent-contains", nargs="*", default=None)
    ap.add_argument("--require-colloquial", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-down-voted", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--client-id", nargs="*", default=None)
    # The speaker set is one contributor. Her Common Voice pseudonym is passed in a file kept OUTSIDE this
    # repository, never on a command line that lands in a shell history or a committed script.
    ap.add_argument("--client-id-file", type=Path, help="contributor ids, one per line, '#' for comments")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="count only; extract nothing")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.archive or not a.locale:
        ap.error("--archive and --locale are required")
    if not a.out_dir and not a.dry_run:
        ap.error("--out-dir is required unless --dry-run")
    if a.client_id_file:
        ids = [x.split("#", 1)[0].strip() for x in a.client_id_file.read_text(encoding="utf-8").splitlines()]
        a.client_id = sorted(set(a.client_id or []) | {x for x in ids if x})
        if not a.client_id:
            ap.error(f"{a.client_id_file} holds no ids")

    root, tsvs = tsvs_from(a.archive, a.locale)
    durs = {}
    for r in read_tsv_bytes(tsvs["clip_durations.tsv"]):
        try:
            durs[r["clip"]] = float(r["duration[ms]"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            pass
    seen, rows = set(), []
    for sp in a.splits:
        for r in read_tsv_bytes(tsvs[f"{sp}.tsv"]):
            if r.get("path") and r["path"] not in seen:
                seen.add(r["path"])
                rows.append(r)
    stage, n = select(rows, durs, a)
    secs = [s["dur"] for s in stage]
    summary = {"archive": a.archive.name, "corpus": root, "locale": a.locale, "pool": len(rows), "kept": len(stage),
               "hours": round(sum(secs) / 3600, 2), "median_s": round(statistics.median(secs), 2) if secs else None,
               "speakers": len({s["client_id"] for s in stage}),
               "filters": {"splits": a.splits, "gender": a.gender, "duration": [a.min_duration, a.max_duration],
                           "require_colloquial": a.require_colloquial, "drop_down_voted": a.drop_down_voted,
                           "accent_contains": a.accent_contains, "seed": a.seed},
               "dropped": n}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if a.dry_run or not stage:
        return 0 if stage else 1
    random.Random(a.seed).shuffle(stage)
    audio = a.out_dir / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    want = {f"{root}/{a.locale}/clips/{s['path']}": s["path"] for s in stage}
    found = 0
    with tarfile.open(a.archive, "r|gz") as tf:
        for m in tf:
            if m.name in want and m.isfile():
                (audio / want[m.name]).write_bytes(tf.extractfile(m).read())
                found += 1
    with (a.out_dir / "transcripts.tsv").open("w", encoding="utf-8", newline="") as fh:
        for s in stage:
            if (audio / s["path"]).exists():
                fh.write(f"{s['path']}\t{s['sentence']}\n")
    summary["extracted"] = found
    (a.out_dir / "selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"extracted {found}/{len(stage)} -> {audio}")
    return 0 if found == len(stage) else 1


def self_test() -> int:
    """The filters, each shown to drop what it should; a red where a guard is emptied."""
    fails = 0

    def check(name, cond):
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        fails += not cond
    a = argparse.Namespace(client_id=None, min_duration=3.0, max_duration=30.0, gender="", accent_contains=None,
                           drop_down_voted=True, require_colloquial=True)
    colloq = next(iter(CANTO))
    rows = [{"path": "ok.mp3", "sentence": "佢" + colloq, "down_votes": "0"},
            {"path": "short.mp3", "sentence": "佢係", "down_votes": "0"},
            {"path": "voted.mp3", "sentence": "佢係", "down_votes": "1"},
            {"path": "formal.mp3", "sentence": "今天天氣很好", "down_votes": "0"}]
    durs = {"ok.mp3": 4.0, "short.mp3": 2.0, "voted.mp3": 4.0, "formal.mp3": 4.0}
    stage, n = select(rows, durs, a)
    check("a colloquial 4 s clip with no down-votes is kept", [s["path"] for s in stage] == ["ok.mp3"])
    check("RED: under 3 s, a down-vote and a formal-register sentence are each dropped by their own filter",
          n["duration"] == 1 and n["down_votes"] == 1 and n["transcript"] == 1)
    a.require_colloquial = False
    check("RED: with the register filter off the formal sentence gets through (the filter is what drops it)",
          "formal.mp3" in [s["path"] for s in select(rows, durs, a)[0]])
    print("PASS" if not fails else f"FAIL: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
