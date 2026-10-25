#!/usr/bin/env python3
"""Select trainable Cantonese clips from the downloaded Common Voice `yue` pool.

The headline pool is 141k clips, but almost none of it is usable as-is and the
dataset card will not tell you which parts. This applies every filter the training
pipeline actually needs, in the order that discards cheapest-first, and reports what
each one costs so the yield is auditable rather than asserted:

* **Duration.** `dataset.py` wants 5-30 s. The median `yue` clip is 3.5 s, so this
  is the filter that removes most of the pool. Reported, not hidden.
* **Gender.** The target voice is female and the existing corpus is ~86% male, so
  female-labelled clips are the point of using this dataset at all. The label is
  self-reported and often empty; empty is excluded rather than guessed.
* **Community votes.** `other` is the un-validated split. `down_votes > 0` means a
  listener actively flagged the clip, which is the only quality signal available
  without listening to 141k files. Clips carrying a down-vote are dropped even
  though they are otherwise eligible.
* **Register.** ~81% of `yue` transcripts carry colloquial Cantonese markers; the
  rest read as Standard Written Chinese, which is a different register from the
  target speaker's code-switched speech. Optional, on by default.

Writes the audio into a flat directory plus the two-column TSV that
`scripts/common/ingest_recordings.py` consumes, so the existing ingestion path is unchanged.

    CV_ROOT=/path/to/common-voice-yue python src/select_common_voice.py --limit 200 --out-dir AUDIT_DIR
    CV_ROOT=/path/to/common-voice-yue python src/select_common_voice.py --splits validated --gender "" \\
        --min-duration 3.0 --max-duration 30.0 --out-dir SELECTED_DIR

CV_ROOT is the Hugging Face mirror's layout (transcript/yue/*.tsv, audio/yue/**/*.tar). For an MDC archive (Common
Voice 23.0 and later), use select_mdc.py, which applies these same filters. `python recipe.py select` runs one of
them with the recipe's filters from recipe.toml.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import statistics
import sys
import tarfile
from pathlib import Path

# Hardcoding this made the selector unrunnable off this one workstation, which
# defeats the point of a repository built to be shared. Default is unchanged.
# The recipe repository has no default: a path on one machine is wrong on every other. Set CV_ROOT.
CV = Path(os.environ.get("CV_ROOT", ""))
CANTO = set("嘅係唔咗喺啲哋嗰噉咁佢乜嘢冇睇嚟返攞畀諗鍾喎嘞囉喇咩啩㗎啦")


def read_tsv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["validated", "other"])
    ap.add_argument("--min-duration", type=float, default=5.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    ap.add_argument("--gender", default="female_feminine",
                    help="Empty string disables the gender filter. For a language "
                         "phase that is usually right: the licensed corpus is 86%% "
                         "male and still produces the best sentence-final particles, "
                         "so register matters more than gender there. Gender is the "
                         "speaker phase's job.")
    # Common Voice `yue` pools the whole Cantonese-speaking world. In the clips
    # originally selected here, only 3.9% were Hong Kong against 30% Guangzhou-region
    # (广州音, 湛江, 佛山, 顺德) and 53% unlabelled -- and sentence-final particles are
    # among the most HK-vs-Guangzhou divergent features in the language, which is
    # exactly where a native listener said the resulting model went wrong.
    ap.add_argument("--accent-contains", nargs="*", default=None,
                    help="Keep only clips whose accents field contains one of these "
                         "substrings, e.g. --accent-contains 'Hong Kong' 香港")
    # BOTH OF THESE WERE UNREACHABLE, AND THAT MADE A REGISTERED EXPERIMENT
    # UNBUILDABLE. --require-colloquial was declared `action="store_true",
    # default=True`, so passing it set what was already set and there was no way to
    # turn it off; down-votes were dropped by an unconditional branch with no flag at
    # all. The docstring above says the register filter is "Optional, on by default"
    # and it was not optional. EXP-046 registers both arms on a DURATION-ONLY chain
    # and could not be built at all until this was fixed.
    #
    # Both default to the previous behaviour, so every selection ever made with this
    # script still reproduces byte for byte. BooleanOptionalAction gives the
    # --no-... form, which is the half that was missing.
    ap.add_argument("--require-colloquial", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="keep only transcripts carrying colloquial Cantonese markers. "
                         "On by default. --no-require-colloquial takes the register "
                         "filter off, which is right for a SPEAKER phase -- register is "
                         "the language phase's concern, and this script's own --gender "
                         "help says gender is the speaker phase's job.")
    ap.add_argument("--drop-down-voted", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="drop clips a listener flagged (down_votes > 0). On by "
                         "default. --no-drop-down-voted keeps them, which is what a "
                         "duration-only chain means.")
    ap.add_argument("--limit", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=4242)
    # Select one contributor -- the speaker phase's whole job. The ids are Common
    # Voice's own pseudonymous hashes, so they are ordinary data here: the audition
    # key in benchmarks/audition/ has carried them since the first commit.
    ap.add_argument("--client-id", nargs="*", default=None,
                    help="keep only these contributor id(s)")
    ap.add_argument("--client-id-file", type=Path,
                    help="a file of contributor ids, one per line, '#' for comments")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not (CV / "transcript/yue").is_dir():
        print(f"CV_ROOT is not a Common Voice yue download (no transcript/yue under {str(CV)!r})", file=sys.stderr)
        return 2
    want_ids = set(args.client_id or [])
    if args.client_id_file:
        if not args.client_id_file.exists():
            print(f"no contributor id file at {args.client_id_file}", file=sys.stderr)
            return 2
        for line in args.client_id_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                want_ids.add(line)
        if not want_ids:
            print(f"{args.client_id_file} holds no ids", file=sys.stderr)
            return 2
    if want_ids:
        print(f"contributor filter: {len(want_ids)} id(s), "
              + ", ".join(sorted(i[:8] + "..." for i in want_ids)))

    durs = {}
    for row in read_tsv(CV / "transcript/yue/clip_durations.tsv"):
        try:
            durs[row["clip"]] = float(row["duration[ms]"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            pass

    seen, rows = set(), []
    for sp in args.splits:
        for r in read_tsv(CV / f"transcript/yue/{sp}.tsv"):
            p = r.get("path")
            if p and p not in seen:
                seen.add(p)
                rows.append(r)
    print(f"pool (dedup across {'+'.join(args.splits)}): {len(rows)} clips")

    stage = []
    n_dur = n_gender = n_votes = n_colloq = n_accent = n_client = 0
    for r in rows:
        # THE DONOR FILTER RUNS FIRST, because when it is set every other counter
        # is a statement about one person and reading them as corpus-wide yields
        # would be wrong.
        if want_ids and (r.get("client_id") or "") not in want_ids:
            n_client += 1
            continue
        d = durs.get(r["path"])
        if d is None or not (args.min_duration <= d <= args.max_duration):
            n_dur += 1
            continue
        if args.gender and (r.get("gender") or "").strip() != args.gender:
            n_gender += 1
            continue
        if args.accent_contains:
            acc = r.get("accents") or ""
            if not any(k in acc for k in args.accent_contains):
                n_accent += 1
                continue
        if args.drop_down_voted:
            try:
                if int(r.get("down_votes") or 0) > 0:
                    n_votes += 1
                    continue
            except ValueError:
                pass
        sent = (r.get("sentence") or "").strip()
        if not sent:
            n_colloq += 1
            continue
        if args.require_colloquial and not any(c in CANTO for c in sent):
            n_colloq += 1
            continue
        stage.append({"path": r["path"], "sentence": sent,
                      "client_id": r.get("client_id", ""), "dur": d})

    if want_ids:
        print(f"  dropped by contributor filter               : {n_client}")
    print(f"  dropped by duration ({args.min_duration:g}-{args.max_duration:g}s): {n_dur}")
    print(f"  dropped by gender != {args.gender}          : {n_gender}")
    print(f"  dropped by accent filter                   : {n_accent}")
    print(f"  dropped by down_votes > 0                   : {n_votes}")
    print(f"  dropped by empty/non-colloquial transcript  : {n_colloq}")
    print(f"  KEPT: {len(stage)} clips")
    if stage:
        secs = [s["dur"] for s in stage]
        print(f"  hours={sum(secs)/3600:.2f}  median={statistics.median(secs):.2f}s  "
              f"speakers={len({s['client_id'] for s in stage})}")

    if not stage:
        print("nothing selected", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(stage)
    if args.limit:
        stage = stage[:args.limit]
        print(f"  limited to {len(stage)}")

    want = {s["path"]: s for s in stage}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    found = 0
    for tar_path in sorted((CV / "audio/yue").rglob("*.tar")):
        remaining = {k for k in want if not (audio_dir / k).exists()}
        if not remaining:
            break
        with tarfile.open(tar_path) as tf:
            for m in tf:
                base = Path(m.name).name
                if base in remaining:
                    src = tf.extractfile(m)
                    if src:
                        (audio_dir / base).write_bytes(src.read())
                        found += 1
        print(f"  {tar_path.name}: extracted so far {found}/{len(want)}")

    tsv = args.out_dir / "transcripts.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as fh:
        for s in stage:
            if (audio_dir / s["path"]).exists():
                fh.write(f"{s['path']}\t{s['sentence']}\n")

    meta = args.out_dir / "selection.json"
    meta.write_text(json.dumps(
        {"kept": len(stage), "extracted": found,
         "filters": {"splits": args.splits, "gender": args.gender,
                     "duration": [args.min_duration, args.max_duration],
                     "require_colloquial": args.require_colloquial},
         "dropped": {"contributor": n_client, "duration": n_dur, "gender": n_gender,
                     "down_votes": n_votes, "transcript": n_colloq}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nextracted {found} clips -> {audio_dir}")
    print(f"wrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
