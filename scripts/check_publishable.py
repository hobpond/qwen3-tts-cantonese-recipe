"""Decide whether a file may cross into the repository intended for publication.

WHY THIS IS A SCRIPT AND NOT A PARAGRAPH. `.gitignore` blocks `*.jsonl` wholesale
because prepared/train manifests carry `audio_codes` that decode back to
near-original audio, and one reached a public fork on 2026-08-24 -- after 2,004
corpus .wav files were already committed, which ignoring does not untrack. The one
narrow exception, for `benchmarks/EXPERIMENTS.jsonl`, documents its own standard:

    verified 0 CJK characters, 0 contributor identifiers, no audio_codes

That verification was done once, by hand, in prose. It is now FALSE: that file
currently holds 8 CJK characters. They are harmless -- 要緊, 關係 and the known-hard
set, written by me in a block note ABOUT a detector -- but the claim gating the file
is no longer true of the file, and nobody would have noticed. Granting a second
exception on the same basis would inherit a gate that has already drifted.

So the standard becomes executable. Run it, do not quote it.

    python scripts/common/check_publishable.py --path benchmarks/QUEUE.jsonl
    python scripts/common/check_publishable.py --self-test

THE TEST THAT MATTERS is not "does this file mention restricted material" but
"could any of it be RECONSTRUCTED from this file". Those give very different
answers, and only the second one decides. A stage log naming a clip is a reference;
a manifest carrying that clip's codec tokens is the clip.

Exit 0 publishable, 1 blocked, 2 usage.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

HAN = re.compile(r"[一-鿿]")
# The audio_codes PAYLOAD, not the field name. Matching the bare name blocked every
# script that merely reads the field -- dataset.py, prepare_data.py, the preflight
# queue's `r.get("audio_codes")` guard. Those carry no codes; they are how codes are
# kept out of things. Requiring the list separates carrying from handling.
CODES = re.compile(r'"audio_codes"\s*:\s*\[')
# ...AND THE SAME PAYLOAD UNDER ANY OTHER KEY, which the rule above cannot see.
#
# WHAT DOES *NOT* EVADE IT, checked first and recorded because it is the plausible
# story and it is wrong. Building the row by interpolation --
#
#     row = '{"audio_codes": [' + str(P) + ']}'
#
# -- evades nothing. `CODES` matches that source line, because the key and its opening
# bracket, `"audio_codes": []` less the close, are right there in it (written closed here
# so this comment measures as an empty array: the recipe repository's copy refuses an
# unclosed one, see check_clean_room.py), and it matches the emitted bytes too, for
# the same reason. Concatenation does not separate the key from the bracket in either
# the writer or the written. Verified against this regex before any of the below was
# written; do not re-derive it.
#
# WHAT ACTUALLY EVADES IT is the key itself, which is a free variable in a way the
# payload is not:
#
#     {"codes": [...]}              a rename, one word
#     {'audio_codes': [...]}        Python repr instead of json.dumps -- CODES
#                                   requires DOUBLE quotes, and `print(row)` on a dict
#                                   writes single ones
#     [[...], [...]]                a bare array in a .json file, no key at all
#
# `check_clean_room`'s base64 rules do not reach any of them either: decimal digits and
# commas are not a base64 run, and the decoded-magic test has nothing to decode. So a
# file carrying the same integers that leaked in 2026-08 passed the ENFORCED direction
# -- `--paths`, the call site `gcs-sync.yml` branches on between `rsync` and `git add`.
# Demonstrated through that call site before this rule existed, not theorised.
#
# The second net therefore refuses on SHAPE: a large integer-only bracketed literal,
# whatever key introduces it, or none. The key was the attacker's free variable; the
# integers are not, the same way the extension is free in check_clean_room.py and the
# audio is not.
#
# THE THRESHOLD IS MEASURED IN BOTH DIRECTIONS, the same way OPAQUE_MIN_B64 is in
# check_clean_room.py, because a number picked for feel is the thing that gets the gate
# switched off.
#
#   WHAT THE REPOSITORY LEGITIMATELY CARRIES. Every tracked file was walked for the
#   largest bracketed region containing nothing but integers, separators and nested
#   brackets. 284 files contain one. The ceiling is **48** -- a 24-pair defect-loci
#   list in benchmarks/exp000/results.json and the same rows in EXPERIMENTS.jsonl.
#   Five files reach 32 or more; NONE reaches 64. Test fixtures, seeds, tensor shapes
#   and segment-index lists all sit far below.
#
#   WHAT A PAYLOAD LOOKS LIKE. Dataset-Cantonese-Training/dataset.py annotates the
#   field `# t, 16` -- sixteen codebook tokens per frame -- and benchmarks/codes/
#   control-n92.json puts the codec at exactly 12.5 frames per second (222 frames /
#   17.76 s, and 140 / 11.2, and 113 / 9.04). That is **200 integers per second of
#   audio**. cc0_voice6 is 343 clips in 0.54 h, a 5.7 s mean clip, so one average row
#   of a prepared manifest is about 1,130 integers.
#
# 256 sits 5.3x above everything the tree actually holds and catches any clip longer
# than 1.28 seconds -- under a quarter of one average clip here.
#
# WHAT IT COSTS: a future integer array between 49 and 255 tokens is refused. The known
# growth path is the defect-loci list, which scales with clips judged; the largest
# listening study run so far judged 82 clips and produced 24 pairs, so reaching 128
# pairs needs a study roughly five times the biggest yet attempted. When that happens
# the answer is to re-measure and raise this number with the new ceiling recorded
# beside it, not to switch the rule off.
#
# WHAT IT LETS THROUGH, stated rather than left to be discovered:
#   * a payload under 256 integers -- less than 1.3 s of audio, under a quarter of one
#     clip. A whole manifest row, and any file of rows, is an order of magnitude past
#     it.
#   * a payload deliberately chunked into sub-256 pieces by someone who has read this
#     constant. No content threshold survives an adversary who knows it; this rule is
#     for the accidental and the shape-shifted, and `CODES` above -- which has no size
#     floor at all -- remains the primary net for the ordinary key form.
#   * FLOATS. The walk requires integers: no decimal point, no exponent. That is what
#     spares the 32-float latency arrays in benchmarks/*.json and every future
#     embedding, probability vector or confusion matrix, which is the single biggest
#     false-positive saving available here. It costs nothing against real codes, which
#     are integral by construction -- dataset.py reads them straight into a
#     `torch.long` tensor -- but a payload rewritten as `1.0, 2.0` evades this rule.
NUMERIC_MIN_TOKENS = 256

_INT_LEAD = frozenset("-0123456789")
_DIGITS = frozenset("0123456789")
_ARRAY_SEP = frozenset(", \t\r\n")


def max_int_array(raw: str) -> int:
    """Largest count of integer tokens in any bracketed region holding NOTHING BUT
    integers, separators and nested brackets.

    A bracket-depth walk rather than a regex, because `audio_codes` is a list of lists
    -- sixteen tokens per frame, one list per frame -- and the count that matters is
    the whole payload, not one frame. A regex for a flat numeric array would read the
    real shape as thousands of separate 16-element arrays and never fire.

    Any character outside the integer/separator/bracket alphabet abandons the region,
    so `[0, "x"]`, `x[0:10]`, `[1, 2, ...]` with an ellipsis and a float array are all
    counted as nothing. A regex character class such as `[0-9]` yields 2 and a slice
    yields 0; neither is anywhere near the threshold.
    """
    best = 0
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "[":
            i += 1
            continue
        depth = 0
        count = 0
        j = i
        ok = True
        while j < n:
            c = raw[j]
            if c == "[":
                depth += 1
                j += 1
            elif c == "]":
                depth -= 1
                j += 1
                if depth == 0:
                    break
            elif c in _ARRAY_SEP:
                j += 1
            elif c in _INT_LEAD:
                k = j + 1
                while k < n and raw[k] in _DIGITS:
                    k += 1
                count += 1
                j = k
            else:
                ok = False
                break
        if ok and depth == 0:
            if count > best:
                best = count
            i = j          # the region is consumed, and its nested brackets with it
        else:
            i += 1
    return best


# Numbered clips under the research corpus's audio directory.
#
# THE DIRECTORY IS OFTEN REACHED THROUGH A VARIABLE. Matching only the literal
# path missed scripts/experiments/codepred/rank_sweep.sh and three lane scripts in the predecessor,
# every one of which sets DATA=$REPO/Dataset-Cantonese-Training and then names
# `$DATA/audio/8026.wav` -- beside REFTEXT holding that clip's verbatim
# transcript. A missed clip is worse than a false positive: a false positive
# gets argued with, a miss reports clean. Any `<var>/audio/<digits>.wav` counts.
# Synthesised renders are safe from this: they are named t09k101_001.wav and
# held-7668.tuned.wav, never bare digits under audio/.
CORPUS_CLIP = re.compile(
    r"Dataset-Cantonese-Training/audio/"
    r"|(?<![\w.])\./audio/\d+\.wav"
    r"|\$\{?\w+\}?/audio/\d+\.wav"
    # ...and pathlib joins it the same way python does:
    #   VOICE = DATA / "audio" / "7668.wav"
    # overnight_plan.py named the reference voice for a whole night of scoring
    # like this, and the two patterns above walked straight past it.
    r'|/\s*"audio"\s*/\s*"\d+\.wav"'
    r'|/\s*"audio/\d+\.wav"')
# Common Voice client_ids are 128 hex characters. Git and HuggingFace object hashes
# are 40, and a model snapshot path carries one -- the first version of this matched
# 40+ and BLOCKED QUEUE.jsonl on
# `snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe`, a model revision with no
# person behind it. A gate that blocks on a model hash gets overridden by hand, and
# an overridden gate is not a gate.
#
# 64 DID NOT SEPARATE THEM EITHER, and "64 or more" is the reason why: a sha256
# digest is exactly 64 hex, so every Docker image digest, pip lock hash and frozen
# manifest checksum in this repository read as a contributor. That blocked
# gcp_fft_startup.sh, gcp_startup.sh, docker/Dockerfile, requirements.lock.txt and
# four benchmark result files, none of which has a person behind it. Match the
# client_id's own length exactly instead -- 128 -- which no content digest reaches.
#
# `client_id` as a bare word is the field NAME, and this project has eight scripts
# whose job is to read that column out of Common Voice -- audit, selection, audition,
# blend verification. None of them carries a contributor. Require the JSON shape of a
# row that holds one, the same handling-versus-carrying line drawn for audio_codes.
#
# The speaker-id rule below is deliberately NOT narrowed to match. It is the one that
# guards the de-identification this repository's history was rewritten twice to
# achieve, and scripts/common/harvest_ear_labels.py trips it on the render levers `spk16`
# and `spk54` -- which sit beside `t0`, `t06`, `tsamp` and `wide` and are not people.
# One cosmetic false positive is the cheaper mistake.
#
# A CC-0 client_id IS NOT A BLOCK, and this control used to say it was. That was a
# third over-match, in the same family as the 40-hex and 64-hex ones above but
# worse, because it contradicted the project's own recorded position rather than
# merely being noisy. `operations/dataset-licensing-and-publication.md` places the
# exposure in the VOICE -- "timbre *is* the identifier" -- and states the rule as
# never distributing weights or rendered audio that reproduce a single donor, never
# claiming a blended voice is anonymous. It does not say the pseudonym may not be
# written down; it discusses "real client_id values" in its own prose, and
# `benchmarks/audition/cvaud1.key.json` has carried them since the initial commit
# as the record of which contributor each audition trial heard.
#
# So the id is now REVIEW, not BLOCK. What stays blocking is a contributor's REAL
# NAME, which is the thing CC-0 genuinely does not cover and which no amount of
# licence makes publishable.
CONTRIBUTOR = re.compile(
    r"\bspk_?\d{2,}\b")
# ...AND THE RULE IS STILL NOT NARROWED. What follows is a carve-out of three named
# tokens, not a change to the pattern, and the difference is the point: narrowing the
# regex would weaken it against identifiers nobody has seen yet, while an enumerated
# list can only ever be wrong about the three things on it.
#
# WHY IT WAS NEEDED. CONTRIBUTOR-IDS fires on 12 of 853 tracked files. Ten of them are
# EXP-039/EXP-041 material -- benchmarks/BENCHMARKS.jsonl, benchmarks/EXPERIMENTS.jsonl
# and eight files under benchmarks/exp041/ -- and every hit in them is `spk343` or
# `spk777`, 278 and 281 occurrences. A reviewer facing twelve undeclared files where
# ten are the same two words in the same experiment stops reading the list, and a list
# nobody reads is the same as no list.
#
# VERIFIED, NOT ASSUMED. BENCHMARKS.jsonl's registered spec for BM-041-343V777 defines
# the arms in its own words: `"spk343": "exp041-spk343: cc0_voice6, 343 clips, 0.54 h"`
# and `"spk777": "exp041-spk777: cc0_voice6_v22, 777 clips, 1.19 h, the SAME
# contributor"`. EXPERIMENTS.jsonl's EXP-039 adds `"spk2457": "The frankenspeaker ...
# 2,457 distinct clips"`. The digits are a CLIP COUNT and the rungs of a data-size
# ladder; the contributor is `cc0_voice6`, which the spec names separately and which
# these three arms SHARE. Two of the three are the same person, so the token cannot be
# a speaker identifier -- it is what distinguishes two datasets cut from one donor.
#
# This project's actual identifiers for real people are the 128-hex Common Voice
# client_id (PSEUDONYM, below) and a donor label such as `cc0_voice6`. `spkNNN` is not
# the form it uses for a person anywhere in the tree.
#
# WHAT IT COSTS: if a contributor were ever assigned the literal id `spk343`, `spk777`
# or `spk2457`, this gate would not see it. That is three specific strings, chosen
# because their digits are recorded clip counts, against ten files of noise removed.
#
# WHAT IS DELIBERATELY *NOT* CARVED OUT: `spk16` and `spk54` in
# scripts/common/harvest_ear_labels.py, which are render levers sitting beside `t0`,
# `t06`, `tsamp` and `wide`. They stay a reported false positive because the file's own
# self-test requires `{"spk_id": {"spk16": 3000}}` to BLOCK -- the same token in a
# speaker slot IS a speaker id, and excusing the word would break the case that proves
# the rule still works. One cosmetic false positive is still the cheaper mistake; ten
# is not.
ARM_NAMES = frozenset({"spk343", "spk777", "spk2457"})
# A pseudonymous Common Voice contributor id: the 128-hex value, or a row carrying
# one. Reported so a human looks, not refused.
PSEUDONYM = re.compile(
    r'"client_id"\s*:\s*"'
    r"|\b[0-9a-f]{128}\b")
# A donor's real-world name, attached to a voice record. This is the line CC-0 does
# not reach: a pseudonym plus a name is no longer a pseudonym.
DONOR_NAME = re.compile(
    r'"(?:donor|contributor|speaker|voice)_?(?:real_?)?name"\s*:\s*"[^"]{2,}"'
    r'|"real_name"\s*:\s*"[^"]{2,}"', re.I)
# A real person's home directory, and therefore a username.
HOMEDIR = re.compile(r"/home/([A-Za-z0-9_.-]+)/|/Users/([A-Za-z0-9_.-]+)/")
CHECKOUT = re.compile(r"qwen3-tts-hk-cantonese-finetune|qwen3-tts-cantonese-lab")


def classify(raw: str) -> list[tuple[str, str, str]]:
    """[(severity, code, detail)]. severity is BLOCK or FIX."""
    out: list[tuple[str, str, str]] = []

    if CODES.search(raw):
        out.append(("BLOCK", "AUDIO-CODES",
                    "carries audio_codes, which decode back to near-original audio. "
                    "This is the exact payload that leaked in 2026-08."))

    n_ints = max_int_array(raw)
    if n_ints >= NUMERIC_MIN_TOKENS:
        out.append(("BLOCK", "NUMERIC-PAYLOAD",
                    f"a bracketed literal of {n_ints} integers. At this project's "
                    f"200 integers per second of audio that is {n_ints / 200:.1f}s of "
                    f"a codec payload, and the largest integer literal anywhere in "
                    f"the tracked tree is 48. The rule above needs the codes key "
                    f"beside the bracket, which is one word an author can rename or "
                    f"drop; this one reads the shape instead."))

    n_clip = len(CORPUS_CLIP.findall(raw))
    if n_clip:
        out.append(("BLOCK", "CORPUS-CLIPS",
                    f"{n_clip} reference(s) to research-corpus audio. Even as bare "
                    f"paths these tie published results to restricted material."))

    contrib = sorted(set(CONTRIBUTOR.findall(raw)) - ARM_NAMES)
    if contrib:
        out.append(("BLOCK", "CONTRIBUTOR-IDS",
                    f"contributor or speaker identifiers: {contrib[:5]}"))

    named = DONOR_NAME.findall(raw)
    if named:
        out.append(("BLOCK", "DONOR-NAME",
                    f"{len(named)} donor record(s) carrying a real-world name. CC-0 "
                    f"waives copyright, not privacy or personality rights, and a "
                    f"pseudonym beside a name is no longer a pseudonym."))

    if PSEUDONYM.search(raw):
        out.append(("REVIEW", "CONTRIBUTOR-PSEUDONYM",
                    "a pseudonymous Common Voice contributor id. Publishable under "
                    "the project's own reading -- the exposure is the voice, not the "
                    "identifier -- provided the surrounding material is CC-0 and no "
                    "real name is attached. If it keys research-corpus audio "
                    "instead, the CORPUS-CLIPS rule above is the one that applies."))

    homes = sorted({a or b for a, b in HOMEDIR.findall(raw)})
    if homes:
        out.append(("FIX", "USERNAME",
                    f"home directories expose a username: {homes}. Rewrite to "
                    f"$HOME or a placeholder before publishing."))

    n_checkout = len(CHECKOUT.findall(raw))
    if n_checkout:
        out.append(("FIX", "CHECKOUT-PATH",
                    f"{n_checkout} absolute path(s) into a named checkout. These "
                    f"make the file inert in any other clone and cross-link the "
                    f"repositories. Rewrite to $REPO_ROOT."))

    han = HAN.findall(raw)
    if han:
        out.append(("REVIEW", "CJK",
                    f"{len(han)} CJK character(s): {''.join(sorted(set(han))[:12])}. "
                    f"Transcripts are corpus content; prose ABOUT the language is "
                    f"not. A human decides which this is."))
    return out


# Fixtures for the NUMERIC-PAYLOAD rule.
#
# CLEAN ROOM: every integer below comes from an arithmetic progression written on this
# line. None of it was read from a manifest, a corpus file or any tokenizer, and none
# of it decodes to anything. What is under test is the gate's reaction to a SHAPE. They
# are also BUILT rather than written out, so this file does not itself acquire a
# 256-integer literal and start tripping the rule it defines -- the same reasoning
# check_clean_room.py's make_fixture() records for its must-be-refused page.
_PAYLOAD_300 = ", ".join(str((i * 7 + 13) % 1024) for i in range(300))
_PAYLOAD_255 = ", ".join(str((i * 7 + 13) % 1024) for i in range(255))
# 16 tokens per frame, 20 frames, nested exactly as dataset.py reads it: 320 integers.
_PAYLOAD_FRAMES = ", ".join(
    "[" + ", ".join(str((f * 16 + c) % 2048) for c in range(16)) + "]"
    for f in range(20))
# The measured ceiling of the real tree: a 24-pair defect-loci list, 48 integers.
_LOCI_48 = "[" + ", ".join(f"[{i}, {(i * 7) % 97}]" for i in range(24)) + "]"
# A 300-float latency array -- past the threshold in token count, and not integers.
_FLOATS_300 = "[" + ", ".join(f"{7.0 + i / 1000:.6f}" for i in range(300)) + "]"

SELF_TEST = [
    ("a prepared manifest -- the thing that actually leaked",
     '{"audio": "./audio/40.wav", "text": "你好", "audio_codes": [1,2,3]}',
     {"AUDIO-CODES", "CORPUS-CLIPS", "CJK"}),
    ("a queue log of synthesised renders",
     '{"t": "2026-09-05", "stage": "topk", "rc": 0, "note": "t09k101_001.wav"}',
     set()),
    ("the same, with a real home directory in it",
     '{"stage": "render", "run_dir": "/home/analyst/longtopk"}',
     {"USERNAME"}),
    ("a stage log naming a checkout",
     '{"stage": "x", "note": "/mnt/d/work/qwen3-tts-hk-cantonese-finetune/scripts/a.py"}',
     {"CHECKOUT-PATH"}),
    ("a model revision hash, which is not a person",
     '{"model": "snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe"}',
     set()),
    ("a benchmark keyed to a corpus speaker",
     '{"speaker": "spk_021", "clips": 1065}',
     {"CONTRIBUTOR-IDS"}),
    ("a corpus clip reached through a shell variable, beside its transcript",
     "DATA=$REPO/Dataset-Cantonese-Training\n"
     "REF=$DATA/audio/8026.wav\n"
     "REFTEXT='近日天氣'",
     {"CORPUS-CLIPS", "CJK"}),
    ("a synthesised render under an audio directory, which is not corpus",
     '{"note": "$OUT/audio/t09k101_001.wav"}',
     set()),
    ("a sha256 content digest, which is not a person",
     '{"image": "bench@sha256:'
     '04446ee2144fdca35715bed3eb3771b7a1f0eda67ccff4cea639789bfca19f41"}',
     set()),
    ("a Common Voice client_id, which is pseudonymous and therefore NOT a block",
     '{"client": "' + "3f566093" + "f" * 120 + '"}',
     {"CONTRIBUTOR-PSEUDONYM"}),
    ("code-handling script, which carries no codes",
     'have_codes = sum(1 for r in rows if r.get("audio_codes"))',
     set()),
    ("a corpus clip joined with pathlib rather than written as a path",
     'VOICE = DATA / "audio" / "7668.wav"',
     {"CORPUS-CLIPS"}),
    ("a script reading the client_id column, which carries no contributor",
     'speakers = {r.get("client_id") for r in rows if r.get("client_id")}',
     set()),
    ("an audition row holding one, which has shipped here since the first commit",
     '{"trial": 1, "client_id": "' + "a" * 128 + '"}',
     {"CONTRIBUTOR-PSEUDONYM"}),
    ("the same row with a real name beside it, which CC-0 does not reach",
     '{"trial": 1, "client_id": "' + "a" * 128 + '", "donor_name": "Chan Siu Ming"}',
     {"CONTRIBUTOR-PSEUDONYM", "DONOR-NAME"}),
    ("a baked speaker slot, which is still a block",
     '{"spk_id": {"spk16": 3000}}',
     {"CONTRIBUTOR-IDS"}),

    # --- NUMERIC-PAYLOAD: the key-split evasion, red then green -----------------
    ("the payload under a renamed key, which is a one-word edit CODES cannot see",
     '{"codes": [' + _PAYLOAD_300 + ']}',
     {"NUMERIC-PAYLOAD"}),
    ("the payload from Python repr rather than json.dumps -- SINGLE quotes, which "
     "is what print() on a dict writes and what CODES' double quotes miss",
     "{'audio_codes': [" + _PAYLOAD_300 + "]}",
     {"NUMERIC-PAYLOAD"}),
    ("codes in the shape dataset.py actually reads -- 16 per frame, nested, under no "
     "key at all. A flat-array regex would see twenty small arrays and never fire",
     "[" + _PAYLOAD_FRAMES + "]",
     {"NUMERIC-PAYLOAD"}),
    ("the ordinary key form still fires the ORIGINAL rule, which keeps its own job: "
     "it has no size floor and refuses three codes where this one would not",
     '{"audio_codes": [[1, 2], [3, 4]]}',
     {"AUDIO-CODES"}),
    ("both nets on one file, which is what an unrenamed full manifest row looks like",
     '{"audio_codes": [' + _PAYLOAD_300 + ']}',
     {"AUDIO-CODES", "NUMERIC-PAYLOAD"}),
    # The green half. A gate that refuses the repository's own benchmark data gets
    # switched off, which is the failure every calibration note in this file exists to
    # prevent.
    ("the largest integer literal in the whole tracked tree: a 24-pair defect-loci "
     "list, which is benchmark data and must never be refused",
     '{"marks": ' + _LOCI_48 + '}',
     set()),
    ("a 300-float latency array: past the count, and floats are not codec tokens",
     '{"latencies_s": ' + _FLOATS_300 + '}',
     set()),
    ("one integer under the threshold -- asserted so what the rule LETS THROUGH is "
     "a test result and not a guess",
     '{"x": [' + _PAYLOAD_255 + ']}',
     set()),
    ("a numeric array with a string in it, which is a record and not a payload",
     '{"rows": [1, 2, "spk", 4]}',
     set()),

    # --- CONTRIBUTOR-IDS: the arm-name carve-out, both directions ----------------
    ("an EXP-041 arm name, whose digits are the clip count its own spec records",
     '{"arms": {"spk343": "cc0_voice6, 343 clips, 0.54 h", '
     '"spk777": "cc0_voice6_v22, 777 clips, 1.19 h, the SAME contributor"}}',
     set()),
    ("the frankenspeaker rung, same reason",
     '{"arm": "spk2457", "clips": 2457}',
     set()),
    ("a token that merely BEGINS like an arm name is still a block -- the carve-out "
     "is three whole words, not a prefix",
     '{"speaker": "spk3430"}',
     {"CONTRIBUTOR-IDS"}),
    ("an arm name beside a real speaker id: carving the arm must not excuse the row",
     '{"arm": "spk343", "speaker": "spk_021"}',
     {"CONTRIBUTOR-IDS"}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.dry_run:
        print(f"  {len(SELF_TEST)} self-test cases")
        print("  DRY RUN OK")
        return 0

    if args.self_test:
        bad = 0
        for name, raw, want in SELF_TEST:
            got = {c for _, c, _ in classify(raw)}
            ok = got == want
            bad += not ok
            print(f"  {'PASS' if ok else '*** FAIL ***'}  {name}")
            if not ok:
                print(f"        expected {sorted(want) or '(clean)'}")
                print(f"        got      {sorted(got) or '(clean)'}")
        # A control nobody has watched fail is not a control. Both new rules are
        # asserted in both directions, so deleting either -- or widening it until it
        # refuses everything -- fails here rather than in the GCS pull.
        for code in ("NUMERIC-PAYLOAD", "CONTRIBUTOR-IDS"):
            red = sum(1 for _, _, w in SELF_TEST if code in w)
            green = sum(1 for _, _, w in SELF_TEST if code not in w)
            ok = red and green
            bad += not ok
            print(f"  {'PASS' if ok else '*** FAIL ***'}  {code} is proven in both "
                  f"directions ({red} red, {green} green)")
        print("  all controls behaved" if not bad
              else f"  {bad} control(s) misbehaved -- do not trust this gate")
        return 1 if bad else 0

    if not args.path:
        print("  give --path, or --self-test")
        return 2
    if not args.path.exists():
        print(f"  no such file: {args.path}")
        return 2

    raw = io.open(args.path, encoding="utf-8", errors="replace").read()
    hits = classify(raw)
    print(f"  {args.path}  ({len(raw)/1024:.0f} KB)")
    if not hits:
        print("  PUBLISHABLE -- nothing restricted, nothing reconstructable, "
              "no local identifiers")
        return 0
    for sev, code, detail in hits:
        print(f"  {sev:<6} {code}: {detail}")
    blocking = [h for h in hits if h[0] == "BLOCK"]
    fixable = [h for h in hits if h[0] == "FIX"]
    review = [h for h in hits if h[0] == "REVIEW"]
    print()
    if blocking:
        print(f"  BLOCKED. {len(blocking)} condition(s) that no rewrite fixes.")
    elif fixable:
        print(f"  NOT YET. {len(fixable)} rewritable condition(s); no blocking one. "
              f"Rewrite, re-run this, then have a human sign off.")
    else:
        print(f"  {len(review)} item(s) for human review; nothing blocking.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
