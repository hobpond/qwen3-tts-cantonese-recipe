#!/usr/bin/env python3
"""Refuse content, not extensions. The clean-room gate for anything entering this repo.

WHY AN EXTENSION GUARD CANNOT EXPRESS THIS HAZARD. `.gitignore` blocks `*.flac`,
`*.wav`, `*.mp3`, `*.jsonl`, `*.parquet` and ~31 literal `Dataset-Cantonese-Training`
paths. It does not block `.html`, `.md`, `.json`, `.csv` or `.tsv`. A listening page is
`.html` with every clip inlined as base64 by `audio_uri.flac_data_uri()` -- so **this
project's own publishing format sits in the gap its own guard does not cover.** The
`benchmarks/**/*.html` rule catches it under `benchmarks/`, and only there: a page
landing in `listen/`, `transfer/` or the repository root is carried by nothing.

The extension is the attacker's free variable. The audio is not. So the rule is a
content rule: it decodes what a file actually carries and refuses on the payload.

WHAT IT REFUSES, and why each rule exists rather than being an obvious idea:

  INLINE-AUDIO   a `data:` URI with an audio or video MIME type, or any long base64
                 run whose decoded first bytes are an audio container. Both halves are
                 needed. The MIME label is what a page writes today; the magic bytes
                 are what survives someone relabelling it `application/octet-stream`,
                 which is a single-word edit.
  BINARY-AUDIO   the file's OWN first bytes are an audio container. `notes.md` holding
                 FLAC bytes is ignored by nothing in `.gitignore`.
  OPAQUE-BLOB    a base64 run past the size threshold whose decode is not audio and
                 not text. Re-encoded, compressed or encrypted audio defeats the magic
                 check, and nothing in this repository legitimately inlines a payload
                 that size into a text file. Calibrated against the tree: see
                 OPAQUE_MIN_B64 below for the measurement, not a guess.

...and then it delegates. Corpus transcripts, clip filenames, `audio_codes` payloads,
donor names and speaker identifiers are already classified by
`scripts/common/check_publishable.py`, whose fifteen regexes carry a written history of
the false positives that shaped each one -- 40-hex model revisions, sha256 digests,
pathlib-joined clip paths, the eight scripts that merely READ `client_id`. That is one
fact with one home. **This file imports it and never restates it.** Its BLOCK rows
refuse here; its REVIEW and FIX rows are printed and do not refuse, because CJK prose
about Cantonese is not a Cantonese transcript and `corpus/public-domain/` is 57 tracked
files of deliberately-published public-domain text.

...and those delegated refusals then meet A DECLARED SELF-REFERENCE CARVE-OUT, because
A GUARD THAT IMPLEMENTS THE `audio_codes` PROHIBITION NECESSARILY NAMES `audio_codes`.
The tracked count used to be printed with the rule "expected to fall, never rise",
which no correct work can satisfy: writing a new guard raises it. See SELF_REFERENCE
below -- an enumerated list, never a heuristic, where each row pins the exact evidence
that was reviewed, so a carved-out file that grows a new clip path or a real-sized
payload still refuses.

    python scripts/check_clean_room.py --self-test
    python scripts/check_clean_room.py --tracked
    python scripts/check_clean_room.py --staged
    python scripts/check_clean_room.py --paths listen/ transfer/

IN THE RECIPE REPOSITORY (copied from the lab, 2026-10-22). Three changes, nothing else:

  * The CC-0 exemption is gone (`cc0_provenance`, `_cc0_exempt`, the `CJK-CC0` note).
    That exemption let proven-CC-0 audio into the lab; this repository carries NO data
    of any licence, so nothing is exempt and the code that decided exemptions would
    only be a way in.
  * The delegated `check_publishable` rules REFUSE on `--staged` and `--tracked` too
    (the lab reports them there). The lab's reason for reporting -- 24 of 835 of its
    own files name the hazard in prose and history -- does not hold here: this tree
    has no corpus prose and no history, so a transcript, a clip filename or a speaker
    id in a staged file is a leak, not a document about one. `--delegated report`
    restores the lab's behaviour if you need to read the notes.
  * SELF_REFERENCE names this repository's own guard files only.

CALL SITE. There is no hook and no CI here. Run it before every commit, by hand:

    python scripts/check_clean_room.py --staged

That is authority, not enforcement -- a thing a person runs, never a thing that stops
a commit. `--self-test` proves the rules fire; `--make-fixture` writes a planted page
that must be refused.

Exit 0 clean, 1 refused, 2 usage.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_publishable  # noqa: E402  the corpus/transcript/identifier rules live there

# A data: URI that declares audio or video AND CARRIES A PAYLOAD.
#
# THE PAYLOAD REQUIREMENT IS NOT COSMETIC -- it is the handling-versus-carrying line
# that check_publishable.py already had to draw for `audio_codes`, and the first draft
# of this file got it wrong in the other direction. Matching the bare label refused
# `scripts/common/audio_uri.py`, whose entire job is to keep this project's audio
# format consistent and which contains the literal
#
#     return "data:audio/flac;base64," + base64.b64encode(raw).decode("ascii")
#
# ...plus `make_listen_test.py`, `inline_page_audio.py`, `make_compare_page.py` and
# every other page builder in the tree. A gate that refuses the fourteen scripts whose
# purpose is producing the pages gets switched off in a week, and then it guards
# nothing. The self-test case for this is deliberately asserted GREEN.
#
# 128 is the separator. No builder in this tree writes 128 consecutive literal base64
# characters; a real inlined clip runs to hundreds of thousands. Verified by running
# --tracked over all 835 tracked files at c677ca7: zero refusals.
AUDIO_DATA_URI = re.compile(
    r"data:\s*(?:audio|video)/[\w.+-]+\s*;\s*base64\s*,\s*[A-Za-z0-9+/]{128,}", re.I)
# An audio MIME label with no payload behind it. Reported as REVIEW, never refused:
# it is what a builder looks like, and also what a truncated page looks like.
AUDIO_URI_LABEL = re.compile(
    r"data:\s*(?:audio|video)/[\w.+-]+\s*;\s*base64\s*,", re.I)
# A standalone base64 run. 256 is the floor at which a payload is worth decoding:
# below it there is no audio frame to find, and above it a run is no longer plausibly
# a hash, a key or an id.
B64_RUN = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")

# OPAQUE-BLOB threshold, MEASURED not guessed. The longest base64-alphabet run in any
# tracked file at c677ca7 is 128 characters (a Common Voice client_id; sha256 digests
# are 64). The largest tracked file is corpus/public-domain/三俠五義.prose.json at
# 3.19 MB of CJK prose, which contains no base64 run at all. 65536 leaves three orders
# of magnitude of headroom over the real tree while still catching a single inlined
# 48 KB clip -- about one second of 16 kHz FLAC, far shorter than any listening-page
# segment this project renders.
OPAQUE_MIN_B64 = 65536

# Container magics, checked against DECODED bytes. Each is the signature at the front
# of a real file, so a re-muxed clip is caught by its container even when the codec
# inside changed.
MAGICS: list[tuple[str, bytes, int]] = [
    ("FLAC", b"fLaC", 0),
    ("OggS", b"OggS", 0),        # Ogg Vorbis / Opus / Speex
    ("ID3", b"ID3", 0),          # MP3 with a tag
    ("AMR", b"#!AMR", 0),
    ("AU", b".snd", 0),
    ("Matroska/WebM", b"\x1a\x45\xdf\xa3", 0),
    ("Musepack", b"MPCK", 0),
    ("WavPack", b"wvpk", 0),
    ("Monkey's Audio", b"MAC ", 0),
    ("Speex", b"Speex", 0),
]


def _magic(head: bytes) -> str | None:
    """Name the audio/video container at the front of `head`, or None."""
    for name, sig, off in MAGICS:
        if head[off:off + len(sig)] == sig:
            return name
    # RIFF....WAVE and FORM....AIFF put the form type at offset 8, so the four bytes
    # in between are a length and must not be compared.
    if head[:4] == b"RIFF" and head[8:12] in (b"WAVE", b"AVI "):
        return "RIFF/WAVE"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "AIFF"
    # ISO base media: mp4, m4a, 3gp. 'ftyp' sits at offset 4, after the box size.
    if head[4:8] == b"ftyp":
        return "ISO-BMFF (mp4/m4a)"
    # A bare MPEG audio frame, no ID3 tag: 11 sync bits, and a layer field that is not
    # the reserved 00. Without the layer test this matches 0xFFE0 in arbitrary binary.
    #
    # THE BOM EXCLUSION IS NOT A LOOPHOLE, IT IS A MEASURED FALSE POSITIVE.
    # `\xff\xfe` is the UTF-16LE byte-order mark and `\xff\xfe\x00\x00` is UTF-32LE's.
    # Both satisfy the test above -- 0xFE & 0xE0 == 0xE0, and 0xFE & 0x06 == 0x06 -- so
    # before this exclusion EVERY UTF-16LE FILE WITH A BOM WAS REFUSED as a "MPEG audio
    # frame". Demonstrated, not theorised: a UTF-16LE copy of an ordinary prose note was
    # refused under that code, and this is a Windows-primary project where saving a
    # document as UTF-16 is a menu item. A gate that refuses ordinary text files with a
    # reason that is visibly wrong gets switched off, and then it guards nothing -- the
    # same failure the AUDIO-URI-BUILDER case above is kept inverted to prevent.
    #
    # WHAT IT COSTS: `\xff\xfe` is also a legal MPEG-1 Layer I frame header, so a bare
    # Layer I stream starting exactly there is no longer caught by its first two bytes.
    # That is paid for rather than dropped: scan_file() now decodes the NUL-stripped
    # view as well, so a UTF-16 carrier is caught by INLINE-AUDIO on its content, and
    # any real MPEG stream long enough to matter carries thousands of later sync words
    # and, base64-encoded, is caught by the same content rules. Layer I is effectively
    # unused; UTF-16LE text is not.
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0 \
            and (head[1] & 0x06) != 0 and head[:2] != b"\xff\xfe":
        return "MPEG audio frame"
    return None


def _decode_head(run: str, want: int = 64) -> bytes:
    """Decode the first bytes of a base64 run, tolerating a run that is not a
    whole number of quanta -- which is the normal case, since the run is a slice of
    a longer string."""
    quantum = (want * 4 // 3 + 3) // 4 * 4
    chunk = run[:quantum]
    chunk = chunk[:len(chunk) - len(chunk) % 4]
    if not chunk:
        return b""
    try:
        return base64.b64decode(chunk, validate=True)
    except (binascii.Error, ValueError):
        return b""


def _looks_textual(raw: bytes) -> bool:
    """Whether decoded bytes are plausibly text. Used only to spare an OPAQUE-BLOB
    verdict on a legitimately-embedded large text payload."""
    if not raw:
        return False
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(c >= 0x20 or c in (9, 10, 13) for c in raw)
    return printable / len(raw) > 0.9


def classify_bytes(head: bytes) -> list[tuple[str, str, str]]:
    """Rules that read the file's own leading bytes. [(severity, code, detail)]."""
    name = _magic(head)
    if name:
        return [("BLOCK", "BINARY-AUDIO",
                 f"the file's own first bytes are a {name} container. The extension "
                 f"is irrelevant: nothing in .gitignore reads content.")]
    return []


# ---------------------------------------------------------------------------
# THE SELF-REFERENCE CARVE-OUT. A DECLARED LIST, NEVER A HEURISTIC.
# ---------------------------------------------------------------------------
#
# WHY IT EXISTS. The `--delegated refuse` count over the tracked tree used to be
# reported as a single number that "is expected to fall, never rise. It rising means
# new source named the hazard." **That rule cannot be satisfied by correct work, and
# it was demonstrated rather than argued.** Four of the 26 files in that baseline are
# `check_clean_room.py`, `check_publishable.py`, `check_bootstrap.py` and
# `assert_dataset.py`, every one of them there because A GUARD THAT IMPLEMENTS THE
# `audio_codes` PROHIBITION CARRIES A SYNTHETIC FIXTURE SHAPED LIKE A PAYLOAD.
#
#   CORRECTED 2026-09-14. This read "...NECESSARILY NAMES `audio_codes`", which is
#   FALSE, and the correction changes how the list below is maintained.
#   `check_publishable.CODES` is `"audio_codes"\s*:\s*\[` -- it requires the JSON shape
#   and DELIBERATELY does not match the bare name, with the reason recorded beside it:
#   matching the bare name blocked every guard that merely handles codes. So naming the
#   field trips NOTHING, and a file mentioning `audio_codes` only in prose needs no
#   carve-out row. What puts these four in the baseline is their FIXTURES -- real
#   objects of the matched shape. That is a narrower problem, and it is why the
#   evidence below is a MEASURED COUNT and not a name exclusion: what must be told
#   apart is a three-integer fixture from a ten-thousand-integer payload, and both are
#   genuine matches.
#
# When TASK-LAB-05 added a
# byte-identity check to `freeze_manifest.py` with the synthetic fixture
# `{"id": "synthetic-0001", "text": "AAAA", "audio_codes": [1, 2, 3]}`, the count rose
# to 27 and read as a breach. It was not one. The counter could not tell a leak from a
# new guard, so it fired on exactly the work the repository wants.
#
# WHY A DECLARED LIST AND NOT A PATTERN. The obvious repair -- "skip files that look
# like guards", "skip `scripts/common/check_*.py`", "skip a hit inside a string
# literal" -- is a NEW AND SILENT WAY TO LET REAL CONTENT THROUGH. A heuristic decides
# at scan time, invisibly, and nobody can enumerate what it currently excuses. This
# list can be read. Every row says which file, which code, WHAT was reviewed, and WHY.
# Adding a row is a diff someone reviews; it cannot happen as a side effect.
#
# WHAT A ROW EXCUSES -- THE FIELD-NAME CLASS OF HIT, NOT THE FILE. A carved-out file
# is scanned exactly like every other file. Its owned rules (INLINE-AUDIO,
# BINARY-AUDIO, OPAQUE-BLOB) are untouched. Its delegated hit is suppressed only while
# the file still shows THE SAME EVIDENCE THAT WAS REVIEWED. New evidence of the same
# code refuses. That is the whole design: the row pins a fact, not a filename.
#
# WHAT "EVIDENCE" IS, AND WHY IT IS NOT UNIFORM ACROSS THE FOUR CODES. For three of
# them the regex match IS the whole fact, so the evidence is the set of DISTINCT
# matched strings -- naming `./audio/40.wav` twice is one clip named twice, and the
# information content is the set.
#
#   CORPUS-CLIPS   the match, EXTENDED RIGHTWARD OVER THE PATH TOKEN. This extension
#                  is load-bearing and is the hole a naive version would leave:
#                  check_publishable's first alternative matches the bare directory
#                  `Dataset-Cantonese-Training/audio/`, so `.../audio/8318.wav` and
#                  `.../audio/9999.wav` produce the IDENTICAL match. Pinning the raw
#                  match would let a carved-out file grow an entire corpus index
#                  behind one reviewed string. Extended, the directory reads as
#                  `Dataset-Cantonese-Training/audio/` -- no clip named -- and every
#                  distinct clip is its own evidence that nobody has declared.
#   CONTRIBUTOR-IDS, DONOR-NAME
#                  the match verbatim; it already carries the identifier or the name.
#
# AUDIO-CODES IS THE EXCEPTION AND THE DANGEROUS ONE. Its pattern ends at the opening
# bracket of `"audio_codes": [ ]` -- the payload, which is the entire hazard, lies
# OUTSIDE the match. Every match is textually identical whether the array holds three
# integers or three million. So its evidence is measured instead of quoted: the array
# is walked with bracket depth and its numeric tokens counted, and the evidence is the
# count. Multiplicity is kept here where it is dropped elsewhere, because two arrays
# of three codes are two payload fragments, not one fact said twice -- without that,
# one reviewed `codes:3` would excuse ten thousand three-element arrays carrying a
# whole clip between them.
#
# WHAT THAT COSTS, STATED RATHER THAN DISCOVERED: an array that does not close inside
# AUDIO_CODES_WINDOW measures as `codes:unbounded`, and no row may excuse one. So
# prose in a carved-out file must CLOSE ITS BRACKETS -- writing the label with a
# dangling `[` in a sentence makes that file refuse until the sentence is repaired or
# the payload is read. That is the rule biting where it should: the scanner cannot
# know that an unterminated array is a sentence rather than a manifest, and guessing
# is the heuristic this whole section exists to avoid.
#
# The delegated severities a carve-out row may touch. check_publishable's REVIEW and
# FIX rows never refused here, so they have nothing to excuse.
DELEGATED_BLOCK_CODES = {"AUDIO-CODES", "CORPUS-CLIPS", "CONTRIBUTOR-IDS",
                         "DONOR-NAME"}
AUDIO_CODES_WINDOW = 4096     # chars scanned for an array's close; see _codes_evidence
# The tail of a path token: what follows the delegated CORPUS-CLIPS match and turns a
# directory reference into a clip reference. `/` is deliberately excluded so the
# evidence stops at the next path component.
_PATH_TAIL = re.compile(r"[\w.*-]*")


def _codes_evidence(raw: str) -> list[str]:
    """Measure every `"audio_codes": [ ... ]` payload. One string per match, in file
    order, kept WITH multiplicity.

    `codes:N` is an array that closed within AUDIO_CODES_WINDOW holding N numeric
    tokens, counted across nesting so `[[0]]` is one code and not zero.
    `codes:unbounded` is an array that did not close in that window -- roughly 1000
    codes, far past anything a fixture writes. An unbounded payload is NEVER
    excusable: see _carved_out(), which refuses to honour a row that names one.
    """
    out: list[str] = []
    for m in check_publishable.CODES.finditer(raw):
        i = m.end() - 1                       # the '[' the delegated pattern ends on
        end = min(len(raw), i + AUDIO_CODES_WINDOW)
        depth = 0
        n = 0
        closed = False
        j = i
        while j < end:
            ch = raw[j]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth <= 0:
                    closed = True
                    break
            elif ch.isdigit():
                while j + 1 < end and (raw[j + 1].isdigit() or raw[j + 1] == "."):
                    j += 1
                n += 1
            j += 1
        out.append(f"codes:{n}" if closed else "codes:unbounded")
    return out


def delegated_evidence(code: str, raw: str) -> tuple[str, ...]:
    """What `raw` actually shows for a delegated BLOCK `code`: everything a carve-out
    row must have reviewed. Sorted, so a row is stable under reordering.

    The patterns are BORROWED FROM check_publishable, never restated here. One fact
    with one home: if that file narrows CORPUS_CLIP tomorrow, this narrows with it and
    there is no second copy to drift.
    """
    if code == "AUDIO-CODES":
        return tuple(sorted(_codes_evidence(raw)))
    if code == "CORPUS-CLIPS":
        found = set()
        for m in check_publishable.CORPUS_CLIP.finditer(raw):
            # A trailing dot is the end of an English sentence, never the end of a
            # path token, and leaving it in makes the same directory reference read
            # as two different facts depending on where in a paragraph it fell.
            found.add((m.group(0)
                       + _PATH_TAIL.match(raw, m.end()).group(0)).rstrip("."))
        return tuple(sorted(found))
    if code == "CONTRIBUTOR-IDS":
        return tuple(sorted({m.group(0)
                             for m in check_publishable.CONTRIBUTOR.finditer(raw)}))
    if code == "DONOR-NAME":
        return tuple(sorted({m.group(0)
                             for m in check_publishable.DONOR_NAME.finditer(raw)}))
    return ()


# THE LIST. Repository-relative POSIX path, delegated code, the exact evidence that
# was read, and why that evidence is the file naming the hazard rather than carrying
# it. Rows are grouped by the reason they exist.
#
# THE BOUNDARY OF THIS LIST IS "FILES WHOSE JOB IS THE PROHIBITION": the guards that
# implement it, their synthetic fixtures, the ignore rules that express it, the
# registry that declares which corpus is restricted, and the documents whose subject
# it is. It deliberately does NOT cover the benchmark result files that trip
# CONTRIBUTOR-IDS on `spk343`/`spk777` -- those are ARM NAMES encoding a clip count
# ("343 clips, 0.54 h"), a false positive of a rule check_publishable says it is
# leaving wide on purpose ("One cosmetic false positive is the cheaper mistake").
# Repairing that belongs in that rule, not in an excuse list here, and widening this
# list to cover data files is how a carve-out becomes the thing it was written to
# prevent.
SELF_REFERENCE: list[tuple[str, str, tuple[str, ...], str]] = [
    # --- the guards themselves: a rule cannot be written without naming its hazard --
    ("scripts/check_publishable.py", "AUDIO-CODES", ('codes:0', 'codes:0', 'codes:1', 'codes:3', 'codes:4'),
     "the self-test fixture for the manifest that leaked in 2026-08. Three "
     "hand-written integers; the file that DEFINES the audio_codes rule cannot "
     "assert the rule without writing the shape it refuses."),
    ("scripts/check_publishable.py", "CORPUS-CLIPS",
     ('$DATA/audio/8026.wav', './audio/40.wav', '/ "audio" / "7668.wav"',
      'Dataset-Cantonese-Training/audio/'),
     "CORPUS_CLIP's own four alternatives, each quoted beside the recorded false "
     "negative that put it there -- the shell-variable form, the pathlib-joined "
     "form -- plus the self-test fixtures that prove each one fires."),
    ("scripts/check_publishable.py", "CONTRIBUTOR-IDS",
     ('spk16', 'spk2457', 'spk343', 'spk3430', 'spk54', 'spk777', 'spk_021'),
     "spk16 and spk54 are quoted in the comment explaining why this rule is "
     "deliberately NOT narrowed -- they are render levers beside t0 and wide, not "
     "people. spk343, spk777 and spk2457 are ARM_NAMES (training arms named by clip "
     "count); spk_021 and spk3430 are self-test fixtures."),
    ("scripts/check_publishable.py", "DONOR-NAME",
     ('"donor_name": "Chan Siu Ming"',),
     "an invented name in the self-test fixture for the donor rule. No such "
     "contributor exists; the name is three common Cantonese syllables."),

    # THE SCANNER'S OWN ROWS. SELF_REFERENCE quotes the evidence of every other row, so
    # this file contains every string those rows excuse, plus its own synthetic
    # fixtures (8431 is the invented clip in the carve-out fixture; the FLAC payloads
    # are hand-written magic bytes). Each tuple below is computed from this file.
    ("scripts/check_clean_room.py", "AUDIO-CODES", ('codes:0', 'codes:0', 'codes:3', 'codes:3', 'codes:3', 'codes:3'),
     "the delegation self-test's fixture and the carve-out fixtures, none holding "
     "more than three integers; the oversized payload is BUILT at runtime from "
     "three, so its source line measures three."),
    ("scripts/check_clean_room.py", "CORPUS-CLIPS", ('$DATA/audio/8026.wav', './audio/40.wav', '/ "audio" / "7668.wav"', 'Dataset-Cantonese-Training/audio/', 'Dataset-Cantonese-Training/audio/1.wav', 'Dataset-Cantonese-Training/audio/8431.wav'),
     "this file's own self-test fixtures and the evidence quoted by the "
     "check_publishable.py row above. Every number names nothing: no corpus is "
     "in this repository and the paths are synthetic."),
    ("scripts/check_clean_room.py", "CONTRIBUTOR-IDS", ('spk16', 'spk2457', 'spk343', 'spk3430', 'spk54', 'spk777', 'spk_021'),
     "quoted from check_publishable.py's row above, and the self-test fixture; "
     "render levers and an invented id, not people."),
    ("scripts/check_clean_room.py", "DONOR-NAME", ('"donor_name": "A Fictional Name"', '"donor_name": "Chan Siu Ming"'),
     "this file's own fixture, which says in its value that it is fictional, and "
     "the literal quoted by the check_publishable.py row."),
]

_CARVED: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {
    (path, code): (evidence, why) for path, code, evidence, why in SELF_REFERENCE}


def _carved_out(carve_key: str | None, code: str, raw: str) -> tuple[bool, str]:
    """Is this delegated BLOCK covered by a declared row? (covered, explanation).

    THREE INDEPENDENT LOCKS, and all three must open.

    1. `carve_key` is set only for THIS REPOSITORY'S OWN tracked or staged files --
       main() passes None for `--paths`. The list says "this repository's file
       legitimately names the hazard", and a file ARRIVING from the bucket that
       merely shares a name is not that file. The enforced call site
       (`gcs-sync.yml`, `--paths "$STAGING"`) therefore cannot reach this carve-out
       at all, whatever the bucket names its contents.
    2. A row exists for exactly this (path, code). A row for AUDIO-CODES excuses
       nothing about CORPUS-CLIPS in the same file.
    3. The evidence found now equals the evidence reviewed then. This is what makes
       the row a pin on a fact rather than a hole punched for a filename.
    """
    if carve_key is None:
        return False, ""
    row = _CARVED.get((carve_key, code))
    if row is None:
        return False, ""
    declared, why = row
    found = delegated_evidence(code, raw)
    if any(e == "codes:unbounded" for e in found):
        # Belt and braces. An unbounded payload cannot appear in a declared row
        # either -- assert_carve_out_rows() fails the self-test if one does -- but a
        # list is edited by hand and this is the one hit no edit may excuse.
        return False, ""
    if found != declared:
        return False, ""
    return True, why


def assert_carve_out_rows() -> list[str]:
    """Structural faults in the list itself, independent of any repository. Returned
    as messages so the self-test can print them; empty means the list is well-formed.
    """
    bad: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path, code, evidence, why in SELF_REFERENCE:
        if (path, code) in seen:
            bad.append(f"duplicate row for {path} / {code}")
        seen.add((path, code))
        if code not in DELEGATED_BLOCK_CODES:
            bad.append(f"{path}: {code} is not a delegated BLOCK code")
        if not evidence:
            bad.append(f"{path} / {code}: empty evidence excuses everything")
        if tuple(sorted(evidence)) != tuple(evidence):
            bad.append(f"{path} / {code}: evidence must be sorted")
        if "codes:unbounded" in evidence:
            bad.append(f"{path} / {code}: an unbounded audio_codes payload is "
                       f"never excusable")
        if len(why) < 40:
            bad.append(f"{path} / {code}: a row must say WHY, at length")
        if "\\" in path:
            bad.append(f"{path}: use repository-relative POSIX paths")
    return bad


def classify_text(raw: str, arriving: bool = True,
                  carve_key: str | None = None) -> list[tuple[str, str, str]]:
    """Rules that read decoded text. [(severity, code, detail)].

    `arriving` says whether this content is entering the repository from outside it
    (a GCS pull, an untrusted drop) rather than already being part of it. It changes
    only the severity of the DELEGATED check_publishable rules -- see the note at the
    delegation below. The three rules this file owns refuse either way.

    `carve_key` is the repository-relative POSIX path when and only when this content
    is one of THIS repository's own tracked or staged files, and is what the
    self-reference carve-out keys on. None everywhere else.
    """
    out: list[tuple[str, str, str]] = []

    labelled = AUDIO_DATA_URI.findall(raw)
    if labelled:
        out.append(("BLOCK", "INLINE-AUDIO",
                    f"{len(labelled)} data: URI(s) declaring an audio or video MIME "
                    f"type with a payload behind it. This is what "
                    f"audio_uri.flac_data_uri() writes and what a listening page is "
                    f"made of."))
    elif AUDIO_URI_LABEL.search(raw):
        out.append(("REVIEW", "AUDIO-URI-BUILDER",
                    "an audio data: URI prefix with no payload behind it. This is what "
                    "a page BUILDER looks like -- audio_uri.py, make_listen_test.py, "
                    "inline_page_audio.py all carry the literal string and no audio. "
                    "Not refused; a human confirms it is the builder and not a "
                    "truncated page."))

    # Every long base64 run, whether or not it sits behind a data: URI, because the
    # MIME label is one word a page author can change.
    found_magic: list[str] = []
    opaque: list[int] = []
    for run in B64_RUN.findall(raw):
        head = _decode_head(run)
        name = _magic(head)
        if name:
            found_magic.append(name)
        elif len(run) >= OPAQUE_MIN_B64 and not _looks_textual(head):
            opaque.append(len(run))

    if found_magic and not labelled:
        out.append(("BLOCK", "INLINE-AUDIO",
                    f"base64 run(s) decoding to an audio container "
                    f"({sorted(set(found_magic))}) with no audio MIME label. A "
                    f"relabelled data: URI is a one-word edit; the magic bytes are "
                    f"not."))
    elif found_magic:
        out.append(("BLOCK", "INLINE-AUDIO",
                    f"payload confirmed by container magic: "
                    f"{sorted(set(found_magic))}."))

    if opaque:
        out.append(("BLOCK", "OPAQUE-BLOB",
                    f"{len(opaque)} base64 run(s) up to {max(opaque)} chars "
                    f"(~{max(opaque) * 3 // 4 // 1024} KB) decoding to neither audio "
                    f"nor text. Re-encoded or compressed audio defeats a magic check; "
                    f"nothing in this tree inlines an opaque payload this size."))

    # Delegate. One fact, one home: check_publishable.py owns the corpus-transcript,
    # clip-path, audio_codes, donor-name and speaker-id rules and the recorded history
    # of every false positive that narrowed them.
    #
    # ITS BLOCK ROWS ARE DOWNGRADED TO `NOTE` UNLESS THE CONTENT IS ARRIVING FROM
    # OUTSIDE THE REPOSITORY, and that is a measurement, not a preference. Run
    # tree-wide over all 835 tracked files at c677ca7, its BLOCK rules fire on **24 of
    # them** -- 12 CONTRIBUTOR-IDS, 11 CORPUS-CLIPS, 3 AUDIO-CODES, 2 DONOR-NAME --
    # including on `check_publishable.py` itself, which trips all four on its own
    # self-test fixtures, and on `assert_dataset.py`, `check_bootstrap.py`,
    # `harvest_ear_labels.py` and eight documents whose subject IS the hazard.
    # (26 of 853 by the time the carve-out above was written; the shape held.)
    #
    # None of the 24 carries corpus audio. Every one of them NAMES the hazard, which
    # is what a guard, a test fixture and a policy document all look like. The rules
    # were calibrated for one question -- "may this DATA FILE be published", run as
    # `--path benchmarks/QUEUE.jsonl` -- and that calibration does not transfer to
    # source and prose. The concept doc says as much in passing: "One cosmetic false
    # positive is the cheaper mistake." Cheap for a one-file manual run; fatal for a
    # gate on every commit, because a gate with a 3% false-refusal rate on the
    # repository's own source gets switched off, and then it guards nothing.
    #
    # So they refuse where their calibration holds -- bucket content arriving by
    # rsync, which is data and never this repository's source -- and they report
    # everywhere else. The count is printed either way, so drift is visible.
    for sev, code, detail in check_publishable.classify(raw):
        if sev != "BLOCK":
            out.append((sev, code, detail))
            continue
        covered, why = _carved_out(carve_key, code, raw)
        if covered:
            out.append(("EXCUSED", code, f"declared self-reference: {why}  "
                                         f"[evidence unchanged since review; new "
                                         f"evidence of this code refuses]"))
        elif not arriving:
            out.append(("NOTE", code, detail + "  [delegated rule, reporting only: "
                                               "see the calibration note in "
                                               "check_clean_room.py. Undeclared: no "
                                               "row in SELF_REFERENCE covers it]"))
        else:
            out.append((sev, code, detail))
    return out


# Severities that refuse. REVIEW, FIX and NOTE are printed and do not refuse: CJK prose
# about Cantonese is not a transcript, `corpus/public-domain/` is 57 files of
# deliberately-published text, and a username in a path is hygiene, not a licence
# breach.
REFUSING = {"BLOCK"}

READ_CAP = 24 * 1024 * 1024   # a listening page runs to tens of MB; read enough of it


def scan_file(path: Path, arriving: bool = True,
              carve_key: str | None = None) -> list[tuple[str, str, str]]:
    try:
        blob = path.open("rb").read(READ_CAP)
    except OSError as exc:
        return [("BLOCK", "UNREADABLE",
                 f"{exc}. An unreadable file cannot be certified, and this gate "
                 f"fails closed.")]
    hits = classify_bytes(blob[:64])
    if hits:
        return hits                                  # refused on its own bytes
    out = classify_text(blob.decode("utf-8", errors="replace"), arriving=arriving,
                        carve_key=carve_key)
    out += _classify_nul_stripped(blob, out, arriving=arriving, carve_key=carve_key)
    return out


def _classify_nul_stripped(blob: bytes, already: list[tuple[str, str, str]],
                           arriving: bool = True,
                           carve_key: str | None = None
                           ) -> list[tuple[str, str, str]]:
    """Scan the same bytes again with NUL bytes removed, to close the UTF-16 evasion.

    WHY THIS EXISTS. The UTF-8 decode above is the only view the gate had, and a
    UTF-16LE file interleaves a NUL after every ASCII character. `errors="replace"`
    keeps those NULs, so `data:audio/flac;base64,` arrives as `d\0a\0t\0a\0:\0...`
    and the base64 run arrives as several hundred one-character runs. Neither
    AUDIO_DATA_URI nor B64_RUN matches. MEASURED: a UTF-16LE note carrying the same
    inlined FLAC payload that is refused under UTF-8 passed the gate with exit 0.
    That is a live evasion on the ENFORCED path -- `gcs-sync.yml` pulls whatever the
    bucket holds, and nothing there guarantees UTF-8.

    WHY STRIPPING NULs RATHER THAN DETECTING AN ENCODING. Every payload these rules
    look for -- a `data:` URI, a base64 run, a container signature inside base64 --
    is entirely ASCII. Deleting NUL bytes turns UTF-16LE *and* UTF-16BE ASCII-range
    text into exactly that ASCII, with no BOM sniff, no chardet, and no guess that
    can be wrong. UTF-8 text contains no NUL at all, so for 833 of this repository's
    837 tracked files this function returns immediately without decoding anything.

    WHY IT CANNOT CRY WOLF. Only files containing a NUL are re-scanned: four tracked
    files today, all of them PNGs, and the tree-wide run stays at 0 refused. A code
    already reported on the UTF-8 view is not reported twice.
    """
    if b"\x00" not in blob:
        return []
    stripped = blob.replace(b"\x00", b"")
    if not stripped or stripped == blob:
        return []
    seen = {code for _, code, _ in already}
    out: list[tuple[str, str, str]] = []
    name = _magic(stripped[:64])
    if name:
        out.append(("BLOCK", "BINARY-AUDIO",
                    f"with NUL bytes removed the leading bytes are a {name} "
                    f"container. A UTF-16 file interleaves a NUL after every ASCII "
                    f"character, which is enough to hide a payload from a UTF-8 "
                    f"decode and not enough to stop it being audio."))
    for sev, code, detail in classify_text(
            stripped.decode("utf-8", errors="replace"), arriving=arriving,
            carve_key=carve_key):
        if code in seen:
            continue
        out.append((sev, code, detail +
                    "  [found in the NUL-stripped view: the payload was hidden from "
                    "the UTF-8 decode by UTF-16 interleaving]"))
    return out


def _git(*args: str) -> list[str]:
    """Collect paths from git as NUL-separated bytes, never as human-facing text.

    A GUARD MUST NOT PARSE GIT'S DISPLAY OUTPUT, AND THIS ONE DID. The first version
    split `git ls-files` on newlines. With `core.quotePath` at its default, git renders
    a non-ASCII path as a C-quoted string, so a file named `tëst-leak.md` arrived as the
    literal seven-character-escaped text `"t\\303\\253st-leak.md"`. The scanner built a
    Path from that, found no such file, and reported CLEAN with exit 0 while a file
    carrying an inlined FLAC data URI sat staged in the index. Demonstrated, not
    theorised: see the self-test case `a quoted non-ASCII path is still scanned`.

    A filename containing a newline defeats line splitting outright, and that is not
    reachable on Windows but is on the Linux runner where CI would execute.

    `-z` makes git emit raw bytes with NUL separators and disables quoting entirely, so
    there is nothing to unquote and nothing to mis-split. `core.quotePath=false` is set
    as well: belt and braces, and it documents the intent for the next reader.

    The lesson is not about filenames. It is that a security check reading a stream
    shaped for humans inherits every escaping rule that stream has, and each one is an
    evasion. The same class was reported independently on the dispatch pillar the same
    day, where five variants -- staged rename, embedded space, non-ASCII, symlinked
    directory, case -- each walked past a guard parsing `git status --porcelain`.
    """
    out = subprocess.run(["git", "-c", "core.quotePath=false", *args, "-z"],
                         capture_output=True)
    if out.returncode:
        print(f"  git {' '.join(args)} failed: "
              f"{out.stderr.decode('utf-8', errors='replace').strip()}")
        raise SystemExit(2)
    return [p.decode("utf-8", errors="surrogateescape")
            for p in out.stdout.split(b"\0") if p]


def collect(args: argparse.Namespace) -> list[Path]:
    if args.tracked:
        return [Path(p) for p in _git("ls-files")]
    if args.staged:
        # --diff-filter=ACMR includes renames, whose -z form emits TWO NUL-separated
        # fields per entry (old path then new). --name-only with -z gives one field per
        # path, and scanning an extra old path is harmless; missing a new one is not.
        return [Path(p) for p in
                _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")]
    files: list[Path] = []
    for root in args.paths:
        r = Path(root)
        if r.is_file():
            files.append(r)
        elif r.is_dir():
            files += [p for p in sorted(r.rglob("*"))
                      if p.is_file() and ".git" not in p.parts]
        else:
            print(f"  no such path: {r}")
            raise SystemExit(2)
    return files


# ---------------------------------------------------------------------------
# Self-test. Every fixture here is SYNTHESISED. No case carries corpus audio, a real
# clip filename, a real transcript or a real speaker id -- the clean-room rule applies
# to the test suite exactly as it applies to everything else, and a fixture built from
# real corpus audio would be the leak it is testing for.
#
# The FLAC bytes below are a hand-written 'fLaC' magic plus a STREAMINFO-shaped header
# and filler. They are not a decodable stream and no audio was involved in making them.
# What is being tested is the gate's reaction to a container signature.
# ---------------------------------------------------------------------------
_FAKE_FLAC = b"fLaC" + bytes([0x80, 0, 0, 0x22]) + bytes(range(34)) + b"\x00" * 400
_FAKE_WAV = b"RIFF" + (2048).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 512
_FAKE_OGG = b"OggS" + b"\x00\x02" + b"\x00" * 600
_B64_FLAC = base64.b64encode(_FAKE_FLAC).decode()
_B64_OGG = base64.b64encode(_FAKE_OGG).decode()
_B64_OPAQUE = base64.b64encode(bytes((i * 37 + 11) % 256 for i in range(60000))).decode()

SELF_TEST: list[tuple[str, str, set[str]]] = [
    # --- the hazard this file exists for -------------------------------------
    ("a listening page: .html, which .gitignore does not cover, with FLAC inlined",
     '<audio controls src="data:audio/flac;base64,' + _B64_FLAC + '"></audio>',
     {"INLINE-AUDIO"}),
    ("the same page with the MIME relabelled, which is a one-word edit",
     '<audio src="data:application/octet-stream;base64,' + _B64_FLAC + '"></audio>',
     {"INLINE-AUDIO"}),
    ("a bare base64 clip in a .md report, behind no data: URI at all",
     "Sample for review:\n\n    " + _B64_OGG + "\n",
     {"INLINE-AUDIO"}),
    ("a clip smuggled through a .json field",
     '{"clip": "' + _B64_FLAC + '", "note": "arm C render"}',
     {"INLINE-AUDIO"}),
    ("a WAV data URI, the other half of the MIME rule",
     'src="data:audio/wav;base64,' + base64.b64encode(_FAKE_WAV).decode() + '"',
     {"INLINE-AUDIO"}),
    # This case was written the other way round first, asserting that a bare audio
    # MIME label is refused. It is the case that caught the defect: run as a BLOCK it
    # refused scripts/common/audio_uri.py -- the one file whose job is keeping this
    # project's audio format consistent -- plus make_listen_test.py,
    # inline_page_audio.py and every other page builder. Kept, inverted, with the
    # reason attached, because the failure it prevents is the gate being switched off.
    ("an audio MIME label with no payload: the BUILDER's shape, reported not refused",
     'src="data:audio/flac;base64,AAAA"',
     {"AUDIO-URI-BUILDER"}),
    ("an opaque blob past the threshold: re-encoded audio defeats a magic check",
     '{"blob": "' + _B64_OPAQUE + '"}',
     {"OPAQUE-BLOB"}),

    # --- the delegated rules, confirming the import is wired ------------------
    ("a prepared manifest -- the payload that actually leaked in 2026-08",
     '{"audio": "./audio/40.wav", "audio_codes": [1,2,3]}',
     {"AUDIO-CODES", "CORPUS-CLIPS"}),
    ("a corpus clip reached through a shell variable",
     "DATA=$REPO/Dataset-Cantonese-Training\nREF=$DATA/audio/8026.wav",
     {"CORPUS-CLIPS"}),
    ("a donor's real name beside a voice record, the line CC-0 does not reach",
     '{"donor_name": "A Fictional Name"}',
     {"DONOR-NAME"}),

    # --- must stay green, or the gate gets disabled ---------------------------
    ("the page BUILDER verbatim from audio_uri.py, which writes data URIs and "
     "carries none",
     'return "data:audio/flac;base64," + base64.b64encode(raw).decode("ascii")',
     {"AUDIO-URI-BUILDER"}),
    ("a sha256 digest: 64 hex, below the 256-char run floor",
     '{"image": "bench@sha256:'
     '04446ee2144fdca35715bed3eb3771b7a1f0eda67ccff4cea639789bfca19f41"}',
     set()),
    ("a 128-hex Common Voice pseudonym: CC-0 and reviewable, never refused",
     '{"client_id": "' + "a" * 128 + '"}',
     {"CONTRIBUTOR-PSEUDONYM"}),
    ("public-domain CJK prose, 57 files of which are deliberately tracked",
     '{"line": "三俠五義 卷一"}',
     {"CJK"}),
    ("a long base64-alphabet run that decodes to text, e.g. an embedded certificate "
     "body or a pasted log -- opaque only if it is not text",
     '{"payload": "'
     + base64.b64encode(b"stage=render rc=0 " * 4000).decode() + '"}',
     set()),
    ("an inline SVG data URI, which is a diagram and not a clip",
     'src="data:image/svg+xml;base64,'
     + base64.b64encode(b"<svg>" + b" " * 400 + b"</svg>").decode() + '"',
     set()),
]

# Cases whose refusal must be proven at the FILE level, not the string level: the
# extension is what .gitignore reads, and the point is that this gate does not.
SELF_TEST_BYTES: list[tuple[str, bytes, set[str]]] = [
    ("a FLAC stream in a file named .md -- ignored by nothing",
     _FAKE_FLAC, {"BINARY-AUDIO"}),
    ("a WAV stream in a file named .json",
     _FAKE_WAV, {"BINARY-AUDIO"}),
    ("ordinary markdown",
     b"# Report\n\nThe arm C render was scored by ear.\n", set()),

    # --- UTF-16, both directions. Added after a negative control showed the gate
    # --- getting BOTH of these wrong: the carrier passed, and the plain text was
    # --- refused as audio. Red and green are asserted together because the fix to
    # --- either one alone is a regression in the other.
    ("a UTF-16LE page with a BOM carrying the same inlined FLAC: the carrier "
     "the UTF-8 decode could not see",
     b"\xff\xfe" + ('<audio src="data:audio/flac;base64,' + _B64_FLAC
                    + '"></audio>').encode("utf-16-le"),
     {"INLINE-AUDIO"}),
    ("the same with NO BOM, which is what a stripped export looks like",
     ('<audio src="data:audio/flac;base64,' + _B64_FLAC
      + '"></audio>').encode("utf-16-le"),
     {"INLINE-AUDIO"}),
    ("UTF-16BE, the other byte order -- NUL-stripping is order-agnostic",
     ('<audio src="data:audio/flac;base64,' + _B64_FLAC
      + '"></audio>').encode("utf-16-be"),
     {"INLINE-AUDIO"}),
    # THE GREEN HALF, and the reason the BOM exclusion in _magic() exists. Before it,
    # this case was refused as a "MPEG audio frame" because the UTF-16LE BOM satisfies
    # the bare-frame sync test. An ordinary note saved as UTF-16 by a Windows editor is
    # not audio, and a gate that says it is gets switched off.
    ("ordinary prose saved as UTF-16LE with a BOM: text, not an MPEG frame",
     b"\xff\xfe" + "# Report\n\nScored by ear, no payload here.\n".encode("utf-16-le"),
     set()),
    ("the same prose as UTF-16BE with a BOM",
     b"\xfe\xff" + "# Report\n\nScored by ear, no payload here.\n".encode("utf-16-be"),
     set()),
]


FIXTURE_NAME = "refused_listen_page.html"


def make_fixture(dest: Path) -> Path:
    """Write a synthetic listening page that this gate must refuse.

    THE FIXTURE IS GENERATED AND NEVER COMMITTED, and that is the point rather than an
    inconvenience. A file that must be refused, committed into the repository, is
    either a permanent refusal in every tree-wide run or a hole punched in the gate to
    excuse it -- and a gate with an excuse list is the shape of the failure this whole
    task exists to prevent. So the red test is a recipe, reproducible by anyone:

        python scripts/check_clean_room.py --make-fixture "$TMP"
        python scripts/check_clean_room.py --paths "$TMP"   # expect exit 1

    Nothing here is corpus-derived. The audio is a 'fLaC' magic plus a
    STREAMINFO-shaped header and deterministic filler; no clip, no transcript, no
    speaker id, no real filename. The page is shaped like the real thing -- the same
    `<audio src="data:audio/flac;base64,...">` that audio_uri.flac_data_uri() emits --
    because what is under test is the gate's reaction to the project's own publishing
    format.
    """
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / FIXTURE_NAME
    clip = base64.b64encode(_FAKE_FLAC * 12).decode()
    path.write_text(
        "<!-- SYNTHETIC FIXTURE. Generated by check_clean_room.py --make-fixture.\n"
        "     Must be REFUSED. Contains no corpus audio: the payload is a hand-written\n"
        "     FLAC container signature plus filler. Do not commit this file. -->\n"
        "<h1>Arm comparison, segment 3</h1>\n"
        '<audio controls src="data:audio/flac;base64,' + clip + '"></audio>\n',
        encoding="utf-8")
    return path


# The carve-out fixtures. SYNTHETIC, like every other fixture in this file: a guard
# that does not exist, a marker constant, and an invented clip number. 8431 indexes
# nothing -- the corpus is not in this repository, and the path is here to prove that
# a clip path NOT on the list refuses even inside a file that is.
_FAKE_GUARD = "scripts/common/a_guard_that_does_not_exist.py"
_SYN_GUARD_SRC = (
    'RESEARCH_MARKERS = ("Dataset-Cantonese-Training/audio/",)\n'
    'FIXTURE = {"id": "synthetic-0001", "text": "AAAA", "audio_codes": [1, 2, 3]}\n')
_SYN_GUARD_PLUS_CLIP = _SYN_GUARD_SRC + (
    'LEAK = "Dataset-Cantonese-Training/audio/8431.wav"\n')
_SYN_GUARD_BIG_CODES = (
    'RESEARCH_MARKERS = ("Dataset-Cantonese-Training/audio/",)\n'
    '{"audio_codes": [' + ", ".join(str((i * 7) % 900) for i in range(2000)) + ']}\n')


def _self_test_carve_out() -> int:
    """Assertions on the carve-out. Returns the number that misbehaved.

    A carve-out is a hole by construction, so the cases that matter are the ones
    where it must NOT open: five of the seven below are red. They run against a
    SYNTHETIC index rather than the production SELF_REFERENCE list, because what is
    under test is the mechanism -- the production list's own correctness is a
    different claim, asserted by assert_carve_out_rows() and by the staleness audit
    that every --tracked run prints.
    """
    bad = 0
    rows = assert_carve_out_rows()
    ok = not rows
    bad += not ok
    print(f"  {'PASS' if ok else '*** FAIL ***'}  "
          "every declared row is well-formed: sorted, non-empty, reasoned")
    for msg in rows:
        print(f"        {msg}")

    def sev(content: str, code: str, key: str | None) -> set[str]:
        return {s for s, c, _ in classify_text(content, arriving=True, carve_key=key)
                if c == code}

    synthetic = {
        (_FAKE_GUARD, "CORPUS-CLIPS"): (
            ("Dataset-Cantonese-Training/audio/",), "a marker constant under test"),
        (_FAKE_GUARD, "AUDIO-CODES"): (
            ("codes:3",), "a three-integer fixture under test"),
    }
    saved = dict(_CARVED)
    try:
        _CARVED.clear()
        _CARVED.update(synthetic)
        cases = [
            # GREEN: the one case the carve-out exists for.
            ("a declared guard's field-name hit is excused, not refused",
             sev(_SYN_GUARD_SRC, "AUDIO-CODES", _FAKE_GUARD), {"EXCUSED"}),
            ("...and so is its marker constant",
             sev(_SYN_GUARD_SRC, "CORPUS-CLIPS", _FAKE_GUARD), {"EXCUSED"}),
            # RED: a carved-out file that grows a clip path nobody declared.
            ("A DECLARED FILE THAT GAINS A CLIP PATH STILL REFUSES -- the row pins "
             "the evidence, not the filename",
             sev(_SYN_GUARD_PLUS_CLIP, "CORPUS-CLIPS", _FAKE_GUARD), {"BLOCK"}),
            ("...while its unchanged audio_codes hit stays excused, so the refusal "
             "names the one thing that moved",
             sev(_SYN_GUARD_PLUS_CLIP, "AUDIO-CODES", _FAKE_GUARD), {"EXCUSED"}),
            # RED: the payload that lies outside the match.
            ("A DECLARED FILE CARRYING A REAL-SIZED audio_codes PAYLOAD STILL "
             "REFUSES -- 2000 codes is not the three that were reviewed",
             sev(_SYN_GUARD_BIG_CODES, "AUDIO-CODES", _FAKE_GUARD), {"BLOCK"}),
            # RED: arriving content wearing a declared file's name.
            ("ARRIVING CONTENT IS NEVER EXCUSED, whatever the bucket called it: the "
             "enforced --paths call site passes no carve key",
             sev(_SYN_GUARD_SRC, "AUDIO-CODES", None), {"BLOCK"}),
            # RED: a row is per-code.
            ("a row for one code excuses nothing about another code in the same "
             "file",
             sev(_SYN_GUARD_SRC, "CONTRIBUTOR-IDS", _FAKE_GUARD) | {"(no hit)"},
             {"(no hit)"}),
        ]
        for name, got, want in cases:
            ok = got == want
            bad += not ok
            print(f"  {'PASS' if ok else '*** FAIL ***'}  {name}")
            if not ok:
                print(f"        expected {sorted(want)}  "
                      f"got {sorted(got) or '(none)'}")
    finally:
        _CARVED.clear()
        _CARVED.update(saved)

    # The measurement the AUDIO-CODES rows depend on, asserted directly: the count is
    # the only thing separating a fixture from a payload, and it is computed here
    # rather than matched by a pattern.
    measured = [
        ("[1, 2, 3]", ["codes:3"]),
        ("[[0]]", ["codes:1"]),
        ("[...]", ["codes:0"]),
        ("[" + ", ".join(str(i % 900) for i in range(2000)) + "]",
         ["codes:unbounded"]),
    ]
    for arr, want in measured:
        got = _codes_evidence('{"audio_codes": ' + arr + '}')
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else '*** FAIL ***'}  "
              f"audio_codes payload {arr[:14]}... measures {want[0]}")
        if not ok:
            print(f"        expected {want}  got {got}")

    # And the extension that stops one reviewed string from excusing a whole corpus
    # index: the directory and a clip under it must be DIFFERENT evidence.
    d = delegated_evidence("CORPUS-CLIPS", "Dataset-Cantonese-Training/audio/\n")
    c = delegated_evidence("CORPUS-CLIPS",
                           "Dataset-Cantonese-Training/audio/" + "8431.wav\n")
    ok = bool(d) and bool(c) and d != c
    bad += not ok
    print(f"  {'PASS' if ok else '*** FAIL ***'}  "
          "a clip path is different evidence from the directory holding it")
    if not ok:
        print(f"        directory {d}  clip {c}")
    return bad


def run_self_test() -> int:
    bad = 0
    for name, raw, want in SELF_TEST:
        got = {c for sev, c, _ in classify_text(raw)}
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else '*** FAIL ***'}  {name}")
        if not ok:
            print(f"        expected {sorted(want) or '(clean)'}")
            print(f"        got      {sorted(got) or '(clean)'}")
    for name, blob, want in SELF_TEST_BYTES:
        # Mirrors scan_file() exactly, including the NUL-stripped second view. It did
        # not before, which is why the UTF-16 evasion could not have been caught here:
        # a self-test that composes the classifier differently from production tests
        # a function nothing calls.
        hits = classify_bytes(blob[:64])
        if not hits:
            hits = classify_text(blob.decode("utf-8", errors="replace"))
            hits = hits + _classify_nul_stripped(blob, hits)
        got = {c for sev, c, _ in hits}
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else '*** FAIL ***'}  {name}")
        if not ok:
            print(f"        expected {sorted(want) or '(clean)'}")
            print(f"        got      {sorted(got) or '(clean)'}")

    # The arriving/not-arriving split is the one design decision in this file that a
    # reader is most likely to think weakens the gate, so it is asserted rather than
    # described. Two properties: the delegated rules downgrade, and the rules this
    # file OWNS do not.
    corpus = 'REF=$DATA/Dataset-Cantonese-Training/audio/1.wav'
    inline = 'src="data:audio/flac;base64,' + _B64_FLAC + '"'
    checks = [
        ("delegated rule refuses on arriving content",
         {s for s, c, _ in classify_text(corpus, arriving=True) if c == "CORPUS-CLIPS"},
         {"BLOCK"}),
        ("delegated rule only NOTEs on content already in the repository",
         {s for s, c, _ in classify_text(corpus, arriving=False) if c == "CORPUS-CLIPS"},
         {"NOTE"}),
        ("an OWNED rule refuses on arriving content",
         {s for s, c, _ in classify_text(inline, arriving=True) if c == "INLINE-AUDIO"},
         {"BLOCK"}),
        ("an OWNED rule refuses REGARDLESS -- the split must not reach these",
         {s for s, c, _ in classify_text(inline, arriving=False) if c == "INLINE-AUDIO"},
         {"BLOCK"}),
    ]
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else '*** FAIL ***'}  {name}")
        if not ok:
            print(f"        expected {sorted(want)}  got {sorted(got) or '(none)'}")

    # The self-reference carve-out, asserted in both directions -- mostly red.
    bad += _self_test_carve_out()

    # PATH COLLECTION IS PART OF THE GATE, AND IT WAS THE HOLE.
    #
    # Every case above tests classification: given content, is the verdict right. None of
    # them could catch the actual defect, which was that a file never reached
    # classification at all. A non-ASCII filename arrived C-quoted from `git ls-files`,
    # resolved to nothing, and the run reported CLEAN with exit 0 while an inlined FLAC
    # sat in the index. This asserts the collector on a real repository, because the bug
    # lived in the seam between git and the scanner and no pure-content test reaches it.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        prev = Path.cwd()
        try:
            os.chdir(td)
            subprocess.run(["git", "init", "-q"], check=True,
                           capture_output=True)
            subprocess.run(["git", "config", "user.email", "t@t"], check=True)
            subprocess.run(["git", "config", "user.name", "t"], check=True)
            # core.quotePath default ON is the condition that produced the evasion.
            subprocess.run(["git", "config", "core.quotePath", "true"], check=True)
            awkward = "tëst-leak.md"
            Path(awkward).write_text(
                'x <audio src="data:audio/flac;base64,' + _B64_FLAC + '"></audio>',
                encoding="utf-8")
            subprocess.run(["git", "add", "-f", "--", awkward], check=True,
                           capture_output=True)
            got = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
            ok = awkward in got and all(not p.startswith('"') for p in got)
            bad += not ok
            print(f"  {'PASS' if ok else '*** FAIL ***'}  "
                  "a quoted non-ASCII path is still scanned")
            if not ok:
                print(f"        expected [{awkward!r}] got {got!r}")
            # And the file it names must actually be readable, or the collector is
            # returning a name nothing can open -- which is how the evasion presented.
            ok2 = bool(got) and Path(got[0]).is_file()
            bad += not ok2
            print(f"  {'PASS' if ok2 else '*** FAIL ***'}  "
                  "the collected path resolves to a real file")
        finally:
            os.chdir(prev)

    # A gate that has never refused anything is untested, and a gate that refuses
    # everything is worse. Both directions are asserted, so a regression that makes
    # classify_text() return nothing -- or everything -- fails here rather than in
    # production.
    reds = sum(1 for _, _, w in SELF_TEST if w & {"INLINE-AUDIO", "OPAQUE-BLOB",
                                                  "BINARY-AUDIO"})
    greens = sum(1 for _, _, w in SELF_TEST if not (w & REFUSING_CODES))
    if not reds:
        print("  *** FAIL ***  no red case for the new rules: the gate is unproven")
        bad += 1
    if not greens:
        print("  *** FAIL ***  no green case: a gate that refuses everything "
              "gets disabled")
        bad += 1
    print(f"\n  {len(SELF_TEST) + len(SELF_TEST_BYTES)} cases, {reds} of them red "
          f"for the rules this file adds")
    print("  all controls behaved" if not bad
          else f"  {bad} control(s) misbehaved -- do not trust this gate")
    return 1 if bad else 0


# Codes that refuse, for the self-test's own green/red accounting.
# Every delegated code that can refuse. Used only by `--self-test` accounting, to count
# the cases expected NOT to refuse: a code missing here is silently counted as a green,
# which weakens the assertion rather than breaking it -- so a gap shows up as a test
# that passes, not as a failure. Enforcement itself is `REFUSING = {"BLOCK"}` above and
# does not read this set.
#
# `NUMERIC-PAYLOAD` added 2026-09-14: `check_publishable` gained it for a large numeric
# array literal irrespective of the key that introduces it, closing the evasion where
# the KEY is the free variable -- a renamed key, or Python's single-quoted repr.
REFUSING_CODES = {"INLINE-AUDIO", "OPAQUE-BLOB", "BINARY-AUDIO", "UNREADABLE",
                  "AUDIO-CODES", "CORPUS-CLIPS", "CONTRIBUTOR-IDS", "DONOR-NAME",
                  "NUMERIC-PAYLOAD"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Content-based clean-room gate. Refuses inlined audio and "
                    "corpus-derived material regardless of file extension.")
    ap.add_argument("--paths", nargs="+", default=[],
                    help="files or directories to scan (the GCS-pull call site)")
    ap.add_argument("--staged", action="store_true",
                    help="scan the git index (pre-commit and CI)")
    ap.add_argument("--tracked", action="store_true",
                    help="scan every tracked file (the baseline green run)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--make-fixture", metavar="DIR", type=Path,
                    help="write the synthetic must-be-refused listening page into DIR "
                         "and exit 0. Generated, never committed -- see make_fixture()")
    ap.add_argument("--quiet", action="store_true",
                    help="print refusals and the summary only")
    ap.add_argument("--delegated", choices=["refuse", "report"], default=None,
                    help="severity for check_publishable's BLOCK rules. Default "
                         "here: refuse on every collector (the recipe repository "
                         "holds no corpus prose; the lab reports them on its own "
                         "tree). 'report' prints them without refusing.")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    if args.self_test:
        return run_self_test()
    if args.make_fixture:
        path = make_fixture(args.make_fixture)
        print(f"  wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
        print(f"  synthetic, no corpus audio. Now scan it; it must exit 1:")
        print(f"    python {Path(__file__).name} --paths {args.make_fixture}")
        return 0
    if not (args.paths or args.staged or args.tracked):
        print("  give --paths, --staged, --tracked, or --self-test")
        return 2

    if args.delegated is not None:
        arriving = args.delegated == "refuse"
    else:
        # THE RECIPE REPOSITORY REFUSES ON EVERY COLLECTOR (see the docstring): it holds
        # no corpus prose, so the lab's report-only calibration for its own tree does
        # not transfer. A guard file that must name a hazard is declared in SELF_REFERENCE.
        arriving = True

    files = collect(args)
    # The carve-out keys on THIS repository's own files, so it is offered only on the
    # collectors that enumerate them. `--paths` is the arriving-content call site and
    # gets None: a file that rsyncs out of the bucket under the name
    # `scripts/check_clean_room.py` is not this file and is not excused.
    own = bool(args.tracked or args.staged)
    refused: list[tuple[Path, list[tuple[str, str, str]]]] = []
    noted = 0
    excused_files = 0
    undeclared_files = 0
    for path in files:
        if not path.exists():
            continue
        hits = scan_file(path, arriving=arriving,
                         carve_key=path.as_posix() if own else None)
        excused_files += any(s == "EXCUSED" for s, _, _ in hits)
        undeclared_files += any(
            s == "NOTE" or (s == "BLOCK" and c in DELEGATED_BLOCK_CODES)
            for s, c, _ in hits)
        blocking = [h for h in hits if h[0] in REFUSING]
        if blocking:
            refused.append((path, blocking))
        elif hits:
            noted += 1
            if not args.quiet:
                for sev, code, detail in hits:
                    print(f"  {sev:<7} {code:<24} {path}")

    mode = "refusing" if arriving else "reporting"
    # Staleness is only answerable against the WHOLE tree: a `--staged` run sees the
    # three files in the index and would call every other row missing.
    stale = audit_carve_out(files) if args.tracked else []
    print()
    if refused:
        print(f"  REFUSED. {len(refused)} of {len(files)} file(s) carry restricted "
              f"content:")
        for path, hits in refused:
            print(f"\n  {path}")
            for sev, code, detail in hits:
                print(f"    {sev} {code}: {detail}")
        print(f"\n  Nothing is committed. A commit that lands corpus-derived audio in "
              f"a public repository cannot be withdrawn from anyone who already "
              f"fetched it, which is why this gate fails closed.")
        _report_carve_out(own, excused_files, undeclared_files, stale)
        return 1

    print(f"  CLEAN. {len(files)} file(s) scanned, 0 refused, {noted} with "
          f"non-blocking notes.")
    print(f"  Delegated check_publishable rules: {mode}.")
    _report_carve_out(own, excused_files, undeclared_files, stale)
    return 0


def _report_carve_out(own: bool, excused: int, undeclared: int,
                      stale: list[str]) -> None:
    """The two numbers, and what a reader is supposed to do about each.

    THE SENTENCE THIS REPLACES SAID THE COUNT "is expected to fall, never rise. It
    rising means new source named the hazard." That could not be satisfied by correct
    work -- a guard implementing the audio_codes prohibition must write audio_codes,
    so writing one raised the number and read as a breach. One number cannot carry
    both facts, so it is now two, and neither of them has a direction it is forbidden
    to move.
    """
    if not own:
        return
    print(f"  Self-reference carve-out: {excused} file(s) excused by a declared row "
          f"in SELF_REFERENCE, {len(SELF_REFERENCE)} row(s) declared.")
    print(f"    An excused file still shows the hazard, and still refuses if its "
          f"evidence changes. RISING IS NORMAL AND EXPECTED: it rises when a new "
          f"guard, fixture or policy document names the hazard and someone declares "
          f"the row. What matters is that each row was read, not how many there are.")
    print(f"  Undeclared delegated hits: {undeclared} file(s).")
    if undeclared:
        print(f"    These are the ones nobody has ruled on. Each is EITHER a file "
              f"that legitimately names the hazard -- read its NOTE/BLOCK row above, "
              f"confirm it carries nothing, and add a row to SELF_REFERENCE with the "
              f"evidence and the reason -- OR restricted content that must come out. "
              f"THE SCANNER CANNOT TELL YOU WHICH; that judgement is the reviewer's "
              f"and the row is where it gets recorded.")
    if stale:
        print(f"  STALE carve-out rows ({len(stale)}). A row that no longer describes "
              f"the tree is a hole standing open; delete or re-review it:")
        for msg in stale:
            print(f"    {msg}")


def audit_carve_out(files: list[Path]) -> list[str]:
    """Declared rows that no longer describe the repository.

    A carve-out is only as honest as its staleness. A row naming a file that has been
    deleted, or a code the file no longer trips, is dead text that a reader will still
    count as reviewed -- and worse, it is a pre-approved excuse waiting for a future
    file at that path. Reported on every `--tracked` run so it cannot rot quietly.
    Rows whose evidence CHANGED are not listed here: those already refused above,
    loudly, which is the point.
    """
    present = {p.as_posix() for p in files}
    out: list[str] = []
    for path, code, _declared, _why in SELF_REFERENCE:
        if path not in present:
            out.append(f"{path} / {code}: not in the scanned set (deleted, renamed, "
                       f"or outside this run's scope)")
            continue
        try:
            raw = Path(path).read_bytes().decode("utf-8", errors="replace")
        except OSError as exc:
            out.append(f"{path} / {code}: unreadable ({exc})")
            continue
        if not delegated_evidence(code, raw):
            out.append(f"{path} / {code}: the file no longer trips this rule -- the "
                       f"row excuses nothing and should be deleted")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
