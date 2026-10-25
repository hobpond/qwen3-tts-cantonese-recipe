#!/usr/bin/env python3
"""Turn a folder of recordings plus transcripts into the JSONL this pipeline eats.

Handles the parts that silently ruin a dataset if you get them wrong: resampling to
the 24 kHz `dataset.py` hard-asserts, downmixing to mono, reporting what the 5-30 s
filter will discard *before* you spend GPU time on tokenisation, and failing loudly
on a transcript that has no audio or audio that has no transcript.

Input is a directory of audio files and one transcript file, either:

  * a two-column TSV of `filename<TAB>transcript`, or
  * one `.txt` beside each audio file with the same stem.

    python src/ingest_recordings.py --audio-dir SELECTED/audio --transcripts SELECTED/transcripts.tsv \\
        --speaker-id cv_yue_v22 --out-dir DATA_DIR --min-duration 4.0

In the recipe, `python recipe.py select` runs this after the selector, and the store's `encode` step adds the codes
(not the lab's prepare_train_evaluate_data.py). --out-dir is required and refused inside this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TARGET_SR = 24000
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus")


def load_transcripts(path: Path, audio_dir: Path) -> dict:
    """Accept a TSV, or sidecar .txt files named like the audio."""
    out = {}
    if path and path.is_file():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                print(f"  line {n}: no tab, skipping: {line[:50]!r}", file=sys.stderr)
                continue
            name, text = line.split("\t", 1)
            out[Path(name.strip()).stem] = text.strip()
        return out

    for p in sorted(audio_dir.iterdir()):
        if p.suffix.lower() in AUDIO_EXT:
            side = p.with_suffix(".txt")
            if side.exists():
                out[p.stem] = side.read_text(encoding="utf-8").strip()
    return out


def trim_edges(y, sr: int, thr_frac: float, margin_s: float):
    """Remove leading and trailing silence, never anything in the middle.

    A pause inside an utterance is prosody and is left alone; only the record-button
    overhead at the ends is removed. The threshold is relative to the clip's own
    median frame energy, so it adapts to each contributor's recording level instead
    of assuming a fixed noise floor across hundreds of microphones.

    `margin_s` is retained beyond the detected speech at both ends. It exists for
    sentence-final particles: they are quiet, they trail, and cutting to the exact
    detected boundary would clip the one feature this project is most sensitive to.
    """
    import numpy as _np
    hop = max(1, sr // 100)
    usable = len(y) // hop * hop
    if usable < hop * 3:
        return y
    frames = y[:usable].reshape(-1, hop)
    rms = _np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    thr = _np.median(rms) * thr_frac
    voiced = _np.where(rms >= thr)[0]
    if len(voiced) < 2:
        return y
    m = int(margin_s * sr / hop)
    lo = max(0, voiced[0] - m) * hop
    hi = min(len(rms), voiced[-1] + 1 + m) * hop
    return y[lo:hi]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path,
                    help="TSV of filename<TAB>text; omit to use sidecar .txt files")
    ap.add_argument("--speaker-id", default="my_speaker")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--path-style", choices=["absolute", "relative"],
                    default="absolute",
                    help="how `audio` and `ref_audio` are written. absolute (the "
                         "default) resolves from any working directory; relative "
                         "only resolves from the manifest's own directory, which "
                         "is upstream's convention and breaks silently elsewhere")
    ap.add_argument("--subdir", default="audio_custom",
                    help="Where converted WAVs land, under --out-dir")
    ap.add_argument("--out-jsonl", default="custom_raw.jsonl")
    ap.add_argument("--min-duration", type=float, default=5.0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    # Crowd-recorded audio carries far more edge silence than a segmented corpus:
    # measured with one method over both, Common Voice clips are 24.2% leading and
    # trailing silence against the licensed corpus's 3.4%. Untrained-on, that teaches
    # the model that an utterance begins and ends with roughly a second of nothing,
    # which is a candidate cause of the padding and rambling seen on short prompts.
    #
    # The margin is not a detail. Cantonese sentence-final particles are quiet and
    # trailing, and they are precisely what a native listener has been catching
    # errors on -- so an aggressive trim would delete the thing we care most about
    # while appearing to clean the data. Default keeps 200 ms beyond detected
    # speech and only ever removes from the ends, validated on particle-final clips.
    #
    # Note the ordering: trimming runs BEFORE the duration filter, so post-trim
    # duration is speech duration. A 5 s floor on trimmed audio is therefore much
    # stricter than the same floor on raw audio -- lower it by roughly the trim rate.
    ap.add_argument("--trim-silence", action="store_true",
                    help="Remove leading/trailing silence, keeping --trim-margin.")
    ap.add_argument("--trim-margin", type=float, default=0.20,
                    help="Seconds of silence to retain at each end (default 0.20).")
    ap.add_argument("--trim-threshold", type=float, default=0.15,
                    help="Speech threshold as a fraction of the clip's median frame "
                         "RMS. Lower is more conservative (keeps more).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would happen without writing anything")
    args = ap.parse_args()

    # REFUSE TO WRITE A DATASET INTO THIS REPOSITORY (in the lab the DEFAULT --out-dir
    # pointed inside it; here --out-dir is required). This script's output is the worst possible thing to put
    # in a clean room: audio files copied verbatim from a corpus, plus a manifest
    # carrying `audio_codes`, which decode back to near-original audio.
    # check_publishable.py's own header records what that costs -- a manifest with
    # audio_codes reached a public fork on 2026-08-24, after 2,004 corpus .wav files
    # had already been committed, which ignoring does not untrack.
    #
    # extract_yue_emo.py and sample_yue_emo_for_audit.py both refuse a repo path.
    # This one did not, and it is the step that produces the training data.
    repo = Path(__file__).resolve().parents[1]          # src/ -> the repository
    if str(args.out_dir.resolve()).startswith(str(repo)):
        print(f"refusing: --out-dir resolves inside the repository ({repo}).\n"
              "This writes corpus audio and a manifest with audio_codes, neither of "
              "which may enter a clean-room repo. Pass an --out-dir outside it.",
              file=sys.stderr)
        return 2

    import librosa
    import numpy as np
    import soundfile as sf

    if not args.audio_dir.is_dir():
        print(f"error: no such directory: {args.audio_dir}", file=sys.stderr)
        return 1

    clips = [p for p in sorted(args.audio_dir.iterdir())
             if p.suffix.lower() in AUDIO_EXT]
    if not clips:
        print(f"error: no audio in {args.audio_dir} "
              f"(looked for {', '.join(AUDIO_EXT)})", file=sys.stderr)
        return 1

    texts = load_transcripts(args.transcripts, args.audio_dir)
    if not texts:
        print("error: no transcripts found. Give --transcripts a TSV, or put a .txt "
              "beside each audio file with the same name.", file=sys.stderr)
        return 1

    # Both of these are set lookups on purpose. The second was written as
    # `any(p.stem == k for p in clips)` inside a comprehension over `texts`, which
    # is O(clips x texts): fine at a few thousand files, and 3.1 billion string
    # comparisons at 55k. It stalled a 55,668-clip ingest for over ten minutes
    # before writing a single output file, with no indication it was doing anything.
    stems = {p.stem for p in clips}
    missing_text = [p.name for p in clips if p.stem not in texts]
    missing_audio = [k for k in texts if k not in stems]

    dest = args.out_dir / args.subdir
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    rows, too_short, too_long, total_s = [], [], [], 0.0
    for i, p in enumerate(clips):
        if p.stem not in texts:
            continue
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if y.ndim > 1:
            y = np.mean(y, axis=-1)
        if sr != TARGET_SR:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=TARGET_SR)
        if args.trim_silence:
            y = trim_edges(y, TARGET_SR, args.trim_threshold, args.trim_margin)
        dur = len(y) / TARGET_SR

        if dur < args.min_duration:
            too_short.append((p.name, dur))
            continue
        if dur > args.max_duration:
            too_long.append((p.name, dur))
            continue

        # ABSOLUTE BY DEFAULT, and this was a real failure rather than a
        # preference. A relative `./<subdir>/<i>.wav` only resolves when the
        # process happens to be running in the manifest's own directory. Upstream
        # gets away with it because its manifests live in
        # Dataset-Cantonese-Training/ and its trainer is run from there; a dataset
        # written to --out-dir anywhere else is silently broken everywhere but one
        # CWD. It cost a run: prepare_data.py reported "missing: 460, no samples
        # left" from the repo directory, and check_bootstrap's train_audio control
        # then refused the lane because the audio the manifest named did not exist.
        # Note the shape of it -- the trainer reads audio_codes and would not have
        # noticed, so this surfaces at RENDER time, hours later.
        # --path-style relative restores the old behaviour for a manifest that is
        # deliberately going to sit beside its audio.
        stored = (str((dest / f"{i}.flac").resolve()) if args.path_style == "absolute"
                  else f"./{args.subdir}/{i}.flac")
        if not args.dry_run:
            # FLAC, PCM_16: the trainer reads it through librosa.load and
            # audio_duration through soundfile, both of which handle it,
            # and a corpus of 24 kHz WAV is twice the size for nothing.
            sf.write(str(dest / f"{i}.flac"), y.astype(np.float32),
                     TARGET_SR, subtype="PCM_16")
        # ref_audio points at the clip itself: it is the voice prompt, and using the
        # clip's own recording is the sane default for single-speaker material.
        rows.append({"audio": stored, "text": texts[p.stem],
                     "ref_audio": stored, "speaker_id": args.speaker_id})
        total_s += dur

    print(f"audio files found   : {len(clips)}")
    print(f"transcripts found   : {len(texts)}")
    if missing_text:
        print(f"  WITHOUT transcript: {len(missing_text)}  e.g. {missing_text[:3]}")
    if missing_audio:
        print(f"  WITHOUT audio     : {len(missing_audio)}  e.g. {missing_audio[:3]}")
    print()
    print(f"usable ({args.min_duration:g}-{args.max_duration:g}s) : "
          f"{len(rows)} clips, {total_s / 3600:.2f} h")
    if too_short:
        lost = sum(d for _, d in too_short) / 3600
        print(f"  dropped, too short: {len(too_short)}  ({lost:.2f} h lost)  "
              f"e.g. {too_short[0][0]} at {too_short[0][1]:.1f}s")
    if too_long:
        print(f"  dropped, too long : {len(too_long)}  "
              f"e.g. {too_long[0][0]} at {too_long[0][1]:.1f}s")

    if not rows:
        print("\nerror: nothing usable. Most likely every clip is under "
              f"{args.min_duration:g}s -- record longer utterances, or lower "
              "--min-duration knowing the model gets less prosodic context per sample.",
              file=sys.stderr)
        return 1

    if total_s / 3600 < 1.0:
        print(f"\nNOTE: {total_s / 3600:.2f} h is thin for a voice. One to five hours "
              "of a single speaker is the usual working range.")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    out = args.out_dir / args.out_jsonl
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}  ({len(rows)} records)")
    print(f"wrote {dest}/  (24 kHz mono FLAC)")
    print("\nNext: python recipe.py cleanup (the store ingests this manifest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
