#!/usr/bin/env python3
"""One way to put audio in a page, and one audio format. FLAC, never MP3.

THE RULE. Everything this project GENERATES is written as FLAC, and every listening
page inlines FLAC. Not WAV, because FLAC holds the identical samples in about half
the bytes and `soundfile` writes it natively -- so ffmpeg stops being something
that has to be on PATH before a render can be stored. And never MP3, which is the
part that is about evidence rather than convenience:

**A lossy codec is not admissible in a test that asks about segments.** These pages
ask a listener to mark INDIVIDUAL CHARACTERS as mispronounced. The evidence lives
in high-frequency detail -- fricatives, checked-tone codas, the quiet trailing
sentence-final particles this project has repeatedly found errors on -- and 64 kbps
MP3 is entitled to alter exactly that. A listening test built on it measures the
codec as much as the model. EXP-000's first session was collected that way before
anyone noticed.

WHY THIS FILE EXISTS AT ALL. Fourteen page builders each carried their own
`mp3_data_uri` helper, each shelling to ffmpeg with its own bitrate. Fourteen
copies of a decision is fourteen places to fix it and fourteen chances to miss one
-- which is how the rule would rot back to MP3. `common/` is where a thing more
than one family uses belongs (scripts/README.md Rule 1), so it lives here and
`check_audio_format.py` fails the build if a copy reappears.

    from audio_uri import flac_data_uri
    src = flac_data_uri(Path("chunk_001.flac"))     # -> data:audio/flac;base64,...
"""
from __future__ import annotations

import base64
import io
from pathlib import Path


def flac_data_uri(src: str | Path, verify: bool = True) -> str:
    """Inline any readable audio file as a lossless FLAC data URI.

    Re-encodes only when the source is not already FLAC, so a FLAC render is
    embedded byte-for-byte rather than decoded and re-compressed.

    `verify` round-trips the encoded bytes and checks the shape and rate survived,
    the way `inline_page_audio.py` already did. It costs a decode and it is the
    difference between shipping a page and shipping a page that plays.
    """
    import soundfile as sf

    src = Path(src)
    if src.suffix.lower() == ".flac":
        raw = src.read_bytes()
        if verify:
            back, sr = sf.read(io.BytesIO(raw))
            if back.size == 0:
                raise SystemExit(f"{src} decoded to nothing")
    else:
        data, sr = sf.read(str(src))
        buf = io.BytesIO()
        # PCM_16 explicitly: FLAC is integer-only and soundfile would otherwise
        # choose a subtype for us.
        sf.write(buf, data, sr, format="FLAC", subtype="PCM_16")
        raw = buf.getvalue()
        if verify:
            back, sr2 = sf.read(io.BytesIO(raw))
            if sr2 != sr or back.shape != data.shape:
                raise SystemExit(f"FLAC round trip changed {src}: "
                                 f"{data.shape}@{sr} -> {back.shape}@{sr2}")
    return "data:audio/flac;base64," + base64.b64encode(raw).decode("ascii")


def write_flac(path: str | Path, samples, sample_rate: int) -> Path:
    """Write generated audio. Forces a .flac suffix rather than trusting the caller."""
    import soundfile as sf

    path = Path(path).with_suffix(".flac")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path
