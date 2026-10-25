# The GGUF runtime: upstream, plus one patch of our own

`src/gguf_backend.py` renders through [qwen3-tts.cpp][up], which is MIT
licensed and **not vendored here**. This directory holds the two things needed to
reproduce the exact binary the recipe expects: the upstream commit it is pinned to, and
the one patch applied on top.

    bash scripts/build_gguf_runtime.sh

That clones upstream at the pin, applies `patches/`, builds with CUDA, and verifies
the result. Everything lands under `~/.qwen-tts-cpp/`, on the native disk rather than
`/mnt` (on WSL, DrvFs is ~8x slower sequentially and ~100x on small files).

[up]: https://github.com/Danmoreng/qwen3-tts.cpp

## Why there is a patch at all

Rendering one clip per process spends most of its time not rendering. Measured on one
short Cantonese line with a q8_0 talker: **1205 ms inside the binary, 6.13 s of wall.**
The rest is process start, 2.3 GB of model file, and CUDA context init. A ten-clip
passage therefore spent longer loading than speaking, and a long arm is far worse,
because the load is paid per clip while the speaking is not.

Upstream has no way to avoid it. `--bench-server` loads once but re-synthesises a
single string; there is no stdin loop, no server binary, and the CLI takes exactly one
`-t` per process. So `0001-serve-resident-mode.patch` adds `--serve`: load once, then
one request per line on stdin.

    stdin   <out_path> TAB <seed> TAB <text>
    stdout  OK TAB <out_path> TAB <ms>      or   ERR TAB <message>

Measured on EXP-000's passage: **50 s one-process-per-clip, 28 s resident.**

## Why it is a patch and not a fork you have to trust

This repository points only at public things: upstream's URL, a commit SHA, and
a diff in this directory that you can read in full. There is no private repository in
the dependency chain, and nothing here needs credentials.

If upstream takes the change, this directory shrinks to a pin and the patch goes away.
Until then the patch is small, self-contained and applies cleanly to the pinned commit
-- `build_gguf_runtime.sh` fails loudly if that ever stops being true rather than
building something subtly different.

## The seed is the reason to care about any of this

The patch sends the seed **per request** rather than fixing it when the process
starts. That is not a detail. With one seed per session, every clip would depend on
how many clips preceded it: the same text would render differently as the second
request than as the first, and a single clip could never be re-rendered on its own to
check it. With a per-request seed, resident output equals one-shot output clip for
clip -- verified by re-rendering EXP-000's ten chunks and getting all ten
byte-identical (md5) to the committed one-shot run.

`gguf_backend.DEFAULT_SEED` is the value the recipe renders with unless a caller passes
its own. Record it with any result, because it is what makes the result reproducible.

## What was considered instead, and why it was not enough

Upstream ships a full public C API (`src/qwen3_tts_c.h`: `qwen3_tts_init`,
`qwen3_tts_load_models_with_name`, `qwen3_tts_synthesize`, `qwen3_tts_free`), and its
symbols are exported with C linkage from `libqwen3_tts.a`. Calling it from `ctypes`
would keep the model loaded with no subprocess, no pipe framing and no scratch WAV at
all -- strictly better than `--serve`, and needing no patch to the CLI.

It was rejected for one reason: **`qwen3_tts_params_t` has no seed field.** The C++
`tts_params` does (`src/qwen3_tts.h`), but the C wrapper does not forward it, so every
synthesis through the C API runs at `seed = -1`, random per call. Trading reproducible
renders for a faster path is the wrong trade when the results are listening tests.

Adding `seed` to that struct is a two-line change and a far better upstream
contribution than `--serve` is. If it lands, the right move is to drop this patch,
bind the C API through `ctypes`, and delete the subprocess entirely.
