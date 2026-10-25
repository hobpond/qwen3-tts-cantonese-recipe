#!/usr/bin/env bash
# Build the GGUF runtime: upstream qwen3-tts.cpp at a pinned commit, plus our patch.
#
# Everything the recipe needs to render through GGUF, from a machine that has none of it:
#
#     bash scripts/build_gguf_runtime.sh
#
# WHY A SCRIPT AND NOT A PARAGRAPH IN A README. The binary this produces has to be the
# same binary the measurements were taken on, or the numbers in gguf_backend.py are
# about someone else's build. The pin, the patch and the CMake flags are the things
# that make it the same, so they live in one runnable place rather than three
# rememberable ones.
#
# ON THE NATIVE DISK, NEVER /mnt. A CUDA build of this tree writes thousands of small
# object files, and on WSL that is 138x slower over DrvFs than on ext4 -- measured in
# the lab on this project's own host. Both the source and the build
# directory default under $HOME for that reason. Cloning to /mnt/d "so it is with the
# other checkouts" turns a 4 minute build into most of an hour.
#
#   --clean       remove the build directory first, then configure from scratch
#   --reclone     remove the source checkout too. Use after changing the pin.
#   --jobs N      parallel compile jobs (default: nproc)
#
# NOTHING BELOW IS HARDCODED TO ONE MACHINE. The toolchain is discovered, and every
# discovery can be overridden, because a fresh checkout is the case that matters:
#
#   CMAKE                cmake to use                    (default: cmake on PATH)
#   CUDACXX / CUDA_HOME  where nvcc is; also searched on PATH, /usr/local/cuda*,
#     / CUDA_PATH        /opt/cuda*, /usr/lib/nvidia-cuda-toolkit, /usr/bin
#   QWEN_TTS_CUDA_ARCH   compute capability            (default: read from nvidia-smi)
#   QWEN_TTS_CUDA_OFF=1  build a CPU-only runtime -- correct, and much slower
#   QWEN_TTS_CPP         source checkout            (default: ~/.qwen-tts-cpp/src)
#   QWEN_TTS_CPP_BUILD   build directory            (default: ~/.qwen-tts-cpp/build)
#   QWEN_TTS_CPP_URL     upstream remote
#   QWEN_TTS_CPP_PIN     upstream commit to build
set -euo pipefail

UPSTREAM_URL="${QWEN_TTS_CPP_URL:-https://github.com/Danmoreng/qwen3-tts.cpp.git}"
# PINNED, not "main". An unpinned dependency means the render path can change under a
# result without anything in this repository changing. Move it deliberately: bump the
# SHA, re-run with --reclone, and re-render something whose output you already know.
UPSTREAM_PIN="${QWEN_TTS_CPP_PIN:-16bb5afcd06311031c72a8488f8d59660dc2fb46}"

SRC="${QWEN_TTS_CPP:-$HOME/.qwen-tts-cpp/src}"
BUILD="${QWEN_TTS_CPP_BUILD:-$HOME/.qwen-tts-cpp/build}"
JOBS="$(nproc 2>/dev/null || echo 4)"
CLEAN=0
RECLONE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --clean)   CLEAN=1 ;;
    --reclone) RECLONE=1; CLEAN=1 ;;
    --jobs)    shift; JOBS="$1" ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

say(){ printf '[gguf-build] %s\n' "$*"; }
die(){ printf '[gguf-build] FATAL: %s\n' "$*" >&2; exit 1; }

REPO=""
for d in "$(cd "$(dirname "$0")" && pwd)" "$(cd "$(dirname "$0")/.." && pwd)" \
         "$(cd "$(dirname "$0")/../.." && pwd)"; do
  [ -f "$d/recipe.toml" ] && REPO="$d" && break
done
[ -n "$REPO" ] || die "cannot locate the repo root by marker search for recipe.toml"
PATCHES="$REPO/external/qwen3-tts-cpp/patches"

CMAKE="${CMAKE:-cmake}"
command -v "$CMAKE" >/dev/null \
  || die "cmake is not on PATH. Install it (apt: cmake, or pip install cmake), or
     set CMAKE=/path/to/cmake."
command -v git >/dev/null || die "git is not on PATH."

# ---- finding the CUDA compiler -------------------------------------------
# FINDING nvcc IS NOT THE SAME AS CMAKE FINDING IT. A non-login shell frequently has
# no CUDA bin directory on PATH, and CMake then fails deep inside ggml-cuda's
# enable_language(CUDA) with "Id flags: -v / No such file or directory" -- an error
# that names neither CUDA nor PATH and sends you looking at the wrong thing.
#
# There is no single right location to hardcode: distributions put it at
# /usr/local/cuda-<ver>, /opt/cuda, /usr/lib/nvidia-cuda-toolkit or plain /usr/bin,
# conda puts it in the environment, and CUDACXX / CUDA_HOME are the variables people
# actually set. So look in all of them, newest versioned toolkit first, and say which
# one was chosen -- a build against an unexpected toolkit is worth seeing.
find_nvcc() {
  local c
  for c in "${CUDACXX:-}" "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}" \
           "${CUDA_PATH:+$CUDA_PATH/bin/nvcc}" "$(command -v nvcc 2>/dev/null || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { printf '%s' "$c"; return 0; }
  done
  for c in $(ls -d /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc \
                   /usr/lib/nvidia-cuda-toolkit/bin/nvcc /usr/bin/nvcc \
                   2>/dev/null | sort -Vr); do
    [ -x "$c" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
}

NVCC=""
if [ -z "${QWEN_TTS_CUDA_OFF:-}" ]; then
  NVCC="$(find_nvcc || true)"
  [ -n "$NVCC" ] || die "no CUDA compiler found. Looked at \$CUDACXX, \$CUDA_HOME,
     \$CUDA_PATH, PATH, /usr/local/cuda*, /opt/cuda*, /usr/lib/nvidia-cuda-toolkit
     and /usr/bin.
     Install the CUDA toolkit and re-run, or point CUDACXX at nvcc, or set
     QWEN_TTS_CUDA_OFF=1 to build a CPU-only runtime (correct, and much slower)."
fi

# ---- which GPU to compile for --------------------------------------------
# Also not hardcodable: 86 is one card. Ask the driver. `native` would be simpler but
# needs CMake >= 3.24 and this has to work on the 3.22 that Ubuntu 22.04 ships.
detect_cuda_arch() {
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d '. ')"
  [ -n "$cap" ] && printf '%s' "$cap"
}
CUDA_ARCH="${QWEN_TTS_CUDA_ARCH:-$(detect_cuda_arch || true)}"
if [ -z "$CUDA_ARCH" ] && [ -z "${QWEN_TTS_CUDA_OFF:-}" ]; then
  # Building for the wrong architecture succeeds and then runs on the CPU, which
  # looks like "GGUF is not much faster" rather than like a misconfiguration.
  CUDA_ARCH=86
  say "WARNING: could not read compute capability from nvidia-smi; assuming $CUDA_ARCH.
             If renders are slow, set QWEN_TTS_CUDA_ARCH to your card's value."
fi

case "$SRC" in
  /mnt/*) say "WARNING: source is under /mnt ($SRC). On WSL this build will be
             dramatically slower -- see the note at the top of this script." ;;
esac

# ---- source ---------------------------------------------------------------
[ $RECLONE -eq 1 ] && [ -d "$SRC" ] && { say "removing $SRC"; rm -rf "$SRC"; }
if [ ! -d "$SRC/.git" ]; then
  say "cloning $UPSTREAM_URL -> $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone --recursive "$UPSTREAM_URL" "$SRC"
fi

say "checking out the pin $UPSTREAM_PIN"
git -C "$SRC" fetch --tags origin
# Reset rather than checkout: a previous run left the patch applied in the worktree,
# and re-applying it onto itself would fail. The patch is re-applied below every time,
# so the tree must start clean at the pin -- which also means a local edit here is
# DISCARDED, deliberately. Edit the patch in external/qwen3-tts-cpp/patches/, not this checkout.
git -C "$SRC" checkout -q --detach "$UPSTREAM_PIN"
git -C "$SRC" reset -q --hard "$UPSTREAM_PIN"
git -C "$SRC" submodule update --init --recursive
git -C "$SRC" clean -qfd -e build

# ---- patches --------------------------------------------------------------
shopt -s nullglob
PATCH_FILES=("$PATCHES"/*.patch)
shopt -u nullglob
if [ ${#PATCH_FILES[@]} -eq 0 ]; then
  say "no patches in $PATCHES -- building stock upstream"
else
  for p in "${PATCH_FILES[@]}"; do
    say "applying $(basename "$p")"
    # --check first so a patch that has drifted from the pin fails HERE, naming
    # itself, rather than producing a binary that quietly lacks the feature.
    git -C "$SRC" apply --check "$p" \
      || die "$(basename "$p") does not apply to $UPSTREAM_PIN.
     The pin moved, or upstream changed the lines it touches. Rebase the patch
     against the pin; do not skip it -- gguf_backend.py falls back to one process
     per clip without it and every render gets roughly twice as slow."
    git -C "$SRC" apply "$p"
  done
fi

# ---- build ----------------------------------------------------------------
[ $CLEAN -eq 1 ] && [ -d "$BUILD" ] && { say "removing $BUILD"; rm -rf "$BUILD"; }
mkdir -p "$BUILD"

# These flags are the ones the recorded measurements were taken with.
# CMAKE_CUDA_ARCHITECTURES=86 is this machine's card (RTX 30-series); override with
# QWEN_TTS_CUDA_ARCH for another. Getting it wrong builds fine and runs on the CPU.
if [ -n "${QWEN_TTS_CUDA_OFF:-}" ]; then
  CUDA_ARGS=(-DQWEN3_TTS_CUDA=OFF -DGGML_CUDA=OFF)
  say "CUDA disabled by QWEN_TTS_CUDA_OFF -- this will render on the CPU"
  say "configuring (CPU only, $JOBS jobs)"
else
  CUDA_ARGS=(-DQWEN3_TTS_CUDA=ON -DGGML_CUDA=ON
             -DCMAKE_CUDA_COMPILER="$NVCC"
             -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  say "configuring (nvcc $NVCC, arch $CUDA_ARCH, $JOBS jobs)"
fi

"$CMAKE" -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  "${CUDA_ARGS[@]}" \
  -DQWEN3_TTS_EMBED_GGML=ON \
  >"$BUILD/configure.log" 2>&1 \
  || { tail -30 "$BUILD/configure.log"; die "cmake configure failed, see $BUILD/configure.log"; }

say "building qwen3-tts-cli"
"$CMAKE" --build "$BUILD" --target qwen3-tts-cli -j "$JOBS" \
  >"$BUILD/build.log" 2>&1 \
  || { tail -40 "$BUILD/build.log"; die "build failed, see $BUILD/build.log"; }

CLI="$BUILD/qwen3-tts-cli"
[ -x "$CLI" ] || die "build reported success but $CLI is not there"

# ---- verify ---------------------------------------------------------------
# A build that succeeds and silently lacks --serve is the failure worth catching:
# everything still works, at half the speed, with nothing to see.
if "$CLI" --help 2>&1 | grep -q -- "--serve"; then
  say "verified: --serve is present (resident mode available)"
else
  die "the built binary has no --serve. The patch did not take effect.
     Re-run with --reclone; if it persists the patch needs rebasing onto the pin."
fi

cat <<EOF

[gguf-build] done.

  binary : $CLI
  source : $SRC @ $UPSTREAM_PIN + $(basename "$PATCHES")/*.patch

Point the recipe at it (add to your shell profile, or set [paths] in recipe.toml):

  export QWEN_TTS_CLI=$CLI
  export QWEN_TTS_CPP=$SRC

Then confirm the recipe agrees:

  python src/gguf_backend.py --check
  python recipe.py check
EOF
