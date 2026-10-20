#!/usr/bin/env bash
set -euo pipefail
# Build the native (decomp) Melee simulator into sim/ — headless, no emulator.
#
#   sim/melee-unlocked/build-linux/port/linux/melee_linux
#
# Source: Hero88go/melee-unlocked (GPL-3.0) pinned by
# tools/patches/melee-unlocked-commit.txt, its doldecomp/melee submodule, and
# our Linux --replay host (tools/patches/melee-unlocked-linux-replay.patch,
# GPL-3.0). Everything is fetched from the public upstreams at build time and
# built against this machine's own melee.iso. NOTHING under sim/ may be
# committed or uploaded: it holds the decompiled game code and compiled
# binaries of it (see docs/research-notes-2026-10-20b.md). sim/ is gitignored.
#
# Toolchain: GCC 14 (the game needs GCC's scalar_storage_order; Clang and
# GCC 13 won't do) + libusb headers. Installed with apt when running as root;
# otherwise extracted from the Ubuntu .debs into sim/toolchain (no sudo).
#
# Usage: bash tools/build_native_sim.sh        # idempotent; re-run to rebuild

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIM_DIR="$REPO_DIR/sim"
SRC="$SIM_DIR/melee-unlocked"
TC="$SIM_DIR/toolchain"
COMMIT=$(cat "$REPO_DIR/tools/patches/melee-unlocked-commit.txt")
PATCH="$REPO_DIR/tools/patches/melee-unlocked-linux-replay.patch"
ISO="$REPO_DIR/melee.iso"
EXE="$SRC/build-linux/port/linux/melee_linux"

# ── toolchain ──
if ! command -v gcc-14 >/dev/null || ! pkg-config --exists libusb-1.0 2>/dev/null; then
    if [[ $EUID -eq 0 ]]; then
        echo "  Installing gcc-14, g++-14, libusb-1.0-0-dev ..."
        apt-get update -qq 2>&1 | tail -1
        apt-get install -y -qq gcc-14 g++-14 libusb-1.0-0-dev ninja-build cmake 2>&1 | tail -1
    elif [[ ! -x "$TC/usr/bin/gcc-14" ]]; then
        echo "  No root: extracting gcc-14 + libusb headers into $TC ..."
        mkdir -p "$TC/debs"
        (cd "$TC/debs" && apt-get download gcc-14 g++-14 cpp-14 gcc-14-x86-64-linux-gnu \
            g++-14-x86-64-linux-gnu cpp-14-x86-64-linux-gnu libgcc-14-dev libstdc++-14-dev \
            gcc-14-base libusb-1.0-0-dev >/dev/null)
        for d in "$TC"/debs/*.deb; do dpkg -x "$d" "$TC"; done
        ln -sf /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 "$TC/usr/lib/x86_64-linux-gnu/libusb-1.0.so"
        sed -i "s|^prefix=.*|prefix=$TC/usr|" "$TC/usr/lib/x86_64-linux-gnu/pkgconfig/libusb-1.0.pc"
        rm -rf "$TC/debs"
    fi
    export PATH="$TC/usr/bin:$PATH"
    export PKG_CONFIG_PATH="$TC/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi
for t in gcc-14 g++-14 cmake ninja; do
    command -v "$t" >/dev/null || { echo "  ERROR: missing $t"; exit 1; }
done

# ── sources (pinned) ──
if [[ ! -d "$SRC/.git" ]]; then
    echo "  Fetching melee-unlocked @ ${COMMIT:0:7} ..."
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" remote add origin https://github.com/Hero88go/melee-unlocked.git
fi
if [[ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]]; then
    git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
    git -C "$SRC" checkout -q -f "$COMMIT"
fi
git -C "$SRC" submodule update -q --init --depth 1 sourceport/extern/melee
python3 "$SRC/tools/prepare_native_sources.py"   # their native patch to the decomp
if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "  Linux replay host patch already applied."
else
    git -C "$SRC" apply "$PATCH"
    echo "  Applied Linux replay host patch."
fi

# ── build (headless: no SDL/GL needed) ──
echo "  Building (headless) ..."
(cd "$SRC" && python3 tools/build_linux.py --headless-build --jobs "$(nproc)" > "$SIM_DIR/build.log" 2>&1) \
    || { echo "  ERROR: build failed, see $SIM_DIR/build.log"; tail -20 "$SIM_DIR/build.log"; exit 1; }

# ── smoke test: boot to the menus headless ──
# The Linux disc loader needs an uncompressed NTSC 1.02 rev 2 image (GALE01).
ISO_SHA1=d4e70c064cc714ba8400a849cf299dbd1aa326fc
if [[ -f "$ISO" ]] && [[ "$(sha1sum "$ISO" | cut -d' ' -f1)" != "$ISO_SHA1" ]]; then
    echo "  ERROR: $ISO is not the NTSC 1.02 rev 2 image the sim needs (SHA-1 $ISO_SHA1)."
    exit 1
fi
if [[ -f "$ISO" ]]; then
    mkdir -p "$SIM_DIR/card"
    (cd "$SRC" && "$EXE" --iso "$ISO" --headless --fast --frames 600 --card-dir "$SIM_DIR/card" \
        > "$SIM_DIR/smoke.log" 2>&1) \
        && echo "  Smoke test OK (600 frames headless)." \
        || { echo "  ERROR: smoke test failed, see $SIM_DIR/smoke.log"; exit 1; }
else
    echo "  (No $ISO yet: skipped the smoke test.)"
fi
echo "  Native sim ready: $EXE"
echo "  Fidelity check:   python3 tools/sim_replay_check.py --batch <slp_dir> --exe $EXE --iso melee.iso"
