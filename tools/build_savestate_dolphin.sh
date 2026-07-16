#!/bin/bash -e
# Build the savestate-enabled Slippi Ishiiruka (ExiAI fork) from source and
# install into the MIMIC repo. Produces TWO variants:
#
#   emulator_ss/Binaries/dolphin-emu   — netplay/ExiAI build (FFW-capable,
#                                        EXI bot inputs) + SAVESTATE/LOADSTATE
#                                        pipe verbs. Drop-in replacement for
#                                        emulator_ffw for savestate work.
#   emulator_pb/Binaries/dolphin-emu   — playback build (replay .slp playback
#                                        via -i comm.json, --cout frame
#                                        protocol) + same pipe verbs.
#                                        NOTE (2026-07-16): replay auto-start
#                                        under this headless build is NOT yet
#                                        working (game boots to menu; gecko
#                                        handshake never fires) — see
#                                        docs/research-notes-2026-07-14.md.
#
# The patch (tools/patches/slippi-ishiiruka-savestate.patch) adds:
#   - Pipes.cpp: "SAVESTATE <path>" / "LOADSTATE <path>" pipe commands.
#     State ops run as host jobs (State::SaveAs/LoadAs use Core::PauseAndLock
#     and must not run on the CPU thread that polls the pipe).
#   - MainNoGUI.cpp: -i/--slippi-input (playback comm file) and --cout
#     (launcher [CURRENT_FRAME] stdout protocol) flags, mirroring the WX GUI.
#
# Python usage after install (see docs/research-notes-2026-07-14.md):
#   ctrl._write(f"SAVESTATE {path}\n"); ctrl.flush()   # host-job latency:
#   ctrl._write(f"LOADSTATE {path}\n"); ctrl.flush()   # 1-150 game frames
#   melee.Console(..., skip_rollback_frames=False)     # REQUIRED to observe
#                                                      # the rewind
#
# Deps (Ubuntu): build-essential, cargo, libudev-dev, libasound2-dev,
#                pip install --user 'cmake<4' ninja
# The bundled libusb must be built WITHOUT udev (its cmake glue mislinks
# netlink symbols when libudev is found) — handled below.

# Our fork carries the patch already applied (branch mimic-savestates,
# forked from vladfi1/slippi-Ishiiruka @ exi-ai-rebase). Unmaintained —
# pinned by the commit file; tools/patches/*.patch is the same diff kept
# as documentation/backup should the fork ever disappear.
FORK_URL=https://github.com/erickfm/slippi-Ishiiruka.git
FORK_BRANCH=mimic-savestates
FORK_COMMIT=$(cat "$(dirname "$0")/patches/slippi-ishiiruka-commit.txt")

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${SRC:-$HOME/projects/slippi-Ishiiruka}"
PATCH="$REPO_DIR/tools/patches/slippi-ishiiruka-savestate.patch"
CMAKE="${CMAKE:-$HOME/.local/bin/cmake}"

if [ ! -d "$SRC" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
    (cd "$SRC" && git checkout "$FORK_COMMIT" && git submodule update --init --recursive)
fi

cd "$SRC"
# Patch already in the fork branch; apply manually only when building from
# the upstream branch instead: git apply "$PATCH"


COMMON_FLAGS="-DLINUX_LOCAL_DEV=true -DDISABLE_WX=true -DENABLE_HEADLESS=true
  -DENABLE_ALSA=false -DENABLE_PULSEAUDIO=false -DENABLE_EVDEV=false
  -DCMAKE_DISABLE_FIND_PACKAGE_Libudev=TRUE"

build_variant () {
    local dir="$1" extra="$2" dest="$3" sysdir="$4"
    mkdir -p "$dir" && cd "$dir"
    PATH="$HOME/.local/bin:$PATH" "$CMAKE" $COMMON_FLAGS $extra ../ > cmake.log 2>&1
    make -j"$(nproc)" > make.log 2>&1
    cp Binaries/dolphin-emu-nogui Binaries/dolphin-emu
    rm -rf Binaries/Sys && cp -r ../Data/Sys Binaries/Sys
    if [ -n "$sysdir" ]; then
        rm -rf Binaries/Sys/GameSettings && mkdir -p Binaries/Sys/GameSettings
        cp -r "../$sysdir/." Binaries/Sys/GameSettings/
    fi
    touch Binaries/portable.txt
    mkdir -p "$dest"
    rsync -a --delete Binaries/ "$dest/Binaries/"
    cd ..
    echo "built + installed: $dest"
}

build_variant build ""                  "$REPO_DIR/emulator_ss" ""
build_variant build-pb "-DIS_PLAYBACK=true" "$REPO_DIR/emulator_pb" "Data/PlaybackGeckoCodes"

echo "done. Point --dolphin-path at emulator_ss/Binaries/dolphin-emu"
