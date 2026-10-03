"""One h2h lane: run a slice of pairings, N matches each, FFW, own-norms.

Usage: python3 tools/run_h2h_lane.py <lane> <slippi_port> <pairs_slice.json> <n_matches>
Writes eval_results/h2h/<A>__vs__<B>.json and replays to replays_h2h/<lane>/.
"""
import sys, json, subprocess, time
from pathlib import Path
import melee

ROOT = Path("/home/erick/projects/MIMIC")
STAGES = "FINAL_DESTINATION,BATTLEFIELD,DREAMLAND,YOSHIS_STORY,FOUNTAIN_OF_DREAMS,POKEMON_STADIUM"

def enum(c):
    return {"ice_climbers": "POPO", "puff": "JIGGLYPUFF"}.get(c, c.upper())

def ddir(c):
    return "hf_eval/fox-master" if c == "fox" else f"hf_eval/{c}"

def main(lane, port, slice_path, n, subdir=""):
    pairs = json.load(open(slice_path))
    out_dir = ROOT / "eval_results/h2h" / subdir; out_dir.mkdir(parents=True, exist_ok=True)
    rep = ROOT / f"replays_h2h/{lane}"; rep.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"lane_{lane}.log"
    lf = open(log, "w")
    def say(m): lf.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); lf.flush()
    say(f"lane {lane} port {port}: {len(pairs)} pairings x {n}")
    for k, (a, b) in enumerate(pairs):
        out = out_dir / f"{a}__vs__{b}.json"
        if out.exists():
            say(f"[{a} vs {b}] exists, skip"); continue
        # melee.Character validation (skip typo'd names early)
        try: melee.Character[enum(a)]; melee.Character[enum(b)]
        except KeyError as e: say(f"[{a} vs {b}] bad enum {e}"); continue
        cmd = [
            "python3", "tools/play.py",
            "--ckpt", f"{ddir(a)}/model.pt", "--data-dir", ddir(a),
            "--opponent", f"{ddir(b)}/model.pt", "--opponent-data-dir", ddir(b),
            "--character", enum(a), "--opponent-character", enum(b),
            "--n-matches", str(n), "--stages", STAGES, "--alternate-ports",
            "--use-exi-inputs", "--enable-ffw", "--gfx-backend", "Null",
            "--disable-audio",
            # DUAL-PAD FFW MUST use emulator_ss (fork 241c13f, fixed 2026-07-16).
            # The stock emulator_ffw AppImage has a keep-alive-FLUSH race that
            # serves stale pads to BOTH EXI-injected controllers in a bot-vs-bot
            # game -> symmetric mistiming (L-cancel ~13-33% vs ~85-91% here),
            # 4x-short matches, and it INVERTS outcomes (measured: bowser 6-0 vs
            # fox on stock FFW, fox 5-0 on emulator_ss AND realtime). The
            # 2026-07-20 campaign ran on the stock build and is invalid. See
            # docs/research-notes-2026-07-22.md.
            "--dolphin-path", "emulator_ss/Binaries/dolphin-emu",
            "--iso-path", "melee.iso", "--slippi-port", str(port),
            "--replay-dir", str(rep), "--seed", "42", "--out", str(out),
        ]
        t0 = time.time()
        # own process group so a hung play.py + its Dolphin child are killed
        # together on timeout (600s bounds a hang; partial --out is kept).
        import os, signal
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception: pass
            say(f"[{a} vs {b}] TIMEOUT(600s) killed pgroup, moving on")
        if out.exists():
            d = json.load(open(out))
            say(f"[{a} vs {b}] {k+1}/{len(pairs)} A={d['a_wins']} B={d['b_wins']} "
                f"n={d['n_matches']} win={d['a_win_rate']*100:.0f}% ({time.time()-t0:.0f}s)")
        else:
            say(f"[{a} vs {b}] FAILED rc={p.returncode} ({time.time()-t0:.0f}s)")
    say(f"lane {lane} DONE")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]),
         sys.argv[5] if len(sys.argv) > 5 else "")
