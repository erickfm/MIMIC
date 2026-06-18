#!/usr/bin/env python3
"""Partition Fox .slp files into per-rank dirs by the FOX player's rank.

The ranked dataset stores per-player rank in `start.players[*].netplay.name`
("Master Player" / "Diamond Player" / "Platinum Player"). For rank-specific
Fox models we want only the target-rank Fox's perspective, so we symlink each
file into `data/fox_<rank>_slp/` keyed on the Fox player's rank. The existing
`slp_to_shards.py --character 1` then keeps that Fox perspective.

Routing per file:
  - exactly one Fox player  -> that Fox's rank dir
  - Fox ditto, same rank    -> that rank dir (both perspectives are valid)
  - Fox ditto, mixed rank   -> EXCLUDE (can't separate via --character alone)
  - no Fox player           -> skip

Idempotent (skips existing symlinks); safe to call repeatedly as pairs stream
in. peppi external char id for Fox = 2.
"""
import argparse, glob, os, re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import peppi_py

FOX_EXT = 2
RANKS = {"master", "diamond", "platinum"}


def _rank_of(netplay):
    if netplay is None or not netplay.name:
        return None
    w = netplay.name.split()[0].lower()
    return w if w in RANKS else None


def classify(path):
    """Return (target_rank, reason). target_rank None means skip/exclude."""
    try:
        g = peppi_py.read_slippi(path, skip_frames=True)
        players = [p for p in g.start.players if p is not None]
        fox = [(p, _rank_of(p.netplay)) for p in players if p.character == FOX_EXT]
        if not fox:
            return (None, "no_fox")
        ranks = {r for _, r in fox if r is not None}
        if len(fox) == 1:
            return (next(iter(ranks)) if ranks else None, "single_fox" if ranks else "no_rank")
        # Fox ditto
        if len(ranks) == 1:
            return (next(iter(ranks)), "same_rank_ditto")
        return (None, "mixed_rank_ditto")
    except Exception as e:
        return (None, f"err:{type(e).__name__}")


def _work(args):
    path, out_root, move = args
    rank, reason = classify(path)
    if rank is not None:
        dst_dir = os.path.join(out_root, f"fox_{rank}_slp")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(path))
        if not os.path.lexists(dst):
            if move:
                os.rename(path, dst)          # same-fs move (instant); raw can be purged
            else:
                os.symlink(os.path.abspath(path), dst)
    return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slp-dir", required=True, help="dir of extracted Fox .slp")
    ap.add_argument("--out-root", default="data", help="root for fox_<rank>_slp dirs")
    ap.add_argument("--move", action="store_true",
                    help="move files into rank dirs (same-fs) instead of symlink")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.slp_dir, "**", "*.slp"), recursive=True)
    print(f"partitioning {len(files)} files from {a.slp_dir} (move={a.move})", flush=True)
    reasons = Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_work, [(f, a.out_root, a.move) for f in files], chunksize=64):
            reasons[r] += 1
    print("reasons:", dict(reasons))
    for rank in sorted(RANKS):
        d = os.path.join(a.out_root, f"fox_{rank}_slp")
        n = len(glob.glob(os.path.join(d, "*.slp"))) if os.path.isdir(d) else 0
        print(f"  fox_{rank}_slp: {n} files (cumulative)")
    excl = reasons.get("mixed_rank_ditto", 0)
    print(f"EXCLUDED mixed-rank Fox dittos this batch: {excl}")


if __name__ == "__main__":
    main()
