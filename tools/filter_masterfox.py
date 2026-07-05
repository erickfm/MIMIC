"""Scan the all-master .slp pool and keep only games where the Fox we'd train
on is itself MASTER rank. Per-port rank is in netplay.name ('Master Player' /
'Diamond Player' / 'Platinum Player'). A game is kept iff every Fox port in it
is master (so whichever Fox slp_to_shards extracts is guaranteed master; this
also drops master-platinum/diamond Fox-dittos that mix a non-master Fox)."""
import glob
from concurrent.futures import ProcessPoolExecutor
from peppi_py import read_slippi

FOX_EXT = 2  # peppi external (CSS-order) id for Fox

def classify(p):
    try:
        g = read_slippi(p, skip_frames=True)
        fox_ranks = []
        for pl in g.start.players:
            if pl is None:
                continue
            if pl.character == FOX_EXT:
                nm = (pl.netplay.name if pl.netplay else "") or ""
                fox_ranks.append(nm.replace(" Player", "").strip().lower())
        if fox_ranks and all(r == "master" for r in fox_ranks):
            return p
    except Exception:
        return None
    return None

def main():
    import sys
    dirs = sys.argv[1:] or ["data/fox_ranked_slp", "data/_held_slp"]
    files = []
    for d in dirs:
        files += glob.glob(d + "/*.slp")
    keep = []
    with ProcessPoolExecutor(max_workers=96) as ex:
        for r in ex.map(classify, files, chunksize=32):
            if r:
                keep.append(r)
    with open("data/masterfox_keep.txt", "w") as f:
        f.write("\n".join(keep) + "\n")
    print(f"scanned={len(files)} master_fox_keep={len(keep)} "
          f"({100.0*len(keep)/max(1,len(files)):.1f}% of pool)")

if __name__ == "__main__":
    main()
