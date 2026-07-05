import glob, random, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from peppi_py import read_slippi

# peppi external id: Fox = 2 (CSS order). We train --character 1 = libmelee Fox.
FOX_EXT = 2

def info(p):
    try:
        g = read_slippi(p, skip_frames=True)
        ranks = []
        fox_ranks = []
        for pl in g.start.players:
            if pl is None:
                continue
            nm = (pl.netplay.name if pl.netplay else "") or ""
            rank = nm.replace(" Player", "").strip().lower() or "?"
            ranks.append(rank)
            if pl.character == FOX_EXT:
                fox_ranks.append(rank)
        return (tuple(sorted(ranks)), tuple(sorted(fox_ranks)))
    except Exception:
        return None

def tier_files(prefix, n):
    out = []
    for d in ("data/fox_ranked_slp", "data/_held_slp"):
        out += glob.glob(f"{d}/{prefix}*.slp")
    random.Random(0).shuffle(out)
    return out[:n]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
for tier in ("master-master", "master-diamond", "master-platinum"):
    files = tier_files(tier, N)
    pair_ct, fox_ct = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(info, files, chunksize=8):
            if r is None:
                pair_ct["PARSE_FAIL"] += 1
                continue
            pair_ct[r[0]] += 1
            for fr in r[1]:
                fox_ct[fr] += 1
    print("=" * 60)
    print(f"{tier}  (n={len(files)})")
    print("  port-rank pairs:", dict(pair_ct))
    print("  FOX-port rank distribution:", dict(fox_ct))
