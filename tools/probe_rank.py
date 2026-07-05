import glob, json
from peppi_py import read_slippi

def sample(prefix, n=1):
    out = []
    for d in ("data/fox_ranked_slp", "data/_held_slp"):
        out += glob.glob(f"{d}/{prefix}*.slp")
        if len(out) >= n:
            break
    return out[:n]

for tier in ("master-master", "master-diamond", "master-platinum"):
    for p in sample(tier, 1):
        print("=" * 70)
        print("FILE", p.split("/")[-1])
        try:
            g = read_slippi(p, skip_frames=True)
            print("  top-level attrs:", [a for a in dir(g) if not a.startswith("_")])
            md = g.metadata
            print("  METADATA TYPE", type(md))
            try:
                print("  METADATA JSON:", json.dumps(md, default=str)[:1500])
            except Exception as e:
                print("  metadata repr:", repr(md)[:1500])
            st = g.start
            print("  START attrs:", [a for a in dir(st) if not a.startswith("_")])
            # hunt for any rank / rating / match fields
            for a in dir(st):
                if any(k in a.lower() for k in ("rank", "rating", "match", "player")):
                    try:
                        print("   start.%s =" % a, getattr(st, a))
                    except Exception:
                        pass
        except Exception as e:
            print("  ERR", type(e).__name__, e)
