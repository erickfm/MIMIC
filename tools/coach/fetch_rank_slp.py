#!/usr/bin/env python3
"""Stream the first N Fox games of one rank from erickfm/melee-ranked-replays, without
downloading the whole tarball. Keeps a game only if every Fox port in it has the wanted rank
(per-port rank is in netplay.name, as in tools/filter_masterfox.py).

  python3 tools/coach/fetch_rank_slp.py --rank diamond --n 1000 --out data/coach_rank/diamond
"""
import argparse, io, tarfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from peppi_py import read_slippi

FOX_EXT = 2   # peppi external (CSS-order) id


def fox_ranks(path):
    g = read_slippi(str(path), skip_frames=True)
    out = []
    for pl in g.start.players:
        if pl is not None and pl.character == FOX_EXT:
            nm = (pl.netplay.name if pl.netplay else "") or ""
            out.append(nm.replace(" Player", "").strip().lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["master", "diamond", "platinum"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--archive", default="a1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"FOX/FOX_{a.rank}-{a.rank}_{a.archive}.tar.gz"
    url = hf_hub_url("erickfm/melee-ranked-replays", name, repo_type="dataset")
    kept = seen = 0
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".slp")):
                    continue
                seen += 1
                p = out / Path(m.name).name
                p.write_bytes(tf.extractfile(m).read())
                try:
                    ok = (lambda rs: rs and all(x == a.rank for x in rs))(fox_ranks(p))
                except Exception:
                    ok = False
                if ok:
                    kept += 1
                else:
                    p.unlink()
                if kept >= a.n:
                    break
    print(f"{name}: kept {kept} of {seen} games -> {out}")


if __name__ == "__main__":
    main()
