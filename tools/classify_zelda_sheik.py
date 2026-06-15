#!/usr/bin/env python3
"""Classify every ZELDA_SHEIK-bucket game into SHEIK vs ZELDA by MAJORITY
in-game form (frames), not just CSS pick. A Zelda picker who transforms and
plays mostly Sheik is labeled SHEIK. Streams the bucket (download/parse/delete).

Output: data/ranked_index/zs_split.jsonl  {filename, p1, p2}  with values in
{ZELDA, SHEIK} per perspective; plus a printed count of real-Zelda games.
"""
import glob, json, subprocess, tempfile, os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import peppi_py
import numpy as np

REPO="erickfm/melee-ranked-replays"
OUT=Path("/home/erick/projects/MIMIC/data/ranked_index/zs_split.jsonl")
DONE=Path("/home/erick/projects/MIMIC/data/ranked_index/zs_done.txt")
ZELDA_EXT, SHEIK_EXT = 18, 19
ZELDA_FORM, SHEIK_FORM = 19, 7   # in-game per-frame char ids

def classify(path):
    g=peppi_py.read_slippi(path, skip_frames=True)
    players=[(i,p.character) for i,p in enumerate(g.start.players) if p is not None]
    if len(players)!=2: return None
    need_frames=any(pick==ZELDA_EXT for _,pick in players)
    gf=peppi_py.read_slippi(path) if need_frames else None
    out=[]
    for i,pick in players:
        if pick==SHEIK_EXT:
            out.append("SHEIK")
        elif pick==ZELDA_EXT:
            try:
                chars=gf.frames.ports[i].leader.post.character.to_numpy()
                z=int((chars==ZELDA_FORM).sum()); s=int((chars==SHEIK_FORM).sum())
                out.append("ZELDA" if z>=s else "SHEIK")
            except Exception:
                out.append("ZELDA")  # pick said zelda, default
        else:
            out.append("OTHER")  # shouldn't happen in this bucket
    return out

def targets():
    api=HfApi()
    return sorted(f for f in api.list_repo_files(REPO,repo_type="dataset")
                  if f.startswith("ZELDA_SHEIK/") and f.endswith(".tar.gz"))

def main():
    done=set(DONE.read_text().split()) if DONE.exists() else set()
    tbs=targets()
    fh=open(OUT,"a")
    zcount=0
    for k,rp in enumerate(tbs):
        if rp in done: continue
        tmp=Path(tempfile.mkdtemp(prefix="zs_"))
        try:
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT","30")
            tp=hf_hub_download(REPO,rp,repo_type="dataset",local_dir=str(tmp/"dl"))
            subprocess.run(["tar","-xzf",tp,"-C",str(tmp)],check=True,capture_output=True)
            files=glob.glob(str(tmp/"**"/"*.slp"),recursive=True)
            nz=0
            for f in files:
                try:
                    c=classify(f)
                    if c is None: continue
                    fh.write(json.dumps({"filename":Path(f).name,"p1":c[0],"p2":c[1]})+"\n")
                    if "ZELDA" in c: nz+=1
                except Exception:
                    pass
            fh.flush()
            with open(DONE,"a") as d: d.write(rp+"\n")
            zcount+=nz
            print(f"[{k+1}/{len(tbs)}] {rp.split('/')[1]}: {len(files)} games, {nz} with real Zelda",flush=True)
        finally:
            subprocess.run(["rm","-rf",str(tmp)],check=False)
    fh.close()
    print(f"\nTOTAL real-Zelda games (majority Zelda frames): {zcount}")

if __name__=="__main__":
    main()
