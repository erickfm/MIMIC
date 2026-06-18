#!/usr/bin/env python3
"""Find real-Zelda games hiding in the buggy MARTH bucket (a1,a2,a4,a5,a6).
The char-ID scramble sends real Zelda (ext 18) into the MARTH folder. Keep
only games where a Zelda picker plays MAJORITY Zelda frames."""
import glob, json, subprocess, tempfile, os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import peppi_py

ZELDA_EXT, ZELDA_FORM, SHEIK_FORM = 18, 19, 7
OUT = Path("data/ranked_index/marth_zelda.jsonl")
DONE = Path("data/ranked_index/mz_done.txt")


def is_majority_zelda(path):
    g = peppi_py.read_slippi(path, skip_frames=True)
    players = [(i, p.character) for i, p in enumerate(g.start.players) if p is not None]
    if len(players) != 2 or not any(pk == ZELDA_EXT for _, pk in players):
        return False
    gf = peppi_py.read_slippi(path)
    for i, pk in players:
        if pk != ZELDA_EXT:
            continue
        try:
            ch = gf.frames.ports[i].leader.post.character.to_numpy()
            z = int((ch == ZELDA_FORM).sum()); s = int((ch == SHEIK_FORM).sum())
            if z >= s:
                return True
        except Exception:
            return True
    return False


def main():
    api = HfApi()
    tbs = sorted(f for f in api.list_repo_files("erickfm/melee-ranked-replays", repo_type="dataset")
                 if f.startswith("MARTH/") and f.endswith(".tar.gz") and "_a3." not in f)
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    fh = open(OUT, "a"); zc = 0
    for k, rp in enumerate(tbs):
        if rp in done:
            continue
        tmp = Path(tempfile.mkdtemp(prefix="mz_"))
        try:
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
            tp = hf_hub_download("erickfm/melee-ranked-replays", rp, repo_type="dataset",
                                 local_dir=str(tmp / "dl"))
            subprocess.run(["tar", "-xzf", tp, "-C", str(tmp)], check=True, capture_output=True)
            files = glob.glob(str(tmp / "**" / "*.slp"), recursive=True); nz = 0
            for f in files:
                try:
                    if is_majority_zelda(f):
                        fh.write(json.dumps({"filename": Path(f).name}) + "\n"); nz += 1
                except Exception:
                    pass
            fh.flush(); open(DONE, "a").write(rp + "\n"); zc += nz
            print(f"[{k+1}/{len(tbs)}] {rp.split('/')[1]}: {len(files)} games, {nz} real Zelda", flush=True)
        finally:
            subprocess.run(["rm", "-rf", str(tmp)], check=False)
    print(f"\nTOTAL real-Zelda in MARTH buckets: {zc}")


if __name__ == "__main__":
    main()
