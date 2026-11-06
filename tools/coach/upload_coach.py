#!/usr/bin/env python3
"""Put the coach's models on Hugging Face so a fresh machine gets them: per character, the policy the
coach uses (coach/<char>/policy/) and its value model (coach/<char>/value.pt), plus
coach/manifest.json listing them. The coach downloads them (serve.py: policy_dir, sync_coach_values).

Only pairs that belong together go up: a value model whose recorded policy (by content, sha1 of
model.pt) is the policy the coach uses for that character now.

  python3 tools/coach/upload_coach.py              # what would go up (nothing is sent)
  python3 tools/coach/upload_coach.py --push       # upload (the owner confirms first)
"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "coach"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--chars", nargs="*")
    a = ap.parse_args()
    from serve import POLICY, HF_REPO, ROOT as R, policy_sha1
    from value import Value, file_sha1
    data = R / "data" / "coach_rank"
    stage = Path(tempfile.mkdtemp(prefix="coach_upload_"))
    manifest = {}
    if a.push:   # keep what's already up there
        try:
            from huggingface_hub import hf_hub_download
            manifest = json.loads(Path(hf_hub_download(HF_REPO, "coach/manifest.json")).read_text())
        except Exception:
            manifest = {}
    for char in a.chars or sorted(POLICY):
        vp = data / f"value_{char.lower()}.pt"
        pdir = R / POLICY[char]
        if not vp.is_file() or not (pdir / "model.pt").is_file():
            continue
        v = Value(str(vp), "cpu")
        if not v.policy_sha1:
            print(f"{char}: value model has no policy sha1 (trained before it was recorded): skipped")
            continue
        if v.policy_sha1 != policy_sha1(pdir):
            print(f"{char}: value model was trained on another policy than {POLICY[char]}: skipped")
            continue
        out = stage / char.lower()
        (out / "policy").mkdir(parents=True)
        for f in pdir.iterdir():
            if f.is_file():
                shutil.copy(f, out / "policy" / f.name)
        shutil.copy(vp, out / "value.pt")
        vc = data / f"vcheck_{char.lower()}_rl.json"
        manifest[char] = dict(policy=POLICY[char], policy_sha1=v.policy_sha1, value_sha1=file_sha1(vp),
                              value_heldout_r2=round(float(v.r2), 3) if v.r2 is not None else None,
                              value_check=vc.name if vc.is_file() else None)
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"{char}: {POLICY[char]} + value ({size / 1e6:.1f} MB)")
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if not a.push:
        print(f"\nstaged in {stage} ({len(manifest)} characters); nothing sent. --push uploads to {HF_REPO}/coach/")
        return
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=HF_REPO, folder_path=str(stage), path_in_repo="coach",
                          commit_message=f"coach models: {', '.join(sorted(manifest))}")
    print(f"uploaded {len(manifest)} characters to {HF_REPO}/coach/")


if __name__ == "__main__":
    main()
