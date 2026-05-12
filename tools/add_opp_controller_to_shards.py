#!/usr/bin/env python3
"""Bake `opp_controller` one-hot into existing shards, parallel to `self_controller`.

The v2 shards store raw opp inputs (`opp_buttons`, `opp_analog`, `opp_c_dir`)
alongside a pre-encoded 56-dim `self_controller` one-hot. To let the WM
encoder treat the opponent symmetrically with self, we re-encode opp's raw
inputs into the same 56-dim layout (37 stick + 9 cstick + N combos + 3
shoulder) and write it back into the shard as `opp_controller`.

No .slp re-parse needed — works directly off the shard's existing raw
opp fields. Idempotent: shards that already have `opp_controller` are
skipped.

    python3 tools/add_opp_controller_to_shards.py --data-dir data/fox_all_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mimic.features import encode_controller_onehot


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--glob", default="*shard_*.pt",
                    help="Glob for shard files.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_combos(data_dir: Path):
    with open(data_dir / "controller_combos.json") as fh:
        data = json.load(fh)
    if "combos" in data:
        combos = [tuple(c) for c in data["combos"]]
        return {c: i for i, c in enumerate(combos)}, len(combos)
    # 7-class rule-based: no combo_to_idx lookup needed
    return {}, data.get("n_combos", 7)


def encode_opp(states: dict, combo_to_idx: dict, n_combos: int) -> torch.Tensor:
    buttons = states["opp_buttons"].numpy()                     # (N, 12)
    analog = states["opp_analog"].numpy().astype(np.float32)    # (N, 4)
    c_dir = states["opp_c_dir"].numpy().astype(np.int64)        # (N,)
    onehot = encode_controller_onehot(
        buttons, analog, c_dir,
        combo_to_idx, n_combos,
        norm_stats=None,   # analog values in the shard are already raw
    )
    return torch.from_numpy(onehot)


def main():
    args = parse_args()
    combo_to_idx, n_combos = load_combos(args.data_dir)
    print(f"combo scheme: {'rule-based 7-class' if not combo_to_idx else f'lookup {n_combos}-class'}")

    shards = sorted(args.data_dir.glob(args.glob))
    print(f"found {len(shards)} shards in {args.data_dir}")

    t0 = time.time()
    skipped = added = 0
    for i, path in enumerate(shards):
        shard = torch.load(path, weights_only=True, mmap=True)
        states = shard["states"]
        if "opp_controller" in states:
            skipped += 1
            continue
        # Need raw opp inputs to encode.
        for key in ("opp_buttons", "opp_analog", "opp_c_dir"):
            if key not in states:
                print(f"  SKIP {path.name}: missing {key}")
                skipped += 1
                break
        else:
            opp_ctrl = encode_opp(states, combo_to_idx, n_combos)
            if opp_ctrl.shape[0] != states["self_controller"].shape[0]:
                print(f"  ERROR {path.name}: opp_controller rows "
                      f"{opp_ctrl.shape[0]} != self_controller rows "
                      f"{states['self_controller'].shape[0]}")
                skipped += 1
                continue
            # Must materialize mmap tensors before save.
            new_states = {k: v.clone() for k, v in states.items()}
            new_states["opp_controller"] = opp_ctrl
            new_shard = dict(shard)
            new_shard["states"] = new_states
            if args.dry_run:
                print(f"  [dry] would write {path.name}  opp_controller "
                      f"shape={tuple(opp_ctrl.shape)}")
            else:
                tmp = path.with_suffix(".pt.tmp")
                torch.save(new_shard, tmp)
                tmp.replace(path)
            added += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(shards):
            dt = time.time() - t0
            rate = (i + 1) / dt
            remaining = (len(shards) - (i + 1)) / rate if rate else 0
            print(f"  [{i + 1}/{len(shards)}] added={added} skipped={skipped} "
                  f"({rate:.1f} shards/s, ~{remaining:.0f}s left)")

    print(f"done: added={added} skipped={skipped} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
