#!/usr/bin/env python3
"""Average the weights of N checkpoints (SWA / LAWA tail-averaging).

Element-wise mean of `model_state_dict` float tensors across the given
checkpoints; non-float tensors and every other checkpoint field (config,
norm_stats, stick/shoulder centers, controller_combos, ...) are copied from the
FIRST checkpoint. The result is a normal checkpoint scoreable with
`train.py --eval-only`.

Usage:
    # explicit list
    python tools/average_checkpoints.py --out checkpoints/avg.pt \
        checkpoints/run_step020000.pt checkpoints/run_step022000.pt
    # last K step-checkpoints of a run (sorted by step)
    python tools/average_checkpoints.py --out checkpoints/avg.pt \
        --run checkpoints/fox-master-20260616-warmrestart --last 5
    # step range
    python tools/average_checkpoints.py --out checkpoints/avg.pt \
        --run checkpoints/fox-master-20260616-warmrestart --min-step 18000 --max-step 26000
"""
import argparse
import glob
import os
import re
import torch


def _step_of(path):
    m = re.search(r"_step0*(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def _list_run_ckpts(prefix):
    paths = glob.glob(f"{prefix}_step*.pt")
    return sorted(paths, key=_step_of)


def main():
    ap = argparse.ArgumentParser(description="Average checkpoint weights (SWA/LAWA)")
    ap.add_argument("checkpoints", nargs="*", help="explicit checkpoint paths to average")
    ap.add_argument("--run", default=None, help="run prefix; selects its _step*.pt files")
    ap.add_argument("--last", type=int, default=None, help="use the last K step-checkpoints of --run")
    ap.add_argument("--min-step", type=int, default=None)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = list(args.checkpoints)
    if args.run:
        run_paths = _list_run_ckpts(args.run)
        if args.min_step is not None or args.max_step is not None:
            lo = args.min_step if args.min_step is not None else -1
            hi = args.max_step if args.max_step is not None else 10 ** 12
            run_paths = [p for p in run_paths if lo <= _step_of(p) <= hi]
        if args.last is not None:
            run_paths = run_paths[-args.last:]
        paths += run_paths
    if not paths:
        raise SystemExit("no checkpoints selected")

    print(f"averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"    step {_step_of(p):>7}  {os.path.basename(p)}")

    base = torch.load(paths[0], map_location="cpu", weights_only=False)
    base_sd = base["model_state_dict"]
    avg = {k: (v.clone().float() if torch.is_floating_point(v) else v.clone())
           for k, v in base_sd.items()}
    for p in paths[1:]:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
        for k in avg:
            if torch.is_floating_point(avg[k]):
                avg[k] += sd[k].float()
    n_float = 0
    for k in avg:
        if torch.is_floating_point(avg[k]):
            avg[k] /= len(paths)
            avg[k] = avg[k].to(base_sd[k].dtype)
            n_float += 1

    out = dict(base)
    out["model_state_dict"] = avg
    out["averaged_from"] = [os.path.basename(p) for p in paths]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"averaged {n_float} float tensors over {len(paths)} ckpts -> {args.out}")


if __name__ == "__main__":
    main()
