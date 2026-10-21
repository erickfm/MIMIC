#!/usr/bin/env python3
"""Head-to-head between two variants of one character in the native sim, with statistics.

Variant A and variant B play a ditto. A takes port 1 in even envs and port 2 in odd ones, so
port advantage cancels. Templates set the stage, rules and costumes: by default 1v1 templates are
turned into dittos of --char (both players' character bytes rewritten, costumes 0 and 1).

Reports A's record, win rate with a Wilson 95% interval, a two-sided exact binomial p-value
against 50% (draws excluded), and the mean stock margin. Use a same-model control (A and B
identical) to check the rig itself comes out at 50%.

  python3 tools/sim_h2h.py --char FOX --a fp32=hf_eval/fox-master:fp32 \
      --b bf16=hf_eval/fox-master:bf16 --templates replays_doc_rt --games 300 --out h2h.json
"""
import argparse, json, math, os, struct, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MELEE_TO_EXT = {"CPTFALCON": 0, "DK": 1, "FOX": 2, "GAMEANDWATCH": 3, "KIRBY": 4, "BOWSER": 5,
                "LINK": 6, "LUIGI": 7, "MARIO": 8, "MARTH": 9, "MEWTWO": 10, "NESS": 11,
                "PEACH": 12, "PIKACHU": 13, "POPO": 14, "JIGGLYPUFF": 15, "SAMUS": 16,
                "YOSHI": 17, "ZELDA": 18, "SHEIK": 19, "FALCO": 20, "YLINK": 21, "DOC": 22,
                "ROY": 23, "PICHU": 24, "GANONDORF": 25}


def make_ditto(src, char, out):
    """A copy of 1v1 template `src` with both players set to `char` (costumes 0 and 1)."""
    d = bytearray(Path(src).read_bytes())
    i = d.find(b"raw[$U#l") + 12
    gs = i + 1 + d[i + 1]               # Game Start follows the Event Payloads event
    if d[gs] != 0x36:
        raise ValueError(f"{src}: Game Start is not the second event")
    k = 0
    for slot in range(4):
        b = gs + 1 + 0x64 + 0x24 * slot
        if d[b + 1] != 3:
            d[b] = MELEE_TO_EXT[char]
            d[b + 3] = k
            k += 1
    if k != 2:
        raise ValueError(f"{src}: not a 1v1 template")
    Path(out).write_bytes(d)


def variant(spec):
    """name=CKPT_DIR[:fp32|bf16] -> (name, policy dict)."""
    name, rest = spec.split("=", 1)
    d, _, prec = rest.partition(":")
    prec = prec or "bf16"
    if prec not in ("fp32", "bf16"):
        raise ValueError(f"precision must be fp32 or bf16, not {prec!r}")
    # fp32 = the Dolphin inference path's numerics (no autocast, no compile); bf16 = the fast path.
    return name, dict(dir=d, amp=prec == "bf16", compile=prec == "bf16")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def binom_p_two_sided(k, n):
    """Exact two-sided binomial test against p = 0.5."""
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) / 2 ** n for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk * (1 + 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", required=True, help="libmelee CHARACTER name both variants play, e.g. FOX")
    ap.add_argument("--a", required=True, metavar="NAME=CKPT_DIR[:fp32|bf16]")
    ap.add_argument("--b", required=True, metavar="NAME=CKPT_DIR[:fp32|bf16]")
    ap.add_argument("--templates", required=True, help="dir of 1v1 .slp templates (made into dittos)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rng", default="online", choices=["online", "offline"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-dir", help="write every game as a .slp here")
    ap.add_argument("--out", help="JSON report path")
    a = ap.parse_args()
    from rlvr.sim.vec_env import VecSim
    (na, pa), (nb, pb) = variant(a.a), variant(a.b)
    if na == nb:
        raise SystemExit("--a and --b need different names (use e.g. ctrlA / ctrlB for a control)")
    tmp = tempfile.mkdtemp(prefix="sim_h2h_templates_")
    templates = []
    for src in sorted(Path(a.templates).glob("*.slp")):
        out = os.path.join(tmp, src.name)
        make_ditto(src, a.char, out)
        templates.append(out)
    assign = lambda e, side, ch: (na, nb)[(e + side) % 2]
    t0 = time.time()
    v = VecSim(templates, {na: pa, nb: pb}, a.envs, a.workers, rng=a.rng, seed=a.seed,
               assign=assign, record_dir=a.record_dir)
    while len(v.results) < a.games:
        v.step()
    games = v.results[:a.games]
    v.close()
    wa = wb = draws = 0
    margin = 0
    for r in games:
        sa, sb = r["final"][na][0], r["final"][nb][0]
        margin += sa - sb
        if sa > sb:
            wa += 1
        elif sb > sa:
            wb += 1
        else:
            draws += 1
    n = wa + wb
    lo, hi = wilson(wa, n)
    rep = dict(char=a.char, a=a.a, b=a.b, games=len(games), a_wins=wa, b_wins=wb, draws=draws,
               a_win_rate=wa / n if n else None, wilson95=[lo, hi], p_two_sided=binom_p_two_sided(wa, n),
               mean_stock_margin_a=margin / len(games), rng=a.rng, seconds=round(time.time() - t0),
               frames=sum(r["frames"] + 123 for r in games))
    print(f"{na} vs {nb} ({a.char} ditto, {len(games)} games, rng={a.rng}): {wa}-{wb}"
          f"{f' ({draws} draws)' if draws else ''}  {na} win rate {rep['a_win_rate']:.3f} "
          f"[95% {lo:.3f}, {hi:.3f}]  p={rep['p_two_sided']:.3f}  stock margin {rep['mean_stock_margin_a']:+.2f}"
          f"  ({rep['seconds']}s)")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(report=rep, games=games), indent=1))


if __name__ == "__main__":
    main()
