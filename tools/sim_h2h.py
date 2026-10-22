#!/usr/bin/env python3
"""Head-to-head between two variants of one character in the native sim, with statistics.

Variant A and variant B play a ditto. Every template is played equally often in both
orientations (A on the template's first port, and A on its second), so port and stage effects
cancel; the envs are a multiple of 2 x templates for that. Each env contributes the same number of
games (not the first N to finish, which would favour short games). The games are split into two
passes with the policies registered in opposite orders (A first, then B first), so nothing that
depends on which policy steps first can favour one side. Templates set the stage, rules and
costumes: by default 1v1 templates are turned into dittos of --char (both players' character bytes
rewritten, costumes 0 and 1).

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


def lcancel_by_variant(games, envs, names):
    """Each variant's L-cancel rate over the recorded games: realized avoidable lag (landing-run
    length minus the move's cancelled floor) == 0, opponent-interrupted landings dropped; the
    rlvr L-cancel reward's definition. Fox-specific floors (CANCELLED_MIN)."""
    from rlvr.online.tasks.l_cancel_online import CANCELLED_MIN, DAMAGE_STATES, DEAD_STATES
    from tools.sim_replay_check import slp_events
    tally = {n: [0, 0] for n in names}
    for g in games:
        if "slp" not in g:
            continue
        states = {}
        for c, p in slp_events(g["slp"]):
            if c == 0x38 and p[5] == 0:
                states.setdefault(p[4], []).append(struct.unpack(">H", p[7:9])[0])
        for port, name in envs[g["env"]]["players"]:
            st, i = states.get(port, []), 0
            while i < len(st):
                if st[i] in CANCELLED_MIN and (i == 0 or st[i - 1] != st[i]):
                    j = i
                    while j < len(st) and st[j] == st[i]:
                        j += 1
                    if j < len(st) and st[j] not in DEAD_STATES and st[j] not in DAMAGE_STATES:
                        tally[name][0 if j - i <= CANCELLED_MIN[st[i]] else 1] += 1
                    i = j
                else:
                    i += 1
    return {n: dict(success=s, miss=m, rate=s / max(1, s + m)) for n, (s, m) in tally.items()}


def damage_by_variant(games, names):
    """Damage dealt per game by each variant, from the recorded games: the sum of every percent
    increase on the opponent (post-frame percent; resets on death don't count). Returns per-variant
    means and the per-game difference (A dealt minus B dealt) with its standard error."""
    from tools.sim_replay_check import slp_events
    na, nb = names
    dealt = {na: [], nb: []}
    diff = []
    for g in games:
        if "slp" not in g:
            continue
        last, taken = {}, {}
        for c, p in slp_events(g["slp"]):
            if c == 0x38 and p[5] == 0:
                pct = struct.unpack(">f", p[21:25])[0]
                if p[4] in last and pct > last[p[4]]:
                    taken[p[4]] = taken.get(p[4], 0.0) + pct - last[p[4]]
                last[p[4]] = pct
        port = {name: pt for pt, name in g["players"]}
        da, db = taken.get(port[nb], 0.0), taken.get(port[na], 0.0)   # what each side dealt
        dealt[na].append(da)
        dealt[nb].append(db)
        diff.append(da - db)
    n = len(diff)
    if n < 2:
        return None
    mean = sum(diff) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diff) / (n - 1))
    return dict(games=n, dealt={k: sum(v) / n for k, v in dealt.items()}, diff=mean, diff_se=sd / math.sqrt(n))


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
    nt = len(templates)
    envs_n = max(2 * nt, a.envs // (2 * nt) * (2 * nt))   # both orientations of every template
    if envs_n != a.envs:
        print(f"--envs {a.envs} -> {envs_n} (a multiple of 2 x {nt} templates)")
    per_env = math.ceil(a.games / (2 * envs_n))          # games per env per pass
    # env e uses template e % nt; A's side flips every nt envs, so each template gets both orientations
    assign = lambda e, side, ch: (na, nb)[(e // nt + side) % 2]
    t0 = time.time()
    games, envs = [], None
    for order in ((na, nb), (nb, na)):   # pass 2 registers the policies in the opposite order
        pols = {n: {na: pa, nb: pb}[n] for n in order}
        rec = os.path.join(a.record_dir, f"pass_{order[0]}_first") if a.record_dir else None
        v = VecSim(templates, pols, envs_n, a.workers, rng=a.rng, seed=a.seed + len(games),
                   assign=assign, record_dir=rec)
        count = [0] * envs_n
        while min(count) < per_env:
            n0 = len(v.results)
            v.step()
            for r in v.results[n0:]:
                count[r["env"]] += 1
        kept = [0] * envs_n
        for r in v.results:   # each env's first per_env games, in completion order
            if kept[r["env"]] < per_env:
                kept[r["env"]] += 1
                games.append(dict(r, order=f"{order[0]}_first", players=v.envs[r["env"]]["players"],
                                  a_side=(r["env"] // nt) % 2, template=r["env"] % nt))
        envs = v.envs
        v.close()
    wa = wb = draws = 0
    margin = 0
    by = {}
    for r in games:
        sa, sb = r["final"][na][0], r["final"][nb][0]
        margin += sa - sb
        k = (r["template"], r["a_side"])
        by.setdefault(k, [0, 0, 0])
        if sa > sb:
            wa += 1
            by[k][0] += 1
        elif sb > sa:
            wb += 1
            by[k][1] += 1
        else:
            draws += 1
            by[k][2] += 1
    n = wa + wb
    lo, hi = wilson(wa, n)
    lc = lcancel_by_variant(games, envs, (na, nb)) if a.record_dir and a.char == "FOX" else None
    dmg = damage_by_variant(games, (na, nb)) if a.record_dir else None
    rep = dict(char=a.char, a=a.a, b=a.b, games=len(games), a_wins=wa, b_wins=wb, draws=draws, lcancel=lc, damage=dmg,
               by_template_side={f"t{t}_aside{sd}": v_ for (t, sd), v_ in sorted(by.items())},
               by_order={o: [sum(1 for r in games if r["order"] == o and r["final"][na][0] > r["final"][nb][0]),
                             sum(1 for r in games if r["order"] == o and r["final"][nb][0] > r["final"][na][0])]
                         for o in (f"{na}_first", f"{nb}_first")},
               a_win_rate=wa / n if n else None, wilson95=[lo, hi], p_two_sided=binom_p_two_sided(wa, n),
               mean_stock_margin_a=margin / len(games), rng=a.rng, seconds=round(time.time() - t0),
               frames=sum(r["frames"] + 123 for r in games))
    print(f"{na} vs {nb} ({a.char} ditto, {len(games)} games, rng={a.rng}): {wa}-{wb}"
          f"{f' ({draws} draws)' if draws else ''}  {na} win rate {rep['a_win_rate']:.3f} "
          f"[95% {lo:.3f}, {hi:.3f}]  p={rep['p_two_sided']:.3f}  stock margin {rep['mean_stock_margin_a']:+.2f}"
          f"  ({rep['seconds']}s)")
    print("  by order (A wins, B wins): " + "  ".join(f"{o} {w}-{l}" for o, (w, l) in rep["by_order"].items()))
    print("  by template/A side (A, B, draws): " + "  ".join(f"{k} {w}-{l}-{d}" for k, (w, l, d) in rep["by_template_side"].items()))
    if dmg:
        print(f"  damage dealt per game: {na} {dmg['dealt'][na]:.1f}%  {nb} {dmg['dealt'][nb]:.1f}%  "
              f"difference {dmg['diff']:+.1f}% ± {dmg['diff_se']:.1f} (1 SE, {dmg['games']} games)")
    if lc:
        print("  L-cancel: " + "  ".join(f"{n} {d['success']}/{d['success'] + d['miss']} = {100 * d['rate']:.1f}%"
                                        for n, d in lc.items()))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(report=rep, games=games), indent=1))


if __name__ == "__main__":
    main()
