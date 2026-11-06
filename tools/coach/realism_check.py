#!/usr/bin/env python3
"""Is the coach's opponent realistic? At real decisions of real games, the player's side presses
exactly what the player really pressed over the decision's stretch, and the opponent is the master
model, sampled k times. If the model opponent behaved like the real one, the real outcome would look
like one more try: its rank among the tries would be uniform, and the tries' mean would match it on
average.

Per decision:
  - tries: the player's real inputs forced for the stretch, the opponent from the model (k of them);
  - control: both sides' real inputs forced (k=1). It must reproduce the real stretch exactly (checked
    against the replay), and it scores the real outcome through the same sim + value-model pipeline
    as the tries, so any offset between the two isn't the pipeline's.
Each is scored like the coach scores tries: damage dealt - taken over the stretch (a stock = 40),
plus V after it.

Runs on the CPU (the GPU may be training):
  python3 tools/coach/realism_check.py --code WAVE#666 --char FALCO --n 60
  python3 tools/coach/realism_check.py --code STUC#911 --char FOX --n 60
"""
import argparse, collections, json, random, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "coach"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def seg_score(fr, mp, op, s_):
    seg = fr[:s_ + 1]
    d = lambda pp: float(np.clip(np.diff([f[pp][5] for f in seg]), 0, None).sum())
    sk = lambda pp: float(np.clip(-np.diff([f[pp][6] for f in seg]), 0, None).sum())
    return d(op) - d(mp) + 40 * (sk(op) - sk(mp))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", required=True, help="the player's connect code")
    ap.add_argument("--char", required=True, help="the player's character (FALCO, FOX, ...)")
    ap.add_argument("--n", type=int, default=60, help="decisions")
    ap.add_argument("--k", type=int, default=16, help="model-opponent tries per decision")
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--replays", default=str(Path.home() / "Slippi"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import serve
    from serve import Library, analyse, support, policy_dir, load_value, stretch_of, reaction, POLICY
    from branch import Brancher
    from value import score
    rng = random.Random(1)
    lib = Library()
    lib._scan(str(Path(a.replays).expanduser()))
    games = [g for g in lib.games if support(g, a.code) is None
             and next(p for p in g["players"] if p.get("code") == a.code)["char"] == a.char]
    rng.shuffle(games)
    V = load_value(a.char, "cpu")
    if V is None:
        sys.exit(f"no usable value model for {a.char}")
    iso = serve.find_iso(serve.load_config())[2]
    br = Brancher("cpu", iso=iso)
    out = Path(a.out or ROOT / "data" / "coach_rank" / f"realism_{a.char.lower()}_{a.code.split('#')[0].lower()}.json")
    rows, t0 = [], time.time()
    for g in games:
        if len(rows) >= a.n:
            break
        an = analyse(g, a.code)
        me, opp = an["me"], an["opp"]
        mp, op = me["port"], opp["port"]
        pol = {mp: policy_dir(me["char"]), op: policy_dir(opp["char"])}
        last = an["last"]
        pts = [p for p in an["points"] if p["frame"] >= 240 and p["frame"] + 80 < last]
        for p in rng.sample(pts, min(a.per_game, len(pts))):
            if len(rows) >= a.n:
                break
            F, s_ = p["frame"], stretch_of(p["wait"])
            try:
                ctl = br.run(g["path"], F, 1, s_ + 1, pol, value_at=(mp, [s_]),
                             force=[(mp, g["path"], s_ + 1), (op, g["path"], s_ + 1)])
                tr = br.run(g["path"], F, a.k, s_ + 1, pol, value_at=(mp, [s_]), force=(mp, g["path"], s_ + 1))
            except Exception as e:
                print(f"  skip {g['id']}@{F}: {e}", flush=True)
                continue
            cf = ctl["rollouts"][0]["frames"]
            if len(cf) <= s_ or s_ not in ctl["value_feats"] or s_ not in tr["value_feats"]:
                continue
            # the control must be the real game: percents and stocks over the stretch, frame for frame
            arr = an["_arrays"]
            i0 = int(np.searchsorted(arr["frames"], F))
            real_pct = [(float(arr["pct"][i0 + j]), float(arr["opct"][i0 + j])) for j in range(s_ + 1)]
            sim_pct = [(f[mp][5], f[op][5]) for f in cf[:s_ + 1]]
            exact = all(abs(x[0] - y[0]) < 0.02 and abs(x[1] - y[1]) < 0.02 for x, y in zip(real_pct, sim_pct))
            real = seg_score(cf, mp, op, s_) + float(score(V(ctl["value_feats"][s_]))[0])
            q = score(V(tr["value_feats"][s_]))
            tries, opp_first = [], []
            for i, ro in enumerate(tr["rollouts"]):
                fr = ro["frames"]
                if len(fr) <= s_:
                    continue
                tries.append(seg_score(fr, mp, op, s_) + float(q[i]))
                opp_first.append(reaction(opp["char"], [f[op][3] for f in fr], [f[op][0] for f in fr],
                                          [f[mp][0] for f in fr]).split(" → ")[0])
            if len(tries) < 2:
                continue
            tries = np.array(tries)
            tie = np.abs(tries - real) < 0.05   # batched vs single forwards differ in the last digits
            rank = (float((tries < real - 0.05).sum()) + 0.5 * float(tie.sum())) / len(tries)
            real_opp = reaction(opp["char"], [f[op][3] for f in cf], [f[op][0] for f in cf],
                                [f[mp][0] for f in cf]).split(" → ")[0]
            rows.append(dict(game=g["id"], frame=F, stretch=s_, did=p["did"], exact=exact, real=round(real, 2),
                             tries_mean=round(float(tries.mean()), 2), tries_sd=round(float(tries.std(ddof=1)), 2),
                             rank=round(rank, 4), real_opp=real_opp, opp_seen=opp_first.count(real_opp),
                             opp_top=collections.Counter(opp_first).most_common(3), k=len(tries)))
            r = rows[-1]
            print(f"[{len(rows)}/{a.n}] {time.time() - t0:5.0f}s {g['id']}@{F} exact={exact} real {real:+.1f} "
                  f"tries {r['tries_mean']:+.1f}±{r['tries_sd']:.1f} rank {rank:.2f} real opp '{real_opp}' "
                  f"seen {r['opp_seen']}/{len(tries)}", flush=True)
            out.write_text(json.dumps(dict(code=a.code, char=a.char, k=a.k, rows=rows), indent=1))
    # summary
    ok = [r for r in rows if r["exact"]]
    ranks = np.array([r["rank"] for r in ok])
    diff = np.array([r["tries_mean"] - r["real"] for r in ok])
    hist = np.histogram(ranks, bins=5, range=(0, 1))[0].tolist() if len(ok) else []
    ks = float(np.max(np.abs(np.sort(ranks) - (np.arange(1, len(ranks) + 1) - 0.5) / len(ranks)))) if len(ok) else None
    summ = dict(decisions=len(rows), exact=len(ok), mean_rank=round(float(ranks.mean()), 3) if len(ok) else None,
                rank_quintiles=hist, ks=round(ks, 3) if ks is not None else None,
                ks_5pct=round(1.36 / np.sqrt(len(ok)), 3) if len(ok) else None,
                tries_minus_real=round(float(diff.mean()), 2) if len(ok) else None,
                tries_minus_real_se=round(float(diff.std(ddof=1) / np.sqrt(len(ok))), 2) if len(ok) > 1 else None,
                real_opp_seen=round(float(np.mean([r["opp_seen"] > 0 for r in ok])), 3) if len(ok) else None,
                real_opp_share=round(float(np.mean([r["opp_seen"] / r["k"] for r in ok])), 3) if len(ok) else None,
                policy=str(POLICY.get(a.char)), seconds=round(time.time() - t0))
    print(json.dumps(summ, indent=1))
    out.write_text(json.dumps(dict(code=a.code, char=a.char, k=a.k, summary=summ, rows=rows), indent=1))


if __name__ == "__main__":
    main()
