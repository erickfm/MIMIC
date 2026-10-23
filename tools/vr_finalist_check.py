"""Finalist checks on recorded sim self-play dittos (2026-10-23): per-VR reward rates for every
VR (trained or not) and behaviour stats that would expose reward hacking, for a candidate's games
vs the BC games.   python3 finalist_check.py RL_DIR BC_DIR"""
import collections, glob, json, sys
import melee
sys.path.insert(0, "/home/erick/projects/MIMIC")
import rlvr.sim.vr as V


def game_stats(path):
    c = melee.Console(is_dolphin=False, path=str(path), allow_old_version=True)
    c.connect()
    h, mods, ports = collections.deque(maxlen=V.HISTORY), None, None
    rew = collections.defaultdict(float)
    beh = collections.Counter()
    n = 0
    last = None
    while True:
        gs = c.step()
        if gs is None:
            break
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            continue
        if mods is None:
            ports = sorted(gs.players)
            mods = {p: [m for m, _ in V.build(",".join(V.VRS), p)] for p in ports}
        h.append(gs)
        n += 1
        last = gs
        for p in ports:
            ps = gs.players[p]
            for m in mods[p]:
                rew[m.id] += m.observe(h)
            name = ps.action.name
            beh["ledge"] += name.startswith("EDGE_")
            beh["offstage"] += bool(ps.off_stage)
            beh["shield"] += name.startswith("SHIELD")
            beh["airborne"] += not ps.on_ground
    for p in ports:
        for m in mods[p]:
            rew[m.id] += m.finalize(h)
    stocks = sorted(int(last.players[p].stock) for p in ports) if last else [0, 0]
    return dict(frames=n, players=len(ports), rew=dict(rew), beh=dict(beh),
                draw=int(stocks[0] == stocks[1]), timeout=int(stocks[0] > 0))


def summarize(d):
    games = [game_stats(f) for f in sorted(glob.glob(f"{d}/**/*.slp", recursive=True))]
    F = sum(g["frames"] * g["players"] for g in games)   # player-frames
    out = dict(games=len(games), mean_len_s=round(sum(g["frames"] for g in games) / len(games) / 60, 1),
               draws=sum(g["draw"] for g in games), timeouts=sum(g["timeout"] for g in games))
    for k in V.VRS:
        out[f"vr/{k}"] = round(sum(g["rew"].get(k, 0) for g in games) / F * 3600, 4)   # per player-minute
    for k in ("ledge", "offstage", "shield", "airborne"):
        out[f"frac/{k}"] = round(sum(g["beh"].get(k, 0) for g in games) / F, 4)
    return out


if __name__ == "__main__":
    a, b = summarize(sys.argv[1]), summarize(sys.argv[2])
    print(f"{'':28s} {'candidate':>12s} {'BC':>12s}")
    for k in a:
        print(f"{k:28s} {a[k]!s:>12s} {b[k]!s:>12s}")
    json.dump(dict(candidate=a, bc=b), open(f"{sys.argv[1]}/finalist_check.json", "w"), indent=1)
