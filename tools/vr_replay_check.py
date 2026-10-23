#!/usr/bin/env python3
"""Validate the sim VRs (rlvr/sim/vr/) on real replays against independent ground truth.

Each replay is parsed twice: by libmelee (the sim's parser, plus the rlvr.sim.vr state-flag hook)
to run the VRs exactly as the sim would, and by slippistats (its own .slp parser) for the truth:

  stock_delta       deaths == stocks lost (+ a death the match ended on); kills + filtered SDs ==
                    opponent deaths; kills == deaths earned by a recent hit or inside one of the
                    player's slippistats combos; (~) vs slippistats' kill combos (did_kill)
  damage_delta      total % dealt / taken == positive percent deltas in slippistats' frames
  l_cancel          success / miss counts vs tools/lcancel_analysis.py (Fox ports)
  combo_length      per-combo move counts vs slippistats ComboComputer.combo_compute
  combo_extend      same combos; rewarded count vs slippistats combos with >= 2 moves and >= 5 %
  tech              situations / punished vs slippistats StatsComputer.tech_compute
  recovery          an array implementation of the same definition over slippistats' frames
                    (position off-stage, Field4 hitstun flag)
  shield_escape     an array implementation over slippistats' state / shield_health
  low_percent_kill  an array implementation (death edge, life-peak %, hit recency)
  neutral_win_loss  slippistats has no neutral stat: an array implementation of the VR's definition
                    over slippistats' frames, plus (~) an approximation rebuilt from both players'
                    slippistats combo starts

  python3 tools/vr_replay_check.py --n 40                       # real master-Fox replays
  python3 tools/vr_replay_check.py --dir <sim record dir> --n 8  # sim-recorded games
"""
import argparse, collections, glob, json, os, sys, types, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

import melee
import rlvr.sim.vr as V
from rlvr.sim.vr import stream as S
from rlvr.sim.vr.outcome import _load_buckets, _FALLBACK_BUCKET, OPP_HIT_MEMORY
from rlvr.sim.vr.situational import (
    CANCELLED_MIN, SHIELD_ACTIONABLE, SHIELD_BREAK_STATES, SHIELD_FAMILY, SHIELD_HIT, SHIELD_LOOKBACK,
    SMALL_SHIELD,
)

# Anonymized ranked replays carry no startAt / lastFrame / player names; slippistats needs them only
# for headers, so fill placeholders (this doesn't touch frame data).
import slippistats.metadata as _SM
_orig_parse = _SM.Metadata._parse.__func__


def _parse(cls, js):
    js = dict(js)
    js.setdefault("startAt", "2000-01-01T00:00:00Z"); js.setdefault("playedOn", "dolphin")
    js.setdefault("lastFrame", 0)
    m = _orig_parse(cls, js)
    m.players = tuple(p if p is not None else types.SimpleNamespace(connect_code=f"P{i}", display_name=f"P{i}")
                      for i, p in enumerate(m.players))
    return m


_SM.Metadata._parse = classmethod(_parse)
from slippistats import ComboComputer, StatsComputer   # noqa: E402
from slippistats.enums.state import Field2, Field4     # noqa: E402

STAGE_NAMES = {"FOUNTAIN_OF_DREAMS": "FOUNTAIN_OF_DREAMS", "YOSHIS_STORY": "YOSHIS_STORY",
               "DREAM_LAND_N64": "DREAMLAND", "POKEMON_STADIUM": "POKEMON_STADIUM",
               "BATTLEFIELD": "BATTLEFIELD", "FINAL_DESTINATION": "FINAL_DESTINATION"}


# ---------------------------------------------------------------- rollback --------------------
FRAME_START, GAME_END = 0x3A, 0x39


def derollback(path, out_path):
    """Rewrite a netplay .slp keeping only the last copy of every re-simulated (rolled-back) frame.
    libmelee keeps a frame's first, predicted copy; slippistats/peppi and the native sim have only
    the final one. Frame events carry their frame index in payload[0:4]; an occurrence of a frame
    begins at its Frame Start event."""
    import struct
    d = Path(path).read_bytes()
    i = d.find(b"raw[$U#l")
    raw_len = struct.unpack(">I", d[i + 8:i + 12])[0]
    start = i + 12
    end = start + raw_len if raw_len else len(d)
    n = d[start + 1]
    sizes = {d[start + 2 + 3 * k]: struct.unpack(">H", d[start + 3 + 3 * k:start + 5 + 3 * k])[0]
             for k in range((n - 1) // 3)}
    head, tail, frames, cur = bytearray(d[start:start + 1 + n]), bytearray(), {}, None
    j = start + 1 + n
    while j < end:
        c = d[j]
        ev = d[j:j + 1 + sizes[c]]
        j += 1 + sizes[c]
        if c == FRAME_START:
            cur = struct.unpack(">i", ev[1:5])[0]
            frames[cur] = bytearray(ev)                    # a later copy replaces the earlier one
        elif c in (0x37, 0x38, 0x3B, 0x3C, 0x3F, 0x40, 0x41, 0x42) and cur is not None:
            frames[cur] += ev
        elif c == GAME_END or frames:
            tail += ev
        else:
            head += ev
    raw = head + b"".join(frames[k] for k in sorted(frames)) + tail
    Path(out_path).write_bytes(d[:i + 8] + struct.pack(">I", len(raw)) + bytes(raw) + d[end:])
    return out_path


# ---------------------------------------------------------------- VRs over the libmelee parse ----
def run_vrs(path):
    c = melee.Console(is_dolphin=False, path=str(path), allow_old_version=True)
    c.connect()
    h, mods, combo_moves, ports = collections.deque(maxlen=V.HISTORY), None, None, None
    trackers, seq = None, collections.defaultdict(dict)
    while True:
        gs = c.step()
        if gs is None:
            break
        if gs.menu_state not in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            continue
        if mods is None:
            ports = sorted(gs.players)
            mods = {p: [m for m, _ in V.build(",".join(V.VRS), p)] for p in ports}
            trackers = {p: S.ComboTracker(p) for p in ports}
            combo_moves = {p: [] for p in ports}
        h.append(gs)
        for p in ports:
            seq[p][int(gs.frame)] = (act_int(gs.players[p]), round(float(gs.players[p].percent), 2))
            for m in mods[p]:
                m.observe(h)
            r = trackers[p].update(h)
            if r is not None:
                combo_moves[p].append(r.n_moves)
    for p in ports:
        for m in mods[p]:
            m.finalize(h)
        r = trackers[p].flush(h)
        if r is not None:
            combo_moves[p].append(r.n_moves)
    meta = {p: {m.id: m.metadata() for m in mods[p]} for p in ports}
    chars = {p: h[-1].players[p].character.value for p in ports}
    return meta, combo_moves, chars, seq


def act_int(ps):
    a = ps.action
    return int(getattr(a, "value", a))


# ---------------------------------------------------------------- slippistats ground truth -------
def _flags(post):
    f = post.flags
    return bool(Field4.HITSTUN in f[3]), bool(Field2.HITLAG in f[1])


def truth(path):
    cc = ComboComputer(str(path))
    pls = cc.players
    stage = STAGE_NAMES.get(cc.replay.start.stage.name, cc.replay.start.stage.name)
    all_combos = {}
    for i, me in enumerate(pls):   # combo_compute appends to player.combos: compute each player once
        me.combos.clear()
        cc.reset_data()
        all_combos[me.port] = list(cc.combo_compute(player=me, opponent=pls[1 - i]))
    out = {}
    for i, me in enumerate(pls):
        opp = pls[1 - i]
        combos, opp_combos = all_combos[me.port], all_combos[opp.port]
        sc = StatsComputer(cc.replay)
        a, b = sc.players
        techs = sc.tech_compute(player=a if a.port == me.port else b, opponent=b if a.port == me.port else a)
        fr = [x.post for x in me.frames]
        ofr = [x.post for x in opp.frames]
        out[me.port + 1] = dict(
            seq={k - 123: (int(x.state), round(float(x.percent), 2)) for k, x in enumerate(fr)},
            stocks_lost=_deaths(fr), opp_stocks_lost=_deaths(ofr),
            kill_combos=sum(bool(c.did_kill) for c in combos),
            dealt=sum(max(0.0, ofr[k].percent - ofr[k - 1].percent) for k in range(1, len(ofr))),
            taken=sum(max(0.0, fr[k].percent - fr[k - 1].percent) for k in range(1, len(fr))),
            combo_moves=[len(c.moves) for c in combos],
            combos_extend=sum(len(c.moves) >= 2 and _peak_damage(c, ofr) >= 5.0 for c in combos),
            techs=len(techs), tech_punished=sum(bool(t.was_punished) for t in techs),
            recovery=_recovery(fr, stage), shield=_shield(fr),
            low_kills=_low_kills(ofr, int(opp.character), combos),
            earned_kills=_earned_kills(ofr, combos),
            neutral=_neutral(combos, opp_combos),
            neutral_direct=_neutral_direct(fr, ofr, stage),
        )
    return out


def _deaths(fr):
    """Stocks lost, plus a final death the match ended before the stock decrement registered
    (the player is in a DEAD state on the last frame: game-ending KO, or an LRAS mid-death)."""
    n = int(fr[0].stocks_remaining) - int(fr[-1].stocks_remaining)
    return n + (1 if 0 <= int(fr[-1].state) <= 10 and int(fr[-1].stocks_remaining) > 0 else 0)


def _peak_damage(c, ofr):
    s = c.start_frame + 123
    e = c.end_frame + 123 if c.end_frame else len(ofr) - 1     # unterminated: runs to the end
    pk = max((ofr[k].percent for k in range(max(s, 0), min(e + 1, len(ofr)))), default=c.start_percent)
    return pk - c.start_percent


def _recovery(fr, stage):
    """Same definition as RecoveryTracker, as an array scan over slippistats' frames."""
    b = S.OFFSTAGE_X.get(stage, 0)
    off = [p.position.y < -5 or p.position.x < -b or p.position.x > b for p in fr]
    n_ok = n_fail = 0
    k = 1
    while k < len(fr):
        if off[k] and not off[k - 1] and _flags(fr[k])[0]:
            j = k + 1
            while j < len(fr):
                st = int(fr[j].state)
                if 0 <= st <= 10:
                    n_fail += 1
                    break
                if 252 <= st <= 263 or not fr[j].is_airborne:
                    n_ok += 1
                    break
                j += 1
            k = j + 1
        else:
            k += 1
    return n_ok + n_fail, n_fail


def _shield(fr):
    """The offline tagger's scan: a shield hit (drop >= 8 vs the last 1-3 frames), walk forward to
    the first actionable shield frame (leaving the shield family first cancels it), open if the
    shield is < 30 there, classify the exit."""
    res = {"escape": 0, "hit": 0, "broken": 0}
    k = 1
    while k < len(fr):
        if int(fr[k].state) in SHIELD_FAMILY and max(
                float(fr[k - d].shield_health) - float(fr[k].shield_health)
                for d in range(1, SHIELD_LOOKBACK + 1) if k - d >= 0) >= SHIELD_HIT:
            j = k
            while j < len(fr) and int(fr[j].state) in SHIELD_FAMILY and int(fr[j].state) not in SHIELD_ACTIONABLE:
                j += 1
            if j < len(fr) and int(fr[j].state) in SHIELD_ACTIONABLE and float(fr[j].shield_health) < SMALL_SHIELD:
                e = j
                while e < len(fr) and int(fr[e].state) in SHIELD_FAMILY:
                    e += 1
                if e < len(fr):
                    x = int(fr[e].state)
                    res["broken" if x in SHIELD_BREAK_STATES else "hit" if S.is_damaged(x) else "escape"] += 1
                k = e + 1
            else:
                k = j + 1
        else:
            k += 1
    return res


def _in_combo(k, combos):
    """Frame k (0-based slippistats frame) inside one of the bot's slippistats combos (an
    unterminated combo — end_frame never set — runs to the end)."""
    for c in combos:
        s, e = c.start_frame + 123, c.end_frame + 123
        if s <= k and (k <= e or c.end_frame == 0):
            return True
    return False


def _earned_kills(ofr, combos):
    """Opponent death edges with a hit in the last OPP_HIT_MEMORY alive frames or inside a bot combo
    (slippistats' own combo windows, not the VR's tracker)."""
    decay, prev, n = 0, False, 0
    for k, p in enumerate(ofr):
        st = int(p.state)
        hs, hl = _flags(p)
        dying = 0 <= st <= 10
        if hs or hl or S.is_damaged(st):
            decay = OPP_HIT_MEMORY
        elif not dying and decay > 0:
            decay -= 1
        if dying and not prev:
            n += decay > 0 or _in_combo(k, combos)
            decay = 0
        prev = dying
    return n


def _low_kills(ofr, opp_char_css, combos):
    """Death edges of the opponent: earned (as _earned_kills) and life-peak % below the bucket."""
    buckets = _load_buckets()
    # slippistats character is the CSS id; map to libmelee's id via name
    lm = melee.Character[_css_name(opp_char_css)].value if _css_name(opp_char_css) else None
    bucket = buckets.get(lm, _FALLBACK_BUCKET)
    decay, peak, prev_dying, kills, low = 0, 0.0, False, 0, 0
    for k, p in enumerate(ofr):
        st = int(p.state)
        hs, hl = _flags(p)
        dying = 0 <= st <= 10
        if hs or hl or S.is_damaged(st):
            decay = OPP_HIT_MEMORY
        elif dying:
            pass
        elif decay > 0:
            decay -= 1
        peak = max(peak, float(p.percent))
        if dying and not prev_dying:
            kills += 1
            if (decay > 0 or _in_combo(k, combos)) and peak < bucket:
                low += 1
            decay, peak = 0, 0.0
        prev_dying = dying
    return kills, low


def _css_name(css):
    from slippistats.enums.character import CSSCharacter
    try:
        n = CSSCharacter(css).name
    except ValueError:
        return None
    return {"CAPTAIN_FALCON": "CPTFALCON", "DONKEY_KONG": "DK", "GAME_AND_WATCH": "GAMEANDWATCH",
            "YOUNG_LINK": "YLINK", "DR_MARIO": "DOC", "ICE_CLIMBERS": "POPO"}.get(n, n)


def _punish_arr(fr, stage):
    """stream.in_punish_state as an array scan over slippistats frames (combo continuation set +
    thrown), from slippistats' own fields."""
    b = S.OFFSTAGE_X.get(stage, 0)
    jy = S.JUGGLE_Y.get(stage)
    out = []
    for k, p in enumerate(fr):
        a = int(p.state)
        hs, hl = _flags(p)
        prev = int(fr[k - 1].state) if k else None
        off = p.position.y < -5 or p.position.x < -b or p.position.x > b
        wd = a == S.ESCAPE_AIR and any(k - d >= 0 and int(fr[k - d].state) == S.LAND_FALL_SPECIAL for d in (1, 2, 3))
        jug = jy is not None and p.is_airborne and p.position.y >= jy
        out.append(S.is_damaged(a) or S.is_grabbed(a) or S.is_cmd_grabbed(a) or hl or hs or S.is_shielding(a)
                   or off or (S.is_dodging(a) and not wd) or S.is_dying(a) or S.is_downed(a)
                   or S.is_teching(a) or S.is_ledge_action(a) or S.is_shield_broken(a) or jug
                   or S.is_special_fall(a) or (prev is not None and S.is_upb_lag(a, prev)) or S.is_thrown(a))
    return out


def _neutral_direct(fr, ofr, stage, K=30, W=4):
    """NeutralWinLossVR's definition as a second implementation over slippistats frames."""
    mp, op = _punish_arr(fr, stage), _punish_arr(ofr, stage)
    wins = losses = 0
    me_last = opp_last = -10 ** 9
    last_win = last_loss = None
    for t in range(len(fr)):
        oe = op[t] and not (op[t - 1] if t else False)
        me = mp[t] and not (mp[t - 1] if t else False)
        if oe and t > 0 and (t - me_last) > K and (ofr[t].percent > ofr[t - 1].percent or S.in_any_grab(int(ofr[t].state))):
            if last_loss is not None and t - last_loss <= W:
                losses -= 1
                last_loss = None
            else:
                wins += 1
                last_win = t
        if me and t > 0 and (t - opp_last) > K and (fr[t].percent > fr[t - 1].percent or S.in_any_grab(int(fr[t].state))):
            if last_win is not None and t - last_win <= W:
                wins -= 1
                last_win = None
            else:
                losses += 1
                last_loss = t
        if mp[t]:
            me_last = t
        if op[t]:
            opp_last = t
    return wins, losses


def _neutral(combos, opp_combos, K=30, W=4):
    """Approximate neutral wins/losses from combo starts: a player's combo start is a neutral win
    unless an opposing combo was active within K frames before it; starts within W frames of each
    other are a trade."""
    mine = [c.start_frame for c in combos]
    theirs = [c.start_frame for c in opp_combos]

    def clean(start, others_combos):
        return not any(c.start_frame - 1 <= start and start <= c.end_frame + K for c in others_combos)

    wins = [s for s in mine if clean(s, opp_combos)]
    losses = [s for s in theirs if clean(s, combos)]
    trades = sum(1 for s in wins if any(abs(s - t) <= W for t in losses))
    return len(wins) - trades, len(losses) - trades


# ---------------------------------------------------------------- comparison ---------------------
def lcancel_truth(path):
    """Per-port (success, miss) by tools/lcancel_analysis.py's landing scan (realized landing lag
    minus the cancelled floor, landings cut short by a hit/death dropped), split by port."""
    import numpy as np
    from lcancel_analysis import LANDING, exit_cat
    from rlvr.state.peppi_adapter import Replay
    r = Replay(path)
    out = {}
    for i, ch in enumerate(r.player_characters):
        if ch != 1:
            continue
        st = np.asarray(r._post[i]["state"]).astype(int)
        succ = miss = 0
        t = 1
        while t < len(st):
            if st[t] in LANDING and st[t - 1] != st[t]:
                j = t
                while j < len(st) and st[j] == st[t]:
                    j += 1
                ex = exit_cat(int(st[j])) if j < len(st) else "grounded"
                if ex not in ("dead", "damage"):
                    if (j - t) - CANCELLED_MIN[int(st[t])] <= 0:
                        succ += 1
                    else:
                        miss += 1
                t = j
            else:
                t += 1
        out[int(r.player_ports[i])] = (succ, miss)          # peppi_adapter ports are 1-based
    return out


def _pearson(xy):
    import math
    n = len(xy)
    if n < 2:
        return float("nan")
    mx = sum(x for x, _ in xy) / n
    my = sum(y for _, y in xy) / n
    sxy = sum((x - mx) * (y - my) for x, y in xy)
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in xy))
    sy = math.sqrt(sum((y - my) ** 2 for _, y in xy))
    return sxy / (sx * sy) if sx and sy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw_slp/fox_master_master")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="checkpoints/vr_check")
    ap.add_argument("--no-derollback", dest="derollback", action="store_false",
                    help="feed libmelee the raw netplay replay (keeps predicted rolled-back frames)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "**", "*.slp"), recursive=True))[:a.n]
    os.makedirs(a.out, exist_ok=True)
    rows = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])   # vr -> [agree, total, vr_sum, truth_sum]
    rows_clean = collections.defaultdict(lambda: [0, 0])          # same, rollback-clean players only
    rows_div = collections.Counter()
    pairs = collections.defaultdict(list)                           # vr -> [(got, want)] per player
    mism = collections.defaultdict(list)
    clean = True

    def cmp(vr, name, got, want, tol=0.0):
        r = rows[vr]
        pairs[vr].append((got, want))
        ok = abs(got - want) <= tol
        r[0] += ok; r[1] += 1; r[2] += got; r[3] += want
        if clean:
            rows_clean[vr][0] += ok; rows_clean[vr][1] += 1
        if not ok and len(mism[vr]) < 20:
            mism[vr].append(dict(file=name, got=got, want=want, rollback_clean=clean))

    for fi, f in enumerate(files):
        try:
            meta, cmoves, chars, seq = run_vrs(derollback(f, os.path.join(a.out, "_derolled.slp")) if a.derollback else f)
            tr = truth(f)
        except Exception as e:   # unparseable replay: skip, but say so
            print(f"skip {f}: {type(e).__name__}: {e}", flush=True)
            continue
        name = os.path.basename(f)
        lct = lcancel_truth(f) if any(c == 1 for c in chars.values()) else None
        for p, m in meta.items():
            t = tr.get(p)
            if t is None:
                continue
            tag = f"{name}:p{p}"
            # rollback: libmelee keeps a re-simulated frame's first (predicted) copy, slippistats the
            # corrected one; count frames where either player's state or percent differs
            div = sum(1 for q in meta for fidx, v in seq[q].items() if tr[q]["seq"].get(fidx) not in (None, v))
            clean = div == 0
            rows_div[clean] += 1
            sd = m["stock_delta"]
            cmp("stock_delta.deaths", tag, sd["deaths"], t["stocks_lost"])
            cmp("stock_delta.kills+sds", tag, sd["kills"] + sd["filtered_opp_sds"], t["opp_stocks_lost"])
            cmp("stock_delta.kills", tag, sd["kills"], t["earned_kills"])
            cmp("stock_delta.kills~kill_combos", tag, sd["kills"], t["kill_combos"])
            cmp("damage_delta.dealt", tag, m["damage_delta"]["damage_dealt"], round(t["dealt"], 1), tol=0.2)
            cmp("damage_delta.taken", tag, m["damage_delta"]["damage_taken"], round(t["taken"], 1), tol=0.2)
            cmp("combo_length.n_combos", tag, len(cmoves[p]), len(t["combo_moves"]))
            cmp("combo_length.moves_exact", tag, int(cmoves[p] == t["combo_moves"]), 1)
            cmp("combo_length.total_moves", tag, m["combo_length"]["total_moves"], sum(t["combo_moves"]))
            cmp("combo_extend.rewarded", tag, m["combo_extend"]["rewarded"], t["combos_extend"])
            cmp("tech.situations", tag, m["tech"]["tech_situations"], t["techs"])
            cmp("tech.punished", tag, m["tech"]["punished"], t["tech_punished"])
            cmp("recovery.recoveries", tag, m["recovery"]["recoveries"], t["recovery"][0])
            cmp("recovery.failed", tag, m["recovery"]["failed"], t["recovery"][1])
            for k in ("escape", "hit", "broken"):
                cmp(f"shield_escape.{k}", tag, m["shield_escape"][k], t["shield"][k])
            cmp("low_percent_kill.low_kills", tag, m["low_percent_kill"]["low_percent_kills"], t["low_kills"][1])
            cmp("neutral_win_loss.wins", tag, m["neutral_win_loss"]["neutral_wins"], t["neutral_direct"][0])
            cmp("neutral_win_loss.losses", tag, m["neutral_win_loss"]["neutral_losses"], t["neutral_direct"][1])
            cmp("neutral_win_loss.wins~approx", tag, m["neutral_win_loss"]["neutral_wins"], t["neutral"][0])
            cmp("neutral_win_loss.losses~approx", tag, m["neutral_win_loss"]["neutral_losses"], t["neutral"][1])
            if chars[p] == 1 and lct is not None and p in lct:
                lc = m["l_cancel"]
                cmp("l_cancel.success", tag, lc["success"], lct[p][0])
                cmp("l_cancel.miss", tag, lc["miss"], lct[p][1])
        print(f"[{fi + 1}/{len(files)}] {name}", flush=True)

    print(f"\nplayers: {rows_div[True]} rollback-clean, {rows_div[False]} with rollback-divergent frames")
    print(f"{'check':34s} {'agree (all)':>13s} {'pct':>6s} {'agree (clean)':>14s} {'pct':>6s} {'VR total':>10s} {'truth total':>12s}")
    for vr, (ok, n, gsum, wsum) in rows.items():
        c_ok, c_n = rows_clean[vr]
        extra = f"   r={_pearson(pairs[vr]):.2f}" if "~" in vr else ""
        print(f"{vr:34s} {ok:5d}/{n:<5d}  {100 * ok / max(n, 1):5.1f}%  {c_ok:5d}/{c_n:<5d}  "
              f"{100 * c_ok / max(c_n, 1):5.1f}% {gsum:10.1f} {wsum:12.1f}{extra}")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "mismatches_" + Path(a.dir).name + ".json")
    json.dump(mism, open(out, "w"), indent=1)
    print(f"\nmismatch examples -> {out}")


if __name__ == "__main__":
    main()
