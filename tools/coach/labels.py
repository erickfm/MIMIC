"""Plain-words labels for what a player did, from their action states.

`describe(states, my_x, opp_x)` turns the first frames of a branch into a short phrase such as
"wavedash in → shine" or "dash back → shield". Character specials are named from the decomp's
motion-state enums (special_states.json, generated from ft/kinds/*/forward.h: id 341 onward is
each character's own list) and a short table of the names players use.
"""
import json
from functools import lru_cache
from pathlib import Path

CLONE_OF = {"FALCO": "FOX", "ROY": "MARTH", "YLINK": "LINK", "DOC": "MARIO", "GANONDORF": "CPTFALCON",
            "PICHU": "PIKACHU"}
# What players call each special (N, S, Hi, Lw); characters not listed get "neutral B" etc.
NICK = {
    "FOX": ("laser", "illusion", "up B", "shine"), "FALCO": ("laser", "phantasm", "up B", "shine"),
    "MARTH": ("shield breaker", "dancing blade", "up B", "counter"),
    "ROY": ("flare blade", "dancing blade", "up B", "counter"),
    "SHEIK": ("needles", "chain", "vanish", "transform"),
    "CPTFALCON": ("falcon punch", "raptor boost", "up B", "falcon kick"),
    "GANONDORF": ("warlock punch", "gerudo dragon", "up B", "wizard's foot"),
    "JIGGLYPUFF": ("rollout", "pound", "sing", "rest"),
    "PEACH": ("toad", "peach bomber", "parasol", "turnip"),
    "LUIGI": ("fireball", "green missile", "up B", "cyclone"),
    "MARIO": ("fireball", "cape", "up B", "tornado"), "DOC": ("pill", "cape", "up B", "tornado"),
    "PIKACHU": ("thunder jolt", "skull bash", "quick attack", "thunder"),
    "PICHU": ("thunder jolt", "skull bash", "agility", "thunder"),
    "SAMUS": ("charge shot", "missile", "screw attack", "bomb"),
    "YOSHI": ("egg lay", "egg roll", "egg throw", "ground pound"),
    "LINK": ("bow", "boomerang", "spin attack", "bomb"), "YLINK": ("bow", "boomerang", "spin attack", "bomb"),
    "DK": ("giant punch", "headbutt", "spinning kong", "hand slap"),
}
SPECIAL_KEYS = (("SpecialN", 0), ("SpecialAirN", 0), ("SpecialS", 1), ("SpecialAirS", 1),
                ("SpecialHi", 2), ("SpecialAirHi", 2), ("SpecialLw", 3), ("SpecialAirLw", 3))
GENERIC = ("neutral B", "side B", "up B", "down B")

ATTACKS = {
    "NAIR": "nair", "FAIR": "forward air", "BAIR": "back air", "UAIR": "up air", "DAIR": "down air",
    "NEUTRAL_ATTACK_1": "jab", "DASH_ATTACK": "dash attack", "UPTILT": "up tilt", "DOWNTILT": "down tilt",
    "UPSMASH": "up smash", "DOWNSMASH": "down smash", "GRAB": "grab", "GRAB_RUNNING": "dash grab",
    "GRAB_WAIT": "grab", "GROUND_ATTACK_UP": "getup attack", "EDGE_ATTACK_SLOW": "ledge attack",
    "EDGE_ATTACK_QUICK": "ledge attack", "THROW_FORWARD": "forward throw", "THROW_BACK": "back throw",
    "THROW_UP": "up throw", "THROW_DOWN": "down throw",
}
PREFIX = (("FTILT", "forward tilt"), ("FSMASH", "forward smash"), ("NEUTRAL_ATTACK", "jab"))
DEFENSE = {"SHIELD_START": "shield", "SHIELD": "shield", "SHIELD_REFLECT": "powershield",
           "SPOTDODGE": "spot dodge", "ROLL_FORWARD": "roll", "ROLL_BACKWARD": "roll",
           "AIRDODGE": "air dodge", "EDGE_ROLL_SLOW": "ledge roll", "EDGE_ROLL_QUICK": "ledge roll",
           "EDGE_GETUP_SLOW": "ledge getup", "EDGE_GETUP_QUICK": "ledge getup",
           "EDGE_JUMP_1_SLOW": "ledge jump", "EDGE_JUMP_1_QUICK": "ledge jump",
           "NEUTRAL_GETUP": "get up", "GETUP_ATTACK": "getup attack",
           "TECH_MISS_UP": "missed tech", "TECH_MISS_DOWN": "missed tech", "NEUTRAL_TECH": "tech in place",
           "FORWARD_TECH": "tech roll", "BACKWARD_TECH": "tech roll", "WALL_TECH": "wall tech",
           "GROUND_ROLL_FORWARD_UP": "getup roll", "GROUND_ROLL_BACKWARD_UP": "getup roll",
           "GROUND_ROLL_FORWARD_DOWN": "getup roll", "GROUND_ROLL_BACKWARD_DOWN": "getup roll"}
MOVE = {"DASHING": "dash", "RUNNING": "run", "WALK_SLOW": "walk", "WALK_MIDDLE": "walk", "WALK_FAST": "walk",
        "CROUCH_START": "crouch", "CROUCHING": "crouch", "KNEE_BEND": "jump", "EDGE_CATCHING": "grab ledge",
        "PLATFORM_DROP": "drop through"}
DIRECTIONAL = {"dash", "run", "walk", "wavedash", "waveland", "jump", "roll", "air dodge", "tech roll",
               "ledge roll", "getup roll"}
IDLE = {"STANDING", "TURNING", "TURNING_RUN", "RUN_BRAKE", "LANDING", "FALLING", "FALLING_FORWARD",
        "FALLING_BACKWARD", "JUMPING_FORWARD", "JUMPING_BACKWARD", "JUMPING_ARIAL_FORWARD",
        "JUMPING_ARIAL_BACKWARD", "CROUCH_END", "SHIELD_RELEASE", "LANDING_SPECIAL", "DEAD_FALL"}


@lru_cache(None)
def _specials():
    raw = json.load(open(Path(__file__).with_name("special_states.json")))
    return {c: {int(k): v for k, v in t.items()} for c, t in raw.items()}


@lru_cache(None)
def _name(a):
    try:
        import melee
        return melee.Action(int(a)).name
    except Exception:
        return str(int(a))


def special_name(char, action):
    t = _specials().get(CLONE_OF.get(char, char) if char not in _specials() else char, {})
    n = t.get(int(action))
    if not n:
        return None
    if n.startswith("Appeal"):
        return "taunt"
    for key, i in SPECIAL_KEYS:
        if n.startswith(key):
            return NICK.get(char, GENERIC)[i]
    return None


def event(char, action):
    """(kind, words) for an action state, or None when it starts nothing worth naming."""
    a = int(action)
    if a >= 341:
        s = special_name(char, a)
        return ("special", s) if s else None
    n = _name(a)
    if n in ATTACKS:
        return ("attack", ATTACKS[n])
    for p, w in PREFIX:
        if n.startswith(p):
            return ("attack", w)
    if n.endswith("_LANDING") and n[:4] in ("NAIR", "FAIR", "BAIR", "UAIR", "DAIR"):
        return None
    if n in DEFENSE:
        return ("defend", DEFENSE[n])
    if n in MOVE:
        return ("move", MOVE[n])
    return None


def describe(char, states, xs, opp_xs, max_frames=60, max_events=3, with_starts=False):
    """states/xs/opp_xs: the player's action state and x and the opponent's x on each frame from
    the branch frame on. Returns a phrase like "wavedash in → shine". with_starts: also return the
    index where each move in the phrase starts."""
    starts = []
    ev, i, n = [], 0, min(len(states), max_frames)
    last, hits = None, 0
    while i < n and len(ev) < max_events:
        a = states[i]
        if i and a == states[i - 1]:
            i += 1
            continue
        start = i
        e = event(char, a)
        name = _name(a)
        if e and e[1] == "jump" and i + 1 < n:   # jump-cancelled move: the jump is not a choice of its own
            nxt = next((k for k in range(i + 1, min(n, i + 6)) if states[k] != a), None)
            if nxt is not None:
                e2 = event(char, states[nxt])
                if e2 and e2[0] in ("attack", "special") and _name(states[nxt]) in ("GRAB", "UPSMASH") or \
                        (e2 and e2[0] == "special" and e2[1] == "shine"):
                    e, i = (e2[0], "jump-cancel " + e2[1]), nxt
        # wavedash / waveland: a jump or air dodge that lands in LANDING_SPECIAL soon after
        if e and e[1] in ("jump", "air dodge") and not e[1].startswith("jump-cancel"):
            j = next((k for k in range(i + 1, min(n, i + 14)) if _name(states[k]) == "LANDING_SPECIAL"), None)
            if j is not None:
                e = ("move", "wavedash" if e[1] == "jump" else "waveland")
                i = j
        if e and e[1] in DIRECTIONAL:
            end = min(len(xs) - 1, i + 8)
            dx = xs[end] - xs[start]
            toward = (opp_xs[start] - xs[start]) * dx > 0
            word = e[1] if abs(dx) < 1.0 else f"{e[1]} {'in' if toward else 'back'}"
            e = (e[0], word)
        if e and last and e[1].split()[0] == last.split()[0] and e[0] == "move":
            ev[-1] = last = e[1] if len(e[1]) > len(last) else last
        elif e and e[1] != last:
            ev.append(e[1])
            starts.append(start)
            last = e[1]
            if e[0] in ("attack", "special"):
                hits += 1
                if hits >= 2:
                    break
        i += 1
        if name in IDLE:
            continue
    label = " → ".join(ev) if ev else "wait"
    return (label, starts) if with_starts else label
