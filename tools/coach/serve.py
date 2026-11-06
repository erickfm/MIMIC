#!/usr/bin/env python3
"""The coach app's server: serves the web app (tools/coach/web) and its API, runs branches.

  python3 tools/coach/serve.py            # then open http://localhost:8765

The browser draws everything (the replay viewer reads the player's own game files, which never
leave the browser). This server reads the replay folder, finds decision points, and runs the
native sim + models to branch from a game (tools/coach/branch.py). Nothing here sends game
files anywhere.
"""
import argparse, collections, hashlib, json, mimetypes, os, queue, re, sys, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WEB = Path(__file__).with_name("web")
HOME = Path(os.environ.get("COACH_HOME", Path.home() / ".cache" / "mimic-coach"))
CONFIG = HOME / "config.json"
STAGES = {2: "Fountain of Dreams", 3: "Pokémon Stadium", 8: "Yoshi's Story", 28: "Dream Land",
          31: "Battlefield", 32: "Final Destination"}
EXT_NAME = {0: "Captain Falcon", 1: "Donkey Kong", 2: "Fox", 3: "Mr. Game & Watch", 4: "Kirby", 5: "Bowser",
            6: "Link", 7: "Luigi", 8: "Mario", 9: "Marth", 10: "Mewtwo", 11: "Ness", 12: "Peach", 13: "Pikachu",
            14: "Ice Climbers", 15: "Jigglypuff", 16: "Samus", 17: "Yoshi", 18: "Zelda", 19: "Sheik", 20: "Falco",
            21: "Young Link", 22: "Dr. Mario", 23: "Roy", 24: "Pichu", 25: "Ganondorf"}
# The policy that plays each character in branches: BC on master replays ("what a master would do").
POLICY = {"FALCO": "hf_eval/falco", "FOX": "runs/fox-M-bc2-mlr01-s1/policy", "LUIGI": "hf_eval/luigi",
          "CPTFALCON": "hf_eval/cptfalcon", "MARTH": "hf_eval/marth", "SHEIK": "hf_eval/sheik",
          "JIGGLYPUFF": "hf_eval/puff", "PEACH": "hf_eval/peach", "SAMUS": "hf_eval/samus",
          "PIKACHU": "hf_eval/pikachu", "YOSHI": "hf_eval/yoshi", "GANONDORF": "hf_eval/ganondorf",
          "DOC": "hf_eval/doc", "MARIO": "hf_eval/mario", "LINK": "hf_eval/link", "YLINK": "hf_eval/ylink",
          "ROY": "hf_eval/roy", "DK": "hf_eval/dk", "BOWSER": "hf_eval/bowser", "NESS": "hf_eval/ness",
          "MEWTWO": "hf_eval/mewtwo", "GAMEANDWATCH": "hf_eval/gameandwatch"}
HF_REPO = "erickfm/MIMIC"
HF_DIR = {"FOX": "fox"}   # otherwise the local hf_eval/<name> folder's name


def policy_dir(char):
    """The policy folder for a character; downloaded from the Hugging Face model repo the first
    time it's needed when it isn't on this machine."""
    d = ROOT / POLICY[char]
    if (d / "model.pt").is_file():
        return str(d)
    name = HF_DIR.get(char, Path(POLICY[char]).name)
    cache = HOME / "models"
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, allow_patterns=[f"{name}/*"], local_dir=str(cache))
    return str(cache / name)


# run-name character keys (tools/run.py CHARS), for finding a character's RL runs
RUN_CHAR = {"FOX": "fox", "FALCO": "falco", "MARTH": "marth", "SHEIK": "sheik", "ZELDA": "zelda",
            "CPTFALCON": "cptfalcon", "JIGGLYPUFF": "puff", "PEACH": "peach", "LUIGI": "luigi",
            "PIKACHU": "pikachu", "SAMUS": "samus", "YOSHI": "yoshi", "GANONDORF": "ganondorf", "DOC": "doc",
            "MARIO": "mario", "LINK": "link", "YLINK": "ylink", "ROY": "roy", "DK": "dk", "BOWSER": "bowser",
            "NESS": "ness", "MEWTWO": "mewtwo", "GAMEANDWATCH": "gameandwatch"}


def live_policy(char):
    """The bot for live play, on the CPU: the character's S RL model (fast enough for 60 fps:
    6.2 ms per frame on 4 threads, measured), else its usual policy (M: ~16 ms, the game then runs
    slower). Returns (dir, kind)."""
    key = RUN_CHAR.get(char)
    runs = ROOT / "runs"
    if key:
        cands = sorted(runs.glob(f"{key}-S-*.rl*-s*/policy/model.pt"), key=lambda q: q.stat().st_mtime)
        if key == "fox" and (runs / "fox-S-bc1-s1.rl1-s0/policy/model.pt").exists():
            cands.append(runs / "fox-S-bc1-s1.rl1-s0/policy/model.pt")   # SF7u100: beat fox-master
        cands = [c for c in cands if (c.parent.parent / "run.json").exists()
                 and json.loads((c.parent.parent / "run.json").read_text()).get("status") in ("done", "legacy")]
        if cands:
            return str(cands[-1].parent), "S RL"
    return policy_dir(char), "default"


COLORS = ["#4cc9f0", "#f72585", "#b5e48c", "#ffb703", "#9d4edd", "#ff6b35", "#06d6a0", "#e0e0e0"]


# ── the Melee ISO: the one Slippi Launcher uses ─────────────────────────────────────────────────

def slippi_launcher_settings():
    for d in (Path.home() / ".config" / "Slippi Launcher",
              Path(os.environ.get("APPDATA", "")) / "Slippi Launcher",
              Path.home() / "Library" / "Application Support" / "Slippi Launcher"):
        f = d / "Settings"
        if f.is_file():
            try:
                d_ = json.loads(f.read_text())
                return d_.get("settings", d_)
            except Exception:
                pass
    return {}


class Iso:
    """The viewer's game files read straight out of the ISO (nothing is extracted or copied)."""

    def __init__(self, path):
        import struct
        self.path = str(path)
        with open(self.path, "rb") as f:
            h = f.read(0x440)
            if h[:6] != b"GALE01":
                raise ValueError("not an NTSC Melee disc")
            if h[7] != 2:
                raise ValueError(f"Melee version 1.0{h[7]}, the coach needs 1.02")
            dol, fst, fsz = struct.unpack(">III", h[0x420:0x42C])
            f.seek(fst)
            d = f.read(fsz)
            f.seek(dol)
            dh = f.read(0xE0)
        n = struct.unpack(">I", d[8:12])[0]
        strs = 12 * n
        self.files = {}
        for i in range(1, n):
            w, off, ln = struct.unpack(">III", d[12 * i:12 * i + 12])
            if w >> 24:
                continue
            name = d[strs + (w & 0xFFFFFF):].split(b"\0", 1)[0].decode()
            self.files[name] = (off, ln)
        end = max(struct.unpack(">I", dh[4 * k:4 * k + 4])[0] + struct.unpack(">I", dh[0x90 + 4 * k:0x94 + 4 * k])[0]
                  for k in range(18))
        self.files["main.dol"] = self.files["Start.dol"] = (dol, end)
        self.mtime = os.stat(self.path).st_mtime

    def read(self, name):
        off, ln = self.files[name]
        with open(self.path, "rb") as f:
            f.seek(off)
            return f.read(ln)


def find_iso(cfg):
    """(Iso or None, where it came from, path tried)."""
    tries = [(cfg.get("iso"), "your setting"), (slippi_launcher_settings().get("isoPath"), "Slippi Launcher"),
             (str(ROOT / "melee.iso"), "this folder")]
    last = None
    for p, src in tries:
        if p and Path(p).is_file():
            try:
                return Iso(p), src, p
            except Exception as e:
                last = (src, p, str(e))
    return None, last[0] if last else None, last[1] if last else None


def default_replay_dir():
    for p in (Path.home() / "Slippi", Path.home() / "Documents" / "Slippi"):
        if p.is_dir():
            return str(p)
    return str(Path.home() / "Slippi")


def load_config():
    try:
        c = json.loads(CONFIG.read_text())
    except Exception:
        c = {}
    c.setdefault("replay_dir", default_replay_dir())
    c.setdefault("me", None)
    return c


def save_config(c):
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(c, indent=1))


def norm_code(s):
    return (s or "").replace("＃", "#").strip() or None


def gid(path):
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


# ── library ──────────────────────────────────────────────────────────────────────────────────

class Library:
    def __init__(self):
        self.lock = threading.Lock()
        self.games, self.by_id, self.scanning, self.scanned_dir = [], {}, False, None
        self.cache_path = HOME / "library.json"
        try:
            self.cache = json.loads(self.cache_path.read_text())
        except Exception:
            self.cache = {}

    def header(self, path):
        from peppi_py import read_slippi
        from sim_play import EXT_TO_MELEE
        st = os.stat(path)
        key = f"{path}|{st.st_mtime_ns}|{st.st_size}"
        if key in self.cache:
            return self.cache[key]
        try:
            g = read_slippi(str(path), skip_frames=True)
        except Exception as e:
            h = dict(error=str(e)[:200])
        else:
            md = g.metadata or {}
            players = []
            for i, pl in enumerate(g.start.players):
                if not pl:
                    continue
                players.append(dict(port=i, char=EXT_TO_MELEE.get(pl.character), char_name=EXT_NAME.get(pl.character, "?"),
                                    code=norm_code(pl.netplay.code) if pl.netplay else None,
                                    name=pl.netplay.name if pl.netplay else None))
            place = {}
            if g.end and g.end.players:
                place = {int(p.port): p.placement for p in g.end.players}
            ver = tuple(g.start.slippi.version[:3])
            h = dict(start=md.get("startAt"), frames=md.get("lastFrame"), stage=STAGES.get(g.start.stage, f"stage {g.start.stage}"),
                     players=players, placement=place, version=".".join(map(str, ver)),
                     end=str(g.end.method.name) if g.end and g.end.method is not None else None)
        self.cache[key] = h
        return h

    def scan(self, folder):
        with self.lock:
            if self.scanning:
                return
            self.scanning = True
        try:
            files = sorted(Path(folder).rglob("*.slp"), key=lambda p: p.name, reverse=True)
            games = []
            for p in files:
                h = self.header(p)
                if "error" in h:
                    continue
                games.append(dict(id=gid(p), path=str(p), **h))
            games.sort(key=lambda g: g.get("start") or "", reverse=True)
            with self.lock:
                self.games, self.by_id = games, {g["id"]: g for g in games}
                self.scanned_dir = folder
            HOME.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        finally:
            self.scanning = False

    def codes(self):
        c = collections.Counter(p["code"] for g in self.games for p in g["players"] if p.get("code"))
        return c.most_common(12)


def support(game, me_code):
    """Why a game can't be analysed, or None."""
    pl = game["players"]
    if len(pl) != 2:
        return "Only 1v1 games can be analysed."
    mine = [p for p in pl if p.get("code") == me_code]
    if not mine:
        return "You aren't in this game."
    ver = tuple(int(x) for x in game["version"].split("."))
    if ver < (3, 18, 0):
        return "Recorded with an older Slippi version that the simulator can't reproduce exactly."
    for p in pl:
        if p["char"] in ("POPO", "NANA"):
            return "Ice Climbers aren't supported yet."
        if p["char"] not in POLICY:
            return f"No model for {p['char_name']} yet."
    return None


def summary(game, me_code):
    pl = game["players"]
    me = next((p for p in pl if p.get("code") == me_code), None)
    opp = next((p for p in pl if p is not me), None) if me else None
    res = None
    if me and game.get("placement"):
        pm = game["placement"].get(str(me["port"]), game["placement"].get(me["port"]))
        if pm is not None:
            res = "win" if pm == 0 else "loss"
    return dict(id=game["id"], start=game.get("start"), frames=game.get("frames"), stage=game["stage"],
                me=me, opp=opp, result=res, players=pl, unsupported=support(game, me_code),
                reviewable=bool(me and value_path(me["char"])),
                file=os.path.basename(game["path"]))


# ── game analysis ────────────────────────────────────────────────────────────────────────────

_game_cache = {}


def analyse(game, me_code):
    """Decision points of the player and both players' percent/stocks per frame."""
    if game["id"] in _game_cache:
        return _game_cache[game["id"]]
    import decision_points as D
    from labels import describe
    me = next(p for p in game["players"] if p.get("code") == me_code)
    opp = next(p for p in game["players"] if p is not me)
    g, frames, st, f1, f4, hl, age, keep = D.load(game["path"], me["port"])
    ok, _ = D.can_act(st, f1, f4, hl, age)
    po_me = g.frames.ports[me["port"]].leader.post
    po_op = g.frames.ports[opp["port"]].leader.post
    x = np.asarray(po_me.position.x, np.float32)[keep]
    ox = np.asarray(po_op.position.x, np.float32)[keep]
    pct = np.asarray(po_me.percent, np.float32)[keep]
    opct = np.asarray(po_op.percent, np.float32)[keep]
    stk = np.asarray(po_me.stocks, np.int32)[keep]
    ostk = np.asarray(po_op.stocks, np.int32)[keep]
    pts = []
    for w in D.windows(frames, st, ok):
        if w["ended_by"] != "player" or w["frame"] < 0:
            continue
        i = w["idx"]
        pts.append(dict(frame=w["frame"], wait=w["length"], came=w["came_from"],
                        did=describe(me["char"], st[i:i + 60].tolist(), x[i:i + 60].tolist(), ox[i:i + 60].tolist()),
                        x=float(x[i]), ox=float(ox[i]), pct=float(pct[i]), opct=float(opct[i])))
    out = dict(id=game["id"], first=int(frames[0]), last=int(frames[-1]),
               me=me, opp=opp, points=pts,
               series=dict(frame0=int(frames[0]), me_pct=np.round(pct, 1).tolist(), opp_pct=np.round(opct, 1).tolist(),
                           me_stocks=stk.tolist(), opp_stocks=ostk.tolist()),
               _arrays=dict(frames=frames, st=st, x=x, ox=ox, pct=pct, opct=opct, stk=stk, ostk=ostk))
    _game_cache[game["id"]] = out
    return out


def real_outcome(an, frame, horizon):
    """What actually happened in the game over the same window: damage dealt/taken, stocks."""
    a = an["_arrays"]
    i = int(np.searchsorted(a["frames"], frame))
    j = min(len(a["frames"]) - 1, i + horizon)
    d = lambda v: float(np.clip(np.diff(v[i:j + 1]), 0, None).sum())
    s = lambda v: int(np.clip(-np.diff(v[i:j + 1]), 0, None).sum())
    from labels import describe
    did = describe(an["me"]["char"], a["st"][i:i + 60].tolist(), a["x"][i:i + 60].tolist(), a["ox"][i:i + 60].tolist())
    return dict(label=did, dealt=round(d(a["opct"]), 1), taken=round(d(a["pct"]), 1),
                stocks_taken=s(a["ostk"]), stocks_lost=s(a["stk"]))


# ── branch jobs ──────────────────────────────────────────────────────────────────────────────

def value_path(char):
    p = ROOT / "data" / "coach_rank" / f"value_{char.lower()}.pt"
    return p if p.is_file() else None


WATCH = 180   # frames a try runs after the decision, for watching (3 s)


def stretch_of(wait):
    """Frames a decision is judged over: from the moment you could act to 20 frames after you
    committed, at most 60 (as in tools/coach/outcome_test.py)."""
    return int(min(60, wait + 20))


def reaction(char, st, x, ox):
    """The opponent's line from a branch, without the action they were already in at the branch
    frame (that isn't a reaction to your choice)."""
    from labels import describe
    k = next((i for i in range(1, len(st)) if st[i] != st[0]), 0)
    return describe(char, st[k:], x[k:], ox[k:]) if k else describe(char, st, x, ox)


class Jobs:
    """One GPU worker thread: branch requests and game reviews, in order."""

    def __init__(self, device):
        self.device = device
        self.q = queue.PriorityQueue()   # (priority, order, ...): the player's own requests first
        self._n = 0
        self.jobs = {}
        self.brancher = None
        self.iso_path = None
        self.values = {}
        self.reviews = {}
        threading.Thread(target=self.loop, daemon=True).start()

    def submit(self, kind, game, an, **kw):
        jid = uuid.uuid4().hex[:12]
        self.jobs[jid] = dict(id=jid, kind=kind, state="queued", progress=0.0, game=game["id"], started=time.time(), **kw)
        self._n += 1
        self.q.put((1 if kw.get("review") else 0, self._n, jid, game, an))
        return self.jobs[jid]

    def value(self, char):
        from value import Value
        if char not in self.values:
            vp = value_path(char)
            self.values[char] = Value(str(vp), self.device) if vp else None
        return self.values[char]

    def loop(self):
        from branch import Brancher
        while True:
            _, _, jid, game, an = self.q.get()
            job = self.jobs[jid]
            try:
                job["state"] = "loading" if self.brancher is None else "running"
                if self.brancher is None:
                    self.brancher = Brancher(self.device, iso=self.iso_path)
                job["state"] = "running"
                if job["kind"] == "review":
                    self.do_review(job, game, an)
                elif job["kind"] == "lab":
                    self.do_lab(jid, job, game, an)
                elif job["kind"] == "patterns":
                    self.do_patterns(jid, job, game, an)
                elif job["kind"] == "earlier":
                    self.do_earlier(jid, job, game, an)
                elif job["kind"] == "test":
                    self.do_test(jid, job, game, an)
                else:
                    self.do_branch(jid, job, game, an)
            except Exception as e:
                traceback.print_exc()
                job.update(state="error", error=str(e)[:300])

    def do_review(self, job, game, an, compare=True):
        """V along the game, and each decision's cost: what your choice did to the expected 5-second
        score, measured over its stretch (score during it + V after it - V before it)."""
        from value import game_value, score
        me = an["me"]
        V = self.value(me["char"])
        if V is None:
            raise RuntimeError(f"No review model for {me['char_name']} yet.")
        pdir = policy_dir(me["char"])
        m = self.brancher.model(pdir)[0]
        with self.brancher.lock:
            frames, out = game_value(game["path"], me["port"], pdir, m, V, self.device)
        sc = score(out)
        a = an["_arrays"]
        fr_idx = {int(f): i for i, f in enumerate(a["frames"])}
        def v_at(f):
            i = int(np.clip(np.searchsorted(frames, f), 0, len(frames) - 1))
            return float(sc[i])
        def realized(f0, f1):
            i, j = fr_idx.get(f0), fr_idx.get(f1)
            if i is None or j is None:
                return None
            d = lambda v: float(np.clip(np.diff(v[i:j + 1]), 0, None).sum())
            st = lambda v: float(np.clip(-np.diff(v[i:j + 1]), 0, None).sum())
            return d(a["opct"]) - d(a["pct"]) + 40 * (st(a["ostk"]) - st(a["stk"]))
        pts = []
        for p in an["points"]:
            s_ = stretch_of(p["wait"])
            r = realized(p["frame"], p["frame"] + s_)
            if r is None or p["frame"] < frames[0]:
                continue
            pts.append(dict(frame=p["frame"], did=p["did"], stretch=s_,
                            cost=round(r + v_at(p["frame"] + s_) - v_at(p["frame"]), 2)))
        costs = np.array([p["cost"] for p in pts]) if pts else np.zeros(0)
        lo = np.sort(costs)[:12] if len(costs) else []
        hi = np.sort(costs)[-8:] if len(costs) else []
        for p in pts:
            p["tag"] = ("costly" if p["cost"] <= min(-8.0, lo[-1] if len(lo) else -8) else
                        "great" if p["cost"] >= max(8.0, hi[0] if len(hi) else 8) else None)
        step = max(1, len(frames) // 1500)
        job.update(state="done", progress=1.0, eval=[[int(f), round(float(v), 1)] for f, v in zip(frames[::step], sc[::step])],
                   points=pts, r2=float(V.r2) if V.r2 is not None else None)
        self.reviews[game["id"]] = dict(job, eval_full=[[int(f), float(v)] for f, v in zip(frames, sc)])
        job["compare"] = {}
        for p in (pts if compare else []):
            if p["tag"]:
                b = self.submit("branch", game, an, frame=p["frame"], k=16, horizon=-1, review=job["id"])
                job["compare"][str(p["frame"])] = dict(job=b["id"], state="queued")
        self.reviews[game["id"]]["compare"] = job["compare"]

    def do_branch(self, jid, job, game, an):
        from labels import describe
        from value import score
        frame, k, horizon = job["frame"], job["k"], job["horizon"]
        me, opp = an["me"], an["opp"]
        out_dir = HOME / "branches" / jid
        out_dir.mkdir(parents=True, exist_ok=True)
        pol = {me["port"]: policy_dir(me["char"]), opp["port"]: policy_dir(opp["char"])}
        V = self.value(me["char"])
        pt = next((p for p in an["points"] if p["frame"] <= frame < p["frame"] + p["wait"] + 30), None)
        s_ = stretch_of(pt["wait"] if pt else 40)
        if horizon <= 0:         # auto: the decision (its stretch) and 2 s after it to watch;
            horizon = s_ + (WATCH if horizon == 0 else 1)   # -1: only what the score needs (review)
            job["horizon"] = horizon
        r = self.brancher.run(game["path"], frame, k, horizon, pol, out_dir=out_dir,
                              progress=lambda f: job.__setitem__("progress", round(f, 3)),
                              value_at=(me["port"], [s_]) if V else None)
        q = score(V(r["value_feats"][s_])) if V and s_ in r["value_feats"] else None
        options = []
        mp, op = me["port"], opp["port"]
        for i, ro in enumerate(r["rollouts"]):
            fr = ro["frames"]
            label = describe(me["char"], [f[mp][3] for f in fr], [f[mp][0] for f in fr], [f[op][0] for f in fr])
            oc = ro["outcome"]
            o = dict(i=i, label=label, group=label.split(" → ")[0],
                     color=COLORS[i % len(COLORS)], url=f"/api/branch/{jid}/{i}.slp",
                     opp_label=reaction(opp["char"], [f[op][3] for f in fr], [f[op][0] for f in fr], [f[mp][0] for f in fr]),
                     dealt=oc[op]["damage_taken"], taken=oc[mp]["damage_taken"],
                     stocks_taken=oc[op]["stocks_lost"], stocks_lost=oc[mp]["stocks_lost"])
            if q is not None and len(fr) > s_:
                seg = fr[:s_ + 1]
                d = lambda pp: float(np.clip(np.diff([f[pp][5] for f in seg]), 0, None).sum())
                st = lambda pp: float(np.clip(-np.diff([f[pp][6] for f in seg]), 0, None).sum())
                o["expected"] = round(d(op) - d(mp) + 40 * (st(op) - st(mp)) + float(q[i]), 1)
            options.append(o)
        you = real_outcome(an, frame, horizon)
        rv = self.reviews.get(game["id"])
        if q is not None and rv:
            # your expected score, judged the same way as the model's tries: what happened over the
            # stretch + V after it (V along your game, from the review)
            seg = real_outcome(an, frame, s_)
            you["expected"] = round(seg["dealt"] - seg["taken"] + 40 * (seg["stocks_taken"] - seg["stocks_lost"])
                                    + _v_near(rv["eval_full"], frame + s_), 1)
        job.update(state="done", progress=1.0, exact=r["exact"], seconds=r["seconds"], stretch=s_,
                   options=options, you=you, valued=q is not None)
        if job.get("review") and rv and "compare" in rv:
            c = rv["compare"].get(str(frame))
            if c is not None:
                exp = [o["expected"] for o in options if "expected" in o]
                c.update(state="done", model=round(float(np.mean(exp)), 1) if exp else None,
                         you=you.get("expected"),
                         best=collections.Counter(o["group"] for o in options).most_common(1)[0][0])
        (out_dir / "result.json").write_text(json.dumps(job))


def _lab(self, jid, job, game, an):
    """The lab: put the fighters where the player set them after frame `at` of a real game (stage,
    characters and everything else come from that game), then the models play it out k ways.
    With `from_job`/`from_try`/`frame`: continue from that frame of an earlier lab try."""
    from labels import describe
    me, opp = an["me"], an["opp"]
    places = {int(p): tuple(map(float, v)) for p, v in job["place"].items()}
    at = int(job["at"])
    pads_slp, F = None, at + 1
    if job.get("from_job"):
        prev = self.jobs[job["from_job"]]
        pads_slp = prev["options"][int(job["from_try"])]["path"]
        F = max(at + 1, int(job["frame"]))
    out_dir = HOME / "branches" / jid
    pol = {me["port"]: policy_dir(me["char"]), opp["port"]: policy_dir(opp["char"])}
    r = self.brancher.run(game["path"], F, job["k"], job["horizon"], pol, out_dir=out_dir,
                          progress=lambda f: job.__setitem__("progress", round(f, 3)),
                          place=(at, places), pads_slp=pads_slp)
    mp, op = me["port"], opp["port"]
    options = []
    for i, ro in enumerate(r["rollouts"]):
        fr = ro["frames"]
        oc = ro["outcome"]
        options.append(dict(i=i, color=COLORS[i % len(COLORS)], url=f"/api/branch/{jid}/{i}.slp", path=ro["slp"],
                            label=describe(me["char"], [f[mp][3] for f in fr], [f[mp][0] for f in fr], [f[op][0] for f in fr]),
                            opp_label=describe(opp["char"], [f[op][3] for f in fr], [f[op][0] for f in fr], [f[mp][0] for f in fr]),
                            dealt=oc[op]["damage_taken"], taken=oc[mp]["damage_taken"],
                            stocks_taken=oc[op]["stocks_lost"], stocks_lost=oc[mp]["stocks_lost"]))
    job.update(state="done", progress=1.0, frame=F, exact=r["exact"], options=options, seconds=r["seconds"])


Jobs.do_lab = _lab

def _earlier(self, jid, job, game, an):
    """Could the spot at `frame` have been avoided? From up to 3 of your decisions in the 3 s before
    it, run the master and score every line at one common end point T (the end of this moment's
    stretch): what happened from that decision to T, plus V at T. Your line is scored the same way
    on the real game. Also counts how many master tries were in about the same spot at `frame`."""
    from labels import describe
    from value import score
    D = int(job["frame"])
    me, opp = an["me"], an["opp"]
    mp, op = me["port"], opp["port"]
    V = self.value(me["char"])
    if V is None:
        raise RuntimeError(f"No review model for {me['char_name']} yet.")
    if game["id"] not in self.reviews:
        self.do_review(dict(), game, an, compare=False)
    rv = self.reviews[game["id"]]
    pt = next((p for p in an["points"] if p["frame"] <= D < p["frame"] + p["wait"] + 30), None)
    T = D + stretch_of(pt["wait"] if pt else 40)
    # earlier decisions: the 3 latest that start 15+ frames before this one and within 3 s
    lo = D - 180
    cands = [p for p in an["points"] if lo <= p["frame"] <= D - 15 and (pt is None or p["frame"] < pt["frame"])]
    picks = []
    for p in reversed(cands):
        if not picks or picks[-1]["frame"] - p["frame"] >= 15:
            picks.append(p)
        if len(picks) == 3:
            break
    if not picks:
        raise RuntimeError("You didn't have another decision in the 3 seconds before this moment.")
    a = an["_arrays"]
    fi = {int(f): i for i, f in enumerate(a["frames"])}
    iD = fi.get(D)
    def realized_real(f0, f1):
        i, j = fi.get(f0), fi.get(f1)
        d = lambda v: float(np.clip(np.diff(v[i:j + 1]), 0, None).sum())
        st = lambda v: float(np.clip(-np.diff(v[i:j + 1]), 0, None).sum())
        return d(a["opct"]) - d(a["pct"]) + 40 * (st(a["ostk"]) - st(a["stk"]))
    v_T_real = _v_near(rv["eval_full"], T)
    pol = {mp: policy_dir(me["char"]), op: policy_dir(opp["char"])}
    rows = []
    for n, p in enumerate(picks):
        E = int(p["frame"])
        steps = T - E
        sub = f"{jid}_{n}"
        out_dir = HOME / "branches" / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        r = self.brancher.run(game["path"], E, job["k"], steps + 1, pol, out_dir=out_dir,
                              progress=lambda f, n=n: job.__setitem__("progress", round((n + f) / len(picks), 3)),
                              value_at=(mp, [steps]))
        if steps not in r["value_feats"]:
            continue
        q = score(V(r["value_feats"][steps]))
        tries = []
        for i, ro in enumerate(r["rollouts"]):
            fr = ro["frames"]
            if len(fr) <= steps:
                continue
            seg = fr[:steps + 1]
            d = lambda pp: float(np.clip(np.diff([f[pp][5] for f in seg]), 0, None).sum())
            st = lambda pp: float(np.clip(-np.diff([f[pp][6] for f in seg]), 0, None).sum())
            k = D - E                                  # this try at the moment you were looking at
            same = None
            if iD is not None and len(fr) > k:
                f = fr[k]
                same = bool(abs(f[mp][0] - a["x"][iD]) < 15 and abs(f[op][0] - a["ox"][iD]) < 15
                            and abs(f[mp][5] - a["pct"][iD]) < 5)
            tries.append(dict(i=i, color=COLORS[i % len(COLORS)], url=f"/api/branch/{sub}/{i}.slp",
                              label=describe(me["char"], [f[mp][3] for f in fr], [f[mp][0] for f in fr], [f[op][0] for f in fr]),
                              opp_label=reaction(opp["char"], [f[op][3] for f in fr], [f[op][0] for f in fr], [f[mp][0] for f in fr]),
                              expected=round(d(op) - d(mp) + 40 * (st(op) - st(mp)) + float(q[i]), 1), same=same))
        if not tries:
            continue
        you = round(realized_real(E, T) + v_T_real, 1)
        ex = np.array([t["expected"] for t in tries])
        rows.append(dict(frame=E, did=p["did"], you=you, master=round(float(ex.mean()), 1),
                         se=round(float(ex.std(ddof=1) / np.sqrt(len(ex))) if len(ex) > 1 else 0.0, 1),
                         same=sum(1 for t in tries if t["same"]), n=len(tries), exact=r["exact"], tries=tries))
    if not rows:
        raise RuntimeError("The simulator couldn't play out the earlier moments in this game.")
    job.update(state="done", progress=1.0, frame=D, end=T, rows=rows)


Jobs.do_earlier = _earlier


def _test(self, jid, job, game, an):
    """Test one option against many opponent responses: from the branch frame, your side presses
    exactly what the chosen try (or you, in the real game) pressed until that first move is under
    way (its first action change + 8 frames, at most the decision's stretch); the master takes over
    from there, and the opponent is sampled fresh in every try. Each try is scored like the others
    (what happened over the stretch + V after it); the spread is the opponent's choices and DI."""
    from labels import describe
    from value import score
    from peppi_py import read_slippi
    F = int(job["frame"])
    me, opp = an["me"], an["opp"]
    mp, op = me["port"], opp["port"]
    V = self.value(me["char"])
    if V is None:
        raise RuntimeError(f"No review model for {me['char_name']} yet.")
    pt = next((p for p in an["points"] if p["frame"] <= F < p["frame"] + p["wait"] + 30), None)
    s_ = stretch_of(pt["wait"] if pt else 40)
    if job.get("source") == "you":
        src = game["path"]
        a = an["_arrays"]
        i0 = int(np.searchsorted(a["frames"], F))
        st, x, ox = (a[k][i0:i0 + 61].tolist() for k in ("st", "x", "ox"))
    else:
        src = next((HOME / "branches" / str(job["from_job"])).glob(f"branch_*_{int(job['from_try'])}.slp"), None)
        if src is None:
            raise RuntimeError("That try is gone; ask what a master would do again.")
        g = read_slippi(str(src))
        keep = np.asarray(g.frames.id) >= F
        pm, po = g.frames.ports[mp].leader.post, g.frames.ports[op].leader.post
        st = np.asarray(pm.state)[keep][:61].tolist()
        x, ox = (np.asarray(v_.position.x)[keep][:61].tolist() for v_ in (pm, po))
    # press exactly what the source pressed until its first move (as the label names it) is under
    # way: 10 frames into it, enough for a jump squat before a wavedash's air dodge. No move: the stretch.
    move, starts = describe(me["char"], st, x, ox, with_starts=True)
    n = min(60, starts[0] + 10) if starts else s_
    out_dir = HOME / "branches" / jid
    out_dir.mkdir(parents=True, exist_ok=True)
    pol = {mp: policy_dir(me["char"]), op: policy_dir(opp["char"])}
    horizon = s_ + WATCH
    r = self.brancher.run(game["path"], F, job["k"], horizon, pol, out_dir=out_dir,
                          progress=lambda f: job.__setitem__("progress", round(f, 3)),
                          value_at=(mp, [s_]), force=(mp, str(src), n))
    q = score(V(r["value_feats"][s_]))
    tries = []
    for i, ro in enumerate(r["rollouts"]):
        fr = ro["frames"]
        if len(fr) <= s_:
            continue
        seg = fr[:s_ + 1]
        d = lambda pp: float(np.clip(np.diff([f[pp][5] for f in seg]), 0, None).sum())
        sk = lambda pp: float(np.clip(-np.diff([f[pp][6] for f in seg]), 0, None).sum())
        opp_label = reaction(opp["char"], [f[op][3] for f in fr], [f[op][0] for f in fr], [f[mp][0] for f in fr])
        tries.append(dict(i=i, url=f"/api/branch/{jid}/{i}.slp",
                          label=describe(me["char"], [f[mp][3] for f in fr], [f[mp][0] for f in fr], [f[op][0] for f in fr]),
                          opp_label=opp_label, opp_first=opp_label.split(" → ")[0],
                          expected=round(d(op) - d(mp) + 40 * (sk(op) - sk(mp)) + float(q[i]), 1)))
    if len(tries) < 2:
        raise RuntimeError("The simulator couldn't play this option out here.")
    ex = np.array([t["expected"] for t in tries])
    by = collections.defaultdict(list)
    for t in tries:
        by[t["opp_first"]].append(t["expected"])
    responses = sorted(([k_, len(v), round(float(np.mean(v)), 1)] for k_, v in by.items()), key=lambda x: -x[1])
    first = move.split(" → ")[0]
    job.update(state="done", progress=1.0, frame=F, stretch=s_, forced=n, move=first,
               same=sum(1 for t in tries if t["label"].split(" → ")[0] == first), horizon=horizon, exact=r["exact"],
               mean=round(float(ex.mean()), 1), se=round(float(ex.std(ddof=1) / np.sqrt(len(ex))), 1),
               sd=round(float(ex.std(ddof=1)), 1), p10=round(float(np.percentile(ex, 10)), 1),
               p50=round(float(np.percentile(ex, 50)), 1), p90=round(float(np.percentile(ex, 90)), 1),
               responses=responses, tries=tries)


Jobs.do_test = _test


def game_state(game, me_code, frame):
    """Both players' position, facing and percent at a frame of a game (the lab's starting point)."""
    from peppi_py import read_slippi
    g = read_slippi(game["path"])
    ids = np.asarray(g.frames.id)
    i = int(np.clip(np.searchsorted(ids, frame), 0, len(ids) - 1))
    out = []
    for p in game["players"]:
        po = g.frames.ports[p["port"]].leader.post
        out.append(dict(port=p["port"], char=p["char"], char_name=p["char_name"], me=p.get("code") == me_code,
                        x=float(np.asarray(po.position.x)[i]), y=float(np.asarray(po.position.y)[i]),
                        facing=1 if float(np.asarray(po.direction)[i]) > 0 else -1,
                        percent=float(np.asarray(po.percent)[i])))
    return dict(frame=int(ids[i]), stage=game["stage"], players=out)


EDGE = {"Battlefield": 68.4, "Dream Land": 77.27, "Final Destination": 85.57, "Yoshi's Story": 56.0,
        "Fountain of Dreams": 63.35, "Pokémon Stadium": 87.75}


def situation(p, stage):
    """A decision's situation in plain words: what you just came out of, where, how far away."""
    c = (p.get("came") or "")
    if c.startswith(("DAMAGE", "TUMBLING", "DOWN_", "LYING", "TECH", "NEUTRAL_TECH", "FORWARD_TECH", "BACKWARD_TECH",
                     "GROUND_GETUP", "THROWN")):
        what = "after getting hit"
    elif c.startswith(("SHIELD", "GUARD")):
        what = "out of shield"
    elif "LANDING" in c:
        what = "after landing"
    elif c.startswith(("EDGE", "CLIFF")):
        what = "on the ledge"
    elif c.startswith(("NAIR", "FAIR", "BAIR", "UAIR", "DAIR", "FTILT", "UPTILT", "DOWNTILT", "FSMASH", "UPSMASH",
                       "DOWNSMASH", "NEUTRAL_ATTACK", "DASH_ATTACK", "GRAB", "THROW")) or c.isdigit():
        what = "after your attack"
    elif c.startswith(("DASHING", "RUNNING", "TURNING", "WALK", "RUN_BRAKE")):
        what = "while moving"
    elif c.startswith(("JUMPING", "FALLING", "KNEE_BEND", "AIRDODGE")):
        what = "in the air"
    else:
        what = "standing"
    edge = EDGE.get(stage, 70.0)
    if abs(p["x"]) > edge + 2:
        where = "off stage"
    else:
        d = abs(p["x"] - p["ox"])
        where = "close" if d < 20 else "mid range" if d < 55 else "far apart"
    return f"{what} · {where}"


def _patterns(self, jid, job, games, me_code):
    """Across many games: review each, group decisions by situation, find the situations that
    cost you most consistently, and run master tries at example moments of each."""
    import random as _r
    rows = []
    for gi, g in enumerate(games):
        job.update(progress=round(0.7 * gi / len(games), 3), message=f"Reviewing game {gi + 1} of {len(games)}…")
        try:
            an = analyse(g, me_code)
            if g["id"] not in self.reviews:
                rj = dict(id=uuid.uuid4().hex[:12], kind="review", state="running")
                self.do_review(rj, g, an, compare=False)
            rv = self.reviews[g["id"]]
        except Exception as e:
            traceback.print_exc()
            continue
        by_f = {p["frame"]: p for p in an["points"]}
        for p in rv["points"]:
            q = by_f.get(p["frame"])
            if q:
                rows.append(dict(game=g["id"], frame=p["frame"], cost=p["cost"], did=p["did"],
                                 sit=situation(q, g["stage"]), first=p["did"].split(" → ")[0]))
    if not rows:
        raise RuntimeError("No reviewable games found.")
    costs = np.array([r["cost"] for r in rows])
    base = float(costs.mean())
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["sit"]].append(r)
    found = []
    for sit, rs in groups.items():
        c = np.array([r["cost"] for r in rs])
        if len(rs) < 12:
            continue
        se = float(c.std(ddof=1) / np.sqrt(len(c)))
        found.append(dict(situation=sit, n=len(rs), mean=round(float(c.mean()) - base, 2), se=round(se, 2),
                          t=(float(c.mean()) - base) / max(se, 1e-6), rows=rs))
    found.sort(key=lambda f: f["t"])
    worst = [f for f in found if f["t"] < -2][:6] or found[:3]
    rng = _r.Random(0)
    for wi, f in enumerate(worst):
        yours = collections.Counter(r["first"] for r in f["rows"])
        f["you"] = [[k, round(v / f["n"], 2)] for k, v in yours.most_common(4)]
        ex = sorted(f["rows"], key=lambda r: r["cost"])[:3] + rng.sample(f["rows"], min(3, len(f["rows"])))
        seen, examples = set(), []
        for r in ex:
            if (r["game"], r["frame"]) not in seen:
                seen.add((r["game"], r["frame"]))
                examples.append(r)
        master = collections.Counter()
        tot = 0
        for ei, r in enumerate(examples[:5]):
            job.update(progress=round(0.7 + 0.3 * (wi + ei / 5) / len(worst), 3),
                       message=f"Asking the master about \"{f['situation']}\" ({ei + 1}/5)…")
            g = next(x for x in games if x["id"] == r["game"])
            an = analyse(g, me_code)
            sub = dict(id=uuid.uuid4().hex[:12], kind="branch", state="running", frame=r["frame"], k=8, horizon=120)
            try:
                self.do_branch(sub["id"], sub, g, an)
            except Exception:
                continue
            for o in sub["options"]:
                master[o["label"].split(" → ")[0]] += 1
                tot += 1
        f["master"] = [[k, round(v / tot, 2)] for k, v in master.most_common(4)] if tot else []
        start_of = {x["id"]: x.get("start") for x in games}
        f["examples"] = [dict(game=r["game"], frame=r["frame"], did=r["did"], cost=r["cost"], start=start_of.get(r["game"]))
                         for r in examples[:6]]
        del f["rows"]
    job.update(state="done", progress=1.0, patterns=worst, decisions=len(rows), games=len(games), base=round(base, 2))


Jobs.do_patterns = _patterns


def _v_near(ev, f):
    fs = np.array([e[0] for e in ev])
    return float(ev[int(np.clip(np.searchsorted(fs, f), 0, len(ev) - 1))][1])


# ── HTTP ─────────────────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, device):
        self.cfg = load_config()
        self.lib = Library()
        self.jobs = Jobs(device)
        self.pattern_cache = {}
        self.iso, self.iso_from, self.iso_path = find_iso(self.cfg)
        self.live, self.cpu, self.cpu_values = {}, None, {}   # live play: sessions, CPU models
        self.jobs.iso_path = self.iso_path if self.iso else None
        threading.Thread(target=self.lib.scan, args=(self.cfg["replay_dir"],), daemon=True).start()

    def cpu_models(self):
        """Models on the CPU for live play (the GPU may be training)."""
        if self.cpu is None:
            import torch
            from branch import Brancher
            torch.set_num_threads(4)
            self.cpu = Brancher("cpu", iso=self.iso_path if self.iso else None)
        return self.cpu

    def cpu_value(self, char):
        if char not in self.cpu_values:
            from value import Value
            vp = value_path(char)
            self.cpu_values[char] = Value(str(vp), "cpu") if vp else None
        return self.cpu_values[char]

    def start_live(self, g, an, frame, speed=1.0):
        from live import LiveSession
        for old in list(self.live.values()):          # one at a time
            old.stop()
        self.live.clear()
        me, opp = an["me"], an["opp"]
        pt = next((p for p in an["points"] if p["frame"] <= frame < p["frame"] + p["wait"] + 30), None)
        bot, kind = live_policy(opp["char"])
        sid = uuid.uuid4().hex[:12]
        sess = LiveSession(self.cpu_models(), g["path"], (me["port"], me["char"]), (opp["port"], opp["char"]),
                           frame, stretch_of(pt["wait"] if pt else 40), bot, policy_dir(me["char"]),
                           self.cpu_value(me["char"]), iso=self.iso_path if self.iso else None, lead=60, after=WATCH,
                           out_dir=HOME / "live" / sid)
        sess.id, sess.bot_kind, sess.game = sid, kind, g["id"]
        self.live[sid] = sess

        def work():
            import torch
            torch.set_num_threads(4)    # per thread: the bot's forward on 4 cores (more only adds contention)
            try:
                sess.prepare()
                sess.run()
            except Exception:
                traceback.print_exc()
        threading.Thread(target=work, daemon=True).start()
        return sess

    def me(self):
        if self.cfg.get("me"):
            return self.cfg["me"]
        codes = self.lib.codes()
        return codes[0][0] if codes else None


def only_player(data, port):
    """A copy of a .slp with every player but `port` (0-based) removed: their frame events and
    their items are dropped and their Game Start slot is marked empty. Used to draw ghosts of
    the player being coached only."""
    import struct
    i0 = data.find(b"raw[$U#l") + 8
    n = struct.unpack_from(">I", data, i0)[0]
    raw, tail = data[i0 + 4:i0 + 4 + n], data[i0 + 4 + n:]
    assert raw[0] == 0x35
    sizes = {raw[2 + 3 * k]: struct.unpack_from(">H", raw, 3 + 3 * k)[0] for k in range((raw[1] - 1) // 3)}
    sizes[0x35] = raw[1]
    out, i = bytearray(), 0
    while i < len(raw):
        c = raw[i]
        if c not in sizes:
            raise ValueError(f"unknown event 0x{c:02x}")
        ev = raw[i:i + 1 + sizes[c]]
        i += 1 + sizes[c]
        if c in (0x37, 0x38) and ev[5] != port:          # pre-/post-frame: player index at 5
            continue
        if c == 0x3B and len(ev) > 0x2A and ev[0x2A] not in (port, 0xFF):   # item: owner at 0x2A
            continue
        if c == 0x36:                                    # game start: 4 player blocks of 0x24 at 0x65
            ev = bytearray(ev)
            for q in range(4):
                if q != port:
                    ev[0x65 + 0x24 * q + 1] = 3          # player type 3 = empty
            ev = bytes(ev)
        out += ev
    return data[:i0] + struct.pack(">I", len(out)) + bytes(out) + tail


class Handler(BaseHTTPRequestHandler):
    app: App = None

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, path, ctype=None):
        path = Path(path)
        if not path.is_file():
            return self.send_json(dict(error="not found"), 404)
        b = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def send_bytes(self, b, ctype="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def stream_live(self, sess):
        """Server-sent events: "f" = one frame's Slippi events (base64), "s" = the session's status."""
        import base64
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = sess.listen()
        try:
            while True:
                try:
                    kind, data = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                if kind == "f":
                    msg = b"event: f\ndata: " + base64.b64encode(data) + b"\n\n"
                else:
                    msg = b"event: s\ndata: " + json.dumps(dict(data, id=sess.id, bot_kind=sess.bot_kind)).encode() + b"\n\n"
                self.wfile.write(msg)
                self.wfile.flush()
                if kind == "s" and data.get("state") in ("done", "error", "stopped"):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if q in sess.listeners:
                sess.listeners.remove(q)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:
            self.route_get()
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            self.send_json(dict(error=str(e)[:300]), 500)

    def do_HEAD(self):
        if self.path.startswith("/gamefiles/"):
            return self.do_GET()
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        try:
            self.route_post()
        except Exception as e:
            traceback.print_exc()
            self.send_json(dict(error=str(e)[:300]), 500)

    def route_get(self):
        a = self.app
        u = urlparse(self.path)
        p = u.path
        if p == "/api/status":
            import torch
            return self.send_json(dict(replay_dir=a.cfg["replay_dir"], replay_dir_exists=Path(a.cfg["replay_dir"]).is_dir(),
                                       me=a.me(), codes=a.lib.codes(), games=len(a.lib.games), scanning=a.lib.scanning,
                                       gpu=torch.cuda.is_available() and a.jobs.device != "cpu",
                                       iso=dict(ok=a.iso is not None, source=a.iso_from, path=a.iso_path)))
        m = re.fullmatch(r"/api/live/(\w+)(/prefix\.slp|/stream|/attempt\.slp)?", p)
        if m:
            sess = a.live.get(m.group(1))
            if sess is None:
                return self.send_json(dict(error="That attempt is over; start again."), 404)
            what = m.group(2)
            if what is None:
                return self.send_json(dict(sess.status(), id=sess.id, bot_kind=sess.bot_kind))
            if what == "/prefix.slp":
                if sess.prefix is None:
                    return self.send_json(dict(error="not ready"), 409)
                return self.send_bytes(sess.prefix)
            if what == "/attempt.slp":
                return self.send_file((sess.result or {}).get("slp") or "/nonexistent", "application/octet-stream")
            return self.stream_live(sess)
        if p == "/api/games":
            me = a.me()
            return self.send_json(dict(me=me, scanning=a.lib.scanning,
                                       games=[summary(g, me) for g in a.lib.games
                                              if any(pl.get("code") == me for pl in g["players"])]))
        m = re.fullmatch(r"/api/game/(\w+)(/replay\.slp)?", p)
        if m:
            g = a.lib.by_id.get(m.group(1))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            if m.group(2):
                return self.send_file(g["path"], "application/octet-stream")
            s = summary(g, a.me())
            if s["unsupported"]:
                return self.send_json(dict(summary=s))
            an = analyse(g, a.me())
            return self.send_json(dict(summary=s, **{k: v for k, v in an.items() if not k.startswith("_")}))
        m = re.fullmatch(r"/api/game/(\w+)/state", p)
        if m:
            g = a.lib.by_id.get(m.group(1))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            f = int(parse_qs(u.query).get("f", ["0"])[0])
            return self.send_json(game_state(g, a.me(), f))
        m = re.fullmatch(r"/api/job/(\w+)", p)
        if m:
            j = a.jobs.jobs.get(m.group(1))
            return self.send_json(j or dict(error="unknown job"), 200 if j else 404)
        m = re.fullmatch(r"/api/branch/([\w]+)/(\d+)\.slp", p)
        if m:
            hit = next((HOME / "branches" / m.group(1)).glob(f"branch_*_{m.group(2)}.slp"), None)
            only = parse_qs(u.query).get("only")
            if hit and only:
                b = only_player(hit.read_bytes(), int(only[0]))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            return self.send_file(hit or "/nonexistent", "application/octet-stream")
        if p.startswith("/gamefiles/"):
            from urllib.parse import unquote
            name = unquote(p[len("/gamefiles/"):])
            if not a.iso or name not in a.iso.files:
                return self.send_json(dict(error="not in the game files"), 404)
            b = a.iso.read(name)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Last-Modified", time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(a.iso.mtime)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(b)
            return
        # static
        if p in ("/", "/index.html"):
            return self.send_file(WEB / "index.html", "text/html; charset=utf-8")
        f = (WEB / p.lstrip("/")).resolve()
        if WEB.resolve() in f.parents and f.is_file():
            return self.send_file(f)
        if p.startswith("/viewer/") and p.rstrip("/") == "/viewer":
            return self.send_file(WEB / "viewer" / "embed.html", "text/html")
        return self.send_json(dict(error="not found"), 404)

    def route_post(self):
        a = self.app
        p = urlparse(self.path).path
        if p == "/api/config":
            b = self.body()
            if "replay_dir" in b:
                d = os.path.expanduser(b["replay_dir"])
                if not Path(d).is_dir():
                    return self.send_json(dict(error="That folder doesn't exist."), 400)
                a.cfg["replay_dir"] = d
                threading.Thread(target=a.lib.scan, args=(d,), daemon=True).start()
            if "me" in b:
                a.cfg["me"] = b["me"]
            if "iso" in b:
                iso, src, path = find_iso(dict(iso=os.path.expanduser(b["iso"])))
                if iso is None or src != "your setting":
                    return self.send_json(dict(error="That file isn't an NTSC 1.02 Melee ISO."), 400)
                a.cfg["iso"] = path
                a.iso, a.iso_from, a.iso_path = iso, src, path
            save_config(a.cfg)
            return self.send_json(dict(ok=True))
        if p == "/api/rescan":
            threading.Thread(target=a.lib.scan, args=(a.cfg["replay_dir"],), daemon=True).start()
            return self.send_json(dict(ok=True))
        if p == "/api/branch":
            b = self.body()
            g = a.lib.by_id.get(b.get("game"))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            why = support(g, a.me())
            if why:
                return self.send_json(dict(error=why), 400)
            an = analyse(g, a.me())
            frame = int(b["frame"])
            frame = max(0, min(frame, an["last"] - 2))
            job = a.jobs.submit("branch", g, an, frame=frame, k=min(32, int(b.get("k", 16))), horizon=int(b.get("horizon", 0)))
            return self.send_json(job)
        if p == "/api/live":
            b = self.body()
            g = a.lib.by_id.get(b.get("game"))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            why = support(g, a.me())
            if why:
                return self.send_json(dict(error=why), 400)
            an = analyse(g, a.me())
            frame = max(60 - 123, min(int(b["frame"]), an["last"] - 2))
            sess = a.start_live(g, an, frame)
            return self.send_json(dict(sess.status(), id=sess.id, bot_kind=sess.bot_kind))
        m = re.fullmatch(r"/api/live/(\w+)/(pad|go|stop)", p)
        if m:
            sess = a.live.get(m.group(1))
            if sess is None:
                return self.send_json(dict(error="That attempt is over; start again."), 404)
            b = self.body()
            if m.group(2) == "pad":
                from live import pad_from_json
                sess.pad = pad_from_json(b)
            elif m.group(2) == "go":
                sess.go(float(b.get("speed", 1.0)))
            else:
                sess.stop()
            return self.send_json(dict(ok=True))
        if p == "/api/test":
            b = self.body()
            g = a.lib.by_id.get(b.get("game"))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            why = support(g, a.me())
            if why:
                return self.send_json(dict(error=why), 400)
            an = analyse(g, a.me())
            frame = max(0, min(int(b["frame"]), an["last"] - 2))
            src = "you" if b.get("you") else "try"
            if src == "try" and not (re.fullmatch(r"\w+", str(b.get("job", ""))) and str(b.get("try", "")).isdigit()):
                return self.send_json(dict(error="bad try"), 400)
            return self.send_json(a.jobs.submit("test", g, an, frame=frame, k=min(48, int(b.get("k", 32))), source=src,
                                                from_job=b.get("job"), from_try=b.get("try"), label=b.get("label")))
        if p == "/api/earlier":
            b = self.body()
            g = a.lib.by_id.get(b.get("game"))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            why = support(g, a.me())
            if why:
                return self.send_json(dict(error=why), 400)
            an = analyse(g, a.me())
            frame = max(0, min(int(b["frame"]), an["last"] - 2))
            return self.send_json(a.jobs.submit("earlier", g, an, frame=frame, k=int(b.get("k", 8))))
        if p == "/api/lab":
            b = self.body()
            g = a.lib.by_id.get(b.get("game"))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            why = support(g, a.me())
            if why:
                return self.send_json(dict(error=why), 400)
            an = analyse(g, a.me())
            at = max(0, min(int(b["at"]), an["last"] - 2))
            job = a.jobs.submit("lab", g, an, at=at, place=b["place"], k=int(b.get("k", 6)),
                                horizon=int(b.get("horizon", 300)), from_job=b.get("from_job"),
                                from_try=b.get("from_try"), frame=b.get("frame"))
            return self.send_json(job)
        if p == "/api/patterns":
            b = self.body()
            me = a.me()
            char = b.get("char")
            games = [g for g in a.lib.games if not support(g, me)
                     and any(pl.get("code") == me and (not char or pl["char_name"] == char) and value_path(pl["char"])
                             for pl in g["players"])][:int(b.get("n", 40))]
            if not games:
                return self.send_json(dict(error="No games to look at yet."), 400)
            key = (char, tuple(g["id"] for g in games))
            hit = a.jobs.jobs.get(a.pattern_cache.get(key, ""))
            if hit and hit["state"] in ("queued", "running", "loading", "done"):
                return self.send_json(hit)
            jid = uuid.uuid4().hex[:12]
            a.pattern_cache[key] = jid
            a.jobs.jobs[jid] = dict(id=jid, kind="patterns", state="queued", progress=0.0, message="Starting…",
                                    started=time.time(), char=char)
            a.jobs._n += 1
            a.jobs.q.put((0, a.jobs._n, jid, games, me))
            return self.send_json(a.jobs.jobs[jid])
        if p == "/api/review":
            b = self.body()
            g = a.lib.by_id.get(b.get("game"))
            if not g:
                return self.send_json(dict(error="unknown game"), 404)
            why = support(g, a.me())
            if why:
                return self.send_json(dict(error=why), 400)
            if g["id"] in a.jobs.reviews:
                rv = a.jobs.reviews[g["id"]]
                if not rv.get("compare") and any(p_["tag"] for p_ in rv["points"]):   # reviewed by the pattern report
                    an = analyse(g, a.me())
                    rv["compare"] = {}
                    for p_ in rv["points"]:
                        if p_["tag"]:
                            bj = a.jobs.submit("branch", g, an, frame=p_["frame"], k=16, horizon=-1, review=rv["id"])
                            rv["compare"][str(p_["frame"])] = dict(job=bj["id"], state="queued")
                    if rv["id"] in a.jobs.jobs:
                        a.jobs.jobs[rv["id"]]["compare"] = rv["compare"]
                return self.send_json({k: v for k, v in rv.items() if k != "eval_full"})
            an = analyse(g, a.me())
            me = an["me"]
            if not value_path(me["char"]):
                return self.send_json(dict(error=f"No review model for {me['char_name']} yet."), 400)
            return self.send_json(a.jobs.submit("review", g, an))
        return self.send_json(dict(error="not found"), 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    a = ap.parse_args()
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Handler.app = App(dev)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Coach is running: open http://localhost:{a.port}  (models on {dev})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
