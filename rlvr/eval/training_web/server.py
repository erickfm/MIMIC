"""Web-based training HUD for the composite-VR RLVR loop. Stdlib HTTP
server + SSE; tails the training log and streams events to a single
self-contained HTML page.

Run alongside training (auto-launched by rlvr/online/loop.py when
training is viewable):

    python3 -m rlvr.eval.training_web.server \
        --log logs/rlvr_run.log --port 8765 --max-updates 50

Open http://localhost:8765 in a browser, or add it as an OBS Browser
Source.

The HUD is VR-suite-centric: it parses the `EVT_EP_VR` JSON event the
actor emits per match (rlvr/online/dolphin_actor.py) and renders one
card per VRModule — name, what it rewards, weight, cumulative reward
contribution, a per-match reward sparkline, and the VR's diagnostic
counts. The composite reward is attributed back to each VR by name.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("rlvr.eval.training_web")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")


# ---------- Log parsers -----------------------------------------------
#
# Three line shapes are consumed:
#   - `update=N ... kl=K ...`     the per-PPO-update line from loop.py
#   - `EVT_MATCH_END result=...`  one per finished match (win/loss tally)
#   - `EVT_EP_VR {json}`          one per match: the per-VR breakdown
# plus `loading frozen opponent: PATH` for the opponent label.

_RE_UPDATE = re.compile(
    r"update=(?P<u>\d+)\s+collected=(?P<c>\d+)\s+valid=(?P<v>\d+)\s+"
    r"reward=(?P<r>[\d\.\-]+)\s+kl=(?P<kl>[\d\.\-]+)"
)
_RE_MATCH_END = re.compile(
    r"EVT_MATCH_END\s+(?:actor=\d+\s+)?result=(?P<r>\S+)\s+"
    r"trainee_stocks=(?P<ts>\d+)\s+opp_stocks=(?P<os>\d+)"
)
_RE_FROZEN = re.compile(r"loading frozen opponent: (\S+)")
# `actor=K` prefix is emitted by every EVT_* line when ActorPool drives
# the run (N actors share one log file). For single-actor runs the
# prefix is `actor=0`. The regexes here all accept it as an optional
# segment so old logs without the prefix still parse. Per-actor demux
# (separate state + UI per actor_id) is a future enhancement; for now
# all actors' events aggregate into one state.
_RE_EP_OPEN = re.compile(r"EVT_EP_OPEN\s+(?:actor=\d+\s+)?frame=(?P<f>\d+)")
_RE_EP_TICK = re.compile(
    r"EVT_EP_TICK\s+(?:actor=\d+\s+)?frame=(?P<f>\d+)\s+"
    r"opp_pct=(?P<p>[\d\.\-]+)"
)
# Substring sentinels — must match BOTH legacy ("EVT_EP_VR {json}")
# and new ("EVT_EP_VR actor=K {json}") forms. The actor=K prefix
# (if present) is skipped before json.loads().
_EVT_VR = "EVT_EP_VR "
_EVT_VR_TICK = "EVT_VR_TICK "
_RE_ACTOR_PREFIX = re.compile(r"^actor=\d+\s+")


# ---------- State + broker --------------------------------------------


class HUDState:
    """Server-side mirror of the dashboard state. Serializable to JSON
    so a new client gets a correct `snapshot` event on connect."""

    def __init__(self) -> None:
        self.opponent_label: str = ""
        self.update: int = 0
        self.max_updates: int = 50
        self.session_start: float = time.time()
        self.last_kl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        self.matches: int = 0
        # vr_id -> {weight, reward_total, counts: {k: n}, spark: [per-match r],
        #          live: {reward, counts}, events: [{frame, delta}], _last_live}
        self.vrs: Dict[str, Dict[str, Any]] = {}
        # Live in-match pulse so the HUD doesn't go silent for ~3 min
        # between EVT_EP_VR events. Updated from EVT_EP_OPEN / EVT_EP_TICK.
        self.live: Dict[str, Any] = {
            "in_match": False, "frame": 0, "opp_pct": 0.0,
            "last_result": "",
        }
        # Session telemetry — one composite-Σ per finished match, one KL
        # per PPO update. Trimmed to keep snapshots small.
        self.composite_history: List[float] = []  # one value per match
        self.kl_history: List[Dict[str, float]] = []  # {update, kl}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opponent_label": self.opponent_label,
            "update": self.update,
            "max_updates": self.max_updates,
            "session_start": self.session_start,
            "last_kl": self.last_kl,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "matches": self.matches,
            "vrs": self.vrs,
            "live": self.live,
            "composite_history": self.composite_history,
            "kl_history": self.kl_history,
            "server_now": time.time(),
        }


class EventBroker:
    """Thread-safe pub-sub. Each subscriber owns a queue.Queue that the
    request handler drains into the SSE response."""

    def __init__(self) -> None:
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def broadcast(self, ev: Dict[str, Any]) -> None:
        payload = json.dumps(ev)
        with self._lock:
            stale = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    stale.append(q)
            for q in stale:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass


# ---------- Log tailer ------------------------------------------------


class LogTailer(threading.Thread):
    """Tails the training log, parses events, mutates the shared
    HUDState, and broadcasts deltas via the broker."""

    daemon = True

    def __init__(self, log_path: Path, state: HUDState, broker: EventBroker):
        super().__init__(name="log-tailer")
        self.log_path = log_path
        self.state = state
        self.broker = broker
        self._stop = threading.Event()
        # Throttle the live tick broadcast: EVT_EP_TICK fires ~10 Hz, push
        # every Nth so the SSE stream stays calm.
        self._tick_skip = 0
        self._TICK_EVERY = 3

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self.log_path.exists() and not self._stop.is_set():
            time.sleep(0.5)
        if self._stop.is_set():
            return
        with self.log_path.open("r") as fh:
            # Replay existing content (so a mid-run launch has history),
            # then tail.
            for line in fh:
                self._consume(line, broadcast=False)
            while not self._stop.is_set():
                line = fh.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                self._consume(line, broadcast=True)

    # --- dispatch ---

    def _consume(self, line: str, broadcast: bool) -> None:
        # EP_TICK is the hottest line; check first so the cheap regex
        # path returns immediately on the high-frequency case.
        m = _RE_EP_TICK.search(line)
        if m:
            self._on_ep_tick(int(m.group("f")), float(m.group("p")), broadcast)
            return
        m = _RE_EP_OPEN.search(line)
        if m:
            self._on_ep_open(int(m.group("f")), broadcast)
            return
        m = _RE_FROZEN.search(line)
        if m:
            self._on_frozen(m.group(1), broadcast)
            return
        m = _RE_UPDATE.search(line)
        if m:
            self._on_update(m, broadcast)
            return
        m = _RE_MATCH_END.search(line)
        if m:
            self._on_match_end(m, broadcast)
            return
        idx = line.find(_EVT_VR_TICK)
        if idx != -1:
            payload = line[idx + len(_EVT_VR_TICK):]
            # Capture per-actor id BEFORE stripping the prefix. Used to
            # demux 4 actors' live-overlay state in _on_vr_tick (else
            # the displayed live counts/rewards bounce between actors).
            m_act = _RE_ACTOR_PREFIX.match(payload)
            actor_id = int(m_act.group(0).split("=")[1].strip()) if m_act else 0
            payload = _RE_ACTOR_PREFIX.sub("", payload, count=1)
            self._on_vr_tick(payload, actor_id, broadcast)
            return
        idx = line.find(_EVT_VR)
        if idx != -1:
            payload = line[idx + len(_EVT_VR):]
            payload = _RE_ACTOR_PREFIX.sub("", payload, count=1)
            self._on_vr(payload, broadcast)
            return

    # --- handlers ---

    def _on_frozen(self, path: str, broadcast: bool) -> None:
        self.state.opponent_label = Path(path).name
        if broadcast:
            self.broker.broadcast({
                "type": "frozen", "label": self.state.opponent_label,
            })

    def _on_update(self, m, broadcast: bool) -> None:
        try:
            u = int(m.group("u"))
        except (TypeError, ValueError):
            return
        try:
            kl = float(m.group("kl"))
        except (TypeError, ValueError):
            kl = self.state.last_kl
        self.state.update = u
        self.state.last_kl = kl
        self.state.kl_history.append({"update": u, "kl": kl})
        if broadcast:
            self.broker.broadcast({
                "type": "update", "update": u, "kl": kl,
                "kl_history": self.state.kl_history,
            })

    def _on_match_end(self, m, broadcast: bool) -> None:
        result = m.group("r")
        if result == "win":
            self.state.wins += 1
        elif result == "loss":
            self.state.losses += 1
        else:
            self.state.draws += 1
        self.state.live["in_match"] = False
        self.state.live["last_result"] = result
        if broadcast:
            self.broker.broadcast({
                "type": "match_end",
                "wins": self.state.wins,
                "losses": self.state.losses,
                "draws": self.state.draws,
                "live": dict(self.state.live),
            })

    def _on_ep_open(self, frame: int, broadcast: bool) -> None:
        self.state.live["in_match"] = True
        self.state.live["frame"] = 0
        self.state.live["opp_pct"] = 0.0
        # Reset every VR's per-match event timeline + the live-delta tracker.
        for entry in self.state.vrs.values():
            entry["events"] = []
            entry["_last_live"] = 0.0
        if broadcast:
            self.broker.broadcast({"type": "live", "live": dict(self.state.live)})

    def _on_ep_tick(self, frame: int, opp_pct: float, broadcast: bool) -> None:
        self.state.live["in_match"] = True
        self.state.live["frame"] = frame
        self.state.live["opp_pct"] = opp_pct
        if not broadcast:
            return
        self._tick_skip = (self._tick_skip + 1) % self._TICK_EVERY
        if self._tick_skip == 0:
            self.broker.broadcast({"type": "live", "live": dict(self.state.live)})

    def _on_vr(self, payload: str, broadcast: bool) -> None:
        """Consume one `EVT_EP_VR {json}` event: a per-match breakdown
        `{n_frames, reward_sum, terminal, result, vrs:{id:{weight,
        reward, ...counts}}}`. VR `metadata()` counts are per-match, so
        they accumulate; the per-match reward feeds each VR's spark."""
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return
        vrs = data.get("vrs")
        if not isinstance(vrs, dict):
            return
        self.state.matches += 1
        match_total = 0.0
        for vid, d in vrs.items():
            if not isinstance(d, dict):
                continue
            entry = self._vr_entry(vid)
            try:
                entry["weight"] = float(d.get("weight", entry["weight"]))
            except (TypeError, ValueError):
                pass
            try:
                r = float(d.get("reward", 0.0))
            except (TypeError, ValueError):
                r = 0.0
            entry["reward_total"] = round(entry["reward_total"] + r, 4)
            entry["spark"].append(round(r, 3))
            if len(entry["spark"]) > 48:
                entry["spark"] = entry["spark"][-48:]
            match_total += r
            for k, v in d.items():
                if k in ("weight", "reward") or isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    entry["counts"][k] = round(
                        entry["counts"].get(k, 0) + v, 1)
            # The just-finished match is now folded into the cumulative
            # totals; clear the live overlay so the card doesn't double-
            # show it until the next EVT_VR_TICK arrives.
            entry["live"] = {"reward": 0.0, "counts": {}}
            entry["_last_live"] = 0.0
        # One composite-Σ point per finished match for the session-wide
        # trajectory plot. Trim to the last 200 matches so snapshots stay
        # small (a 100-update run is 600 matches but 200 is plenty for
        # visual trend).
        self.state.composite_history.append(round(match_total, 4))
        if len(self.state.composite_history) > 200:
            self.state.composite_history = self.state.composite_history[-200:]
        if broadcast:
            self.broker.broadcast({
                "type": "vr",
                "vrs": self.state.vrs,
                "matches": self.state.matches,
                "composite_history": self.state.composite_history,
            })

    def _vr_entry(self, vid: str) -> Dict[str, Any]:
        """Get-or-create a VR card entry with all expected sub-fields:
        `live` (in-progress overlay), `events` (per-match firing times
        for the per-card event strip), and `_last_live` (server-side
        bookkeeping for delta computation in _on_vr_tick)."""
        entry = self.state.vrs.get(vid)
        if entry is None:
            entry = {"weight": 1.0, "reward_total": 0.0,
                     "counts": {}, "spark": [],
                     "live": {"reward": 0.0, "counts": {}},
                     "events": [], "_last_live": 0.0}
            self.state.vrs[vid] = entry
        else:
            entry.setdefault("live", {"reward": 0.0, "counts": {}})
            entry.setdefault("events", [])
            entry.setdefault("_last_live", 0.0)
        return entry

    def _on_vr_tick(self, payload: str, actor_id: int,
                    broadcast: bool) -> None:
        """Consume an `EVT_VR_TICK {json}` event — the composite task's
        in-progress per-VR state, emitted by the actor every ~30 game
        frames (~2 Hz). Updates each VR's `live` overlay; the card UI
        displays `reward_total + live.reward` so numbers climb live.

        Multi-actor: each of N actors emits its own ticks. Per-actor
        live state is kept under `entry["live"]["per_actor"][actor_id]`,
        and `entry["live"]["reward"]` / `entry["live"]["counts"]` are
        SUMS across all actors. Without this, the displayed live values
        would bounce between actors as each one's tick lands.

        Per-actor `_last_live` keys track delta-from-prior-tick for the
        event timeline; events are still written to the shared
        `entry["events"]` list (no per-actor demux on that yet)."""
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        frame = int(self.state.live.get("frame", 0))
        for vid, d in data.items():
            if not isinstance(d, dict):
                continue
            entry = self._vr_entry(vid)
            try:
                entry["weight"] = float(d.get("weight", entry["weight"]))
            except (TypeError, ValueError):
                pass
            try:
                new_live = float(d.get("reward", 0.0))
            except (TypeError, ValueError):
                new_live = 0.0

            # Per-actor state for proper aggregation.
            pa_map = entry["live"].setdefault("per_actor", {})
            pa = pa_map.setdefault(str(actor_id),
                                   {"reward": 0.0, "counts": {},
                                    "_last_live": 0.0})
            delta = new_live - pa["_last_live"]
            if abs(delta) > 0.01:
                entry["events"].append({"frame": frame,
                                         "delta": round(delta, 3)})
                if len(entry["events"]) > 80:
                    entry["events"] = entry["events"][-80:]
            pa["_last_live"] = new_live
            pa["reward"] = new_live
            pa_counts: Dict[str, Any] = {}
            for k, v in d.items():
                if k in ("weight", "reward") or isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    pa_counts[k] = v
            pa["counts"] = pa_counts

            # Aggregate (sum) across all actors for display.
            entry["live"]["reward"] = round(
                sum(p["reward"] for p in pa_map.values()), 4)
            agg_counts: Dict[str, float] = {}
            for p in pa_map.values():
                for k, v in p["counts"].items():
                    agg_counts[k] = agg_counts.get(k, 0) + v
            entry["live"]["counts"] = agg_counts
        if broadcast:
            self.broker.broadcast({"type": "vr_tick", "vrs": self.state.vrs})


# ---------- HTML page -------------------------------------------------


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>RLVR live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0c0d10;
    --card: #101116;
    --fg: #e8e8ea;
    --fg2: #7c7e88;
    --fg3: #3a3c44;
    --line: #1c1e25;
    --good: #5fcf67;
    --bad: #a85258;
  }
  * { box-sizing: border-box; }
  html, body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.45;
    margin: 0; padding: 0; height: 100%;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .dim  { color: var(--fg2); }

  .root {
    display: grid;
    grid-template-rows: auto auto auto auto 1fr auto;
    height: 100vh;
    padding: 18px 28px;
    gap: 12px;
  }

  /* session telemetry: composite-Σ-per-match + KL-per-update */
  .telemetry {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .tel-cell {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px 12px 6px;
    display: grid;
    grid-template-columns: 1fr auto;
    grid-template-rows: auto auto;
    column-gap: 10px;
    align-items: baseline;
  }
  .tel-label { color: var(--fg2); font-size: 11px; }
  .tel-value {
    grid-row: 1; grid-column: 2; align-self: baseline;
    font-size: 13px; color: var(--fg);
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .tel-label { grid-row: 1; grid-column: 1; }
  .tel-svg { width: 100%; height: 22px; display: block;
             grid-column: 1 / -1; grid-row: 2; }

  /* per-card event timeline — dots at frame positions */
  .event-bar { width: 100%; height: 8px; display: block; margin-top: 4px; }
  .event-bar-label {
    display: flex; justify-content: space-between;
    font-size: 10px; color: var(--fg3); margin-top: 2px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
  }

  /* live match pulse — fills the silence between EVT_EP_VR events */
  .live-strip {
    color: var(--fg2); font-size: 12px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
  }
  .live-strip .pulse {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    margin-right: 7px; transform: translateY(-1px);
    background: var(--good);
    animation: pulse 1.4s ease-in-out infinite;
  }
  .live-strip.idle .pulse { background: var(--fg3); animation: none; }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: translateY(-1px) scale(1); }
    50%     { opacity: 0.55; transform: translateY(-1px) scale(1.2); }
  }

  /* PPO-update spinner — shown after the 6th EP_VR of a batch, hidden
     when the next `update=N` line lands. */
  .live-spinner {
    display: none;
    width: 10px; height: 10px; margin-right: 7px;
    border: 1.6px solid var(--line);
    border-top-color: var(--good);
    border-radius: 50%;
    vertical-align: -1px;
    animation: spin 0.9s linear infinite;
  }
  .live-strip.ppo .pulse { display: none; }
  .live-strip.ppo .live-spinner { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* header */
  .header { display: flex; align-items: baseline; justify-content: space-between; }
  .title { font-size: 18px; font-weight: 500; letter-spacing: 0.02em; }
  .title span.dot { color: var(--fg3); margin: 0 10px; }
  .title .task { color: var(--fg2); font-size: 14px; }
  .update-wrap { display: flex; align-items: center; gap: 12px; }
  .update-text { color: var(--fg2); font-size: 13px; }
  .update-bar {
    width: 180px; height: 2px; background: var(--line);
    border-radius: 1px; overflow: hidden;
  }
  .update-bar > div { height: 100%; background: var(--fg2); transition: width 0.4s ease; }

  .players {
    color: var(--fg2); font-size: 13px;
    padding-bottom: 4px; border-bottom: 1px solid var(--line);
  }
  .players strong { color: var(--fg); font-weight: 500; }
  .players .arrow { color: var(--fg3); padding: 0 12px; }

  /* VR card grid */
  .vr-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 12px;
    align-content: start;
    overflow-y: auto;
    padding-right: 4px;
  }
  .vr-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px 14px 10px;
  }
  .vr-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
  .vr-name { font-weight: 500; font-size: 14px; }
  .vr-desc { color: var(--fg2); font-size: 11px; }
  .vr-weight { color: var(--fg3); font-size: 12px; white-space: nowrap; }
  .vr-reward {
    font-size: 26px; font-weight: 500;
    margin: 6px 0 2px; line-height: 1.1;
  }
  .vr-reward .unit { font-size: 12px; color: var(--fg3); margin-left: 6px; }
  .vr-reward .vr-live {
    font-size: 11px; color: var(--fg2); margin-left: 10px;
    padding: 1px 6px; border: 1px solid var(--line); border-radius: 4px;
  }
  .vr-spark { width: 100%; height: 30px; display: block; }
  .vr-counts {
    color: var(--fg2); font-size: 12px; margin-top: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .vr-empty { color: var(--fg3); font-size: 13px; font-style: italic; }

  /* footer */
  .footer {
    border-top: 1px solid var(--line);
    padding-top: 10px;
    color: var(--fg2); font-size: 13px;
    display: flex; gap: 20px; align-items: baseline;
  }
  .footer .sep { color: var(--fg3); }
  .footer .good { color: var(--good); }
  .footer .bad { color: var(--bad); }
</style>
</head>
<body>
  <div class="root">
    <div class="header">
      <div class="title">
        RLVR live<span class="dot">·</span><span class="task mono" id="taskName">composite VR</span>
      </div>
      <div class="update-wrap">
        <span class="update-text mono" id="updateText">update —/—</span>
        <div class="update-bar"><div id="updateBar" style="width:0%"></div></div>
      </div>
    </div>

    <div class="players">
      <strong>P1</strong> · trainee policy<span class="arrow">↔</span><strong>P2</strong> · frozen baseline <span class="dim" id="oppLabel"></span>
    </div>

    <div class="live-strip idle" id="liveStrip">
      <span class="pulse"></span>
      <span class="live-spinner"></span>
      <span id="liveText">waiting for first match…</span>
    </div>

    <div class="telemetry">
      <div class="tel-cell">
        <span class="tel-label">composite Σ per match</span>
        <span class="tel-value" id="telCompositeVal">—</span>
        <svg class="tel-svg" viewBox="0 0 200 22" preserveAspectRatio="none">
          <line id="telCompositeZero" x1="0" y1="11" x2="200" y2="11"
                stroke="var(--fg3)" stroke-width="0.5" stroke-dasharray="2 3"/>
          <path id="telCompositePath" d="" fill="none"
                stroke="var(--good)" stroke-width="1.4"
                vector-effect="non-scaling-stroke"
                stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="tel-cell">
        <span class="tel-label">KL to baseline · per update</span>
        <span class="tel-value" id="telKlVal">—</span>
        <svg class="tel-svg" viewBox="0 0 200 22" preserveAspectRatio="none">
          <path id="telKlPath" d="" fill="none"
                stroke="var(--bad)" stroke-width="1.4"
                vector-effect="non-scaling-stroke"
                stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
      </div>
    </div>

    <div class="vr-grid" id="vrGrid">
      <div class="vr-empty">waiting for the first match…</div>
    </div>

    <div class="footer">
      <span class="mono" id="ftSession">0s</span>
      <span class="sep">·</span>
      <span class="mono" id="ftMatches">0 matches</span>
      <span class="sep">·</span>
      <span class="mono"><span class="good" id="ftWins">0</span>–<span class="bad" id="ftLosses">0</span>–<span id="ftDraws">0</span> W–L–D</span>
      <span class="sep">·</span>
      <span class="mono" id="ftComposite">Σ +0.00</span>
      <span class="sep">·</span>
      <span class="mono" id="ftHealth">kl —</span>
    </div>
  </div>

<script>
// What each VR rewards — shown under the name so the UI is self-explaining.
const VR_DESC = {
  stock_delta:      'stocks won / lost',
  damage_delta:     'damage dealt / taken',
  neutral_win_loss: 'neutral exchanges',
  combo_length:     'combo length',
  low_percent_kill: 'early kills',
  tech:             'tech success',
  recovery:         'recovery success',
};

const state = {
  update: 0,
  max_updates: 50,
  session_start: Date.now() / 1000,
  last_kl: 0,
  opponent_label: '',
  wins: 0, losses: 0, draws: 0, matches: 0,
  vrs: {},
  live: { in_match: false, frame: 0, opp_pct: 0, last_result: '' },
  composite_history: [],   // one value per match
  kl_history: [],          // [{update, kl}]
  _vrFlash: {},            // vid -> { endTime, sign }  (transient flash state)
  _vrPrevLive: {},         // vid -> last seen live.reward (for delta detect)
};

// Trigger a card flash when a VR's live.reward jumped from the last tick.
// Threshold 0.015 — bigger than damage_delta's smallest single-hit reward
// (0.003 × ~3% = 0.009) so noise doesn't constantly flash, but small enough
// that real hits land. 800 ms fade so consecutive events look like a held
// glow rather than rapid pop-pops.
const FLASH_THRESHOLD = 0.015;
const FLASH_MS = 800;
function detectFlashes(newVrs) {
  for (const vid of Object.keys(newVrs)) {
    const newLive = (newVrs[vid].live || {}).reward || 0;
    const oldLive = state._vrPrevLive[vid] || 0;
    const delta = newLive - oldLive;
    if (Math.abs(delta) > FLASH_THRESHOLD) {
      state._vrFlash[vid] = {
        endTime: Date.now() + FLASH_MS,
        sign: delta > 0 ? 'pos' : 'neg',
      };
    }
    state._vrPrevLive[vid] = newLive;
  }
}
function flashStyle(id) {
  const f = state._vrFlash[id];
  if (!f) return '';
  const remain = (f.endTime - Date.now()) / FLASH_MS;
  if (remain <= 0) return '';
  const a = remain;     // 1.0 → 0.0 linear fade
  const rgb = f.sign === 'pos' ? '95, 207, 103' : '168, 82, 88';
  return ` style="border-color: rgba(${rgb}, ${a}); box-shadow: 0 0 18px rgba(${rgb}, ${a * 0.3}), inset 0 0 0 1px rgba(${rgb}, ${a * 0.2})"`;
}

function applySnapshot(s) {
  state.update = s.update || 0;
  state.max_updates = s.max_updates || 50;
  state.session_start = s.session_start || (Date.now() / 1000);
  state.last_kl = s.last_kl || 0;
  state.opponent_label = s.opponent_label || '';
  state.wins = s.wins || 0;
  state.losses = s.losses || 0;
  state.draws = s.draws || 0;
  state.matches = s.matches || 0;
  state.vrs = s.vrs || {};
  if (s.live) state.live = s.live;
  state.composite_history = s.composite_history || [];
  state.kl_history = s.kl_history || [];
}

function applyEvent(ev) {
  if (ev.type === 'snapshot')  { applySnapshot(ev.state); return; }
  if (ev.type === 'frozen')    { state.opponent_label = ev.label || ''; return; }
  if (ev.type === 'update')    {
    state.update = ev.update; state.last_kl = ev.kl;
    if (ev.kl_history) state.kl_history = ev.kl_history;
    return;
  }
  if (ev.type === 'match_end') {
    state.wins = ev.wins; state.losses = ev.losses; state.draws = ev.draws;
    if (ev.live) state.live = ev.live;
    return;
  }
  if (ev.type === 'vr')      {
    state.vrs = ev.vrs || {}; state.matches = ev.matches || 0;
    if (ev.composite_history) state.composite_history = ev.composite_history;
    // Match-end folds live → total; reset our live-delta tracker so the
    // next match's first tick isn't seen as a giant negative jump.
    for (const vid of Object.keys(state.vrs)) state._vrPrevLive[vid] = 0;
    return;
  }
  if (ev.type === 'vr_tick') {
    state.vrs = ev.vrs || {};
    detectFlashes(state.vrs);
    return;
  }
  if (ev.type === 'live')    { if (ev.live) state.live = ev.live; return; }
}

// --- render ---

const $ = id => document.getElementById(id);

function fmtSession(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  return `${m}m ${String(sec).padStart(2, '0')}s`;
}

function fmtCount(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

// Sparkline: per-match reward for one VR, with a zero baseline. Returns
// {line, zeroY, lastX, lastY} in a 100x30 viewBox. The caller appends
// the live in-progress value as the tail point during a match so the
// line grows in real time.
function sparkPaths(vals, w, h) {
  if (!vals || vals.length < 2) {
    return { line: '', zeroY: h, lastX: w, lastY: h };
  }
  let mn = 0, mx = 0;
  for (const v of vals) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const range = (mx - mn) || 1;
  const y = v => h - ((v - mn) / range) * h;
  let line = '';
  let lastX = 0, lastY = h;
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w;
    const yv = y(v);
    line += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + yv.toFixed(1);
    lastX = x; lastY = yv;
  });
  return { line, zeroY: y(0), lastX, lastY };
}

function renderTelemetry() {
  // Composite Σ per match — one point per finished match, zero-anchored.
  {
    const vals = state.composite_history || [];
    const sp = sparkPaths(vals, 200, 22);
    const path = $('telCompositePath');
    const zero = $('telCompositeZero');
    if (path) path.setAttribute('d', sp.line);
    if (zero) {
      zero.setAttribute('y1', sp.zeroY.toFixed(1));
      zero.setAttribute('y2', sp.zeroY.toFixed(1));
    }
    const v = vals.length ? vals[vals.length - 1] : null;
    const cum = vals.reduce((a, b) => a + b, 0);
    $('telCompositeVal').textContent = (v === null)
      ? '—'
      : `last ${v >= 0 ? '+' : ''}${v.toFixed(2)} · Σ ${cum >= 0 ? '+' : ''}${cum.toFixed(1)}`;
  }
  // KL per update — always ≥ 0; no zero baseline needed.
  {
    const vals = (state.kl_history || []).map(p => p.kl);
    const sp = sparkPaths(vals, 200, 22);
    const path = $('telKlPath');
    if (path) path.setAttribute('d', sp.line);
    const v = vals.length ? vals[vals.length - 1] : null;
    $('telKlVal').textContent = (v === null) ? '—' : v.toFixed(4);
  }
}

function vrCard(id, vr) {
  // Live overlay: in-progress match contribution from EVT_VR_TICK. Folded
  // into the displayed numbers so they climb during a match; zeroed at
  // match end (when the per-match value is committed to reward_total).
  const live = vr.live || { reward: 0, counts: {} };
  const liveR = live.reward || 0;
  const cumR  = vr.reward_total || 0;
  const total = cumR + liveR;
  const sign = total > 1e-9 ? 'good' : (total < -1e-9 ? 'bad' : 'fg2');
  const color = sign === 'good' ? 'var(--good)'
              : sign === 'bad'  ? 'var(--bad)' : 'var(--fg2)';
  const desc = VR_DESC[id] || '';
  const weight = (vr.weight !== undefined ? vr.weight : 1);
  // Sparkline values: the completed-match history, plus the in-progress
  // live tail if we're mid-match. Result: the line literally grows
  // every ~2 Hz as the current match accumulates reward, ending at the
  // match-final value when EVT_EP_VR lands.
  const inMatch = (state.live || {}).in_match;
  const sparkVals = inMatch
    ? [...(vr.spark || []), liveR]
    : (vr.spark || []);
  const sp = sparkPaths(sparkVals, 100, 30);
  // Live "now" marker — a thin vertical tick at the live point. Used a
  // line not a circle because the SVG has preserveAspectRatio="none"
  // (so circles get stretched into wide ovals); vector-effect on the
  // stroke keeps the visual width 2px regardless of viewport scaling.
  const liveMarker = (inMatch && sparkVals.length >= 1)
    ? `<line x1="${sp.lastX.toFixed(1)}" y1="${Math.max(0, sp.lastY - 4).toFixed(1)}"
             x2="${sp.lastX.toFixed(1)}" y2="${Math.min(30, sp.lastY + 4).toFixed(1)}"
             stroke="${color}" stroke-width="2"
             vector-effect="non-scaling-stroke" stroke-linecap="round">
         <animate attributeName="stroke-opacity" values="1;0.4;1"
                  dur="1.4s" repeatCount="indefinite"/>
       </line>` : '';
  // Counts: sum cumulative + live, generic over whatever keys this VR exposes.
  const liveC = live.counts || {};
  const cumC  = vr.counts || {};
  const keys  = Array.from(new Set([...Object.keys(cumC), ...Object.keys(liveC)]));
  const counts = keys.map(k => `${k} ${fmtCount((cumC[k] || 0) + (liveC[k] || 0))}`)
                     .join('  ·  ') || '—';
  const sign_str = total >= 0 ? '+' : '';
  // Tiny live-tag when this match's contribution is non-zero.
  const liveTag = Math.abs(liveR) > 1e-9
    ? `<span class="vr-live mono">${liveR >= 0 ? '+' : ''}${liveR.toFixed(2)} live</span>`
    : '';
  return `<div class="vr-card"${flashStyle(id)}>
    <div class="vr-head">
      <div>
        <span class="vr-name">${id}</span>
        <span class="vr-desc">${desc}</span>
      </div>
      <span class="vr-weight mono">w ${weight}</span>
    </div>
    <div class="vr-reward mono" style="color:${color}">${sign_str}${total.toFixed(2)}<span class="unit">reward Σ</span>${liveTag}</div>
    <svg class="vr-spark" viewBox="0 0 100 30" preserveAspectRatio="none">
      <line x1="0" y1="${sp.zeroY.toFixed(1)}" x2="100" y2="${sp.zeroY.toFixed(1)}"
            stroke="var(--fg3)" stroke-width="0.6" stroke-dasharray="2 3"/>
      <path d="${sp.line}" fill="none" stroke="${color}" stroke-width="1.6"
            stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${liveMarker}
    </svg>
    <div class="vr-counts mono">${counts}</div>
  </div>`;
}

function render() {
  // Header.
  const ids = Object.keys(state.vrs);
  $('taskName').textContent = ids.length
    ? `composite VR · ${ids.length} VRs` : 'composite VR';
  $('updateText').textContent = `update ${state.update}/${state.max_updates}`;
  $('updateBar').style.width =
    `${(state.update / Math.max(1, state.max_updates)) * 100}%`;
  $('oppLabel').textContent = state.opponent_label ? `— ${state.opponent_label}` : '';

  // Live in-match pulse / PPO-update spinner. "In PPO" is derived
  // purely from existing state: episodes-since-last-update = matches -
  // update*6; when that hits 6, the 6-episode batch is done and PPO is
  // running. The next `update=N` line resets it.
  const L = state.live || {};
  const strip = $('liveStrip');
  const EP_PER_UPDATE = 6;
  const epsThisUpdate = state.matches - state.update * EP_PER_UPDATE;
  const inPPO = epsThisUpdate >= EP_PER_UPDATE;

  if (inPPO) {
    if (!state._ppoStart) state._ppoStart = Date.now() / 1000;
    strip.classList.remove('idle');
    strip.classList.add('ppo');
    const elapsed = Math.round(Date.now() / 1000 - state._ppoStart);
    $('liveText').textContent =
      `running PPO update ${state.update + 1} · weights updating · ${elapsed}s elapsed`;
  } else {
    state._ppoStart = null;
    strip.classList.remove('ppo');
    if (L.in_match) {
      strip.classList.remove('idle');
      const sec = (L.frame / 60).toFixed(1);
      $('liveText').textContent =
        `match ${epsThisUpdate + 1} of ${EP_PER_UPDATE} this batch · frame ${L.frame} (${sec}s) · opp ${L.opp_pct.toFixed(0)}%`;
    } else if (L.last_result) {
      strip.classList.add('idle');
      $('liveText').textContent =
        `between matches · last result: ${L.last_result} · ${epsThisUpdate}/${EP_PER_UPDATE} this batch`;
    } else {
      strip.classList.add('idle');
      $('liveText').textContent = 'waiting for first match…';
    }
  }

  // Session telemetry — composite Σ + KL trajectories.
  renderTelemetry();

  // VR card grid — high-weight VRs first, then by name.
  const grid = $('vrGrid');
  if (ids.length === 0) {
    if (!grid.querySelector('.vr-empty')) {
      grid.innerHTML = '<div class="vr-empty">waiting for the first match…</div>';
    }
  } else {
    ids.sort((a, b) => {
      const wa = state.vrs[a].weight ?? 1, wb = state.vrs[b].weight ?? 1;
      return wb - wa || a.localeCompare(b);
    });
    grid.innerHTML = ids.map(id => vrCard(id, state.vrs[id])).join('');
  }

  // Footer.
  const now = Date.now() / 1000;
  $('ftSession').textContent = fmtSession(now - state.session_start);
  $('ftMatches').textContent = `${state.matches} matches`;
  $('ftWins').textContent = state.wins;
  $('ftLosses').textContent = state.losses;
  $('ftDraws').textContent = state.draws;
  const composite = ids.reduce((a, id) => a + (state.vrs[id].reward_total || 0), 0);
  $('ftComposite').textContent = `Σ ${composite >= 0 ? '+' : ''}${composite.toFixed(2)}`;
  const kl = state.last_kl;
  const health = kl > 1.0 ? 'drifting' : (kl > 0.2 ? 'elevated' : 'stable');
  $('ftHealth').textContent = `kl ${kl.toFixed(4)} · ${health}`;
}

// SSE.
const sse = new EventSource('/events');
sse.onmessage = e => {
  try { applyEvent(JSON.parse(e.data)); } catch (err) { console.error(err); }
};
sse.onerror = () => console.warn('SSE error');

// Tighter render cadence so the per-card flash fade looks smooth
// (FLASH_MS / render_interval ≈ 8 frames per fade).
setInterval(render, 100);
render();
</script>
</body>
</html>
"""


# ---------- HTTP server -----------------------------------------------


class HUDHandler(BaseHTTPRequestHandler):
    server: "HUDServer"  # set by server class

    def log_message(self, format: str, *args) -> None:  # silence default
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        broker: EventBroker = self.server.broker
        state: HUDState = self.server.state
        # Snapshot first so the page is correct immediately.
        try:
            snap = {"type": "snapshot", "state": state.to_dict()}
            self.wfile.write(b"data: " + json.dumps(snap).encode("utf-8") + b"\n\n")
            self.wfile.flush()
        except Exception:
            return
        q = broker.subscribe()
        try:
            while True:
                try:
                    payload = q.get(timeout=15.0)
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                except queue.Empty:
                    try:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
                except Exception:
                    break
        finally:
            broker.unsubscribe(q)


class HUDServer(ThreadingHTTPServer):
    def __init__(self, addr, state: HUDState, broker: EventBroker):
        super().__init__(addr, HUDHandler)
        self.state = state
        self.broker = broker


# ---------- Main -------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-updates", type=int, default=50,
                    help="Total PPO updates for this run; used for the "
                         "header progress bar.")
    ap.add_argument("--no-open", action="store_true",
                    help="Don't auto-open the browser.")
    args = ap.parse_args()

    state = HUDState()
    state.max_updates = args.max_updates
    broker = EventBroker()
    tailer = LogTailer(args.log, state, broker)
    tailer.start()

    server = HUDServer(("127.0.0.1", args.port), state, broker)
    url = f"http://localhost:{args.port}"
    log.info("training HUD serving at %s  (log=%s)", url, args.log)

    if not args.no_open:
        try:
            webbrowser.open(url, new=0, autoraise=False)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tailer.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
