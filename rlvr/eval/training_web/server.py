"""Web-based training HUD. Stdlib HTTP server + SSE; tails the
training log and streams events to a single self-contained HTML page.

Run alongside training (auto-launched by rlvr/online/loop.py when
training is viewable):

    python3 -m rlvr.eval.training_web.server \
        --log logs/bvb_comboext.log --port 8765

Open http://localhost:8765 in a browser, or add it as an OBS Browser
Source.
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
from typing import Any, Dict, List, Optional


log = logging.getLogger("rlvr.eval.training_web")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)s]  %(message)s")


# ---------- Log parsers (same as rlvr/eval/training_hud.py) -----------

_RE_UPDATE = re.compile(
    r"update=(?P<u>\d+)\s+collected=(?P<c>\d+)\s+valid=(?P<v>\d+)\s+"
    r"reward=(?P<r>[\d\.\-]+)\s+kl=(?P<kl>[\d\.\-]+)\s+"
    r"clip_frac=(?P<cf>[\d\.\-]+)\s+results=(?P<res>\{[^}]*\})"
)
_RE_EVENT = re.compile(
    r"EVT_EP\s+frame=(?P<f>\d+)\s+result=(?P<r>\S+)\s+reward=(?P<rw>[\d\.\-]+)"
    r"(?:\s+damage=(?P<d>[\d\.\-]*))?"
    r"(?:\s+start_pct=(?P<sp>[\d\.\-]*))?"
)
_RE_OPEN = re.compile(
    r"EVT_EP_OPEN\s+frame=(?P<f>\d+)\s+start_pct=(?P<sp>[\d\.\-]*)"
)
_RE_TICK = re.compile(
    r"EVT_EP_TICK\s+frame=(?P<f>\d+)\s+opp_pct=(?P<p>[\d\.\-]+)"
)
_RE_MATCH_END = re.compile(
    r"EVT_MATCH_END\s+result=(?P<r>\S+)\s+"
    r"trainee_stocks=(?P<ts>\d+)\s+opp_stocks=(?P<os>\d+)"
)
_RE_FROZEN = re.compile(r"loading frozen opponent: (\S+)")


# ---------- State + broker --------------------------------------------


class HUDState:
    """Server-side mirror of the dashboard state. Serializable to JSON
    so a new client can be sent a `snapshot` event on connect."""

    def __init__(self):
        self.task_name: str = "combo_extend"
        self.opponent_path: str = ""
        self.opponent_label: str = "frozen model"
        self.update: int = 0
        self.max_updates: int = 50
        self.session_start: float = time.time()
        self.last_kl: float = 0.0

        self.window_open: bool = False
        self.window_open_time: float = 0.0
        self.window_open_frame: int = 0
        self.window_start_pct: float = 0.0
        self.window_trajectory: List[List[float]] = []  # [elapsed_frame, damage_pct]

        self.last_close_time: float = -1e9
        self.last_close_result: str = ""

        self.recent_events: List[Dict[str, Any]] = []
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        # Session totals (across all episodes, not just shown ones).
        self.windows_opened: int = 0
        self.windows_combo: int = 0
        self.windows_single_hit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "opponent_path": self.opponent_path,
            "opponent_label": self.opponent_label,
            "update": self.update,
            "max_updates": self.max_updates,
            "session_start": self.session_start,
            "last_kl": self.last_kl,
            "window_open": self.window_open,
            "window_open_time": self.window_open_time,
            "window_open_frame": self.window_open_frame,
            "window_start_pct": self.window_start_pct,
            "window_trajectory": self.window_trajectory,
            "last_close_time": self.last_close_time,
            "last_close_result": self.last_close_result,
            "recent_events": self.recent_events,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "windows_opened": self.windows_opened,
            "windows_combo": self.windows_combo,
            "windows_single_hit": self.windows_single_hit,
            "server_now": time.time(),
        }


class EventBroker:
    """Thread-safe pub-sub. Each subscriber owns a queue.Queue that
    the server's request handler drains and writes to the SSE
    response."""

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

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        # Wait for the log file to exist.
        while not self.log_path.exists() and not self._stop.is_set():
            time.sleep(0.5)
        if self._stop.is_set():
            return
        with self.log_path.open("r") as fh:
            # Read existing content so we replay it into the state
            # (so the dashboard starts with the right history on a
            # mid-run launch).
            for line in fh:
                self._consume(line, broadcast=False)
            # Then tail.
            while not self._stop.is_set():
                line = fh.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                self._consume(line, broadcast=True)

    # --- event handlers ---

    def _consume(self, line: str, broadcast: bool) -> None:
        m = _RE_FROZEN.search(line)
        if m:
            self._on_frozen(m.group(1), broadcast)
            return
        m = _RE_OPEN.search(line)
        if m:
            self._on_open(m, broadcast)
            return
        m = _RE_TICK.search(line)
        if m:
            self._on_tick(m, broadcast)
            return
        m = _RE_EVENT.search(line)
        if m:
            self._on_close(m, broadcast)
            return
        m = _RE_UPDATE.search(line)
        if m:
            self._on_update(m, broadcast)
            return
        m = _RE_MATCH_END.search(line)
        if m:
            self._on_match_end(m, broadcast)
            return

    def _on_frozen(self, path: str, broadcast: bool) -> None:
        self.state.opponent_path = path
        self.state.opponent_label = f"frozen model · {Path(path).name}"
        if broadcast:
            self.broker.broadcast({
                "type": "frozen",
                "path": path,
                "label": self.state.opponent_label,
            })

    def _on_open(self, m, broadcast: bool) -> None:
        try:
            frame = int(m.group("f"))
        except (TypeError, ValueError):
            frame = 0
        try:
            start_pct = float(m.group("sp") or "0")
        except ValueError:
            start_pct = 0.0
        self.state.window_open = True
        self.state.window_open_time = time.time()
        self.state.window_open_frame = frame
        self.state.window_start_pct = start_pct
        self.state.window_trajectory = [[0, 0.0]]
        self.state.windows_opened += 1
        if broadcast:
            self.broker.broadcast({
                "type": "window_open",
                "open_time": self.state.window_open_time,
                "start_pct": start_pct,
                "frame": frame,
                "windows_opened": self.state.windows_opened,
            })

    def _on_tick(self, m, broadcast: bool) -> None:
        if not self.state.window_open:
            return
        try:
            f = int(m.group("f"))
            p = float(m.group("p"))
        except (TypeError, ValueError):
            return
        elapsed = max(0, f - self.state.window_open_frame)
        damage = max(0.0, p - self.state.window_start_pct)
        self.state.window_trajectory.append([elapsed, damage])
        if broadcast:
            self.broker.broadcast({
                "type": "window_tick",
                "elapsed_frame": elapsed,
                "damage_pct": damage,
            })

    def _on_close(self, m, broadcast: bool) -> None:
        result = m.group("r")
        try:
            reward = float(m.group("rw"))
        except (TypeError, ValueError):
            reward = 0.0
        try:
            damage = float(m.group("d") or "0")
        except (TypeError, ValueError):
            damage = 0.0
        try:
            start_pct = float(m.group("sp") or "0")
        except ValueError:
            start_pct = 0.0
        try:
            close_frame = int(m.group("f"))
        except (TypeError, ValueError):
            close_frame = 0
        # Combo duration = frames the window was open / 60 fps.
        duration_frames = max(0, close_frame - self.state.window_open_frame)
        duration_sec = duration_frames / 60.0
        ev = {
            "type": "window_close",
            "result": result,
            "reward": reward,
            "damage": damage,
            "start_pct": start_pct,
            "close_time": time.time(),
            "duration_sec": duration_sec,
        }
        self.state.window_open = False
        self.state.last_close_time = ev["close_time"]
        self.state.last_close_result = result
        # Only keep the trajectory for one beat post-close (the
        # frontend handles the fade); reset.
        self.state.window_trajectory = []
        # Add to recent_events only if a real extension landed.
        if result in ("combo", "combo_kill"):
            self.state.windows_combo += 1
            self.state.recent_events.insert(0, {
                "t": ev["close_time"],
                "kind": "ko" if result == "combo_kill" else "combo",
                "reward": reward,
                "damage": damage,
                "start_pct": start_pct,
                "duration_sec": duration_sec,
            })
            self.state.recent_events = self.state.recent_events[:5]
        elif result == "single_hit":
            self.state.windows_single_hit += 1
        ev["windows_combo"] = self.state.windows_combo
        ev["windows_single_hit"] = self.state.windows_single_hit
        if broadcast:
            self.broker.broadcast(ev)

    def _on_update(self, m, broadcast: bool) -> None:
        try:
            u = int(m.group("u"))
        except (TypeError, ValueError):
            return
        try:
            kl = float(m.group("kl"))
        except (TypeError, ValueError):
            kl = 0.0
        self.state.update = u
        self.state.last_kl = kl
        if broadcast:
            self.broker.broadcast({
                "type": "update",
                "update": u,
                "kl": kl,
            })

    def _on_match_end(self, m, broadcast: bool) -> None:
        result = m.group("r")
        if result == "win":
            self.state.wins += 1
        elif result == "loss":
            self.state.losses += 1
        else:
            self.state.draws += 1
        if broadcast:
            self.broker.broadcast({
                "type": "match_end",
                "result": result,
                "wins": self.state.wins,
                "losses": self.state.losses,
                "draws": self.state.draws,
            })


# ---------- HTTP server -----------------------------------------------


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
    --fg: #e8e8ea;
    --fg2: #7c7e88;
    --fg3: #3a3c44;
    --line: #1c1e25;
    --bad: #a85258;
    --v0: #441b5f;   /* viridis ~0.05 */
    --v25: #3e497f;
    --v50: #1f968b;
    --v75: #5fcf67;
    --v100: #fde725;
  }
  * { box-sizing: border-box; }
  html, body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    line-height: 1.45;
    margin: 0;
    padding: 0;
    height: 100%;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .dim  { color: var(--fg2); }
  .dim2 { color: var(--fg3); }

  /* layout grid */
  .root {
    display: grid;
    grid-template-rows: auto auto 1fr auto auto;
    height: 100vh;
    padding: 18px 28px;
    gap: 16px;
  }

  /* header */
  .header { display: flex; align-items: baseline; justify-content: space-between; }
  .title { font-size: 18px; font-weight: 500; letter-spacing: 0.02em; }
  .title span.dot { color: var(--fg3); margin: 0 10px; }
  .task { color: var(--fg2); }
  .update-wrap { display: flex; align-items: center; gap: 12px; }
  .update-text { color: var(--fg2); font-size: 13px; }
  .update-bar {
    width: 180px; height: 2px; background: var(--line); border-radius: 1px;
    overflow: hidden;
  }
  .update-bar > div { height: 100%; background: var(--fg2); transition: width 0.4s ease; }

  .players { color: var(--fg2); font-size: 13px; padding-top: 2px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }
  .players strong { color: var(--fg); font-weight: 500; }
  .players .arrow { color: var(--fg3); padding: 0 14px; }

  /* plot panel */
  .plot {
    position: relative;
    padding: 18px 0 8px 0;
    border-bottom: 1px solid var(--line);
    transition: background 0.4s ease;
  }
  .plot.buzz { background: rgba(168, 82, 88, 0.07); }
  .plot.win  { background: rgba(94, 201, 98, 0.06); }

  /* Floating "+0.32" reward badge shown on successful close. */
  .reward-badge {
    position: absolute;
    pointer-events: none;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 22px;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: var(--fg);
    opacity: 0;
    transform: translateY(0);
    transition: opacity 0.45s ease-out, transform 0.45s ease-out;
  }
  .reward-badge.show {
    opacity: 1;
    transform: translateY(-28px);
  }
  .reward-badge .label {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 12px;
    color: var(--fg2);
    margin-left: 8px;
    letter-spacing: 0.04em;
    text-transform: lowercase;
  }

  .plot-head { display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 12px; }
  .plot-status { font-size: 13px; }
  .plot-status .dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--fg3); margin-right: 7px; transform: translateY(-1px);
    transition: background 0.2s ease;
  }
  .plot-status.open .dot { background: var(--v50); animation: pulse 1.6s ease-in-out infinite; }
  .plot-status.bad .dot { background: var(--bad); }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: translateY(-1px) scale(1); }
    50%     { opacity: 0.7; transform: translateY(-1px) scale(1.25); }
  }
  .plot-readout { font-size: 15px; }
  .plot-readout .cur { color: var(--fg); }
  .plot-readout .peak { color: var(--fg2); font-size: 12px; margin-left: 10px; }

  .plot-svg-wrap { width: 100%; }
  .plot-svg { width: 100%; height: 220px; display: block; }
  .gridline { stroke: var(--line); stroke-width: 1; stroke-dasharray: 2 4; }
  .axis-label { fill: var(--fg3); font-size: 11px; }
  .traj { fill: none; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }

  /* events list (no header label per user spec) */
  .events { padding: 8px 0 12px 0; }
  .ev-row {
    display: grid;
    grid-template-columns: 60px 1fr 200px 50px;
    column-gap: 16px;
    align-items: center;
    padding: 5px 0;
    font-size: 13px;
  }
  .ev-row .rew { text-align: right; }
  .ev-bar {
    height: 8px; background: var(--line); border-radius: 4px; overflow: hidden;
    position: relative;
  }
  .ev-bar > div {
    height: 100%; border-radius: 4px;
    transition: width 0.4s ease;
  }
  .ev-desc { color: var(--fg); }
  .ev-age  { color: var(--fg3); text-align: right; font-size: 12px; }
  .ev-empty { color: var(--fg3); padding: 8px 0; font-size: 13px; font-style: italic; }

  /* footer */
  .footer {
    border-top: 1px solid var(--line);
    padding-top: 10px;
    color: var(--fg2);
    font-size: 13px;
    display: flex;
    gap: 22px;
  }
  .footer .sep { color: var(--fg3); }
</style>
</head>
<body>
  <div class="root">
    <div class="header">
      <div class="title">
        RLVR live<span class="dot">·</span><span class="task mono" id="taskName">—</span>
      </div>
      <div class="update-wrap">
        <span class="update-text mono" id="updateText">update —/—</span>
        <div class="update-bar"><div id="updateBar" style="width: 0%"></div></div>
      </div>
    </div>

    <div class="players" id="players">
      <strong>P1</strong> · trainee policy<span class="arrow">↔</span><strong>P2</strong> · frozen model <span class="dim" id="oppLabel"></span>
    </div>

    <div class="plot" id="plotPanel">
      <div class="plot-head">
        <div class="plot-status" id="plotStatus"><span class="dot"></span><span id="plotStatusText">no window open</span></div>
        <div class="plot-readout">
          <span class="cur mono" id="plotCur">—</span>
          <span class="peak mono" id="plotPeak"></span>
        </div>
      </div>
      <div class="plot-svg-wrap" style="position: relative;">
        <svg class="plot-svg" id="plotSvg" viewBox="0 0 1000 220" preserveAspectRatio="none">
          <g id="plotGrid"></g>
          <path id="plotPath" class="traj" d=""></path>
        </svg>
        <div id="rewardBadge" class="reward-badge"></div>
      </div>
    </div>

    <div class="events" id="eventsList">
      <div class="ev-empty">waiting for first extension...</div>
    </div>

    <div class="footer">
      <span class="mono" id="ftSession">0s</span>
      <span class="sep">·</span>
      <span class="mono">wins <span id="ftWins">0</span></span>
      <span class="sep">·</span>
      <span class="mono">losses <span id="ftLosses">0</span></span>
      <span class="sep">·</span>
      <span class="mono">combos <span id="ftCombos">0</span></span>
      <span class="sep">·</span>
      <span class="mono">updates <span id="ftUpdates">0</span></span>
    </div>
  </div>

<script>
const VIRIDIS = [
  [0.00, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.50, [33, 145, 140]],
  [0.75, [94, 201, 98]],
  [1.00, [253, 231, 37]],
];
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < VIRIDIS.length - 1; i++) {
    const [t0, c0] = VIRIDIS[i], [t1, c1] = VIRIDIS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return `rgb(${c0[0]+(c1[0]-c0[0])*f|0}, ${c0[1]+(c1[1]-c0[1])*f|0}, ${c0[2]+(c1[2]-c0[2])*f|0})`;
    }
  }
  return `rgb(${VIRIDIS[VIRIDIS.length-1][1].join(',')})`;
}

const state = {
  task_name: 'combo_extend',
  max_updates: 50,
  update: 0,
  session_start: Date.now() / 1000,
  opponent_label: '',
  window_open: false,
  window_open_time: 0,
  window_open_frame: 0,
  window_start_pct: 0,
  window_trajectory: [],   // [[elapsed_frame, damage_pct], ...]
  last_close_time: 0,
  last_close_result: '',
  recent_events: [],
  wins: 0,
  losses: 0,
  draws: 0,
  windows_opened: 0,
  windows_combo: 0,
  windows_single_hit: 0,
};

function applySnapshot(s) {
  state.task_name = s.task_name;
  state.max_updates = s.max_updates;
  state.update = s.update;
  state.session_start = s.session_start;
  state.opponent_label = s.opponent_label || '';
  state.window_open = s.window_open;
  state.window_open_time = s.window_open_time;
  state.window_open_frame = s.window_open_frame;
  state.window_start_pct = s.window_start_pct;
  state.window_trajectory = s.window_trajectory || [];
  state.last_close_time = s.last_close_time;
  state.last_close_result = s.last_close_result;
  state.recent_events = s.recent_events || [];
  state.wins = s.wins; state.losses = s.losses; state.draws = s.draws;
  state.windows_opened = s.windows_opened || 0;
  state.windows_combo = s.windows_combo || 0;
  state.windows_single_hit = s.windows_single_hit || 0;
}

function applyEvent(ev) {
  if (ev.type === 'snapshot') { applySnapshot(ev.state); return; }
  if (ev.type === 'frozen')   { state.opponent_label = ev.label; return; }
  if (ev.type === 'update')   { state.update = ev.update; return; }
  if (ev.type === 'window_open') {
    state.window_open = true;
    state.window_open_time = ev.open_time;
    state.window_open_frame = ev.frame;
    state.window_start_pct = ev.start_pct;
    state.window_trajectory = [[0, 0]];
    if (ev.windows_opened !== undefined) state.windows_opened = ev.windows_opened;
    return;
  }
  if (ev.type === 'window_tick') {
    if (!state.window_open) return;
    state.window_trajectory.push([ev.elapsed_frame, ev.damage_pct]);
    return;
  }
  if (ev.type === 'window_close') {
    state.window_open = false;
    state.last_close_time = ev.close_time;
    state.last_close_result = ev.result;
    const success = (ev.result === 'combo' || ev.result === 'combo_kill');
    if (success) {
      state.recent_events.unshift({
        t: ev.close_time,
        kind: ev.result === 'combo_kill' ? 'ko' : 'combo',
        reward: ev.reward, damage: ev.damage, start_pct: ev.start_pct,
        duration_sec: ev.duration_sec || 0,
      });
      state.recent_events = state.recent_events.slice(0, 5);
      // Trigger the floating reward badge animation.
      flashRewardBadge(ev.reward, ev.result === 'combo_kill');
    }
    // Hold trajectory for a brief afterimage (longer on success so the
    // viewer's eye lands on the green tint + climbing line).
    state._closing_traj = state.window_trajectory;
    state._closing_until = (Date.now() / 1000) + (success ? 0.55 : 0.45);
    state._closing_success = success;
    state.window_trajectory = [];
    if (ev.windows_combo !== undefined) state.windows_combo = ev.windows_combo;
    if (ev.windows_single_hit !== undefined) state.windows_single_hit = ev.windows_single_hit;
    return;
  }
  if (ev.type === 'match_end') {
    state.wins = ev.wins; state.losses = ev.losses; state.draws = ev.draws;
    return;
  }
}

// --- render ---

const $ = id => document.getElementById(id);

function fmtSession(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2,'0')}m`;
  return `${m}m ${String(sec).padStart(2,'0')}s`;
}

function render() {
  // Header
  $('taskName').textContent = state.task_name;
  $('updateText').textContent = `update ${state.update}/${state.max_updates}`;
  $('updateBar').style.width = `${(state.update / Math.max(1, state.max_updates)) * 100}%`;
  $('oppLabel').textContent = state.opponent_label ? `— ${state.opponent_label.replace(/^frozen model · /, '')}` : '';

  // Plot
  const now = Date.now() / 1000;
  const psStatus = $('plotStatus');
  const buzz = (!state.window_open && state.last_close_result &&
                state.last_close_result !== 'combo' &&
                state.last_close_result !== 'combo_kill' &&
                (now - state.last_close_time) < 0.45);
  const win  = (!state.window_open && state._closing_success &&
                (now - state.last_close_time) < 0.55);
  $('plotPanel').classList.toggle('buzz', buzz);
  $('plotPanel').classList.toggle('win', win);
  psStatus.classList.remove('open', 'bad');
  if (state.window_open) {
    psStatus.classList.add('open');
    const dt = now - state.window_open_time;
    $('plotStatusText').textContent = `window open · ${dt.toFixed(1)}s`;
  } else if (buzz) {
    psStatus.classList.add('bad');
    $('plotStatusText').textContent = `no extension · ${state.last_close_result}`;
  } else if ((now - state.last_close_time) < 0.6 && state.last_close_result) {
    $('plotStatusText').textContent = `closed · ${state.last_close_result}`;
  } else {
    $('plotStatusText').textContent = 'no window open';
  }

  // Pick trajectory to draw.
  let traj;
  let traj_fade = 1.0;
  if (state.window_open) {
    traj = state.window_trajectory;
  } else if (state._closing_traj && now < (state._closing_until || 0)) {
    traj = state._closing_traj;
    const dur = state._closing_success ? 0.55 : 0.45;
    traj_fade = Math.max(0.2, ((state._closing_until || 0) - now) / dur);
  } else {
    traj = [];
  }

  // Y autoscale on damage.
  let yMax = 25;
  if (traj.length > 0) {
    const peak = traj.reduce((a, p) => Math.max(a, p[1]), 0);
    yMax = Math.max(25, Math.ceil((peak + 1) / 25) * 25);
  }
  // X: pad to 90 frames or actual.
  const xMax = traj.length ? Math.max(traj[traj.length-1][0], 90) : 90;

  // Gridlines.
  const grid = $('plotGrid');
  grid.innerHTML = '';
  const W = 1000, H = 220, padL = 50, padR = 14, padT = 8, padB = 18;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const nLines = yMax / 25 + 1;
  for (let i = 0; i < nLines; i++) {
    const y = padT + innerH - (i * 25 / yMax) * innerH;
    grid.insertAdjacentHTML('beforeend',
      `<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" class="gridline"/>`);
    grid.insertAdjacentHTML('beforeend',
      `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" class="axis-label mono">${i*25}</text>`);
  }

  // Trajectory path.
  const path = $('plotPath');
  if (traj.length >= 1) {
    const peak = traj.reduce((a, p) => Math.max(a, p[1]), 0);
    const color = viridis(Math.min(1, peak / 100));
    let d = '';
    for (let i = 0; i < traj.length; i++) {
      const [ef, dmg] = traj[i];
      const x = padL + (ef / xMax) * innerW;
      const y = padT + innerH - Math.min(1, dmg / yMax) * innerH;
      d += (i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`);
    }
    path.setAttribute('d', d);
    path.setAttribute('stroke', color);
    path.style.opacity = traj_fade;
  } else {
    path.setAttribute('d', '');
  }

  // Readout.
  if (traj.length) {
    const cur = traj[traj.length - 1][1];
    const peak = traj.reduce((a, p) => Math.max(a, p[1]), 0);
    $('plotCur').textContent = `+${cur.toFixed(0)}% dealt`;
    $('plotPeak').textContent = `peak ${peak.toFixed(0)}`;
  } else {
    $('plotCur').textContent = '';
    $('plotPeak').textContent = '';
  }

  // Events list.
  const list = $('eventsList');
  if (state.recent_events.length === 0) {
    if (!list.querySelector('.ev-empty')) {
      list.innerHTML = `<div class="ev-empty">waiting for first extension...</div>`;
    }
  } else {
    let html = '';
    for (const ev of state.recent_events) {
      const color = viridis(Math.min(1, ev.reward));
      const width = (Math.max(0, Math.min(1, ev.reward)) * 100).toFixed(0);
      const desc = ev.kind === 'ko'
        ? `KO from ${ev.start_pct.toFixed(0)}%`
        : `combo · ${ev.damage.toFixed(0)}% damage`;
      const dur = (ev.duration_sec || 0).toFixed(1);
      html += `<div class="ev-row">
        <span class="rew mono">+${ev.reward.toFixed(2)}</span>
        <div class="ev-bar"><div style="width:${width}%; background:${color};"></div></div>
        <div class="ev-desc">${desc}</div>
        <span class="ev-age mono">${dur}s</span>
      </div>`;
    }
    list.innerHTML = html;
  }

  // Footer.
  $('ftSession').textContent = fmtSession(now - state.session_start);
  $('ftWins').textContent = state.wins;
  $('ftLosses').textContent = state.losses;
  $('ftCombos').textContent = state.windows_combo;
  $('ftUpdates').textContent = state.update;
}

// Reward badge animation. Positions the badge at the right end of the
// last-known trajectory point, then fades up + out.
function flashRewardBadge(reward, isKO) {
  const badge = $('rewardBadge');
  badge.classList.remove('show');
  // Force layout flush so the transition restarts.
  void badge.offsetWidth;
  const label = isKO ? 'KO' : 'combo';
  const color = viridis(Math.min(1, reward));
  badge.style.color = color;
  badge.innerHTML = `+${reward.toFixed(2)}<span class="label">${label}</span>`;
  // Place near top-right of plot (above the latest point).
  badge.style.right = '12px';
  badge.style.top = '10px';
  badge.classList.add('show');
  // Schedule un-show (CSS transitions back) so it can re-fire.
  setTimeout(() => badge.classList.remove('show'), 450);
}

// SSE.
const sse = new EventSource('/events');
sse.onmessage = e => {
  try { applyEvent(JSON.parse(e.data)); } catch (err) { console.error(err); }
};
sse.onerror = () => console.warn('SSE error');

// Render loop.
setInterval(render, 60);
render();
</script>
</body>
</html>
"""


class HUDHandler(BaseHTTPRequestHandler):
    server: "HUDServer"  # set by server class
    _logged_paths: set = set()

    def log_message(self, format: str, *args) -> None:  # silence default
        pass

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
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
                    # Heartbeat to keep the proxy / browser happy.
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
    ap.add_argument("--no-open", action="store_true",
                    help="Don't auto-open the browser.")
    args = ap.parse_args()

    state = HUDState()
    broker = EventBroker()
    tailer = LogTailer(args.log, state, broker)
    tailer.start()

    server = HUDServer(("127.0.0.1", args.port), state, broker)
    url = f"http://localhost:{args.port}"
    log.info("training HUD serving at %s  (log=%s)", url, args.log)

    if not args.no_open:
        # Best-effort browser launch.
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
