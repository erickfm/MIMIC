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
    r"EVT_MATCH_END\s+result=(?P<r>\S+)\s+"
    r"trainee_stocks=(?P<ts>\d+)\s+opp_stocks=(?P<os>\d+)"
)
_RE_FROZEN = re.compile(r"loading frozen opponent: (\S+)")
_EVT_VR = "EVT_EP_VR "


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
        # vr_id -> {weight, reward_total, counts: {k: n}, spark: [per-match r]}
        self.vrs: Dict[str, Dict[str, Any]] = {}

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
        idx = line.find(_EVT_VR)
        if idx != -1:
            self._on_vr(line[idx + len(_EVT_VR):], broadcast)
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
        if broadcast:
            self.broker.broadcast({"type": "update", "update": u, "kl": kl})

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
                "wins": self.state.wins,
                "losses": self.state.losses,
                "draws": self.state.draws,
            })

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
        for vid, d in vrs.items():
            if not isinstance(d, dict):
                continue
            entry = self.state.vrs.get(vid)
            if entry is None:
                entry = {"weight": 1.0, "reward_total": 0.0,
                         "counts": {}, "spark": []}
                self.state.vrs[vid] = entry
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
            for k, v in d.items():
                if k in ("weight", "reward") or isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    entry["counts"][k] = round(
                        entry["counts"].get(k, 0) + v, 1)
        if broadcast:
            self.broker.broadcast({
                "type": "vr",
                "vrs": self.state.vrs,
                "matches": self.state.matches,
            })


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
    grid-template-rows: auto auto 1fr auto;
    height: 100vh;
    padding: 18px 28px;
    gap: 14px;
  }

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
};

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
}

function applyEvent(ev) {
  if (ev.type === 'snapshot')  { applySnapshot(ev.state); return; }
  if (ev.type === 'frozen')    { state.opponent_label = ev.label || ''; return; }
  if (ev.type === 'update')    { state.update = ev.update; state.last_kl = ev.kl; return; }
  if (ev.type === 'match_end') {
    state.wins = ev.wins; state.losses = ev.losses; state.draws = ev.draws;
    return;
  }
  if (ev.type === 'vr') { state.vrs = ev.vrs || {}; state.matches = ev.matches || 0; return; }
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
// {line, zeroY} in a 100x30 viewBox.
function sparkPaths(vals, w, h) {
  if (!vals || vals.length < 2) return { line: '', zeroY: h };
  let mn = 0, mx = 0;
  for (const v of vals) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const range = (mx - mn) || 1;
  const y = v => h - ((v - mn) / range) * h;
  let line = '';
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w;
    line += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y(v).toFixed(1);
  });
  return { line, zeroY: y(0) };
}

function vrCard(id, vr) {
  const total = vr.reward_total || 0;
  const sign = total > 1e-9 ? 'good' : (total < -1e-9 ? 'bad' : 'fg2');
  const color = sign === 'good' ? 'var(--good)'
              : sign === 'bad'  ? 'var(--bad)' : 'var(--fg2)';
  const desc = VR_DESC[id] || '';
  const weight = (vr.weight !== undefined ? vr.weight : 1);
  const sp = sparkPaths(vr.spark || [], 100, 30);
  const counts = Object.entries(vr.counts || {})
    .map(([k, v]) => `${k} ${fmtCount(v)}`).join('  ·  ') || '—';
  const sign_str = total >= 0 ? '+' : '';
  return `<div class="vr-card">
    <div class="vr-head">
      <div>
        <span class="vr-name">${id}</span>
        <span class="vr-desc">${desc}</span>
      </div>
      <span class="vr-weight mono">w ${weight}</span>
    </div>
    <div class="vr-reward mono" style="color:${color}">${sign_str}${total.toFixed(2)}<span class="unit">reward Σ</span></div>
    <svg class="vr-spark" viewBox="0 0 100 30" preserveAspectRatio="none">
      <line x1="0" y1="${sp.zeroY.toFixed(1)}" x2="100" y2="${sp.zeroY.toFixed(1)}"
            stroke="var(--fg3)" stroke-width="0.6" stroke-dasharray="2 3"/>
      <path d="${sp.line}" fill="none" stroke="${color}" stroke-width="1.6"
            stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
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

setInterval(render, 250);
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
