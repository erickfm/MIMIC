import { readIso, keepFiles, keptFiles, forgetFiles } from "/iso.js";
import { PadReader, KEYS, samePad } from "/pad.js";

const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
const api = async (path, body) => {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmtTime = (f) => { const s = Math.max(0, f) / 60; return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
const fmtTimeTenth = (f) => { const t = Math.max(0, f) / 60; return `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`; };
const fmtTimeFine = (f) => { const s = Math.max(0, f) / 60; return `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`; };
const fmtDate = (iso) => { if (!iso) return ""; const d = new Date(iso); return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " · " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }); };
function toast(msg, ms = 2600) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => (t.hidden = true), ms); }
// " · about 40 s left", from how fast the job has gone since it started moving (model loading and
// queueing don't count); nothing until there's enough to go on
function timer() {
  let t1 = null, p1 = 0, pLast = null, txt = "";
  return (p) => {
    if (!(p > 0) || p >= 1) return "";
    if (p === pLast) return txt;    // re-estimate only when progress moves (it comes in steps)
    pLast = p;
    const now = performance.now();
    if (t1 === null) { t1 = now; p1 = p; return (txt = ""); }
    const el = (now - t1) / 1000, dp = p - p1;
    if (el < 4 || dp < 0.05) return (txt = "");
    const r = el * (1 - p) / dp;
    return (txt = r < 10 ? " · a few seconds left" : r < 90 ? ` · about ${Math.round(r / 5) * 5} s left` : ` · about ${Math.round(r / 60)} min left`);
  };
}
// live play: before the moment the game replays as it was; the countdown says when the player takes over
const readyMsg = (frames) => `Get ready… ${(frames / 60).toFixed(1)} s`;
const crumbs = (html) => ($("#crumbs").innerHTML = html);

let gameFiles = null;   // File[] from an ISO picked in this browser (only when the server has none)
let serverIso = false; // the server reads the game files from Slippi Launcher's ISO
// the viewer: game files from the server when it has the ISO, else handed over from this browser
const viewerSrc = (you) => `/viewer/embed.html?mini=1&controllers=off&hudslots=game&you=${you}` + (serverIso ? "&files=/gamefiles/" : "&renderer=svg");
const GHOST_ALPHA = 0.45;   // ghost opacity in the 3D view (the viewer's default is 0.9)
function feedViewer(iframe, send, url, name, frame) {
  try { const st = iframe.contentDocument.createElement("style"); st.textContent = "html,body,.bg-white{background:#05070b!important}"; iframe.contentDocument.head.appendChild(st); } catch {}
  if (!serverIso && gameFiles) send({ type: "gameFiles", files: gameFiles });
  send({ type: "load", url, name, frame });
  send({ type: "ghostAlpha", alpha: GHOST_ALPHA });
  if (!serverIso && gameFiles) setTimeout(() => send({ type: "renderer", renderer: "3d" }), 50);
}

// ── routing ──────────────────────────────────────────────────────────────────────────────
async function route() {
  cleanup?.(); cleanup = null;
  const h = location.hash.slice(1) || "/";
  if (gameFiles === null) gameFiles = (await keptFiles().catch(() => null)) || false;
  const st = await api("/api/status").catch(() => null);
  if (!st) { view.innerHTML = `<div class="setup"><h1>Can't reach the coach</h1><p class="lead">The coach program isn't running. Start it and reload this page.</p></div>`; return; }
  serverIso = !!st.iso?.ok;
  if (h.startsWith("/setup") || !(serverIso || gameFiles) || !st.me || !st.replay_dir_exists) return setup(st);
  try {
    const lm = h.match(/^\/lab\/(\w+)(?:\?f=(-?\d+))?/);
    if (lm) return await lab(lm[1], lm[2] != null ? +lm[2] : null);
    if (h.startsWith("/patterns")) return await patterns(st);
    const m = h.match(/^\/game\/(\w+)(?:\?f=(-?\d+))?/);
    if (m) return await game(m[1], st, m[2] != null ? +m[2] : null);
    return await library(st);
  } catch (e) {
    view.innerHTML = `<div class="setup"><h1>Couldn't open this page</h1><p class="lead">${esc(e.message)}</p><a class="btn" href="#/">Back to your games</a></div>`;
  }
}
let cleanup = null;
window.addEventListener("hashchange", route);

// ── setup ────────────────────────────────────────────────────────────────────────────────
async function setup(st) {
  crumbs("<b>Setup</b>");
  view.innerHTML = `
  <div class="setup">
    <h1>Set up the coach</h1>
    <p class="lead">Checked once. Nothing leaves your computer.</p>
    <div class="step" id="s1"><h2><span class="n">1</span>Your Melee disc</h2>
      <p id="isoLead">The replay viewer draws the game with its own models, read from your copy of Melee (NTSC 1.02 ISO).</p>
      <div class="drop" id="drop"><div id="isoMsg">Drop your Melee ISO here, or</div>
        <div style="margin-top:10px"><label class="btn"><input type="file" id="isoInput" accept=".iso,.gcm" hidden>Choose ISO…</label></div>
        <div class="bar" id="isoBar" hidden style="margin-top:14px"><i></i></div></div>
      <div class="err" id="isoErr"></div>
    </div>
    <div class="step" id="s2"><h2><span class="n">2</span>Your replays</h2>
      <p>The folder where Slippi saves your games.</p>
      <div class="row"><input type="text" id="dir" value="${esc(st.replay_dir)}"><button class="btn" id="dirSave">Use this folder</button></div>
      <div class="dim" id="dirMsg" style="margin-top:8px"></div>
      <div class="err" id="dirErr"></div>
      <div id="who" style="margin-top:12px"></div>
    </div>
    <div class="row" style="justify-content:flex-end;margin-top:18px"><button class="btn primary" id="go" disabled>Open my games</button></div>
  </div>`;
  const s1 = $("#s1"), s2 = $("#s2");
  const refresh = async () => {
    const s = await api("/api/status");
    serverIso = !!s.iso?.ok;
    s1.classList.toggle("ok", serverIso || !!gameFiles);
    if (serverIso) {
      $("#isoLead").textContent = `Using the Melee ISO ${s.iso.source === "Slippi Launcher" ? "Slippi Launcher plays" : "set up on this computer"}. Nothing to do here.`;
      $("#drop").hidden = true;
    } else if (gameFiles) $("#isoMsg").textContent = `Melee files ready (${gameFiles.length} files). Drop another ISO to replace them.`;
    s2.classList.toggle("ok", s.replay_dir_exists && s.games > 0 && !!s.me);
    $("#dirMsg").textContent = !s.replay_dir_exists ? "That folder doesn't exist."
      : s.scanning ? "Looking through your replays…"
      : s.games ? `Using ${s.replay_dir}: ${s.games} games found.`
      : "No Slippi replays in this folder. Slippi Launcher shows where it saves them under Settings → Replays.";
    if (!s.codes.length || s.scanning) $("#who").innerHTML = "";
    else {
      $("#who").innerHTML = `<div class="row"><span class="dim" style="flex:none">You are</span><select id="me">${s.codes.map(([c, n]) => `<option value="${esc(c)}" ${c === s.me ? "selected" : ""}>${esc(c)} — ${n} games</option>`).join("")}</select></div>`;
      $("#me").onchange = async (e) => { await api("/api/config", { me: e.target.value }); refresh(); };
    }
    $("#go").disabled = !((serverIso || gameFiles) && s.replay_dir_exists && s.games > 0 && s.me);
    if (s.scanning) setTimeout(refresh, 700);
  };
  const takeIso = async (file) => {
    $("#isoErr").textContent = "";
    try {
      const list = await readIso(file);
      $("#isoBar").hidden = false; $("#isoMsg").textContent = "Keeping the files the viewer needs…";
      await keepFiles(list, (f) => ($("#isoBar").firstChild.style.width = `${(f * 100).toFixed(0)}%`));
      gameFiles = await keptFiles();
      $("#isoBar").hidden = true;
      refresh();
    } catch (e) { $("#isoErr").textContent = e.message; $("#isoBar").hidden = true; }
  };
  $("#isoInput").onchange = (e) => e.target.files[0] && takeIso(e.target.files[0]);
  const drop = $("#drop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; f && takeIso(f); };
  $("#dirSave").onclick = async () => {
    $("#dirErr").textContent = "";
    try { await api("/api/config", { replay_dir: $("#dir").value }); } catch (e) { $("#dirErr").textContent = e.message; }
    refresh();
  };
  $("#go").onclick = () => { location.hash = "#/"; };
  refresh();
}

// ── library ──────────────────────────────────────────────────────────────────────────────
async function library(st) {
  crumbs(`<b>Your games</b>`);
  view.innerHTML = `<div class="lib"><div class="empty">Loading your games…</div></div>`;
  const d = await api("/api/games");
  const games = d.games;
  const chars = [...new Set(games.map((g) => g.me?.char_name).filter(Boolean))];
  let filter = sessionStorage.getItem("libFilter") || "all";
  const draw = () => {
    const list = games.filter((g) => filter === "all" || g.me?.char_name === filter);
    const ok = list.filter((g) => !g.unsupported).length;
    view.innerHTML = `<div class="lib">
      <div class="libhead"><div><h1>Your games</h1><div class="dim">${esc(d.me)} · ${list.length} games · ${ok} ready to review</div></div><div class="grow"></div><a class="btn" href="#/patterns">Your patterns</a></div>
      <div class="chips">${["all", ...chars].map((c) => `<button class="chip ${c === filter ? "on" : ""}" data-c="${esc(c)}">${c === "all" ? "All" : esc(c)}</button>`).join("")}</div>
      <div class="games">${list.slice(0, 400).map(row).join("") || `<div class="empty">No games here.</div>`}</div></div>`;
    view.querySelectorAll(".chip").forEach((b) => (b.onclick = () => { filter = b.dataset.c; sessionStorage.setItem("libFilter", filter); draw(); }));
  };
  const row = (g) => `<a class="game ${g.unsupported ? "off" : ""}" ${g.unsupported ? `title="${esc(g.unsupported)}"` : `href="#/game/${g.id}"`}>
      <div class="when">${esc(fmtDate(g.start))}</div>
      <div class="who"><b>${esc(g.me?.char_name)}</b><span class="vs">vs</span><b>${esc(g.opp?.char_name)}</b> <span class="faint">· ${esc(g.opp?.code || g.opp?.name || "")}</span>${g.unsupported ? `<div class="faint" style="font-size:12px">${esc(g.unsupported)}</div>` : ""}</div>
      <div class="stage dim">${esc(g.stage)}</div>
      <div class="len dim mono">${g.frames ? fmtTime(g.frames) : ""}</div>
      <div>${g.result ? `<span class="pill ${g.result}">${g.result === "win" ? "Win" : "Loss"}</span>` : ""}</div></a>`;
  draw();
}

// ── game view ────────────────────────────────────────────────────────────────────────────
async function game(id, st, startAt = null) {
  view.innerHTML = `<div class="lib"><div class="empty">Loading the game…</div></div>`;
  const g = await api(`/api/game/${id}`);
  const s = g.summary;
  crumbs(`<a href="#/">Your games</a> › <b>${esc(s.me.char_name)} vs ${esc(s.opp.char_name)}</b> <span class="faint">· ${esc(s.stage)} · ${esc(fmtDate(s.start))}</span>`);
  if (s.unsupported) { view.innerHTML = `<div class="setup"><h1>Can't review this game</h1><p class="lead">${esc(s.unsupported)}</p><a class="btn" href="#/">Back to your games</a></div>`; return; }
  const youPort = s.me.port + 1;
  view.innerHTML = `
  <div class="gameview">
    <div class="stagecol">
      <div class="screen"><iframe id="player" src="${viewerSrc(youPort)}" allow="fullscreen"></iframe><div class="over" id="over">Loading the viewer…</div></div>
      <div class="controls">
        <button class="btn icon" id="back10" title="Back 10 frames (Shift+←)">«</button>
        <button class="btn icon" id="back1" title="Back 1 frame (←)">‹</button>
        <button class="btn icon" id="play" title="Play / pause (Space)">▶</button>
        <button class="btn icon" id="fwd1" title="Forward 1 frame (→)">›</button>
        <button class="btn icon" id="fwd10" title="Forward 10 frames (Shift+→)">»</button>
        <button class="btn" id="speed" title="Playback speed">1×</button>
        <div class="clock" id="clock"></div>
        <div class="grow"></div>
        <button class="btn" id="toLab" title="Set up this moment however you like and play it out">Open in Lab</button>
        <button class="btn" id="prevDp" title="Previous decision ([)">‹ Decision</button>
        <button class="btn" id="nextDp" title="Next decision (])">Decision ›</button>
      </div>
      <div class="timeline" id="tl"><canvas id="tlc"></canvas></div>
      <div class="legend" id="legend"><span><i style="background:var(--me)"></i>Your percent</span><span><i style="background:var(--opp)"></i>Opponent's percent</span><span><i style="background:#8b93a7;height:8px;width:2px"></i>Your decisions</span><span id="revState" class="faint"></span></div>
    </div>
    <aside class="side">
      <div class="sec">
        <h3>This moment</h3>
        <div class="moment" id="moment"></div>
        <button class="btn primary ask" id="ask">What would a master do? <kbd style="color:#06222c;border-color:#0a5b73">M</kbd></button>
        <button class="btn tryit" id="tryIt" title="Play this moment yourself against the bot, from the exact game state">Try it yourself <kbd>T</kbd></button>
        <div class="progress" id="prog" hidden><div class="dim" id="progMsg"></div><div class="bar"><i></i></div></div>
        <div class="hint"><span><kbd>Space</kbd> play</span><span><kbd>←</kbd><kbd>→</kbd> frame</span><span><kbd>[</kbd><kbd>]</kbd> decisions</span></div>
      </div>
      <div class="opts" id="opts"><div class="dim" style="padding:6px 4px">Pause on a moment, or pick one from Biggest swings below, then ask what a master would do. You'll see what the master most likely does from that exact spot, in slow motion over your game.</div></div>
      <div class="sec" id="revSec" hidden><h3 id="revHead">Biggest swings</h3><div id="revList" class="revlist"></div></div>
    </aside>
  </div>`;
  const frame0 = g.series.frame0, last = g.last;
  const pts = g.points.map((p) => p.frame);
  let cur = startAt ?? g.points[0]?.frame ?? 0, running = false, ready = false, branch = null, focus = null, hidden = new Set(), review = null, alive = true;
  let back = null, backRow = null, backEx = 0;   // "could you have avoided this spot?" result and the row being watched
  const iframe = $("#player");
  const send = (m) => iframe.contentWindow?.postMessage(m, "*");
  const replayUrl = `/api/game/${id}/replay.slp`;

  const onMsg = (e) => {
    if (e.source !== iframe.contentWindow) return;
    const m = e.data || {};
    if (m.type === "frame") {
      cur = m.frame; running = m.running; tick();
      if (!live.on() && loop && running && cur >= loop[1]) { seek(loop[0]); send({ type: "play" }); }
      live.frame();
    }
    else if (m.type === "loaded") { ready = true; $("#over").hidden = true; send({ type: "ghostAlpha", alpha: GHOST_ALPHA }); live.loaded(); }
    else if (m.type === "renderer") {
      const bad = m.renderer === "3d" && m.reason;
      $("#over").hidden = !bad;
      if (bad) $("#over").textContent = `The 3D view can't run: ${m.reason}`;
    }
  };
  window.addEventListener("message", onMsg);
  iframe.addEventListener("load", () => feedViewer(iframe, send, replayUrl, s.file, cur));

  // timeline
  const tl = $("#tl"), cv = $("#tlc"), ctx = cv.getContext("2d");
  const maxPct = Math.max(120, ...g.series.me_pct, ...g.series.opp_pct);
  const fx = (f, w) => ((f - frame0) / (last - frame0)) * w;
  function drawTl() {
    const r = tl.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    cv.width = r.width * dpr; cv.height = r.height * dpr;
    const w = cv.width, h = cv.height, pad = 6 * dpr, top = 8 * dpr, bot = h - 18 * dpr;
    ctx.clearRect(0, 0, w, h);
    // stock-loss bands
    const stockLine = (arr, col) => { for (let i = 1; i < arr.length; i++) if (arr[i] < arr[i - 1]) { ctx.fillStyle = col; ctx.fillRect(fx(frame0 + i, w) - dpr, top, 2 * dpr, bot - top); } };
    stockLine(g.series.me_stocks, "rgba(76,201,240,.35)"); stockLine(g.series.opp_stocks, "rgba(242,95,92,.35)");
    // branch window
    if (branch?.frame != null) { ctx.fillStyle = "rgba(76,201,240,.10)"; ctx.fillRect(fx(branch.frame, w), 0, fx(branch.frame + (branch.horizon || 300), w) - fx(branch.frame, w), h); }
    const line = (arr, col) => { ctx.strokeStyle = col; ctx.lineWidth = 1.6 * dpr; ctx.beginPath(); const step = Math.max(1, Math.floor(arr.length / (w / dpr))); for (let i = 0; i < arr.length; i += step) { const x = fx(frame0 + i, w), y = bot - (arr[i] / maxPct) * (bot - top); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); };
    if (review?.eval) {
      // expected score over the next 5 s (the value model): above the middle = going your way
      const mid = (top + bot) / 2, span = (bot - top) / 2, cap = 40;
      const ev = review.eval;
      const y = (v) => mid - Math.max(-1, Math.min(1, v / cap)) * span;
      for (const [sign, col] of [[1, "rgba(87,214,141,.30)"], [-1, "rgba(242,95,92,.30)"]]) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(fx(ev[0][0], w), mid);
        for (const [f, v] of ev) ctx.lineTo(fx(f, w), sign > 0 ? Math.min(mid, y(v)) : Math.max(mid, y(v)));
        ctx.lineTo(fx(ev[ev.length - 1][0], w), mid); ctx.closePath(); ctx.fill();
      }
      ctx.fillStyle = "rgba(255,255,255,.12)"; ctx.fillRect(0, mid, w, dpr);
      ctx.globalAlpha = 0.45; line(g.series.opp_pct, "#f25f5c"); line(g.series.me_pct, "#4cc9f0"); ctx.globalAlpha = 1;
      for (const p of review.points) if (p.tag) {
        ctx.fillStyle = p.tag === "costly" ? "#f25f5c" : "#57d68d";
        ctx.beginPath(); ctx.arc(fx(p.frame, w), top + 4 * dpr, 3.6 * dpr, 0, 7); ctx.fill();
      }
    } else { line(g.series.opp_pct, "#f25f5c"); line(g.series.me_pct, "#4cc9f0"); }
    // decision ticks
    ctx.fillStyle = "rgba(139,147,167,.55)";
    for (const f of pts) ctx.fillRect(fx(f, w), bot + 4 * dpr, Math.max(1, dpr * .8), 9 * dpr);
    // playhead
    const x = fx(cur, w);
    ctx.fillStyle = "#fff"; ctx.fillRect(x - dpr, 0, 2 * dpr, h);
  }
  const seekFromX = (cx) => { const r = tl.getBoundingClientRect(); const f = Math.round(frame0 + ((cx - r.left) / r.width) * (last - frame0)); seek(Math.max(frame0, Math.min(last, f))); };
  let drag = false;
  tl.onpointerdown = (e) => { loop = null; drag = true; tl.setPointerCapture(e.pointerId); send({ type: "pause" }); seekFromX(e.clientX); };
  tl.onpointermove = (e) => drag && seekFromX(e.clientX);
  tl.onpointerup = () => (drag = false);
  const ro = new ResizeObserver(drawTl); ro.observe(tl);

  function costOf(f) {
    const p = review?.points.find((x) => x.frame === f);
    if (!p) return "";
    const c = p.cost, cls = c <= -8 ? "neg" : c >= 8 ? "pos" : "zero";
    return ` <span class="net ${cls}" title="How this choice changed your expected score over the next 5 seconds">${c > 0 ? "+" : ""}${c.toFixed(0)}</span>`;
  }
  function nearestPoint(f) { let best = null; for (const p of g.points) { if (p.frame <= f) best = p; else break; } return best; }
  function tick() {
    $("#clock").innerHTML = `<b>${fmtTimeFine(cur)}</b> <span class="faint">frame ${cur}</span>`;
    $("#play").textContent = running ? "❚❚" : "▶";
    // while a try is on repeat, this panel stays on the moment you asked about
    const watching = loop && branch && backRow == null;
    const ref = watching ? branch.frame : cur;
    const p = nearestPoint(ref);
    const at = p && ref - p.frame < p.wait + 30 ? p : null;
    const head = watching ? `Watching the tries from ${fmtTimeFine(branch.frame)}: ${cur < branch.frame ? `starts in ${((branch.frame - cur) / 60).toFixed(2)} s` : `${((cur - branch.frame) / 60).toFixed(2)} s in`}.`
      : at ? (cur === at.frame ? "You could act here." : `${((cur - at.frame) / 60).toFixed(2)} s after you could act.`) : null;
    $("#moment").innerHTML = at
      ? `${head}<div style="margin-top:4px">You did: <span class="did">${esc(at.did)}</span>${watching ? "" : costOf(at.frame)}</div>`
      : `<span class="dim">${head || `${fmtTimeFine(cur)} into the game.`}</span>`;
    drawTl();
  }
  function seek(f) { cur = f; send({ type: "seek", frame: f }); tick(); }
  const step = (d) => { send({ type: "pause" }); seek(Math.max(frame0, Math.min(last, cur + d))); };
  $("#play").onclick = () => send({ type: "toggle" });
  $("#back1").onclick = () => step(-1); $("#fwd1").onclick = () => step(1);
  $("#back10").onclick = () => step(-10); $("#fwd10").onclick = () => step(10);
  const jumpDp = (dir) => { loop = null; send({ type: "pause" }); const f = dir > 0 ? pts.find((x) => x > cur) : [...pts].reverse().find((x) => x < cur); if (f != null) seek(f); };
  $("#prevDp").onclick = () => jumpDp(-1); $("#nextDp").onclick = () => jumpDp(1);
  $("#toLab").onclick = () => { send({ type: "pause" }); location.hash = `#/lab/${id}?f=${cur}`; };

  // ask the model
  // speed: the viewer runs at `fps` game frames per second (60 = normal)
  const SPEEDS = [[60, "1×"], [30, "½×"], [15, "¼×"]];
  let speed = 60;
  function setSpeed(fps) {
    speed = fps;
    // the viewer reads the speed when playback starts: restart it if it's playing
    if (running) { send({ type: "pause" }); send({ type: "speed", fps }); send({ type: "play" }); } else send({ type: "speed", fps }); const b = $("#speed"); if (b) b.textContent = SPEEDS.find((x) => x[0] === fps)[1]; }
  $("#speed").onclick = () => { const i = SPEEDS.findIndex((x) => x[0] === speed); setSpeed(SPEEDS[(i + 1) % SPEEDS.length][0]); };
  let loop = null;     // [start, end] while watching a try: play it on repeat
  async function ask(at) {
    const f = typeof at === "number" ? at : cur;
    send({ type: "pause" });
    $("#ask").disabled = true; $("#prog").hidden = false; $("#progMsg").textContent = "Starting…";
    $("#prog .bar i").style.width = "0%";
    try {
      const job = await api("/api/branch", { game: id, frame: f, k: 16 });
      let j = job; const left = timer();
      while (j.state !== "done" && j.state !== "error") {
        await new Promise((r) => setTimeout(r, 300));
        j = await api(`/api/job/${job.id}`);
        $("#progMsg").textContent = j.state === "queued" ? "Waiting for the simulator…" : j.state === "loading" ? "Loading the models (first time only)…" : `Trying it ${j.k} ways from this exact moment…${left(j.progress)}`;
        $("#prog .bar i").style.width = `${Math.round((j.progress || 0) * 100)}%`;
      }
      if (j.state === "error") throw new Error(j.error);
      branch = j; back = null; backRow = null;
      groups = makeGroups(j); sel = 0; ex = 0; tests = []; testRow = null; testEx = 0;
      document.querySelector(".side")?.classList.add("has-branch");
      setSpeed(15);
      showOptions(); watch();
    } catch (e) { toast(e.message, 5000); }
    $("#ask").disabled = false; $("#prog").hidden = true;
  }
  $("#ask").onclick = () => ask();

  const raw = (o) => o.dealt - o.taken + 40 * ((o.stocks_taken || 0) - (o.stocks_lost || 0));
  const valued = () => branch?.valued && branch?.you?.expected != null;
  const net = (o) => (o.expected != null && valued() ? o.expected : raw(o));
  const sgn = (v) => { const r = Math.round(v); return r > 0 ? `+${r}` : `${r === 0 ? 0 : r}`; };
  const ME_COL = "#4cc9f0", OPP_COL = "#f25f5c";   // the timeline's "your percent" / "opponent's percent" colors
  const GCOL = ["#4cc9f0", "#f72585", "#57d68d", "#ffb703", "#b388ff", "#ff7b54", "#2ec4b6", "#e0e0e0"];
  let groups = [], sel = 0, ex = 0;     // sel: chosen group (-1 = what you did); ex: which try of it
  let tests = [], testRow = null, testEx = 0, testing = false;   // "test this move" results; the one being watched
  // the tries grouped by the master's first move, most likely first
  function makeGroups(b) {
    const by = new Map();
    for (const o of b.options) { const k = o.group || o.label; if (!by.has(k)) by.set(k, []); by.get(k).push(o); }
    const top = (arr) => { const c = {}; for (const x of arr) { const k = x.split(" → ")[0]; c[k] = (c[k] || 0) + 1; } return Object.entries(c).sort((p, q) => q[1] - p[1])[0]; };
    function summarize(label, tries, b, other = false) {
      const v = tries.map(net), m = v.reduce((x, y) => x + y, 0) / v.length;
      const sd = v.length > 1 ? Math.sqrt(v.reduce((x, y) => x + (y - m) ** 2, 0) / (v.length - 1)) : null;
      const [oppTop, oppN] = other ? ["", 0] : top(tries.map((o) => o.opp_label || "")) || ["", 0];
      return { label, tries, other, share: tries.length / b.options.length, mean: m, se: sd != null ? sd / Math.sqrt(v.length) : null, oppTop, oppN };
    }
    const list = [...by.entries()].map(([label, tries]) => summarize(label, tries, b))
      .sort((p, q) => q.tries.length - p.tries.length || q.mean - p.mean);
    // moves the master picked only once go together at the end
    const many = list.filter((g) => g.tries.length > 1), once = list.filter((g) => g.tries.length === 1);
    const out = once.length > 1 ? [...many, summarize(`Other moves, once each`, once.flatMap((g) => g.tries), b, true)] : list;
    return out.map((g, n) => ({ ...g, color: GCOL[n % GCOL.length] }));
  }
  function setGhosts() {
    if (backRow != null && back) {
      const r = back.rows[backRow], o = r.tries[backEx % r.tries.length];
      send({ type: "ghosts", ghosts: [
        { url: `${o.url}?only=${s.me.port}`, label: "@", color: ME_COL },
        { url: `${o.url}?only=${s.opp.port}`, label: "@", color: OPP_COL }] });
      return;
    }
    if (testRow != null && tests[testRow]) {
      const t = tests[testRow], o = t.tries[testEx % t.tries.length];
      send({ type: "ghosts", ghosts: [
        { url: `${o.url}?only=${s.me.port}`, label: "@", color: ME_COL },
        { url: `${o.url}?only=${s.opp.port}`, label: "@", color: OPP_COL }] });
      return;
    }
    if (!branch || sel < 0) { send({ type: "ghosts", ghosts: [] }); return; }
    // the try as two ghosts so the players can be told apart: you (the master playing your
    // character) in your color, the opponent in theirs, each with a tag
    const gp = groups[sel], o = gp.tries[ex % gp.tries.length];
    send({ type: "ghosts", ghosts: [
      { url: `${o.url}?only=${s.me.port}`, label: "@", color: ME_COL },
      { url: `${o.url}?only=${s.opp.port}`, label: "@", color: OPP_COL }] });
  }
  // play the chosen line from just before the moment, on repeat
  function watch() {
    if (!branch) return;
    const a = Math.max(frame0, branch.frame - 20), e = Math.min(last, branch.frame + (branch.horizon || 90));
    loop = [a, e];
    setGhosts();
    send({ type: "range", start: a, end: e });
    seek(a); send({ type: "play" });
    drawTl();
  }
  function showOptions() {
    const b = branch, you = b.you, k = b.options.length;
    const yv = net(you);
    const row = (g, n) => {
      const gap = g.mean - yv, cls = !valued() ? (g.mean > 2 ? "pos" : g.mean < -2 ? "neg" : "zero") : gap >= 8 ? "pos" : gap <= -8 ? "neg" : "zero";
      const on = sel === n && testRow == null && backRow == null, o = g.tries[ex % g.tries.length];
      return `<div class="opt ${on ? "focus" : ""}" data-g="${n}">
        <div class="dot" style="background:${on ? ME_COL : "#3a4150"}"></div>
        <div><div class="lbl">${esc(g.label)}</div>
          <div class="res"><span class="share"><i style="width:${Math.round(g.share * 100)}%;background:#8b93a7"></i></span> ${g.tries.length} of ${k} tries${g.other ? `: ${esc(g.tries.map((o) => o.group || o.label).join(", "))}` : ""}${g.oppTop ? ` · opponent: ${esc(g.oppTop)}${g.tries.length > 1 ? ` (${g.oppN} of ${g.tries.length})` : ""}` : ""}</div>
          ${on ? `<div class="res" style="margin-top:6px">${g.tries.length > 1 ? `<button class="btn tiny" data-step="-1">‹</button> try ${ex % g.tries.length + 1} of ${g.tries.length} <button class="btn tiny" data-step="1">›</button> · ` : ""}this try: ${esc(o.label)}; opponent: ${esc(o.opp_label || "")} · <a class="dl" href="${o.url}" download="branch_${b.frame}_${o.i}.slp" title="Save as a Slippi replay">.slp</a></div>
          <button class="btn tiny testbtn" data-test="1" ${testing ? "disabled" : ""} title="Play this exact move 32 times against fresh opponents">Test this move ×32</button>
          <button class="btn tiny testbtn" data-try="1" title="Play this moment yourself, with this try as a ghost to follow">Try this yourself</button>` : ""}
        </div>
        <div style="text-align:right"><span class="net ${cls}">${sgn(g.mean)}</span>${g.se != null ? `<div class="faint" style="font-size:11px">±${g.se.toFixed(0)}</div>` : ""}</div></div>`;
    };
    $("#opts").innerHTML = `
      <div class="summary">What the master does from ${fmtTimeFine(b.frame)}, most likely first (${k} tries). ${valued()
        ? `Numbers: expected damage dealt − taken over the next 5 s after the choice (a stock = 40). Gaps under 8 are luck.`
        : `The number is damage dealt − taken over the try (a stock counts 40).`}${b.exact ? "" : " <span style='color:var(--warn)'>The simulator didn't match the replay exactly here, so treat these as approximate.</span>"}</div>
      <div class="faint" style="font-size:12.5px;line-height:1.5"><b style="color:#4cc9f0">Blue ghost</b>: the master playing your ${esc(s.me.char_name)}. <b style="color:#f25f5c">Red ghost</b>: your opponent, played by a master model so they react. Solid: your real game.</div>
      <div id="tests"></div>
      <div class="opt you ${sel === -1 && testRow == null && backRow == null ? "focus" : ""}" data-you="1"><div class="dot" style="background:transparent;border:2px solid #e8ebf2"></div>
        <div><div class="lbl">You: ${esc(you.label)}</div><div class="res">what you actually did · <button class="btn tiny testbtn" data-testyou="1" ${testing ? "disabled" : ""} title="Play what you did 32 times against fresh opponents">Test ×32</button></div></div><div>${`<span class="net zero">${sgn(yv)}</span>`}</div></div>
      ${groups.map(row).join("")}
      
      <div class="back" id="back"></div>`;
    $("#opts").querySelectorAll(".opt[data-g]").forEach((el) => (el.onclick = (e) => {
      if (e.target.closest("a")) return;
      const n = +el.dataset.g;
      if (e.target.closest("[data-test]")) { const gp = groups[n], o = gp.tries[ex % gp.tries.length]; runTest({ job: b.id, try: o.i }, o.group || gp.label, `${b.id}:${o.i}`); return; }
      if (e.target.closest("[data-try]")) { const gp = groups[n], o = gp.tries[ex % gp.tries.length]; tryIt({ frame: b.frame, ghost: o.url, label: gp.label, master: gp.mean }); return; }
      const st = e.target.closest("[data-step]");
      if (st) { ex = (ex + +st.dataset.step + groups[n].tries.length) % groups[n].tries.length; }
      else if (sel !== n) { sel = n; ex = 0; }
      backRow = null; testRow = null; showOptions(); watch();
    }));
    $("#opts .opt.you").onclick = (e) => {
      if (e.target.closest("[data-testyou]")) { runTest({ you: true }, `You: ${you.label.split(" → ")[0]}`, "you"); return; }
      sel = -1; backRow = null; testRow = null; showOptions(); watch();
    };
    drawTests();
    drawBack();
  }

  // ── Try it yourself (makeLive, below) ──
  const live = makeLive({
    id, s, send, iframe, panel: () => $("#opts"), cur: () => cur,
    busy: (b) => { $("#tryIt").disabled = b; $("#ask").disabled = b; if (b) loop = null; },
    youScore: (a) => (branch?.frame === a.frame && valued() ? net(branch.you) : null),
    watch: (st) => {
      loop = [Math.max(frame0, st.moment - 20), Math.min(last, st.end)];
      send({ type: "range", start: loop[0], end: loop[1] }); setSpeed(15); seek(loop[0]); send({ type: "play" });
    },
    back: (a) => {
      if (!alive) return;
      send({ type: "load", url: replayUrl, name: s.file, frame: a?.frame ?? cur });
      if (branch) showOptions();
      else $("#opts").innerHTML = `<div class="dim" style="padding:6px 4px">Pause on a moment, or pick one from Biggest swings below, then ask what a master would do, or try it yourself.</div>`;
    },
  });
  const tryIt = (args) => {
    if (!args) { const p = nearestPoint(cur); args = { frame: p && cur - p.frame < p.wait + 30 ? p.frame : cur }; }
    live.start(args);
  };
  $("#tryIt").onclick = () => tryIt();

  // Test one move against many opponents: your side presses exactly that move, the opponent is
  // fresh each time. Ranked by expected return; "best" only when it beats the next by more than luck.
  async function runTest(body, label, key) {
    if (testing) return;
    testing = true; showOptions();
    const msg = $("#tests");
    try {
      const job = await api("/api/test", { game: id, frame: branch.frame, k: 32, label, ...body });
      let j = job; const left = timer();
      while (j.state !== "done" && j.state !== "error") {
        await new Promise((r) => setTimeout(r, 300));
        j = await api(`/api/job/${job.id}`);
        if ($("#testProg")) $("#testProg").innerHTML = `<div class="testing">Testing <b>${esc(label)}</b> against 32 opponents… ${Math.round((j.progress || 0) * 100)}%${left(j.progress)}</div>`;
      }
      if (j.state === "error") throw new Error(j.error);
      if (!alive || branch?.frame !== j.frame) return;
      // the same move from another try gets a number so the two can be told apart
      const same = tests.filter((t) => t.key !== key && t.base === label).length;
      tests = tests.filter((t) => t.key !== key).concat([{ ...j, key, base: label, label: same ? `${label} (${same + 1})` : label }]);
      tests.sort((p, q) => q.mean - p.mean);
      testRow = tests.findIndex((t) => t.key === key); testEx = 0;
    } catch (e) { toast(e.message, 5000); }
    testing = false; showOptions();
    if (testRow != null) { watch(); $("#tests")?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  }
  function drawTests() {
    const el = $("#tests");
    if (!el) return;
    const prog = testing ? `<div id="testProg"><div class="testing">Testing against 32 opponents…</div></div>` : "";
    if (!tests.length) { el.innerHTML = prog; return; }
    const lo = Math.min(...tests.map((t) => t.p10)), hi = Math.max(...tests.map((t) => t.p90));
    const X = (v) => (hi > lo ? ((v - lo) / (hi - lo)) * 100 : 50);
    const [a, b2] = tests;
    const clear = b2 && a.mean - b2.mean > 2 * Math.hypot(a.se, b2.se);
    el.innerHTML = `${prog}<h3 style="margin:4px 0 6px">Tested moves</h3>
      <div class="faint" style="font-size:12px;margin-bottom:6px">Each move played 32 times against fresh opponents (different reactions and DI). Number: average expected return ± uncertainty. Bar: the middle 80% of outcomes.${tests.length > 1 ? (clear ? "" : " The top two are within luck of each other: too close to call.") : " Test another move to compare."}</div>
      ${tests.map((t, n) => `<div class="opt ${testRow === n ? "focus" : ""}" data-t="${n}"><div class="dot" style="background:${testRow === n ? ME_COL : "#3a4150"}"></div>
        <div><div class="lbl">${esc(t.label)}${n === 0 && clear ? ` <span class="best">best</span>` : ""}</div>
          <div class="range"><i style="left:${X(t.p10)}%;width:${Math.max(1, X(t.p90) - X(t.p10))}%"></i><b style="left:${X(t.mean)}%"></b></div>
          <div class="res">from ${sgn(t.p10)} to ${sgn(t.p90)} · if the opponent: ${t.responses.slice(0, 3).map((r) => `${esc(r[0])} (${r[1]}×) ${sgn(r[2])}`).join(", ")}${t.same != null && t.same < t.tries.length ? `<br><span style="color:var(--warn)">The move came out in ${t.same} of ${t.tries.length} tries; the opponent's actions changed the rest.</span>` : ""}</div>
          ${testRow === n ? (() => { const o = t.tries[testEx % t.tries.length]; return `<div class="res" style="margin-top:6px"><button class="btn tiny" data-tstep="-1">‹</button> try ${testEx % t.tries.length + 1} of ${t.tries.length} <button class="btn tiny" data-tstep="1">›</button> · ${sgn(o.expected)} · you: ${esc(o.label)}; opponent: ${esc(o.opp_label)} · <a class="dl" href="${o.url}" download="test_${t.frame}_${o.i}.slp">.slp</a></div>
          <button class="btn tiny testbtn" data-ttry="1" title="Play this moment yourself, with this try as a ghost to follow">Try this yourself</button>`; })() : ""}
        </div>
        <div style="text-align:right"><span class="net ${n === 0 && clear ? "pos" : "zero"}">${sgn(t.mean)}</span><div class="faint" style="font-size:11px">±${t.se < 1 ? t.se.toFixed(1) : t.se.toFixed(0)}</div></div></div>`).join("")}`;
    el.querySelectorAll(".opt[data-t]").forEach((row) => (row.onclick = (e) => {
      if (e.target.closest("a")) return;
      const n = +row.dataset.t, st = e.target.closest("[data-tstep]");
      if (e.target.closest("[data-ttry]")) { const t = tests[n], o = t.tries[testEx % t.tries.length]; tryIt({ frame: t.frame, ghost: o.url, label: t.label, master: t.mean }); return; }
      if (st) { const L = tests[n].tries.length; testEx = (testEx + +st.dataset.tstep + L) % L; }
      else { testRow = testRow === n ? null : n; testEx = 0; backRow = null; }
      showOptions(); watch();
    }));
  }

  // Could the spot have been avoided? The master from your earlier decisions, scored at the same end point.
  function drawBack() {
    const el = $("#back");
    if (!el) return;
    if (!branch?.valued) { el.innerHTML = ""; return; }
    if (!back) {
      el.innerHTML = `<button class="btn" id="goBack" style="width:100%">Could you have avoided this spot?</button>
        <div class="hint" style="margin-top:6px">Plays the master from your decisions in the 3 seconds before this, and scores every line at the same moment.</div>`;
      $("#goBack").onclick = goBack;
      return;
    }
    const sgn = (v) => { const r = Math.round(v); return r > 0 ? `+${r}` : `${r === 0 ? 0 : r}`; };
    el.innerHTML = `<h3 style="margin:14px 0 6px">Before this moment</h3>
      <div class="hint" style="margin:0 0 8px">Each line is scored at ${fmtTimeFine(back.end)}: damage dealt − taken from that decision on, plus the expected next 5 seconds. Gaps under 8 are within luck.</div>
      ${back.rows.map((r, n) => {
        const gap = r.master - r.you, cls = gap >= 8 ? "pos" : gap <= -8 ? "neg" : "zero";
        return `<div class="opt ${backRow === n ? "focus" : ""}" data-b="${n}"><div></div>
          <div><div class="lbl">${((back.frame - r.frame) / 60).toFixed(1)} s earlier · you: ${esc(r.did)}</div>
          <div class="res">Master's tries ${sgn(r.master)} (±${r.se.toFixed(0)}), yours ${sgn(r.you)}. ${r.same} of ${r.n} tries were in this same spot.${r.exact ? "" : " <span style='color:var(--warn)'>Approximate.</span>"}</div>
          ${backRow === n ? (() => { const o = r.tries[backEx % r.tries.length]; return `<div class="res" style="margin-top:6px"><button class="btn tiny" data-bstep="-1">‹</button> try ${backEx % r.tries.length + 1} of ${r.tries.length} <button class="btn tiny" data-bstep="1">›</button> · master: ${esc(o.label)}; opponent: ${esc(o.opp_label || "")} · <a class="dl" href="${o.url}" download="earlier_${r.frame}_${o.i}.slp">.slp</a></div>`; })() : ""}</div>
          <div style="text-align:right"><span class="net ${cls}">${sgn(gap)}</span></div></div>`;
      }).join("")}`;
    el.querySelectorAll(".opt[data-b]").forEach((row) => (row.onclick = (e) => {
      if (e.target.closest("a")) return;
      const n = +row.dataset.b, st = e.target.closest("[data-bstep]");
      if (st) { const L = back.rows[n].tries.length; backEx = (backEx + +st.dataset.bstep + L) % L; }
      else { backRow = backRow === n ? null : n; backEx = 0; testRow = null; }
      showOptions();
      const r = back.rows[n];
      if (backRow != null) { loop = [Math.max(frame0, r.frame - 30), Math.min(last, back.end + 30)]; send({ type: "range", start: loop[0], end: loop[1] }); seek(loop[0]); send({ type: "play" }); }
      else watch();
    }));
  }
  async function goBack() {
    const btn = $("#goBack"); btn.disabled = true; btn.textContent = "Starting…";
    try {
      const job = await api("/api/earlier", { game: id, frame: branch.frame, k: 8 });
      let j = job; const left = timer();
      while (j.state !== "done" && j.state !== "error") {
        await new Promise((r) => setTimeout(r, 400));
        j = await api(`/api/job/${job.id}`);
        if ($("#goBack")) $("#goBack").textContent = j.state === "queued" ? "Waiting for the simulator…" : `Playing out the earlier moments… ${Math.round((j.progress || 0) * 100)}%${left(j.progress)}`;
      }
      if (j.state === "error") throw new Error(j.error);
      if (!alive) return;
      back = j; backRow = null; drawBack();
    } catch (e) { toast(e.message, 5000); drawBack(); }
  }

  async function loadReview() {
    $("#revState").textContent = "Reviewing the game…";
    try {
      let j = await api("/api/review", { game: id }); const left = timer();
      while (j.state && j.state !== "done" && j.state !== "error") {
        await new Promise((r) => setTimeout(r, 500)); j = await api(`/api/job/${j.id}`);
        if (alive && j.progress > 0) $("#revState").textContent = `Reviewing the game… ${Math.round(j.progress * 100)}%${left(j.progress)}`;
      }
      if (j.state === "error") throw new Error(j.error);
      if (!alive) return;
      review = j;
      const ev = j.eval, half = Math.max(1, Math.round(15 / Math.max(1, (ev[1]?.[0] ?? 2) - ev[0][0])));
      review.eval = ev.map(([f], i) => { let sum = 0, n = 0; for (let k = Math.max(0, i - half); k <= Math.min(ev.length - 1, i + half); k++) { sum += ev[k][1]; n++; } return [f, sum / n]; });
      $("#legend").innerHTML = `<span><i style="background:rgba(87,214,141,.6);height:8px"></i>Going your way</span><span><i style="background:rgba(242,95,92,.6);height:8px"></i>Going their way</span><span><i style="background:#f25f5c;width:7px;height:7px;border-radius:50%"></i>Costly decision</span><span><i style="background:#57d68d;width:7px;height:7px;border-radius:50%"></i>Great decision</span>`;
      drawList();
      drawTl(); tick();
      // master tries run in the background at each flagged moment; show them as they land
      while (alive && review && Object.values(review.compare || {}).some((c) => c.state !== "done" && c.state !== "error")) {
        await new Promise((r) => setTimeout(r, 1500));
        const k = await api(`/api/job/${review.id}`).catch(() => null) || await api("/api/review", { game: id }).catch(() => null);
        if (!k || !review || !alive) break;
        review.compare = k.compare; drawList();
      }
    } catch (e) { $("#revState").textContent = ""; if (!/No review model/.test(e.message)) toast(e.message, 4000); }
  }
  loadReview();
  function drawList() {
    const tagged = review.points.filter((p) => p.tag).sort((a, b) => a.frame - b.frame);
    $("#revSec").hidden = !tagged.length;
    const pending = Object.values(review.compare || {}).filter((c) => c.state !== "done" && c.state !== "error").length;
    $("#revHead").innerHTML = `Biggest swings <span class="faint" style="text-transform:none;letter-spacing:0;font-weight:400">${pending ? `· comparing with the master… ${tagged.length - pending}/${tagged.length}` : "· you vs the master (gaps under 8 are within luck)"}</span>`;
    $("#revList").innerHTML = tagged.map((p) => {
      const c = review.compare?.[String(p.frame)];
      const vs = c?.state === "done" && c.model != null && c.you != null ? c.you - c.model : null;
      const right = vs == null
        ? `<span class="c ${p.tag}" title="How much your expected score moved">${p.cost > 0 ? "+" : ""}${p.cost.toFixed(0)}</span>`
        : `<span class="c ${vs <= -8 ? "costly" : vs >= 8 ? "great" : ""}" title="Your choice minus the master's average, in expected score">${vs > 0 ? "+" : ""}${vs.toFixed(0)}</span>`;
      const sub = vs != null && vs <= -8 && c.best ? `<span class="faint sub">master: ${esc(c.best)}</span>` : "";
      return `<button class="rev ${p.tag}" data-f="${p.frame}"><span class="mono">${fmtTimeTenth(p.frame)}</span><span class="lbl">${esc(p.did)}${sub}</span>${right}</button>`;
    }).join("");
    $("#revList").querySelectorAll(".rev").forEach((b) => (b.onclick = () => { send({ type: "pause" }); send({ type: "range", start: null }); seek(+b.dataset.f); ask(+b.dataset.f); }));
  }

  const onKey = (e) => {
    if (e.target.closest("input,select,textarea")) return;
    if (live.on()) return;
    if ((e.key === "t" || e.key === "T") && !$("#tryIt").disabled) { e.preventDefault(); tryIt(); return; }
    if ((e.key === "r" || e.key === "R") && live.again()) { e.preventDefault(); return; }
    if (e.code === "Space") { e.preventDefault(); send({ type: "toggle" }); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
    else if (e.key === "]") jumpDp(1);
    else if (e.key === "[") jumpDp(-1);
    else if (e.key === "m" || e.key === "M") !$("#ask").disabled && ask();
  };
  window.addEventListener("keydown", onKey);
  cleanup = () => { alive = false; live.end(true); window.removeEventListener("message", onMsg); window.removeEventListener("keydown", onKey); ro.disconnect(); };
  tick();
}

// ── Try it yourself: play a moment live against the bot (server: live.py, on the CPU) ──
// c: {id, s, send, iframe, panel(), cur(), busy(on), youScore(args), watch(status), back(args)}.
// args: {frame, ghost?, label?, master?} for a moment of the game, or {lab: {at, place, states}}.
function makeLive(c) {
  let on = false, sess = null, src = null, onLoaded = null, again = null, args = null, speed = 1;
  const pad = new PadReader();
  const panel = (html) => { c.panel().innerHTML = `<div class="live">${html}</div>`; };
  const q = (sel) => c.panel().querySelector(sel);
  const controllerLine = () => `<div class="ctl"><span class="dot2"></span>Controller: <b class="ctlName">${esc(pad.source())}</b>
      ${navigator.usb && !pad.adapter ? `<button class="btn tiny gcAdapter">Use a GameCube adapter</button>` : ""}</div>
      <div class="faint keys" ${pad.source() === "keyboard" ? "" : "hidden"}>${KEYS.map(([k, v]) => `<span><kbd>${k}</kbd> ${v}</span>`).join("")}</div>`;
  const wireController = () => {
    const b = q(".gcAdapter");
    if (b) b.onclick = async () => { try { await pad.connectAdapter(); toast("GameCube adapter connected"); } catch (e) { if (e?.name !== "NotFoundError") toast(e.message || String(e), 6000); } };
  };
  const ghosts = () => (args.ghost ? [{ url: `${args.ghost}?only=${c.s.me.port}`, label: "@", color: ME_COL }] : []);

  async function start(a) {
    if (on) return;
    args = a; again = null;
    c.send({ type: "pause" });
    on = true; pad.start(); c.busy(true);
    panel(`<h3>Try it yourself</h3><div class="dim">Getting the moment ready…</div>${controllerLine()}`);
    wireController();
    try {
      let j = await api("/api/live", a.lab ? { game: c.id, lab: a.lab } : { game: c.id, frame: a.frame });
      sess = j;
      while (j.state === "preparing") { await new Promise((r) => setTimeout(r, 200)); j = await api(`/api/live/${sess.id}`); }
      if (j.state === "error") throw new Error(j.error);
      sess = j;
      const speeds = [[1, "Full speed"], [0.5, "Half"], [0.25, "Quarter"]].filter(([v]) => v <= j.max_speed + 1e-6);
      if (!speeds.length) speeds.push([j.max_speed, `${Math.round(j.max_speed * 100)}%`]);
      if (!speeds.some(([v]) => v === speed)) speed = speeds[0][0];
      panel(`<h3>Try it yourself</h3>
        <div class="lead2">${a.lab ? "Your setup, played live: you control your character from the first frame." : `You take over at <b>${fmtTimeFine(j.moment)}</b>, after 1 second of what really happened, to get ready.`}${a.ghost ? ` The <b style="color:${ME_COL}">blue ghost</b> is the master's try (${esc(a.label)}) to follow.` : ""}</div>
        <div class="faint">Your opponent: a model of a master ${esc(c.s.opp.char_name)}${j.bot_kind === "S RL" ? "" : `, running on this computer's processor${j.max_speed < 1 ? `: the game runs at up to ${Math.round(j.max_speed * 100)}% speed until this character's faster model is trained` : ""}`}.</div>
        ${controllerLine()}
        <div class="speeds">${speeds.map(([v, n]) => `<button class="btn tiny ${v === speed ? "on" : ""}" data-sp="${v}">${n}</button>`).join("")}</div>
        <button class="btn primary big liveGo">Start <kbd style="color:#06222c;border-color:#0a5b73">Enter</kbd></button>
        <button class="btn liveCancel">Cancel</button>`);
      wireController();
      c.panel().querySelectorAll("[data-sp]").forEach((b) => (b.onclick = () => { speed = +b.dataset.sp; c.panel().querySelectorAll("[data-sp]").forEach((x) => x.classList.toggle("on", x === b)); }));
      q(".liveCancel").onclick = () => end(true);
      await new Promise((res) => {
        // Start: the button, Enter (even a quick tap), or Start on the controller
        const enter = (e) => { if (e.key === "Enter") { e.preventDefault(); done(); } };
        const done = () => { window.removeEventListener("keydown", enter, true); res(); };
        window.addEventListener("keydown", enter, true);
        q(".liveGo").onclick = done;
        const wait = () => { if (!on) return done(); if (pad.read().buttons.includes("start")) done(); else requestAnimationFrame(wait); };
        requestAnimationFrame(wait);
      });
      if (!on) return;
      await play();
    } catch (e) { toast(e.message || String(e), 6000); end(true); }
  }

  async function play() {
    const j = sess;
    panel(`<div class="bigmsg liveMsg">${j.start < j.moment ? readyMsg(j.moment - j.start) : "Your turn!"}</div><div class="bar thin"><i class="liveBar"></i></div>${controllerLine()}`);
    // the replay up to the start, then the frames as they're played
    await new Promise((res) => { onLoaded = res; c.send({ type: "load", url: `/api/live/${j.id}/prefix.slp`, name: "attempt.slp", frame: j.start - 1 }); });
    c.send({ type: "live", sizes: j.sizes });
    c.send({ type: "range", start: null });
    c.send({ type: "speed", fps: 60 });
    c.send({ type: "ghosts", ghosts: ghosts() });
    c.iframe.blur(); c.iframe.style.pointerEvents = "none"; window.focus();
    src = new EventSource(`/api/live/${j.id}/stream`);
    src.addEventListener("f", (e) => {
      const bin = atob(e.data), u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      c.iframe.contentWindow?.postMessage({ type: "append", bytes: u.buffer }, "*", [u.buffer]);
    });
    src.addEventListener("s", (e) => {
      const st = JSON.parse(e.data);
      if (st.state === "done") done(st);
      else if (st.state === "error") { toast(st.error || "Something went wrong", 6000); end(true); }
    });
    await api(`/api/live/${j.id}/go`, { speed });
    // the controller, sent whenever it changes (and every 100 ms anyway); a button pressed while a
    // send is under way goes with the next send, so a quick tap still counts
    let last = null, lastT = 0, inflight = false, held = new Set();
    const loop = () => {
      if (!on || sess !== j) return;
      const p = pad.read(), t = performance.now();
      p.buttons.forEach((b) => held.add(b));
      const n = q(".ctlName"); if (n) n.textContent = pad.source();
      if (!inflight && (!samePad(p, last) || held.size > p.buttons.length || t - lastT > 100)) {
        const out = { ...p, buttons: [...new Set([...p.buttons, ...held])] };
        held = new Set();
        inflight = true; last = p; lastT = t;
        fetch(`/api/live/${j.id}/pad`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(out) })
          .catch(() => {}).finally(() => { inflight = false; });
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  function frame() {
    if (!on || !sess) return;
    const j = sess, m = q(".liveMsg"), b = q(".liveBar"), cur = c.cur();
    if (!m) return;
    m.textContent = cur < j.moment ? readyMsg(j.moment - cur) : "Your turn!";
    m.classList.toggle("go", cur >= j.moment);
    if (b) b.style.width = `${Math.max(0, Math.min(100, ((cur - j.start) / (j.end - j.start)) * 100))}%`;
  }

  function done(st) {
    src?.close(); src = null;
    pad.stop(); on = false;
    c.iframe.style.pointerEvents = "";
    c.busy(false);
    const r = st.result || {}, a = args;
    const you = a.lab ? null : c.youScore(a);
    const sg = (v) => { const x = Math.round(v); return x > 0 ? `+${x}` : `${x === 0 ? 0 : x}`; };
    const line = (name, v, cls = "") => v == null ? "" : `<div class="cmp ${cls}"><span>${name}</span><b>${sg(v)}</b></div>`;
    panel(`<h3>Your attempt</h3>
      ${line("You, just now", r.expected ?? r.realized, "me")}
      ${a.master != null ? line(`The master (${esc(a.label)})`, a.master) : ""}
      ${line("You, in the game", you)}
      ${st.unsettled?.length ? `<div class="warnline">${st.unsettled.map((q) => (q === c.s.me.port ? "You" : esc(c.s.opp.char_name))).join(" and ")} couldn't get into the position you picked, so this started from where the game put ${st.unsettled.length > 1 ? "them" : "it"}.</div>` : ""}
      <div class="faint" style="margin:6px 0 10px">${r.expected != null ? "Expected damage dealt − taken over the next 5 s after the choice (a stock = 40)." : "Damage dealt − taken over the decision (a stock = 40); this character has no review model yet."} This attempt: dealt ${r.dealt}%, took ${r.taken}%. One attempt has luck in it; try a few times.</div>
      <div class="row2"><button class="btn primary liveAgain">Again <kbd style="color:#06222c;border-color:#0a5b73">R</kbd></button>
      <button class="btn liveWatch">Watch it</button>
      <a class="btn" href="/api/live/${st.id}/attempt.slp" download="attempt_${st.moment}.slp">Save .slp</a>
      <button class="btn liveDone">Done</button></div>`);
    again = () => { again = null; start(a); };
    q(".liveAgain").onclick = again;
    q(".liveWatch").onclick = () => {
      onLoaded = () => { c.send({ type: "ghosts", ghosts: ghosts() }); c.watch(st); };
      c.send({ type: "load", url: `/api/live/${st.id}/attempt.slp`, name: "attempt.slp", frame: st.moment - 20 });
    };
    q(".liveDone").onclick = () => end(false);
  }

  function end(stopServer) {
    if (stopServer && sess && on) fetch(`/api/live/${sess.id}/stop`, { method: "POST" }).catch(() => {});
    src?.close(); src = null;
    const was = args;
    pad.stop(); on = false; again = null; onLoaded = null; sess = null;
    c.iframe.style.pointerEvents = "";
    c.busy(false);
    c.back(was);
  }

  return {
    start, end, frame,
    on: () => on,
    again: () => { if (!again) return false; again(); return true; },
    loaded: () => { if (onLoaded) { const f = onLoaded; onLoaded = null; f(); } },
  };
}

// ── patterns ─────────────────────────────────────────────────────────────────────────────
async function patterns(st) {
  crumbs(`<a href="#/">Your games</a> › <b>Your patterns</b>`);
  const d = await api("/api/games");
  const chars = [...new Set(d.games.filter((g) => !g.unsupported && g.reviewable).map((g) => g.me?.char_name).filter(Boolean))];
  let char = chars.includes(sessionStorage.getItem("patChar")) ? sessionStorage.getItem("patChar") : chars[0];
  const draw = (body) => {
    view.innerHTML = `<div class="lib">
      <div class="libhead"><div><h1>Your patterns</h1><div class="dim">Situations that keep costing you, across your recent games, and what the master does there instead.</div></div></div>
      <div class="chips">${chars.map((c) => `<button class="chip ${c === char ? "on" : ""}" data-c="${esc(c)}">${esc(c)}</button>`).join("")}</div>
      ${body}</div>`;
    view.querySelectorAll(".chip").forEach((b) => (b.onclick = () => { char = b.dataset.c; sessionStorage.setItem("patChar", char); run(); }));
  };
  const bars = (list) => list.map(([k, v]) => `<div class="pbar"><span>${esc(k)}</span><i style="width:${Math.round(v * 100)}%"></i><b>${Math.round(v * 100)}%</b></div>`).join("") || `<div class="faint">—</div>`;
  async function run() {
    draw(`<div class="labcard"><div id="pmsg" class="dim">Starting…</div><div class="bar" style="margin-top:8px"><i id="pbar"></i></div><div class="faint" style="margin-top:8px;font-size:12px">This looks through your 40 most recent games and asks the master about the situations that stand out. The first time, each game gets reviewed, which takes a while (it keeps going if you leave this page); reviewed games are instant after that.</div></div>`);
    let key = `pat:${char}`, j;
    try {
      j = await api("/api/patterns", { char, n: 40 }); const left = timer();
      while (j.state !== "done" && j.state !== "error") {
        await new Promise((r) => setTimeout(r, 1000));
        if (!location.hash.startsWith("#/patterns")) return;
        j = await api(`/api/job/${j.id}`);
        const m = $("#pmsg"); if (m) m.textContent = (j.message || "Working…") + left(j.progress);
        const b = $("#pbar"); if (b) b.style.width = `${Math.round((j.progress || 0) * 100)}%`;
      }
      if (j.state === "error") throw new Error(j.error);
    } catch (e) { draw(`<div class="empty">${esc(e.message)}</div>`); return; }
    draw(`<div class="summary" style="padding:0 0 12px">${j.decisions} decisions in ${j.games} games. Each number is how much a situation costs you per time, in expected score over the next 5 seconds, compared with your average decision (± is the uncertainty). Many situations are checked, so treat one that's barely past its ± with care.</div>
      <div class="pats">${j.patterns.map((f) => `<div class="pat">
        <div class="pathead"><h2>${esc(f.situation[0].toUpperCase() + f.situation.slice(1))}</h2><span class="net ${f.mean <= -2 ? "neg" : "zero"}">${f.mean > 0 ? "+" : ""}${f.mean.toFixed(1)} <span class="faint" style="font-weight:400">± ${f.se.toFixed(1)}</span></span></div>
        <div class="dim" style="margin-bottom:10px">${f.n} times</div>
        <div class="cols"><div><h3>You usually</h3>${bars(f.you)}</div><div><h3>The master</h3>${bars(f.master)}</div></div>
        <h3 style="margin-top:12px">Examples</h3>
        <div class="exs">${f.examples.map((e) => `<a href="#/game/${e.game}?f=${e.frame}" class="ex"><span class="faint" style="min-width:64px">${e.start ? new Date(e.start).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : ""}</span><span class="mono">${fmtTime(e.frame)}</span> ${esc(e.did)} <span class="net ${e.cost <= -8 ? "neg" : "zero"}">${e.cost > 0 ? "+" : ""}${e.cost.toFixed(0)}</span></a>`).join("")}</div>
      </div>`).join("") || `<div class="empty">Nothing stands out yet: no situation costs you clearly more than your average.</div>`}</div>`);
  }
  if (!chars.length) { draw(`<div class="empty">No games to look at yet.</div>`); return; }
  run();
}

// ── lab ──────────────────────────────────────────────────────────────────────────────────
// Stage outlines for the editor (game units, y up). The main stage's top is y = 0.
const STAGE_GEO = {
  "Battlefield": { top: [-68.4, 68.4], under: [[68.4, 0], [65, -6], [36, -19], [39, -21], [30, -29], [29, -35], [10, -40], [-10, -40], [-29, -35], [-30, -29], [-39, -21], [-36, -19], [-65, -6], [-68.4, 0]], plats: [[-57.6, -20, 27.2], [20, 57.6, 27.2], [-18.8, 18.8, 54.4]], blast: [-224, -108.8, 224, 200] },
  "Dream Land": { top: [-77.25, 77.25], under: [[77.25, 0], [76.5, -11], [65.75, -36], [-65.75, -36], [-76.5, -11], [-77.25, 0]], plats: [[-61.4, -31.7, 30.1], [31.7, 63.1, 30.2], [-19, 19, 51.4]], blast: [-255, -123, 255, 250] },
  "Final Destination": { top: [-85.6, 85.6], under: [[85.6, 0], [85.6, -10], [65, -20], [65, -30], [60, -47], [50, -55], [-50, -55], [-60, -47], [-65, -30], [-65, -20], [-85.6, -10], [-85.6, 0]], plats: [], blast: [-246, -140, 246, 188] },
  "Yoshi's Story": { top: [-56, 56], under: [[56, 0], [56, -4], [40, -20], [-40, -20], [-56, -4], [-56, 0]], plats: [[-59.5, -28, 23.45], [28, 59.5, 23.45], [-15.75, 15.75, 42]], blast: [-175.7, -91, 173.6, 169] },
  "Fountain of Dreams": { top: [-63.35, 63.35], under: [[63.35, 0.6], [63.35, -4.5], [59.3, -15], [55, -27], [48, -38], [41, -42], [19, -49.5], [10, -62], [8.8, -90], [-8.8, -90], [-10, -62], [-19, -49.5], [-41, -42], [-48, -38], [-55, -27], [-59.3, -15], [-63.35, -4.5], [-63.35, 0.6]], plats: [[-49.5, -28, 16.1], [28, 49.5, 22.1], [-14.25, 14.25, 42.75]], blast: [-198.75, -146.25, 198.75, 202.5] },
  "Pokémon Stadium": { top: [-87.75, 87.75], under: [[87.75, 0], [87.75, -4], [70, -20], [-70, -20], [-87.75, -4], [-87.75, 0]], plats: [[-55, -25, 25], [25, 55, 25]], blast: [-230, -111, 230, 180] },
};

async function lab(id, f0) {
  view.innerHTML = `<div class="lib"><div class="empty">Loading the lab…</div></div>`;
  const g = await api(`/api/game/${id}`);
  const s = g.summary;
  if (s.unsupported) { view.innerHTML = `<div class="setup"><h1>Can't use this game in the Lab</h1><p class="lead">${esc(s.unsupported)}</p></div>`; return; }
  const st = await api(`/api/game/${id}/state?f=${f0 ?? g.points[0]?.frame ?? 600}`);
  const at = st.frame;
  crumbs(`<a href="#/">Your games</a> › <a href="#/game/${id}">${esc(s.me.char_name)} vs ${esc(s.opp.char_name)}</a> › <b>Lab</b> <span class="faint">· ${esc(st.stage)} · from ${fmtTime(at)}</span>`);
  const geo = STAGE_GEO[st.stage] || STAGE_GEO["Final Destination"];
  const floors = [[geo.top[0], geo.top[1], 0], ...geo.plats];
  const floorAt = (x, y, below = 3, above = 6) => floors.find(([a, b, fy]) => x >= a && x <= b && y >= fy - below && y <= fy + above);
  const players = st.players.map((p) => ({ ...p, x: +p.x.toFixed(1), y: +p.y.toFixed(1), percent: Math.round(p.percent),
                                           state: floorAt(p.x, p.y, 1, 3) ? "stand" : "air" }));
  const start = JSON.parse(JSON.stringify(players));
  view.innerHTML = `
  <div class="labview">
    <div class="labcol">
      <div class="labcard">
        <h3>Set up the moment</h3>
        <p class="dim" style="margin:0 0 10px">Drag the characters. Click one to turn it around. Pick what each one is doing. It starts from ${fmtTime(at)} of your game.</p>
        <svg id="ed" class="editor"></svg>
        <div class="pcts" id="pcts"></div>
        <div class="row" style="margin-top:12px"><button class="btn" id="reset">Back to the game</button><div class="grow"></div><button class="btn" id="tryLab" title="Play your setup yourself against the bot">Try it yourself</button><button class="btn primary" id="go">Play it out</button></div>
        <div class="progress" id="prog" hidden><div class="dim" id="progMsg"></div><div class="bar"><i></i></div></div>
      </div>
      <div class="opts" id="opts" style="padding:0"></div>
    </div>
    <div class="labright">
      <div class="screen"><iframe id="player" src="${viewerSrc(s.me.port + 1)}"></iframe><div class="over" id="over" hidden></div></div>
      <div class="controls">
        <button class="btn icon" id="back1" title="Back 1 frame">‹</button>
        <button class="btn icon" id="play" title="Play / pause (Space)">▶</button>
        <button class="btn icon" id="fwd1" title="Forward 1 frame">›</button>
        <div class="clock" id="clock"></div><div class="grow"></div>
        <button class="btn" id="cont" disabled title="Play it out again from the paused frame of the chosen try">Continue from here</button>
      </div>
    </div>
  </div>`;
  // editor
  const svg = $("#ed");
  const [bx0, , bx1] = geo.blast;
  const vx0 = Math.max(bx0, -130), vx1 = Math.min(bx1, 130), vy0 = -60, vy1 = 110;
  svg.setAttribute("viewBox", `${vx0} ${-vy1} ${vx1 - vx0} ${vy1 - vy0}`);
  const pathOf = (pts) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x} ${-y}`).join(" ") + " Z";
  const COL = (p) => (p.me ? "#4cc9f0" : "#f25f5c");
  function drawEd() {
    svg.innerHTML = `
      <rect x="${vx0}" y="${-vy1}" width="${vx1 - vx0}" height="${vy1 - vy0}" fill="#0b0e15"/>
      <path d="${pathOf(geo.under)}" fill="#1c2230" stroke="#39415a" stroke-width=".6"/>
      ${geo.plats.map(([a, b, y]) => `<line x1="${a}" x2="${b}" y1="${-y}" y2="${-y}" stroke="#8b93a7" stroke-width="1.6" stroke-linecap="round"/>`).join("")}
      ${players.map((p, i) => {
        const h = p.state === "crouch" ? 4.2 : 7, cy = -h;
        return `<g class="fighter" data-i="${i}" transform="translate(${p.x} ${-p.y})">
        <ellipse cx="0" cy="${cy}" rx="${p.state === "crouch" ? 5.4 : 4.6}" ry="${h}" fill="${COL(p)}" fill-opacity=".85" stroke="#fff" stroke-width=".5"/>
        ${p.state === "shield" ? `<circle cx="0" cy="${cy}" r="9" fill="${COL(p)}" fill-opacity=".18" stroke="${COL(p)}" stroke-width=".8"/>` : ""}
        <path d="M ${p.facing * 3} ${cy - 2} L ${p.facing * 9} ${cy} L ${p.facing * 3} ${cy + 2} Z" fill="#fff"/>
        <text x="0" y="${cy - h - 3}" text-anchor="middle" font-size="5.5" fill="#e8ebf2">${esc(p.me ? "You" : p.char_name)}</text></g>`;
      }).join("")}`;
    svg.querySelectorAll(".fighter").forEach((el) => {
      el.onpointerdown = (e) => {
        const i = +el.dataset.i, p = players[i];
        el.setPointerCapture(e.pointerId);
        const pt = svg.createSVGPoint();
        const toGame = (ev) => { pt.x = ev.clientX; pt.y = ev.clientY; const q = pt.matrixTransform(svg.getScreenCTM().inverse()); return [q.x, -q.y]; };
        const [sx, sy] = toGame(e), ox = p.x, oy = p.y;
        let moved = false;
        el.onpointermove = (ev) => {
          const [x, y] = toGame(ev);
          if (Math.hypot(x - sx, y - sy) > 1) moved = true;
          p.x = Math.max(geo.blast[0] + 5, Math.min(geo.blast[2] - 5, ox + x - sx));
          p.y = Math.max(geo.blast[1] + 5, Math.min(geo.blast[3] - 5, oy + y - sy));
          el.setAttribute("transform", `translate(${p.x} ${-p.y})`);
        };
        el.onpointerup = () => {
          el.onpointermove = null;
          if (!moved && p.state !== "ledge") p.facing = -p.facing;
          else if (moved) placeAfterDrag(p);
          drawEd(); drawPcts();
        };
      };
    });
  }
  function snap(p) {   // dropped just above a floor: stand on it
    const f = floorAt(p.x, p.y);
    if (f) { p.y = f[2] + 0.5; return true; }
    return false;
  }
  // hanging on a ledge: just past the nearer edge of the main stage and below it, facing in (the sim
  // falls past the ledge and catches it: measured to work 4-8 out and 8-12 below)
  function toLedge(p) {
    // one fighter per ledge: the other side's when this one is taken
    const taken = (sd) => players.some((o) => o !== p && o.state === "ledge" && Math.sign(o.x) === sd);
    let side = p.x >= 0 ? 1 : -1;
    if (taken(side)) side = -side;
    p.x = side * (Math.abs(geo.top[side > 0 ? 1 : 0]) + 6); p.y = -10; p.facing = -side;
  }
  const nearLedge = (p) => { const e = Math.abs(geo.top[p.x >= 0 ? 1 : 0]); return Math.abs(p.x) > e - 4 && Math.abs(p.x) < e + 18 && p.y > -30 && p.y < 6; };
  function placeAfterDrag(p) {
    if (p.state === "ledge" && nearLedge(p)) { toLedge(p); return; }
    if (snap(p)) { if (p.state === "air" || p.state === "ledge") p.state = "stand"; }
    else p.state = "air";
  }
  // a ground state while in the air drops the fighter to the floor under it
  function setState(p, state) {
    if (state === "ledge") { toLedge(p); p.state = state; return true; }
    if (state === "air") { if (floorAt(p.x, p.y, 1, 3)) p.y += 20; p.state = state; return true; }
    const under = floors.filter(([a, b, fy]) => p.x >= a && p.x <= b && fy <= p.y + 6).sort((u, v) => v[2] - u[2])[0];
    if (!under) { toast("Put it over the stage first."); return false; }
    p.y = under[2] + 0.5; p.state = state; return true;
  }
  const STATES = [["stand", "Standing"], ["crouch", "Crouching"], ["shield", "Shielding"], ["air", "In the air"], ["ledge", "On the ledge"]];
  function drawPcts() {
    const order = players.map((p, i) => [p, i]).sort((a, b) => b[0].me - a[0].me);   // you first
    $("#pcts").innerHTML = order.map(([p, i]) => `<label class="pct"><span style="color:${COL(p)}">${esc(p.me ? `You (${p.char_name})` : `Opponent (${p.char_name})`)}</span>
      <input type="range" min="0" max="200" value="${p.percent}" data-i="${i}"><b class="mono">${p.percent}%</b></label>
      <div class="states">${STATES.map(([k, n]) => `<button class="btn tiny ${p.state === k ? "on" : ""}" data-p="${i}" data-st="${k}">${n}</button>`).join("")}</div>`).join("");
    $("#pcts").querySelectorAll("[data-st]").forEach((b) => (b.onclick = () => { if (setState(players[+b.dataset.p], b.dataset.st)) { drawEd(); drawPcts(); } }));
    $("#pcts").querySelectorAll("input").forEach((inp) => (inp.oninput = () => { players[+inp.dataset.i].percent = +inp.value; inp.nextElementSibling.textContent = `${inp.value}%`; }));
  }
  drawEd(); drawPcts();
  $("#reset").onclick = () => { players.splice(0, players.length, ...JSON.parse(JSON.stringify(start))); drawEd(); drawPcts(); };
  const setup = () => ({ place: Object.fromEntries(players.map((p) => [p.port, [p.x, p.y, p.facing, p.percent]])),
                         states: Object.fromEntries(players.map((p) => [p.port, p.state])) });

  // viewer
  const iframe = $("#player");
  const send = (m) => iframe.contentWindow?.postMessage(m, "*");
  let cur = at, running = false, result = null, focus = 0;
  const onMsg = (e) => {
    if (e.source !== iframe.contentWindow) return;
    const m = e.data || {};
    if (m.type === "frame") { cur = m.frame; running = m.running; tickLab(); live.frame(); }
    else if (m.type === "loaded") { send({ type: "ghostAlpha", alpha: GHOST_ALPHA }); live.loaded(); }
  };
  window.addEventListener("message", onMsg);
  iframe.addEventListener("load", () => feedViewer(iframe, send, `/api/game/${id}/replay.slp`, s.file, at));
  function tickLab() {
    $("#clock").innerHTML = result ? `<b>${((cur - result.frame) / 60).toFixed(2)} s</b> <span class="faint">into try ${focus + 1}</span>` : `<span class="faint">your game at ${fmtTimeFine(cur)}</span>`;
    $("#play").textContent = running ? "❚❚" : "▶";
    $("#cont").disabled = !result || cur <= result.frame || running;
  }
  $("#play").onclick = () => send({ type: "toggle" });
  $("#back1").onclick = () => { send({ type: "pause" }); send({ type: "seek", frame: cur - 1 }); };
  $("#fwd1").onclick = () => { send({ type: "pause" }); send({ type: "seek", frame: cur + 1 }); };

  async function playOut(from) {
    $("#go").disabled = true; $("#cont").disabled = true; $("#prog").hidden = false; $("#progMsg").textContent = "Starting…";
    try {
      const body = { game: id, at, k: 6, horizon: 300, ...setup() };
      if (from) Object.assign(body, from);
      let j = await api("/api/lab", body); const left = timer();
      while (j.state !== "done" && j.state !== "error") {
        await new Promise((r) => setTimeout(r, 300));
        j = await api(`/api/job/${j.id}`);
        $("#progMsg").textContent = j.state === "loading" ? "Loading the models (first time only)…" : `Playing it out ${j.k} ways…${left(j.progress)}`;
        $("#prog .bar i").style.width = `${Math.round((j.progress || 0) * 100)}%`;
      }
      if (j.state === "error") throw new Error(j.error);
      result = j; focus = 0;
      showTry();
    } catch (e) { toast(e.message, 5000); }
    $("#go").disabled = false; $("#prog").hidden = true;
  }
  function showTry() {
    const r = result, o = r.options[focus];
    send({ type: "load", url: o.url, name: `try ${focus + 1}`, frame: r.frame - 1 });
    setTimeout(() => {
      send({ type: "ghosts", ghosts: [] });
      send({ type: "range", start: r.frame - 1, end: r.frame + 300 }); send({ type: "play" });
    }, 400);
    const raw = (x) => x.dealt - x.taken + 40 * (x.stocks_taken - x.stocks_lost);
    $("#opts").innerHTML = `${r.unsettled?.length ? `<div class="warnline">${r.unsettled.map((q) => (q === s.me.port ? "You" : esc(s.opp.char_name))).join(" and ")} couldn't get into the position you picked, so these start from where the game put ${r.unsettled.length > 1 ? "them" : "it"}.</div>` : ""}<div class="summary">${r.options.length} ways it plays out from your setup (5 seconds each). Models of master players play both sides. Click one to watch it up close; pause anywhere and "Continue from here" to play out that moment again.</div>` +
      r.options.map((x) => `<div class="opt ${x.i === focus ? "focus" : ""}" data-i="${x.i}"><div class="dot" style="background:${x.color}"></div>
        <div><div class="lbl">You: ${esc(x.label)}</div><div class="res">Opponent: ${esc(x.opp_label)}<br>dealt ${x.dealt.toFixed(0)}% · took ${x.taken.toFixed(0)}%${x.stocks_taken ? ` · took a stock` : ""}${x.stocks_lost ? ` · lost a stock` : ""}</div></div>
        <div style="text-align:right"><span class="net ${raw(x) > 2 ? "pos" : raw(x) < -2 ? "neg" : "zero"}">${raw(x) > 0 ? "+" : ""}${raw(x).toFixed(0)}</span><div><a class="dl" href="${x.url}" download="lab_${x.i}.slp">.slp</a></div></div></div>`).join("");
    $("#opts").querySelectorAll(".opt").forEach((el) => (el.onclick = (e) => { if (e.target.closest("a")) return; focus = +el.dataset.i; showTry(); }));
    tickLab();
  }
  $("#go").onclick = () => playOut(null);
  $("#cont").onclick = () => { send({ type: "pause" }); playOut({ from_job: result.id, from_try: focus, frame: cur }); };
  // Try it yourself: the setup, played live (makeLive)
  let labAlive = true;
  const live = makeLive({
    id, s, send, iframe, panel: () => $("#opts"), cur: () => cur,
    busy: (b) => { $("#go").disabled = b; $("#tryLab").disabled = b; },
    youScore: () => null,
    watch: (st) => { send({ type: "range", start: st.moment - 20, end: st.end }); send({ type: "speed", fps: 15 }); send({ type: "seek", frame: st.moment - 20 }); send({ type: "play" }); },
    back: () => {
      if (!labAlive) return;
      send({ type: "speed", fps: 60 });
      if (result) showTry(); else { send({ type: "load", url: `/api/game/${id}/replay.slp`, name: s.file, frame: at }); $("#opts").innerHTML = ""; }
    },
  });
  $("#tryLab").onclick = () => live.start({ lab: { at, ...setup() } });
  const onKey = (e) => {
    if (e.target.closest("input") || live.on()) return;
    if ((e.key === "r" || e.key === "R") && live.again()) { e.preventDefault(); return; }
    if (e.code === "Space") { e.preventDefault(); send({ type: "toggle" }); }
  };
  window.addEventListener("keydown", onKey);
  cleanup = () => { labAlive = false; live.end(true); window.removeEventListener("message", onMsg); window.removeEventListener("keydown", onKey); };
  tickLab();
}

route();
