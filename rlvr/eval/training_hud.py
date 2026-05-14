"""Live training dashboard — minimal, pleasant, viewer-focused.

Tails the training log, renders a 960x540 pygame window:

  +-- HEADER ----------------------------------------+
  |  RLVR live · combo_extend       update X/50  ▓▓▓ |
  +-- PLAYERS ---------------------------------------+
  |  P1: trainee (learning combo extension)          |
  |  P2: frozen BC baseline (fixed opponent)         |
  +-- HERO STATS ------------------------------------+
  |       42                    87                   |
  |  STOCKS TAKEN            COMBOS                  |
  +-- RECENT REWARDS --------------------------------+
  |  +0.85  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░  combo · 65% damage     |
  |  +1.00  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  KO from 45%            |
  |  +0.18  ▓▓░░░░░░░░░░░░░░  combo · 14% damage     |
  +-- TREND -----------------------------------------+
  |  combo rate    ▁▂▃▅▇▇▆▇▇▇▇▇   45%                |
  +-- FOOTER ----------------------------------------+
  |  12 matches · 2m 14s · health: stable            |
  +--------------------------------------------------+

Bar color is viridis (perceptually-uniform) so reward magnitude is
encoded in a way humans actually read it. Everything else is
greyscale to keep the eye on the bars.

Auto-launched by rlvr/online/loop.py when training is viewable.
"""
from __future__ import annotations

import argparse
import re
import time
from collections import deque
from pathlib import Path

try:
    import pygame
except ImportError:
    raise SystemExit("pygame missing. pip3 install --user --break-system-packages pygame")


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
_RE_MATCH_END = re.compile(r"match ended, returning to menu")
_RE_FROZEN = re.compile(r"loading frozen opponent: (\S+)")


# ---------------------------- Palette ---------------------------------
# Quiet greyscale base; viridis only for reward bars.
BG       = (16, 18, 22)
INK      = (228, 230, 234)
DIM      = (140, 144, 154)
SUBDIM   = (78, 82, 92)
LINE     = (28, 32, 40)

# Reward-bar colormap = viridis (matplotlib).
# Anchor points sampled at 0/0.25/0.5/0.75/1.0; lerped between.
_VIRIDIS = [
    (0.00, (68, 1, 84)),       # deep purple
    (0.25, (59, 82, 139)),     # blue
    (0.50, (33, 145, 140)),    # teal
    (0.75, (94, 201, 98)),     # green
    (1.00, (253, 231, 37)),    # yellow
]


def viridis(t: float):
    """Return RGB for t ∈ [0, 1] along the viridis colormap."""
    t = max(0.0, min(1.0, t))
    for i in range(len(_VIRIDIS) - 1):
        t0, c0 = _VIRIDIS[i]
        t1, c1 = _VIRIDIS[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
    return _VIRIDIS[-1][1]


def _lerp(a, b, t):
    return a + (b - a) * t


def _pick_font(size, bold=False):
    for name in ("notosans", "inter", "dejavusans", "liberationsans", "sans"):
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f is not None:
                return f
        except Exception:
            pass
    return pygame.font.SysFont(None, size, bold=bold)


class HUDState:
    def __init__(self):
        self.task_name = "combo_extend"
        self.task_description = "learning to extend punishes on opp"
        self.opponent = "frozen BC"
        self.update = 0
        self.max_updates = 50
        self.session_start = time.time()

        self.stocks_taken_total = 0
        self.combos_total = 0
        self.stocks_disp = 0.0
        self.combos_disp = 0.0

        self.update_combo_rates = deque(maxlen=40)
        self.last_kl = 0.0
        self.last_event_time = 0.0  # for subtle pulse on bar appearance

        self.recent_events = deque(maxlen=5)
        self.matches_total = 0

        # Live episode-window state. window_open is True between the
        # most recent EVT_EP_OPEN and the matching EVT_EP close. The
        # HUD plots a live opp-percent trajectory while this is True.
        self.window_open = False
        self.window_open_time = 0.0
        self.window_open_frame = 0
        self.window_start_pct = 0.0
        # (elapsed_frames_since_open, opp_pct) tuples for the live plot.
        self.window_trajectory = []
        # Last completed trajectory + result, kept for brief afterimage.
        self.last_trajectory = []
        self.last_close_time = -1e9
        self.last_close_result = ""
        # "Buzz wrong" timer: when a window closes with single_hit /
        # sub_threshold / aborted, briefly flash the plot panel red.
        self.buzz_until = -1e9

    def feed_event(self, m):
        result = m.group("r")
        reward = float(m.group("rw"))
        try:
            dmg = float(m.group("d") or "0")
        except (TypeError, ValueError):
            dmg = 0.0
        try:
            sp = float(m.group("sp") or "0")
        except ValueError:
            sp = 0.0
        kind = None
        if result == "combo_kill":
            self.stocks_taken_total += 1
            self.combos_total += 1
            kind = "ko"
        elif result == "combo":
            self.combos_total += 1
            kind = "combo"
        if kind is not None:
            self.recent_events.appendleft({
                "t": time.time(), "kind": kind,
                "reward": reward, "damage": dmg, "start_pct": sp,
            })
            self.last_event_time = time.time()
        # Mark the window as closed regardless of kind (sub_threshold
        # and single_hit close too, just don't enter the recent list).
        self.window_open = False
        self.last_close_time = time.time()
        self.last_close_result = result
        # Snapshot the trajectory so it can be drawn briefly post-close.
        self.last_trajectory = list(self.window_trajectory)
        self.window_trajectory = []
        # Buzz on non-rewarded outcomes (the bot tried but didn't
        # extend / didn't damage). Combo + combo_kill don't buzz.
        if result not in ("combo", "combo_kill"):
            self.buzz_until = time.time() + 0.6

    def feed_open(self, m):
        self.window_open = True
        self.window_open_time = time.time()
        try:
            self.window_open_frame = int(m.group("f") or "0")
        except ValueError:
            self.window_open_frame = 0
        try:
            self.window_start_pct = float(m.group("sp") or "0")
        except ValueError:
            self.window_start_pct = 0.0
        # Seed trajectory at (0, start_pct) so the line starts at the
        # right Y from frame 0.
        self.window_trajectory = [(0, self.window_start_pct)]

    def feed_tick(self, m):
        if not self.window_open:
            return
        try:
            f = int(m.group("f"))
            p = float(m.group("p"))
        except (TypeError, ValueError):
            return
        elapsed = max(0, f - self.window_open_frame)
        self.window_trajectory.append((elapsed, p))

    def feed_update(self, m):
        self.update = int(m.group("u"))
        res_str = m.group("res")
        total = int(m.group("v"))
        c = re.search(r"'combo':\s*(\d+)", res_str)
        ck = re.search(r"'combo_kill':\s*(\d+)", res_str)
        n = (int(c.group(1)) if c else 0) + (int(ck.group(1)) if ck else 0)
        rate = n / max(1, total)
        self.update_combo_rates.append(rate)
        self.last_kl = float(m.group("kl"))

    def feed_match_end(self):
        self.matches_total += 1

    def feed_frozen(self, path):
        self.opponent = f"frozen {Path(path).stem}"

    def tick(self, dt):
        self.stocks_disp = _lerp(self.stocks_disp,
                                  float(self.stocks_taken_total),
                                  min(1.0, dt * 5.0))
        self.combos_disp = _lerp(self.combos_disp,
                                  float(self.combos_total),
                                  min(1.0, dt * 5.0))


class HUDRenderer:
    def __init__(self, w=960, h=540):
        self.w = w
        self.h = h
        self.f = {
            "hero":  _pick_font(78, bold=True),
            "h":     _pick_font(20, bold=True),
            "body":  _pick_font(16),
            "small": _pick_font(14),
            "tiny":  _pick_font(12),
        }

    def draw(self, screen, st: HUDState):
        screen.fill(BG)
        self._header(screen, st)
        self._live_plot(screen, st)
        self._rewards(screen, st)
        self._footer(screen, st)

    def _hline(self, screen, y):
        pygame.draw.line(screen, LINE, (24, y), (self.w - 24, y), 1)

    def _header(self, screen, st):
        # Two-line header. Top: title + task. Bottom: P1/P2 identity
        # + update progress bar. All on a single visual block.
        y1 = 12
        title = self.f["h"].render("RLVR live", True, INK)
        screen.blit(title, (24, y1))
        dot = self.f["h"].render("·", True, SUBDIM)
        screen.blit(dot, (24 + title.get_width() + 10, y1))
        sub = self.f["body"].render(st.task_name, True, DIM)
        screen.blit(sub, (24 + title.get_width() + 28, y1 + 4))

        # Update + progress bar (right).
        upd = self.f["body"].render(f"update {st.update}/{st.max_updates}",
                                     True, DIM)
        bar_w, bar_h = 200, 3
        bar_x = self.w - bar_w - 24
        bar_y = y1 + 9
        screen.blit(upd, (bar_x - upd.get_width() - 14, y1 + 2))
        pygame.draw.rect(screen, LINE,
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h),
                         border_radius=2)
        fill = int(bar_w * st.update / max(1, st.max_updates))
        if fill > 0:
            pygame.draw.rect(screen, DIM,
                             pygame.Rect(bar_x, bar_y, fill, bar_h),
                             border_radius=2)

        # P1 vs P2 identity line.
        y2 = y1 + 40
        line = (f"P1 · trainee  (default Fox)   "
                f"vs   P2 · {st.opponent}  (green Fox)")
        screen.blit(self.f["small"].render(line, True, DIM), (24, y2))
        self._hline(screen, y2 + 28)

    def _players(self, screen, st):
        y = 64
        # P1: trainee
        p1 = self.f["body"].render("P1", True, INK)
        screen.blit(p1, (24, y))
        p1d = self.f["body"].render(
            f"  trainee  ·  {st.task_description}", True, DIM)
        screen.blit(p1d, (24 + p1.get_width(), y))
        # P2: opponent
        p2 = self.f["body"].render("P2", True, INK)
        screen.blit(p2, (24, y + 22))
        p2d = self.f["body"].render(
            f"  {st.opponent}  ·  fixed (no learning)", True, DIM)
        screen.blit(p2d, (24 + p2.get_width(), y + 22))
        self._hline(screen, y + 52)

    def _live_plot(self, screen, st):
        """Live plot: X = game-frames since window open, Y = opp
        percent (auto-scaled). Trajectory drawn live. On bad close,
        panel briefly tints red."""
        x = 24
        y = 130
        w = self.w - 48
        h = 200

        # Buzz: red tint on bad close, fades over 0.6s.
        buzz_t = max(0.0, st.buzz_until - time.time())
        # Pick traj to draw + post-close fade alpha.
        post_close_age = time.time() - st.last_close_time
        if st.window_open:
            traj = st.window_trajectory
            fade_alpha = 1.0
        elif post_close_age < 1.2:
            traj = st.last_trajectory
            fade_alpha = max(0.25, 1.0 - post_close_age / 1.2)
        else:
            traj = []
            fade_alpha = 0.0

        # Dynamic Y range. Round peak up to nearest 25%, floor 50%.
        if traj:
            peak = max(p for _, p in traj)
            y_max = max(50.0, ((int(peak / 25) + 1) * 25.0))
        else:
            y_max = 100.0

        # Plot frame.
        if buzz_t > 0:
            a = buzz_t / 0.6
            bg_color = (
                int(BG[0] + (50 - BG[0]) * a),
                int(BG[1] + (10 - BG[1]) * a),
                int(BG[2] + (14 - BG[2]) * a),
            )
        else:
            bg_color = BG
        pygame.draw.rect(screen, bg_color,
                         pygame.Rect(x, y, w, h), border_radius=4)
        pygame.draw.rect(screen, LINE,
                         pygame.Rect(x, y, w, h), width=1, border_radius=4)

        # Status label (single line, top-left).
        if st.window_open:
            elapsed = time.time() - st.window_open_time
            label = f"window open · {elapsed:.1f}s"
            label_color = INK
        elif buzz_t > 0:
            label = f"no extension · {st.last_close_result}"
            label_color = (235, 110, 120)
        elif post_close_age < 1.2:
            label = f"closed · {st.last_close_result}"
            label_color = DIM
        else:
            label = "no window open"
            label_color = SUBDIM
        screen.blit(self.f["small"].render(label, True, label_color),
                    (x + 14, y + 10))

        # Right side: live percent + peak (compact).
        if traj:
            cur_pct = traj[-1][1]
            peak = max(p for _, p in traj)
            cur_s = self.f["h"].render(f"{cur_pct:.0f}%", True, INK)
            screen.blit(cur_s, (x + w - cur_s.get_width() - 14, y + 6))
            peak_s = self.f["small"].render(f"peak {peak:.0f}%",
                                             True, DIM)
            screen.blit(peak_s, (x + w - peak_s.get_width() - 14,
                                  y + 6 + cur_s.get_height() + 2))

        # Plot region.
        plot_x = x + 50
        plot_y = y + 36
        plot_w = w - 64
        plot_h = h - 56

        # Y-axis: 3-4 gridlines depending on y_max.
        if y_max <= 75:
            ticks = (0, 25, 50, 75)
        elif y_max <= 150:
            ticks = (0, 50, 100, 150)
        else:
            step = int((y_max + 49) // 50) * 50 // 3
            step = max(50, step)
            ticks = tuple(range(0, int(y_max) + 1, step))
        for pct in ticks:
            if pct > y_max:
                continue
            yy = plot_y + plot_h - int(plot_h * pct / y_max)
            lbl = self.f["tiny"].render(f"{pct}", True, SUBDIM)
            screen.blit(lbl, (x + 14, yy - 6))
            pygame.draw.line(screen, LINE,
                             (plot_x, yy), (plot_x + plot_w, yy), 1)

        # Trajectory line.
        if len(traj) >= 1:
            max_elapsed = max(traj[-1][0], 45)
            pts = []
            for ef, pct in traj:
                px = plot_x + int(plot_w * ef / max(1, max_elapsed))
                py = (plot_y + plot_h
                      - int(plot_h * min(pct, y_max) / y_max))
                pts.append((px, py))
            peak = max(p for _, p in traj)
            line_color = viridis(min(1.0, peak / 150.0))
            line_color = tuple(int(c * fade_alpha) for c in line_color)
            if len(pts) >= 2:
                pygame.draw.lines(screen, line_color, False, pts, 3)
            pygame.draw.circle(screen, line_color, pts[-1], 5)

        self._hline(screen, y + h + 8)

    def _rewards(self, screen, st):
        y = 350
        screen.blit(self.f["small"].render("recent extensions", True, DIM),
                    (24, y))
        rows_top = y + 22
        row_h = 24
        bar_x = 90
        bar_w = self.w - 24 - 240 - bar_x
        bar_h = 10
        for i, ev in enumerate(st.recent_events):
            ry = rows_top + i * row_h
            rstr = f"+{ev['reward']:.2f}"
            rcolor = INK if ev["reward"] > 0.5 else DIM
            screen.blit(self.f["body"].render(rstr, True, rcolor),
                        (24, ry))
            pygame.draw.rect(screen, LINE,
                             pygame.Rect(bar_x, ry + 6, bar_w, bar_h),
                             border_radius=2)
            t = max(0.0, min(1.0, ev["reward"]))
            fill = int(bar_w * t)
            if fill > 0:
                pygame.draw.rect(screen, viridis(t),
                                 pygame.Rect(bar_x, ry + 6, fill, bar_h),
                                 border_radius=2)
            age = time.time() - ev["t"]
            if ev["kind"] == "ko":
                desc = f"KO from {ev['start_pct']:.0f}%"
            else:
                desc = f"combo · {ev['damage']:.0f}%"
            descx = bar_x + bar_w + 14
            screen.blit(self.f["body"].render(desc, True, INK),
                        (descx, ry))
            ages = self.f["small"].render(f"{age:.0f}s", True, SUBDIM)
            screen.blit(ages, (self.w - ages.get_width() - 24, ry + 2))
        if not st.recent_events:
            screen.blit(self.f["body"].render("waiting...", True, SUBDIM),
                        (24, rows_top))

    def _trend(self, screen, st):
        y = 444
        label = self.f["small"].render("COMBO RATE / UPDATE", True, DIM)
        screen.blit(label, (24, y))
        spark_x = 220
        spark_y = y + 1
        spark_w = self.w - 24 - 220 - 70
        spark_h = 18
        pygame.draw.rect(screen, LINE,
                         pygame.Rect(spark_x, spark_y, spark_w, spark_h),
                         border_radius=2)
        if len(st.update_combo_rates) >= 2:
            data = list(st.update_combo_rates)
            pts = []
            for i, v in enumerate(data):
                xi = spark_x + int(i * spark_w / max(1, len(data) - 1))
                yi = spark_y + spark_h - int(min(1.0, v) * (spark_h - 4)) - 2
                pts.append((xi, yi))
            pygame.draw.lines(screen, DIM, False, pts, 2)
        v = st.update_combo_rates[-1] if st.update_combo_rates else 0.0
        latest = self.f["body"].render(f"{v * 100:.0f}%", True, INK)
        screen.blit(latest, (self.w - latest.get_width() - 24, y))
        self._hline(screen, y + 28)

    def _footer(self, screen, st):
        y = self.h - 24
        elapsed = int(time.time() - st.session_start)
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        ttext = f"{hrs}h {mins:02d}m" if hrs else f"{mins}m {secs:02d}s"
        kl = st.last_kl
        if kl > 1.0:
            health = "drifting"
        elif kl > 0.2:
            health = "elevated"
        else:
            health = "stable"
        text = f"{st.matches_total} matches  ·  {ttext}  ·  health: {health}"
        screen.blit(self.f["small"].render(text, True, SUBDIM), (24, y))


def run_hud(log_path: Path, width: int = 960, height: int = 540, fps: int = 30):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("RLVR Training HUD")
    clock = pygame.time.Clock()

    state = HUDState()
    renderer = HUDRenderer(width, height)

    fh = None
    if log_path.exists():
        fh = log_path.open("r")
        fh.seek(0, 2)

    last = time.time()
    running = True
    while running:
        now = time.time()
        dt = min(0.1, now - last)
        last = now

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        if fh is None and log_path.exists():
            fh = log_path.open("r")
        if fh is not None:
            while True:
                line = fh.readline()
                if not line:
                    break
                m = _RE_FROZEN.search(line)
                if m:
                    state.feed_frozen(m.group(1))
                m = _RE_OPEN.search(line)
                if m:
                    state.feed_open(m)
                m = _RE_TICK.search(line)
                if m:
                    state.feed_tick(m)
                m = _RE_EVENT.search(line)
                if m:
                    state.feed_event(m)
                m = _RE_UPDATE.search(line)
                if m:
                    state.feed_update(m)
                if _RE_MATCH_END.search(line):
                    state.feed_match_end()

        state.tick(dt)
        renderer.draw(screen, state)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    if fh is not None:
        fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    run_hud(args.log, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
