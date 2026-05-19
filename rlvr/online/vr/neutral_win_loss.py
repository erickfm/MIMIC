"""neutral-win-loss VR — winning / losing the neutral exchange.

`+1` when the bot wins neutral (the opponent enters punish state on an
ascending edge AND the bot has not been in punish state for the preceding
`K` frames — excludes counter-hits), `−1` the mirror. A trade (both enter
punish within `W` frames) nets 0. See docs/vr-proposals/neutral-win-loss.md.

Trade handling is streaming: an edge emits its `±1` immediately; if the
opposite edge follows within `W` frames it is recognised as a trade and a
correction (`∓1`) is emitted that cancels it. Net 0, a few frames apart.
"""
from __future__ import annotations

from rlvr.online.slippi_stream import get_opponent, get_player, in_any_grab, in_punish_state
from rlvr.online.vr.composite import VRModule

_NEG_INF = -10 ** 9


class NeutralWinLossVR(VRModule):
    id = "neutral_win_loss"

    def __init__(self, self_port: int = 1, win: float = 1.0, loss: float = 1.0,
                 clean_frames: int = 30, trade_window: int = 4):
        self.self_port = self_port
        self.win = win
        self.loss = loss
        self.K = clean_frames     # frames the hitter must be clean of punish
        self.W = trade_window     # both-enter-punish-within-W = a trade
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._self_punish_prev = False
        self._opp_punish_prev = False
        self._self_last_punish_t = _NEG_INF
        self._opp_last_punish_t = _NEG_INF
        self._last_win_t = None    # frame of an unconsumed neutral-win edge
        self._last_loss_t = None
        self._wins = 0
        self._losses = 0
        self._trades = 0

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        self_ps = get_player(gs, self.self_port)
        opp_ps = get_opponent(gs, self.self_port)
        if self_ps is None or opp_ps is None:
            return 0.0
        t = self._t
        self._t += 1

        self_punish = in_punish_state(self_ps)
        opp_punish = in_punish_state(opp_ps)
        # Ascending edges — computed against the previous frame's flags,
        # before this frame's last-punish timestamps are updated.
        opp_edge = opp_punish and not self._opp_punish_prev
        self_edge = self_punish and not self._self_punish_prev

        prev_gs = state_history[-2] if len(state_history) >= 2 else None
        reward = 0.0

        # --- neutral WIN: opponent entered punish, bot clean for K frames --
        if opp_edge:
            bot_clean = (t - self._self_last_punish_t) > self.K
            prev_opp = get_opponent(prev_gs, self.self_port) if prev_gs else None
            dealt = prev_opp is not None and float(opp_ps.percent) > float(prev_opp.percent)
            safe = dealt or in_any_grab(opp_ps)   # damage-or-grab safeguard
            if bot_clean and safe:
                if self._last_loss_t is not None and (t - self._last_loss_t) <= self.W:
                    reward += self.loss          # trade — cancel the prior −loss
                    self._last_loss_t = None
                    self._losses -= 1
                    self._trades += 1
                else:
                    reward += self.win
                    self._wins += 1
                    self._last_win_t = t

        # --- neutral LOSS: bot entered punish, opponent clean for K frames -
        if self_edge:
            opp_clean = (t - self._opp_last_punish_t) > self.K
            prev_self = get_player(prev_gs, self.self_port) if prev_gs else None
            took = prev_self is not None and float(self_ps.percent) > float(prev_self.percent)
            safe = took or in_any_grab(self_ps)
            if opp_clean and safe:
                if self._last_win_t is not None and (t - self._last_win_t) <= self.W:
                    reward -= self.win           # trade — cancel the prior +win
                    self._last_win_t = None
                    self._wins -= 1
                    self._trades += 1
                else:
                    reward -= self.loss
                    self._losses += 1
                    self._last_loss_t = t

        if self_punish:
            self._self_last_punish_t = t
        if opp_punish:
            self._opp_last_punish_t = t
        self._self_punish_prev = self_punish
        self._opp_punish_prev = opp_punish
        return reward

    def metadata(self) -> dict:
        return {"neutral_wins": self._wins, "neutral_losses": self._losses,
                "trades": self._trades}
