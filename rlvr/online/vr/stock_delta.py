"""stock-delta VR — the objective, decomposed per stock.

`+1` when the opponent loses a stock to the bot, `−1` when the bot loses a
stock. See docs/vr-proposals/stock-delta.md.

SD handling (design interview): the opponent-side `+1` is *gated* — it
fires only if the opponent took a hit shortly before dying, filtering
opponent self-destructs (exogenous noise the bot did not earn). The
bot-side `−1` is **never gated**: a bot self-destruct is the bot's own
failure and must be penalized.

The gate is a streaming tracker (`OppHitRecencyTracker`), not a backward
scan: the `stock` count decrements a long, variable time after the
killing hit — a top/star KO leaves the opponent in a DEAD action state
for 1.5-3 s before `stock` ticks down, so a fixed look-back window from
the decrement frame misses the hit entirely.
"""
from __future__ import annotations

from rlvr.online.slippi_stream import (
    OppHitRecencyTracker, get_opponent, get_player,
)
from rlvr.online.vr.composite import VRModule

# Alive-frames of "recently hit" memory for the opponent-SD gate. DEAD
# frames are transparent (see OppHitRecencyTracker), so this only bounds
# the gap of *alive*, non-hit frames between the last hit and the death.
OPP_HIT_MEMORY = 90


class StockDeltaVR(VRModule):
    id = "stock_delta"

    def __init__(self, self_port: int = 1, hit_memory: int = OPP_HIT_MEMORY):
        self.self_port = self_port
        self.hit_memory = hit_memory
        self.reset()

    def reset(self) -> None:
        self._prev_self_stock = None
        self._prev_opp_stock = None
        self._kills = 0
        self._deaths = 0
        self._filtered_opp_sds = 0
        self._opp_hit = OppHitRecencyTracker(self.hit_memory)

    def observe(self, state_history) -> float:
        gs = state_history[-1]
        self_ps = get_player(gs, self.self_port)
        opp_ps = get_opponent(gs, self.self_port)
        if self_ps is None or opp_ps is None:
            return 0.0
        # Advance the SD-gate tracker every frame — DEAD frames included;
        # they are transparent and keep the pre-death hit "remembered".
        self._opp_hit.update(opp_ps)
        self_stock = int(self_ps.stock)
        opp_stock = int(opp_ps.stock)
        reward = 0.0

        if self._prev_opp_stock is not None and opp_stock < self._prev_opp_stock:
            n = self._prev_opp_stock - opp_stock
            if self._opp_hit.recently_hit:
                reward += float(n)          # the bot earned the kill
                self._kills += n
            else:
                self._filtered_opp_sds += n  # opponent self-destruct — no +1
            self._opp_hit.reset()            # next opponent life starts fresh

        if self._prev_self_stock is not None and self_stock < self._prev_self_stock:
            n = self._prev_self_stock - self_stock
            reward -= float(n)               # always — a bot SD is its own failure
            self._deaths += n

        self._prev_self_stock = self_stock
        self._prev_opp_stock = opp_stock
        return reward

    def finalize(self, state_history) -> float:
        """Reconcile a final death the per-frame loop did not observe (rare
        edge: the actor stops calling observe before the last stock-0
        frame). Normally returns 0."""
        if not state_history:
            return 0.0
        gs = state_history[-1]
        self_ps = get_player(gs, self.self_port)
        opp_ps = get_opponent(gs, self.self_port)
        reward = 0.0
        if (opp_ps is not None and self._prev_opp_stock is not None
                and int(opp_ps.stock) < self._prev_opp_stock):
            n = self._prev_opp_stock - int(opp_ps.stock)
            if self._opp_hit.recently_hit:
                reward += float(n)
                self._kills += n
        if (self_ps is not None and self._prev_self_stock is not None
                and int(self_ps.stock) < self._prev_self_stock):
            n = self._prev_self_stock - int(self_ps.stock)
            reward -= float(n)
            self._deaths += n
        return reward

    def metadata(self) -> dict:
        return {"kills": self._kills, "deaths": self._deaths,
                "filtered_opp_sds": self._filtered_opp_sds}
