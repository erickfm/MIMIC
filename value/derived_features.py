"""Engineered per-frame features derived from shard state.

Mirrors slippistats's canonical event-detection logic
(`slippistats/stats/combo_computer.py` + `slippistats/stats/common.py`)
but operates directly on our shard tensors rather than reparsing .slps.

What we compute, per frame, per game (using slippistats's terminology):

Action-state indicators (binary):
  - opp_in_damaged          opp action in DAMAGE_START..DAMAGE_END (75-91)
  - opp_in_grabbed          opp action in CAPTURE_START..CAPTURE_END (223-232)
  - opp_in_tech_miss/down   opp action in DOWN_START..DOWN_END (183-198)
  - opp_in_tech_success     opp action in TECH_START..TECH_END (199-204)
  - opp_in_shielding        opp action in GUARD_START..GUARD_END (178-182)
  - opp_in_dodging          opp action in DODGE_START..DODGE_END (233-236)
  - opp_in_dying            opp action in DYING_START..DYING_END (0-10)
  - opp_in_hitstun          opp_hitstun_left_raw > 0 (from shard's already-stored hitstun)
  - opp_in_punish_state     damaged OR grabbed OR in_hitstun OR in_tech_miss
                            (matches slippistats's combo-continuation condition)
  ... and mirrored for self.

Combo tracking (per slippistats's combo_compute state machine):
  - combo_on_opp_active     bool: we have an active combo on opp
  - combo_on_opp_hits       int: hits landed on opp in current combo
  - combo_on_opp_damage     float: percent damage dealt in current combo
  - combo_on_opp_frames     int: frames since current combo started
  - combo_on_self_active    bool (mirror)
  - combo_on_self_hits      int (mirror)
  - combo_on_self_damage    float (mirror)
  - combo_on_self_frames    int (mirror)

Per-game cumulative counts (running total over game so far):
  - game_opp_stocks_taken   stocks self has knocked off opp
  - game_self_stocks_lost   stocks opp has knocked off self
  - game_combos_won         combos self has completed against opp
  - game_combos_lost        combos opp has completed against self
  - game_damage_dealt       cumulative percent dealt by self
  - game_damage_taken       cumulative percent taken by self

Frame-since timers (clamped at 600 to avoid runaway):
  - frames_since_self_landed_hit
  - frames_since_self_took_hit
  - frames_since_in_neutral
  - frames_since_combo_on_opp_ended
  - frames_since_combo_on_self_ended

All outputs are numpy float32 arrays of length (total_frames_in_shard,).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


# slippistats action-state ranges (from slippistats/enums/state.py:191-211).
# Values are libmelee Action enum integers, which is exactly what shards store.
DAMAGE_START, DAMAGE_END = 75, 91
CAPTURE_START, CAPTURE_END = 223, 232
GUARD_START, GUARD_END = 178, 182
DOWN_START, DOWN_END = 183, 198
TECH_START, TECH_END = 199, 204
DODGE_START, DODGE_END = 233, 236
DYING_START, DYING_END = 0, 10

# The "punish-continuation" set (matches slippistats's combo-active check).
# Combo continues while opp is in any of these states (or in_hitstun, checked
# separately because hitstun is from a bitflag we don't have, so we use
# hitstun_frames_left > 0 as a proxy).
def _in_range(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return ((arr >= lo) & (arr <= hi)).astype(np.float32)


# slippistats uses the percent column to detect a hit landed. The shard
# stores percent in self_numeric[:, PERCENT_COL] but it's normalized. We
# need to detect a percent INCREASE between consecutive frames.
PERCENT_COL = 2  # self_numeric / opp_numeric column index
STOCK_COL = 3


def compute_derived_features(
    shard_states: Dict[str, torch.Tensor],
    offsets: torch.Tensor,
    n_games: int,
) -> Dict[str, np.ndarray]:
    """Walk each game in the shard and emit per-frame engineered features.

    Returns a dict of numpy arrays, each shape (total_frames,). Concatenated
    across games using the same offsets array as the rest of the shard, so
    you can index by absolute frame index.
    """
    # Pull the columns we need once, convert to numpy.
    opp_action = shard_states["opp_action"].numpy()
    self_action = shard_states["self_action"].numpy()
    opp_hitstun_raw = shard_states["opp_numeric"][:, 11].numpy()  # hitstun_left (col 11)
    self_hitstun_raw = shard_states["self_numeric"][:, 11].numpy()
    opp_percent_norm = shard_states["opp_numeric"][:, PERCENT_COL].numpy()
    self_percent_norm = shard_states["self_numeric"][:, PERCENT_COL].numpy()
    opp_stock_norm = shard_states["opp_numeric"][:, STOCK_COL].numpy()
    self_stock_norm = shard_states["self_numeric"][:, STOCK_COL].numpy()

    total_frames = opp_action.shape[0]

    # --- Vectorized binary state indicators (action-range checks) ---
    out: Dict[str, np.ndarray] = {}
    out["opp_in_damaged"] = _in_range(opp_action, DAMAGE_START, DAMAGE_END)
    out["opp_in_grabbed"] = _in_range(opp_action, CAPTURE_START, CAPTURE_END)
    out["opp_in_downed"] = _in_range(opp_action, DOWN_START, DOWN_END)
    out["opp_in_tech_success"] = _in_range(opp_action, TECH_START, TECH_END)
    out["opp_in_shielding"] = _in_range(opp_action, GUARD_START, GUARD_END)
    out["opp_in_dodging"] = _in_range(opp_action, DODGE_START, DODGE_END)
    out["opp_in_dying"] = _in_range(opp_action, DYING_START, DYING_END)
    # hitstun_frames_left is normalized in the shard via log_max(20) per CLAUDE.md
    # (or raw if older shards). Either way, > -1 means hitstun > 0.
    # The transform is log1p(clamp(x,0,120))/log1p(120). At x=0, normalized=0.
    # So opp_hitstun_raw > 0 means actual hitstun > 0.
    out["opp_in_hitstun"] = (opp_hitstun_raw > 0).astype(np.float32)
    out["self_in_damaged"] = _in_range(self_action, DAMAGE_START, DAMAGE_END)
    out["self_in_grabbed"] = _in_range(self_action, CAPTURE_START, CAPTURE_END)
    out["self_in_downed"] = _in_range(self_action, DOWN_START, DOWN_END)
    out["self_in_tech_success"] = _in_range(self_action, TECH_START, TECH_END)
    out["self_in_shielding"] = _in_range(self_action, GUARD_START, GUARD_END)
    out["self_in_dodging"] = _in_range(self_action, DODGE_START, DODGE_END)
    out["self_in_dying"] = _in_range(self_action, DYING_START, DYING_END)
    out["self_in_hitstun"] = (self_hitstun_raw > 0).astype(np.float32)

    # Punish-state composite (matches slippistats combo-active set)
    out["opp_in_punish_state"] = np.clip(
        out["opp_in_damaged"] + out["opp_in_grabbed"]
        + out["opp_in_downed"] + out["opp_in_hitstun"],
        0, 1,
    )
    out["self_in_punish_state"] = np.clip(
        out["self_in_damaged"] + out["self_in_grabbed"]
        + out["self_in_downed"] + out["self_in_hitstun"],
        0, 1,
    )

    # --- Per-game sequential state machine for combo tracking ---
    combo_on_opp_active = np.zeros(total_frames, dtype=np.float32)
    combo_on_opp_hits = np.zeros(total_frames, dtype=np.float32)
    combo_on_opp_damage = np.zeros(total_frames, dtype=np.float32)
    combo_on_opp_frames = np.zeros(total_frames, dtype=np.float32)
    combo_on_self_active = np.zeros(total_frames, dtype=np.float32)
    combo_on_self_hits = np.zeros(total_frames, dtype=np.float32)
    combo_on_self_damage = np.zeros(total_frames, dtype=np.float32)
    combo_on_self_frames = np.zeros(total_frames, dtype=np.float32)
    game_opp_stocks_taken = np.zeros(total_frames, dtype=np.float32)
    game_self_stocks_lost = np.zeros(total_frames, dtype=np.float32)
    game_combos_won = np.zeros(total_frames, dtype=np.float32)
    game_combos_lost = np.zeros(total_frames, dtype=np.float32)
    game_damage_dealt = np.zeros(total_frames, dtype=np.float32)
    game_damage_taken = np.zeros(total_frames, dtype=np.float32)
    frames_since_self_landed_hit = np.full(total_frames, 600, dtype=np.float32)
    frames_since_self_took_hit = np.full(total_frames, 600, dtype=np.float32)
    frames_since_in_neutral = np.full(total_frames, 600, dtype=np.float32)
    frames_since_combo_on_opp_ended = np.full(total_frames, 600, dtype=np.float32)
    frames_since_combo_on_self_ended = np.full(total_frames, 600, dtype=np.float32)

    offsets_l = offsets.tolist() if torch.is_tensor(offsets) else offsets

    for g in range(n_games):
        gs = offsets_l[g]
        ge = offsets_l[g + 1]
        # State machine variables (reset per game)
        opp_combo_open = False
        opp_combo_start = 0           # frame within game
        opp_combo_start_pct = 0.0     # normalized percent at start
        opp_combo_last_hit_action = -1  # to dedupe hits from same animation
        opp_hits_in_combo = 0
        self_combo_open = False
        self_combo_start = 0
        self_combo_start_pct = 0.0
        self_combo_last_hit_action = -1
        self_hits_in_combo = 0

        cum_dealt = 0.0
        cum_taken = 0.0
        opp_stocks_taken = 0
        self_stocks_lost = 0
        combos_won = 0
        combos_lost = 0

        prev_opp_pct = opp_percent_norm[gs]
        prev_self_pct = self_percent_norm[gs]
        prev_opp_stk = opp_stock_norm[gs]
        prev_self_stk = self_stock_norm[gs]
        prev_self_action_local = self_action[gs]
        prev_opp_action_local = opp_action[gs]

        ts_self_landed = 600
        ts_self_took = 600
        ts_neutral = 600
        ts_opp_combo_ended = 600
        ts_self_combo_ended = 600

        for fi in range(gs, ge):
            cur_opp_pct = opp_percent_norm[fi]
            cur_self_pct = self_percent_norm[fi]
            cur_opp_stk = opp_stock_norm[fi]
            cur_self_stk = self_stock_norm[fi]
            cur_self_action = self_action[fi]
            cur_opp_action = opp_action[fi]

            # Stock changes (normalized stock decreases when player loses a stock)
            if cur_opp_stk < prev_opp_stk - 1e-3:
                opp_stocks_taken += 1
            if cur_self_stk < prev_self_stk - 1e-3:
                self_stocks_lost += 1

            # Damage events: percent increased = took damage
            opp_dmg_taken = max(0.0, cur_opp_pct - prev_opp_pct)
            self_dmg_taken = max(0.0, cur_self_pct - prev_self_pct)
            cum_dealt += opp_dmg_taken
            cum_taken += self_dmg_taken

            # --- Combo on opp (we are punishing opp) ---
            opp_in_punish = bool(out["opp_in_punish_state"][fi])
            if opp_in_punish:
                if not opp_combo_open:
                    # Start new combo
                    opp_combo_open = True
                    opp_combo_start = fi
                    opp_combo_start_pct = prev_opp_pct
                    opp_combo_last_hit_action = -1
                    opp_hits_in_combo = 0
                # Detect a new hit within combo (opp took damage AND animation
                # changed since last recorded hit)
                if opp_dmg_taken > 1e-4:
                    action_changed = (cur_self_action != opp_combo_last_hit_action)
                    if action_changed:
                        opp_hits_in_combo += 1
                        opp_combo_last_hit_action = cur_self_action
                        ts_self_landed = 0
            else:
                if opp_combo_open:
                    # Combo just ended
                    opp_combo_open = False
                    combos_won += 1
                    ts_opp_combo_ended = 0

            # --- Combo on self (opp is punishing us) ---
            self_in_punish = bool(out["self_in_punish_state"][fi])
            if self_in_punish:
                if not self_combo_open:
                    self_combo_open = True
                    self_combo_start = fi
                    self_combo_start_pct = prev_self_pct
                    self_combo_last_hit_action = -1
                    self_hits_in_combo = 0
                if self_dmg_taken > 1e-4:
                    action_changed = (cur_opp_action != self_combo_last_hit_action)
                    if action_changed:
                        self_hits_in_combo += 1
                        self_combo_last_hit_action = cur_opp_action
                        ts_self_took = 0
            else:
                if self_combo_open:
                    self_combo_open = False
                    combos_lost += 1
                    ts_self_combo_ended = 0

            # Neutral detection: neither player in punish state, neither shielding/dodging
            in_neutral = (not opp_in_punish and not self_in_punish
                          and out["opp_in_shielding"][fi] == 0
                          and out["self_in_shielding"][fi] == 0
                          and out["opp_in_dodging"][fi] == 0
                          and out["self_in_dodging"][fi] == 0
                          and out["opp_in_dying"][fi] == 0
                          and out["self_in_dying"][fi] == 0)
            if in_neutral:
                ts_neutral = 0

            # Write out
            combo_on_opp_active[fi] = float(opp_combo_open)
            combo_on_opp_hits[fi] = float(opp_hits_in_combo)
            combo_on_opp_damage[fi] = float(cur_opp_pct - opp_combo_start_pct) \
                if opp_combo_open else 0.0
            combo_on_opp_frames[fi] = float(fi - opp_combo_start) \
                if opp_combo_open else 0.0
            combo_on_self_active[fi] = float(self_combo_open)
            combo_on_self_hits[fi] = float(self_hits_in_combo)
            combo_on_self_damage[fi] = float(cur_self_pct - self_combo_start_pct) \
                if self_combo_open else 0.0
            combo_on_self_frames[fi] = float(fi - self_combo_start) \
                if self_combo_open else 0.0
            game_opp_stocks_taken[fi] = float(opp_stocks_taken)
            game_self_stocks_lost[fi] = float(self_stocks_lost)
            game_combos_won[fi] = float(combos_won)
            game_combos_lost[fi] = float(combos_lost)
            game_damage_dealt[fi] = float(cum_dealt)
            game_damage_taken[fi] = float(cum_taken)
            frames_since_self_landed_hit[fi] = float(min(600, ts_self_landed))
            frames_since_self_took_hit[fi] = float(min(600, ts_self_took))
            frames_since_in_neutral[fi] = float(min(600, ts_neutral))
            frames_since_combo_on_opp_ended[fi] = float(min(600, ts_opp_combo_ended))
            frames_since_combo_on_self_ended[fi] = float(min(600, ts_self_combo_ended))

            # Advance timers
            ts_self_landed = min(600, ts_self_landed + 1)
            ts_self_took = min(600, ts_self_took + 1)
            ts_neutral = min(600, ts_neutral + 1)
            ts_opp_combo_ended = min(600, ts_opp_combo_ended + 1)
            ts_self_combo_ended = min(600, ts_self_combo_ended + 1)

            prev_opp_pct = cur_opp_pct
            prev_self_pct = cur_self_pct
            prev_opp_stk = cur_opp_stk
            prev_self_stk = cur_self_stk

    out["combo_on_opp_active"] = combo_on_opp_active
    out["combo_on_opp_hits"] = combo_on_opp_hits
    out["combo_on_opp_damage"] = combo_on_opp_damage
    out["combo_on_opp_frames"] = combo_on_opp_frames
    out["combo_on_self_active"] = combo_on_self_active
    out["combo_on_self_hits"] = combo_on_self_hits
    out["combo_on_self_damage"] = combo_on_self_damage
    out["combo_on_self_frames"] = combo_on_self_frames
    out["game_opp_stocks_taken"] = game_opp_stocks_taken
    out["game_self_stocks_lost"] = game_self_stocks_lost
    out["game_combos_won"] = game_combos_won
    out["game_combos_lost"] = game_combos_lost
    out["game_damage_dealt"] = game_damage_dealt
    out["game_damage_taken"] = game_damage_taken
    out["frames_since_self_landed_hit"] = frames_since_self_landed_hit
    out["frames_since_self_took_hit"] = frames_since_self_took_hit
    out["frames_since_in_neutral"] = frames_since_in_neutral
    out["frames_since_combo_on_opp_ended"] = frames_since_combo_on_opp_ended
    out["frames_since_combo_on_self_ended"] = frames_since_combo_on_self_ended

    return out


def derived_feature_names() -> List[str]:
    """Names of all derived features in the order they would be stacked."""
    return [
        # action-state indicators
        "opp_in_damaged", "opp_in_grabbed", "opp_in_downed",
        "opp_in_tech_success", "opp_in_shielding", "opp_in_dodging",
        "opp_in_dying", "opp_in_hitstun", "opp_in_punish_state",
        "self_in_damaged", "self_in_grabbed", "self_in_downed",
        "self_in_tech_success", "self_in_shielding", "self_in_dodging",
        "self_in_dying", "self_in_hitstun", "self_in_punish_state",
        # combo state
        "combo_on_opp_active", "combo_on_opp_hits",
        "combo_on_opp_damage", "combo_on_opp_frames",
        "combo_on_self_active", "combo_on_self_hits",
        "combo_on_self_damage", "combo_on_self_frames",
        # cumulative
        "game_opp_stocks_taken", "game_self_stocks_lost",
        "game_combos_won", "game_combos_lost",
        "game_damage_dealt", "game_damage_taken",
        # timers
        "frames_since_self_landed_hit",
        "frames_since_self_took_hit",
        "frames_since_in_neutral",
        "frames_since_combo_on_opp_ended",
        "frames_since_combo_on_self_ended",
    ]
