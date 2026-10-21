"""Streaming L-cancel episodes for RL in the native sim.

Same episodes and reward as rlvr/online/tasks/l_cancel_online.py, which defines them on Dolphin:
an episode starts when the bot enters an aerial attack (not already in one, airborne, no hitstun)
and ends when it leaves the aerial. If it ends in a LANDING_AIR state, the reward is realized
avoidable lag: the length of that landing-state run, minus the move's L-cancelled floor
(CANCELLED_MIN, Fox-specific), is 0 → 1.0, else 0.0. Landings the opponent interrupts (the run
exits into a damage or dead state) and non-LANDING_AIR exits (autocancels, getting hit) carry no
input-timing signal and are dropped, as there.

On Dolphin the landing run is measured after the match from the .slp; here the sim's post-frames
are exact, so the run is followed live and the episode resolves the frame it exits.
"""
from rlvr.online.tasks.l_cancel_online import (AERIAL_STATES, CANCELLED_MIN, DAMAGE_STATES,
                                               DEAD_STATES, GENERIC_LANDING_STATES,
                                               LANDING_AIR_STATES)
from rlvr.online.trajectory import Episode

TASK_ID = "l_cancel_sim"


class LCancelTracker:
    """One per learning (env, side) slot. Feed observe() every frame (the policy's current
    frame), then record() the decision taken on it. finished holds scored episodes."""

    def __init__(self):
        self.finished = []
        self.stats = dict(success=0, miss=0, dropped=0)
        self.reset()

    def reset(self):
        """A new match: open and pending episodes are abandoned."""
        self.prev_action = None
        self.active = None       # Episode while the aerial is in progress
        self.pending = None      # (episode, landing_state, run_length) while landing

    def observe(self, frame, action, on_ground, hitstun):
        action = int(action)
        if self.pending is not None:
            ep, state, run = self.pending
            if action == state:
                self.pending = (ep, state, run + 1)
            else:
                self.pending = None
                if action in DEAD_STATES or action in DAMAGE_STATES:
                    self.stats["dropped"] += 1       # opponent cut the landing short
                else:
                    avoidable = max(0, run - CANCELLED_MIN[state])
                    ep.terminal_reward = 1.0 if avoidable == 0 else 0.0
                    ep.metadata.update(result="l_cancel_success" if avoidable == 0 else "l_cancel_missed",
                                       realized_lag=run, avoidable_lag=avoidable, exit_state=action)
                    ep.end_game_frame = int(frame)
                    self.stats["success" if avoidable == 0 else "miss"] += 1
                    self.finished.append(ep)
        if self.active is not None:
            if action not in AERIAL_STATES or action in LANDING_AIR_STATES or action in GENERIC_LANDING_STATES:
                ep, self.active = self.active, None
                if action in LANDING_AIR_STATES and ep.frames:
                    ep.metadata["landing_state"] = action
                    self.pending = (ep, action, 1)
                else:
                    self.stats["dropped"] += 1       # autocancel / interrupted: no timing signal
        elif (self.prev_action is not None and action in AERIAL_STATES
              and self.prev_action not in AERIAL_STATES and not on_ground and hitstun == 0):
            self.active = Episode(task_id=TASK_ID, start_game_frame=int(frame))
        self.prev_action = action

    def recording(self):
        return self.active is not None
