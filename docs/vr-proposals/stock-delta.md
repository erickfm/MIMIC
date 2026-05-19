# VR Proposal: `stock-delta`

## Definition

Reward the policy `+1` each time the opponent loses a stock and `−1`
each time the bot loses a stock. Symmetric.

Summed over a match, the per-event rewards equal the **final stock
differential** — and a Melee game is won by `sign(stock margin)`. So
this VR is not a *proxy* for the outcome, it is the **complete dense
decomposition of the outcome itself**.

> Originally drafted as positive-only `stock-taken`; made symmetric on
> request. Symmetry is exactly what removes the trade-indifference
> degeneracy of the positive-only version: a stock-for-stock trade
> nets `+1 − 1 = 0`, correctly valued as neutral rather than as a win.

## Reward events

- Opponent's stock count drops between two consecutive in-game frames
  → `+1`.
- Bot's stock count drops between two consecutive in-game frames
  → `−1`.
- **Magnitude:** `±1` per stock (Melee removes one stock at a time).
  Every stock is worth the same — no last-stock bonus.
- **Detection:** two integer comparisons of `stock` against the
  previous in-game frame (`state_history[-1]` vs `state_history[-2]`),
  one for each player.
- **Delivery:** `EpisodeOutcome.per_frame_reward` — `±1` on the
  decrement frame, `0` elsewhere.

Episode boundaries (whole match / per-stock / windowed) are a
training-loop decision, **not** part of this VR — "player X lost a
stock → `±1`" is well-defined as a per-frame reward under any episode
definition.

**Implementation note — final-stock reconciliation.** When the loser
loses their last stock (`1 → 0`) the match ends almost immediately;
the `1 → 0` frame may not be observed as a per-frame event before
`should_end` fires. `compute_outcome` should cross-check the summed
per-frame stock events against the final stock counts and attach any
unobserved last death as `terminal_reward`, so the episode's total
reward equals the exact final stock margin.

## Feasibility — trivial

Pure libmelee, **zero slippistats**. `melee.PlayerState.stock` is an
`int` populated live every frame. The detector is two integer
comparisons:

    opp_now  < opp_prev   ->  reward += (opp_prev  - opp_now)
    self_now < self_prev  ->  reward -= (self_prev - self_now)

No streaming state machine, no `Action`-range constants, no
faithful-port checklist — contrast `combo_extend`, which needed a
6-round streaming state-machine port of the slippistats combo logic.

**Timing caveat.** `stock` does not decrement on the kill frame. The
victim plays a DYING animation first; the counter drops several frames
later when the death completes — the same lag the `combo_extend` port
handles by keeping `DYING` (0–10) in its keep-alive set. The reward
therefore attaches a few frames *after* the killing blow connected.
Fine for a reward signal; just note the credited frame is not the
causal frame.

## Open decision

**SD attribution.** Count *every* stock change on both sides
(self-destructs included), or only bot-attributed kills / non-SD bot
deaths?

**Recommendation: count everything, symmetrically.** An SD loses a
stock and loses the game exactly as a kill does — the game does not
care how the stock was lost, so neither should the reward. It needs
zero extra logic, and the bot cannot farm opponent SDs (the opponent
is an independent frozen policy). Kill-attribution would require
inferring "was the player in hitstun from the other's hit just before
death" — i.e. re-deriving the `combo_extend` hitstun tracking — for
marginal signal-quality gain.

(The positive-only / trade-indifference question from the earlier
draft is now resolved by symmetry and no longer open.)

## Causal status

Maximal. Every prior VR candidate (combo damage, hitstun, l-cancels)
is a proxy that *correlates* with winning. `stock-delta`'s per-event
rewards **sum to the final stock margin**, and the win is
`sign(stock margin)` — so this VR is the outcome itself, densified,
not a correlate of it. There is no common-cause concern: nothing is
being substituted for the outcome.

## Reward-hacking risk

Un-hackable: the signal is the objective, it is symmetric, and it sums
to the game result. The trade-indifference degeneracy of the
positive-only draft is eliminated. The bot cannot manufacture opponent
stock losses directly — the opponent is a separate frozen policy.

## Eval

Head-to-head win-rate vs frozen BC:

    tools/play.py --ckpt <trained> --opponent <frozen-BC> \
        --n-matches N --out reports/stock-delta.json

Expect win-rate to genuinely move — this is the objective, decomposed.
`avg_a_stocks − avg_b_stocks` in the report should track the trained
reward directly.

## Relation to existing tasks

`combo_extend` / `l_cancel` / `shield_escape` are all proxies *for*
the stock margin. `stock-delta` is the ground-truth outcome they
approximate — so it doubles as the **reference VR** for the
correlation-vs-causation test: train on a proxy, train on
`stock-delta`, compare win-rate. If the proxy adds nothing over
`stock-delta`, it was a corollary.
