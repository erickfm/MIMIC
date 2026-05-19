# VR Proposal: `low-percent-kill`

## Definition

An *extra* reward layered on `stock-delta`'s opponent-stock-loss event:
when the opponent loses a stock, if their percent at death was below a
per-character "low-percent" threshold, add a bonus. It rewards
*efficient* kills — taking the stock early off a hard read, a gimp, or
a strong conversion — over grinding the opponent to 150%+.

This is **not a standalone VR**. It rides on the `stock-delta`
opp-stock event; with no kill there is nothing to bonus. It is a
re-weighting of `stock-delta` toward early kills, not a separate
signal.

## Reward spec

- **Fires when:** the opponent loses a stock (the `stock-delta` `+1`
  event) AND the opponent's percent at death `< BUCKET[opp_character]`.
- **Bonus:** `+0.5` on top of `stock-delta`'s `+1` — a low-% kill is
  worth `+1.5`, a normal kill `+1.0`. Magnitude is tunable (open
  decision).
- **Death percent:** the opponent's percent the frame before the stock
  decrement — robustly, `max(opp.percent)` over the ~12 frames before
  the decrement (percent holds through the death animation, then
  resets to 0 on respawn; the window guards the reset-timing edge).
- **SD gate (recommended):** apply the bonus only if the opponent was
  in hitstun / a DAMAGE action-state shortly before death. A clean
  self-destruct at low percent should not earn a "skillful early kill"
  bonus — and SDs cluster exactly in the low-% range. See open
  decisions.
- **Delivery:** adds to the per-frame reward on the stock-decrement
  frame.

## Per-character bucket table

`BUCKET[char]` = the **15th percentile of percent-at-death**, measured
from a 40-shard (~2,300-game, ~12k-death) sample of `data/fox_all_v2`
(all-ranks ranked replays). A kill below the bucket is in the earliest
~15% of kills against that character.

| char (id)        | bucket  | char (id)      | bucket  |
|------------------|---------|----------------|---------|
| Falco (22)       | < 90%   | Jigglypuff (15)| < 110%  |
| Marth (18)       | < 95%   | Sheik (7)      | < 120%  |
| CptFalcon (2)    | < 100%  | Peach (9)      | < 125%  |
| Fox (1)          | < 105%  | Yoshi (13)     | < 135%  |
|                  |         | DK (3)         | < 140%  |

Provenance notes:
1. **Absolute percents run high** — `fox_all_v2` is the full ranked
   ladder, not tournament play, so kill conversion is poor. These are
   *relative* p15 cuts, not tournament intuition.
2. Falco's bucket (90) sits well below Fox's (105) because Falco has a
   fat low tail of gimp deaths; Jigglypuff's (110) stays high because
   Puff's elite recovery means it is rarely gimped and dies mostly to
   clean kill moves.

**Characters not yet measured** (< 150 deaths in the sample — Luigi,
Pikachu, Samus, Ganondorf, Ice Climbers, etc.): fallback `< 110%`
until measured. **Luigi is a production frozen-BC opponent and must be
measured before any real run** — a supplementary scan over more shards
will fill it (and the rest of the cast) in.

## Feasibility — easy

No new slippistats. Death detection and the percent read are exactly
`stock-delta`'s machinery (a stock decrement + the pre-decrement
percent). The two additions are both cheap:
- `BUCKET` is a `{char_id: float}` dict; the opponent's character is
  `libmelee.PlayerState.character`.
- The SD gate reuses `combo_extend`'s `DAMAGE` range (75-91) and the
  `hitstun_frames_left` field — no new state machine.

## Open decisions

1. **Bonus magnitude.** Proposed `+0.5` (low-% kill = `+1.5` total).
   `+1.0` makes an early kill worth a full extra stock; `+0.25` is a
   gentle nudge. Tunable.
2. **SD gate.** Recommended: require the opponent to have been in
   hitstun / a DAMAGE state shortly before death, so a clean SD at low
   percent does not earn the bonus. Imperfect for *slow* gimps (opp
   hit offstage, flails out of hitstun, then dies) — those may miss
   the gate. Alternatives: (a) no gate — any low-% opp death bonuses
   (simplest, rewards opponent SDs); (b) full kill-attribution — most
   accurate, but costs the `combo_extend` hitstun-tracking machinery.
   The heuristic gate is the recommended middle ground.

## Causal status

Neither a proxy nor a corollary — it is a *re-weighting* of
`stock-delta` (the objective itself) toward early kills, never a
substitute. A normal-percent kill still earns the full `stock-delta`
`+1`; the bonus only adds. So there is no Goodhart-substitution risk.
The open question is whether biasing toward early kills *helps*
win-rate — plausible (early kills are efficient: less time at risk,
less damage required) but a hypothesis to test, not a given.

## Reward-hacking risk

Low. The bonus is gated on an actual stock being taken — it cannot be
farmed independently. Because it is *additive* (not instead-of), there
is no perverse incentive to pass up a safe 130% kill — that kill still
pays `+1`. The one residual concern, rewarding the bot for an
opponent's low-% self-destruct, is what the SD gate addresses.

## Eval

Head-to-head win-rate vs frozen BC, plus a direct behavioral check:
measure the trained bot's *own* kill-percent distribution and confirm
it shifts earlier. If kills move earlier but win-rate does not, the
early-kill bias was cosmetic — a conversation-1 outcome.

## Relation to existing tasks

A layer on `stock-delta` — same event, conditioned on death percent.
Complements `combo_extend` (which rewards damage *during* a punish) by
rewarding the punish *ending in a kill, early*. Independent of
`neutral-win-loss`. Together the four decompose a stock cleanly:
`neutral-win` (won the exchange) → `combo_extend` (converted it) →
`low-percent-kill` (closed it out early) → `stock-delta` (closed it
out at all).
