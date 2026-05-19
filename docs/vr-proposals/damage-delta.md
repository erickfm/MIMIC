# VR Proposal: `damage-delta`

## Definition

Two dense reward terms on percent change: `+λ_give ×` damage dealt to
the opponent, `−λ_take ×` damage taken by the bot. They are the
**dense companions to `stock-delta`** — where `stock-delta` is the
sparse objective (one event per ~stock), these put signal on every
exchange.

`λ_give` and `λ_take` are **separate knobs**, not one locked symmetric
coefficient (see Coefficients). Both terms are *shaping*, not
objectives — run them only anchored under `stock-delta`.

## Reward events

- Opponent percent rises by Δ → `+λ_give·Δ`.
- Bot percent rises by Δ → `−λ_take·Δ`.
- **Positive deltas only.** A negative delta is a death/respawn reset
  (percent → 0), not healing — ignore it.
- **Detection:** frame-to-frame `PlayerState.percent` diff per player.
  Pure libmelee, one comparison per player per frame.
- **Delivery:** per-frame.

## Coefficients — `λ_take / λ_give` is the aggression knob

Unlike `stock-delta`, where taken and given are locked 1:1 (a stock is
a stock — losing one and taking one matter *exactly* equally to the
game result), damage given and taken are the same physical quantity
but **different levers**: `λ_give` shapes offense/conversion,
`λ_take` shapes defense/risk-tolerance. Their *ratio* sets how much
damage the bot will accept to deal damage:

- `λ_take = λ_give` — neutral trader.
- `λ_take < λ_give` — accepts unfavorable-looking trades to stay aggressive.
- `λ_take > λ_give` — defensive, protects its own percent.

Start at `λ_take = λ_give` and tune from observed behavior. Both small
in absolute terms: a full stock of damage (~120%) should sum to well
under `stock-delta`'s `±1`, so damage never outvotes the objective —
`λ ≈ 0.002–0.004` per percent point.

## Feasibility — trivial

Pure libmelee — `PlayerState.percent` frame-to-frame deltas. No
slippistats, no state machine, no porting. As easy as `stock-delta`.

## Why keep damage-taken at all

A positive-only damage reward (give only) recreates the
trade-indifference failure from `stock-delta`'s first draft: the bot's
own percent costs it nothing, so it trades percent recklessly — eats a
clean 30% to deal three 10% pokes. The `−λ_take` term prices that.

Damage-taken also **complements** `neutral-loss` and `tech`, which
penalize the lost-exchange *event* (binary). `damage-taken` adds the
*magnitude* — taking 10% vs 60% out of the same lost exchange should
not score equally.

## Causal status

Damage is the textbook **means-not-end proxy** — never run as an
objective (a damage-only policy goes damage-positive / stock-negative;
see hacking). Its role is to **densify** the sparse `stock-delta`
signal so the policy has gradient on every exchange. Close to
potential-based shaping with Φ = percent-differential — policy-
invariant in the limit, useful for learning speed.

## Reward-hacking risk

Two failure modes, both addressed by small λ + the `stock-delta`
anchor:

- **damage-positive, stock-negative** — maximizing dealt damage →
  spreads damage thin, fishes pokes, won't commit to kill confirms.
  Mitigated by small `λ_give` and the `stock-delta` anchor (raw damage
  cannot outvote stocks).
- **passivity** — too large a `λ_take` → the bot disengages to dodge
  the penalty. Mitigated by small `λ_take` and `stock-delta` (you
  cannot win by disengaging — stocks will not move). The bot also
  cannot zero out `damage-taken` by hiding; it is in a fight.

## Eval

Never run alone. The meaningful eval is an ablation: `stock-delta` vs
`stock-delta + damage-delta` — does the dense shaping speed learning /
lift win-rate over the sparse objective alone?

## Relation to existing tasks

Damage-dealt overlaps the **existing `combo_extend_online.py`** — that
task rewards *damage dealt during a combo episode* (`n_moves` is only
its gate, not its reward). `damage-delta` is all damage, ungated and
symmetric; it broadly subsumes `combo_extend`'s damage reward —
running both double-counts combo damage.

It does **not** meaningfully overlap `combo-length` (VR #5), which
rewards *move count* — nair-shine-nair is 3 moves regardless of the
percent it dealt. Move count and damage correlate loosely but are
different quantities: a 6-move combo can be lower-damage than a
2-move one.

Damage-taken overlaps `neutral-loss` / `tech` only at the event level
(see "Why keep damage-taken"). Across the suite the per-VR
**coefficients now matter as much as the designs** — a coherent
magnitude-balancing pass is needed before training.
