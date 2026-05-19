# VR Proposal: `neutral-win-loss`

## Definition

Reward the policy `+1` each time it wins a neutral exchange (lands the
opening hit of a punish from neutral) and `−1` each time it loses one
(gets hit out of neutral). This is the upstream lever in the causal
chain `neutral win → first hit → hitstun → combo → damage → stock →
game` — the candidate "common cause" the downstream combo/damage VRs
are corollaries of.

A "neutral win" is the *first* hit of an exchange where neither player
was already in advantage/disadvantage — distinct from a
combo-continuation hit (`combo-length` territory) and from a
counter-hit (the hitter was itself being punished).

## Reward events

- **Neutral win (`+1`):** the opponent transitions into punish state
  on an ascending edge (was not in punish the prior frame — excludes
  combo continuation), AND the bot was not in punish state for the
  preceding `K` frames (excludes counter-hits).
- **Neutral loss (`−1`):** the mirror — the bot enters punish state on
  a clean edge, the opponent clean for the preceding `K` frames.
- **Trade:** both players enter punish within `W` frames of each other
  → `0` for both (a trade is genuinely ambiguous).
- **Magnitude:** flat `±1` per exchange. The *conversion* of the
  exchange (damage, kill) is deliberately not folded in — that is
  `combo-length` / `stock-delta`. This VR rewards winning the exchange,
  full stop.
- Reuse the damage-or-grab safeguard (a punish-state edge counts only
  if it coincides with damage dealt or a grab) so a spurious
  state-machine trigger is not scored.
- **Delivery:** per-frame, `±1` on the exchange-resolving frame.
- `K ≈ 30`, `W ≈ 4` (tunable).

## Feasibility — medium; the machinery is already ported

slippistats does *not* classify combo openings (no neutral-win /
counter-attack / trade categorization in the library — that lives only
in the Slippi desktop app). But the hard part is the punish state
machine, and that lives in `rlvr/online/slippi_stream.py`
(`in_punish_state`, ascending-edge detection). `neutral-win-loss` is
that edge detector run for both players plus the K-frame clean-hitter
filter — not a from-scratch port.

## Causal status

The first genuine *proxy* in the suite — not the objective.
`neutral-win` is high up the causal chain, the candidate common cause.
But winning neutral does not guarantee a stock (you can win neutral and
drop the conversion). It is a prime subject *for* the
correlation-vs-causation test: train on it, check win-rate; if win-rate
stays flat it was a corollary.

## Reward-hacking risk

Real (unlike `stock-delta`). "Land the first hit" can be farmed with
tiny safe pokes that technically win neutral but never convert.
Symmetry mitigates partially — the bot is also penalized `−1` for
getting poked, so reckless poking exposes it — but a *risk-free* poke
is still farmable. `neutral-win-loss` must never run without
`stock-delta` anchoring it.

## Eval

Head-to-head win-rate vs the pre-RLVR baseline. The conversation-1
experiment: `stock-delta` alone vs `stock-delta + neutral-win`. Watch
for neutral-win count rising while stock margin stays flat — the
poke-farming hack.

## Relation to existing tasks

A neutral win is a *subset* of combo openings — the ones from neutral,
not counter-hits. Complements `combo-length` (how the opening is
converted) and `stock-delta` (closing it out). Together: `neutral-win`
(won the exchange) → `combo-length` (converted it) → `low-percent-kill`
(closed it early) → `stock-delta` (closed it at all).
