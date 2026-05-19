# VR Proposal: `combo-length`

## Definition

Reward longer combos, scored by **move count** — the number of distinct
moves landed in a combo. A "move" is a group of hits sharing one bot
action state: Fox's multi-hit dair drill is *one* move; jab → grab →
up-air is *three*. Rewards stringing more moves together.

## Reward spec

- **Episode = a combo.** Detected by the punish-sequence state machine:
  start on the ascending edge of the opponent entering punish state;
  close after `COMBO_END_GAP = 45` consecutive out-of-punish frames, or
  on a stock change.
- **Terminal reward**, computed at combo close from the final move
  count: `reward = (clip(n_moves, 2, CAP) − 2) / (CAP − 2)` → 2 moves =
  `0` (the minimum combo, not "long"), `CAP`+ moves = `1.0`. `CAP = 8`.
- **Damage-floor gate:** the combo must have dealt ≥ 5% total damage to
  be eligible — otherwise `0`. A pure gate, not a damage reward; it
  blocks racking up move count with weak non-committal hits.
- `n_moves < 2` → `0` (a single move is not a combo).

## Feasibility — easy; the machinery is already ported

The move-counter — the slippistats-faithful "what counts as a distinct
move" logic — lives in `rlvr/online/slippi_stream.py` (`MoveCounter`,
`ComboTracker`). `combo-length` reuses the episode detection and the
move counter wholesale; the only new code is the scoring line.

## Causal status

The most corollary-suspect VR in the suite. Move count is a late-chain
combo-quality metric — a *correlate* of "good conversion skill," not
obviously a cause of winning. Conversation 1 predicts this is exactly
the kind of VR where the VR improves but win-rate may not. A prime
causal-test candidate: reward it, check win-rate; if win-rate stays
flat, move count was a corollary.

## Reward-hacking risk

The highest in the suite. Move count is farmable — many weak,
low-commitment hits that extend a combo without converting it, style
over substance. It also pulls *against* `low-percent-kill`: the early
kill is often the *short* combo (grab → up-throw → up-air). Mitigations:
the 5%-damage floor gates out the most degenerate weak-hit farming; the
`stock-delta` anchor keeps the real objective in the reward. A residual
incentive to over-extend past the optimal kill confirm remains — run
only anchored by `stock-delta`, and watch for combo length rising while
stock margin does not.

## Eval

Head-to-head win-rate vs the pre-RLVR baseline. Behavioral check: the
trained bot's average combo move-count should rise. Conversation-1
check: does win-rate follow? If combos get longer but win-rate is flat,
this VR was a corollary.

## Relation to existing tasks

Replaces, with `damage-delta`, the retired `combo_extend` task —
`combo_extend` rewarded combo *damage* gated by move-count;
`damage-delta` now owns the damage and `combo-length` owns the move
count. Move count and damage are different quantities (a 6-move combo
can be lower-damage than a 2-move one), so the two genuinely complement
rather than duplicate. Note `combo-length` and `low-percent-kill` can
pull against each other at the kill-confirm decision.
