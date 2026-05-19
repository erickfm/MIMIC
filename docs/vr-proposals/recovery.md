# VR Proposal: `recovery`

## Definition

A penalty when the bot **fails to recover** — it is knocked off-stage
and dies before making it back. Penalty-shaped: a failed recovery
costs, a successful one is `0`. There is no positive recovery reward,
so there is nothing to farm — the bot has no incentive to go
off-stage, only an incentive, *once knocked off*, to recover well.

The off-stage *defensive* counterpart to the (dropped) `edgeguard`.
Like `tech`, it shapes how the bot resolves a sub-game it is already
in — and because recovery is itself a mixup (the recovering player
picks option + timing, the edgeguarder reads), it benefits from the
same re-freeze loop: a predictable recovery gets read and killed,
which across re-freeze iterations pressures recovery unpredictability.

## Reward spec

- **Episode start (`should_start`):** the bot transitions into being
  *knocked* off-stage — the ascending edge of `off_stage` **while in
  hitstun** (`hitstun_frames_left > 0`). This is the anti-farming
  gate: a *voluntary* drift off-stage is not in hitstun and opens no
  episode. (Matches slippistats' commented recovery-sketch start
  condition, `player_just_offstage and player_in_hitstun`.)
- **Episode end (`should_end`):**
  - *success* — the bot is back: lands on stage/platform, or grabs the
    ledge (LEDGE range 252-263). Episode closes, reward `0`.
  - *failure* — the bot loses a stock while the episode is open.
    Reward `−P`.
- **Penalty `−P`:** flat, proposed `−0.25`. This is *additional* to
  `stock-delta`'s `−1` for the death — a deliberate, tunable shaping
  choice: an off-stage death costs `−1.25`, an on-stage death `−1`,
  encoding "a failed recovery is a more avoidable way to lose a
  stock." Penalty-only — a clean recovery is `0`, never positive.
- **Delivery:** the penalty on the stock-loss frame.

## Feasibility — medium; no slippistats to port

slippistats' `recovery_compute` is commented out — there is no stat to
port, only the design sketch (which the episode boundaries above
follow). But every primitive exists and is already in use:

- `off_stage` flag — `combo_extend` already reads it.
- `hitstun_frames_left` — `combo_extend` uses it.
- LEDGE range 252-263 — `combo_extend` has the constant.
- stock-decrement detection — `stock-delta`.

So this is assembling existing primitives into a small off-stage state
machine — `tech`-grade, no new porting.

## Open decisions

1. **Penalty magnitude.** Proposed `−0.25` (additional to
   `stock-delta`'s `−1`). Tunable.
2. **Success definition.** Land on stage/platform **or** grab ledge —
   faithful to the slippistats sketch. Consequence: a ledge-trump /
   2-frame death *after* grabbing the ledge is a separate situation (a
   future ledge VR), not a recovery failure — the recovery episode has
   already closed at "grabbed ledge." Alternative: require fully back
   on-stage and actionable.

## Causal status

Defensive and conditional — it only fires once the bot is already
knocked off-stage (that upstream event is `neutral-loss` /
`damage-taken` territory). Neither a proxy for the objective nor a
corollary: penalty-shaped and gated, it cannot substitute for
anything. It shapes the *quality* of recovery decisions; the
unpredictability benefit is delivered by the re-freeze loop, as with
`tech`.

## Reward-hacking risk

The farming concern — "go off-stage to collect recovery reward" — is
designed out twice over: (1) there is no positive reward to collect;
(2) the episode only opens on being *knocked* off (off-stage +
hitstun), not on a voluntary trip. Getting knocked off also costs
`damage-taken` + `neutral-loss`, so being hit off-stage is
net-negative regardless. No perverse over-caution either: the bot does
not choose to be knocked off-stage, and the penalty is conditional on
already being there — its only penalty-minimizing move is to recover
successfully.

## Eval

Head-to-head vs the pre-RLVR baseline. Behavioral checks: recovery-
success rate up, off-stage deaths down. The unpredictability angle,
like `tech`, only materializes across re-freeze iterations.

## Relation to existing tasks

Sibling of `tech` — both penalty-shaped defensive mixup sub-games,
both re-freeze-loop-friendly. It is the off-stage *defense* axis; the
off-stage *offense* axis (`edgeguard`) remains intentionally dropped.
Reuses `combo_extend`'s `off_stage` / `hitstun` / LEDGE primitives and
`stock-delta`'s stock-decrement detection — another consumer of the
shared combo/state module those tasks want.
