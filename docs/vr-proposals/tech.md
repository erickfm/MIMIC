# VR Proposal: `tech`

## Definition

A small penalty when the bot is **punished out of a tech situation** —
it enters a knockdown/tech situation and exits it directly into a
damaged state. It pressures the policy off predictable, readable tech
options (including readable *no-tech* options). Across re-freeze
iterations of the baseline opponent, this drives toward unpredictable,
robust tech play.

The goal — unpredictable teching — is a mixed-strategy goal, and a
mixed strategy is only optimal against an opponent that adapts. The
adaptive opponent here is the **re-freeze loop**: train vs frozen
baseline → bot best-responds to its tech-chase → re-freeze the
baseline to the new policy → repeat. That is iterated best response /
fictitious play, the procedure that converges to a mixed equilibrium.
Unpredictability emerges *across* iterations; within any single run
the bot best-responds to the current baseline, which is expected.
Because the baseline clones *good* play, each best-response is to a
strong tech-chaser, so the equilibrium it walks toward is robust tech
play, not a cheap exploit.

## Reward spec

- **Episode:** one tech situation. `should_start` = the bot enters
  `is_teching` (action in TECH range 199-204 **or** DOWN range
  183-198) on an ascending edge. `should_end` = the bot exits
  `is_teching`.
- **Penalty:** at exit, if the exit action state `is_damaged` (75-91)
  → the tech was punished → `−0.15` (tunable; "slight"). A clean exit
  → `0`.
- **Penalty-only.** A clean tech is `0`, never positive — the best the
  bot can do is never get punished. There is nothing to farm.
- The tech *option* itself (tech-in-place / roll-L / roll-R / no-tech)
  is neither rewarded nor penalized — only the outcome. This is
  deliberate: it lets the policy find its own mix, and **"no-tech on
  purpose" is a first-class option** — a missed tech that is not
  punished costs nothing.
- **Delivery:** the penalty on the frame the tech situation resolves.

## Feasibility — medium, faithfully portable

slippistats *has* this — `tech_compute` is implemented (not commented
out) and `TechData` carries a `was_punished` field. The port:

- **Tech situation** = `common.py:is_teching` — TECH (199-204) ∪ DOWN
  (183-198) ∪ wall/ceiling reflect. The DOWN range means missed-tech /
  no-tech states are *inside* the same situation tracking — no special
  case.
- **`was_punished`** = slippistats' exact rule: the player exits the
  tech situation directly into `is_damaged`. Tight and correct — it
  catches "hit out of the tech," not an unrelated hit later in
  neutral. No arbitrary punish window.

Reuses `combo_extend`'s DOWN / TECH / DAMAGE range constants — no new
conceptual machinery. `get_tech_type()` (the ~20-case option
classifier) is *not* needed for the reward; pull it in only for
the option-distribution diagnostic below.

## Open decisions

1. **Penalty magnitude.** Proposed `−0.15`. Tunable.
2. **Tech-option diagnostic.** Recommend logging `get_tech_type()` per
   episode as metadata — the bot's tech-option distribution is the
   signal that tells you *when to re-freeze*: if it collapses toward
   one option, the bot has found a pure exploit of the current
   baseline and it is time to re-freeze. Without this you are
   re-freezing blind.

## Causal status

Defensive, and conditional — it can only fire once the bot is already
in a tech situation (i.e. already got hit and knocked down; that
upstream event is `stock-delta` / `neutral-loss` territory). It is not
a proxy for the objective and not a corollary — it shapes *how* the
bot resolves tech situations. Whether it moves win-rate depends on the
re-freeze loop actually running; a single run against a static
baseline yields a best-response, not the mixed equilibrium.

## Reward-hacking risk

Low within the reward itself (penalty-only — nothing to farm; the
floor is 0). There is no perverse "avoid teching" incentive either:
the bot does not choose to enter tech situations (it gets knocked
there), and the penalty is conditional on already being in one.

The one thing that *looks* like hacking — the bot collapsing onto a
single tech option that beats the current baseline — is not a failure
mode here, it is the expected within-run best-response and the
designed trigger for a re-freeze. Watch the tech-option distribution
(open decision 2) and re-freeze when it collapses.

## Eval

Head-to-head win-rate vs the pre-RLVR baseline will register the
within-run best-response, but **cannot directly measure
unpredictability** — it only ever plays one opponent. The real
progress signal for this VR is the re-freeze iteration: does the
tech-option mix stabilize into a genuine spread across iterations,
rather than oscillating between pure options?

## Relation to existing tasks

The first purely *defensive* VR — the combo / stock / kill VRs are all
offensive. Independent of them. Reuses `combo_extend`'s action-range
constants (the shared-combo-module factoring is now wanted by four
tasks: `combo_extend`, `neutral-win`, `combo-length`, `tech`). The
*offensive* side — the bot punishing the opponent's tech — is already
covered by `neutral-win` / `combo_extend` (a tech-chase punish is an
opening) and is intentionally out of scope here.
