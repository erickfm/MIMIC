# Automated Discovery of Verifiable Rewards for Melee RLVR

## The Problem

The primary signal for learning to play Melee — winning the game — is extremely sparse. A single game produces thousands of frames of gameplay but only one bit of outcome signal. Reinforcement learning with verifiable rewards (RLVR) addresses this by identifying denser intermediate objectives that are always aligned with winning. However, identifying these objectives currently requires deep domain expertise. An expert knows that reducing landing lag (via L-canceling, edge canceling, etc.) is always beneficial — there is no game state where more landing lag would have been preferable. But this kind of judgment is difficult to automate. Language models, when prompted to generate reward targets, tend to produce plausible-sounding but incorrect suggestions — incentivizing wavedash frequency, for instance, which is context-dependent and can produce degenerate policies.

We want a method that systematically discovers verifiable rewards from replay data without requiring domain expertise to assess each candidate.

## What Makes a Valid Verifiable Reward

Through analysis of known good VRs (L-cancel rate, stock-taking, damage conversion) and known bad ones (wavedash frequency, aerial rate, distance from opponent), a few properties emerge that seem to characterize valid targets:

**Strict dominance.** There exists no game state in the dataset where the opposite direction is preferable. Less landing lag is always better. Taking a stock is always better. This is the core invariant — if a metric's relationship to winning ever flips sign depending on context, it cannot safely be used as a reward signal without risking degenerate optimization.

**Density.** The metric triggers frequently enough to provide useful gradient signal during training. Winning happens once per game. Taking stocks happens a few times. L-canceling happens dozens of times. VRs at different density levels serve different roles in training — sparse ones provide direction, dense ones provide traction.

**Unambiguous direction.** The metric must be structured such that "improve this number" is always correct. If the optimal direction depends on context, either the metric needs to be narrowed until the context is explicit enough to resolve ambiguity, or it's not a valid VR.

These properties together describe a kind of **context-free monotonic sub-objective of winning** — something you always want more (or less) of, that you can measure, and that occurs often enough to learn from.

## The Decomposition Intuition

Valid VRs seem to live at multiple levels of a natural hierarchy rooted in the win condition:

- Winning the game (maximally sparse, maximally aligned)
- Stock differential (less sparse, still clearly aligned)
- Damage dealt and taken (denser, still aligned)
- Properties of individual interactions (combo conversion, neutral wins)
- Properties of individual actions (landing lag, DI quality)

Each level decomposes the one above into denser components. The levels correspond roughly to different time scales of the game — stocks change over tens of seconds, percent changes over seconds, mechanical execution varies frame-to-frame.

## The Degeneration Problem

A naive approach — train a win probability model V(s) on the full game state and interrogate which features matter — is likely to degenerate. Stocks and percent explain the vast majority of win probability variance. All other features (positioning, mechanical execution, micro-decisions) contribute to winning only *through* their effect on stocks and percent over long causal chains. A single model asked "what predicts winning?" will answer "stocks and percent" and stop there. The dense mechanical signals that RLVR is specifically meant to provide will be invisible in the noise floor.

This is the central technical challenge: the features we most want to discover as VRs are the ones least visible to a model trained directly on the outcome.

## Proposed Direction: Recursive Temporal Decomposition

Rather than asking one model to connect frame-level mechanics to game-level outcomes, decompose the problem into layers where each model makes a single hop in the causal chain.

The rough shape:

1. Train a model predicting win probability from game state. Confirm it's dominated by stocks and percent. This isn't a failure — it's the first level. Stocks and percent become the first VRs (sparse, high-impact).

2. Train a model predicting a denser sub-objective — something like damage dealt over short time windows. At this resolution, stocks and percent are approximately constant, so the model is forced to find other explanatory features. These might include things like: winning neutral exchanges, converting hits into follow-up damage, successfully punishing defensive options.

3. Repeat: for each important feature discovered at one level, train a model predicting it from finer-grained state features. What predicts combo conversion? What predicts winning neutral? Each hop pushes into denser, more mechanical territory.

At each level, candidate VRs are features whose relationship to the predicted sub-objective is sign-consistent across the dataset — the strict dominance check, applied locally.

An alternative (or complementary) framing that avoids needing to hand-define sub-objectives at each level: use time scale as the decomposition axis. Predict ΔV over progressively shorter windows — game-length, 10 seconds, 2 seconds, per-frame. At each time resolution, different features dominate simply because the macro features (stocks, percent) barely vary within short windows. The mechanical signals emerge naturally at fine time scales because they're the only things that change frame-to-frame.

## Open Questions

Much of the above is conjecture shaped by domain intuition. The following are things we don't know and expect to learn through implementation.

**Does V(s) actually degenerate as expected?** It's possible that a sufficiently expressive model trained on enough data picks up subtle signals beyond stocks and percent. It's also possible the degeneration is even worse than anticipated — that percent alone explains nearly everything and even stock count is redundant given percent context. The actual explained-variance breakdown will tell us a lot about how to proceed.

**What's the right granularity for sub-objective windows?** The time-scale decomposition sounds clean but the specific window sizes matter. Too large and you get the same degeneration. Too small and you might get noise — a single frame's ΔV is probably not meaningful. There may be natural breakpoints tied to game structure (per-stock, per-interaction, per-action) that work better than fixed time windows.

**Do sign-consistent features actually emerge at each level?** It's plausible that most features have context-dependent relationships even at narrow time scales. If very few features pass the strict dominance filter at any level, the method produces few VRs and we need to rethink. Conversely, if many features pass, we need a way to prioritize.

**How sensitive are the discovered VRs to the model architecture and training details of V?** If two differently-trained value functions surface very different candidate VRs, the method may not be robust enough to trust. Consistency across model variations would increase confidence.

**Can this actually find things an expert wouldn't think of?** The motivating hope is that the decomposition surfaces non-obvious VRs — mechanical patterns that consistently contribute to winning but aren't part of conventional Melee wisdom. If it only rediscovers L-canceling and DI, it's useful but not transformative. If it finds novel targets, that's a much stronger result.

**How does this interact with character-specificity?** Some VRs may be universal (landing lag reduction) while others are character-specific (certain combo routes, recovery patterns). The decomposition may need to be run per-character, or the sign-consistency check may naturally filter out character-specific features when run on mixed data.

**What happens when the agent optimizes a discovered VR?** A feature may be sign-consistent in human replay data but become exploitable under optimization. Humans never L-cancel 100% of the time, so the data always shows "more L-canceling = better." But there could be discovered features where the relationship holds in the human data range but breaks down when pushed to extremes. Validating VRs under optimization pressure, not just in the offline dataset, may be necessary.

## From V(s) to Actual Verifiable Rewards

V(s) is a discovery tool — it tells you what matters. It is not itself the reward. This distinction is important because using V(s) directly as a reward signal is just a learned reward model (RLHF territory), which is neither verifiable nor stable under optimization. The whole point of RLVR is that the reward is a deterministic, auditable function of game state.

So the pipeline has two phases with different roles:

**Phase 1: Discovery.** The recursive decomposition of V(s) surfaces features that are sign-consistent predictors of sub-objectives at various time scales. The output of this phase is a ranked list of feature-direction pairs — "landing lag is consistently bad for combo conversion," "opponent percent increasing is consistently good for stock-taking," etc. These are statistical findings about the dataset, not yet rewards.

**Phase 2: Operationalization.** Each finding gets translated into a concrete, deterministic function over game state. This is where the finding becomes a verifiable reward. The function takes a trajectory (or window of frames) and returns a scalar, with no learned model in the loop.

Some findings translate trivially. "Landing lag hurts combo conversion" becomes: count frames of landing lag after aerials, reward their reduction. The game state directly encodes action states and frame counters, so this is a straightforward check. "Taking stocks is good" becomes: reward stock transitions. These are already verifiable by definition — they're just reading game state.

The interesting cases are findings that don't map to a single clean check. The decomposition might surface something like "having frame advantage after an exchange predicts winning the next interaction." Frame advantage isn't a single field in the game state — it's a derived quantity involving both players' action states, remaining animation frames, and positioning. The VR for this needs to be a small piece of logic that computes frame advantage from raw state and rewards being positive. It's still deterministic and verifiable, but it requires defining the computation.

There may also be findings that resist operationalization entirely — features that V(s) identifies as important but that can't be cleanly expressed as a state function. These might point to reward targets that require more thought, or they might indicate that the discovered feature is a proxy for something else that IS directly measurable.

The degree to which this translation step can be automated is an open question. The simplest version is: the discovery phase produces a report, a human reads it and writes reward functions. A more ambitious version uses the discovered feature-importance structure to automatically generate reward code — plausible for simple features (direct state readings, thresholds on counters), harder for derived quantities. How far automation can go here will probably become clearer once we see what kinds of features the decomposition actually surfaces.

One important constraint on the operationalized VR: it should be cheap to compute. These rewards run at every frame during training. If a VR requires running a neural network or complex simulation to evaluate, it's no longer practically verifiable in the relevant sense — the training loop can't afford it and you can't easily audit what it's doing. The reward functions should be simple enough that a human can read the code and confirm "yes, this is measuring what we think it's measuring."

## Starting Point

The most concrete first step is training V(s) on the existing ranked replay dataset and examining what it learns — both the dominant features and, critically, the residuals. What's left unexplained after accounting for stocks and percent? That residual structure will shape everything that follows.
