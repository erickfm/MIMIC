# Research notes — 2026-05-12

## tl;dr

**Lever A from 2026-04-21 (`enable_ffw=True` for online RL) is not a
1-line change on the bundled emulator.** It requires vladfi1's
`slippi-Ishiiruka:exi-ai-rebase` fork, which our `emulator.tar.gz`
isn't, and libmelee's own docstring warns the required `use_exi_inputs`
gecko-code path is "likely incompatible with netplay" — so even with
the fork, the Discord bot's `play_netplay.py` can't use it. Lever C
(scenario-seeded mini-episodes) is the better target for the rare-
event skills (shield-escape) that motivated A anyway.

## The dependency chain

libmelee's `Console.__init__` enforces this gate:

```python
self.use_exi_inputs = use_exi_inputs
if enable_ffw and not use_exi_inputs:
    raise ValueError("Must use exi inputs to enable ffw mode.")
```

with the docstring:

> `use_exi_inputs (bool)`: Enable gecko code for exi dolphin inputs.
> This is necessary for fast-forward mode which ignores dolphin's
> normal polling. Must be used with a compatible Ishiiruka branch
> such as `vladfi1/slippi-Ishiiruka:exi-ai-rebase`. Note that this
> will likely be incompatible with netplay.
>
> `enable_ffw (bool)`: Enable fast-forward mode. Useful for bot
> training. Must have use_exi_inputs=True.

So: FFW → exi inputs → vladfi1 fork → no netplay.

## What our bundled emulator actually is

`strings emulator/squashfs-root/usr/bin/dolphin-emu` enumerates EXI
device classes:

```
CEXISlippi, CEXIETHERNET, CEXIMemoryCard, CEXIAgp, CEXIIPL,
CEXIMic, CEXIAD16, CEXIDummy, CEXIGecko
```

**No `CEXIAi`** — vladfi1's fork adds that device to expose AI
controller inputs via EXI. Confirms mainline project-slippi/Ishiiruka.

The FFW symbols that *do* exist —
`SlippiPlaybackStatus::shouldFFWFrame`, `setHardFFW` — are for Slippi
replay-playback fast-forward (skipping ahead through a recorded .slp),
not live-rollout training fast-forward via exi inputs. Different
feature, different code path, irrelevant to online RL.

## What this changes for the three levers

| lever | claim 2026-04-21 | revised 2026-05-12 |
|-------|------------------|--------------------|
| A — `enable_ffw=True` | 1-line, potentially 5-10× | fork swap + netplay loss |
| B — savestate across matches (F2/F1 + xdotool) | ~25% gain | unchanged |
| C — scenario-seeded mini-episodes | 60-90×; mandatory for rare events | unchanged, and now the leading candidate |

The lever that was supposed to be the cheap win is actually the most
expensive one (build/maintain a second emulator, lose netplay path).
The lever that was supposed to be expensive (C) is now the obvious
next step — it's the only one that actually unlocks shield-escape
(which produced zero episodes in 17 minutes of live play because
master Fox avoids pressured shield).

## Sunset of world-model track

Today's commit `8b83622` (`world-model: sunset snapshot`) captures
the final state of that branch: DDP polish, log-backfill helper, opp-
controller shard backfill, viz tool, rollout JSONs for baseline /
discsym-xl / oppsym at 32k, and three CLAUDE.md updates (BC-vs-WM
opp_controller asymmetry, wandb step=step pitfall, viz tool entry).
Branch is `world-model`, 2 commits ahead of `origin/world-model`,
working tree clean.

The pivot motivation: WM val-loss is Huber-saturated and rollout
drift never beat a "predict-current-frame" baseline at meaningful
horizons. The next-best track is to make online RL fast enough that
we don't *need* a learned simulator — which lever C does.

## Next session

If picking up here: build the scenario-seeding library. Concrete
plan, no code changed yet:

1. **State extraction** — scan replay corpus for "pressured shield"
   frames (shield dropped by ≥8 and currently <30/60). Capture
   savestate via Dolphin's serialize-state mechanism (need to check
   what libmelee exposes; if nothing, use Dolphin's hotkey-driven
   state save and read the file).
2. **State library** — pickle ~50 such states keyed by skill +
   character. Lives at `data/scenarios/<skill>/<char>/*.state`.
3. **Episode driver** — in `rlvr/online/dolphin_actor.py`, add a path
   that loads a state, runs the policy for N frames (30-60), measures
   outcome via the skill's verifier, then loads the next state. The
   in-match free-play loop stays as the fallback for tasks that don't
   have a scenario set.
4. **Verifier reuse** — `rlvr/tasks/shield_escape.py`'s verifier is
   already pure (`__call__(prompt, sampled_ctrl) -> float`), so it
   plugs in unchanged. The harder piece is detecting episode
   termination — when has the character successfully escaped, vs.
   still being pressured?
5. **Validate the 60-90× claim** on a few hundred episodes before
   committing to a full training run. The 2026-04-21 number is a
   napkin estimate (60-90s full match vs 1s of in-game time per
   30-60 frame mini-episode); savestate-load overhead in Dolphin is
   unmeasured and could be the bottleneck.

Watch out for: Dolphin savestate format is engine-version specific
(state from one Ishiiruka build won't load in another), so save the
binary + state library as a pair. Also: rollback netplay won't fire
in offline mode, so any rollback-sensitive frame state needs to be
captured at a known-stable frame.
