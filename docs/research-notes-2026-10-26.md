# 2026-10-26 — Sim RL recipe: more updates win; RL small Fox beats fox-master in the sim; gains hold in Dolphin

Night of 10-25 → 10-26, owner-approved, one seed per variant, nothing promoted. Base for every RL run:
`checkpoints/smallfox/base_full56k` (S, full attention, cond-heads, 56k steps), reward stock 1 + damage 1,
SF2's recipe otherwise (`docs/research-notes-2026-10-25.md`). Scripts `checkpoints/smallfox/queue_{m,n,o,q,r,s}.sh`;
result lines in `checkpoints/smallfox/results.txt`. Sim h2h: native sim, bf16, `replays_doc_rt` templates, seed 41.

## Decisions per update vs number of updates (queue_m)

| Run | Decisions per update × updates | vs full56k (192 games) | vs SF2 (96) | KL per decision |
|---|---|---|---|---|
| SF2 | 16,384 × 30 | 121–71 (63.0%) | — | 0.003–0.007 |
| SF5 | 32,768 × 30 | 130–62 (67.7%) | 57–39 | 0.0018 |
| SF6 | 65,536 × 30 | 131–61 (68.2%) | 53–43 | 0.0030 |
| SF7 @ u60 | 16,384 × 60 | 67–29 (69.8%, 96 games) | 56–40 | — |
| SF7 @ u100 | 16,384 × 100 | **140–52 (72.9%)** | **64–32** | 0.0134 |

The batch gain flattens between 32k and 64k; more updates keep helping (u100 beats its own u60 checkpoint 63–33)
and training reward was still rising at update 100. SF7 used less compute than SF6. The lr is constant, so the u60
checkpoint stands in for a 60-update run.

## Round robin, all SF models (queue_o; 96 games per pair, the vs-SF2 screens reused)

| Model | Total vs the other 7 | Win rate |
|---|---|---|
| SF7u100 | 432–240 | 64.3% |
| SF7u60 | 362–310 | 53.9% |
| SF6 | 356–316 | 53.0% |
| SF3 (lr 2e-5) | 339–333 | 50.4% |
| SF5 | 331–341 | 49.3% |
| SF4 (no critic) | 319–353 | 47.5% |
| SF2 | 312–360 | 46.4% |
| SF1 (4,096 per update) | 237–435 | 35.3% |

SF7u100 beats every other model (52–44 vs SF6 is its closest; 63–33 or more vs four of them). SF2–SF6 are within
noise of each other. Pairwise matrix: `checkpoints/smallfox/rr/`.

## vs fox-master (queue_n, sim, 192 games)

fox-master = production M (20M params, full 102k-game pool, 280k steps, pure BC, AdamW, independent heads).

- full56k (S, no RL) vs fox-master: 59–133 (30.7%, p < 0.001, stock margin −0.72).
- **SF7u100 vs fox-master: 109–83 (56.8% [49.7, 63.6], p = 0.071, stock margin +0.15).**

RL on a model a quarter the size, trained on less data, takes it from 31% to 57% against the production bot.
Sim only.

## Do sim RL gains hold in real Dolphin? (queue_q, queue_r)

SF5 vs full56k, 200 matches in Dolphin: `emulator_ss` FFW, blocking input, alternating ports, 6 stages,
production fp32 inference (`tools/play.py`), in 10-match chunks (`checkpoints/smallfox/dolphin/run_chunks.py`).
Frame-delta histogram all 1s (no dropped frames), 0 timeouts.

- **Dolphin: SF5 124–76 (62.0% [55.1, 68.4], p < 0.001)**; by port order 66–34 / 58–42. Sim: 130–62 (67.7% [60.8, 73.9]).
  The intervals overlap; the gain transfers.

Canaries (`checkpoints/smallfox/dolphin/analyze.py`; 192 of 200 replays parsed; human = master-Fox dittos from
`data/raw_slp/fox_master_master`, 1,336 player-games):

| | center-stick | L-cancel | aerial landings / 1k frames | stick JS vs human | state JS vs human |
|---|---|---|---|---|---|
| SF5 | 32.9% | 89.4% | 3.91 | 0.0065 | 0.0080 |
| full56k | 33.4% | 88.2% | 3.24 | 0.0059 | 0.0062 |
| human | 34.0% | 92.0% | 3.96 | — | — |

Largest action-state shifts SF5 vs full56k [human]: up-smash 2.85% vs 1.70% [1.54%], DAMAGE_FLY_TOP 4.83% vs 5.87%
[4.97%], nair 2.75 vs 2.15 [2.47], bair 3.11 vs 2.71 [3.31], uair 1.47 vs 1.13 [1.39], dashing 7.77 vs 8.31 [8.57],
DEAD_FLY_STAR 0.72 vs 1.07 [1.00]. RL plays more aerials (landing rate now at the human level), more up-smash (the
one shift past the human rate), and gets launched less. No camping or degenerate states among the shifts.

Analysis gotcha: `play.py` costumes stay per port across `--alternate-ports` flips (port 1 = costume 1 in every
replay), so costume can't identify the policy; `analyze.py` maps replays to models from play.py's log
("match i/N starting (A on port P)").
