# 2026-10-29 — Muon LR at S and M, first RL on M

Local 4090, 10-28 8:13 PM → 10-29 9:42 AM Pacific (owner: "stay busy till 10 am"; S run paused 8:27–9:37 PM).
Logs: `checkpoints/s_mlr01_20261028/`, `checkpoints/overnight_20261029/`. One training seed per model; h2h in the
native sim, bf16, 192 games per seed.

| Run | vs | Seeds 41 / 42 / 43 | Total | Win rate [95%] |
|---|---|---|---|---|
| `fox-S-bc1-mlr01-s1` (`bc-S-v1`, Muon 0.01) | `fox-S-bc1-s1` (Muon 0.02) | 94–98, 99–93, 108–84 | 301–275 | 52.3% [48.2, 56.3] |
| `fox-M-bc3-mlr005-s1` (`bc-M-v3`, Muon 0.005) | `fox-M-bc2-mlr01-s1` (Muon 0.01) | 93–99, 87–105, 87–105 | 267–309 | 46.4% [42.3, 50.4] |
| `fox-M-bc2-mlr01-s1.rl2-s1` (`rl-v2`) | its base `fox-M-bc2-mlr01-s1` | 150–42, 158–34, 153–39 | 461–115 | 80.0% [76.6, 83.1] |
| same | `fox-M-master` | 151–41, 155–37, 161–31 | 467–109 | 81.1% [77.7, 84.1] |
| same | `fox-M-bc1-legacy` (cond-heads M) | 142–50, 139–53 | 281–103 | 73.2% [68.5, 77.4] |

- Muon LR: at S, 0.01 ties 0.02; at M, 0.01 beats 0.02 (57.6%, 10-28) and 0.005 (above), 0.04 collapses. 0.01 is
  the best LR tried at M and no worse than 0.02 at S. The S default is unchanged (owner's call).
- First RL on M: `rl-v2` (100 updates) on the bc-M-v3-recipe model, 219 min, fits on the 4090. Unlike XS (74.5% vs
  its base, 37.7% vs S), the gain holds against models it never trained against: 81.1% vs production `fox-M-master`
  and 73.2% vs cond-heads M, which its base recipe lost to (`fox-M-bc2-s1` 37.5% vs cond-heads M; `fox-M-bc2-mlr01-s1`
  itself was not played vs either). SF7 (S + RL) for reference: 56.8% vs `fox-M-master`, 46.2% vs cond-heads M.

**Default changed (owner, 10-29):** `train.py --muon-lr` defaults to 0.01 for every size (S was 0.02); the
per-size `MUON_LR_DEFAULT` from 10-28 is gone. New recipe `recipes/bc-S-v2.yaml` = `bc-S-v1` at 0.01, so `bc-S-v2`
and `bc-M-v3` differ only in model size. `tools/fast_proxy.sh` pins 0.02 to keep `results.tsv` comparable.

## RL-on-M vs Phillip's released Fox agents (Dolphin, realtime)

Owner asked whether our best model beats a comparable Phillip. Harness: `tools/vs_phillip.py` (new), run in a venv
of vladfi1/slippi-ai (commit 5ada33e, its libmelee fork 0.47.6, onnxruntime CPU) with torch added. Phillip runs
from the released ONNX files (Google Drive folder linked in slippi-ai's README) through slippi-ai's own
`eval_lib.Agent`, console delay 0, so its trained 18-frame delay is its whole delay. Ours
(`fox-M-bc2-mlr01-s1.rl2-s1`) runs fp32 with no delay, as in production. One Slippi Dolphin 3.6.4 (realtime,
window), blocking input, both pads flushed once per frame by `console.step`. Ports alternate every game; stages
rotate FD, Battlefield, Dreamland (no Randall or FoD platforms, which Phillip can't see). Fox ditto, 20 games each.
MIMIC's model inputs built under libmelee 0.47.6 match 0.45.1 exactly on every frame of a test replay (both
perspectives), so the version change doesn't alter what our model sees.

| Phillip agent | Name it played as | Result (ours) | Stocks left (ours / Phillip) | Mean game |
|---|---|---|---|---|
| `fox_d18_ditto_v4` (imitation + RL; RL names Cody, Hax, Aklo, SFAT) | Cody (from RL) | 0–20 | 0 / 63 | 99 s |
| `fox_d18_imitation_v3` (imitation only) | Master Player (default) | 20–0 | 49 / 0 | 125 s |

Phillip's RL Fox 4-stocked us in 10 of 20 games; we 3- or 2-stocked its imitation-only Fox in every game. Not
separated: the imitation agent played as the generic "Master Player" name while the RL agent played as "Cody",
and both handicap Phillip by an 18-frame reaction delay that ours doesn't have. Replays and per-game JSON:
`checkpoints/vs_phillip_20261029/` (not tracked).
