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
