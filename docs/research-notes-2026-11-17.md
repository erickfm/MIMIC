# 2026-11-17 — box 2 results (coach rl-v4, Fox RL curve, cond-heads order)

Box 2 (vast.ai 60995156, 2× RTX 5090) ran from the evening of 2026-11-16 and was destroyed at
2:38 AM on 2026-11-17 when the vast credit ran low. Its queues were cut short; this note records
what finished. Logs copied from the box are in `runs/box2-logs/` (not tracked).

## Coach models: S + rl-v4 (200 updates, lr 2e-5)

Each run vs its own BC parent, 192 games, h2h seed 41, bf16. The rl-v2 column is the same parent's
100-update run from 2026-11-05/08, for reference (different h2h, same rig).

| character | rl-v4 run | vs BC | rl-v2 vs BC |
|---|---|---|---|
| fox | `fox-S-bc1-s1.rl4-s1` | 162–30 | |
| falco | `falco-S-bc2-hf28k-s1.rl4-s1` | 169–23 | 157–35 |
| marth | `marth-S-bc2-hf21k-s1.rl4-s1` | 181–11 | 169–23 |
| sheik | `sheik-S-bc2-hf17k-s1.rl4-s1` | 178–14 | 155–37 |
| puff | `puff-S-bc2-hf14k-s1.rl4-s1` | stopped at u100 | 164–28 |
| cptfalcon | `cptfalcon-S-bc2-hf12k-s1.rl4-s1` | stopped at u65 | 160–32 |

- An rl-v4 run took ~185 min per character on one 5090 (falco 185.9, marth 185.4, sheik 182.8).
- Puff and Captain Falcon: the `rl_u*.pt` checkpoints were copied, the optimizer state was not, and
  there is no `policy/` export or h2h. Their `run.json` status is `stopped`. Finishing them is a
  rerun, or a `--resume` with a fresh AdamW.
- The other 17 characters were never started. The coach keeps using their rl-v2 models.
- The per-character u50/u100/u150 curve h2hs did not run.

## Fox: win rate vs RL updates (S, rl-v4)

`fox-S-bc1-s1.rl4-s1` checkpoints vs the BC parent `fox-S-bc1-s1`, 192 games each, seed 41, bf16.

| updates | record | win rate |
|---|---|---|
| 25 | 122–70 | 63.5% |
| 50 | 144–48 | 75.0% |
| 75 | 158–34 | 82.3% |
| 100 | 152–40 | 79.2% |
| 125 | 161–31 | 83.9% |
| 150 | 170–22 | 88.5% |
| 175 | 168–24 | 87.5% |
| 200 | 162–30 | 84.4% |

One training seed, one h2h seed; a single 192-game match is about ±5 points (95%) at these rates.
The u25–u175 run folders were left on the box; the numbers are from `runs/box2-logs/curve.gpu0.log`.

## Cond-heads chain order: all 24 orders on the fast proxy

`tools/fast_proxy.sh co-ORDER S 1900 1 --arch cond_order=ORDER` on `data/fox_hf_proxy` (a 1/22
train + 1/13 val slice of the HF Fox shards, `runs/box2-logs/make_proxy_data.py`), seed 1, tail-SWA,
fixed-subset val. Letters: s = shoulder, c = c-stick, m = main stick, b = buttons. `scmb` is the
current order.

| first head | orders (val/total) |
|---|---|
| s | scmb 0.7766, scbm 0.7767, smcb 0.7765, smbc 0.7765, sbcm 0.7766, sbmc 0.7772 |
| c | csmb 0.7770, csbm 0.7770, cmsb 0.7765, cmbs 0.7761, cbsm 0.7763, cbms 0.7753 |
| m | mscb 0.7762, msbc 0.7768, mcsb 0.7766, mcbs 0.7763, mbsc 0.7756, mbcs 0.7758 |
| b | bscm 0.7766, bsmc 0.7763, bcsm 0.7763, bcms 0.7757, bmsc 0.7763, bmcs 0.7759 |

- Range 0.7753–0.7772 (0.0019, 0.25%). The proxy's seed gap, measured on its usual dataset, is
  ~0.0007; it was not re-measured on `fox_hf_proxy`.
- Four orders are more than 0.0007 below `scmb`: cbms (−0.0013), mbsc (−0.0010), bcms (−0.0009),
  mbcs (−0.0008). For scale, cond-heads vs no cond-heads on this proxy is about −0.03.

### h2h of the four vs `scmb` (local 4090, 2026-11-17)

Proxy checkpoints (`checkpoints/fastproxy/co-*.avg.pt`), Fox ditto in the sim, 192 games, seed 41,
bf16, `checkpoints/fastproxy/h2h/cond_order_h2h.sh`. Sim inference samples the heads in each
model's own order (`_forward_cond`).

| order | val vs scmb | record vs scmb | win rate [95%] | p | stock margin |
|---|---|---|---|---|---|
| cbms | −0.0013 | 108–84 | 56.2% [49.2, 63.1] | 0.097 | +0.22 |
| mbsc | −0.0010 | 89–103 | 46.4% [39.4, 53.4] | 0.35 | −0.07 |
| bcms | −0.0009 | 91–101 | 47.4% [40.5, 54.4] | 0.52 | −0.10 |
| mbcs | −0.0008 | 95–97 | 49.5% [42.5, 56.5] | 0.94 | −0.03 |

- Pooled over the four: 383–385 (49.9%).
- No order differs from `scmb` at the 95% level. cbms, the best on val, has the only record above
  50%; its two passes split 61–35 (cbms registered first) and 47–49 (scmb first), the other three
  orders split evenly between passes.
- These are 1,900-step proxy models, one training seed and one h2h seed each. Not tested at S 56k.
- `scmb` stays the default. Policy folders and full reports: `checkpoints/fastproxy/h2h/co-*`.
