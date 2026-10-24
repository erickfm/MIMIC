# 2026-10-24 — 8-hour budget: an S-stream base on the full Fox pool, then RL

Owner gave the machine and an 8 ± 1 h budget (00:00–08:00 PDT). Plan, script
`checkpoints/smallfox/train_stream_then_rl.sh`: download the HF Fox shards, train an S-stream base,
eval, then S2's RL recipe on it with the usual screen and confirmations.

## Data

`erickfm/mimic-shards-v2` fox/ is 1,523 shards, 1.52 TB (97,490 train games). The download
(`checkpoints/smallfox/dl_fox.py`: random shard order, so any prefix is a random sample; manifest
of complete shards at a deadline) ran 16 workers at ~170 MB/s, cut off at 00:20 to keep training
inside the budget: **647 train shards, ~46,277 games** (5× the local `mastfox_c9_local`), plus 24 val
shards. Data dir `data/fox_full_hf/fox` (`tensor_manifest.json` = the subset; the full one is
`tensor_manifest_full.json`). The machine rebooted once mid-download; jobs were restarted by hand.

## Base: S-stream, 56k steps

S (5.4M), cond-heads, Muon, `--attn-window 180 --seq-len 1440`, 48 × 2 = 96 chunks/step (batch 96
in one step OOMs at 1,440 frames), 56k steps (5.4M windows, ~1.2 passes over the windows), 2.8
step/s, 5.6 h (00:21–05:56). No crash, no OOM.

`tools/eval_last_frame.py` on the 24 HF val shards, paired by game:

| Model | Context | Last-frame loss |
|---|---|---|
| 5-minute proxy (`fp-F5-base-s1-postclean`, S-full, 1,900 steps on 5% of c9) | 180 | 0.7749 ± 0.0105 |
| **`fox-20261024-S-stream56k`** | 1,075 | **0.6805 ± 0.0094** (−0.0944 ± 0.0043) |

Not comparable with fox-master's 0.6812 (different val games). Some HF val games may be in the
proxy's training data, which would flatter the proxy.

## RL on it (S2's recipe: stock 1 + damage 1, 30 updates)

| Run | Base | Screen (192) | p |
|---|---|---|---|
| S2 (yesterday) | 5-minute proxy | 122–70 (63.5%); 61.2% over 1,152 confirmed | < 0.001 |
| **SS2** | S-stream 56k | **108–84 (56.2%)** | 0.097 |

Not significant, so no confirmation ran. Final KL 0.0027 (S2: 0.010). Damage per minute rose (+0.04 →
+0.56) but stock delta per minute fell (−0.21 → −0.74). Single seed. A stronger base may need more
updates or a looser leash to move; or the 30-update recipe tuned on a weak base doesn't carry over.

Stream RL was slow: **142 s per PPO update** (re-forwarding 1,075 frames per decision, ×4 epochs)
vs ~11 s for the full-context S, and rollouts at 1.5k env frames/s vs ~4k (per-slot history copies in
`BatchedPolicy.context`). 30 updates took 82 min.

## Not done / next

- Fixed-set comparison of the new base against fox-master (same val games), and an h2h of the
  S-stream base vs fox-master.
- Stream RL speed: batch the history gather in `context()` into one copy; bf16 autocast for the PPO
  re-forward; fewer epochs.
- RL on the new base with more updates (60) or a second seed.
