# 2026-04-14 — RoPE sweep on Falco, final answer: stick with Shaw relpos

## Setup

All runs were Falco on `data/falco_v2` (9,110 games, v2 shards), identical
recipe except for position encoding: d_model=512, 6 layers (12 for one
run), 8 heads, seq_len=256, batch 512, 32,768 steps, dropout 0.1,
`--self-inputs`, `mimic_flat` encoder, 7-class button head.

## Motivation

Fox RoPE (fox-20260413-rope-32k.pt) matched the reference Fox relpos on
val metrics, which suggested RoPE was a drop-in replacement. When I ran
Falco RoPE the next day (falco-20260413-rope-31k.pt), the same recipe
landed at **val 0.76, btn F1 87.1%, main F1 ~55%**, behind the Falco relpos
reference at **val 0.68, btn F1 88.2%, main F1 58.5%**. A consistent
~0.08 val-loss gap.

Hypothesis ladder, cheapest-first:
1. RoPE's default θ=10000 wastes frequency resolution on long distances
   we don't use (seq_len=256 << 10000). Lower θ or learnable freqs might
   close the gap.
2. Exponential distance decay (xPos) might better match Melee's recency
   prior.
3. It's a capacity issue — more layers brute-forces the gap shut.
4. It's structural — Shaw's learned per-offset bias table is doing work
   that no parameterized rotation scheme can replicate at this scale.

## Runs

| Run | Preset | Change | Mean Δ val vs plain rope | Verdict |
|---|---|---|---|---|
| falco-20260413-rope-31k (ref) | `mimic-rope` | — | 0 | baseline |
| falco-20260414-xpos64 | `mimic-xpos-64` | xPos, scale_base=64 | +0.002 | killed at val 13, noise |
| falco-20260414-ropelt | `mimic-rope-lt` | θ=1000 | +0.0002 | killed at val 30, noise |
| falco-20260414-ropelf | `mimic-rope-lf` | learnable freqs | +0.004 | killed at val 32, slightly worse |
| falco-20260414-ropedeep | `mimic-rope-deep` | 12 layers | +0.005 | killed at val 17, slightly worse |

All four variants fell within ±0.01 of plain RoPE at every checkpoint. None
approached the relpos reference.

## Conclusion

**The ~0.08 Falco RoPE → relpos gap is structural, not a frequency or
capacity problem.** Most plausible explanation: Shaw's 513 free parameters
(one per relative offset in [-256, 256]) let the model memorize
distance-keyed biases like "frame +2 after a KNEE_BEND is always JUMP."
RoPE's parameterized rotation spectrum can approximate those but
apparently can't match them at this scale on this data.

## Rule going forward

- **Canonical preset for new training runs: `mimic` (6-layer relpos).**
  Do not use `mimic-rope` for production checkpoints.
- RoPE's ~2× training throughput is real (~17 step/s vs ~8 step/s on the
  5090) but doesn't compensate for the val-loss loss.
- RoPE variants can stay in the preset table as dead code — they're
  useful if someone else wants to re-verify the sweep, and the sweep
  itself took about 40 minutes of wall clock total (most runs were
  killed early).

## Things that were NOT tried (and why)

- **Relpos + RoPE hybrid** — apply RoPE rotation to Q/K AND add Shaw's
  learned bias on attention logits. Would likely match or beat relpos.
  Requires ~1 hour of code work in `mimic/model.py`'s attention
  implementations. Deferred — relpos alone is already good enough.
- **ALiBi + RoPE** — linear distance decay added to RoPE scores. Similar
  argument: unclear upside over plain relpos.
- **Fox RoPE re-run to confirm the Fox result was real** — the Fox RoPE
  match-to-relpos might have been a lucky seed or a data-specific quirk.
  Not investigated; not going to matter now that we're committed to relpos.

## Artifacts kept

- `checkpoints/falco-20260413-rope-31k.pt` — the original Falco RoPE
  result that started the sweep. Kept for historical A/B.

## Artifacts cleaned

- All `falco-20260414-*` intermediate + best checkpoints from the four
  sweep variants. None were worth keeping.
