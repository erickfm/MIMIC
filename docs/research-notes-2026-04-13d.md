# 2026-04-13d — HAL → MIMIC rename

## Motivation

MIMIC bootstrapped from HAL for a known-good baseline, then diverged a lot
(7-class button head, v2 shard alignment, RoPE, netplay/Discord, shared
inference utils). The codebase still had `hal_*` names everywhere —
CLI flags, class names, config fields, encoder registry strings, data dir
names, metadata JSON filenames — which made it look like a HAL fork when
it isn't anymore. Renamed the active code path wholesale to use `mimic_*`
and left the historical / HAL-comparison tooling (`tools/run_hal_model.py`,
research notes) alone.

## Scope

**Renamed on disk:**
- `data/fox_hal_v2` → `data/fox_v2`
- In every active `data/*_v2/` dir: `hal_norm.json` → `mimic_norm.json`
  (falco_v2, cptfalcon_v2, luigi_v2, fox_v2, falco_wavedash)
- `tools/run_mimic_via_hal_loop.py` → `tools/play_vs_cpu.py`
- `tools/build_hal_norm.py` → `tools/build_mimic_norm.py`
- `tools/verify_hal_pipeline.py` — deleted (one-shot HAL parity check,
  obsolete)

**Renamed in code (with backwards-compat aliases):**
- Model presets: `"hal"`, `"hal-rope"`, `"hal-rope-lt"`, `"hal-learned"`,
  `"hal-rope-lf"`, `"hal-flex"`, `"hal-ropeflex"`, `"hal-rope-deep"`,
  `"hal-xpos"`, `"hal-selrope"`, `"hal-ropenope"`, `"hal-xpos-64"`
  → `"mimic"`, `"mimic-rope"`, etc. Legacy names added as aliases in
  `MODEL_PRESETS` so saved checkpoints with `model_preset="hal-rope"`
  still load.
- Classes: `HALFlatEncoder` → `MimicFlatEncoder`,
  `HALTransformerBlock` → `MimicTransformerBlock`,
  `HALPredictionHeads` → `MimicPredictionHeads`. Legacy names assigned
  as module-level aliases.
- Encoder registry: `"hal_flat"` → `"mimic_flat"` (legacy key retained).
- `ModelConfig` fields: added `mimic_mode`, `mimic_minimal_features`,
  `mimic_controller_encoding`. Legacy `hal_*` fields kept;
  `__post_init__` migrates them onto the new fields and keeps both in
  sync so old `ModelConfig(**ckpt["config"])` calls still work.
- Functions: `load_hal_norm` → `load_mimic_norm` (reads `mimic_norm.json`
  first, falls back to `hal_norm.json`). `hal_normalize`,
  `hal_normalize_array` → `mimic_normalize`, `mimic_normalize_array`.
- Constants: `HAL_STICK_CLUSTERS_37`, `HAL_CSTICK_CLUSTERS_9`,
  `HAL_SHOULDER_CLUSTERS_3` → `MIMIC_*`. Legacy names aliased.
- Constructor kwargs: `hal_minimal_features=` → `mimic_minimal_features=`,
  `hal_controller_encoding=` → `mimic_controller_encoding=`. Legacy
  kwargs still accepted.

**Renamed CLI flags (with `--hal-*` aliases):**
- `--hal-mode` → `--mimic-mode`
- `--hal-minimal-features` → `--mimic-minimal-features`
- `--hal-controller-encoding` → `--mimic-controller-encoding`
- `--hal-norm` (slp_to_shards) → `--mimic-norm`

**HuggingFace repo `erickfm/MIMIC`:**
Re-uploaded with `mimic_norm.json` in each character dir. The tarball
paths for model.pt did not change (deduplicated by content on HF). See
`tools/upload_models_to_hf.py` for the new provenance.

## Backwards compatibility matrix

| Old thing | Still works? | How |
|---|---|---|
| `ModelConfig(hal_mode=True, ...)` | ✅ | `__post_init__` migrates to `mimic_mode` |
| `ckpt["config"]` with `hal_mode=True` | ✅ | Same path |
| `--model hal-rope` | ✅ | `MODEL_PRESETS["hal-rope"]` aliased to `"mimic-rope"` |
| `--encoder hal_flat` | ✅ | `ENCODER_REGISTRY["hal_flat"]` → `MimicFlatEncoder` |
| `--hal-mode` / `--hal-minimal-features` / `--hal-controller-encoding` | ✅ | argparse `--x/--y` alias form |
| `from mimic.model import HALFlatEncoder` | ✅ | Module-level alias |
| `features.load_hal_norm(d)` | ✅ | Direct alias of `load_mimic_norm` |
| `data/*_v2/hal_norm.json` (on-disk) | ✅ | Falls back if `mimic_norm.json` missing |
| Existing checkpoint files on disk | ✅ | Verified all 4 load + construct; names kept |

## Verification

Smoke-tested:
- All imports resolve (old + new names)
- `build_encoder("mimic_flat", ...)` constructs
- `build_encoder("hal_flat", ...)` constructs
- `load_mimic_norm` loads from all 4 `data/*_v2/` dirs
- `load_mimic_model(path)` on all 4 production checkpoints:
  `fox-20260413-rope-32k.pt`, `falco-20260412-relpos-28k.pt`,
  `cptfalcon-20260412-relpos-27k.pt`, `luigi-20260412-relpos-5k.pt`.
  All report `mimic_mode=True` after __post_init__ migration.
- `train.py --help` shows new `--mimic-*` flags + legacy aliases.
- `ast.parse` on every touched file (no syntax errors).
- Re-pushed `erickfm/MIMIC` on HF successfully.

## Not renamed (intentional)

- Research notes under `docs/` — historical, left as-is.
- `tools/run_hal_model.py` — runs Eric Gu's actual HAL checkpoints; still
  useful as a reference implementation and explicitly out of the rename
  scope.
- Internal plumbing variable names inside `train.py`, `eval.py`,
  `inference.py` (function params like `hal_mode=`, local vars like
  `_hal_norm`). These are opaque pass-through — they work correctly via
  the backcompat aliases and touching them is churn without value.
- `tools/validate_checkpoint.py`'s internal `HALModel` reimplementation —
  legacy HAL-comparison code.
