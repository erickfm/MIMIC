#!/usr/bin/env python3
"""Evaluate HAL and/or MIMIC checkpoints on the same val data.

Produces per-head CE losses in HAL's format for direct comparison.
Supports both HAL-format checkpoints (keys like stage_emb.weight)
and MIMIC-format checkpoints (keys like encoder.stage_emb.weight).

Usage:
    # Evaluate MIMIC checkpoint
    python tools/validate_checkpoint.py \
        --checkpoint checkpoints/my_model.pt \
        --data-dir data/fox_hal_norm \
        --n-batches 64

    # Compare HAL vs MIMIC side-by-side
    python tools/validate_checkpoint.py \
        --checkpoint checkpoints/hal_original.pt \
        --checkpoint-b checkpoints/my_model.pt \
        --data-dir data/fox_hal_norm \
        --n-batches 64
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mimic.model import FramePredictor, ModelConfig, MODEL_PRESETS
from mimic.dataset import StreamingMeleeDataset
from mimic.features import (
    load_controller_combos, HAL_CSTICK_CLUSTERS_9,
    HAL_STICK_CLUSTERS_37, HAL_SHOULDER_CLUSTERS_3,
)


def collate_fn(batch):
    bs, bt = {}, {}
    for k in batch[0][0]:
        bs[k] = torch.stack([b[0][k] for b in batch])
    for k in batch[0][1]:
        bt[k] = torch.stack([b[1][k] for b in batch])
    return bs, bt


def detect_checkpoint_type(state_dict):
    """Detect if checkpoint is HAL format or MIMIC format."""
    keys = list(state_dict.keys())
    if any(k.startswith("encoder.") or k.startswith("blocks.") for k in keys):
        return "mimic"
    if any(k.startswith("transformer.") or k == "stage_emb.weight" for k in keys):
        return "hal"
    return "unknown"


def load_hal_model(checkpoint_path, device):
    """Load HAL's original checkpoint into a compatible model."""
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    # Build HAL model inline (same as run_hal_model.py)
    import torch.nn as nn

    def _skew(QEr):
        padded = F.pad(QEr, (1, 0))
        B, nh, nr, nc = padded.shape
        return padded.reshape(B, nh, nc, nr)[:, :, 1:, :]

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(512, 1536)
            self.c_proj = nn.Linear(512, 512)
            self.Er = nn.Parameter(torch.randn(1024, 64))
            self.attn_dropout = nn.Dropout(0.2)
            self.resid_dropout = nn.Dropout(0.2)
            self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))

        def forward(self, x):
            B, L, D = x.size()
            q, k, v = self.c_attn(x).split(D, 2)
            k = k.view(B, L, 8, 64).transpose(1, 2)
            q = q.view(B, L, 8, 64).transpose(1, 2)
            v = v.view(B, L, 8, 64).transpose(1, 2)
            s = 1024 - L
            Et = self.Er[s:, :].T
            S = _skew(q @ Et)
            a = ((q @ k.transpose(-2, -1)) + S) / 8.0
            a = a.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
            y = (self.attn_dropout(F.softmax(a, -1)) @ v).transpose(1, 2).contiguous().view(B, L, D)
            return self.resid_dropout(self.c_proj(y))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(512)
            self.attn = Attn()
            self.ln_2 = nn.LayerNorm(512)
            self.mlp = nn.ModuleDict(dict(c_fc=nn.Linear(512, 2048), c_proj=nn.Linear(2048, 512)))
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            h = F.gelu(self.mlp.c_fc(self.ln_2(x)))
            return x + self.dropout(self.mlp.c_proj(h))

    class HALModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage_emb = nn.Embedding(6, 4)
            self.character_emb = nn.Embedding(27, 12)
            self.action_emb = nn.Embedding(396, 32)
            self.transformer = nn.ModuleDict(dict(
                proj_down=nn.Linear(164, 512), drop=nn.Dropout(0.2),
                h=nn.ModuleList([Block() for _ in range(6)]), ln_f=nn.LayerNorm(512)))

            def _h(i, o):
                return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, i // 2), nn.GELU(), nn.Linear(i // 2, o))
            self.shoulder_head = _h(512, 3)
            self.c_stick_head = _h(515, 9)
            self.main_stick_head = _h(524, 37)
            self.button_head = _h(561, 5)

        def forward(self, seq):
            # Stage remap
            stage = (seq["stage"] - 1).clamp(min=0)
            # Numeric: select and reorder to HAL order
            sn, on = seq["self_numeric"], seq["opp_numeric"]
            if sn.shape[-1] > 7:
                sn = sn[..., [0, 1, 2, 3, 4, 13]]
                on = on[..., [0, 1, 2, 3, 4, 13]]
            elif sn.shape[-1] == 7:
                sn = sn[..., [0, 1, 2, 3, 4, 6]]
                on = on[..., [0, 1, 2, 3, 4, 6]]
            sf = seq["self_flags"][..., [0, 2, 3]].float() * 2.0 - 1.0
            of = seq["opp_flags"][..., [0, 2, 3]].float() * 2.0 - 1.0
            _O = [2, 3, 7, 8, 4, 6, 5, 0, 1]
            gs = torch.cat([
                torch.cat([sn, sf], -1)[..., _O],
                torch.cat([on, of], -1)[..., _O],
            ], -1)

            x = torch.cat([
                self.stage_emb(stage),
                self.character_emb(seq["self_character"]),
                self.character_emb(seq["opp_character"]),
                self.action_emb(seq["self_action"]),
                self.action_emb(seq["opp_action"]),
                gs, seq["self_controller"],
            ], -1)

            x = self.transformer.drop(self.transformer.proj_down(x))
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)

            s = self.shoulder_head(x)
            c = self.c_stick_head(torch.cat([x, s.detach()], -1))
            m = self.main_stick_head(torch.cat([x, s.detach(), c.detach()], -1))
            b = self.button_head(torch.cat([x, s.detach(), c.detach(), m.detach()], -1))
            return {"shoulder_val": s, "c_dir_logits": c, "main_xy": m, "btn_logits": b}

    model = HALModel().to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_mimic_model(checkpoint_path, device):
    """Load MIMIC checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = FramePredictor(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def compute_losses(model, state, target, stick_centers, shoulder_centers):
    """Compute per-head CE losses matching HAL's format."""
    with torch.no_grad():
        preds = model(state)

    # Main stick
    main_pred = preds["main_xy"].float()
    # Re-cluster targets from raw x/y using HAL's 37 clusters
    if "main_x" in target and stick_centers is not None:
        xy = torch.stack([target["main_x"], target["main_y"]], dim=-1)
        dists = torch.cdist(xy.reshape(-1, 2), stick_centers)
        main_tgt = dists.argmin(dim=-1).reshape(xy.shape[:-1])
    else:
        main_tgt = target["main_cluster"].long()
    loss_main = F.cross_entropy(main_pred.reshape(-1, 37), main_tgt.reshape(-1))

    # Shoulder: combined max(L,R) → 3 classes [0.0, 0.4, 1.0]
    shldr_pred = preds["shoulder_val"].float()
    sc = shoulder_centers
    if "l_shldr" in target and "r_shldr" in target:
        # Compute from raw shoulder values
        l_vals = target["l_shldr"].float().reshape(-1)
        r_vals = target["r_shldr"].float().reshape(-1)
    else:
        l_bin = target["l_bin"].long().reshape(-1).clamp(max=len(sc) - 1)
        r_bin = target["r_bin"].long().reshape(-1).clamp(max=len(sc) - 1)
        l_vals, r_vals = sc[l_bin], sc[r_bin]
    combined = torch.max(l_vals, r_vals)
    shldr_tgt = (combined.unsqueeze(-1) - sc).abs().argmin(-1)
    loss_shoulder = F.cross_entropy(shldr_pred.reshape(-1, 3), shldr_tgt)

    # C-stick
    c_pred = preds["c_dir_logits"].float()
    c_tgt = target["c_dir"]
    n_cdir = c_pred.size(-1)
    if c_tgt.size(-1) == n_cdir:
        cdir_classes = c_tgt.argmax(dim=-1)
    elif n_cdir == 9 and c_tgt.size(-1) == 5:
        _MAP = torch.tensor([0, 4, 3, 2, 1], dtype=torch.long, device=c_tgt.device)
        cdir_classes = _MAP[c_tgt.argmax(dim=-1)]
    else:
        cdir_classes = c_tgt.argmax(dim=-1)
    loss_cstick = F.cross_entropy(c_pred.reshape(-1, n_cdir), cdir_classes.reshape(-1))

    # Buttons
    btn_pred = preds["btn_logits"].float()
    if "btns_single" in target:
        btn_tgt = target["btns_single"].long()
    else:
        btn_raw = target.get("btns", target.get("btns_float")).float()
        a = btn_raw[..., 0] > 0.5
        b = btn_raw[..., 1] > 0.5
        jump = (btn_raw[..., 2] > 0.5) | (btn_raw[..., 3] > 0.5)
        z = btn_raw[..., 4] > 0.5
        btn_tgt = torch.full(btn_raw.shape[:-1], 4, device=btn_raw.device, dtype=torch.long)
        btn_tgt[z] = 3; btn_tgt[jump] = 2; btn_tgt[b] = 1; btn_tgt[a] = 0
    loss_buttons = F.cross_entropy(btn_pred.reshape(-1, 5), btn_tgt.reshape(-1))

    total = loss_main + loss_shoulder + loss_cstick + loss_buttons
    return {
        "loss_total": total.item(),
        "loss_main_stick": loss_main.item(),
        "loss_shoulder": loss_shoulder.item(),
        "loss_c_stick": loss_cstick.item(),
        "loss_buttons": loss_buttons.item(),
    }


def evaluate(model, dataloader, device, n_batches, stick_centers, shoulder_centers):
    """Run evaluation over n_batches."""
    sums = {}
    count = 0
    for i, (state, target) in enumerate(dataloader):
        if i >= n_batches:
            break
        for k in state:
            state[k] = state[k].to(device, non_blocking=True)
        for k in target:
            target[k] = target[k].to(device, non_blocking=True)
        losses = compute_losses(model, state, target, stick_centers, shoulder_centers)
        for k, v in losses.items():
            if math.isfinite(v):
                sums[k] = sums.get(k, 0.0) + v
        count += 1
    return {k: v / count for k, v in sums.items()} if count > 0 else {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint(s) on val data")
    parser.add_argument("--checkpoint", required=True, help="Primary checkpoint to evaluate")
    parser.add_argument("--checkpoint-b", default=None, help="Second checkpoint for comparison")
    parser.add_argument("--data-dir", required=True, help="Data directory with shards + metadata")
    parser.add_argument("--n-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = args.data_dir

    # Load data
    combos, combo_map, _ = load_controller_combos(data_dir)
    val_ds = StreamingMeleeDataset(
        data_dir=data_dir, sequence_length=256, reaction_delay=1, split="val",
        hal_controller_encoding=True, controller_combo_map=combo_map, n_controller_combos=5,
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=4, drop_last=True)

    stick_centers = torch.tensor(HAL_STICK_CLUSTERS_37, dtype=torch.float32, device=device)
    shoulder_centers = torch.tensor(HAL_SHOULDER_CLUSTERS_3, dtype=torch.float32, device=device)

    # Evaluate checkpoint A
    print(f"Loading {args.checkpoint}...")
    sd_a = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(sd_a, dict) and "model_state_dict" in sd_a:
        ckpt_type_a = detect_checkpoint_type(sd_a["model_state_dict"])
    else:
        sd_a = {k.removeprefix("module."): v for k, v in sd_a.items()}
        ckpt_type_a = detect_checkpoint_type(sd_a)

    if ckpt_type_a == "hal":
        model_a = load_hal_model(args.checkpoint, device)
        name_a = "HAL Original"
    else:
        model_a, _ = load_mimic_model(args.checkpoint, device)
        name_a = f"MIMIC ({Path(args.checkpoint).stem})"

    print(f"Evaluating {name_a} on {args.n_batches} batches...")
    metrics_a = evaluate(model_a, val_dl, device, args.n_batches, stick_centers, shoulder_centers)

    # Evaluate checkpoint B (if provided)
    metrics_b = None
    name_b = None
    if args.checkpoint_b:
        print(f"\nLoading {args.checkpoint_b}...")
        sd_b = torch.load(args.checkpoint_b, map_location=device, weights_only=False)
        if isinstance(sd_b, dict) and "model_state_dict" in sd_b:
            ckpt_type_b = detect_checkpoint_type(sd_b["model_state_dict"])
        else:
            sd_b = {k.removeprefix("module."): v for k, v in sd_b.items()}
            ckpt_type_b = detect_checkpoint_type(sd_b)

        if ckpt_type_b == "hal":
            model_b = load_hal_model(args.checkpoint_b, device)
            name_b = "HAL Original"
        else:
            model_b, _ = load_mimic_model(args.checkpoint_b, device)
            name_b = f"MIMIC ({Path(args.checkpoint_b).stem})"

        print(f"Evaluating {name_b} on {args.n_batches} batches...")
        metrics_b = evaluate(model_b, val_dl, device, args.n_batches, stick_centers, shoulder_centers)

    # Print results
    print("\n" + "=" * 60)
    if metrics_b:
        print(f"{'Metric':20s} {name_a:>18s} {name_b:>18s}")
        print("-" * 60)
        for k in ["loss_total", "loss_main_stick", "loss_buttons", "loss_c_stick", "loss_shoulder"]:
            va = metrics_a.get(k, float("nan"))
            vb = metrics_b.get(k, float("nan"))
            print(f"{k:20s} {va:>18.4f} {vb:>18.4f}")
    else:
        print(f"{'Metric':20s} {name_a:>18s}")
        print("-" * 42)
        for k in ["loss_total", "loss_main_stick", "loss_buttons", "loss_c_stick", "loss_shoulder"]:
            va = metrics_a.get(k, float("nan"))
            print(f"{k:20s} {va:>18.4f}")


if __name__ == "__main__":
    main()
