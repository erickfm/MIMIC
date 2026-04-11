#!/usr/bin/env python3
"""Inspect exactly what goes into and comes out of the model for a given frame.

Usage:
    # Show frame 533 from shard 0 of fox_hal_full
    python tools/inspect_frame.py --checkpoint checkpoints/hal-7class-relpos_best.pt \
        --data-dir data/fox_hal_full --shard 0 --frame 533

    # Show frame at position 200 in a 256-frame window starting at frame 5000
    python tools/inspect_frame.py --checkpoint checkpoints/hal-7class-relpos_best.pt \
        --data-dir data/fox_hal_full --shard 0 --window-start 5000 --pos 200

    # Show a range of frames around a transition
    python tools/inspect_frame.py --checkpoint checkpoints/hal-7class-relpos_best.pt \
        --data-dir data/fox_hal_full --shard 0 --frame 533 --context 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import torch
import torch.nn.functional as F
from mimic.model import FramePredictor, ModelConfig
from mimic.features import (
    HAL_STICK_CLUSTERS_37, HAL_CSTICK_CLUSTERS_9, HAL_SHOULDER_CLUSTERS_3,
)

BTN_NAMES = ["A", "B", "Z", "JUMP", "TRIG", "A_TRIG", "NONE"]
STICK_NAMES = {i: f"({HAL_STICK_CLUSTERS_37[i][0]:.2f},{HAL_STICK_CLUSTERS_37[i][1]:.2f})" for i in range(37)}
CSTICK_NAMES = {i: f"({HAL_CSTICK_CLUSTERS_9[i][0]:.2f},{HAL_CSTICK_CLUSTERS_9[i][1]:.2f})" for i in range(9)}
SHOULDER_NAMES = {0: "off", 1: "0.4", 2: "full"}

HAL_STATE_KEYS = [
    "stage", "self_character", "opp_character", "self_action", "opp_action",
    "self_numeric", "opp_numeric", "self_flags", "opp_flags", "self_controller",
    "self_port", "opp_port",
]

# After the encoder reorders, the 9 numeric features per player are:
# [percent, stock, facing, invulnerable, jumps_left, on_ground, shield, pos_x, pos_y]
ENCODER_NUM_NAMES = ["percent", "stock", "facing", "invuln", "jumps", "grounded", "shield", "pos_x", "pos_y"]


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in ModelConfig.__dataclass_fields__}
    cfg = ModelConfig(**cfg_dict)
    model = FramePredictor(cfg)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, cfg


def decode_controller(ctrl_vec):
    """Decode a 56-dim one-hot controller vector into human-readable parts."""
    main_idx = ctrl_vec[:37].argmax().item()
    cstick_idx = ctrl_vec[37:46].argmax().item()
    btn_idx = ctrl_vec[46:53].argmax().item()
    shldr_idx = ctrl_vec[53:56].argmax().item()
    return {
        "main": f"cl{main_idx} {STICK_NAMES[main_idx]}",
        "c_stick": f"cl{cstick_idx} {CSTICK_NAMES[cstick_idx]}",
        "button": BTN_NAMES[btn_idx],
        "shoulder": SHOULDER_NAMES[shldr_idx],
    }


def show_frame(t, abs_idx, states, targets, preds, ctrl_shifted, encoder_input=None):
    """Print full input/output for one frame position."""
    print(f"┌─── Frame {abs_idx} (window pos {t}) ───────────────────────────────────")

    # --- ENCODER INPUT: what the model actually sees ---
    print(f"│")
    print(f"│  ENCODER INPUT (what goes into the model):")
    print(f"│")

    # Categoricals → embeddings
    stage = states["stage"][abs_idx].item()
    self_char = states["self_character"][abs_idx].item()
    opp_char = states["opp_character"][abs_idx].item()
    self_act = states["self_action"][abs_idx].item()
    opp_act = states["opp_action"][abs_idx].item()
    print(f"│    stage={stage}  self_char={self_char}  opp_char={opp_char}  self_action={self_act}  opp_action={opp_act}")

    # Numerics (after encoder reorder to HAL order)
    if encoder_input is not None:
        # Use the actual concatenated vector the encoder produced
        enc = encoder_input[t]
        # encoder output is: stage(4) + self_char(12) + opp_char(12) + self_act(32) + opp_act(32)
        #                    + self_num(9) + opp_num(9) + controller(56) = 166
        offset = 4 + 12 + 12 + 32 + 32  # = 92
        self_num = enc[offset:offset+9]
        opp_num = enc[offset+9:offset+18]
        ctrl = enc[offset+18:offset+18+56]

        print(f"│")
        print(f"│    self gamestate (9 features, HAL order):")
        for i, name in enumerate(ENCODER_NUM_NAMES):
            print(f"│      {name:>10s}: {self_num[i].item():+.4f}")

        print(f"│    opp gamestate:")
        for i, name in enumerate(ENCODER_NUM_NAMES):
            print(f"│      {name:>10s}: {opp_num[i].item():+.4f}")

        print(f"│")
        print(f"│    controller input (56-dim one-hot from prev frame):")
        c = decode_controller(ctrl)
        print(f"│      main: {c['main']}  c_stick: {c['c_stick']}  button: {c['button']}  shoulder: {c['shoulder']}")
    else:
        # Fall back to raw shard values
        sn = states["self_numeric"][abs_idx]
        on = states["opp_numeric"][abs_idx]
        sf = states["self_flags"][abs_idx]
        of_ = states["opp_flags"][abs_idx]

        # Reproduce encoder's reorder: pick [0,1,2,3,4,13] from 22-col, add flags [0,2,3], reorder
        if sn.shape[0] > 7:
            sn6 = sn[[0,1,2,3,4,13]]
            on6 = on[[0,1,2,3,4,13]]
        elif sn.shape[0] == 7:
            sn6 = sn[[0,1,2,3,4,6]]
            on6 = on[[0,1,2,3,4,6]]
        else:
            sn6 = sn[:6]
            on6 = on[:6]

        sf3 = sf[[0,2,3]] * 2.0 - 1.0
        of3 = of_[[0,2,3]] * 2.0 - 1.0
        self_9 = torch.cat([sn6, sf3])[[2,3,7,8,4,6,5,0,1]]
        opp_9 = torch.cat([on6, of3])[[2,3,7,8,4,6,5,0,1]]

        print(f"│")
        print(f"│    self gamestate (9 features, HAL order):")
        for i, name in enumerate(ENCODER_NUM_NAMES):
            print(f"│      {name:>10s}: {self_9[i].item():+.4f}")

        print(f"│    opp gamestate:")
        for i, name in enumerate(ENCODER_NUM_NAMES):
            print(f"│      {opp_9[i].item():+.4f}")

        # Controller from offset
        print(f"│")
        if t == 0:
            print(f"│    controller input: ALL ZEROS (offset shifted out)")
        else:
            c_vec = ctrl_shifted[t]
            if c_vec.sum().item() == 0:
                print(f"│    controller input: ALL ZEROS")
            else:
                c = decode_controller(c_vec)
                print(f"│    controller input: main={c['main']}  c_stick={c['c_stick']}  button={c['button']}  shoulder={c['shoulder']}")

    # --- TARGET ---
    btn_tgt = targets["btns_single"][abs_idx].item()
    main_tgt = targets["main_cluster"][abs_idx].item()
    l_tgt = targets["l_bin"][abs_idx].item()
    r_tgt = targets["r_bin"][abs_idx].item()
    cdir_tgt = targets["c_dir"][abs_idx].argmax().item()

    print(f"│")
    print(f"│  TARGET (ground truth):")
    print(f"│    button:   {BTN_NAMES[btn_tgt]}")
    print(f"│    main:     cl{main_tgt} {STICK_NAMES[main_tgt]}")
    print(f"│    c_stick:  cl{cdir_tgt} {CSTICK_NAMES[cdir_tgt]}")
    print(f"│    shoulder: L={SHOULDER_NAMES[l_tgt]} R={SHOULDER_NAMES[r_tgt]}")

    # --- PREDICTION ---
    btn_logits = preds["btn_logits"][0, t].float()
    btn_probs = F.softmax(btn_logits, dim=-1)
    main_logits = preds["main_xy"][0, t].float()
    main_probs = F.softmax(main_logits, dim=-1)
    shldr_logits = preds["shoulder_val"][0, t].float()
    shldr_probs = F.softmax(shldr_logits, dim=-1)
    cdir_logits = preds["c_dir_logits"][0, t].float()
    cdir_probs = F.softmax(cdir_logits, dim=-1)

    btn_pred = btn_probs.argmax().item()
    main_pred = main_probs.argmax().item()
    shldr_pred = shldr_probs.argmax().item()
    cdir_pred = cdir_probs.argmax().item()

    print(f"│")
    print(f"│  PREDICTION:")
    top5_btn = btn_probs.topk(min(5, len(BTN_NAMES)))
    print(f"│    button:   {BTN_NAMES[btn_pred]} {'✓' if btn_pred == btn_tgt else '✗'}")
    print(f"│      probs:  {', '.join(f'{BTN_NAMES[i]}={v:.4f}' for v,i in zip(top5_btn.values.tolist(), top5_btn.indices.tolist()))}")
    print(f"│    main:     cl{main_pred} {STICK_NAMES[main_pred]} {'✓' if main_pred == main_tgt else '✗'}")
    print(f"│    c_stick:  cl{cdir_pred} {CSTICK_NAMES[cdir_pred]} {'✓' if cdir_pred == cdir_tgt else '✗'}")
    print(f"│    shoulder: {SHOULDER_NAMES[shldr_pred]} {'✓' if shldr_pred == l_tgt else '✗'}")
    print(f"└──────────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Inspect model input/output for a specific frame")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--shard", type=int, default=0, help="Train shard index")
    parser.add_argument("--frame", type=int, default=None, help="Absolute frame index in shard")
    parser.add_argument("--window-start", type=int, default=None, help="Start of 256-frame window")
    parser.add_argument("--pos", type=int, default=None, help="Position within window (0-255)")
    parser.add_argument("--context", type=int, default=0, help="Show N frames before and after")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    W = cfg.max_seq_len  # 256

    # Load shard
    shards = sorted(glob.glob(f"{args.data_dir}/train_shard_*.pt"))
    shard = torch.load(shards[args.shard], map_location="cpu", weights_only=False)
    states = shard["states"]
    targets = shard["targets"]
    n_frames = len(states["self_action"])

    # Determine window and position
    if args.frame is not None:
        # Center a window around this frame
        if args.window_start is not None:
            window_start = args.window_start
            pos = args.frame - window_start
        else:
            window_start = max(0, args.frame - W + 1)
            pos = args.frame - window_start
    elif args.window_start is not None and args.pos is not None:
        window_start = args.window_start
        pos = args.pos
    else:
        parser.error("Provide --frame or both --window-start and --pos")

    assert 0 <= pos < W, f"Position {pos} out of range [0, {W})"
    assert window_start + W <= n_frames, f"Window {window_start}:{window_start+W} exceeds shard ({n_frames} frames)"

    # Build window with controller offset
    ctrl = states["self_controller"][window_start:window_start+W].clone()
    shifted = torch.zeros_like(ctrl)
    shifted[1:] = ctrl[:-1]

    window = {}
    for k in HAL_STATE_KEYS:
        if k == "self_controller":
            window[k] = shifted.unsqueeze(0).to(device)
        else:
            window[k] = states[k][window_start:window_start+W].unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        preds = model(window)

    # Also get the encoder's combined input (pre-projection) for exact inspection
    # Hook into the encoder to capture the concatenated vector
    encoder_combined = {}
    def hook_fn(module, input, output):
        # The last thing forward() does is self.drop(self.proj(combined))
        # We want combined. Grab input to self.proj instead.
        pass

    # Actually, let's capture by running encoder manually
    enc = model.encoder
    with torch.no_grad():
        seq = window
        stage_idx = seq["stage"]
        if enc.stage_emb.num_embeddings == 6:
            stage_idx = (stage_idx - 1).clamp(min=0)
        stage_e = enc.stage_emb(stage_idx)
        self_char_e = enc.char_emb(seq["self_character"])
        opp_char_e = enc.char_emb(seq["opp_character"])
        self_act_e = enc.action_emb(seq["self_action"])
        opp_act_e = enc.action_emb(seq["opp_action"])

        sn = seq["self_numeric"]
        on = seq["opp_numeric"]
        if sn.shape[-1] > 7:
            sn = sn[..., [0,1,2,3,4,13]]
            on = on[..., [0,1,2,3,4,13]]
        elif sn.shape[-1] == 7:
            sn = sn[..., [0,1,2,3,4,6]]
            on = on[..., [0,1,2,3,4,6]]
        sf = seq["self_flags"][..., [0,2,3]].float() * 2.0 - 1.0
        of_ = seq["opp_flags"][..., [0,2,3]].float() * 2.0 - 1.0
        _HAL_ORDER = [2,3,7,8,4,6,5,0,1]
        self_num = torch.cat([sn, sf], dim=-1)[..., _HAL_ORDER]
        opp_num = torch.cat([on, of_], dim=-1)[..., _HAL_ORDER]

        parts = [stage_e, self_char_e, opp_char_e, self_act_e, opp_act_e, self_num, opp_num]
        if enc._hal_ctrl_enc and not enc._no_self_inputs and "self_controller" in seq:
            parts.append(seq["self_controller"])
        combined = torch.cat(parts, dim=-1)  # (1, W, 166)

    encoder_input = combined[0].cpu()  # (W, 166)

    # Show frames
    frames_to_show = list(range(max(0, pos - args.context), min(W, pos + args.context + 1)))
    for t in frames_to_show:
        show_frame(t, window_start + t, states, targets, preds, shifted, encoder_input)
        print()


if __name__ == "__main__":
    main()
