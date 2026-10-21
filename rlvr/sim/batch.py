"""GPU-batched MIMIC policy for N sim envs: the batched equivalent of PlayerState + decode_and_press.

Per env it keeps the 180-frame model window on the GPU and advances it by one frame per step
(the oldest frame drops out, the newest goes in), exactly like PlayerState's deque, including the
mock-frame prefill at the start of an episode. The previous decision's controller feature is the
one-hot of the sampled head indices: for every (main, c, shoulder, button) combination that equals
encode_controller_onehot_single of what decode_and_press sends (checked exhaustively, 2026-10-20).
Neutral (no previous decision) = indices (0, 0, 0, NONE).

Sampling draws from the same distribution as _safe_sample's full-distribution path
(softmax(logits / T); argmax when T <= 0 or a row is not finite), via the Gumbel-max trick
(argmax(logits / T + Gumbel noise) is an exact sample of softmax(logits / T)): one kernel per head
and capturable in a CUDA graph. top-k / top-p are not implemented here.

Speed: the per-step work (shift the windows, forward_last, sample) is captured once in a CUDA
graph and replayed (graph=True), so a step costs a few launches instead of hundreds.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

from mimic.features import HAL_CSTICK_CLUSTERS_9, HAL_STICK_CLUSTERS_37
from rlvr.sim.fast_policy import forward_last

HEADS = ("main_xy", "c_dir_logits", "shoulder_val", "btn_logits")
SHOULDER_VALUES = (0.0, 0.4, 1.0)
BTN_NONE = 6
# 7-class button head -> PADStatus bits (decode_and_press: A, B, Z, X for JUMP, L for TRIG, A+L)
BTN_BITS = np.array([0x0100, 0x0200, 0x0010, 0x0400, 0x0040, 0x0140, 0x0000], np.uint16)
_MAIN = np.asarray(HAL_STICK_CLUSTERS_37, np.float32).astype(np.float64)
_C = np.asarray(HAL_CSTICK_CLUSTERS_9, np.float32).astype(np.float64)


def _stick_raw(v):
    return np.round((np.clip(v, 0.0, 1.0) - 0.5) * 160).astype(np.int8)   # half-even, = Python round


def pads_from_indices(idx):
    """(N, 4) int head indices (main, c, shoulder, button) -> (N, 8) uint8 wire pads (see melee_sim)."""
    idx = np.asarray(idx)
    m, c, s, b = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
    out = np.zeros((len(idx), 8), np.uint8)
    bits = BTN_BITS[b]
    out[:, 0] = bits & 0xFF
    out[:, 1] = bits >> 8
    out[:, 2] = _stick_raw(_MAIN[m, 0]).view(np.uint8)
    out[:, 3] = _stick_raw(_MAIN[m, 1]).view(np.uint8)
    out[:, 4] = _stick_raw(_C[c, 0]).view(np.uint8)
    out[:, 5] = _stick_raw(_C[c, 1]).view(np.uint8)
    trig = np.round(np.asarray(SHOULDER_VALUES)[s] * 140).astype(np.uint8)
    out[:, 6] = np.where((b == 4) | (b == 5), 255, trig)   # digital L drives the trigger fully
    return out


class BatchedPolicy:
    def __init__(self, model, seq_len, mock_frame, n_envs, device="cuda", amp=True, compile=True):
        """mock_frame: build_mock_frame(ctx) (each value one frame, leading dim 1)."""
        self.model = model.eval()
        self.n = n_envs
        self.device = device
        self.amp = amp
        self.T = seq_len
        self.mock = {k: v[0].to(device) for k, v in mock_frame.items()}   # one frame per key
        self.keys = [k for k in mock_frame if k != "self_controller"]
        self.buf = {k: v.unsqueeze(0).unsqueeze(0).repeat(n_envs, self.T, *[1] * v.dim()).clone()
                    for k, v in self.mock.items()}
        self.last_idx = torch.zeros(n_envs, 4, dtype=torch.long, device=device)
        self.last_idx[:, 3] = BTN_NONE
        fwd = lambda b: forward_last(self.model, b)
        self.forward = torch.compile(fwd, mode="max-autotune-no-cudagraphs", dynamic=False) if compile else fwd
        # Static inputs for the captured step: this frame's features (host-pinned staging + device).
        self.stage = {k: torch.zeros((n_envs,) + tuple(v.shape), dtype=v.dtype).pin_memory()
                      for k, v in self.mock.items() if k != "self_controller"}
        self.new = {k: torch.zeros_like(v, device=device) for k, v in self.stage.items()}
        self.inv_temp = torch.ones((), device=device)
        self.greedy = torch.zeros((), dtype=torch.bool, device=device)
        self.graph = None
        self.use_graph = True

    def reset(self, env_ids):
        """Start new episodes: mock-filled windows and no previous decision."""
        if len(env_ids) == 0:
            return
        e = torch.as_tensor(env_ids, device=self.device)
        for k, v in self.mock.items():
            self.buf[k][e] = v
        self.last_idx[e] = torch.tensor([0, 0, 0, BTN_NONE], device=self.device)

    def _controller(self):
        i = self.last_idx
        return torch.cat([Fn.one_hot(i[:, 0], 37), Fn.one_hot(i[:, 1], 9),
                          Fn.one_hot(i[:, 3], 7), Fn.one_hot(i[:, 2], 3)], dim=1).float()

    def _device_step(self):
        """Everything after the host-to-device copy: in-place, fixed addresses (graph-capturable)."""
        for k, v in self.new.items():
            b = self.buf[k]
            b[:, :-1].copy_(b[:, 1:].clone())
            b[:, -1].copy_(v.to(b.dtype))
        b = self.buf["self_controller"]
        b[:, :-1].copy_(b[:, 1:].clone())
        b[:, -1].copy_(self._controller())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            preds = self.forward(self.buf)
        idx = torch.stack([_gumbel_sample(preds[h][:, -1].float(), self.inv_temp, self.greedy)
                           for h in HEADS], 1)
        self.last_idx.copy_(idx)

    @torch.no_grad()
    def step(self, feats, temperature=1.0):
        """feats: {key: (N, ...) array} of this frame's features (no controller). Pushes the frame,
        runs the policy, samples. Returns (N, 4) head indices (main, c, shoulder, button), CPU."""
        for k in self.stage:
            self.stage[k].numpy()[:] = feats[k]
            self.new[k].copy_(self.stage[k], non_blocking=True)
        self.greedy.fill_(temperature <= 0)
        self.inv_temp.fill_(1.0 / temperature if temperature > 0 else 1.0)
        if not self.use_graph:
            self._device_step()
        elif self.graph is None:
            s = torch.cuda.Stream()   # warm up (compile, cuBLAS handles) off the capture
            s.wait_stream(torch.cuda.current_stream())
            snapshot = {k: v.clone() for k, v in self.buf.items()}, self.last_idx.clone()
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._device_step()
            torch.cuda.current_stream().wait_stream(s)
            for k, v in snapshot[0].items():   # the warm-up steps must not count
                self.buf[k].copy_(v)
            self.last_idx.copy_(snapshot[1])
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._device_step()
            for k, v in snapshot[0].items():   # nor the capture itself
                self.buf[k].copy_(v)
            self.last_idx.copy_(snapshot[1])
            self.graph.replay()
        else:
            self.graph.replay()
        return self.last_idx.cpu().numpy()


def _gumbel_sample(logits, inv_temp, greedy):
    """argmax(logits / T + Gumbel) ~ softmax(logits / T); argmax of the raw logits when greedy or
    when the row is not finite (the _safe_sample fallbacks)."""
    finite = torch.isfinite(logits).all(dim=1)
    safe = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    u = torch.rand_like(safe).clamp_(1e-20, 1.0)
    noisy = safe * inv_temp - torch.log(-torch.log(u))
    draw = torch.argmax(noisy, dim=1)
    plain = torch.argmax(safe, dim=1)
    return torch.where(finite & ~greedy, draw, plain)
