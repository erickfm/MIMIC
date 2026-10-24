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

Sliding-window models (cfg.attn_window > 0) stream instead: one frame per step through
mimic/kv_cache.RelPosKVCache, which is exactly the computation they were trained with. Their
episodes start from an empty cache (a training chunk that starts at a game's first frame), not
from mock-filled windows. They also keep the episode's last receptive-field frames
(num_layers x (W - 1) + 1) in a ring buffer, so context() can hand RL the exact input behind a
decision: a window-attention forward over those frames gives the streamed output at its last
position.
"""
import numpy as np
import torch
import torch.nn.functional as Fn

from mimic.features import HAL_CSTICK_CLUSTERS_9, HAL_STICK_CLUSTERS_37
from mimic.kv_cache import RelPosKVCache
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
        raw = getattr(model, "_orig_mod", model)
        self.kv = (RelPosKVCache(raw, n_envs, dtype=torch.bfloat16 if amp else None)
                   if getattr(raw.cfg, "attn_window", 0) > 0 else None)
        if self.kv is not None:   # each env's frames since its episode start, up to the receptive field
            self.R = raw.cfg.num_layers * (raw.cfg.attn_window - 1) + 1
            self.hist = {k: torch.zeros((n_envs, self.R) + tuple(v.shape), dtype=v.dtype, device=device)
                         for k, v in self.mock.items()}
            self.hist_len = torch.zeros(n_envs, dtype=torch.long, device=device)
            self.hist_g = torch.zeros(1, dtype=torch.long, device=device)   # frames written (all envs)
        fwd = lambda b: forward_last(self.model, b)
        self.forward = (torch.compile(fwd, mode="max-autotune-no-cudagraphs", dynamic=False)
                        if compile and self.kv is None else fwd)
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
        if self.kv is not None:
            self.kv.reset(e)
            self.hist_len[e] = 0
        else:
            for k, v in self.mock.items():
                self.buf[k][e] = v
        self.last_idx[e] = torch.tensor([0, 0, 0, BTN_NONE], device=self.device)

    def _controller(self):
        i = self.last_idx
        return torch.cat([Fn.one_hot(i[:, 0], 37), Fn.one_hot(i[:, 1], 9),
                          Fn.one_hot(i[:, 3], 7), Fn.one_hot(i[:, 2], 3)], dim=1).float()

    def _state(self):
        """Mutable device state: what warm-up / capture steps must restore."""
        st = {f"buf.{k}": v for k, v in self.buf.items()}
        st["last_idx"] = self.last_idx
        if self.kv is not None:
            st.update(kv_K=self.kv.K, kv_V=self.kv.V, kv_t=self.kv.t, hist_len=self.hist_len,
                      hist_g=self.hist_g, **{f"hist.{k}": v for k, v in self.hist.items()})
        return st

    def _device_step(self):
        """Everything after the host-to-device copy: in-place, fixed addresses (graph-capturable)."""
        if self.kv is not None:   # stream this frame through the KV cache
            frame = {k: v.unsqueeze(1).to(self.buf[k].dtype) for k, v in self.new.items()}
            frame["self_controller"] = self._controller().unsqueeze(1)
            slot = self.hist_g % self.R
            for k, v in frame.items():
                self.hist[k].index_copy_(1, slot, v.to(self.hist[k].dtype))
            self.hist_len.add_(1).clamp_(max=self.R)
            self.hist_g.add_(1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp, cache_enabled=False):
                preds = self.kv.step(frame)
            self._sample(preds)
            return
        for k, v in self.new.items():
            b = self.buf[k]
            b[:, :-1].copy_(b[:, 1:].clone())
            b[:, -1].copy_(v.to(b.dtype))
        b = self.buf["self_controller"]
        b[:, :-1].copy_(b[:, 1:].clone())
        b[:, -1].copy_(self._controller())
        # cache_enabled=False: required when this runs inside CUDA-graph capture (PyTorch's graph
        # docs); with the cache on, the cast weights live in memory the graph can later overwrite.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp, cache_enabled=False):
            preds = self.forward(self.buf)
        self._sample(preds)

    def _sample(self, preds):
        if "cond_sample" in preds:   # --cond-heads: sampled in order inside the heads
            idx = preds["cond_sample"][:, -1]
        else:
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
        heads = self.model.heads
        if getattr(heads, "cond", False):
            # cond heads sample inside the captured graph at a baked-in temperature
            if self.graph is not None and heads.sample_temperature != float(temperature):
                raise ValueError("a --cond-heads BatchedPolicy cannot change temperature after capture")
            heads.sample_temperature = float(temperature)
        if not self.use_graph:
            self._device_step()
        elif self.graph is None:
            s = torch.cuda.Stream()   # warm up (compile, cuBLAS handles) off the capture
            s.wait_stream(torch.cuda.current_stream())
            snapshot = {k: v.clone() for k, v in self._state().items()}
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._device_step()
            torch.cuda.current_stream().wait_stream(s)
            for k, v in self._state().items():   # the warm-up steps must not count
                v.copy_(snapshot[k])
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._device_step()
            for k, v in self._state().items():   # nor the capture itself
                v.copy_(snapshot[k])
            self.graph.replay()
        else:
            self.graph.replay()
        return self.last_idx.cpu().numpy()


    @torch.no_grad()
    def refresh(self):
        """Sliding-window models: recompute every env's cached keys/values from its stored history
        with the model's current weights. Call after an optimizer step on the model: the cache
        otherwise keeps keys/values from the old weights for up to a receptive field of frames,
        and the rollout would sample from a mix of old and new policies. In place, so a captured
        CUDA graph stays valid. Exact: the R stored frames cover every cached position's own
        receptive field."""
        if self.kv is None:
            return
        from rlvr.sim.ppo import collate
        live = [e for e in range(self.n) if int(self.hist_len[e]) > 0]
        if not live:
            return
        raw = getattr(self.model, "_orig_mod", self.model)
        D, w = raw.cfg.d_model, self.kv.w
        for i in range(0, len(live), 16):
            rows = live[i:i + 16]
            batch, last = collate(self.context(rows), self.device)
            qkv = []
            hooks = [blk.self_attn.c_attn.register_forward_hook(lambda m, inp, out: qkv.append(out))
                     for blk in raw.blocks]
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp, cache_enabled=False):
                    raw.features(batch)
            finally:
                for h in hooks:
                    h.remove()
            for j, e in enumerate(rows):
                L = int(last[j]) + 1
                n = min(L, w)
                t = int(self.kv.t[e])                      # frames seen; the last one is t - 1
                slots = torch.arange(t - n, t, device=self.device) % w
                for li, out in enumerate(qkv):
                    k = out[j, L - n:L, D:2 * D].view(n, -1, raw.blocks[li].self_attn.hs).transpose(0, 1)
                    v = out[j, L - n:L, 2 * D:].view(n, -1, raw.blocks[li].self_attn.hs).transpose(0, 1)
                    self.kv.K[li, e][:, slots] = k.to(self.kv.K.dtype)
                    self.kv.V[li, e][:, slots] = v.to(self.kv.V.dtype)

    def context(self, slots):
        """The exact model input behind each listed slot's latest decision, on the CPU: one dict
        per slot of {key: (L, ...)} frames, oldest first. Full-causal models: the 180-frame window
        (mock frames included). Sliding-window models: the frames since the episode start, up to
        the receptive field; a forward over them gives the streamed output at the last position."""
        s = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        if self.kv is None:   # one batched, non-blocking device-to-host copy per key, one sync
            got = {k: v[s].to("cpu", non_blocking=True) for k, v in self.buf.items()}
            torch.cuda.current_stream().synchronize()
            return [{k: got[k][i] for k in got} for i in range(len(slots))]
        lens = self.hist_len[s].tolist()
        g, Lm = int(self.hist_g), max(lens)
        idx = (torch.arange(g - Lm, g, device=self.device) % self.R)[None]   # one gather for all slots
        got = {k: v[s[:, None], idx].to("cpu", non_blocking=True) for k, v in self.hist.items()}
        torch.cuda.current_stream().synchronize()
        return [{k: got[k][i, Lm - L:] for k in got} for i, L in enumerate(lens)]


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
