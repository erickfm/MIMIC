# GPU Inventory

## Machines

| Machine | Host | Port | GPUs | GPU Model | VRAM/GPU | Notes |
|---------|------|------|------|-----------|----------|-------|
| A | 203.57.40.63 | 10015 | 6 | RTX 4090 | 24 GB | |
| B | 38.65.239.14 | 28750 | 7 | RTX 4090 | 24 GB | |
| C | 38.65.239.56 | 45107 | 8 | RTX 4090 | 24 GB | |
| D | 66.222.152.184 | 10003 | 6 | RTX 4090 | 24 GB | 20 GB writable disk -- cannot fit dataset |

**Total: 27 GPUs (21 usable -- Machine D lacks disk space for the 71 GB dataset)**

## SSH Access

```bash
ssh -p 10015 root@203.57.40.63   # Machine A
ssh -p 28750 root@38.65.239.14   # Machine B
ssh -p 45107 root@38.65.239.56   # Machine C
ssh -p 10003 root@66.222.152.184 # Machine D
```

## Batch Size Limits (RTX 4090, 24 GB VRAM)

Empirically determined max batch sizes with `hybrid16` encoder, seq_len=60:

| Model Preset | Params | Max Batch Size |
|-------------|--------|----------------|
| tiny | ~6M | 128 |
| small | ~16M | 128 |
| medium | ~32M | 128 |
| base | ~54M | 64 |
| deep | ~29M | 64 |
| wide-shallow | ~62M | 32 |
| xlarge | ~105M | 32 |
| xxlarge | ~232M | 16 |

For longer context lengths (small model):

| Seq Len | Max Batch Size |
|---------|----------------|
| 30 | 128 |
| 60 | 128 |
| 120 | 64 |
| 180 | 48 |
| 240 | 32 |
| 360 | 16 |

## Software

All machines run:
- Python 3.11
- PyTorch with CUDA (bf16 AMP, torch.compile)
- Dataset: `data/full` (71 GB parquet shards from HuggingFace `erickfm/frame-melee`)
- Code: `/root/FRAME` (git clone of this repo)
