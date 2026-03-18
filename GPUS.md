# GPU Machines

| Machine | Host | Port | GPUs | Model | Status |
|---------|------|------|------|-------|--------|
| A | 203.57.40.63 | 10015 | 6 | RTX 4090 24 GB | OFFLINE |
| B | 38.65.239.14 | 28750 | 7 | RTX 4090 24 GB | OFFLINE |
| C | 194.14.47.19 | 22824 | 8 | RTX 5090 32 GB | Active |
| D | 142.127.93.36 | 11559 | 8 | RTX 5090 32 GB | Active |
| E | 66.222.138.178 | 11335 | 8 | RTX 5090 32 GB | Active (upload) |
| F | 74.2.96.10 | 18619 | 8 | RTX 5090 32 GB | Active |

```bash
ssh -p 10015 root@203.57.40.63   # Machine A (offline)
ssh -p 28750 root@38.65.239.14   # Machine B (offline)
ssh -p 22824 root@194.14.47.19   # Machine C
ssh -p 11559 root@142.127.93.36  # Machine D
ssh -p 11335 root@66.222.138.178 # Machine E
ssh -p 18619 root@74.2.96.10     # Machine F
```
