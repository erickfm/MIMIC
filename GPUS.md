# GPU Machines

| Machine | Host | Port | GPUs | Model |
|---------|------|------|------|-------|
| A | 203.57.40.63 | 10015 | 6 | RTX 4090 24 GB |
| B | 38.65.239.14 | 28750 | 7 | RTX 4090 24 GB |
| C | 38.65.239.56 | 45107 | 8 | RTX 4090 24 GB |

```bash
ssh -p 10015 root@203.57.40.63   # Machine A
ssh -p 28750 root@38.65.239.14   # Machine B
ssh -p 45107 root@38.65.239.56   # Machine C
```
