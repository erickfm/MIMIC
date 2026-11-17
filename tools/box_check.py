#!/usr/bin/env python3
"""Check a rented GPU box before trusting it with runs. Start it right after setup.sh (it waits until
torch can be imported), or any time on an idle box.

  python3 tools/box_check.py                    # report + runs/box_check.json, exit 0 ok / 1 warn / 2 fail
  python3 tools/box_check.py --ref OTHER.json   # also compare against a known-good box's report
  python3 tools/box_check.py --gpus 1           # only GPU 1 (e.g. while GPU 0 is busy)

Why: the 2026-11-16 vast.ai box had one bad RTX 5090 (PCIe stuck at gen 2 under load, half the power
of its twin, BC training 0.4-0.7 vs 7.4 step/s on the other GPU) that looked healthy in nvidia-smi
at idle. Every check below runs the GPU under load, one GPU at a time, and compares GPUs of the same
model with each other: a GPU more than 25% slower than the box's best one fails.

Per GPU (each in its own subprocess): bf16 and fp32 (TF32) matmul throughput over ~10 s, a small
transformer forward+backward in bf16 at the BC batch shape (768 x 180), pinned host<->device copy
bandwidth, and during the load the PCIe link generation/width, power draw and clock (nvidia-smi).
Box: usable CPU cores (affinity + cgroup quota), single- and all-core CPU speed (the sim is
CPU-bound), RAM, free disk and write speed where the repo lives, and download speed from Hugging Face.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOW = 0.75          # a GPU below this fraction of the best same-model GPU on the box fails


def smi(gpu, fields):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), f"--query-gpu={','.join(fields)}",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return dict(zip(fields, [v.strip() for v in out.split(",")]))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── one GPU (runs in a subprocess with only that GPU visible) ───────────────────────────────────
def gpu_worker(phys):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda:0")
    res = {"gpu": phys, "name": torch.cuda.get_device_name(0),
           "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}

    samples, stop = [], threading.Event()
    fields = ["pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current",
              "pcie.link.width.max", "power.draw", "power.limit", "clocks.sm", "clocks.max.sm",
              "utilization.gpu", "temperature.gpu"]

    def sampler():
        while not stop.is_set():
            samples.append(smi(phys, fields))
            time.sleep(0.5)

    def timed(fn, seconds):
        fn(); torch.cuda.synchronize()
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            fn(); n += 1
        torch.cuda.synchronize()
        return n / (time.time() - t0)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    try:
        n = 8192
        a, b = (torch.randn(n, n, device=dev, dtype=torch.bfloat16) for _ in range(2))
        res["bf16_tflops"] = round(timed(lambda: a @ b, 10) * 2 * n**3 / 1e12, 1)
        a, b = a.float(), b.float()
        res["tf32_tflops"] = round(timed(lambda: a @ b, 6) * 2 * n**3 / 1e12, 1)
        del a, b

        # BC-shaped transformer step: 768 windows x 180 frames, S width (256), 4 layers, bf16 autocast
        layer = torch.nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True, dropout=0.0)
        model = torch.nn.TransformerEncoder(layer, 4).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(768, 180, 256, device=dev)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(180, device=dev)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, mask=mask, is_causal=True).float().pow(2).mean()
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        res["train_steps_s"] = round(timed(step, 8), 2)
        del model, opt, x

        host = torch.empty(1 << 28, dtype=torch.float32, pin_memory=True)    # 1 GiB
        devb = torch.empty_like(host, device=dev)
        res["h2d_gb_s"] = round(timed(lambda: devb.copy_(host, non_blocking=True), 3) * 4 * 2**28 / 1e9, 1)
        res["d2h_gb_s"] = round(timed(lambda: host.copy_(devb, non_blocking=True), 3) * 4 * 2**28 / 1e9, 1)
    finally:
        stop.set(); th.join()

    busy = [s for s in samples if (num(s.get("utilization.gpu")) or 0) >= 50] or samples
    pick = lambda k, f=max: f(v for v in (num(s.get(k)) for s in busy) if v is not None) if busy else None
    res.update(pcie_gen_load=pick("pcie.link.gen.current", min), pcie_gen_max=pick("pcie.link.gen.max"),
               pcie_width_load=pick("pcie.link.width.current", min), pcie_width_max=pick("pcie.link.width.max"),
               power_w_load=pick("power.draw"), power_limit_w=pick("power.limit"),
               sm_clock_load=pick("clocks.sm"), sm_clock_max=pick("clocks.max.sm"),
               temp_c_max=pick("temperature.gpu"))
    return res


# ── box ─────────────────────────────────────────────────────────────────────────────────────────
def cpu_burn(_):
    t0, x = time.time(), 0
    for i in range(3_000_000):
        x = (x * 1103515245 + i) % 2147483647
    return time.time() - t0


def cpu_check():
    cores = len(os.sched_getaffinity(0))
    quota = None
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        quota = None if q == "max" else round(int(q) / int(p), 1)
    except Exception:
        pass
    single = cpu_burn(0)
    n = int(min(cores, quota or cores))
    t0 = time.time()
    with ProcessPoolExecutor(n) as ex:
        list(ex.map(cpu_burn, range(n)))
    allc = time.time() - t0
    return {"cores": cores, "cgroup_cpus": quota, "single_core_s": round(single, 2),
            "all_core_speedup": round(n * single / allc, 1), "all_core_workers": n}


def disk_check(path):
    free = shutil.disk_usage(path).free / 2**30
    buf = os.urandom(64 << 20)
    with tempfile.NamedTemporaryFile(dir=path, delete=True) as f:
        t0 = time.time()
        for _ in range(16):                      # 1 GiB
            f.write(buf)
        f.flush(); os.fsync(f.fileno())
        write = 16 * 64 / 1024 / (time.time() - t0)
    return {"free_gb": round(free), "write_gb_s": round(write, 2)}


def net_check(mb=200):
    try:
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        url = hf_hub_url("erickfm/mimic-shards-v2", "fox/train_shard_000_000.pt", repo_type="dataset")
        t0, n = time.time(), 0
        with requests.get(url, headers={**build_hf_headers(), "Range": f"bytes=0-{mb * 2**20 - 1}"},
                          stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                n += len(chunk)
        return {"hf_download_mb_s": round(n / 2**20 / (time.time() - t0))}
    except Exception as e:
        return {"hf_download_mb_s": None, "error": f"{type(e).__name__}: {e}"[:200]}


# ── verdict ─────────────────────────────────────────────────────────────────────────────────────
def judge(rep, ref):
    issues = []                                   # (level, message): level 1 warn, 2 fail
    gpus = rep["gpus"]
    for key in ("bf16_tflops", "train_steps_s", "h2d_gb_s"):
        by_model = {}
        for g in gpus:
            if g.get(key) is not None:
                by_model.setdefault(g["name"], []).append(g)
        for name, gs in by_model.items():
            best = max(g[key] for g in gs)
            for g in gs:
                if g[key] < SLOW * best:
                    issues.append((2, f"GPU {g['gpu']}: {key} {g[key]} is {g[key] / best:.0%} of the best {name} here ({best})"))
    for g in gpus:
        if "error" in g:
            issues.append((2, f"GPU {g['gpu']}: {g['error']}"))
            continue
        if g["pcie_gen_load"] and g["pcie_gen_max"] and g["pcie_gen_load"] < g["pcie_gen_max"]:
            issues.append((2 if g["pcie_gen_load"] <= 2 else 1,
                           f"GPU {g['gpu']}: PCIe gen {g['pcie_gen_load']:.0f} under load (max {g['pcie_gen_max']:.0f})"))
        if g["pcie_width_load"] and g["pcie_width_max"] and g["pcie_width_load"] < g["pcie_width_max"]:
            issues.append((1, f"GPU {g['gpu']}: PCIe x{g['pcie_width_load']:.0f} under load (max x{g['pcie_width_max']:.0f})"))
        if g["power_w_load"] and g["power_limit_w"] and g["power_w_load"] < 0.5 * g["power_limit_w"]:
            issues.append((1, f"GPU {g['gpu']}: peak {g['power_w_load']:.0f} W under matmul load, limit {g['power_limit_w']:.0f} W"))
        if g["temp_c_max"] and g["temp_c_max"] >= 87:
            issues.append((1, f"GPU {g['gpu']}: reached {g['temp_c_max']:.0f} C"))
        if ref:
            for rg in ref.get("gpus", []):
                if rg.get("name") == g["name"]:
                    for key in ("bf16_tflops", "train_steps_s"):
                        if rg.get(key) and g.get(key) is not None and g[key] < SLOW * rg[key]:
                            issues.append((2, f"GPU {g['gpu']}: {key} {g[key]} vs {rg[key]} on the reference box"))
                    break
    c = rep["cpu"]
    if c["cgroup_cpus"] and c["cgroup_cpus"] < c["cores"]:
        issues.append((1, f"CPU: cgroup quota {c['cgroup_cpus']} of {c['cores']} visible cores"))
    if c["all_core_speedup"] < 0.5 * c["all_core_workers"]:
        issues.append((1, f"CPU: {c['all_core_workers']} workers only {c['all_core_speedup']}x one core"))
    if rep["disk"]["free_gb"] < 200:
        issues.append((1, f"disk: {rep['disk']['free_gb']} GB free"))
    if not rep["net"].get("hf_download_mb_s"):
        issues.append((1, f"network: HF download failed ({rep['net'].get('error', '')})"))
    elif rep["net"]["hf_download_mb_s"] < 50:
        issues.append((1, f"network: HF download {rep['net']['hf_download_mb_s']} MB/s"))
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", help="comma-separated physical GPU indices (default: all)")
    ap.add_argument("--ref", help="a box_check.json from a known-good box to compare against")
    ap.add_argument("--out", default=str(ROOT / "runs" / "box_check.json"))
    ap.add_argument("--wait-torch", type=int, default=3600, help="seconds to wait for torch (setup.sh running)")
    a = ap.parse_args()

    t0 = time.time()
    while subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
                         capture_output=True).returncode != 0:
        if time.time() - t0 > a.wait_torch:
            sys.exit("torch with CUDA never became importable")
        print("waiting for torch (setup.sh still installing?) ...", flush=True)
        time.sleep(20)

    n_gpu = int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else list(range(n_gpu))
    busy = [g for g in gpus if (num(smi(g, ["memory.used"])["memory.used"]) or 0) > 1000]
    if busy:
        print(f"warning: GPU(s) {busy} already have >1 GB in use: their numbers will read low", flush=True)

    rep = {"host": os.uname().nodename, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpus": []}
    for g in gpus:
        print(f"GPU {g} ...", flush=True)
        p = subprocess.run([sys.executable, __file__, "--_worker", str(g)], capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}, timeout=600)
        try:
            rep["gpus"].append(json.loads(p.stdout.strip().splitlines()[-1]))
        except Exception:
            rep["gpus"].append({"gpu": g, "name": "?", "error": (p.stderr or p.stdout)[-300:]})
    print("CPU, disk, network ...", flush=True)
    rep["cpu"], rep["disk"], rep["net"] = cpu_check(), disk_check(ROOT), net_check()

    ref = json.loads(Path(a.ref).read_text()) if a.ref else None
    issues = judge(rep, ref)
    rep["issues"] = [{"level": "FAIL" if l == 2 else "WARN", "msg": m} for l, m in issues]
    level = max([l for l, _ in issues], default=0)
    rep["verdict"] = ["OK", "WARN", "FAIL"][level]

    cols = ["gpu", "name", "bf16_tflops", "tf32_tflops", "train_steps_s", "h2d_gb_s", "d2h_gb_s",
            "pcie_gen_load", "pcie_gen_max", "pcie_width_load", "power_w_load", "power_limit_w", "sm_clock_load", "temp_c_max"]
    print("\n" + "\t".join(cols))
    for g in rep["gpus"]:
        print("\t".join(str(g.get(c, "")) for c in cols))
    print(f"\nCPU {rep['cpu']}\ndisk {rep['disk']}\nnet {rep['net']}")
    for i in rep["issues"]:
        print(f"{i['level']}: {i['msg']}")
    print(f"\nverdict: {rep['verdict']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"report: {a.out}")
    sys.exit(level)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker":
        print(json.dumps(gpu_worker(int(sys.argv[2]))))
    else:
        main()
