from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset_lab.bootstrap import setup
setup()

import argparse
import math
import multiprocessing as mp
import subprocess
import time


def cpu_worker() -> None:
    value = 0.1
    while True:
        for _ in range(200_000):
            value = math.sin(value) ** 2 + 0.000001


def cpu_load(percent: int) -> None:
    cores = max(1, mp.cpu_count())
    count = max(1, round(cores * min(percent, 75) / 100))
    processes = [mp.Process(target=cpu_worker) for _ in range(count)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()


def gpu_temperature() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(output.splitlines()[0].strip())
    except Exception:
        return None


def gpu_load(percent: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable for GPU load injector")
    duty = min(max(percent / 100.0, 0.05), 0.50)
    left = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    right = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    last_temperature_check = 0.0
    while True:
        now = time.monotonic()
        if now - last_temperature_check >= 2.0:
            temperature = gpu_temperature()
            last_temperature_check = now
            if temperature is not None and temperature >= 82:
                time.sleep(2.0)
                continue
        started = time.perf_counter()
        for _ in range(4):
            left = torch.mm(left, right)
        torch.cuda.synchronize()
        active = time.perf_counter() - started
        time.sleep(active * (1.0 / duty - 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["cpu", "gpu"])
    parser.add_argument("percent", type=int)
    args = parser.parse_args()
    if args.kind == "cpu":
        cpu_load(args.percent)
    else:
        gpu_load(args.percent)


if __name__ == "__main__":
    mp.freeze_support()
    main()
