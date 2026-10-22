import torch
import time
import argparse
from torch.profiler import profile, ProfilerActivity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size",   type=int, required=True)
    p.add_argument("--iters",  type=int, required=True)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        x = torch.randn(args.size, args.size, device="xpu")

        for _ in range(args.warmup):
            torch.matmul(x, x)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            torch.matmul(x, x)
        torch.xpu.synchronize()
        t1 = time.perf_counter()

    print(f"wall_s={t1 - t0:.6f}")

    t_a0 = time.perf_counter()
    avgs = prof.key_averages()
    t_a1 = time.perf_counter()

    print(f"n_events={sum(e.count for e in avgs)}")
    print(f"analysis_s={t_a1 - t_a0:.6f}")


if __name__ == "__main__":
    main()
