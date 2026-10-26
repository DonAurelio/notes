import torch
import time
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size",   type=int, required=True)
    p.add_argument("--iters",  type=int, required=True)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
