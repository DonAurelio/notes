#!/usr/bin/env python3
"""Kineto narrow scope: profile() opens just before the step loop, after DDP init and synchronize()."""

import os
import socket
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent / "workloads"))
from llama3_demo_common import (
    TinyLlama,
    build_parser,
    get_local_rank,
    get_mpi_env,
    initialize_process_group,
    select_device,
    synchronize,
    validate_args,
)


def run() -> None:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    validate_args(parser, args)

    rank, world_size, local_size = get_mpi_env()
    local_rank = get_local_rank(rank, local_size)
    device = select_device(args.device, local_rank)
    default_backends = {"cpu": "gloo", "cuda": "nccl", "xpu": "xccl"}
    backend = args.backend or default_backends[device.type]

    initialize_process_group(rank, world_size, backend, device)

    if rank == 0:
        print(
            f"[rank0] mode=kineto-narrow host={socket.gethostname()} world_size={world_size} "
            f"local_size={local_size} backend={backend} device={device}"
        )
        print(f"[rank0] args={vars(args)}")

    torch.manual_seed(1234 + rank)
    model = TinyLlama(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_len=args.seq_len,
        use_checkpoint=args.checkpoint,
    ).to(device)
    ddp_options = {}
    if device.type != "cpu":
        ddp_options.update(device_ids=[local_rank], output_device=local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, broadcast_buffers=False, **ddp_options
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    if args.fp32 or (device.type == "cpu" and not args.bf16):
        use_autocast = False
        autocast_dtype = torch.float32
    else:
        use_autocast = True
        autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    use_scaler = use_autocast and autocast_dtype == torch.float16
    scaler = (
        torch.amp.GradScaler(device=device.type, enabled=True) if use_scaler else None
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + rank)

    def get_batch():
        inputs = torch.randint(
            0,
            args.vocab_size,
            (args.batch_size, args.seq_len),
            generator=generator,
            device=device,
            dtype=torch.long,
        )
        targets = torch.roll(inputs, shifts=-1, dims=1)
        return inputs, targets

    activities = [ProfilerActivity.CPU]
    if device.type in ("cuda", "xpu"):
        activities.append(
            ProfilerActivity.CUDA if device.type == "cuda" else ProfilerActivity.XPU
        )

    # Narrow scope: synchronize all ranks, then open profiler just before the step loop
    synchronize(device)
    model.train()

    with profile(activities=activities) as prof:
        start_time = time.time()
        for step in range(args.steps):
            inputs, targets = get_batch()
            optimizer.zero_grad(set_to_none=True)

            autocast = (
                torch.amp.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=True
                )
                if use_autocast
                else nullcontext()
            )
            with autocast:
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if step % 5 == 0:
                reduced_loss = loss.detach().clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                average_loss = reduced_loss / world_size
                if rank == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"step {step:04d} loss={average_loss.item():.4f} "
                        f"elapsed={elapsed:.2f}s"
                    )

        synchronize(device)
        wall_s = time.time() - start_time

    if rank == 0:
        print(f"wall_s={wall_s:.6f}")

        t_a0 = time.time()
        avgs = prof.key_averages()
        t_a1 = time.time()

        n_events = sum(e.count for e in avgs)
        analysis_s = t_a1 - t_a0

        print(f"n_events={n_events}")
        print(f"analysis_s={analysis_s:.6f}")

        print("=== key_averages breakdown ===")
        for e in sorted(avgs, key=lambda x: x.count, reverse=True):
            print(f"  count={e.count:6d}  key={e.key}")
        print("=== end breakdown ===")

        print("done")

    dist.destroy_process_group()


if __name__ == "__main__":
    run()
