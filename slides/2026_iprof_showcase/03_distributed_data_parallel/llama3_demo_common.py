#!/usr/bin/env python3
"""Shared TinyLlama workload used by the baseline and ITT demo entry points."""

import argparse
import math
import os
import socket
import time
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


TaskFactory = Callable[[str, str], ContextManager]


class Regions:
    """Return either real ITT tasks or no-op contexts for identical control flow."""

    def __init__(self, domain: str, task_factory: Optional[TaskFactory] = None):
        self.domain = domain
        self.task_factory = task_factory

    @property
    def enabled(self) -> bool:
        return self.task_factory is not None

    def task(self, name: str) -> ContextManager:
        if self.task_factory is None:
            return nullcontext()
        return self.task_factory(name, self.domain)


def get_mpi_env() -> tuple[int, int, int]:
    """Read rank information from common Open MPI, PMI, and PALS variables."""

    rank = (
        os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("PALS_RANKID")
    )
    world_size = (
        os.environ.get("OMPI_COMM_WORLD_SIZE")
        or os.environ.get("PMI_SIZE")
        or os.environ.get("PALS_WORLD_SIZE")
    )
    local_size = (
        os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
        or os.environ.get("PMI_LOCAL_SIZE")
        or os.environ.get("PALS_LOCAL_SIZE")
    )
    return (
        int(rank) if rank is not None else 0,
        int(world_size) if world_size is not None else 1,
        int(local_size) if local_size is not None else 1,
    )


def get_local_rank(rank: int, local_size: int) -> int:
    local_rank = (
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MPI_LOCALRANKID")
        or os.environ.get("PMI_LOCAL_RANK")
        or os.environ.get("PALS_LOCAL_RANKID")
    )
    return int(local_rank) if local_rank is not None else rank % max(1, local_size)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def rotate_half(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


class RoPECache(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.einsum("t,f->tf", positions, inv_freq)
        embedding = torch.cat([frequencies, frequencies], dim=-1)
        self.register_buffer("cos", embedding.cos(), persistent=False)
        self.register_buffer("sin", embedding.sin(), persistent=False)

    def forward(self, sequence_length: int):
        return self.cos[:sequence_length], self.sin[:sequence_length]


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: RoPECache):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("model dimension must be divisible by the number of heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = rope
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch, sequence_length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(sequence_length)
        q, k = apply_rope(q, k, cos, sin)
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        attention = F.softmax(attention.masked_fill(causal_mask, float("-inf")), dim=-1)
        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch, sequence_length, channels)
        return self.projection(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_hidden: int,
        rope: RoPECache,
        layer_index: int,
        regions: Regions,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.regions = regions
        self.norm1 = RMSNorm(dim)
        self.attention = CausalSelfAttention(dim, n_heads, rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def forward(self, x):
        with self.regions.task(f"Layer.{self.layer_index}"):
            with self.regions.task(f"Attn.{self.layer_index}"):
                x = x + self.attention(self.norm1(x))
            with self.regions.task(f"MLP.{self.layer_index}"):
                x = x + self.mlp(self.norm2(x))
        return x


class TinyLlama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        regions: Regions,
        use_checkpoint: bool,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rope = RoPECache(dim // n_heads, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    mlp_hidden=4 * dim,
                    rope=self.rope,
                    layer_index=layer_index,
                    regions=regions,
                )
                for layer_index in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.use_checkpoint = use_checkpoint

    def forward(self, token_ids):
        hidden = self.token_embedding(token_ids)
        for block in self.blocks:
            if self.use_checkpoint:
                # Recomputed block forwards appear inside Backward in this mode.
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def select_device(device_choice: str, local_rank: int) -> torch.device:
    if device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda is unavailable")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if device_choice == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("--device xpu requested, but torch.xpu is unavailable")
        torch.xpu.set_device(local_rank)
        return torch.device("xpu", local_rank)
    return torch.device("cpu")


def initialize_process_group(
    rank: int, world_size: int, backend: str, device: torch.device
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    kwargs = {"backend": backend, "rank": rank, "world_size": world_size}
    if device.type != "cpu":
        kwargs["device_id"] = device
    dist.init_process_group(**kwargs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    if device.type == "cpu":
        dist.barrier()
    else:
        dist.barrier(device_ids=[device.index])


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cuda")
    parser.add_argument(
        "--backend",
        default=None,
        help="distributed backend (defaults: cpu=gloo, cuda=nccl, xpu=xccl)",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="checkpoint transformer blocks (demonstrates recompute during Backward)",
    )
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true", help="use bfloat16 autocast")
    precision.add_argument("--fp32", action="store_true", help="disable autocast")
    parser.add_argument("--itt-domain", default="LlamaDemo", help="ITT domain name")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "--steps": args.steps,
        "--batch-size": args.batch_size,
        "--seq-len": args.seq_len,
        "--vocab-size": args.vocab_size,
        "--dim": args.dim,
        "--layers": args.layers,
        "--heads": args.heads,
        "--lr": args.lr,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.dim % args.heads != 0:
        parser.error("--dim must be divisible by --heads")
    if (args.dim // args.heads) % 2 != 0:
        parser.error("the per-head dimension must be even for rotary embeddings")


def run(task_factory: Optional[TaskFactory] = None, description: str = __doc__) -> None:
    """Run the shared workload with real ITT tasks or no-op regions."""

    parser = build_parser(description)
    args = parser.parse_args()
    validate_args(parser, args)
    regions = Regions(args.itt_domain, task_factory)

    with regions.task("Init.MPI"):
        rank, world_size, local_size = get_mpi_env()
        local_rank = get_local_rank(rank, local_size)

    device = select_device(args.device, local_rank)
    default_backends = {"cpu": "gloo", "cuda": "nccl", "xpu": "xccl"}
    backend = args.backend or default_backends[device.type]

    with regions.task("Init.ProcessGroup"):
        initialize_process_group(rank, world_size, backend, device)

    if rank == 0:
        mode = "ITT" if regions.enabled else "baseline"
        print(
            f"[rank0] mode={mode} host={socket.gethostname()} world_size={world_size} "
            f"local_size={local_size} backend={backend} device={device}"
        )
        print(
            f"[rank0] MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')}"
        )
        print(f"[rank0] args={vars(args)}")

    with regions.task("Init.Model"):
        torch.manual_seed(1234 + rank)
        model = TinyLlama(
            vocab_size=args.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            n_heads=args.heads,
            max_seq_len=args.seq_len,
            regions=regions,
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
        with regions.task("Data.Generate"):
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

    synchronize(device)
    start_time = time.time()
    model.train()

    for step in range(args.steps):
        with regions.task(f"Step.{step}"):
            inputs, targets = get_batch()
            optimizer.zero_grad(set_to_none=True)

            with regions.task("Forward"):
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

            with regions.task("Backward"):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with regions.task("Optimizer.Step"):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        if step % 5 == 0:
            with regions.task("Metrics.AllReduce"):
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
    if rank == 0:
        print("done")
    dist.destroy_process_group()
