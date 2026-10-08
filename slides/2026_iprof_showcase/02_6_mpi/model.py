import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

x = torch.randn(2048, 2048, device="xpu")
y = torch.matmul(x, x)
torch.xpu.synchronize()

local_sum = y.sum().item()
total_sum = comm.allreduce(local_sum, op=MPI.SUM)
print(f"rank {rank}: local_sum={local_sum:.4f} total_sum={total_sum:.4f}")
