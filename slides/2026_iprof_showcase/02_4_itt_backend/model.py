import torch
from torch.profiler import itt

x = torch.randn(2048, 2048, device="xpu")
with torch.autograd.profiler.emit_itt():
    itt.range_push("my_matmul")
    y = torch.matmul(x, x)
    itt.range_pop()
torch.xpu.synchronize()
