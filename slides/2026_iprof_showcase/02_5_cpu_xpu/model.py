import torch

xc = torch.randn(2048, 2048, device="cpu")
yc = torch.matmul(xc, xc)

xg = torch.randn(2048, 2048, device="xpu")
yg = torch.matmul(xg, xg)
torch.xpu.synchronize()
