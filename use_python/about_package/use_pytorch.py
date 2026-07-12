import torch
from torch.utils.cpp_extension import load
import numpy as np

lib = load(name="vecadd",
           sources=["vecadd.cu"],
           extra_cflags=["-std=c++20"],)

a = torch.randn(1024).cuda().float().contiguous()
b = torch.randn(1024).cuda().float().contiguous()
c = torch.zeros_like(a).cuda().float().contiguous()

lib.vecadd(a, b, c, 1024)

expected = (a + b).detach().cpu().numpy()
c = c.detach().cpu().numpy()

np.testing.assert_array_almost_equal(c, expected)

print("Test succeeded")
