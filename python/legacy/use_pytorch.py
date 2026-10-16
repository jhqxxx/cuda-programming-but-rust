from pathlib import Path

import torch
from torch.utils.cpp_extension import load
import numpy as np

# .cu 的路径基于 __file__，而不是当前工作目录 ——
# 否则必须 cd 到本目录才能跑（load() 只在 cwd 里找相对路径）。
lib = load(name="vecadd",
           sources=[str(Path(__file__).parent / "vecadd.cu")],
           extra_cflags=["-std=c++20"],)

a = torch.randn(1024).cuda().float().contiguous()
b = torch.randn(1024).cuda().float().contiguous()
c = torch.zeros_like(a).cuda().float().contiguous()

lib.vecadd(a, b, c, 1024)

expected = (a + b).detach().cpu().numpy()
c = c.detach().cpu().numpy()

np.testing.assert_array_almost_equal(c, expected)

print("Test succeeded")
