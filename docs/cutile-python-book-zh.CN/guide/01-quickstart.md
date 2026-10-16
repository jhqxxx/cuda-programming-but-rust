# 快速开始

本页带你完成 cuTile Python 的环境准备，并跑通第一个示例。

## 环境要求

cuTile Python 需要：

- Linux x86_64、Linux aarch64 或 Windows x86_64
- 计算能力（compute capability）为 8.x、9.x、10.x、11.x 或 12.x 的 GPU
- NVIDIA 驱动 **r580 或更高**
- Python 3.10、3.11、3.12、3.13、3.14 或 3.14t

## 安装 cuTile Python

cuTile Python 依赖 CUDA TileIR 编译器 `tileiras`，
而 `tileiras` 又依赖 CUDA Toolkit 里的 `ptxas` 和 `libnvvm`。
这些依赖既可以按虚拟环境装上对应的 Python 包，也可以使用系统级 CTK 里的那份。

如果你的系统没有全局安装 CUDA Toolkit（13.1+），
可以给 cuTile Python 带上可选的 `[tileiras]`：
它会把 `nvidia-cuda-tileiras`、`nvidia-cuda-nvcc` 和 `nvidia-nvvm`
装进你的 Python 虚拟环境。

```bash
pip install --upgrade cuda-tile[tileiras]
```

注意：`nvidia-cuda-tileiras`、`nvidia-cuda-nvcc` 和 `nvidia-nvvm` 三者的版本
**major.minor 必须一致**。

cuTile Python 支持多个 `tileiras` 版本。若要用某个指定版本，例如 13.3：

```bash
pip install cuda-toolkit[tileiras,nvvm,nvcc]>=13.3
```

另一种情况：如果你已经全局装了 CUDA Toolkit（13.1+），
那么可以把 cuTile Python 当独立包装 —— 它会自动从 CUDA Toolkit 的位置去找 `tileiras`。

```bash
pip install cuda-tile
```

## 其他包

cuTile Python 的部分示例还会用到别的 Python 包。

本页的快速开始示例用到 cupy：

```bash
pip install cupy-cuda13x
```

`samples/` 目录下的示例还用到 pytest、torch 和 numpy。

PyTorch 的安装说明见 <https://pytorch.org/get-started/locally/>。
pytest 和 numpy 可以这样装：

```bash
pip install pytest numpy
```

## 示例代码

下面这个例子做的是向量加法 —— CUDA 里最典型的第一个 kernel —— 只不过这里用 cuTile
做基于 tile 的编程：用一个 1 维 tile 把两个 1 维向量相加。

这个例子展示了 cuTile kernel 的通用结构：

- 从 GPU 显存加载一个或多个 tile
- 对 tile 做运算，得到新的 tile
- 把结果 tile 写回 GPU 显存

具体到这里，kernel 从向量 `a` 和 `b` 各加载一个 tile，
分别叫 `a_tile` 和 `b_tile`；两者相加得到第三个 tile `result`；
最后把 `result` tile 存回输出向量 `c`。
更多示例见 cuTile Python 仓库。

```python
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
演示最朴素的向量加法，展示如何对向量做逐元素运算。
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # 取 1 维的 pid
    pid = ct.bid(0)

    # 加载输入 tile
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # 逐元素相加
    result = a_tile + b_tile

    # 存回结果
    ct.store(c, index=(pid, ), tile=result)

def test():
    # 准备输入数据
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    rng = cp.random.default_rng()
    a = rng.random(vector_size)
    b = rng.random(vector_size)
    c = cp.zeros_like(a)

    # 启动 kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,  # 1 维的处理器网格
              vector_add,
              (a, b, c, tile_size))

    # 拷回主机仅仅是为了比对
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # 校验结果
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")

if __name__ == "__main__":
    test()
```

在命令行里这样运行。如果一切配置正确，测试会打印示例通过：

```bash
$ python3 samples/quickstart/VectorAdd_quickstart.py
✓ vector_add_example passed!
```

想跑更多 cuTile Python 示例，按同样方式直接调用 sample 文件即可：

```bash
$ python3 samples/FFT.py
# 输出略
```

也可以用 pytest 跑全部示例：

```bash
$ pytest samples
========================= test session starts =========================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0
rootdir: /home/ascudiero/sw/cutile-python
configfile: pytest.ini
collected 6 items

samples/test_samples.py ...... [100%]

========================= 6 passed in 30.74s ==========================
```

## 开发者工具

NVIDIA Nsight Compute 可以像分析 SIMT CUDA kernel 那样分析 cuTile Python kernel。
装好 Nsight Compute 之后，用下面的命令给本页的向量加法 kernel 生成 profile：

```bash
ncu -o VecAddProfile --set detailed python3 VectorAdd_quickstart.py
```

然后把这份 profile 加载到 Nsight Compute 的图形界面里，选中 `vector_add` kernel，
就能看到它的各项统计信息。

> **注意**
>
> 采集 cuTile Python kernel 的详细统计信息，要求驱动不低于
> r580.126.09（Linux）或 r582.16（Windows）。
