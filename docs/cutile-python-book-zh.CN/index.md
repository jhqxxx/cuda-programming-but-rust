# cuTile Python

cuTile 是面向 NVIDIA GPU 的并行编程模型，同时也是一套基于 Python 的 DSL。
它会自动利用 tensor cores、tensor memory accelerator 等先进硬件能力，
并在不同的 NVIDIA GPU 架构之间保持可移植性 —— 无需改动代码即可用上最新的硬件特性。

cuTile **kernel** 是在由 **block** 组成的逻辑 **grid** 上并行执行的 GPU 程序。
`@ct.kernel` 装饰器把一个 Python 函数标记为 kernel 的入口点。
kernel **不能**由宿主代码直接调用；宿主必须用 `ct.launch()` 函数把 kernel 排入 GPU 执行队列：

**片段**

```python
import cuda.tile as ct
import cupy

TILE_SIZE = 16

# cuTile kernel：两个稠密向量相加，在 GPU 上并行执行。
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# 宿主侧函数：负责启动上面的 kernel。
def vector_add(a: cupy.ndarray, b: cupy.ndarray, result: cupy.ndarray):
    assert a.shape == b.shape == result.shape
    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
```

**完整示例**

```python
import cuda.tile as ct
import cupy

TILE_SIZE = 16

# cuTile kernel：两个稠密向量相加，在 GPU 上并行执行。
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# 宿主侧函数：负责启动上面的 kernel。
def vector_add(a: cupy.ndarray, b: cupy.ndarray, result: cupy.ndarray):
    assert a.shape == b.shape == result.shape
    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))

import numpy as np

rng = cupy.random.default_rng()
a = rng.random(128)
b = rng.random(128)
result = cupy.zeros_like(a)

vector_add(a, b, result)

a_np = cupy.asnumpy(a)
b_np = cupy.asnumpy(b)
result_np = cupy.asnumpy(result)

expected = a_np + b_np
np.testing.assert_array_almost_equal(result_np, expected)
```

kernel 用 `ct.load()`、`ct.store()` 这类函数在 **arrays** 和 **tiles** 之间搬运数据。
arrays 和 tiles 都是类张量（tensor-like）的数据结构：各自有 shape（每个轴上的元素个数）
和 dtype（元素的数据类型）。但两者之间有关键区别：

- **Arrays** 存放在 global memory 中。它们可变，并具有物理的、带 stride 的内存布局。
  在 kernel 代码内，它们只支持一小组操作，主要和向/从 tile 加载与存储数据有关。
  多种 Python 对象 —— 包括 PyTorch tensor 和 CuPy array —— 都可以作为 array，
  通过 kernel 参数从宿主代码传给 kernel。

- **Tiles** 是不可变的值，没有确定的存储位置，只存在于 kernel 代码中。
  tile 的各维长度必须是编译期常量，且为 2 的幂。
  tile 支持大量操作，包括逐元素算术、矩阵乘、归约、形状变换等。

安装说明与一个完整可运行的示例，见[快速开始](guide/01-quickstart.md)。
