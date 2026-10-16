# 实用的心智模型

cuTile Rust 的 kernel 是在 tile 上运作的。编译器与运行时负责把这些 tile 映射到 CUDA 线程和内存上。

## 基于 tile 的编程

CUDA C++ 的 kernel 通常从单个线程和显式的线程索引出发：

```cpp
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

而 cuTile Rust 从数据中 tile 形状的区域出发：

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    c: &mut Tensor<f32, S>,
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
) {
    let tile_a = a.load_like(c);
    let tile_b = b.load_like(c);
    c.store(tile_a + tile_b);
}
```

kernel 只描述**一个** tile 形状区域上发生了什么。编译器与运行时负责把这部分工作映射到 CUDA 执行单元上。

## tile 程序与 tile block

**tile 程序**是并发执行的逻辑单位。每个 tile 程序把入口函数运行一次，并处理输出中的一块区域。它可以查询自己在各轴上的网格坐标与网格范围：

```rust
let pid0 = program_id(0);
let n0 = num_programs(0);
```

tile 程序、tile block、tile 线程这几个词指的是同一个逻辑单位。API 中使用 `program_id(axis)` 和 `num_programs(axis)`；元组形式的 `get_tile_block_id()` 和 `get_num_tile_blocks()` 会一次性返回三个轴的值。

能放上可用流式多处理器（SM）的 tile block 会同时运行。全部 tile block 在逻辑上是并发的：它们之间的相对顺序是未指定的，除非 kernel 之外存在显式的同步机制，否则 kernel 不能依赖某个 tile block 先于另一个运行。

## 网格几何

对 tensor 做分区会生成一个逻辑网格。一个 `[128, 256]` 的 tensor 按 `[32, 64]` 分区，会生成 `(4, 4, 1)` 个 tile block：

```text
Tensor shape:    [128, 256]
Partition shape: [ 32,  64]
Grid:            (ceil(128 / 32), ceil(256 / 64), 1)
```

tensor 的第 0 维映射到网格的 `x`，第 1 维映射到 `y`，第 2 维映射到 `z`。对于秩更低的 tensor，末尾的网格维度取 1。

在 kernel 内部，program id 用来选出对应的逻辑区域：

```rust
let pid_m = program_id(0);
let part_x = x.partition(shape![BM, BK]);
let tile_x = part_x.load([pid_m, k_tile]);
```

对于可变输出，被选中的子 tensor 会直接交给该 tile block。

## 内存层级

NVIDIA GPU 有多级内存：

| 内存层级 | 作用 |
|---|---|
| 寄存器 | 最快的存储；计算期间 tile 就住在这里 |
| 共享内存 | 硬件 block 内共享的快速片上存储 |
| L2 缓存 | 跨 SM 共享的、由硬件管理的缓存 |
| HBM | 存放 tensor 的大容量全局内存 |

在 cuTile Rust 中，你从 HBM 里的 tensor 加载数据，并在寄存器里的 tile 上计算。当共享内存、缓存、线程、warp、Tensor Core 以及张量内存加速器（TMA）指令这些机制有用时，Tile IR 编译器与运行时会决定如何经由它们来中转数据。

```rust
let tile = x.load_like(z); // HBM -> tile。
let y = tile * 2.0f32;           // 驻留在寄存器中的计算。
z.store(y);                      // Tile -> HBM。
```

你仍然需要选择 tile 形状和访存模式。编译器负责处理线程级的映射。

## 合并访存与跨步

当相邻线程访问相邻地址时，GPU 搬运内存的效率最高。tile 加载在设计上就会为规整布局产生合并的访存模式，因此行主序的连续 tile 通常是很好的起点。

跨步访存会降低有效带宽，因为内存请求变得分散了：

```text
Good: memory[0], memory[1], memory[2], ...
Bad:  memory[0], memory[1024], memory[2048], ...
```

性能不佳时，检查一下算法的逻辑访问模式是否破坏了连续的 tile 加载。

## 并发与并行

**并行**意味着工作真正在不同的硬件单元上同时运行。**并发**意味着相互独立的工作同时处于进行中，并且可能以任意顺序被调度。一次 kernel 启动两者兼具：一部分 tile block 在可用的 SM 上并行运行，而所有 tile block 从程序员视角看都是并发的。

宿主侧的 `DeviceOp` 组合用的是同一套思路。相互独立的操作可以在不同的流上重叠。有依赖的操作则必须串联，或者放到同一条流上以强制排序。

## 从 CUDA 或 Triton 迁移过来

常见的 CUDA 与 Triton 概念在 cuTile Rust 里对应如下：

| 底层关注点 | cuTile Rust 中的表达方式 |
|---|---|
| 线程索引计算 | 已分区的 tensor 与 tile block id |
| 共享内存中转 | tile 加载与编译器托管的中转 |
| Tensor Core 的放置 | `mma` / `mmaf_scaled` |
| kernel 流水线组合 | `DeviceOp` 组合子与 CUDA 图 |
| 外部 kernel | 通过 `AsyncKernelLaunch` 做 PTX/CUBIN 互操作 |

当某个算法需要手写的 CUDA kernel，或者需要来自其它编译器的 PTX/CUBIN 产物时，请参见[互操作](interoperability.md)。
