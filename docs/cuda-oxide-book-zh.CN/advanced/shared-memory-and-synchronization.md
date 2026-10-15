# 共享内存与同步

CUDA 线程块中的每个线程都可以访问一块小而快的暂存区，称为**共享内存**。它位于片上，紧挨着 SM 的执行单元 —— 大约比全局内存快 100 倍，比 L1 缓存快约 10 倍。代价是容量：视架构不同，每个 SM 通常为 48–228 KB，且在该 SM 上运行的所有 block 之间共享。

cuda-oxide 通过 `SharedArray` 和 `DynamicSharedArray` 暴露共享内存，两者的设计目标是使用起来像 Rust 数组，同时编译到 PTX 地址空间 3。本章介绍如何使用它们、何时需要同步，以及能把朴素 kernel 变成快速 kernel 的分块模式。

> **参见**
>
> [CUDA Programming Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/#shared-memory)，其中有关于 bank 结构、广播规则以及各架构容量的硬件细节。

---

## 共享内存为什么重要

想想 [异步 MLP 流水线](../projects/async-mlp-pipeline.md)项目里的朴素 GEMM。每个线程通过从全局内存读取 A 的一整行和 B 的一整列，来计算一个输出元素。对 64×64 的矩阵来说，这就是*每个线程* 128 次全局加载，而块里的其他线程又反复加载其中许多相同的元素。硬件会尽职地从 DRAM —— 或者运气好的话，从 L2 缓存 —— 把这些元素一个个取回来。

共享内存改变了这笔经济账。一个线程块协作地把 A 的一个 **tile** 和 B 的一个 tile 加载到共享内存中，同步，然后每个线程都从这个 tile 里读取。每次全局加载被所有复用该元素的线程分摊。对于 16×16 的 tile，全局内存流量降低为原来的 1/16。

![分块计算模式。线程协作地把 A 和 B 的 tile 从全局内存加载到 `SharedArray` 中，同步，从快速的片上内存计算部分积，再次同步，然后沿 K 维度对下一个 tile 重复这一过程。](images/shared-memory-tiling.svg)

*分块计算模式。线程协作地把 A 和 B 的 tile 从全局内存加载到 `SharedArray` 中，同步，从快速的片上内存计算部分积，再次同步，然后沿 K 维度对下一个 tile 重复这一过程。*

---

## SharedArray —— 静态共享内存

`SharedArray<T, N, ALIGN>` 是一块在编译期分配于共享内存中的定长数组。在 kernel 内部把它声明为 `static mut`：

```rust
use cuda_device::thread::Runtime2DIndex;
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

const TILE: usize = 16;

#[kernel]
pub fn tiled_sgemm(
    m: u32, n: u32, k: u32,
    a: &[f32], b: &[f32],
    mut c: DisjointSlice<f32, Runtime2DIndex>,
) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

    let n_sz = n as usize;
    let row = thread::index_2d_row();
    let col = thread::index_2d_col();
    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;

    let mut sum = 0.0f32;
    let mut t = 0u32;

    while t < k / TILE as u32 {
        let tile_offset = t as usize * TILE;

        // 阶段 1：协作加载
        unsafe {
            TILE_A[ty * TILE + tx] = a[row * k as usize + tile_offset + tx];
            TILE_B[ty * TILE + tx] = b[(tile_offset + ty) * n_sz + col];
        }

        // 所有线程都必须完成加载，之后任何线程才能读取
        thread::sync_threads();

        // 阶段 2：从共享内存计算
        let mut i = 0usize;
        while i < TILE {
            unsafe {
                sum += TILE_A[ty * TILE + i] * TILE_B[i * TILE + tx];
            }
            i += 1;
        }

        // 在下一次迭代覆盖 tile 之前必须同步
        thread::sync_threads();

        t += 1;
    }

    // 行宽来自 `c`，在宿主侧被绑定为同一个 `n`。
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        if let Some(c_elem) = c.get_mut(c_idx) {
            *c_elem = sum;
        }
    }
}
```

### 声明规则

`SharedArray` 必须在 kernel 函数体内声明为 `static mut`。这会告诉编译器把它分配到 PTX 共享地址空间（`.shared`）。`UNINIT` 常量会跳过初始化 —— 在线程写入之前，其内容是不确定的。

| 参数 | 含义                                                                           |
| :-------- | :-------------------------------------------------------------------------------- |
| `T`       | 元素类型（`f32`、`u32` 等）                                                 |
| `N`       | 元素个数（编译期固定）                                        |
| `ALIGN`   | 字节对齐（默认 0 = 自然对齐；TMA 目标请用 128）      |

> **提示**
>
> `SharedArray` 在设计上就是 `!Sync` 的 —— 它包装了 `UnsafeCell`，从而阻止编译器假定它在多线程间不可变。这是正确的：共享内存本质上可被块中所有线程修改，同步由程序员负责。

### API 概貌

```rust
impl<T, const N: usize, const ALIGN: usize> SharedArray<T, N, ALIGN> {
    pub const UNINIT: Self;
    pub const fn len() -> usize;
    pub fn as_ptr(&self) -> *const T;
    pub fn as_mut_ptr(&mut self) -> *mut T;
}

// 索引（通过 static mut 进行不安全访问）
impl Index<usize> for SharedArray<T, N, ALIGN> { ... }
impl IndexMut<usize> for SharedArray<T, N, ALIGN> { ... }
```

索引在 debug 构建中会做边界检查。在 release 构建中（以及在 GPU 上），边界检查会被省略。如果你越界索引，就会得到未定义行为 —— 与 Rust 中任何 `static mut` 访问的规则相同。

---

## DynamicSharedArray —— 运行时确定大小的分配

有时你在编译期并不知道 tile 的大小，或者想在一个共享内存池上划分出多个逻辑数组。`DynamicSharedArray` 从动态共享内存区域分配，该区域的大小在启动时通过 `LaunchConfig::shared_mem_bytes` 设置：

```rust
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, thread};

#[kernel]
pub fn reduce_dynamic(input: &[f32], n: u32, mut output: DisjointSlice<f32>) {
    let tid = thread::threadIdx_x() as usize;

    // 获取指向动态共享内存区域的指针
    let smem: *mut f32 = DynamicSharedArray::<f32>::get();

    unsafe {
        // 从全局内存加载
        let idx = thread::index_1d();
        *smem.add(tid) = if idx.get() < n as usize {
            input[idx.get()]
        } else {
            0.0
        };
    }

    thread::sync_threads();

    // 树形归约
    let mut stride = thread::blockDim_x() as usize / 2;
    while stride > 0 {
        if tid < stride {
            unsafe {
                *smem.add(tid) += *smem.add(tid + stride);
            }
        }
        thread::sync_threads();
        stride /= 2;
    }

    if tid == 0 {
        let block_idx = thread::blockIdx_x() as usize;
        unsafe {
            *output.get_unchecked_mut(block_idx) = *smem;
        }
    }
}
```

这样启动：

```rust
let config = LaunchConfig {
    grid_dim: ((n + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
};
```

### 划分动态共享内存

如果你需要从同一个动态池中得到多个数组，请使用 `offset()`：

```rust
let pool_a: *mut f32 = DynamicSharedArray::<f32>::get();
let pool_b: *mut f32 = DynamicSharedArray::<f32>::offset(
    256 * std::mem::size_of::<f32>()
);
```

`offset` 接收的是相对于池起始位置的**字节偏移量**。请确保总量不超过 `shared_mem_bytes` —— 这里没有任何运行时保护。

### 对齐

`DynamicSharedArray` 默认采用 16 字节对齐。对于 TMA 操作（Hopper 及以上），请使用 `DynamicSharedArray<f32, 128>` 以获得所需的 128 字节对齐。对齐信息会编码到 PTX 的 `.align` 指令中。

---

## 同步：sync_threads()

`thread::sync_threads()` 是一个**块级屏障**。它编译为 PTX 的 `bar.sync 0`，并保证两件事：

1. **块中的所有线程**都到达屏障后，任何线程才会继续执行。
2. 那些线程的**所有内存写入**在屏障之后对所有线程可见（它对共享内存起到内存栅栏的作用）。

如果在加载阶段和计算阶段之间没有 `sync_threads()`，某些线程可能会在另一个线程写入之前就读到共享内存的某个位置。硬件并不保证 warp 内共享内存存储的任何顺序 —— 即使同一 warp 中的线程，在没有屏障的情况下也可能看到过期的值。

### 何时需要同步

规则很简单：**在读取别的线程写入的数据之前同步**。

| 情形                                                      | 需要同步吗？                                                               |
| :--------------------------------------------------------------| :------------------------------------------------------------------------|
| 线程 A 写 `TILE[i]`，线程 B 读 `TILE[i]`            | 需要                                                                      |
| 线程 A 写 `TILE[i]`，线程 A 读 `TILE[i]`            | 不需要（同一个线程）                                                         |
| 为下一轮循环覆盖某个 tile                 | 需要（在新一轮加载覆盖线程可能仍在读取的数据之前） |
| 从 `DisjointSlice` 读取（每个线程读自己的索引） | 不需要                                                                       |

上面的分块 GEMM 每次迭代有两个同步点：一个在加载之后（计算之前），一个在计算之后（下一次加载覆盖 tile 之前）。漏掉任何一个都会造成数据竞争。

> **提示**
>
> 一个常见的错误是把 `sync_threads()` 放进并非所有线程都会进入的条件分支里。块中的每个线程都必须执行到同一个 `sync_threads()` 调用，否则 kernel 会死锁。如果你需要分叉的控制流，请调整结构，让屏障位于分支之外。

---

## 共享内存与其他方案对比

| 方案                      | 时延        | 容量              | 程序员的工作量                |
| :---------------------------- | :------------- | :-------------------- | :------------------------------- |
| 全局内存（朴素）         | 约 500 周期    | GB 级                   | 无                             |
| L1/L2 缓存（隐式）        | 约 30–100 周期 | 128 KB–40 MB          | 无                             |
| **共享内存（显式）**  | **约 5 周期**  | **每个 SM 48–228 KB**  | **分块、同步、bank 感知** |
| 寄存器                     | 约 1 周期       | 每个 SM 64K × 32 位   | 由编译器管理                 |

当缓存不够用时，共享内存就是程序员手中的工具。L1 和 L2 缓存会自动帮忙，但它们受访问模式和淘汰策略的摆布。共享内存给你显式的控制权：加载什么、何时加载、保留多久，都由你决定。

---

## Bank 冲突

共享内存被划分为 32 个 **bank**（每个 warp lane 一个）。如果同一 warp 中的两个线程访问映射到同一个 bank 的不同地址，这些访问会被串行化 —— 这就是**bank 冲突**。代价是：2 路冲突带来 2 倍时延，32 路冲突最多可达 32 倍。

映射方式很直接：连续的 32 位字映射到连续的 bank。所以 `TILE[0]` 在 bank 0，`TILE[1]` 在 bank 1，……，`TILE[32]` 又回到 bank 0。一个常见的无冲突访问模式是：

```rust
// 每个线程读取不同的列：线程 k 读取 TILE[row + k]
// 如果 TILE_WIDTH = 32（或 32 的倍数），加入填充：SharedArray<f32, 33 * 16>
```

对于 16×16 的分块 GEMM，内层循环中没有 bank 冲突，因为 `TILE_A[ty * 16 + i]` 读取的是一行（连续元素 = 不同的 bank），而 `TILE_B[i * 16 + tx]` 读取的是一列，跨距为 16。

> **参见**
>
> [CUDA Programming Guide — Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-programming-guide/#shared-memory-5-x)，其中详细讨论了 bank 冲突规则与填充策略。

---

## 把它们串起来

下面是把这个演进过程从朴素版本应用到 MLP 流水线中 GEMM 的结果：

| 版本                   | 每线程全局加载次数（64×64） | 每线程共享内存加载次数 | 加速比  |
| :------------------------ | :------------------------------ | :---------------------- | :------- |
| 朴素（`sgemm_naive`）     | 128                             | 0                       | 1×       |
| 分块（16×16 tile）       | 8（4 次迭代 × 2 个 tile）      | 128                     | 约 4–10×   |
| 分块 + 双缓冲  | 8                               | 128（重叠）        | 约 6–15×   |

分块版本用共享内存带宽和计算换取全局内存带宽 —— 当 kernel 受限于内存带宽（memory-bound）时，这是一笔划算的交易。双缓冲变体把下一个 tile 的加载与当前 tile 的计算重叠起来，从而隐藏更多时延，但每个矩阵需要两个 `SharedArray`，同步也更复杂。

> **参见**
>
> - [Warp 级编程](warp-level-programming.md) —— 对于小规模归约，用基于 shuffle 的归约替代共享内存
> - [Tensor Memory Accelerator](tensor-memory-accelerator.md) —— 用硬件加速的全局→共享拷贝取代手写加载循环
