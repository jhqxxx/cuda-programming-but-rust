# 集群编程

**线程块集群**是一组线程块，它们被保证同时运行在同一个 GPC（Graphics Processing Cluster）内的 SM 上。集群在 Hopper（SM 90）中引入，它带来了标准 CUDA 所没有的能力：**跨块共享内存访问**，而无需经过全局内存。

在普通 CUDA 中，线程块彼此独立。块 0 无法读取块 1 的共享内存。有了集群，硬件会把集群中所有成员的共享内存映射到一个统一的**分布式共享内存（DSMEM）**地址空间。集群中的任何线程都可以读取任何块的共享内存 —— 直接读取，以共享内存的时延，没有全局内存往返。

cuda-oxide 通过 `cuda_device::cluster` 和 `#[cluster_launch]` 属性暴露集群。本章介绍编程模型、DSMEM 访问模式，以及集群如何与 TMA 多播结合来构建高性能矩阵 kernel。

> **参见**
>
> [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/#thread-block-clusters)，其中有硬件规格、GPC 约束以及最大集群规模。

---

## 集群模型

![一个 4 块集群。每个块有自己的共享内存，但硬件把四者映射到统一的 DSMEM 空间。在 `cluster_sync()` 屏障之后，各块可以读取彼此的共享内存。虚线箭头表示跨块 DSMEM 读取。](images/cluster-dsmem-topology.svg)

*一个 4 块集群。每个块有自己的共享内存，但硬件把四者映射到统一的 DSMEM 空间。在 `cluster_sync()` 屏障之后，各块可以读取彼此的共享内存。虚线箭头表示跨块 DSMEM 读取。*

集群在启动时定义。硬件把集群中的所有块调度到同一个 GPC 内的 SM 上，以保证低时延的跨块通信。最大集群规模取决于架构 —— Hopper 支持每个集群最多 8 个块。

### 声明集群 kernel

在 cuda-oxide 中，你用 `#[cluster_launch]` 标注 kernel：

```rust
use cuda_device::{SharedArray, cluster, cluster_launch, kernel, thread};

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(/* ... */) {
    // 这个 kernel 以 X 维度上 4 个块为一组集群运行
    let rank = cluster::block_rank();
    let size = cluster::cluster_size();
    // ...
}
```

`#[cluster_launch(x, y, z)]` 属性：
1. 注入一个 `__cluster_config<X, Y, Z>()` 函数，用于设置 PTX 的
   `.reqnctapercluster` 指令。
2. 在宿主侧，启动必须使用带匹配 `cluster_dim` 参数的 `launch_kernel_ex`。

---

## 集群标识

每个线程都可以查询自己在集群层次中的位置：

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();       // 本集群内的 0..cluster_size()-1
let size = cluster::cluster_size();     // 集群中块的总数
let cidx = cluster::cluster_idx();      // 该集群在 grid 中的编号
let ncls = cluster::num_clusters();     // grid 中集群的总数
```

`block_rank()` 是 DSMEM 访问的关键标识 —— 它告诉你该读哪个块的共享内存。可以把它看作集群内的"块级设备 ID"。

要获得完整的三维集群位置：

```rust
let cx = cluster::cluster_ctaidX();     // 块在集群中的 X 位置
let cy = cluster::cluster_ctaidY();     // 块在集群中的 Y 位置
let cz = cluster::cluster_ctaidZ();     // 块在集群中的 Z 位置
```

---

## 分布式共享内存（DSMEM）

集群的核心特性是跨块共享内存访问。读取远程块的共享内存有两种方式：

### 方法 1：map_shared_rank + 普通解引用（任意类型）

`map_shared_rank` 接收一个本地共享内存指针，返回一个指向另一个块共享内存中相同偏移处的真实指针。它位于集群共享地址空间中，所以直接解引用即可：

```rust
use cuda_device::{SharedArray, cluster, thread};

static mut DATA: SharedArray<u32, 256> = SharedArray::UNINIT;

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn halo_exchange(/* ... */) {
    let tid = thread::threadIdx_x() as usize;
    let rank = cluster::block_rank();

    // 每个块写入自己的共享内存
    unsafe { DATA[tid] = compute_value(rank, tid); }

    thread::sync_threads();    // 先做本地屏障
    cluster::cluster_sync();   // 再做集群级屏障

    // 从下一个块的共享内存读取
    let neighbor = (rank + 1) % cluster::cluster_size();
    let remote_ptr = unsafe {
        cluster::map_shared_rank(DATA.as_ptr().add(tid), neighbor)
    };
    let neighbor_val = unsafe { *remote_ptr };
}
```

最后一行是普通的 Rust 读取。编译器会让映射后的指针保持位于集群共享地址空间，因此这次解引用会编译成 `ld.shared::cluster`。写入同理：用 `map_shared_rank_mut` 映射，通过指针赋值，存储会编译成 `st.shared::cluster`。这种方式适用于任意类型，并且需要 sm_90 及以上。

### 方法 2：dsmem_read_u32（固定宽度的 u32 读取）

`dsmem_read_u32` 在一次调用中完成映射并读取一个 `u32`。它接收*本地*指针加上目标 rank，并自己完成 rank 映射：

```rust
let neighbor_val = unsafe {
    cluster::dsmem_read_u32(
        DATA.as_ptr() as *const u32,
        neighbor,  // 目标 rank
    )
};
```

两种方法发出的是同一条 `ld.shared::cluster` PTX 指令。通过 `map_shared_rank` 的普通解引用是主要方法：它适用于任意类型并支持写入。当你只需要一个 `u32`、并希望把映射和加载合并为一次调用时，可以用 `dsmem_read_u32`。

---

## 同步

集群在块级同步与全局同步之间引入了一个新的同步层级：

| 原语                        | 作用域   | 何时使用                                          |
| :------------------------------- | :------ | :------------------------------------------------ |
| `thread::sync_threads()`         | 块   | 在单个块内同步               |
| `cluster::cluster_sync()`        | 集群 | 在集群的所有块之间同步    |
| `mbarrier_arrive_cluster(addr)`  | 集群 | 向远程块中的屏障发信号             |

访问 DSMEM 的正确同步顺序永远是：

1. **写入**本地共享内存
2. **`sync_threads()`** —— 确保所有本地线程都已写入
3. **`cluster_sync()`** —— 确保所有块都已到达这一点
4. 通过 DSMEM **读取**远程共享内存

漏掉任何一次同步都会造成数据竞争。只有 `sync_threads()` 而没有
`cluster_sync()` 意味着你的块准备好了，但邻居可能还没有。只有
`cluster_sync()` 而没有 `sync_threads()` 意味着整个集群已同步，但你所在块自己的写入可能还不可见。

---

## 集群与 TMA 多播

集群解锁了 TMA 最强大的特性之一：**多播拷贝**。一次 TMA 加载可以同时把同一个 tile 写入多个块的共享内存：

```rust
use cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast;

// CTA 掩码：第 0..3 位置位 → 集群中全部 4 个块都收到该 tile
let cta_mask: u16 = 0b1111;

unsafe {
    cp_async_bulk_tensor_2d_g2s_multicast(
        smem_dst, desc, tile_x, tile_y, bar_ptr, cta_mask
    );
}
```

没有多播时，每个块都要发出自己的 TMA 拷贝 —— 四次独立的全局内存读取。有了多播，TMA 引擎只读取一次数据，再把它分发给全部四个块。当 GEMM kernel 中集群的每个块都需要 A 或 B 的同一个 tile 时，这一点尤其有价值。

`cta_mask` 是一个位掩码，当 rank *i* 需要接收该拷贝时，第 *i* 位置位。你可以有选择地只向集群的一个子集多播。

---

## 一个实际示例：halo 交换

集群的一个常见用途是 stencil 计算中的 **halo 交换**。每个块处理网格的一个 tile，并需要来自邻居的边界元素。没有集群时，这需要全局内存的写入和读取（或者小心地做流同步）。有了集群，它就变成了本地操作：

```rust
use cuda_device::{DisjointSlice, SharedArray, cluster, cluster_launch, kernel, thread};

const TILE_W: usize = 256;
const HALO: usize = 1;
const SMEM_W: usize = TILE_W + 2 * HALO;

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn stencil_1d(input: &[f32], mut output: DisjointSlice<f32>, n: u32) {
    static mut SMEM: SharedArray<f32, { SMEM_W }> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let rank = cluster::block_rank();
    let global_idx = rank as usize * TILE_W + tid;

    // 加载内部区域（为 halo 槽位偏移 HALO）
    unsafe {
        if global_idx < n as usize {
            SMEM[tid + HALO] = input[global_idx];
        }
    }

    thread::sync_threads();
    cluster::cluster_sync();

    // 从上一个块加载左侧 halo
    if tid == 0 && rank > 0 {
        let prev_rank = rank - 1;
        unsafe {
            SMEM[0] = f32::from_bits(cluster::dsmem_read_u32(
                SMEM.as_ptr().add(TILE_W) as *const u32, prev_rank
            ));
        }
    }

    // 从下一个块加载右侧 halo
    if tid == 0 && rank < cluster::cluster_size() - 1 {
        let next_rank = rank + 1;
        unsafe {
            SMEM[TILE_W + HALO] = f32::from_bits(cluster::dsmem_read_u32(
                SMEM.as_ptr().add(HALO) as *const u32, next_rank
            ));
        }
    }

    thread::sync_threads();

    // 三点 stencil：output[i] = 0.25 * left + 0.5 * center + 0.25 * right
    if global_idx < n as usize {
        let left = unsafe { SMEM[tid + HALO - 1] };
        let center = unsafe { SMEM[tid + HALO] };
        let right = unsafe { SMEM[tid + HALO + 1] };

        unsafe {
            *output.get_unchecked_mut(global_idx) = 0.25 * left + 0.5 * center + 0.25 * right;
        }
    }
}
```

没有集群时，halo 交换需要把边界元素写入全局内存、通过事件或流同步，再把它们读回来。有了集群，它只需一次 `cluster_sync()` 加一次 `dsmem_read_u32` —— 共享内存的时延，没有全局内存。

---

## 约束与最佳实践

| 约束                     | 细节                                                               |
| :----------------------------- | :------------------------------------------------------------------- |
| 最大集群规模           | 8 个块（视架构而定）                                    |
| 调度保证           | 集群中的所有块同时运行在同一个 GPC 上                |
| DSMEM 时延                  | 与本地共享内存相近（约 5–10 周期）                        |
| 必须声明集群维度   | `#[cluster_launch(x, y, z)]` + 启动配置中的宿主 `cluster_dim`   |
| 块数必须可整除  | 每个维度上的 grid 块数必须是集群规模的倍数     |
| 混用集群/非集群      | 同一个 kernel 中不支持                                     |

> **提示**
>
> 集群会带来调度约束：硬件必须把所有块放在同一个 GPC 上。如果集群相对于 GPC 容量过大，占用率就会下降。从较小的集群（2–4 个块）开始，测量之后再考虑扩大。

> **参见**
>
> - [Tensor Memory Accelerator](tensor-memory-accelerator.md) —— TMA 多播需要集群才能完成跨 CTA 投递
> - [矩阵乘法加速器](matrix-multiply-accelerators.md) —— CG2 模式使用集群对来获得更宽的 MMA tile
> - [共享内存与同步](shared-memory-and-synchronization.md) —— DSMEM 所扩展的逐块基础
