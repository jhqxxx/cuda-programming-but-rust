# API 快速参考

本附录是 cuda-oxide 设备端与宿主端 API 的精简参考。完整文档请在工作区根目录
运行 `cargo doc --no-deps --open`。

---

## 属性与宏

### Kernel 与设备属性

```rust
use cuda_device::{cluster_launch, cooperative_launch, device, kernel, launch_bounds};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { /* ... */ }

#[kernel]
pub fn unrolled(mut data: DisjointSlice<f32>) {
    let mut i = 0;
    #[unroll(4)]
    while i < 16 { /* ... */ i += 1; }
}

#[kernel]
#[launch_bounds(256, 2)]
pub fn tuned_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cooperative_launch]
pub fn grid_sync_kernel(data: &mut [f32]) { /* ... */ }

#[device]
fn helper(x: f32) -> f32 { x * x }
```

| Attribute                                   | Purpose                                                             |
|:--------------------------------------------|:--------------------------------------------------------------------|
| `#[cuda_module]`                            | 把一个模块的 kernel 收集为带类型的宿主模块，带有 `load` 与各启动器 |
| `#[kernel]`                                 | 把函数标记为 GPU kernel 入口点（PTX 中的 `.entry`）       |
| `#[device]`                                 | 把辅助函数或 `extern "C"` 块标记为参与设备编译 |
| `#[unroll]` / `#[unroll(N)]`               | 请求完全展开，或按因子 `N >= 2` 展开            |
| `#[launch_bounds(max_threads, min_blocks)]` | 用于寄存器分配的占用率提示                             |
| `#[constant]`                               | 把一个 `ConstantMemory<T>` 静态量放到常量内存中，并提供宿主端 `set_<name>` |
| `#[launch_contract(...)]`                   | 声明 kernel 所需的启动形状，从而解锁安全的（非 `unsafe`）启动 |
| `#[cluster_launch(x, y, z)]`                | 设置编译期 cluster 维度（Hopper+）                       |
| `#[cooperative_launch]`                     | 通过 `#[cuda_module]` 以协作方式启动（启用 `grid::sync()`）  |
| `#[convergent]`                             | 标记为 convergent（barrier 语义）                              |
| `#[pure]`                                   | 标记为无副作用                                            |
| `#[readonly]`                               | 标记为只读                                                   |

请仅在 `#[kernel]` 或 `#[device]` 函数内的显式计数 `while` 循环上使用这些注解。
基于 range 的 `for` 循环目前尚不能被展开 pass 识别。支持嵌套循环与多条
`continue` 路径。完全的 `#[unroll]` 会保留 `break` 路径与多个退出目标。

部分展开 `#[unroll(N)]` 要求正步长、`<` 或 `<=` 判断、不变的上限，并且除了
正常的头部判断外没有其他出口。其他请求会告警并且不展开。

单个注解最多可产生 1,024 份循环体拷贝、8,192 个克隆基本块和 65,536 个克隆
操作。更大的请求会告警并且不展开。

### 调试与 PTX 宏

```rust
use cuda_device::{gpu_assert, gpu_printf, ptx_asm};

gpu_printf!("thread %d: val = %f\n", idx as i32, val as f64);
gpu_assert!(val.is_finite());
gpu_assert!(val >= 0.0, "expected non-negative value");

let y: u32;
unsafe {
    ptx_asm!("add.u32 %0, %1, %1;", out("=r") y, in("r") x, options(register_only));
}
```

| Macro                               | Purpose                                                           |
|:------------------------------------|:------------------------------------------------------------------|
| `gpu_printf!(fmt, args...)`         | 设备侧格式化输出（下沉为 `vprintf`）                |
| `gpu_assert!(condition)`            | 运行时断言；失败时调用 `trap()`                      |
| `gpu_assert!(condition, "message")` | 带 CUDA 诊断信息的运行时断言；消息必须是字面量 |
| `ptx_asm!(...)`                     | unsafe 的 CUDA 内联 PTX                                            |

消息形式会下沉为 CUDA 的设备侧 `__assertfail` 系统调用。驱动会上报消息与调用点
元数据，而同步操作会返回 `CUDA_ERROR_ASSERT`。

---

## 编译期 policy 配置

```rust
use cuda_device::config::{
    Atom, AtomKind, AtomSpec, Block, Cluster, ColumnMajor, Global, Layout, MemorySpace, Policy,
    PolicyId, Register, RowMajor, Scope, Shape, Shape1, Shape2, Shape3, Shared, TensorMemory,
    Thread, Tile, TileSpec, Warp, WarpGroup,
};
```

policy 使用零大小的 Rust 类型来描述编译期的 kernel 配置。一个泛型 kernel 会为
每一种具体的 policy 类型单态化一次；policy 不作为运行时 kernel 实参传递。

```rust
trait VectorPolicy: Policy {
    type BlockTile: TileSpec;
    type ElementAtom: AtomSpec;

    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum SmallTilePolicy {}

impl Policy for SmallTilePolicy {
    const ID: PolicyId =
        PolicyId::new(0x706f_6c69_6379_5f63, 1);
}
```

| API                  | Description                                                                             |
| :------------------- |:----------------------------------------------------------------------------------------|
| `Policy`             | 具名编译期 kernel policy 的最小基础 trait                               |
| `PolicyId`           | 显式的稳定身份，包含项目专属命名空间与 policy 局部值 |
| `Shape`              | 暴露静态形状的秩、各维长度与经检查的元素数量的 trait                |
| `Shape1<D0>`         | 一维编译期形状                                                      |
| `Shape2<D0, D1>`     | 二维编译期形状                                                      |
| `Shape3<D0, D1, D2>` | 三维编译期形状                                                      |
| `Tile<S, L, M, Q>`   | 仅含元数据的描述，组合形状、布局、内存空间与执行作用域    |
| `TileSpec`           | 对 tile 描述各组成部分的类型级访问                               |
| `Atom<K, S, Q>`      | 仅含元数据的描述，涵盖操作、逻辑足迹与参与线程 |
| `AtomKind`           | 标识特定领域操作的开放标记 trait                               |
| `AtomSpec`           | 对 atom 的操作种类、形状与作用域的类型级访问                         |
| `Layout`             | 内存顺序元数据的开放 trait                                                    |
| `RowMajor`           | 最右坐标连续的布局                                         |
| `ColumnMajor`        | 最左坐标连续的布局                                          |
| `MemorySpace`        | 描述 CUDA 存储位置的开放 trait                                           |
| `Global`             | 设备全局内存标记                                                             |
| `Shared`             | 每 block 共享内存标记                                                          |
| `Register`           | 线程本地寄存器存储标记                                                    |
| `TensorMemory`       | 硬件 tensor memory 标记                                                           |
| `Scope`              | 描述在某个操作上协作的线程的开放 trait                           |
| `Thread`             | 单线程执行作用域                                                           |
| `Warp`               | 单 warp 执行作用域                                                           |
| `WarpGroup`          | 硬件 warpgroup 执行作用域                                                      |
| `Block`              | 线程块执行作用域                                                           |
| `Cluster`            | 线程块 cluster 执行作用域                                                    |

`Tile` 与 `Atom` 仅仅是描述。它们不分配存储、不提供指针访问、不发射 GPU 指令、
不同步线程，也不建立任何安全性质。特定领域的 policy trait 才赋予这些描述符
意义，并校验受支持的组合。

`PolicyId` 的值由 policy 库显式提供。它们不是从 Rust 的 `TypeId`、类型名、
编译器 mangle 或哈希推导出来的。当 policy 生成的行为保持不变时，请让该 ID 保持
稳定；当行为改变时，请分配一个新的值。

policy 关联的常量目前可用于编译期的 `launch_bounds` 与部分循环 `unroll`
表达式：

```rust
use cuda_device::{kernel, launch_bounds};

#[kernel]
#[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
pub unsafe fn transform<P: VectorPolicy>(
    input: *const u32,
    output: *mut u32,
    count: u32,
) {
    let mut lane = 0;

    #[unroll(P::UNROLL)]
    while lane < count {
        // ...
        lane += 1;
    }
}
```

泛型 policy 表达式需要 `#![feature(generic_const_exprs)]`。
`launch_contract` 的字段、cluster 维度以及动态共享内存大小目前仍必须是字面量。

关于两个具体 policy 生成独立 PTX 特化以及 policy 专属预备启动的示例，见
[`policy_config`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/policy_config)
示例。

---

## 线程标识

```rust
use cuda_device::thread;

let idx     = thread::index_1d();                            // ThreadIndex<'_, Index1D>
let idx2d   = thread::index_2d::<128>();                     // Option<ThreadIndex<'_, Index2D<128>>>
let idx2d_r = thread::index_2d_runtime(&out);                // Option<ThreadIndex<'_, Runtime2DIndex>>
//            `out` 是该索引将要寻址的 DisjointSlice<T, Runtime2DIndex>；
//            它的行宽由宿主通过 cuda_host::RowWidth 绑定一次。
let idx32   = thread::index_1d_u32(launch_context);          // ThreadIndex32<'_>
let pos32   = thread::coord_2d_u32(launch_context);          // ThreadCoord2D32<'_>

let tid_x  = thread::threadIdx_x();    // u32
let bid_x  = thread::blockIdx_x();     // u32
let bdim_x = thread::blockDim_x();     // u32
```

| Function                                    | Returns                                          | Description                                                |
|:--------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------|
| `thread::index_1d()`                        | `ThreadIndex<'_, Index1D>`                       | 唯一的线性索引（1D grid）                             |
| `thread::index_2d::<S>()`                   | `Option<ThreadIndex<'_, Index2D<S>>>`            | const stride 的 2D 索引；stride 不匹配是类型错误 |
| `thread::index_2d_runtime(&slice)`          | `Option<ThreadIndex<'_, Runtime2DIndex>>`        | 运行时确定宽度的 2D 索引；行宽从切片读取      |
| `thread::index_1d_u32(launch_context)`      | `ThreadIndex32<'_>`                              | 以 `u32` 表示的 1-D 索引；要求坐标已按 `u32` 检查     |
| `thread::coord_2d_u32(launch_context)`      | `ThreadCoord2D32<'_>`                            | 以 `u32` 表示的 2-D 行/列；要求坐标已按 `u32` 检查|
| `thread::index_2d_row()`                    | `usize`                                          | 2D 行索引                                               |
| `thread::index_2d_col()`                    | `usize`                                          | 2D 列索引                                            |
| `thread::threadIdx_{x,y,z}()`               | `u32`                                            | block 内的线程索引                                  |
| `thread::blockIdx_{x,y,z}()`                | `u32`                                            | grid 内的 block 索引                                    |
| `thread::blockDim_{x,y,z}()`                | `u32`                                            | block 维度                                           |

当计算出的列超出 stride 时，`thread::index_2d::<S>()` 与
`thread::index_2d_runtime(&slice)` 会返回 `None`，从而在非对齐的 2D kernel 中
跳过右边缘的尾巴。

`index_2d::<S>` 是安全的 const stride 形式；const 泛型把 stride 编码进见证类型，
因此线程无法使用不同的 stride。`index_1d` 要求 Y/Z 维度未启用，2D 索引要求
Z 维度未启用。匹配的 `PreparedLaunch<K>` 无需设备端检查即可证明这一点。否则，
设备会拒绝错误的秩：`index_1d` 会创建无效见证，2D 辅助函数返回 `None`。裸启动
仍然是 unsafe 的，因为它其他的内存与启动义务未被检查。`index_2d_runtime` 覆盖
那些 stride 只在运行时才已知的启动：行宽随切片一起传递，由宿主一次性写入启动
数据包（`cuda_host::RowWidth`），因此既没有 `unsafe`，也不存在每次调用可能被
线程们各执一词的 stride。见证存储该线程的 `(row, col)` 坐标，被寻址的切片会
根据它自己的行宽解析这些坐标。完整讨论见[安全模型](../gpu-safety/the-safety-model.md)。

---

## 安全并行写入 —— DisjointSlice

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

| Method                  | Signature                                        | Description                                                          |
|:------------------------|:-------------------------------------------------|:---------------------------------------------------------------------|
| `get_mut_indexed`       | `() -> Option<(&mut T, ThreadIndex<'_, IS>)>`    | 一次调用形式：铸造见证并解析它。Index1D / Index2D。 |
| `get_mut`               | `(ThreadIndex<'_, IS>) -> Option<&mut T>`        | 由显式见证进行的带边界检查可变访问               |
| `get_unchecked_mut`     | `(usize) -> &mut T`                              | unsafe、不检查的访问                                             |
| `len`                   | `() -> usize`                                    | 元素个数                                                   |

`get_mut_indexed` 受 `IndexSpace: IndexFormula` 约束（`Index1D` 与 `Index2D<S>`
实现了它）。对于 `Runtime2DIndex` 切片，请使用显式的
`thread::index_2d_runtime(&slice)` + `get_mut(idx)` 组合；切片会针对它自己的、
由宿主绑定的行宽（`slice.row_width()`）解析见证的坐标。

对于固定大小的 tile，请使用 `DisjointSlice<T, LinearTiles<N>>::tile_thread32`
或 `DisjointSlice<T, RowMajorTiles<R, C, S>>::tile_2d32`。每个方法都会对完整的
tile 检查一次，随后 `at_const` 访问已知位置时无需再做运行时边界检查。`S` 是
调用方声明的逻辑行宽，必须与缓冲区布局一致。见
[检查一次 tile](../gpu-safety/the-safety-model.md#check-a-tile-once)。

---

## 共享内存

```rust
use cuda_device::{DynamicSharedArray, SharedArray, thread};

#[kernel]
pub fn tiled(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    unsafe { TILE[tid] = data[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... 从 TILE 读取 ...
}

#[kernel]
pub fn dynamic(data: &[f32]) {
    static mut BUF: DynamicSharedArray<f32> = DynamicSharedArray::UNINIT;
    // 大小在启动时通过 LaunchConfig::shared_mem_bytes 设置
}
```

| Type                      | Description                                               |
|:--------------------------|:----------------------------------------------------------|
| `SharedArray<T, N>`       | 编译期确定大小、作用域为 block 的共享内存            |
| `SharedArray<T, N, 128>`  | 带 128 字节对齐（TMA 目标所必需）   |
| `DynamicSharedArray<T>`   | 运行时确定大小的共享内存（通过 `LaunchConfig` 设置）      |

两者都是 `!Sync` —— 并发访问需要显式的 barrier。

---

## 同步

### Block 级

```rust
thread::sync_threads();   // 等价于 __syncthreads()
```

### 受管理的 Barrier（Hopper+）

```rust
use cuda_device::{ManagedBarrier, Ready, TmaBarrierHandle, Uninit};

// typestate 生命周期：Uninit → Ready → Invalidated
let bar: TmaBarrierHandle<Uninit> = TmaBarrierHandle::from_static(ptr);
let bar: TmaBarrierHandle<Ready> = unsafe { bar.init(thread_count) };
let token = bar.arrive();
bar.wait(token);
unsafe { bar.inval() };
```

| Operation                   | Description                                      |
|:----------------------------|:-------------------------------------------------|
| `.init(count)`              | 以期望的到达次数初始化 barrier   |
| `.arrive()`                 | 发出到达信号，返回 `BarrierToken`           |
| `.arrive_expect_tx(bytes)`  | 到达并设置期望的 TX 字节数（用于 TMA）  |
| `.wait(token)`              | 阻塞直到所有到达 + TX 完成           |
| `.inval()`                  | 使 barrier 失效（清理）                     |

---

## Warp 原语

```rust
use cuda_device::warp;

let lane = warp::lane_id();      // 0–31
let wid  = warp::warp_id();

// Shuffle
let partner = warp::shuffle_xor_f32(val, mask);
let from_above = warp::shuffle_down_f32(val, delta);
let from_below = warp::shuffle_up_f32(val, delta);
let from_lane  = warp::shuffle_f32(val, src_lane);

// u32 是无后缀形式
let partner_u = warp::shuffle_xor(val_u32, mask);

// Vote
let all_true = warp::all(predicate);
let any_true = warp::any(predicate);
let mask     = warp::ballot(predicate);
let count    = warp::popc(predicate); // == ballot(predicate).count_ones()
```

### Shuffle 操作

无后缀名接受 `u32`；`_f32`、`_f64` 与 `_u64` 是其他位宽。每个还有一个接受显式
成员掩码的 `_sync` 形式。没有 `_i32` 变体 —— 请把 `i32` 重新解释后使用 `u32`
形式，因为 shuffle 只是搬运比特，并不解释它们。

| Function                                    | Description                       |
|:--------------------------------------------|:----------------------------------|
| `shuffle_xor(val, mask)` + `_f32/_f64/_u64` | 与 lane `id ^ mask` 交换    |
| `shuffle_down(val, delta)` + `_f32/_f64/_u64` | 从 lane `id + delta` 读取     |
| `shuffle_up(val, delta)` + `_f32/_f64/_u64` | 从 lane `id - delta` 读取       |
| `shuffle(val, src)` + `_f32/_f64/_u64`      | 从指定 lane 读取           |

### Vote 操作

| Function       | Returns  | Description                                  |
|:---------------|:---------|:---------------------------------------------|
| `all(pred)`    | `bool`   | 若谓词对所有 lane 成立则为真        |
| `any(pred)`    | `bool`   | 若谓词对任一 lane 成立则为真         |
| `ballot(pred)` | `u32`    | 谓词为真的 lane 的位掩码     |
| `popc(pred)`   | `u32`    | 谓词为真的 lane 计数       |

---

## 原子操作

### 带作用域的 GPU 原子操作

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);

// 在 kernel 中：
COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
let old = COUNTER.load(AtomicOrdering::Acquire);
```

| Scope                                        | Types                         |
|:---------------------------------------------|:------------------------------|
| `DeviceAtomic{U32,I32,U64,I64,F16,F32,F64}`  | `.gpu` 作用域                  |
| `BlockAtomic{U32,I32,U64,I64,F16,F32,F64}`   | `.cta` 作用域                  |
| `SystemAtomic{U32,I32,U64,I64,F16,F32,F64}`  | `.sys` 作用域（CPU-GPU 共享） |

共二十一种类型：三种作用域各含七种值宽度。`F16` 变体提供与 `F32`/`F64` 相同的
接口 —— `load`、`store`、`fetch_add`、`fetch_sub`、`swap` —— 其中
`fetch_add`/`fetch_sub` 下沉为硬件 `atom.add.noftz.f16`。

`core::sync::atomic` 类型（`AtomicU32`、`AtomicBool` 等）同样会编译为 GPU
代码，默认使用 system 作用域。

在 nvptx64 上，`core::sync::atomic::AtomicPtr<T>` 对泛型地址空间的指针值支持
`load`、`store`、`swap` 与 `compare_exchange`，并在下沉过程中保留指针类型。
本地原子存储使用普通的线程私有访问。全局与共享存储保留原子 ordering 和作用域，
即使某个辅助函数接收到的是泛型指针也是如此。遗留 NVVM 对指针交换与比较交换
使用带作用域的 PTX，同时保留指针类型的操作数与结果。

---

## TMA —— Tensor Memory Accelerator（Hopper+）

```rust
use cuda_device::tma::TmaDescriptor;
use cuda_device::tma::{cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_g2s};

// 宿主：构建描述符（128 字节，不透明）
// 设备：发起异步批量拷贝
cp_async_bulk_tensor_2d_g2s(smem_ptr, &desc, coord_x, coord_y, barrier_ptr);
cp_async_bulk_commit_group();
```

| Function                                      | Description                          |
|:----------------------------------------------|:-------------------------------------|
| `cp_async_bulk_tensor_{1..5}d_g2s(...)`       | 全局 → 共享的异步批量拷贝      |
| `cp_async_bulk_tensor_{1..5}d_s2g(...)`       | 共享 → 全局的异步批量拷贝      |
| `cp_async_bulk_tensor_2d_g2s_multicast(...)`  | 多播到 cluster 中所有 CTA     |
| `cp_async_bulk_commit_group()`                | 提交尚未完成的拷贝            |
| `cp_async_bulk_wait_group(n)`                 | 等待直到剩余组数 ≤ n         |

---

## Cluster 编程（Hopper+）

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();        // 本 block 在 cluster 中的 rank
let size = cluster::cluster_size();      // cluster 中的 block 数量
cluster::cluster_sync();                 // 跨所有 cluster block 的 barrier

// 分布式共享内存。两者都是 `unsafe fn`，并且都接受*本地*
// 指针加目标 rank；它们自己完成 rank 映射，因此
// 绝不要把 `map_shared_rank` 已返回的指针再传回去。

// `map_shared_rank` 返回一个指向目标 block 共享内存的真实指针
// （cluster 共享地址空间）。通过它进行的普通读写
// 会编译为 ld.shared::cluster / st.shared::cluster，并在 sm_90+ 上直接可用。
let remote_ptr = unsafe { cluster::map_shared_rank(local_ptr, target_rank) };
let val = unsafe { *remote_ptr };  // ld.shared::cluster

// 固定宽度替代方案：在一次调用中映射并读取一个 u32。
let val = unsafe { cluster::dsmem_read_u32(local_u32_ptr, target_rank) };
```

---

## Tensor Core —— WGMMA（Hopper，SM 90）

```rust
use cuda_device::wgmma;

wgmma::wgmma_fence();
wgmma::wgmma_commit_group();
wgmma::wgmma_wait_group::<0>();
```

Warpgroup MMA：4 个 warp（128 个线程）从共享内存发起矩阵乘加。操作数由 SMEM
描述符描述；累加器位于寄存器中。

---

## Tensor Core —— tcgen05（Blackwell，SM 100+）

```rust
use cuda_device::SharedArray;
use cuda_device::tcgen05::{TmemGuard, TmemReady, TmemUninit};

static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

let guard = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
let guard = unsafe { guard.alloc() };   // TmemUninit → TmemReady
// ... 发射 MMA，通过 guard.address() 读取结果 ...
let _guard = unsafe { guard.dealloc() }; // TmemReady → TmemDeallocated
```

单线程发射 MMA 到专用的 Tensor Memory（TMEM）。`TmemGuard` 以 typestate 管理
TMEM 的生命周期：`TmemUninit → TmemReady → TmemDeallocated`。
N_COLS 必须是 [32, 512] 范围内的 2 的幂。

---

## 宿主侧：Kernel 启动

### 带类型的同步

```rust
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();
let module = kernels::load(&ctx).unwrap();

let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
let mut output = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

// SAFETY: 这是一次 1D 启动，且所有 buffer 都包含 n 个元素。
unsafe {
    module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut output)
}
.unwrap();
```

### 带类型的异步

```rust
use cuda_async::simt::device_operation::DeviceOperation;

let module = kernels::load_async(0)?;
// SAFETY: 这是一次 1D 启动，buffer 包含 n 个元素，且 module/scheduler 共享同一个上下文。
let op = unsafe {
    module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut output)
}?;

op.sync()?;       // 阻塞
// 或者：op.await?;  // 使用 tokio 的异步
```

生成的裸调用是 unsafe 的，因为 `LaunchConfig` 并未与某个 kernel 绑定。带有
`#[launch_contract]` 的 kernel 则改用 `LaunchConfig1D/2D/3D` 创建一个经过检查的
`PreparedLaunch<K>`，然后安全地启动。`cuda_launch!` 与 `cuda_launch_async!`
仍是 unsafe 的低层 API，用于显式模块加载与自定义启动代码。

### LaunchConfig

| Method                                                   | Description                                  |
|:---------------------------------------------------------|:---------------------------------------------|
| `LaunchConfig::for_num_elems(n)`                         | 为 `n` 个元素自动配置 grid/block   |
| `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }` | 直接构造结构体                   |

---

## 宿主侧：虚拟内存管理

### VMM 生命周期

| API                                         | Purpose                                     |
|:--------------------------------------------|:--------------------------------------------|
| `vmm::allocation_granularity(device)`       | 查询所需的分配粒度   |
| `vmm::align_size(size, granularity)`        | 把大小向上取整到所需粒度    |
| `PhysicalAllocation::new(device, size)`     | 分配物理内存                    |
| `VirtualReservation::new(size, alignment)`  | 预留一段虚拟地址范围             |
| `Mapping::new(va, size, &physical, offset)` | 把物理内存映射到某个 VA 范围         |
| `vmm::set_access(va, size, devices)`        | 授予选定设备读/写访问权限 |

映射必须在其虚拟预留与物理分配之前被 drop。

---

## 宿主侧：对等访问

| API                                   | Purpose                                           |
|:--------------------------------------|:--------------------------------------------------|
| `peer::can_access_peer(from, to)`     | 查询拓扑是否支持直接访问 |
| `peer::enable_peer_access(from, to)`  | 启用单向对等访问                |
| `peer::disable_peer_access(from, to)` | 禁用单向对等访问               |

对等访问是有方向的。当两个设备都需要发起访问时，请启用两个方向。

---

## 宿主侧：Kernel 家族

| Type | Purpose |
|:-----|:--------|
| `KernelFamily<Id, Entry, Meta, N>` | 一组固定的、提前编译好的变体 |
| `KernelVariant<Id, Entry, Meta>` | 稳定的 ID、可调用的入口与 policy 元数据 |
| `KernelProblem<Variant>` | 校验某个变体是否合格 |
| `KernelSelector<Problem, Variant, Id>` | 在已经合格的变体中进行选择 |
| `KernelSelectionCache<Problem, Id>` | 存储稳定的选择 ID |
| `NoKernelSelectionCache` | 禁用缓存 |
| `SelectionMode::Auto` | 使用经过校验的缓存结果，或调用 selector |
| `SelectionMode::Force(id)` | 绕过缓存与 selector，但仍校验合格性 |
| `SelectedVariant` | 返回被选中的变体及其来源 |
| `SelectionSource` | 报告来源是 override、缓存还是 selector |

`KernelFamily::try_new` 会拒绝空的家族、空白家族名以及重复的变体 ID。家族名与
revision 共同构成缓存命名空间。只要变体的语义、成员、顺序或选择策略发生变化，
就请递增 revision。

完整的选择模型与示例见 [Kernel 家族](../gpu-programming/kernel-families.md)。

---

## 调试设施

```rust
use cuda_device::debug;

let t = debug::clock64();       // 周期计数器
debug::trap();                  // 中止 kernel
debug::breakpoint();            // cuda-gdb 断点
unsafe { cuda_device::barrier::nanosleep(1000) }; // 休眠约 1μs
debug::prof_trigger::<7>();     // Nsight profiler 触发器
```

---

## 快速参考表

### cuda-device 模块

| Module               | Description                                                      | Min SM   |
|:---------------------|:-----------------------------------------------------------------|:---------|
| `thread`             | 线程/block ID、`index_1d`、`sync_threads`                     | All      |
| `config`             | 编译期 policy、shape、tile、atom、layout、内存空间与作用域 | All |
| `disjoint`           | `DisjointSlice<T>` —— 由启动证明完成带类型的写入    | All      |
| `shared`             | `SharedArray<T, N>`、`DynamicSharedArray<T>`                     | All      |
| `warp`               | Shuffle、vote、match、lane/warp ID                               | All      |
| `atomic`             | 带作用域的原子操作（device/block/system）                             | sm_70+   |
| `debug`              | `clock64`、`trap`、`breakpoint`、`gpu_printf!`                   | All      |
| `fence`              | `threadfence_block` / `threadfence` / `threadfence_system`       | All      |
| `grid`               | grid 作用域的 `sync`（协作式 kernel 启动）                 | sm_70+   |
| `cooperative_groups` | 带类型的句柄、warp/block 归约与前缀和                   | All      |
| `barrier`            | `ManagedBarrier` —— 用于 TMA/MMA 的异步 mbarrier                    | sm_90+   |
| `cluster`            | 线程块 cluster、DSMEM                                     | sm_90+   |
| `tma`                | `TmaDescriptor`、批量 tensor 拷贝（1D–5D）                      | sm_90+   |
| `wgmma`              | Warpgroup MMA（fence/commit/wait）                                | sm_90    |
| `tcgen05`            | 第五代 tensor core、TMEM、`TmemGuard`                          | sm_100+  |
| `cusimd`             | `CuSimd<T, N>`、`Float2`/`Float4`                                | All      |
| `clc`                | Cluster Launch Control                                           | sm_100+  |

### Crate 映射

| Crate             | Role                                                                   |
|:------------------|:-----------------------------------------------------------------------|
| `cuda-device`     | 设备 intrinsics 与类型（`#![no_std]`）                             |
| `cuda-macros`     | proc macro（`#[kernel]`、`#[device]`、`gpu_printf!`、`ptx_asm!`）      |
| `cuda-host`       | 带类型的模块加载与低层启动辅助                     |
| `cuda-core`       | 针对上下文、流、buffer、VMM 与 P2P 的安全 RAII 封装        |
| `cuda-async`      | `DeviceOperation`、`DeviceFuture`、`DeviceBox<T>`                      |
| `cuda-bindings`   | 对 `cuda.h` 的裸 `bindgen` FFI                                          |
| `cargo-oxide`     | cargo 子命令（`cargo oxide run`、`build`、`inspect`、`clean`、`sanitize`、`debug`） |
