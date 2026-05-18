# API 快速参考

本附录是 cuda-oxide 设备端和主机端 API 的浓缩参考。如需完整文档，请在工作区根目录运行 `cargo doc --no-deps --open`。

---

## 属性和宏

### kernel和设备属性

```rust
use cuda_device::{kernel, device, launch_bounds, cluster_launch};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { /* ... */ }

#[kernel]
#[launch_bounds(256, 2)]
pub fn tuned_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(data: &mut [f32]) { /* ... */ }

#[device]
fn helper(x: f32) -> f32 { x * x }
```

| 属性     | 用途     |
|:------------|:----------|
| `#[kernel]` | 将函数标记为 GPU kernel入口点（PTX 中的 `.entry`）   |
| `#[device]` | 将辅助函数或 `extern "C"` 块标记为设备编译    |
| `#[launch_bounds(max_threads, min_blocks)]` | 占用率提示，用于寄存器分配      |
| `#[cluster_launch(x, y, z)]`  | 设置编译时簇维度（Hopper+）    |
| `#[convergent]`  | 标记为 convergent（屏障语义）   |
| `#[pure]`   | 标记为无副作用      |
| `#[readonly]`  | 标记为只读  |

### 输出宏

```rust
use cuda_device::{gpu_printf, gpu_assert};

gpu_printf!("thread %d: val = %f\n", idx as i32, val as f64);
gpu_assert!(val >= 0.0);
```

| 宏   | 用途  |
|:-----|:------|
| `gpu_printf!(fmt, args...)`  | 设备端格式化输出（降级为 `vprintf`）              |
| `gpu_assert!(condition)`     | 运行时断言；失败时调用 `trap()`                   |

---

## 线程标识

```rust
use cuda_device::thread;

let idx     = thread::index_1d();                            // ThreadIndex<'_, Index1D>
let idx2d   = thread::index_2d::<128>();                     // Option<ThreadIndex<'_, Index2D<128>>>
let idx2d_r = unsafe { thread::index_2d_runtime(stride) };   // Option<ThreadIndex<'_, Runtime2DIndex>>

let tid_x  = thread::threadIdx_x();    // u32
let bid_x  = thread::blockIdx_x();     // u32
let bdim_x = thread::blockDim_x();     // u32
```

| 函数  | 返回值   | 描述     |
|:-----|:---------|:-------|
| `thread::index_1d()`    | `ThreadIndex<'_, Index1D>`   | 唯一线性索引（1D 网格）    |
| `thread::index_2d::<S>()`   | `Option<ThreadIndex<'_, Index2D<S>>>`    | 常量步长 2D 索引；步长不匹配是类型错误   |
| `unsafe thread::index_2d_runtime(s)`   | `Option<ThreadIndex<'_, Runtime2DIndex>>` | 运行时步长 2D 索引；调用者需保证 `s` 统一     |
| `thread::index_2d_row()` | `usize`  | 2D 行索引  |
| `thread::index_2d_col()` | `usize` | 2D 列索引  |
| `thread::threadIdx_{x,y,z}()`  | `u32`  | 块内线程索引  |
| `thread::blockIdx_{x,y,z}()` | `u32` | 网格内块索引    |
| `thread::blockDim_{x,y,z}()`  | `u32`   | 块维度    |

`thread::index_2d::<S>()` 和 `thread::index_2d_runtime(s)` 当计算出的列超过步长时返回 `None` —— 用于在非对齐的 2D kernel中跳过右侧边缘尾部。

`index_2d::<S>` 是安全的默认选择；常量泛型将步长编码在见证类型中，因此两个线程无法通过传递不同步长而产生冲突的索引。`index_2d_runtime` 是用于仅在运行时知道步长的启动的逃生舱口；调用者通过编写 `unsafe` 承担“每个线程使用相同步长”的义务。完整讨论见[安全模型](../gpu-safety/the-safety-model.md)。

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

| 方法    | 签名   | 描述    |
|:----|:-----|:---------|
| `get_mut_indexed`  | `() -> Option<(&mut T, ThreadIndex<'_, IS>)>`     | 一步调用：生成见证并解析。适用于 Index1D / Index2D。          |
| `get_mut`    | `(ThreadIndex<'_, IS>) -> Option<&mut T>`   | 通过显式见证进行边界检查的可变访问     |
| `get_unchecked_mut`   | `(usize) -> &mut T` | 不安全，无检查   |
| `len`  | `() -> usize`     | 元素数量   |

`get_mut_indexed` 受限于 `IndexSpace: IndexFormula`（由 `Index1D` 和 `Index2D<S>` 实现）。对于 `Runtime2DIndex` 切片，请使用显式的 `unsafe { thread::index_2d_runtime(s) }` + `get_mut(idx)` 对。

---

## 共享内存

```rust
use cuda_device::{SharedArray, DynamicSharedArray, thread};

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

| 类型  | 描述       |
|:------|:--------------|
| `SharedArray<T, N>`  | 编译时大小，块作用域共享内存     |
| `SharedArray<T, N, 128>`  | 128 字节对齐（TMA 目标所需）    |
| `DynamicSharedArray<T>`   | 运行时大小共享内存（通过 `LaunchConfig` 设置）   |

两者都是 `!Sync` —— 并发访问需要显式屏障。

---

## 同步

### 块级同步

```rust
thread::sync_threads();   // 等价于 __syncthreads()
```

### 托管屏障（Hopper+）

```rust
use cuda_device::{ManagedBarrier, TmaBarrierHandle, Uninit, Ready};

// 类型状态生命周期：Uninit → Ready → Invalidated
let bar: TmaBarrierHandle<Uninit> = TmaBarrierHandle::from_static(ptr);
let bar: TmaBarrierHandle<Ready> = unsafe { bar.init(thread_count) };
let token = bar.arrive();
bar.wait(token);
unsafe { bar.inval() };
```

| 操作  | 描述  |
|:-----|:--------|
| `.init(count)`   | 初始化屏障，设置预期到达计数      |
| `.arrive()`    | 标记到达，返回 `BarrierToken`    |
| `.arrive_expect_tx(bytes)` | 到达并设置预期 TX 字节数（用于 TMA）  |
| `.wait(token)`  | 阻塞直到所有到达 + TX 完成    |
| `.inval()`   | 使屏障无效（清理）    |

---

## 线程束原语

```rust
use cuda_device::warp;

let lane = warp::lane_id();      // 0–31
let wid  = warp::warp_id();

// 洗牌
let partner = warp::shuffle_xor_f32(val, mask);
let from_above = warp::shuffle_down_f32(val, delta);
let from_below = warp::shuffle_up_f32(val, delta);
let from_lane  = warp::shuffle_f32(val, src_lane);

// i32 变体
let partner_i = warp::shuffle_xor_i32(val, mask);

// 投票
let all_true = warp::all(predicate);
let any_true = warp::any(predicate);
let mask     = warp::ballot(predicate);
let count    = warp::popc(mask);
```

### 洗牌操作

| 函数   | 描述        |
|:------|:---------|
| `shuffle_xor_{f32,i32}(val, mask)`    | 与通道 `id ^ mask` 交换         |
| `shuffle_down_{f32,i32}(val, delta)`  | 从通道 `id + delta` 读取        |
| `shuffle_up_{f32,i32}(val, delta)`    | 从通道 `id - delta` 读取        |
| `shuffle_{f32,i32}(val, src)`         | 从指定通道读取                  |

### 投票操作

| 函数   | 返回值  | 描述                                  |
|:-------|:--------|:---------------|
| `all(pred)`    | `bool`  | 如果所有通道的谓词都为真，则返回真      |
| `any(pred)`    | `bool`  | 如果任一通道的谓词为真，则返回真        |
| `ballot(pred)` | `u32`   | 谓词为真的通道的位掩码                 |
| `popc(mask)`   | `u32`   | 设置位的数量                          |

---

## 原子操作

### 作用域 GPU 原子操作

```rust
use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);

// 在kernel中：
COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
let old = COUNTER.load(AtomicOrdering::Acquire);
```

| 作用域   | 类型         |
|:------------|:-------------------|
| `DeviceAtomic{U32,I32,U64,I64,F32,F64}`  | `.gpu` 作用域                 |
| `BlockAtomic{U32,I32,U64,I64,F32,F64}`   | `.cta` 作用域                 |
| `SystemAtomic{U32,I32,U64,I64,F32,F64}`  | `.sys` 作用域（CPU-GPU 共享） |

`core::sync::atomic` 类型（`AtomicU32`、`AtomicBool` 等）也会编译为 GPU 代码，默认为系统作用域。

---

## TMA —— 张量内存加速器（Hopper+）

```rust
use cuda_device::tma::TmaDescriptor;
use cuda_device::tma::{cp_async_bulk_tensor_2d_g2s, cp_async_bulk_commit_group};

// 主机：构建描述符（128 字节，不透明）
// 设备：发出异步批量拷贝
cp_async_bulk_tensor_2d_g2s(smem_ptr, &desc, coord_x, coord_y, barrier_ptr);
cp_async_bulk_commit_group();
```

| 函数     | 描述        |
|:-----------|:-------------|
| `cp_async_bulk_tensor_{1..5}d_g2s(...)`  | 全局 → 共享 异步批量拷贝       |
| `cp_async_bulk_tensor_{1..5}d_s2g(...)`  | 共享 → 全局 异步批量拷贝       |
| `cp_async_bulk_tensor_2d_g2s_multicast(...)`  | 多播到簇中的所有 CTA   |
| `cp_async_bulk_commit_group()`    | 提交未完成的拷贝      |
| `cp_async_bulk_wait_group(n)`   | 等待直到剩余组数 ≤ n     |

---

## 簇编程（Hopper+）

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();        // 此块在簇中的秩
let size = cluster::cluster_size();      // 簇中的块数
cluster::cluster_sync();                 // 跨簇所有块的屏障

// 分布式共享内存
let remote_ptr = cluster::map_shared_rank(local_ptr, target_rank);
let val = cluster::dsmem_read_u32(remote_ptr);
```

---

## 张量核心 —— WGMMA（Hopper，SM 90）

```rust
use cuda_device::wgmma;

wgmma::wgmma_fence();
wgmma::wgmma_commit_group();
wgmma::wgmma_wait_group::<0>();
```

线程束组 MMA：4 个线程束（128 个线程）从共享内存发出矩阵乘加。操作数通过 SMEM 描述符描述；累加器在寄存器中。

---

## 张量核心 —— tcgen05（Blackwell，SM 100+）

```rust
use cuda_device::tcgen05::{TmemGuard, TmemUninit, TmemReady};
use cuda_device::SharedArray;

static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

let guard = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
let guard = unsafe { guard.alloc() };   // TmemUninit → TmemReady
// ... 发出 MMA，通过 guard.address() 读取结果 ...
let _guard = unsafe { guard.dealloc() }; // TmemReady → TmemDeallocated
```

单线程 MMA 发起到专用张量内存（TMEM）。`TmemGuard` 使用类型状态管理 TMEM 生命周期：`TmemUninit → TmemReady → TmemDeallocated`。N_COLS 必须是 2 的幂，范围为 [32, 512]。

---

## 主机端：kernel启动

### 类型化同步

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();
let module = kernels::load(&ctx).unwrap();

let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
let mut output = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut output).unwrap();
```

### 类型化异步

```rust
use cuda_async::device_operation::DeviceOperation;

let module = kernels::load_async(0)?;
let op = module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut output)?;

op.sync()?;       // 阻塞
// 或：op.await?;  // 与 tokio 异步
```

`cuda_launch!` 和 `cuda_launch_async!` 仍然作为较低级 API 可用，用于显式模块加载和自定义启动代码。

### LaunchConfig

| 方法    | 描述           |
|:--------|:-----------|
| `LaunchConfig::for_num_elems(n)`  | 为 `n` 个元素自动配置网格/块   |
| `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }` | 直接构造结构体    |

---

## 调试工具

```rust
use cuda_device::debug;

let t = debug::clock64();       // 周期计数器
debug::trap();                  // 中止kernel
debug::breakpoint();            // cuda-gdb 断点
cuda_device::barrier::nanosleep(1000); // 睡眠 ~1μs
debug::prof_trigger::<7>();     // Nsight 性能分析触发器
```

---

## 快速参考表

### cuda-device 模块

| 模块  | 描述     | 最低 SM   |
|:-----|:--------|:---------|
| `thread`             | 线程/块 ID，`index_1d`，`sync_threads`                        | 所有     |
| `disjoint`           | `DisjointSlice<T>` —— 安全并行写入                            | 所有     |
| `shared`             | `SharedArray<T, N>`，`DynamicSharedArray<T>`                  | 所有     |
| `warp`               | 洗牌，投票，匹配，通道/线程束 ID                               | 所有     |
| `atomic`             | 作用域原子操作（设备/块/系统）                                 | sm_70+   |
| `debug`              | `clock64`，`trap`，`breakpoint`，`gpu_printf!`                | 所有     |
| `fence`              | `threadfence_block` / `threadfence` / `threadfence_system`    | 所有     |
| `grid`               | 网格作用域 `sync`（协作kernel启动）                              | sm_70+   |
| `cooperative_groups` | 类型化句柄，线程束/块归约和扫描                                | 所有     |
| `barrier`            | `ManagedBarrier` —— 用于 TMA/MMA 的异步 mbarrier              | sm_90+   |
| `cluster`            | 线程块簇，DSMEM                                              | sm_90+   |
| `tma`                | `TmaDescriptor`，批量张量拷贝（1D–5D）                        | sm_90+   |
| `wgmma`              | 线程束组 MMA（fence/commit/wait）                             | sm_90    |
| `tcgen05`            | 第五代张量核心，TMEM，`TmemGuard`                             | sm_100+  |
| `cusimd`             | `CuSimd<T, N>`，`Float2`/`Float4`                             | 所有     |
| `clc`                | 簇启动控制                                                   | sm_100+  |

### Crate 映射

| Crate  | 角色 |
|:----|:---|
| `cuda-device`     | 设备内联函数和类型（`#![no_std]`）                                     |
| `cuda-macros`     | 过程宏（`#[kernel]`，`#[device]`，`gpu_printf!`）                      |
| `cuda-host`       | 类型化模块加载以及底层启动辅助函数                                     |
| `cuda-core`       | 安全的 RAII 包装器（`CudaContext`，`CudaStream`，`DeviceBuffer<T>`）    |
| `cuda-async`      | `DeviceOperation`，`DeviceFuture`，`DeviceBox<T>`                       |
| `cuda-bindings`   | 到 `cuda.h` 的原始 `bindgen` FFI                                       |
| `cargo-oxide`     | Cargo 子命令（`cargo oxide run`，`build`，`debug`）                     |

| [上一页](./从源码编译.md) | [下一页](./支持的特性.md) |
| :--- | ---: |