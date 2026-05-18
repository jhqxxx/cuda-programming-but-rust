# 术语表

术语定义基于它们在 cuda-oxide 项目和本书中的用法。

---

## ABI 标量化

将复合类型分解为主机和设备均认可的形式的过程。切片在所有调用边界处展平：`&[T]` 变成 `(ptr, len)` 对，在函数内部通过 `insertvalue`/`extractvalue` 重建。按值传递的结构体和闭包在内部设备到设备调用中以相同方式展平，但在**kernel**边界处它们作为单个 byval `.param` 传递（一个主机包槽位）——这与主机启动器推送的内容匹配，并避免了展平的设备声明与单槽位主机包之间的不匹配。

## 块（线程块）

在同一流式多处理器上执行并共享访问共享内存的一组线程。块内的线程可以通过 `sync_threads()` 同步。由 `blockIdx` 标识，由 `blockDim` 决定大小。

## 簇（线程块簇）

Hopper+（sm_90）特性：一组最多 16 个线程块，保证在同一 GPC 上协同调度。簇中的块可以通过分布式共享内存（DSMEM）访问彼此的共享内存，并通过 `cluster_sync()` 同步。使用 `#[cluster_launch]` 配置。

## 代码生成后端

`rustc_codegen_cuda` crate —— 以动态库形式加载的自定义 rustc 后端。它在编译期间拦截 MIR，将其通过 `dialect-mir` → `mem2reg` → `dialect-llvm` → LLVM IR → PTX 降级，并将 PTX 与普通主机二进制文件一同发出。

## `cuda-async`

异步执行层。提供 `DeviceOperation`（惰性 GPU 工作描述）、`DeviceFuture`（绑定流的执行）和 `DeviceBox<T>`（设备拥有的内存）。使用 `zip!`、`and_then` 和 `value()` 组合工作。

## `cuda-device`

`#![no_std]` 设备端 crate，提供所有 GPU 内联函数和类型：线程标识、共享内存、线程束原语、屏障、TMA、张量核心、原子操作和调试工具。

## `cuda-core`

围绕 CUDA 驱动 API 的安全 RAII 包装器：`CudaContext`、`CudaStream`、`DeviceBuffer<T>` 和模块加载。处理主机端的 GPU 上下文和内存管理。

## `DeviceOperation`

GPU 工作的惰性、可组合描述（分配、kernel启动或数据传输）。直到调用 `.sync()` 或 `.await` 才执行。可以通过 `zip!`（并行）和 `and_then`（顺序）组合。

## `DisjointSlice<T, IndexSpace>`

kernel的安全可变输出抽象。仅接受一个 `ThreadIndex`，其 `IndexSpace` 与自身类型参数匹配，提供边界检查的 `Option<&mut T>` 返回。通过构造防止数据竞争 —— 每个线程只能写入自己的元素。`get_mut_indexed()` 快捷方式一步生成见证并将其解析为可变引用。

## 分布式共享内存（DSMEM）

Hopper+ 特性，允许簇中的块直接读取和写入彼此的共享内存，而无需经过全局内存。通过 `cluster::map_shared_rank()` 将本地共享指针转换为远程地址来访问。

## 网格

kernel启动中线程的顶层组织。网格是线程块的三维数组，大小由 `gridDim` 决定。线程总数为 `gridDim × blockDim`。

## HMM（异构内存管理）

Linux kernel特性（6.1.24+），允许 GPU 通过缺页异常直接访问主机内存，无需显式 `cudaMemcpy`。cuda-oxide 利用 HMM 进行闭包中的引用捕获 —— GPU 透明地读取主机地址。也称为统一内存管理（UMM）。

## 通道

线程束内的单个线程。通道编号为 0–31，由 `warp::lane_id()` 标识。线程束洗牌和投票操作在通道之间通信，无需共享内存或屏障。

## LTOIR（链接时优化 IR）

用于设备端链接时优化的中间表示。cuda-oxide 为 libNVVM 发出 NVVM IR，libNVVM 可以将其编译为 LTOIR，以便使用 `nvJitLink` 将 Rust 设备代码与 CUDA C++ 设备代码链接。

## `ManagedBarrier<State, Kind, ID>`

Hopper+ 上用于异步操作的类型状态屏障。在编译时跟踪其生命周期（`Uninit → Ready → Invalidated`）和用途（`TmaBarrier`、`MmaBarrier`、`GeneralBarrier`）。无效的状态转换是编译错误。

## 单态化

Rust 编译器为每个使用的具体类型生成泛型函数的专用副本的过程。cuda-oxide 完全支持设备上的单态化 —— `scale::<f32>` 和 `scale::<f64>` 各自成为独立的 PTX 函数。

## Pliron

受 MLIR 启发的用 Rust 编写的 IR 框架，用作 cuda-oxide 编译流水线中的中间表示。MIR 被导入为 Pliron IR，通过dialect pass 转换，并导出为 LLVM IR。

## PTX（并行线程执行）

NVIDIA 用于 GPU kernel的低级虚拟 ISA。cuda-oxide 编译器将 PTX 作为其主要输出，CUDA 驱动程序在加载时将其 JIT 编译为本地 GPU 机器码（SASS）。

## `SharedArray<T, N, ALIGN>`

编译时大小的、块作用域的共享内存数组。在kernel函数中声明为 `static mut`。可选的 alignment 参数（对于 TMA 目标使用 `ALIGN=128`）。访问需要 `unsafe`，因为共享内存是 `!Sync`。

## SM（流式多处理器）

NVIDIA GPU 上的主要处理单元。每个 SM 都有自己的寄存器、共享内存、线程束调度器和包括张量核心在内的执行流水线。线程块由硬件调度器调度到 SM 上。

## `sync_threads()`

块级屏障：线程块中的所有线程必须到达此点，任何线程才能继续越过它。等价于 CUDA C++ 中的 `__syncthreads()`。降级为 `llvm.nvvm.barrier0()`。

## 张量核心

专用的矩阵乘加硬件单元。WGMMA（Hopper，sm_90）以线程束组粒度从共享内存操作。tcgen05（Blackwell，sm_100+）使用单线程发起到专用张量内存（TMEM）。

## `ThreadIndex<'kernel, IndexSpace>`

一个不透明见证，只能由可信的索引函数构造。有三种形式：

- `thread::index_1d() -> ThreadIndex<'_, Index1D>`。始终返回见证；每个线程无条件唯一（`threadIdx.x < blockDim.x` 由硬件强制执行）。
- `thread::index_2d::<S>() -> Option<ThreadIndex<'_, Index2D<S>>>`。行步长是常量泛型，因此 `DisjointSlice<T, Index2D<S>>` 只接受具有匹配 `S` 的见证 —— 混合步长是类型错误。
- `unsafe thread::index_2d_runtime(s) -> Option<ThreadIndex<'_, Runtime2DIndex>>`。当步长仅在启动时已知时的逃生舱口。`unsafe` 是契约：每个将 `Runtime2DIndex` 馈入同一 `DisjointSlice` 的线程必须使用相同的 `s`。

该见证是 `!Send + !Sync + !Copy + !Clone` 且具有 `'kernel` 作用域，因此线程不能通过共享内存传递它，它也不能超过kernel体生命周期。

## TMA（张量内存加速器）

Hopper+ 硬件单元，用于全局内存和共享内存之间的异步批量拷贝。通过 `TmaDescriptor`（在主机上构建的 128 字节不透明描述符）和 `cp_async_bulk_tensor_*` 内联函数操作。完成状态由 `ManagedBarrier` 跟踪。

## `TmemGuard<State, N_COLS>`

Blackwell 张量内存（TMEM）的类型状态包装器 —— 用于 tcgen05 MMA 操作的专用累加器存储。管理 TMEM 生命周期：`TmemUninit → TmemReady → TmemDeallocated`。无效的状态转换是编译错误。

## 线程束

在 SM 上以锁步方式执行指令的一组 32 个线程。线程束是最小的调度单元。线程束级操作（洗牌、投票）在约 1 个周期内在通道之间交换数据，无需共享内存或同步屏障。

## 线程束组

在 Hopper 上为 WGMMA 操作集体执行的四个连续线程束（128 个线程）。线程束组是线程束组级矩阵乘加的发源单元。

## WGMMA（线程束组矩阵乘加）

Hopper（sm_90）张量核心指令集。四个线程束集体从共享内存操作数执行 MMA 到寄存器累加器。通过 `ManagedBarrier` 进行异步执行，使用 commit/wait 机制。

[上一页](./rust+GPU生态系统.md) 