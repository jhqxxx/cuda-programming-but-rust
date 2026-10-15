# 术语表

本术语表中的术语按它们在 cuda-oxide 项目及本书中的用法进行定义。

---

## ABI Scalarization

将复合类型分解为宿主与设备都认可的形状的过程。切片在所有调用边界处都会被
展平：`&[T]` 变为 `(ptr, len)` 对，并在函数内部通过
`insertvalue`/`extractvalue` 重建。按值传递的结构体与闭包在设备到设备的内部
调用中以同样的方式展平，但在 **kernel** 边界处它们改为以单个 byval `.param`
（一个宿主数据包槽位）的形式传递 —— 这与宿主启动器推入的内容一致，避免了
展平后的设备声明与单槽位宿主数据包之间的不匹配。

## Block (Thread Block)

一组在同一 Streaming Multiprocessor 上执行、并共享对共享内存的访问的线程。
block 内的线程可以通过 `sync_threads()` 同步。由 `blockIdx` 标识，由
`blockDim` 确定大小。

## Cluster (Thread Block Cluster)

Hopper+（sm_90）的一项特性：最多 16 个线程块组成的一组，保证在同一 GPC 上
协同调度。cluster 中的 block 可以通过 Distributed Shared Memory（DSMEM）
访问彼此的共享内存，并通过 `cluster_sync()` 同步。用 `#[cluster_launch]`
进行配置。

## Codegen Backend

`rustc_codegen_cuda` crate —— 一个以 dylib 形式加载的自定义 rustc backend。
它在编译期间拦截 MIR，将其经由 `dialect-mir` → `mem2reg` → 带注解的循环
展开 → LLVM dialect → LLVM IR → PTX 逐级下沉。它在生成常规宿主二进制的同时
产出 PTX。

## `cuda-async`

异步执行层，一个与 cutile-rs 共享、维护在仓库根目录（`cuda-async/`）的
crate。cuda-oxide 模型位于 `cuda_async::simt` 下：
`DeviceOperation`（惰性的 GPU 工作描述）、`DeviceFuture`（绑定到流的执行）
以及 `DeviceBox<T>`（设备拥有的内存）。用 `zip!`、`and_then` 和 `value()`
组合工作。

## `cuda-device`

`#![no_std]` 的设备侧 crate，提供全部 GPU intrinsics 与类型：线程标识、
共享内存、warp 原语、barrier、TMA、tensor core、原子操作以及调试设施。

## `cuda-core`

对 CUDA Driver API 的安全 RAII 封装：`CudaContext`、`CudaStream`、
`DeviceBuffer<T>` 以及模块加载。在宿主侧处理 GPU 上下文与内存管理。与
cutile-rs 共享并维护在仓库根目录（`cuda-core/`）；除
`simt::LaunchConfig`、`simt::vmm`、`simt::peer` 和 `simt::memory` 之外，
cuda-oxide 的接口都在 crate 根处重新导出。

## `DeviceOperation`

对 GPU 工作（分配、kernel 启动或数据传输）的惰性、可组合的描述。在被调用
`.sync()` 或 `.await` 之前不会执行。可以用 `zip!`（并行）和 `and_then`
（顺序）组合。

## `DisjointSlice<T, IndexSpace>`

kernel 的一种安全可变输出抽象。它只接受 `IndexSpace` 与其自身类型参数匹配的
`ThreadIndex`，提供带边界检查的 `Option<&mut T>` 返回值。配合相匹配的预备
启动几何，每个线程只能写入属于自己的元素。`get_mut_indexed()` 这个快捷方法
在一次调用中铸造见证（witness）并将其解析为可变引用。

## Distributed Shared Memory (DSMEM)

Hopper+ 的一项特性，允许 cluster 内的 block 直接读写彼此的共享内存，而无需
经过全局内存。通过 `cluster::map_shared_rank()` 访问，把本地共享指针转换为
远程地址。

## Grid

kernel 启动中线程的顶层组织方式。grid 是一个三维的线程块数组，由 `gridDim`
确定大小。线程总数为 `gridDim × blockDim`。

## HMM (Heterogeneous Memory Management)

Linux 内核的一项特性（6.1.24+），让 GPU 可以通过缺页异常直接访问宿主内存，
而无需显式的 `cudaMemcpy`。cuda-oxide 利用 HMM 实现闭包中的引用捕获 ——
GPU 透明地读取宿主地址。也称为 Unified Memory Management（UMM）。

## Lane

warp 中的单个线程。lane 编号为 0–31，由 `warp::lane_id()` 标识。warp shuffle
与 vote 操作在 lane 之间通信，不需要共享内存或 barrier。

## LTOIR (Link-Time Optimized IR)

用于设备侧链接时优化的一种中间表示。cuda-oxide 为 libNVVM 生成 NVVM IR，
后者可将其编译为 LTOIR，以便使用 `nvJitLink` 把 Rust 设备代码与 CUDA C++
设备代码链接起来。

## `ManagedBarrier<State, Kind, ID>`

Hopper+ 上用于异步操作的 typestate barrier。在编译期跟踪其生命周期
（`Uninit → Ready → Invalidated`）及其用途（`TmaBarrier`、`MmaBarrier`、
`GeneralBarrier`）。非法的状态转换是编译错误。

## Monomorphization

Rust 编译器为每个具体类型生成泛型函数专用副本的过程。cuda-oxide 在设备端
完全支持单态化 —— `scale::<f32>` 和 `scale::<f64>` 各自成为独立的 PTX 函数。

## `PreparedLaunch<K>`

一个可复用的宿主侧凭证，证明某项启动配置已针对确切的 kernel `K` 检查过。
生成的安全方法接受这个带标记（branded）的值。把裸的 `LaunchConfig` 直接传给
生成的方法是 unsafe 的，因为它不能证明该 kernel 的索引或资源需求。

## Pliron

一个受 MLIR 启发、用 Rust 编写的 IR 框架，在 cuda-oxide 编译管线中充当中间
表示。MIR 被导入 Pliron IR，经 dialect pass 变换，再导出为 LLVM IR。

## PTX (Parallel Thread Execution)

NVIDIA 面向 GPU kernel 的底层虚拟 ISA。cuda-oxide 编译器以 PTX 作为其主要
输出，CUDA 驱动在加载时将其 JIT 编译为原生 GPU 机器码（SASS）。

## `SharedArray<T, N, ALIGN>`

编译期确定大小、作用域为 block 的共享内存数组。在 kernel 函数中声明为
`static mut`。可选对齐参数（用于 TMA 目标时使用 `ALIGN=128`）。访问需要
`unsafe`，因为共享内存是 `!Sync` 的。

## SM (Streaming Multiprocessor)

NVIDIA GPU 上的主要处理单元。每个 SM 拥有自己的寄存器、共享内存、warp
调度器，以及包括 Tensor Core 在内的执行流水线。线程块由硬件调度器调度到
SM 上。

## `sync_threads()`

block 级 barrier：线程块中的所有线程都必须到达此点，任何线程才能继续向后
执行。等价于 CUDA C++ 中的 `__syncthreads()`。下沉为
`llvm.nvvm.barrier0()`。

## Tensor Cores

专用的矩阵乘加硬件单元。WGMMA（Hopper，sm_90）以 warpgroup 为粒度、从共享
内存取操作数进行运算。tcgen05（Blackwell，sm_100+）采用单线程发射，并使用
专用的 Tensor Memory（TMEM）。

## `ThreadIndex<'kernel, IndexSpace>`

一个不透明的见证（witness），只能由受信任的索引函数构造。有三种形式：

- `thread::index_1d() -> ThreadIndex<'_, Index1D>`。总是返回一个见证。
  仅当 block 与 grid 的 Y/Z 维度都为 1 时它才是唯一的；`domain = 1` 的
  预备启动可证明这一点。
- `thread::index_2d::<S>() -> Option<ThreadIndex<'_, Index2D<S>>>`。
  行 stride 是一个 const 泛型，因此 `DisjointSlice<T, Index2D<S>>`
  只接受 `S` 匹配的见证 —— 混用 stride 是类型错误。
- `thread::index_2d_runtime(&slice) -> Option<ThreadIndex<'_, Runtime2DIndex>>`。
  用于仅在启动时才已知的 stride，且是安全的：行宽存在于切片中，由宿主
  一次性写入启动数据包。见证存储该线程的 `(row, col)` 坐标，而被寻址的
  切片会根据自己的宽度解析它们，因此按构造，索引同一切片的每个线程都
  使用相同的行宽。

该见证是 `!Send + !Sync + !Copy + !Clone` 且作用域为 `'kernel`，因此线程
无法通过共享内存将其洗白（launder），它也不能比 kernel 函数体存活得更久。

## TMA (Tensor Memory Accelerator)

Hopper+ 上的硬件单元，用于在全局内存与共享内存之间进行异步批量拷贝。通过
`TmaDescriptor`（在宿主上构建的 128 字节不透明描述符）与
`cp_async_bulk_tensor_*` intrinsics 工作。完成情况由 `ManagedBarrier` 跟踪。

## `TmemGuard<State, N_COLS>`

Blackwell Tensor Memory（TMEM）的 typestate 封装 —— TMEM 是 tcgen05 MMA
操作专用的累加器存储。它管理 TMEM 的生命周期：
`TmemUninit → TmemReady → TmemDeallocated`。非法的转换是编译错误。

## Warp

一组 32 个线程，在 SM 上以锁步（lockstep）方式执行指令。warp 是最小的调度
单位。warp 级操作（shuffle、vote）在 lane 之间交换数据，约 1 个周期完成，
不需要共享内存或同步 barrier。

## Warpgroup

四个连续的 warp（128 个线程），在 Hopper 上为 WGMMA 操作而协同工作。
warpgroup 是 warpgroup 级矩阵乘加的发射单位。

## WGMMA (Warpgroup Matrix Multiply-Accumulate)

Hopper（sm_90）的 tensor core 指令集。四个 warp 协同从共享内存操作数发射
MMA，将结果累加进寄存器累加器。异步执行，通过 `ManagedBarrier`
提交/等待（commit/wait）。
