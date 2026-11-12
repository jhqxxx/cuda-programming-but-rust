# 术语表

本术语表给出这些关键术语在整本 cuTile Rust 手册中的含义。

词条标题保留英文原文，以便按字母顺序检索；内部锚点也因此与原文保持一致。

---

## Tile

在 kernel 执行期间位于 GPU **寄存器**中的多维数组片段。tile 是 cuTile Rust 中最基本的计算单位：你把数据从 tensor 加载进 tile，在 tile 上计算，再把结果存回去。tile 具有编译期静态形状，由类型 `Tile<E, S>` 表示，其中 `E` 是元素类型，`S` 是形状（例如 `Tile<f32, {[16, 16]}>`）。

## Tensor

存放在 GPU **全局内存**（HBM）中的多维数组。tensor 以 kernel 参数的形式传入 —— 只读输入用 `&Tensor<E, S>`，可写输出用 `&mut Tensor<E, S>`。tensor 不支持直接算术运算；数据必须先加载进 tile。

## Partition

把一个 tensor 在逻辑上划分成由等大小子区域组成的网格，每个子区域由一个 tile block 处理。"partition" 这个词在宿主侧和设备侧都会出现，但指的是不同的东西。

**宿主侧分区（仅适用于可变 tensor）。** 对 `Tensor<T>` 调用 `.partition([M, N])` 会产生 `Partition<Tensor<T>>`。这是一个宿主侧包装，它在原有 tensor 之外记录了 `partition_shape`（即 tile 维度）。传给 kernel 启动器的、放在 `&mut Tensor<E, S>` 参数位置上的，正是宿主侧的 `Partition<Tensor<T>>`。宿主侧 `Partition` 中保存的 `partition_shape` 决定了 kernel 看到的静态形状 `S` —— 例如传入 `partition_shape = [32, 64]` 的 `Partition`，意味着 kernel 收到的是 `&mut Tensor<T, {[32, 64]}>`。

只有可变 tensor 必须在宿主侧分区。这是因为每一个 `&mut Tensor` 子区域都恰好只由一个 tile block 写入，从而满足 Rust 对可变内存的独占访问要求：同一时刻至多有一个写者可以访问给定区域。通过在启动前分区，系统保证了不会有任意两个 tile block 写入重叠的内存。

**共享 tensor 引用（无需宿主侧分区）。** 只读输入在宿主侧以 `&Tensor<T>`、`Tensor<T>`、`Arc<Tensor<T>>` 或 `&TensorView<T>` 传入，对应 kernel 签名中的 `&Tensor<E, S>`。它们**不需要**在宿主侧分区 —— 多个 tile block 可以安全地同时读取同一个 tensor 或相互重叠的区域，因此没有独占访问约束需要强制。取而代之的是，共享 tensor 在设备侧分区，以获得更大的访问灵活性。

**设备侧分区。** 在 kernel 内部，对 `&Tensor` 调用 `.partition(shape![M, N])` 会创建一个只读的 `Partition` 视图，可以按索引加载单个 tile（例如 `part.load([i, j])`）。共享 tensor 引用正是这样被切成可加载的 tile 的。由于分区发生在设备侧，同一个 `&Tensor` 可以在同一个 kernel 内以不同方式分区 —— 或者用不同的索引模式访问。例如在 GEMM 中，输入矩阵 `x` 和 `y` 在 kernel 体内各自使用不同的形状分区（分别是 `shape![BM, BK]` 和 `shape![BK, BN]`），尽管二者都是从宿主侧以未分区形式传入的。

生成的启动器代码对每一个 `&mut Tensor` 参数都接受 `Partition<Tensor<T>>`、`Partition<&mut Tensor<T>>` 或映射分区，对每一个 `&Tensor` 参数都接受 `&Tensor<T>`、`Tensor<T>`、`Arc<Tensor<T>>` 或 `&TensorView<T>`。

**网格维度。** 宿主侧分区的网格由 tensor 形状除以分区形状并向上取整得到：`grid[i] = ceil(tensor_shape[i] / partition_shape[i])`。结果被映射到一个三维元组 `(x, y, z)`，对于秩小于 3 的 tensor，末尾维度取 1。例如一个 `[128, 256]` 的 tensor 按 `[32, 64]` 分区，产生 `(4, 4, 1)` 的网格。

**启动网格推断。** 在 kernel 启动时，启动器对每个可变输出参数的宿主侧包装调用 `.grid()`，并收集得到的网格。对于普通的 `&mut Tensor` 参数，这就是宿主侧 `Partition` 的网格。对于 `MappedPartitionMut` 参数，这就是映射后的物理 tile block 网格。如果没有通过 `.grid()` 或 `.const_grid()` 显式指定网格，启动网格就会从这些网格**推断**出来。当存在多个可变输出参数时，所有推断出的网格必须一致，否则启动会以错误失败。宿主侧对 tensor 的分区，正是这样决定了 kernel 会运行多少个 tile block。

## Tile Program

GPU 上并发执行的基本单位 —— kernel 函数的一个实例，作为单一逻辑执行线程运行一次，并处理数据的一个分区。这就是你所编写的东西：tile 程序没有可见的内部线程；它加载 tile、在 tile 上计算、存储结果。每个 tile 程序由它在各轴上的网格索引 `program_id(axis)` 标识，网格范围由 `num_programs(axis)` 给出（与 Triton 的 `tl.program_id` / `tl.num_programs` 一致）；元组形式的 `get_tile_block_id()` 和 `get_num_tile_blocks()` 会一次性返回全部三个轴的值。它也被称为 *tile block*（强调它在启动网格中的位置及其到硬件的映射）或 *tile thread*（强调其单线程编程模型）。

## Tile Block

与 [Tile Program](#tile-program) 是同一个单位，只是从启动网格与硬件侧来看：启动网格是 tile block 的三维排列，cuTile Rust 编译器会根据目标架构把每一个映射到一个或多个底层 CUDA 执行单元（thread block、cluster 或 warp）。从程序员视角看，tile block 仍然是一个单线程上下文，处理一个 tile 的数据。

## Tile Thread

[Tile Program](#tile-program) 的别名，强调单线程编程模型：每个 tile 线程作为单一逻辑执行线程把 kernel 函数运行一次。

## Concurrent Execution（并发执行）

多个 tile block 各自独立推进。GPU 调度器在资源可用时把 block 分配到流式多处理器（SM）上。有些 block 同时运行，另一些则在等待；它们之间的相对执行顺序是未指定的。

在宿主侧，并发也会通过 CUDA 流与 async/await 出现：提交到不同流的多个 `DeviceOp` 可以在时间上重叠，异步运行时会调度它们，而不要求程序员指定确切的执行顺序。

## Parallel Execution（并行执行）

多个 tile block 在不同的 SM 上同时执行。一次 kernel 启动同时具有并行与并发：能放到可用 SM 上的 block 并行运行，其余的等待被调度。关于宿主编程中同一组区分，见[《The Rust Programming Language》第 17 章](https://doc.rust-lang.org/book/ch17-00-async-await.html#parallelism-and-concurrency)。

## Streaming Multiprocessor (SM)

NVIDIA GPU 上的主要处理单元。每个 SM 有自己的寄存器、共享内存以及包括 Tensor Core 在内的执行流水线。tile block 由 GPU 的硬件调度器调度到 SM 上。如果资源（寄存器、共享内存、线程槽位）足够，单个 SM 可以并发运行多个 tile block。关于 SM 资源的架构相关细节，见 [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。

## Tensor Cores

专用硬件单元（Volta 架构及以后提供），可以在单条指令中完成小规模的矩阵乘累加。cuTile Rust 中的 `mma()` 内建函数映射到 Tensor Core 指令。Tensor Core 对 tile 维度施加对齐要求（例如，根据元素类型不同，维度通常必须是 8 或 16 的倍数）。

## Block Scaling（块缩放）

一种低精度矩阵乘法布局，其中一个缩放值由一组固定大小的输入取值共享，通常沿 K 方向。在本书使用的 NVFP4 E4M3 缩放布局中，这个比例是每 16 个逻辑 FP4 值对应一个 FP8 缩放值。cuTile Rust 的 `mmaf_scaled()` 操作消费逻辑低精度操作数 tile 及其缩放 tile，然后累加到 `f32`。

## FP4 E2M1FN

4 位浮点编码，在 cuTile Rust 中名为 `f4e2m1fn`，在 Tile IR 中名为 `f4E2M1FN`。这 4 位分别编码 1 个符号位、2 个指数位和 1 个尾数位。它是 `mmaf_scaled()` 使用的逻辑 FP4 操作数类型。

## Packed FP4 Pair

两个 `f4e2m1fn` 取值的可按字节寻址存储表示。在 cuTile Rust 中它就是 `f4e2m1fnx2`；kernel 在调用 `mmaf_scaled()` 之前会把它解包成逻辑 `f4e2m1fn` tile。

## Nibble（半字节）

4 位，即半个字节。打包 FP4 存储每字节使用两个半字节：cuTile Rust 的 `f4e2m1fnx2` 把第一个逻辑 FP4 值存放在低半字节，第二个逻辑 FP4 值存放在高半字节。

## NVFP4

一种 NVIDIA 的 FP4 推理格式，基于 4 位 E2M1 取值加上块缩放 tensor。FP4 载荷通常以每字节两个取值的可按字节寻址数据形式交换。在 cuTile Rust 中，这种存储由 `f4e2m1fnx2` 表示；kernel 在使用块缩放 MMA 之前会把它解包成逻辑 `f4e2m1fn` tile。

## Global Memory (HBM)

GPU 的主存 —— High Bandwidth Memory（高带宽内存）。全局内存容量最大，但比共享内存和寄存器慢。`Tensor` 的数据就存放在全局内存中。

## Registers (RMEM)

GPU 上最快的存储，为 tile block 内每个线程私有。`Tile` 的数据在计算期间就位于寄存器中。每个 SM 的寄存器堆是固定的，因此更大的 tile 会消耗更多寄存器，可能降低占用率。

## Shared Memory (SMEM)

tile block 内各线程共享的片上内存。它比寄存器慢，但比全局内存快。[Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/) 编译器会为 tile kernel 管理共享内存。关于容量与时延细节，见 [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。

## Const Generics

kernel 函数上的编译期常量参数，例如 `const BM: i32`。const 泛型让编译器能够优化寄存器分配、展开循环，并生成架构相关的代码。改变 const 泛型的取值会产生一个新的编译变体。另见 [Const Generic Arrays](#const-generic-arrays)。

## Const Generic Arrays

对 **Rust 编程语言的一项扩展**，它允许 const 泛型参数具有数组类型 —— 例如 `const S: [i32; 2]`。标准 Rust 只支持标量 const 泛型（整数、`bool`、`char`），因此这种语法在普通 Rust 代码中并不合法。cuTile Rust 编译器识别数组形式的 const 泛型，并借助它们在编译期把 tile 形状传播通过整个类型系统。

const 泛型数组是按 tile 形状参数化 kernel 的惯用方式：

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, {[-1, -1]}>,
    y: &Tensor<f32, {[-1, -1]}>,
) { ... }
```

这里 `S` 是从启动时传入的宿主侧分区形状推断出来的。由于 `S` 是编译期常量，编译器可以为每一种不同的形状特化生成的代码。`S` 的一个新取值会产生一个新的编译变体，这与标量 const 泛型完全一样。

## Dynamic Dimensions

在 kernel 签名中以 `-1` 指定的 tensor 形状维度（例如 `Tensor<f32, {[-1, -1]}>`）。动态维度可以在多次 kernel 启动之间变化而不产生新的编译变体。它们不带来编译期优化收益，但为那些经常变化的问题规模提供了灵活性。

## JIT Compilation

cuTile Rust 通常在首次启动时通过一条多阶段流水线编译 kernel 入口函数：Rust AST → Tile IR 字节码 → cubin。生成的启动器会使用当前活跃设备与解析出的 kernel 特化，把已编译的 kernel 缓存在内存中。特化包含入口函数标识、类型与 const 泛型、编译选项、可选的常量网格，以及 tensor 或标量的特化提示。完整的缓存行为与仅编译 API 见[编译](../guide/jit-compilation.md)。

## DeviceOp

对 GPU 工作 —— 分配、kernel 启动或数据传输 —— 的一种惰性描述，在被 `.sync_on(&stream)`、`.sync()` 或 `.await` 调用之前不会执行。`DeviceOp` 可以用 `zip!`、`.then()`、`.map()`、`.shared()` 以及 `.first()`/`.last()` 组合，在提交 GPU 工作之前构建出数据流图。kernel 启动器通过 `IntoDeviceOp` 与 `KernelInput` 直接接受 `Tensor<T>`、`Arc<Tensor<T>>`、`&Tensor<T>`、标量以及 `DeviceOp` 参数。返回类型与输入类型匹配 —— 你放进去什么，就拿回什么。

## DeviceFuture

`DeviceFuture` 是一个已经被分配了资源的 future —— 具体来说，是一条用于执行的设备流 —— 但尚未开始 GPU 工作。当 `DeviceOp` 被调度时（例如通过 `into_future()`）会创建 `DeviceFuture`，此时调度策略会选定一条流。真正的 GPU 工作要等到 `DeviceFuture` 第一次被 poll 时才会提交，而那次 poll 发生在你 `.await` 它或 `tokio::spawn` 它的时候。

## Broadcasting

把一个较小的 tile（或标量）复制扩展，以匹配一个较大 tile 的形状。广播是一种编译期变换 —— 不会分配额外内存。例如 `a.broadcast(y.shape())` 把一个标量扩展成与 `y` 的分区形状匹配的 tile。

## Kernel Fusion

把多个逻辑操作合并成一个 kernel，使中间结果留在寄存器里，而不必写回全局内存再读回来。融合 softmax 是典型例子：求最大值、相减、取指数、求和、相除，全部在一次 kernel 启动中完成。

## Arithmetic Intensity

计算操作数（FLOPs）与内存操作数（传输的字节数）的比值。算术强度越高，意味着 GPU 利用率越好。算术强度低的 kernel（例如逐元素加法）是**内存受限**的；算术强度高的 kernel（例如矩阵乘法）是**计算受限**的。

## CUDA Stream

GPU 操作的有序队列。同一条流上的操作按提交顺序执行；不同流上的操作可以并发执行。cuTile Rust 默认的异步调度策略以轮转方式把工作分配到由四条流组成的池上。

## Occupancy

活跃 warp 数与 SM 所能支持的最大 warp 数之比。更高的占用率通常能提升 GPU 通过切换 warp 来隐藏内存延迟的能力。占用率受寄存器用量、共享内存用量以及线程块大小的影响。

## Warp

一组 32 个 GPU 线程，以锁步方式执行指令。warp 是 SM 上最小的调度单位。32 的倍数的 tile 尺寸与 warp 级执行对齐得很好。
