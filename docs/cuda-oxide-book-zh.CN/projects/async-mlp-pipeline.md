# 项目：异步 MLP 流水线

你已经学习了如何定义 kernel、搬移数据、用组合子组合操作，以及跨流调度工作。现在是时候把它们全部串起来了。

本章端到端地走一遍 `async_mlp` —— cuda-oxide 仓库中最完整的示例 —— 从第一条 `use` 语句到 stdout 上最后一行 `[ReLU OK]`。读完本章，你将构建出一个多 kernel 的 GPU 流水线，它并发处理四个批次，并且用上了工具箱里的每一种异步模式。如果你是一直按顺序读到这里的，那就把这章当作最终 boss 吧。

> **参见**
>
> [并发执行](../async-programming/concurrent-execution.md)一章深入介绍了并发流水线背后的*概念*。本章聚焦于*代码* —— 一个完整、可构建的项目，你可以运行、修改并做性能分析。

---

## 我们要构建什么

GPU 上一个玩具级的**多层感知机（MLP）前向传播**：

```text
input [64×64]  ──►  GEMM(input, W0)  ──►  hidden [64×64]
                                              │
                            MatVec(hidden, W1) ──►  output [64]
                                                       │
                                                 ReLU(output) ──►  result [64]
```

三个 kernel、四个阶段（包括最后的设备到宿主拷贝）、四个批次、四条 CUDA 流、一组共享权重，而且没有任何宿主线程在那儿空等。下面是整体图景：

![四个 MLP 前向传播并发运行。共享权重作为 `Arc<DeviceBox>` 上传一次，然后被廉价地克隆到每个批次。轮转调度器把批次分配到四条流上，错开的流水线在 GPU 时间线上相互重叠。](../async-programming/images/concurrent-batches.svg)

*四个 MLP 前向传播并发运行。共享权重作为 `Arc<DeviceBox>` 上传一次，然后被廉价地克隆到每个批次。轮转调度器把批次分配到四条流上，错开的流水线在 GPU 时间线上相互重叠。*

顺序方式一次只处理一个批次 —— 四倍的工作量，四倍的等待。并发方式让它们重叠起来：如果 GPU 还有空闲的 SM，你大致能在单个批次的时间加上一点错开量内完成。对几次 `Arc::clone()` 调用来说，这个回报相当不错。

---

## 项目结构

示例位于 `cuda-oxide/crates/rustc-codegen-cuda/examples/async_mlp/`，是一个独立的 Cargo workspace 成员：

```text
async_mlp/
├── Cargo.toml        ← 依赖：cuda-device、cuda-core、cuda-async、tokio
└── src/
    └── main.rs       ← kernel + 宿主代码，单个文件
```

所有东西 —— 设备 kernel 和宿主编排 —— 都在一个文件里。`#[kernel]` 属性告诉编译器哪些函数要变成 PTX；其余部分按普通 Rust 编译。没有单独的 `.cu` 文件，没有头文件的把戏，也没有构建系统的双重人格。

### 依赖

`Cargo.toml` 只引入我们真正需要的 crate：

```toml
[dependencies]
cuda-device   = { path = "../../../cuda-device" }       # #[kernel]、DisjointSlice、thread::*
cuda-core     = "0.3.1"                                 # 与 cutile-rs 共享：CudaModule、simt::LaunchConfig
cuda-async    = "0.3.1"                                 # 与 cutile-rs 共享：simt::DeviceOperation、zip!、and_then
tokio       = { version = "1", features = ["rt", "macros"] }
```

`cuda-device` 提供设备侧 API（内建指令、安全的可变切片）。`cuda-core` 负责模块加载与启动配置。`cuda-async` 提供 `DeviceOperation` 图、组合子和流池调度器。`tokio` 是负责轮询 future 的宿主异步运行时。

---

## GPU kernel

下面三个函数由 `rustc-codegen-cuda` 编译成 PTX。它们从不在宿主上执行 —— `#[kernel]` 属性会把每个函数重命名为 `cuda_oxide_kernel_<hash>_<name>`，这样代码生成后端就能识别并提取它们。

### sgemm_naive —— 矩阵乘法

```rust
use cuda_device::{DisjointSlice, kernel, thread};

use cuda_device::thread::Runtime2DIndex;

#[kernel]
pub fn sgemm_naive(
    m: u32, n: u32, k: u32,
    alpha: f32, a: &[f32], b: &[f32],
    beta: f32, mut c: DisjointSlice<f32, Runtime2DIndex>,
) {
    let row = thread::index_2d_row();
    let col = thread::index_2d_col();

    // 行宽来自 `c`，在宿主侧被绑定为同一个 `n`。
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        // `Some` 已经保证 col < n —— 不需要手动检查
        if row < m as usize {
            let n_sz = n as usize;
            let k_sz = k as usize;
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i < k_sz {
                sum = sum + a[row * k_sz + i] * b[i * n_sz + col];
                i = i + 1;
            }
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
```

每个线程计算输出矩阵的一个元素。二维线程索引直接映射到 (row, col) 位置。`DisjointSlice` 会做边界检查，并要求匹配的索引空间类型。`c` 的行宽不是每次调用的参数：宿主在启动时把它一次性绑定到切片里，`index_2d_runtime(&c)` 再把它读回来，因此从构造上就保证每个线程解析出的行宽是一致的。剩下的证明是显式的：Z 处于非活跃状态，且原始的二维启动形状与 kernel 匹配。

> **提示**
>
> 这是刻意实现的*朴素* GEMM —— 一个线程一个元素，没有共享内存分块，也没有合并访问（coalescing）技巧。生产级 GEMM 会使用[共享内存](../advanced/shared-memory-and-synchronization.md)一章中的技术。但为了演示异步组合，正确性比性能更重要。

### matvec_naive —— 矩阵-向量乘法

```rust
#[kernel]
pub fn matvec_naive(
    _m: u32, n: u32,
    mat: &[f32], vec_in: &[f32],
    mut vec_out: DisjointSlice<f32>,
) {
    let row = thread::index_1d();
    if let Some(out_elem) = vec_out.get_mut(row) {
        let n_sz = n as usize;
        let mut sum = 0.0f32;
        let mut j = 0usize;
        while j < n_sz {
            sum = sum + mat[row.get() * n_sz + j] * vec_in[j];
            j = j + 1;
        }
        *out_elem = sum;
    }
}
```

每个输出元素一个线程，一维索引。`_m` 参数没有使用，但能让调用约定与 BLAS 风格的接口保持一致。

### relu —— 激活函数

```rust
#[kernel]
pub fn relu(input: &[f32], mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = output.get_mut(idx) {
        let val = input[idx.get()];
        *out_elem = if val > 0.0f32 { val } else { 0.0f32 };
    }
}
```

逐元素的 `max(0, x)`。在流水线中，`input` 和 `output` 指向同一个缓冲区 —— 这是完全合法的原地（in-place）写法，因为每个线程读写的是同一个索引，而且启动是一维的。

### 需要注意的地方

几个在这三个 kernel 中反复出现的模式：

| 模式                                   | 作用                                                                                               |
| :---------------------------------------- | :----------------------------------------------------------------------------------------------------------|
| `thread::index_1d()` / `index_2d::<S>()`  | 根据 block/grid 维度计算全局线程索引                                                |
| `DisjointSlice<f32>`                      | 带边界检查的可变输出；启动几何信息补全了唯一性证明                              |
| `if let Some(elem) = slice.get_mut(idx)`  | 让超出数据规模的线程安静下来的边界检查                                                    |
| 用 `while` 循环而不是 `for`            | 风格选择 —— 带 range 的 `for` 循环在设备上也能用，但 `while` 让循环边界更显式 |

---

## 连接宿主与设备

有三个辅助函数把原始 CUDA driver 调用包装成 `DeviceOperation`。它们用 `with_context` 在执行时（而不是在调用点）拿到调度器分配的流。这是关键的洞见：你现在构建好配方，调度器稍后挑选流。

### h2d —— 宿主到设备

```rust
fn h2d(host_data: Vec<f32>) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = host_data.len();
        let num_bytes = n * mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(dptr, host_data.as_ptr(), num_bytes, stream.cu_stream())
                .unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}
```

`host_data` 向量被 `move` 捕获。闭包在调度器执行该操作时运行 —— 那时它才拥有真正的 CUDA 流。`malloc_async` 和 `memcpy_htod_async` 都是流序（stream-ordered）的，所以分配和拷贝会在调度器选定的流上串行化。

### zeros —— 零初始化的设备缓冲区

```rust
fn zeros(n: usize) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let num_bytes = n * mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memset_d8_async(dptr, 0, num_bytes, stream.cu_stream()).unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}
```

与 `h2d` 相同的模式，只是用 `memset_d8_async` 取代了 `memcpy`。GEMM kernel 使用 `beta = 0.0`，所以初始内容并不重要，但清零是防御性做法。

### d2h —— 设备到宿主

```rust
fn d2h(dev: DeviceBox<[f32]>) -> impl DeviceOperation<Output = Vec<f32>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = dev.len();
        let num_bytes = n * mem::size_of::<f32>();
        let mut host = vec![0.0f32; n];
        unsafe {
            memcpy_dtoh_async(
                host.as_mut_ptr(), dev.cu_deviceptr(),
                num_bytes, stream.cu_stream(),
            ).unwrap();
        }
        let _ = &dev;
        value(host)
    })
}
```

`let _ = &dev;` 这一行看着像空操作，但它让 `dev` 在闭包中一直存活到流同步为止。没有它，`dev` 会在 `memcpy_dtoh_async` 调用之后、流完成拷贝之前被 drop —— 这是设备侧经典的 use-after-free。

---

## 组合流水线

每个批次都是一个由组合子构建出来的 `DeviceOperation`。组装它时不会有任何 GPU 工作发生 —— 它只是对未来工作的惰性描述。下面是各个部件如何拼在一起：

![单个批次的组合子流水线。`zip!` 并行分配三个缓冲区。`and_then` 链把 GEMM → MatVec → ReLU → D2H 串起来，并通过 `value()` 在每个阶段之间传递设备句柄。](../async-programming/images/combinator-dataflow.svg)

*单个批次的组合子流水线。`zip!` 并行分配三个缓冲区。`and_then` 链把 GEMM → MatVec → ReLU → D2H 串起来，并通过 `value()` 在每个阶段之间传递设备句柄。*

### 第 1 步：初始化运行时

```rust
const DIM: usize = 64;
const BLOCK: u32 = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;
```

`init_device_contexts(0, 1)` 把设备 0 设为默认设备，并初始化设备上下文映射（容量为 1）。轮转流池在首次使用时惰性创建。由我们的 `#[kernel]` 函数编译出的内嵌模块只加载一次，并由带类型的模块句柄共享。

### 第 2 步：上传共享权重

```rust
    let w0_host: Vec<f32> = (0..DIM * DIM)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let w1_host: Vec<f32> = (0..DIM)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();

    let (w0, w1): (Arc<DeviceBox<[f32]>>, Arc<DeviceBox<[f32]>>) =
        zip!(h2d(w0_host).arc(), h2d(w1_host).arc()).await?;
```

两次独立的上传，用 `zip!` 捆绑在一起，让它们共享同一条流。`.arc()` 把每个结果包进 `Arc` —— 一个引用计数的指针，克隆一次的代价大约是一纳秒。四个批次、四次克隆、零次设备拷贝。

### 第 3 步：构建每个批次的流水线

魔法就在这里。对每个批次，我们构建一条四阶段链：

```rust
    use cuda_async::simt::launch::{AsyncKernelLaunchBuilder, OwnedAsyncKernelLaunch};

    let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
        .and_then(move |(input, hidden, output)| {
            // 阶段 1：GEMM —— hidden = input @ W0
            let func = module.load_function("sgemm_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((
                DIM as u32, DIM as u32, DIM as u32,
                1.0f32,
                input.cu_deviceptr(), input.len() as u64,
                w0.cu_deviceptr(), w0.len() as u64,
                0.0f32,
                hidden.cu_deviceptr(), hidden.len() as u64,
                DIM as u32,  // hidden 的行宽：DisjointSlice<_, Runtime2DIndex>
                             // 所接收的第三个 packet 字
            ));
            // 安全性：packet/配置与 sgemm_naive 匹配（包括 `c` 的宽度字），
            // 调度器使用模块的上下文，且 owned 包装器会保留其分配。
            let launch = unsafe { builder.finalize_unchecked(gemm_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (input, w0, hidden));
            launch.and_then(move |(_input, _w0, hidden)| {
                value((hidden, output, w1, module))
            })
        })
        .and_then(move |(hidden, output, w1, module)| {
            // 阶段 2：MatVec —— output = hidden @ W1
            let func = module.load_function("matvec_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((
                DIM as u32, DIM as u32,
                hidden.cu_deviceptr(), hidden.len() as u64,
                w1.cu_deviceptr(), w1.len() as u64,
                output.cu_deviceptr(), output.len() as u64,
            ));
            // 安全性：packet/配置与 matvec_naive 匹配，调度器使用
            // 模块的上下文，且 owned 包装器会保留其分配。
            let launch = unsafe { builder.finalize_unchecked(matvec_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (hidden, w1, output));
            launch.and_then(move |(_hidden, _w1, output)| value((output, module)))
        })
        .and_then(move |(output, module)| {
            // 阶段 3：ReLU —— result = max(0, output)
            let func = module.load_function("relu").unwrap();
            let relu_out: DeviceBox<[f32]> = output;
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((
                relu_out.cu_deviceptr(), relu_out.len() as u64,
                relu_out.cu_deviceptr(), relu_out.len() as u64,
            ));
            // 安全性：packet/配置与 relu 匹配，调度器使用模块的
            // 上下文，且 owned 包装器会保留 output。
            let launch = unsafe { builder.finalize_unchecked(relu_cfg) };
            OwnedAsyncKernelLaunch::new(launch, relu_out)
        })
        .and_then(d2h);
```

有几点值得放慢脚步细看：

**`value()` 接力棒。** 每个 `and_then` 闭包消费上一阶段的输出，并返回一个 `DeviceOperation`。kernel 启动返回的是 `()`，所以你需要用 `value()` 把下一阶段所需的设备句柄传递下去。把它想成接力棒 —— kernel 跑完，接力棒交出去。

**类型标注。** `zip!` + `and_then` 链产生的深层嵌套泛型超出了 Rust 的类型推导能力。你需要为闭包参数写显式标注：

```rust
.and_then(move |(hidden, output, w1, module): (
    DeviceBox<[f32]>,
    DeviceBox<[f32]>,
    Arc<DeviceBox<[f32]>>,
    Arc<CudaModule>,
)| { ... })
```

这是唯一的人体工学上的粗糙之处。要让 `zip!` 工作，还需要导入 `Zippable` trait。

**原地 ReLU。** 阶段 3 把 `relu_out` 同时作为 `input` 和 `output` 传给 kernel。由于每个线程在同一个索引上读 `input[idx]`、写 `output[idx]`，这是安全的 —— 没有线程会读到别的线程写入的值。

**原始启动的边界。** builder 本身是惰性的。`finalize_unchecked` 是原始 packet 与几何信息变成可运行状态的地方，所以每次调用都需要一份局部安全性证明。`OwnedAsyncKernelLaunch` 会保持其缓冲区存活。当 kernel 声明了启动契约时，优先使用带 `PreparedLaunch<K>` 的生成式 owned-async 方法；对于像 `c` 这样运行时才确定宽度的输出，生成的方法接收 `cuda_host::RowWidthOwned::new(buffer, width)`，并替你编组第三个 packet 字。

### 第 4 步：派生并收集

```rust
    handles.push(tokio::spawn(pipeline.into_future()));
```

`.into_future()` 把惰性的 `DeviceOperation` 转换成 `DeviceFuture`。此时调度策略会挑选一条流（批次 0 → 流 0，批次 1 → 流 1，轮转进行）。`tokio::spawn` 把这个 future 交给 Tokio 运行时。

在第一次轮询时，流水线的 `execute()` 会把所有 GPU 工作提交到已分配的流上，并注册一个 `cuLaunchHostFunc` 回调。future 返回 `Poll::Pending`。GPU 完成后，回调唤醒该任务。没有宿主线程空转。

```rust
    for (i, handle) in handles.into_iter().enumerate() {
        let result: Vec<f32> = handle.await.expect("Tokio task panicked")?;
        let all_non_negative = result.iter().all(|&v| v >= 0.0);
        println!("Batch {}: {} elements, first 8 = {:?}{}",
            i, result.len(), &result[..8.min(result.len())],
            if all_non_negative { " [ReLU OK]" } else { " [ReLU FAILED]" }
        );
    }
```

ReLU 的合理性检查不是深度学习验证 —— 它只是确认激活函数确实跑过了。每个输出都应当非负。如果你看到 `[ReLU FAILED]`，说明上游有非常严重的问题。

---

## 构建、运行、验证

```bash
cargo oxide run async_mlp
```

预期输出：

```text
=== Async MLP Pipeline ===

Allocating model weights...
  W0: 64x64 on device (Arc refcount=1)
  W1: 64 on device (Arc refcount=1)

Launched 4 batches concurrently, awaiting results...

Batch 0: 64 elements, first 8 = [0.0020799995, 0.0, ...] [ReLU OK]
Batch 1: 64 elements, first 8 = [0.0, 0.0, ...] [ReLU OK]
Batch 2: 64 elements, first 8 = [0.0, 0.00108, ...] [ReLU OK]
Batch 3: 64 elements, first 8 = [0.00244, 0.0025, ...] [ReLU OK]

SUCCESS: All batches completed.
```

Arc 引用计数从 1 开始（`w0` 和 `w1` 各有一个引用）。在处理批次期间它们会暂时升到 5（原始引用加上四次克隆），并在批次完成时回落。如果你增加更多批次，引用计数会相应增长 —— 没有拷贝，也没有重新分配。

### 用 Nsight Systems 做性能分析

想亲眼看到流的重叠：

```bash
nsys profile --trace=cuda cargo oxide run async_mlp
nsys-ui report.nsys-rep
```

在时间线视图中，找四排并行的 kernel 块 —— 每条流一排。如果 kernel 在一条流上被串行化了，说明轮转调度器没有被用上（很可能是没有调用 `init_device_contexts`，或者只配置了一条流）。

---

## 它为什么"真"

这仍然是个玩具 —— 64×64 的矩阵、假权重、三个 kernel。但它的*结构*与生产级 GPU 流水线是一样的：

| 生产模式                                | async_mlp 中的对应物                                |
| :------------------------------------------------ | :-------------------------------------------------- |
| 模型权重加载一次，在多个请求间共享 | `zip!(h2d(w0).arc(), h2d(w1).arc())`                |
| 每个请求的推理流水线                    | `and_then` 链：GEMM → MatVec → ReLU → D2H        |
| 并发的请求处理                     | 每个批次 `tokio::spawn(pipeline.into_future())`    |
| 基于流的 GPU 调度                       | 轮转 `init_device_contexts(0, 1)`            |
| 非阻塞的宿主                                 | 对 join 句柄 `.await`，而不是 `.sync()`             |

把矩阵从 64 放大到 4096，把朴素 kernel 换成基于共享内存分块的版本（参见[共享内存](../advanced/shared-memory-and-synchronization.md)），再加几层，你就得到了一个真实推理服务器的骨架。

---

## 扩展这个示例

一些可以继续走下去的思路：

**加一层 softmax。** 写一个 `#[kernel]`，在 64 元素的输出向量上计算[数值稳定的 softmax](https://en.wikipedia.org/wiki/Softmax_function)。在 ReLU 之后再接一个 `.and_then` 把它串起来。

**在更大维度下做性能分析。** 把 `DIM` 改成 512 或 1024。观察 GEMM 如何主宰时间线。然后用 `SharedArray` 把 `sgemm_naive` 换成分块版本，看看加速效果。

**加入错误恢复。** 把 kernel 启动闭包中的 `.unwrap()` 换成正确的 `Result` 传播。使用[并发执行](../async-programming/concurrent-execution.md#concurrent-error-handling)一章里的三臂 `match` 模式，分别处理 GPU 错误和任务 panic。

**多设备。** 传入 `init_device_contexts(0, 2)` 来管理两块 GPU。构建一个批次拆分器，把偶数批次路由到 GPU 0，奇数批次路由到 GPU 1。

> **参见**
>
> - [设备操作模型](../async-programming/the-device-operation-model.md) —— 惰性 GPU 图是如何工作的
> - [组合子与组合](../async-programming/combinators-and-composition.md) —— 详解 `and_then`、`zip!`、`value()`、`.arc()`
> - [调度与流](../async-programming/scheduling-and-streams.md) —— 轮转、流池、调度策略
> - [并发执行](../async-programming/concurrent-execution.md) —— 本章一切内容背后的理论
