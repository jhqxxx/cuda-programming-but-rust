# 项目：异步 MLP 流水线

你已经学会了如何定义kernel、移动数据、使用组合器组合操作以及跨流调度工作。是时候把它们全部整合起来了。

本章从头到尾讲解 `async_mlp` —— cuda-oxide 仓库中最完整的示例 —— 从第一个 `use` 语句到标准输出上的最终 `[ReLU OK]`。最终你将构建一个多kernel GPU 流水线，并发处理四个批次，使用工具包中的每一个异步模式。如果你一直在按顺序阅读各章，不妨将此视为最终的 Boss。

> 另请参阅：[并发执行](../异步GPU编程/并发执行.md) 一章深入介绍了并发流水线背后的**概念**。本章聚焦于**代码** —— 一个完整的、可构建的项目，你可以运行、修改和剖析。

---

## 我们正在构建什么

GPU 上的一个玩具**多层感知机（MLP）前向传播**：

```text
input [64×64]  ──►  GEMM(input, W0)  ──►  hidden [64×64]
                                              │
                            MatVec(hidden, W1) ──►  output [64]
                                                       │
                                                 ReLU(output) ──►  result [64]
```

三个kernel，四个阶段（包括最后的设备到主机拷贝），四个批次，四个 CUDA 流，一组共享的权重，零个在等待的主机线程。大图如下：

![concurrent](../assets/concurrent-batches.svg)

<figcaption>四个 MLP 前向传播并发运行。共享权重作为 <code>Arc&lt;DeviceBox&gt;</code> 一次性上传，并廉价克隆到每个批次中。轮询调度器将四个批次分发到四个流上，错开的流水线在 GPU 时间线上重叠。</figcaption>
</figure>

顺序方法一次处理一个批次 —— 四倍的工作量，四倍的等待时间。并发方法将它们重叠：如果 GPU 有空闲的 SM，你将在大约一个批次加上少量错开的时间内完成。这只需要调用几次 `Arc::clone()`，回报相当不错。

---

## 项目结构

示例位于 `crates/rustc-codegen-cuda/examples/async_mlp/`，是一个独立的 Cargo 工作区成员：

```text
async_mlp/
├── Cargo.toml        ← 依赖：cuda-device, cuda-core, cuda-async, tokio
└── src/
    └── main.rs       ← kernel + 主机代码，单文件
```

所有代码 —— 设备kernel和主机编排 —— 都在一个文件中。`#[kernel]` 属性告诉编译器哪些函数成为 PTX；其余部分作为普通 Rust 编译。没有单独的 `.cu` 文件，没有头文件体操，没有构建系统的分裂人格。

### 依赖

`Cargo.toml` 精确地引入了我们需要的 crate：

```toml
[dependencies]
cuda-device   = { path = "../../../cuda-device" }       # #[kernel], DisjointSlice, thread::*
cuda-core     = { path = "../../../cuda-core" }         # CudaModule, LaunchConfig
cuda-async    = { path = "../../../cuda-async" }        # DeviceOperation, zip!, and_then, spawn
tokio         = { version = "1", features = ["rt", "macros"] }
```

- `cuda-device` 提供设备端 API（内联函数、安全的可变切片）。
- `cuda-core` 处理模块加载和启动配置。
- `cuda-async` 提供 `DeviceOperation` 图、组合器和流池调度器。
- `tokio` 是轮询 future 的主机异步运行时。

---

## GPU kernel

这三个函数由 `rustc-codegen-cuda` 编译成 PTX。它们永远不会在主机上执行 —— `#[kernel]` 属性将每个函数重命名为 `cuda_oxide_kernel_<hash>_<name>`，以便代码生成后端识别并提取它们。

### sgemm_naive —— 矩阵乘法

```rust
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_device::thread::Runtime2DIndex;

#[kernel]
pub fn sgemm_naive(
    m: u32, n: u32, k: u32,
    alpha: f32, a: &[f32], b: &[f32],
    beta: f32, mut c: DisjointSlice<f32, Runtime2DIndex>,
) {
    let n_sz = n as usize;
    let row = thread::index_2d_row();
    let col = thread::index_2d_col();

    // SAFETY: every thread sees the same `n_sz` (same kernel arg).
    if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_sz) } {
        // col < n_sz guaranteed by `Some` -- no manual check needed
        if row < m as usize {
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

每个线程计算输出矩阵的一个元素。二维线程索引直接映射到 (row, col) 位置。`DisjointSlice<f32>` 是安全的可变视图 —— 它在编译时保证每个线程写入不同的元素，因此没有数据竞争，不需要 `unsafe`。

> **提示**：这是一个有意为之的**朴素** GEMM —— 一个线程一个元素，没有共享内存分块，没有合并技巧。生产级 GEMM 会使用[共享内存](../advanced/shared-memory-and-synchronization.html)一章中的技术。但对于演示异步组合，正确性优先于性能。

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

每个输出元素一个线程，一维索引。`_m` 参数未使用，但保持调用约定与 BLAS 风格接口一致。

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

逐元素 `max(0, x)`。在流水线中，`input` 和 `output` 指向同一个缓冲区 —— 这是一种完全有效的就地模式，因为每个线程读取并写入同一个索引。

### 值得注意的模式

在所有三个kernel中重复出现的一些模式：

| 模式 | 作用 |
|------|------|
| `thread::index_1d()` / `index_2d::<S>()` | 从块/网格维度计算全局线程索引 |
| `DisjointSlice<f32>` | 安全的可变输出 —— 编译器保证无别名 |
| `if let Some(elem) = slice.get_mut(idx)` | 边界检查，使超出数据大小的线程静默 |
| `while` 循环代替 `for` | 风格选择 —— 带范围的 `for` 循环也可以在设备上工作，但 `while` 使循环边界明确 |

---

## 桥接主机和设备

三个辅助函数将原始 CUDA 驱动调用包装成 `DeviceOperation` 值。它们使用 `with_context` 在执行时接收调度器的流 —— 而不是在调用点。这是关键见解：你现在构建“菜谱”，调度器稍后选择流。

### h2d —— 主机到设备

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

`host_data` 向量通过 `move` 捕获。闭包在调度器执行操作时运行 —— 此时它拥有一个真实的 CUDA 流。`malloc_async` 和 `memcpy_htod_async` 是按流顺序的，因此分配和拷贝在调度器选择的流上串行化。

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

与 `h2d` 相同的模式，但使用 `memset_d8_async` 而不是 `memcpy`。GEMM kernel使用 `beta = 0.0`，因此初始内容无关紧要，但置零是一种防御性做法。

### d2h —— 设备到主机

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

`let _ = &dev;` 这一行看起来像空操作，但它使 `dev` 在闭包中保持存活，直到流同步。没有它，`dev` 会在 `memcpy_dtoh_async` 调用之后、流完成拷贝之前被释放 —— 经典的设备端释放后使用。

---

## 组合流水线

每个批次都是一个由组合器构建的单个 `DeviceOperation`。组装时不会发生任何 GPU 工作 —— 它是对未来工作的惰性描述。以下是各部分的组合方式：

![concurrent](../assets/combinator-dataflow.svg)

<figcaption>单个批次的组合器流水线。<code>zip!</code> 并行分配三个缓冲区。<code>and_then</code> 链将 GEMM → MatVec → ReLU → D2H 序列化，通过 <code>value()</code> 在每个阶段传递设备句柄。</figcaption>
</figure>

### 步骤 1：初始化运行时

```rust
const DIM: usize = 64;
const BLOCK: u32 = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;
}
```

`init_device_contexts(0, 1)` 将设备 0 设置为默认设备，并初始化设备上下文映射（容量为 1）。轮询流池在首次使用时惰性创建。从我们的 `#[kernel]` 函数编译的嵌入式模块被加载一次，并由类型化的模块句柄共享。

### 步骤 2：上传共享权重

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

两个独立的上传，用 `zip!` 捆绑在一起，以便它们可以共享一个流。`.arc()` 将每个结果包装在 `Arc` 中 —— 一个引用计数指针，克隆大约花费一纳秒。四个批次，四次克隆，零次设备拷贝。

### 步骤 3：构建每个批次的流水线

这就是魔法所在。对于每个批次，我们构建一个四阶段链：

```rust
    let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
        .and_then(move |(input, hidden, output)| {
            // Stage 1: GEMM — hidden = input @ W0
            let func = module.load_function("sgemm_naive").unwrap();
            let mut launch = AsyncKernelLaunch::new(Arc::new(func));
            launch.push_args((
                DIM as u32, DIM as u32, DIM as u32,
                1.0f32,
                input.cu_deviceptr(), input.len() as u64,
                w0.cu_deviceptr(), w0.len() as u64,
                0.0f32,
                hidden.cu_deviceptr(), hidden.len() as u64,
            )).set_launch_config(gemm_cfg);
            launch.and_then(move |()| value((hidden, output, w1, module)))
        })
        .and_then(move |(hidden, output, w1, module)| {
            // Stage 2: MatVec — output = hidden @ W1
            let func = module.load_function("matvec_naive").unwrap();
            let mut launch = AsyncKernelLaunch::new(Arc::new(func));
            launch.push_args((
                DIM as u32, DIM as u32,
                hidden.cu_deviceptr(), hidden.len() as u64,
                w1.cu_deviceptr(), w1.len() as u64,
                output.cu_deviceptr(), output.len() as u64,
            )).set_launch_config(matvec_cfg);
            launch.and_then(move |()| value((output, module)))
        })
        .and_then(move |(output, module)| {
            // Stage 3: ReLU — result = max(0, output)
            let func = module.load_function("relu").unwrap();
            let relu_out: DeviceBox<[f32]> = output;
            let mut launch = AsyncKernelLaunch::new(Arc::new(func));
            launch.push_args((
                relu_out.cu_deviceptr(), relu_out.len() as u64,
                relu_out.cu_deviceptr(), relu_out.len() as u64,
            )).set_launch_config(relu_cfg);
            launch.and_then(move |()| value(relu_out))
        })
        .and_then(d2h);
```

有几件事值得放慢脚步：

**`value()` 接力棒。** 每个 `and_then` 闭包消耗前一阶段的输出，并返回一个 `DeviceOperation`。kernel启动返回 `()`, 因此你需要 `value()` 将下一阶段需要的设备句柄向前传递。把它想象成一个接力棒 —— kernel运行，接力棒传递。

**类型注解。** `zip!` + `and_then` 链产生的深度嵌套泛型超出了 Rust 的类型推断能力。你需要在闭包参数上给出显式注解：

```rust
.and_then(move |(hidden, output, w1, module): (
    DeviceBox<[f32]>,
    DeviceBox<[f32]>,
    Arc<DeviceBox<[f32]>>,
    Arc<CudaModule>,
)| { ... })
```

这是唯一的人机工程学缺陷。`zip!` 正常工作还需要导入 `Zippable` trait。

**就地 ReLU。** 阶段 3 将 `relu_out` 同时作为 `input` 和 `output` 传递给kernel。由于每个线程读取 `input[idx]` 并写入相同索引的 `output[idx]`，这是安全的 —— 没有线程读取另一个线程的写入。

### 步骤 4：生成并收集

```rust
    handles.push(tokio::spawn(pipeline.into_future()));
```

`.into_future()` 将惰性的 `DeviceOperation` 转换为 `DeviceFuture`。此时调度策略选择一个流（批次 0 → 流 0，批次 1 → 流 1，轮询）。`tokio::spawn` 将 future 交给 Tokio 运行时。

在第一次轮询时，流水线的 `execute()` 将所有 GPU 工作提交到指定的流，并注册一个 `cuLaunchHostFunc` 回调。future 返回 `Poll::Pending`。当 GPU 完成时，回调唤醒任务。没有主机线程空转。

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

ReLU 健全性检查并不是深度学习验证 —— 它只是确认激活函数运行了。每个输出都应该非负。如果你看到 `[ReLU FAILED]`，说明上游出了严重问题。

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

Arc 引用计数从 1 开始（`w0` 和 `w1` 各一个引用）。在批处理期间，它们会暂时上升到 5（原始 + 四个克隆），并在批次完成时下降。如果你添加更多批次，引用计数会相应缩放 —— 没有拷贝，没有重新分配。

### 使用 Nsight Systems 分析

要查看流重叠的实际效果：

```bash
nsys profile --trace=cuda cargo oxide run async_mlp
nsys-ui report.nsys-rep
```

在时间线视图中，寻找四行并行的kernel块 —— 每个流一行。如果kernel串行化在单个流上，则说明没有使用轮询调度器（很可能 `init_device_contexts` 没有被调用，或者只配置了一个流）。

---

## 是什么让它“真实”

这仍然是一个玩具 —— 64×64 矩阵、假权重、三个kernel。但**结构**与生产级 GPU 流水线相同：

| 生产模式 | async_mlp 对应 |
|----------|----------------|
| 模型权重加载一次，跨请求共享 | `zip!(h2d(w0).arc(), h2d(w1).arc())` |
| 每个请求的推理流水线 | `and_then` 链：GEMM → MatVec → ReLU → D2H |
| 并发请求处理 | 每个批次 `tokio::spawn(pipeline.into_future())` |
| 基于流的 GPU 调度 | 轮询策略 `init_device_contexts(0, 1)` |
| 非阻塞主机 | join handle 上的 `.await`，而不是 `.sync()` |

将矩阵从 64 放大到 4096，用分块共享内存版本（参见[共享内存](../advanced/shared-memory-and-synchronization.html)）替换朴素kernel，添加更多层，你就得到了一个真正的推理服务器的骨架。

---

## 扩展示例

一些可以进一步探索的想法：

**添加 softmax 层。** 编写一个 `#[kernel]`，在 64 元素输出向量上计算[数值稳定的 softmax](https://en.wikipedia.org/wiki/Softmax_function)。在 ReLU 之后用另一个 `.and_then` 链接它。

**在更大维度上分析。** 将 `DIM` 改为 512 或 1024。观察 GEMM 主导时间线。然后用使用 `SharedArray` 的分块版本替换 `sgemm_naive`，观察加速效果。

**添加错误恢复。** 将kernel启动闭包中的 `.unwrap()` 替换为正确的 `Result` 传播。使用[并发执行](../异步GPU编程/并发执行.md)一章中的三臂 `match` 模式，分别处理 GPU 错误和任务 panic。

**多设备。** 传递 `init_device_contexts(0, 2)` 来管理两个 GPU。构建一个批次拆分器，将偶数批次路由到 GPU 0，奇数批次路由到 GPU 1。

> 另请参阅：
> - [DeviceOperation 模型](../异步GPU编程/DeviceOperation模型.md) —— 惰性 GPU 图如何工作
> - [组合器与组合](../异步GPU编程/组合子与组合.md) —— 详细讲解 `and_then`、`zip!`、`value()`、`.arc()`
> - [调度与流](../异步GPU编程/调度和流.md) —— 轮询策略、流池、调度策略
> - [并发执行](../异步GPU编程/并发执行.md) —— 本章所有内容背后的理论

| [上一页](../异步GPU编程/并发执行.md) |  |
| :--- | ---: |
