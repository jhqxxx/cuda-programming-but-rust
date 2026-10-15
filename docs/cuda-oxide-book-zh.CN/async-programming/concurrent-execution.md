# 并发执行

你已经知道如何用 `DeviceOperation` 描述 GPU 工作、用 `and_then` 和 `zip!` 组合多阶段流水线，并让调度策略来分配流。现在到了收获的时候：**同时**运行多条流水线。本章端到端地走一遍 `async_mlp` 示例，展示 Tokio 任务、轮转调度器和 Rust 的所有权模型如何协作，在一块 GPU 上并发处理四个批次。

> **参见**
>
> [CUDA Programming Guide -- Multi-Device System](https://docs.nvidia.com/cuda/cuda-programming-guide/#multi-device-system)，了解 CUDA 在多 GPU 上下文、对等访问以及跨设备内存方面的规则。

## 场景

你正在运行一个三层 MLP 前向传播：**GEMM**（矩阵乘）、**MatVec**（矩阵-向量乘）和 **ReLU**（激活）。GPU 上加载了两个权重矩阵，另有四批输入数据要处理。每个批次彼此独立 —— 批次 0 不依赖批次 1 —— 但它们共享同一组权重。

![四个 MLP 前向传播并发运行。顶部：共享权重作为 `Arc<DeviceBox>` 上传一次，然后被廉价地克隆到每个批次。中部：GPU 时间线 —— 轮转把批次分配到四个流上，错开的流水线相互重叠。底部：顺序处理的耗时约为单个批次的 4 倍；如果 GPU 还有空闲的 SM，并发处理约为 1.3 倍。](images/concurrent-batches.svg)

*四个 MLP 前向传播并发运行。顶部：共享权重作为 `Arc<DeviceBox>` 上传一次，然后被廉价地克隆到每个批次。中部：GPU 时间线 —— 轮转把批次分配到四个流上，错开的流水线相互重叠。底部：顺序处理的耗时约为单个批次的 4 倍；如果 GPU 还有空闲的 SM，并发处理约为 1.3 倍。*

顺序方式一次只处理一个批次：

```text
Batch 0:  ████ GEMM ████ MatVec ██ ReLU █ D2H █
                                                  Batch 1:  ████ GEMM ████ ...
```

并发方式则让它们在多个流上重叠：

```text
Stream 0:  ████ GEMM ████ MatVec ██ ReLU █ D2H █
Stream 1:    ████ GEMM ████ MatVec ██ ReLU █ D2H █
Stream 2:      ████ GEMM ████ MatVec ██ ReLU █ D2H █
Stream 3:        ████ GEMM ████ MatVec ██ ReLU █ D2H █
```

如果 GPU 有足够的 SM 同时运行多个 kernel，重叠版本会明显地更快完成。而且由于每条流水线都是一个流上的一条 `and_then` 链，批次内部的各个阶段仍然严格有序 —— 不需要任何跨流同步。

## 第 1 步：初始化运行时

一切都从 `init_device_contexts` 开始。它会创建 CUDA 上下文、设置调度策略（默认是带四个流的轮转），并让线程局部的状态可用于 `.sync()` 和 `.await`：

```rust
use cuda_async::simt::device_context::init_device_contexts;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;
```

嵌入的 module 通过线程局部的异步上下文加载。带类型的 module 句柄可以被廉价地克隆进每个批次的流水线。

## 第 2 步：上传共享权重

这个模型有两个权重矩阵：W0（DIM x DIM）和 W1（DIM）。在四次前向传播的整个过程中，它们都需要驻留在设备上。这正是 `zip!` 和 `.arc()` 大显身手的地方：

```rust
    let w0_host: Vec<f32> = (0..DIM * DIM)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let w1_host: Vec<f32> = (0..DIM)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.01)
        .collect();

    let (w0, w1): (Arc<DeviceBox<[f32]>>, Arc<DeviceBox<[f32]>>) = zip!(
        h2d(w0_host).arc(),
        h2d(w1_host).arc()
    ).await?;
```

`zip!` 把两个相互独立的 H2D 传输打包成一个操作。`.arc()` 把每个结果包进 `Arc`，这样权重就能被共享。`.await` 把合并后的操作调度到池中某个流上，并等待它完成。

这一行之后，`w0` 和 `w1` 就是 `Arc<DeviceBox<[f32]>>` —— 克隆廉价、可以安全共享，并且钉在设备上。

## 第 3 步：构建并 spawn 批次流水线

现在到了有趣的部分。对每个批次，你构建一条惰性流水线（此时还没有 GPU 工作），然后把它交给 `tokio::spawn`：

```rust
    let num_batches = 4;
    let mut handles = vec![];

    for batch_idx in 0..num_batches {
        let w0 = w0.clone();       // Arc 克隆：约 1 ns
        let w1 = w1.clone();
        let module = module.clone();

        let batch_data: Vec<f32> = (0..DIM * DIM)
            .map(|i| ((i + batch_idx * 37) % 13) as f32 * 0.1)
            .collect();

        let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
            .and_then(move |(input, hidden, output)| {
                // 阶段 1：GEMM —— hidden = input × W0
                // ... 构建并 finalize 一次自有（owned）的启动，然后用 and_then 串接 ...
            })
            .and_then(move |(hidden, output, w1, module)| {
                // 阶段 2：MatVec —— output = hidden × W1
                // ...
            })
            .and_then(move |(output, module)| {
                // 阶段 3：ReLU —— result = max(0, output)
                // ...
            })
            .and_then(d2h);  // 阶段 4：把结果拷贝到宿主

        handles.push(tokio::spawn(pipeline.into_future()));
    }
```

下面拆解一下这里发生了什么：

1. **`pipeline` 是一个 `DeviceOperation`。** 它描述了整个前向传播，但不执行任何 GPU 工作。构建它纯粹是宿主侧的计算 —— 分配闭包和结构体。

2. **`.into_future()` 把它转换成 `DeviceFuture`。** 这会触发调度策略，由策略挑选一个流。批次 0 拿到流 0，批次 1 拿到流 1，依此类推 —— 轮转。

3. **`tokio::spawn` 把这个 future 交给 Tokio 运行时。** 执行器有空闲容量时，运行时就会轮询它。第一次轮询时，流水线的 `execute()` 运行，把所有 GPU 工作提交到被分配的那个流上。

4. **运行时被解放出来了。** 第一次轮询之后，任务进入挂起状态。GPU 在四个流上同时埋头算数。没有任何宿主线程坐在那里等待。

5. **当 GPU 完成**某条流水线的工作时，`cuLaunchHostFunc` 回调触发，唤醒对应的 Tokio 任务。运行时再次轮询它，交付 `Vec<f32>` 结果。

## 第 4 步：收集结果

```rust
    for (i, handle) in handles.into_iter().enumerate() {
        let result: Vec<f32> = handle.await??;
        println!("Batch {}: {} elements, first 4 = {:?}",
            i, result.len(), &result[..4]);
    }
```

两个 `?` 解开了两层错误：外层的 `JoinError`（以防 Tokio 任务 panic）和内层的 `DeviceError`（以防 GPU 工作失败）。在生产系统里，你会分别处理它们。

## `.await` vs `.sync()` vs `tokio::spawn`

看过这三者的实际表现之后，下面说明各自适用的场景：

**`.sync()`** 会阻塞调用线程，直到 GPU 完成。它适合脚本、测试，以及任何没有异步运行时的场景。简单、无需仪式感，但也没有并发 —— 宿主线程会卡在等待上：

```rust
let result = pipeline.sync()?;
```

**`.await`** 在 GPU 工作期间让出当前的异步任务。同一 Tokio 线程上的其他任务因此可以继续推进。就吞吐而言这比 `.sync()` 好，但在任务内部仍然是顺序的 —— 操作完成之前，`.await` 之后的代码不会执行：

```rust
let result = pipeline.await?;
```

**`tokio::spawn(op.into_future())`** 把流水线作为一个完全独立的任务启动。发起 spawn 的代码立即继续执行，结果稍后通过 join 句柄送达。这才是实现真正并发的方式 —— 多条流水线同时在不同流上运行：

```rust
let handle = tokio::spawn(pipeline.into_future());
// ... spawn 更多流水线，做其他工作 ...
let result = handle.await??;
```

> **提示**
>
> 对于一条依赖链（比如那个四阶段前向传播），优先使用 `and_then`，而不是一串顺序的 `.await`。`and_then` 链完全在一个流上运行，阶段之间没有任何调度开销。一串顺序的 `.await` 会让每个操作都经过调度策略，可能落在不同的流上，从而需要跨流同步。

## 所有权模式

在 Rust 中做并发 GPU 编程，意味着 Rust 的所有权规则在主动为你工作 —— 偶尔也会挡你的路。以下这些模式会反复出现。

### 通过 `and_then` 闭包传递数据

每个 `and_then` 闭包都以 `move` 捕获数据。kernel 启动产生的输出是 `()`，因此你需要显式地把下一阶段所需的缓冲区传递下去：

```rust
launch_gemm(input, hidden, w0)
    .and_then(move |()| {
        // input 已被 kernel 消耗。hidden 和 module 得以保留，
        // 因为它们只是被捕获，并未被消耗。
        value((hidden, output, w1, module))
    })
```

闭包返回一个 `Value`，其中包含一个元组，装着下一阶段所需的一切。这个元组就是阶段之间传递的「接力棒」。

### 用 `Arc` 共享不可变数据

模型权重、查找表，以及其他跨流水线共享的不可变数据，都应该用 `Arc` 包裹。`.arc()` 组合子会对 `DeviceOperation` 的输出自动完成这件事。对于你已经持有的数据，`Arc::new()` 就行：

```rust
let weights = h2d(weight_data).arc().await?;  // Arc<DeviceBox<[f32]>>
for batch in batches {
    let w = weights.clone();  // 廉价的引用计数递增
    tokio::spawn(forward_pass(batch, w).into_future());
}
```

### 让设备内存保持存活

`DeviceBox` 必须一直存活，直到 GPU 用完它。在 `and_then` 链中这是自动的 —— 闭包持有 `DeviceBox`，它会一直存活到下一阶段取用它。危险之处在于 `with_context` 闭包：你在其中执行异步拷贝，而 `DeviceBox` 可能在拷贝完成之前就被丢弃：

```rust
fn d2h(dev: DeviceBox<[f32]>) -> impl DeviceOperation<Output = Vec<f32>> {
    with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let mut host = vec![0.0f32; dev.len()];
        unsafe {
            memcpy_dtoh_async(
                host.as_mut_ptr(), dev.cu_deviceptr(),
                dev.len() * std::mem::size_of::<f32>(),
                stream.cu_stream(),
            ).unwrap();
        }
        // dev 被闭包捕获，会一直存活到闭包返回。
        // 结果被消费之前流会同步，
        // 所以异步拷贝会在 dev 被丢弃之前完成。
        value(host)
    })
}
```

关键在于：`dev` 被 `move` 闭包捕获，直到闭包返回才会被丢弃。既然在 `DeviceFuture` 交付结果之前流会同步，异步拷贝就会在 `dev` 被释放之前完成。

<a id="concurrent-error-handling"></a>

## 错误处理

GPU 工作的错误会像任何 Rust 代码一样沿着 `Result` 链传播。在跑并发批次时，通常你希望即使某个批次失败也继续处理：

```rust
for (i, handle) in handles.into_iter().enumerate() {
    match handle.await {
        Ok(Ok(result)) => {
            println!("Batch {i}: {} elements", result.len());
        }
        Ok(Err(device_err)) => {
            eprintln!("Batch {i}: GPU error: {device_err}");
        }
        Err(join_err) => {
            eprintln!("Batch {i}: task panicked: {join_err}");
        }
    }
}
```

这三个分支分别对应：成功；CUDA 驱动或调度错误（例如显存不足、无效的启动配置）；以及 Tokio 任务 panic。

## 多设备执行

对于有多块 GPU 的系统，给 `init_device_contexts` 传入更大的设备数量：

```rust
init_device_contexts(0, 2)?;  // 默认设备 0，可容纳 2 台设备
```

这会把默认设备设为 GPU 0，并为两台设备准备好线程局部的映射表。每台设备的 CUDA 上下文、调度策略和流池都是在首次使用时惰性创建的。所有由策略驱动的操作（`.sync()`、`.await`）都会使用默认设备，除非你显式指向另一台。`ExecutionContext` 携带设备 ID，因此 GPU 0 上的操作永远不会干扰 GPU 1 上的流。

> **提示**
>
> 多设备编程需要留意内存的放置。分配在 GPU 0 上的 `DeviceBox` 在未启用对等访问时无法从 GPU 1 访问。在构建设备相关的操作时，可以用 `with_context` 检查 `ctx.get_device_id()`。

## 性能调优

### 流池大小

默认的四流池对大多数工作负载都够用，但如果你在追求最后几个百分点的吞吐，合适的池大小取决于你的具体情况：

| 工作负载                    | 建议的池大小 | 原因                                     |
|:------------------------------|:--------------------|:------------------------------------------|
| 每次启动一个大型 kernel | 1--2                | kernel 本身已经把 GPU 打满了       |
| 大量小 kernel               | 4--8                | 在流之间重叠启动开销   |
| kernel + memcpy 混合         | 2--4                | 让计算与数据传输重叠        |
| 时延敏感的服务     | 每个请求 1 个       | 避免队头阻塞               |

### 性能分析

Nsight Systems 是可视化流占用情况的标准工具：

```bash
nsys profile --trace=cuda cargo oxide run async_mlp
nsys-ui report.nsys-rep
```

留意同一流上 kernel 之间的空隙（启动开销）、池中闲置的流（工作分配不均衡），以及跨流意外的串行化（依赖缺失或多余）。

### 常见陷阱

| 陷阱                                | 你会看到的现象                                         | 修复方法                                          |
|:---------------------------------------|:-----------------------------------------------------|:---------------------------------------------|
| 在异步上下文中调用 `.sync()`      | 阻塞 Tokio 线程，让其上的所有任务都停摆    | 改用 `.await`                         |
| `DeviceBox` 在流同步之前被丢弃 | use-after-free 崩溃或结果损坏            | 把所有权保持在 `and_then` 闭包中        |
| 流太多                       | 调度开销超过了重叠带来的收益          | 做性能分析，减小池大小                    |
| 缺少 `init_device_contexts`         | 第一个操作就报 `DeviceError::Context`                   | 在程序启动时调用一次                   |

> **参见**
>
> [异步 MLP 流水线](../projects/async-mlp-pipeline.md)项目章节包含完整的 `async_mlp` 源码、构建说明与预期输出。
