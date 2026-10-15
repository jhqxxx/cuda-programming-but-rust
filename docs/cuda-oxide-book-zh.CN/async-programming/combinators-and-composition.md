# 组合子与组合

单次 kernel 启动很少是故事的全部。真实的 GPU 工作负载可能要上传输入数据、做两个矩阵相乘、应用激活函数，再把结果拷贝回宿主 —— 四个不同的操作，且必须严格按这个顺序发生。本章展示 cuda-oxide 的组合子系统如何让你把这些片段拼成一条流水线，并且在你执行它的那一刻之前始终保持惰性、与流无关。

> **参见**
>
> [DeviceOperation 模型](the-device-operation-model.md)，了解组合子所构建于其上的基础 trait 与执行方法。

## 问题：多步骤的 GPU 工作

设想你在构建一次简单的神经网络前向传播。步骤如下：

1. 把输入批次从**上传**到设备（host → device）。
2. **GEMM** —— 把输入与权重矩阵 W0 相乘。
3. **ReLU** —— 对结果应用激活函数。
4. 把输出**下载**回宿主。

每一步都依赖前一步。上传完成之前无法运行 GEMM，ReLU 完成之前也无法下载。在同步的世界里，你会写四次独立的 `.sync()` 调用：

```rust
let input  = h2d(batch_data).sync()?;       // 上传
let hidden = launch_gemm(input, w0).sync()?; // GEMM
let output = launch_relu(hidden).sync()?;    // ReLU
let result = d2h(output).sync()?;            // 下载
```

这能行，但有个问题。每次 `.sync()` 都会经过调度策略，而调度策略可能每次都分配*不同的*流。不同流之间没有顺序保证 —— GEMM 可能在另一个流上的上传完成之前就开始了。而且即便策略碰巧选了同一个流，每次 `.sync()` 都会阻塞宿主线程，提交一个操作，再次阻塞，提交下一个，如此往复。你要为「把菜谱交给厨房、等待菜肴、再交下一个菜谱」这个往返付出四遍代价。

你真正想要的是把整份菜单写在一张卡片上，一次交给厨房。

## `and_then` —— 「当这个完成时，做那个」

`and_then` 组合子串联两个操作：运行第一个，把它的输出传给一个闭包，而该闭包产生第二个操作。两者都在**同一个流**上运行，因此 CUDA 的顺序保证意味着第二个总能看到第一个的结果：

```rust
let pipeline = h2d(batch_data)
    .and_then(|input| launch_gemm(input, w0))
    .and_then(|hidden| launch_relu(hidden))
    .and_then(|output| d2h(output));

let result: Vec<f32> = pipeline.sync()?;
```

整条链是一个单独的 `DeviceOperation`。你编写它时不会发生任何 GPU 工作 —— 你只是在描述这个序列。当你调用 `.sync()` 时，调度策略挑选**一个**流，整条链就在该流上从上到下运行。一趟厨房、一次等待、四道菜。

### 数据如何在链中流动

每个 `and_then` 闭包把上一阶段的 `Output` 作为参数接收。闭包必须返回一个新的 `DeviceOperation`，它成为链中的下一环：

```text
h2d(batch)         → DeviceBox<[f32]>
    │
    └─ and_then ─► launch_gemm(input, w0)   → DeviceBox<[f32]>
                        │
                        └─ and_then ─► launch_relu(hidden)   → DeviceBox<[f32]>
                                            │
                                            └─ and_then ─► d2h(output)   → Vec<f32>
```

整条链的类型由编译器推断。你永远不需要写出中间类型 —— 它们会自动流经这些闭包。

### 用 `value()` 携带额外数据

Kernel 启动产生 `()` —— 它们通过对设备内存的副作用起作用，而不是通过返回值。但下一阶段需要缓冲区句柄、模块引用，可能还有其他元数据。技巧就是用 `value()` 把下一阶段需要的所有东西打包起来：

```rust
let pipeline = launch_gemm(input, hidden)
    .and_then(move |()| {
        // kernel 产生了 ()，但我们仍然持有来自外围作用域的 `hidden` 和 `module`。
        // 把它们打包给下一阶段。
        value((hidden, module))
    })
    .and_then(move |(hidden, module)| {
        launch_relu(hidden, module)
    });
```

`move` 关键字很重要 —— 每个闭包按所有权捕获它需要的数据。当闭包运行时，它消费这些值，并通过 `value()` 把它们向前传递。这正是 Rust 所有权系统按设计发挥作用的地方：确保每个缓冲区在同一时刻只被一个阶段使用，且没有悬垂引用。

### `and_then_with_context` —— 当你需要流时

有时阶段之间的闭包需要执行原始 CUDA 操作 —— 一次异步内存拷贝、一次事件记录或一次同步调用。这些操作需要 `CUstream` 句柄，而普通的 `and_then` 闭包拿不到它。`and_then_with_context` 同时传递上一阶段的结果和 `ExecutionContext`：

```rust
let pipeline = launch_kernel(input)
    .and_then_with_context(|ctx, gpu_result| {
        let stream = ctx.get_cuda_stream();
        copy_result_to_staging(stream, gpu_result)
    });
```

请少用它 —— 大多数流水线完全可以只用 `and_then` 以及那些内部使用 `with_context` 的辅助函数（`h2d`、`d2h`、`zeros`）来构建。

## `zip!` —— 打包相互独立的工作

并非所有事情都是顺序的。在运行前向传播之前，你需要分配三个缓冲区：输入、用于隐藏层的临时缓冲区，以及输出缓冲区。这些分配是相互独立的 —— 谁都不依赖谁。但每一个都返回一个你之后要用到的值。

如果你对这三者都用 `and_then`，最终会为了把所有结果向前传递而笨拙地嵌套闭包：

```rust
// 别这么做 —— 它能工作，但很难读
let pipeline = h2d(batch_data)
    .and_then(|input| {
        zeros(DIM * DIM).and_then(move |hidden| {
            zeros(DIM).and_then(move |output| {
                value((input, hidden, output))
            })
        })
    });
```

`zip!` 通过把相互独立的操作合并成单个操作来解决这个问题，该操作返回它们结果的元组：

```rust
use cuda_async::zip;

let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM));
// pipeline: impl DeviceOperation<Output = (DeviceBox, DeviceBox, DeviceBox)>

let (input, hidden, output) = pipeline.sync()?;
```

清爽多了。`zip!` 接受两个或三个参数，并在同一个流上按顺序执行它们。结果被收集成元组并一起返回。

> **提示**
>
> `zip` 这个名字来自数据组合的模式 —— 两个相互独立的结果被 zip 成一个元组 —— 而不是来自并行执行。所有分支都在同一个流上按顺序运行。要跨流实现真正的并发执行，请参见[并发执行](concurrent-execution.md)。

### 把 `zip!` 与 `and_then` 结合起来

真正的威力在你把它们结合起来时才展现。`zip!` 处理相互独立的准备工作，`and_then` 处理有依赖的流水线：

```rust
let pipeline = zip!(h2d(batch_data), zeros(DIM * DIM), zeros(DIM))
    .and_then(|(input, hidden, output)| launch_gemm(input, hidden, w0))
    .and_then(|hidden| launch_relu(hidden))
    .and_then(|result| d2h(result));
```

这读起来几乎像伪代码：「分配三个缓冲区，然后 GEMM，然后 ReLU，然后下载。」整个东西就是一个 `DeviceOperation`，当你 `.sync()` 或 `.await` 它时会在一个流上运行。

## `.arc()` —— 跨流水线共享结果

在批处理场景中，你可能会加载一次模型权重，并在多次前向传播中共享它们。每条批次流水线都需要对权重的引用，但 `DeviceOperation` 会按值消费它的输出。你无法把同一个 `DeviceBox` move 进四个不同的闭包。

`.arc()` 把输出包装进 `Arc<T>`，使其可以廉价地克隆：

```rust
let (w0, w1) = zip!(
    h2d(w0_host).arc(),
    h2d(w1_host).arc()
).await?;

// w0: Arc<DeviceBox<[f32]>>  -- 把它克隆进每个批次
for batch in batches {
    let w0 = w0.clone();
    let w1 = w1.clone();
    tokio::spawn(build_forward_pass(batch, w0, w1).into_future());
}
```

权重缓冲区驻留在设备上，通过 `Arc` 共享，每条批次流水线持有一个廉价的 `Arc::clone`。只要还有任何批次在使用它们，权重就会保持存活，而 Rust 的引用计数会自动处理清理。

## `unzip!` —— 拆分成对的结果

偶尔你会有一个产生元组的操作，但想把它各个元素送入不同的下游链。`unzip!` 把一个产生二元组的操作拆分成两个共享执行过程的独立操作 —— 源最多运行一次：

```rust
use cuda_async::unzip;

let pair_op = zip!(allocate_a(), allocate_b());
let (op_a, op_b) = unzip!(pair_op);

let result_a = op_a.and_then(|a| process_a(a));
let result_b = op_b.and_then(|b| process_b(b));
```

在底层，`unzip!` 创建了一个共享的 memoization 节点。哪条分支先执行就触发源，第二条读取缓存的结果。这对于把一个共享的准备步骤拆分成分岔的下游流水线很有用。

## 把它们全部放在一起：一次 MLP 前向传播

![用组合子构建的 MLP 前向传播中的数据流。`zip!` 在顶部打包三个相互独立的分配。四个 `and_then` 阶段依次串联：GEMM、MatVec、ReLU 和 D2H。类型通过 `value()` 元组在各阶段之间流动。整条链是一个 `DeviceOperation` —— 在 `.sync()` 或 `.await` 之前什么都不会执行。](images/combinator-dataflow.svg)

*用组合子构建的 MLP 前向传播中的数据流。`zip!` 在顶部打包三个相互独立的分配。四个 `and_then` 阶段依次串联：GEMM、MatVec、ReLU 和 D2H。类型通过 `value()` 元组在各阶段之间流动。整条链是一个 `DeviceOperation` —— 在 `.sync()` 或 `.await` 之前什么都不会执行。*

下面是根据 `async_mlp` 示例简化的版本，演示了所有组合子协同工作。这个函数返回单个 `DeviceOperation`，描述一个批次的整个前向传播：

```rust
use cuda_async::simt::launch::{AsyncKernelLaunchBuilder, OwnedAsyncKernelLaunch};

fn forward_pass(
    batch: Vec<f32>,
    w0: Arc<DeviceBox<[f32]>>,
    w1: Arc<DeviceBox<[f32]>>,
    module: Arc<CudaModule>,
) -> impl DeviceOperation<Output = Vec<f32>> {
    // 准备：分配输入 + 临时缓冲区（相互独立 → 用 zip 打包）
    zip!(h2d(batch), zeros(DIM * DIM), zeros(DIM))
        // 阶段 1：GEMM —— hidden = input × W0
        .and_then(move |(input, hidden, output)| {
            let func = module.load_function("sgemm_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((DIM as u32, DIM as u32, DIM as u32,
                               1.0f32,
                               input.cu_deviceptr(), input.len() as u64,
                               w0.cu_deviceptr(), w0.len() as u64,
                               0.0f32,
                               hidden.cu_deviceptr(), hidden.len() as u64));
            // 安全性：packet/config 与 sgemm_naive 匹配，调度器使用该模块的
            // 上下文，且 owned 包装器保留其各项分配。
            let launch = unsafe { builder.finalize_unchecked(gemm_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (input, w0, hidden));
            // kernel 返回 ()。把仍然需要的缓冲区向前传递。
            launch.and_then(move |(_input, _w0, hidden)| {
                value((hidden, output, w1, module))
            })
        })
        // 阶段 2：MatVec —— output = hidden × W1
        .and_then(move |(hidden, output, w1, module)| {
            let func = module.load_function("matvec_naive").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((DIM as u32, DIM as u32,
                               hidden.cu_deviceptr(), hidden.len() as u64,
                               w1.cu_deviceptr(), w1.len() as u64,
                               output.cu_deviceptr(), output.len() as u64));
            // 安全性：packet/config 与 matvec_naive 匹配，调度器使用该模块的
            // 上下文，且 owned 包装器保留其各项分配。
            let launch = unsafe { builder.finalize_unchecked(matvec_cfg) };
            let launch = OwnedAsyncKernelLaunch::new(launch, (hidden, w1, output));
            launch.and_then(move |(_hidden, _w1, output)| {
                value((output, module))
            })
        })
        // 阶段 3：ReLU —— result = max(0, output)
        .and_then(move |(output, module)| {
            let func = module.load_function("relu").unwrap();
            let mut builder = AsyncKernelLaunchBuilder::new(Arc::new(func));
            builder.push_args((output.cu_deviceptr(), output.len() as u64,
                               output.cu_deviceptr(), output.len() as u64));
            // 安全性：packet/config 与 relu 匹配，调度器使用该模块的
            // 上下文，且 owned 包装器保留 output。
            let launch = unsafe { builder.finalize_unchecked(relu_cfg) };
            OwnedAsyncKernelLaunch::new(launch, output)
        })
        // 阶段 4：把结果下载到宿主
        .and_then(d2h)
}
```

研究一下数据如何在这条流水线中流动：

1. `zip!` 产生三个 `DeviceBox` 句柄。
2. 第一个 `and_then` 消费这三者，启动 GEMM，并通过 `value()` 把它仍然需要的缓冲区打包成元组。
3. 第二个 `and_then` 解包该元组，启动 MatVec，只向前传递下一阶段需要的东西。
4. 第三个 `and_then` 原地启动 ReLU。
5. 最后一个 `and_then` 调用 `d2h` 把结果拷贝到宿主。

整个函数返回 `impl DeviceOperation<Output = Vec<f32>>`。什么都还没有执行。你可以 `.sync()` 它、`.await` 它，或者把它交给 `tokio::spawn`，让调度策略去决定使用哪个流。

`AsyncKernelLaunchBuilder` 是惰性的，可以安全地填充。只有 `finalize_unchecked(raw_config)` 会跨过 unsafe 边界并创建一个可运行的操作。`OwnedAsyncKernelLaunch` 让每个阶段的缓冲区保持存活，直到该次启动把它们返回。应用代码应优先使用带 `PreparedLaunch<K>` 的生成式 owned-async 方法；当模块是按名字加载、且没有可用的生成合约时，上面这种低层形式就很有用。

## 速查表

| 组合子                    | 它做什么                                            | 何时使用                        |
|:------------------------------|:--------------------------------------------------------|:--------------------------------------|
| `.and_then(f)`                | 运行自身，然后在同一流上运行 `f(result)`           | 有依赖的顺序阶段           |
| `.and_then_with_context(f)`   | 类似 `and_then`，但 `f` 还能拿到 `ExecutionContext`| 当闭包需要原始流时 |
| `.apply(f)`                   | `and_then` 的别名                                    | 哪个在语境中读起来更顺就用哪个     |
| `zip!(a, b)`                  | 运行两者，返回 `(A, B)`                               | 相互独立的准备步骤               |
| `zip!(a, b, c)`               | 运行全部三者，返回 `(A, B, C)`                       | 同上，用于三个操作数              |
| `.arc()`                      | 把输出包装进 `Arc<T>`                                 | 跨流水线共享结果      |
| `unzip!(op)`                  | 把产生元组的操作拆分成两个                     | 分岔的下游链           |
| `value(x)`                    | 把 `x` 包装成空操作                           | 把宿主数据喂入流水线      |
| `with_context(f)`             | 推迟构造直到流已知            | 包装原始 CUDA 驱动调用        |

> **参见**
>
> [并发执行](concurrent-execution.md)一章展示了如何用 `tokio::spawn` 并发运行多条流水线，而[调度与流](scheduling-and-streams.md)一章解释了调度策略如何把流水线分配到各个 CUDA 流上。
