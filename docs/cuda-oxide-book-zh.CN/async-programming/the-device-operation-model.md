# DeviceOperation 模型

在[编写 GPU 程序](../gpu-programming/launching-kernels.md)一章中，你已经看到：带类型的同步启动会把工作入队到一个显式的流上，而带类型的异步启动则返回一个惰性句柄，把流的选择推迟到之后。本章深入探讨这个惰性句柄背后的抽象 —— **`DeviceOperation`** trait —— 并解释为什么把「GPU 应该做什么」与「它在哪个流上运行」解耦，是 cuda-oxide 中可组合异步 GPU 编程的基础。

> **参见**
>
> [CUDA Programming Guide -- Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/#asynchronous-concurrent-execution)，了解 `DeviceOperation` 所构建于其上的底层 CUDA 流与事件模型。

## 为什么需要惰性操作？

在 CUDA C++ 中，你通过创建多个 `cudaStream_t` 句柄，并显式地把 kernel 启动和内存拷贝放到它们上面来构建并发。程序员要在每个调用点决定使用哪个流。这就把 GPU 工作的*定义*与*调度决策*耦合在一起，使得在工作已经定义之后再组合和重排它们变得困难。

cuda-oxide 采取不同的做法。`DeviceOperation` 描述 GPU 工作，但不绑定到任何流。你可以用组合子（`and_then`、`zip!`）组合操作，把它们跨函数边界传递，把它们存进集合里，只在最后一刻才决定如何调度它们。这与 Rust 的 `Iterator` 背后的思想相同 —— 惰性地构建流水线，在调用点立即执行。

| 方式             | 由谁选择流  | 可组合？                      |
|:---------------------|:------------------|:---------------------------------|
| 带类型的同步启动    | 调用者            | 否，会立即入队         |
| 带类型的异步启动   | 调度策略 | 是，返回 `DeviceOperation`   |

![DeviceOperation 的生命周期。阶段 1：带类型的异步方法构建一个惰性配方（尚未产生 GPU 工作）。阶段 2：调度策略从它的池中选择一个流。阶段 3：`execute()` 提交 GPU 工作以及一个 `cuLaunchHostFunc` 回调。阶段 4：回调触发，唤醒异步运行时并交付结果。底部：从最简单（`.sync()`）到最手动（`async_on`）的四种执行方法。](images/device-operation-lifecycle.svg)

*DeviceOperation 的生命周期。阶段 1：带类型的异步方法构建一个惰性配方（尚未产生 GPU 工作）。阶段 2：调度策略从它的池中选择一个流。阶段 3：`execute()` 提交 GPU 工作以及一个 `cuLaunchHostFunc` 回调。阶段 4：回调触发，唤醒异步运行时并交付结果。底部：从最简单（`.sync()`）到最手动（`async_on`）的四种执行方法。*

## 配方与厨房

把一个 `DeviceOperation` 想象成一张**菜谱卡**。卡片描述了这道菜的每一步 —— 要把哪些食材组合起来、用什么温度、做多长时间 —— 但它并没有说*由哪个厨房*来烹制。你可以把卡片交给任何厨房、复印它、把两张卡片订在一起做成一套多道菜的套餐，或者把它归档留待以后使用。只有当有人走进厨房并开始照做时，这道菜才开始烹制。

在 cuda-oxide 的模型中：

- **菜谱**就是 `DeviceOperation` —— 对 GPU 工作的惰性描述。
- **厨房**就是 CUDA stream —— 工作真正在其中运行的顺序队列。
- **主厨**就是 `SchedulingPolicy` —— 决定由哪个厨房处理每份菜谱的逻辑。
- **菜肴**就是 `Output` —— 一切完成后你得到的结果。

正是这种分离让整个系统具备可组合性。你可以写一个函数，返回「上传数据、运行 GEMM、应用 ReLU」的菜谱，而不必关心它会在哪个流上执行。调用者可以继续往菜谱上串接更多步骤，在指定的流上运行它，或者把它交给调度策略然后就走开。

## 你的第一次异步启动

创建 `DeviceOperation` 最简单的方式是使用生成的 `{kernel}_async` 方法。它看起来和同步方法一样，只是没有 stream 参数，并且它返回的是菜谱而不是立即烹制：

```rust
use cuda_async::simt::device_context::init_device_contexts;
use cuda_core::simt::LaunchConfig;

// 一次性初始化：为调度创建一个流池
init_device_contexts(0, 1)?;

// 构建菜谱（此时尚无 GPU 工作）
let module = kernels::load_async(0)?;
// 安全性：这是一维的，缓冲区包含 1024 个元素，且 module/scheduler
// 都使用设备 0 的上下文。
let op = unsafe {
    module.vecadd_async(
        LaunchConfig::for_num_elems(1024),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )
}?;

// 现在开始烹制：挑选一个流、启动、等待结果
op.sync()?;
```

在创建 `op` 的那一刻，GPU 上什么都没有发生。该方法构建了一个 `AsyncKernelLaunch` 值，它记住了要调用哪个函数、传递哪些参数、如何配置 grid —— 但它并没有触碰任何流。它只是一张放在台面上的菜谱卡。

这里的 `unsafe` 是为了接受一个未经证明的原始 `LaunchConfig`，而不是因为惰性构建。合约化的 kernel 使用经过检查的 `PreparedLaunch<K>` 以及安全的生成异步方法：

```text
raw config -> unsafe async recipe
prepared K -> safe async recipe for K
```

当你调用 `.sync()` 时，调度策略从它的池中选择一个流，提交 kernel，并阻塞直到该流空闲。这一行就是菜谱变成已经做好的一道菜的地方。

## `DeviceOperation` 由什么构成

在幕后，`DeviceOperation` 是一个 trait。任何描述 GPU 工作的类型都可以实现它。该 trait 有一个必需方法和一个关联类型：

```rust
pub trait DeviceOperation: Send + Sized + IntoFuture {
    type Output: Send;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<Self::Output, DeviceError>;
}
```

**`Output`** 是该操作完成时产生的 Rust 值。对于一次 kernel 启动，它是 `()` —— kernel 运行是为了它对设备内存产生的副作用。对于一次设备到宿主的拷贝，它可能是 `Vec<f32>`。对于一次内存分配，它可能是拥有该指针的 `DeviceBox<[f32]>`。

**`execute`** 是真正发生 GPU 工作的地方。它接收一个 `ExecutionContext` —— 被分配到的那个厨房 —— 并把工作提交到其中的流。该方法是 `unsafe` 的，因为当它返回时 GPU 工作可能仍在进行中；调用者有责任在读取结果之前进行同步。

`Send` 约束意味着操作可以跨线程移动（这对 `tokio::spawn` 至关重要）。`IntoFuture` 约束则是让 `.await` 得以工作的原因 —— 稍后会详细说明。

你很少需要自己实现 `DeviceOperation`。这个 crate 提供了一组实现了它的类型，你可以用组合子来组合它们：

- **`AsyncKernelLaunch`** —— 由带类型的异步启动方法产生。启动一个 kernel。
- **`Value<T>`** —— 包装一个宿主侧的值。不产生 GPU 工作。立即返回 `T`。
- **`AndThen`** —— 串联两个操作：运行 A，把结果喂给 B。
- **`Zip`** —— 运行两个操作，并把两个结果作为元组返回。
- **`StreamOperation`** —— 把构造推迟到流已知时。

这些是每一条异步流水线的构建块。[组合子与组合](combinators-and-composition.md)一章会详细讲解每一个。

## `ExecutionContext` —— 流所在之处

当一份菜谱被执行时，它需要知道自己身处哪个厨房。`ExecutionContext` 携带了这一信息：

```rust
pub struct ExecutionContext {
    device: usize,              // 哪块 GPU
    cuda_stream: Arc<CudaStream>,   // 哪个流
    cuda_context: Arc<CudaContext>, // 哪个 CUDA 上下文
}
```

操作永远不会自己创建流。调度策略（在[调度与流](scheduling-and-streams.md)中介绍）创建 `ExecutionContext` 并将其传入 `execute`。这正是这种分离的核心：操作描述*做什么*，上下文提供*在哪里做*。

在 `execute` 的实现内部，你通过 `ctx.get_cuda_stream()` 访问流，通过 `ctx.get_cuda_context()` 访问 CUDA 上下文。对大多数操作来说这就足够了 —— 在流上入队一个 kernel 或一次内存拷贝，然后就完成了。

## 运行菜谱：四种执行方式

一旦你有了一个 `DeviceOperation`，就需要触发它。cuda-oxide 给你四条路径，从「全都帮我做完」一直到「我自己来搞定」。

### `.sync()` —— 阻塞并等待

最简单的选项。调度策略挑选一个流，运行该操作，并阻塞调用线程直到该流空闲：

```rust
let result: Vec<f32> = d2h_operation.sync()?;
```

这非常适合脚本、测试，以及任何你只想立刻拿到答案的地方。不需要 Tokio 运行时。

### `.await` —— 让出并恢复

在异步运行时内部，`.await` 做同样的事，但不会阻塞线程。它把操作转换成一个 `DeviceFuture`，提交 GPU 工作，并让出当前任务。当 GPU 完成时，它唤醒该任务并交付结果：

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_device_contexts(0, 1)?;

    let module = kernels::load_async(0)?;
    // 安全性：这是一维的，缓冲区包含 1024 个元素，且 module/scheduler
    // 都使用设备 0 的上下文。
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(1024),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .await?;

    Ok(())
}
```

当 GPU 在工作时，Tokio 运行时可以自由地去轮询其他任务 —— 没有线程会空闲地等待硬件。这是并发运行多条 GPU 流水线的关键，我们会在[并发执行](concurrent-execution.md)中探讨它。

### `.sync_on(&stream)` —— 由你选择流

当你需要指定的流时 —— 为了与已有的 CUDA 库互操作，或者为保证与该流上其他工作的顺序关系 —— `sync_on` 让你直接提供它，并阻塞直到完成：

```rust
let stream = ctx.new_stream()?;
operation.sync_on(&stream)?;
```

### `unsafe async_on(&stream)` —— 发射后不管

最手动的方式。它把工作提交到流上并立即返回，**不做**同步。调用者必须确保在读取结果之前该流已经同步。这在需要把许多操作批量放到一个流上、最后只做一次同步时很有用：

```rust
let stream = ctx.new_stream()?;
unsafe { op_a.async_on(&stream)? };
unsafe { op_b.async_on(&stream)? };
stream.synchronize()?;  // 现在两者都完成了
```

## 用 `value()` 提升宿主数据

流水线中并非每一步都涉及 GPU。有时你需要把宿主侧的一个值 —— 一个配置参数、一组维度、一个预先加载好的权重向量 —— 喂进一串设备操作中。`value()` 函数把任意 `Send` 类型包装成一个空操作的 `DeviceOperation`，它会立即返回该值：

```rust
use cuda_async::simt::device_operation::value;

let weights = vec![1.0f32; 1024];
let op = value(weights);  // impl DeviceOperation<Output = Vec<f32>>
```

单看 `value()` 似乎毫无用处。它的威力在组合中才显现出来。如果你要把一次宿主到设备的传输和一个配置结构体 zip 在一起，`value()` 能让配置融入这条流水线：

```rust
let (device_buf, config) = zip!(
    h2d(raw_data),
    value(ModelConfig { dim: 64, layers: 3 })
).sync()?;
```

`zip!` 的两条分支都必须是 `DeviceOperation`。`value()` 就是让宿主数据与设备工作配合良好的适配器。

## 用 `with_context` 与流对话

有些操作在执行时需要访问流本身。内存分配（`malloc_async`）、异步拷贝（`memcpy_htod_async`）以及事件记录都需要原始的 `CUstream` 句柄。但请记住 —— `DeviceOperation` 在被创建时并不知道它将在哪个流上运行。流是稍后由调度策略分配的。

`with_context` 弥合了这一鸿沟。它创建一个操作，其主体被推迟到 `ExecutionContext` 可用时才执行：

```rust
use cuda_async::simt::device_operation::{value, with_context};
use cuda_core::simt::memory::{malloc_async, memcpy_htod_async};

fn h2d(host_data: Vec<f32>) -> impl DeviceOperation<Output = DeviceBox<[f32]>> {
    with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let n = host_data.len();
        let num_bytes = n * std::mem::size_of::<f32>();
        unsafe {
            let dptr = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(dptr, host_data.as_ptr(), num_bytes, stream.cu_stream())
                .unwrap();
            value(DeviceBox::from_raw_parts(dptr, n, ctx.get_device_id()))
        }
    })
}
```

闭包接收 `ExecutionContext`，并且必须返回另一个 `DeviceOperation`。这里它返回一个 `Value`，包装了刚刚分配的设备指针。内部操作会在同一个流上立即执行。

这个模式 —— 用 `with_context` 包装原始驱动调用，最后返回 `value()` —— 就是你把任何底层 CUDA 操作变成可组合构建块的方式。`async_mlp` 示例用它来实现 `h2d`、`d2h` 和 `zeros` 辅助函数，它们可以干净地嵌入 `and_then` 链中。

> **提示**
>
> `with_context` 是需要 `CUstream` 的原始驱动调用的逃生舱。对于 kernel 启动，优先使用合约化的生成异步方法：它们负责参数编组和缓冲区生命周期，并接受经过检查的 `PreparedLaunch<K>`。未合约化的生成方法仍然需要 unsafe 的原始启动证明。

## GPU 如何告诉 Rust 它完成了

当你 `.await` 一个 `DeviceOperation` 时，底层会发生一些有趣的事情。该操作变成一个 `DeviceFuture` —— 一个实现了 Rust 的 `std::future::Future` 的类型 —— 而异步运行时会轮询它。但一个基于轮询的系统怎么知道*硬件*何时完成了它的工作呢？

答案是 `cuLaunchHostFunc`，一个 CUDA 驱动 API，它把一个宿主侧回调入队到流中。当该流上所有先前的 GPU 工作完成时，驱动会在一个驱动线程上调用该回调。cuda-oxide 用它来构建一座连接 CUDA 与 Rust 异步模型的零忙等桥梁。

`DeviceFuture` 是一个三态状态机：

```text
  Idle ───poll()───► Executing ───callback fires───► Complete
                         │                               │
                   (submit GPU work               (return result
                    + enqueue callback)             to the runtime)
```

在**第一次轮询**时，该 future：

1. 调用操作的 `execute()`，把 GPU 工作提交到流。
2. 在**同一个流**上、紧接 GPU 工作之后入队一个 `cuLaunchHostFunc` 回调。CUDA 保证流的顺序性：这个回调在 kernel 完成之前不会触发。
3. 返回 `Poll::Pending`。异步运行时挂起该任务并继续做别的事。

当 **GPU 完成**该 kernel 时，CUDA 驱动在一个驱动线程上调用宿主回调。回调设置一个 `AtomicBool` 标志并唤醒该任务的 `AtomicWaker`。异步运行时注意到这次唤醒，并重新轮询该 future。

在**第二次轮询**时，该 future 看到标志并返回 `Poll::Ready(Ok(result))`。任务带着该值恢复执行。

关键性质是：**当 GPU 工作时，没有任何宿主线程自旋或休眠**。异步执行器可以自由地运行其他任务 —— 包括其他流上的其他 `DeviceFuture`。这就是 cuda-oxide 在不给每个 GPU 操作专门分配一个线程的情况下实现真正并发执行的方式。

> **参见**
>
> [组合子与组合](combinators-and-composition.md)一章展示了如何用这些原语构建多阶段流水线，而[调度与流](scheduling-and-streams.md)解释了调度策略如何选择一个流，并创建把一切串联起来的 `ExecutionContext`。
