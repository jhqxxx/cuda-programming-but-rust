# 调度与流

前面几章建立了一套描述 GPU 工作的词汇 —— `DeviceOperation`、`and_then` 链、`zip!` 打包。但描述不等于执行。在某个时刻，菜谱必须送到厨房。本章讲的就是厨房：CUDA stream 是什么，cuda-oxide 的调度策略如何把工作分配给它们，以及当一个 `DeviceOperation` 变成正在运行的 `DeviceFuture` 时幕后发生了什么。

> **参见**
>
> [CUDA Programming Guide -- Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/#streams)，了解调度策略所构建于其上的底层 CUDA 流语义。

## 结账通道

把一个 CUDA stream 想象成超市的**结账通道**。每条通道按顺序处理顾客 —— 排在前面的人先被服务。但多条通道彼此独立地运行，所以 1 号通道在给一位顾客结账时，2 号通道可以为另一笔订单装袋。商店的吞吐更高，因为不同通道上的工作在时间上重叠了。

GPU 的工作方式相同。**流**是一个按顺序排队的操作队列。在单个流内，一切都顺序执行 —— kernel A 在 kernel B 开始之前完成。但如果你把 kernel A 放到流 0、把 kernel B 放到流 1，它们就可以在硬件上重叠：

```text
Stream 0:  ┌─── Kernel A ───┐
           └────────────────┘
Stream 1:       ┌─── Kernel B ───┐
                └────────────────┘

           ◄─── time ────────────────►
```

用一个流时，一切都是串行的。用多个流时，相互独立的工作发生重叠，GPU 保持更繁忙。这是 GPU 上并发的基本机制。

## 为什么你很少直接接触流

在 CUDA C++ 中，程序员创建流、决定每个操作进入哪个流，并在一个流上的工作依赖另一个流上的结果时手动插入事件。这很强大但很繁琐，而且它把每个函数都耦合到某个特定的并发策略上。

cuda-oxide 插入了一层间接：**调度策略**。你不再自己选择流，而是把你的 `DeviceOperation` 交给策略，由它替你挑选流。这意味着同一条流水线可以在单个流上运行（用于调试）、在一个四流池上运行（用于吞吐），或者在你自写的自定义策略上运行 —— 全都无需改动流水线代码。

## `SchedulingPolicy` trait

调度策略只回答一个问题：「给定这个操作，它应该在哪个流上运行？」该 trait 有三个方法：

- **`init`** —— 在启动时调用一次，用于创建 CUDA 流。
- **`schedule`** —— 挑选一个流，并把操作包装成 `DeviceFuture` 以供 `.await` 使用。
- **`sync`** —— 挑选一个流，执行该操作，并阻塞直到它完成。

策略是 `Sync` 的，意味着单个实例会在一台设备上的所有操作之间共享。流的选择必须是线程安全的。

## `StreamPoolRoundRobin` —— 默认策略

当你调用 `init_device_contexts(0, 1)` 时，cuda-oxide 会创建一个带有四个 CUDA 流的 `StreamPoolRoundRobin`。每当一个操作被调度时，一个原子计数器前进，池中的下一个流被选中：

```text
Operation 1  →  Stream 0  ──► ████████
Operation 2  →  Stream 1  ──►    ████████         (overlaps 1)
Operation 3  →  Stream 2  ──►       ████████      (overlaps 1, 2)
Operation 4  →  Stream 3  ──►          ████████
Operation 5  →  Stream 0  ──►                ████████  (waits for 1)
```

选择是无锁的 —— 对一个 `AtomicUsize` 做一次 `fetch_add`，再对池大小取模。与 GPU 工作的开销相比，这点开销可以忽略不计。

四个流是个不错的默认值。它给 GPU 足够的在飞工作，以便把 kernel 执行与内存传输重叠起来，又不会带来过度的上下文切换开销。对大多数工作负载来说，你永远不需要考虑它 —— 策略就是能正常工作。

### 轮转策略大放异彩的场合

- **批量推理：** 每个批次都是一条独立的流水线。轮转把批次分配到各个流上，使计算重叠。
- **计算与传输混合：** 当一个流运行 kernel 时，另一个流在拷贝数据。GPU 的拷贝引擎与计算单元同时工作。
- **大量小 kernel：** 重叠启动开销减小了 kernel 之间的间隔，让 GPU 保持更繁忙。

### 何时该重新考虑

- **依赖密集的链：** 如果你构建的是一条单独的 `and_then` 链（比如上一章中的前向传播），这条链终究还是完全在一个流上运行。只有当你要调度*多个相互独立*的操作时，轮转才起作用。
- **超大 kernel：** 单个能把 GPU 跑满的 kernel 从多流调度中得不到任何好处。多出来的流只会闲置。

## `SingleStream` —— 一条通道，严格顺序

> **注意**
>
> `SingleStream` 已在调度内部实现，但**目前尚未接入 `GlobalSchedulingPolicy`**，也未通过 `init_device_contexts` 暴露。默认设置总是使用 `StreamPoolRoundRobin`。本节描述的是面向未来 API 表面所做的设计意图。

为了调试，或者当你需要保证*所有*操作之间的顺序时，`SingleStream` 会把一切都路由到同一个流上。每个操作都能看到此前每一个操作的结果，从而排除任何与流相关的并发 bug：

```text
Operation 1  →  Stream 0  ──► ████████
Operation 2  →  Stream 0  ──►          ████████   (waits for 1)
Operation 3  →  Stream 0  ──►                  ████████
```

> **提示**
>
> 如果你怀疑你的 GPU 流水线中有并发 bug，切换到 `SingleStream` 是最快的检查方式。如果 bug 消失了，那就是不同流上操作之间缺少依赖。如果它依然存在，问题就在别处。

## 搭建运行时

在任何异步操作能够运行之前，你要初始化线程局部的设备上下文：

```rust
use cuda_async::simt::device_context::init_device_contexts;

init_device_contexts(0, 1)?;
```

第一个参数是默认的 GPU 序号；第二个是要管理的设备数量。在底层，这会注册一个线程局部对象，它在首次使用时为每台设备惰性地创建一个 `StreamPoolRoundRobin`。该池默认持有四个流。

在你的程序开始时调用它一次，在任何 `.sync()` 或 `.await` 之前。在同一线程上调用两次会返回错误。

## 当你 `.await` 时会发生什么

![左：DeviceFuture 的三态状态机。在第一次轮询时，提交 GPU 工作和一个 `cuLaunchHostFunc` 回调 —— 然后返回 `Poll::Pending`。当 GPU 完成时，回调设置一个 `AtomicBool` 并唤醒该任务。在第二次轮询时，结果被交付。右：`.sync()`（线程全程阻塞）与 `.await`（GPU 工作时线程去运行其他任务）的对比。](images/device-future-polling.svg)

*左：DeviceFuture 的三态状态机。在第一次轮询时，提交 GPU 工作和一个 `cuLaunchHostFunc` 回调 —— 然后返回 `Poll::Pending`。当 GPU 完成时，回调设置一个 `AtomicBool` 并唤醒该任务。在第二次轮询时，结果被交付。右：`.sync()`（线程全程阻塞）与 `.await`（GPU 工作时线程去运行其他任务）的对比。*

下面是一个操作从构造到完成的完整旅程：

```text
module.kernel_async(&prepared, ...)  ← build a checked recipe (no GPU work)
        │
        ▼
  AsyncKernelLaunch            ← a DeviceOperation, lazy and stream-agnostic
        │
        │  .await
        ▼
  IntoFuture::into_future()    ← scheduling policy picks a stream
        │
        ▼
  DeviceFuture                 ← bound to a stream, ready for polling
        │
        │  first poll()
        ▼
  execute() on stream 2        ← GPU work is submitted
  cuLaunchHostFunc on stream 2 ← host callback enqueued after the kernel
  return Poll::Pending
        │
        │  ... GPU is working, host thread is free ...
        │
        │  callback fires on a CUDA driver thread
        │  → sets AtomicBool, wakes AtomicWaker
        │
        │  second poll()
        ▼
  return Poll::Ready(Ok(()))   ← result delivered to the caller
```

关键的洞见是：在第一次 `poll()` 与回调之间，**没有任何宿主线程被占用**。异步运行时挂起该任务并运行其他任务。GPU 完成时通过 `cuLaunchHostFunc` 回调通知运行时。这就是为什么对并发工作负载而言 `.await` 比 `.sync()` 更具扩展性 —— 你可以有几十个在飞操作，而不必为每一个占用一个线程。

## 手动控制流

大多数时候，调度策略替你处理流。但在某些情况下你需要直接控制 —— 与期望特定流的 CUDA 库互操作、对计算与传输做细粒度重叠，或者单独对某个 kernel 做性能分析。cuda-oxide 为这些场景暴露了完整的流 API。

### 创建流

```rust
let ctx = CudaContext::new(0)?;
let default = ctx.default_stream();  // 每个上下文的默认（空）流
let custom  = ctx.new_stream()?;     // 一个新的非阻塞流
```

默认流在 CUDA 中有特殊的同步语义（它会隐式地与大多数其他流串行化）。由 `new_stream()` 创建的非阻塞流没有这个约束，这也是调度策略只使用它们的原因。

### Fork 与 join

一个常见模式是从父流 fork 出一个子流，在子流上运行独立的工作，再把结果 join 回来。`fork` 创建一个新流，它对父流的当前位置有隐式依赖 —— 在父流上所有先前的工作完成之前，子流不会开始。`join` 做相反的事：父流在继续之前等待子流完成。

```rust
let main = ctx.default_stream();

// 在 main 上上传数据
let buf_a = DeviceBuffer::from_host(&main, &data_a)?;
let buf_b = DeviceBuffer::from_host(&main, &data_b)?;

// Fork：子流能看到这些上传
let child_1 = main.fork()?;
let child_2 = main.fork()?;

// 安全性：cfg 与 process 的索引/资源以及两个缓冲区的边界匹配。
// 并行运行独立的工作。
unsafe {
    module.process(&child_1, cfg, &mut buf_a)?;
    module.process(&child_2, cfg, &mut buf_b)?;
}

// Join：main 等待两个子流
main.join(&child_1)?;
main.join(&child_2)?;

// 安全性：cfg 与 combine 的索引/资源以及两个输入的边界匹配。
unsafe { module.combine(&main, cfg, &buf_a, &buf_b) }?;
```

这对应的 GPU 时间线看起来像：

```text
main:      ██ upload_a ██ upload_b ██ ──fork──────────────── join ──► ██ combine ██
                                         |                    ^  ^
child_1:                                 └─► ██ process_a ██ ─┘  |
                                         |                       |
child_2:                                 └─► ██ process_b ██ ────┘
```

在底层，`fork` 和 `join` 使用 CUDA 事件 —— 在一个流上 `cuEventRecord`，在另一个流上 `cuStreamWaitEvent`。这些事件是 GPU 侧的同步令牌；在 fork 或 join 期间没有宿主线程阻塞。

### 用事件做细粒度顺序控制

当 `fork`/`join` 太粗粒度时，你可以直接使用事件来在不同流中的特定点之间建立顺序：

```rust
// 在 kernel 完成后于流 A 上记录一个事件
let event = stream_a.record_event(None)?;

// 流 B 等待那个特定的点，然后才继续
stream_b.wait(&event)?;
```

CUDA 事件不属于任何流 —— 它是一个独立的同步令牌。`record` 把它打在一个流上的特定点；`wait` 把一条依赖插入另一个流。事件就是会合点：在流 A 上的事件触发之前，流 B 上 `wait` 之后的一切都不会运行。

事件也是测量 GPU 执行时间的标准方式。注意计时要求事件**不带** `CU_EVENT_DISABLE_TIMING` 创建 —— 传入显式标志以启用计时：

```rust
// bindgen 把 CUDA 枚举展平了，所以这个常量是 crate 根级别的条目，
// 以它的枚举命名，而不是某个模块内的变体。
use cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT as CU_EVENT_DEFAULT;

let start = stream.record_event(Some(CU_EVENT_DEFAULT))?;
// 安全性：config 与参数满足 my_kernel 的要求。
unsafe { module.my_kernel(&stream, config) }?;
let end = stream.record_event(Some(CU_EVENT_DEFAULT))?;
end.synchronize()?;
println!("Kernel took {:.2} ms", start.elapsed_ms(&end)?);
```

这测量的是实际 GPU 时间，而不是宿主侧调度开销。`record_event(None)` 默认创建禁用计时的事件，这类事件不能与 `elapsed_ms` 一起使用。

> **提示**
>
> 对于日常的异步流水线，你永远不需要手动创建流、事件或 fork/join。调度策略和 `and_then` 链会自动处理顺序。手动 API 的存在是为了互操作、性能分析和高级优化。

## 选择合适的方式

| 情形                                     | 推荐方式                                 |
|:----------------------------------------------|:-----------------------------------------------------|
| 简单脚本，一个合约化 kernel          | `module.kernel_async(&prepared, ...).sync()`         |
| 多阶段流水线（GEMM → ReLU → D2H）      | `and_then` 链，策略挑选一个流            |
| 相互独立的批次并发运行      | 对每个批次 `tokio::spawn`，由轮转策略分配   |
| 调试疑似流顺序 bug     | 切换到 `SingleStream`                             |
| 与已有 CUDA 库互操作         | 用它们的流调用 `.sync_on(&stream)`                |
| 单独对某个 kernel 做性能分析       | 在启动前后插入显式事件                    |
| 最大吞吐                            | 用 Nsight Systems 做性能分析，调整池大小          |

> **参见**
>
> [CUDA Programming Guide -- Events](https://docs.nvidia.com/cuda/cuda-programming-guide/#events)，了解 CUDA 事件的完整规范。[并发执行](concurrent-execution.md)一章展示了这些调度概念在真实多批次工作负载中的应用。
