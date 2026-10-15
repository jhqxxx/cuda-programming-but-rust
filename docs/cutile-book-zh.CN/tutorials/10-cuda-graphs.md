# 10. CUDA 图

CUDA 图让你把整个 GPU 工作负载捕获一次，然后重放很多次，从而消除逐次启动的开销。本教程用 DeviceOp 组合子搭出一个多层前向传播，把它捕获成一张 CUDA 图，再在 token 循环里重放它。

---

## 动机

每次 kernel 启动都涉及 CPU 侧的工作：选择流、准备参数、调用驱动。对于那些会重复同一张操作图的工作负载（例如 transformer 的前向传播），这种逐次启动的开销可能占主导 —— 在 batch 很小、kernel 相对于其启动成本跑得很快时尤其如此。

CUDA 图把整个操作序列记录一次，然后用**一次驱动调用**重放它。GPU 会提前看到完整的图，从而能更激进地调度内部工作。

```text
Without graphs:                     With graphs:

  CPU: launch → wait → launch →      CPU: launch_graph → wait
       wait → launch → wait               (single call)
  GPU: ████   ████   ████            GPU: ████████████████
       gaps between kernels                no gaps
```

---

## 模型

我们要搭一个极简的 transformer 风格层堆叠：每一层执行 RMSNorm → Q 投影（matvec）→ O 投影（matvec）→ 残差相加。隐状态依次穿过所有层。

```text
input
  │
  ├─ Layer 0: RMSNorm → Q MatVec → O MatVec → Add(residual, hidden)
  │
  ├─ Layer 1: RMSNorm → Q MatVec → O MatVec → Add(residual, hidden)
  │
  └─ … (n_layers)
```

### kernel

三个 cutile kernel 承担计算。每个都遵循"输出优先"的约定（`&mut Tensor` 作为第一个参数）：

```rust
#[cutile::module]
mod kernels {
    use cutile::core::*;

    /// RMS 归一化：out = rms_norm(x) * w
    #[cutile::entry()]
    pub fn rms_norm<const D: i32, const BS: i32>(
        out: &mut Tensor<f32, { [1, D] }>,
        x: &Tensor<f32, { [-1, D] }>,
        w: &Tensor<f32, { [D] }>,
        eps: f32,
    ) { /* tile 级实现 */ }

    /// 矩阵-向量乘法：out = x @ w^T
    #[cutile::entry()]
    pub fn matvec<const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<f32, { [BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        w: &Tensor<f32, { [-1, K] }>,
    ) { /* tile 级实现 */ }

    /// 逐元素加法：out = a + b
    #[cutile::entry()]
    pub fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) { /* tile 级实现 */ }
}
```

### 模型状态

权重在所有前向调用之间共享。缓冲区只分配一次，每个 token 都复用 —— 图重放时写回同一块内存：

```rust
struct LayerWeights {
    norm_w: Arc<Tensor<f32>>,
    wq: Arc<Tensor<f32>>,
    wo: Arc<Tensor<f32>>,
}

struct LayerBuffers {
    norm: Arc<Tensor<f32>>,
    q: Arc<Tensor<f32>>,
    o: Arc<Tensor<f32>>,
    residual: Arc<Tensor<f32>>,
}
```

---

## 构建惰性图

把前向传播构建成一个惰性的 `DeviceOp`，以便做图捕获：

```rust
fn build_forward(
    cfg: &Config,
    weights: &[LayerWeights],
    input: Arc<Tensor<f32>>,
    buffers: Vec<LayerBuffers>,
) -> (impl DeviceOp<Output = ()>, SharedDeviceOp<Tensor<f32>>) {
    let mut ops: Vec<BoxedDeviceOp<()>> = Vec::with_capacity(buffers.len());
    let mut hidden: SharedDeviceOp<Tensor<f32>> =
        cuda_async::device_operation::shared(input);

    for (w, bufs) in weights.iter().zip(buffers) {
        // RMSNorm：hidden(1,d) × norm_w → norm(1,d)
        let norm = rms_norm(
            bufs.norm.try_partition([1, cfg.d]).expect("sole buffer owner"),
            hidden.clone().reshape(&[1, cfg.d]),
            w.norm_w.clone(),
            cfg.eps,
        )
        .generics(cfg.rms_generics())
        .first()
        .unpartition()
        .shared();

        // Q 投影：norm @ wq^T → q
        let q = matvec(
            bufs.q.try_partition([cfg.bn]).expect("sole buffer owner"),
            norm.clone(),
            w.wq.clone(),
        )
        .generics(cfg.mv_generics())
        .first()
        .unpartition()
        .shared();

        // O 投影：q @ wo^T → o
        let o = matvec(
            bufs.o.try_partition([cfg.bn]).expect("sole buffer owner"),
            q.clone().reshape(&[1, cfg.d]),
            w.wo.clone(),
        )
        .generics(cfg.mv_generics())
        .first()
        .unpartition()
        .shared();

        // 残差相加：hidden + o → residual
        let residual = add(
            bufs.residual.try_partition([cfg.block]).expect("sole buffer owner"),
            hidden.clone().reshape(&[cfg.d]),
            o.clone(),
        )
        .first()
        .unpartition()
        .shared();

        hidden = residual.clone();

        // 把该层的工作留在图里；数据由那些固定缓冲区承载。
        ops.push(
            zip!(norm, q, o, residual)
                .map(|_| ())
                .boxed(),
        );
    }

    // DeviceOpVec 产生 Vec<()>；把它丢弃，这样捕获结果就是 CudaGraph<()>。
    (DeviceOpVec::new(ops).map(|_| ()), hidden)
}
```

`build_forward` 里这些组合子各自的角色：

- **`.shared()`** —— 每个中间结果都被共享，以便既能喂给下一个 kernel，又能进入最终的缓冲区收集。底层计算只执行一次；下游消费者拿到的是 `Arc::clone()`。
- **`.first()`** —— kernel 启动返回一个包含所有参数的元组，`.first()` 只取出输出（也就是 `&mut Tensor` 参数）。
- **`try_partition`** —— 通过证明独占所有权（Arc 引用计数 == 1）把 `Arc<Tensor<T>>` 转换成 `Partition`。当该 Arc 还有其它持有者时返回 `Err`；上面的 `.expect(...)` 会把它变成 panic。（`shared(...)` 是 `cuda_async::device_operation::shared`；它不在 prelude 里。）
- **`DeviceOpVec`** —— 收集各层图工作的装箱操作。它的输出是 `Vec<()>`，所以末尾的 `.map(|_| ())` 才让捕获出来的图成为 `CudaGraph<()>`。

---

## 捕获图

`.graph_on(stream)` 会在 CUDA 的流捕获模式下把该操作执行一次，把所有 GPU 工作记录进一张可重放的图：

```rust
let input: Tensor<f32> = api::rand([cfg.d], None).sync_on(&stream)?;
let input_arc: Arc<Tensor<f32>> = unsafe { input.into_shared_alias() };
let buffers: Vec<_> = (0..cfg.n_layers)
    .map(|_| LayerBuffers::allocate(cfg.d, &stream))
    .collect::<Result<_, _>>()?;
unsafe { stream.synchronize() }?;

// 构建惰性图（没有 GPU 工作）。
let (forward_op, output_shared) = build_forward(&cfg, &weights, input_arc, buffers);

// 捕获：执行一次、记录全部内容、返回 CudaGraph。
let mut graph = forward_op.graph_on(stream.clone())?;

// 从这次捕获执行中取出输出。
let output = output_shared.sync_on(&stream)?;
```

捕获之后：
- `graph` 持有已记录下来的 CUDA 图，可以重放。
- `input` 是那个固定的输入缓冲区，它的指针已经被烘焙进图里。
- `output` 指向最后一层的残差缓冲区，图在重放时会改写它。

---

## Module 模式

实现 `Module`，通过 `forward` 来调用图：

```rust
trait Module {
    type Input: Send + ?Sized;
    type Output: Send;
    fn forward(&mut self, input: &Self::Input) -> Result<Self::Output, DeviceError>;
}

struct GraphModel {
    graph: CudaGraph<()>,
    input: Tensor<f32>,
    output: Arc<Tensor<f32>>,
}

impl Module for GraphModel {
    type Input = Tensor<f32>;
    type Output = Arc<Tensor<f32>>;

    fn forward(&mut self, input: &Self::Input) -> Result<Self::Output, DeviceError> {
        // 把新的 embedding 拷贝进那个已烘焙好的输入缓冲区。
        self.graph.update(api::memcpy(&mut self.input, input))?;
        // 用一次驱动调用重放整个前向传播。
        self.graph.launch().sync_on(self.graph.stream())?;
        Ok(self.output.clone())
    }
}
```

每次 `forward` 调用：

1. **`graph.update(memcpy(…))`** —— 把新的输入数据拷进预分配的输入缓冲区。这会在图的流上执行，因此会在图启动之前完成。
2. **`graph.launch().sync_on(…)`** —— 重放所有已捕获的 kernel。GPU 看到完整的操作序列，可以激进地调度。
3. **返回 `output.clone()`** —— 输出的 `Arc` 指向图写入的同一块设备内存。不需要拷贝。

---

<a id="putting-it-together"></a>
## 推理循环

```rust
fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let cfg = Config { d: 2048, n_layers: 22, block: 128, bn: 16, bk: 16, eps: 1e-5 };

    // 分配权重（本示例用随机值）。
    let weights: Vec<LayerWeights> = (0..cfg.n_layers)
        .map(|_| Ok(LayerWeights {
            norm_w: api::rand([cfg.d], None).sync_on(&stream)?.into(),
            wq: api::rand([cfg.d, cfg.d], None).sync_on(&stream)?.into(),
            wo: api::rand([cfg.d, cfg.d], None).sync_on(&stream)?.into(),
        }))
        .collect::<Result<_, Error>>()?;

    // 构建并捕获。
    let mut model = GraphModel::new(&cfg, &weights, &stream)?;

    // 推理循环 —— 每次调用都是一次单独的图启动。
    let n_tokens = 512;
    for _ in 0..n_tokens {
        let embedding: Tensor<f32> = api::rand([cfg.d], None).sync_on(&stream)?;
        let _output = model.forward(&embedding)?;
    }

    Ok(())
}
```

---

## 替代方案：`CudaGraph::scope`

上面的组合子做法对预分配缓冲区要求 `Arc<Tensor<T>>` + `try_partition`。`CudaGraph::scope` 提供了一种使用 `&mut` 借用的命令式替代方案：

```rust
let mut input: Tensor<f32> = api::ones::<f32>(&[cfg.d]).sync_on(&stream)?;
let mut buffers: Vec<LayerBuffers> = /* 预分配 */;

let graph = CudaGraph::scope(&stream, |s| {
    for (w, bufs) in weights.iter().zip(buffers.iter_mut()) {
        let hidden_2d = input.view(&[1, cfg.d])?;

        s.record(rms_norm(
            (&mut bufs.norm).partition([1, cfg.d]),
            &hidden_2d,
            &w.norm_w,
            cfg.eps,
        ).generics(cfg.rms_generics()))?;

        s.record(matvec(
            (&mut bufs.q).partition([cfg.bn]),
            &bufs.norm,
            &w.wq,
        ).generics(cfg.mv_generics()))?;

        let q_2d = bufs.q.view(&[1, cfg.d])?;
        s.record(matvec(
            (&mut bufs.o).partition([cfg.bn]),
            &q_2d,
            &w.wo,
        ).generics(cfg.mv_generics()))?;

        s.record(add(
            (&mut bufs.residual).partition([cfg.block]),
            &input,
            &bufs.o,
        ))?;

        s.record(api::memcpy(&mut input, &bufs.residual))?;
    }
    Ok(())
})?;

graph.launch().sync_on(&stream)?;
```

| | 组合子（`.graph()`） | 作用域（`CudaGraph::scope`） |
|---|---|---|
| 缓冲区所有权 | `Arc<Tensor<T>>` + `try_partition` | `&mut Tensor<T>` 借用 |
| 先写后读 | 通过 `.shared()` + 克隆 | 通过 `record()` 释放借用 |
| 失败方式 | `try_partition` 返回运行时 `Err`（引用计数 != 1） | 编译错误（借用冲突） |
| 可组合性 | 可与 `.then()`、`.map()` 串联 | 命令式顺序代码 |

`s.record(op)` 只接受实现了 `GraphNode` 的操作 —— kernel 启动和 `memcpy`。分配类操作（`api::zeros`、`dup` 等）会在编译期被拒绝，因为它们的地址在图重放时可能发生变化。

---

## 适用场景

| 场景 | 用 CUDA 图？ | 原因 |
|---|---|---|
| 反复执行同一张操作图很多次 | **是** | 摊薄捕获成本；消除逐次启动开销 |
| 每次迭代形状都变 | 否 | 已捕获的图把 tensor 维度烘焙固定了 |
| 每次迭代控制流都变 | 否 | 已捕获的图把分支结构烘焙固定了 |
| 迭代次数很少 | 也许 | 捕获成本（约等于一次执行）必须被摊薄 |
| 对单个 kernel 做性能分析 | 否 | 图重放会显示为一个单独的事件 |

---

## 要点

| 概念 | 含义 |
|---|---|
| **`DeviceOp` 图** | GPU 工作的惰性组合 —— 在被驱动之前不执行 |
| **`.graph_on(stream)`** | 把整个操作捕获成可重放的 `CudaGraph` |
| **`CudaGraph::scope`** | 使用 `&mut` 借用的命令式图捕获 |
| **`graph.launch()`** | 返回一个重放全部已捕获工作的 `DeviceOp` —— 用 `.sync_on()`、`.sync()` 或 `.await` 执行 |
| **`graph.update(op)`** | 在重放之前于图的流上运行一个 DeviceOp（例如用 memcpy 写入新输入） |
| **预分配缓冲区** | 图写入固定内存；通过 `memcpy` 改变输入 |
| **图中的 `.shared()`** | 每个中间结果在捕获期间只执行一次；克隆共享同一结果 |

> **注意**：`build_forward` 中的权重 tensor 以 `Arc<Tensor<T>>` 传入，因为它们要通过 `.clone()` 在各层之间共享。在只读权重（不跨 spawn 出来的任务共享）的同步上下文里，你也可以改成传 `&Tensor<T>` —— 借用检查器会保证权重比图捕获活得更久。关于借用输入的细节，见[教程 7](07-intro-to-async.md#borrowed-inputs-and-spawn-safety)。

---

### 练习 1：再加一张图

再捕获一张图来计算反向传播（或者它的简化版）。你会怎样安排前向图与反向图的先后顺序？

### 练习 2：测量加速比

在 `graph.launch()` 与非图路径（每次迭代都重建并执行 `DeviceOp`）两侧加上计时。加速比随 `n_layers` 如何变化？

### 练习 3：动态输入形状

当前做法在捕获时就把 tensor 维度烘焙固定了。要支持可变的序列长度，需要改动什么？（提示：考虑为常见的尺寸各捕获一张图。）

---

## 完整参考示例

基于作用域的参考示例（带基准测试）：

```bash
cargo run -p cutile-examples --example cuda_graphs
```

DeviceOp 组合子版本：

```bash
cargo run -p cutile-examples --example cuda_graphs_deviceop
```

---

## 参见

- [设备操作](../guide/device-operations.md) —— CUDA 图与同步、异步执行方式之间的关系
- [宿主 API：CUDA 图集成](../reference/host-api.md#cuda-graph-integration) —— `.graph_on(stream)` 与 `CudaGraph::scope` 的签名
