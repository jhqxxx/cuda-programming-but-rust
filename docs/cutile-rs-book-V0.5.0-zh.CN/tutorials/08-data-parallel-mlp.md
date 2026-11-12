# 8. 数据并行 MLP

每块 GPU 各持有一份这个单层 MLP 的副本，并处理各自独立的一批数据：

```text
Input → Linear → ReLU → Output

Where:
  Linear: hidden = input @ weights
  ReLU:   output = max(0, hidden)
```

---

<a id="the-code"></a>
## 多 GPU 前向传播

```rust
#[cutile::module]
mod data_parallel_module {

    use cutile::core::*;

    // 矩阵 × 矩阵。
    #[cutile::entry()]
    fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK, BN]);
        let pid_m = program_id(0);
        let pid_n = program_id(1);
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid_m, i]);
            let tile_y = part_y.load([i, pid_n]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    // 矩阵 × 向量。
    #[cutile::entry()]
    pub fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_y = y.partition(shape![BK]);
        let pid = program_id(0);
        let mut tile_z = z.load().reshape(shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid, i]);
            let tile_y = part_y.load([i]).reshape(shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z.reshape(shape![BM]));
    }

    // ReLU。
    #[cutile::entry()]
    fn relu<const D: i32>(input_output: &mut Tensor<f32, { [D] }>) {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, shape![D]);
        let input = input_output.load();
        input_output.store(max_tile(zero_tile, input));
    }
}

use cuda_async::error::DeviceError;
use data_parallel_module::{gemm, relu, matvec};

#[tokio::main]
async fn main() -> Result<(), DeviceError> {

    use std::sync::Arc;
    use cuda_async::device_operation::*;
    use data_parallel_module::{gemm, relu, matvec};
    use cutile::api;
    use cutile::tensor::{Unpartition, Partition, Tensor, ToHostVec};
    use cutile::tile_kernel::{PartitionOp, TileKernel};
    use cuda_async::device_context::global_policy;
    use cutile::api::dup;
    use tokio::task::JoinHandle;

    // 取得各设备的调度策略。
    let num_devices = 2;
    let devices = {
        let mut r = vec![];
        for _ in 0..num_devices {
            // 假装我们有多块设备……
            // 如果你确实有多块设备，请把 0 换成 i。
            r.push(global_policy(0)?);
        }
        r
    };

    let dim = 16;
    let block_dim = 4;
    let fully_connected_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let output_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let w0 = api::randn(0.0f32, 1.0, [dim, dim], None); // impl DeviceOp
    let w1 = api::randn(0.0f32, 1.0, [dim], None); // impl DeviceOp
    let w: (Arc<Tensor<f32>>, Arc<Tensor<f32>>) = zip!(w0.map(Into::into), w1.map(Into::into))
        .schedule(&devices[0])?.await?;
    let mut joins = vec![];
    for i in 1..num_devices {
        let w_copy = tokio::spawn(zip!(dup(&w.0).map(Into::into), dup(&w.1).map(Into::into)).schedule(&devices[i])?);
        joins.push(w_copy);
    }
    let mut model_weights = vec![w];
    for join in joins {
        model_weights.push(join.await.unwrap()?);
    }

    // 为每块设备上的每一批数据异步计算前向传播。
    let mut futures: Vec<JoinHandle<Result<Partition<Tensor<f32>>, cuda_async::error::DeviceError>>> = vec![];
    for i in 0..num_devices {
        let w = &model_weights[i];
        let (w0, w1) = (w.0.clone(), w.1.clone());
        let data = api::randn(0.0, 1.0, [dim, dim], None);
        // 统一的启动器：直接传入输出分区和各个输入。
        let out0 = api::zeros(&[dim, dim]).partition([block_dim, block_dim]);
        let out0 = gemm(out0, data, w0)
            .generics(fully_connected_layer.to_vec())
            .first()
            .unpartition();
        let out1 = api::zeros(&[dim]).partition([block_dim]);
        let out1 = matvec(out1, out0, w1)
            .generics(output_layer.to_vec())
            .first()
            .unpartition();
        let (out1,) = relu(out1.partition([block_dim])).unzip();
        futures.push(tokio::spawn(out1.schedule(&devices[i])?));
    }

    // 等待结果。
    let mut outputs: Vec<Tensor<f32>> = vec![];
    for future in futures.into_iter() {
        let tensor = future.await.unwrap()?.unpartition();
        outputs.push(tensor);
    }
    for output in outputs {
        println!("{:?}", output.to_host_vec().await?);
    }

    Ok(())
}
```

---

<a id="key-pattern-compose-device-operations-then-spawn"></a>
## 派生设备操作

这个循环把每一次前向传播都安排下去，而不等待 GPU 完成：

```rust
let mut futures: Vec<JoinHandle<Result<Partition<Tensor<f32>>, cuda_async::error::DeviceError>>> = vec![];
for i in 0..num_devices {
    // 取得设备 i 上模型权重的引用。
    let w = &model_weights[i];
    let (w0, w1) = (w.0.clone(), w.1.clone());
    // 采样随机数据。虽然这里的采样过程只是一种模拟，
    // 但它可以替换成真正采样一批数据的流程。
    let data = api::randn(0.0, 1.0, [dim, dim], None);
    // 统一的启动器：直接传入输出分区和各个输入。
    let out0 = api::zeros(&[dim, dim]).partition([block_dim, block_dim]);
    let out0 = gemm(out0, data, w0)
        .generics(fully_connected_layer.to_vec())
        .first()
        .unpartition();
    // 最终输出：matvec + relu。
    let out1 = api::zeros(&[dim]).partition([block_dim]);
    let out1 = matvec(out1, out0, w1)
        .generics(output_layer.to_vec())
        .first()
        .unpartition();
    // 施加 ReLU。先分区以便按 tile 派发，再 unzip 取出结果。
    let (out1,) = relu(out1.partition([block_dim])).unzip();
    // 现在 out1 承载着我们想用策略 i 去调度的工作。
    // 用该策略调用 schedule 会生成一个 device future，
    // 它已准备好在该策略选定的流上执行。通过为这个 device future spawn 一个任务，
    // 我们把工作提交给异步运行时（tokio）去执行。随后
    // 把任务句柄收集进 futures 向量。
    futures.push(tokio::spawn(out1.schedule(&devices[i])?));
}
```

为每块设备上的每一次前向传播 spawn 出任务之后，我们在继续之前等待结果：

```rust
let mut outputs: Vec<Tensor<f32>> = vec![];
for future in futures.into_iter() {
    let tensor = future.await.unwrap()?.unpartition();
    outputs.push(tensor);
}
```

---

## 要点

| 概念 | 含义 |
|---------|---------------|
| **设备操作** | 可串联的、与资源无关的 DAG |
| **tokio::spawn** | 并发地跑多个批次 |
| **schedule(policy)** | 指定某个具体的调度策略 |
| **惰性执行** | 先构建好流水线，再在 `.await` 或 `spawn()` 时执行 |

---

### 练习 1：融合 kernel

我们怎样才能把上面这些 kernel 融合成一个？这样会减少计算的内存占用吗？

### 练习 2：让数据传输与计算重叠

要构造一条让数据传输与计算重叠的流水线，我们需要改动什么？

---

## 参见

- [设备操作](../guide/device-operations.md) —— 流调度与 `DeviceOp` 的生命周期
- [宿主 API](../reference/host-api.md) —— `.schedule()`、调度策略以及各种执行方式
