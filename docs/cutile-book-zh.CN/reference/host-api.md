# 宿主 API

宿主 API 涵盖 tensor、CUDA 上下文与流、kernel 启动，以及 `DeviceOp` 的执行。入门材料见[宿主代码与设备代码](../guide/host-vs-device.md)、[张量与 tile](../guide/tensors-and-tiles.md)和[设备操作](../guide/device-operations.md)。

---

## tensor 的创建与视图

### `api::*` 构造函数

所有创建函数都返回 `DeviceOp` —— 分配与初始化发生在操作运行时，而不是在它被构造时。

| 函数 | 输出 | 说明 |
|---|---|---|
| `api::zeros::<T>(shape: &[usize])` | `DeviceOp<Output = Tensor<T>>` | 全零 |
| `api::ones::<T>(shape: &[usize])` | `DeviceOp<Output = Tensor<T>>` | 全一 |
| `api::full::<T>(val, shape: &[usize])` | `DeviceOp<Output = Tensor<T>>` | 用标量值填充 |
| `api::fill::<T>(tensor, val)` | `DeviceOp<Output = Tensor<T>>` | 填充一个已有 tensor 并返回它 |
| `api::arange::<T>(len: usize)` | `DeviceOp<Output = Tensor<T>>` | `[0, 1, 2, ..., len-1]`（一维） |
| `api::linspace(start: f32, stop: f32, n: usize)` | `DeviceOp<Output = Tensor<f32>>` | 从 `start` 到 `stop` 均匀分布的 `n` 个取值 |
| `api::eye(n: usize)` | `DeviceOp<Output = Tensor<f32>>` | `n × n` 单位矩阵 |
| `api::eye_rect(rows: usize, cols: usize)` | `DeviceOp<Output = Tensor<f32>>` | `rows × cols`，主对角线上为 1 |
| `api::convert::<From, To>(src: Arc<Tensor<From>>)` | `DeviceOp<Output = Tensor<To>>` | 转换 tensor 的元素类型 |
| `api::rand::<T, RANK>(shape: [usize; RANK], seed: Option<u64>)` | `DeviceOp<Output = Tensor<T>>` | 来自 cuRAND 的均匀分布 `[0, 1)`（`T: RandUniform`） |
| `api::randn::<T, RANK>(mean: T, std: T, shape: [usize; RANK], seed: Option<u64>)` | `DeviceOp<Output = Tensor<T>>` | 来自 cuRAND 的正态分布 `N(mean, std²)`（`T: RandNormal`） |
| `api::randn_f16(mean: f16, std: f16, shape: [usize; RANK], seed: Option<u64>)` | `DeviceOp<Output = Tensor<f16>>` | `f16` 的正态分布（先生成 `f32` 再转换；cuRAND 没有原生 `f16`） |

形状约定在整个模块中并不统一：`zeros`/`ones`/`full` 接受任意长度的 `&[usize]` 切片；`rand`/`randn` 接受 `[usize; RANK]` 数组（秩是 const 泛型）。`RANK` 参数通常从数组字面量推断出来。

```rust
use cutile::api;

let z = api::zeros::<f32>(&[1024]).sync_on(&stream)?;
let m = api::ones::<f32>(&[256, 256]).sync_on(&stream)?;
let r = api::randn(0.0f32, 1.0, [32, 64, 128], None).sync_on(&stream)?;   // 三维 N(0, 1)
let u = api::rand::<f32, 1>([1024], Some(42)).sync_on(&stream)?;          // 固定种子的均匀分布
let idx = api::arange::<i32>(1024).sync_on(&stream)?;
let I = api::eye(64).sync_on(&stream)?;
```

### tensor 与 `DeviceOp` 的形状辅助方法

宿主侧的重塑是零拷贝的元数据变更。它们要求新形状保持元素总数不变，并且对于借用形式的视图而言要求存储连续。

| API | 说明 |
|---|---|
| `tensor.reshape(&shape)` | 消耗 tensor 并返回带新形状的 `Result<Tensor<T>, Error>`（`Reshape` trait）。 |
| `(&arc_tensor).reshape(&shape)` | 返回 `Result<Arc<Tensor<T>>, Error>`：一个新的 `Arc<Tensor<T>>`，与原来共享同一份分配，只是元数据形状不同。 |
| `device_op.reshape(&shape)` | 重塑某个 `DeviceOp` 产出的 `Tensor<T>` 或 `Arc<Tensor<T>>`。 |
| `tensor.partition(shape)` | 消耗一个 tensor，并创建一个用于 kernel 启动的可变输出分区。若该 tensor 的存储与其它 tensor 或视图共享（例如某个 `(&arc_tensor).reshape(..)` 别名），则 panic。 |
| `arc_tensor.try_partition(shape)` | 仅当 `Arc<Tensor<T>>` 只有唯一持有者时才消耗它（否则返回 `Err`），然后对它做 `partition`，同样会在存储被共享时 panic。 |
| `partition.unpartition()` | 从 kernel 返回的分区中取回拥有所有权的 tensor。 |

```rust
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{Reshape, Tensor, TryPartition};
use cutile::tile_kernel::PartitionOp;
use std::sync::Arc;

let x = api::arange::<f32>(32).reshape(&[4, 8]).sync_on(&stream)?;
let z = api::zeros::<f32>(&[32]).partition([4]);

let weights: Arc<Tensor<f32>> = api::ones::<f32>(&[4, 8]).sync_on(&stream)?.into();
let weights_2d = (&weights).reshape(&[8, 4])?;
drop(weights); // weights_2d 现在是该存储的唯一持有者；继续让 `weights` 存活会让 `partition` panic
let partitioned = weights_2d.try_partition([2, 4])?;
```

### tensor 元数据与重新解释

`Tensor<T>` 在设备分配之外还保存了形状与布局元数据。这些访问器不会与 GPU 同步：

| API | 说明 |
|---|---|
| `tensor.shape()` | 以 `&[i32]` 表示的运行时维度 |
| `tensor.strides()` | 以 `&[i32]` 表示的运行时步长 |
| `tensor.size()` | 元素个数 |
| `tensor.num_bytes()` | tensor 视图中的字节数 |
| `tensor.is_contiguous()` | 该视图是否连续 |
| `tensor.device_id()` | 该分配所在的 CUDA 设备序号 |
| `tensor.device_pointer()` | 用于互操作的带类型、不拥有所有权的 `DevicePointer<T>` |
| `arc_tensor.reinterpret::<U>(&shape)` | 零拷贝地重新解释为 `Arc<Tensor<U>>` |

`reinterpret` 要求 `Arc<Tensor<T>>`、连续的存储、总字节数一致，并且指针对齐兼容：

```rust
use cutile::tensor::Tensor;
use std::sync::Arc;

let raw: Arc<Tensor<u32>> = api::arange::<u32>(4).sync_on(&stream)?.into();
let floats: Arc<Tensor<f32>> = raw.reinterpret::<f32>(&[4])?;
assert_eq!(floats.shape(), &[4]);
```

打包的 FP4 模型字节可以走同一条零拷贝元数据路径。如果某个互操作层给你的是字节存储，请在启动那些期望带类型打包 FP4 tensor 的 kernel 之前，把它重新解释为 `f4e2m1fnx2`：

```rust
use cutile::api::{self, DeviceOpReshape};
use cutile::cuda_core::f4e2m1fnx2;
use cutile::tensor::Tensor;
use std::sync::Arc;

let packed_bytes: Arc<Vec<u8>> = Arc::new(model_bytes);
let raw: Arc<Tensor<u8>> = api::copy_host_vec_to_device(&packed_bytes)
    .reshape(&[m, k / 2])
    .sync_on(&stream)?
    .into();
let fp4: Arc<Tensor<f4e2m1fnx2>> = raw.reinterpret::<f4e2m1fnx2>(&[m, k / 2])?;
```

`reinterpret` 不会重新打包，也不会校验 FP4 载荷。它保留字节、只改变 tensor 的元素类型元数据，因此 `model_bytes` 的生产者需要负责低半字节、高半字节的打包顺序。

### `TensorView`：零拷贝视图与切片

`TensorView` 提供对一个 tensor 的零拷贝借用视图，可以具有不同的形状或偏移。视图会借用底层 tensor —— 视图存在期间该 tensor 不能被修改。偏移在宿主侧施加，因此把视图传给 kernel 时，kernel 拿到的是指向正确起始地址的指针，而没有任何数据搬运。

| 方法 | 说明 |
|---|---|
| `tensor.view(&shape)` | 在不拷贝的前提下重塑为给定形状。元素总数必须一致。 |
| `tensor.slice(&ranges)` | 借用一个矩形子区域（每个维度一个 numpy 风格的区间）。 |
| `view.slice(&ranges)` | 继续链式切片；偏移会累加。 |

```rust
let tensor = api::arange::<f32>(1024).sync_on(&stream)?;

// 不拷贝地重塑。
let matrix = tensor.view(&[32, 32])?;

// 切片：借用一个子区域（numpy 风格的区间）。
let first_half = tensor.slice(&[0..512])?;       // 元素 0-511
let row_slice = matrix.slice(&[1..3])?;          // 第 1-2 行，所有列
let block = matrix.slice(&[1..3, 2..6])?;        // 第 1-2 行，第 2-5 列

// 链式切片会累加偏移。
let inner = tensor.slice(&[100..200])?.slice(&[10..20])?;  // = tensor[110..120]
```

把视图和切片作为 `&Tensor` 参数传给 kernel，即可在不分配、不拷贝的前提下处理某个已有 tensor 的一部分。

---

## 宿主-设备与设备-设备传输

在 CPU 与 GPU 之间，或者在两个设备 tensor 之间搬运数据，使用的 API 都返回 `DeviceOp` —— 拷贝在操作运行时才被调度，而不是在构造时：

| API | 返回 | 说明 |
|---|---|---|
| `api::copy_host_vec_to_device::<T>(vec: &Arc<Vec<T>>)` | `DeviceOp<Output = Tensor<T>>` | 把宿主 `Vec<T>` 拷贝到一个新的设备 `Tensor<T>` |
| `api::copy_device_to_host_vec::<T>(tensor: &Arc<Tensor<T>>)` | `DeviceOp<Output = Vec<T>>` | 把设备 `Tensor<T>` 拷贝到宿主 `Vec<T>` |
| `tensor.to_host_vec()` | `DeviceOp<Output = Vec<T>>` | `copy_device_to_host_vec` 的方法形式（推荐） |
| `device_op.to_host_vec()` | `DeviceOp<Output = Vec<T>>` | 把某个 `DeviceOp` 产出的 `Tensor<T>` 拷贝到宿主 |
| `api::dup(&tensor)` / `tensor.dup()` | `DeviceOp<Output = Tensor<T>>` | 分配一个新 tensor 并做设备到设备的拷贝 |
| `api::memcpy(&mut dst, &src)` | `DeviceOp<Output = ()>` | 设备到设备地拷贝进一个已有 tensor，尤其用于 CUDA 图更新 |

```rust
// 宿主 -> 设备
let data: Arc<Vec<f32>> = Arc::new(vec![1.0; 1024]);
let tensor: Tensor<f32> = api::copy_host_vec_to_device(&data).sync_on(&stream)?;

// 设备 -> 宿主
let result: Vec<f32> = tensor.to_host_vec().sync_on(&stream)?;

// 设备 -> 设备
let copy = tensor.dup().sync_on(&stream)?;
```

宿主侧的 `Vec` 必须在操作完成之前保持存活 —— 异步拷贝会一直从它读取，直到流同步为止。`Arc<Vec<T>>` 让共享访问变得简单。`to_host_vec` 在 `Tensor<T>`、`Arc<Tensor<T>>` 和 `&Arc<Tensor<T>>` 上都可用；它们返回同一个 `DeviceOp<Output = Vec<T>>`。它也可以用在 `DeviceOp<Output = Tensor<T>>` 上，这是 kernel 链之后最常见的形态：

```rust
let host: Vec<f32> = kernel(out.partition([128]), &input)
    .first()
    .unpartition()
    .to_host_vec()
    .sync_on(&stream)?;
```

`api::memcpy` 在已经分配好的 tensor 之间拷贝；若源与目标的元素个数不同，它会 panic。它是在重放之前更新图输入缓冲区的常用方式：

```rust
graph.update(api::memcpy(&mut input_buffer, &new_input))?;
graph.launch().sync_on(&stream)?;
```

---

## 设备与流

每个宿主程序都从一个 `Device` 开始，再加上一条或多条用于调度 GPU 工作的 `Stream`：

```rust
use cuda_core::Device;

let device = Device::new(0)?;              // 设备序号 0
let stream = device.new_stream()?;         // 由该设备拥有的一条新流
```

| 方法 | 返回 | 说明 |
|---|---|---|
| `Device::new(ordinal: usize)` | `Result<Arc<Device>, DriverError>` | 创建一个绑定到某个 GPU 序号的设备句柄 |
| `Device::device_count()` | `Result<i32, DriverError>` | 支持 CUDA 的设备数量 |
| `device.ordinal()` | `usize` | 该句柄代表的 GPU 序号 |
| `device.name()` | `Result<String, DriverError>` | 设备名称 |
| `device.new_stream()` | `Result<Arc<Stream>, DriverError>` | 在该设备上创建一条新流 |
| `unsafe Device::borrow_raw(cu_ctx, cu_device, ordinal)` | `Arc<Device>` | 借用外部拥有的 CUDA 上下文/设备，用于互操作 |
| `unsafe Device::borrow_with_owner(cu_ctx, cu_device, ordinal, owner)` | `Arc<Device>` | 同上，但持有一个 `Arc<dyn ForeignOwner>`，使外部所有者比该句柄活得更久 |
| `unsafe Stream::borrow_raw(cu_stream, &device)` | `Arc<Stream>` | 借用外部拥有的 CUDA 流，用于互操作 |
| `unsafe Stream::borrow_with_owner(cu_stream, &device, owner)` | `Arc<Stream>` | 同上，但持有一个 `Arc<dyn ForeignOwner>` |
| `unsafe Module::borrow_raw(...)` / `unsafe Function::borrow_raw(...)` | CUDA module/function 包装器 | 借用外部拥有的 CUDA 句柄 |

设备用 `Arc` 包装以便跨线程共享；流同样用 `Arc` 包装，可以传给 `.sync_on(&stream)` 以显式指定流调度。

默认的轮转调度策略会为大多数工作负载自动处理流的分配 —— 这些 API 是为你确实需要显式流控制时准备的（调试、确定性排序、配合 `AsyncKernelLaunch`，或者在专用流上让计算与传输重叠）。

`borrow_raw` 系列构造函数不取得底层 CUDA 句柄的所有权，因此不会在 drop 时销毁它们。它们是 `unsafe fn`：调用者要保证这些句柄在借用被使用期间始终有效。`borrow_with_owner` 变体会让一个 `Arc<dyn ForeignOwner>` 保持存活，这样拥有它的那个运行时的对象就不会被提前 drop。当你与另一个拥有上下文、流、模块或函数的运行时集成时，请使用它们。

---

## kernel 启动配置

有若干类型用于配置 kernel 如何编译与启动。

**`CompileOptions`** —— 对入口级 `optimization_hints` 的运行时覆盖，通常用于自动调优：

```rust
use cutile::tile_kernel::CompileOptions;

let opts = CompileOptions::default()
    .occupancy(4)
    .num_cta_in_cga(2)
    .max_divisibility(16)
    .num_worker_warps_per_cta(4); // 字节码 13.3+；请使用 4 或 8。

let result = my_kernel(args).compile_options(opts).grid(grid).await?;
```

Tile IR 13.4 把 `num_worker_warps_per_cta` 限制为 4 或 8。Tile IR 13.3 接受 `[1, 32]` 范围内 2 的幂（汇编器会夹取到它支持的 worker 数）。JIT 会检查所选版本的取值范围。

不同的 `CompileOptions` 取值会触发各自独立的 JIT 编译，并且是 kernel 缓存键的一部分。

生成的 `#[cutile::entry]` 启动器还暴露了启动时的配置方法：

| 方法 | 说明 |
|---|---|
| `.grid((x, y, z))` | 设置一个显式的运行时启动网格，而不是从已分区的 tensor 输入推断。 |
| `.const_grid((x, y, z))` | 设置一个编译期常量网格，从而启用依赖网格的优化。 |
| `.compile_options(opts)` | 为本次编译覆盖占用率、cluster/CTA、worker warp 与整除性提示。 |
| `.generics(values)` | 当类型与 const 泛型参数无法被推断时，手动绑定它们。 |

JIT 编译器按以下顺序解析 `tileiras`：若设置了 `CUTILE_TILEIRAS_PATH` 则优先用它，然后是 `$CUDA_TOOLKIT_PATH/bin/tileiras`，接着是默认的 CUDA 13.2+ 安装目录（`/usr/local/cuda-13.4`、`/usr/local/cuda-13.3`、`/usr/local/cuda-13.2`、`/usr/local/cuda-13`、`/usr/local/cuda`），最后是通过常规 `PATH` 查找 `tileiras`。设置 `CUTILE_TILEIRAS_PATH` 可以强制指定某个具体的二进制：

```bash
CUTILE_TILEIRAS_PATH=/opt/cuda-tile/bin/tileiras cargo test -p cutile
```

所选二进制支持的字节码版本决定了输出，而不是用来构建 Rust crate 的 CUDA 头文件。默认是双方共同支持的最高版本（13.2、13.3 或 13.4）。`CUTILE_BYTECODE_VERSION` 可以把版本固定为其中之一，但无法强制使用所选汇编器不支持的版本。请在启动之前配置这些变量；已经解析过的启动点会保留它们当初的工具链快照。

**`LaunchConfig`** —— 用于 `AsyncKernelLaunch`（在 `#[cutile::entry]` 路径之外启动的裸 CUDA kernel）的 grid/block/共享内存规格：

```rust
use cuda_core::LaunchConfig;

LaunchConfig {
    grid_dim: ((n + 255) / 256, 1, 1),    // 三维的线程块网格
    block_dim: (256, 1, 1),                // 三维的线程块
    shared_mem_bytes: 0,                   // 每块的动态共享内存
}
```

**`AsyncKernelLaunch`** —— 把一次 CUDA 驱动的 kernel 启动包装成一个 `DeviceOp`。用 `push_arg`（安全，用于 `DType` 标量）或 `push_device_ptr`（`unsafe`，用于裸设备指针）构建参数列表，设置启动配置，然后 `.await` 或 `.sync_on()`：

```rust
use cuda_async::launch::AsyncKernelLaunch;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher.push_arg(num_elements as u32);
launcher.push_arg(scale);
let input_ptr = input.device_pointer();
let output_ptr = output.device_pointer();
unsafe {
    launcher
        .push_device_ptr(input_ptr.cu_deviceptr())
        .push_device_ptr(output_ptr.cu_deviceptr());
}
launcher.set_launch_config(LaunchConfig {
    grid_dim: ((num_elements as u32 + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
});
launcher.await?;  // 作为一个 DeviceOp 执行
```

完整演练以及那个把 `unsafe` 隐藏在调用点之下的包装模式，见[互操作](../guide/interoperability.md)。

**`.generics(Vec<String>)`** —— `#[cutile::entry]` 生成的启动器接受该方法，用来在运行时绑定 const 泛型与类型参数：

```rust
let generics = vec![
    "f32".to_string(),  // E
    "16".to_string(),   // BM
    "16".to_string(),   // BN
    "8".to_string(),    // BK
    "128".to_string(),  // K
];
gemm(z, x, y).generics(generics).sync_on(&stream)?;
```

泛型取值是 kernel 缓存键的一部分：每一个独特的组合都会触发它自己的一次 JIT 编译。

---

<a id="programmatic-dependent-launch"></a>
### 程序化依赖启动（PDL）

生成的 kernel builder 上有一个 unsafe、消耗 self 的 `.programmatic_dependent_launch()` 方法。它默认关闭，并且与 `CompileOptions` 相互独立。它要求 Tile IR 13.4、sm_90 或更新，以及提供 `cuLaunchKernelEx` 的驱动；不受支持的请求会失败，而不是静默退化成普通的串行化。`cuda_async::AsyncKernelLaunch` 也为更底层的调用者暴露了一个 unsafe 的可变 builder 方法。

把两个 kernel 入队到同一条流上，在**消费者**上启用 PDL：

```rust
unsafe {
    kernels::producer(input.device_pointer()).grid((1, 1, 1))
        .then(|_| kernels::consumer(input.device_pointer(), output.device_pointer())
            .programmatic_dependent_launch().grid((1, 1, 1)))
}.sync_on(&stream)?;
```

在生产者的每个 tile block 里，当依赖它的 kernel 可以开始时，把它的存储串接到 `gdc_launch_dependents_tko(Some(stored))`；`Tensor::store` 会返回可供串接的令牌。每一个生产者 tile block 都必须发出信号或完成。在消费者中，调用 `gdc_wait_tko(None)`，并让每一个依赖前驱的内存操作都消费它的结果令牌：对于 tensor 输入，在创建它的任何视图之前先 `input.set_token(ready)`（unsafe），这样之后通过该 tensor 的每一次加载都被串接在等待之后；对于指针操作，直接传入该令牌。仅靠源码顺序并不能建立那种依赖关系。启动信号只是允许调度；等待才在消费数据之前建立依赖完成。

请把令牌安装在函数级绑定上，位于条件与循环之外。当 `input` 声明在函数级时，`unsafe { input.set_token(ready) }` 块是合法的。块局部的接收者，以及在安装之后重新绑定该 tensor，都会被 JIT 拒绝；显式的令牌更新目前还不参与控制流的传递。已有视图会保留它们的旧令牌，因此请在安装之后再创建视图。裸的 `make_partition_view` 构造函数之所以是 unsafe，是因为调用者提供的令牌否则可能丢弃安全视图加载所要求的顺序。

等待之前的工作不得与尚未完成的生产者工作发生竞争。两个 kernel 都不得依赖实际的重叠来推进，并且所有资源都必须在两个 kernel 都用完它们之后仍然存活。裸指针不会保留拥有它们的 tensor。PDL 是一项 unsafe 的 opt-in，因为这些义务超出了安全 tensor API 的常规流顺序保证。

完整可执行的生产者/消费者示例见 `cutile-examples/examples/pdl.rs`（含流顺序与 PDL 的计时对比），或者测试 `cutile/tests/gpu/tile_ir_13_4.rs::dependent_launch_waits_for_producer_data`。

## Future 类比

`DeviceOp` 之于 GPU 工作，正如 `Future` 之于异步 I/O。二者都是对工作的惰性描述，在被驱动之前不会执行：

| 概念 | `std::future::Future` | `DeviceOp` |
|---|---|---|
| 它表示什么 | 异步计算 | GPU 计算 |
| 何时运行 | `.await` 或 `poll()` 时 | `.sync()`、`.sync_on()` 或 `.await` 时 |
| 串联 | 通过 `FutureExt` 的 `.then()`、`.map()` | `DeviceOp` 上的 `.then()`、`.map()` |
| 汇聚 | `join!` | `zip!` |
| 扇出 | 不适用（单一消费者） | `.unzip()` |
| 共享访问 | `FutureExt::shared()` | `.shared()` |
| 类型擦除 | `BoxFuture` | `.boxed()` → `BoxedDeviceOp` |
| 输出包装 | `Poll<T>` | `Result<T, DeviceError>` |

异步运行时通过 `poll()` 驱动 `Future`。`DeviceOp` 通过 `execute()` 提交 GPU 工作。用 `.await` 或 `.into_future()` 把它转换成 future 会得到一个 `DeviceFuture`，运行时通过轮询它来获知 GPU 完成。

---

## 组合子参考

所有组合子都遵循既有的 Rust 惯例。"先例"这一列指出该设计灵感来自哪个标准库或 `futures` crate 中的方法。

### 组合

| 组合子 | 签名 | 先例 | 作用 |
|---|---|---|---|
| `zip!(a, b, …)` | `(impl DeviceOp, …) → impl DeviceOp<Output=(A, B, …)>` | `Iterator::zip` | 把 N 个操作合并成一个产生元组的操作 |
| `.unzip()` | `impl DeviceOp<Output=(A, B, …)> → (impl DeviceOp<Output=A>, …)` | `Iterator::unzip` | 把元组操作拆成相互独立的逐元素操作 |
| `.then(f)` | `self → f(Self::Output) → impl DeviceOp<Output=O>` | `FutureExt::then` | 在**同一条流上**串接后续 GPU 工作 |
| `.map(f)` | `self → f(Self::Output) → O`（无 GPU 工作） | `FutureExt::map` | 在不发出 GPU 工作的前提下变换输出 |
| `.inspect(f)` | `self → f(&Self::Output)`（透传） | `FutureExt::inspect` | 为调试窥视输出；原样返回它 |

### 选择

| 组合子 | 签名 | 先例 | 作用 |
|---|---|---|---|
| `.first()` | `impl DeviceOp<Output=(A, B, …)> → impl DeviceOp<Output=A>` | `slice::first` | 取出元组输出的第一个元素 |
| `.last()` | `impl DeviceOp<Output=(A, B, …)> → impl DeviceOp<Output=Z>` | `slice::last` | 取出元组输出的最后一个元素 |

### 共享与类型擦除

| 组合子 | 签名 | 先例 | 作用 |
|---|---|---|---|
| `.shared()` | `self → SharedDeviceOp<Self::Output>`（要求 `Self: 'static`、`Output: Sync`） | `FutureExt::shared` | 可克隆、只执行一次；输出是 `Arc<T>` |
| `shared(arc)` | `Arc<T> → SharedDeviceOp<T>` | — | 把一个已有的 `Arc` 包装成预先算好的 `SharedDeviceOp` |
| `.boxed()` | `self → BoxedDeviceOp<Self::Output>` | `FutureExt::boxed` | 为异构集合做类型擦除 |

### 执行

| 方法 | 流由谁选择 | 是否阻塞？ | 适用场景 |
|---|---|---|---|
| `.sync()` | 默认策略（轮转） | 是 | 快速脚本 |
| `.sync_on(&stream)` | 你显式指定的流 | 是 | 确定性排序、调试 |
| `.await` | 默认策略（轮转） | 否（挂起任务） | 异步生产代码 |
| `.into_future()` | 默认策略 | 否（返回 `DeviceFuture`） | 手动处理 future |
| `.schedule(&policy)` | 你提供的 `Arc<dyn SchedulingPolicy>` | 否（返回 `Result<DeviceFuture, DeviceError>`） | 多设备派发 |
| `.graph()` | 默认策略（轮转） | 是（捕获 + 同步） | CUDA 图捕获 |
| `.graph_on(stream)` | 你显式指定的流 | 是（捕获 + 同步） | 在特定流上做 CUDA 图捕获 |

> **注意**：如果有任何 kernel 输入是 `&Tensor<T>`（借用形式），该操作就不是 `'static` 的，因而无法用于 `tokio::spawn`。请在同一作用域内使用 `.sync_on()` 或 `.await`，或者改用 `Arc<Tensor<T>>` 以便 spawn 任务。

---

## 受支持的 kernel 参数类型

| kernel 参数 | 宿主类型 | 返回类型 |
|---|---|---|
| `&Tensor<T, S>` | `Tensor<T>`、`Arc<Tensor<T>>`、`&Tensor<T>` 或 `&TensorView<T>` | 与输入相同 |
| `&mut Tensor<T, S>` | `Partition<Tensor<T>>`、`Partition<&mut Tensor<T>>`，或来自 `.map(..)` 的 `MappedLaunchPartition<…>` 形式 | 与输入相同 |
| 标量（`f32`、`i32` 等） | 同一个标量 | 同一个标量 |
| `*mut T`、`*const T`（仅 unsafe） | `DevicePointer<T>` | `DevicePointer<T>` |

借用形式的分区（`Partition<&mut Tensor<T>>`）会就地写入 —— 不需要 `unpartition()`。用 `(&mut tensor).partition(shape)` 创建它。

裸指针入口点是 `unsafe fn`。用 `tensor.device_pointer()` 从某个 tensor 取得带类型的设备指针，并确保该指针在 kernel 启动期间始终有效：

```rust
let backing = api::zeros::<f32>(&[1024]).sync_on(&stream)?;
let ptr = backing.device_pointer();
unsafe { raw_ptr_kernel(ptr, 1024) }.sync_on(&stream)?;
```

---

<a id="ownership-model"></a>
## 所有权模型

启动器以它接收时相同的所有权形式返回各个参数。

### 只读输入（`&Tensor` 参数）

| 输入 | 返回 | 可用于 `tokio::spawn`？ |
|---|---|---|
| `Tensor<T>` | `Tensor<T>` | 是 |
| `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | 是 |
| `&'a Tensor<T>` | `&'a Tensor<T>` | 否（不是 `'static`） |

### 可变输出（`&mut Tensor` 参数）

| 输入 | 返回 | 需要 `unpartition()`？ |
|---|---|---|
| `Partition<Tensor<T>>`（拥有所有权） | `Partition<Tensor<T>>` | 需要 |
| `Partition<&'a mut Tensor<T>>`（借用） | `Partition<&'a mut Tensor<T>>` | 不需要 —— tensor 被就地写入 |

借用形式用 `(&mut tensor).partition(shape)` 创建：

### 拥有所有权：`Tensor<T>`

直接传入一个 tensor —— 启动器会在内部把它包进 `Arc` 交给 kernel，之后再把所有权解包回来（由于引用计数为 1，这是安全的）：

```rust
let output = my_kernel(
    api::zeros(&[1024]).partition([128]),
    api::ones::<f32>(&[1024]),  // DeviceOp<Output=Tensor<f32>>
)
.first()
.unpartition()
.sync_on(&stream)?;
```

适用于不需要共享访问、只使用一次的 tensor。

### 共享：`Arc<Tensor<T>>`

当同一个 tensor 要传给多个 kernel 时，用 `Arc` 包装：

```rust
let x: Arc<Tensor<f32>> = api::ones(&[1024]).sync_on(&stream)?.into();

let a = kernel_a(out_a, x.clone()).sync_on(&stream)?;
let b = kernel_b(out_b, x.clone()).sync_on(&stream)?;
```

这是既有代码中最常见的模式。

### 借用：`&Tensor<T>`

当你希望保留所有权并避免 `Arc` 开销时，传引用。借用检查器会保证 tensor 比 kernel 活得更久：

```rust
let weights: Tensor<f32> = api::ones(&[1024]).sync_on(&stream)?;

// 借用 —— 没有 Arc 分配，没有引用计数。
let result = my_kernel(out_partition, &weights).sync_on(&stream)?;

// weights 在这里仍然可用。
```

由于 `&Tensor<T>` 不是 `'static` 的，`tokio::spawn` 会拒绝那些借用了 tensor 的操作：

```rust
let op = my_kernel(out, &weights);  // 借用了 weights
tokio::spawn(op.into_future());      // ← 编译错误：不是 'static
```

### `.shared()`：克隆 + 只执行一次

`.shared()` 把一个 `DeviceOp` 转换成可 `Clone` 的 `SharedDeviceOp<T>`。它要求该操作是 `'static` 的，且其输出是 `Sync` 的。底层操作**最多只运行一次**；每个克隆拿到的都是缓存结果的 `Arc::clone()`：

```rust
let x = api::ones::<f32>(&[32, 32]).shared();

let a = kernel_a(x.clone()).sync()?;  // x 在这里执行（第一个运行的克隆）
let b = kernel_b(x.clone()).sync()?;  // 使用缓存的 Arc<Tensor<f32>>
```

输出类型会变化：`DeviceOp<Output=T>` 变成 `Output=Arc<T>` 的 `SharedDeviceOp`。

对于预先算好的取值（例如权重 tensor），用 `shared()` 自由函数直接包装一个 `Arc<T>`：

```rust
use cuda_async::device_operation::shared;

let w: Arc<Tensor<f32>> = /* 已加载的权重 */;
let w_op: SharedDeviceOp<Tensor<f32>> = shared(w);
```

### `.unwrap_arc()`

`.shared()` 产生的是 `Arc<T>` 输出（`unzip` 会保留元组的元素类型）。当你需要取回拥有所有权的 `T` 时（例如要把它分区），使用 `.unwrap_arc()`：

```rust
let x: Arc<Tensor<f32>> = api::ones(&[1024]).shared().sync()?;

let owned: Tensor<f32> = value(x).unwrap_arc().sync()?;
let partitioned = owned.partition([128]);
```

若该 Arc 有多个持有者，它会 panic。

### IntoDeviceOp：自动包装

`IntoDeviceOp` trait 让 kernel 启动器既能接受 `DeviceOp`，也能接受普通值：

| 类型 | 包装为 |
|---|---|
| 任意 `impl DeviceOp<Output=T>` | 直接透传 |
| `Tensor<T>` | `Value<Tensor<T>>` |
| `Arc<T>` | `Value<Arc<T>>` |
| `&'a Tensor<T>` | `Value<&'a Tensor<T>>` |
| `&Arc<T>` | `Value<Arc<T>>`（克隆该 Arc） |
| `&'a TensorView<'a, T>` | `Value<&'a TensorView<'a, T>>` |
| `f32`、`f64`、`i8`、`i16`、`i32`、`i64`、`u8`、`u16`、`u32`、`u64`、`usize`、`bool`、`f16`、`bf16` | `Value<T>` |
| `Partition<Tensor<T>>`、`Partition<&'a mut Tensor<T>>` | `Value<Partition<…>>` |
| `MappedLaunchPartition<Partition<…>>` | `Value<MappedLaunchPartition<…>>` |
| `DevicePointer<T>` | `Value<DevicePointer<T>>` |

```rust
// 下面这些都可以作为 &Tensor 类型 kernel 参数的输入：
my_kernel(out, tensor);              // Tensor<T>
my_kernel(out, arc_tensor);          // Arc<Tensor<T>>
my_kernel(out, &tensor);             // &Tensor<T>
my_kernel(out, api::ones(&[1024]));  // DeviceOp<Output=Tensor<T>>
```

---

## 调度模型

### 流的分配

当你调用 `.sync()` 或 `.await` 时，该操作会向**默认设备的调度策略**请求一条流。默认策略是带 4 条流的 `StreamPoolRoundRobin`：

```text
op_a.sync()  →  Stream 0
op_b.sync()  →  Stream 1
op_c.sync()  →  Stream 2
op_d.sync()  →  Stream 3
op_e.sync()  →  Stream 0  （回绕）
```

连续且相互独立的操作会落到不同的流上，从而实现 GPU 重叠。用 `.then()` 串联的操作共享父操作的流，从而保持数据依赖的顺序。

### 显式指定流：`.sync_on()`

完全绕过策略。所有使用同一条流的操作都按调用顺序执行：

```rust
let stream = device.new_stream()?;
let a = op_a.sync_on(&stream)?;  // Stream X
let b = op_b.sync_on(&stream)?;  // Stream X —— 保证在 op_a 之后
```

### 可用的策略

| 策略 | 行为 |
|---|---|
| `StreamPoolRoundRobin`（默认） | 在 N 条流之间轮转（默认 4 条） |
| `SingleStream` | 所有操作都在一条流上 —— 严格顺序 |
| 自定义 `impl SchedulingPolicy` | 实现 `fn next_stream()` 来使用你自己的策略 |

### `.then()` 的保证

`.then()` 是表达数据依赖的推荐方式。两个操作共享同一条流，因此第二个一定能看到第一个的输出被完整写入 —— 无需手动同步：

```rust
let result = allocate_buffer()
    .then(|buf| fill_kernel(buf))      // 同一条流
    .then(|buf| process_kernel(buf))   // 同一条流
    .sync()?;
```

**不可重入：** 在任意给定线程上，同一时刻只允许一个 DeviceOp 处于执行状态。在 `then` 闭包内部调用 `sync_on`、`sync` 或 `.await` 会返回运行时错误。这可以防止跨流访问正在执行中的 tensor 所导致的 CUDA 数据竞争。如果你确实需要嵌套执行，并且已经核实不存在跨流数据竞争，请使用 `unsafe then_unchecked`。

---

## 错误传播

所有执行方法都返回 `Result<T, DeviceError>`。错误会穿过组合子传播：如果 `.then()` 链中任意一个操作失败，错误会短路返回给调用者。

### DeviceError 的变体

| 变体 | 何时出现 |
|---|---|
| `Driver(DriverError)` | CUDA 驱动调用失败（OOM、非法参数等） |
| `Context { device_id, message }` | 设备上下文断言失败 |
| `KernelCache(String)` | 已声明，但当前没有任何代码路径会构造它 |
| `Scheduling(String)` | 没有可用流，或者策略配置有误 |
| `Launch(String)` | 启动器生成的前置条件校验失败（`kernel_launch_assert`：参数形状、已声明的 `preconditions`） |
| `Internal(String)` | cuda-async 内部有 bug，或者一次嵌套执行尝试被不可重入守卫拒绝 |
| `Anyhow(String)` | 从 `anyhow::Error` 或 `cutile::error::Error` 转换而来；JIT 编译、网格推断与 tensor 错误都会走到这里 |

### 错误处理模式

```rust
// 模式 1：用 ? 向上传播
let x = api::zeros(&[1024]).sync_on(&stream)?;

// 模式 2：匹配具体错误
match my_kernel(args).sync_on(&stream) {
    Ok(result) => { /* 使用 result */ }
    Err(DeviceError::Launch(msg)) => {
        eprintln!("launch precondition failed: {msg}");
    }
    Err(DeviceError::Anyhow(msg)) => {
        eprintln!("compile, grid, or tensor error: {msg}");
    }
    Err(e) => return Err(e.into()),
}
```

### cutile::error::Error 与 DeviceError

`cutile::error::Error` 是顶层错误类型，它把 `DeviceError` 与其它错误类别（I/O、形状不匹配等）一起包装起来。只做 GPU 工作的函数返回 `DeviceError`；混合了宿主与设备工作的函数（例如那些示例）返回 `cutile::error::Error`。

---

<a id="cuda-graph-integration"></a>
## CUDA 图集成

### 组合子方式：`.graph_on(stream)`

任意 `DeviceOp` 都可以被捕获成一张可重放的 CUDA 图：

```rust
let forward_op = build_forward(&cfg, &weights, input, buffers);
let mut graph = forward_op.graph_on(stream.clone())?;
let output = graph.take_output().unwrap();

// 重放循环 —— 不重建图，不重新编译 kernel。
for token in tokens {
    graph.update(api::memcpy(&mut input_buf, &token))?;
    graph.launch().sync_on(&stream)?;
}
```

这要求对共享缓冲区使用 `Arc<Tensor<T>>` + `try_partition`。

### 作用域方式：`CudaGraph::scope`

`CudaGraph::scope` 提供了一种命令式的替代方案，它使用 `&mut` 借用而不是 `Arc`。每次 `s.record(op)` 记录一个图节点，并立刻释放借用。一次 `record` 调用写入的缓冲区可以被下一次读取：

```rust
let mut output = api::zeros::<f32>(&[d]).sync_on(&stream)?;
let weights = api::ones::<f32>(&[d]).sync_on(&stream)?;

let graph = CudaGraph::scope(&stream, |s| {
    s.record(kernel1((&mut output).partition([128]), &weights))?;
    s.record(kernel2((&mut output).partition([64]), &weights))?;
    Ok(())
})?;

graph.launch().sync_on(&stream)?;
```

`record` 只接受实现了 `GraphNode` 的操作 —— kernel 启动与 `memcpy`。分配类操作（`zeros`、`ones`、`dup`）会在编译期被拒绝，因为它们的地址在重放时可能变化。

### `GraphNode` trait

`GraphNode` 是一个标记 trait，用于那些可以安全记录进 CUDA 图的操作。只有不分配、不释放设备内存的操作才实现它：

| 实现 `GraphNode` 的类型 | 为什么安全 |
|---|---|
| 宏生成的 kernel 启动器 | 只做 kernel 启动 —— 不分配也不释放 |
| `Memcpy`（`api::memcpy`） | 在预分配的缓冲区之间拷贝 |
| `Value<T>`（`value(x)`） | 没有 GPU 工作 |

### CudaGraph 的方法

| 方法 | 作用 |
|---|---|
| `.graph()` / `.graph_on(stream)` | 把一个 `DeviceOp` 捕获成 `CudaGraph<T>` |
| `CudaGraph::scope(&stream, \|s\| { … })` | 使用 `&mut` 借用的作用域捕获 |
| `s.record(op: impl GraphNode)` | 在作用域内记录一个图节点 |
| `graph.take_output()` | 从这次捕获执行中取回输出 |
| `graph.update(op)` | 在图的流上运行一个 `DeviceOp`（例如拷贝新的输入） |
| `graph.launch()` | 返回一个重放已捕获图的 `DeviceOp` |

所有设备指针都在捕获时被烘焙固定。要改变输入，请预分配一个缓冲区、把它传入操作，并在每次启动之前用 `memcpy` 写入新数据。完整演练见[教程 10](../tutorials/10-cuda-graphs.md)。

---

## 其它公开 API

上面各节未涵盖的公开宿主侧 API：

| API | 说明 |
|---|---|
| `unsafe Tensor::from_foreign(owner: Arc<dyn DeviceAllocation>, shape, strides)` | 借用由外部框架（cudarc、torch、VMM 区间）拥有的设备内存；所有者通过引用计数保持存活，可寻址范围在构造时被校验 |
| `unsafe Tensor::borrow_raw_parts(dptr, device_id, shape, strides)` | 把裸设备指针借用为一个 `Tensor<T>` |
| `tensor.partition_prefix(shape)` | 启动网格可以覆盖 block 网格各轴前缀的输出分区（`partition` 保持严格相等） |
| `partition.map(map_shape, num_tile_blocks)` | 用于持久化 kernel 的 `MappedLaunchPartition`，它会遍历 `MappedPartitionMut` 的索引；启动网格为 `(num_tile_blocks, 1, 1)` |
| `cutile::jit_cache::{enable, enable_default, disable, is_enabled, stats}` | Opt-in 的持久化磁盘 cubin 缓存；默认关闭 |
| 启动器的 `.compile()` / `.compile_on(&stream)` | 在不启动的前提下 JIT 编译并缓存该特化 |
| 启动器的 `.specialize()`、`.l1_cache_key()`、`.l2_cache_key()` | 在不编译、不启动的前提下解析特化标识以及内存/磁盘缓存键 |
| `api::meta::<T>(&shape)` | 携带形状与 dtype 但不做分配的占位 tensor，用于 `.compile()` 预热 |
| `CompileOptions::{opt_level, device_debug, lineinfo, sanitize_memcheck}` | 设备编译器开关；每一项都是缓存键的一部分（见[调试与性能分析](../guide/debugging-and-profiling.md)） |
| `cutile::bench::{do_bench, do_bench_paired, BenchOptions, Measurement}` | 基于设备事件的 kernel 计时（见[性能](../guide/performance.md)） |
| `cuda_async::device_context::{set_default_device, global_policy, with_device_policy}` | 设置该线程的默认设备；读取或借用一个设备的调度策略 |

---

## 参见

- [设备操作](../guide/device-operations.md) —— 关于流、调度与组合模式的教程式指南
- [教程 10](../tutorials/10-cuda-graphs.md) —— 端到端的 CUDA 图示例
- [互操作](../guide/interoperability.md) —— 把自定义 CUDA C++ kernel 集成进 DeviceOp 模型
