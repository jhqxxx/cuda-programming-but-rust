# 编写你的第一个内核 — cuda-oxide

本节将介绍安装 cuda-oxide、创建项目、编写 GPU 内核并运行它 —— 全部使用纯 Rust 完成。

---

## 安装 cargo-oxide

如果尚未安装，请先安装构建工具：

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

验证你的环境配置是否正确：

这会检查兼容的 GPU、CUDA toolkit、LLVM 和代码生成后端。在继续之前，请先修复它报告的所有问题（详见[安装指南](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)）。

---

## 创建项目

使用 `cargo oxide new` 脚手架创建一个新项目：

```bash
cargo oxide new my_first_kernel
cd my_first_kernel
```

这会生成一个可立即运行的项目：

```
my_first_kernel/
├── Cargo.toml          # 依赖 cuda-device、cuda-host、cuda-core
├── rust-toolchain.toml # 固定所需的 nightly 工具链
└── src/
    └── main.rs          # 内核代码 + 主机代码在同一文件中
```

构建并运行它：

你应该会看到 `PASSED: all 1024 elements correct`。生成的模板是一个向量加法内核 —— 不错的起点，但我们来看点更有趣的。

---

## 内核剖析

这里是一个略有不同的向量加法：元素级加法被提取到一个普通的辅助函数中。内核和辅助函数与主机代码一起位于同一个文件中：

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

/// 普通辅助函数 —— 无需注解。
/// 编译器通过 `vecadd` 的调用关系自动发现它。
fn add(a: f32, b: f32) -> f32 {
    a + b
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = add(a[i], b[i]);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("加载嵌入模块失败");
    module
        .vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
        .unwrap();

    let c_host = c_dev.to_host_vec(&stream).unwrap();
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: 全部 {} 个元素正确", N);
    } else {
        eprintln!("FAILED: {} 个错误", errors);
        std::process::exit(1);
    }
}
```

这里发生了很多事情。我们来拆解关键部分。

### 单源编译

内核和主机代码**位于同一个文件**中，通过单次 `cargo` 命令调用进行编译。代码生成后端拦截编译过程，将 `#[kernel]` 函数路由到 MIR-to-PTX 流水线，其余部分委托给标准 LLVM。最终二进制文件包含原生主机代码以及嵌入的设备产物。

### `#[kernel]`

将函数标记为**可启动的内核入口点** —— 相当于 GPU 版的 `main`。该函数通过以下流水线编译为 PTX：

```
Rust 源码 → MIR → Pliron IR → LLVM IR → PTX
```

同一个函数对主机编译器也可见，用于类型检查，但其函数体永远不会在 CPU 上被调用。

### 设备函数（自动发现）

上面的 `add` 辅助函数**没有任何注解**。当编译器处理 `#[kernel]` 时，它会遍历调用图并自动发现内核调用的每个函数。这些函数会被编译为 PTX 设备函数，并由后端内联 —— 你无需手动标记它们。

> **注意**
> 
> `#[device]` 属性存在，但用途不同：它将函数标记为独立的设备编译根（用于构建 C++ 消费的 Rust 设备库），或在 `#[device] extern "C" { ... }` 块中用于声明与 CUDA C++ LTOIR 进行 FFI 的外部设备函数。对于从内核调用的私有辅助函数，你**不需要** `#[device]`。

### `#[cuda_module]`

`#[cuda_module]` 包装内联内核模块并生成类型化的主机 API：

```rust
let module = kernels::load(&ctx)?;
module.vecadd(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?;
```

加载器从主机二进制文件中读取嵌入的设备产物，缓存内核函数句柄，并将每个 `#[kernel]` 暴露为 Rust 方法。方法签名与内核签名一致，设备切片映射为 `DeviceBuffer` 借用。

`load_kernel_module` 和 `cuda_launch!` 作为底层 API 仍然可用，用于手动加载附属产物和自定义启动代码。

### 参数标量化

切片通过主机/设备 ABI 时，以其 `(ptr, len)` 组件传递 —— 主机将它们作为两个内核参数传递，设备编译器在入口基本块中重新组装切片。按值传递的结构体和闭包则作为一个 byval `.param` 传递，因此主机数据包将整个聚合体作为单个槽位推送（这与启动器实际行为一致，避免了按字段声明的不匹配）。所有这些都是完全透明的 —— 内核签名看起来仍然是普通的 Rust：

```
主机:   module.vecadd(..., &data, ...)
          → 提取切片的 (ptr, len)，传递两个参数

PTX:    .entry kernel(.param .u64 ptr, .param .u64 len, ...)
          → 接收扁平化的切片参数

设备:   内核体看到的是统一的 &[T] 切片
          → 编译器在入口处重建
```

### 动态结构体布局

当你向 GPU 传递结构体时，cuda-oxide 会向 rustc 查询每个字段的**精确字节偏移量**，并在设备端用显式填充重建布局。这意味着**不需要** `#[repr(C)]` —— 普通 Rust 结构体即可直接使用，即使在 HMM（GPU 直接访问主机内存）场景下也是如此。

---

## 走向异步

对于多内核流水线或并发工作负载，cuda-oxide 提供了基于 Tokio 的异步执行模型。我们来脚手架创建一个异步项目，并讲解其中的差异。

### 创建异步项目

```bash
cargo oxide new my_async_kernel --async
cd my_async_kernel
cargo oxide run
```

`--async` 标志会生成一个已预配置 `tokio` 和 `cuda-async` 依赖的项目。

### 完整示例

以下是生成的异步 vecadd 模板（为可读性做了少量格式调整）：

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_core::LaunchConfig;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::device_box::DeviceBox;
    use cuda_core::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    // 1. 初始化设备上下文映射（默认设备 0，1 个设备）。
    //    轮询流池在首次使用时惰性创建。
    init_device_contexts(0, 1)?;

    // 2. 从异步设备上下文加载嵌入的内核模块。
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    // 3. 分配设备内存并拷贝主机数据。
    let (a_dev, b_dev, mut c_dev) =
        cuda_async::device_context::with_cuda_context(0, |ctx| {
            let stream = ctx.default_stream();
            let bytes = N * mem::size_of::<f32>();
            unsafe {
                let a = malloc_async(stream.cu_stream(), bytes).unwrap();
                let b = malloc_async(stream.cu_stream(), bytes).unwrap();
                let c = malloc_async(stream.cu_stream(), bytes).unwrap();
                memcpy_htod_async(a, a_host.as_ptr(), bytes, stream.cu_stream()).unwrap();
                memcpy_htod_async(b, b_host.as_ptr(), bytes, stream.cu_stream()).unwrap();
                stream.synchronize().unwrap();
                (
                    DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
                )
            }
        })?;

    // 4. 启动 —— 返回一个惰性的 DeviceOperation，尚未执行 GPU 工作。
    module
        .vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )?
        .sync()?;  // 阻塞直到 GPU 完成。

    // 5. 将结果拷贝回主机。
    let mut c_host = vec![0.0f32; N];
    cuda_async::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    // 6. 验证。
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: 全部 {} 个元素正确", N);
    } else {
        eprintln!("FAILED: {} 个错误", errors);
        std::process::exit(1);
    }

    Ok(())
}
```

### 与同步版本相比的变化

内核本身是**完全相同的** —— 异步只改变你在主机端启动和管理 GPU 工作的方式。

**`{kernel}_async` 替代 `{kernel}`**

返回一个惰性的 `DeviceOperation` 而非立即启动。直到你显式调度它之前，不会发生任何 GPU 工作。这让你可以在提交资源之前构建计算图。

**`init_device_contexts(default_device, num_devices)`**

初始化线程本地设备上下文映射，设置默认 GPU 序号和多设备使用容量。轮询流池在首次使用时惰性创建。随后操作按轮询顺序分配到流，在无需手动流管理的情况下最大化 GPU 利用率。

**`DeviceBox` 替代 `DeviceBuffer`**

设备内存的异步安全包装器。与流池配合工作，并支持通过 `malloc_async` 进行异步分配。

**`.sync()` 与 `.await`**

`.sync()` 阻塞调用线程直到 GPU 完成 —— 当你主机端无事可做时使用。`.await` 挂起当前 Tokio 任务，在等待期间让其他任务继续执行 —— 当你有并发主机工作或多个 GPU 流水线在运行时，使用它。

**`and_then` / `zip!`**

用 `.and_then(|result| next_op)` 链式连接依赖操作。用 `zip!(op_a, op_b)` 并发运行独立操作 —— 两者都提交到流池，当两者都完成时组合结果可用。这些组合子让你能够以声明式方式表达复杂的多内核流水线。

> **提示**
> 
> 更完整的异步示例，请参阅 `async_mlp` —— 一个多内核前向传播（GEMM、MatVec、ReLU），使用 `and_then` 链式调用、`zip!` 并行分配，以及跨并发批次的 `Arc` 共享权重。在 cuda-oxide 工作区中运行 `cargo oxide run async_mlp` 即可体验。