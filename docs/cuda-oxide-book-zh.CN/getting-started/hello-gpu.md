# 编写你的第一个 kernel

本节会带你走完安装 cuda-oxide、创建项目、编写 GPU kernel 并运行它的全过程 —— 全程只用纯 Rust。

---

## 安装 cargo-oxide

如果还没装，用固定版本的 nightly 工具链安装这个构建工具：

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

验证你的环境配置正确：

```bash
cargo oxide doctor
```

它会检查兼容的 GPU、CUDA 工具包、LLVM 以及 codegen 后端。继续之前请先修掉它报告的所有问题（细节见[安装](installation.md)）。

---

## 创建项目

用 `cargo oxide new` 生成新项目的脚手架：

```bash
cargo oxide new my_first_kernel
cd my_first_kernel
```

这会生成一个开箱可跑的项目：

```text
my_first_kernel/
├── Cargo.toml          # dependencies on cuda-device, cuda-host, cuda-core
├── README.md           # doctor + run instructions
├── .gitignore          # target/ and generated device artifacts
├── rust-toolchain.toml # pins the required nightly toolchain
└── src/
    └── main.rs          # kernel + host code in one file
```

构建并运行它：

```bash
cargo oxide run
```

你应当看到 `PASSED: all 1024 elements correct`。生成的模板是一个向量加法 kernel —— 是个不错的起点，但我们来看点更有意思的。

---

## 一个 kernel 的解剖

下面这个向量加法有点花样：逐元素加法被抽成了一个普通的辅助函数。kernel 和辅助函数与宿主代码都在同一个文件里：

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

/// Plain helper function -- no annotation needed.
/// The compiler discovers it automatically because `vecadd` calls it.
fn add(a: f32, b: f32) -> f32 {
    a + b
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = add(a[i], b[i]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
    module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

    let c_host = c_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
    Ok(())
}
```

这里发生了不少事。我们把关键部分逐一拆开。

### 单源编译

kernel 与宿主代码位于**同一个文件**，用一次 `cargo` 命令调用就完成编译。codegen 后端会截获编译过程，把 `#[kernel]` 函数导向 MIR-to-PTX 流水线，其余一切交给标准 LLVM。最终的二进制里包含本机宿主代码以及嵌入的设备产物。

### `#[kernel]`

把一个函数标记为**可启动的 kernel 入口点** —— 相当于 GPU 世界的 `main`。该函数会经由这条流水线编译成 PTX：

```text
Rust source → MIR → Pliron IR → LLVM IR → PTX
```

同一个函数对宿主编译器也可见，用于类型检查，但它的函数体永远不会在 CPU 上被调用。

### 设备函数（自动发现）

上面的 `add` 辅助函数**没有任何标注**。当编译器处理一个 `#[kernel]` 时，它会遍历调用图，自动发现该 kernel 调用的每一个函数。这些函数会作为设备函数编译成 PTX，并由后端内联 —— 你不需要给它们加标记。

> **注意**：`#[device]` 属性确实存在，但用途不同：它把一个函数标记为独立的设备编译根（用于构建供 C++ 消费的 Rust 设备库），或者用在 `#[device] extern "C" { ... }` 块中，为与 CUDA C++ LTOIR 的 FFI 声明外部设备函数。对于从 kernel 调用的私有辅助函数，你**不需要** `#[device]`。

### `#[cuda_module]`

`#[cuda_module]` 包装这个内联的 kernel 模块，并生成带类型的宿主 API：

```rust
// SAFETY: this package owns the embedded device bundle produced for the
// kernels module above.
let module = unsafe { kernels::load(&ctx)? };
let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;
```

加载器从宿主二进制中读取嵌入的设备产物、缓存 kernel 函数句柄，并把每个 `#[kernel]` 暴露为一个 Rust 方法。该方法的签名与 kernel 签名一一对应，设备侧切片映射为 `DeviceBuffer` 借用。由于 `vecadd` 声明了 `#[launch_contract]`，模块还会生成 `prepare_vecadd`：它接收一个保持秩的 `LaunchConfig1D`，对照契约和当前设备的限值做校验，并返回一个 `PreparedLaunch<vecadd>` 证明，供安全的 `vecadd` 方法消费。加载是剩下唯一的一个 unsafe 步骤 —— 由调用者担保嵌入的产物与该模块声明的 ABI 一致。没有契约的 kernel 只暴露裸的、unsafe 的启动方法，因为裸 `LaunchConfig` 无法证明一个按一维索引的 kernel 拿到的是一维启动。完整的 prepared 路径见[启动 kernel](../gpu-programming/launching-kernels.md)。

`load_kernel_module` 与 `cuda_launch!` 作为更底层的 API 依然可用，用于手动加载 sidecar 产物和自定义启动代码。`cuda_launch!` 无法对照 kernel 签名检查参数，因此它是 unsafe 的，必须包在 `unsafe { }` 里。

### 参数标量化

切片以它们的 `(ptr, len)` 分量跨越宿主/设备 ABI —— 宿主把它们当作两个 kernel 参数传入，设备编译器在入口块中把切片重新组装起来。按值传递的结构体和闭包则以一个 byval `.param` 传输，因此宿主的数据包会把整个聚合体作为一个槽位推入（这与启动器实际的做法一致，避免与逐字段声明产生不匹配）。这一切都完全透明 —— kernel 签名看上去仍然是普通的 Rust：

```text
Host:   module.vecadd(&stream, &prepared, &data, ...)
          → extracts (ptr, len) for the slice, passes two args

PTX:    .entry kernel(.param .u64 ptr, .param .u64 len, ...)
          → receives flat slice parameters

Device: kernel body sees unified &[T] slice
          → compiler reconstructs at entry
```

### 动态结构体布局

当你把结构体传给 GPU 时，cuda-oxide 会向 rustc 查询每个字段**确切的字节偏移**，并在设备侧用显式填充重建布局。这意味着 `#[repr(C)]` **并非必需** —— 普通的 Rust 结构体直接可用，即使跨越 HMM（GPU 直接访问宿主内存）也一样。

---

## 走向异步

对于多 kernel 流水线或并发工作负载，cuda-oxide 提供了基于 Tokio 的异步执行模型。我们来生成一个异步项目，并走一遍差异。

### 创建异步项目

```bash
cargo oxide new my_async_kernel --async
cd my_async_kernel
cargo oxide run
```

`--async` 标志会生成一个预先配置好 `tokio` 与 `cuda-async` 依赖的项目。

### 完整示例

下面是生成的异步 vecadd 模板（为可读性做了少量格式调整）：

```rust
use cuda_async::simt::device_context::init_device_contexts;
use cuda_async::simt::device_operation::DeviceOperation;
use cuda_core::simt::LaunchConfig;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_box::DeviceBox;
    use cuda_core::simt::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    // 1. Initialize the device context map (default device 0, 1 device).
    //    The round-robin stream pool is created lazily on first use.
    init_device_contexts(0, 1)?;

    // 2. Load the embedded kernel module from the async device context.
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    // 3. Allocate device memory and copy host data.
    let (a_dev, b_dev, mut c_dev) =
        cuda_async::simt::device_context::with_cuda_context(0, |ctx| {
            let stream = ctx.default_stream();
            let num_bytes = N * mem::size_of::<f32>();
            unsafe {
                let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
                memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
                memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
                stream.synchronize().unwrap();
                (
                    DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                    DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
                )
            }
        })?;

    // 4. Launch -- returns a lazy DeviceOperation, no GPU work yet.
    // SAFETY: this is a 1D launch and `vecadd` guards its index against the
    // output length before writing.
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .sync()?;  // Block until the GPU finishes.

    // 5. Copy results back to host.
    let mut c_host = vec![0.0f32; N];
    cuda_async::simt::device_context::with_cuda_context(0, |ctx| {
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

    // 6. Verify.
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
```

### 与同步版本相比改了什么

kernel 本身**完全一样** —— 异步改变的只是你在宿主侧如何启动和管理 GPU 工作。

- **`{kernel}_async` 而不是 `{kernel}`** —— 返回一个惰性的 `DeviceOperation`，而不是立即启动。在你显式调度它之前，不会有任何 GPU 工作发生。这让你能在提交资源之前先把计算图搭好。

- **`init_device_contexts(default_device, num_devices)`** —— 初始化线程局部的设备上下文映射，设定默认 GPU 序号以及多设备使用时的容量。轮转流池在首次使用时惰性创建。此后操作会按轮转顺序分配到各条流上，无需手动管理流就能最大化 GPU 占用率。

- **`DeviceBox` 而不是 `DeviceBuffer`** —— 设备内存的异步安全包装。它与流池配合工作，并通过 `malloc_async` 支持异步分配。

- **`.sync()` 与 `.await`** —— `.sync()` 会阻塞调用线程直到 GPU 完成 —— 当你在宿主侧没有别的事可做时用它。`.await` 会挂起当前 Tokio 任务，让其它任务在等待期间继续推进 —— 当你还有并发的宿主工作或多个 GPU 流水线在飞时用它。

- **`and_then` / `zip!`** —— 用 `.and_then(|result| next_op)` 串联有依赖的操作。用 `zip!(op_a, op_b)` 并发运行相互独立的操作 —— 两者都被提交到流池，两个都完成后就能拿到合并的结果。这些组合子让你能以声明式表达复杂的多 kernel 流水线。

> **提示**：更完整的异步示例见 `async_mlp` —— 一个多 kernel 的前向传播（GEMM、MatVec、ReLU），用 `and_then` 串联、用 `zip!` 做并行分配，并在并发批次之间用 `Arc` 共享权重。在 cuda-oxide 工作区里用 `cargo oxide run async_mlp` 运行它。
