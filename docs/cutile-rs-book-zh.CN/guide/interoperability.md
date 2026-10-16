# 互操作

cuTile Rust 可以启动外部 CUDA kernel，也可以借用其它库的句柄：

- **集成外部 PTX 或 CUBIN kernel** —— 用于 CUDA C++ kernel、cuda-oxide 生成的 PTX，或者其它你希望与 cuTile kernel 一起启动的 CUDA 模块产物。
- **借用外部 CUDA 句柄** —— 把来自其它 Rust 绑定 crate（cudarc、Candle、手写 FFI）的 `CUcontext` / `CUstream` 包装起来，让 cuTile kernel 能在你已经持有的句柄上运行。
- **从其它 tile DSL 迁移** —— 从 Triton 和 cuTile Python 的概念映射。

外部 CUDA kernel 与 tile kernel 参与同一套 `DeviceOp` 执行模型 —— 共享流、用 `.then()` 串联，并且避免不必要的同步。

---

## 集成外部 PTX 或 CUBIN kernel

### 来自 CUDA C++

把你的 CUDA C++ kernel 编译成 PTX（可移植）或 `.cubin`（架构专用）：

```bash
# PTX —— 可跨 GPU 架构移植，加载时 JIT 编译。
nvcc -ptx -arch=compute_80 my_kernel.cu -o my_kernel.ptx

# cubin —— 针对单一架构预编译，没有 JIT 开销。
nvcc -cubin -arch=sm_80 my_kernel.cu -o my_kernel.cubin
```

> **架构可移植性：** `.cubin` 文件只能在它被编译时指定的那个确切的 SM 架构上运行。用 `-arch=sm_80` 编译出的代码无法在 `sm_100` GPU 上加载。PTX 没有这个问题 —— CUDA 驱动会在加载时为目标 GPU JIT 编译它，代价是一次性的编译延迟。除非你需要彻底消除 JIT 开销，否则优先使用 PTX。

### 来自 cuda-oxide

[cuda-oxide](https://nvidia.github.io/cuda-rust/cuda-oxide/) 是 CUDA Rust 的 SIMT kernel 编程模型；它与 cuTile Rust 并排放在 [cuda-rust 仓库](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide)里。它文档化的输出路径是 PTX：用 `cargo oxide build` 或 `cargo oxide pipeline` 构建或查看生成的 PTX，然后用 `load_module_from_ptx` 加载该 PTX，并把入口函数包装成 `AsyncKernelLaunch` 或带类型的 `DeviceOp`。

如果你需要 CUBIN，请把 PTX 交给常规 CUDA 工具链，作为一个单独的步骤来编译。

### 加载与启动

加载编译好的模块，并拿到入口函数的句柄：

```rust
use cuda_async::device_context::{load_module_from_file, load_module_from_ptx};

// 从 cubin：
let module = load_module_from_file("my_kernel.cubin", device_id)?;

// 或者从 PTX（加载时 JIT 编译）：
let ptx_src = include_str!("my_kernel.ptx");
let module = load_module_from_ptx(ptx_src, device_id)?;

let function = Arc::new(module.load_function("my_kernel_entry")?);
```

通过 `AsyncKernelLaunch` 启动 kernel，它是一个包装了 CUDA 驱动 kernel 启动 API 的 `DeviceOp`：

```rust
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher.push_arg(num_elements as u32);
launcher.push_arg(scale);
// 安全性：input 和 output 都是有效的设备分配，至少包含
// num_elements 个 f32 元素。output 被独占写入；input 是只读的。
// 两者都会保持分配状态，直到本次操作完成。
unsafe {
    launcher
        .push_device_ptr(input.device_pointer().cu_deviceptr())
        .push_device_ptr(output.device_pointer().cu_deviceptr());
}
launcher.set_launch_config(LaunchConfig {
    grid_dim: ((num_elements as u32 + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
});

// 作为 DeviceOp 执行 —— 与异步模型集成。
launcher.await?;
```

实现了 `DType` 的标量参数使用安全的 `push_arg` 方法。设备指针则需要 `unsafe { push_device_ptr() }`。调用者必须确认这些分配在目标 GPU 上有效、对 kernel 来说足够大，并且不存在冲突的访问。参数顺序与类型必须和 kernel 的签名一致。Rust 与 CUDA 驱动都无法校验这些条件；弄错可能导致未定义行为或 GPU 故障。

要防止数据竞争，请使用流来排序：在同一条流上用 `.then()` 串联的操作会按顺序执行，并且能看到彼此的写入。不同流上的操作则需要显式同步。

> **为什么生成的 cuTile kernel 不需要 `unsafe`：** `#[cutile::entry]` 宏生成的启动器在内部同样会调用 `push_device_ptr`，但它们是以安全方式调用的，因为框架掌控了两端 —— 设备指针来自框架管理的分配，而所有权模型（用 `Partition` 表示独占访问、用 `Arc<Tensor>` 表示共享读）在类型层面排除了别名。自定义 kernel 绕开了这一切：你推入的指针既不是框架分配的，框架也无法追踪，因此安全责任落到了你身上。

---

## 包装成安全的 DeviceOp

把自定义 kernel 的启动包装进一个实现了 `DeviceOp` 的结构体。结构体的带类型字段会强制约束正确的参数签名；`unsafe` 被限制在 `execute` 内部：

```rust
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_async::scheduling_policies::SchedulingPolicy;
use cuda_core::{Function, LaunchConfig};
use std::future::IntoFuture;

pub struct ScaleKernel {
    function: Arc<Function>,
    n: u32,
    scale: f32,
    input: Arc<Tensor<f32>>,
    output: Tensor<f32>,
}

impl DeviceOp for ScaleKernel {
    type Output = (Arc<Tensor<f32>>, Tensor<f32>);

    // execute 之所以是 unsafe，是因为它入队异步 GPU 工作而不做同步 ——
    // 返回的 tensor 可能仍在执行中。
    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let mut launcher = AsyncKernelLaunch::new(self.function);
        launcher.push_arg(self.n);
        launcher.push_arg(self.scale);
        // 安全性：input 和 output 都是框架管理的 Tensor 分配。
        // input 是共享的（Arc，只读）；output 被独占写入。
        unsafe {
            launcher
                .push_device_ptr(self.input.device_pointer().cu_deviceptr())
                .push_device_ptr(self.output.device_pointer().cu_deviceptr());
        }
        launcher.set_launch_config(LaunchConfig {
            grid_dim: ((self.n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        });
        unsafe { launcher.execute(ctx)? };
        Ok((self.input, self.output))
    }
}

// IntoFuture 是 DeviceOp 的父 trait。每个自定义 DeviceOp
// 都需要这段样板代码，才能支持 `.await` 和 `.sync()`。
impl IntoFuture for ScaleKernel {
    type Output = Result<(Arc<Tensor<f32>>, Tensor<f32>), DeviceError>;
    type IntoFuture = DeviceFuture<(Arc<Tensor<f32>>, Tensor<f32>), ScaleKernel>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}
```

这正是 `#[cutile::entry]` 宏为 tile kernel 生成安全启动器时所用的同一套模式 —— 调用点没有任何 `unsafe`。包装好之后，自定义 kernel 就能通过常见的 `DeviceOp` 组合子与 tile kernel 组合在一起。下面这条流水线先跑一个 tile 加法（`z = x + y`），再跑被包装的 scale kernel（`w = scale * z`）：

```rust
let (z_part, _x, _y) =
    tile_add::add(z.partition([tile_size]), x.clone(), y.clone()).await?;
let z: Tensor<f32> = z_part.unpartition();

let w: Tensor<f32> = zeros(&[num_elements]).await?;
let (_z, w) = ScaleKernel {
    function: scale_function,
    n: num_elements as u32,
    scale,
    input: Arc::new(z),
    output: w,
}
.await?;
```

完整、可运行的版本见 [`interop.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-rs-examples/examples/interop.rs)。

---

## 底层驱动访问

如果需要更直接的控制，用 `with_context` 拿到 CUDA 流并直接下发驱动 API 调用：

```rust
use cuda_async::device_operation::{with_context, value, DeviceOp};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{malloc_async, memcpy_htod_async, free_async};

let host_data: Vec<f32> = vec![1.0; num_elements];
let num_bytes = num_elements * std::mem::size_of::<f32>();

// host_data 是以引用方式捕获的 —— 它的生命周期必须长于这次 await，
// 这样异步 memcpy 才能在流同步之前一直从它读取。
let op = with_context(|ctx: &ExecutionContext| {
    let stream = ctx.get_cuda_stream();

    let dptr = unsafe {
        let dptr = malloc_async(num_bytes, stream).expect("device allocation");
        memcpy_htod_async(dptr, host_data.as_ptr(), num_elements, stream)
            .expect("host-to-device copy");
        dptr
    };

    value(dptr)
});

let dptr = op.await?;

// 清理：在一条流上释放设备内存。
with_context(move |ctx: &ExecutionContext| {
    unsafe { free_async(dptr, ctx.get_cuda_stream()) }.expect("device free");
    value(())
})
.await?;
```

`unsafe` 块内部的驱动调用必须满足 CUDA API 的安全要求。

---

## 借用外部 CUDA 句柄

如果你的应用已经通过别的 Rust 绑定 crate —— cudarc、Candle，或者手写的 `bindgen` 封装 —— 持有了 CUDA 句柄，你可以把它们包装成 cuTile 的 `Device`、`Stream`、`Module` 或 `Function`，而无需转移所有权。

`borrow_raw` 系列构造函数接收的是裸 C 原语（不透明句柄用 `*mut c_void`，`CUdevice` 用 `c_int`），而不是 `cuda_bindings` 的类型别名，因此不同绑定 crate 之间的名义类型差异不会造成阻碍：

```rust
use core::ffi::{c_int, c_void};
use cuda_core::{Device, Stream};

// `foreign` 是来自另一个 Rust 绑定 crate（cudarc、Candle、bindgen 封装……）
// 的句柄集合。它的 `CUcontext` / `CUstream` 类型别名与 cuTile 的名义上不同，
// 但底层 C ABI 完全一致 —— 在边界处转型，句柄即可原样流通。

// 安全性：调用者保证这些句柄有效、生命周期长于返回的包装器，
// 并且不会被并发销毁。
let device = unsafe {
    Device::borrow_raw(
        foreign.cu_ctx as *mut c_void,
        foreign.cu_device as c_int,
        foreign.ordinal,
    )
};
let stream = unsafe { Stream::borrow_raw(foreign.cu_stream as *mut c_void, &device) };

// 现在 cuTile kernel 就运行在这个借来的流上。
let result = my_kernel(out, x).sync_on(&stream)?;
```

借用来的包装器在被 drop 时会跳过销毁：`Stream` 不会调用 `cuStreamDestroy`，`Module` 不会卸载，`Device` 也不会释放主上下文。句柄生命周期的完全控制权仍留在源框架手中。

`Module::borrow_raw` 和 `Function::borrow_raw` 对预编译模块和已解析函数遵循同样的模式。端到端演练见 `cutile-rs-examples/examples/` 下的 `cudarc_interop` 示例。

---

## 从其它 DSL 迁移

**从 Triton。** [Triton](https://triton-lang.org/) 和 cuTile Rust 都让你以 tile 级操作来编写 kernel。Triton 中需要显式 warp 特化的许多模式，在 cuTile Rust 里由编译器隐式处理：

| Triton（手动） | cuTile Rust（自动） |
|-----------------|------------------------|
| 指定 producer warp 把 tile 从全局内存预取到共享内存 | 编译器为 `load_tile` 操作生成共享内存中转 |
| 指定 consumer warp 在共享内存 tile 上计算 | 编译器把 tile 算术映射到 Tensor Core 与寄存器 |
| 用 `tl.range` 里的 `warp_specialize` 做软件流水 | 在受支持的架构上，编译器使用张量内存加速器（TMA）指令实现硬件辅助的流水 |
| 手动在 warp 之间布置 `tl.dot` | `mma()` 直接映射到 Tensor Core 指令；线程/warp 的分配由编译器管理 |
| 为占用率调 `num_warps` 和 `num_stages` | `occupancy` 与 `num_cta_in_cga` 优化提示引导编译器 |

对于无法映射到 tile 模型的模式，请用 Triton 编译该 kernel，然后按上文所述通过 `AsyncKernelLaunch` 集成它。由于 Triton 输出 PTX，可以直接加载：

```rust
let module = load_module_from_ptx(triton_generated_ptx, device_id)?;
let function = Arc::new(module.load_function("gemm_kernel")?);
```

**从 cuTile Python。** 如果你熟悉 [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/)，kernel 侧的概念可以直接对应：

| cuTile Python | cuTile Rust |
|---------------|-------------|
| `@ct.kernel` | `#[cutile::entry()]` |
| `ct.load()` | `x.load_like(z)` |
| `ct.store()` | `tensor.store()` |
| `ct.bid(0)` | 通过分区隐式获得 |
| `ct.launch()` | 异步操作 + `.await` |

两个前端使用同一套底层的 Tile IR 编译流水线，生成等价的 GPU 代码；差别在于宿主语言及其类型系统。

---

接下来阅读[调试与性能分析](debugging-and-profiling.md)了解排查方法。互操作 kernel 使用的是[设备操作](device-operations.md)中描述的 `DeviceOp` 模型。
