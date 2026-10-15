# 虚拟内存管理与对等访问

CUDA 虚拟内存管理（VMM）把物理内存分配、虚拟地址预留、映射以及访问权限拆分为相互独立的操作。

它比 `DeviceBuffer` 更底层，后者在一次操作中完成内存分配并返回可直接使用的设备指针。当应用需要显式控制物理内存映射到何处、或哪些 GPU 可以访问它时，VMM 就派上了用场。

对等访问（P2P）是一种相关但独立的机制。它允许一个 CUDA 上下文通过受支持的互连（例如 NVLink 或 PCIe）直接访问与另一块 GPU 关联的内存。

本章涵盖：

* 查询两块 GPU 是否支持对等访问。
* 启用和禁用有方向的对等访问。
* 使用 VMM 分配物理设备内存。
* 预留虚拟地址区间。
* 把物理分配映射到这些区间中。
* 把映射授予一块或多块设备。
* 把一份物理分配映射到多个 GPU 可见的地址区间。
* 同步以及资源销毁顺序。

> **参见**
>
> 完整的集成测试见 [`cuda-core/tests/simt_vmm_p2p.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-core/tests/simt_vmm_p2p.rs)。
>
> 对应的实现为：
>
> * [`cuda-core/src/simt/vmm.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-core/src/simt/vmm.rs)
> * [`cuda-core/src/simt/peer.rs`](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-core/src/simt/peer.rs)

## VMM 与 P2P 解决的是不同问题

VMM 控制物理内存与虚拟地址之间的关系：

```text
Physical allocation
        |
        v
Virtual address reservation
        |
        v
Mapping
        |
        v
Per-device access permissions
```

P2P 控制一个 CUDA 上下文能否访问与另一块设备关联的内存：

```text
Context on GPU 0
        |
        | enable_peer_access(&gpu0, &gpu1)
        v
Memory accessible through GPU 1
```

多 GPU 应用可以同时使用这两种机制：

1. 在一台设备上分配物理内存。
2. 预留虚拟地址区间。
3. 把物理分配映射到这些区间中。
4. 授予两台设备对这些映射的访问权限。
5. 在所需的方向上启用对等访问。

两者互不蕴含。VMM 映射仍然需要显式的访问权限，而当某个上下文需要直接访问与对等上下文关联的内存时，也必须启用对等访问。

## 硬件与拓扑要求

对等访问取决于物理 GPU 拓扑、CUDA 驱动以及平台配置。同一台机器中的两块 GPU 未必能够直接相互访问。

在启用对等访问之前，务必先查询支持情况：

```rust
use cuda_core::simt::peer;
use cuda_core::{CudaContext, DriverError};

fn check_peer_access() -> Result<(), DriverError> {
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if peer::can_access_peer(&ctx0, &ctx1)? {
        println!("GPU 0 can access GPU 1");
    } else {
        println!("GPU 0 cannot access GPU 1");
    }

    Ok(())
}
```

`can_access_peer(from, to)` 只是一个能力查询。它不会改变任何一个上下文。

> **注意**
>
> 对等访问是有方向的。GPU 0 到 GPU 1 的支持不应被视为反方向也已启用的证明。

## 启用对等访问

使用 `peer::enable_peer_access` 允许一个上下文访问与另一块设备关联的内存：

```rust
use cuda_core::simt::peer;
use cuda_core::{CudaContext, DriverError};

fn enable_bidirectional_peer_access() -> Result<(), DriverError> {
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if !peer::can_access_peer(&ctx0, &ctx1)? {
        return Ok(());
    }

    peer::enable_peer_access(&ctx0, &ctx1)?;
    peer::enable_peer_access(&ctx1, &ctx0)?;

    // 现在两个上下文都可以发起对等访问了。

    peer::disable_peer_access(&ctx0, &ctx1)?;
    peer::disable_peer_access(&ctx1, &ctx0)?;

    Ok(())
}
```

方向由参数顺序决定：

| 调用                               | 效果                                    |
| :--------------------------------- | :-------------------------------------- |
| `enable_peer_access(&ctx0, &ctx1)` | GPU 0 的上下文可以访问 GPU 1 的内存     |
| `enable_peer_access(&ctx1, &ctx0)` | GPU 1 的上下文可以访问 GPU 0 的内存     |
| 两次调用                           | 双向对等访问                            |

当访问已经启用时调用 `enable_peer_access` 会成功。同样，禁用当前并未启用的访问会被视为成功的空操作。

> **警告**
>
> 在禁用对等访问之前，请先完成所有使用对等内存的操作。如果在 kernel 或拷贝仍在使用对等可见内存时禁用访问，可能会使这些操作失效。

## VMM 生命周期

一次 VMM 分配有四个显式阶段：

1. 查询设备的分配粒度。
2. 分配物理内存。
3. 预留一个虚拟地址区间，并把物理内存映射进去。
4. 授予一台或多台设备对该映射的访问权限。

主要的 API 有：

| API                                         | 用途                                    |
| :------------------------------------------ | :-------------------------------------- |
| `vmm::allocation_granularity(device)`       | 查询所需的分配粒度                      |
| `vmm::align_size(size, granularity)`        | 把大小向上取整到该粒度                  |
| `PhysicalAllocation::new(device, size)`     | 分配物理内存                            |
| `VirtualReservation::new(size, alignment)`  | 预留一个虚拟地址区间                    |
| `Mapping::new(va, size, &physical, offset)` | 把物理内存映射到某个 VA 区间            |
| `vmm::set_access(va, size, devices)`        | 授予设备对已映射区间的访问权限          |

## 分配粒度

CUDA VMM 操作要求分配大小和映射大小都是设备分配粒度的整数倍。

在分配之前使用 `allocation_granularity` 和 `align_size`：

```rust
use cuda_core::simt::vmm;
use cuda_core::{CudaContext, DriverError};

fn aligned_allocation_size(requested: usize) -> Result<usize, DriverError> {
    let ctx = CudaContext::new(0)?;
    let granularity = vmm::allocation_granularity(ctx.cu_device())?;

    Ok(vmm::align_size(requested, granularity))
}
```

不要假设像 4 KiB 这样页大小的值就能满足 CUDA VMM 的粒度要求。所需的值取决于设备。

## 单 GPU VMM 映射

下面的示例分配物理内存，把它映射到一个虚拟地址区间，通过该映射写入数据，然后读回。

```rust
use cuda_core::simt::vmm;
use cuda_core::{CudaContext, DriverError};

fn single_gpu_vmm_roundtrip() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let granularity = vmm::allocation_granularity(ctx.cu_device())?;
    let allocation_size = vmm::align_size(4096, granularity);

    // 阶段 1：分配物理内存。
    let physical =
        vmm::PhysicalAllocation::new(ctx.cu_device(), allocation_size)?;

    // 阶段 2：预留虚拟地址空间。
    let reservation =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    // 阶段 3：把物理分配映射到预留区间中。
    let mapping = vmm::Mapping::new(
        reservation.base(),
        allocation_size,
        &physical,
        0,
    )?;

    // 阶段 4：授予 GPU 0 对该映射的读/写访问权限。
    vmm::set_access(
        reservation.base(),
        allocation_size,
        &[ctx.cu_device()],
    )?;

    let input: Vec<u8> = (0..64).map(|i| (i * 3 + 7) as u8).collect();

    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            reservation.base(),
            input.as_ptr(),
            input.len(),
            stream.cu_stream(),
        )?;
    }

    ctx.synchronize()?;

    let mut output = vec![0u8; input.len()];

    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            reservation.base(),
            output.len(),
            stream.cu_stream(),
        )?;
    }

    ctx.synchronize()?;

    assert_eq!(output, input);

    // 资源必须按依赖顺序销毁。
    drop(mapping);
    drop(reservation);
    drop(physical);

    Ok(())
}
```

`Mapping::new` 本身并不会让被映射的地址变得可访问。在 `vmm::set_access` 至少授予一台设备访问权限之前，该映射始终不可用。

## 为两块 GPU 映射同一份物理分配

VMM 允许把一份物理分配映射到多个虚拟地址区间中。

下面的模式会：

* 在 GPU 0 上分配物理内存。
* 为 GPU 0 创建一个虚拟地址映射。
* 为 GPU 1 创建另一个虚拟地址映射。
* 授予两台设备对这两个映射的访问权限。
* 通过 GPU 0 的映射写入。
* 通过 GPU 1 的映射读取同一份物理内存。

```rust
use cuda_core::simt::{peer, vmm};
use cuda_core::{CudaContext, DriverError, IntoResult};
use std::mem::MaybeUninit;

fn gpu_count() -> Result<usize, DriverError> {
    unsafe {
        cuda_core::init(0)?;
    }

    let mut count = MaybeUninit::uninit();

    unsafe {
        cuda_core::sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

fn cross_gpu_vmm_roundtrip() -> Result<(), DriverError> {
    if gpu_count()? < 2 {
        println!("This example requires at least two CUDA devices.");
        return Ok(());
    }

    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if !peer::can_access_peer(&ctx0, &ctx1)? {
        println!("The selected devices do not support peer access.");
        return Ok(());
    }

    peer::enable_peer_access(&ctx0, &ctx1)?;
    peer::enable_peer_access(&ctx1, &ctx0)?;

    let granularity =
        vmm::allocation_granularity(ctx0.cu_device())?;
    let allocation_size =
        vmm::align_size(4096, granularity);

    // 物理内存只分配一次，在 GPU 0 上。
    let physical =
        vmm::PhysicalAllocation::new(
            ctx0.cu_device(),
            allocation_size,
        )?;

    // GPU 0 的虚拟地址区间。
    let reservation0 =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    let mapping0 = vmm::Mapping::new(
        reservation0.base(),
        allocation_size,
        &physical,
        0,
    )?;

    vmm::set_access(
        reservation0.base(),
        allocation_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )?;

    // GPU 1 的虚拟地址区间，由同一份物理分配支撑。
    let reservation1 =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    let mapping1 = vmm::Mapping::new(
        reservation1.base(),
        allocation_size,
        &physical,
        0,
    )?;

    vmm::set_access(
        reservation1.base(),
        allocation_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )?;

    let input: Vec<u32> = (0..256).collect();
    let byte_length =
        input.len() * std::mem::size_of::<u32>();

    let stream0 = ctx0.default_stream();
    ctx0.bind_to_thread()?;

    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            reservation0.base(),
            input.as_ptr(),
            byte_length,
            stream0.cu_stream(),
        )?;
    }

    ctx0.synchronize()?;

    let mut output = vec![0u32; input.len()];
    let stream1 = ctx1.default_stream();
    ctx1.bind_to_thread()?;

    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            reservation1.base(),
            byte_length,
            stream1.cu_stream(),
        )?;
    }

    ctx1.synchronize()?;

    assert_eq!(output, input);

    // 在销毁其对应的预留和底层分配之前先销毁映射。
    drop(mapping1);
    drop(mapping0);
    drop(reservation1);
    drop(reservation0);
    drop(physical);

    peer::disable_peer_access(&ctx0, &ctx1)?;
    peer::disable_peer_access(&ctx1, &ctx0)?;

    Ok(())
}
```

## 虚拟地址不保证相同

`VirtualReservation::new(size, 0)` 请求 CUDA 驱动选择一个虚拟地址。因此，由同一份物理分配支撑的两个预留可能具有不同的基地址：

```text
GPU 0 VA: 0x0000_7f10_0000_0000
                         \
                          same physical allocation
                         /
GPU 1 VA: 0x0000_7e80_0000_0000
```

当前 API 支持共享物理支撑，但它并不保证每块 GPU 都在相同的数值虚拟地址上看到这块内存。

代码必须使用与当前访问路径相关联的那个预留所返回的 `base()`。

> **警告**
>
> 不要把这个模式描述为一个完整的对称堆（symmetric heap）。对称堆要求参与的各设备具有等价的虚拟地址，而当前由驱动选择预留的 API 并不保证这一点。

## 所有权与销毁顺序

VMM 包装类型通过 RAII 拥有 CUDA 资源：

| 类型                 | CUDA 资源                      | drop 操作          |
| :------------------- | :----------------------------- | :----------------- |
| `PhysicalAllocation` | 通用物理分配句柄               | `cuMemRelease`     |
| `VirtualReservation` | 预留的虚拟地址区间             | `cuMemAddressFree` |
| `Mapping`            | 物理到虚拟的映射               | `cuMemUnmap`       |

依赖顺序为：

```text
PhysicalAllocation   VirtualReservation
        ^                    ^
         \                  /
          \                /
               Mapping
```

一个 `Mapping` 会消费一个 `PhysicalAllocation` 和一个 `VirtualReservation`，因此两者都必须比它活得更久。

按相反顺序销毁资源：

```rust
ctx0.synchronize()?;
ctx1.synchronize()?;

drop(mapping1);
drop(mapping0);

drop(reservation1);
drop(reservation0);

drop(physical);
```

多 GPU 操作的完整拆除顺序是：

1. 停止提交使用这些映射的操作。
2. 同步所有正在使用它们的流或上下文。
3. drop 所有 `Mapping` 值。
4. drop 所有 `VirtualReservation` 值。
5. drop `PhysicalAllocation`。
6. 如果不再需要，禁用对等访问。

> **警告**
>
> Rust 并未在类型系统中编码这些映射依赖关系。一个 `Mapping` 并不会在其整个生命周期内借用它的 `PhysicalAllocation` 或 `VirtualReservation`。应用需要自行负责维持所需的 drop 顺序。

## 不要把 VMM 指针包装进拥有型缓冲区

一个 VMM 虚拟地址归其 `Mapping`、`VirtualReservation` 和 `PhysicalAllocation` 所有。

除非 API 明确定义了非拥有型转换，否则不要用该指针构造拥有型的 `DeviceBuffer`。拥有型缓冲区可能会尝试用 `cuMemFree` 释放该指针，而 VMM 资源期望的是 `cuMemUnmap`、`cuMemAddressFree` 和 `cuMemRelease`。

只把原始 `CUdeviceptr` 传给那些接受非拥有型设备地址的 API。

## 错误处理与不受支持的系统

可移植的多 GPU 应用应当分别处理这些情况：

| 情况                              | 推荐行为                                  |
| :-------------------------------- | :---------------------------------------- |
| 设备少于两台                      | 跳过 P2P 路径                             |
| `can_access_peer` 返回 `false`    | 使用经宿主中转的传输或另一条路径          |
| VMM 分配失败                      | 报告 CUDA 驱动错误                        |
| 访问权限设置失败                  | 销毁映射并回退                            |
| 对等访问已经启用                  | 继续                                      |
| 对等访问已经禁用                  | 继续                                      |

当机器上的 GPU 少于两台、或所选拓扑不支持 P2P 时，cuda-oxide 集成测试会提前返回。这使测试套件可以在单 GPU 开发系统上运行，同时在兼容的多 GPU 宿主上仍然演练该特性。

## API 摘要

### VMM

| API                                        | 说明                                         |
| :----------------------------------------- | :------------------------------------------- |
| `vmm::allocation_granularity(device)`      | VMM 操作所需的大小粒度                       |
| `vmm::align_size(size, granularity)`       | 把请求的大小向上取整                         |
| `PhysicalAllocation::new(device, size)`    | 在设备上分配物理内存                         |
| `PhysicalAllocation::handle()`             | 返回原始分配句柄                             |
| `PhysicalAllocation::size()`               | 返回物理分配的大小                           |
| `VirtualReservation::new(size, alignment)` | 预留一个虚拟地址区间                         |
| `VirtualReservation::base()`               | 返回预留的基地址                             |
| `VirtualReservation::size()`               | 返回预留区间的大小                           |
| `Mapping::new(va, size, physical, offset)` | 把物理内存映射到虚拟区间中                   |
| `Mapping::va()`                            | 返回已映射的虚拟地址                         |
| `Mapping::size()`                          | 返回已映射的大小                             |
| `vmm::set_access(va, size, devices)`       | 授予所选设备读/写访问权限                    |

### P2P

| API                                   | 说明                     |
| :------------------------------------ | :----------------------- |
| `peer::can_access_peer(from, to)`     | 查询拓扑支持情况         |
| `peer::enable_peer_access(from, to)`  | 启用有方向的对等访问     |
| `peer::disable_peer_access(from, to)` | 禁用有方向的对等访问     |

## 下一步

对于生产环境的多 GPU 代码，请为 P2P 不可用的系统添加一条显式的回退路径。一种常见的回退做法是通过锁页的宿主内存中转数据，而不是假设每种多 GPU 拓扑都支持直接访问。
