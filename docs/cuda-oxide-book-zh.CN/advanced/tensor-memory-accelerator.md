# Tensor Memory Accelerator (TMA)

从 Hopper 架构（SM 90）开始，NVIDIA GPU 中包含了一个专用硬件单元 —— **Tensor Memory Accelerator** —— 它可以在全局内存与共享内存之间搬移数据，而*不占用*线程的执行资源。不再需要块中的每个线程各自发出加载指令，而是由一个线程把描述符交给 TMA 引擎，硬件异步地完成整个传输。其余 127 个线程可以自由地做计算、加载其他数据，或者干脆在屏障处等待。

cuda-oxide 通过 `TmaDescriptor`、`cp_async_bulk_tensor_*` 函数族以及 `ManagedBarrier` 类型状态（typestate）API 暴露 TMA。本章介绍初始化设置、拷贝模式，以及 TMA 如何与屏障集成来构建高效的加载/计算流水线。

> **参见**
>
> [CUDA Programming Guide — Asynchronous Data Copies using TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/#tensor-memory-access)，其中有对硬件单元、swizzle 模式和支持的张量维度的完整描述。

---

## TMA 解决的问题

在[共享内存](shared-memory-and-synchronization.md)一章中，块中的每个线程都要参与从全局内存加载一个 tile：

```rust
TILE_A[ty * TILE + tx] = a[row * k + tile_offset + tx];
```

这能工作，但意味着块里全部 128（或 256、或 1024）个线程都要花时间计算地址并发出加载指令。对于较大的 tile，加载阶段可能主导整个 kernel 的运行时间。

TMA 用一条硬件指令取代了逐线程的加载循环。一个线程说"把这个二维区域从全局内存拷到共享内存"，剩下的都由 TMA 引擎处理 —— 包括地址计算、步长（stride）处理以及无 bank 冲突的 swizzle 写入。

![TMA 异步批量拷贝流水线。宿主构建一个编码了张量布局的 `TmaDescriptor`。在设备上，一个线程发出拷贝，所有线程在 mbarrier 上等待。TMA 引擎异步执行传输，完成时向屏障发信号，随后线程继续从共享内存计算。](images/tma-async-pipeline.svg)

*TMA 异步批量拷贝流水线。宿主构建一个编码了张量布局的 `TmaDescriptor`。在设备上，一个线程发出拷贝，所有线程在 mbarrier 上等待。TMA 引擎异步执行传输，完成时向屏障发信号，随后线程继续从共享内存计算。*

---

## TmaDescriptor —— 张量映射（tensor map）

`TmaDescriptor` 是一个 128 字节的不透明结构体，它编码了 TMA 引擎需要知道的关于一个张量的一切：基地址、维度、元素类型、步长以及 swizzle 模式。它使用 CUDA driver API 在**宿主**上构建：

```rust
use cuda_device::tma::TmaDescriptor;

// 在宿主上（简化版 —— 真正的调用有很多参数）
unsafe {
    cuTensorMapEncodeTiled(
        &mut desc as *mut TmaDescriptor as *mut _,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                       // 二维张量
        global_ptr,              // 全局内存中的基地址
        &dims,                   // [rows, cols]
        &strides,                // 每个维度的字节步长
        &box_dims,               // 每次拷贝的 tile 大小
        &elem_strides,           // box 内的元素步长
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, // 无 bank 冲突的 swizzle
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    );
}
```

随后描述符作为普通参数传给 kernel。在设备上，它被视为 `*const TmaDescriptor` —— 不透明、只读，并且它所描述的那块分配在其生命周期内始终有效。

> **提示**
>
> swizzle 模式是对性能影响最大的参数。`SWIZZLE_128B` 会重排共享内存中的字节布局，使得来自不同 warp 的 128 字节访问落到不同的 bank 上。如果不做 swizzle，朴素的二维 tile 加载常常会产生 bank 冲突。

---

## 全局到共享的拷贝（G2S）

`cp_async_bulk_tensor_*_g2s` 系列函数发出从全局内存到共享内存目标的 TMA 拷贝。它们支持一维到五维张量：

```rust
use cuda_device::barrier::Barrier;
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};

unsafe fn load_tile(
    smem_dst: *mut u8,
    desc: *const TmaDescriptor,
    tile_x: i32,
    tile_y: i32,
    bar: *mut Barrier,
) {
    cp_async_bulk_tensor_2d_g2s(smem_dst, desc, tile_x, tile_y, bar);
}
```

坐标（`tile_x`、`tile_y`）以描述符中编码的 box 维度为单位，指定要拷贝哪个 tile。屏障指针告诉 TMA 引擎去哪里发送完成信号。

### 只有一个线程发出拷贝

TMA 拷贝**不是**块级的集合操作。应当**恰好由一个线程**调用 `cp_async_bulk_tensor_*_g2s`。其他所有线程通过屏障机制参与：

```rust
if thread::threadIdx_x() == 0 {
    let token = bar.arrive_expect_tx(tile_bytes as u32);
    unsafe {
        cp_async_bulk_tensor_2d_g2s(dst, desc, x, y, bar.as_ptr() as *mut _);
    }
    bar.wait(token);
} else {
    let token = bar.arrive();
    bar.wait(token);
}
```

`arrive_expect_tx(bytes)` 告诉屏障：除了线程到达之外，还要再等待 `bytes` 字节的事务完成。当所有线程都已到达*并且* TMA 引擎已交付全部预期字节时，屏障才会触发。

---

## mbarrier 之舞

TMA 的完成追踪依赖 `ManagedBarrier`（或原始的 `mbarrier_*` 函数）。类型状态 API 在类型层面强制约束生命周期：

```rust
use cuda_device::SharedArray;
use cuda_device::barrier::{Barrier, ManagedBarrier, Ready, TmaBarrierHandle, Uninit};

#[kernel]
pub fn tma_load_kernel(desc: *const TmaDescriptor) {
    static mut BAR: SharedArray<Barrier, 1, 128> = SharedArray::UNINIT;

    let bar: TmaBarrierHandle<Ready> = unsafe {
        TmaBarrierHandle::<Uninit>::from_static(BAR.as_mut_ptr())
            .init_by(block_size, 0) // 线程 0 初始化，包含 fence + sync
    };

    // 发出 TMA 拷贝
    if thread::threadIdx_x() == 0 {
        let token = bar.arrive_expect_tx(TILE_BYTES as u32);
        unsafe {
            cp_async_bulk_tensor_2d_g2s(
                smem_ptr, desc, tile_x, tile_y, bar.as_ptr() as *mut _
            );
        }
        bar.wait(token);
    } else {
        let token = bar.arrive();
        bar.wait(token);
    }

    // 共享内存现在包含该 tile —— 可以安全读取
    // ...

    unsafe { bar.inval(); }
}
```

生命周期为：`Uninit` → `init()` → `Ready` → `arrive()` / `wait()` →
`inval()` → `Invalidated`。类型系统会阻止你在未初始化的屏障上调用 `wait`，或在已失效的屏障上调用 `arrive`。

### 关键的屏障规则

| 规则                                                  | 原因                                                                            |
| :---------------------------------------------------- | :----------------------------------------------------------------------------- |
| `init_by` 包含 `fence_proxy_async_shared_cta()`   | TMA 引擎需要能在共享内存中看到已初始化的屏障               |
| `arrive_expect_tx` 在拷贝之前，而不是之后         | 引擎开始写入之前，屏障必须知道预期的字节数 |
| 所有线程都必须到达（或被计入）         | 屏障在 `expected_count` 次到达 + `TX` 字节后触发                        |
| 完成后调用 `inval()`（如果不复用）               | 释放屏障的硬件资源                                         |

---

## 共享到全局的拷贝（S2G）

TMA 也支持相反方向。该模式使用提交组（commit group）而不是屏障：

```rust
use cuda_device::tma::{
    cp_async_bulk_commit_group, cp_async_bulk_tensor_2d_s2g, cp_async_bulk_wait_group,
};

unsafe {
    cp_async_bulk_tensor_2d_s2g(smem_src, desc, tile_x, tile_y);
    cp_async_bulk_commit_group();
    cp_async_bulk_wait_group(0); // 等待所有未完成的组
}
```

S2G 拷贝不使用屏障。相反，`commit_group` 把未完成的拷贝捆绑成一个组，`wait_group(n)` 则阻塞到至多还有 `n` 个组在途为止。`wait_group(0)` 会等待全部完成。

---

## 对齐要求

共享内存中的 TMA 目标必须 **128 字节对齐**。cuda-oxide 通过 `SharedArray` 和 `DynamicSharedArray` 上的 `ALIGN` 参数提供这一保证：

```rust
static mut TILE: SharedArray<f32, 1024, 128> = SharedArray::UNINIT;

// 或者使用动态共享内存：
let dst: *mut u8 = DynamicSharedArray::<u8, 128>::get();
```

如果目标不是 128 字节对齐，TMA 引擎会静默地产生错误结果或触发错误。这里没有运行时检查 —— 对齐必须在构造上就是正确的。

---

## 多播（multicast）变体

在启用集群的 Hopper 及更新架构上，TMA 可以把一次全局加载同时多播到多个 CTA 的共享内存中：

```rust
use cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast;

unsafe {
    cp_async_bulk_tensor_2d_g2s_multicast(
        dst, desc, x, y, bar_ptr,
        cta_mask,  // 集群中目标 CTA 的 u16 位掩码
    );
}
```

这对于 GEMM 风格的 kernel 特别有用 —— 其中 A 或 B 的同一个 tile 会被多个线程块需要。不必让每个块都加载自己的一份拷贝，一次 TMA 操作就能服务所有块。

> **参见**
>
> [集群编程](cluster-programming.md)介绍了完整的集群模型与分布式共享内存，它们与 TMA 多播天然搭配。

---

## TMA 与手动加载对比

| 特性                    | 手动的逐线程加载                  | TMA 批量拷贝                               |
| :-------------------------- | :----------------------------------- | :------------------------------------------ |
| 需要的线程            | 块中的全部线程                 | 1 个线程发出；硬件执行          |
| 地址计算         | 每线程计算（索引运算）              | 编码在描述符中                       |
| 用 swizzle 消除 bank 冲突  | 手动填充                       | 描述符层面的 swizzle 模式               |
| 完成追踪         | `sync_threads()`                     | 带 TX 追踪的 `mbarrier`                 |
| 与计算重叠        | 不可能（线程都在加载）   | 加载下一个 tile 的同时计算当前 tile |
| 最低架构要求        | 任何 CUDA GPU                         | Hopper（SM 90+）                             |

关键优势不在原始带宽 —— TMA 与手动加载可以达到相近的峰值吞吐。优势在于**线程利用率**：当 TMA 加载下一个 tile 时，所有线程都可以在当前 tile 上计算。这正是现代 GEMM 实现所依赖的**多级软件流水线**模式。

> **参见**
>
> - [共享内存与同步](shared-memory-and-synchronization.md) —— TMA 所取代的手动加载模式
> - [矩阵乘法加速器](matrix-multiply-accelerators.md) —— TMA 为 Tensor Core 供数的场景
