# 简介

**cuTile Rust** 是面向 Rust 的一个安全的、基于 tile 的并行编程模型。kernel 会被自动映射到 Tensor Core、张量内存加速器（TMA）指令以及其他架构专用单元上，因此同一份源码可以跨多个 NVIDIA GPU 架构运行。

在宿主侧，API 负责设备端 tensor 的分配、把可变 tensor 切分以实现安全的并行访问，以及为只读 tensor 提供 `Arc` 包装的共享。kernel 启动器由 `#[cutile::entry]` 函数生成，并在首次使用时 JIT 编译；之后的启动会复用缓存好的二进制。执行过程是异步的，通过 `DeviceOp` 表达 —— 它是对 GPU 工作的惰性宿主侧描述。

---

## 第一个 kernel

cuTile Rust 的 kernel 是在一个逻辑上的 tile block 网格上并发执行的 GPU 程序。`#[cutile::entry()]` 属性把一个 Rust 函数标记为*入口点*：一个你可以从 Rust 程序里调用、但在 GPU 上执行的函数。

```rust
use cutile::prelude::*;
use my_module::add;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

fn main() -> Result<(), cuda_async::error::DeviceError> {
    let device = cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;

    let x = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
    let y = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
    let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;

    let _ = add((&mut z).partition([4, 4]), &x, &y).sync_on(&stream)?;
    Ok(())
}
```

`main` 运行在 CPU 上，负责在 GPU 上启动 `add`。`#[cutile::module]` 宏为 `add` 生成宿主侧启动器，而它的 `#[cutile::entry()]` 属性则把它标记为 kernel 入口点。

首次调用时，入口函数会经历 Rust AST -> Tile IR 字节码 -> cubin 的编译流程。之后以相同编译变体调用时，会复用缓存好的二进制。

---

## kernel 参数与启动

在宿主侧，生成的启动器对每个 kernel 参数都接受几种形式：

| kernel 参数 | 宿主侧传入 | kernel 看到的内容 |
|---|---|---|
| `&Tensor<T, S>` | `&Tensor<T>`、`Arc<Tensor<T>>` 或 `Tensor<T>` | `&Tensor<T, S>`（只读） |
| `&mut Tensor<T, S>` | `Partition<&mut Tensor<T>>` 或 `Partition<Tensor<T>>` | `&mut Tensor<T, S>`（一个 tile 形状的区域） |
| 标量（`f32` 等） | 同一个标量 | 同一个标量 |

分区（partition）会把一个 tensor 按固定的 tile 形状切成互不相交的区域，例如对二维 tensor 用 `partition([4, 4])`。每个 tile block 拿到一个分区元素，cuTile Rust 正是借此让 kernel 每次只获得对一个区域的可变访问，同时保证写入不重叠。

基于借用的形式（`&Tensor`、`Partition<&mut Tensor>`）让你可以在不移动 tensor 的前提下传入它们。kernel 通过借用直接写入 —— 不需要 `unpartition()`，也不需要捕获返回值。

最简单的启动模式是把所有东西都借出去：

```rust
let x = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
let y = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;

// 基于借用：z 会被就地写入。
let _ = add((&mut z).partition([4, 4]), &x, &y).sync_on(&stream)?;
```

启动器同样接受惰性的 `DeviceOp` 参数 —— 在 `.sync()` 或 `.await` 之前，一切都保持惰性：

```rust
let z = api::zeros(&[32, 32]).partition([4, 4]);
let x = api::ones::<f32>(&[32, 32]);
let y = api::ones::<f32>(&[32, 32]);

let (_z, _x, _y) = add(z, x, y).sync()?;
```

要做串联时，用 `.then()` 在同一条流上组合操作：

```rust
let result = allocate()
    .then(|buf| fill_kernel(buf))
    .then(|buf| process_kernel(buf))
    .sync()?;
```

---

## 张量与 tile

kernel 在 **tensor** 和 **tile** 之间搬运数据。tensor 是位于 GPU 全局内存中的多维数组。tile 是位于寄存器中的不可变多维数据片段。kernel 通常遵循同一个套路：

![数据流：从 tensor 加载到 tile，在寄存器中计算，再把结果存回 tensor](../_static/images/data-flow.svg)

1. 把 tensor 数据加载到 tile 中。
2. 在 tile 上做计算。
3. 把 tile 的结果存回 tensor。

tile 的形状是类型系统的一部分，所以许多形状和 dtype 相关的错误在 kernel 运行之前就会被发现。

---

## 适用场景

**在以下情况使用 cuTile Rust：**
- 你需要库中没有的自定义 GPU kernel
- 你希望通过融合多个操作来提升性能
- 你正在构建性能关键的 ML 基础设施
- 你需要在 GPU 代码上获得 Rust 的安全保证

**在以下情况不要使用 cuTile Rust：**
- 标准库操作（cuBLAS、cuDNN）已经够用
- 你需要跨 GPU *厂商* 的最大可移植性
- 你需要 Python 或 C++ 的 tile 编程支持；请改用 [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) 或 [CUDA Tile C++](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/)

> **注意**：如果你的算法需要 warp 级原语、显式的流水线控制、手动内存管理，或者自定义的 CUDA C++ kernel，请参见[互操作](interoperability.md)；自定义 kernel 可以和你的 tile kernel 参与同一套 `DeviceOp` 执行模型。

---

接下来阅读[宿主与设备代码](host-vs-device.md)，或者直接跳到[教程](../tutorials/01-hello-world.md)。
