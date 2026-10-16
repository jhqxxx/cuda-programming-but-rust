# 宿主代码与设备代码

cuTile Rust 程序由两部分组成。宿主代码运行在 CPU 上，负责常规的 Rust 控制流、tensor 分配、流的选取、kernel 启动以及结果回读。设备代码运行在 GPU 上，描述每个 tile block 所要完成的工作。

```rust
use cutile::prelude::*;
use kernels::scale;

fn main() -> Result<(), cuda_async::error::DeviceError> {
    let device = cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;

    let x = api::ones::<f32>(&[32, 32]).sync_on(&stream)?;
    let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;

    let _ = scale((&mut z).partition([4, 4]), &x, 2.0f32).sync_on(&stream)?;
    Ok(())
}

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        alpha: f32,
    ) {
        let tile_x = x.load_like(z);
        z.store(tile_x * alpha);
    }
}
```

`main` 是宿主代码。它构造 tensor、切分可写输出，并同步返回的 `DeviceOp`。`scale` 是设备代码。它每个 tile block 执行一次，每次只看到一个可变的输出分区。

## 模块与入口点

`#[cutile::module]` 标记一个 Rust 模块，表示其中的函数可以被编译到 GPU 上。`#[cutile::entry()]` 则把某个函数标记为 kernel 入口点。入口点可以通过生成的启动器 API 从宿主代码调用。

入口点遵循四条规则：

1. 必须位于 `#[cutile::module]` 内部。
2. 可写的 tensor 参数使用静态 tile 形状，例如 `Tensor<f32, S>` 或 `Tensor<f32, { [BM, BN] }>`。
3. 只读的 tensor 参数可以使用动态维度，例如 `Tensor<f32, { [-1, -1] }>`。
4. kernel 把结果写入 tensor 参数，而不是通过返回值返回。

`#[cutile::module]` 内部未加标记的函数是设备函数。它们可以被入口点或其他设备函数调用，并在编译期被内联，但不能被直接启动。

## kernel 启动器

生成的启动器接受与设备侧 kernel 签名相对应的宿主侧取值：

| kernel 参数 | 宿主侧传入 | 设备侧视图 |
|---|---|---|
| `&Tensor<T, S>` | `&Tensor<T>`、`Arc<Tensor<T>>` 或 `Tensor<T>` | 只读 tensor |
| `&mut Tensor<T, S>` | `Partition<&mut Tensor<T>>` 或 `Partition<Tensor<T>>` | 可写分区 |
| 标量（`f32`、`i32` 等） | 同一个标量 | 同一个标量 |

可变 tensor 在启动前会被切分，从而让每个 tile block 写入互不相交的区域：

```rust
let mut z = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;
let _ = scale((&mut z).partition([4, 4]), &x, 2.0f32).sync_on(&stream)?;
```

只读 tensor 可以被借用、移动，或者用 `Arc` 共享。多个 tile block 可以并发读取同一个 tensor。

## 宿主类型与设备类型

两侧会出现同名的类型，但宿主类型与设备类型承载的信息并不相同：

| 类型 | 所在侧 | 参数化于 | 用途 |
|---|---|---|---|
| `Tensor<T>` | 宿主 | 元素类型 | 由 CPU 代码管理的 GPU 内存分配 |
| `Partition<Tensor<T>>` | 宿主 | 元素类型 | 拥有所有权的可写启动分区 |
| `Partition<&mut Tensor<T>>` | 宿主 | 元素类型 | 借用形式的可写启动分区 |
| `Arc<Tensor<T>>` | 宿主 | 元素类型 | 共享的只读输入 |
| `Tensor<E, S>` | 设备 | 元素类型与形状 | kernel 的 tensor 参数 |
| `Partition<E, S>` | 设备 | 元素类型与形状 | 只读 tensor 在设备侧的 tile 化视图 |
| `MappedPartitionMut<E, S, M>` | 设备 | 元素类型、tile 形状与映射形状 | 进阶的可写输出，允许一个 tile block 处理多个逻辑输出 tile |
| `Tile<E, S>` | 设备 | 元素类型与静态形状 | 驻留在寄存器中的计算值 |

宿主 tensor 使用运行时形状，因为分配大小本身就是普通的运行时数据。设备 tensor 和 tile 则把形状信息放进类型系统，这样编译器既能检查操作，也能对生成的代码做特化。

需要自定义输出网格遍历方式的 kernel（例如持久化 GEMM）会用到 `MappedPartitionMut`。宿主侧通过给可变分区做一次映射来创建它：

```rust
let z = z.partition([BM, BN]).map([4, 1], num_tile_blocks);
```

设备侧入口点按值接收它：

```rust
fn kernel<const MAP_SHAPE: [i32; 2]>(
    mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
) {
    for out_idx in z.iter_indices() {
        // 计算一个逻辑输出 tile。
        z.store(tile, out_idx);
    }
}
```

## DeviceOp 基础

tensor 构造函数、kernel 启动器、宿主侧回读以及组合辅助函数都返回 `DeviceOp`。`DeviceOp` 是对 GPU 工作的惰性描述。在被同步、被 `await`，或者被捕获进 CUDA 图之前，什么都不会执行。

```rust
let z = api::zeros::<f32>(&[32, 32]); // 此时还没有真正分配。
let z = z.sync_on(&stream)?;          // 操作在这里才真正执行。
```

kernel 启动器同样是 `DeviceOp`：

```rust
let op = scale((&mut z).partition([4, 4]), &x, 2.0f32);
let _ = op.sync_on(&stream)?;
```

完整的宿主侧执行模型在[设备操作](device-operations.md)中描述。

## 首次使用时的编译

调用生成的启动器时，可能会在启动之前先编译设备代码。一个**特化（specialization）**是指针对某个具体入口函数、目标 GPU 和一组编译期输入所编译出的一个 kernel 变体。某个特化首次启动时，会把捕获到的 Rust AST 编译成 Tile IR 字节码，再编译成 cubin。之后使用同一特化启动时会复用缓存好的二进制。

特化取决于元素类型、const 泛型取值、编译选项以及其他编译期参数。动态的 tensor 维度可以跨多次启动变化，而不会产生新的特化。[JIT 编译](jit-compilation.md)详细介绍了缓存与特化规则。

---

接下来阅读[张量与 tile](tensors-and-tiles.md)。
