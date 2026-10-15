# 9. 指针加法

裸设备指针让 kernel 能够直接访问内存。这个向量加法的例子把裸指针与异步执行结合使用 —— 此时调用者必须保证内存在 kernel 完成之前一直存活。

> **警告**：使用裸指针会绕开 cutile 的安全保证。只有在确实没有别的方式实现某个 kernel、或者要进一步压榨性能时，才使用它们。

---

```rust
use std::future::IntoFuture;
use cutile;
use cutile::api::{arange, ones, zeros};
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        let tensor = unsafe { make_tensor_view(ptr_tile, shape, strides, new_token_unordered()) };
        tensor
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(z_ptr, len) };
        let x_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(x_ptr, len) };
        let y_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(y_ptr, len) };
        let pid = program_id(0);
        let tile_shape = shape![4i32];
        let tile_x = x_tensor.partition(tile_shape).load([pid]);
        let tile_y = y_tensor.partition(tile_shape).load([pid]);
        z_tensor
            .partition_mut(tile_shape)
            .store(tile_x + tile_y, [pid]);
    }
}

use my_module::add_ptr;

#[tokio::main]
async fn main() -> Result<(), cutile::error::Error> {
    let len = 2usize.pow(5);
    let tile_size = 4usize;

    // 初始化 tensor。
    let z: Tensor<f32> = zeros(&[len]).await?;
    let x: Tensor<f32> = ones(&[len]).await?;
    let y: Tensor<f32> = ones(&[len]).await?;

    // 取出设备指针。
    let z_ptr = z.device_pointer();
    let x_ptr = x.device_pointer();
    let y_ptr = y.device_pointer();

    // 准备 kernel 启动。注意：由于我们传的是指针，所以必须用 unsafe。
    let op = unsafe { add_ptr(z_ptr, x_ptr, y_ptr, len as i32) }.grid((
        (len / tile_size) as u32,
        1,
        1,
    ));

    // spawn 一个异步任务来执行这个操作。
    let op_handle = tokio::spawn(op.into_future());

    // 注意：设备操作这些 tensor 期间，程序员并不受 UB 的保护！
    // 我们不需要结果。设备指针是 Copy 的（它们会被拷贝到设备内存）。
    let _ = op_handle.await.expect("Failed to execute tokio task.")?;

    let x_host = x.to_host_vec().await?;
    let y_host = y.to_host_vec().await?;
    let z_host = z.to_host_vec().await?;
    for i in 0..z_host.len() {
        let x_i: f32 = x_host[i];
        let y_i: f32 = y_host[i];
        let answer = x_i + y_i;
        println!("{} + {} = {}", x_i, y_i, z_host[i]);
        assert_eq!(answer, z_host[i], "{} != {} ?", answer, z_host[i]);
    }
    Ok(())
}
```

**输出：**

```text
1 + 1 = 2
1 + 1 = 2
... (32 lines total)
```

---

## kernel 逐步解析

辅助函数 `get_tensor` 用裸设备指针、长度和步长构造出一个 `Tensor` 视图。由于编译器无法验证该指针是否有效、也无法验证它引用的内存是否比 kernel 活得更久，这个函数 —— 以及 kernel 入口点 —— 都必须标记为 `unsafe`。

在 kernel 内部，手工构造出来的 tensor 与安全版本的 tensor 一样被分区和加载。唯一的差别在于它们的创建方式。

在宿主侧，`device_pointer()` 为一个已有 tensor 返回一个带类型的 `DevicePointer<T>` 句柄。kernel 启动器会把该句柄转换成 unsafe kernel 所期望的裸设备指针。kernel 完成之后，原来的 tensor 仍然拥有底层内存，因此我们可以用 `to_host_vec()` 把结果读回来。

---

<a id="the-danger-of-async-raw-pointers"></a>
## 异步执行下的指针生命周期

`tokio::spawn` 会在 kernel 完成之前就返回。在这段时间里 drop 或重新分配某个 tensor，可能让 kernel 访问已释放的内存，从而导致未定义行为。请在释放 tensor 之前 await 任务句柄。

---

## 要点

| 概念 | 含义 |
|---------|---------------|
| **`device_pointer()`** | 为某个 tensor 返回一个带类型的设备指针句柄。 |
| **`make_tensor_view`** | 在 kernel 内部由指针、形状和步长构造出一个 tensor。 |
| **`unsafe` kernel** | 只要裸指针绕开了类型系统的保证，就必须使用。 |
| **生命周期责任** | 程序员必须保证被指向的内存在 kernel 期间一直存活。 |

---

### 练习 1：加上边界检查

如果 `len` 不能被 tile 大小整除会发生什么？在 kernel 中加一个保护来处理余数。

### 练习 2：就地缩放

修改 kernel，用额外传入的一个标量参数对输出做常数倍缩放。

### 练习 3：安全封装

为这个 unsafe kernel 写一个安全的 Rust 封装函数，在宿主层面于启动之前校验指针长度。

---

## 参见

- [互操作](../guide/interoperability.md) —— 用 `AsyncKernelLaunch` 而非裸指针来集成预编译 CUDA C++ kernel 的结构化做法
- [张量与 tile](../guide/tensors-and-tiles.md) —— 使用基于指针的 kernel 之前，先了解 tensor 与 tile 的基础
- [DSL API](../reference/dsl-api.md#low-level-memory-ops) —— 指针 tile、tensor 视图、指针加载/存储以及原子操作
