# cuTile Rust

**cuTile Rust** 把 Rust 编写的 tile 程序编译成 CUDA kernel。

---

## 项目状态

这是一个处于早期研究阶段、正在积极开发中的项目。请预期其中存在 bug、尚未完成的功能，以及破坏性的 API 变更。你在自己项目中实际使用的反馈，能帮助我们决定接下来做什么。

---

## 快速上手

```rust
use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Error> {
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (_z, _x, _y) = kernel::add(z, x, y).sync()?;
    Ok(())
}
```

这个例子把**宿主侧**的 tensor 准备与**设备侧**的 tile 程序分开了。宿主负责构造 tensor、把可变输出切分成 128 个元素的块，并用 `.sync()` 启动生成的操作。

在 kernel 签名里，`z` 是独占的可变输出；`x` 和 `y` 是共享的只读输入。kernel 会把与输出分区形状匹配的输入 tile 相加，并把结果存入 `z`。

---

## 它带来了什么

**启动边界上的所有权**
可变 tensor 在启动前被切分成互不相交的若干块；不可变 tensor 则通过 `Arc` 共享。借用检查器覆盖的不只是宿主侧代码，还包括 GPU kernel 的参数。

**Tile 程序**
tile kernel 是写成作用于数据 tile 上的单线程程序。编译器负责把 tile 映射到 warp、block 和 Tensor Core 上；你不需要直接管理共享内存或线程索引。

**CUDA Tile IR 编译**
tile kernel 经由 CUDA Tile IR —— NVIDIA 的 tile 级编译器 IR —— 下降为 GPU cubin。在 B200 上，安全 API 在持久化 GEMM 上可达 2.07 PFlop/s（相当于 cuBLAS 的 96.4%）；安全映射版本的 kernel 与裸指针 Rust 基线的差距落在测量噪声范围内。

---

## 目录

### 教程

1. [你好，世界](tutorials/01-hello-world.md)
2. [向量加法](tutorials/02-vector-addition.md)
3. [SAXPY](tutorials/03-saxpy.md)
4. [矩阵乘法](tutorials/04-matrix-multiplication.md)
5. [融合 softmax](tutorials/05-fused-softmax.md)
6. [Flash Attention](tutorials/06-flash-attention.md)
7. [异步入门](tutorials/07-intro-to-async.md)
8. [数据并行 MLP](tutorials/08-data-parallel-mlp.md)
9. [指针加法](tutorials/09-pointer-addition.md)
10. [CUDA 图](tutorials/10-cuda-graphs.md)
11. [NVFP4 推理](tutorials/11-nvfp4-inference.md)
12. [DGX Spark 推理](tutorials/12-dgx-spark-inference.md)

### 用户指南

1. [简介](guide/introduction.md)
2. [宿主与设备](guide/host-vs-device.md)
3. [张量与 tile](guide/tensors-and-tiles.md)
4. [JIT 编译](guide/jit-compilation.md)
5. [设备操作](guide/device-operations.md)
6. [性能](guide/performance.md)
7. [自动调优](guide/autotuning.md)
8. [边界检查的放置](guide/bounds-check-placement.md)
9. [互操作](guide/interoperability.md)
10. [调试与性能分析](guide/debugging-and-profiling.md)
11. [实用的心智模型](guide/useful-mental-models.md)

### 参考

1. [DSL API](reference/dsl-api.md)
2. [宿主 API](reference/host-api.md)
3. [兼容性](reference/compatibility.md)
4. [术语表](reference/glossary.md)
