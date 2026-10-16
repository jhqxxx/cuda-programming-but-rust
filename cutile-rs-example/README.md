# cuTile Rust 学习示例

本目录存放 [`cutile`](https://crates.io/crates/cutile)（cuTile Rust）的学习示例。
每个 `.rs` 文件就是一个独立可运行的 example 二进制。

> 目录名不叫 `examples/`，所以 Cargo **不会自动发现**这里的文件 —— 每个示例都必须在
> 根目录 `Cargo.toml` 里注册一条 `[[example]]`。这是刻意的取舍：换来的是示例目录名
> 可以和仓库里 `cpp/`、`python/` 的风格保持一致。

> **这是 tile 模型那条线。** cuTile 的函数体描述的是「一个 tile（≈ 一个 thread block）
> 该干什么」，和 SIMT 模型（一个线程干什么）是两种不同的抽象。仓库里 SIMT 的 Rust 线
> 在 [`cuda-oxide-example/`](../cuda-oxide-example/)，**两边的示例不互为对照**。

## 当前示例

| 文件 | 主题 | 根 `Cargo.toml` 里的注册名 |
| --- | --- | --- |
| `00-add.rs` | 向量加法 `z = x + y` | `00-add` |
| `01-hello-world.rs` | `program_id` / `num_programs` | `01-hello-world` |
| `02-vec-add.rs` | 🚧 空文件，还没写、也没注册 | — |

## 目录约定

| 约定 | 说明 |
| --- | --- |
| 文件名 | 两位数序号 + 短横线 + 主题，如 `00-add.rs`、`02-saxpy.rs`、`03-tiled-gemm.rs` |
| 入口 | 顶层写 `fn main()`，直接可运行 |
| 注册 | 根 `Cargo.toml` 里加一条 `[[example]]`（`name` 与文件名去掉 `.rs` 后一致） |
| 自检 | 在 `main` 里用 `assert!` 校验结果，避免只打印不验证 |

## 新增一个示例

1. 新建文件，例如 `cutile-rs-example/02-vec-add.rs`。
2. 在根目录 `Cargo.toml` **末尾**追加：

   ```toml
   [[example]]
   name = "02-vec-add"
   path = "cutile-rs-example/02-vec-add.rs"
   ```

3. 编译 / 运行：

   ```bash
   cargo check --example 02-vec-add   # 只做类型检查与 JIT 编译，不占用 GPU
   cargo run   --example 02-vec-add   # 真正在 GPU 上执行
   ```

   `path` 是相对于根 `Cargo.toml` 的路径，可以指向任意目录或文件名，不必和 `name` 同名。

## 示例模板

```rust
//! 示例：向量加法 z = x + y
//!
//! 运行：cargo run --example <名字>

use anyhow::Result;
use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    // kernel 体描述的是「一个 tile（= 一个 thread block）该干什么」。
    // `{[B]}` 是 tile 形状（B 由 partition 推断），`{[-1]}` 表示动态维。
    #[cutile::entry]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>, // 输出参数（&mut）约定放第一个
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z); // 取「本 block 负责的那块」x
        let ty = y.load_like(z);
        z.store(tx + ty); // 写回本 block 负责的那块 z
    }
}

fn main() -> Result<()> {
    // api::* 返回的是惰性 DeviceOp，不 sync() 就没有 GPU 工作。
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);

    // 先物化成具体 Tensor，再 partition（纯元数据，不做 GPU 工作），
    // 这样才能直接查 grid。grid = 张量 / partition = 1024/128 = 8。
    let z = api::zeros::<f32>(&[1024]).sync()?.partition([128]);
    assert_eq!(z.grid()?, (8, 1, 1));

    // 一次调用 = 一次 kernel 启动，在 grid 个 block 上并行跑 grid 份。
    // 返回元组按「形参声明顺序」位置对应；输入也会被一并返回。
    let (z, _x, _y) = kernel::add(z, x, y).sync()?;

    // 显存里的数据要先拷回主机才能打印。
    let z_host: Vec<f32> = z.unpartition().to_host_vec().sync()?;
    assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    println!("z[:8] = {:?}", &z_host[..8]);
    Ok(())
}
```

## 执行模型速记

- `api::*` 里的算子（`zeros` / `ones` / `arange` / `randn` …）**都是惰性的**，返回一个
  `DeviceOp`（本质是 future），不 `await` / `sync` 就不会有 GPU 工作发生。
- 两种驱动方式：
  - `.sync()` —— 用默认调度策略执行并阻塞等待结果。
  - `.sync_on(&stream)` —— 在指定流上执行并阻塞等待；同一条流上的操作保证按调用顺序执行。
- 因为是阻塞式 API，普通 `fn main()` 就够了，**不需要 tokio**。只有想用 `.await`
  把多个算子流水线化组合时，才需要引入异步运行时。
- 惰性算子的一个好用处：先拼好整条流水线再 `sync`，中间结果可以复用同一块显存。

## 环境要求

| 项 | 本机实测版本 |
| --- | --- |
| CUDA Toolkit | 13.3（`/usr/local/cuda`，`nvcc` 在 PATH 中） |
| GPU | NVIDIA GeForce RTX 3080 |
| 驱动 | 615.71.09（R615 = CUDA 13.4） |
| Rust | 1.99.0 |
| cutile | 0.4.0，取自 `NVIDIA/cuda-rust` 的 `bc46b1ff`（git，非 crates.io） |

`cargo check --example ...` 只需要工具链；`cargo run --example ...` 需要有可用的 GPU 和驱动。

## 参考

- crate 文档：`cargo doc -p cutile --no-deps --open`
- 官方 Book：<https://nvlabs.github.io/cutile-rs/>
- **源码与测试**：`cutile` 来自上游 git 仓库（见根 `Cargo.toml`），cargo 的检出在
  `~/.cargo/git/checkouts/cuda-rust-*/bc46b1f/cutile-rs/`：
  - `cutile/tests/gpu/*.rs` —— 能在 GPU 上跑通的完整用例
  - `cutile/tests/*.rs` —— 多为不依赖 GPU 的编译期用例
  - `cutile-rs-examples/examples/` —— 官方示例集（含 async 系列）
