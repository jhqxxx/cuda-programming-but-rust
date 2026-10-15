# cuTile Rust 学习示例

本目录存放 [`cutile`](https://crates.io/crates/cutile)（cuTile Rust）的学习示例。
每个 `.rs` 文件就是一个独立可运行的 example 二进制。

> 目录名不叫 `examples/`，所以 Cargo **不会自动发现**这里的文件 —— 每个示例都必须在
> 根目录 `Cargo.toml` 里注册一条 `[[example]]`。这是刻意的取舍：换来的是示例目录名
> 可以和仓库里 `use_cpp/`、`use_python/` 的风格保持一致。

## 目录约定

| 约定 | 说明 |
| --- | --- |
| 文件名 | 两位数序号 + 短横线 + 主题，如 `01-vector-add.rs`、`02-saxpy.rs`、`03-tiled-gemm.rs` |
| 入口 | 顶层写 `fn main()`，直接可运行 |
| 注册 | 根 `Cargo.toml` 里加一条 `[[example]]`（`name` 与文件名去掉 `.rs` 后一致） |
| 自检 | 在 `main` 里用 `assert!` 校验结果，避免只打印不验证 |

## 新增一个示例

1. 新建文件，例如 `cutile-example/01-vector-add.rs`。
2. 在根目录 `Cargo.toml` **末尾**追加：

   ```toml
   [[example]]
   name = "01-vector-add"
   path = "cutile-example/01-vector-add.rs"
   ```

3. 编译 / 运行：

   ```bash
   cargo check --example 01-vector-add   # 只做类型检查与 JIT 编译，不占用 GPU
   cargo run   --example 01-vector-add   # 真正在 GPU 上执行
   ```

   `path` 是相对于根 `Cargo.toml` 的路径，可以指向任意目录或文件名，不必和 `name` 同名。

## 示例模板

```rust
//! 示例 01：向量加法
//!
//! 运行：cargo run --example 01-vector-add

use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    // 注意：`&mut` 输出参数必须放在第一个位置。
    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let x = api::ones::<f32>(&[32]).sync_on(&stream)?;
    let y = api::ones::<f32>(&[32]).sync_on(&stream)?;
    let mut z = api::zeros::<f32>(&[32]).sync_on(&stream)?;

    // 把 32 个元素切成 8 个长度为 4 的 tile，交给 8 个线程块。
    kernels::add((&mut z).partition([4]), &x, &y).sync_on(&stream)?;

    let host: Vec<f32> = z.to_host_vec().sync_on(&stream)?;
    println!("{host:?}");
    assert!(host.iter().all(|&v| (v - 2.0).abs() < 1e-6));

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
| 驱动 | 595.99.02 |
| Rust | 1.99.0 |

`cargo check --example ...` 只需要工具链；`cargo run --example ...` 需要有可用的 GPU 和驱动。

## 参考

- crate 文档：`cargo doc -p cutile --no-deps --open`
- 官方 Book：<https://nvlabs.github.io/cutile-rs/>
- 最有价值的参考资料是 crate 自带的测试：`~/.cargo/registry/src/*/cutile-0.4.0/tests/`
  （`tests/gpu/*.rs` 是能在 GPU 上跑通的完整例子，`tests/*.rs` 多为不依赖 GPU 的编译期用例）
