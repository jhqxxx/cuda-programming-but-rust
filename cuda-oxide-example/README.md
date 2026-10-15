# cuda-oxide 学习示例

本目录存放 [cuda-oxide](https://github.com/NVIDIA/cuda-rust)（NVIDIA 的 Rust GPU 编译器后端）
的学习示例。

## 为什么这里是一个「独立区域」

本仓库根目录钉的是 **stable 1.99.0**，而 cuda-oxide 必须要 **nightly**：

- 它的 codegen 后端 `rustc-codegen-cuda` 直接链接 rustc 内部结构，依赖只有 nightly
  才提供的 `rustc-dev` 组件；
- rustup 是从当前目录**向上**查找 `rust-toolchain.toml` 的，而一个 cargo 工作区只能被
  一个工具链构建 —— 两者无法共存在同一个工作区里。

所以本目录自带一份 `rust-toolchain.toml`，并且在仓库根 `Cargo.toml` 里被
`exclude` 掉。这也是上游 `cuda-rust` 仓库自己对 `cuda-oxide/`、`cutile-rs/` 采用的做法。

```
cuda-programming-but-rust/
├── Cargo.toml              # [workspace] exclude = ["cuda-oxide-example"]
├── rust-toolchain.toml     # 1.99.0 stable → 主 crate + cutile-example
├── cutile-example/         # 用 stable：[[example]] 注册，cargo run --example
└── cuda-oxide-example/     # 用 nightly：nightly-2026-10-11
    ├── rust-toolchain.toml # 兜底：即使某个例子漏写也能拿到 nightly
    └── 01-vector-add/      # 每个例子 = 独立、自包含的 cuda-oxide 项目
        ├── Cargo.toml      # [package] + [workspace] + 三个 cuda-* git 依赖
        ├── rust-toolchain.toml # 让例子能整目录拷走
        └── src/main.rs
```

### 与 `cutile-example/` 的关键差异

`cutile-example/` 用的是根 `Cargo.toml` 里的 `[[example]]` 注册机制，`cuda-oxide-example/`
**不能**照抄这套：`#[kernel]` / `#[cuda_module]` 必须由 `cargo oxide` 驱动的自定义
codegen 后端编译，普通 `cargo build --example` 编译不了。

还有一个必须遵守的约束：**独立模式下 `cargo oxide run` 会忽略例子名**。新版源码
（`commands/build_run.rs`）里是：

```rust
let example_dir = if ctx.is_workspace {
    resolve_example_dir(ctx, example)   // 仓库内部：按名字取子目录
} else {
    ctx.workspace_root.clone()          // 独立项目：只认当前目录
};
```

所以「多个例子放在一个项目下，用 `cargo oxide run <名字>` 切换」**只在 cuda-oxide
仓库内部（workspace 模式）可用**。独立项目下每个例子必须是自己的项目根，
`cd` 进去后直接 `cargo oxide run`（不带参数）。

## 目录约定

| 约定 | 说明 |
| --- | --- |
| 目录名 | 两位数序号 + 短横线 + 主题，如 `01-vector-add`、`02-saxpy`、`03-shared-memory` |
| 包名 | **不能以数字开头**（Cargo 限制），所以目录 `01-vector-add` 对应 `name = "vector_add"` |
| 结构 | 每个例子是一个**完整独立项目**：`Cargo.toml` + `rust-toolchain.toml` + `src/main.rs`，且 `Cargo.toml` 必须带一个空的 `[workspace]` 把自己声明为工作区根 |
| 依赖 | `cuda-device` / `cuda-host` / `cuda-core` 三者必须来自**同一 git 仓库 + 同一 `rev`**，否则 cargo oxide 会拒绝构建 |
| 自检 | 在 `main` 里用 `assert!` 校验结果，避免只打印不验证 |

> 运行方式是 `cd <例子目录>` 后直接 `cargo oxide run`，**不带例子名**。

## 运行

进入想跑的例子目录再执行；`rust-toolchain.toml` 在上一层，rustup 会自动向上找到它。

```bash
cd cuda-oxide-example/01-vector-add

cargo oxide doctor   # 检查环境：工具链、组件、nvcc、GPU、后端缓存
cargo oxide setup    # 首次：构建并缓存 codegen 后端（全局共享，只需一次）
cargo oxide run      # 构建本例并在 GPU 上运行
```

## ⚠️ 首次运行的一次性成本

配置组合是「git 依赖钉到 `bc46b1ff` + nightly-2026-10-11」，首次要付两项成本：

1. **下载 `nightly-2026-10-11`**（含体积很大的 `rustc-dev`）—— 已完成。
   用国内镜像会快很多（官方源曾出现连接卡在初始拥塞窗口、只跑 12 KB/s 的情况）：

   ```bash
   RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup \
     rustup toolchain install nightly-2026-10-11 \
     -c rust-src -c rustc-dev -c rust-analyzer -c clippy -c rustfmt -c llvm-tools \
     --profile minimal
   ```

   注意 `rustup toolchain install` 的 `--component` **只能接一个值**，多个组件必须写成多个 `-c`。

2. **编译 codegen 后端** —— 产出 `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`
   （约 340 MB）。它是**全局共享、按 commit 键控**的，所有例子只在第一次构建。

### 踩过的坑

- **cargo-oxide 必须和仓库布局匹配。** 仓库已从独立的 `NVlabs/cuda-oxide` 合并成
  `NVIDIA/cuda-rust` monorepo，后端源码从 `crates/rustc-codegen-cuda` 挪到了
  `cuda-oxide/crates/rustc-codegen-cuda`。合并前那份 cargo-oxide 找不到新位置，会报：

  ```
  Failed to run cargo build for rustc-codegen-cuda: No such file or directory (os error 2)
  ```

  解决办法是从当前 checkout 重装（版本号不变，但代码是新的）：

  ```bash
  cargo uninstall cargo-oxide
  cargo install --path /home/jhq/rust_code/cuda-rust/cuda-oxide/crates/cargo-oxide
  ```

- 别把例子做成「一个项目根 + 多个子目录」，独立模式下 `cargo oxide run <名字>` 会忽略名字
  （理由见上文）。

- 工具链与 commit 不匹配时 cargo oxide 会**直接报错拒绝**（`refuse_unloadable_backend`），
  不会静默产出坏结果。

## 示例模板

```rust
//! 示例 NN：一句话说明
//!
//! 运行：cd NN-xxx && cargo oxide run

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// ---- Device 侧：编译成 PTX ----
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn my_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let raw = idx.get();
        if let Some(out) = output.get_mut(idx) {
            *out = input[raw] * 2.0;
        }
    }
}

// ---- 主机侧：编译成 x86_64 机器码 ----
fn main() {
    let ctx = CudaContext::new(0).expect("无法创建 CUDA 上下文");
    let stream = ctx.default_stream();

    const N: usize = 1000;
    let host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("加载模块失败");
    // SAFETY：启动配置与内核资源需求匹配，缓冲区覆盖其访问范围。
    unsafe {
        module.my_kernel(&stream, LaunchConfig::for_num_elems(N as u32), &dev, &mut out_dev)
    }
    .expect("内核启动失败");

    let out = out_dev.to_host_vec(&stream).unwrap();
    assert!(out.iter().zip(&host).all(|(o, h)| (*o - h * 2.0).abs() < 1e-5));
    println!("✓ 校验通过");
}
```

## 执行模型速记

- **主机代码和 device 代码写在同一个文件里**，一次编译，产物是一个同时含 x86_64 机器码
  和嵌入式 PTX 的二进制 —— 不需要 CUDA C++ 那种 `.cu` + `__global__` 的分离写法。
- 内核必须放在 `#[cuda_module]` 模块里并标 `#[kernel]`；它同时存在于主机 MIR 和
  device PTX 两个世界（主机侧只做类型检查，永远不调用）。
- 可变输出参数用 `DisjointSlice<T>`，它向编译器承诺「每个线程只写自己的元素」，
  从而让 Rust 的别名规则成立，**不需要 `unsafe`**。
- 启动走类型化 API：`kernels::load(&ctx)` 从嵌入的 PTX 加载模块，再调
  `module.<kernel_name>(...)`；`LaunchConfig::for_num_elems` 按元素数自动算网格尺寸。
- 越界处理靠 `get_mut` / `get` 返回的 `Option`，不靠 `if idx < n` 这类裸判断 ——
  网格长度通常是线程块大小的整数倍，末尾块常有空转线程。

## 环境要求

| 项 | 本机实测版本 |
| --- | --- |
| CUDA Toolkit | 13.3（`/usr/local/cuda`，`nvcc` 在 PATH 中） |
| GPU | NVIDIA GeForce RTX 3080 |
| 驱动 | 595.99.02 |
| Rust | `nightly-2026-10-11`（由本目录 `rust-toolchain.toml` 钉定） |
| cargo-oxide | 0.2.1，从本地 checkout `bc46b1ff` 编译（`~/.cargo/bin/cargo-oxide`） |

## 参考

- 仓库：<https://github.com/NVIDIA/cuda-rust>
- 本地 checkout：`/home/jhq/rust_code/cuda-rust`
- 中文文档：`docs/cuda-oxide-book-zh.CN/`
- 最有价值的参考资料是上游自带的 200+ 个例子：
  `/home/jhq/rust_code/cuda-rust/cuda-oxide/crates/rustc-codegen-cuda/examples/`
  （每个都是一个可直接 `cargo oxide run` 的独立 crate）
