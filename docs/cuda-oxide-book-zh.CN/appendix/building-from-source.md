# 从源码构建

本附录逐步介绍如何从一份全新的检出开始，为 cuda-oxide 搭建开发环境。如果你
只是想运行一个示例，[编写你的第一个 Kernel](../getting-started/hello-gpu.md)
一章更快。

---

## 环境要求

| Dependency       | Version                       | Purpose                                                     |
|:-----------------|:----------------------------- |:------------------------------------------------------------|
| **Rust nightly** | `nightly-2026-10-11`（固定） | 带 `rustc-dev` 的编译器工具链，用于 codegen backend |
| **CUDA 工具链** | 13.0+（含 cuRAND 头文件）   | Driver API、`nvcc`、PTX 汇编器；bindgen 需要 `curand.h`   |
| **Clang**        | 21+（`clang-21` 包）          | 宿主 `cuda-bindings` 中的 `bindgen` 需要 clang 的头文件     |
| **Linux**        | 在 Ubuntu 24.04 上测试过      | 不支持 Windows 与 macOS                         |
| **GPU**          | sm_80, sm_90, sm_100a         | 硬件目标                                             |

## 克隆仓库

```bash
git clone https://github.com/NVlabs/cuda-oxide.git
cd cuda-oxide
```

## 安装 Rust 工具链

CUDA Oxide 组件自带了 `cuda-oxide/rust-toolchain.toml`，其中固定了确切的
nightly 版本与组件。在 `cuda-oxide/` 目录内运行的命令，rustup 会自动采用它；
仓库根目录与 `cutile-rs/` 仍使用 stable：

```toml
# cuda-oxide/rust-toolchain.toml
[toolchain]
channel = "nightly-2026-10-11"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
```

如果你需要手动安装：

```bash
rustup toolchain install nightly-2026-10-11
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain nightly-2026-10-11
```

`rust-src` 提供用于交叉编译的标准库源码，`rustc-dev` 暴露 codegen backend 所
链接的编译器内部结构，而 `llvm-tools` 安装工具链自带的、用于生成 PTX 的
`llc`（`cargo oxide doctor` 也需要它）。另外三个构建时并不需要：
`rust-analyzer` 支撑 IDE 支持，`clippy` 是 CI 运行的 lint 关卡，`rustfmt`
支撑 `cargo oxide fmt`。

## 安装 CUDA

确保 CUDA 工具链在你的 `PATH` 中：

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version   # 应打印 13.x 或更高版本
```

如果你要在没有 GPU 的系统上构建（例如 CI），仍然需要工具链来提供 `ptxas` 与
头文件，但你无法运行 kernel。

## 安装 LLVM（通常可选）

codegen 管线会生成 LLVM IR 并调用 `llc` 来产出 PTX。固定的 Rust 工具链
（`nightly-2026-10-11`）已经通过 `llvm-tools` 组件自带了启用了 NVPTX backend
的 LLVM 23，因此推荐的做法是：

```bash
rustup component add llvm-tools
```

该组件已经列在 `rust-toolchain.toml` 中，因此在新克隆的仓库上 rustup 会自动
安装它；对较旧的克隆来说，运行上面的命令就是一次性的修复手段。管线会在
`<sysroot>/lib/rustlib/<host>/bin/llc` 处自动检测这个 `llc`。

如果你更想使用系统 LLVM（为了某个特定的补丁级别，或者因为你已经装了一个），
管线会回退到 `PATH` 中的 `llc-23` / `llc-22` / `llc-21`。LLVM 21 是最低要求
—— 更早的版本会拒绝 cuda-oxide 生成的 TMA / tcgen05 / WGMMA intrinsic 签名。

```bash
# Ubuntu / Debian
sudo apt install llvm-21
```

如果你的发行版软件包没有提供 `llvm-21`，请使用 LLVM 的 apt 辅助脚本：

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

```bash
# 验证 NVPTX 支持
llc-21 --version | grep nvptx
```

你应该会在目标列表中看到 `nvptx64 - NVIDIA PTX 64-bit`。

要固定某个特定的二进制（rustup 的、发行版的，或自定义构建的），请设置
`CUDA_OXIDE_LLC=/path/to/llc`。管线的完整解析顺序是：

1. `$CUDA_OXIDE_LLC`（如果已设置）
2. Rust 工具链 `llvm-tools` 中的 llc
3. `PATH` 上的 `llc-23`，然后是 `llc-22`，然后是 `llc-21`，最后是裸的 `llc`

> **注意**
>
> 较旧的 `llc` 二进制（LLVM 20 及更早）在通过
> `CUDA_OXIDE_LLC=/path/to/llc-20` 指向时，仍能编译较简单的 kernel，但任何
> 使用现代 TMA / tcgen05 / WGMMA intrinsics 的示例（`tma_copy`、`gemm_sol`、
> `tcgen05_matmul`、`wgmma`、……）都会失败，报出
> `Intrinsic has incorrect argument type!`，直到你升级到 LLVM 21+。

## 安装 Clang（用于宿主 `cuda-bindings`）

宿主 `cuda-bindings` crate 会运行 `bindgen`，后者加载 libclang，并需要 clang
自己的 resource-dir 中的 `stddef.h` —— 仅有 `libclang1-*` 运行库是不够的。

```bash
sudo apt install clang-21   # 或者 libclang-common-21-dev
```

`cargo oxide doctor` 会事先校验这一点。

## 构建工作区

主工作区包含面向用户的 crate（`cuda-device`、`cuda-host`、`cuda-macros` 等）
以及构建工具（`cargo-oxide`）。宿主运行时（`cuda-bindings`、`cuda-core`、
`cuda-async`）与 cutile-rs 共享，来自 crates.io：

```bash
cargo build
```

> **注意**
>
> codegen backend（`cuda-oxide/crates/rustc-codegen-cuda/`）**有意**不作为
> 工作区成员，因为它需要特殊的 nightly 特性以及不同的构建流程。
> `cargo-oxide` 会透明地处理它的构建。

## 安装 cargo-oxide

`cargo-oxide` 是驱动完整编译管线的 cargo 子命令。在仓库内部，它通过工作区别名
即可使用。若要在仓库之外独立使用，请用固定的 nightly 工具链安装它：

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

首次运行时，`cargo-oxide` 会自动获取并构建 codegen backend dylib。

## 验证安装

```bash
# 检查所有前置条件
cargo oxide doctor

# 编译并运行第一个规范示例
cargo oxide run vecadd
```

`cargo oxide doctor` 会校验你的 Rust 工具链、CUDA 工具链（包括供使用数学
intrinsics 的 kernel 所用的 libNVVM / nvJitLink / libdevice）、LLVM 安装以及
codegen backend。如果一切配置正确，`cargo oxide run vecadd` 会把一个 Rust
kernel 编译为 PTX，在 GPU 上启动它，并打印成功消息。

## 常用命令

```bash
# 构建并运行一个示例
cargo oxide run <example>

# 仅打印生成的 PTX
cargo oxide inspect <example>

# 显示完整的编译管线（MIR → LLVM IR → PTX）
cargo oxide pipeline <example>

# 删除本地构建输出与生成的文件
cargo oxide clean

# 在 NVIDIA Compute Sanitizer 下运行
cargo oxide sanitize <example> --tool memcheck

# 用 cuda-gdb 调试
cargo oxide debug <example> --tui

# 为 libNVVM/nvJitLink 互操作性构建 NVVM IR
cargo oxide build <example> --emit-nvvm-ir --arch sm_120
```

## 构建本书

文档位于 `cuda-oxide-book/`，使用 Sphinx 与 MyST Markdown。要在本地构建并
提供服务：

```bash
cd cuda-oxide/cuda-oxide-book
make setup      # 创建 venv、安装依赖
source .venv/bin/activate
make livehtml   # 在 http://localhost:8000 上启动开发服务器
```

## 生成 API 文档

标准 `cargo doc` 可以对工作区 crate 使用：

```bash
cd cuda-oxide
cargo doc --no-deps --open
```

这会为 `cuda-device`、`cuda-host`、`cuda-macros` 以及所有其他工作区成员生成
rustdoc。codegen backend 被排除在外，因为它不是工作区成员，而共享的宿主运行时
（`cuda-core`、`cuda-async`）的文档来自 cutile-rs。

## 工作区结构

```text
repository/
├── Cargo.toml              # 稳定版共享宿主工作区
├── cuda-bindings/          # 共享的 CUDA 绑定
├── cuda-core/              # 共享宿主运行时
├── cuda-core-derive/       # cuda-core 派生宏
├── cuda-async/             # 共享异步运行时
├── cutile-rs/              # Tile DSL（嵌套导入）
└── cuda-oxide/
    ├── Cargo.toml          # nightly SIMT 工作区
    ├── rust-toolchain.toml # 固定的 nightly + 组件
    ├── crates/
    │   ├── cuda-device/          # 设备 intrinsics（#![no_std]）
    │   ├── cuda-host/            # 宿主启动 API
    │   ├── cuda-macros/          # proc macro（#[kernel]、#[device]、gpu_printf!）
    │   │                         # （共享宿主 crate 位于 git 根目录；
    │   │                         #  SIMT API 位于它们的 simt 模块下）
    │   ├── cargo-oxide/          # cargo 子命令
    │   ├── rustc-codegen-cuda/   # Codegen backend（非工作区成员）
    │   ├── mir-importer/         # MIR → Pliron IR 转换
    │   ├── mir-lower/            # `dialect-mir` → LLVM dialect 下沉
    │   ├── dialect-mir/          # 为 Rust MIR 建模的 pliron dialect
    │   ├── llvm-export/          # 重新导出 pliron-llvm 的 shim + 文本 .ll 导出
    │   ├── dialect-nvvm/         # NVVM intrinsics dialect
    │   ├── libnvvm-sys/          # 对 libNVVM 的 dlopen 绑定
    │   ├── nvjitlink-sys/        # 对 nvJitLink 的 dlopen 绑定
    │   ├── reserved-oxide-symbols/ # 共享命名约定
    │   └── fuzzer/               # 差分测试支持
    └── cuda-oxide-book/        # 本书（Sphinx + MyST）
```

## 故障排查

- **`llc` not found or missing NVPTX** —— 最快的修复是
  `rustup component add llvm-tools` —— 固定工具链的 `llc` 就是启用了 NVPTX
  的 LLVM 23，会被自动选用。
  否则请安装系统 LLVM 21+（`sudo apt install llvm-21`）；管线会先探测 rustup
  的 `llc`，然后是 `PATH` 上的 `llc-23` → `llc-22` → `llc-21`。要固定某个
  特定的二进制，请设置 `CUDA_OXIDE_LLC=/path/to/llc`。

- **`Intrinsic has incorrect argument type!`（来自 `llc`）** —— 你的 `llc`
  早于 LLVM 21，无法下沉现代的 TMA / tcgen05 / WGMMA intrinsic 签名。请安装
  `llvm-21` 并重新运行。

- **`error[E0463]: can't find crate for rustc_middle`** —— 你缺少 `rustc-dev`
  组件。请运行：
  `rustup component add rustc-dev --toolchain nightly-2026-10-11`。

- **CUDA driver version mismatch** —— 工具链版本（编译期）与驱动版本（运行时）
  必须兼容。运行 `nvidia-smi` 检查驱动版本，运行 `nvcc --version` 检查工具链。

- **`cargo oxide doctor` fails on codegen backend** —— 该 backend 在首次使用时
  构建。如果构建失败，请检查 `rust-src` 是否已安装，以及 nightly 版本是否与
  `rust-toolchain.toml` 一致。
