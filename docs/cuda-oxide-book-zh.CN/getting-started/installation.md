# 安装

本节会带你走完在你自己机器上跑通 `cargo oxide run vecadd` 所需的全部步骤。

---

## 前置依赖

| 要求 | 版本 | 说明 |
|------------------|---------------------|---------------------------------------------------------------|
| **Linux** | 在 Ubuntu 24.04 上验证过 | 其它发行版可能可用，但未测试 |
| **NVIDIA GPU** | Ampere+（sm_80+） | 驱动 580+（CUDA 13.x 驱动；在运行时加载） |
| **CUDA Toolkit** | 13.0+ | `nvcc`、`cuda.h` 与 `curand.h` 必须可用 |
| **LLVM** | 21+ | 必须包含 NVPTX 后端 |
| **Clang** | 21+ | `clang-21` —— 宿主侧 `cuda-bindings` 的 `bindgen` 需要它 |
| **Rust** | Nightly（已固定版本） | 固定在 `cuda-oxide/rust-toolchain.toml` |

宿主运行时在构建期不链接 `libcuda`。共享 crate `cuda-bindings` 会在首次调用驱动时加载它，因此二进制可以在没有驱动的情况下启动，并在那次调用上以 `CUDA_ERROR_NOT_INITIALIZED` 失败；错误信息中会列出加载器尝试过的库文件。如果驱动的 CUDA 主版本比工具包更旧（例如 12.x 驱动配 13.x 构建），会以同样的方式失败，并给出"驱动过旧"的信息。

> **注意**：cuda-oxide 目前**只支持 Linux**。不支持 Windows。

---

## Dev Container

仓库在 `cuda-oxide/.devcontainer/` 下包含一个 cuda-oxide 的 devcontainer。用它是最快获得可复现开发环境的途径 —— CUDA Toolkit 13.0、LLVM 21、Clang 21 以及固定版本的 Rust nightly 都已装好。

宿主机不需要安装 CUDA Toolkit，但需要：

- 一块 NVIDIA GPU
- 与 CUDA 13.0 兼容的 NVIDIA 驱动（R580 或更新）
- 装有 NVIDIA Container Toolkit 的 Docker

如果编辑器支持 devcontainer，打开检出的 `cuda-oxide/` 目录并选择 "Reopen in Container"。编辑器会读取它的 `.devcontainer/devcontainer.json`、构建镜像，并用 `--gpus=all` 请求 GPU 访问权限。容器挂载整个仓库，因此共享的宿主 crate 依然可用，并把工作目录打开在 `/workspaces/cuda-rust/cuda-oxide`。

只用命令行的话，在仓库根目录执行：

```bash
npx -y @devcontainers/cli up --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json
```

然后在其中运行命令：

```bash
npx -y @devcontainers/cli exec --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide doctor
npx -y @devcontainers/cli exec --workspace-folder ./cuda-oxide \
  --config ./cuda-oxide/.devcontainer/devcontainer.json cargo oxide run vecadd
```

如果宿主驱动太旧，容器内诸如 `nvidia-smi`、`cargo oxide doctor` 或 `cargo oxide run vecadd` 这类 GPU 命令会失败。请升级宿主机的 NVIDIA 驱动，而不是在容器里装另一套 CUDA Toolkit。

如果你使用 devcontainer，可以跳过下面手动配置 CUDA、LLVM、Clang 和 Rust 的各个小节。

---

## Nix / flake.nix

仓库还提供了 `cuda-oxide/flake.nix`，它给出一个可复现的开发 shell（CUDA 13、LLVM 22、Clang、固定版本的 Rust nightly）。需要启用 flakes 的 [Nix](https://nixos.org/download/)、宿主机的 NVIDIA 驱动，以及 Linux（x86\_64 或 aarch64）。

在仓库根目录进入 cuda-oxide 工作区；shell 中已包含 `cargo-oxide`：

```bash
cd cuda-oxide
nix develop
cargo oxide run vecadd
```

要在不克隆仓库的前提下引导一个新项目：

```bash
nix run 'github:NVIDIA/cuda-rust?dir=cuda-oxide#new' -- my-project
cd my-project && nix develop
```

这会通过 `cargo oxide new` 生成脚手架，并放入一个 `flake.nix` 来继承本仓库的开发 shell。shellHook 会在 NixOS 与非 NixOS 系统上自动发现宿主机的 NVIDIA 驱动库；如果宿主驱动太旧，请升级它，而不是改动 Nix shell 内部的东西。

要把同一套开发环境加到已有项目里，请在该项目中运行：

```bash
nix flake init -t 'github:NVIDIA/cuda-rust?dir=cuda-oxide'
nix develop
```

如果你使用 Nix flake，可以跳过下面手动配置 CUDA、LLVM、Clang 和 Rust 的各个小节。

---

## CUDA Toolkit

从 [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) 页面安装 CUDA Toolkit，然后确保它在你的 `PATH` 上：

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

验证安装：

```bash
nvcc --version
```

> **提示**：如果你把 CUDA 装到了非默认位置，请把 `CUDA_TOOLKIT_PATH` 设成它的根目录（即包含 `include/cuda.h` 的那个目录）。如果 `CUDA_TOOLKIT_PATH` 和 `CUDA_HOME` 都没有设置，则会按顺序尝试这些根目录，第一个含有 CUDA 13.0 或更新版本 `cuda.h` 的胜出：`/usr/local/cuda-13.3`、`/usr/local/cuda-13.2`、`/usr/local/cuda-13`、`/usr/local/cuda`。装了多套工具包时，起决定作用的是这个顺序 —— 不是最高的版本，也不是 `/usr/local/cuda` 这个符号链接（它排在最后）。`cargo oxide doctor` 会报告它解析到了哪个根目录，以及更早的那些为什么被否掉。

<a id="installation-toolkit-driver-compatibility"></a>
### 工具包与驱动的兼容性

CUDA 工具包与 NVIDIA 驱动是两套独立的安装。较新的工具包往往可以借助 CUDA 次版本兼容性在较旧的驱动上运行，但那种兼容性并不覆盖更新的 PTX：

```text
cubin ───────────────────────────► driver loads finished GPU code
PTX   ──► driver compiles it ─────► finished GPU code
```

cuda-oxide 的常规路径是从 LLVM 的 `llc` 得到 PTX，所以单是安装更新的 CUDA 工具包并不会改变那个 PTX 版本。NVVM 前向执行路径则从所选工具包的 nvJitLink 得到 PTX。如果那个 PTX 比驱动能理解的新，模块加载会返回 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`（错误 222）。

例如，当前验证期间所用的工具链产生了：

```text
LLVM llc ───────────────► PTX 8.7 ──► driver 580 ✓
CUDA 13.3 nvJitLink ───► PTX 9.3 ──► driver 580 ✗ error 222
```

这些是观测到的版本，而不是这两条路径的永久属性。诊断另一套工具链时，请检查生成 PTX 的 `.version` 指令。

CUDA 13.x 的次版本兼容从驱动 580 开始，而 CUDA 13.3 对应的完全支持驱动是 610.43.02。因此 580 驱动可以加载兼容的成品 cubin，但无法 JIT CUDA 13.3 产出的 PTX 9.3。当前驱动对照表见 [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。

升级驱动，或者为这条命令选择兼容的工具包：

```bash
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 cargo oxide run <example> --arch sm_86
```

受支持的驱动范围与 PTX 限制见 NVIDIA 的[次版本兼容性指南](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

---

## LLVM 21+（可选）

cuda-oxide 使用 LLVM 的 NVPTX 后端把 LLVM IR 下降为 PTX。

通常 Rust 工具链自带的 `llc` 就够了。

安装 LLVM 21 或更新版本，并确保 `llc-21`（或 `llc-22`）在你的 `PATH` 上：

```bash
# Ubuntu / Debian
sudo apt install llvm-21
```

如果你的发行版没有提供 `llvm-21` 包，请用 LLVM 的 apt 辅助脚本：

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

验证 NVPTX 目标存在：

```bash
llc-21 --version | grep nvptx
```

你应当看到已注册目标中含 `nvptx64` 的一行。流水线会按 `llc-23`、`llc-22`、`llc-21` 的顺序自动发现；需要时可以用 `CUDA_OXIDE_LLC=/usr/bin/llc-21` 固定某个具体的二进制。

> **警告**：没有 NVPTX 后端的原版 LLVM 构建不可用。Ubuntu 的 `llvm-21` 包默认包含它，但如果你从源码构建 LLVM，必须给 CMake 传 `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"`。

> **重要**
>
> **为什么需要 LLVM 21？** 我们生成的 TMA / tcgen05 / WGMMA 内联函数是 LLVM 20 及更早版本的 `llc` 无法处理的。简单的 kernel 可能仍能与较旧的 `llc` 一起工作（设置 `CUDA_OXIDE_LLC=/path/to/llc-20`），但任何 Hopper / Blackwell 相关的东西都需要 21+。

---

## Clang（宿主侧 `cuda-bindings`）

宿主侧的 `cuda-bindings` crate 会运行 [`bindgen`](https://github.com/rust-lang/rust-bindgen)，后者在运行时加载 `libclang.so`，并且需要 clang 自己的资源目录下的 `stddef.h` —— 光有一个 `libclang1-*` 运行时包是不够的。三个包合起来才覆盖这两半：

```bash
sudo apt install libclang-21-dev libclang-cpp21-dev libclang-common-21-dev
```

`cargo oxide doctor` 会提前抓到这个问题；否则症状是宿主构建期间出现 `'stddef.h' file not found`。

> **提示**
>
> **如果 doctor 仍然报告找不到 clang：**
> 问题可能是带版本号的二进制（例如 `clang-21`）已经装了，但没有映射到 bindgen 所需的那个不带版本号的 `clang` 命令。装一下 clang 的元包即可修复，它会自动管理这些依赖：
>
> ```bash
> sudo apt install clang libclang-dev
> ```

> **注意**
>
> **全新的 Ubuntu 24.04 / DGX-OS：** 按上面的方式通过 `apt.llvm.org/llvm.sh` 安装 LLVM 21 之后，带版本号的 `clang-21` / `clang++-21` 二进制是存在的，但 `cargo oxide doctor` 所查找的那些不带版本号的别名还没有。用 `update-alternatives` 加上它们：
>
> ```bash
> sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-21   100
> sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100
> ```
>
> 已在 Asus GX10 / NVIDIA DGX Spark 上验证。

---

## Rust 工具链

CUDA Oxide 工作区自带 `cuda-oxide/rust-toolchain.toml`，它固定了确切的 nightly 版本和所需组件。当你第一次在 `cuda-oxide/` 里运行 `cargo` 命令时，rustup 会自动装好正确的工具链。仓库根目录与 `cutile-rs/` 下的命令则使用 stable。

如果需要手动安装：

```bash
rustup toolchain install nightly-2026-10-11
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain nightly-2026-10-11
```

其中三个组件是 codegen 后端和 doctor 所需要的：

- `rust-src` —— Rust 标准库源码，交叉编译到 NVPTX 目标时需要。
- `rustc-dev` —— 后端链接所依赖的编译器内部实现。
- `llvm-tools` —— 工具链自带的 `llc`，用于把 LLVM IR 下降为 PTX（doctor 的下限检查）。

另外三个构建时并不需要：`rust-analyzer` 提供 IDE 支持，`clippy` 是 CI 跑的 lint 闸门，`rustfmt` 支撑 `cargo oxide fmt`。

---

## cargo-oxide

`cargo-oxide` 是驱动整条构建流水线的 cargo 子命令（`cargo oxide run`、`build`、`debug`、`pipeline` 等等 —— 本节末尾的命令参考把它们全列了出来）。

**在 cuda-oxide 仓库内部**，它通过工作区别名开箱即用 —— 不需要额外的安装步骤。

**要在仓库之外使用**（你自己的项目），请用固定版本的 nightly 工具链安装它：

```bash
cargo +nightly-2026-10-11 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

首次运行时，`cargo-oxide` 会自动获取并构建 codegen 后端。之后的运行会复用构建缓存。

要发现某个 cuda-oxide 检出中可用的示例：

```bash
cargo oxide list
cargo oxide list --json
```

该命令会报告每个示例的用途，以及任何有记录的 GPU、架构、CUDA Toolkit 或外部 SDK 要求。

等你开始构建 kernel 之后，这些是日常好用的辅助命令：

```bash
# Print generated PTX without the full MIR/LLVM pipeline dump
cargo oxide inspect vecadd

# Remove local target/ dirs and generated device artifacts (PTX, LLVM IR,
# LTOIR, cubin, and their sidecar metadata)
cargo oxide clean
```

`inspect` 是 `cargo oxide pipeline` 的轻量对应物。`clean` 只碰项目本地的产物；它不会动 `~/.cargo/cuda-oxide/` 下的共享后端缓存。

### 命令参考

`cargo oxide --help` 报告的完整集合：

| 命令 | 说明 |
|---------|-------------|
| `run` | 构建并运行一个示例或项目 |
| `sanitize` | 在 NVIDIA Compute Sanitizer 下构建并运行一个示例或项目 |
| `build` | 构建一个示例或项目（只编译，不运行） |
| `fuzz-schedule` | 通过扰动某个示例生成的 PTX，找出对调度敏感的失败 |
| `test` | 通过 cuda-oxide 后端运行 Cargo 测试 |
| `emit-ltoir` | 一步把某个 crate 的设备代码编译成二进制 LTOIR 产物 |
| `pipeline` | 以详细输出展示完整的编译流水线（MIR -> PTX/NVVM IR） |
| `debug` | 带调试信息构建并启动 `cuda-gdb` |
| `list` | 列出 cuda-oxide 工作区自带的示例 |
| `inspect` | 构建一个示例或项目并打印生成的 PTX |
| `fmt` | 格式化所有 crate（SIMT 工作区、codegen 后端、示例） |
| `new` | 生成一个新的独立 cuda-oxide 项目脚手架 |
| `clean` | 删除项目本地的构建产物与生成的 cuda-oxide 产物 |
| `doctor` | 检查你的环境是否配置正确 |
| `setup` | 构建并缓存 codegen 后端 |
| `update` | 刷新缓存的 codegen 后端（或在工作区内运行 `setup`） |

其中四个值得单独说明，因为它们做了 `cargo` 自身做不到的事：

```bash
# Run the test suite with device code compiled by the cuda-oxide backend.
# Arguments after `--` go to cargo; with none it is a plain `cargo test`.
cargo oxide test -- --lib

# Format every scope the fmt CI gate checks: the SIMT workspace, the codegen
# backend, the cuda-macros device-only fixture, and every manifest under
# examples/ including nested ones. Each is its own [workspace], so a single
# `cargo fmt` at the root misses most of them.
cargo oxide fmt
cargo oxide fmt --check

# Rebuild the cached backend after changing compiler crates or the toolchain
# pin. Inside the workspace this advises `setup`; --force runs it.
cargo oxide update

# Produce the LTOIR a tile or C++ kernel links against. LTOIR is
# architecture-specific, so --arch is required rather than inferred.
cargo oxide emit-ltoir my_kernels --arch sm_90
```

每个命令都接受 `--help`，它会列出该命令接受的各个选项。

---

## 验证你的安装

运行内置的诊断检查：

```bash
cargo oxide doctor
```

`cargo oxide doctor` 会一次性校验你的 Rust 工具链、CUDA 头文件（`cuda.h`）、CUDA 工具包（包括使用数学内联函数的 kernel 所需的 libNVVM / nvJitLink / libdevice）、LLVM 安装、NVIDIA 驱动 / GPU，以及 codegen 后端。

`cargo-oxide` 本身构建和运行都不需要 CUDA 工具包或 NVIDIA 驱动，所以即使在一台什么都没装的机器上 `doctor` 也能运行，并准确诊断出缺了什么（例如缺少 `cuda.h`，或者没有驱动）。驱动 / GPU 那项检查仅供参考：只有 `cargo oxide run` 需要 GPU，`build` 和 `pipeline` 没有 GPU 也能工作。

然后端到端地构建并运行一个示例：

```bash
cargo oxide run vecadd
```

如果一切配置正确，这会把一个 Rust kernel 编译成 PTX、在 GPU 上启动它，并打印一条成功信息。

> **提示**
>
> **常见问题：**
>
> - `No working llc found` —— 优先用 `rustup component add llvm-tools --toolchain nightly-2026-10-11`，或者安装 LLVM 21+（`sudo apt install llvm-21`）、把 `/usr/lib/llvm-21/bin` 加到你的 `PATH`，或者设置 `CUDA_OXIDE_LLC=/usr/bin/llc-21`。
> - 构建宿主侧 `cuda-bindings` 时 `'stddef.h' file not found` —— 安装 clang 的开发头文件：`sudo apt install clang-21`（或 `libclang-common-21-dev`）。
> - `cuda.h not found` —— 把 `CUDA_TOOLKIT_PATH` 设成你的 CUDA 安装根目录，或者确保 `/usr/local/cuda/include/cuda.h` 存在。
> - 缺少 `rust-src` / `llvm-tools` 组件 —— 运行 `rustup component add rust-src llvm-tools --toolchain nightly-2026-10-11`。
