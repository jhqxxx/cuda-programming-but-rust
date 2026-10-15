# CUDA Programming But Rust 🦀⚡

> **English** → [README.en.md](./README.en.md)

<p align="center">
  <b>用 Rust 写 GPU kernel，不再写 .cu 文件。</b>
</p>

一个 CUDA 学习仓库：重点是**用 Rust 写 GPU kernel**，并整理了三本官方文档的中文翻译。
另外还放着一批早先写的 CUDA C++ 与 Python 示例 —— 它们已基本定型，后续少有改动，
仅作参考。

---

## 两条 Rust 路线

Rust 生态目前有两条可用的 GPU 编程路线，本仓库各有一块练习区：

| | [cuTile Rust](./cutile-example/) | [cuda-oxide](./cuda-oxide-example/) |
| --- | --- | --- |
| 定位 | 面向 tile 的高层编程模型 | 自定义 rustc codegen 后端，直接把 kernel 编成 PTX |
| 工具链 | **stable**（仓库根 `1.99.0`） | **nightly-2026-10-11**（需 `rustc-dev`） |
| 驱动方式 | `cargo run --example <名字>` | `cargo oxide run` |
| 示例位置 | `cutile-example/*.rs`，在根 `Cargo.toml` 注册 `[[example]]` | `cuda-oxide-example/<名字>/`，每个例子是独立项目 |

两条路线对工具链的要求互斥（stable vs nightly），所以在仓库里被**物理隔离**：
`cuda-oxide-example/` 自带 `rust-toolchain.toml`，并在根 `Cargo.toml` 里被 `exclude` 掉。
具体细节和踩过的坑见各自的 README。

## 参考示例（C++ / Python）

这两部分是早先的示例，已基本定型，后续不再随 Rust 侧演进，留作查阅与参考：

- **CUDA C++** —— `use_cpp/examples/`，9 个示例，CMake 构建：设备属性、原子操作、
  常量内存、显式内存、统一内存、共享内存、流与事件、矩阵转置（朴素 / 共享内存优化）
- **Python** —— `use_python/about_package/`，Numba、PyCUDA、PyTorch 三种绑定

## 目录结构

```
.
├── cutile-example/             # cuTile Rust 学习示例（stable）
├── cuda-oxide-example/         # cuda-oxide 学习示例（nightly）
│   └── 01-vector-add/          # 向量加法：c[i] = a[i] + b[i]
├── use_cpp/                    # CUDA C++ 参考示例（已定型）
├── use_python/                 # Python 参考示例（已定型）
├── docs/                       # 中文文档
│   ├── cuda-oxide-book-zh.CN/        # cuda-oxide 手册（42 篇）
│   ├── cutile-book-zh.CN/            # cuTile 手册（29 篇）
│   ├── CUDA编程指南-V13.3-zh.CN/      # CUDA 编程指南（15 篇）
│   ├── atomic.md                     # 原子操作与内存序详解
│   └── barrier.md                    # mbarrier 详解
└── assets/img/
```

## 快速开始

```bash
# cuda-oxide 示例（需要 nightly，首次会自动构建 codegen 后端）
cd cuda-oxide-example/01-vector-add
cargo oxide doctor   # 环境自检：工具链、nvcc、GPU、后端缓存
cargo oxide run      # 编译并在 GPU 上运行

# CUDA C++ 示例
cd use_cpp && mkdir -p build && cd build && cmake .. && make
./transpose_with_shared_mem
```

## 配套文档

`docs/` 下有三本官方文档的中文翻译：

| 文档 | 说明 |
| --- | --- |
| [cuda-oxide 手册](./docs/cuda-oxide-book-zh.CN/README.md) | cuda-oxide 官方 Book |
| [cuTile 手册](./docs/cutile-book-zh.CN/README.md) | cuTile Rust 官方 Book |
| [CUDA 编程指南 v13.3](./docs/CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA C++ 编程指南 |

## 环境（本机实测）

| 项 | 版本 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080（compute capability 8.6） |
| 驱动 | 595.99.02 |
| CUDA Toolkit | 13.3（`/usr/local/cuda`） |
| Rust（主工作区） | 1.99.0 stable |
| Rust（cuda-oxide） | nightly-2026-10-11 |
| cargo-oxide | 0.2.1，从 `NVIDIA/cuda-rust` 的 `bc46b1ff` 编译 |

## 进展

🚧 持续补充中。目前 cuda-oxide 侧有 `01-vector-add` 可跑通，cuTile 侧已定好目录约定与模板。

## 参考

- cuda-rust: <https://github.com/NVIDIA/cuda-rust>
- CUDA 编程指南：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

