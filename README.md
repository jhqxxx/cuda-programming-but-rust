# CUDA Programming But Rust 🦀⚡

> **English** → [README.en.md](./README.en.md)

<p align="center">
  <b>三个语言 × 两个编程模型，对照着学 GPU 编程。</b>
</p>

一个 CUDA 学习仓库：Rust / C++ / Python 三条线并行推进，每条线都覆盖 **tile** 和
**SIMT** 两种编程模型，并整理了四本官方文档的中文翻译。

Rust 是起点，也是这个仓库的特色；C++ 与 Python 同样是 GPU 编程绕不开的语言。

---

## 两个编程模型

GPU 编程真正的分水岭不是语言，而是**你用哪一级抽象描述并行**：

| | tile 模型 | SIMT 模型 |
| --- | --- | --- |
| 编程单位 | 一块数据（tile），≈ 一个 thread block | 一个线程 |
| 你描述的是 | 「我这个块处理哪一块数据」 | 「我这个线程算哪个下标」 |
| 索引从哪来 | 框架给（`program_id`） | 自己算（`threadIdx + blockIdx * blockDim`） |
| 线程数 | 后端决定，你只给「每块多大」 | 你显式指定 `blockDim` |
| 典型入口 | cuTile（Rust / Python / C++）、Triton | CUDA C++ / cuda-oxide / Numba / PyCUDA |

这是**两种不同的抽象，不是同一写法的两种语法** —— 所以示例只和同模型的实现对照。

## 三语言 × 两模型

| | Rust | C++ | Python |
| --- | --- | --- | --- |
| **tile** | ✅ [`cutile-rs-example/`](./cutile-rs-example/) —— cuTile Rust | ✅ [`cpp/examples/tile/`](./cpp/examples/tile/) —— CUDA Tile C++ | ⬜ [`python/examples/tile/`](./python/examples/tile/) —— cuTile Python（环境已验证，示例待写） |
| **SIMT** | ✅ [`cuda-oxide-example/`](./cuda-oxide-example/) —— cuda-oxide | ⬜ [`cpp/examples/simt/`](./cpp/examples/simt/) —— CUDA C++ | ⬜ [`python/examples/simt/`](./python/examples/simt/) —— Numba / PyCUDA / PyTorch |

tile 那一行的三种语言**共享同一个编译器后端**（CUDA Tile IR → `tileiras`），所以是真正的
「同模型跨语言」对照，API 一一对应。C++ 侧从 **CUDA Toolkit 13.3** 起可用：
`nvcc --enable-tile -std=c++20`。

> 这里早前写过「C++ 无用户级入口」，**是错的** —— 漏看了 `cuda_tile.h` 转发到的
> `crt/cuda_tile.h`（4064 行 / 177 KB），也没查 `nvcc --help`。更正记录见
> [`cpp/examples/tile/README.md`](./cpp/examples/tile/README.md)。

## Rust 侧的两条路线

Rust 生态目前有两条可用的 GPU 编程路线，正好对应上表的两种模型：

| | [cuTile Rust](./cutile-rs-example/) | [cuda-oxide](./cuda-oxide-example/) |
| --- | --- | --- |
| 定位 | 面向 tile 的高层编程模型 | 自定义 rustc codegen 后端，直接把 kernel 编成 PTX |
| 工具链 | **stable**（仓库根 `1.99.0`） | **nightly-2026-10-11**（需 `rustc-dev`） |
| 驱动方式 | `cargo run --example <名字>` | `cargo oxide run` |
| 示例位置 | `cutile-rs-example/*.rs`，在根 `Cargo.toml` 注册 `[[example]]` | `cuda-oxide-example/<名字>/`，每个例子是独立项目 |

两条路线对工具链的要求互斥（stable vs nightly），所以在仓库里被**物理隔离**：
`cuda-oxide-example/` 自带 `rust-toolchain.toml`，并在根 `Cargo.toml` 里被 `exclude` 掉。
具体细节和踩过的坑见各自的 README。

## C++ 与 Python 两条线

Rust 之外还有两条并行维护的线，示例同样按模型分目录：

- **CUDA C++** —— `cpp/`，CMake 构建。`examples/simt/` 放 SIMT 示例（待写）；
  `examples/tile/` 放 CUDA Tile C++ 示例（`00-add` 已跑通，CMake 为目标自动带上
  `--enable-tile` 和 `-std=c++20`）；`legacy/` 保留早期 9 个示例备查：
  设备属性、原子操作、常量内存、显式内存、统一内存、共享内存、流与事件、
  矩阵转置（朴素 / 共享内存优化）
- **Python** —— `python/`，uv 管理。`examples/simt/`（Numba / PyCUDA / PyTorch 扩展）与
  `examples/tile/`（cuTile Python）；`legacy/` 保留早期 4 个文件备查

## 目录结构

```
.
├── cutile-rs-example/          # Rust · tile   —— cuTile（stable）
├── cuda-oxide-example/         # Rust · SIMT   —— cuda-oxide（nightly）
│   └── 01-vector-add/          # 向量加法：c[i] = a[i] + b[i]
├── cpp/                        # C++ 线
│   ├── common/                 # cuda_utils.h / vector_ops.cu
│   ├── examples/
│   │   ├── simt/               # 线程级 CUDA C++
│   │   └── tile/               # 块级 CUDA Tile C++（--enable-tile + C++20）
│   └── legacy/                 # 早期 9 个示例，只读备查
├── python/                     # Python 线
│   ├── examples/
│   │   ├── simt/               # Numba / PyCUDA / PyTorch
│   │   └── tile/               # cuTile Python（cuda-tile）
│   └── legacy/                 # 早期 4 个文件，只读备查
├── docs/                       # 中文文档
│   ├── cuda-oxide-book-zh.CN/         # cuda-oxide 手册（42 篇）
│   ├── cutile-rs-book-zh.CN/          # cuTile Rust 手册（29 篇）
│   ├── cutile-python-book-zh.CN/      # cuTile Python 手册（13 篇）
│   ├── CUDA编程指南-V13.3-zh.CN/      # CUDA 编程指南（15 篇）
│   ├── atomic.md                      # 原子操作与内存序详解
│   └── barrier.md                     # mbarrier 详解
└── assets/img/
```

## 快速开始

```bash
# Rust · tile（用仓库根的 stable 工具链）
cargo run --example 00-add

# Rust · SIMT（需要 nightly，首次会自动构建 codegen 后端）
cd cuda-oxide-example/01-vector-add
cargo oxide doctor   # 环境自检：工具链、nvcc、GPU、后端缓存
cargo oxide run      # 编译并在 GPU 上运行

# C++ 线
cd cpp && mkdir -p build && cd build && cmake .. && make
./tile_01-hello-world                # tile 模型（CMake 已带上 --enable-tile + C++20）
./legacy_transpose_with_shared_mem   # SIMT 老示例
```

## 配套文档

`docs/` 下有四本官方文档的中文翻译：

| 文档 | 说明 |
| --- | --- |
| [cuda-oxide 手册](./docs/cuda-oxide-book-zh.CN/README.md) | cuda-oxide 官方 Book |
| [cuTile Rust 手册](./docs/cutile-rs-book-zh.CN/README.md) | cuTile Rust 官方 Book |
| [cuTile Python 手册](./docs/cutile-python-book-zh.CN/README.md) | cuTile Python（`cuda.tile`）官方文档 |
| [CUDA 编程指南 v13.3](./docs/CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA C++ 编程指南 |

另外 `docs/atomic.md`、`docs/barrier.md` 是本仓库自己整理的两篇专题笔记。

## 环境（本机实测）

| 项 | 版本 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080（compute capability 8.6） |
| 驱动 | 615.71.09（R615 = CUDA 13.4） |
| CUDA Toolkit | 13.3（`/usr/local/cuda`） |
| Rust（主工作区） | 1.99.0 stable |
| Rust（cuda-oxide） | nightly-2026-10-11 |
| cargo-oxide | 0.2.1，从 `NVIDIA/cuda-rust` 的 `bc46b1ff` 编译 |

## 进展

🚧 持续补充中。

| 线 | 状态 |
| --- | --- |
| Rust · tile（cuTile） | ✅ `00-add`、`01-hello-world` 可跑通，目录约定与模板已定 |
| Rust · SIMT（cuda-oxide） | ✅ `01-vector-add` 可跑通，codegen 后端缓存已就绪 |
| C++ · SIMT | ⬜ 目录与 CMake 已就绪，示例待写 |
| C++ · tile | ✅ `01-hello-world` 可跑通（CUDA Tile C++，`--enable-tile` + C++20） |
| Python · SIMT | ⬜ uv 环境已就绪；`examples/` 待写，`legacy/` 三份已跑通 |
| Python · tile | ⬜ cuTile Python 已在环境里跑通，`examples/tile/` 待写 |

## 参考

- cuda-rust: <https://github.com/NVIDIA/cuda-rust>
- CUDA 编程指南：<https://docs.nvidia.com/cuda/cuda-programming-guide>

