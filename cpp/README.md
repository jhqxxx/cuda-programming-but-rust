# CUDA C++ 示例

CUDA C++ 是 GPU 编程的基础功，所以这里是练习的重点。

`examples/` 下按**编程模型**分两个子目录 —— 传统 CUDA C++ 和 tile 模型是两种不同的
抽象，混在同一列里对照会误导：

```
cpp/
├── CMakeLists.txt
├── common/        # 共享：cuda_utils.h、vector_ops.cu
├── examples/
│   ├── simt/      # 线程级 CUDA C++  ← SIMT 新代码写这里
│   └── tile/      # 块级 CUDA Tile C++（需 --enable-tile + C++20）
└── legacy/        # 早期随手写的 9 个示例，保留原样备查
```

| 目录 | 模型 | 编程单位 | 状态 |
| --- | --- | --- | --- |
| `examples/simt/` | SIMT | 一个线程 | ⬜ 待写 |
| `examples/tile/` | tile | 一个块（tile） | ✅ 可跑（CUDA Tile C++，工具包 13.3 起提供） |
| `legacy/` | SIMT | 一个线程 | ✅ 可跑，只读备查 |

## 构建与运行

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make -j

./simt_00-add            # examples/simt/00-add.cu → 目标名 simt_00-add
./tile_01-hello-world    # examples/tile/01-hello-world.cu → 目标名 tile_01-hello-world
                         #   tile 目标已自动带上 --enable-tile 和 -std=c++20
./legacy_transpose_v1    # legacy/ 的目标统一带 legacy_ 前缀
```

不需要 legacy 时：`cmake -DBUILD_LEGACY=OFF ..`

**目标名规则**：`<模型>_<文件名>`。加模型前缀是因为两个子目录里可能出现同名文件，
而且编号在各自模型内独立计数。

## 主题对照表

同一主题在三语言、两模型下各有实现。**只和同模型的实现对照**：

| 主题 | SIMT（Rust） | SIMT（C++） | SIMT（Python） | tile（Rust） | tile（C++） | tile（Python） |
| --- | --- | --- | --- | --- | --- | --- |
| 向量加法 `z = x + y` | `cuda-oxide-example/01-vector-add/` | `examples/simt/` ⬜ | `python/examples/simt/` ⬜ | `cutile-rs-example/00-add.rs` ✅ | ⬜ | `python/examples/tile/` ⬜ |
| 网格与线程索引 | — | ⬜ | ⬜ | `cutile-rs-example/01-hello-world.rs` ✅ | `examples/tile/01-hello-world.cu` ✅ | `python/examples/tile/01-hello-world.py` ✅ |

> ⚠️ **编号还没统一**：Rust 的 SIMT 线把向量加法编成 `01`，tile 线编成 `00`。
> 新示例按「一个主题一个编号」写，之后再把 Rust 侧对齐。

`legacy/` 里的 9 个旧示例（`shared_mem`、`transpose`、`atomic` 等）都属 SIMT，
还没进编号体系 —— 需要哪个就照着在 `examples/simt/` 里写一份编号版。

## 约定

- **文件名**：两位编号 + slug，如 `01-hello-world.cu`；CMake 目标名自动加模型前缀
- 每个示例自带 `main`，并且**自检结果**（`vectorApproximatelyEqual` / `assert`），
  别只打印不验证
- 公共工具放 `common/`，用 `#include "cuda_utils.h"`，不要复制粘贴
- `legacy/` 视为只读；要升级某个旧示例，就在 `examples/simt/` 里新建编号版，不要原地改
- **跨模型不要对照**：`examples/simt/00-add.cu` 该对照的是
  `cuda-oxide-example/01-vector-add/`，不是 `cutile-rs-example/00-add.rs`

## 环境

| 项 | 本机实测 |
| --- | --- |
| CUDA Toolkit | 13.3（`/usr/local/cuda`） |
| CUDA Tile C++ | ✅ 可用（`nvcc --enable-tile`，需 `-std=c++20`；工具包 13.3 起提供） |
| GPU | NVIDIA GeForce RTX 3080（sm_86） |
| CMake | ≥ 3.18 |
| `CMAKE_CUDA_ARCHITECTURES` | 86 |

## `common/cuda_utils.h` 里有什么

| 名字 | 作用 |
| --- | --- |
| `CUDA_CHECK(expr)` | 检查 `cudaError_t`，失败时打印文件:行号 + 错误串并退出 |
| `initArray<T>(arr, len)` | 随机初始化（浮点 / 整数自动分派） |
| `vectorApproximatelyEqual<T>(A, B, len)` | 逐元素比较，浮点容差 1e-5 |
| `INDX(row, col, ld)` | 行主序二维索引 |

`common/vector_ops.cu` 提供 `vecAdd` / `serialVecAdd`，目前只有 `legacy/` 里的
三个示例在 include 它；`examples/` 里的编号示例刻意自包含，便于单独阅读。
