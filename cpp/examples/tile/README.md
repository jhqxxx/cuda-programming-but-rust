# tile 示例（C++）

> **这个目录是有意空着的。** 它记录一个查证过的结论：**C++ 目前没有可以直接写
> tile kernel 的用户级 API。**

## 为什么空着

查了三处：

| 来源 | 结论 |
| --- | --- |
| `/usr/local/cuda/include/cuda_tile.h`（CUDA 13.3 自带） | 50 行，去掉 license 后**只有一行**：`#include "crt/cuda_tile.h"`。这是**设备端运行时头**，给 Tile IR 的编译产物用，不是给人写 kernel 的 API |
| `/usr/local/cuda/lib64/` | 没有任何 `*tile*` 库 |
| 上游 CUDA Tile IR（`cuda-tile`）项目 | 自称 *"MLIR-based intermediate representation and compiler infrastructure"*，核心组件是 MLIR dialect、bytecode、conformance test suite、Python IR 绑定。这是**编译器基础设施** |

也就是说，C++ 侧的 tile 入口停在 **MLIR IR 层**。要在 C++ 里写 tile 算子，先得把
LLVM/MLIR + CUDA Tile IR 那一整套工具链建起来 —— `cuda-tile-rs` 的 `build.rs` 就是
先 clone 一个 pin 住的 LLVM commit，再编 LLVM + MLIR。那是编译器开发，和「练算子」
不是同一条路。

- CUDA Tile 官方入口：<https://developer.nvidia.com/cuda/tile>
- CUDA Tile IR 规范：<https://docs.nvidia.com/cuda/tile-ir/13.1/index.html>

## 三条语言线的对照

| 语言 | tile 模型入口 | 状态 |
| --- | --- | --- |
| Rust | `cutile-rs-example/` —— cuTile，面向用户的 tile DSL | ✅ 可跑 |
| Python | Triton（面向用户）；cuda-tile 的 Python IR 绑定 | ⬜ 见 `python/examples/tile/` |
| **C++** | 只有 MLIR 层的 CUDA Tile IR | ⚠️ 无实用入口 |

## 什么时候回来补

等 CUDA Toolkit 出现可用的 C++ tile 编程接口（头文件 + 库），或者确实需要深入
Tile IR 本身的时候。在那之前，C++ 侧投入 SIMT（`../simt/`）的收益高得多 ——
那才是 CUDA 的通用基础。
