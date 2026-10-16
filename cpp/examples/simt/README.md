# SIMT 示例（CUDA C++）

**SIMT** = Single Instruction, Multiple Threads。编程单位是**一个线程**：
你自己算下标，自己判边界。

```cpp
__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;  // 我在哪
    if (idx < n) C[idx] = A[idx] + B[idx];                  // 我负责一个元素
}
```

启动时你得亲手给出 `grid` / `block` 的维度，warp 调度完全暴露在你面前。

传统 CUDA C++ 就属于这一类，本仓库 C++ 侧的主体放在这里。

## 这个目录放什么

按主题编号的 CUDA C++ 示例，例如 `00-add.cu`（向量加法 `z = x + y`）。

每个主题在另外两条语言线上的对应实现：

| 模型 | Rust | C++ | Python |
| --- | --- | --- | --- |
| **SIMT** | `cuda-oxide-example/01-vector-add/` | 本目录 | `python/examples/simt/` |
| **tile** | `cutile-rs-example/00-add.rs` | `../tile/` | `python/examples/tile/` |

> ⚠️ **编号现状**：Rust 侧向量加法在 SIMT 线叫 `01-vector-add`、在 tile 线叫 `00-add`，
> 两条线的编号还没对齐。写新示例时按「一个主题一个编号」来，后续再统一。

## 约定

- 文件名 = 两位编号 + slug，如 `00-add.cu`
- **CMake 目标名带模型前缀**：`simt_00-add`（不加前缀会和 `../tile/` 下的同名文件撞车）
- 每个示例自带 `main` 并**自检结果**（`vectorApproximatelyEqual` / `assert`），
  别只打印不验证
- 公共工具用 `#include "cuda_utils.h"`（`../../common` 已在 include 路径上），不要复制粘贴

## 和 tile 模型的差别

不是「同一种写法的两种语法」，而是两种不同的抽象：

| | SIMT（本目录） | tile（`../tile/`） |
| --- | --- | --- |
| 你描述的是 | **一个线程**做什么 | **一个 thread block** 处理哪一块数据 |
| 索引 | 自己算 `threadIdx + blockIdx * blockDim` | 框架给 `program_id` |
| 线程数 | 你显式指定 `blockDim` | 后端决定，你只给「每块多大」 |

所以**只和同模型的实现对比** —— 拿 `00-add.cu` 去对照 `cutile-rs-example/00-add.rs`
会误导，该对照的是 `cuda-oxide-example/01-vector-add/`。
