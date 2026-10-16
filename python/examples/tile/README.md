# tile 示例（Python）

## 用哪个：cuTile Python（官方）

**本目录用 [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/)**，
PyPI 包名 `cuda-tile`，模块 `cuda.tile`。Triton 作为备选路线另行安排（见下）。

理由按重要性排序：

### 1. 和 Rust 侧是同一个模型、同一套编译器 —— 本仓库唯一真正的「同模型跨语言」

`cutile-rs-example/` 用的是 **cuTile Rust**，这里用 **cuTile Python**：两者是同一个编程模型的
两个语言绑定，都降到 **TileIR**，再由同一个 `tileiras` 编成 cubin。所以它们是 1:1 可对照的
—— 但**命名各叫各的**，不要跨语言照抄（比如 `program_id(0)` 在 Python 里叫 `ct.bid(0)`）。

完整的对照表和实测踩过的坑统一放在 [`docs/cross-language/tile/`](../../../docs/cross-language/tile/)：

| 主题 | 文件 |
| --- | --- |
| 块索引、grid 尺寸、设备端打印、启动 | [`01-block-index.md`](../../../docs/cross-language/tile/01-block-index.md) |


### 2. 文档就在本仓库里

`docs/cutile-python-book-zh.CN/` 是这套官方文档的中文译本（12 个导航页全部完成）。

### 3. 本机实测可用（RTX 3080 / sm_86）

环境门槛逐项核对：

| 要求 | 本机 |
| --- | --- |
| Python 3.10 ~ 3.14 | 3.12.9 ✓ |
| NVIDIA 驱动 ≥ r580 | 615.71.09（R615 = CUDA 13.4）✓ |
| GPU 计算能力 8.x ~ 12.x | RTX 3080 = **8.6** ✓ |
| Linux x86_64 | ✓ |
| `tileiras` + `ptxas` + `libnvvm` | 系统 CUDA 13.3 自带 **`/usr/local/cuda/bin/tileiras`** ✓ |

实测（**不需要 `[tileiras]` extra** —— 因为系统 CTK 13.3 里已经有 `tileiras`）：

```bash
uv pip install cuda-tile cupy-cuda13x
python vecadd.py
# ✓ vector_add 在 RTX 3080 (sm_86) 上通过
```

`cuda-tile` 会自动去 CUDA Toolkit 的位置找 `tileiras`。
只有当系统没装 CTK 13.1+ 时，才需要 `pip install cuda-tile[tileiras]`
（它会把 `nvidia-cuda-tileiras` / `nvidia-cuda-nvcc` / `nvidia-nvvm` 装进 venv）。

### 4. 和 SIMT 线共用同一种输入

cuTile Python 直接吃实现了 CUDA Array Interface / DLPack 的对象 ——
**cupy ndarray 和 torch tensor 都可以**，和 `../simt/`（PyTorch 扩展）共用数据准备代码。

## 已知限制（Ampere / sm_86）

部分张量核专属特性在 Ampere 上不可用，会抛 `TileUnsupportedFeatureError`：

- FP8 相关操作（编译到 sm80 系列会报错，见 release notes 1.2.0）
- Hopper 才有的能力（块缩放 MMA `ct.mma_scaled`、`use_fast_acc` 等）

普通逐元素、归约、`ct.mma` 的常规路径不受影响。这也是选它做**学习示例**没问题、
但不要指望在 3080 上跑 FP8 算子的原因。

## 备选：Triton

[triton](https://github.com/triton-lang/triton) 也是 tile 模型，资料和生态比 cuTile 广得多，
值得单独学。但它是**另一套模型 + 另一套编译器**：

| | cuTile Python | Triton |
| --- | --- | --- |
| 编译器 | 降到 **TileIR** → `tileiras`（与 Rust 侧同一个） | 自带（Triton IR → LLVM/PTX） |
| 内核入口 | `@ct.kernel` | `@triton.jit` |
| 块索引 | `ct.bid(0)` | `tl.program_id(0)` |
| 编译期常量 | `ct.Constant[int]` | `X: tl.constexpr` |
| 边界处理 | `padding_mode=ct.PaddingMode.ZERO` 等 | 手工 `mask` |
| 与 `cutile-rs-example/` 对照 | ✅ 同一模型 | ✗ 不同模型 |
| 生态/资料量 | 新，NVIDIA 官方 | 大，事实标准 |

**所以不要把它放进本目录** —— 那又变成跨模型对照了。要加就另开一个目录
（例如 `python/examples/triton/`），或者放到 `python/legacy/` 的思路里当作「另一条路线」。

## 计划写的内容

- `00-add/` —— 向量加法，和 `cutile-rs-example/00-add.rs`、`cpp/examples/simt/00-add.cu` 同主题
- 之后往矩阵乘、softmax、LayerNorm 走（tile 模型的主场）

## 更正

> 本文件早先有一节把 `cuda-tile` 写成「IR 层的 Python 绑定，编译器视角」，**那是错的**。
>
> PyPI 上的 `cuda-tile` 就是**完整的用户级 kernel DSL**（本目录要用的就是它）。
> 写错的原因是我把它和上游 Rust monorepo 里那个同名的 MLIR 项目
> （`cutile-rs/cuda-tile-rs/cuda-tile`，提供 Tile IR dialect 的 C++/Python 绑定）
> 混为一谈了 —— 那个是编译器基础设施，和这里的 kernel DSL 不是一回事。

## 状态

✅ `01-hello-world.py` —— 块索引 / grid 尺寸，三个语言版本输出一致。
依赖（`cuda-tile`、`cupy`）都在 `python/pyproject.toml` 的基础依赖里，`uv sync` 即可用。
