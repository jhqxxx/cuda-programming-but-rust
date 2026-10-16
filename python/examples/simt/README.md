# SIMT 示例（Python）

Python 写 SIMT kernel 有三条主流绑定路线，三条都是「写线程级代码」：

| 库 | 写法 | 特点 | 参考实现 |
| --- | --- | --- | --- |
| **PyTorch 扩展** ✅ | 写 `.cu`，用 `torch.utils.cpp_extension.load` 加载 | 写的是真 CUDA C++，能直接吃 `torch.Tensor`，`.cu` 还能被 C++ 侧复用 | `python/legacy/use_pytorch.py` + `vecadd.cu` |
| Numba | 给 Python 函数加 `@cuda.jit` | 用 Python 子集写 kernel，原型最快 | `python/legacy/use_numba.py` |
| PyCUDA | `SourceModule` 里内嵌 CUDA C 源码 | 最薄的包装，暴露驱动 API | `python/legacy/use_pycuda.py` |

**本目录只用 PyTorch 扩展那条。** 理由：

- 它的「内核」就是一份 `.cu`，和 `cpp/examples/simt/` 同构 —— 同一个文件可以让 CMake
  编（C++ 线）也可以让 `load()` 编（Python 线），正好对上本仓库「同模型跨语言对照」的结构
- 写的是完整 CUDA C++（模板、`__shared__`、warp intrinsics 都能用）：Numba 只支持
  Python 子集，PyCUDA 只到 CUDA C
- 和 tile 线的 Triton 一样直接吃 `torch.Tensor`，两条线能共用张量与对比代码

另外两条的参考实现在 `python/legacy/`，保留作对照，**不再扩展**。

## 这个目录放什么

按主题编号的 SIMT 示例，每个主题一个子目录，内含 CUDA 源 + Python 驱动脚本：

```
simt/
└── 00-add/
    ├── add.cu       # kernel + pybind11 导出（可以和 cpp/examples/simt/ 共用）
    └── add.py       # 加载并验证
```

加载方式两种：短例子用 `torch.utils.cpp_extension.load_inline`（源码内联，不必建
`.cu` 文件）；内核一长就拆成独立 `.cu`，用 `load(name=..., sources=[...])`。

`.cu` 路径用 `Path(__file__).parent` 拼，不要用裸相对路径 —— 否则必须 `cd` 到本目录
才能跑。

## 与 tile 路线的差别

**只和同模型的实现对比**：本目录该对照的是 `../../../cuda-oxide-example/01-vector-add/`
（Rust 的 SIMT 线）和 `../../../cpp/examples/simt/`，**不是** Triton 版
（`../tile/`）—— 两者的抽象层级不同：

| | SIMT（本目录） | tile（`../tile/`） |
| --- | --- | --- |
| 你描述的是 | 一个**线程**做什么 | 一个**块**处理哪一块数据 |
| 索引 | `cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x` | `tl.program_id(0)` |

## 状态

⬜ 暂无内容。
