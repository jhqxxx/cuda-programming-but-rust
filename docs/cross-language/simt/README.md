# SIMT 跨语言对照

**占位** —— SIMT 线的示例还没写，这里先留着位置。

## 为什么不能和 `../tile/` 混在一起

`tile` 和 `simt` 是**两种不同的抽象**（块级 vs 线程级），不是一个东西的两种写法。
把两边的对照表放一起，会让人误以为「Rust 的这块等价于 C++ 的那块」——
所以本目录在 `docs/cross-language/` 下和 `tile/` **平级分开**。

参考：根 [README](../../../README.md) 的「两个编程模型」一节。

## 将来会写什么

等示例落地后，按同样的格式补（每个主题一篇，编号和示例对齐）：

| 计划 | 主题 | 对应示例 |
| --- | --- | --- |
| `01-thread-index.md` | 全局线程索引 `threadIdx + blockIdx * blockDim` | `01-vector-add`（Rust `cuda-oxide-example/` · C++ `cpp/examples/simt/` · Python `python/examples/simt/`） |

预计差异会比 tile 线**更大**：SIMT 三语言用的根本不是同一套编译器
（Rust 走 cuda-oxide 的 codegen 后端，C++ 走 nvcc，Python 走 Numba / PyCUDA /
PyTorch 扩展），所以连「语义一致」这个前提都不一定成立，对照时得更小心。
