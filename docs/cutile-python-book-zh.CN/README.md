# cuTile Python 手册 中文翻译版

[cuTile Python](https://docs.nvidia.com/cuda/cutile-python/index.html) 官方文档的中文译本。

> **免责声明**
>
> 本译文主要由 AI 完成，无法保证与原文完全一致。技术术语、代码标识符与 API 名称一律保留英文原文；若译文与[原文](https://docs.nvidia.com/cuda/cutile-python/index.html)存在冲突，请以原文为准。若有谬误，敬请指正。

## 原文信息

| 项 | 值 |
| --- | --- |
| 原文标题 | cuTile Python |
| 原文链接 | <https://docs.nvidia.com/cuda/cutile-python/index.html> |
| 版权 | Copyright © 2025, NVIDIA Corporation |
| 翻译所依据的抓取日期 | 2026-10-16 |

## cuTile Python 是什么

cuTile 是面向 NVIDIA GPU 的**并行编程模型**，同时也是一套基于 Python 的 **DSL**。
它会自动利用 tensor cores、tensor memory accelerator 等先进硬件能力，同时在不同
NVIDIA GPU 架构之间保持可移植性 —— 无需改动代码即可用上最新的硬件特性。

一句话概括它和 SIMT 的分工：**cuTile 让你描述「一个 block 处理哪一块数据」，
而不是「一个线程算哪个下标」。**

## 目录

上游文档的导航是一个**扁平列表**，共 12 页（上游**没有**「参考 / Reference」这一节）。
本译本的对应关系：

| # | 上游页面 | 本译本 |
| --- | --- | --- |
| 1 | cuTile Python | [index.md](index.md) |
| 2 | Release Notes | [release-notes.md](release-notes.md) |
| 3 | Quickstart | [guide/01-quickstart.md](guide/01-quickstart.md) |
| 4 | Execution Model | [guide/02-execution.md](guide/02-execution.md) |
| 5 | Data Model | [guide/03-data.md](guide/03-data.md) |
| 6 | Memory Model | [guide/04-memory-model.md](guide/04-memory-model.md) |
| 7 | Operations | [guide/05-operations.md](guide/05-operations.md) |
| 8 | Interoperability | [guide/06-interoperability.md](guide/06-interoperability.md) |
| 9 | Performance Tuning | [guide/07-performance.md](guide/07-performance.md) |
| 10 | Compilation and Export | [guide/08-compilation.md](guide/08-compilation.md) |
| 11 | Debugging | [guide/09-debugging.md](guide/09-debugging.md) |
| 12 | Known Issues | [guide/10-known-issues.md](guide/10-known-issues.md) |

### 未翻译的部分：105 个自动生成的 API 页

上游站点还有一批**自动生成的逐符号 API 页**：`generated/cuda.tile.*.html`，共 105 个。
它们**不出现在左侧导航**中 —— 唯一入口是上面第 7 项
[操作](guide/05-operations.md) 页里的分类表格：点表里的函数名（例如 `load`）
才会跳到对应的 `generated/cuda.tile.load.html`。

**这 105 页本译本尚未翻译。** 需要逐函数查阅时请直接看上游：

- 入口（分类表格）：<https://docs.nvidia.com/cuda/cutile-python/operations.html>
- 单个函数示例：<https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.load.html>

## 翻译进度

上游共 **12 个导航页** + **105 个不在导航里的自动生成页**：

| 部分 | 状态 |
| --- | --- |
| 12 个导航页（含发行说明） | ✅ 已全部翻译 |
| 105 个自动生成的 API 页 | ⬜ 未翻译，见上 |

## 相关文档

本仓库里另有几份相关的中文翻译，可对照阅读：

| 文档 | 说明 |
| --- | --- |
| [cuTile Rust 手册](../cutile-rs-book-zh.CN/README.md) | 同一编程模型的 Rust 版本（tile 模型） |
| [cuda-oxide 手册](../cuda-oxide-book-zh.CN/README.md) | Rust 的 SIMT 路线（自定义 rustc codegen 后端） |
| [CUDA 编程指南 v13.3](../CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA C++ 编程指南（SIMT） |

本手册讲的是 **tile 模型**的 Python 实现；如果要和 SIMT 的写法对照，请按模型配对阅读
（Python 侧的 SIMT 对应 Triton / Numba / PyCUDA / PyTorch 扩展，而不是本手册）。
