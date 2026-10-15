# cuTile Rust 手册 中文翻译版

[The cuTile Rust Book](https://nvlabs.github.io/cutile-rs/) 的中文译本。

> **免责声明**
>
> 本译文主要由 AI 完成，无法保证与原文完全一致。技术术语、代码标识符与 API 名称一律保留英文原文；若译文与[原文](https://nvlabs.github.io/cutile-rs/)存在冲突，请以原文为准。若有谬误，敬请指正。

## 原书信息

| 项 | 值 |
| --- | --- |
| 原文标题 | The cuTile Rust Book |
| 原文链接 | <https://nvlabs.github.io/cutile-rs/> |
| 上游仓库 | <https://github.com/NVIDIA/cuda-rust> |
| 版权 | Copyright (c) 2025-2026, NVIDIA Corporation |
| 许可证 | Apache License 2.0 |

本译本是对上游文档的翻译（衍生作品），依据 Apache-2.0 许可证分发，同样受该许可证约束。代码块中的示例代码保持原样，未作改动（仅注释被译为中文）。

## 目录

### 教程

1. [你好，世界](tutorials/01-hello-world.md) —— tile kernel 的启动、program id、编译流水线
2. [向量加法](tutorials/02-vector-addition.md) —— 分区（partitioning）与加载/存储模式
3. [SAXPY](tutorials/03-saxpy.md) —— 广播与就地操作
4. [矩阵乘法](tutorials/04-matrix-multiplication.md) —— 分块 GEMM、`mma`、const 泛型推断、持久化 GEMM
5. [融合 softmax](tutorials/05-fused-softmax.md) —— kernel 融合、数值稳定性、归约
6. [融合多头注意力](tutorials/06-flash-attention.md) —— 在线 softmax 与注意力分块
7. [异步执行入门](tutorials/07-intro-to-async.md) —— `DeviceOp`、同步 vs 异步、spawn 安全性
8. [数据并行 MLP](tutorials/08-data-parallel-mlp.md) —— 多设备调度与任务派生
9. [指针加法](tutorials/09-pointer-addition.md) —— 裸指针与异步安全性陷阱
10. [CUDA 图](tutorials/10-cuda-graphs.md) —— 捕获与重放
11. [NVFP4 推理](tutorials/11-nvfp4-inference.md) —— 打包 FP4 与块缩放 MMA
12. [DGX Spark 推理](tutorials/12-dgx-spark-inference.md) —— 统一内存上的大模型推理

### 用户指南

1. [简介](guide/introduction.md)
2. [宿主代码与设备代码](guide/host-vs-device.md)
3. [张量与 tile](guide/tensors-and-tiles.md)
4. [JIT 编译](guide/jit-compilation.md)
5. [设备操作](guide/device-operations.md)
6. [性能](guide/performance.md)
7. [自动调优](guide/autotuning.md)
8. [边界检查的放置](guide/bounds-check-placement.md)
9. [互操作](guide/interoperability.md)
10. [调试与性能分析](guide/debugging-and-profiling.md)
11. [实用的心智模型](guide/useful-mental-models.md)

### 参考

1. [DSL API](reference/dsl-api.md)
2. [宿主 API](reference/host-api.md)
3. [兼容性](reference/compatibility.md)
4. [术语表](reference/glossary.md)

## 本译本的目录说明

为了便于日后与上游同步 diff，本译本**保留原书目录结构与英文文件名**（`tutorials/`、`guide/`、`reference/`），正文内的相对链接与图片引用也与原文保持一致 —— 因此图片资源放在 `_static/images/` 下，而非仓库中其它译本使用的 `assets/`。

原文使用 Sphinx + MyST 构建，本译本面向纯 Markdown 阅读，做了以下等价转换：

| 原文写法（MyST / Sphinx） | 本译本写法 |
| --- | --- |
| `:::{note}` / `:::{tip}` 提示块 | `> **注意**` / `> **提示**` 引用块 |
| `:::{dropdown} Answer` 练习答案 | `> **答案**` 引用块 |
| `::::{grid}` 卡片网格（首页） | Markdown 列表与加粗小标题 |
| `{figure}` 指令 | 标准 `![说明](路径)` 图片语法 |
| `{raw} html` 时序图（设备操作篇） | 等价的 ```text 纯文本图 |
| `{toctree}` 目录树 | [首页](index.md) 与本文的 Markdown 目录 |
| `(anchor)=` MyST 锚点目标 | `<a id="anchor"></a>` 不可见锚点 |

最后一行的转换是必要的：标题被译成中文后，原文的英文锚点（如 `#tiled-matrix-multiply`）会失效。用 `<a id="...">` 保留原锚点名，可以让译本内部的所有交叉链接（含指向具体小节的那些）继续正常工作，同时在渲染结果中不可见。

## 环境与依赖

原文中的示例针对 `cutile` crate（cuTile Rust）。本仓库根目录下另有 `cutile-example/` 目录，收录了可直接在本机运行的学习示例，说明见 [`cutile-example/README.md`](../../cutile-example/README.md)。

相关环境要求（本机实测）：CUDA Toolkit 13.3、Rust 1.99.0、NVIDIA RTX 3080。
