# cuda-oxide 手册 v0.1.0 zh.CN

The cuda-oxide Book v0.1.0 的中文翻译版。

> **免责声明**：翻译主要由 AI 完成，无法保证译文与原文完全一致。若有谬误，敬请指正。

- 原文仓库：<https://github.com/NVlabs/cuda-oxide>
- 原文在线文档：<https://nvlabs.github.io/cuda-oxide/>
- 许可证：Apache-2.0（© 2025-2026 NVIDIA Corporation）

[开始阅读 →](./index.md)

---

## 关于本译本

- **目录结构、文件名与原书保持一致**，因此正文里的交叉链接和图片引用路径与上游逐字相同，便于后续与上游比对更新。
- 图片资源按原书布局放在各章的 `images/` 目录下，首页横幅在 `_static/images/`。
- 原文由 Sphinx + MyST 构建，本译本已转成普通 Markdown。转换对照表：

  | 原文（MyST） | 本译本 |
  | --- | --- |
  | `` ```{note} `` / `:::{note}` | `> **注意**` |
  | `` ```{tip} `` / `:::{tip}` | `> **提示**` |
  | `:::{warning}` | `> **警告**` |
  | `` ```{important} `` / `:::{important}` | `> **重要**` |
  | `:::{seealso}` | `> **参见**` |
  | `` ```{figure} `` / `` ```{image} `` | 标准 `![说明](路径)` |
  | `` ```{toctree} `` | 本页目录与 [index.md](./index.md) 中的链接列表 |
  | `::::{grid}` + `:::{grid-item-card}` | 加粗小标题 + 段落 |

- 代码块**逐字保留**（包括注释与字符串字面量），可直接复制运行。

## 章节一览

### 入门
1. [安装](getting-started/installation.md)
2. [Hello GPU](getting-started/hello-gpu.md)

### 编写 GPU 程序
1. [执行模型](gpu-programming/execution-model.md)
2. [kernel 与设备函数](gpu-programming/kernels-and-device-functions.md)
3. [内存与数据搬运](gpu-programming/memory-and-data-movement.md)
4. [虚拟内存与对等访问](gpu-programming/virtual-memory-and-peer-access.md)
5. [启动 kernel](gpu-programming/launching-kernels.md)
6. [kernel 家族](gpu-programming/kernel-families.md)
7. [闭包与泛型](gpu-programming/closures-and-generics.md)
8. [错误处理与调试](gpu-programming/error-handling-and-debugging.md)

### GPU 上的安全性
1. [安全模型](gpu-safety/the-safety-model.md)
2. [边界检查](gpu-safety/bounds-checks.md)

### 异步 GPU 编程
1. [DeviceOperation 模型](async-programming/the-device-operation-model.md)
2. [组合子与组合](async-programming/combinators-and-composition.md)
3. [调度与流](async-programming/scheduling-and-streams.md)
4. [并发执行](async-programming/concurrent-execution.md)
5. [重叠传输与计算](async-programming/overlapping-transfers-and-compute.md)

### 构建一个真正的应用程序
1. [异步 MLP 流水线](projects/async-mlp-pipeline.md)

### 进阶 GPU 特性
1. [共享内存与同步](advanced/shared-memory-and-synchronization.md)
2. [warp 级编程](advanced/warp-level-programming.md)
3. [张量内存加速器](advanced/tensor-memory-accelerator.md)
4. [矩阵乘法加速器](advanced/matrix-multiply-accelerators.md)
5. [集群编程](advanced/cluster-programming.md)

### 深入编译器
1. [架构概述](compiler/architecture-overview.md)
2. [pliron](compiler/pliron.md)
3. [rustc-public](compiler/rustc-public.md)
4. [rustc-codegen-cuda](compiler/rustc-codegen-cuda.md)
5. [MIR 导入器](compiler/mir-importer.md)
6. [编译器优化](compiler/compiler-optimizations.md)
7. [MLIR 方言](compiler/mlir-dialects.md)
8. [下降流水线](compiler/lowering-pipeline.md)
9. [添加新的内联函数](compiler/adding-new-intrinsics.md)
10. [生成式内联函数目录](compiler/catalog-generated-intrinsics.md)
11. [模糊测试与差异测试](compiler/fuzzing-and-differential-testing.md)

### 附录
1. [从源码构建](appendix/building-from-source.md)
2. [API 快速参考](appendix/api-quick-reference.md)
3. [支持的特性](appendix/supported-features.md)
4. [CUDA C++ 对比](appendix/cuda-cpp-comparison.md)
5. [生态系统](appendix/ecosystem.md)
6. [术语表](appendix/glossary.md)
