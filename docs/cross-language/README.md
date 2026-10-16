# 跨语言对照（Rust / C++ / Python）

同一件事在三种语言里怎么写、哪里一样、哪里不一样。

## 为什么需要这个目录

三种语言的 **tile** 实现共享同一个编译器后端（CUDA Tile IR → `tileiras`），
所以语义一致、API 也一一对应 —— 但**前端的命名和签名各叫各的**。光看一种语言照着
写另一种，必然会踩坑。最先撞上的就是取块索引：

| | Rust | C++ | Python |
| --- | --- | --- | --- |
| 取块索引 | `program_id(0)` | `ct::bid().x` | `ct.bid(0)` ← **不叫 `program_id`** |

这个目录就是这些「同模型、不同拼法」的速查表，外加每个主题实测踩过的坑。

## 怎么读

1. **按模型分目录**：`tile/` 和 `simt/` 是两种不同的抽象，**不要横向对照**
   （原因见根 [README](../../README.md)）
2. **只信实测**：每条对照都在下面那套环境里真跑过，不是照抄官方文档
3. **编号和示例对齐**：`tile/01-block-index.md` 对应三语言里的 `01-hello-world.*`，
   学的时候可以对着开三个窗口看

## 目录

| 文件 | 主题 | 对应示例 |
| --- | --- | --- |
| [`tile/01-block-index.md`](./tile/01-block-index.md) | 块索引、grid 尺寸、设备端打印、启动 | `01-hello-world`（[Rust](../../cutile-rs-example/01-hello-world.rs) / [C++](../../cpp/examples/tile/01-hello-world.cu) / [Python](../../python/examples/tile/01-hello-world.py)） |

`simt/` 目前只有占位，等 SIMT 线的示例写起来再补。

## 和示例 README 的关系

**本目录是唯一来源。** [`cpp/examples/tile/README.md`](../../cpp/examples/tile/README.md)
和 [`python/examples/tile/README.md`](../../python/examples/tile/README.md) 里原本各有一张
对照表，已经改成指过来，避免同一张表三处维护。

## 实测环境

本目录所有结论都在这套环境上跑过：

| 项 | 版本 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080（compute capability 8.6） |
| 驱动 | 615.71.09（R615 = CUDA 13.4） |
| CUDA Toolkit | 13.3（`/usr/local/cuda`） |
| cuTile Rust | `cutile` 0.4.0（git `bc46b1ff`），Rust 1.99.0 stable |
| CUDA Tile C++ | CTK 13.3 自带，`nvcc --enable-tile`（**需 `-std=c++20`**） |
| cuTile Python | `cuda-tile` 1.6.0，Python 3.12.9（uv 管理） |
