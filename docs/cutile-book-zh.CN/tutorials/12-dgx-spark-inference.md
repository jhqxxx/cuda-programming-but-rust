# 12. 在 DGX Spark 上做 LLM 推理

DGX Spark 的统一内存可以用 16 位精度装下 Qwen3-32B。我们将用 [Grout](https://github.com/huggingface/grout) 来运行它 —— 一个基于 cuTile Rust 构建的推理引擎，先从较小的 Qwen3-4B 模型开始。

下面的测量数据来自一台 DGX Spark（GB10、`sm_121`、20 个 CPU 核心、128 GB 统一 LPDDR5X 内存），运行 DGX OS、580 驱动和 CUDA 13.4。使用 cuTile Rust 0.4.0（[cuda-rust 仓库](https://github.com/NVIDIA/cuda-rust)中的 tag `cutile-rs/v0.4.0`）以及 commit [`deb7427`](https://github.com/elibol/grout/commit/deb74274c731d4ca9e7f9b356a91892398498309) 的 Grout。

---

## 前置条件

这里使用的 DGX OS 镜像自带 CUDA 13.0，对 cuTile 来说太旧了，而且缺少 Tile IR 汇编器。请在保留 13.0 的同时另装一套较新的工具包：

```bash
sudo apt-get install -y cuda-toolkit-13-4 cuda-tileiras-13-4
sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
```

然后把工具链指向新的工具包（把下面这些加到 `~/.bashrc`）：

```bash
export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_TOOLKIT_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH
```

`CUDA_TOOLKIT_PATH` 的优先级高于 `CUDA_HOME`。设置它同时也会覆盖 Grout 的 `.cargo/config.toml` 中 `/usr/local/cuda-13` 这一默认值。

确认两个 CUDA 工具都报告 13.4 版本，且 Cargo 使用的是 nightly：

```bash
nvcc --version
tileiras --version
cargo --version
```

580 驱动是 CUDA 13.0 的驱动。CUDA 的次版本兼容性让 13.x 工具包可以在它上面运行，而 Tile IR 汇编器输出的是原生的 `sm_121` 代码而不是 PTX，因此本教程中的任何环节都不需要更新的驱动。

---

## 目录布局

下面这些命令把模型目录和 cuTile Rust 的检出放在 Grout 旁边：

```text
~/dev/
  cuda-rust/     # cuda-rust 检出，位于 tag cutile-rs/v0.4.0
  cutile-rs/     # 指向 cuda-rust/cutile-rs 的软链接（Grout 针对它构建）
  grout/         # 推理引擎
  hf_models/
    qwen3_4b/
    qwen3_32b/
```

```bash
mkdir -p ~/dev && cd ~/dev
git clone --branch cutile-rs/v0.4.0 https://github.com/NVIDIA/cuda-rust.git
ln -s cuda-rust/cutile-rs cutile-rs
git clone https://github.com/elibol/grout.git
git -C grout checkout deb74274c731d4ca9e7f9b356a91892398498309
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-4B  --local-dir hf_models/qwen3_4b    # about 8 GB
hf download Qwen/Qwen3-32B --local-dir hf_models/qwen3_32b   # about 66 GB
```

Grout 固定的那个 commit 依赖 cuTile Rust 0.4.0。它的 `[patch.crates-io]` 使用同级的 `cutile-rs` 路径，上面那个软链接会把它解析到 cuda-rust 的检出目录。

用 release 模式构建 Grout。在 Spark 上，依赖编译过一次之后，整个构建用不到一分钟：

```bash
cd ~/dev/grout && cargo build --release
```

这会构建出推理二进制。自动调优器需要 `benchmarks` feature；见下面的[自动调优](#autotuning)。

---

## 第一次运行：Qwen3-4B

```bash
cargo run --release -- \
  --model ../hf_models/qwen3_4b \
  --prompt "Explain unified memory on DGX Spark in three sentences." \
  --max-new-tokens 200 --profile
```

这些运行使用 Grout 内置的调优默认值。固定的那个 revision 自带 `sm_100` 和 `sm_120` 的记录；由于没有 `benchmarks/tuning/sm_121/` 目录，Grout 会直接使用默认值而不打印警告。

首次运行时，cuTile 会把 Grout 的 kernel 从 Rust 经 Tile IR 编译成 `sm_121` 机器码并缓存到磁盘。之后的运行会从缓存加载已编译的 kernel。

| Qwen3-4B，16 位权重 | 首次运行 | 预热后运行 |
|---|---|---|
| 模型加载 | 9.9 s | 10.0 s |
| kernel 编译 | 11.7 s（37 个 kernel） | 0 s（缓存命中） |
| 解码 | 29.7 tokens/s | 29.5 tokens/s |
| 峰值内存占用 | 15 GB | 15 GB |

模型加载时间主要花在从磁盘读取权重上。

`--profile` 会打印 prefill 与 decode 步骤的平均值。`GROUT_PROFILE_OPS=1` 会额外打印一张逐操作的表格；`GROUT_PROFILE_SYNC_OPS=1` 会在每个操作之后做同步，从而把 GPU 执行时间也计入。该表格覆盖 prefill 和预热阶段。decode 会重放一张 CUDA 图，所以它的各个操作不会被单独计时。

```bash
GROUT_PROFILE_OPS=1 GROUT_PROFILE_SYNC_OPS=1 cargo run --release -- \
  --model ../hf_models/qwen3_4b \
  --prompt "Explain unified memory on DGX Spark in three sentences." \
  --max-new-tokens 200 --profile
```

用这些设置来观察单个操作，然后在测量吞吐时把它们去掉：同步本身会改变执行时序。

---

<a id="the-model-that-does-not-fit-a-gpu-qwen3-32b"></a>
## Qwen3-32B

Qwen3-32B 有 328 亿参数，16 位精度下约 66 GB，另外还有随上下文长度增长的 KV cache（32K token 时约 9 GB）。两者都能放进 Spark 的 128 GB 统一内存。

```bash
cargo run --release -- \
  --model ../hf_models/qwen3_32b \
  --prompt "What is 2 + 2? Then explain why in one paragraph." \
  --max-new-tokens 64 --max-seq-len 4096 --profile
```

| Qwen3-32B，16 位权重 | 实测 |
|---|---|
| 解码 | 3.7 tokens/s |
| 内存中的权重 | 66 GB |

两个模型使用同样的 37 个已编译 kernel。Qwen3-32B 每生成一个 token 要读取更多的权重数据。

---

## 时间花在哪

解码一个 token 需要把每一个权重读一遍。按每 token 66 GB 计算，3.7 tokens/s 对应约 245 GB/s 的持续访存；Spark 的 LPDDR5X 峰值接近 273 GB/s。对 4B 模型，每 token 8 GB、29.5 tokens/s 对应约 236 GB/s。两种情况下，解码都跑在内存屋顶线的大约 90%。

`GROUT_ATTN_BN_DECODE` 设置 Grout 解码注意力的 tile 大小（默认 32）。下面几次运行手工改变了它：

| `GROUT_ATTN_BN_DECODE` | Qwen3-4B 解码 |
|---|---|
| 16 | 29.1 tokens/s |
| 32 | 29.5 tokens/s |
| 64 | 28.9 tokens/s |

改变 tile 尺寸对解码影响很小：每个 token 仍然需要同样多的权重数据。**Prefill** 是计算受限的，会对 tile 形状有响应。请用长提示（数千 token）配合逐 kernel 的性能分析器来测量调优效果。

**更低精度的权重**用 8 位或 4 位格式可以把每 token 的字节数减半甚至减到四分之一。[NVFP4 教程](./11-nvfp4-inference.md)介绍了 4 位推理所需的 kernel。

不同的 tile 尺寸会改变浮点累加的顺序，这在 logits 几乎并列时可能改变贪心解码选出的 token。16 和 64 那两次运行彼此产生了相同的文本，但在第二句之后与默认值产生了分歧。

<a id="autotuning"></a>
### 自动调优

从 Grout 的检出构建自动调优器：

```bash
cargo build --release --features benchmarks --bin grout_autotune
target/release/grout_autotune --help
```

要做 prefill 注意力的扫描，请使用 `--site prefill_attention`，并准备一个包含 `pp_512.txt`、`pp_2048.txt`、`pp_8192.txt` 的 `--prompt-dir`。Grout 的 `benchmarks/make_prompts.py` 用模型的 tokenizer 生成这些文件；它需要 Python 的 `transformers` 包。

调优器默认把记录和试验日志写到 `benchmarks/tuning/sm_121/` 下。`--budget-min` 限制每个提示长度桶的搜索时间。重新运行同一条命令会从试验日志恢复。如果分配恢复以退出码 3 退出，请在一个新进程里重新运行它。

上面的测量用的是内置默认值。请在 Spark 上生成记录，并与默认值比较时序之后再采用它们。把 `sm_120` 的记录留在它们自己的目录里；那些是在 RTX 5090 上调出来的。

---

<a id="notes-on-unified-memory"></a>
## 统一内存

Spark 的 CPU 与 GPU 通过一个一致的地址空间（ATS 模式）共享物理内存。

- **宿主内存就是设备内存。** 一个可分页的宿主缓冲区的地址在 GPU 上是有效的：用 `Tensor::from_foreign` 包装一个 `Vec` 并在其上启动 kernel 是可以工作的。驱动的指针属性查询不会把这类指针归类，因此在需要以那种方式校验设备指针的代码上，这个平台需要额外的检查。
- **回读到新分配的可分页内存很慢。** 把一个 256 MB 的 tensor 拷到一个新分配的宿主 `Vec` 只跑到 0.13 GB/s，而设备到宿主方向是 18.6 GB/s，宿主 `memcpy` 是 24 GB/s。目标端的缺页占了主导。当回读处于热点路径上时，请复用或预先触碰（pre-touch）目标缓冲区，或者使用锁页（pinned）宿主内存。

---

## 要点

| 概念 | 在 DGX Spark 上的含义 |
|---|---|
| **首次启动** | kernel 编译在首次使用时耗时约 12 s；之后的运行从磁盘缓存加载 |
| **内存容量** | 128 GB 统一内存可以装下 16 位精度的 32B 模型 |
| **解码屋顶线** | 内存带宽决定每秒 token 数；两个模型都跑在该上限的约 90% |
| **调优在哪里有用** | prefill 等计算受限的阶段；16 位下解码的 tile 尺寸几乎无影响 |
| **统一内存** | 宿主指针在设备上可用；回读到新分配的可分页内存是慢路径 |

---

## 参见

- [NVFP4 推理](./11-nvfp4-inference.md) —— 更低精度推理所需的 kernel
- [性能](../guide/performance.md) —— kernel 调优
- [JIT 编译](../guide/jit-compilation.md) —— 编译与 kernel 缓存
- [互操作](../guide/interoperability.md) —— `Tensor::from_foreign` 与外部拥有的内存
