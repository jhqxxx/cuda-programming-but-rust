# CUDA Programming But Rust 🦀⚡

> **English** → [README.en.md](./README.en.md)

<p align="center">
  <img src="https://img.shields.io/badge/rustc--codegen--cuda-v0.1.0-blue" alt="cuda-oxide v0.1.0"/>
  <img src="https://img.shields.io/badge/Rust-nightly--2026--04--03-orange" alt="Rust nightly-2026-04-03"/>
    <img src="https://img.shields.io/badge/license-Apache--2.0-lightgrey" alt="Apache 2.0"/>
</p>

<p align="center">
  <b>用 Rust 写 CUDA Kernel，完全摆脱 .cu 文件！</b>
</p>

这是一个 **cuda-oxide**（[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)）学习项目，使用纯 Rust 语法编写 CUDA GPU 程序，涵盖从基础到高级的各种特性。**正在持续更新中 🚧**

---

## 📖 配套文档

| 文档 | 说明 |
|------|------|
| [📘 cuda-oxide 手册 v0.1.0 中文版](./cuda-oxide-zh.CN/README.md) | cuda-oxide 官方手册的中文翻译 |
| [📙 CUDA 编程指南 v13.3 中文版](./CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA 编程指南的中文翻译 |

---

## ✨ 特性总览

### 🦀 Rust CUDA Kernels（`src/`）

| 模块 | 文件 | 内容 |
|------|------|------|
| **elementwise** | [`src/elementwise/`](./src/elementwise/mod.rs) | 逐元素加法：f32、f32x4 向量化、f16 半精度，以及 `#[device]` 辅助函数 |
| **generic** | [`src/generic/`](./src/generic/mod.rs) | 泛型 kernel：`scale<T>`、`add<T>`、闭包捕获与高阶函数 |
| **constant_memory** | [`src/constant_memory/`](./src/constant_memory/mod.rs) | CUDA 常量内存：标量常量、数组常量、主机端 setter |
| **shared_mem** | [`src/shared_mem/`](./src/shared_mem/mod.rs) | 共享内存：`SharedArray` 静态声明、动态共享内存、对齐控制（16/128/256 字节）、混合对齐分区 |
| **barrier** | [`src/barrier/`](./src/barrier/mod.rs) | 线程屏障：mbarrier 同步屏障、屏障保护下的共享内存数据交换 |
| **device_global** | [`src/device_global/`](./src/device_global/mod.rs) | 设备全局变量：跨 kernel 共享状态、静态常量表、静态初始化器边界（NaN 位模式） |
| **atomic** | [`src/atomic/`](./src/atomic/mod.rs) | 原子操作：19 个测试，覆盖多种操作/类型/作用域/内存序 |
| **printf** | [`src/printf/`](./src/printf/mod.rs) | GPU 端 `gpu_printf!`：整数/浮点/布尔/格式化输出 |
| **utils** | [`src/utils/`](./src/utils/mod.rs) | 工具函数：`get_launch_config_1d` 便捷启动配置 |

### 🔬 C++ CUDA 对照（`use_cpp/`）

用于与 Rust 实现做对照学习的传统 CUDA C++ 示例：

| 示例 | 文件名 | 说明 |
|------|--------|------|
| 设备属性查询 | [`get_device_attr.cu`](./use_cpp/examples/get_device_attr.cu) | 查询 GPU 设备属性 |
| 原子操作 | [`atomic_test.cu`](./use_cpp/examples/atomic_test.cu) | CUDA 原子操作 |
| 常量内存 | [`const_mem.cu`](./use_cpp/examples/const_mem.cu) | 常量内存使用 |
| 显式内存管理 | [`explicit_mem.cu`](./use_cpp/examples/explicit_mem.cu) | `cudaMalloc`/`cudaMemcpy` 等 |
| 共享内存 | [`shared_mem.cu`](./use_cpp/examples/shared_mem.cu) | 共享内存 tile 加载 |
| 统一内存 | [`unified_mem.cu`](./use_cpp/examples/unified_mem.cu) | `cudaMallocManaged` 统一内存 |
| Stream + Event | [`stream_event.cu`](./use_cpp/examples/stream_event.cu) | CUDA 流和事件 |
| 矩阵转置 v1 | [`transpose_v1.cu`](./use_cpp/examples/transpose_v1.cu) | 朴素矩阵转置 |
| 矩阵转置（共享内存） | [`transpose_with_shared_mem.cu`](./use_cpp/examples/transpose_with_shared_mem.cu) | 共享内存优化的矩阵转置 |

使用 CMake 构建：
```bash
cd use_cpp && cmake -B build && make -C build
```

### 🐍 Python CUDA 对照（`use_python/`）

使用 Python CUDA 框架的参考实现：

| 文件 | 说明 |
|------|------|
| [`about_package/vecadd.cu`](./use_python/about_package/vecadd.cu) | CUDA C++ 模板 kernel 文件 |
| [`about_package/use_numba.py`](./use_python/about_package/use_numba.py) | Numba CUDA 示例 |
| [`about_package/use_pycuda.py`](./use_python/about_package/use_pycuda.py) | PyCUDA 示例 |
| [`about_package/use_pytorch.py`](./use_python/about_package/use_pytorch.py) | PyTorch CUDA 示例 |
| [`kernels/elementwise.py`](./use_python/kernels/elementwise.py) | Python 逐元素操作 kernel |

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Linux | Ubuntu 24.04 已验证，其他发行版可能可用 |
| NVIDIA GPU | Ampere+ (sm_80+) | 建议驱动 ≥ 545 |
| CUDA Toolkit | 12.x+ | 需要 `nvcc` 和 `cuda.h` |
| LLVM | 21+ | 必须包含 NVPTX 后端 |
| Clang | 21+ | `clang-21` — bindgen 编译 `cuda-bindings` 所需 |
| Rust | Nightly (pinned) | 由 `rust-toolchain.toml` 锁定 |

> ⚠️ cuda-oxide 目前仅支持 Linux，不支持 Windows。

### 安装注意事项

完整安装说明请查看 [cuda-oxide 官方安装文档](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html)，以下为要点：

- **Dev Container** 或 **Nix flake** 是最快的可重现环境方式
- **手动安装** 需要安装：CUDA Toolkit → LLVM 21+（含 NVPTX 后端）→ Clang 21+ → Rust nightly（`rust-src` + `rustc-dev` 组件）→ `cargo-oxide`
- `cargo oxide doctor` 可一键诊断环境
- 如果使用 Dev Container，可跳过 CUDA Toolkit、LLVM、Clang、Rust 的手动安装

### 在本项目中运行

```bash
# 克隆项目
git clone https://github.com/jhqxxx/cuda-programming-but-rust.git
cd cuda-programming-but-rust

# 运行所有测试（直接使用 Rust 测试框架）
cargo oxide test

# 仅运行特定模块的测试
cargo oxide test -- elementwise
cargo oxide test -- shared_mem
cargo oxide test -- constant_memory
cargo oxide test -- generic
cargo oxide test -- printf
```

---

## 🧪 测试与运行结果

当前共有 **43 个测试**，全部通过 ✅

```
test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### 测试覆盖

| 模块 | 测试数 | 覆盖内容 |
|------|--------|----------|
| `elementwise` | 4 | f32 加法、f32x4 向量化、f16 半精度、device 辅助函数 |
| `generic` | 6 | scale(f32/f16)、add(f32/f16)、closure(f32/f16) |
| `constant_memory` | 2 | 线性变换、多项式 |
| `shared_mem` | 6 | 静态单/双 TILE、动态基础/分区/128对齐/混合对齐 |
| `barrier` | 2 | mbarrier 同步屏障、屏障保护下的共享内存数据交换 |
| `device_global` | 3 | 设备全局变量、静态常量表、静态初始化器边界（NaN 位模式等） |
| `atomic` | 19 | 多种原子操作/类型、`.gpu` 与 `.cta` 作用域、多种内存序 |
| `printf` | 1 | 整数/浮点/格式化/布尔/返回值 |

---

## 📦 项目结构

```
cuda-programming-but-rust/
├── Cargo.toml              # 项目依赖（cuda-device, cuda-host, cuda-core）
├── rust-toolchain.toml     # nightly 工具链配置
├── README.md               # 本文件
├── src/                    # Rust CUDA 源代码
│   ├── lib.rs              # 库入口
│   ├── main.rs             # 二进制入口
│   ├── elementwise/        # 逐元素操作 kernel
│   ├── generic/            # 泛型 kernel
│   ├── constant_memory/    # 常量内存 kernel
│   ├── shared_mem/         # 共享内存 kernel
│   ├── barrier/            # 线程屏障 kernel（mbarrier）
│   ├── device_global/      # 设备全局变量 kernel
│   ├── atomic/             # 原子操作 kernel
│   ├── printf/             # GPU printf kernel
│   └── utils/              # 工具函数
├── use_cpp/          # 对照：C++ CUDA 实现
│   ├── CMakeLists.txt
│   ├── common/             # 公共头文件和 CUDA 工具
│   └── examples/           # C++ CUDA 示例
├── use_python/             # 对照：Python CUDA 实现
│   ├── about_package/      # Numba/PyCUDA/PyTorch 示例
│   └── kernels/            # Python kernel
├── cuda-oxide-zh.CN/       # cuda-oxide 手册中文翻译
└── CUDA编程指南-V13.3-zh.CN/ # CUDA 编程指南中文翻译
```

---

## 🧠 学习路线

1. **入门** → [cuda-oxide 手册 第一章](./cuda-oxide-zh.CN/01-入门/01-安装.md)
2. **逐元素操作** → [`elementwise`](./src/elementwise/mod.rs) — 最简单的 kernel
3. **泛型和闭包** → [`generic`](./src/generic/mod.rs) — Rust 泛型的 GPU 应用
4. **常量内存** → [`constant_memory`](./src/constant_memory/mod.rs) — 只读参数缓存
5. **共享内存** → [`shared_mem`](./src/shared_mem/mod.rs) — 块内协作优化
6. **线程屏障** → [`barrier`](./src/barrier/mod.rs) — mbarrier 与块内同步
7. **设备全局变量** → [`device_global`](./src/device_global/mod.rs) — 跨 kernel 共享状态
8. **原子操作** → [`atomic`](./src/atomic/mod.rs) — 原子性与内存序
9. **调试输出** → [`printf`](./src/printf/mod.rs) — GPU 端调试技巧

---

## 🔧 技术栈

- **[cuda-oxide](https://github.com/NVlabs/cuda-oxide)** — NVIDIA 开源的 Rust CUDA 编译器后端，将 Rust 代码直接编译为 PTX
- **Rust nightly** — 利用 `rustc_codegen_cuda` 自定义 codegen 后端
- **CUDA 12.x** — 底层 CUDA 驱动 API（通过 `cuda-core` crate 封装）
- **CMake** — C++ CUDA 示例的构建系统
- **Python** — PyCUDA / Numba / PyTorch CUDA 参考实现

---

## 📚 参考资源

- [cuda-oxide 官方文档](https://nvlabs.github.io/cuda-oxide/index.html)
- [NVlabs/cuda-oxide GitHub](https://github.com/NVlabs/cuda-oxide)
- [NVIDIA CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## 📄 许可证

[Apache-2.0](./LICENSE)

## ⚠️ 注意

- 本项目仍在开发中，内容会持续更新
- cuda-oxide 目前处于早期阶段，API 可能发生变化
- 文档翻译由 AI 辅助完成，如有谬误欢迎指正
