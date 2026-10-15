# Rust + GPU 生态系统

cuda-oxide 是若干让 GPU 计算可以从 Rust 访问的活跃尝试之一。每个项目解决的是
问题的一个不同侧面 —— 图形着色器、隐式卸载、CUDA 编程、安全的驱动绑定 ——
而整个版图正在快速向前推进。本附录简要勾勒了 cuda-oxide 相对于其邻近项目的
位置，以及这些项目之间的相互关系。

## cuda-oxide 的定位

| Project          | Approach                          | Target              | Scope                                          |
|:-----------------|:----------------------------------|:--------------------|:-----------------------------------------------|
| **cuda-oxide**   | `rustc` codegen backend           | NVIDIA PTX/SASS     | CUDA programming model in safe Rust            |
| **Rust-GPU**     | `rustc` → SPIR-V                  | Vulkan/Metal/DX     | Graphics shaders and compute via SPIR-V        |
| **rust-cuda**    | `rustc` → NVVM IR                 | NVIDIA PTX          | Rust language model on NVIDIA GPUs             |
| **CubeCL**       | Embedded DSL + JIT runtime        | CUDA/ROCm/WGPU      | Cross-vendor compute kernels from a Rust DSL   |
| **std::offload** | `rustc` + LLVM offload            | NVIDIA/AMD/Intel    | Implicit offload of CPU code                   |
| **cudarc**       | Safe CUDA driver bindings         | NVIDIA              | Host-side bindings to the CUDA driver          |
| **wgpu**         | WebGPU API + WGSL/Naga            | Cross-platform      | Portable compute via shader languages          |

“Scope”一列刻画的是每个项目所优化的**设计中心**，而不是它的功能上限。其中
若干项目在边缘处存在重叠，单个应用可能会同时引入不止一个项目。

## cuda-oxide 与 rust-cuda

最接近的邻居 —— 也是最常与 cuda-oxide 混淆的一个 —— 是 `rust-cuda`。两个项目
都使用 `rustc` codegen backend 面向 NVIDIA GPU，从远处看它们几乎可以互换。
但靠近来看，就我们对这两个项目的解读，它们的设计中心指向不同方向：

- **rust-cuda** 关注的是把 **Rust 带到 NVIDIA GPU**：诸如 `async`/`.await`
  这样的 Rust 人体工学、部分标准库在设备端运行，以及一套以 Rust 为先、
  对 CUDA 概念做抽象的程序设计模型。
- **cuda-oxide** 关注的是把 **CUDA 带入 Rust**：kernel 编写、设备 intrinsics、
  SIMT 执行模型，以及用安全 Rust 原生表达的 CUDA 编程模型 —— 在精神上更接近
  用 C++ 写一个 `__global__` 函数，而不是写一个恰好能在 GPU 上运行的通用
  Rust 函数。

两个方向都很有价值，我们相信这两个项目是互补的。随着两个项目的成熟，我们
一直在与 rust-cuda 的维护者合作，并期望继续保持这种合作。

## 其他邻近项目

- **Rust-GPU** 通过 SPIR-V 面向图形导向的 GPU 编程，可覆盖 Vulkan、Metal 与
  DirectX。它支持计算着色器，但设计中心是图形；如果你需要跨厂商可移植性或
  着色器互操作性，Rust-GPU 是合适的工具。
- **CubeCL** 是一个嵌入式 DSL：你用 `#[cube]` 标注一个 Rust 函数，CubeCL
  运行时按需将其 JIT 编译为 CUDA、ROCm/HIP 或 WGSL。它不是 `rustc` backend
  —— `#[cube]` proc-macro 会把函数体重写成 CubeCL IR，随后运行时再将其下沉
  到宿主 GPU 所暴露的任一 backend。其设计中心是*跨厂商可移植性*（同一个
  kernel 可在 NVIDIA、AMD 与 WGPU 上运行），而不是完整的 Rust 语言覆盖；
  作为交换，CubeCL 放弃了在 kernel 内使用任意 Rust 构造的能力，转而使用
  一套刻意受限的 DSL 表面。CubeCL 是 [Burn](https://github.com/tracel-ai/burn)
  ML 框架背后的计算 backend，也是它的主要展示应用。cuda-oxide 与 CubeCL
  在很大程度上是互补的：当你需要通过受控 DSL 让同一个 kernel 跨 GPU 厂商
  运行时用 CubeCL；当你需要在 NVIDIA 硬件上针对完整 CUDA 编程模型编写地道
  的安全 Rust 时用 cuda-oxide。
- **std::offload** 是一项 Rust 语言特性（目前为 nightly），它使用 LLVM 的
  offload runtime 把 CPU 循环隐式迁移到加速器上。程序设计模型不同：用户编写
  CPU 风格的代码，由编译器/运行时处理卸载。cuda-oxide 是显式的；offload 是
  隐式的。
- **cudarc** 提供对 CUDA driver API 的安全 Rust 绑定。它是一个宿主侧库，用于
  启动在别处编写的 kernel（通常是 PTX 或 CUDA C++）。cuda-oxide 的宿主侧层
  （`cuda-core`，与 cutile-rs 共享）与 `cuda-host` 的启动宏紧密集成，但
  cuda-oxide 产出的 PTX 是可移植的 —— 如果你想在一个不依赖 cuda-oxide 其余
  技术栈的项目中启动 cuda-oxide kernel，cudarc 是个不错的选择。
- **wgpu** 是 WebGPU API 的 Rust 实现 —— 一个对 Vulkan、Metal 与 DirectX 12
  （以及浏览器中的 WebGPU）的运行时抽象。计算与图形着色器用 WGSL（WebGPU
  着色语言）编写，并由 Naga 翻译为各 backend 特有的代码（SPIR-V、MSL、
  HLSL）。它处于技术栈的不同层次 —— wgpu 是运行时 + 着色器 DSL，而
  cuda-oxide 是一个编译 Rust 本身的 `rustc` codegen backend。

## 参与项目

如果你正在构建一个 Rust + GPU 项目 —— 或者正在评估上述哪一个项目适合你的
需求 —— 我们很乐意与你交流心得。欢迎加入
[Discord](https://discord.gg/ZUEr4AhH5C) 社区提出疑问、参与设计讨论、获取
公告，也可以通过
[GitHub Discussions](https://github.com/NVlabs/cuda-oxide/discussions)
联系团队，或在仓库中提交 issue。
