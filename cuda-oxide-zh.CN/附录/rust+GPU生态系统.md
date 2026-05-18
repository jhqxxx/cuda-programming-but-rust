# Rust + GPU 生态系统

cuda-oxide 是多个致力于从 Rust 访问 GPU 计算的活跃项目之一。每个项目解决问题的不同侧面——图形着色器、隐式卸载、CUDA 编程、安全的驱动绑定——整个领域正在快速发展。本附录简要介绍 cuda-oxide 相对于其他项目的位置，以及这些项目之间的关系。

## cuda-oxide 的定位

|项目|方法|目标|范围|
|:---|:---|:---|:---|
|**cuda-oxide**|`rustc` 代码生成后端|NVIDIA PTX/SASS|安全 Rust 中的 CUDA 编程模型|
|**Rust-GPU**|`rustc` → SPIR-V|Vulkan/Metal/DX|通过 SPIR-V 进行图形着色器和计算|
|**rust-cuda**|`rustc` → NVVM IR|NVIDIA PTX|NVIDIA GPU 上的 Rust 语言模型|
|**CubeCL**|嵌入式 DSL + JIT 运行时|CUDA/ROCm/WGPU|从 Rust DSL 生成跨厂商计算kernel|
|**std::offload**|`rustc` + LLVM offload|NVIDIA/AMD/Intel|CPU 代码的隐式卸载|
|**cudarc**|安全的 CUDA 驱动绑定|NVIDIA|CUDA 驱动的主机端绑定|
|**wgpu**|WebGPU API + WGSL/Naga|跨平台|通过着色器语言进行可移植计算|

“范围”列捕捉的是每个项目优化的**设计中心**，而非功能上限。其中几个项目在边缘处有所重叠，单个应用程序可能会使用多个项目。

## cuda-oxide 和 rust-cuda

最接近的邻居——也是最常与 cuda-oxide 混淆的项目——是 `rust-cuda`。两个项目都使用 `rustc` 代码生成后端来针对 NVIDIA GPU，从远处看它们似乎可以互换。仔细观察，我们发现两者的设计中心指向不同的方向：

- **rust-cuda** 的重点是**将 Rust 带到 NVIDIA GPU**：Rust 的人机工程学特性如 `async`/`.await`、在设备上运行的部分标准库，以及抽象掉 CUDA 概念的 Rust 优先编程模型。
- **cuda-oxide** 的重点是**将 CUDA 带入 Rust**：kernel编写、设备内联函数、SIMT 执行模型，以及用安全 Rust 原生表达的 CUDA 编程模型——更接近于用 C++ 编写 `__global__` 函数的精神，而不是编写一个恰好运行在 GPU 上的泛型 Rust 函数。

两种方向都很有价值，我们相信这两个项目是互补的。随着两个项目的成熟，我们一直在与 rust-cuda 的维护者合作，并期望继续这样做。

## 其他邻居

- **Rust-GPU** 通过 SPIR-V 面向图形导向的 GPU 编程，可覆盖 Vulkan、Metal 和 DirectX。计算着色器得到支持，但设计中心是图形；如果你需要跨厂商可移植性或着色器互操作，Rust-GPU 是正确的工具。
- **CubeCL** 是一个嵌入式 DSL：你用 `#[cube]` 注解一个 Rust 函数，CubeCL 运行时按需 JIT 编译为 CUDA、ROCm/HIP 或 WGSL。它不是 `rustc` 后端——`#[cube]` 过程宏将函数体重写为 CubeCL IR，然后运行时将其降级到主机 GPU 暴露的任何后端。其设计中心是**跨厂商可移植性**（单个kernel可在 NVIDIA、AMD 和 WGPU 上运行），而非完整的 Rust 语言覆盖；作为交换，CubeCL 放弃了在kernel中使用任意 Rust 构造的能力，并在一个有意识地受限的 DSL 表面上工作。CubeCL 是 [Burn](https://github.com/tracel-ai/burn) ML 框架背后的计算后端，这也是其主要展示应用。cuda-oxide 和 CubeCL 在很大程度上是互补的：当你需要一个kernel通过受控 DSL 跨 GPU 厂商运行时使用 CubeCL；当你需要在 NVIDIA 硬件上针对完整的 CUDA 编程模型编写地道的安全 Rust 时使用 cuda-oxide。
- **std::offload** 是一个 Rust 语言特性（当前为 nightly），使用 LLVM 的卸载运行时将 CPU 循环隐式移动到加速器。不同的编程模型：用户编写 CPU 风格的代码，编译器/运行时处理卸载。cuda-oxide 是显式的；offload 是隐式的。
- **cudarc** 为 CUDA 驱动程序 API 提供安全的 Rust 绑定。它是一个主机端库，用于启动在其他地方（通常是 PTX 或 CUDA C++）编写的kernel。cuda-oxide 自带其主机端绑定层（`cuda-core`），与 `cuda-host` 的启动宏紧密集成，但 cuda-oxide 生成的 PTX 是可移植的——如果你希望从一个不依赖 cuda-oxide 栈其余部分的项目中启动 cuda-oxide kernel，cudarc 是一个不错的选择。
- **wgpu** 是 WebGPU API 的 Rust 实现——一个在 Vulkan、Metal 和 DirectX 12（以及浏览器中的 WebGPU）之上的运行时抽象。计算和图形着色器使用 WGSL（WebGPU 着色语言）编写，并由 Naga 翻译为后端特定代码（SPIR-V、MSL、HLSL）。不同层次的栈——wgpu 是一个运行时 + 着色器 DSL，cuda-oxide 是一个编译 Rust 本身的 `rustc` 代码生成后端。

## 参与项目

如果你正在构建一个 Rust + GPU 项目——或评估上述哪个项目适合你的需求——我们很乐意交流心得。联系 cuda-oxide 团队的最佳方式是通过 [GitHub Discussions](https://github.com/NVlabs/cuda-oxide/discussions) 或在仓库中提交 issue。
