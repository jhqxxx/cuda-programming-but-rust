# CUDA-Oxide 示例学习指南

`rustc-codegen-cuda` 的 `examples/` 目录包含 **100+ 个示例**，涵盖了从"Hello World"级别的向量加法到 Hopper/Blackwell 架构的张量核心等高级特性。下面是针对新手的系统化学习路径。

---

## 🗺️ 学习路径概览

将示例按学习顺序分为几个阶段：

```
第一阶段：基础入门  →  第二阶段：核心概念  →  第三阶段：进阶特性  →  第四阶段：高级主题
```

---

## 第一阶段：基础入门（新手起点）

这些示例展示了最基本的 CUDA 编程模式，**建议全部跑一遍**。

| 示例 | 说明 | 学习要点 |
|------|------|----------|
| **`vecadd`** | 向量加法 — CUDA 的 "Hello World" | `#[kernel]`、`DisjointSlice`、`c.get_mut_indexed()`、线程索引 |
| **`generic`** | 泛型 kernel — `scale<T>`、`add<T>` | 泛型 kernel 的定义与单态化、trait 约束 |
| **`printf`** | 设备端 `printf` | GPU 上的打印调试、FFI 调用 |
| **`constant_memory_simple`** | 常量内存基础 | `#[constant_memory]`、只读缓存 |
| **`helper_fn`** | 辅助函数 | 从 kernel 调用普通设备函数 |

**运行方式**：
```bash
cargo oxide run vecadd
cargo oxide run generic
# 查看完整编译流水线
cargo oxide pipeline vecadd
```

---

## 第二阶段：核心 GPU 概念

理解 GPU 编程的核心机制。

### 🔹 内存层次

| 示例 | 说明 | 关键概念 |
|------|------|----------|
| **`sharedmem`** | 共享内存（片上快速内存） | `SharedArray<T, N>`、`thread::sync_threads()`、线程协作 |
| **`dynamic_smem`** | 动态共享内存分配 | 运行时确定共享内存大小 |
| **`constant_memory`** | 常量内存（含复杂系数） | `#[constant_memory]`、`#[coefficients]` |
| **`constant_memory_coeffs`** | 常量系数 | 编译时初始化的常量数据 |
| **`device_global`** | 设备全局变量 | 跨 kernel 调用的设备端全局状态 |

### 🔹 同步与协作

| 示例 | 说明 | 关键概念 |
|------|------|----------|
| **`barrier`** | 线程屏障 | `__syncthreads`、块内同步 |
| **`atomics`** | 原子操作（20 个测试用例） | `AtomicU32/I32/U64`、内存序、fence 拆分、跨 CTA 原子操作 |
| **`atomic_f16`** | f16 原子正确性检查 | 半精度浮点原子操作 |
| **`warp_reduce`** | Warp 级归约 | Warp 内协作、shuffle 指令 |
| **`coop_groups_demo`** | 协作组（Cooperative Groups） | 细粒度线程分组 |

### 🔹 内存管理与指针

| 示例 | 说明 | 关键概念 |
|------|------|----------|
| **`abi_hmm`** | 异构内存管理（HMM）+ 统一 ABI | GPU 直接访问主机内存、结构体内存布局匹配、闭包捕获 |
| **`device_closures`** | 设备端闭包 | `move` 闭包（传值）与非 `move` 闭包（引用+HMM） |
| **`copy_nonoverlapping`** | 内存复制 | `ptr::copy_nonoverlapping` 在设备上的行为 |
| **`ptr_offset_from`** | 指针偏移计算 | 设备端指针运算 |
| **`transmute_scalar_struct`** | 标量结构体 transmute | 类型转换、内存布局兼容性 |

---

## 第三阶段：进阶 GPU 编程

### 🔹 数学运算

| 示例 | 说明 |
|------|------|
| **`libdevice_math`** | 使用 NVIDIA libdevice 数学库（`sin`、`cos`、`exp` 等） |
| **`libm_math`** | 使用 Rust `libm` 数学库 |
| **`math_atan` / `math_tan` / `math_hyperbolic`** | 具体的数学函数测试 |
| **`asin_acos_smoke` / `cbrt_smoke` / `fmaxmin_smoke`** | 数学函数冒烟测试 |
| **`extern_libdevice`** | FFI 调用 libdevice 函数 |
| **`dotprod`** | 点积运算 |
| **`complex_mul`** | 复数乘法 |
| **`redux_sum` / `redux_minmax`** | 归约求和、归约求最值 |
| **`compound`** | 复合运算 |
| **`carrying_mul_add`** | 带进位的乘加运算 |
| **`primitive_stress`** | 基础类型运算压力测试 |
| **`numeric_stress`** | 数值运算压力测试 |

### 🔹 类型系统与 ABI

| 示例 | 说明 |
|------|------|
| **`cuda_module_contract`** | 类型化 launch ABI 的参数编组 |
| **`manual_launch_generic`** | 底层泛型 launch API |
| **`manual_launch_libdevice`** | 底层 libdevice launch API |
| **`enum_abi_multi_payload`** | 多 payload 枚举的 ABI |
| **`enum_array_match`** | 枚举数组 match |
| **`tuple_return`** | 元组返回值 |
| **`union_aggregate`** | 联合体聚合类型 |
| **`repr_u32_enum_stride`** | `#[repr(u32)]` 枚举的内存布局 |
| **`struct_field_layout`** | 结构体字段布局 |
| **`option_ref_transmute` / `option_ref_unwrap_or`** | `Option<&T>` 的 transmute/unwrap |
| **`function_item_call`** | 函数项调用 |
| **`bool_phi_cmp`** | 布尔 PHI 节点比较 |

### 🔹 异步与流

| 示例 | 说明 |
|------|------|
| **`async_vecadd`** | 异步向量加法 — 使用 `cuda-async` 的 CUDA 流 |
| **`async_mlp`** | 多层感知机 — 使用异步流的神经网络 |
| **`future_apis`** | Future API — 异步编程模式 |

### 🔹 跨 crate 与模块化

| 示例 | 说明 |
|------|------|
| **`cross_crate_kernel`** | 在库 crate 中定义 kernel |
| **`cross_crate_embedded`** | 跨 crate 嵌入 PTX |
| **`cuda_module_in_lib`** | 在库中定义 `cuda_module` |
| **`cuda_module_nested`** | 嵌套模块定义 |
| **`export_name_policy`** | 导出命名策略 |

---

## 第四阶段：高级 / 架构特定特性

### 🔹 Hopper (sm_90) 架构

| 示例 | 说明 |
|------|------|
| **`wgmma`** | Warpgroup MMA — Hopper 张量核心矩阵运算 |
| **`tma_copy`** | 张量内存加速器（TMA）— 异步 2D 复制 |
| **`cluster`** | 线程块集群（Cluster） |
| **`clc`** | Cluster Launch Control |
| **`mcast_barrier_test`** | 多播屏障测试 |
| **`cp_async_small` / `cp_async_zfill`** | 异步复制（小规模/填充模式） |

### 🔹 Blackwell (sm_100+) 架构

| 示例 | 说明 |
|------|------|
| **`tcgen05`** | 第五代张量核心基础 |
| **`tcgen05_matmul`** | TC Gen5 矩阵乘法 |
| **`tma_multicast`** | TMA 多播 — CTA 间数据共享 |

### 🔹 矩阵乘法（GEMM）

| 示例 | 说明 |
|------|------|
| **`gemm`** | 基础 GEMM 实现 |
| **`gemm_sol`** / **`gemm_sol_final`** | 优化后的 GEMM 实现 |
| **`tiled_gemm`** | 分块 GEMM |
| **`cutile_inter_kernel`** | cuTile 跨 kernel GEMM |

### 🔹 FFI 与互操作

| 示例 | 说明 |
|------|------|
| **`cpp_consumes_rust_device`** | C++ 主机调用 Rust 生成的 PTX |
| **`device_ffi_test`** | Rust kernel 调用外部 C++/CCCL 函数（LTOIR） |
| **`mathdx_ffi_test`** | cuFFTDx + cuBLASDx FFI |
| **`inline_ptx`** | 内联 PTX 汇编 |

### 🔹 调试与测试

| 示例 | 说明 |
|------|------|
| **`debug`** | 调试支持 |
| **`vecadd` 的 `cargo oxide pipeline`** | 查看完整编译流水线 |
| **`error_*` 系列** | 预期失败的诊断/回归测试（见下文） |

---

## ❌ `error_*` 系列示例

这些示例**预期会编译失败**，分为两类：

| 示例 | 类型 | 含义 |
|------|------|------|
| `error` | 诊断夹具 | `core::fmt` 从设备代码可达 |
| `error_drop_glue` | 支持缺口 | 有副作用的析构函数 |
| `error_heap_alloc` | 诊断夹具 | `__rust_alloc` 从设备代码可达 |
| `error_missing_device_attr` | 诊断夹具 | 缺少 `#[device]` 属性 |
| `error_set_discriminant_niche` | 支持缺口 | Niche 编码 |
| `error_set_discriminant_uninhabited` | 诊断夹具 | 无效枚举变体 |
| `error_static_initializer_provenance` | 支持缺口 | 设备全局指针重定位 |
| `error_wgmma_mma_unimplemented` | 支持缺口 | WGMMA MMA 未实现 |

**新手可以跳过这些**，它们主要用于开发者回归测试。

---

## 🎯 新手推荐学习路线

### 第 1 步：跑通基础
```bash
# 从最简单的开始
cargo oxide run vecadd
cargo oxide run generic
cargo oxide run constant_memory_simple
cargo oxide run helper_fn
cargo oxide run printf
```

### 第 2 步：理解 GPU 内存层次
```bash
cargo oxide run sharedmem
cargo oxide run constant_memory
cargo oxide run atomics
cargo oxide run abi_hmm
```

### 第 3 步：深入 GPU 编程
```bash
cargo oxide run barrier
cargo oxide run warp_reduce
cargo oxide run libdevice_math
cargo oxide run dotprod
cargo oxide run async_vecadd
```

### 第 4 步：探索高级特性
```bash
# 需要 Hopper GPU (sm_90) 或更高
cargo oxide run wgmma
cargo oxide run tma_copy
cargo oxide run cluster
cargo oxide run gemm
```

### 第 5 步：查看编译流水线
```bash
# 了解 Rust MIR → PTX 的完整编译过程
cargo oxide pipeline vecadd
```

---

## 💡 重要提示

1. **每个示例都是一个独立的 Cargo crate**，有完整的 Cargo.toml 和 `src/` 目录。
2. **大多数示例有详细的 README.md**，阅读它们是理解示例的最佳方式。
3. **使用 `cargo oxide run <name>` 运行**——它会自动设置正确的编译器标志。
4. **GPU 要求**：
   - 大多数基础示例需要任意 CUDA 显卡
   - `wgmma`、`tma_copy`、`cluster` 需要 **Hopper (sm_90)** 或更新
   - `tcgen05` 需要 **Blackwell (sm_100+)** 
5. **推荐先读 `vecadd` 和 `generic` 的 README**——它们详细解释了核心概念。
6. 如果想了解某个具体类型（如 `DisjointSlice`）的工作原理，可以查看 `cuda-device` crate 的源码。