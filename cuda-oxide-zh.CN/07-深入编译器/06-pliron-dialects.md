# Pliron dialect

cuda-oxide 并非通过一次宏大的转换将 Rust 降级为 PTX。它使用了三种自定义的 pliron dialect，每种dialect对不同的抽象层次进行建模。本章将详细介绍这三种dialect —— 它们的类型、它们的操作，以及它们如何组合在一起形成编译流水线。

如果你还没有阅读过 [Pliron —— Pliron IR（类 MLIR）](pliron.md) 一章，现在是阅读的好时机。其中的概念（操作、类型、属性、区域、`Ptr<T>`、定值-使用链）是本章所有内容的构建块。

---

## 三种dialect概览

| dialect    | 目的           | 层次        |
| :--- | :-------- | :----- |
| **dialect-mir** | 对 Rust MIR 语义进行建模       | 最高层 —— Rust 类型、元组、枚举、切片、检查算术                   |
| **dialect-llvm**| 对 LLVM IR 进行建模            | 中间层 —— 扁平类型、GEP、支持 PHI 的控制流                        |
| **dialect-nvvm**| 对 NVIDIA GPU 内联函数进行建模 | 正交 —— 线程索引、线程束、TMA、WGMMA、tcgen05                    |

`dialect-nvvm` 是“正交”的，而不是栈中的一个层，因为它的操作与 `dialect-llvm` 操作**并存**于同一个函数体中，而不是位于它们之下。线程束洗牌和整数加法可以共存于同一个函数体中。

数据流经流水线的过程如下：

```text
dialect-mir ──(mem2reg)──▶ dialect-mir (SSA) ──(DialectConversion)──▶ dialect-llvm + dialect-nvvm ops ──(export.rs)──▶ 文本 LLVM IR ──(llc)──▶ PTX
```

每个箭头都是一个定义良好的转换。前两个发生在 pliron 内部；最后一个是 LLVM 的 NVPTX 后端做它最擅长的事情。

---

## dialect-mir —— Rust 层

`dialect-mir` 将 Rust 的类型系统和控制流语义保留为 pliron 操作。这是有意为之：我们希望在将 Rust 概念（元组、枚举、检查算术、地址空间）展平到 LLVM 的类型系统之前，先对它们进行推理。

### 类型

该dialect定义了七种自定义类型，它们镜像了 Rust 的复合类型：

| 类型     | 示例          | 描述          |
| :------- | :----------- | :---------- |
| `mir.tuple`          | `mir.tuple<i32, f32, i64>`                                 | 异构元组                                                       |
| `mir.ptr`            | `mir.ptr<f32, mutable, addrspace: 1>`                      | 带 GPU 地址空间的指针                                           |
| `mir.array`          | `mir.array<f32, 256>`                                      | 固定大小数组                                                   |
| `mir.struct`         | `mir.struct<"Point", [f32, f32]>`                          | 带布局信息的命名结构体                                         |
| `mir.slice`          | `mir.slice<f32, addrspace: 1>`                             | 胖指针（指针 + 长度）                                           |
| `mir.disjoint_slice` | `mir.disjoint_slice<f32>`                                  | 安全检查的切片 —— 每个线程访问唯一元素                           |
| `mir.enum`           | `mir.enum<"Option_i32", [("None", []), ("Some", [i32])]>`  | 带判别式和变体负载的 Rust 枚举                                  |

`mir.ptr` 和 `mir.slice` 上的地址空间追踪数据在 GPU 内存层次结构中的位置：

| 地址空间 | 含义                                    |
| :------- | :-------------------------------------- |
| 0        | 通用（运行时解析）                      |
| 1        | 全局（设备 DRAM）                        |
| 3        | 共享（每块 SRAM）                        |
| 4        | 常量（只读缓存）                        |
| 5        | 局部（每线程栈，溢出到 DRAM）            |
| 6        | 张量内存（Blackwell TMEM）              |

### 操作

`dialect-mir` 在 11 个类别中定义了 54 个操作：

| 类别  | 示例      | 数量 |
| :--------- | :------- | ---: |
| 函数       | `mir.func`    |    1 |
| 控制流     | `mir.goto`, `mir.cond_br`, `mir.return`, `mir.assert`, `mir.unreachable`      |    5 |
| 常量       | `mir.constant`, `mir.float_constant`, `mir.undef`        |    3 |
| 内存       | `mir.alloca`, `mir.load`, `mir.store`, `mir.assign`, `mir.ref`, `mir.ptr_offset`, `mir.shared_alloc`, `mir.global_alloc`, `mir.extern_shared`         |    9 |
| 算术       | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, `mir.checked_add`, `mir.checked_sub`, `mir.checked_mul`, `mir.neg`, `mir.not`, `mir.shr`, `mir.shl`, `mir.bitand`, `mir.bitor`, `mir.bitxor` |   15 |
| 比较       | `mir.eq`, `mir.ne`, `mir.lt`, `mir.le`, `mir.gt`, `mir.ge`          |    6 |
| 聚合       | `mir.extract_field`, `mir.insert_field`, `mir.construct_struct`, `mir.construct_tuple`, `mir.construct_array`, `mir.extract_array_element`, `mir.field_addr`, `mir.array_element_addr`               |    8 |
| 枚举       | `mir.get_discriminant`, `mir.construct_enum`, `mir.enum_payload`       |    3 |
| 转换       | `mir.cast`             |    1 |
| 存储       | `mir.storage_live`, `mir.storage_dead`   |    2 |
| 调用       | `mir.call`       |    1 |

操作数量很多，但它们可以自然地分组。如果你了解 Rust MIR（或者阅读过 [rustc_public 章节](rustc-public.md)），每个操作都直接映射到一个 MIR 概念。

### IR 示例

以下是一些实际 `dialect-mir` 操作的示例。为了可读性进行了简化 —— 实际打印形式包含更多元数据。

**检查加法**（Rust：`let sum = a + b`，其中 `a, b: i32`）：

```text
// mir.checked_add 返回一个元组 (result, overflow_flag)
%checked = mir.checked_add %a, %b : i32
%sum     = mir.extract_field %checked, 0 : mir.tuple<i32, i1>
%overflowed = mir.extract_field %checked, 1 : mir.tuple<i32, i1>
mir.assert %overflowed == false, "attempt to add with overflow" -> bb1
```

**结构体构造和字段访问**（Rust：`point.x`）：

```text
%point = mir.construct_struct %x, %y : mir.struct<"Point", [f32, f32]>
%x_val = mir.extract_field %point, 0 : mir.struct<"Point", [f32, f32]>
```

**共享内存分配**（GPU 特定部分）：

```text
%shmem = mir.shared_alloc : mir.ptr<f32, mutable, addrspace: 3>
mir.store %value, %shmem : f32
```

### 验证

每个 MIR 操作在构造时都会验证类型一致性。这可以在导入错误有机会传播到降级流水线并在三个 pass 之后表现为晦涩的 LLVM 错误之前，及早捕获它们。

检查内容的示例：

- `mir.add` 验证两个操作数具有相同的类型。
- `mir.cond_br` 验证条件是 `i1`（布尔值）。
- `mir.extract_field` 验证字段索引在范围内，并且结果类型与字段类型匹配。
- `mir.store` 验证值类型与指针的指向类型匹配。

`DisjointSlice` 的安全保证（“一个线程，一个元素”）通过 `ThreadIndex` 在类型系统层面强制执行 —— 只有硬件派生的线程索引可以访问该切片。没有单独的编译器 pass 用于不相交访问验证；安全性来自 Rust 类型系统和 `cuda-device` 的 API 设计。

---

## dialect-llvm —— LLVM 层

`dialect-llvm` 将 LLVM IR 建模为 pliron 操作。它提供了与文本 `.ll` 文件几乎 1:1 的映射 —— 每条 LLVM 指令都有一个对应的 pliron 操作，并且类型直接映射到 LLVM 的类型系统。

### 类型

| 类型          | 示例      | 描述          |
| :------------ | :------- | :---------- |
| 整数          | `i1`, `i8`, `i16`, `i32`, `i64`, `i128`   | Pliron 内置，直接使用                                 |
| 浮点数        | `half`, `float`, `double`                 | Pliron 内置（`FP16Type`, `FP32Type`, `FP64Type`）    |
| `llvm.ptr`    | `ptr addrspace(1)`                        | 带可选地址空间的不透明指针                             |
| `llvm.struct` | `{ i32, float }` 或 `%MyStruct`           | 命名或匿名，可能不透明                                 |
| `llvm.array`  | `[256 x float]`                           | 固定大小数组                                           |
| `llvm.vector` | `<4 x float>`                             | SIMD 向量                                              |
| `llvm.func`   | `(i32, ptr) -> void`                      | 函数签名                                               |
| `llvm.void`   | `void`                                    | 单元类型                                               |

注意没有 Rust 特定类型。当代码到达 `dialect-llvm` 时，元组变成了结构体，枚举变成了判别式索引的结构体，切片变成了指针-长度对。降级 pass（在[降级流水线](lowering-pipeline.md)中介绍）处理所有这些展平操作。

### 操作

该dialect定义了 62 个操作：

| 类别       | 示例         | 数量 |
| :--------- | :-------------- | ---: |
| 算术       | `add`, `sub`, `mul`, `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`, ...          |   19 |
| 转换       | `zext`, `sext`, `trunc`, `fpext`, `fptrunc`, `sitofp`, `uitofp`, `fptosi`, `fptoui`, `ptrtoint`, `inttoptr`, `addrspacecast`, `bitcast`  |   13 |
| 控制流     | `br`, `cond_br`, `switch`, `return`, `unreachable`  |    5 |
| 内存       | `load`, `store`, `alloca`, `gep`     |    4 |
| 原子       | `atomic_load`, `atomic_store`, `atomicrmw`, `cmpxchg`, `fence`    |    5 |
| 比较       | `icmp`, `fcmp`      |    2 |
| 聚合       | `extract_value`, `insert_value`, `extractelement`   |    3 |
| 调用       | `call`, `call_intrinsic`           |    2 |
| 内联汇编   | `inline_asm`, `inline_asm_multi`         |    2 |
| 常量       | `constant`, `zero`, `undef`        |    3 |
| 符号       | `func`, `global`, `addressof`         |    3 |
| 选择       | `select`         |    1 |

如果你以前阅读过 LLVM IR，这里没有什么会让你惊讶。操作名称故意与 LLVM 中的对应名称相同，在 IR 中带有 `llvm.` 前缀。

### 导出引擎

dialect-llvm 的皇冠上的明珠是 `export.rs` —— 该模块将 pliron IR 模块转换为有效的文本 LLVM IR。这不仅仅是“打印每个操作”；在导出期间会发生几个重要的转换：

**块参数变为 PHI 节点。** Pliron IR（类 MLIR）将合并点建模为块参数 —— 基本块之间类似函数调用的调用约定。LLVM IR 改用 PHI 节点。导出器从分支操作数构建前驱映射，并在每个非入口块的顶部发出 `phi` 指令。

**值命名。** 一个预 pass 为每个值分配顺序的 SSA 名称（`%v0`, `%v1`, ...）。常量被特殊处理：`llvm.constant` 的结果被映射到它们的字面值（而不是 `%vN` 名称），这样 PHI 节点就可以引用稍后出现在输出中的块中的常量。

**NVVM 内联函数名称转换。** Pliron 标识符使用下划线；LLVM 内联函数使用点。导出器将以 `llvm_` 开头的所有名称中的下划线替换为点：`llvm_nvvm_read_ptx_sreg_tid_x` 变为 `llvm.nvvm.read.ptx.sreg.tid.x`。这是一个机械转换，而不是查表。

**Convergent 属性标记。** 屏障、洗牌和投票内联函数必须标记为 `convergent`，以防止 LLVM 将它们提升出控制流。导出器通过对（点形式的）名称进行前缀模式匹配来识别这些内联函数，并在其调用点附加 `#0`，在模块级别发出 `attributes #0 = { convergent }`。

**kernel元数据。** 标记为kernel的函数获得 `ptx_kernel` 调用约定和一个 `!nvvm.annotations` 元数据条目。

以下是一个简单的向量加法kernel导出后的 LLVM IR 示例：

```llvm
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

define ptx_kernel void @vecadd(ptr addrspace(1) %v0, i64 %v1,
                                ptr addrspace(1) %v2, i64 %v3,
                                ptr addrspace(1) %v4, i64 %v5) {
entry:
    %v6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
    %v7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
    %v8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
    %v9 = mul i32 %v8, %v7
    %v10 = add i32 %v9, %v6
    ; ... 边界检查、加载、加法、存储 ...
    ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @vecadd, !"kernel", i32 1}
attributes #0 = { convergent }
```

注意切片已被标量化：每个 Rust `&[f32]` 变成了一个 `ptr addrspace(1)` 和一个 `i64` 长度。这发生在降级 pass 中；当 `dialect-llvm` 看到它们时，它们已经是扁平化的参数了。

---

## dialect-nvvm —— GPU 层

`dialect-nvvm` 将 NVIDIA 的 GPU 内联函数包装为类型化的 pliron 操作。这些操作并不构成降级链中的一个“层级” —— 它们在 `dialect-mir` → `dialect-llvm` 降级 pass 期间被插入，并与 `dialect-llvm` 操作共存于同一个函数体中。在导出时，它们变成对 `@llvm.nvvm.*` 内联函数的 `call` 指令。

### 架构覆盖范围

该dialect按模块组织，每个模块针对一组 GPU 特性：

| 模块       | 描述      | 操作数 | 最低 SM | GPU 系列   |
| :--------- | :------- | ----: | :------ | :--------- |
| `thread`   | 线程/块索引、`barrier0`、线程栅栏  |    18 | 全部    | 所有 GPU   |
| `warp`     | 通道 ID、洗牌、投票、匹配     |    18 | 全部    | 所有 GPU   |
| `grid`     | 协作 `grid_sync`  |     1 | sm_70   | Volta+     |
| `debug`    | 时钟、陷阱、断点、`vprintf` |     6 | 全部    | 所有 GPU   |
| `atomic`   | 原子加载/存储/RMW/cmpxchg  |     4 | sm_70   | Volta+     |
| `cluster`  | 线程块簇 + DSMEM   |    11 | sm_90   | Hopper+    |
| `mbarrier` | 异步屏障 + 代理栅栏 + 纳秒睡眠  |    10 | sm_90   | Hopper+    |
| `tma`      | 张量内存加速器（批量 G2S/S2G）  |    15 | sm_90   | Hopper+    |
| `wgmma`    | 线程束组矩阵乘加      |     5 | sm_90   | Hopper+    |
| `stmatrix` | 共享内存矩阵存储 + bf16 转换      |     5 | sm_90   | Hopper+    |
| `tcgen05`  | 张量核心第 5 代 + TMEM    |    24 | sm_100  | Blackwell+ |
| `clc`      | 簇启动控制    |     6 | sm_100  | Blackwell+ |

总共有 123 个操作。大多数用户只会遇到前三个模块（线程索引、线程束洗牌、屏障）。其余用于高级 GPU 编程 —— TMA、矩阵加速器和 Blackwell 的张量内存 —— 在[高级 GPU 特性](../advanced/tensor-memory-accelerator.md)章节中介绍。

### 从 Rust 到 PTX：内联函数的旅程

每个 NVVM 操作映射到三个层次的命名：

| Pliron 操作   | LLVM 内联函数     | PTX 指令    |
| :---- | :---------------- | :---------------- |
| `ReadPtxSregTidXOp`             | `llvm.nvvm.read.ptx.sreg.tid.x`                       | `mov.u32 %r1, %tid.x`                       |
| `Barrier0Op`                    | `llvm.nvvm.barrier0`                                  | `bar.sync 0`                                |
| `ShflSyncBflyI32Op`             | `llvm.nvvm.shfl.sync.bfly.i32`                        | `shfl.sync.bfly.b32`                        |
| `CpAsyncBulkTensorG2sTile2dOp`  | `llvm.nvvm.cp.async.bulk.tensor.2d.tile.g2s.im2col.*` | `cp.async.bulk.tensor.2d.tile.g2s ...`      |

第一列是 `dialect-nvvm` 中的 Rust 结构体名称。第二列是 `export.rs` 发出的内容（在下划线到点的转换之后）。第三列是 `llc` 产生的内容。你永远不需要手动编写任何这些 —— 当 `mir-lower` 看到对 `cuda-device` 内联函数（如 `thread::index_x()` 或 `warp::shfl_sync_bfly()`）的调用时，它们会被自动生成。

### 验证策略

NVVM 操作使用最小的结构验证：每个操作检查其操作数计数和结果计数，少数操作检查结果类型（线程索引操作要求 `i32` 结果；`tcgen05` 加载检查其 32 寄存器和 4 寄存器变体的确切结果计数）。

这是有意为之。NVVM 操作由 `mir-lower` 机器生成 —— 它们永远不会由用户手写。LLVM 的 NVPTX 后端在下游提供全面的类型验证。为每个 NVVM 操作添加完整的类型检查会使dialect代码量翻倍，而没有任何实际的好处。

> **注意**：
> GPU 架构要求（sm_70, sm_90, sm_100）已在文档中记录，但未在 pliron 级别强制执行。架构验证发生在后面，当 `llc` 以特定的 `-mcpu=sm_XX` 标志被调用时。如果你在面向 Volta 时使用了 Hopper 内联函数，`llc` 会大声告诉你。

---

## dialect如何交互

以下是单个 Rust 操作在穿过所有三个抽象层次时的生命周期：

```text
Rust 源代码：   let sum = a + b;        // a, b: f32

dialect-mir：   %sum = mir.add %a, %b : f32
                ↓  (DialectConversion)
dialect-llvm：  %v5 = fadd float %v3, %v4
                ↓  (export.rs)
LLVM IR：       %v5 = fadd float %v3, %v4
                ↓  (llc --mcpu=sm_80)
PTX：           add.f32 %f3, %f1, %f2;
```

`dialect-mir` → `dialect-llvm` 这一步是有趣的工作发生的地方：对 `f32` 的 `mir.add` 变成 `fadd`（浮点加法），而对 `i32` 的 `mir.add` 变成 `add`（整数加法）。像 `mir.checked_add` 这样的检查操作会展开为 `llvm.add`、一个用于溢出标志的常量 `i1 false`，以及一个 `insertvalue` 到结构体中 —— GPU 路径省略了溢出检测（因为 GPU 整数算术会回绕）。降级 pass 处理所有这些翻译。

对于 GPU 特定的操作，`dialect-nvvm` 参与进来：

```text
Rust 源代码：   let tid = thread::threadIdx_x();

dialect-mir：   %tid = mir.call @cuda_oxide_device_<hash>_thread_index_x()
                ↓  (DialectConversion, 识别内联函数)
dialect-nvvm：  %v2 = nvvm.read_ptx_sreg_tid_x : i32
                ↓  (export.rs)
LLVM IR：       %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
                ↓  (llc)
PTX：           mov.u32 %r1, %tid.x;
```

降级 pass 通过其完全限定名（FQDN）识别对 `cuda_device` 内联函数的调用，并将它们替换为相应的 `dialect-nvvm` 操作。不需要通用的“函数调用”机制 —— 内联函数变成了直接的硬件指令。

### 完整视图

综合来看，一个编译后的kernel体包含 `dialect-llvm` 和 `dialect-nvvm` 操作的混合：

```text
llvm.func @vecadd(...) {
  entry:
    %tid    = nvvm.read_ptx_sreg_tid_x     // NVVM: 线程索引
    %ntid   = nvvm.read_ptx_sreg_ntid_x    // NVVM: 块大小
    %ctaid  = nvvm.read_ptx_sreg_ctaid_x   // NVVM: 块索引
    %offset = llvm.mul %ctaid, %ntid        // LLVM: 整数运算
    %idx    = llvm.add %offset, %tid        // LLVM: 整数运算
    %cmp    = llvm.icmp slt %idx, %len      // LLVM: 边界检查
    llvm.cond_br %cmp, bb1, bb2             // LLVM: 分支
  bb1:
    %p_a    = llvm.gep %a, %idx             // LLVM: 指针算术
    %val_a  = llvm.load %p_a                // LLVM: 内存访问
    %p_b    = llvm.gep %b, %idx
    %val_b  = llvm.load %p_b
    %sum    = llvm.fadd %val_a, %val_b      // LLVM: 浮点加法
    %p_c    = llvm.gep %c, %idx
    llvm.store %sum, %p_c
    llvm.br bb2
  bb2:
    llvm.return void
}
```

顶部的 `dialect-nvvm` 操作计算全局线程索引。其他一切都是标准的 `dialect-llvm` —— 加载、存储、算术、分支。导出引擎将所有这些序列化为一个 `.ll` 文件，`llc` 将其编译为 PTX。

---

关于这些dialect如何通过降级 pass 连接，请参阅[降级流水线](./lowering-pipeline.md)。

