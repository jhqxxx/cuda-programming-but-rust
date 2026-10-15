# Pliron 方言

cuda-oxide 并不是通过一次英勇的变换就把 Rust 下降为 PTX。它在下行过程中横跨
三个 pliron 方言，每个方言建模不同层次的抽象：两个在本地定义（`dialect-mir`、
`dialect-nvvm`），以及由上游 `pliron-llvm` crate 提供的 LLVM 方言。本章逐一
介绍这三者 —— 它们的类型、它们的操作，以及它们如何组合成编译流水线。

还有两个 pliron 方言存在于这棵代码树中，但不在该路径上，本章不予介绍：
`dialect-iket`，即 in-kernel 事件追踪的面向编译器形式；以及 `dialect-ptx`，
一种结构化的终端 PTX 方言，既可以直接构建，也可以从解析后的 PTX 源码投射而来。
每个 crate 的 README 就是参考。

如果你还没有读过 [Pliron -- Pliron IR (MLIR-like)](pliron.md) 这一章，现在正是
好时机。那里的概念（操作、类型、属性、region、`Ptr<T>`、def-use 链）是本页一切
内容的构建块。

---

## 三个方言一览

| 方言             | 用途                         | 层次                                                              |
| :--------------- | :--------------------------- | :---------------------------------------------------------------- |
| **dialect-mir**  | 建模 Rust MIR 语义           | 最高 —— Rust 类型、元组、枚举、切片、带检查的算术                  |
| **LLVM 方言**    | 建模 LLVM IR                 | 中间 —— 扁平类型、GEP、可容纳 PHI 的控制流                         |
| **dialect-nvvm** | 建模 NVIDIA GPU 内联函数     | 正交 —— 线程索引、warp、TMA、WGMMA、tcgen05                        |

LLVM 方言不是一个 cuda-oxide crate：它的建模（操作、类型、属性、op 接口）位于
上游的 `pliron-llvm` crate 中，cuda-oxide 将其作为依赖使用。

`dialect-nvvm` 之所以是"正交"而不是堆栈中的一层，是因为它的操作出现在 LLVM
方言操作*旁边*，而不是它们之下。一个 warp shuffle 和一个整数加法可以共存于
同一个函数体中。

数据如下流过整条流水线：

```text
dialect-mir ──(mem2reg)──▶ dialect-mir (SSA) ──(annotated unroll)──▶ dialect-mir
  ──(DialectConversion)──▶ LLVM dialect + dialect-nvvm ops ──(export)──▶ LLVM IR ──(llc)──▶ PTX
```

每个箭头都是对 pliron 或 LLVM IR 的一次明确定义的变换。最后一个箭头是 LLVM 的
NVPTX 后端在做它最擅长的事。

---

## dialect-mir —— Rust 层

`dialect-mir` 把 Rust 的类型系统和控制流语义保留为 pliron 操作。这是刻意的：
我们希望在把 Rust 概念（元组、枚举、带检查的算术、地址空间）扁平化为 LLVM 的
类型系统*之前*，先对它们进行推理。

### 类型

该方言定义了九个自定义类型，对应 Rust 自身的类型：

| 类型                 | 示例                                                      | 描述                                                          |
| :------------------- | :-------------------------------------------------------- | :------------------------------------------------------------ |
| `mir.tuple`          | `mir.tuple<i32, f32, i64>`                                | 异构元组                                                      |
| `mir.ptr`            | `mir.ptr<f32, mutable, addrspace: 1>`                     | 带 GPU 地址空间的指针                                         |
| `mir.array`          | `mir.array<f32, 256>`                                     | 固定大小数组                                                  |
| `mir.struct`         | `mir.struct<"Point", [f32, f32]>`                         | 带布局信息的具名结构体                                        |
| `mir.union`          | `mir.union<"Repr", [a, b], [i32, f32], 4, 4>`             | Rust union —— 每个字段都是同一段字节的视图                     |
| `mir.slice`          | `mir.slice<f32, addrspace: 1>`                            | 胖指针（ptr + 长度）                                          |
| `mir.disjoint_slice` | `mir.disjoint_slice<f32>`                                 | 携带类型化索引空间的带边界检查切片                             |
| `mir.enum`           | `mir.enum<"Option_i32", [("None", []), ("Some", [i32])]>` | 带判别值和变体载荷的 Rust 枚举                                 |
| `mir.fp16`           | `mir.fp16`                                                | IEEE 754 binary16，即 Rust 的 `f16`                           |

`mir.ptr` 和 `mir.slice` 上的地址空间跟踪数据位于 GPU 内存层次结构的何处：

| 地址空间 | 含义                                   |
| :------- | :------------------------------------- |
| 0        | 通用（运行时解析）                     |
| 1        | 全局（设备 DRAM）                      |
| 3        | 共享（每 block 的 SRAM）               |
| 4        | 常量（只读缓存）                       |
| 5        | 本地（每线程栈，溢出到 DRAM）          |
| 6        | 张量内存（Blackwell TMEM）             |

### 操作

`dialect-mir` 定义了 62 个操作，分为 12 个类别，每个类别对应
`cuda-oxide/crates/dialect-mir/src/ops/` 下的一个模块：

| 类别         | 示例                                                                                                                                                                                                | 数量 |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---: |
| 函数         | `mir.func`                                                                                                                                                                                          |     1 |
| 控制流       | `mir.goto`, `mir.cond_br`, `mir.return`, `mir.assert`, `mir.unreachable`, `mir.unroll_hint`                                                                                                         |     6 |
| 常量         | `mir.constant`, `mir.float_constant`, `mir.undef`                                                                                                                                                   |     3 |
| 内存         | `mir.alloca`, `mir.load`, `mir.store`, `mir.assign`, `mir.ref`, `mir.ptr_offset`, `mir.memcpy`, `mir.memmove`, `mir.shared_alloc`, `mir.global_alloc`, `mir.extern_shared`                          |    11 |
| 算术         | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, `mir.checked_add`, `mir.checked_sub`, `mir.checked_mul`, `mir.neg`, `mir.not`, `mir.shr`, `mir.shl`, `mir.bitand`, `mir.bitor`, `mir.bitxor` |    15 |
| 比较         | `mir.eq`, `mir.ne`, `mir.lt`, `mir.le`, `mir.gt`, `mir.ge`, `mir.cmp`                                                                                                                               |     7 |
| 聚合         | `mir.extract_field`, `mir.insert_field`, `mir.construct_struct`, `mir.construct_tuple`, `mir.construct_array`, `mir.construct_slice`, `mir.construct_disjoint_slice`, `mir.extract_array_element`, `mir.field_addr`, `mir.array_element_addr` |    10 |
| 枚举         | `mir.get_discriminant`, `mir.set_discriminant`, `mir.construct_enum`, `mir.enum_payload`                                                                                                            |     4 |
| 转换         | `mir.cast`                                                                                                                                                                                          |     1 |
| 存储         | `mir.storage_live`, `mir.storage_dead`                                                                                                                                                              |     2 |
| 调用         | `mir.call`                                                                                                                                                                                          |     1 |
| 调试         | `mir.dbg_value`                                                                                                                                                                                     |     1 |

操作很多，但它们归入自然的几组。如果你了解 Rust MIR（或者读过
[rustc_public 章节](rustc-public.md)），每个操作都直接对应一个 MIR 概念。

### IR 长什么样

以下是实践中几个 `dialect-mir` 操作的示例。为便于阅读做了简化 —— 实际的打印
形式包含更多元数据。

**带检查的加法**（Rust：`let sum = a + b`，其中 `a, b: i32`）：

```text
// mir.checked_add returns a tuple (result, overflow_flag)
%checked = mir.checked_add %a, %b : i32
%sum     = mir.extract_field %checked, 0 : mir.tuple<i32, i1>
%overflowed = mir.extract_field %checked, 1 : mir.tuple<i32, i1>
%ok = mir.not %overflowed : i1
mir.assert %ok
mir.goto bb1
```

当条件为真时，`mir.assert` 在其 block 内继续执行，否则陷入 trap。它与分支保持
分离，这样合并 block 时就不会抹掉这次检查。LLVM 下降会在断言处拆分该 block，
并添加一条显式分支，指向后续操作或一个 trap block。

**结构体构造与字段访问**（Rust：`point.x`）：

```text
%point = mir.construct_struct %x, %y : mir.struct<"Point", [f32, f32]>
%x_val = mir.extract_field %point, 0 : mir.struct<"Point", [f32, f32]>
```

**共享内存分配**（GPU 特有的部分）：

```text
%shmem = mir.shared_alloc : mir.ptr<f32, mutable, addrspace: 3>
mir.store %value, %shmem : f32
```

### 校验

每个 MIR 操作在构造时都会校验类型一致性。这能及早捕获导入 bug —— 在它们有机会
穿过下降流水线、并在三个 pass 之后以晦涩的 LLVM 错误浮现之前。

会检查的内容示例：

- `mir.add` 校验两个操作数具有相同的类型。
- `mir.cond_br` 校验条件为 `i1`（布尔值）。
- `mir.extract_field` 校验字段索引在边界内，且结果类型与字段的类型匹配。
- `mir.store` 校验值的类型与指针的指向类型匹配。

`DisjointSlice` 只接受匹配的 `ThreadIndex`，因此类型系统会检查设备侧的索引空间。
宿主启动几何信息通过 `PreparedLaunch<K>` 或不安全的裸启动义务来完成唯一性证明。
并不存在一个单独的分离访问编译器 pass；安全性来自这些 API。

---

## LLVM 方言 —— LLVM 层

LLVM 方言把 LLVM IR 建模为 pliron 操作。它提供到文本 `.ll` 文件的近乎 1:1 映射
—— 每条 LLVM 指令都有一个对应的 pliron 操作，类型也直接映射到 LLVM 的类型系统。
该方言本身（操作、类型、属性、op 接口）定义在上游的 `pliron-llvm` crate 中；
cuda-oxide 使用它，并通过轻量的 `llvm-export` crate 重新导出，后者还带有文本
`.ll` 导出器以及一些 pliron-llvm 未附带的 GPU 专用扩展（具名地址空间、fp16
位辅助函数）。

### 类型

| 类型          | 示例                                    | 描述                                                 |
| :------------ | :-------------------------------------- | :--------------------------------------------------- |
| 整数          | `i1`, `i8`, `i16`, `i32`, `i64`, `i128` | pliron 内建，直接使用                                |
| 浮点          | `half`, `float`, `double`               | pliron 内建（`FP16Type`、`FP32Type`、`FP64Type`）    |
| `llvm.ptr`    | `ptr addrspace(1)`                      | 带可选地址空间的不透明指针                           |
| `llvm.struct` | `{ i32, float }` 或 `%MyStruct`         | 具名或匿名，可以是不透明的                           |
| `llvm.array`  | `[256 x float]`                         | 固定大小数组                                         |
| `llvm.vector` | `<4 x float>`                           | SIMD 向量                                            |
| `llvm.func`   | `(i32, ptr) -> void`                    | 函数签名                                             |
| `llvm.void`   | `void`                                  | 单位类型                                             |

注意这里没有 Rust 特有的类型。当代码到达 LLVM 方言时，元组已经变成结构体，
枚举已经变成以判别值索引的结构体，切片已经变成指针-长度对。下降 pass（在
[下降流水线](lowering-pipeline.md) 中介绍）负责所有这些扁平化工作。

### 操作

在所锁定的 `pliron` 修订版上，该方言定义了 69 个操作：

| 类别         | 示例                                                                                                                                    | 数量 |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------- | ---: |
| 算术         | `add`, `sub`, `mul`, `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg`, ...                                                                |    19 |
| 转换         | `zext`, `sext`, `trunc`, `fpext`, `fptrunc`, `sitofp`, `uitofp`, `fptosi`, `fptoui`, `ptrtoint`, `inttoptr`, `addrspacecast`, `bitcast` |    13 |
| 控制流       | `br`, `cond_br`, `switch`, `return`, `unreachable`, `indirectbr`                                                                        |     6 |
| 内存         | `load`, `store`, `alloca`, `gep`                                                                                                        |     4 |
| 原子         | `atomic_load`, `atomic_store`, `atomicrmw`, `cmpxchg`, `fence`                                                                          |     5 |
| 比较         | `icmp`, `fcmp`                                                                                                                          |     2 |
| 聚合         | `extract_value`, `insert_value`, `extract_element`, `insert_element`, `shuffle_vector`                                                  |     5 |
| 调用         | `call`, `call_intrinsic`                                                                                                                |     2 |
| 内联汇编     | `inline_asm`                                                                                                                            |     1 |
| 常量         | `constant`, `zero`, `undef`, `poison`                                                                                                   |     4 |
| 符号         | `func`, `global`, `addressof`                                                                                                           |     3 |
| 选择         | `select`                                                                                                                                |     1 |
| 其他         | `freeze`, `va_arg`, `blockaddress`, `blocktag`                                                                                          |     4 |

`llvm-export` 在这些之上还添加了一个自己的操作 `llvm.dbg_value`，以及上文提到的
地址空间和 fp16 辅助函数。

由于该方言位于上游，这张表会随着 `pliron` 锁定的移动而移动，而不是随本仓库的
变化而变化；`pliron-llvm` 的 `src/ops.rs` 就是它正在计数的清单。

如果你以前读过 LLVM IR，这里不会让你感到意外。操作名有意与它们的 LLVM 对应物
同名，在 IR 中加上 `llvm.` 前缀。

### 导出引擎

`llvm-export` 的王冠上的明珠是它的导出模块（`cuda-oxide/crates/llvm-export/src/export/`）
—— 把 pliron IR 模块转换为合法文本 LLVM IR 的代码。这是 cuda-oxide 保留在本地的
部分：pliron-llvm 只能通过 `llvm-sys` 桥接发出真正的 `.ll`，而 cuda-oxide 避免
这样做。这不仅仅是"打印每个操作"；导出期间会发生若干非平凡的变换：

**block 参数变成 PHI 节点。** Pliron IR（类似 MLIR）把合并点建模为 block 参数
—— 基本块之间一种函数式调用约定。LLVM IR 则使用 PHI 节点。导出器从分支操作数
构建前驱映射，并在每个非入口 block 的顶部发出 `phi` 指令。

**值命名。** 一个预 pass 会为每个值分配顺序的 SSA 名（`%v0`、`%v1`、……）。
常量会被特殊处理：`llvm.constant` 的结果映射到其字面值（而不是 `%vN` 名），
这样 PHI 就能引用出现在输出中较后位置的 block 中的常量。

**NVVM 内联函数名转换。** Pliron 标识符使用下划线；LLVM 内联函数使用点号。
导出器转换所有以 `llvm_` 开头的名称，把下划线替换为点号：
`llvm_nvvm_read_ptx_sreg_tid_x` 变成 `llvm.nvvm.read.ptx.sreg.tid.x`。这是机械
变换，不是查找表。

**convergent 属性标记。** Barrier、shuffle 和 vote 内联函数必须标记为
`convergent`，以防止 LLVM 把它们提升出控制流。导出器通过对（点号形式的）名称
做前缀模式匹配来识别它们，并在其调用点追加 `#0`，在模块层级发出
`attributes #0 = { convergent }`。

**Kernel 元数据。** 被标记为 kernel 的函数会获得 `ptx_kernel` 调用约定和一条
`!nvvm.annotations` 元数据条目。

以下是一个简单向量加法 kernel 导出的 LLVM IR 的样子：

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

注意切片已经被标量化：每个 Rust `&[f32]` 变成一个 `ptr addrspace(1)` 和一个
`i64` 长度。这发生在下降 pass 中；当 LLVM 方言看到它们时，它们已经是扁平参数。

---

## dialect-nvvm —— GPU 层

`dialect-nvvm` 把 NVIDIA 的 GPU 内联函数包装为类型化的 pliron 操作。这些操作
并不构成下降链中的"一层" —— 它们是在 `dialect-mir` → LLVM 方言的下降 pass
期间被插入的，并与 LLVM 方言操作共存于同一个函数体中。在导出时，它们变成对
`@llvm.nvvm.*` 内联函数的 `call` 指令。

### 架构覆盖范围

在目录 SHA-256 为 `c243e568` 时（这是每个 `ops/generated/` 文件头部中的戳记），
该方言在 42 个模块中持有 576 个操作，它们来自两个不同的地方。这种划分是首先
需要了解的事情，因为它决定了你在哪里 —— 以及是否 —— 会添加一个操作。如果头部
戳记不再以 `c243e568` 开头，那么本页上的计数早于你正在阅读的目录。

**手写**的，直接位于 `cuda-oxide/crates/dialect-nvvm/src/ops/` 下。这些是带有
定制校验或下降、内联函数目录未描述的操作。共有七个模块和 26 个操作：

| 模块      | 描述                                                        | 操作 |
| :-------- | :---------------------------------------------------------- | ---: |
| `asm`     | `inline_ptx`                                                |    1 |
| `atomic`  | 原子 load/store/RMW/cmpxchg/fence                            |    5 |
| `cluster` | Cluster 索引与 cluster 计数寄存器                             |    2 |
| `debug`   | `assertfail`、`vprintf`                                     |    2 |
| `grid`    | 协作式 `grid_sync`                                           |    1 |
| `memory`  | 带字节偏移的通用地址到共享地址转换                            |    1 |
| `wgmma`   | Warpgroup MMA 描述符；bf16/f16 在 m64n64k16，bf16 在 m64n128k16，tf32 在 m64n64k8 |  14 |

**生成**的，位于 `ops/generated/` 下，由 `cuda-intrinsics-gen` 从
`intrinsics/catalog.json` 生成。那里的每个文件都以 `// @generated ... DO NOT
EDIT.` 开头，手工编辑其中一个会被下一次生成器运行撤销。这是绝大多数 —— 35 个
模块和 550 个操作，从 1029 个目录条目解析而来，因为若干内联函数可以共享同一个
结构化操作：

| 领域                        | 模块                                                                          | 操作 |
| :-------------------------- | :---------------------------------------------------------------------------- | ---: |
| 张量核心 Gen 5 + TMEM       | `tcgen05`                                                                     | 210 |
| 张量内存加速器              | `tma`                                                                         | 111 |
| 特殊寄存器                  | `sreg`                                                                        |  44 |
| 打包（寄存器内 SIMD）       | `packed_alu`, `packed_conversion`, `packed_atomic`                            |  51 |
| 异步拷贝与屏障              | `cp_async`, `mbarrier_extended`, `mbarrier_basic`, `sync`                     |  35 |
| Warp 级                     | `warp_shuffle`, `redux`, `vote`, `warp_match`, `warp_barrier`, `active_mask`, `elect` |  39 |
| 矩阵片段搬移                | `ldmatrix`, `register_mma`, `stmatrix`, `wgmma_control`, `movmatrix`, `sparse_mma` |  22 |
| 执行与调试控制              | `execution_control`, `debug_control`                                          |  11 |
| Cluster                     | `clc`, `cluster_barrier`, `cluster_memory`                                    |  10 |
| 标量数学                    | `dotprod`, `scalar_arithmetic`, `scalar_conversion`, `scalar_math`, `extended_minmax`, `prmt` |   9 |
| 整数 min/max（DPX）         | `integer_minmax`                                                              |   8 |

架构要求是按内联函数而不是按模块记录的 —— 目录记录每一个内联函数的 PTX 版本和
最低 SM，而 `intrinsics/generated-reference.md` 会把它们与各自预期发出的 PTX
并排渲染出来。该文件与操作一起重新生成，所以应查阅它，而不是依赖这里手工维护
的计数。

大多数用户只会遇到特殊寄存器、warp shuffle 和屏障。其余用于高级 GPU 编程 ——
TMA、矩阵加速器以及 Blackwell 的张量内存 —— 在
[高级 GPU 特性](../advanced/tensor-memory-accelerator.md) 各章中介绍。如果你要
添加一个操作，请先阅读 [添加新的内联函数](adding-new-intrinsics.md)：它走的是
目录路径，而几乎所有新的内联函数都走这条路。

### 从 Rust 到 PTX：一个内联函数的旅程

每个 NVVM 操作都要经过三层命名：

| Pliron 操作                    | LLVM 内联函数                                          | PTX 指令                                |
| :----------------------------- | :---------------------------------------------------- | :------------------------------------- |
| `ReadPtxSregTidXOp`            | `llvm.nvvm.read.ptx.sreg.tid.x`                       | `mov.u32 %r1, %tid.x`                  |
| `Barrier0Op`                   | `llvm.nvvm.barrier0`                                  | `bar.sync 0`                           |
| `ShflSyncBflyI32Op`            | `llvm.nvvm.shfl.sync.bfly.i32`                        | `shfl.sync.bfly.b32`                   |
| `CpAsyncBulkTensorG2sTile2dOp` | `llvm.nvvm.cp.async.bulk.tensor.2d.tile.g2s.im2col.*` | `cp.async.bulk.tensor.2d.tile.g2s ...` |

第一列是 `dialect-nvvm` 中的 Rust 结构体名。第二列是 `llvm-export` 发出的内容
（经过下划线到点号的转换之后）。第三列是 `llc` 产生的内容。你永远不必手工编写
其中任何一个：`mir-importer` 在翻译对 `cuda-device` 内联函数（例如
`thread::threadIdx_x()` 或 `warp::shuffle_xor_sync()`）的调用时发出该操作，
后续阶段再从它派生出另外两种形式。

### 校验策略

NVVM 操作使用最小的结构化校验：每个操作检查其操作数数量和结果数量，少数操作会
校验结果类型（线程索引操作要求 `i32` 结果；`tcgen05` 加载会检查其 32 寄存器与
4 寄存器变体的准确结果数量）。

这是刻意的。NVVM 操作由 `mir-importer` 机器生成 —— 它们从不由用户手写。LLVM
的 NVPTX 后端在下游提供全面的类型验证。为每个 NVVM 操作添加完整类型检查会让该
方言的代码量翻倍，却没有实际收益。

> **注意**
>
> GPU 架构要求（sm_70、sm_90、sm_100）有文档记录，但在 pliron 层面并不强制执行。
> 架构验证发生在稍后，即 `llc` 以特定的 `-mcpu=sm_XX` 标志被调用时。如果你使用
> 一个 Hopper 内联函数却面向 Volta，`llc` 会告诉你 —— 而且很大声。

---

## 方言如何相互作用

以下是一个单一 Rust 操作经过全部三个抽象层次的生命周期：

```text
Rust source:   let sum = a + b;        // a, b: f32

dialect-mir:   %sum = mir.add %a, %b : f32
                ↓  (DialectConversion)
LLVM dialect:  %v5 = fadd float %v3, %v4
                ↓  (llvm-export)
LLVM IR:       %v5 = fadd float %v3, %v4
                ↓  (llc --mcpu=sm_80)
PTX:           add.f32 %f3, %f1, %f2;
```

`dialect-mir` → LLVM 方言这一步才是有趣工作发生的地方：`f32` 上的 `mir.add`
变成 `fadd`（浮点加法），而 `i32` 上的 `mir.add` 变成 `add`（整数加法）。像
`mir.checked_add` 这样的带检查操作会展开为一个 `llvm.add`、一个表示溢出标志的
常量 `i1 false`，以及一次到结构体的 `insertvalue` —— GPU 路径省略了溢出检测
（因为 GPU 整数算术是回绕的）。下降 pass 处理所有这些转换。

对于 GPU 特有的操作，`dialect-nvvm` 登场：

```text
Rust source:   let tid = thread::threadIdx_x();
                ↓  (mir-importer matches "cuda_device::thread::threadIdx_x")
dialect-nvvm:  %v2 = nvvm.read_ptx_sreg_tid_x : i32
                ↓  (mir-lower, then llvm-export)
LLVM IR:       %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
                ↓  (llc)
PTX:           mov.u32 %r1, %tid.x;
```

`mir-importer` 中的翻译器通过完全限定名（FQDN）识别对 `cuda_device` 内联函数
的调用；匹配 `threadIdx_x` 的那个分支由 `intrinsics/catalog.json` 生成到其分发
逻辑中。该调用永远不会变成 `mir.call`：翻译器当场发出 `dialect-nvvm` 操作，
于是内联函数最终成为一条直接的硬件指令。

### 全貌

把这一切放在一起，一个编译后的 kernel 函数体包含 LLVM 方言和 `dialect-nvvm`
操作的混合：

```text
llvm.func @vecadd(...) {
  entry:
    %tid    = nvvm.read_ptx_sreg_tid_x     // NVVM: 线程索引
    %ntid   = nvvm.read_ptx_sreg_ntid_x    // NVVM: block 大小
    %ctaid  = nvvm.read_ptx_sreg_ctaid_x   // NVVM: block 索引
    %offset = llvm.mul %ctaid, %ntid        // LLVM: 整数运算
    %idx    = llvm.add %offset, %tid        // LLVM: 整数运算
    %cmp    = llvm.icmp slt %idx, %len      // LLVM: 边界检查
    llvm.cond_br %cmp, bb1, bb2             // LLVM: 分支
  bb1:
    %p_a    = llvm.gep %a, %idx             // LLVM: 指针运算
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

顶部的 `dialect-nvvm` 操作计算全局线程索引。其余都是标准的 LLVM 方言 ——
加载、存储、算术、分支。导出引擎把所有这些序列化为一个 `.ll` 文件，然后
`llc` 把它编译为 PTX。

---

关于下降 pass 如何连接这些方言，见
[下降流水线](lowering-pipeline.md)。
