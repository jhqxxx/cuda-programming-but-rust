# 降级流水线

前面的章节从两端构建了 IR：[MIR 导入器](mir-importer.md)将稳定 MIR 翻译成 `dialect-mir`，而 [Pliron dialect](mlir-dialects.md)描述了 `dialect-mir`、`dialect-llvm` 和 `dialect-nvvm`。本章是关于它们之间的桥梁 —— 将带有 Rust 风味的 IR 转换成 LLVM 实际可以编译的形式的 pass。

如果你了解 Rust 类型，你将要发现 LLVM 对其中的多少种一无所知。

---

## 降级的含义

`dialect-mir` 使用 Rust 语言。它了解元组、枚举、切片、检查算术和 GPU 地址空间。LLVM IR 对这些一无所知。它只有扁平的整数和浮点类型、`getelementptr`、PHI 节点，并且对任何具有超过一层抽象的东西都持普遍怀疑态度。

**降级**是用等效的 `dialect-llvm` 操作序列逐个替换每个 `dialect-mir` 操作的过程，直到没有 `dialect-mir` 操作剩余。元组变成匿名结构体。切片变成指针-长度对。检查加法变成 LLVM 溢出内联函数后跟一个提取操作。每个 Rust 概念都被展平为 LLVM 可以消化的东西。

完成这一切的 pass 位于 `crates/mir-lower/` 中，并使用 pliron 的 `DialectConversion` 框架。它是流水线中最大的一次转换，本章的其余部分将介绍其工作原理。

---

## DialectConversion —— 降级框架

降级使用 pliron 的 `DialectConversion` + `DialectConversionRewriter`，而不是手动的遍历和替换 pass。该框架自动处理 IR 遍历、定义在使用之前排序、类型转换和块参数修补。

### 工作原理

每个 `dialect-mir` 和 `dialect-nvvm` 操作通过 `MirToLlvmConversion` 操作接口（在 `conversion_interface.rs` 中定义）声明如何降级自身。该接口有一个单独的方法：`convert(ctx, rewriter, op, operands_info)`。每个操作的实现位于 `convert/interface_impls.rs` 中，它分派到按类别组织的转换器函数。

对于模块中的每个 `MirFuncOp`，`convert_func`（在 `lowering.rs` 中）执行以下操作：

1. **创建一个 LLVM 函数**，其签名取决于该函数是否为kernel入口点。非kernel函数将每个聚合参数展平为其标量字段，以适应内部 CUDA ABI。kernel入口点保持切片展平（`(ptr, len)`），但将结构体和闭包作为单个 byval 值传递，因为主机启动器将整个聚合作为一个包槽位推送（参见下文 {ref}`参数标量化 <lowering-argument-scalarization>`）。
2. **传播 GPU 元数据**（`gpu_kernel`、`maxntid`、`cluster_dim_*`）。
3. 使用 `inline_region` 将 MIR 块移动到 LLVM 函数中。原始块被保留 —— 不需要手动块映射。
4. **构建入口序言**，根据到达 LLVM 层的任何形状为每个参数重建 MIR 级别的值（内部调用和切片的展平标量的 `insertvalue`；kernel边界处的 byval 聚合直接传递）。
5. 运行 `DialectConversion`，它遍历每个 MIR 操作并调用其 `MirToLlvmConversion::rewrite` 实现，将其替换为 LLVM 操作。

### 转换器模块

转换函数按类别组织成模块：

| 类别   | 模块     | 处理内容   |
| :----------- | :------ | :------ |
| 算术         | `convert/ops/arithmetic.rs`     | `add`→`add`，`sub`→`sub`，`checked_add`→`add`+`extractvalue`    |
| 内存         | `convert/ops/memory.rs`         | `mir.load`→`load`，`mir.store`→`store`，`shared_alloc`→全局 + `addrspacecast` |
| 控制流       | `convert/ops/control_flow.rs`   | `mir.goto`→`br`，`mir.cond_br`→`cond_br`，`mir.return`→`return`            |
| 聚合         | `convert/ops/aggregate.rs`      | 结构体/元组字段访问 → GEP 或 `extractvalue`/`insertvalue`                  |
| 转换         | `convert/ops/cast.rs`           | `IntToInt`→`zext`/`sext`/`trunc`，`FloatToFloat`→`fpext`/`fptrunc` 等     |
| 调用         | `convert/ops/call.rs`           | `mir.call`→`call`，参数展平和 `::` 到 `__` 名称转换                        |
| GPU 内联函数 | `convert/intrinsics/*.rs`       | NVVM 操作 → LLVM 内联函数调用或内联 PTX                                    |
| 常量         | `convert/ops/constants.rs`      | `mir.constant`→`llvm.constant`   |

框架自动分派到这些转换器 —— 每个操作的 `MirToLlvmConversion` 实现都会调用正确的转换器函数。复杂性存在于每个转换器内部，在那里 Rust 语义与 LLVM 现实相遇。

---

## 类型转换

在转换操作之前，必须先转换其类型。LLVM 的类型系统有意地比 Rust 简单 —— 整数上没有符号性，没有元组，没有枚举，没有胖指针。一切必须被展平。

| MIR 类型      | LLVM 类型    | 说明     |
| :------- | :----- | :------ |
| `IntegerType(32, Unsigned)`    | `IntegerType(32, Signless)`  | LLVM 整数不携带符号 —— 符号性在*操作*上，而不是类型上      |
| `MirTupleType<i32, f32>`       | `{ i32, float }`    | 元组变成匿名结构体           |
| `MirSliceType<f32>`    | `{ ptr, i64 }`   | 胖指针分解 —— 指针 + 长度         |
| `MirStructType`    | `{ fields..., [N x i8] }`    | 显式填充数组以匹配 rustc 的布局    |
| `MirPtrType<f32, addrspace:3>` | `ptr addrspace(3)`    | 保留地址空间的不透明指针    |
| `MirArrayType<f32, 256>`  | `[256 x float]`   | 直接映射 —— 数组对于 LLVM 也足够简单    |
| `MirEnumType`    | `{ discriminant, [M x i8] }` | 判别式 + 按最大变体大小分配的有效负载区域    |

整数符号性值得强调。在 Rust 中，`i32` 和 `u32` 是不同的类型。在 LLVM 中，两者都只是 `i32`。符号信息转移到操作上：有符号小于比较是 `icmp slt`，无符号的是 `icmp ult`。类型转换器丢弃符号性，操作转换器在发出比较和除法指令时将其重新拾起。


### 参数标量化

kernel入口点需要特殊处理。CUDA 驱动程序不理解 Rust 的胖指针，因此 `&[f32]` 必须在 ABI 两侧分别作为指针和长度到达。另一方面，按值传递的结构体和闭包确实与单个主机包槽位匹配，因此kernel入口将它们作为单个 byval `.param` 接收 —— 否则设备会期望 N 个展平的参数，而主机只会推送一个，导致与后续每个切片不匹配。

因此，降级 pass 区分kernel入口规则和内部调用规则：

| MIR kernel参数    | LLVM kernel签名           |
|:----------|:-------------------|
| `&[f32]`    | `ptr addrspace(1) %ptr, i64 %len`       |
| `T`（标量）  | 直接传递        |
| `struct { a, b }`     | 一个 byval `{a, b}` 值       |
| 闭包（N 个捕获）  | 一个 byval 闭包结构体值       |
| 零大小聚合  | 丢弃（无 LLVM 参数，无主机包槽位）  |

切片情况仍然在入口块中使用经典的“从展平形式重建”模式：

```text
MIR:  fn kernel(slice: &[f32])
      → 入口参数：%slice : MirSliceType

LLVM: fn kernel(ptr addrspace(1) %ptr, i64 %len)
      → 入口块重建：
          %slice  = insertvalue {ptr, i64} undef, %ptr, 0
          %slice2 = insertvalue {ptr, i64} %slice, %len, 1
```

结构体/闭包情况跳过重建 —— byval 值已经是正确的形状 —— 函数的其余部分看到它时就好像什么都没发生。内部设备到设备调用继续保持与以前相同的聚合展平方式，因此此规则的成本仅存在于kernel边界。

---

## 有趣的转换

大多数转换是直接的：整数上的 `mir.add` 变成 `llvm.add`，`mir.load` 变成 `llvm.load`，`mir.goto` 变成 `llvm.br`。有趣的情况是单个 MIR 操作扩展为多个 LLVM 操作，或者 GPU 特定关注点完全改变了翻译。

### 检查算术

在 debug 构建中，Rust 检查每个整数算术操作是否溢出。MIR 使用像 `mir.checked_add` 这样的操作对此建模，该操作返回一个 `(result, overflow_flag)` 元组。LLVM 没有这样的概念，但它确实有溢出内联函数：

```text
MIR:  %result = mir.checked_add %a, %b : i32  → mir.tuple<i32, bool>

LLVM: %sum = add i32 %a, %b
      %overflow = extractvalue {i32, i1} @llvm.sadd.with.overflow.i32(%a, %b), 1
```

溢出标志输入到一个断言中，MIR 导入器已经将该断言降低为以不可达块为目标的条件分支。在 GPU 上，这实际上意味着：如果你设法触发了整数溢出，kernel就会陷入陷阱。这不是最优雅的错误处理方式，但 CUDA 工具链目前不支持栈展开。

### 共享内存

CUDA 中的共享内存是块作用域的 SRAM —— 快速、小型、静态声明。在 `dialect-mir` 中，它是一个 `mir.shared_alloc` 操作。在 LLVM IR 中，共享内存必须是地址空间 3 中的模块级全局变量：

```text
MIR:  %shmem = mir.shared_alloc : mir.array<f32, 256>

LLVM: @shmem_0 = addrspace(3) global [256 x float] zeroinitializer
      %ptr = addrspacecast [256 x float] addrspace(3)* @shmem_0 to ptr
```

`addrspacecast` 产生一个通用指针，函数的其余部分可以在不关心地址空间的情况下使用它。LLVM 中的 NVPTX 后端处理其余部分 —— 它知道 `addrspace(3)` 意味着共享内存，并生成适当的 `st.shared` / `ld.shared` 指令。

### 枚举降级

Rust 枚举在代数上是丰富的。LLVM 没有标记联合的概念。降级 pass 通过将枚举表示为具有两个字段的结构体来弥合差距：一个判别式（告诉你哪个变体处于活动状态）和一个按最大变体大小分配的有效负载区域：

```text
MIR:  %opt = mir.construct_enum "Some", (%val) : mir.enum<"Option_i32">

LLVM: %tmp = insertvalue { i8, [4 x i8] } zeroinitializer, i8 1, 0
      %result = insertvalue { i8, [4 x i8] } %tmp, <val into payload area>
```

判别式是 `i8 1`，因为 `Some` 是 `Option` 的变体 1。有效负载是 `[4 x i8]` —— 四个字节，足以容纳一个 `i32`。变体访问反向进行：读取判别式，根据它分支，然后 `extractvalue` 有效负载并位转换为期望的类型。

它并不优雅，但这正是 C 编译器几十年来处理标记联合的方式。LLVM 的优化器在清理冗余的 insertvalue/extractvalue 链方面相当出色。

---

## GPU 内联函数转换

`dialect-nvvm` 操作 —— 线程索引、线程束洗牌、屏障、TMA 批量拷贝 —— 不会被降低为通用的 `dialect-llvm` 操作。它们被降低为 LLVM 内联函数调用或内联 PTX 汇编，具体取决于 LLVM 是否为该操作提供了内置内联函数。

### 策略 1：LLVM 内联函数调用

对于 LLVM 已经提供目标特定内联函数的操作，转换会发出对该内联函数的 `call`：

```text
nvvm.read_ptx_sreg_tid_x
  → call i32 @llvm_nvvm_read_ptx_sreg_tid_x()

nvvm.shfl_sync_bfly_i32
  → call i32 @llvm_nvvm_shfl_sync_bfly_i32(i32 -1, i32 %val, i32 %mask, i32 31)
```

注意线程束洗牌：面向用户的 `cuda_device` API 接受两个参数（值和通道掩码），但 LLVM 内联函数接受四个参数（membermask、value、delta、clamp）。降级 pass 填补了缺失的参数 —— `membermask = -1`（所有通道）和 `clamp = 31`（完整线程束宽度） —— 因此用户永远不必考虑它们。

### 策略 2：内联 PTX 汇编

较新的 GPU 指令通常缺少 LLVM 内联函数。对于这些，降级 pass 使用 LLVM 的 `asm` 语法发出内联 PTX 汇编：

```text
nvvm.wgmma_fence_sync
  → call void asm sideeffect convergent "wgmma.fence.sync.aligned;", ""()

nvvm.mbarrier_arrive
  → call i64 asm sideeffect convergent "mbarrier.arrive.shared.b64 $0, [$1];", "=l,r"(ptr %bar)
```

`convergent` 属性在这里至关重要。它告诉 LLVM：“不要将此指令移过、复制或推测跨越控制流。”没有它，LLVM 可能会将屏障提升出条件分支，或将线程束级指令下沉到同步点之后，导致 GPU 挂起或计算垃圾数据 —— 这两种情况都不会产生有用的错误消息。

---

## 块参数到 PHI 节点

Pliron IR（类 MLIR）使用**块参数**在基本块之间进行值传递。LLVM 使用 **PHI 节点**。它们表达相同的概念 —— “这个值来自不同的前驱” —— 但语法差异足够大，以至于导出步骤需要一个真正的转换，而不仅仅是漂亮的打印。

Pliron 风格（块参数）：

```text
^loop_header(%sum: f32, %i: i64):
    ...
    br ^loop_header(%new_sum, %new_i)
```

LLVM IR 风格（PHI 节点）：

```text
loop_header:
    %sum = phi float [ 0.0, %preheader ], [ %new_sum, %body ]
    %i = phi i64 [ 0, %preheader ], [ %new_i, %body ]
```

导出器使用两遍方法处理此转换：

1. **预遍：为每个值命名。** 在发出任何代码之前，导出器遍历所有块并为每个值分配顺序的 SSA 名称（`%v0`、`%v1`、...）。这很关键，因为 PHI 节点可以引用在列表中*后面*出现的块中的值 —— 循环回边在文本中指向前方，但在控制流中指向前方。没有预命名，这些引用将是未定义的。

2. **构建前驱映射。** 对于每个块，导出器通过检查函数中的每个分支指令来收集 `(predecessor_block, values_passed)` 对。

3. **发出 PHI 节点。** 在每个非入口块的入口处，导出器为每个块参数发出一个 PHI 节点，用前驱映射中的值和前驱标签填充。

预遍是微妙的部分。考虑一个循环：循环头中的 PHI 引用来自循环体的 `%new_sum`，但循环体在文本输出中出现在循环头*之后*。如果在发出期间动态分配名称，`%new_sum` 还没有名称。预遍通过预先命名所有内容消除了这个问题。

---

## 符号名称清理

函数名称流经多个阶段，每个阶段都应用自己的约束：

```text
rustc_public (FQDN)            helper_fn::cuda_oxide_device_<hash>_vecadd
  ↓ body.rs (:: → __)
dialect-mir                    helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ call.rs (:: → __)
dialect-llvm                   helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ export.rs (去除前缀)
文本 LLVM IR                   @vecadd
  ↓ llc
PTX                            vecadd

```

沿此路径发生三个转换：

1. **`::` 到 `__`** —— `body.rs`（函数定义）和 `call.rs`（调用目标）都将 Rust 路径分隔符替换为双下划线，以生成有效的 pliron/LLVM 标识符。由于双方应用相同的转换，定义和调用点匹配。

2. **设备前缀去除** —— `export.rs` 通过 `reserved_oxide_symbols::device_base_name` 从 `#[device]` 函数名称中去除保留的 `cuda_oxide_device_<hash>_` 前缀（以及任何前面的 FQDN crate 前缀）。此前缀存在于 MIR 级检测中，但不应出现在最终的 LLVM IR、PTX 或 LTOIR 输出中。

3. **设备外部前缀去除** —— 对于 `#[device] unsafe extern "C"` 函数，`call.rs` 通过 `reserved_oxide_symbols::device_extern_base_name` 去除 `cuda_oxide_device_extern_<hash>_` 前缀，以便 LLVM IR 引用外部 LTOIR（例如 CCCL 库）导出的原始符号名称。

> **注意**：
> 这种手动清理将在框架升级后被 pliron 的 `Legaliser` 替换。`Legaliser` 系统性地处理 `::` 到 `_` 的转换和冲突检测。

---

## PTX 生成

在 `dialect-llvm` 导出为文本 `.ll` 文件之后，最后一步是调用 `llc` —— LLVM 的静态编译器 —— 来生成 PTX 汇编：

```bash
llc -march=nvptx64 -mcpu=sm_90 kernel.ll -o kernel.ptx
```

### 目标选择

流水线按以下顺序在 `PATH` 上探测 `llc`。LLVM 21 是最低版本 —— 较早的版本拒绝 cuda-oxide 发出的 TMA / tcgen05 / WGMMA 内联函数签名。

| 优先级 | llc 版本 | 目标                     | PTX 版本 |
| :----- | :------- | :----------------------- | :------- |
| 1st    | `llc-22` | `sm_100a` (Blackwell DC) | PTX 8.x  |
| 2nd    | `llc-21` | `sm_100` / `sm_120`      | PTX 8.x  |

如果两者都不可用，流水线将失败并显示清晰的错误信息。你可以通过设置 `CUDA_OXIDE_LLC=/path/to/llc` 来选择特定（可能是较旧的）二进制文件，但只有简单的kernel保证可以在 LLVM 20 及以下版本上编译。

如果选定的目标与物理 GPU 不匹配，CUDA 驱动程序会在加载时即时编译 PTX。第一次启动大约需要 30 毫秒，用于驱动程序翻译；后续启动使用缓存的二进制文件。在实践中，你很少会注意到 —— JIT 很快，并且缓存在多次运行之间持久存在。

`CUDA_OXIDE_TARGET` 环境变量在你需要特定目标时覆盖自动检测。例如，`sm_100a` 启用了 Blackwell 特定的 `tcgen05` 特性，这些特性在通用的 `sm_100` 目标下不可用。

`cargo oxide run` 在后端基于特性的默认值之上添加了第二层自动检测：当既未设置 `--arch` 也未设置 `CUDA_OXIDE_TARGET` 时，它会查询 CUDA 设备 0 的计算能力，并将其转发给后端，以便生成的模块保证可以在本地 GPU 上加载。完整的优先级顺序是 `--arch` > `CUDA_OXIDE_TARGET` > 主机 CC（仅对 `run` 有效） > 后端基于特性的默认值。`cargo oxide build` 和 `cargo oxide pipeline` 故意跳过主机 CC 步骤，以便它们仍然可用于交叉编译。

> **注意**：
> 为什么是 LLVM 21？`tma_copy`、`gemm_sol` 和 `tcgen05_matmul` 使用的 2D 批量 TMA 加载内联函数在 LLVM 21 中获得了带有 `addrspace(7)` 和 `cta_group` 参数的 10 操作数形式。较旧的 `llc` 版本会以 `Intrinsic has incorrect argument type!` 拒绝它。与其为每个 LLVM 版本维护单独的内联函数发射器，我们将 21 设置为最低版本。

---

## 整合起来

以下是 `lower_mir_to_llvm` 处理模块时的完整事件序列：

```text
1. 注册 `dialect-llvm` 类型和操作
2. 对于模块中的每个 MirFuncOp：
   a. 创建带有展平类型签名的 `llvm.func`
   b. inline_region：将 `dialect-mir` 块移动到 LLVM 函数中
   c. 构建入口序言（从扁平参数重建聚合）
   d. 运行 DialectConversion：
      ├── 遍历每个 `dialect-mir`/`dialect-nvvm` 操作（定义在使用之前顺序）
      ├── 为每个操作调用 MirToLlvmConversion::rewrite
      ├── 转换器通过 DialectConversionRewriter 发出 `dialect-llvm` 操作
      └── 框架自动修补块参数类型
3. 将 `dialect-llvm` 导出为文本 LLVM IR（.ll）（带 PHI 节点转换）
4. 调用 llc 生成 .ptx
```

在第 4 步之后，你有了一个 CUDA 驱动程序可以加载和执行的 `.ptx` 文件。从 `mir.checked_add` 到 `add.s32` 的旅程完成了。

---

降级流水线将带有 Rust 风味的 IR 转换为 GPU 就绪的 LLVM IR。有关添加新 GPU 操作的动手演练，请参阅[添加新内联函数](./添加新内联函数.md)。

