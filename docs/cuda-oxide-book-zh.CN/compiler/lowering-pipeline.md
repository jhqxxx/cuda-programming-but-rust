# 下降流水线

前面的章节构建并准备好了 IR。
[MIR Importer](mir-importer.md) 把 Stable MIR 翻译为 `dialect-mir`。
[编译器优化](compiler-optimizations.md) 解释了 `mir-transforms` 如何就地重写
那个 IR，而 [Pliron 方言](mlir-dialects.md) 描述了 `dialect-mir`、LLVM 方言
以及 `dialect-nvvm`。

本章介绍下一座桥：把 Rust 风味的 IR 变成 LLVM 能编译的东西。

对于 NVVM 构建，一个单独的 `nvvm-transforms` pass 会在本次下降之后运行。
它让 `mir-lower` 专注于把 `dialect-mir` 转换为 LLVM 方言，然后在文本导出之前
为旧版或现代的 libNVVM 调整该 LLVM 模块。

如果你了解 Rust 类型，你即将发现其中有多少是 LLVM 从未听说过的。

---

## 下降意味着什么

`dialect-mir` 说 Rust。它了解元组、枚举、切片、带检查的算术以及 GPU 地址空间。
LLVM IR 对这些一无所知。它有扁平的整数和浮点类型、`getelementptr`、PHI 节点，
以及对任何抽象层次超过一层的东西的普遍怀疑。

**下降**是这样一个过程：把每个 `dialect-mir` 操作逐个替换为等价的 LLVM 方言
操作序列，直到不再有任何 `dialect-mir` 操作残留。元组变成匿名结构体。切片
变成指针-长度对。带检查的加法变成 LLVM 溢出内联函数加一次提取。每个 Rust 概念
都被扁平化为 LLVM 能消化的东西。

完成所有这些的 pass 位于 `cuda-oxide/crates/mir-lower/`，并使用 pliron 的
`DialectConversion` 框架。它是整条流水线中最大的一次变换，本章其余部分讲的就是
它如何工作。

---

## DialectConversion —— 下降框架

下降使用 pliron 的 `DialectConversion` + `DialectConversionRewriter`，而不是
手工的遍历-替换 pass。该框架自动处理 IR 遍历、定义先于使用的顺序、类型转换
以及 block 参数修补。

### 它如何工作

每个 `dialect-mir` 和 `dialect-nvvm` 操作通过 `MirToLlvmConversion` op 接口
（定义在 `conversion_interface.rs` 中）声明自己如何下降。该接口只有一个方法：
`convert(ctx, rewriter, op, operands_info)`。每个操作的实现位于
`convert/interface_impls.rs`，它分发到按类别组织的转换函数。

对于模块中的每个 `MirFuncOp`，`convert_func`（位于 `lowering.rs`）：

1. **创建一个 LLVM 函数**，其签名取决于该函数是否是 kernel 入口点。非 kernel
   函数为内部 CUDA ABI 把每个聚合参数扁平化为其标量字段。Kernel 入口点让切片
   保持扁平（`(ptr, len)`），但把结构体和闭包作为单个 byval 值传入，因为宿主
   启动器会把整个聚合作为一个 packet slot 推入（见下文
   [参数标量化](#lowering-argument-scalarization)）。
2. **传播 GPU 元数据**（`gpu_kernel`、`maxntid`、`cluster_dim_*`）。
3. **使用 `inline_region`** 把 MIR block 移入 LLVM 函数。原始 block 会被保留
   —— 不需要手工做 block 映射。
4. **构建入口序言**，为每个参数从 LLVM 层面到达时的形状重建 MIR 层面的值
   （对内部调用和切片做扁平标量的 `insertvalue`；在 kernel 边界对 byval 聚合
   直接透传）。
5. **运行 `DialectConversion`**，它遍历每个 MIR 操作并调用其
   `MirToLlvmConversion::rewrite` 实现，用 LLVM 操作替换它。

### 转换器模块

转换函数按类别组织为模块：

| 类别          | 模块                          | 处理内容                                                                         |
| :------------ | :---------------------------- | :------------------------------------------------------------------------------- |
| 算术          | `convert/ops/arithmetic.rs`   | `add`→`add`、`sub`→`sub`、`checked_add`→`add`+`extractvalue`                     |
| 内存          | `convert/ops/memory/*.rs`     | `mir.load`→`load`、`mir.store`→`store`、`shared_alloc`→global + `addrspacecast`  |
| 控制流        | `convert/ops/control_flow.rs` | `mir.goto`→`br`、`mir.cond_br`→`cond_br`、`mir.return`→`return`                  |
| 聚合          | `convert/ops/aggregate/*.rs`  | 结构体/元组字段访问 → GEP 或 `extractvalue`/`insertvalue`                        |
| 转换          | `convert/ops/cast.rs`         | `IntToInt`→`zext`/`sext`/`trunc`、`FloatToFloat`→`fpext`/`fptrunc` 等            |
| 调用          | `convert/ops/call.rs`         | `mir.call`→`call`，带参数扁平化和 `::` 到 `__` 的名称转换                        |
| GPU 内联函数  | `convert/intrinsics/*.rs`     | NVVM 操作 → LLVM 内联函数调用或内联 PTX                                          |
| 常量          | `convert/ops/constants.rs`    | `mir.constant`→`llvm.constant`                                                   |

框架会自动分发到这些模块 —— 每个操作的 `MirToLlvmConversion` 实现调用正确的
转换函数。复杂性存在于每个转换器内部，也就是 Rust 语义与 LLVM 现实交汇之处。

---

## 类型转换

在操作能被转换之前，它们的类型必须被转换。LLVM 的类型系统刻意比 Rust 的更简单
—— 整数没有符号性，没有元组，没有枚举，没有胖指针。一切都必须被扁平化。

| MIR 类型                       | LLVM 类型                    | 备注                                                                           |
| :----------------------------- | :--------------------------- | :----------------------------------------------------------------------------- |
| `IntegerType(32, Unsigned)`    | `IntegerType(32, Signless)`  | LLVM 整数不携带符号 —— 符号性在*操作*上，而不在类型上                           |
| `MirTupleType<i32, f32>`       | `{ i32, float }`             | 元组变成匿名结构体                                                             |
| `MirSliceType<f32>`            | `{ ptr, i64 }`               | 胖指针分解 —— 指针 + 长度                                                       |
| `MirStructType`                | `{ fields..., [N x i8] }`    | 显式填充数组以匹配 rustc 的布局                                                 |
| `MirPtrType<f32, addrspace:3>` | `ptr addrspace(3)`           | 保留地址空间的不透明指针                                                        |
| `MirArrayType<f32, 256>`       | `[256 x float]`              | 直接映射 —— 数组即使对 LLVM 来说也足够简单                                       |
| `MirEnumType`                  | `{ discriminant, [M x i8] }` | 判别值 + 按最大变体确定大小的载荷                                                |

整数符号性这一情形值得强调。在 Rust 中，`i32` 和 `u32` 是不同的类型。在 LLVM
中，两者都只是 `i32`。符号信息转移到操作上：有符号小于比较是 `icmp slt`，
无符号的是 `icmp ult`。类型转换器丢弃符号性，而操作转换器在发出比较和除法指令
时把它捡回来。

<a id="lowering-argument-scalarization"></a>

### 参数标量化

Kernel 入口点需要特殊处理。CUDA 驱动不理解 Rust 胖指针，所以 `&[f32]` 必须在
ABI 两侧都作为单独的指针和长度到达。另一方面，按值传递的结构体和闭包确实与
单个宿主 packet slot 匹配，因此 kernel 入口把它们作为一个 byval `.param` 接收
—— 否则设备会期待 N 个扁平参数，而宿主只推入一个，从而与之后的每一个切片都
错位。

因此下降 pass 区分了 kernel 入口规则与内部调用规则：

| MIR kernel 参数       | LLVM kernel 签名                           |
|:----------------------|:-------------------------------------------|
| `&[f32]`              | `ptr addrspace(1) %ptr, i64 %len`          |
| `T`（标量）           | 透传                                        |
| `struct { a, b }`     | 一个 byval `{a, b}` 值                      |
| 闭包（N 个捕获）      | 一个 byval 闭包结构体值                     |
| 零大小聚合            | 丢弃（没有 LLVM 参数，也没有宿主 packet slot） |

切片这一情形仍然在入口 block 中使用经典的重建自扁平模式：

```text
MIR:  fn kernel(slice: &[f32])
      → entry arg: %slice : MirSliceType

LLVM: fn kernel(ptr addrspace(1) %ptr, i64 %len)
      → entry block reconstructs:
          %slice  = insertvalue {ptr, i64} undef, %ptr, 0
          %slice2 = insertvalue {ptr, i64} %slice, %len, 1
```

结构体/闭包这一情形跳过重建 —— byval 值已经是正确的形状 —— 函数其余部分看到
它就仿佛什么都没特别发生过。内部设备到设备的调用一如既往地继续扁平化聚合，
因此这条规则的代价只存在于 kernel 边界处。

---

## 有趣的转换

大多数转换都是直截了当的：整数上的 `mir.add` 变成 `llvm.add`，`mir.load` 变成
`llvm.load`，`mir.goto` 变成 `llvm.br`。有趣的是那些单个 MIR 操作展开为多个
LLVM 操作的转换，或者 GPU 特有的考量完全改变翻译方式的转换。

### 带检查的算术

在 debug 构建中，Rust 会检查每个整数算术操作的溢出。MIR 用像
`mir.checked_add` 这样返回 `(result, overflow_flag)` 元组的操作来建模这一点。
LLVM 没有这样的概念，但它确实有溢出内联函数：

```text
MIR:  %result = mir.checked_add %a, %b : i32  → mir.tuple<i32, bool>

LLVM: %sum = add i32 %a, %b
      %overflow = extractvalue {i32, i1} @llvm.sadd.with.overflow.i32(%a, %b), 1
```

溢出标志会送入一个断言，MIR importer 已经把它下降为指向不可达 block 的条件
分支。在 GPU 上，这实际上意味着：如果你设法触发了整数溢出，kernel 就会陷入
trap。这算不上最优雅的错误处理，但 CUDA 工具链目前不支持栈展开。

### 共享内存

CUDA 中的共享内存是 block 作用域的 SRAM —— 快、小、静态声明。在 `dialect-mir`
中，它是一个 `mir.shared_alloc` 操作。在 LLVM IR 中，共享内存必须是地址空间 3
中的模块级全局变量：

```text
MIR:  %shmem = mir.shared_alloc : mir.array<f32, 256>

LLVM: @shmem_0 = addrspace(3) global [256 x float] zeroinitializer
      %ptr = addrspacecast [256 x float] addrspace(3)* @shmem_0 to ptr
```

`addrspacecast` 产生一个通用指针，函数其余部分可以使用它而不必担心地址空间。
LLVM 中的 NVPTX 后端处理其余部分 —— 它知道 `addrspace(3)` 意味着共享内存，
并生成相应的 `st.shared` / `ld.shared` 指令。

### 枚举下降

Rust 枚举在代数上很丰富。LLVM 没有带标签联合的概念。下降 pass 通过把枚举表示
为带两个字段的结构体来弥补这一差距：一个判别值（告诉你哪个变体是活动的）和
一块按最大变体确定大小的载荷区域：

```text
MIR:  %opt = mir.construct_enum "Some", (%val) : mir.enum<"Option_i32">

LLVM: %tmp = insertvalue { i8, [4 x i8] } zeroinitializer, i8 1, 0
      %result = insertvalue { i8, [4 x i8] } %tmp, <val into payload area>
```

判别值是 `i8 1`，因为 `Some` 是 `Option` 的变体 1。载荷是 `[4 x i8]` —— 四个
字节，足以容纳一个 `i32`。变体访问按相反方向工作：读取判别值，对它分支，然后
`extractvalue` 载荷并 bitcast 到期望的类型。

这不优雅，但这正是 C 编译器几十年来处理带标签联合的方式。LLVM 的优化器非常
擅长清理冗余的 insertvalue/extractvalue 链。

---

## GPU 内联函数转换

`dialect-nvvm` 操作 —— 线程索引、warp shuffle、屏障、TMA 批量拷贝 —— 不会被
下降为通用的 LLVM 方言操作。它们被下降为 LLVM 内联函数调用或内联 PTX 汇编，
取决于 LLVM 是否为该操作提供了内建内联函数。

### 策略 1：LLVM 内联函数调用

对于 LLVM 已经提供目标专用内联函数的操作，转换会发出对该内联函数的 `call`：

```text
nvvm.read_ptx_sreg_tid_x
  → call i32 @llvm_nvvm_read_ptx_sreg_tid_x()

nvvm.shfl_sync_bfly_i32
  → call i32 @llvm_nvvm_shfl_sync_bfly_i32(i32 -1, i32 %val, i32 %mask, i32 31)
```

注意这个 warp shuffle：面向用户的 `cuda_device` API 接受两个参数（值和 lane
掩码），但 LLVM 内联函数接受四个（membermask、value、delta、clamp）。下降 pass
补上了缺失的参数 —— `membermask = -1`（所有 lane）和 `clamp = 31`（完整 warp
宽度）—— 因此用户从不需要考虑它们。

### 策略 2：内联 PTX 汇编

较新的 GPU 指令往往缺乏 LLVM 内联函数。对于这些指令，下降 pass 使用 LLVM 的
`asm` 语法发出内联 PTX 汇编：

```text
nvvm.wgmma_fence_sync
  → call void asm sideeffect convergent "wgmma.fence.sync.aligned;", ""()

nvvm.mbarrier_arrive
  → call i64 asm sideeffect convergent "mbarrier.arrive.shared.b64 $0, [$1];", "=l,r"(ptr %bar)
```

这里的 `convergent` 属性至关重要。它告诉 LLVM："不要把这行指令跨控制流移动、
复制或推测执行。"没有它，LLVM 可能会把屏障提升出条件分支，或者把一条 warp 级
指令下沉到同步点之后，结果是 GPU 挂起或计算出垃圾 —— 这两者都不会产生有用的
错误消息。

---

## block 参数到 PHI 节点

Pliron IR（类似 MLIR）使用 **block 参数**在基本块之间传递值。LLVM 使用 **PHI
节点**。它们表达相同的概念 —— "这个值来自不同的前驱" —— 但语法差异足够大，
导出步骤需要一次真正的变换，而不仅仅是美化打印。

Pliron 风格（block 参数）：

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

导出器用两遍的方法处理这一转换：

1. **预 pass：为每个值命名。** 在发出任何代码之前，导出器遍历所有 block，为
   每个值分配顺序的 SSA 名（`%v0`、`%v1`、……）。这至关重要，因为 PHI 节点
   可以引用在列表中*较后*出现的 block 中的值 —— 循环回边在文本中向前指，但在
   控制流中向后指。若不预先命名，那些引用将是未定义的。

2. **构建前驱映射。** 对于每个 block，导出器通过检查函数中的每条分支指令来
   收集 `(predecessor_block, values_passed)` 对。

3. **发出 PHI 节点。** 在每个非入口 block 的入口处，导出器为每个 block 参数
   发出一个 PHI 节点，用前驱映射中的值和前驱标签填充。

预 pass 是微妙的部分。考虑一个循环：循环头部的 PHI 引用来自循环体的
`%new_sum`，但循环体在文本输出中出现在头部*之后*。如果我们在发出过程中即时
分配名称，`%new_sum` 就还没有名称。预 pass 通过提前为一切命名消除了这个问题。

---

## 调试位置

rustc 已经知道大多数 MIR 操作来自源程序中的何处。cuda-oxide 在下降 kernel 时
保留该信息：

```text
rustc Span
  ↓
pliron Location
  ↓
LLVM !dbg metadata
  ↓
PTX .loc / DWARF
  ↓
cuda-gdb source line
```

常见情形是源码中与函数同文件的一行：

```llvm
define ptx_kernel void @vecadd(...) !dbg !func {
  %idx = call i64 @cuda_device____internal__index_1d(...), !dbg !loc
}

!loc = !DILocation(line: 39, column: 13, scope: !func)
```

这表示："这条指令属于第 39 行第 13 列，在这个函数内部。"它刚好足以让 cuda-gdb
停在正确的 Rust 行上。

### 跨文件 span

有些源码 span 指向辅助文件。例如，`src/main.rs` 中的某个 kernel 在调用
`thread::index_1d()` 时可能踏入 `cuda-device/src/thread.rs`。

导出器绝不能把 `thread.rs:292` 直接附加到 kernel 的 `src/main.rs` 作用域上。
那会告诉调试器"main.rs 的第 292 行"，而这正是那种把调试变成考古的小谎言。

相反，cuda-oxide 创建一个文件专用的调试作用域：

```text
src/main.rs function scope
  |
  +-- cuda-device/src/thread.rs file scope
        |
        +-- line 292, column 19
```

在 LLVM 元数据中这看起来像：

```llvm
!thread_scope = !DILexicalBlockFile(scope: !kernel_scope, file: !thread_file, discriminator: 0)
!loc = !DILocation(line: 292, column: 19, scope: !thread_scope)
```

当 Pliron 给出显式的调用点位置时，导出器还会记录辅助代码是从何处被触达的：

```llvm
!call = !DILocation(line: 39, column: 13, scope: !kernel_scope)
!loc = !DILocation(line: 292, column: 19, scope: !thread_scope, inlinedAt: !call)
```

把它读作："我们正在执行 `thread.rs:292`，它是从 `main.rs:39` 触达的。"

### 变量

完整调试模式在行位置之上添加了第一批变量记录：

```llvm
%tid_slot = alloca i32
call void @llvm.dbg.declare(metadata ptr %tid_slot, metadata !tid, metadata !DIExpression())

!tid = !DILocalVariable(name: "tid", scope: !func, file: !file, line: 31, type: !u32)
```

模型是每个 debug 构建都使用的那一个：

```text
dbg.declare -> "this source variable lives at this address"
```

由于该地址在变量的整个作用域内是稳定的，cuda-gdb 可以在该作用域内任意断点处
读取它。为了让那些地址保持真实，full-debug 是一种 `-G` 风格的构建：它**跳过**
Pliron `mem2reg`（以便栈槽存活），**跳过** LLVM `opt -O2`，并以 `-O0` 运行
`llc`。把一个局部变量提升为 SSA 值会把它可检查的范围缩小到其寄存器的活跃期，
而这正是一次优化构建最终把作用域内的局部变量显示为 `<optimized out>` 的原因。

聚合局部变量（结构体、元组、数组）会得到一个 `DICompositeType`，其成员携带
rustc 的真实布局偏移，因此 `info locals` 会打印它们的字段：

```text
out = DisjointSlice {ptr: 0x..., len: 1}
```

Pliron `mem2reg` 仍然有一条感知提升的挽救路径：当它提升一个带调试标记的栈槽时，
它会发出一个 `mir.dbg_value`（"这个源变量在此处有这个 SSA 值"），后者下降为
`llvm.dbg.value`。当前的 `full` 层级不运行 `mem2reg`，因此以栈槽为后端的局部
变量通常依赖 `dbg.declare`。

`ReferencePropagation` 可以把一次指针赋值替换为一个内部 `AssignRef` 事件：

```text
ptr = &slice[i] -> AssignRef(ptr, slice[i])
```

完整模式会跨 stable MIR 携带该事件。对于受支持的借用切片索引形式，它在同一
程序点把地址写入该变量的调试器栈槽：

```text
AssignRef -> address of slice[i] -> ptr's debugger slot
```

导入器先校验完整的事件集。无效或不支持的事件会在翻译之前省略该目标，因此不会
发出部分插桩。默认、行表和 release 构建保持不变。提升挽救仍然是为未来的*优化*
调试层级打下的基础。

### 变量作用域与内联

变量元数据还有一条额外规则：`arg: 1` 表示"由这个调试作用域命名的函数的第 1 个
形式参数"。在 MIR 内联之后，一个 kernel 函数体中的两个变量都可以诚实地是
`arg: 1`：

```text
caller kernel:  data  -> arg #1 of caller
inlined helper: self  -> arg #1 of helper
```

所以 cuda-oxide 会随每个被翻译的函数一起携带 rustc 的 MIR 源作用域表。局部调试
记录只存储它来自的源作用域 id；LLVM 导出器把该 id 变成正确的 DWARF 形状：

```text
kernel DISubprogram
  |
  +-- helper DISubprogram
        |
        +-- DILocation(..., inlinedAt: kernel callsite)
```

这就是为什么 `data` 和 `self` 都能保持 `arg: 1`，而 LLVM 不会以自相矛盾为由
拒绝该模块。

当前的放置策略是保守的。如果 `mem2reg` 创建了一个行为类似 PHI 值的 block 参数，
cuda-oxide 会在被提升的 load/store 源点记录该变量。它暂时还不会仅仅因为编译器
创建了一个合并值就添加一条急切的 block 入口调试记录：

```text
then -> merge(10)
else -> merge(20)

merge(%x):
  ... old load site becomes dbg.value(%x, "x") ...
```

这是正确的，但不是最大程度的精确。一旦源位置和词法作用域规则明确，未来的 pass
可以让 `x` 在合并 block 中更早可见。

还有一个给编译器维护者的细节：`mir.dbg_value` 仅用于调试，但在今天的 Pliron
内部它仍然有一个普通的操作数使用。因此 cuda-oxide 在这个调试挽救点之后不运行
Pliron DCE。如果我们以后添加它，DCE 需要把调试使用视为非语义的，就像 LLVM
对待 `dbg.value` 那样。

导出器生成的 PHI 节点通常不会获得 `!dbg`：它们是 SSA 簿记，而不是用户编写的
源操作。后续的变量元数据可以通过 `dbg.value` 引用 PHI 值，但 PHI 本身不应在
cuda-gdb 中成为一个意外的停靠点。

---

## 符号名净化

函数名流经若干阶段，每个阶段施加自己的约束：

```text
rustc_public (FQDN)            helper_fn::cuda_oxide_device_<hash>_vecadd
  ↓ body.rs (:: → __)
dialect-mir                    helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ call.rs (:: → __)
LLVM dialect                   helper_fn__cuda_oxide_device_<hash>_vecadd
  ↓ llvm-export (strip prefix)
Textual LLVM IR                @vecadd
  ↓ llc
PTX                            vecadd
```

这条路径上发生了三次转换：

1. **`::` 到 `__`** —— `body.rs`（函数定义）和 `call.rs`（调用目标）都把 Rust
   路径分隔符替换为双下划线，以产生合法的 pliron/LLVM 标识符。由于两侧施加
   相同的转换，定义和调用点相互匹配。

2. **设备前缀剥离** —— `export.rs` 通过
   `reserved_oxide_symbols::device_base_name` 从 `#[device]` 函数名中剥离保留的
   `cuda_oxide_device_<hash>_` 前缀（以及任何在前的 FQDN crate 前缀）。该前缀
   用于 MIR 级检测，但不应出现在最终的 LLVM IR、PTX 或 LTOIR 输出中。

3. **设备 extern 前缀剥离** —— 对于 `#[device] unsafe extern "C"` 函数，
   `call.rs` 通过 `reserved_oxide_symbols::device_extern_base_name` 剥离
   `cuda_oxide_device_extern_<hash>_` 前缀，使 LLVM IR 引用由外部 LTOIR
   导出的原始符号名（例如 CCCL 库）。

> **注意**
>
> 当框架升级后，这种手工净化将被 pliron 的 `Legaliser` 取代。`Legaliser`
> 会系统化地处理 `::` 到 `_` 的转换以及碰撞检测。

---

## PTX 生成

在 LLVM 方言被导出为文本 `.ll` 文件之后，最后一步是调用 `llc` —— LLVM 的静态
编译器 —— 来产生 PTX 汇编：

```bash
llc -march=nvptx64 -mcpu=sm_90 kernel.ll -o kernel.ptx
```

### 目标选择

流水线按以下顺序探测 `llc`。LLVM 21 是最低要求 —— 更早的版本会拒绝 cuda-oxide
发出的 TMA / tcgen05 / WGMMA 内联函数签名。

| 优先级 | 来源                                                      | 备注                                                                  |
| :----- | :-------------------------------------------------------- | :-------------------------------------------------------------------- |
| 第 1   | `$CUDA_OXIDE_LLC`（若已设置）                             | 调用方提供的覆盖；指向哪个二进制就用哪个。                             |
| 第 2   | Rust 工具链的 `llvm-tools` llc                            | `<sysroot>/lib/rustlib/<host>/bin/llc`（通过 `rustup` 自动安装）。     |
| 第 3   | `PATH` 上的 `llc-23`                                      | 发行版 / `apt.llvm.org` 安装的 LLVM 23。                               |
| 第 4   | `PATH` 上的 `llc-22`                                      | 发行版 / `apt.llvm.org` 安装的 LLVM 22。                               |
| 第 5   | `PATH` 上的 `llc-21`                                      | 发行版 / `apt.llvm.org` 安装的 LLVM 21。                               |
| 第 6   | `PATH` 上的 `llc`                                         | 仅作报告用的回退；若早于 LLVM 21 则在运行时报错。                       |

所锁定的 Rust 工具链（`nightly-2026-10-11`）附带启用了 NVPTX 的 LLVM 23，
因此 `rustup component add llvm-tools` 是推荐的上手路径。针对 `llc-23` /
`llc-22` / `llc-21` 的 PATH 探测作为回退保留给已有 LLVM 安装的用户。如果所有
探测都失败，流水线会以清晰的错误失败。你可以通过设置
`CUDA_OXIDE_LLC=/path/to/llc` 选择特定的（可能更旧的）二进制，但在 LLVM 20
及以下只有简单 kernel 保证能编译。

如果所选目标与物理 GPU 不匹配，CUDA 驱动会在加载时 JIT 编译 PTX。首次启动
大约花费 30ms 来让驱动翻译；后续启动使用缓存的二进制。实践中你很少会注意到
—— JIT 很快，而且缓存在多次运行之间是持久的。

在你需要特定目标的情况下，`CUDA_OXIDE_TARGET` 环境变量会覆盖自动检测。例如，
`sm_100a` 启用通用 `sm_100` 目标下不可用的 Blackwell 专用 `tcgen05` 特性。

`cargo oxide run` 在后端基于 feature 的默认值之上增加了一层自动检测：当既未
设置 `--arch` 也未设置 `CUDA_OXIDE_TARGET` 时，它会查询 CUDA 设备 0 的计算能力
并将其转发给后端，从而保证生成的模块能在本地 GPU 上加载。完整的优先级是
`--arch` > `CUDA_OXIDE_TARGET` > host CC（仅对 `run`）> 后端基于 feature 的
默认值。`cargo oxide build` 和 `cargo oxide pipeline` 刻意跳过 host-CC 这一步，
以便它们仍可用于交叉编译。

如果下降后的模块调用了 CUDA libdevice，cuda-oxide 会自动从 `llc` 切换到 NVVM
路径。它按相同的优先级解析目标，然后才选择类型化指针或非透明指针的 NVVM IR。
显式的 NVVM/LTOIR 命令仍然要求 `--arch`；基于 feature 的回退只在普通构建发现
需要 libdevice 时适用。

> **注意**
>
> 为什么是 LLVM 21？`tma_copy`、`gemm_sol` 和 `tcgen05_matmul` 使用的二维批量
> TMA 加载内联函数在 LLVM 21 中获得了带 `addrspace(7)` 和 `cta_group` 参数的
> 10 操作数形式。更旧的 `llc` 版本会以
> `Intrinsic has incorrect argument type!` 拒绝它。与其为每个 LLVM 版本维护
> 单独的内联函数发射器，我们把 21 设为最低要求。

### 选择匹配的 `opt`

`llc` 并不是流水线运行的唯一 LLVM 二进制：中端优化阶段需要 `opt`，而两者必须
来自**同一个 LLVM 主版本**。混用它们正是这一排序所要防止的失败 —— 一个发行版
的 `opt` 可能发出另一个发行版的 `llc` 不接受的 IR。

因此先按上表解析 `llc`，并从 `llc --version` 读出它的主版本号，然后选择匹配的
`opt`：

| 优先级 | 来源                                               | 备注                                                                  |
| :----- | :------------------------------------------------- | :-------------------------------------------------------------------- |
| 第 1   | `$CUDA_OXIDE_OPT`（若已设置）                       | 始终被尊重。与所选 `llc` 的主版本不匹配时不会被静默纠正 —— 而是记录一条诊断，同时点名这两个二进制。 |
| 第 2   | 所选 `llc` 旁边的 `opt`                             | LLVM 安装会把工具并排放置。仍会做版本检查：不匹配的同伴会被拒绝，只有在无法读取 `llc` 自身版本时才未经校验地接受它。 |
| 第 3   | Rust 工具链的 `llvm-tools` `opt`                    | `<sysroot>/lib/rustlib/<host>/bin/opt`。过滤到与 `llc` 相同的主版本。  |
| 第 4   | `PATH` 上的 `opt-22` / `opt-21` / `opt`             | 过滤到与 `llc` 相同的主版本。                                         |

如果不存在同主版本的 `opt`，解析会记录一条诊断，点名每个被拒绝的候选。独立
API 把请求的优化视为严格行为并失败；旧式 rustc 路径则回退为不优化运行。

由于第 2 步取决于哪一个 `llc` 胜出，单独锁定 `CUDA_OXIDE_LLC` 通常就足够了
—— 匹配的 `opt` 通常能在它旁边找到，只有在两者被分到不同目录时才需要同时设置
`CUDA_OXIDE_OPT`。

### 流水线寻找的另外两个输入

仅当设备代码调用进 libdevice 且 PTX 路径以 IR 级链接解析那些调用时，才需要
`llvm-link`。而 `libdevice.10.bc` 更常被需要：NVVM 路径会把它添加到它最终化的
每个模块中，无论 kernel 是否调用进 libdevice。两者都有自己的覆盖项：

| 变量 | 选择 | 未设置时的发现方式 |
| :------- | :------ | :------------------- |
| `CUDA_OXIDE_LLVM_LINK` | `llvm-link` 二进制 | 与 `opt` 相同的四步 —— 先在所选 `llc` 旁边，然后 sysroot，然后 `PATH` 上的带版本名，过滤到 `llc` 的主版本。当该变量*确实*被设置时，它像 `CUDA_OXIDE_OPT` 一样被尊重，若其主版本与 `llc` 不同则记录一条诊断并点名这两个二进制 |
| `CUDA_OXIDE_LIBDEVICE` | `libdevice.10.bc` 位码文件；不存在的路径会被静默跳过 | 对 `CUDA_TOOLKIT_PATH`、`CUDA_HOME`、`CUDA_PATH`、`/usr/local/cuda`、`/opt/cuda` 中的每一个取 `<root>/nvvm/libdevice/libdevice.10.bc` |

当任一部分缺失时，两条路径的反应不同。普通构建不会失败：后端的路径决策看到
IR 级链接不可用，于是回退到 NVVM 路径，也就是目标选择一节中描述的同一自动
切换。独立 API 的 `Linking::Libdevice` 则是严格的，会直接在一开始就失败，并
以一个 `LibdeviceUnavailable` 错误点名缺失的部分：`libdevice.10.bc`，或者一个
与所选 `llc` 主版本相同的 `llvm-link`。

## 旧版 NVVM IR 中的原子操作

"LLVM 级原子"是指在 NVVM 输入中的一条原子指令。它并非关于 GPU 是否有原子硬件
的陈述。CUDA 12 的 LLVM 7 NVVM 方言只接受一个子集：

| LLVM 构造 | CUDA 12 NVVM IR | 今天的 cuda-oxide 旧式路径 |
|:----------------|:----------------|:-----------------------------|
| 原子 load 或 store | 不作为 LLVM 原子指令支持 | 拒绝；它需要另一种下降方式 |
| `fence` | LLVM `fence` 不受支持；需要一种 NVVM 内存屏障操作 | 拒绝，等待精确的顺序/作用域映射 |
| `cmpxchg` | `i32`/`i64`，在 `compute_90+` 上还有 `i128`；全局/共享指针，或已知指向那里的通用指针 | 对通用、全局和共享地址空间中 relaxed（monotonic）顺序的 `i32`/`i64` 保持原生；成功顺序强于 relaxed，或 block/system 作用域，会被重写为内联 PTX。其他宽度（包括每种计算能力上的 `i128`）、其他地址空间，以及失败顺序强于成功顺序，都会被拒绝 |
| `atomicrmw` | 在 `i32`/`i64` 上的整数 `xchg`、`add`、`sub`、`and`、`or`、`xor`、`max`、`min`、`umax` 和 `umin`；在 `compute_90+` 上的 `i128 xchg`；相同的地址空间限制 | 对通用、全局和共享地址空间中这十种操作在 monotonic 顺序下的 `i32`/`i64` 保持原生。其他宽度、种类、顺序和地址空间会被拒绝 |
| NVVM 原子内联函数 | 提供选定的额外操作，包括浮点原子加 | 通用/全局/共享地址空间中 `f32`/`f64` 上 relaxed、设备作用域的 `atomicrmw fadd` 会被下降为精确的旧式内联函数；其他*浮点*原子 RMW 操作仍被拒绝 |

常规的 LLVM 到 PTX 路径保持其现有的原子支持。上表仅描述旧式 NVVM 合法化器。

有两条规则解释了那一列。第一条是：任何其类型、地址空间、顺序和作用域无法被
证明可存活的东西都不会被发出：[#921](https://github.com/NVlabs/cuda-oxide/pull/921)
和 [#923](https://github.com/NVlabs/cuda-oxide/pull/923) 是通过证明那些性质而
不是放松要求来接纳整数 RMW 与比较交换的，这就是为什么仍然保留的拒绝情形正是
证明不成立的那些情形。

第二条是：**作用域绝不会为了凑合而被削弱**。libNVVM 的旧式方言会丢弃它无法
表达的东西：一个有顺序的 `cmpxchg` 会下降为赤裸的 `atom.cas.b32`/`.b64`，没有
顺序限定符也没有环绕的 fence，PTX 随后会把它视为 relaxed。因此，block 或
system 作用域的整数原子操作，以及成功顺序强于 relaxed 的比较交换，会被重写为
显式声明作用域和顺序的内联 PTX，而不是交给 libNVVM。该重写需要 `sm_70` 或
更新。失败顺序强于成功顺序的情形永远不会被重写；它会被直接拒绝。

monotonic 是 `atomicrmw` 在这里能达到的唯一顺序，因为 MIR 下降已经把 Rust 更强
的顺序拆分为单独的 fence，所以指令本身不携带任何顺序。

旧式规范还会接受但忽略 `cmpxchg` 的 `weak` 标记和失败顺序。

关于确切接受的类型和操作，见
[CUDA 12.4 NVVM IR 规范](https://docs.nvidia.com/cuda/archive/12.4.0/nvvm-ir-spec/index.html)。

---

## 汇总

以下是 `lower_mir_to_llvm` 处理一个模块时完整的事件序列：

```text
1. LLVM dialect types and operations are registered automatically (link-time)
2. For each MirFuncOp in the module:
   a. Create `llvm.func` with flattened type signature
   b. inline_region: move `dialect-mir` blocks into the LLVM function
   c. Build entry prologue (reconstruct aggregates from flat args)
   d. Run DialectConversion:
      ├── Walk every `dialect-mir`/`dialect-nvvm` op (def-before-use order)
      ├── Invoke MirToLlvmConversion::rewrite for each op
      ├── Converter emits LLVM dialect op(s) via DialectConversionRewriter
      └── Framework patches block arg types automatically
3. Export the LLVM dialect to textual LLVM IR (.ll) (with PHI node conversion)
4. Invoke llc to produce .ptx
```

在第 4 步之后，你就得到了一个 CUDA 驱动可以加载并执行的 `.ptx` 文件。从
`mir.checked_add` 到 `add.s32` 的旅程完成了。

---

下降流水线把 Rust 风味的 IR 变成 GPU 就绪的 LLVM IR。关于添加新 GPU 操作的
实践性演练，见 [添加新的内联函数](adding-new-intrinsics.md)。
