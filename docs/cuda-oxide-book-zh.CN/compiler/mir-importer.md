# MIR Importer

前面几章解释了 rustc 如何产出 Stable MIR（[rustc_public](rustc-public.md)），
以及 pliron 如何提供 IR 框架（[Pliron](pliron.md)）。本章是两者相遇的地方：
`mir-importer` 接过 rustc 交给我们的 Stable MIR，把它翻译成 `dialect-mir`
—— 那个保留 Rust 语义的 pliron 方言。翻译器最初产出的是一种
alloca/load/store 形式 —— 生成成本低、容易推理，而且对 pliron 来说是恒等
映射。随后一个 `pliron::opts::mem2reg` pass 会把这些槽位提升回 SSA 形式。
接着 cuda-oxide 运行它的[编译器优化](compiler-optimizations.md)（从带注解的
循环展开开始），然后再下降到 LLVM 方言。

但翻译只是一半的工作。`mir-importer` 还编排*整条*编译流水线：翻译、校验、
优化 `dialect-mir`、下降、导出、生成 PTX。它既是翻译器，也是舞台监督。

这个 crate 位于 `cuda-oxide/crates/mir-importer`，分成两部分：

- **`translator/`** —— MIR 到 pliron 的翻译逻辑（有意思的部分）。
- **`pipeline.rs`** —— 把每个阶段串起来的编排（负责任的部分）。

---

## 流水线编排

在深入翻译细节之前，先看全景。`run_pipeline()` 函数是 `rustc-codegen-cuda`
在收集完设备函数之后调用的入口点。它接收一组 `CollectedFunction` 结构体和一个
`PipelineConfig`，然后运行这些阶段：

```text
第 1 步：  把 Rust MIR 翻译成 `dialect-mir`
第 2 步：  校验 `dialect-mir` 模块
第 3 步：  运行 `pliron::opts::mem2reg`，把所有 alloca 槽位提升回 SSA
第 4 步：  施加 `dialect-mir` 优化（目前是带注解的循环展开）
第 5 步：  把 `dialect-mir` 下降到 LLVM 方言（经由 mir-lower）
第 6 步：  对 NVVM 构建，用 nvvm-transforms 调整 LLVM 方言
第 7 步：  把 LLVM 方言导出为文本形式的 LLVM IR（.ll）
第 8 步：  用 llc 编译，或用 libNVVM 与 nvJitLink 编译
```

完整的变量调试构建会跳过第 3、4 步，好让源码变量留在稳定的内存位置上，
供 cuda-gdb 使用。

每个 `CollectedFunction` 携带流水线需要知道的关于某个设备函数的一切：

```rust
pub struct CollectedFunction {
    pub instance: Instance,
    pub is_kernel: bool,
    pub export_name: String,
}
```

`instance` 是来自 `rustc_public` 的单态化函数。`is_kernel` 区分 kernel 入口点
与设备辅助函数（kernel 会在 LLVM IR 里得到特殊元数据，好让 NVPTX 后端把它们
作为 `.entry` 点输出）。`export_name` 是最终 PTX 中出现的符号名 —— 对设备
函数来说，这通常是完全限定名（FQDN），与同一个函数上 `CrateDef::name()`
返回的名字一致。

对每个函数，流水线会：

1. 通过 `instance.body()` 取回 MIR body。
2. 调用 `translator::body::translate_body()` 生成该函数的 `dialect-mir`
   表示（对 local 使用 `mir.alloca` 槽位），然后把它追加到整次运行共享的
   那一个 `builtin.module` 上。
3. 在模块上运行 pliron 的校验器，尽早抓住结构性错误 —— 类型不匹配、操作数
   缺失、支配关系破损 —— 以免它们在下游变成晦涩的 LLVM 失败。
4. 运行 `pliron::opts::mem2reg`，把 alloca 槽位提升回 `dialect-mir` 内的
   SSA 值。
5. 运行[编译器优化](compiler-optimizations.md)，目前是展开带有 `#[unroll]`
   或 `#[unroll(N)]` 请求且受支持的循环。
6. 运行 `lower_mir_to_llvm`（来自 `mir-lower` crate），通过
   `DialectConversion` 把每一个 `dialect-mir` 操作下降到与之等价的 LLVM
   方言操作。
7. 对 NVVM 构建，以目标 GPU 选定的方言运行 `nvvm-transforms`。普通 PTX
   构建跳过这一步。
8. 把 LLVM 方言模块导出为文本 `.ll` 字符串，并用 `llc` 编译它；对 NVVM
   输出来说则是用 libNVVM 和 nvJitLink。

任何一步失败，流水线都会停下来并返回带类型的错误（`NoBody`、`Translation`、
`Verification`、`Lowering`、`Export` 或 `PtxGeneration`），并附带足够的上下文
来诊断问题。没有静默损坏，也没有神秘的空输出文件。

---

## 翻译架构

`translator/` 目录是 Stable MIR 变成 pliron IR 的地方。每个模块处理 MIR 结构
的一个层次，它们组合得很整齐：

| 模块       | 用途                                                                    |
| :----------- | :--------------------------------------------------------------------------|
| `body`       | 函数级翻译、alloca 槽位铺设、FQDN 名字清洗    |
| `block`      | 基本块翻译的协调器                                        |
| `statement`  | 语句翻译（赋值、存储）                              |
| `terminator` | terminator 翻译（goto、call、return、基于 FQDN 的内联函数分发） |
| `rvalue`     | 表达式翻译（二元运算、转换、聚合）                         |
| `types`      | Rust 类型到 `dialect-mir` 类型的转换                                 |
| `values`     | MIR local → alloca 槽位映射（`ValueMap`）+ 槽位地址空间推断    |

调用流遵循 MIR 的结构，自上而下：

```text
pipeline::run_pipeline()
  └─ body::translate_body()               // 每个收集到的函数一次
       ├─ emit_entry_allocas()            // 每个非 ZST local 一个 mir.alloca
       │     └─ SlotAddrSpaceMap::analyze // 指针槽位的地址空间推断
       └─ 对每个基本块：
            └─ block::translate_block()
                  ├─ statement::translate_statement()
                  │     └─ rvalue::translate_rvalue()
                  └─ terminator::translate_terminator()
```

`translate_body()` 搭建函数签名、创建与 MIR 基本块对应的 pliron 块、清洗
函数名（把 `instance.name()` 给出的 FQDN 中的 `::` 转成 `__`）、在入口块顶部
为每个非 ZST local 发出一条 `mir.alloca`，并把传入的函数参数存进各自的槽位。
每个非入口块都没有参数 —— 跨基本块的数据流由 alloca 槽位承载，而不是由块参数
承载。然后它顺序遍历每个块，逐条翻译语句与 terminator。

`translate_statement()` 处理块内的平坦操作 —— 赋值、storage live/dead 标记
以及判别值写入。当一个赋值涉及右侧表达式（MIR 的 `Rvalue`）时，它会委托给
`translate_rvalue()`，由后者处理二元运算、一元运算、转换、聚合构造、判别值
读取、指针算术，以及 Rust 编译出来的另外十来种东西。

`translate_terminator()` 处理块末尾的操作：`Goto`、`SwitchInt`、`Call`、
`Return`、`Assert`、`Drop` 和 `Unreachable`。内联函数分发也住在这里 ——
但它有自己的小节，见下文。

---

## SSA 难题（以及我们如何推迟它）

这是翻译中最棘手的部分，值得仔细解释。

Rust MIR *不*处于严格的 SSA 形式。local（变量）是有名字的存储位置，任何块
都能读或写。如果 `_3` 在 `bb0` 里被赋值，它可以被 `bb1`、`bb5` 或任何其它
地方随意使用 —— MIR 不在乎。

Pliron IR（类 MLIR）最终期待的是**严格 SSA**：一个值必须支配它的每一次使用，
如果一个值需要从一个块流到另一个块，就必须把它作为**块参数**显式传递。你
不能直接跨块去抓一个 local。

我们用两个阶段化解这个矛盾：

1. **importer：alloca + load/store。** `mir-importer` 刻意*不*直接构造 SSA。
   每个非 ZST 的 MIR local 都由单个栈槽支撑 —— 在入口块顶部发出一条
   `mir.alloca` 并记录在 `ValueMap` 里。对该 local 的每次写入变成向它槽位的
   `mir.store`；每次读取变成 `mir.load`。因此分支 terminator 全都**零操作数**，
   每个非入口块都**没有参数**：所有跨块数据流都经由 alloca 槽位流动，而不是
   经由块参数。

2. **`pliron::opts::mem2reg`：槽位 → SSA。** 在 `dialect-mir` 模块通过校验
   之后，`pipeline.rs` 运行 pliron 内置的 `mem2reg` pass。它把每一个合格的
   alloca 提升回 SSA 值，把每次 load 改接到到达的定义上，并在值沿多条控制
   路径汇合的地方插入块参数（pliron 表述 phi 节点的方式）。取过地址的槽位
   —— 那些我们确实需要留在栈上的 —— 原样保留，留给 `dialect-mir` → LLVM
   方言的下降去翻译成真正的 `alloca`。

下面是这个问题的缩微版。给定 `_1` 在 `bb0` 中写入、在 `bb1` 中读取的 MIR：

```text
// Rust 的 MIR
bb0: { _1 = 42_i32; goto -> bb1; }
bb1: { _0 = _1;     return; }
```

importer 先产出这种基于 alloca 的 `dialect-mir`（在 `mem2reg` 之前）：

```text
^bb0:
  %s1 = mir.alloca           : !mir.ptr<i32>
  %c  = mir.constant 42_i32  : i32
  mir.store %c, %s1
  mir.goto ^bb1                     // 零操作数；_1 经由 %s1 流动
^bb1:                               // 没有块参数
  %r = mir.load %s1 : i32
  mir.return %r : i32
```

在 `pliron::opts::mem2reg` 提升 `%s1` 之后，同一个函数变成 SSA 形式的
`dialect-mir`，只有在确实需要汇合到达定义的地方才会出现块参数：

```text
^bb0:
  %c = mir.constant 42_i32 : i32
  mir.goto ^bb1(%c : i32)
^bb1(%r : i32):
  mir.return %r : i32
```

（像上面这样后继只有一个前驱的函数，最终会得到一个块参数；而存在多个到达
定义的汇合点，才是 `mem2reg` 引入那些不平凡的 phi 式参数的地方。）

要点是：importer 从不做活跃性分析，也从不在块之间穿引值。所有"哪个值在
哪里是活跃的？"的推理都推迟给 `mem2reg`，它一个 pass 就能为整个 `dialect-mir`
模块正确解决这个问题。`translator/terminator/mod.rs` 的模块文档字符串里以
内联源码形式带了同一个完整示例，便于快速查阅。

---

## 类型翻译

`types` 模块把（通过 `rustc_public` 看到的）Rust 类型转换成 `dialect-mir`
类型。大多数映射都很直接，但有几个值得留意：

| Rust 类型             | `dialect-mir` 类型            | 说明                                |
| :-------------------- | :---------------------------- | :----------------------------------- |
| `i32`、`u64` 等    | `IntegerType`                 | 带符号性跟踪             |
| `f32`, `f64`          | `Float32Type` / `Float64Type` | 标准 IEEE 754                    |
| `bool`                | `IntegerType(1)`              | 1 位整数，一如既往       |
| `(A, B, C)`           | `MirTupleType`                | 异质的积类型           |
| `&[T]`                | `MirSliceType`                | 指针 + 长度                     |
| `DisjointSlice<T>`    | `MirDisjointSliceType`        | 带边界检查、带类型索引的可变切片 |
| `struct Foo`          | `MirStructType`               | 字段偏移来自 rustc 布局 |
| `*mut T` / `*const T` | `MirPtrType`                  | 带 GPU 地址空间               |
| `enum Option<T>`      | `MirEnumType`                 | 判别值 + 变体              |

### 动态结构体布局

这里有一个微妙之处，能为用户省下真实的头疼。考虑这个结构体：

```rust
struct Extreme {
    a: u8,
    b: i128,
}
```

Rust 的布局算法可能为了对齐而重排字段：

```text
用户写的：        struct Extreme { a: u8, b: i128 }
rustc 可能布局：  [b: i128 @ offset 0][a: u8 @ offset 16]
MirStructType：   mem_to_decl 映射、偏移、total_size
LLVM struct：     { i128, i8, [15 x i8] }   // 显式填充
```

cuda-oxide 会向 rustc 查询每个字段精确的字节偏移，并带着显式填充字节构造
结构体类型。`MirStructType` 存储一个 `mem_to_decl` 映射（内存顺序到声明
顺序）、逐字段偏移以及总大小。下降到 LLVM 时，填充会被实体化为字段之间的
`[N x i8]` 数组。

实际的好处是：在宿主与设备代码之间共享的类型**不要求 `#[repr(C)]`**。
cuda-oxide 自动匹配 rustc 的布局，所以你的结构体可以用 Rust 默认的
`repr(Rust)` 布局，编译器会在两边都做正确的事。少一个要记住的属性，也就少
一个会踩到的坑。

---

## 内联函数分发

当翻译器遇到一个 `Call` terminator 时，它不会立刻发出一条 `mir.call` 操作。
它首先检查被调用者是不是一个*已知内联函数* —— 也就是来自 `cuda_device`、
直接映射到某条 GPU 硬件指令、而不是有函数体的函数。

`try_dispatch_intrinsic()` 函数按被调用者的**完全限定名（FQDN）**匹配，名字
来自 `CrateDef::name()`：

```rust
match name {
    // 精简版；真正的分支还会匹配 re-export 与 ABI 别名路径
    "cuda_device::thread::threadIdx_x"  => emit_generated_nvvm_intrinsic(ReadPtxSregTidXOp),
    "cuda_device::thread::sync_threads" => { /* 原地构建一个 Barrier0Op */ }
    // ... 100 多个内联函数
    _ => translate_as_normal_call()
}
```

大多数分支不是手写的。任何由 `intrinsics/catalog.json` 描述的东西，其分支都会
被生成到 `cuda-oxide/crates/mir-importer/src/translator/terminator/intrinsics/generated/`
下按族划分的模块中。上面两个分支都属于这一类，warp shuffle 也是：匹配的
字符串是带掩码的 `_sync` 形式（`shuffle_xor_sync`、`shuffle_xor_f32_sync`、
`shuffle_xor_u64_sync`，以及 `shuffle_up_*`/`shuffle_down_*` 系列）。
裸的 `warp::shuffle_xor` 永远不是匹配字符串。它是
`cuda-oxide/crates/cuda-device/src/warp.rs` 里的一个 `#[inline(always)]`
包装函数，负责提供整 warp 的掩码（`u32::MAX`）并调用 `shuffle_xor_sync`，
所以分发器只会拦截 `_sync` 调用。`terminator/mod.rs` 里手写的分支以及生成
文件旁边的 `intrinsics/` 模块，装的是目录文件没有描述的那些情况。

匹配时使用完整的 FQDN（例如 `cuda_device::thread::threadIdx_x`，而不只是
`threadIdx_x`），以避免不同模块中同名函数的歧义。同一个 FQDN 也被用作
非泛型、非内联函数调用的调用目标名 —— 收集器产出匹配的名字，下降层在两侧
都把 `::` 转成 `__`。

如果该函数是一个被识别的内联函数，翻译器会直接发出对应的 `dialect-nvvm`
操作 —— 没有函数体，没有调用开销，只有那条硬件指令。线程索引、warp shuffle、
barrier、共享内存操作、TMA 批量拷贝以及矩阵乘指令，全都走这条路径。

如果该函数*不是*内联函数，它就落到普通路径上：发出一条 `mir.call` 操作，
按符号名引用被调用者。被调用者的函数体已经单独翻译过（它也在收集到的函数
列表里），所以一切都能接上。

> **注意**
>
> 内联函数分发表是往 cuda-oxide 里添加新 GPU 操作的主要扩展点。如果 NVIDIA
> 发布了新指令而你想把它暴露出来，就给 `cuda_device` 加一个函数、加一个
> `dialect-nvvm` 操作，并在这里加一个匹配分支。分步指南见
> [添加新的内联函数](adding-new-intrinsics.md)。

---

## 处理展开路径

MIR 忠实地建模了 Rust 的 panic 语义。每次函数调用都有两个可能的后继 ——
一个返回目标和一个展开（unwind）目标：

```text
_2 = mul(_1, _3) -> [return: bb1, unwind: bb2]
```

在 CPU 上，展开路径很重要：它运行析构函数、展开栈，然后要么捕获 panic，
要么中止进程。在 GPU 上，CUDA 工具链今天并不暴露这种能力 —— nvcc/ptxas 会
剥掉 landing pad，没有任何异常处理基础设施能活到 PTX。硬件本身*可以*支持
展开（Volta 之后的绝对分支加逐线程调用栈追踪就足够了），但编译器和运行时
没有把它接起来。NVIDIA 有一个进行中的项目要为汽车安全领域添加 C++ 异常
支持；cuda-oxide 当前的设计与那项工作是向前兼容的。

目前，cuda-oxide 把**所有展开路径都视为不可达**。如果运行时真的发生了 panic
—— 比如 debug 模式下的整数溢出，或者显式的 `panic!()` —— GPU 会 trap，
kernel 会崩溃。这在语义上等价于 `panic=abort`，而不需要用户去设置那个标志。

实践中，翻译器只是在每个 `Call` 与 `Assert` terminator 里忽略展开目标，
只生成返回路径上的分支。那些展开块永远不会被翻译。它们消失了，就像从未
存在过。

panic 的*消息*也随之消失。一个 terminator 是对 `core::panicking` 发散调用的
块，会下降成 `nvvm.trap` + `mir.unreachable`，别无其它 —— 它的语句存在的唯一
目的就是构造那个被丢弃的调用会消费的东西（消息 `&str`、`format_args!` 的各个
片段），而该块没有后继，所以它们写的东西永远不可能被读到。跳过它们也正是
带消息的 panic 能编译的前提：实体化的 `&str` 常量没有对应的设备下降。

这并没有听起来那么可怕。Rust 的借用检查器和类型系统阻止了大多数会导致 panic
的 bug。至于漏过去的那些（数组边界检查、对 `None` 做 unwrap），GPU trap
本来就是正确的行为 —— GPU 线程在 kernel 执行到一半时面对一个逻辑错误，本来
也无法做任何有用的"恢复"。

---

## 把它们串起来

我们跟踪一个简单的 kernel 走完整个流水线，看看各块如何连接。下面是一个向量
加法 kernel：

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

在 `mir-importer` 把 Stable MIR 翻译成 `dialect-mir`（并且
`pliron::opts::mem2reg` 把 alloca 槽位提升回 SSA）之后，结果大致长这样
（已简化，为清晰起见省略了大量细节）：

```text
mir.func @vecadd(%a: mir.slice<f32>, %b: mir.slice<f32>,
                 %c: mir.disjoint_slice<f32>) {
^entry:
    %idx = nvvm.read_ptx_sreg_tid_x : i32
    %len = mir.extract_field %c[1]       // 切片长度
    %in_bounds = mir.lt %idx, %len
    mir.cond_br %in_bounds, ^compute, ^exit

^compute:
    %a_val = mir.load ...                // a[idx]
    %b_val = mir.load ...                // b[idx]
    %sum = mir.add %a_val, %b_val : f32
    mir.store %sum, ...                  // c[idx] = sum
    mir.goto ^exit

^exit:
    mir.return
}
```

有几处值得注意：

- **`thread::index_1d()`** 被当作内联函数分发，变成了
  `nvvm.read_ptx_sreg_tid_x` —— 一次直接的 GPU 寄存器读取，而不是函数调用。
- **`DisjointSlice::get_mut()`** 变成一个边界检查（`mir.lt`）加一次条件
  分支。Rust 中的 `if let Some` 模式变成了显式的控制流。
- **这里没有块参数。** 记住 `mir-importer` 自己发出的*每一个*分支 terminator
  都是零操作数，每个非入口块都没有参数 —— 值通过 alloca 槽位跨越块边界，
  直到 `pliron::opts::mem2reg` 运行。在这个函数里，`mem2reg` 可以看出
  `^compute` 或 `^exit` 处没有任何东西需要汇合（分支两侧都没有值存活），
  所以它把槽位提升掉，而没有引入任何块参数。一个更循环化、且有值存活跨过
  回边的 kernel，在 `mem2reg` 之后会得到 `mir.goto ^header(%i, %acc)` 和
  `^header(%i: i32, %acc: f32)`，也就是 pliron 表述 phi 节点的方式。
- **没有展开路径。** 原始 MIR 在每个可能 panic 的操作上都有展开目标。
  它们都不见了。

从这里开始，流水线接手：

1. **校验** —— pliron 检查每个操作的类型是否匹配、每个块的参数是否正确、
   支配关系是否成立。
2. **优化** —— `mir-transforms` 改写受支持且带注解的循环，并清理它改动的
   控制流。
3. **下降** —— `lower_mir_to_llvm` 把 `mir.add` 变成 `llvm.fadd`、把
   `mir.load` 变成 `llvm.load`、把 `mir.slice` 变成指针加长度的 LLVM
   结构体，等等。
4. **按需准备 NVVM IR** —— `nvvm-transforms` 把操作转换成所选 libNVVM
   目标接受的 LLVM 形式。普通 PTX 构建跳过这一步。
5. **导出** —— LLVM 方言被打印成文本 `.ll` 文件，并带上相应的
   `!nvvm.annotations` 元数据，把 `vecadd` 标记为 kernel 入口点。
6. **编译** —— LLVM 的 NVPTX 后端通常把 `.ll` 编译成 `.ptx`。NVVM 构建
   则改用 libNVVM 和 nvJitLink。结果写在宿主二进制旁边。

`dialect-mir` 忠实地捕捉 Rust 语义。在下降它之前，cuda-oxide 可以做有针对性
的、保持行为的改写。第一个见[编译器优化](compiler-optimizations.md)。
随后[下降流水线](lowering-pipeline.md)把结果转换成 LLVM 能理解的东西。
