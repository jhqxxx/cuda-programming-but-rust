# Pliron —— Pliron IR（类 MLIR）

cuda-oxide 并非一步将 Rust 直接编译成 PTX。它通过一系列中间表示逐步降级代码，每种中间表示捕获不同层次的抽象。实现这一点的框架是 **pliron** —— 一个用纯 Rust 编写的可扩展编译器 IR 框架，受 LLVM 的 MLIR 启发。我们将在该框架之上构建的 IR 称为 **Pliron IR**。

本章解释上游 MLIR 是什么、为何 pliron 作为替代方案存在，以及其核心数据结构如何工作。如果你从未构建过编译器，不用担心 —— 这里不需要编译器经验，只需要熟悉 Rust。

## 什么是 MLIR（以及你为什么要关心它）？

LLVM IR 是一个固定的指令集。它有大约 70 个操作码（`add`、`load`、`br`、`getelementptr` 等等），如果你的领域不能干净地映射到这些操作码上，那就糟糕了 —— 你将高层概念压平为底层指令，希望优化器能重建你的意图。

**MLIR**（多级中间表示）采用了不同的方法。MLIR 不提供单个固定的指令集，而是为你提供一个定义*许多*指令集的框架 —— 这些指令集称为**dialect** —— 每个dialect都针对特定领域定制。dialect是操作、类型和属性的集合，它们对你实际关心的概念进行建模。

关键思想很简单：

1. 定义一种dialect，其中包含与你的领域匹配的操作。
2. 编写将一种dialect转换为另一种dialect的 **pass**（降级）。
3. 将 pass 串联起来，直到得到后端可以消费的东西。

考虑 Triton（PyTorch 背后的 GPU 编译器）如何使用 MLIR。Python GPU 代码首先降级为 **TTIR**，这是一种张量级 IR，其中像“将此标量广播到整个张量”这样的操作是一等公民。在这个层次上，Triton 可以应用领域特定的优化 —— 例如，将 `splat + mul`（广播标量，然后逐元素相乘）替换为单个本地向量-标量乘法。当你在 IR 中拥有张量操作时，这种优化很容易表达，而在压平为 LLVM IR 之后几乎不可能恢复。

cuda-oxide 面临类似的挑战。我们需要表示三个非常不同的抽象层次：

| 层级              | 建模内容                                                      | 示例操作                                     |
| :---------------- | :------------------------------------------------------------ | :------------------------------------------- |
| **`dialect-mir`** | Rust 语义 —— 元组、枚举、切片、检查算术                        | `mir.extract_field`、`mir.get_discriminant`  |
| **`dialect-llvm`** | 接近机器的操作 —— 整数运算、内存、控制流                      | `llvm.add`、`llvm.load`、`llvm.br`           |
| **`dialect-nvvm`** | GPU 内联函数 —— 线程索引、线程束洗牌、TMA、WGMMA               | `nvvm.read_ptx_sreg_tid_x`、`nvvm.shfl_sync` |

如果没有可扩展的 IR，我们将不得不要么将 Rust 枚举塞进 LLVM IR（丢失语义信息），要么为流水线的每个层级构建单独的 IR 框架。MLIR 允许我们在一个系统中将所有三个定义为dialect，并通过类型化的 pass 在它们之间进行降级。

## pliron 登场

MLIR 的dialect和降级模型是 cuda-oxide 的正确抽象：在 IR 中保持高层语义，同时逐步向 GPU 后端可以消费的代码降级。问题在于实现的契合度。上游 MLIR 是 LLVM C++ 生态系统的一部分，而 cuda-oxide 是一个围绕 Rust crate、Rust 类型和 Cargo 工作流构建的 Rust 编译器项目。

[Pliron](https://github.com/vaivaswatha/pliron) 是一个受 MLIR 启发的、用**纯 Rust** 编写的可扩展编译器 IR 框架。它遵循相同的概念模型 —— 操作、区域、基本块、类型、属性、dialect、pass —— 同时自然地融入 Rust 原生的编译器栈。

| 方面              | pliron                                   | 上游 MLIR                      |
| :---------------- | :--------------------------------------- | :----------------------------- |
| 语言              | 纯 Rust                                  | C++ 及 Python 绑定             |
| 构建系统          | `cargo build`                            | CMake + 完整 LLVM 构建          |
| 需要的 DSL        | 无 —— 只需 Rust 宏                       | TableGen、ODS                  |
| 调试              | 标准 Rust 工具（`dbg!`、rust-gdb）       | 在 C++ 模板上使用 gdb          |
| 可扩展性          | 以 Rust crate 形式添加dialect               | 针对 LLVM 头文件的 C++ 扩展    |
| 依赖重量          | 一个 crate（git 依赖）                   | 数 GB 的 LLVM 构建产物         |

对于 cuda-oxide，这意味着整个编译器 —— 从 MIR 导入到 LLVM IR 导出 —— 与项目的其余部分保持在同一个 Rust 构建和调试环境中。Pliron 为我们提供了所需的可扩展 IR 结构，而无需将编译器移出 Rust 生态系统。

## 核心数据结构

Pliron 的 IR 由几个核心类型构建而成。理解这些是阅读（和编写）dialect代码的关键。

### Context

`Context` 是拥有*所有* IR 数据的中心数据结构 —— 操作、基本块、区域、类型、属性以及dialect注册。可以把它看作整个编译单元的 arena 分配器。

```rust
// 简化 —— 真正的 Context 有更多字段
pub struct Context {
    operations: SlotMap<Ptr<Operation>, Operation>,
    basic_blocks: SlotMap<Ptr<BasicBlock>, BasicBlock>,
    regions: SlotMap<Ptr<Region>, Region>,
    dialects: HashMap<DialectName, Dialect>,
    type_store: TypeStore,     // 唯一化（去重）的类型
    // 属性不集中存储 —— 参见下面的“类型和属性”
}
```

Pliron 使用**分代竞技场**（来自 `slotmap` crate）而不是 `Box` 分配的堆节点。这为你提供：

- **O(1) 插入和删除** —— 没有树重平衡，没有链表遍历。
- **稳定的索引** —— 插入或删除元素不会使其他索引失效。
- **分代版本控制** —— 每个槽位带有一个代计数器。如果你删除一个操作，该槽位被重用，任何旧的引用都会有陈旧的分代，在运行时失败而不是静默读取垃圾数据。

IR 的每一部分都存储在 Context 内部。你永远不会跨函数边界持有直接的 `&mut Operation` —— 你持有 `Ptr<Operation>`，并在需要时通过 Context 解引用它。

### Operation

**操作**是 IR 图中的一个节点。它有操作数（输入）、结果（输出）、属性（编译时元数据），以及可选的区域（用于函数体、循环嵌套等嵌套结构）。

每个操作都属于一个dialect。命名约定是 `dialect_name.op_name`：

```rust
#[pliron_op(name = "mir.func", dialect = "mir")]
pub struct MirFuncOp;

#[pliron_op(name = "nvvm.read_ptx_sreg_tid_x", dialect = "nvvm")]
pub struct ReadPtxSregTidXOp;

#[pliron_op(name = "llvm.add", dialect = "llvm")]
pub struct AddOp;
```

`#[pliron_op(...)]` 过程宏（来自 `pliron-derive`）生成了将操作注册到其dialect、分配操作码以及连接 `Op` trait 的样板代码。你通过实现 `Verify`、`Printable` 和 `Parsable` 来定义操作的语义。

### 类型和属性

**类型**表示 IR 中的数据类型。Pliron 的内置dialect提供了整数（`i1`、`i32`、`i64`）和浮点数（`f32`、`f64`）。cuda-oxide 的 MIR dialect通过这些 Rust 特有的类型进行扩展：

```rust
#[pliron_type(name = "mir.tuple", dialect = "mir")]
pub struct MirTupleType {
    pub types: Vec<Ptr<TypeObj>>,
}

#[pliron_type(name = "mir.slice", dialect = "mir")]
pub struct MirSliceType {
    pub element_ty: Ptr<TypeObj>,
}

#[pliron_type(name = "mir.enum", dialect = "mir")]
pub struct MirEnumType {
    pub name: String,
    pub discriminant_ty: Ptr<TypeObj>,
    pub variant_names: Vec<String>,
    // ...
}
```

**属性**将编译时元数据附加到操作 —— 常量值、标志、谓词类型、转换类型等等。

类型和属性的存储方式不同：

- **类型是唯一化的**（去重）。如果你创建两个 `MirTupleType { types: vec![i32, f32] }` 实例，pliron 只存储一份副本并返回相同的指针。类型相等变成指针比较而不是深层结构比较。
- **属性默认不唯一化**。每个操作内联携带自己的属性值。这符合 MLIR 的“属性”工作方式 —— MLIR 最初也对属性进行唯一化，发现可变性和每个操作的状态很笨拙，于是引入了作为非存储、每个操作替代方案的属性。Pliron 跳过了那个弯路，直接从类似 property 的设计开始。

如果你确实想对属性进行去重（例如，一个 4 KB 的常量查找表被许多操作引用），可以通过 `uniqued_any` 工具按每个属性选择加入唯一化。模式是将重量级负载包装在 `UniquedKey<T>` 中，并将键存储在属性内部：

```rust
struct SomeData { /* 大或比较昂贵的负载 */ }

#[pliron_attr(name = "...", dialect = "...")]
pub struct SomeDataAttr(pub UniquedKey<SomeData>);
```

现在 `SomeDataAttr` 是一个通过键进行比较的小句柄；底层的 `SomeData` 在唯一化表中只存在一次。

### Ptr\<T\> —— 安全的竞技场引用

`Ptr<T>` 是 pliron 中等价于指向竞技场的指针。在底层，它由两个字段组成：

```rust
pub struct Ptr<T> {
    index: u32,           // 竞技场中的槽位索引
    version: NonZeroU32,  // 分代版本
    _phantom: PhantomData<T>,
}
```

**版本**字段是使其安全的因素。当你删除一个操作时，pliron 会递增该槽位的代计数器。如果之后有人试图解引用一个旧的 `Ptr<Operation>`，其版本不再匹配槽位的当前代，则访问失败，而不是读取新的（不相关的）占用者。

`PhantomData<T>` 确保编译时的类型安全 —— 你不能意外地将 `Ptr<Operation>` 用于索引 `BasicBlock` 竞技场。编译器不会让你这么做。

> **注意**：
> 你通过 Context 与竞技场内容交互：
> - `ptr.deref(ctx)` 返回共享引用（`&T`）。
> - `ptr.deref_mut(ctx)` 返回独占引用（`&mut T`）。
> 这遵循 Rust 的借用模型 —— Context 是所有者，你通过它进行借用。

## 定值-使用链（内存安全）

在任何编译器 IR 中，你需要不断回答两个问题：

- **使用-定值**：“这个值在哪里定义的？”（给定一个使用，找到定义。）
- **定值-使用**：“这个值在哪里被使用？”（给定一个定义，找到所有使用。）

传统编译器使用原始指针和手动记账实现这些链。当你删除一个操作时忘记更新使用列表？悬垂指针。替换一个值但遗漏了一个使用？过时引用。欢迎来到你在 `opt -O2` 中调试段错误的下午。

Pliron 使用 Rust 的类型系统实现定值-使用链。pliron 中的 `Value` 是一个枚举：

```rust
pub enum Value {
    OpResult { op: Ptr<Operation>, index: usize },
    BlockArgument { block: Ptr<BasicBlock>, index: usize },
}
```

每个值要么是操作的结果，要么是基本块的参数。每个定义跟踪其所有使用的集合，每个使用存储一个指向其定义的指针。当你调用 `replace_all_uses_with` 时，两边会自动更新。

这意味着：

- **没有悬垂引用** —— 删除操作会更新所有使用列表。
- **没有过时使用** —— 替换值会传播到每个消费者。
- **IR 转换期间没有段错误** —— 借用检查器和分代竞技场阻止了困扰 C++ 编译器框架的整个错误类别。

> **注意**：
> 如果你熟悉 LLVM 的 `Value` / `Use` / `User` 系统，pliron 的设计服务于相同的目的。区别在于 LLVM 使用侵入式链表和原始 `Value*` 指针实现，而 pliron 使用竞技场索引和 `HashSet<UseNode>` 实现。语义相似，但所有权模型直接适合 Rust IR 转换。

## 操作接口

你想要对整个 IR 做的大部分事情 —— 验证它、打印它、将其降级到另一种dialect —— 自然地表达为“每个实现接口 `X` 的操作都知道如何做 `X`”。Pliron 通过**操作接口**使这种模式成为一等公民：一些小的 Rust trait，用 `#[op_interface]` 标记，任何操作（或类型、或属性）都可以实现。

你可以像普通 Rust trait 一样定义接口，并在其上添加 `#[op_interface]` 属性：

```rust
use pliron::derive::op_interface;

#[op_interface]
pub trait MirToLlvmConversion {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}
```

每个操作使用 `#[op_interface_impl]` 实现接口：

```rust
use pliron::derive::op_interface_impl;

#[op_interface_impl]
impl MirToLlvmConversion for MirAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        // ...将 MirAddOp 降级为一个或多个 dialect-llvm 操作...
    }
}
```

然后框架针对接口进行分发，而不是针对具体的操作类型。一个将 MIR 降级为 LLVM 的 pass 大致如下：

```rust
// 在 DialectConversion::rewrite 内部，对于遇到的每个操作：
if let Some(converter) = op_cast::<dyn MirToLlvmConversion>(op) {
    converter.convert(ctx, rewriter, operands_info)?;
}
```

如果你明天添加一个新的 MIR 操作并编写其 `#[op_interface_impl]`，降级 pass 会自动拾取它 —— 无需更新中心化的 match 语句，无需增加枚举。相同的模式也用于验证（`#[op_interface] trait Verify`）、打印以及任何其他横切行为。

相同的 `#[op_interface]` / `#[op_interface_impl]` 机制也适用于类型和属性：一个类型可以实现 `dyn MemorySemantics`，一个常量属性可以实现 `dyn TypedAttr`，等等。

> **注意**：
> 在底层，`op_cast` 是一个小的运行时查找（一次哈希表探测），将操作的具体类型映射到宏注册的接口实现。它仅在 pass 分发期间触发，不在 IR 构建的热路径中。好处是添加一个dialect就是添加一个 crate —— 永远不会修改中心化的枚举或泛型纠缠的函数签名。

## cuda-oxide 如何使用 pliron

cuda-oxide 定义了三个dialect，每个都是独立的 crate。编译器流水线将它们注册到 `Context` 上（从 pliron 0.14 开始，pliron 的 `builtin` dialect会自动注册），因此kernel作者和 pass 作者永远不需要考虑dialect设置 —— 你唯一要做的就是依赖相应的 crate。

### dialect-mir —— Rust 语义

`dialect-mir` 将 Rust 的中级 IR 捕获为 pliron 操作，保留了如果我们直接降级到 LLVM 会丢失的语义信息。

- **函数定义**：`MirFuncOp` —— 每个设备函数的入口点，带有与 Rust 函数签名匹配的类型化块参数。
- **算术和比较**：检查和非检查的二元操作（`mir.add`、`mir.sub`、`mir.eq`、`mir.lt`……）保留 Rust 的溢出语义。
- **聚合类型**：`MirTupleType`、`MirStructType`、`MirEnumType`、`MirSliceType`、`MirArrayType` —— 一等 Rust 复合类型，带有如 `mir.extract_field` 和 `mir.get_discriminant` 的操作。
- **内存和控制流**：`mir.load`、`mir.store`、`mir.ref`、`mir.goto`、`mir.cond_br`、`mir.return` —— 带有 GPU 地址空间跟踪（`global`、`shared`、`local`、`tmem`）。

### dialect-llvm —— 接近机器的 IR

`dialect-llvm` 将 LLVM IR 建模为 pliron 操作，提供与文本 `.ll` 文件的一一映射。

- **算术和转换**：所有 19 个 LLVM 二元操作（`llvm.add` 到 `llvm.frem`），加上 13 个转换操作（`llvm.sext`、`llvm.trunc`、`llvm.bitcast`……）。
- **控制流**：`llvm.br`、`llvm.cond_br`、`llvm.switch`、`llvm.return`、`llvm.unreachable` —— 在导出时块参数转换为 PHI 节点。
- **文本导出**：`dialect_llvm::export` 模块发出有效的 LLVM IR 文本，包括 `@llvm.used` 数组和用于 GPU kernel的 `!nvvm.annotations` 元数据。

### dialect-nvvm —— GPU 内联函数

`dialect-nvvm` 将 LLVM 的 NVPTX 后端内联函数包装为类型化的 pliron 操作。

- **线程索引**：`nvvm.read_ptx_sreg_tid_x`、`nvvm.read_ptx_sreg_ctaid_x`、`nvvm.barrier0` —— `thread::index_1d()` 的构建块。
- **线程束级原语**：洗牌（`nvvm.shfl_sync`）、投票和归约操作，用于线程束协作算法。
- **加速器操作**：TMA 批量拷贝、WGMMA 矩阵乘加（Hopper）以及 tcgen05 张量核心操作（Blackwell） —— [高级 GPU 特性](../高级GPU特性/矩阵乘法加速器.md) 章节背后的硬件指令。

每个dialect在 [Pliron dialect](./pliron-dialects.md) 中都有详细说明。
