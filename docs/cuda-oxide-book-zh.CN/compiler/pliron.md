# Pliron —— Pliron IR（类 MLIR）

cuda-oxide 并不是一步就把 Rust 编译成 PTX。它让代码经过一系列中间表示逐级
下降，每一层捕捉不同层次的抽象。让这件事成为可能的框架就是 **pliron** —— 一个
可扩展的编译器 IR 框架，用纯 Rust 写成，灵感来自 LLVM 的 MLIR。我们把构建在
这个框架上的 IR 称为 **Pliron IR**。

本章解释上游的 MLIR 是什么、pliron 为什么作为替代方案存在，以及它的核心
数据结构如何工作。如果你从未写过编译器，别担心 —— 这里不要求任何编译器经验，
只需要对 Rust 有实用的了解。

## 什么是 MLIR（以及你为什么该关心）？

LLVM IR 是一套固定的指令集。它大约有 70 个 opcode（`add`、`load`、`br`、
`getelementptr` 等等），如果你的领域不能干净地映射到这些 opcode 上，那就
没办法了 —— 你只能把高层概念展平成底层指令，然后祈祷优化器能重建你的本意。

**MLIR**（Multi-Level Intermediate Representation，多级中间表示）走了另一条
路。MLIR 不是给你一套固定指令集，而是给你一套框架来定义*许多*指令集 ——
称为**方言** —— 每一个都针对特定领域量身定制。一个方言就是一组操作、类型和
属性，用来建模你真正在乎的概念。

核心想法很直白：

1. 定义一个方言，其操作贴合你的领域。
2. 编写 **pass**，把一个方言变换成另一个（下降）。
3. 把 pass 串起来，直到得到后端可以消费的东西。

看看 Triton（PyTorch 背后的 GPU 编译器）如何使用 MLIR。Python GPU 代码首先
下降到 **TTIR**，这是一种张量级 IR，其中"把这个标量广播到一个张量上"这样的
操作是一等公民。在那个层面，Triton 可以施加领域特有的优化 —— 比如把一个
`splat + mul`（广播一个标量，然后逐元素相乘）替换成一条原生的向量-标量乘法。
当你的 IR 里有张量操作时，这种优化表达起来轻而易举；而一旦展平到 LLVM IR，
就几乎不可能再恢复出来。

cuda-oxide 面临类似的挑战。我们需要表示三个非常不同的抽象层次：

| 层级              | 建模的内容                                                | 示例操作                           |
| :----------------- | :------------------------------------------------------------ | :------------------------------------------- |
| **`dialect-mir`**  | Rust 语义 —— 元组、枚举、切片、带检查的算术   | `mir.extract_field`、`mir.get_discriminant`  |
| **LLVM 方言**   | 接近机器的操作 —— 整数运算、内存、控制流 | `llvm.add`、`llvm.load`、`llvm.br`           |
| **`dialect-nvvm`** | GPU 内联函数 —— 线程索引、warp shuffle、TMA、WGMMA  | `nvvm.read_ptx_sreg_tid_x`、`nvvm.shfl_sync` |

如果没有可扩展的 IR，我们要么得把 Rust 枚举硬塞进 LLVM IR（丢失语义信息），
要么为流水线的每个层次单独造一套 IR 框架。MLIR 让我们把这三者都定义成同一个
系统中的方言，并用类型良好的 pass 在它们之间下降。

## pliron 登场

MLIR 的"方言 + 下降"模型正是 cuda-oxide 需要的抽象：在 IR 中保留高层语义，
同时逐步向 GPU 后端能消费的代码下降。问题在于实现的契合度。上游 MLIR 属于
LLVM 的 C++ 生态，而 cuda-oxide 是一个围绕 Rust crate、Rust 类型和 Cargo
工作流构建的 Rust 编译器项目。

[Pliron](https://github.com/vaivaswatha/pliron) 是一个受 MLIR 启发、可扩展的
编译器 IR 框架，用**纯 Rust** 写成。它遵循同一套概念模型 —— 操作、region、
基本块、类型、属性、方言、pass —— 同时又能自然融入 Rust 原生的编译器技术栈。

| 方面            | pliron                                   | 上游 MLIR                      |
| :---------------- | :--------------------------------------- | :--------------------------------- |
| 语言          | 纯 Rust                                | C++ 加 Python 绑定           |
| 构建系统      | `cargo build`                            | CMake + 完整 LLVM 构建            |
| 需要的 DSL     | 无 —— 只需 Rust macro                 | TableGen、ODS                      |
| 调试         | 标准 Rust 工具（`dbg!`、rust-gdb） | 在 C++ 模板上用 gdb               |
| 可扩展性     | 把方言加成一个 Rust crate            | 针对 LLVM 头文件做 C++ 扩展 |
| 依赖重量 | 一个 crate（git 依赖）               | 数 GB 的 LLVM 构建产物  |

对 cuda-oxide 而言，这意味着整个编译器 —— 从 MIR 导入到 LLVM IR 导出 ——
都留在与项目其余部分相同的 Rust 构建与调试环境里。pliron 给了我们所需的
可扩展 IR 结构，而不必让编译器离开 Rust 生态。

## 核心数据结构

Pliron 的 IR 由少量核心类型构建而成。理解这些是读懂（并写出）方言代码的关键。

### Context

`Context` 是核心数据结构，它拥有*全部* IR 数据 —— 操作、基本块、region、
类型、属性以及方言注册。可以把它看作整个编译单元的 arena 分配器。

```rust
// 简化版 —— 真正的 Context 字段更多
pub struct Context {
    operations: SlotMap<Ptr<Operation>, Operation>,
    basic_blocks: SlotMap<Ptr<BasicBlock>, BasicBlock>,
    regions: SlotMap<Ptr<Region>, Region>,
    dialects: HashMap<DialectName, Dialect>,
    type_store: TypeStore,     // 经唯一化（去重）的类型
    // 属性不集中存储 —— 见下文"类型与属性"。
}
```

Pliron 使用**分代 arena**（来自 `slotmap` crate），而不是用 `Box` 分配堆节点。
这带给你：

- **O(1) 插入与删除** —— 没有树的重平衡，也没有链表遍历。
- **稳定的索引** —— 插入或删除元素不会让其它索引失效。
- **分代版本** —— 每个槽位带一个代计数器。如果你删除了一个操作、该槽位又被
  复用，任何旧的引用都会带有过期的代号，在运行时失败，而不是悄悄读到垃圾数据。

IR 的每一部分都存放在 Context 里。你永远不会跨函数边界持有直接的
`&mut Operation` —— 你持有 `Ptr<Operation>`，在需要时通过 Context 解引用它。

### 操作

一个**操作**是 IR 图中的单个节点。它有操作数（输入）、结果（输出）、属性
（编译期元数据），还可以有 region（用于函数体、循环嵌套这类嵌套结构）。

每个操作都属于某个方言。命名约定是 `dialect_name.op_name`：

```rust
#[pliron_op(name = "mir.func", dialect = "mir")]
pub struct MirFuncOp;

#[pliron_op(name = "nvvm.read_ptx_sreg_tid_x", dialect = "nvvm")]
pub struct ReadPtxSregTidXOp;

#[pliron_op(name = "llvm.add", dialect = "llvm")]
pub struct AddOp;
```

`#[pliron_op(...)]` proc macro（来自 `pliron-derive`）会生成样板代码，把这个
操作注册到它的方言、给它分配 opcode，并接好 `Op` trait。操作的语义则通过
实现 `Verify`、`Printable` 和 `Parsable` 来定义。

### 类型与属性

**类型**表示 IR 中的数据类型。Pliron 的内置方言提供整数（`i1`、`i32`、`i64`）
和浮点数（`f32`、`f64`）。cuda-oxide 的 MIR 方言用 Rust 特有的类型扩展了它们：

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

**属性**把编译期元数据附加到操作上 —— 常量值、标志、谓词种类、转换种类等等。

类型和属性的存储方式不同：

- **类型是唯一化的**（去重）。如果你创建两个
  `MirTupleType { types: vec![i32, f32] }` 实例，pliron 只存一份，
  并返回同一个指针。类型相等性变成指针比较，而不是深层的结构比较。
- **属性默认不唯一化。** 每个操作内联携带自己的属性值。这与 MLIR 的
  "properties" 的做法一致 —— MLIR 最初也对属性做唯一化，后来发现变更和
  逐操作状态都很别扭，于是引入 properties 作为不存储、逐操作的替代方案。
  pliron 跳过了这段弯路，直接从 properties 形态的设计开始。

如果你确实想对某个属性去重（例如一张很多操作都引用的 4 KB 常量查找表），
可以通过 `uniqued_any` 工具按属性选择启用唯一化。套路是把重载荷包在
`UniquedKey<T>` 里，再把 key 存进属性中：

```rust
struct SomeData { /* 很大或比较开销高昂的载荷 */ }

#[pliron_attr(name = "...", dialect = "...")]
pub struct SomeDataAttr(pub UniquedKey<SomeData>);
```

现在 `SomeDataAttr` 只是一个小句柄，按 key 比较；底层的 `SomeData` 在
唯一化表里只存一份。

### Ptr\<T\> —— 安全的 arena 引用

`Ptr<T>` 是 pliron 中指向 arena 的指针。在底层，它由两个字段组成：

```rust
pub struct Ptr<T> {
    index: u32,           // arena 中的槽位索引
    version: NonZeroU32,  // 分代版本
    _phantom: PhantomData<T>,
}
```

**version** 字段正是安全性的来源。当你删除一个操作时，pliron 会把这个槽位的
代计数器加一。如果之后有人试图解引用一个旧的 `Ptr<Operation>`，而它的版本
已经与该槽位当前的代号不匹配，这次访问就会失败，而不是读到新的（不相关的）
占用者。

`PhantomData<T>` 则在编译期保证类型安全 —— 你无法不小心用一个
`Ptr<Operation>` 去索引 `BasicBlock` 的 arena。编译器不会允许你这么做。

> **注意**
>
> 你通过 Context 与 arena 的内容交互：
>
> - `ptr.deref(ctx)` 返回共享引用（`&T`）。
> - `ptr.deref_mut(ctx)` 返回独占引用（`&mut T`）。
>
> 这遵循 Rust 的借用模型 —— Context 是所有者，你通过它来借用。

## def-use 链（内存安全）

在任何编译器 IR 里，你都需要不断回答两个问题：

- **use-def**："这个值在哪里定义？"（给定一次使用，找到定义。）
- **def-use**："这个值在哪里被使用？"（给定一个定义，找到所有使用。）

传统编译器用裸指针加手工记账来实现这些链。删除一个操作时忘了更新 use 列表？
悬垂指针。替换一个值但漏掉了一次使用？过期引用。欢迎开始你排查
`opt -O2` 段错误的一整个下午。

Pliron 用 Rust 的类型系统实现 def-use 链。pliron 中的 `Value` 是一个枚举：

```rust
pub enum Value {
    OpResult { op: Ptr<Operation>, index: usize },
    BlockArgument { block: Ptr<BasicBlock>, index: usize },
}
```

每个值要么是某个操作的结果，要么是某个基本块的参数。每个定义都追踪它全部
使用构成的集合，而每次使用都存有一个指回其定义的指针。当你调用
`replace_all_uses_with` 时，两侧都会自动更新。

这意味着：

- **没有悬垂引用** —— 移除一个操作会更新所有 use 列表。
- **没有过期使用** —— 替换一个值会传播到每一个消费者。
- **IR 变换期间不会段错误** —— 借用检查器与分代 arena 阻止了困扰 C++ 编译器
  框架的整类 bug。

> **注意**
>
> 如果你熟悉 LLVM 的 `Value` / `Use` / `User` 系统，pliron 的设计目的相同。
> 区别在于 LLVM 用侵入式链表和裸 `Value*` 指针实现它，而 pliron 用 arena
> 索引和 `HashSet<UseNode>` 实现。语义相似，但所有权模型能直接契合 Rust 的
> IR 变换。

## Op 接口

你想在整个 IR 上做的多数事情 —— 校验它、打印它、把它下降到另一个方言 ——
都可以自然地表达为"每个实现了接口 `X` 的 op 都知道怎么做 `X`"。pliron 通过
**op 接口**把这种模式变成一等公民：小的 Rust trait，用 `#[op_interface]`
标记，任何 op（或类型、属性）都可以实现它。

你像定义普通 Rust trait 一样定义接口，顶上加上 `#[op_interface]` 属性：

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

每个 op 用 `#[op_interface_impl]` 实现该接口：

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
        // ...把 MirAddOp 下降成一个或多个 LLVM 方言操作...
    }
}
```

框架随后针对接口做分发，而不是针对某个具体的 op 类型。一个把 MIR 下降到
LLVM 的 pass 大致长这样：

```rust
// 在 DialectConversion::rewrite 内部，对遇到的每个 op：
if let Some(converter) = op_cast::<dyn MirToLlvmConversion>(op) {
    converter.convert(ctx, rewriter, operands_info)?;
}
```

如果你明天加一个新的 MIR op，并写好它的 `#[op_interface_impl]`，下降 pass
会自动接上它 —— 不需要更新中心化的 match 语句，也不需要一个枚举继续长大。
同样的模式也用于校验（`#[op_interface] trait Verify`）、打印以及任何其它
横切行为。

同一套 `#[op_interface]` / `#[op_interface_impl]` 机制也适用于类型和属性：
一个类型可以实现 `dyn MemorySemantics`，一个常量属性可以实现 `dyn TypedAttr`，
等等。

> **注意**
>
> 在底层，`op_cast` 只是一次很小的运行时查找（一次哈希表探测），把 op 的具体
> 类型映射到宏注册的接口实现上。它只在 pass 分发时触发，不在 IR 构建的热路径
> 上。好处是：添加一个方言就是添加一个 crate —— 永远不必去改一个中心化的
> 枚举，也不必与泛型纠缠的函数签名搏斗。

## cuda-oxide 如何使用 pliron

下降路径用到三个方言：cuda-oxide 在本地定义 `dialect-mir` 和 `dialect-nvvm`
（各自一个 crate），并从上游的 `pliron-llvm` crate 消费 LLVM 方言。还有两个
本地方言位于该路径之外 —— `dialect-iket` 用于 kernel 内事件追踪，`dialect-ptx`
用于结构化终端 PTX —— 下面的小节只覆盖上述三个。注册是自动的：链接进二进制
的每个方言、op、类型和属性都会在创建 `Context` 时自行注册
（`Context::default` 会运行所有链接期注册），所以 kernel 作者和 pass 作者
从不需要考虑方言的搭建 —— 依赖那个 crate 就是你要做的全部事情。

### dialect-mir —— Rust 语义

`dialect-mir` 把 Rust 的中级 IR 捕捉为 pliron 操作，保留了直接下降到 LLVM
时就会丢失的语义信息。

- **函数定义**：`MirFuncOp` —— 每个设备函数的入口点，块参数带类型，
  与 Rust 函数签名对应。
- **算术与比较**：带检查与不带检查的二元运算（`mir.add`、`mir.sub`、
  `mir.eq`、`mir.lt` 等），保留 Rust 的溢出语义。
- **聚合类型**：`MirTupleType`、`MirStructType`、`MirEnumType`、
  `MirSliceType`、`MirArrayType` —— 一等的 Rust 复合类型，配有
  `mir.extract_field`、`mir.get_discriminant` 等操作。
- **内存与控制流**：`mir.load`、`mir.store`、`mir.ref`、
  `mir.goto`、`mir.cond_br`、`mir.return` —— 并带 GPU 地址空间
  追踪（`global`、`shared`、`local`、`tmem`）。

### LLVM 方言 —— 接近机器的 IR

LLVM 方言把 LLVM IR 建模为 pliron 操作，与文本 `.ll` 文件提供 1:1 映射。
它的建模位于上游的 `pliron-llvm` crate 中；cuda-oxide 通过 `llvm-export`
crate 重新导出它（并加上文本导出器）。

- **算术与转换**：全部 19 个 LLVM 二元运算（从 `llvm.add` 到
  `llvm.frem`），外加 13 个转换操作（`llvm.sext`、`llvm.trunc`、
  `llvm.bitcast` 等）。
- **控制流**：`llvm.br`、`llvm.cond_br`、`llvm.switch`、`llvm.return`、
  `llvm.unreachable` —— 导出时块参数会被翻译成 PHI 节点。
- **文本导出**：`llvm_export::export` 模块输出合法的 LLVM IR 文本，
  包括 GPU kernel 所需的 `@llvm.used` 数组与 `!nvvm.annotations` 元数据。

### dialect-nvvm —— GPU 内联函数

`dialect-nvvm` 把 LLVM 的 NVPTX 后端内联函数包装成带类型的 pliron 操作。

- **线程索引**：`nvvm.read_ptx_sreg_tid_x`、`nvvm.read_ptx_sreg_ctaid_x`、
  `nvvm.barrier0` —— `thread::index_1d()` 的构建积木。
- **warp 级原语**：shuffle（`nvvm.shfl_sync`）、vote 以及归约操作，
  用于 warp 协作式算法。
- **加速器操作**：TMA 批量拷贝、WGMMA 矩阵乘累加（Hopper），以及
  tcgen05 张量核心操作（Blackwell）—— [高级 GPU 特性](../advanced/tensor-memory-accelerator.md)
  各章背后的硬件指令。

每个方言都在 [Pliron 方言](mlir-dialects.md)中有详细讲解。
