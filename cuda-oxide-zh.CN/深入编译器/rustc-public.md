# rustc_public —— 稳定 MIR

cuda-oxide 并没有发明自己的 Rust 解析器或类型系统。它借助真正的 Rust 编译器，拦截 `rustc` 在类型检查和单态化之后产生的内部表示，并将*那个*编译成 PTX。本章解释 cuda-oxide 读取的中间表示（MIR）、它读取时通过的稳定性层（`rustc_public`），以及连接这两个世界的桥接模式。

## 什么是 MIR？

在 `rustc` 解析完源代码、解析名称、检查类型并消除所有语法糖（闭包、`for` 循环、`?` 运算符）之后，它会产生 **MIR** —— **中级中间表示**。MIR 是一种简化的、面向控制流的程序形式，与你编写的抽象语法树相比，它更接近机器会执行的形式。

MIR 是繁重工作的发生地：

- **借用检查** —— Rust 的所有权规则是在 MIR 上验证的，而不是在你的源代码上。
- **优化** —— 常量传播、拷贝传播、死存储消除和内联都在 MIR 上操作。
- **单态化** —— 泛型函数被实例化为每个类型参数集的具体版本。

在正常的 Rust 编译中，MIR 被降级为 LLVM IR（或者如果你使用 `cranelift` 后端，则是 Cranelift IR），然后从那里到机器码。但如果你能在 MIR 到达 LLVM 之前拦截它并对其做其他事情 —— 比如将其编译为 GPU 的 PTX 呢？

这正是 cuda-oxide 所做的。

## 稳定性问题

这里有一个问题。MIR 是 `rustc` 的**内部**中间表示。它从未被设计为公共 API。类型会被重命名，枚举变体会被重新排序，字段会出现和消失 —— 所有这些都在连续的 nightly 版本之间发生，有时甚至在连续的*提交*之间发生。编译器团队没有义务保持其任何部分的稳定性，他们也不会。

如果 cuda-oxide 直接消费 `rustc_middle` 类型，那么每次 nightly 更新都会像打地鼠游戏一样：某个东西移动了，某个东西坏了，有人花一个周末修补编译错误而不是编写 GPU 代码。

这就是 `rustc_public` 的用武之地。

## 什么是 rustc_public？

`rustc_public`（以前称为 `stable_mir`）是 Rust 编译器内部的**稳定接口**。它允许工具开发者 —— 验证器、linter、像 cuda-oxide 这样的代码生成后端 —— 进行分析和代码生成，而不会在编译器的管道每次变动时都出现问题。

该实现位于 `rustc` 仓库中的两个 crate 中：

| Crate  | 角色    |
| :---- | :--------- |
| `rustc_public`        | 面向用户的公共 API。为 `Body`、`BasicBlock`、`Local`、`Place`、`Ty`、`StatementKind`、`TerminatorKind` 以及 MIR 的其余部分定义稳定类型。最终将发布到 crates.io 上。 |
| `rustc_public_bridge` | 翻译层。在 `rustc_public` 类型和编译器内部的真实 `rustc_middle` 类型之间进行转换。                                                                         |

稳定 API 覆盖了 cuda-oxide 最关心的类型：

- **`Body`** —— 单个函数的 MIR（基本块、局部变量、类型）。
- **`BasicBlock`** —— 一系列直线语句后跟一个终止符。
- **`Local` 和 `Place`** —— 变量和内存位置。
- **`Ty`** —— 完整的 Rust 类型系统：基本类型、引用、元组、代数数据类型（ADT）、函数指针、闭包。
- **`StatementKind`** —— 赋值、存储注解、判别式读取。
- **`TerminatorKind`** —— 分支、调用、返回、断言、丢弃。
- **`Instance`** —— 一个单态化的函数（具体类型已填充）。

> **注意**：
> `rustc_public` 的工作由 AWS 的 [Kani](https://github.com/model-checking/kani) 团队（Rust 的形式化验证）以及其他需要稳定编译器访问的项目推动。cuda-oxide 受益于他们的工作，而无需自己维护桥接层。

## cuda-oxide 如何挂钩

### CodegenBackend trait

在 `rustc` 深处，编译是围绕一个名为 `CodegenBackend` 的 trait 组织的。其关键方法是 `codegen_crate`，它接收一个 `TyCtxt` —— 编译器的“上帝对象”，包含当前编译的所有类型信息、MIR 体和元数据 —— 并且必须产生编译输出。

通常，`rustc_codegen_llvm` 实现这个 trait 并通过 LLVM 将 MIR 转换为机器码。cuda-oxide 提供了 `CudaCodegenBackend`，它**包装**了 LLVM 后端而不是替换它。这是一个深思熟虑的设计选择：cuda-oxide 不是 LLVM 的完整替代品，而是一个处理 GPU 端的专家，同时让 LLVM 做 LLVM 最擅长的事情。

包装流程如下：

1. `rustc` 调用 `CudaCodegenBackend::codegen_crate(tcx)`。
2. cuda-oxide 拦截，运行其**收集器**以识别所有设备函数（kernel及其传递调用的函数），并进入稳定 MIR 上下文，通过 `mir-importer` 将它们编译成 PTX。
3. PTX 输出与构建产物一起写入磁盘。
4. cuda-oxide 将主机代码委托给包装的 LLVM 后端，该后端将其正常编译为本机二进制文件。

结果是一次 `cargo` 调用同时产生了一个本机主机二进制文件*和*一个 PTX 模块，而不需要两个单独的工具链或分裂的构建系统。

### 进入稳定 MIR 上下文

在 `codegen_crate` 内部，cuda-oxide 接收 `rustc_middle` 类型 —— 内部的、不稳定的类型。但是执行实际 MIR 到 Pliron IR 翻译的 `mir-importer` crate 完全基于 `rustc_public` 类型。为了跨越边界，cuda-oxide 使用了桥接层：

```rust
// 在 codegen_crate() 内部：
let result = rustc_internal::run(tcx, || {
    // 现在在稳定 MIR 上下文中
    let stable_instance = rustc_internal::stable(func.instance);
    let body = stable_instance.body().unwrap();
    // 送入 cuda-oxide 流水线
    mir_importer::run_pipeline(&functions, &config)
});
```

`rustc_internal::run(tcx, || { ... })` 设置了一个作用域上下文，在其中可以调用稳定 MIR 查询。在闭包内部，`rustc_internal::stable()` 将内部 `rustc_middle::ty::Instance<'tcx>` 转换为其稳定版本 `rustc_public::mir::mono::Instance`。从那里，调用 `Instance::body()` 通过稳定 API 检索 MIR —— 不需要直接接触 `rustc_middle`。

## 线程局部上下文管理

你可能会好奇为什么桥接层需要一个特殊的 `run()` 作用域，而不是仅仅传递一个上下文对象。答案是生命周期纠缠。

`TyCtxt<'tcx>` 从编译器的竞技场分配器借用数据。`'tcx` 生命周期与编译会话相关联，并且不能逃出竞技场的作用域。你不能将 `TyCtxt` 存储在结构体中、从函数返回它或将其发送到另一个线程。编译器的解决方案是**作用域线程局部存储**：上下文仅在你处于作用域内时可用，并且类型系统（加上运行时检查）防止它泄漏出去。

桥接层设置了两个嵌套的线程局部变量（TLV）：

| TLV     | 类型       | 目的      |
| :------ | :-------- | :----------- |
| `compiler_interface::TLV` | `&dyn CompilerInterface` | 高级查询：`local_crate()`、`all_local_items()`、入口点查找 |
| `rustc_internal::TLV`     | `&Container`             | 通过 `Tables` 映射进行稳定到内部类型的翻译                 |

两者都指向相同的底层 `Container` 结构体，但提供不同的访问模式：

- **`with()`** 访问外部 TLV 以进行高级编译器查询。
- **`with_container()`** 访问内部 TLV 以在稳定类型和内部类型之间进行转换。

这种两级设计将查询接口与原始翻译机制分开，因此只需要问“给我本地 crate 中的所有函数”的代码不必知道内部 ID 映射。

如果你曾经使用过 Web 框架的请求作用域上下文（比如 Actix 的 `web::Data` 或 Axum 的提取器），其心智模型类似：数据在请求（这里是编译）期间存在，框架使其可用，而不必通过每个函数签名传递它。

## 桥接模式

`Container` 的核心是一个 **`Tables`** 结构体 —— `rustc` 内部 ID 与稳定 API 类型之间的双向映射。当你调用 `rustc_internal::stable(instance)` 时，桥接层在表中查找（或创建）相应的稳定 ID。当稳定 API 需要代表你查询编译器 —— 比如获取函数的 MIR 体 —— 它会以相反方向通过表来恢复内部类型。

```text
    rustc_middle::ty::Instance<'tcx>
              │
              ▼
         ┌─────────┐
         │  Tables │  (双向：内部 ↔ 稳定)
         └─────────┘
              │
              ▼
    rustc_public::mir::mono::Instance
```

一些值得了解的实现细节：

- **内部可变性** —— `Tables` 使用 `RefCell`，因为代码库的多个部分在单个翻译过程中需要对映射进行可变访问。这是安全的，因为访问是单线程的（由线程局部存储保证）。
- **缓存** —— 一旦类型或实例被翻译，结果就会存储在表中。重复查找会命中缓存而不是重新计算。
- **自动清理** —— 当 `rustc_internal::run()` 返回时，线程局部存储被拆除，表被丢弃。无需手动清理，不可能有过时引用。

从 cuda-oxide 的角度来看，桥接层是不可见的。`mir-importer` crate 只看到 `rustc_public` 类型 —— 它从不导入 `rustc_middle`，从不处理 `'tcx` 生命周期，也从不直接接触 `Tables`。所有这些复杂性都封装在 `rustc_internal::run()` 和 `rustc_internal::stable()` 后面，它们位于 `rustc-codegen-cuda` crate 中，处于编译器与 cuda-oxide 流水线之间的边界处。

## MIR 长什么样

在 cuda-oxide 可以翻译 MIR 之前，了解 MIR 实际*是*什么会有所帮助。下面是一个简单的函数及其 MIR：

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

```text
// MIR for `add`:
// _0: i32              (返回位置)
// _1: i32              (参数 `x`)
// _2: i32              (参数 `y`)
// _3: (i32, bool)      (用于检查算术的临时变量)
//
// bb0: {
//     _3 = CheckedAdd(_1, _2);
//     assert(!(_3.1), "attempt to compute `{} + {}`, which would overflow") -> bb1;
// }
// bb1: {
//     _0 = (_3.0);
//     return;
// }
```

需要注意的几点：

- **局部变量**被编号。`_0` 总是返回位置（结果存放的地方）。`_1`、`_2`……是函数参数。编号更高的局部变量是编译器引入的临时变量。
- **基本块**（`bb0`、`bb1`、……）是语句的直线序列。每个块以恰好一个**终止符**结束，该终止符转移控制权 —— 分支、调用、返回或断言。
- **`CheckedAdd`** 返回一个元组 `(i32, bool)`。`bool` 是溢出标志。`assert` 终止符检查它，并要么继续到 `bb1`，要么 panic。在 debug 构建中，这会捕获整数溢出；在 release 构建中，该检查会被优化掉。
- **没有嵌套表达式**。源代码中的 `x + y` 在 MIR 中变成了两个独立的操作：计算检查加法，然后提取结果。每个中间值都有自己的局部变量。这种扁平结构使得 MIR 很容易被像 cuda-oxide 这样的工具消费 —— 没有递归表达式树，只有每个基本块中扁平的列表操作。

> **注意**：
> 你可以通过向 `rustc` 传递 `--emit=mir` 来查看任何函数的 MIR，或者访问 [Rust Playground](https://play.rust-lang.org/) 并选择 MIR 输出。一旦你习惯了局部变量编号，这是一种非常可读的格式。

### 这对 cuda-oxide 为什么重要

MIR 扁平的、显式的结构使得在其之上构建 GPU 编译器成为可能。考虑另一种选择：如果 cuda-oxide 操作在 Rust 的 AST 或 HIR（高层 IR）上，它将不得不处理闭包、方法解析、trait 分发、类型推断以及一百多种在 MIR 产生时*已经*解决的语言特性。通过读取 MIR，cuda-oxide 获得了一个表示，其中泛型已经被单态化，闭包已经被降级为结构体，控制流已经显式化。`mir-importer` crate 将其翻译成 Pliron IR（一种类 MLIR 框架），然后从那里[降级流水线](lowering-pipeline.md)将其一路带到 PTX。

## Nightly 版本固定

即使 `rustc_public` 提供了稳定的 *API*，桥接层（`rustc_public_bridge`）仍然是针对编译器内部编译的，并且没有独立版本。在实践中，这意味着特定版本的 cuda-oxide 适用于特定的 nightly 版本。

cuda-oxide 通过 `rust-toolchain.toml` 固定到确切的 nightly 版本：

```toml
[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "rust-analyzer"]
```

这个固定保证了可重现的构建：任何克隆仓库的人都会得到相同的编译器、相同的 MIR 形状和相同的桥接行为。当更新固定版本时，流程是：

1. 在 `rust-toolchain.toml` 中增加 nightly 日期。
2. 修复任何 `rustc_public` API 变更（通常很小 —— 这正是稳定 API 的意义所在）。
3. 运行完整的测试套件，验证所有示例仍然可以编译并生成正确的 PTX。
4. 庆祝，或者回退并尝试下周的 nightly。

> **注意**：
> 随着 `rustc_public` 成熟并朝着发布到 crates.io 的方向发展，对特定 nightly 版本的依赖将会放松。长期目标是让 cuda-oxide 能够与任何足够新的稳定 Rust 工具链一起工作 —— 但我们还没有达到那个阶段。

---

现在你已经了解了 cuda-oxide 如何与 Rust 编译器通信，下一章将介绍当这种对话到达代码生成后端时会发生什么：[代码生成器](./rustc-codegen-cuda.md)。

| [上一页](./pliron.md) | [下一页](./rustc-codegen-cuda.md) |
| :--- | ---: |