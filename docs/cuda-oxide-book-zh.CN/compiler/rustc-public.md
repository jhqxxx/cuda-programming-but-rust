# rustc_public —— Stable MIR

cuda-oxide 没有自己发明 Rust 解析器或类型系统。它搭上真正的 Rust 编译器，
拦截 `rustc` 在类型检查与单态化之后产出的内部表示，然后把*那个东西*编译成
PTX。本章解释 cuda-oxide 读取的中间表示（MIR）、它赖以读取的稳定层
（`rustc_public`），以及连接这两个世界的桥接模式。

## 什么是 MIR？

在 `rustc` 解析你的源码、解析名称、检查类型，并把所有语法上的便利之处
（闭包、`for` 循环、`?` 运算符）脱糖之后，它会产出 **MIR** —— **中级中间
表示**。MIR 是把你的程序简化成面向控制流的形式，它看起来比你写下的抽象
语法树更接近机器将要执行的东西。

MIR 正是苦活发生的地方：

- **借用检查** —— Rust 的所有权规则是针对 MIR 校验的，而不是针对你的源码。
- **优化** —— 常量传播、拷贝传播、死存储消除以及内联都在 MIR 上操作。
- **单态化** —— 泛型函数会为每一组类型参数盖印出具体的版本。

在正常的 Rust 编译中，MIR 会下降到 LLVM IR（如果你用的是 `cranelift` 后端，
那就是 Cranelift IR），再往下变成机器码。但如果你能在 MIR 到达 LLVM *之前*
拦截它，拿它做别的事情 —— 比如把它编译成 GPU 用的 PTX —— 呢？

cuda-oxide 做的正是这件事。

## 稳定性问题

有个陷阱。MIR 是 `rustc` 的**内部**中间表示。它从来不是按公开 API 设计的。
类型会改名，枚举变体会重排，字段会出现也会消失 —— 全都发生在相邻的 nightly
版本之间，有时甚至在相邻的*提交*之间。编译器团队没有义务保持其中任何部分
稳定，而他们确实也没有。

如果 cuda-oxide 直接消费 `rustc_middle` 类型，那么每次 nightly 更新都会变成
打地鼠：某个东西挪了位，某个东西坏掉了，有人就得花一个周末去修编译错误，
而不是写 GPU 代码。

这就是 `rustc_public` 出现的地方。

## 什么是 rustc_public？

`rustc_public`（以前叫 `stable_mir`）是通向 Rust 编译器内部结构的**稳定
接口**。它让工具开发者 —— 验证器、linter、像 cuda-oxide 这样的代码生成
后端 —— 能够做分析和代码生成，而不必每次编译器的内部管道一变就崩掉。

它的实现位于 `rustc` 仓库内的两个 crate 中：

| crate                 | 角色                                                                                                                                                                                                  |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rustc_public`        | 面向用户的公开 API。为 `Body`、`BasicBlock`、`Local`、`Place`、`Ty`、`StatementKind`、`TerminatorKind` 以及 MIR 的其余部分定义稳定类型。最终会发布到 crates.io。 |
| `rustc_public_bridge` | 翻译层。在 `rustc_public` 类型与编译器内部真正的 `rustc_middle` 类型之间转换。                                                                         |

这个稳定 API 覆盖了 cuda-oxide 最关心的那些类型：

- **`Body`** —— 单个函数的 MIR（基本块、local、类型）。
- **`BasicBlock`** —— 一串直线型语句，末尾跟一个 terminator。
- **`Local` 与 `Place`** —— 变量与内存位置。
- **`Ty`** —— 完整的 Rust 类型系统：基础类型、引用、元组、
  代数数据类型、函数指针、闭包。
- **`StatementKind`** —— 赋值、存储标注、判别值读取。
- **`TerminatorKind`** —— 分支、调用、返回、断言、drop。
- **`Instance`** —— 一个单态化的函数（具体类型已经填好）。

> **注意**
>
> `rustc_public` 的工作由 AWS 的 [Kani](https://github.com/model-checking/kani)
> 团队（面向 Rust 的形式化验证）以及其它需要稳定编译器访问途径的项目推动。
> cuda-oxide 受益于他们的工作，而不必自己维护这座桥。

## cuda-oxide 如何接入

### CodegenBackend trait

在 `rustc` 深处，编译是围绕一个名为 `CodegenBackend` 的 trait 组织的。它
的关键方法是 `codegen_crate`，它接收一个 `TyCtxt` —— 编译器的上帝对象，
包含当前这次编译的全部类型信息、MIR body 与元数据 —— 并且必须产出编译结果。

正常情况下，`rustc_codegen_llvm` 实现这个 trait，通过 LLVM 把 MIR 变成机器码。
cuda-oxide 提供 `CudaCodegenBackend`，它**包装**LLVM 后端，而不是替换它。
这是刻意的设计选择：cuda-oxide 不是 LLVM 的完整替代品，它是一个专家，负责
GPU 这一侧，同时让 LLVM 去做它最擅长的事。

包装后的流程大致如下：

1. `rustc` 调用 `CudaCodegenBackend::codegen_crate(tcx)`。
2. cuda-oxide 拦截，运行它的**收集器**来识别所有设备函数（kernel 及其间接
   被调用者），并进入稳定 MIR 上下文，通过 `mir-importer` 把它们编译成 PTX。
3. PTX 输出被写到构建产物旁边的磁盘上。
4. cuda-oxide 把宿主代码委托给被包装的 LLVM 后端，后者照常把它编译成本地
   二进制。

结果是一次 `cargo` 调用同时产出本地宿主二进制*和*一个 PTX 模块，不需要两套
工具链，也不需要拆分的构建系统。

### 进入稳定 MIR 上下文

在 `codegen_crate` 内部，cuda-oxide 收到的是 `rustc_middle` 类型 —— 内部的、
不稳定的那种。真正执行 MIR 到 Pliron IR 翻译的 `mir-importer` crate 把
`rustc_public` 当作它的主要 MIR 接口。为了跨越这条边界，cuda-oxide 使用那座桥：

```rust
// 在 codegen_crate() 内部：
let result = rustc_internal::run(tcx, || {
    // 现在处于稳定 MIR 上下文中
    let stable_instance = rustc_internal::stable(func.instance);
    let body = stable_instance.body().unwrap();
    // 送入 cuda-oxide 流水线
    mir_importer::run_pipeline(&functions, &config)
});
```

`rustc_internal::run(tcx, || { ... })` 搭建一个有作用域的上下文，在其中稳定
MIR 查询可用。在闭包内部，`rustc_internal::stable()` 把一个内部的
`rustc_middle::ty::Instance<'tcx>` 转换成它在稳定 API 中的对应物
`rustc_public::mir::mono::Instance`。从那里调用 `Instance::body()` 就通过稳定
API 取到 MIR —— 完全不需要直接接触 `rustc_middle`。

## 线程局部上下文管理

你可能会想，为什么这座桥需要一个特殊的 `run()` 作用域，而不是直接把一个
上下文对象传来传去。答案是生命周期的纠缠。

`TyCtxt<'tcx>` 从编译器的 arena 分配器借用数据。它可以被存进结构体，也可以
在函数之间传递，但它的 `'tcx` 生命周期必须留在编译器会话之内。`rustc_public`
使用**有作用域的线程局部存储**，这样它的公开查询方法就能找到当前上下文，而
不必在每次调用时都要求传一个 `TyCtxt` 参数。

这座桥设立了一个线程局部变量（TLV）：

| TLV                       | 存储的值        | 用途                                                   |
| :------------------------ | :------------------ | :-------------------------------------------------------- |
| `compiler_interface::TLV` | `Cell<*const ()>`    | 对当前 `CompilerInterface` 引用的作用域访问 |

这个指针在内部被擦除了类型；`compiler_interface::with()` 会把
`CompilerInterface` 引用恢复出来，供 `local_crate()`、`all_local_items()`
这类查询使用。像 `rustc_internal::stable(instance)` 这样的转换会用
`with_bridge()`，它通过 `with()` 访问同一个接口，并借用它的翻译表。不存在
第二个线程局部上下文。

例如，先转换一个 instance，再取它的 MIR body，全程用的是同一个上下文：

```text
rustc_internal::run(tcx, || {
    stable_instance = stable(instance) -> with_bridge() -> with() -> tables
    stable_instance.body() -----------------------------> with() -> query
})
```

在活动作用域之外做查询会 panic；开始一个嵌套的 `run()` 会返回错误。当前的
设计取代了旧的 `Container` 与双上下文实现，见
[rust-lang/rust#147923](https://github.com/rust-lang/rust/pull/147923)。

## 桥接模式

`CompilerInterface` 拥有一个 **`Tables`** 结构 —— 一张 `rustc` 内部 ID 与
稳定 API 类型之间的双向映射。当你调用 `rustc_internal::stable(instance)` 时，
桥会在表里查找（或创建）对应的稳定 ID。当稳定 API 需要代表你向编译器发起
查询时 —— 比如说，取某个函数的 MIR body —— 它会沿着相反方向穿过这些表，
把内部类型找回来。

```text
    rustc_middle::ty::Instance<'tcx>
              │
              ▼
         ┌─────────┐
         │  Tables │  （双向：internal ↔ stable）
         └─────────┘
              │
              ▼
    rustc_public::mir::mono::Instance
```

几个值得了解的实现细节：

- **内部可变性** —— `CompilerInterface` 把表和编译器上下文放在 `RefCell` 里。
  每次查询都在当前线程上借用它们；重叠的可变借用会在运行时被拒绝。
- **缓存** —— 一个类型或 instance 一旦被翻译过，结果就会存进表里。重复查找
  会命中缓存，而不是重新计算。
- **自动清理** —— 当 `rustc_internal::run()` 返回时，作用域线程局部值被取消
  设置，接口及其表被丢弃。稳定 ID 引用的是这些表；不要在之后的上下文里复用
  它们。

`rustc-codegen-cuda` 中的 driver 用 `rustc_public::rustc_internal::run()`
打开作用域，并用 `stable()` 转换 instance。`mir-importer` 翻译器主要消费
`rustc_public` 类型，但有些检查仍会通过桥去查 `rustc_middle`。例如，
grid 常量校验会从编译器的内部表示中读取声明的引用生命周期以及内联所有权。
importer 自己不管桥接表。

## MIR 长什么样

在 cuda-oxide 能翻译 MIR 之前，先搞清楚 MIR 究竟*是*什么会很有帮助。下面是
一个简单的函数及其 MIR：

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

```text
// `add` 的 MIR：
// _0: i32          （返回位置）
// _1: i32          （参数 `x`）
// _2: i32          （参数 `y`）
// _3: (i32, bool)  （带检查算术的临时值）
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

有几处值得注意：

- **local 是编号的**。`_0` 永远是返回位置（结果去的地方）。`_1`、`_2`……
  是函数参数。编号更大的 local 是编译器引入的临时值。
- **基本块**（`bb0`、`bb1`……）是直线型的语句序列。每个块都以恰好一个
  **terminator** 结束，由它转移控制 —— 分支、调用、返回或断言。
- **`CheckedAdd`** 返回元组 `(i32, bool)`。这个 `bool` 是溢出标志。`assert`
  terminator 检查它，然后要么继续到 `bb1`，要么 panic。在 debug 构建里这能
  捕获整数溢出；在 release 构建里这个检查会被优化掉。
- **没有嵌套的表达式。** 源码里的 `x + y` 在 MIR 中变成两个独立操作：先算
  带检查的加法，再取出结果。每一个中间值都有自己的 local。正是这种平坦的
  结构，让 MIR 易于被 cuda-oxide 这样的工具消费 —— 没有递归的表达式树，
  只有每个块一份平坦的操作列表。

> **注意**
>
> 你可以给 `rustc` 传 `--emit=mir` 来查看任何函数的 MIR，或者访问
> [Rust Playground](https://play.rust-lang.org/) 并选择 MIR 输出。一旦你
> 习惯了 local 的编号方式，它出乎意料地好读。

### 为什么这对 cuda-oxide 重要

MIR 平坦而显式的结构，正是能在它之上构建 GPU 编译器的原因。想想另一种可能：
如果 cuda-oxide 直接在 Rust 的 AST 或 HIR（高层 IR）上操作，它就得自己处理
闭包、方法解析、trait 分发、类型推断，以及上百种在 MIR 产出时*已经*被解析
掉的语言特性。通过读取 MIR，cuda-oxide 得到的是一个泛型已经单态化、闭包已经
下降为结构体、控制流已经显式化的表示。`mir-importer` crate 把它翻译成
Pliron IR（一个类 MLIR 的框架），[下降流水线](lowering-pipeline.md)再从那里
把它送到 PTX。

## nightly 固定

即便 `rustc_public` 提供了稳定的 *API*，桥接层（`rustc_public_bridge`）仍然
是针对编译器内部结构编译的，并没有独立版本。实践中这意味着某个具体版本的
cuda-oxide 只能配某一个具体的 nightly。

cuda-oxide 通过 `rust-toolchain.toml` 固定到一个精确的 nightly 发布：

```toml
[toolchain]
channel = "nightly-2026-10-11"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
```

这个固定保证了可复现的构建：克隆这个仓库的人拿到同一个编译器、同样的 MIR
形态、同样的桥接行为。更新这个固定时，流程是：

1. 改动 `cuda-oxide/rust-toolchain.toml` 里的 nightly 日期。Rustup 会从独立的
   `rustc-codegen-cuda` 工作区向上查找并找到那个组件固定，所以不需要再放一份
   嵌套的副本。`cuda-oxide/scripts/check-toolchain-parity.sh` 会检查
   `cargo oxide new` 写出的脚手架、devcontainer，以及本书和各 README 里引用的
   代码块与带日期的命令 —— 运行它来找出漏网之鱼。
2. 修掉所有 `rustc_public` 的 API 变化（通常都很小 —— 这正是稳定 API 的意义）。
3. 跑完整测试套件，确认所有示例仍能编译并产出正确的 PTX。
4. 庆祝一下，或者回滚，下周的 nightly 再试。

> **注意**
>
> 随着 `rustc_public` 逐渐成熟并向 crates.io 发布靠拢，与特定 nightly 的耦合
> 会放松。长期目标是让 cuda-oxide 能配任何足够新的稳定 Rust 工具链 —— 但我们
> 还没到那一步。

---

现在你已经理解了 cuda-oxide 如何与 Rust 编译器对话，下一章讲的是当这场对话
抵达代码生成后端时会发生什么：[代码生成器](rustc-codegen-cuda.md)。
