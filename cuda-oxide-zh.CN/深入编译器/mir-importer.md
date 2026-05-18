# MIR 导入器

前面的章节解释了 rustc 如何产生稳定 MIR（[rustc_public](rustc-public.md)）以及 pliron 如何提供 IR 框架（[Pliron](pliron.md)）。本章是两者的交汇点：`mir-importer` 获取 rustc 交给我们的稳定 MIR，并将其翻译成 `dialect-mir` —— 保留 Rust 语义的 pliron dialect。翻译器最初发出 alloca/load/store 形式 —— 生成成本低、易于推理，并且在输入时就是 pliron 的身份。随后通过 `pliron::opts::mem2reg` pass 将这些槽位提升回 SSA 形式，使 `dialect-mir` 准备好降级到 `dialect-llvm`。

但翻译只是工作的一半。`mir-importer` 还编排了*整个*编译流水线：翻译、验证、降级、导出和生成 PTX。它既是翻译器，也是阶段管理器。

该 crate 位于 `crates/mir-importer`，分为两部分：

- **`translator/`** —— MIR 到 pliron 的翻译逻辑（有趣的部分）。
- **`pipeline.rs`** —— 将每个阶段串联在一起的编排逻辑（负责任的部分）。

---

## 流水线编排

在深入翻译细节之前，先看大图。`run_pipeline()` 函数是 `rustc-codegen-cuda` 在收集设备函数后调用的入口点。它接受一个 `CollectedFunction` 列表和一个 `PipelineConfig`，然后运行六个阶段：

```text
步骤 1：  翻译 Rust MIR → `dialect-mir`
步骤 2：  验证 `dialect-mir` 模块
步骤 3：  运行 `pliron::opts::mem2reg` 将 alloca 槽位提升回 SSA
步骤 4：  降级 `dialect-mir` → `dialect-llvm`（通过 mir-lower）
步骤 5：  导出 `dialect-llvm` 为文本 LLVM IR（.ll）
步骤 6：  运行 llc 将 .ll 编译为 .ptx
```

每个 `CollectedFunction` 包含流水线需要知道的关于设备函数的一切：

```rust
pub struct CollectedFunction {
    pub instance: Instance,
    pub is_kernel: bool,
    pub export_name: String,
}
```

`instance` 是来自 `rustc_public` 的单态化函数。`is_kernel` 区分kernel入口点和设备辅助函数（kernel在 LLVM IR 中获得特殊元数据，以便 NVPTX 后端将其作为 `.entry` 点发出）。`export_name` 是最终 PTX 中出现的符号名称 —— 对于设备函数，这通常是一个完全限定名（FQDN），与 `CrateDef::name()` 为同一函数返回的名称匹配。

对于每个函数，流水线执行以下操作：

1. 通过 `instance.body()` 检索 MIR 体。
2. 调用 `translate_function()` 生成一个包含 `dialect-mir` 表示的 pliron 模块（为局部变量使用 `mir.alloca` 槽位）。
3. 在模块上运行 pliron 的验证器，以尽早捕获结构错误 —— 类型不匹配、缺少操作数、支配关系破坏 —— 防止它们在下游变成晦涩的 LLVM 失败。
4. 运行 `pliron::opts::mem2reg` 将 alloca 槽位提升回 `dialect-mir` 中的 SSA 值。
5. 运行 `lower_mir_to_llvm`（来自 `mir-lower` crate），通过 `DialectConversion` 将每个 `dialect-mir` 操作降级为其 `dialect-llvm` 等价物。
6. 将 `dialect-llvm` 模块导出为文本 `.ll` 字符串，写入磁盘，并调用 `llc` 生成最终的 `.ptx` 文件。

如果任何步骤失败，流水线停止并返回一个类型化的错误（`NoBody`、`Translation`、`Verification`、`Lowering`、`Export` 或 `PtxGeneration`），并提供足够的上下文来诊断问题。没有静默损坏，没有神秘的空输出文件。

---

## 翻译架构

`translator/` 目录是稳定 MIR 变成 pliron IR 的地方。每个模块处理 MIR 结构的一个层级，它们组合得很整洁：

| 模块  | 目的       |
| :------- | :-------|
| `body`       | 函数级翻译，alloca 槽位播种，FQDN 名称清理    |
| `block`      | 基本块翻译协调器                                        |
| `statement`  | 语句翻译（赋值、存储）                              |
| `terminator` | 终止符翻译（goto、call、return、基于 FQDN 的内联函数分发） |
| `rvalue`     | 表达式翻译（二元操作、转换、聚合）                         |
| `types`      | Rust 类型到 `dialect-mir` 类型的转换                                 |
| `values`     | MIR 局部变量 → alloca 槽位映射（`ValueMap`）+ 槽位地址空间推断    |

调用流程遵循 MIR 结构的自顶向下：

```text
translate_function()
  └─ body::translate_body()
       ├─ emit_entry_allocas()            // 每个非 ZST 局部变量一个 mir.alloca
       │     └─ SlotAddrSpaceMap::analyze // 指针槽位地址空间推断
       └─ For each basic block:
            └─ block::translate_block()
                  ├─ statement::translate_statement()
                  │     └─ rvalue::translate_rvalue()
                  └─ terminator::translate_terminator()
```

`translate_body()` 设置函数的签名，创建与 MIR 基本块对应的 pliron 块，清理函数名称（将 `instance.name()` 的 FQDN 中的 `::` 转换为 `__`），在入口块顶部为每个非 ZST 局部变量发出一个 `mir.alloca`，并将传入的函数参数存储到它们各自的槽位中。每个非入口块都不带参数 —— 跨块数据流由 alloca 槽位携带，而不是由块参数携带。然后它按顺序遍历每个块，逐个翻译语句和终止符。

`translate_statement()` 处理块内的扁平操作 —— 赋值、存储存活/死亡标记以及判别式写入。当赋值涉及右侧表达式（MIR `Rvalue`）时，它委托给 `translate_rvalue()`，后者处理二元操作、一元操作、转换、聚合构造、判别式读取、指针算术以及 Rust 编译到的其他十几种操作。

`translate_terminator()` 处理块结束操作：`Goto`、`SwitchInt`、`Call`、`Return`、`Assert`、`Drop` 和 `Unreachable`。这也是内联函数分发所在 —— 但这将在下面单独一节中介绍。

---

## SSA 挑战（以及我们如何推迟它）

这是翻译中最棘手的部分，值得仔细解释。

Rust MIR 不是严格的 SSA 形式。局部变量（变量）是命名的存储位置，任何块都可以读取或写入。如果 `_3` 在 `bb0` 中赋值，它可以在 `bb1`、`bb5` 或任何其他地方自由使用 —— MIR 不关心。

Pliron IR（类 MLIR）最终期望**严格的 SSA**：一个值必须支配每次使用，如果一个值需要从一个块流向另一个块，它必须作为**块参数**显式传递。你不能简单地在块之间跨过去抓取一个局部变量。

我们分两个阶段解决这个矛盾：

1. **导入器：alloca + load/store。** `mir-importer` 故意不直接构造 SSA。每个非 ZST 的 MIR 局部变量由一个栈槽位支撑 —— 在入口块顶部发出一个 `mir.alloca` 并记录在 `ValueMap` 中。对局部变量的每次写入变成 `mir.store` 到其槽位；每次读取变成 `mir.load`。因此，分支终止符都是**零操作数**的，每个非入口块都是**无参数**的：所有跨块数据流都通过 alloca 槽位，而不是块参数。

2. **`pliron::opts::mem2reg`：槽位 → SSA。** 在 `dialect-mir` 模块验证之后，`pipeline.rs` 运行 pliron 内置的 `mem2reg` pass。它将每个符合条件的 alloca 提升回 SSA 值，将每个负载重新连接到其到达定义，并在一个值沿多个控制路径合并的任何地方插入块参数（pliron 中 phi 节点的拼写）。地址被取用的槽位 —— 那些我们真正需要保留在栈上的 —— 保持不变，供 `dialect-mir` → `dialect-llvm` 降级翻译成真正的 `alloca`。

下面是一个小问题。给定 MIR，其中 `_1` 在 `bb0` 中写入，在 `bb1` 中读取：

```text
// Rust MIR
bb0: { _1 = 42_i32; goto -> bb1; }
bb1: { _0 = _1;     return; }
```

导入器首先发出这个基于 alloca 的 `dialect-mir`（`mem2reg` 之前）：

```text
^bb0:
  %s1 = mir.alloca           : !mir.ptr<i32>
  %c  = mir.constant 42_i32  : i32
  mir.store %c, %s1
  mir.goto ^bb1                     // 零操作数；_1 通过 %s1 流动
^bb1:                               // 无块参数
  %r = mir.load %s1 : i32
  mir.return %r : i32
```

在 `pliron::opts::mem2reg` 提升 `%s1` 之后，同一个函数变成了 SSA 形式的 `dialect-mir`，块参数仅在确实需要合并到达定义的地方出现：

```text
^bb0:
  %c = mir.constant 42_i32 : i32
  mir.goto ^bb1(%c : i32)
^bb1(%r : i32):
  mir.return %r : i32
```

（只有一个前驱的后继函数，如上面的例子，最终只有一个块参数；具有多个到达定义的合并点才是 `mem2reg` 引入非平凡 phi 风格参数的地方。）

要点：导入器从不运行活跃性分析，也从不跨块传递值。所有“哪个值在哪里活跃”的推理都推迟到 `mem2reg`，它已经可以在一次遍历中为整个 `dialect-mir` 模块正确解决。`translator/terminator/mod.rs` 模块的文档字符串以内联源代码形式包含了相同的工作示例，以便快速参考。

---

## 类型翻译

`types` 模块将 Rust 类型（通过 `rustc_public` 所见）转换为 `dialect-mir` 类型。大多数映射是直接的，但有几个值得注意：

| Rust 类型   | `dialect-mir` 类型    | 说明        |
| :------- | :--------- | :--------- |
| `i32`, `u64`, 等    | `IntegerType`                 | 带符号性跟踪             |
| `f32`, `f64`          | `Float32Type` / `Float64Type` | 标准 IEEE 754                    |
| `bool`                | `IntegerType(1)`              | 1 位整数，遵循传统       |
| `(A, B, C)`           | `MirTupleType`                | 异构乘积类型           |
| `&[T]`                | `MirSliceType`                | 指针 + 长度                     |
| `DisjointSlice<T>`    | `MirDisjointSliceType`        | 安全验证的可变切片        |
| `struct Foo`          | `MirStructType`               | 带有来自 rustc 布局的字段偏移量 |
| `*mut T` / `*const T` | `MirPtrType`                  | 带有 GPU 地址空间               |
| `enum Option<T>`      | `MirEnumType`                 | 判别式 + 变体              |

### 动态结构体布局

这里有一个微妙之处，可以为用户省去真正的麻烦。考虑这个结构体：

```rust
struct Extreme {
    a: u8,
    b: i128,
}
```

Rust 的布局算法可能会为了对齐而重新排序字段：

```text
用户编写：      struct Extreme { a: u8, b: i128 }
rustc 可能布局： [b: i128 @ offset 0][a: u8 @ offset 16]
MirStructType：    mem_to_decl 映射、偏移量、总大小
LLVM struct：      { i128, i8, [15 x i8] }   // 显式填充
```

cuda-oxide 向 rustc 查询每个字段的确切字节偏移量，并使用显式填充字节构建结构体类型。`MirStructType` 存储一个 `mem_to_decl` 映射（内存顺序到声明顺序）、每个字段的偏移量以及总大小。当降级到 LLVM 时，填充被物化为字段之间的 `[N x i8]` 数组。

实际结果是：**`#[repr(C)]` 不是必需的**，对于在主机和设备代码之间共享的类型。cuda-oxide 自动匹配 rustc 的布局，因此你的结构体可以使用 Rust 的默认 `repr(Rust)` 布局，编译器将在两边都做正确的事情。少记一个属性，少踩一个坑。

---

## 内联函数分发

当翻译器遇到 `Call` 终止符时，它不会立即发出 `mir.call` 操作。首先，它检查被调用者是否是一个*已知的内联函数* —— 来自 `cuda_device` 的函数，直接映射到 GPU 硬件指令，而不是一个有主体的函数。

`try_dispatch_intrinsic()` 函数根据被调用者的**完全限定域名（FQDN）**进行匹配，该名称从 `CrateDef::name()` 获得：

```rust
match name {
    "cuda_device::thread::threadIdx_x" => emit_nvvm_intrinsic(ReadPtxSregTidXOp),
    "cuda_device::warp::shuffle_xor"   => emit_warp_shuffle_i32(ShflSyncBflyI32Op),
    "cuda_device::sync::syncthreads"   => emit_nvvm_intrinsic(Barrier0Op),
    // ... 100+ 内联函数
    _ => translate_as_normal_call()
}
```

使用完整的 FQDN（例如 `cuda_device::thread::threadIdx_x`，而不仅仅是 `threadIdx_x`）进行匹配，以避免不同模块中同名函数之间的歧义。相同的 FQDN 也用作非泛型、非内联函数调用的调用目标名称 —— 收集器产生匹配的名称，降级层在两边都将 `::` 转换为 `__`。

如果该函数是已识别的内联函数，翻译器直接发出相应的 `dialect-nvvm` 操作 —— 没有函数体，没有调用开销，只有硬件指令。线程索引、线程束洗牌、屏障、共享内存操作、TMA 批量拷贝和矩阵乘法指令都走这条路径。

如果该函数*不是*内联函数，则回退到正常路径：发出一个 `mir.call` 操作，通过符号名称引用被调用者。被调用者的主体将被单独翻译（它也在收集的函数列表中），因此所有链接都能正确工作。

> **注意**：
> 内联函数分发表是为 cuda-oxide 添加新 GPU 操作的主要扩展点。如果 NVIDIA 发布了新指令并且你想暴露它，你需要向 `cuda_device` 添加一个函数，添加一个 `dialect-nvvm` 操作，并在此处添加一个匹配分支。有关逐步指南，请参阅[添加新内联函数](adding-new-intrinsics.md)。

---

## 处理展开路径

MIR 忠实地模拟了 Rust 的 panic 语义。每个函数调用都有两个可能的后继 —— 一个返回目标和一个展开目标：

```text
_2 = mul(_1, _3) -> [return: bb1, unwind: bb2]
```

在 CPU 上，展开路径很重要：它运行析构函数、展开栈，然后要么捕获 panic，要么中止进程。在 GPU 上，CUDA 工具链目前不暴露此功能 —— nvcc/ptxas 会剥离着陆垫，没有异常处理基础设施能存活到 PTX。硬件本身*可以*支持展开（绝对分支 + Volta 之后的每线程调用栈跟踪是足够的），但编译器和运行时没有连接它。NVIDIA 有一个正在进行的项目，为汽车安全添加 C++ 异常支持；当前的 cuda-oxide 设计与该工作向前兼容。

目前，cuda-oxide 将**所有展开路径视为不可达**。如果运行时发生 panic —— 例如，调试模式下的整数溢出或显式的 `panic!()` —— GPU 会陷入陷阱，kernel崩溃。这在语义上等同于 `panic=abort`，而无需用户设置标志。

在实践中，翻译器只是忽略每个 `Call` 和 `Assert` 终止符中的展开目标，只生成返回路径分支。展开块永远不会被翻译。它们消失了，就像从未存在过一样。

这并不像听起来那么可怕。Rust 的借用检查器和类型系统阻止了大部分会导致 panic 的错误。对于那些漏掉的错误（数组边界检查、对 `None` 调用 `unwrap`），GPU 陷阱无论如何都是正确的行为 —— GPU 线程没有办法从kernel中间的逻辑错误中“恢复”。

---

## 整合起来

让我们跟踪一个简单的kernel通过整个流水线，看看所有部分如何连接。这是一个向量加法kernel：

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

在 `mir-importer` 将稳定 MIR 翻译成 `dialect-mir`（并且 `pliron::opts::mem2reg` 已将 alloca 槽位提升回 SSA）之后，结果看起来像这样（简化，为清晰起见省略了许多细节）：

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

需要注意的几点：

- **`thread::index_1d()`** 作为内联函数被分发，变成了 `nvvm.read_ptx_sreg_tid_x` —— 直接读取 GPU 寄存器，而不是函数调用。
- **`DisjointSlice::get_mut()`** 变成了边界检查（`mir.lt`）和一个条件分支。Rust 中的 `if let Some` 模式变成了显式的控制流。
- **块参数**这里不存在。请记住，`mir-importer` 本身发出的是**零操作数**的每个分支终止符，并且每个非入口块都是无参数的 —— 值通过 alloca 槽位跨越块边界，直到 `pliron::opts::mem2reg` 运行。在这个函数中，`mem2reg` 可以看到没有什么需要在 `^compute` 或 `^exit` 处合并（没有值在分支后存活），因此它提升槽位而不引入任何块参数。一个更复杂的kernel，其值跨越回边存活，在 `mem2reg` 之后会以 `mir.goto ^header(%i, %acc)` 和 `^header(%i: i32, %acc: f32)` 结束，这是 pliron 中 phi 节点的拼写。
- **没有展开路径。** 原始 MIR 中每个可能 panic 的操作都有展开目标。它们都消失了。

从这里开始，流水线接手：

1. **验证** —— pliron 检查每个操作的类型是否匹配，每个块的参数是否正确，以及支配关系是否成立。
2. **降级** —— `lower_mir_to_llvm` 将 `mir.add` 转换为 `llvm.fadd`，`mir.load` 转换为 `llvm.load`，`mir.slice` 转换为一个指针加长度的 LLVM 结构体，依此类推。
3. **导出** —— `dialect-llvm` 被打印为文本 `.ll` 文件，并带有适当的 `!nvvm.annotations` 元数据，将 `vecadd` 标记为kernel入口点。
4. **llc** —— LLVM 的 NVPTX 后端将 `.ll` 编译为 `.ptx`，结果写入主机二进制文件旁边。

`dialect-mir` 忠实地捕获了 Rust 语义。下一步是将其降级为 LLVM 可以理解的内容 —— 在[降级流水线](./lowering-pipeline.md)中介绍。

| [上一页](./rustc-codegen-cuda.md) | [下一页](./pliron-dialects.md) |
| :--- | ---: |