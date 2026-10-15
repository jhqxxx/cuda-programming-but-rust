# 已支持的特性

本附录给出 cuda-oxide 的特性矩阵：每一项编译器能力、运行时 API 与硬件特性，
以及它们当前的支持状态。这些数据取自编译器/运行时的源码与测试套件。

**图例：** **Full** = 已测试且可用，**Partial** = 已发布且可用但存在已知缺口
（在该行描述中说明），**Planned** = 在路线图上，**N/A** = 不适用或尚未发现
相关需求。

---

## 编译器：内存模型

| Feature | Status | Description |
|:--------|:-------|:------------|
| HMM / Unified Memory Management | **Full** | GPU 无需 `cudaMemcpy` 即可直接读写宿主内存。闭包中的引用捕获利用 HMM 访问宿主指针。需要 Turing+ GPU、Linux 6.1.24+、CUDA 12.2+。 |
| Unified Struct ABI (no `#[repr(C)]`) | **Full** | 设备端结构体布局与宿主完全一致。编译器查询 rustc 的实际布局，并在 LLVM IR 中用显式填充复现它。可用于 `#[repr(Rust)]` 默认布局。 |
| Dynamic Layout Matching | **Full** | 编译器查询 rustc 的 `fields_by_offset_order()` 与字节偏移，构建字段顺序和显式填充字节都正确的 LLVM 结构体。与 LLVM 的 datalayout 无关。 |
| Packed Layouts (`#[repr(packed)]`) | **Partial** | 通过指针取字段地址（`addr_of!((*p).field)`）时使用 rustc 的精确字节偏移，因此 `read_unaligned`/`write_unaligned` 往返可用，包括 `packed(N)`。字节级忠实的 packed 结构体也可使用 packed LLVM 存储进行按值构造/加载/存储以及 `[Packed; N]` 元素寻址。当所选的自然或 packed LLVM 表示能精确复现 rustc 的字段偏移与存储大小时，可递归提升的 packed 常量数组走不可变设备全局路径。重叠的或以其他方式无法表示的布局仍会被拒绝。 |
| Pointer Distance (`offset_from`) | **Full** | `ptr_offset_from` / `ptr_offset_from_unsigned` intrinsics（以及 `offset_from`、`offset_from_unsigned`、`byte_offset_from`、`byte_offset_from_unsigned` 方法）下沉为地址差除以 rustc 报告的 pointee 大小，返回 `isize`（有符号）或 `usize`（无符号）。pointee 为零大小时报错。 |
| Volatile Load/Store | **Full** | `core::ptr::read_volatile` / `write_volatile` 在 MIR 导入、mem2reg（volatile 访问永不提升）、MIR 到 LLVM 的下沉以及文本导出（`load volatile` / `store volatile`）中都携带显式的 volatile 位。在 PTX 中生成 `ld.volatile` / `st.volatile`。 |
| Bulk Copy (`copy_nonoverlapping`) | **Full** | `core::ptr::copy_nonoverlapping` 下沉为一个 `mir.memcpy` op，随后是 `llvm.memcpy`，元素计数按 pointee 换算为字节数。intrinsic 的重载后缀由操作数的地址空间和长度宽度导出。 |

## 编译器：类型系统

| Feature | Status | Description |
|:--------|:-------|:------------|
| Generics and Monomorphization | **Full** | 带 trait bound 的泛型 kernel 与设备函数。单态化实例从 rustc MIR 中收集。支持 const 泛型。 |
| Enums (`Option<T>`, `Result<T,E>`, custom) | **Full** | 完整的枚举支持，包括判别值提取与载荷访问。枚举的模式匹配可用。 |
| Struct Construction and Field Access | **Full** | 结构体字面量、字段访问、按值传递与返回值。用户自定义结构体无需标注即可使用。 |
| Array Types (`[T; N]`) | **Full** | 静态构造、常量索引与运行时索引访问。数组值常量（裸的与嵌套的）会被具现化。可变数组自动提升为内存支撑的形式。 |
| `CuSimd<T, N>` SIMD Type | **Full** | 带命名访问器（`x`/`y`/`z`/`w`）的泛型 SIMD 寄存器类型，支持运行时与编译期索引、`to_array` 转换。 |
| ABI Scalarization | **Full** | 切片在 kernel 边界处被标量化（`&[T]` -> `(ptr, len)`，并在函数内部重建）。结构体与闭包按值作为单个 byval `.param` 传递；字段展平在设备到设备的内部调用中仍然适用。 |
| Grid-constant parameters | **Full** | `#[grid_constant] value: &T` 按值传递 `T`，并保留一个 grid 范围内的只读地址。宿主启动是 unsafe 的。要求 pointee 非零大小且无内部可变性、引用具有启动作用域，以及 Volta+ 目标。 |

数组值常量支持基本类型叶子（整数、`f16`、`f32`、`f64`）、嵌套数组，以及由受
支持的标量、枚举、元组和零大小字段递归组成的元组。元组元素 stride 与字段
偏移来自 rustc 布局，包括内部填充与尾部填充；直接的元组值常量使用同一套
感知布局的解码器。结构体常量（直接的和按引用提升的）同样在 rustc 布局偏移处
读取每个字段，因此带填充的、重排序的、`#[repr(C)]` 的以及嵌套的形状都能正确
解码，而结构体的存储大小就是它的填充后大小，这为常量内部带填充结构体的数组
固定了元素 stride。无指针的已初始化 union 常量从 rustc 求值得到的存储映像
具现化，而不去猜测活跃字段：已初始化的字节被保留，未初始化的非活跃字节保持
`undef`，字节映像被 transmute 为与布局精确一致的 union 类型。这包括直接的
union、嵌套在元组或结构体常量中的 union、运行时索引的 `[U; N]`，以及
`MaybeUninit<T>` 常量。元素为可递归提升结构体的裸数组与标量和元组表走同一条
不可变设备全局路径，从而避免只读使用时每线程一份本地表拷贝。这种提升同样
接受受支持的裸指针/引用叶子。它们求值得到的字节映像保持字节精确，而指针槽位
被保留为符号化的设备全局重定位，包括指向被引用设备静态量的非零字节加数。
重定位目标会被显式具现化，提升全局的去重同时考虑重定位标识以及类型和字节，
因此指向不同静态量的字节完全相同表不会发生别名。

形如 `const R: &[Struct; N] = &TABLE` 的指向数组的指针常量，在每一个元素都可
递归提升且转换后的存储大小与 rustc 布局一致时，使用同一个被提升的不可变全局
量。当外层指针选择了后备分配的一个子区间时，该区间内的重定位源偏移会被重定基
到被提升的初始化器中，同时保留其目标与加数。不受支持的裸数组常量在可用时保留
现有的逐元素回退；当不存在正确的回退时，指向数组的指针常量继续采用 fail-closed
（安全失败）策略。零字节且过度对齐的结构体叶子（例如 `repr(align(N))` 的
ZST）仍走现有的对对齐敏感的值路径，而不走这条提升路径。当下沉能够用精确的
packed LLVM 结构体复现其记录的字段偏移时，提升也包含 `repr(packed)` 与
`repr(packed(N))` 结构体；其值形式与引用形式会去重为同一个不可变初始化器。
重叠的或以其他方式无法表示的结构体布局仍在不可变提升范围之外，并保持现有的
fail-closed 行为。

数组、元组与结构体 **const** 值中不采用不可变表提升路径的裸指针字段，会按字段
通过 `MirGlobalAllocOp` 具现化，包括指向某个静态量的非零字节加数（见
`struct_constant_provenance`、`tuple_constant_provenance`、
`tuple_array_provenance`）。`array_constants` 回归测试还覆盖了一张同时含有
零加数静态引用和非零静态子对象加数的被提升带指针元组表，并验证优化后的代码
不会保留每线程一份的表仓库。

聚合常量中的切片胖指针字段同样受支持，只要其数据指针重定位到某个设备静态量，
且 pointee 是同元素的数组转切片视图。它们字面的 `usize` 长度元数据会被独立
解码，包括非零静态字节加数与嵌套聚合字段偏移。仅含裸指针的 union 常量通过
重建一个带类型的指针载体（而不是 transmute 占位字节）来保留同样的重定位来源
信息，包括非零加数。兼容的 `SharedRef` 备选项必须具有相同的转换后 pointee
类型；`UniqueRef` 的 union 常量仍被拒绝。裸指针备选项之间只允许 pointee 视图
不同，同时保持相同的裸指针种类、可变性与地址空间。存储恰好为一个完全初始化、
自然对齐的指针字、其指针备选项是同一类型的泛型裸指针、其整数备选项是全宽度
的无重定位指针/整数 union，可以改用 rustc 求值得到的字节映像。导入器只把该
映像 transmute 为整数字段并将其插入 union，因此不会具现化任何非活跃的指针
备选项。带重定位的指针/整数 union、union 中的胖指针或嵌套指针存储、过度对齐
或带填充的指针 union、不受支持的胖指针元数据，以及指向数组的 union 常量
（`&[U; N]`）仍被拒绝。顶层的仅含裸指针的设备全局 union 初始化器可以在字节
零处保留一个全宽度重定位；混合指针/整数的设备全局初始化器仍被拒绝。

枚举常量保留指向设备静态量的载荷重定位，包括非零字节加数。这包括 niche 编码
的 `Option<&T>` 与直接带标签的枚举布局，既适用于直接的裸引用载荷，也适用于
嵌套在元组、结构体或数组载荷字段中的指针。携带重定位的枚举常量还可以嵌套在
元组、结构体与数组常量内部。匿名提升分配仍不受支持。

## 编译器：闭包

| Feature | Status | Description |
|:--------|:-------|:------------|
| Move Closures (`FnOnce`) | **Full** | 按值捕获的闭包。整个闭包结构体作为一个 byval kernel 实参推入。`move \|x\| x * factor` 模式。 |
| Reference Closures (`Fn`/`FnMut`) | **Full** | 按引用捕获的非 move 闭包。闭包结构体（其中含宿主指针）仍作为一个 byval 实参传递；GPU 通过那些指针经 HMM 读取。 |
| Host-to-Device Closures | **Full** | 在宿主上定义、传给泛型 kernel 的闭包。已测试带捕获系数的多项式求值。 |
| Device-Internal Closures | **Full** | 完全在设备上创建和使用的闭包，包括传给设备函数的闭包。 |

## 编译器：控制流

| Feature | Status | Description |
|:--------|:-------|:------------|
| Match Expressions (integer switch) | **Full** | 整数上的多路 match。生成条件分支链。 |
| Match on Enums | **Full** | 对 `Option<T>` 与自定义枚举的模式匹配。判别值提取 + 载荷访问。 |
| For Loops (range, iterator, enumerate) | **Full** | 完整的迭代器脱糖：基于 range、`slice.iter()`、`enumerate()`、嵌套循环、`break`、`continue`。 |
| While Loops / If-Else | **Full** | 基线控制流得到完全支持。 |
| Break and Continue | **Full** | for/while 循环中的 `break` 与 `continue`，包括提前退出。 |
| Loop Unroll Annotations | **Partial** | `#[unroll]` 与 `#[unroll(N)]` 请求对显式计数的 `while` 循环展开。嵌套循环与多条 `continue` 路径可用；完全展开会保留 `break` 路径与多个退出目标，而部分展开要求循环是正步长的 `<`/`<=` 循环、具有循环不变量上限，且只有正常的头部出口。请求上限为 1,024 份拷贝、8,192 个克隆块和 65,536 个克隆操作。 |
| Monomorphization-Dead Branches | **Partial** | 泛型特化后变为死代码的分支（例如 `if M::ENABLED` 的 const false 分支）会被符号收集、panic 检查和指针地址空间推断忽略，因此死分支中仅用于 panic 的钩子可以编译通过。只有 rustc 自身会折叠的 switch 才会被裁剪：一个常量判别操作数，或一个直接的单次赋值常量。多步常量拷贝链保留两个分支，这与 rustc 的宿主端单态化一致；这是有意为之，而不是一个通用的常量传播 pass。 |

## 编译器：算术与类型转换

| Feature | Status | Description |
|:--------|:-------|:------------|
| 64-bit Arithmetic | **Full** | 完整的 64 位整数算术，包括移位、位运算与描述符字段打包。 |
| Type Casting (all kinds) | **Full** | IntToInt、IntToFloat、FloatToInt、FloatToFloat、Transmute（bitcast）、PtrToPtr、PtrToInt、IntToPtr，以及指针强制转换。 |
| Packed bf16x2 FMA | **Full** | `bf16x2::fma_bf16x2(a, b, c)` 下沉为 PTX `fma.rn.bf16x2`，每个 `u32` 含两条 bf16 通道。sm_80+。 |

## 编译器：互操作性

| Feature | Status | Description |
|:--------|:-------|:------------|
| Bi-directional LTOIR Support | **Full** | Rust kernel 调用 CUDA C++ 设备函数，**并且** C++ 调用 Rust 设备函数。经由 NVVM IR → libNVVM → LTOIR → nvJitLink。 |
| Device FFI (`extern "C"`) | **Full** | 用 `#[device] extern "C" { fn ... }` 声明外部 LTOIR 函数。已演示 CUB/CCCL 集成。 |
| MathDx FFI (cuFFTDx / cuBLASDx) | **Full** | 通过 LTOIR 使用 cuFFTDx（8/16/32 点线程级 FFT）与 cuBLASDx（32x32x32 block 级 GEMM）。 |
| Tile interop | **Partial** | kernel 间互操作目前可用：一个 [cutile-rs Tile kernel](https://github.com/NVIDIA/cuda-rust/tree/main/cutile-rs) 与一个 cuda-oxide SIMT PTX kernel 可以在同一个宿主进程中、在同一条 CUDA 流上、基于共享的设备 tensor 运行。kernel 内的 Tile 互操作仍在进行中，跟踪于 [#96](https://github.com/NVlabs/cuda-oxide/issues/96)。 |
| Cross-Crate Kernels | **Full** | 在库 crate 中定义的 kernel 与设备函数，在二进制 crate 的使用处进行单态化。 |

## 编译器：函数

| Feature | Status | Description |
|:--------|:-------|:------------|
| `#[kernel]` Attribute | **Full** | 将函数标记为 GPU kernel 入口点（`ptx_kernel` 调用约定）。单个文件可含多个 kernel。 |
| `#[device]` Helper Functions | **Full** | 可从 kernel 调用的设备侧辅助函数。`#[inline(always)]` 被保留为 LLVM 的 `alwaysinline` 属性（与 convergent group 以及任何 `!dbg` 作用域一并生成），因此 `opt` 会尊重内联意图。 |
| Standalone `#[device]` Functions | **Full** | 在没有任何 kernel 的情况下编译的设备函数。为 C++ 消费提供干净的导出名。 |
| Multi-Kernel Modules | **Full** | 单个源文件中的多个 `#[kernel]` 函数编译为一个 PTX 模块。 |

## 编译器：编译管线

| Feature | Status | Description |
|:--------|:-------|:------------|
| Unified Single-Source Compilation | **Full** | 宿主代码与设备代码位于同一文件。自定义 rustc codegen backend 拦截代码生成。无需 `#[cfg]`。 |
| PTX Output | **Full** | 默认输出：Rust MIR → `dialect-mir` → `mem2reg` → 带注解的循环展开 → LLVM dialect → LLVM IR → `llc` → PTX。目标涵盖 sm_80 到 sm_100a。 |
| NVVM IR Output | **Full** | 对 Blackwell 之前的 GPU 选择 LLVM 7 的带类型指针语法，对 Blackwell 及更新的 GPU 使用不透明指针语法。生成的模块由 libNVVM 校验，不受支持的遗留操作会产生编译错误。 |
| LTOIR Linking | **Full** | 通过 libNVVM 与 nvJitLink 进行设备侧 LTO。 |
| Float Math Intrinsics (libdevice) | **Full** | Rust 的 `f32`/`f64` 数学方法（`sin`、`cos`、`exp`、`pow`、`sqrt`、……）在 Blackwell 之前及 Blackwell GPU 上都下沉为 CUDA libdevice（`__nv_*`）。cuda-oxide 会自动选择匹配的 NVVM IR 语法。在 Blackwell 上，运行时还可以 JIT 由标准 Blackwell 之前目标（例如 `sm_86`）产出的 PTX。 |
| Pipeline Inspection | **Full** | `cargo oxide pipeline <example>` 展示导入后的以及经过 `mem2reg` 之后的 MIR、LLVM dialect、导出的 LLVM IR 以及 PTX。 |
| PTX Inspect | **Full** | `cargo oxide inspect <example>` 构建并打印生成的 PTX，而不输出完整的管线转储。 |
| Local Clean | **Full** | `cargo oxide clean` 删除项目本地的 `target/` 目录与生成的设备产物（`.ptx`、`.ll`、`.opt.ll`、`.ltoir`、`.cubin`、`.target`、`.options`、`.cubin.target`），绝不删除共享的 `~/.cargo/cuda-oxide/` 缓存。 |
| Compute Sanitizer Wrapper | **Full** | `cargo oxide sanitize <example>` 构建示例并在 NVIDIA Compute Sanitizer（`memcheck`、`racecheck`、`initcheck` 或 `synccheck`）下运行宿主二进制。 |
| cuda-gdb Source Debugging | **Full** | `cargo oxide debug` 在 PTX 路径上构建设备调试信息，并启动 `cuda-gdb`。遗留 NVVM IR 尚不支持调试元数据。 |
| cuda-gdb Local / Argument Inspection | **Partial** | 完整模式把受支持的局部变量保留在内存中。它禁用 `ScalarReplacementOfAggregates` 与 `SingleUseConsts`，保留 `ReferencePropagation` 与常规 MIR 内联，并且只把 `DisjointSlice::get_mut` 外联（outline）。常见的标量、指针、聚合、闭包、枚举以及选定的投影都可用；当前限制见 [调试与错误处理](../gpu-programming/error-handling-and-debugging.md)。 |

## 编译器：内联 PTX

| Feature | Status | Description |
|:--------|:-------|:------------|
| `ptx_asm!` Macro | **Partial** | 带 `%0` 操作数、`in`、`out` 与 `inout` 的 CUDA 内联 PTX；`out` 上带 `=` 前缀约束、`inout` 上带 `+` 前缀约束时最多 16 个输出操作数；最多 16 个显式输入；CUDA 寄存器约束 `h`、`r`、`l`、`q`、`f` 与 `d`；立即数整数约束 `n`；编译期字符串约束 `C`；`clobber("memory")`；以及用于纯寄存器片段的 `options(register_only)`。有多个输出操作数时，宏按声明顺序写回元组结果。默认情况下，片段被视为有副作用，并留在其当前控制流中。仅对可以安全跨越发散控制流移动的纯片段使用 `options(register_only, may_diverge)`；**绝不要**对 `.sync` 指令或集合操作使用它。超过 16 个输出操作数尚未实现。 |

---

## 运行时库：安全性

| Feature | Status | Description |
|:--------|:-------|:------------|
| `DisjointSlice<T, IndexSpace>` | **Full** | 带边界检查的并行写输出切片。`IndexSpace` 拒绝不匹配的布局；唯一性还要求匹配的预备启动几何（或一个裸的 unsafe 证明）。 |
| `ThreadIndex<'kernel, IndexSpace>` | **Full** | 不透明、不可转移的见证。`index_1d` 的唯一性要求 Y/Z 维度未启用，这由 `domain = 1` 的预备启动或裸 unsafe 启动的调用者来证明。 |
| Proof-carrying static views | **Full** | 一个经过检查的 `u32` 线程索引证明了一个完整元素或 tile，随后 `at_const` 访问编译期位置时无需再做运行时边界检查。 |
| `PreparedLaunch<K>` | **Full** | 经过检查、可复用、为确切 kernel 打上标记的启动几何。裸的 `LaunchConfig` 生成方法是 unsafe 的。 |
| `ManagedBarrier` Typestate | **Full** | 编译期的 barrier 生命周期：`Uninit → Ready → Invalidated`。非法的转换是编译错误。 |

## 运行时库：原子操作

| Feature | Status | Description |
|:--------|:-------|:------------|
| Device-Scope Atomics | **Full** | `DeviceAtomic{U32,I32,U64,I64,F16,F32,F64}`，作用域为 `.gpu`。支持全部 5 种 ordering。 |
| Block-Scope Atomics | **Full** | `BlockAtomic{U32,I32,U64,I64,F16,F32,F64}`，作用域为 `.cta`。 |
| System-Scope Atomics | **Full** | `SystemAtomic{U32,I32,U64,I64,F16,F32,F64}`，作用域为 `.sys`。用于 CPU-GPU 共享数据。 |
| `core::sync::atomic` Support | **Full** | 标准库原子类型下沉为 PTX `atom.sys` 指令。 |

## 运行时库：共享内存

| Feature | Status | Description |
|:--------|:-------|:------------|
| Static Shared Memory | **Full** | `SharedArray<T, N, ALIGN>` —— 编译期确定大小、作用域为 block。对齐可选，最高 256B。 |
| Dynamic Shared Memory | **Full** | `DynamicSharedArray<T, ALIGN>` —— 运行时确定大小，通过 `LaunchConfig::shared_mem_bytes` 设置。 |
| Distributed Shared Memory (DSMEM) | **Full** | 在 cluster 内直接访问其他 block 的共享内存。用 `map_shared_rank()` 做地址映射。sm_90+。 |

## 运行时库：线程与同步

| Feature | Status | Description |
|:--------|:-------|:------------|
| Thread/Block/Grid Intrinsics | **Full** | `threadIdx`、`blockIdx`、`blockDim`、`gridDim`。索引见证是带布局类型的；它们的唯一性还取决于启动维度是否匹配。`index_2d_runtime(&slice)` 针对切片自身的行宽解析，该行宽由宿主绑定一次。见[安全模型](../gpu-safety/the-safety-model.md)。 |
| Block Synchronization | **Full** | `sync_threads()` —— 线程块 barrier。 |
| Async Barriers (mbarrier) | **Full** | Hopper+ 的硬件异步 barrier：init、arrive、test_wait、try_wait、inval。 |
| Cluster Synchronization | **Full** | 对 cluster 中所有 block 的 `cluster_sync()`。sm_90+。 |
| Fence Operations | **Full** | 用于 TMA 可见性的 `fence_proxy_async_shared_cta()`，以及 `nanosleep(ns)`。 |

## 运行时库：Warp

| Feature | Status | Description |
|:--------|:-------|:------------|
| Warp Shuffle Operations | **Full** | `shuffle`、`shuffle_xor`、`shuffle_down`、`shuffle_up`。无后缀形式接受 `u32`；还有 `_f32`、`_u64`、`_f64` 变体，以及每个的无 `_sync` 形式。 |
| Warp Vote Operations | **Full** | `all(pred)`、`any(pred)`、`ballot(pred)` → 位掩码。 |
| Lane/Warp ID | **Full** | `lane_id()`（0–31）、`warp_id()`。直接读取寄存器。 |
| Warp Reduction (`redux.sync`) | **Full** | 单指令的整 warp 归约。sm_80+ 上的整数：`redux_sync_add`、`_min_{u32,i32}`、`_max_{u32,i32}`、`_and`、`_or`、`_xor`。`sm_100a`/`sm_100f`/`sm_103a`/`sm_103f` 上带可选 `.abs` 与 `.NaN` 的 `f32` min/max。没有 `f64` 形式。 |

## 运行时库：协作组

| Feature | Status | Description |
|:--------|:-------|:------------|
| Typed Group Handles | **Full** | `Grid`、`Cluster`、`ThreadBlock`、`WarpTile<N>`（N ∈ {1,2,4,8,16,32}）、`CoalescedThreads`。 |
| Group Universal API | **Full** | 每个 group 句柄上的 `size()`、`thread_rank()`、`sync()`。 |
| Warp Tile Partitioning | **Full** | `ThreadBlock::tiled_partition::<N>()` 把 block 划分为固定大小的 `WarpTile<N>` 组；block 的线程数必须能被 `N` 整除。`coalesced_threads()` 具现化活跃 lane 组。 |
| Warp Collectives | **Full** | `ballot`、`all`、`any`、`shfl`、`shfl_xor`、`shfl_down`、`shfl_up`（`u32` 与 `f32`）；`match_any` / `match_all`（`i32` 与 `i64`）；`active_mask`。 |
| Warp Reductions / Scans | **Full** | `warp_reduce`、`warp_scan`（含前缀）。`u32`/`i32`/`f32` 的 `Sum`/`Min`/`Max`；`u32` 的 `BitAnd`/`BitOr`/`BitXor`。 |
| Block Reductions / Scans | **Full** | `block_reduce`、`block_scan`（含前缀）。`NUM_WARPS` 是共享内存中以 warp 总数计的容量，必须至少为 `ceil(block_threads / 32)`；支持末尾不满的 warp。操作/类型矩阵与 warp 变体相同。 |
| Cooperative Kernel Launch | **Full** | 在 `#[cuda_module]` kernel 上使用 `#[cooperative_launch]`（或 `unsafe { cuda_launch! { cooperative: true, ... } }`）可为 grid 级 barrier 启用 `Grid::sync()`。 |

## 运行时库：调试

| Feature | Status | Description |
|:--------|:-------|:------------|
| `gpu_printf!` Macro | **Full** | 带完整格式说明符支持的格式化 GPU 输出。下沉为 `vprintf`。 |
| `gpu_assert!` Macro | **Full** | 无消息形式在失败时调用 `trap()`。字符串字面量消息形式调用 CUDA 的设备侧 `__assertfail`，上报消息与调用点元数据，并浮现 `CUDA_ERROR_ASSERT`。 |
| Debug Intrinsics | **Full** | `clock()`、`clock64()`、`trap()`、`breakpoint()`、`prof_trigger::<N>()`。 |

## 运行时库：Kernel 启动

| Feature | Status | Description |
|:--------|:-------|:------------|
| `#[cuda_module]` Typed Launch | **Full** | 内嵌模块加载，具有带类型的同步/异步实参。裸的配置方法是 unsafe 的。 |
| `#[launch_contract]` / `PreparedLaunch<K>` | **Full** | 检查维度、确切的 block 形状、资源、能力、上下文以及 kernel 标识。 |
| `cuda_launch!` Macro | **Full** | 用于运行时加载模块的 unsafe 低层启动；需要 `unsafe { }`。 |
| `cuda_launch_async!` Macro | **Full** | unsafe 低层惰性启动；需要 `unsafe { }`。 |
| `#[launch_bounds]` | **Full** | 占用率提示：每 block 最大线程数、每 SM 最少 block 数。 |
| `#[cluster_launch]` | **Full** | 编译期 cluster 维度。在 PTX 中生成 `.reqnctapercluster`。 |

## 运行时库：TMA

| Feature | Status | Description |
|:--------|:-------|:------------|
| TMA Bulk Tensor Copy (1D–5D) | **Full** | `cp_async_bulk_tensor_{1..5}d_g2s`。128 字节的 TMA 描述符。sm_90+。 |
| TMA Multicast | **Full** | 单次 TMA 加载广播到 cluster 中所有 CTA。完整 multicast 需要 sm_100a。 |
| TMA Commit/Wait Groups | **Full** | 用 `cp_async_bulk_commit_group`、`cp_async_bulk_wait_group` 跟踪异步完成。 |

---

## 运行时库：矩阵与 Tensor Core

| Feature | Status | Description |
|:--------|:-------|:------------|
| Warp-Level MMA (`wmma`) | **Full** | 仅寄存器形式的 `mma.sync` 形状、`movmatrix`，以及 warp 协作的 `ldmatrix` 加载。 |
| Sparse MMA | **Full** | 在同一个 `wmma` 模块中，与稠密形状并存的稀疏结构化 `mma.sp` 形状。 |
| Warpgroup MMA (`wgmma`) | **Partial** | Hopper `sm_90a`：fence/commit/wait 流水线、共享内存描述符，以及以 `bf16`/`f16` 为输入、`f32` 为累加的 `m64n64k16` MMA。缺口：下沉只覆盖特定的、经过验证的循环模式（`bf16` 在直线代码、计数 K 循环与部分等待流水线中可用；`f16` 仅在直线代码中可用），且 `tf32` 调用会被拒绝，等待 [#1076](https://github.com/NVlabs/cuda-oxide/issues/1076)。 |
| Tensor Core Gen 5 (`tcgen05`) | **Full** | Blackwell sm_100+：TMEM 分配/释放、MMA、`stmatrix`、CTA 对（cg2）变体。 |
| Accumulator Fragment Algebra (`mma_frag`) | **Full** | `m16n8k16` 累加器的索引代数，使某个 lane 可以寻址 `[f32; 4]` 片段中属于自己的槽位。 |
| FP8 / FP6 / FP4 Formats | **Partial** | 转换（`e4m3`/`e5m2` 的 `convert::cvt_*`）与矩阵路径（FP8 的 `mma.sync` 形状、`mxf8f6f4` 形状、tcgen05 描述符）都已发布；`mma_mxf8f6f4` 示例会编译它们。缺口：这些格式上没有*算术运算*。没有像 `f16` 和 `bf16` 那样的 add/mul/min/max，因此这些值以被打包的位模式携带，并在使用前转换。 |

---

## 尚未实现

| Feature | Status | Notes |
|:--------|:-------|:------|
| Rust `asm!` macro | **Planned** | CUDA 内联 PTX 请使用 `ptx_asm!`。Rust MIR `InlineAsm` 的直接下沉尚未实现。 |
| Dynamic Dispatch (`dyn Trait`) | **N/A** | 请使用带静态分派的泛型。尚未发现真实需求。 |
| Heap Allocation (`Box`, `Vec`) | **N/A** | CUDA 有设备侧堆（kernel 中的 `malloc`/`free`），编译器也放行 `alloc` crate —— 但今天还没有接上设备侧的 `#[global_allocator]`。即便接上了，设备端 `malloc` 也极慢（串行化、易碎片化、非合并访问）。请使用切片与 `SharedArray`。 |
| `String` / `format_args!` | **N/A** | 格式化输出请使用 `gpu_printf!`。 |
| Panic / Unwinding | **N/A** | panic 路径在 MIR 中存在，但编译器会剥离 `core::panicking::*` 以及所有 unwind 边。GPU 硬件*可以*支持 unwinding（绝对分支 + Volta 之后的逐线程调用栈跟踪），但 CUDA 工具链（nvcc/ptxas）今天并未暴露它 —— 没有任何 landing pad 能存活到 PTX。如果运行时走到 panic 路径，GPU 会 trap（与 `panic=abort` 相同）。NVIDIA 有一个活跃项目要为汽车安全场景给 CUDA 增加 C++ 异常支持；当前 cuda-oxide 的设计与该工作是向前兼容的。今天请使用 `gpu_assert!()` + `trap()` 做显式的运行时检查。 |
| Standard Library (`std`/`alloc`) | **N/A** | `std` 被禁止。`alloc` 被收集器放行但没有支撑用的分配器。只有 `core` 是完全可用的。`Option`、`Result`、迭代器都能工作。 |
| Texture Memory | **N/A** | 鉴于 Hopper+ 上已有 TMA，优先级较低。 |
