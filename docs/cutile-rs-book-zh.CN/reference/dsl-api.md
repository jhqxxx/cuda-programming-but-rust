# DSL API

> **状态**：该 API 正在积极开发中，请预期它会发生变化。

cuTile Rust 的 DSL 让你用 Rust 语法编写 GPU kernel 代码，并编译为 [CUDA Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/)。在 `#[cutile::module]` 块内部，你使用本文档所记录的类型和操作来编写类 Rust 代码。编译器把这段代码翻译成 Tile IR 字节码，再调用 `tileiras` 为 NVIDIA GPU 生成 GPU 代码。

所有 DSL 类型与函数都可以在模块块内通过 `use cutile::core::*;` 获得。

---

## 函数

### 模块与入口点

GPU kernel 定义在 `#[cutile::module]` 块内部。每一个带有 `#[cutile::entry()]` 标记的函数都会成为一个可启动的 kernel。

```rust
#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,       // 可变：已分区的输出
        a: &Tensor<f32, { [-1] }>,         // 不可变：只读输入
        b: &Tensor<f32, { [-1] }>,         // 不可变：只读输入
    ) {
        let pid = program_id(0);
        let tile_a: Tile<f32, S> = a.load_tile(shape!(S), [pid]);
        let tile_b: Tile<f32, S> = b.load_tile(shape!(S), [pid]);
        output.store(tile_a + tile_b);
    }
}
```

**入口点参数：**

| 参数类型 | 说明 | 宿主侧类型 |
|---|---|---|
| `&mut Tensor<E, S>` | 可变输出（必须放在第一位） | `Partition<Tensor<E>>` 或 `Partition<&mut Tensor<E>>` |
| `&Tensor<E, S>` | 不可变输入 | `&Tensor<E>`、`Arc<Tensor<E>>`、`Tensor<E>` 或 `&TensorView<E>` |
| `f32`、`i32` 等 | 标量值 | 同一类型，按值传入 |
| `*mut T`、`*const T` | 裸设备指针 | `DevicePointer<T>` |

**约定：** 可变输出 tensor 永远是第一个参数。

**入口属性：**

| 属性 | 类型 | 说明 |
|---|---|---|
| `print_ir = true` | bool | 在 JIT 编译期打印两样产物：生成的入口点包装（Rust）以及编译后的 Tile IR 文本 |
| `unchecked_accesses` | bool | **Unsafe 豁免。** 不发出*任何*安全检查 —— kernel 内、启动时以及编译期都不做校核。要求 `unsafe fn`。由调用者断言正确性 |
| `deny_in_kernel_checks` | bool | **安全且严格。** 若有任何安全检查会残留在 kernel 内（即无法在编译期消除、也无法提升到启动时），则编译期报错。检查依然会被完整校核 —— 它只是禁止那些会占用设备寄存器的检查。诊断信息会指出那个无法离开 kernel 的检查及其原因 |
| `preconditions = ( ... )` | expr | 声明的形状事实，由生成的启动器在 kernel 运行前校验，并被编译器在消除安全检查时直接假定。两种形式：`dim(a, i) == dim(b, j)`（两个轴范围相等）与 `dim(t, k) % d == 0`（某个轴范围是整数字面量 `d` 的倍数）。形状违反已声明事实的启动会被拒绝，并给出指明该事实的错误 |
| `optimization_hints = expr` | expr | 把一个 `OptimizationHints` 表达式传给编译器，用于目标相关的优化 |
| `dump_mlir_dir = "path"` | string | 把编译后的 Tile IR 文本写入指定目录下的文件 |

`unchecked_accesses` 会禁用校核并要求 `unsafe`。
`deny_in_kernel_checks` 保留校核，但拒绝任何残留在 kernel 内部的检查。

```rust
#[cutile::entry(print_ir = true)]
fn debug_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }

// Unsafe：跳过所有检查，不做校核。
#[cutile::entry(unchecked_accesses)]
unsafe fn fast_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }

// 安全 + 严格：若有任何安全检查会留在 kernel 内，则编译失败。
#[cutile::entry(deny_in_kernel_checks = true)]
fn zero_check_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }

#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
fn traced_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }
```

**模块属性：**

| 属性 | 类型 | 说明 |
|---|---|---|
| `tile_rust_crate = true` | bool | 内部使用：在 cutile crate 自身内部定义 kernel 时使用（把导入路径从 `cutile::` 改成 `crate::`） |

```rust
#[cutile::module]                          // 标准用户模块
mod my_kernels { ... }

#[crate::module(tile_rust_crate = true)]   // cutile crate 内部使用
pub mod creation { ... }
```

### 入口点与设备函数

带有 `#[cutile::entry()]` 标记的函数会被编译成 GPU kernel 入口点 —— 它们可以由宿主侧带网格配置地启动。

模块内未加标记的函数是**设备函数** —— 它们在编译期被内联到调用点。它们不能被直接启动，但可以从入口点或其它设备函数中调用。

```rust
#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    // 设备函数：会被内联进调用者
    fn relu<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        select(gt_tile(x, zero), x, zero)
    }

    #[cutile::entry()]
    fn apply_relu<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile: Tile<f32, S> = input.load_tile(shape!(S), [pid]);
        output.store(relu(tile));  // 设备函数调用，会被内联
    }
}
```

### 跨模块的设备函数调用

定义在某个 `#[cutile::module]` 中的设备函数，可以从另一个模块的入口点里调用。当模块 B 写下 `use crate::activations::*` 时，会发生两件事：

1. **Rust 的类型检查器**在编译期解析这些函数签名。
2. **宏**把模块 A 记录为模块 B 的依赖，并把模块 A 的 AST 纳入为 JIT 编译所收集的模块列表。

在 JIT 阶段，编译器能看到所有被依赖模块的 AST，并把其中任意模块的设备函数内联进入口点。

每个模块由它的全限定路径（通过 `module_path!()`）标识，因此像 `cutile::core` 这样的共享依赖即使被多个模块导入，也会自动去重。

```rust
/// 模块 A：可复用的激活设备函数（没有入口点）。
#[cutile::module]
mod activations {
    use cutile::core::*;

    pub fn relu<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        max_tile(x, zero)
    }

    pub fn square<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        x * x
    }
}

/// 模块 B：调用模块 A 中设备函数的 kernel。
#[cutile::module]
mod my_kernels {
    use cutile::core::*;
    use crate::activations::{relu, square};  // 从模块 A 导入

    #[cutile::entry()]
    fn apply_relu_square<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let pid = program_id(0);
        let tile: Tile<f32, S> = input.load_tile(shape!(S), [pid]);
        let activated: Tile<f32, S> = relu(tile);   // 从模块 A 内联
        output.store(square(activated));             // 从模块 A 内联
    }
}
```

可运行的版本见 `cutile-rs-examples/examples/inter_module.rs`。

### `cutile::core` 的内部机制

`cutile::core`（定义在 `_core.rs` 中）本身就是一个包含内置 DSL 操作的 `#[cutile::module]`。当你写下 `use cutile::core::*;` 时，宏会把它记录为一个依赖并纳入其 AST。随后 JIT 编译器从 core 的 AST 中解析对 `constant`、`iota`、`mma`、`reduce_sum` 等的调用，并把它们内联进你的入口点 —— 机制与用户自定义的跨模块调用完全相同。

---

## 类型

### Tile

`Tile<E: ElementType, const S: [i32; N]>` —— GPU 取值的多维数组。这是最基本的计算类型。tile 驻留在寄存器中，由 warp/block 中的所有线程并行处理。

```rust
// 形状是 const 泛型
let a: Tile<f32, { [128] }>;          // 一维：128 个 f32 元素
let b: Tile<f32, { [16, 16] }>;       // 二维：16x16 矩阵
let c: Tile<i32, { [4, 8, 2] }>;      // 三维

// 标量 tile（秩 0）
let s: Tile<f32, { [] }>;
```

**方法：**
- `tile.shape()` —— 返回该 tile 的 `Shape<S>`
- `tile.broadcast(shape)` —— 广播到更大的形状
- `tile.reshape(shape)` —— 重塑形状（必须保持元素总数不变）
- `tile.transpose()` —— 转置一个秩为 2 的 tile

**算术运算符：** `+`、`-`、`*`、`/` 被重载为同形状、同类型 tile 之间的逐元素运算。整数 `/` 和 `%` 遵循 Rust 语义：商向零截断（`-7 / 2 == -3`），余数取被除数的符号（`-7 % 2 == -1`）；无符号元素类型按无符号除法。若需要向正无穷取整的商，请使用 `ceil_div`。

```rust
let z: Tile<f32, { [128] }> = a + b;     // 逐元素相加
let w: Tile<f32, { [128] }> = a * b - c; // 链式算术
```

### Tensor

`Tensor<E: ElementType, const S: [i32; N]>` —— 设备侧的 tensor 视图。它表示位于 GPU 全局内存中的内存。与 `Tile` 不同，tensor 必须显式地加载/存储。

静态维度在编译期已知；动态维度用 `-1` 标记：

```rust
fn kernel(
    output: &mut Tensor<f32, { [128, 128] }>,  // 静态：128x128
    input: &Tensor<f32, { [-1, -1] }>,          // 动态：形状在运行时给出
)
```

**方法：**
- `tensor.shape()` —— 返回该 tensor 的 `Shape<S>`
- `tensor.load()` —— 加载整个 tile（仅适用于 `&mut Tensor`，加载输出 tile）
- `tensor.store(tile)` —— 把一个 tile 存入 tensor
- `tensor.load_tile(shape, [indices])` —— 在指定的分区索引处加载一个 tile
- `tensor.partition(shape)` —— 创建一个用于按 block 索引加载的 `Partition` 视图
- `tensor.partition_permuted(shape, dim_map)` —— 创建一个带置换的 `Partition` 视图
- `tensor.partition_mut(shape)` —— 创建一个可变的 `PartitionMut` 视图（要求 `&mut Tensor`；这是安全的，因为通过该视图进行的每一次未证明的存储本身就是 `unsafe`）
- `unsafe tensor.partition_full_mut(shape)` —— 覆盖整个 tensor、不带 tile block 偏移地创建一个可变 `PartitionMut` 视图，用于由调度驱动的 kernel

### Partition / PartitionMut

`Partition<'a, E, const D: [i32; N]>` —— tensor 的只读分区视图，把它切成由 program ID 索引的 tile。

`PartitionMut<'a, E, const D: [i32; N]>` —— 可变的的分区视图。

```rust
let part: Partition<f32, { [128, 128] }> = input.partition(shape![128, 128]);
let tile: Tile<f32, { [128, 128] }> = part.load([pid_m, pid_n]);
```

### Shape / Array

`Shape<const D: [i32; N]>` —— 编译期形状描述符。用 `shape!` 创建：

```rust
let shape: Shape<{ [128, 64] }> = shape![128, 64];
let shape: Shape<S> = tensor.shape();
```

`Array<const D: [i32; N]>` —— 编译期数组（用于步长与索引）。

#### const 泛型数组（CGA）语法

形状可以使用字面量维度、标量 const 泛型，或者一个 const 泛型数组（`const S: [i32; N]`）：

**1. 显式取值** —— 当维度是已知字面量时：

```rust
fn kernel(
    output: &mut Tensor<f32, { [128, 64] }>,   // 固定 128x64
    input: &Tensor<f32, { [-1, -1] }>,          // 动态（运行时）形状
)
```

**2. const 泛型参数** —— 当维度在启动时指定时：

```rust
fn gemm<const BM: i32, const BN: i32, const BK: i32>(
    output: &mut Tensor<f32, { [BM, BN] }>,    // 形状来自泛型
    a: &Tensor<f32, { [-1, -1] }>,
)
```

**3. const 泛型数组（CGA）** —— 当整个形状是一个单一的泛型参数时：

```rust
fn add<const S: [i32; 1]>(                      // S 就是整个形状
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, { [-1] }>,
)
```

在 `const S: [i32; N]` 中，秩 `N` 必须是定义时就固定下来的字面量。它本身不能是泛型的，因此这种形式无法让同一个 kernel 同时适用于一维和二维 tensor：

```rust
// 不支持：泛型秩
fn add<const N: usize, const S: [i32; N]>(output: &mut Tensor<f32, S>) { ... }

// 请改为按秩分别编写 kernel：
fn add_1d<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }
fn add_2d<const S: [i32; 2]>(output: &mut Tensor<f32, S>) { ... }
```

**该用哪一种：**

| 写法 | 适用时机 |
|---|---|
| `{ [128, 64] }` | 维度是固定的字面量 |
| `{ [BM, BN] }` | 每个维度是相互独立的泛型（例如 GEMM 的 tile 尺寸） |
| `const S: [i32; 1]` | 整个形状是泛型的，但秩已知 |
| `{ [-1] }` | 维度是动态的（由宿主在运行时提供） |

动态维度（`-1`）在运行时从 tensor 的实际形状解析。静态维度被烘焙进已编译的 kernel。两者可以混用：`{ [-1, 128] }` 表示"第一轴动态、第二轴静态"。

### PointerTile

`PointerTile<P: Pointer, const D: [i32; N]>` —— 设备指针组成的 tile。用于 scatter/gather 与原子操作。

```rust
let ptr: PointerTile<*mut f32, { [] }> = pointer_to_tile(raw_ptr);
let offset_ptrs: PointerTile<*mut f32, { [128] }> = ptr.broadcast(shape![128]).offset_tile(offsets);
```

### Token

`Token` —— 内存操作的顺序令牌。令牌用于在加载与存储之间强制顺序，而不需要完整的屏障。

```rust
let token: Token = new_token_unordered();
let joined: Token = join_tokens(&[token_a, token_b]);
```

### ElementType

所有可用于 tile 的标量类型都实现了该 trait：

| 类型 | 说明 |
|------|-------------|
| `f16` | IEEE 754 半精度 |
| `bf16` | Brain 浮点格式 |
| `f32` | 单精度 |
| `f64` | 双精度 |
| `i8`、`i16`、`i32`、`i64` | 有符号整数 |
| `u8`、`u16`、`u32`、`u64` | 无符号整数 |
| `bool` | 布尔 |
| `tf32` | TensorFloat-32（NVIDIA） |
| `f8e4m3fn`、`f8e5m2` | FP8（仅用于存储；做计算时请转换成 `f16`/`f32`） |
| `f8e8m0fnu` | 用于块缩放低精度 MMA 的 FP8 缩放格式 |
| `f4e2m1fn` | 逻辑 FP4 E2M1 tile 元素；在 MMA 之前需从打包存储解包 |
| `f4e2m1fnx2` | 两个 `f4e2m1fn` 取值的可按字节寻址打包存储；在 MMA 之前需解包 |
| `i4` | Tile IR 的 4 位无符号整数类型，用于量化以及 pack/unpack 路径 |

`f4e2m1fn` 与 `i4` 是亚字节的 Tile IR 元素类型。在宿主与 kernel 边界上，打包 FP4 tensor 存储请使用 `f4e2m1fnx2` 或 `u8`，然后在块缩放 MMA 之前解包成逻辑 `f4e2m1fn` tile。`i4` 不是 NVFP4 类型，目前也不是 cuTile Rust 中受支持的高层整数 MMA 操作数。

---

## 操作

### 与 Tile IR 操作的对应关系

本参考中的函数是 [CUDA Tile IR 操作](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html) 的 Rust DSL 表面。`cutile::core` 中的大多数底层操作都直接对应一个 `cuda_tile.*` 操作。Rust 名称通常要么是去掉 `cuda_tile.` 前缀的 Tile IR 操作名，要么是一个带有面向 Rust 的名称与类型签名的小包装。

示例：

| Rust DSL 表面 | Tile IR 操作族 |
|---|---|
| `constant`、`iota`、`broadcast`、`reshape`、`transpose`、`permute`、`cat`、`extract`、`select` | 核心 tile 操作，如 `cuda_tile.constant`、`cuda_tile.iota`、`cuda_tile.broadcast`、`cuda_tile.reshape`、`cuda_tile.permute`、`cuda_tile.cat`、`cuda_tile.extract`、`cuda_tile.select` |
| `addf`、`mulf`、`fma`、`exp`、`sqrt`、`maxf`、`cmpf` 以及运算符重载 | 浮点操作，如 `cuda_tile.addf`、`cuda_tile.mulf`、`cuda_tile.fma`、`cuda_tile.exp`、`cuda_tile.sqrt`、`cuda_tile.maxf`、`cuda_tile.cmpf` |
| `addi`、`muli`、`shli`、`shri`、`andi`、`ori`、`xori`、`cmpi` 以及运算符重载 | 整数与位操作，如 `cuda_tile.addi`、`cuda_tile.muli`、`cuda_tile.shli`、`cuda_tile.andi`、`cuda_tile.cmpi` |
| `convert_tile`、`bitcast`、`exti`、`trunci`、`int_to_ptr`、`ptr_to_int`、`ptr_to_ptr` | 转换操作，如 `cuda_tile.bitcast`、`cuda_tile.exti`、`cuda_tile.trunci`、`cuda_tile.int_to_ptr`、`cuda_tile.ptr_to_int`、`cuda_tile.ptr_to_ptr` |
| `pack`、`unpack` | 秩为 1 的字节打包与解包操作，如 `cuda_tile.pack` 与 `cuda_tile.unpack` |
| `Tile<f4e2m1fn, ...>::pack`、`Tile<f4e2m1fnx2, ...>::unpack` | Rust 辅助方法：把带形状的 FP4 tile 展平、调用秩为 1 的 `cuda_tile.pack`/`cuda_tile.unpack`，再把结果 reshape 回来 |
| `load_ptr_tko`、`store_ptr_tko`、`atomic_rmw_tko`、`atomic_cas_tko`、`new_token_unordered`、`join_tokens` | 内存、原子与令牌操作，如 `cuda_tile.load_ptr_tko`、`cuda_tile.store_ptr_tko`、`cuda_tile.atomic_rmw_tko`、`cuda_tile.atomic_cas_tko`、`cuda_tile.make_token`、`cuda_tile.join_tokens` |
| `load_like`/`load_tile_like`、`tensor.load_tile`、`tensor.store`、分区与 tensor 视图辅助方法 | 视图操作，如 `cuda_tile.load_view_tko`、`cuda_tile.store_view_tko`、`cuda_tile.make_tensor_view`、`cuda_tile.make_partition_view`，以及编译器生成的形状/步长管线 |
| `assume_div_by`、`assume_bounds_*`、`cuda_tile_print!`、`cuda_tile_assert!` | 编译器与调试操作，如 `cuda_tile.assume`、`cuda_tile.print_tko`、`cuda_tile.assert` |

有些 Tile IR 操作是有意归编译器所有的，而不是公开的 DSL 函数。`cuda_tile.module`、`cuda_tile.entry`、`cuda_tile.return` 以及控制流操作，都是由 Rust 模块、入口属性、`return`、`if`、`for`、`loop`、`while`、`break` 和 `continue` 语法生成的。Tile IR 13.4 还允许在 `loop`/`while` 内部使用 kernel 的 `return`，包括该循环内部的条件分支。这会**退出 kernel**，而不只是退出循环。它不支持在内联的辅助函数内部使用，也不支持位于任何 `for` 循环之下。其它形式的嵌套 return 仍然是编译错误；请改用条件把剩余语句保护起来。`for` 内部的 `break` 同样是编译错误（`cuda_tile.for` 无法提前退出；请改用 `while`/`loop`）。`continue` 在每一种循环形式下都受支持。

Tile IR 的属性，例如内存顺序、内存作用域、比较谓词、舍入模式、溢出行为以及 flush-to-zero 模式，都尽可能用 Rust 的标记类型与 trait 来表示。这让 Rust 类型检查器能够在 JIT 编译器下降该操作之前，就拒绝不受支持的属性组合。

### 裸 Tile IR 的版本化表面

裸内建函数位于 `cutile::core`；视图构造函数及其类型也位于 `cutile::tileir`。它们面向的是那些能够维持文档所述 unsafe 不变式的调用者。既有的便捷 API 保持不变。JIT 会同时检查选定汇编器的字节码版本与实际的目标架构。即使某个操作本身出现得更早，其类型、属性与操作数配置要求依然适用。

| API / 选项 | 要求 | 含义 |
|---|---|---|
| `unsafe insert(source, destination, indices)` | 13.4 | 替换一个子 tile；每个轴一个标量 i32 索引，单位为源 tile |
| `unsafe fpowi(base, exponent)` | 13.4 | 形状相同的有符号整数指数量 tile；指数最多 32 位 |
| `unsafe fpowf(base, exponent)` | 13.2 | 既有浮点 `pow` 操作的别名 |
| `unsafe gdc_launch_dependents_tko(Option<Token>)` | 13.4 | 发出信号表示依赖该 kernel 的网格可以启动；其本身不发布内存 |
| `unsafe gdc_wait_tko(Option<Token>)` | 13.4 | 等待依赖完成；把后续的依赖内存操作串接到它返回的令牌上 |
| `unsafe memory_fence_alias_tko(token)` | 13.4 | 在改变访问路径时建立所需的 Tile IR 别名栅栏 |
| `unsafe alloca(num_elem, alignment, allocation::Local/Global)` | 13.3 | 逐 tile block 的分配；元素个数需为编译期非负值，对齐需为 2 的幂 |
| `unsafe ftoi_saturating(tile, saturating::Enabled/Disabled)` | 启用时为 13.4 | 浮点转整数，采用 `NearestIntToZero`；饱和是 opt-in 的 |
| `unsafe ptr_with_attr_none(pointer_tile)` | 13.4 | 显式的 `ptr_attr<none>`，与"未作分类"不同；派生视图会继承它 |
| `unsafe mmaf_with_fast_acc(a, b, accumulator, fast_acc)` | 启用时为 13.3 | 显式的 `fast_acc::Enabled/Disabled`；常规的 MMA 类型/形状约束依然适用 |
| `unsafe exp_with_rounding(tile, rounding)` | Approx 需 13.3 | `rounding::Full` 或 f32 的 `rounding::Approx` |
| `unsafe tanh_with_rounding(tile, rounding)` | 13.2 | `rounding::Full` 或 f32 的 `rounding::Approx` |
| `rounding::NearestIntToZero` | 13.2 | 浮点转整数的舍入模式 |
| `rounding::NearestAway` | 13.4 | 浮点转换模式，仅适用于受支持的源/目标组合 |
| `f8e5m3fnu` | 13.4 且 sm_107+ | 无符号缩放格式；使用该缩放类型的缩放式 MMA 需要 sm_107 |

`load_view_raw`、`store_view_raw` 与 `atomic_red_view_raw` 接受分区视图、gather/scatter 视图或 strided 视图。后两种视图类型要求 13.3；视图上的原子归约同样要求 13.3，并要求分区或 strided 视图不带 padding。索引是按 tensor 轴顺序排列的数组或元组；gather/scatter 视图的稀疏轴接受一个秩为 1 的整数 tile，其余轴接受标量 i32 索引。在元素类型允许的前提下，padding 支持 `None`、`Zero`、`Nan`、`PosInf` 与 `NegInf`。

```rust
let (value, done): (Tile<f32, {[4]}>, Token) = unsafe {
    load_view_raw(&view, [0i32], None, ordering::Weak, scope::Device,
        tma::Disabled, None, [false], shape![4])
};
let stored = unsafe {
    store_view_raw(&view, value, [0i32], Some(done), ordering::Weak,
        scope::Device, tma::Disabled, None, [false])
};
```

这些调用暴露了可选的输入令牌、顺序/作用域，以及对加载与存储而言的 TMA 与延迟提示。每个 `inbounds` 布尔值对应一个索引；任意一个为 `true` 都要求 13.4，并承诺该轴在界内。为 `false` 并不会让不带 padding 的裸访问变得安全。调用者必须确保存储有效且存活、对齐正确、访问无竞争，并且满足所有必需的令牌依赖。裸存储不会通过它们的 `&view` 参数建立独占所有权。

`assume_div_by_raw(value, divisor: u64, every: Option<i64>, along: Option<i64>)`、`assume_bounds_raw(value, lower: Option<i64>, upper: Option<i64>)` 与 `assume_same_elements_raw(value, groups: [i64; N])` 在 13.2+ 上暴露了全宽度、任意秩的谓词。它们是 unsafe 的承诺，而非运行时检查。除数必须是 2 的幂且不大于 2^62；`every` 与 `along` 必须同时提供。边界必须能放进有符号操作数的位宽。

在 13.3+ 上，`#[cutile::module(producer = "my compiler")]` 会写入 producer 元数据，而 `#[cuda_tile::global(constant = true, visibility = "private")]` 会把一个已有的 `Global` 静态量标记为常量/私有。默认仍是可变且公开的；常量 global 无法通过 `Global` API 写入。

### 内存：加载与存储

| 函数 | 签名 | 说明 |
|---|---|---|
| `tensor.load()` | `&mut Tensor<E, S> -> Tile<E, S>` | 加载输出 tile |
| `tensor.store(tile)` | `(&mut Tensor<E, S>, Tile<E, S>)` | 把一个 tile 存入 tensor |
| `tensor.load_tile(shape, idx)` | `(&Tensor<E, S>, Shape<R>, [i32; N]) -> Tile<E, R>` | 在某个分区索引处加载 |
| `load_tile_mut(tensor)` | `(&mut Tensor<E, S>) -> Tile<E, S>` | 加载输出 tile（便捷函数） |
| `src.load_like(dst)` | `(&Tensor, &Tensor) -> Tile` | 在 dst 的 tile block 位置从 src 加载（秩 1-3）；也有自由函数 `load_tile_like(src, dst)`。src 必须与 dst 背后那个完整 tensor 的形状一致（而不是 dst 每个程序拿到的子块），并且 dst 必须是已分区的可变输出 |

```rust
// 模式 1：在可变 tensor 上直接加载/存储
let tile: Tile<f32, { [128] }> = load_tile_mut(output);
output.store(tile * scale_tile);

// 模式 2：在程序位置上加载
let pid = program_id(0);
let tile: Tile<f32, { [128] }> = input.load_tile(shape![128], [pid]);

// 模式 3：类位置加载（positional）
let tile_x: Tile<f32, { [16, 16] }> = x.load_like(output);
```

### 网格与程序 ID

| 函数 | 签名 | 说明 |
|---|---|---|
| `program_id(axis)` | `(i32) -> i32` | 本 tile 程序在网格轴 `axis` 上的索引（0、1 或 2） |
| `num_programs(axis)` | `(i32) -> i32` | 启动网格在轴 `axis` 上的范围（0、1 或 2） |
| `get_tile_block_id()` | `() -> (i32, i32, i32)` | 三个 (x, y, z) 程序索引组成的元组 |
| `get_num_tile_blocks()` | `() -> (i32, i32, i32)` | 三个 (x, y, z) 网格范围组成的元组 |

对每个轴 `k` 都有 `0 <= program_id(k) < num_programs(k)`。

```rust
let pid = program_id(0);
let n = num_programs(0);
// 网格跨步循环
for i in (pid..total).step_by(n as usize) { ... }
```

### 算术（逐元素）

除了运算符重载（`+`、`-`、`*`、`/`、`%`；对有符号与无符号元素类型都一致：整数 `/` 向零截断，`%` 取被除数的符号，与 Rust 完全相同）之外，还可以使用这些显式函数：

| 函数 | 签名 | 说明 |
|---|---|---|
| `absi(x)` | `Tile<E, S> -> Tile<E, S>` | 绝对值（整数） |
| `absf(x)` | `Tile<E, S> -> Tile<E, S>` | 绝对值（浮点） |
| `negi(x, overflow)` | `(Tile<E, S>, overflow::Mode) -> Tile<E, S>` | 取负（整数）；`overflow::{None, NoSignedWrap, NoUnsignedWrap, NoWrap}` |
| `negf(x)` | `Tile<E, S> -> Tile<E, S>` | 取负（浮点） |
| `fma(lhs, rhs, acc, rounding, ftz)` | `(Tile<E, S>, Tile<E, S>, Tile<E, S>, rounding::Mode, ftz::Mode) -> Tile<E, S>` | 融合乘加：`lhs * rhs + acc` |
| `pow(base, exp)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 幂 |
| `ceil_div(a, b)` | `(E, E) -> E` | 向上取整除法（标量） |
| `true_div(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 真（浮点）除法 |
| `mulhii(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 整数乘法的高位 |

```rust
// 绝对值
let abs_x: Tile<f32, S> = absf(x);
let abs_i: Tile<i32, S> = absi(int_tile);

// 融合乘加：a * b + c（单条指令，中间不做舍入）
let result: Tile<f32, S> = fma(a, b, c, rounding::NearestEven, ftz::Disabled);

// 幂
let squared: Tile<f32, S> = pow(x, broadcast_scalar(2.0f32, x.shape()));

// 取负
let neg_x: Tile<f32, S> = negf(x);
let neg_i: Tile<i32, S> = negi(int_tile, overflow::None);
```

### 数学（浮点）

| 函数 | 签名 | 说明 |
|---|---|---|
| `exp(x)` | `Tile<E, S> -> Tile<E, S>` | e^x |
| `exp2(x, ftz)` | `(Tile<E, S>, ftz::Mode) -> Tile<E, S>` | 2^x |
| `log(x)` | `Tile<E, S> -> Tile<E, S>` | 自然对数 |
| `log2(x)` | `Tile<E, S> -> Tile<E, S>` | 以 2 为底的对数 |
| `sqrt(x, rounding, ftz)` | `(Tile<E, S>, rounding::Mode, ftz::Mode) -> Tile<E, S>` | 平方根 |
| `rsqrt(x, ftz)` | `(Tile<E, S>, ftz::Mode) -> Tile<E, S>` | 平方根倒数（1/sqrt(x)） |
| `sin(x)` | `Tile<E, S> -> Tile<E, S>` | 正弦 |
| `cos(x)` | `Tile<E, S> -> Tile<E, S>` | 余弦 |
| `tan(x)` | `Tile<E, S> -> Tile<E, S>` | 正切 |
| `sinh(x)` | `Tile<E, S> -> Tile<E, S>` | 双曲正弦 |
| `cosh(x)` | `Tile<E, S> -> Tile<E, S>` | 双曲余弦 |
| `tanh(x)` | `Tile<E, S> -> Tile<E, S>` | 双曲正切 |
| `ceil(x)` | `Tile<E, S> -> Tile<E, S>` | 向上取整 |
| `floor(x)` | `Tile<E, S> -> Tile<E, S>` | 向下取整 |
| `maxf(lhs, rhs, nan, ftz)` | `(Tile<E, S>, Tile<E, S>, nan::Mode, ftz::Mode) -> Tile<E, S>` | 浮点最大值；`nan::Enabled` 会传播 NaN |
| `minf(lhs, rhs, nan, ftz)` | `(Tile<E, S>, Tile<E, S>, nan::Mode, ftz::Mode) -> Tile<E, S>` | 浮点最小值；`nan::Enabled` 会传播 NaN |
| `addf(lhs, rhs, rounding, ftz)`、`subf(…)`、`mulf(…)`、`divf(…)` | `(Tile<E, S>, Tile<E, S>, rounding::Mode, ftz::Mode) -> Tile<E, S>` | 显式的浮点加/减/乘/除，可控制舍入与 flush-to-zero（`+ - * /` 运算符使用默认值） |

没有单独的 `*_ftz` 函数：flush-to-zero 是通过给基础操作传 `ftz::Enabled` 来选择的。

```rust
// softmax 的数值处理：减去最大值，再取指数
let max_val: Tile<f32, { [BM] }> = reduce_max(x, 1i32);
let shifted: Tile<f32, S> = x - max_val.reshape(shape![BM, 1]).broadcast(x.shape());
let softmax_exp: Tile<f32, S> = exp(shifted);

// RMS 归一化
let sq: Tile<f32, S> = x * x;
let mean_sq: Tile<f32, { [BM] }> = reduce_sum(sq, 1i32);
let rms: Tile<f32, { [BM] }> =
    rsqrt(mean_sq + broadcast_scalar(1e-6f32, mean_sq.shape()), ftz::Disabled);

// 激活函数
let gelu_approx: Tile<f32, S> = x * (constant(1.0f32, x.shape()) + tanh(x));
let swish: Tile<f32, S> = x / (constant(1.0f32, x.shape()) + exp(negf(x)));

// exp2 在 GPU 上比 exp 快 —— 转换方式：exp(x) = exp2(x * log2(e))
let log2_e: f32 = 1.4426950408889634f32;
let fast_exp: Tile<f32, S> = exp2(x * broadcast_scalar(log2_e, x.shape()), ftz::Disabled);

// Flush-to-zero：传 ftz::Enabled 把非规格化数当作零（在某些硬件上更快，仅限 f32）
let clamped: Tile<f32, S> =
    maxf(x, broadcast_scalar(0.0f32, x.shape()), nan::Disabled, ftz::Enabled);
let sum: Tile<f32, S> = addf(a, b, rounding::NearestEven, ftz::Enabled);
let product: Tile<f32, S> = mulf(a, b, rounding::NearestEven, ftz::Enabled);
let fma_result: Tile<f32, S> = fma(a, b, c, rounding::NearestEven, ftz::Enabled);
```

### 比较

| 函数 | 签名 | 说明 |
|---|---|---|
| `eq_tile(a, b)` | `-> Tile<bool, S>` | 相等 |
| `ne_tile(a, b)` | `-> Tile<bool, S>` | 不相等 |
| `gt_tile(a, b)` | `-> Tile<bool, S>` | 大于 |
| `ge_tile(a, b)` | `-> Tile<bool, S>` | 大于等于 |
| `lt_tile(a, b)` | `-> Tile<bool, S>` | 小于 |
| `le_tile(a, b)` | `-> Tile<bool, S>` | 小于等于 |
| `select(cond, a, b)` | `(Tile<bool, S>, Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 条件选择 |
| `min(a, b)` / `max(a, b)` | `(E, E) -> E` | 标量最小/最大值 |
| `min_tile(a, b)` / `max_tile(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 逐元素最小/最大值 |

```rust
let mask: Tile<bool, { [128] }> = lt_tile(indices, len_tile);
let result: Tile<f32, { [128] }> = select(mask, values, zeros);
```

### 创建

| 函数 | 签名 | 说明 |
|---|---|---|
| `constant(value, shape)` | `(E, Shape<S>) -> Tile<E, S>` | 用常量填充一个 tile |
| `iota(shape)` | `(Shape<S>) -> Tile<E, S>` | 顺序索引 `[0, 1, 2, ...]`（仅一维） |
| `broadcast_scalar(val, shape)` | `(E, Shape<S>) -> Tile<E, S>` | 把一个标量广播成 tile 形状 |

```rust
let zeros: Tile<f32, { [128] }> = constant(0.0f32, shape![128]);
let indices: Tile<i32, { [64] }> = iota(shape![64]);  // [0, 1, 2, ..., 63]
let scale: Tile<f32, { [16, 16] }> = broadcast_scalar(2.0f32, shape![16, 16]);
```

### 形状操作

| 函数 | 签名 | 说明 |
|---|---|---|
| `tile.reshape(shape)` | `Tile<E, S> -> Tile<E, R>` | 重塑形状（保持元素总数） |
| `tile.broadcast(shape)` | `Tile<E, S> -> Tile<E, R>` | 广播到更大的形状 |
| `tile.transpose()` | `Tile<E, [M, N]> -> Tile<E, [N, M]>` | 秩为 2 的转置 |
| `reshape(tile, shape)` | `(Tile<E, S>, Shape<R>) -> Tile<E, R>` | 自由函数形式的 reshape |
| `broadcast(tile, shape)` | `(Tile<E, S>, Shape<R>) -> Tile<E, R>` | 自由函数形式的 broadcast |
| `permute(tile, permutation)` | `(Tile<E, A>, Array<I>) -> Tile<E, R>` | 转置/置换维度（例如 `const_array![1, 0]`）；结果形状 `R` 在调用处注明 |
| `cat(a, b, dim)` | `(Tile<E, SLhs>, Tile<E, SRhs>, i32) -> Tile<E, SOut>` | 沿某个维度拼接 |
| `extract(tile, indices)` | `(Tile<E, SIn>, [Tile<i32, {[]}>; N]) -> Tile<E, SOut>` | 在秩为 0 的索引 tile 处提取子 tile；结果形状 `SOut` 在调用处注明 |
| `shape[index]` | `(Shape<S>, usize) -> i32` | 从形状中读取某一个运行时维度 |

```rust
let row: Tile<f32, { [128] }> = iota(shape![128]);
let col: Tile<f32, { [128, 1] }> = row.reshape(shape![128, 1]);
let matrix: Tile<f32, { [128, 64] }> = col.broadcast(shape![128, 64]);
let n_cols: i32 = matrix.shape()[1];
```

<a id="reduction-and-scan"></a>
### 归约与扫描

| 函数 | 签名 | 说明 |
|---|---|---|
| `reduce_sum(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | 沿一个维度求和归约 |
| `reduce_max(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | 最大值归约 |
| `reduce_min(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | 最小值归约 |
| `reduce_prod(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | 乘积归约 |
| `reduce(tile, dim, identity, f)` | `(Tile<E, S>, i32, E, Fn(E, E) -> E) -> Tile<E, S>` | 自定义归约；其类型标注使用输入形状（只有 `reduce_min/max/sum/prod` 会在返回类型中折叠轴） |
| `scan_sum(tile, dim, reverse, identity)` | `(Tile<E, S>, i32, reverse::Mode, E) -> Tile<E, S>` | 前缀和 |
| `scan(tile, dim, reverse, identity, f)` | `(Tile<E, S>, i32, reverse::Mode, E, Fn(E, E) -> E) -> Tile<E, S>` | 自定义前缀扫描 |

```rust
// 把 [128, 64] tile 的每一行求和成 [128]（沿轴 1 归约）
let row_sums: Tile<f32, { [128] }> = reduce_sum(matrix, 1i32);

// 沿轴 0 的前缀和
let prefix: Tile<f32, { [128] }> = scan_sum(row, 0i32, reverse::Forward, 0.0f32);
```

<a id="matrix-multiply"></a>
### 矩阵乘法

| 函数 | 签名 | 说明 |
|---|---|---|
| `mma(a, b, c)` | `(Tile<E1, {[M,K]}>, Tile<E1, {[K,N]}>, Tile<E2, {[M,N]}>) -> Tile<E2, {[M,N]}>` | 矩阵乘累加；累加器的元素类型可以与输入不同（例如 `f16` 输入、`f32` 累加器） |
| `mmaf_scaled(a, b, c, a_scale, b_scale)` | `(Tile<E, A>, Tile<E, B>, Tile<f32, C>, Tile<S, AS>, Tile<S, BS>) -> Tile<f32, C>` | 块缩放浮点矩阵乘累加 |

在硬件支持时映射到 Tensor Core。

```rust
let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
for k in 0i32..(K/BK) {
    let a_tile: Tile<f32, { [16, 8] }> = a_part.load([pid_m, k]);
    let b_tile: Tile<f32, { [8, 16] }> = b_part.load([k, pid_n]);
    acc = mma(a_tile, b_tile, acc);
}
```

`mmaf_scaled` 用于 FP4/FP8 的块缩放矩阵乘法。打包的 FP4 tensor 应表示为 `Tensor<f4e2m1fnx2, ...>`，用 `tile.unpack(shape![...])` 解包成逻辑 `Tile<f4e2m1fn, ...>` 操作数，再连同一并的 FP8 缩放 tile 传给 `mmaf_scaled`。裸的 `pack` 与 `unpack` 操作作用于秩为 1 的 tile；FP4 的 tile 方法会发出所需的展平/打包或解包/重塑序列。完整模式见[使用 NVFP4/MXFP8 做推理](../tutorials/11-nvfp4-inference.md)。

<a id="low-level-memory-ops"></a>
### 底层访存操作

这些 API 贴近 Tile IR 的内存/视图操作。除非你是在构建自定义视图、裸指针 kernel，或者面向编译器的辅助工具，否则请优先使用上面的高层方法（`tensor.load_tile`、`partition.load`、`partition_mut.store`、`x.load_like(z)`、`tensor.store`）。

#### 视图构造与查询

视图构造函数从更底层的元数据创建带类型的 tensor 或分区视图。裸视图构造函数是 `unsafe` 的：调用者必须维持顺序、别名、布局与生命周期方面的不变式。特别地，一个只读视图不能丢弃 tensor 的令牌而与更早的存储产生竞争。

| 函数 | 签名 | 说明 |
|---|---|---|
| `unsafe make_tensor_view(base, shape, strides, token)` | `(PointerTile<*mut E, {[]}>, Shape<D>, Array<C>, Token) -> Tensor<E, D>` | 从基指针构建一个 tensor 视图 |
| `unsafe make_partition_view(tensor, tile, padding, dim_map, token)` | `(&Tensor<E, S>, Shape<R>, padding::Mode, dim_map::Mode, Token) -> Partition<E, R>` | 构建一个只读分区视图 |
| `unsafe make_partition_view_mut(tensor, tile, padding, token)` | `(&Tensor<E, S>, Shape<R>, padding::Mode, Token) -> PartitionMut<E, R>` | 构建一个可变分区视图 |
| `get_tensor_shape(tensor)` | `&Tensor<E, S> -> [i32; N]` | 查询 tensor 视图的运行时形状 |
| `get_index_space_shape(partition)` | `&Partition<E, S> -> [i32; N]` | 查询分区的 tile 网格形状 |
| `get_tensor_token(tensor)` | `&Tensor<E, S> -> Token` | 读取 tensor 视图的内存令牌 |
| `unsafe set_tensor_token(tensor, token)` | `(&Tensor<E, S>, Token)` | 在控制流区域之外，为某个函数级 tensor 绑定安装一个顺序依赖 |
| `get_partition_token(partition)` | `&Partition<E, S> -> Token` | 读取只读分区的令牌 |
| `get_partition_token_mut(partition)` | `&PartitionMut<E, S> -> Token` | 读取可变分区的令牌 |
| `num_tiles(partition, axis)` | `(&Partition<E, S>, i32) -> i32` | 某个分区轴上 tile 的数量 |
| `unsafe load_tensor(ptrs, idx, shape, strides)` | `(&Tensor<i64, S>, [i32; N], Shape<R>, Array<C>) -> Tensor<T, R>` | 从整数指针 tensor 加载一个带步长的 tensor 视图 |

```rust
let shape = input.shape();
let token = get_tensor_token(input);
// 安全性：保留 tensor 自身的顺序令牌，并确保该视图在其借用期内使用。
let part = unsafe {
    make_partition_view(input, shape, padding::None, dim_map::Identity, token)
};
let tiles_m: i32 = num_tiles(&part, 0);
```

#### 视图的加载与存储

这些是作用于分区视图的直接访存操作。它们显式地暴露了 Tile IR 的顺序、作用域、延迟与 TMA 控制。

| 函数 | 签名 | 说明 |
|---|---|---|
| `load_view_tko(view, index, ordering, scope, latency, tma)` | `(&Partition<E, S>, [i32; N], O, Sc, Option<i32>, T) -> Tile<E, S>` | 从只读分区加载一个 tile |
| `unsafe load_view_tko_mut(view, index, ordering, scope, latency, tma)` | `(&PartitionMut<E, S>, [i32; N], O, Sc, Option<i32>, T) -> Tile<E, S>` | 从可变分区加载；带 unsafe 的别名契约 |
| `unsafe store_view_tko_mut(view, tile, index, ordering, scope, latency, tma)` | `(&mut PartitionMut<E, S>, Tile<E, S>, [i32; N], O, Sc, Option<i32>, T) -> Token` | 把一个 tile 存入可变分区 |

```rust
let pid = program_id(0);
let tile: Tile<f32, S> =
    load_view_tko(&part, [pid], ordering::Weak, scope::TileBlock, None, tma::Enabled);
```

#### 基于指针的加载与存储

| 函数 | 签名 | 说明 |
|---|---|---|
| `unsafe load_ptr_tko(ptrs, ordering, Option<scope>, mask, fill, token, Latency<N>)` | `-> (Tile<E, S>, Token)` | 通过指针的 scatter-gather 加载 |
| `unsafe store_ptr_tko(ptrs, values, ordering, Option<scope>, mask, token, Latency<N>)` | `-> Token` | 通过指针的 scatter-gather 存储 |
| `pointer_to_tile(ptr)` | `P -> PointerTile<P, {[]}>` | 把裸指针转换成标量指针 tile |
| `tile_to_pointer(ptile)` | `PointerTile<P, {[]}> -> P` | 转换回裸指针 |
| `addptr(ptile, offset)` | `(PointerTile<P, D>, i32) -> PointerTile<P, D>` | 用标量偏移一个指针 tile |
| `addptr_tile(ptile, offsets)` | `(PointerTile<P, D>, Tile<I, D>) -> PointerTile<P, D>` | 用索引 tile 偏移一个指针 tile |
| `broadcast_ptr(ptile, shape)` | `(PointerTile<P, S>, Shape<R>) -> PointerTile<P, R>` | 把指针 tile 广播到更大的形状 |
| `reshape_ptr(ptile, shape)` | `(PointerTile<P, S>, Shape<R>) -> PointerTile<P, R>` | 重塑指针 tile 的形状 |

`load_ptr_tko` 与 `store_ptr_tko` 会解引用裸设备指针，且都是 `unsafe fn`：请在 `unsafe` 块中调用它们。下面的片段为简洁起见省略了该块。

```rust
let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(ptr);
let ptrs: PointerTile<*mut f32, { [128] }> = base.broadcast(shape![128]).offset_tile(offsets);
let (values, token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);
```

#### 原子操作

模块作用域的原子存储使用 `Global<A, { [] }>`，其中 `A` 实现了密封的 `Atomic` trait：

```rust
static COUNTER: Global<AtomicI32, { [] }> = Global::new(0i32);
// 在 kernel 内部：
let one: Tile<i32, { [] }> = constant(1i32, shape![]);
let (previous, token) = COUNTER.atomic_add(one, ordering::Relaxed, scope::Device);
```

受支持的标记有 `AtomicI32`、`AtomicU32`、`AtomicI64`、`AtomicU64`、`AtomicF32` 与 `AtomicF64`。它们的 `Atomic::Value` 类型就是对应的标量；这些是设备 DSL 的标记，不是宿主侧的原子对象。目前只支持标量 global。

global 的 `load` 接受 `Relaxed`/`Acquire`；`store` 接受 `Relaxed`/`Release`；`atomic_add` 接受 `Relaxed`/`Acquire`/`Release`/`AcqRel`。它们都要求 `Device` 或 `System` 作用域。Rust 与 JIT 都会拒绝弱序或 tile block 作用域的 global 访问。先 load 再 store **不是**原子自增：请使用 `atomic_add`。

顺序契约仍然是 Tile IR 的显式令牌模型，而不是 `std::sync::atomic` 对周边内存的语义。global 操作会返回完成令牌，但不接受输入令牌。它们不会自动为其它 tensor 或裸指针访问建立顺序：不要单独用它们做发布协议，那还需要显式的令牌依赖。

下面的裸内建函数仍然是不安全的，并保留了 Tile IR 更宽泛的顺序/作用域选择。

| 函数 | 签名 | 说明 |
|---|---|---|
| `unsafe atomic_rmw_tko(ptrs, arg, mode, ordering, scope, mask, token)` | `(PointerTile<*mut E, S>, Tile<E, S>, atomic::Mode, ordering::AtomicMode, scope::Mode, Option<Tile<bool, S>>, Option<Token>) -> (Tile<E, S>, Token)` | 原子读-改-写 |
| `unsafe atomic_cas_tko(ptrs, cmp, val, ordering, scope, mask, token)` | `(PointerTile<*mut E, S>, Tile<E, S>, Tile<E, S>, ordering::AtomicMode, scope::Mode, Option<Tile<bool, S>>, Option<Token>) -> (Tile<E, S>, Token)` | 原子比较并交换 |

**RMW 模式：** `atomic::{Add, AddF, And, Or, Xor, Max, Min, Umax, Umin, Xchg}`

**内存顺序：** `ordering::{Relaxed, Acquire, Release, AcqRel}`（原子操作用；load/store 还接受 `Weak`）

**作用域：** `scope::{TileBlock, Device, System}`

两个原子操作都会解引用裸设备指针，且都是 `unsafe fn`，与 `load_ptr_tko`/`store_ptr_tko` 一样：

```rust
unsafe {
    atomic_rmw_tko(ptrs, increments, atomic::Add, ordering::Relaxed, scope::Device, None, None);
    atomic_cas_tko(ptrs, expected, desired, ordering::AcqRel, scope::System, None, None);
}
```

#### 令牌

令牌用于跟踪内存操作之间的顺序依赖。一次加载返回的令牌保证：任何消费该令牌的操作都在这此加载完成之后才执行。`join_tokens` 把多个令牌合并成一个，确保所有被合并的操作都在结果令牌被使用之前完成。

这样就能在不使用完整屏障的前提下实现细粒度排序：相互独立的加载可以并行执行，而一次存储只需等待它所依赖的那些具体加载。

| 函数 | 签名 | 说明 |
|---|---|---|
| `new_token_unordered()` | `() -> Token` | 创建一个全新的顺序令牌（不提供任何顺序保证） |
| `join_tokens(tokens)` | `(&[Token]) -> Token` | 合并多个令牌：结果会等待所有输入 |

```rust
// 把令牌贯穿"加载 → 计算 → 存储"这条链：
let token: Token = new_token_unordered();

// 加载返回一个新令牌，保证加载已完成
let (data, load_token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(src_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);

// 在加载到的数据上计算
let result: Tile<f32, { [128] }> = data * data;

// 存储使用该加载令牌：写入之前等待加载完成
let store_token: Token =
    store_ptr_tko(dst_ptrs, result, ordering::Weak, None::<scope::TileBlock>, None, Some(load_token), Latency::<0>);
```

```rust
// 在依赖性的存储之前，合并两次相互独立的加载的令牌：
let (a_data, a_token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(a_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);
let (b_data, b_token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(b_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);

// 两次加载都必须完成，然后才能存储
let combined: Token = join_tokens(&[a_token, b_token]);
let result: Tile<f32, { [128] }> = a_data + b_data;
let _: Token =
    store_ptr_tko(out_ptrs, result, ordering::Weak, None::<scope::TileBlock>, None, Some(combined), Latency::<0>);
```

### 位运算

| 函数 | 签名 | 说明 |
|---|---|---|
| `andi(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 按位与 |
| `ori(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 按位或 |
| `xori(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 按位异或 |
| `shli(lhs, rhs, overflow)` | `(Tile<E, S>, Tile<E, S>, overflow::Mode) -> Tile<E, S>` | 左移 |
| `shri(lhs, rhs)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | 右移（有符号 `E` 为算术右移，无符号为逻辑右移） |

```rust
// 屏蔽低 8 位
let mask: Tile<i32, S> = constant(0xFF, x.shape());
let low_byte: Tile<i32, S> = andi(x, mask);

// 左移 2 位（相当于乘 4）
let shift: Tile<i32, S> = constant(2, x.shape());
let shifted: Tile<i32, S> = shli(x, shift, overflow::None);

// 用异或翻转位
let toggled: Tile<i32, S> = xori(x, mask);
```

### 类型转换

| 函数 | 签名 | 说明 |
|---|---|---|
| `convert_tile(tile)` | `Tile<FROM, S> -> Tile<TO, S>` | 转换元素类型 |
| `convert_scalar(x)` | `impl Scalar -> S` | 转换标量类型 |
| `scalar_to_tile(x)` | `impl Scalar -> Tile<E, {[]}>` | 把标量包装成秩为 0 的 tile |
| `tile_to_scalar(tile)` | `Tile<E, {[]}> -> S` | 把秩为 0 的 tile 转回标量 |
| `ftof(tile, rounding)` | `Tile<EIn, S> -> Tile<EOut, S>` | 带舍入模式的浮点转浮点 |
| `ftoi(tile, rounding)` | `Tile<EIn, S> -> Tile<EOut, S>` | 带舍入模式的浮点转整数 |
| `itof(tile, rounding)` | `Tile<EIn, S> -> Tile<EOut, S>` | 带舍入模式的整数转浮点 |
| `bitcast(tile)` | `Tile<EIn, S> -> Tile<EOut, S>` | 重新解释位（不做转换） |
| `exti(tile)` | `Tile<EIn, S> -> Tile<EOut, S>` | 整数扩展（符号/零扩展） |
| `trunci(tile, overflow)` | `Tile<EIn, S> -> Tile<EOut, S>` | 带溢出模式的整数截断 |
| `int_to_ptr(tile)` | `Tile<SRC_T, S> -> PointerTile<*mut PTR_T, S>` | 整数转指针 |
| `ptr_to_int(ptrs)` | `PointerTile<*mut E, S> -> Tile<E, S>` | 指针转整数 tile |
| `ptr_to_ptr(ptrs)` | `PointerTile<*mut EIn, S> -> PointerTile<*mut EOut, S>` | 指针类型转换 |

```rust
// 浮点转整数
let indices: Tile<i32, { [128] }> = iota(shape![128]);
let float_indices: Tile<f32, { [128] }> = convert_tile(indices);

// Bitcast：把 f32 的位重新解释为 u32（不做取值转换）
let float_tile: Tile<f32, { [128] }> = constant(1.0f32, shape![128]);
let bits: Tile<u32, { [128] }> = bitcast(float_tile);  // 0x3F800000

// 整数的扩展与截断
let small: Tile<i16, { [64] }> = constant(42i16, shape![64]);
let wide: Tile<i32, { [64] }> = exti(small);     // 符号扩展 i16 -> i32
let narrow: Tile<i16, { [64] }> = trunci(wide, overflow::None);
```

### 编译器提示

| 函数 | 签名 | 说明 |
|---|---|---|
| `assume_div_by::<_, N>(val)` | `T -> T` | 断言取值能被 N 整除 |
| `assume_bounds_lower::<_, L>(val)` | `T -> T` | 断言取值 >= L |
| `assume_bounds_upper::<_, U>(val)` | `T -> T` | 断言取值 <= U |
| `assume_bounds::<_, L, U>(val)` | `T -> T` | 断言 L <= 取值 <= U |

这些是 `unsafe` 的 —— 假设不正确会产生未定义行为。它们能启用向量化加载、简化索引算术等编译器优化。

```rust
// 告诉编译器某个维度是 16 的倍数（启用更宽的向量加载）
let dim: i32 = unsafe { assume_div_by::<_, 16>(dim) };

// 给索引进界（启用基于范围的优化）
let idx: i32 = unsafe { assume_bounds::<_, 0, 1024>(idx) };

// 组合：非负且对齐
let stride: i32 = unsafe { assume_bounds_lower::<_, 0>(stride) };
let stride: i32 = unsafe { assume_div_by::<_, 4>(stride) };
```
### 调试

| 宏 | 说明 |
|---|---|
| `cuda_tile_print!(fmt, args...)` | printf 风格的 GPU 打印 |
| `cuda_tile_assert!(cond, msg)` | GPU 断言 |

```rust
let pid0 = program_id(0);
let pid1 = program_id(1);
let pid2 = program_id(2);
cuda_tile_print!("Program ({}, {}, {})\n", pid0, pid1, pid2);

// 断言一个条件 —— 为假时中止 kernel
cuda_tile_assert!(len > 0, "Length must be positive");

// 打印标量用于调试（每个程序都会运行，因此输出可能交错）
let offset = pid0 * 128;
cuda_tile_print!("offset = {}\n", offset);
```
