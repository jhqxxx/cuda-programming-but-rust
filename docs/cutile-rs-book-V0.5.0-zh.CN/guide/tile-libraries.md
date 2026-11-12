# Tile 库

一个 **tile 库**（`.tilelib`）把设备函数打包起来，供 kernel 像调用普通函数那样调用：一个 tile 的 FFT、一个 tile 的 Cholesky 分解，等等。Tile C++ 通过 `__experimental_foreign_decl__` 声明调用它们，cuTile Python 通过 `cuda.tile_preview.foreign_call` 调用，cuTile Rust 则通过 `#[cutile::foreign]` 声明调用。

> **注意**
>
> tile 库是 Tile IR 13.5 的公开预览特性。它们需要 CUDA 13.5 的 `tileiras`，以及一套针对同一工具包构建的 tile 库。

<a id="finding-tile-libraries"></a>

## 寻找 tile 库

先从 NVIDIA 的 [Tile IR 受支持库列表](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#supported-libraries)入手。随着各家包陆续发布，那里会链接到提供方。tile 库是与 CUDA Toolkit 分开发布的；只安装 CUDA 13.5 并不会装上所有提供方的 `.tilelib` 文件。

按提供方的安装说明，为你的工具包取得 `.tilelib` 及其依赖。它的接口决定了你能用的函数名、常量参数、形状、类型和架构。然后按下面的方式声明函数，并把安装路径传给 [`TileLibrarySpec`](#configuring-tile-libraries)。提供方列表描述的是 Tile IR 支持情况；cuTile Rust 的[当前操作数限制](#limitations)同样适用，其中包括外部调用不支持指针操作数。

本章里的 `solver.tilelib`、`fft_path` 以及各提供方库的路径都是示意性的配置示例。它们不是随附的下载项，也不保证是真实的包文件名。请把它们换成提供方包里的路径与声明。

### 来自 CUDA Oxide 的 Rust SIMT 函数

CUDA Oxide 也能从 Rust SIMT 函数构建 tile 库：用 `#[tile_export]` 标记它们，再用 `cargo oxide tilelib` 打包。导出定义、构建命令与当前的导出限制见 [`tilelib_relu` 示例](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/crates/rustc-codegen-cuda/examples/tilelib_relu/README.md)。

要在 cuTile Rust 里调用这些导出，请写出匹配的 [`#[cutile::foreign] unsafe fn` 声明](#declaring-a-foreign-function)，并在宿主侧用 `TileLibrarySpec` [配置生成的 `.tilelib`](#configuring-tile-libraries)。导出的函数名、具体 tile 形状、元素类型以及输入/输出的访问方式都要对上。这是设备函数层面的互操作：SIMT 函数在 Tile kernel 内部运行。

这用到了 CUDA 13.5 的预览特性，要求库、调用方的字节码和 `tileiras` 工具链彼此匹配；预览库不能跨任意工具包版本移植。Oxide 当前的导出限制会叠加在 [cuTile Rust 的限制](#limitations)之上。记录在案的 Oxide 执行验证用的是 Tile C++ 调用方；它并没有建立起 Oxide → cuTile Rust 的执行测试。

### 可复用的 Rust kernel

如果你需要 attention、归一化、transformer 工具函数、融合 QKV 与 NVFP4 MoE 的 Rust 实现，请看 [`cutile-kernels` 目录](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-kernels/README.md)以及 [`nvfp4_moe` 示例](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-examples/examples/nvfp4_moe.rs)。[融合 QKV 指南](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-examples/fused_qkv.md)介绍了它的正确性示例与可选的基准测试。

这些是从宿主侧启动的 Rust 源码 kernel。`.tilelib` 提供的则是 kernel 内部通过 `#[cutile::foreign]` 调用的设备函数。

<a id="declaring-a-foreign-function"></a>

## 声明外部函数

外部函数是 `#[cutile::module]` 内部一个没有函数体的函数，写起来与 tile 库在它自己的 Tile IR 接口里携带的那份声明一致：

```text
preview$func extern @cholesky<[
    Function: !cuda_tile<preview$string>, FillMode: !cuda_tile<preview$string>,
    Batch: index, M: index, T: !cuda_tile<preview$Datatype>
  ]>(
    %A: tile<[Batch]x[M]x[M]xpreview$sym_t<T>> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>},
    %info: tile<[Batch]xi32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
  ) -> (tile<[Batch]x[M]x[M]xpreview$sym_t<T>>, tile<[Batch]xi32>)
```

```rust
#[cutile::module]
pub mod solver {
    use cutile::core::*;

    /// # Safety
    ///
    /// 该声明与 `solver.tilelib` 的 `cholesky` 一致。
    #[cutile::foreign]
    pub unsafe fn cholesky<const Batch: i32, const M: i32, T: ElementType>(
        Function: &str,
        FillMode: &str,
        #[inout] A: Tile<T, { [Batch, M, M] }>,
    ) -> (Tile<T, { [Batch, M, M] }>, Tile<i32, { [Batch] }>);   // `info` 是纯输出
}
```

- 声明指定的是**库里的一个函数**，而不是某个库：由宿主决定 kernel 链接哪些库（见[配置 tile 库](#configuring-tile-libraries)）。
- `name = "..."` 是函数在库里的名字；默认取 Rust 里的名字。
- **常量实参**先排 `&str` 参数，再排泛型参数，两组各自保持顺序：`name: &str` 参数对应字符串，`const N: i32` 对应 `index`，`T: ElementType` 对应数据类型（`f32`、`f64`……）。它们与 Tile IR 声明里的常量参数、Tile C++ 声明的模板参数、cuTile Python `foreign_call` 的 `constant_params` 一一对应。如果库里排列顺序不同，就用 `const_args = "..."` 按那个顺序列出所有名字，例如 `#[cutile::foreign(const_args = "Batch, Function, M, T, FillMode")]`。
- **其余每个参数都是调用方传入的操作数**，按声明里的顺序排列：默认只读，加 `#[inout]` 表示读写。标量是 0 阶 tile。
- **返回值**以更新后的 `#[inout]` 操作数开头。后续的每个返回值都是纯输出操作数（上面例子里的 `info`）：它不是参数，调用时会在参数之后以零填充的形式传入。这正是 Tile C++ `__experimental_foreign_decl__` 声明的形态。
- 声明可以保留库里的命名（`Batch`、`A`）；本来会对此报警的 lint 对外部声明是关闭的。
- 外部函数声明为 `unsafe fn`，每次调用都要放在 `unsafe` 块里（见[安全性](#safety)）。

声明必须与库的接口完全一致：`tileiras` 会拿调用去核对库所携带的那份声明。

<a id="safety"></a>

### 安全性

编译器只看得到声明，看不到库的实现，所以外部函数是 `unsafe fn`。漏掉 `unsafe` 的声明是错误，在 `unsafe` 块之外调用也是错误。请在声明的 `# Safety` 小节里写明调用方的义务，即使除了下面这些之外没有别的。外部函数的调用方要保证：

- 声明与库的接口一致：名字、常量实参及其顺序、操作数的类型与形状、哪些操作数是 `#[inout]`、以及返回值。`tileiras` 会拒绝其中一部分不一致，但不是全部。
- 库针对目标架构、针对本次调用的常量实参实现了该函数。
- 函数只读写自己的操作数与返回值。外部函数接收的是 tile，而 tile 是值，所以今天所有库都满足这一点。一旦支持指针操作数，这就变成库的义务。

如果某个包装函数对它接受的每个输入都能满足上述义务，它就可以是一个安全的设备函数，内部在 `unsafe` 块里调用那个声明。

### 字符串常量

字符串常量就是一个 `&str` 参数。调用时传入的字符串由 JIT 在编译 kernel 时解析，和 `print_tko` 的格式串一样：

- 字面量：`cholesky::<1, 32, f64>("potrf", "lower", a)`；
- 取值为字面量的 `const` 项，用它的名字或路径：`const FILL: &str = "lower";`、`consts::FILL`；
- 绑定到上述之一的局部变量或函数参数 —— 这样包装函数就能接收 `fill: &str` 再往下传。但在运行期的 `if`、循环、`{}` 或 `unsafe {}` 块里被重新赋值的局部变量，在它之后（或在循环内部）就不再是常量：在那时传它会是编译错误；
- 标签类型的关联常量，`Lower::NAME`；对绑定到标签类型的泛型参数 `F`，用 `F::NAME`。如果一个标签类型在两个 trait 里都定义了 `NAME`，请写明 trait：`<F as FillMode>::NAME`。

字符串会成为本次调用特化的一部分：字符串不同的两次调用，就是两个不同的库函数。任何其它表达式（运行时算出的值，或 `concat!`）都是编译错误。

标签类型按它书写处的路径解析，与 Rust 一致 —— 通过 import 和 `pub use` 再导出。两个模块可以各自声明一个 `NAME` 不同的 `Lower`：每次调用传的是它所指名的那个 `Lower` 的字符串。函数或方法返回的标签，则是它在该函数所在模块里返回值类型所指名的那个。

### 与 Tile C++、cuTile Python 的对照

| | Tile C++ | cuTile Python | cuTile Rust |
|---|---|---|---|
| 声明 | `__experimental_foreign_decl__` 函数模板 | 无：每次调用都是 `foreign_call(...)` | `#[cutile::foreign]` 无函数体的 `unsafe fn` |
| 库中的名字 | `__experimental_foreign_name__("fft")` | `symbol_name="fft"` | `name = "fft"`（默认：Rust 名） |
| 常量参数 | 模板参数（`size_t`、`fixedString`、`typename`） | `constant_params=(int, str, dtype)` | `&str` 参数，然后是泛型参数（`const i32`、`ElementType`） |
| 读写操作数 | `[[cutile::experimental_inout]]` | 位置参数 `inputs` | `#[inout]` |
| 纯输出操作数 | 隐含，来自 `multi_outputs` | 显式传入 | 隐含：`#[inout]` 操作数之后的返回值 |
| 标量 | 0 阶 tile | 0 阶 tile | 0 阶 tile |
| 库 | `-Xtileiras -l=path` | `Tilelib(path, version)` | 宿主侧的 `TileLibrarySpec::new(path)` |

## 调用外部函数

调用写起来和调用普通设备函数一样，只是要放在 `unsafe` 块里。能由操作数推断出来的泛型实参可以省略；其余的用 turbofish 传，其中 `_` 让编译器推断剩下的部分：

```rust
let (factor, info) = unsafe { cholesky::<_, _, _>("potrf", "lower", a_tile) };
```

一次调用会降级成 Tile IR 的 `preview$call`，例如 `preview$call @cholesky<["potrf", "lower", 1, 32, f64]>(%a, %info0, %token)`，其中 `%info0` 是零填充的纯输出操作数。

外部调用的排序与访存一样：一次调用发生在它之前所有 load、store、原子操作与调用之后，也发生在它之后的所有这些操作之前，包括跨循环迭代以及在內联的辅助函数里。中间没有调用穿插的访存之间仍然是无序的，就和没有外部调用的 kernel 一样。

`#[inout]` 描述的是库如何使用这个操作数：更新后的值是调用的返回值，而调用方的 tile 本身不变 —— 和其他值一样。

两个模块可以各自声明同名（Rust 名）的外部函数；每次调用只会到达它所指名的那个。

<a id="configuring-tile-libraries"></a>

## 配置 tile 库

在配置这些路径之前，先按[寻找 tile 库](#finding-tile-libraries)取得兼容的提供方包。

库由宿主选择，kernel 链接哪些 tile 库也由宿主决定 —— 所以同一份 kernel 源码可以链接库的 debug 版或 release 版，也可以链接分别为不同架构构建的版本：

```rust
use cutile::tile_kernel::{set_default_tile_libraries, CompileOptions, TileLibrarySpec};

// 每个 kernel 都会链接的库，作用域是本进程的剩余生命周期。
set_default_tile_libraries([TileLibrarySpec::new("/opt/tilelibs/solver.tilelib")]);

// 在默认库之外，为这一次启动再链接一个库。
let (y,) = kernels::fft_kernel(y)
    .compile_options(CompileOptions::new().tile_library(TileLibrarySpec::new(fft_path)))
    .sync()?;
```

- 默认库（`set_default_tile_libraries`、`add_default_tile_library`）与单次启动的 `CompileOptions::tile_library` 列表会被合并、规范化、排序并去重，然后才交给 `tileiras` 链接。相对路径相对于当前目录。添加一个已经是默认项的库不会有任何变化。
- 没有外部调用的 kernel 不链接任何库。有外部调用却没配置库的 kernel 会编译失败，错误信息里会点名它调用的那些库函数。
- *实现提供方*（implementation-provider）库在链接时会加载一个共享库，为每次调用生成代码。请把它加载的每个文件、以及每个会改变所生成代码的设置都声明出来，这样任一方变化时缓存都会失效：

  ```rust
  let solver = TileLibrarySpec::new(mathdx.join("lib/solver.tilelib"))
      .dependency(mathdx.join("lib/libmathdx_tile.so"))
      .setting("mathdx-version", "26.01");
  ```

<a id="limitations"></a>

## 限制

- 操作数与返回值只能是 tile 和标量；指针操作数尚不支持。
- 常量参数只能是 `index`、字符串和数据类型参数；形状（tuple）参数不行。
- tile 库必须是针对链接它的那个 `tileiras` 构建的。
- 字符串常量必须是字面量、取值为字面量的 `const`，或取值为字面量的关联常量。JIT 不会去求值其它常量表达式。
- 跨模块的名字解析有它自己的限制；见[名字解析](../reference/dsl-api.md#name-resolution)。
