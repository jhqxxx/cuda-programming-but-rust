# 编译

cuTile Rust 既支持普通 kernel 启动时的 JIT 编译，也提供面向工具链与预编译流水线的仅编译 API。在两条路径下，编译器都会先解析出一个 kernel **特化**，把它下降为 Tile IR 字节码，再调用 CUDA Tile IR 汇编器为目标 GPU 生成代码。**特化（specialization）**是指针对某一个入口函数、某一个目标架构，以及一组会影响生成 GPU 代码的取值所编译出的具体变体。

![cuTile Rust 从 Rust 到 GPU 执行的编译流水线](../_static/images/compilation-pipeline.svg)

```text
Rust AST -> Tile IR bytecode -> cubin
```

对于启动时 JIT，编译好的 kernel 会缓存在宿主进程内。再次启动同一个特化时会复用缓存的 cubin。

## 启动时 JIT

对于生成的 `#[cutile::entry]` 启动器，编译发生在首次启动时，而不是在 Rust crate 本身构建的时候：

```rust
let op = add((&mut z).partition([16, 16]), &x, &y); // 构造一个 DeviceOp。
let _ = op.sync_on(&stream)?;                       // 可能先编译，再启动。
```

如果缓存中已经存在匹配的特化，启动会直接进行，不会重新编译。重启进程会清空内存中的缓存。

## 什么会导致特化

特化的机制与 Rust 的单态化类似：像 `fn f<T, const N: usize>(...)` 这样的泛型函数，会为每一组具体的 `T` 和 `N` 各编译一份。cuTile 会把入口函数的类型参数与 const 泛型参数纳入 kernel 缓存键。改变它们就会产生一个新的 GPU 特化。

### 常见的重新编译触发因素

**入口函数的参数。** `#[cutile::entry]` 签名里的类型参数和 const 泛型参数会对生成的 GPU 代码产生特化。这些取值并不总是写在 `.generics(...)` 旁边；它们也可以从传给启动器的宿主侧取值推断出来：

- `z.partition([BM, BN])` 可以为 `&mut Tensor<T, { [BM, BN] }>` 绑定 tile 形状。
- `z.partition([BM, BN]).map([MAP_M, MAP_N], num_tile_blocks)` 可以绑定输出 tile 形状，以及 `MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>` 的 `MAP_SHAPE`。
- 把 tensor 传给带有静态或泛型维度的参数（例如 `&Tensor<T, { [K, -1] }>`）时，可以从 tensor 的形状中绑定这些静态维度。对于希望可以自由变化而无需重新编译的维度，请使用 `-1`。

对于映射分区，`MAP_SHAPE` 会影响 JIT，因为它是设备侧类型的一部分。`num_tile_blocks` 则不同：它控制的是启动网格的推断，也就是有多少个物理 tile block 去遍历这个映射分区。它本身不会创建新的缓存条目。

运行时的 tensor 数据和运行时的标量值可以变化而不触发重新编译。动态 tensor 维度（`-1`）是在运行时传入的；具体的维度取值并不属于入口函数的泛型参数。

**目标架构。** 一个 cubin 是针对某一个目标 GPU 架构生成的。在另一个 SM 目标上运行同一个入口函数，需要另一份编译产物。

### 显式的编译期输入

其余由用户控制的重新编译触发因素是显式的。只有当编译器确实应该针对该取值做特化时，才使用这些 API：

- `.const_grid((x, y, z))` 把启动网格作为编译期常量嵌入。改变它会创建一个独立的特化。当你希望启动网格只在运行时决定时，用 `.grid(...)`。
- `.compile_options(opts)` 把调优选择（例如占用率、架构相关的调度设置、整除性提示）嵌入进去。不同的选项会产生独立的缓存条目。

像 `.grid(...)`、标量值、tensor 内容、动态维度（`-1`）以及 `MappedPartitionMut` 的 `num_tile_blocks` 这些运行时输入，本身不会创建缓存条目。

> **注意**：cuTile 也可能针对由形状与步长元数据推导出的 tensor 特化提示，额外缓存若干优化变体。这是一种性能取舍：当编译器掌握诸如"形状维度或步长可被 2 的幂整除""某个步长恰好等于 `1`"这类事实时，它就能生成更好的代码。这些事实可以带来更简单的索引、更强的对齐假设以及更高效的内存访问。这些提示会影响缓存复用，但它们并不是额外传给 kernel 入口函数的取值。

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    z.store(x.load_like(z));
}
```

把 `S` 从 `[16, 16]` 改成 `[32, 32]` 会创建一个新的特化。把 `x` 的运行时形状从 `[1024, 1024]` 改成 `[2048, 1024]` 不会改变入口函数的泛型参数，因为这两个维度在 kernel 签名里都是动态的。

## 为缓存复用而设计

把入口函数的签名看作**特化的边界**。只有当某个取值改变时确实应该产生不同的 GPU 代码，才把它放进类型泛型或 const 泛型里。当同一份编译好的 kernel 应该能处理多种问题时，就把变化的问题规模保留为动态 tensor 维度。

tile 形状的维度是"保持尺寸动态"这条规则的主要例外。Tile IR 要求 tile 形状必须是编译期常量，而 cuTile Rust 通过 `Tile<T, S>`、`tensor.partition(S)` 这类 API 把它暴露出来。`BM`、`BN`、`BK` 这些维度常常是 const 泛型，因为改变它们就改变了编译器要下降到 Tile IR 的 tile 类型。

tile 形状的维度与完整 tensor 的维度扮演着不同的角色。对于可变输出，像 `&mut Tensor<f32, { [BM, BN] }>` 这样的签名通常接收宿主侧的 `z.partition([BM, BN])`；其中的泛型形状是暴露给 kernel 的**输出 tile 形状**，而不是完整的输出 tensor 形状。这是预期行为，不会为每一个问题规模都创建一个特化。对于只读 tensor，泛型形状维度描述的是 kernel 可见的 tensor 形状。给预期会变化的输入维度用 `-1`，只有当某些输入维度确实应该成为特化的一部分时，才使用泛型/静态的输入维度。

```rust
#[cutile::entry()]
fn normalize<const BM: i32, const BN: i32>(
    z: &mut Tensor<f32, { [BM, BN] }>,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    let tile = x.load_like(z);
    let row_max = reduce_max(tile, 1i32);
    z.store(tile - row_max.reshape(shape![BM, 1]).broadcast(z.shape()));
}
```

改变 `BM` 或 `BN` 会产生新的特化，因为它们出现在入口函数签名里。改变 `x` 的完整形状则可以复用同一个入口函数特化，因为这两个维度在签名里都是 `-1`；形状校验仍然会在启动时执行。

在热点路径上，请让 tile 形状、元素类型、目标架构、const 网格以及编译选项的集合保持足够小。过度的特化会增加首次启动的延迟和内存占用。

Tensor 与标量的特化提示会按"能否被 2 的幂整除"分档。默认情况下，自动推断出的提示上限为 `16`：像 `16`、`32`、`64`、`1024` 这些维度都会记录为同一个 `divisor = 16`，因此在都能被 `16` 整除的取值之间变化动态维度，不会通过该维度的整除性提示产生新的缓存条目。但从一个能被 `8` 整除的值改成能被 `16` 整除的值，则可能产生不同的特化提示，从而产生不同的缓存条目。

`max_divisibility` 控制编译器基于这些提示所能发出的最强整除性假设。它是一个**上限**，而不是要求你凭空造出更强的事实。例如 `max_divisibility = 4` 会让一个已知能被 `16` 整除的值以 `div_by<4>` 的假设进行编译。修改 `max_divisibility` 本身就属于编译选项变更，因此会创建独立的缓存条目。目前大于 `16` 的取值并不会强化自动推断出的提示，因为推断本身已经被限制在 `16`。

## 编译选项与提示

入口级的优化提示写在 `#[cutile::entry]` 属性上：

```rust
#[cutile::entry(
    optimization_hints = (
        sm_120 = (
            num_cta_in_cga = 2,
            occupancy = 2,
            max_divisibility = 16,
            num_worker_warps_per_cta = 4, // 字节码 13.3+；合法取值：1、2、4、8、16、32。
        ),
        sm_90 = (
            num_cta_in_cga = 1,
        ),
    )
)]
fn optimized_kernel<const S: [i32; 2]>(...) { ... }
```

宿主侧可以用 `CompileOptions` 覆盖这些取值，这在自动调优时很有用：

```rust
use cutile::tile_kernel::CompileOptions;

let opts = CompileOptions::default()
    .occupancy(4)
    .num_cta_in_cga(2)
    .num_worker_warps_per_cta(4); // 字节码 13.3+；合法取值：1、2、4、8、16、32。

let _ = optimized_kernel(args)
    .compile_options(opts)
    .sync_on(&stream)?;
```

`num_worker_warps_per_cta` 要求字节码版本 13.3 或更高。它必须是闭区间 `[1, 32]` 内的 2 的幂（`1`、`2`、`4`、`8`、`16` 或 `32`）。

不同的 `CompileOptions` 取值会产生独立的 JIT 缓存条目。

当某次具体的加载或存储需要延迟提示或张量内存加速器（TMA）提示时，可以使用底层访存 API 上的逐操作提示：

```rust
let tile = load_view_tko(
    &partition,
    idx,
    ordering::Weak,
    scope::TileBlock,
    Some(4),
    tma::Enabled,
);
```

请在测量之后再使用提示来引导编译器。对大多数 kernel 来说，编译器的默认选择就是合适的起点。

## kernel 缓存键

启动时，cuTile 会使用当前活跃设备与 kernel 特化去查找已编译 kernel 的缓存。下面的伪代码使用了定义这次查找的内部字段名：

```rust
match TileFunctionKey {
    module_name,
    function_name,
    function_generics,
    stride_args,
    spec_args: [
        (tensor_param, SpecializationBits {
            shape_div,
            stride_div,
            stride_one,
            base_ptr_div,
            elements_disjoint,
        }),
        ...
    ],
    scalar_hints: [
        (integer_or_pointer_param, DivHint { divisor, max }),
        ...
    ],
    grid,
    compile_options,
    source_hash,
    device_id,
    gpu_name,
    compiler_version,
    tileiras_fingerprint,
} {
    cached => reuse_compiled_kernel(),
    missing => compile_and_insert_into_cache(),
}
```

| 字段 | 什么会改变它 |
|---|---|
| `module_name`、`function_name` | 正在启动的 `#[cutile::module]` 模块与 `#[cutile::entry]` 函数。 |
| `function_generics` | 入口函数的类型参数与 const 泛型参数，包括作为签名一部分的 tile 形状（如 `BM`、`BN`、`BK`）。 |
| `stride_args` | 为 tensor 参数记录的步长为 1 的模式。 |
| `spec_args` | tensor 特化提示。目前包含逐维度的形状整除性、逐维度的步长整除性、每个步长是否恰好为 `1`、基指针整除性，以及由步长推导出的不重叠位。 |
| `scalar_hints` | 由整数标量参数和裸指针参数推导出的 `DivHint`。整数提示基于取值的整除性；指针提示基于指针对齐。 |
| `grid` | 仅当使用了 `.const_grid(...)` 时为 `Some((x, y, z))`。运行时的 `.grid(...)` 会让它保持为 `None`。 |
| `compile_options` | 逐次启动的编译选项，例如占用率、架构相关的调度设置以及整除性上限。 |
| `source_hash` | 由 `#[cutile::module]` 生成的、模块源码文本的哈希。修改模块中任意一个 kernel 或设备函数（并重新构建）都会改变它，因此绝不会跨源码变更复用陈旧的已编译 kernel。 |
| `device_id` | 当前活跃的 CUDA 设备。编译产物是按设备加载的。 |
| `gpu_name` | 目标架构（`sm_90`、`sm_120`……），普通启动时由设备推断得到。一个 cubin 只针对一种架构生成。 |
| `compiler_version` | 生成该入口的 `cutile-compiler` crate 版本。 |
| `tileiras_fingerprint` | 解析出的汇编器二进制的 `tileiras --version` 输出（若不可用则退化为其路径、大小与 mtime）。升级工具包，或把 `CUTILE_TILEIRAS_PATH` 指向另一个二进制，都会改变它。 |

不在此模式中的取值都是普通的启动输入或数据输入。tensor 的内容、浮点标量值，以及通过 `.grid(...)` 传入的运行时网格值，本身都不会创建缓存条目。

<a id="bytecode-version"></a>
## 持久化磁盘缓存

上面的 kernel 缓存位于进程内存中，会随进程结束而消失。cuTile 还可以把编译好的 cubin 持久化到磁盘，这样新进程对于任何先前进程已经编译过的 kernel，都能跳过 `tileiras` 后端。

磁盘持久化**默认关闭，且没有任何环境变量能打开它**。程序必须显式启用：

```rust
// ~/.cache/cutile/kernels（会遵循 $XDG_CACHE_HOME），2 GiB LRU 目标：
cutile::jit_cache::enable_default()?;

// 或者自定义位置与容量：
use cutile::jit_cache::FileSystemJitStore;
let store = FileSystemJitStore::builder("/data/kernel-cache")
    .capacity_bytes(512 << 20)
    .open()?;
cutile::jit_cache::enable(std::sync::Arc::new(store));
```

`cutile::jit_cache::disable()` 会停止所有磁盘读写；已在内存缓存中的 kernel 不受影响。`enable` 背后的 `JitStore` trait 是一个面向字节的 get/put 接口，因此可以用自定义后端（对象存储、网络缓存）替换文件系统实现。

命中磁盘缓存会跳过 `tileiras`。但每次内存未命中时，编译器前端仍会运行，以校验启动参数。

持久化键是对完整 `tileiras` 输入做 SHA-256 得到的：

- 字节码版本（`major.minor` 与 `tag`；见[字节码版本](#bytecode-version)）。
- 序列化后的 Tile IR 字节码，包含被内联的依赖模块。
- 目标架构。
- 优化级别以及第二阶段开关（`device_debug`、`lineinfo`、`sanitize_memcheck`）。
- 解析出的 `tileiras` 二进制的指纹。

上述输入发生变化就会让旧条目失效。版本号相同但实际不同的汇编器二进制不会被区分开。

每个条目都带有一个带校验和的头部，每次命中都会校验。不完整、损坏或不匹配的条目会被重新编译。该校验和**不用于认证代码**：任何拥有写权限的人都可以构造一个合法条目，其中包含任意的设备代码。请把自定义存储与共享存储的写入权限限制在可信用户范围内。

缓存 I/O 错误会被计数，但不会让启动失败；编译会在没有磁盘缓存的情况下继续。要查看缓存使用情况：

- `cutile::jit_cache::stats()` —— 命中数、未命中数、写入条目数、写入字节数、软 I/O 错误数。
- `cutile::jit_cache::jit_backend_compile_count()` / `jit_disk_hit_count()` —— 配合 `jit_compile_count()`，在无失败的情况下满足 `compiles == backend + disk_hits`。
- `CUTILE_JIT_TIMING=1` —— 每个 kernel 的计时行里会包含 `stage2_source=disk|tileiras`。

淘汰策略是按文件 mtime 做 LRU，并配有一对高低水位（默认：超过容量就开始回收，删到最旧的条目只剩 80%）。配置的容量是软目标而非硬上限：淘汰是概率性触发的，并且使用非阻塞锁，因此短暂超标是可能的。多个进程可以共享同一个缓存目录；写入是原子的，淘汰则通过锁文件协调。

并发的多个进程可能各自编译同一个尚不存在的条目。建议先在一个进程里预热共享缓存，再启动并行 worker，以避免重复劳动。

参见 `cutile-rs-examples/examples/jit_disk_cache.rs`；跑两遍即可看到第二个进程命中磁盘。要自行实现后端（对象存储、数据库），参见 `cutile-rs-examples/examples/jit_custom_store.rs`。

## 计算缓存键

使用生成的启动器可以推导出一次普通启动会用到的那两个确切的键：

| 需求 | API |
|---|---|
| 只要内存中的 L1 键 | `launcher.l1_cache_key()?` |
| 只要持久化的 L2 键 | `launcher.l2_cache_key()?` |
| 某个已解析启动特化的两个键 | `launcher.specialize()?` |
| 某个仅编译特化的 L2 键 | `KernelCompiler::l2_cache_key()?` |

### 直接获取单个键

只需要一个键时，调用启动器上对应的辅助方法：

```rust
let l1_key = my_kernels::copy(
    cutile::api::meta::<f32>(&[1024]).partition([128]),
    cutile::api::meta::<f32>(&[1024]),
)
.generics(vec!["f32".into(), "128".into()])
.l1_cache_key()?;
```

或者直接推导 L2 键：

```rust
let l2_key = my_kernels::copy(
    cutile::api::meta::<f32>(&[1024]).partition([128]),
    cutile::api::meta::<f32>(&[1024]),
)
.generics(vec!["f32".into(), "128".into()])
.l2_cache_key()?;
```

`l1_cache_key()` 返回完整的结构化 `TileFunctionKey`，而不是简写过的 `display_hash()` 调试标签。`l2_cache_key()` 返回持久化缓存所用的、由 64 个小写字符组成的 SHA-256。

这两个终止调用都会消耗掉各自的启动器。如果两个键都需要，请把启动特化解析一次，而不是把启动器构造两遍。

### 一次拿到两个键

`specialize()` 会在不编译、不启动 kernel 的前提下，解析出这次启动的特化标识。它会把启动输入实体化，并返回一个 `Specialization`，其中包含完整的 L1 键和一个惰性的模块 AST 提供者：

```rust
let specialization = my_kernels::copy(
    cutile::api::meta::<f32>(&[1024]).partition([128]),
    cutile::api::meta::<f32>(&[1024]),
)
.generics(vec!["f32".into(), "128".into()])
.specialize()?;

let l1_key = specialization.l1_cache_key().clone();
let l2_key = specialization.l2_cache_key()?;
```

`Specialization::l1_cache_key()` 只是借用已经解析好的键，因此它返回 `&TileFunctionKey` 而不是 `Result`。`specialize()` 可能在实体化启动输入时失败，`l2_cache_key()` 可能在运行编译器前端或序列化字节码时失败，所以这两处调用都带 `?`。

### 与真实的 L2 查找保持一致

L2 推导会运行编译器前端并序列化 Tile IR，因为持久化键要对前端输出做哈希。辅助方法与真实的磁盘缓存查找共用同一个"字节码到键"的函数：

```text
launcher.l2_cache_key()
    ↓
Specialization::l2_cache_key()
    ↓
KernelCompiler::l2_cache_key()
    ├── compiler frontend
    ├── Tile IR bytecode serialization
    ▼
current_l2_key_for_bytecode()
    ▼
l2_key(...)

compile_bytecode_cached()
    ▼
current_l2_key_for_bytecode()
    ▼
l2_key(...)
```

字节码版本来自序列化器。启动器上的辅助方法会从启动设备解析目标；仅编译的调用方则需要在 `KernelCompiler` 上自行配置。默认优化级别与当前 `tileiras` 指纹在内部解析。这保证了辅助方法返回的正是 JIT 会去查询的那个 L2 键。

### 辅助方法做了哪些工作

两个启动器辅助方法都会实体化上游的 `DeviceOp` 输入，以便推导步长与特化提示。任意上游操作都可能在这一步执行设备工作；缓存工具可以使用 `api::meta` 输入来提供 tensor 元数据，而无需分配设备显存。

输入实体化之后：

- L1 推导不会构造模块 AST，也不会运行编译器前端。
- L2 推导会构造 AST、运行前端并序列化字节码。
- 两个辅助方法都不会读写 `JitStore`、不会运行目标 kernel 的 `tileiras` 后端、不会加载 CUDA 模块，也不会填充 L1 缓存。

带 `_on` 的变体（`specialize_on`、`l1_cache_key_on`、`l2_cache_key_on`）会从显式指定的流中选择设备，而不是走默认设备策略。以上这些 API 都只做键的推导；调用方仍需自行负责 `JitStore::contains`、`delete`、缓存播种以及预热编排。

## 仅编译 API

大多数用户通过启动生成的 `#[cutile::entry]` 启动器来编译 kernel。cuTile 还暴露了 `cutile::compile_api::KernelCompiler`，用于预编译，以及其他需要 Tile IR 文本或字节码、但不启动 kernel 的仅编译工作流。

```rust
use cutile::compile_api::KernelCompiler;

let artifacts = KernelCompiler::new(my_kernels::__module_ast_self, "my_kernels", "tile_math")
    .generics(vec!["32".into()])
    .strides(&[("output", &[1])])
    .target("sm_80")
    .compile()?;

let ir_text = artifacts.ir_text();
let bytecode = artifacts.bytecode()?;
```

对于只需要持久化键的缓存工具，用同一个 builder 调用 `l2_cache_key()`：

```rust
let l2_key = KernelCompiler::new(my_kernels::__module_ast_self, "my_kernels", "tile_math")
    .generics(vec!["32".into()])
    .strides(&[("output", &[1])])
    .target("sm_80")
    .l2_cache_key()?;
```

这会运行前端和规范的字节码序列化器，但不会编译 cubin，也不会访问已配置的 `JitStore`。目标来自 `.target(...)`；字节码版本、默认优化级别以及当前 `tileiras` 指纹都在内部选定。需要刻意模拟另一套工具链的高级工具，仍然可以直接调用更底层的 `jit_cache::l2_key(...)` 并显式传入各项输入。

`KernelCompiler` 接收的特化信息与启动器通常从参数推断出的那套一致：入口函数的泛型参数、tensor 步长与特化提示、标量提示、目标架构、可选的常量网格，以及编译选项。它适用于预编译流水线、工具、测试，以及对生成的 IR 或字节码做纯 CPU 校验。由于仅编译模式下没有活跃的 CUDA 设备，所以由 `.target("sm_...")` 显式提供目标架构。

仅编译的预编译与启动时的 JIT 缓存是两回事。`KernelCompiler` 只为你描述的那个特化返回产物；它不会分配 tensor、不会启动 CUDA 工作、不会把编译好的函数插入进程内的 JIT 缓存，也不会产生宿主侧的结果值。之后通过生成启动器的调用依然会走正常的缓存查找，并且可能在首次启动时编译 —— 除非那条启动路径被改造成消费预编译产物。

## 检查编译器输出

有两个入口属性可用于检查。`print_ir = true` 会在 JIT 编译期间打印生成的入口包装、源码 kernel 以及编译后的 Tile IR 文本：

```rust
#[cutile::entry(print_ir = true)]
fn debug_kernel<const S: [i32; 2]>(...) { ... }
```

`dump_mlir_dir` 会把编译后的 Tile IR 文本写入文件：

```rust
#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
fn debug_kernel<const S: [i32; 2]>(...) { ... }
```

没有任何属性可以用于替换成手改的 IR：编译器总是汇编它自己的输出。

环境变量 `CUTILE_DUMP` 与 `CUTILE_DUMP_FILTER` 会把编译器的输出打印到 stderr，每个已编译模块打印一次：

| 变量 | 说明 |
|---|---|
| `CUTILE_DUMP` | 逗号分隔的阶段：`ir`（Tile IR 模块的文本形式）、`bytecode` 或 `bc`（编码后的镜像，解码回文本），或者用 `all` 表示两者。目前只会输出这些阶段；`ast`、`resolved`、`typed`、`instantiated` 这些 pass 级别的名字会被接受，但不产生任何输出。 |
| `CUTILE_DUMP_FILTER` | 逗号分隔的 `module::function` 路径或裸函数名。转储是按模块进行的，所以带限定名的条目会按它的模块部分来筛选（`kernels::fmha_prefill` 会转储从 `kernels` 编译出的每一个 kernel）；裸函数名则匹配所有模块。 |

`TILE_IR_DUMP=1` 是 `CUTILE_DUMP=ir` 的旧别名。

## 字节码版本

编译器能以 13.2 和 13.3 两个版本写出 Tile IR 字节码，并且会按工具链选择其一，以确保永远不会把更新的默认版本交给更旧的 `tileiras`：

1. 若设置了 `CUTILE_BYTECODE_VERSION`（例如 `13.2`），它覆盖其他所有规则。
2. 否则由解析出的工具包的 `cuda.h` 决定：CUDA 13.2 输出 13.2，CUDA 13.3 及更新版本输出 13.3。
3. 再否则 —— 即 `CUTILE_TILEIRAS_PATH` 指向一个裸二进制，或者在 `PATH` 上找到 `tileiras` —— 编译器会用一个小型代表性 kernel 按从新到旧的顺序探测每个版本。

最终结果会被夹取到可输出的范围内。比 CUDA 13.2 更旧的工具包，或者探测无法进行（临时目录不可写、二进制无法启动或崩溃、或者它拒绝了所有版本），都会产生一个指明原因的编译错误，而不是静默回退。选定的版本是持久化缓存键的一部分，而 `CompileArtifacts::bytecode()` 写出的正是启动时会使用的那份协商结果。

## 编译失败

大多数编译期错误都是普通的类型错误或形状错误：tile 形状不兼容、不支持的元素类型组合、非法的归约轴，或者缺少 `#[cutile::entry()]`。

JIT 失败通常指向本机 CUDA 安装或目标架构：

| 现象 | 可能原因 | 检查 |
|---|---|---|
| `invalid GPU architecture` | 工具链不支持所选的 SM | `tileiras --help`、CUDA Toolkit 版本 |
| Failed to load generated kernel | 驱动与工具包不匹配 | `nvidia-smi`、`nvcc --version` |
| 在 CUDA 或 Tile IR 库内部发生段错误 | 工具包路径损坏或库不兼容 | `CUDA_TOOLKIT_PATH`、动态库搜索路径 |
| 首次启动时 OOM | 编译期间宿主内存耗尽 | 系统内存、特化数量 |

在修改环境变量或清除坏掉的内存缓存条目之后，重启进程以强制重新编译。

---

接下来阅读[设备操作](device-operations.md)。
