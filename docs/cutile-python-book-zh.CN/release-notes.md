# 发行说明（Release Notes）

> 本页按版本倒序记录 cuTile Python 的变更。函数名与小节标题保留原文结构。

## 1.6.0（2026-09-09）

本版增加了对 **CTK 13.4** 特性的支持，包括：程序化依赖启动（PDL）、不做检查的内存访问、
`ct.insert()`，以及 Rubin 上 FP4 块缩放 MMA 的 `float8_e5m3fnu` 缩放 dtype。
它还通过静态数组 stride 和可移植的 TileIR bytecode 导出，带来了对编译与执行的更细粒度控制。
支持的 Python 子集扩展了用户自定义上下文管理器、dataclass 方法、非静态循环中的 `break`，
以及整数 `divmod`。自动调优、JAX 互操作、诊断信息和浮点行为也都有改进。

### CTK 13.4 特性

- 新增**程序化依赖启动（Programmatic Dependent Launch）**支持：`ct.grid_dependency_control_launch_dependents()`、`ct.grid_dependency_control_wait()`，以及 `ct.launch()` 的 `programmatic_dependent_launch` 选项。
- 给 `ct.load()`、`ct.store()` 以及 `TiledView.load()` / `TiledView.store()` 方法新增 `check_bounds` 选项，默认为 `True`。设为 `False` 表示声明「每个 tile 元素都在数组边界内」，从而跳过边界检查以获得更快的访问。
- 新增 `ct.insert()` 与 `Tile.insert()`：通过替换一个较大 tile 的子 tile 来创建新 tile。这个操作是 `ct.extract()` 的**逆操作**；原 tile 保持不变。
- 在 Rubin（`sm_107`）上，为 FP4 块缩放 MMA 新增 `float8_e5m3fnu` 缩放 dtype 支持。

### 特性

- 支持把数组的 **stride 维度特化为启动时的取值**，使其在 kernel 内部成为编译期常量。用法是把 `ct.ArrayAnnotation` 作为 `Annotated` 元数据，并列出要特化的维度，例如 `Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0, 1))]`。一旦指定了任何静态 stride，调度器就**不再**为所有维度自动推断 `stride == 1`，所以若连续维也应保持为编译期常量，需要把它显式列出来。
- 新增 `ct.ensure_constant()`，静态断言某个值是编译期常量。
- 浮点的 `astype()` 转换新增 **IEEE 舍入模式**支持。
- 给 `ct.min()`、`ct.max()`、`ct.argmin()`、`ct.argmax()`、`ct.minimum()`、`ct.maximum()` 新增 `propagate_nan` 选项。默认忽略 NaN；开启后 NaN 会传播 —— min/max/minimum/maximum 返回 NaN，argmin/argmax 返回第一个 NaN 的索引。

### Python 特性

- 支持非静态 `for` 循环中的 `break`。
- 支持用 `@contextlib.contextmanager` 创建的用户自定义上下文管理器。
- 支持用户自定义 dataclass 方法：`__post_init__()`、`__call__()`、`__getitem__()`、`__setitem__()`、`__repr__()`、`__str__()`。
- 新增 `ct.divmod()`，并支持内置 `divmod()`（仅限整数输入）。
- 允许在 `ct.static_eval()` 内部构造基本的符号表达式。

### 改进

- 当某个数组维度的 stride 被推断为 1 时，除非用 `ArrayAnnotation(static_stride_dims=...)` 显式选中，否则不再把它作为静态 stride 编码进数组类型。这允许控制流中更宽松的类型统一。
- `ct.tune.exhaustive_search()` 的 `kernel` 参数现在可以接受一个函数，把每个配置映射到待调优的 kernel。传固定 kernel 的行为保持不变。
- 当 `output_format="tileir_bytecode"` 时，`export_kernel()` 可以在不指定 `gpu_code` 的情况下导出 TileIR bytecode。这要求 bytecode 版本 13.3 或更高。
- 当 `Array.slice()` 的 `start` 和 `stop` 都是编译期常量时，把切出的轴特化为编译期常量维度。
- 在 TileIR 13.4 或更高版本上，当 `x` 是非受限浮点、`y` 是取值落在有符号 32 位整数范围内的整数时，把 `x ** y` 和 `ct.pow(x, y)` 下降为整数指数的 `FPowI`。其它情况仍然把指数提升为浮点并使用 `FPowF`。
- 改进「函数返回类型在不同控制流路径上不一致」这类问题的诊断信息。
- `TuningResult.failures` 每项的第二元素从「异常类名」改为「异常类本身」。

### Bug 修复

- 让 `ct.argmin()` 与 `ct.argmax()` 默认忽略 NaN，与 `ct.min()`、`ct.max()` 保持一致。
- 修复 `ct.launch()` 总是为 CUDA device 0 编译 kernel（而不是为实际启动所在设备编译）的问题。现在 kernel 按启动设备的架构编译；同架构的设备仍可共享已编译的 kernel。
- 修复开启 `CUDA_TILE_ENABLE_CRASH_DUMP=1` 时出现的 `AttributeError`。
- 修复「运行时循环条件会更新循环携带的标量」时触发的编译器内部错误。

### 工具

- 新增 `cutile-cache log`，一个用于查看编译历史的命令行工具。缓存元数据包括 mangled 后的 kernel 名、编译器版本、编译日期与耗时；在 TileIR 13.4 下还包括 `tileiras` 的编译备注。

### 生态

- JAX FFI 集成新增对 tuple 参数的支持。

## 1.5.0（2026-07-08）

本版通过把选定的数组形状维度变为编译期常量、以及声明标量的整除性假设，
带来了对 kernel 特化的更细粒度控制。自动调优更快，并且能隔离挂起的 kernel。
支持的 Python 子集现在包括 tuple 推导式、tuple 取值的 kernel 参数、枚举、字典，
以及变参关键字参数。

### 特性

- 新增 `ct.assume_divisible_by(x, divisor)` —— 一个声明「整数标量能被常量整除」的编译器提示。编译器会把这一事实沿算术运算传播，从而可以证明派生索引与指针偏移的对齐，启用更宽的内存操作。
- 支持把数组的 **shape 维度特化为启动时的取值**，使其在 kernel 内部成为编译期常量。用法是把 `ct.ArrayAnnotation` 作为 `Annotated` 元数据并列出要特化的维度，例如 `Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(0, -1))]`。
- 给 `ct.tune.exhaustive_search()` 新增 `single_run_timeout_sec` 参数，避免某个挂起的 kernel 拖死整轮搜索。
- 支持把 Python tuple 作为 kernel 参数传入。元素可以是数组、标量、列表或嵌套 tuple。注解既可以作用于整个 tuple，也可以作用于单个元素：`ct.Constant[tuple[int, float]]` 让整个 tuple 成为编译期常量；`tuple[ct.Constant[int], float]` 只让第一个元素为常量。

### Python 特性

- 支持 tuple 推导式。
- 支持 tuple 上的 `in` 与 `not in` 运算符。
- 对字典、用户自定义函数中的变参关键字参数（例如 `def foo(**kwargs)`）、字典解包（例如 `foo(x, **y)`）提供有限支持。
- kernel 内部支持 Python `enum.Enum` 的比较与构造。枚举成员可以用 `==`、`!=` 比较，也可以从常量值构造，例如 `Color(0)`。
- 支持打印 dataclass 实例。
- 允许 frozen dataclass 实例在设备代码中作为全局对象使用。

### Bug 修复

- 修复「在某 block 内创建的假设变量被该 block 之外使用」时，整除性自动传播的 bug。
- 修复严格类型数值常量（例如 `ct.uint32(-3)`、`ct.int8(300)`、`ct.float16(0.2)`）未按其 dtype 转换的 bug。现在超出范围的整数会按 dtype 的范围回绕，浮点值会按 dtype 的精度与范围做舍入或截断。
- 尊重安装在用户自定义函数上的包装器，包括用 `functools.wraps()` 的装饰器添加的包装器。
- 拒绝 `load()`、`store()`、`num_tiles()` 的 `order` 参数中出现重复的轴。

### 改进

- 扩展 `ct.arange()`，新增可选的 `start` 和 `step` 参数：`ct.arange(size, start=0, step=1, dtype=...)`。`size` 必须是常量整数，而 `start` 和 `step` 可以是动态值。例如 `ct.arange(8, start=7, step=-1, dtype=ct.int32)` 生成一个倒序范围。
- 通过对慢配置提前退出，改进穷举搜索的性能。

### ABI 变更

- 为带静态形状注解或 tuple 参数的 kernel 引入 **calling convention v2**。

## 1.4.0（2026-05-26）

本版的重点是与 **CTK 13.3** 的兼容，它引入了 Hopper（sm_90）GPU 支持、块缩放 MMA、
新的浮点 dtype、pack/unpack 操作、原子 store 操作、高级索引的 load/store、
支持间隙或重叠访问的 tiled view，以及对大数组和标量的支持。
此外还加入了对 Python 3.14（含 free-threading）、星号 `*` 表达式、frozen dataclass 的支持，
以及与 JAX 的集成。

### CTK 13.3 特性

- 支持 Hopper（sm_90 系列）GPU。
- 新增受限浮点 dtype `float8_e8m0fnu` 与 `float4_e2m1fn`。
- 新增 `ct.mma_scaled()`，用于块缩放的矩阵乘法累加。
- 新增 `ct.pack_to_bytes()`：把 tile 展平并把其原始字节重新解释为一维 uint8 tile；`ct.unpack_from_bytes()` 是其逆操作。
- 新增 `ct.load_advanced_indexing()` 与 `ct.store_advanced_indexing()`：在**一个**维度上做 gather/scatter，同时在其它维度上做切片。
- 新增 `atomic_store_add` 以及 `TiledView` 上更多的原子方法，用于在给定 tile 索引处对 tiled view 做逐元素原子读-改-写。
- 支持带 `traversal_steps` 的 `Array.tiled_view()`，用于创建 tile 相互重叠或留有间隙的 tile 空间。
- `ByTarget` 现在接受 `default` 值，作用于所有架构（例如 `ByTarget(sm_100=8, default=2)`），这使生成的 TileIR bytecode 可以**与 GPU 架构无关**。
- 新增 `ct.astile()`：从标量（产出 0 维 tile）或（可能嵌套的）标量 tuple 创建 tile，嵌套层次决定 tile 的形状。
- 支持大数组（>20 亿元素）：
  - 新增 `ct.IndexedWithInt64` 注解，用于 shape 或 stride 超出 32 位整数范围的数组 kernel 参数。不加注解的数组继续用 `int32` 表示 shape 与 stride。
  - 新增 `ct.ScalarInt64` 注解，强制把标量整数 kernel 参数推断为 `int64`（默认是 `int32`）。
- 给 `ct.mma()` 新增 `use_fast_acc` 选项，在 Hopper GPU 上为 FP8 输入（`float8_e4m3fn`、`float8_e5m2`）启用快速累加模式。
- 给 `ct.kernel` 新增 kernel 提示 `num_worker_warps`。
- `ct.atomic_add()` 现在在 Hopper（sm_90）及更新架构上支持 `bfloat16` 操作数。
- `ct.exp()` 新增可选的 `rounding_mode` 参数（f32 下支持 `RoundingMode.FULL` 和 `RoundingMode.APPROX`）。

### Python 特性

- 支持 Python 3.14 与 Python 3.14t（free-threading）。
- 支持 frozen dataclass。
- 支持把 tuple 作为带星号的函数实参传入，例如 `foo(a, *b, c)`。
- 支持用户自定义函数中的变参，例如 `def foo(*args)`。

### 改进

- 当常量索引超出 tile grid 的范围时，`ct.extract()` 现在会抛编译期的 `TileTypeError`。动态索引不受影响。
- `ct.floordiv()` 与 `//` 运算符现在支持浮点操作数。

### 生态

- 通过 `ct.jax.cutile_call()` 支持 JAX。

## 1.3.0（2026-04-20）

### 特性

- 新增提前编译与导出 API：`compilation.export_kernel()`。详见[编译与导出](guide/08-compilation.md)。
- 新增自动调优 API：`tune.exhaustive_search()`，以及以下辅助设施：
  - `kernel.replace_hints()`：得到带更新提示的 kernel。
  - `compiler_timeout()`：临时设置 `tileiras` 编译器的超时。
  详见[性能调优](guide/07-performance.md)。
- 新增 `Array.tiled_view()`：以固定的 tile 形状和 padding 模式创建数组的平铺视图。

### 改进

- 支持在 `cuda.tile.load` 与 `cuda.tile.store` 操作上指定 `memory_order` 和 `memory_scope`。
- 改进 `print()`，支持 tuple 与嵌套 fstring。

### Bug 修复

- 修复「受限浮点 dtype 配简单 reduce 和 scan 时未抛出正确 `TileTypeError`」的 bug。

### ABI 变更

- 把 kernel ABI 约定改为**省略**标注了 `cuda.tile.Constant` 的参数。

## 1.2.0（2026-03-05）

### CTK 13.2 特性

- 支持 Ampere 与 Ada（sm80 系列）GPU。
- 支持 `pip install cuda-tile[tileiras]`，从而在**不**安装系统级 CTK 的情况下使用 Python 环境里的 `tileiras`。
- 新增 `ct.atan2(y, x)`，计算 y/x 的反正切。
- `ct.tanh()` 新增可选的 `rounding_mode` 参数，支持 `RoundingMode.FULL` 与 `RoundingMode.APPROX`。
- 为 sm80 系列 GPU 编译 FP8 操作将抛出 `TileUnsupportedFeatureError`。
- `ct.printf()` 与 `ct.print()` 不再需要把 `ct.kernel` 的 `opt_level` 设为 0。

### 特性

- 新增 `ct.static_iter` 关键字，启用编译期的 `for` 循环。
- 新增 `ct.static_assert` 关键字，用于在编译期断言条件为真。
- 新增 `ct.static_eval` 关键字，用宿主 Python 解释器做编译期求值。
- 新增 `ct.scan()`，用于自定义前缀扫描。
- 新增 `ct.isnan()`。
- 新增 `print()` 与 `ct.print()`，支持 Python 风格的打印和 f-string。
- 给 `ct.gather()` 与 `ct.scatter()` 新增可选的 `mask` 参数，用于自定义布尔掩码。
- `+` 运算符现在可用于拼接 tuple。
- 支持解包嵌套 tuple（例如 `a, (b, c) = t`）以及用方括号解包（例如 `[a, b] = 1, 2`）。
- 新增 bytecode→cubin 磁盘缓存，避免重复编译未改动的 kernel。由 `CUDA_TILE_CACHE_DIR` 与 `CUDA_TILE_CACHE_SIZE` 控制。

### Bug 修复

- 修复 `nan != nan` 返回 False 的 bug。
- 修复「辅助函数在一个没有提前返回的 `while` 循环之后返回」时报 “potentially undefined variable `$retval`” 的问题。
- 修复下划线文本只有一个字符宽时，错误信息里缺少列指示符的问题。
- 补上「解包 tuple 时值过多」的检查。例如 `a, b = 1, 2, 3` 现在会报错，而不是静默丢弃多余的值。
- 修复 uint16 与 uint64 的共同 dtype 被错误设为 uint32 的 bug。

### 改进

- 取消标量与零维 tile 之间的区别 —— 二者现在完全可互换。
- 对常量布尔 `x` 使用 `~x` 会抛 `TypeError`，以避免与布尔 Tile 上的 `~x` 结果不一致。
- 把 `TileUnsupportedFeatureError` 加入公开 API。

## 1.1.0（2026-01-30）

### 特性

- 支持嵌套函数与 lambda。
- 支持通过 `ct.reduce()` 做自定义归约。
- 新增 `Array.slice(axis, start, stop)`：创建沿单个轴切片得到的数组视图。结果与原数组**共享内存**（不复制数据）。

### Bug 修复

- 修复「多个轴以非递增顺序指定」时的归约问题。
- 修复模式匹配（FusedMultiplyAdd）试图删除一个被新操作使用的值时的 bug。

### 改进

- 允许带类型注解的赋值。类型注解会被忽略。
- 支持内置数值类型（bool、int、float）的构造函数，例如 `float('inf')`。
- 解除对递归辅助函数调用的禁令，改为限制递归深度。新增异常类 `TileRecursionError`，在函数调用内联过程中达到递归上限时于编译期抛出。
- 改进控制流语句中类型不匹配的错误信息。
- 放宽「变量在不同分支被赋成不同类型」的类型检查规则：现在只有在之后**使用了**该变量时才算错误。
- 「可能未定义变量」的检测更严格：如果变量首次赋值发生在 `for` 循环内部、之后在循环外被使用，现在会报错 —— 因为循环可能执行零次，导致使用了未定义的变量。
- 错误信息中带上完整的 cuTile traceback。改进代码位置的格式：加入函数名、去掉多余字符以缩短行宽。
- 把 CUDA 驱动的加载推迟到 kernel 启动时。
- 在公开 API 中暴露 `TileError` 基类。
- 为完整性新增 `ct.abs()`。

## 1.0.1（2025-12-18）

### Bug 修复

- 修复哈希函数中的一个 bug —— 它会导致「特化很多」的 kernel 出现性能回退。
- 修复「循环内的 if 语句可能触发编译器内部错误」的 bug。
- 修复 `SliceType.__eq__` 的比较逻辑。

### 改进

- 改进 `ct.cat()` 的错误信息。
- 支持 `is not None` 比较。

## 1.0.0（2025-12-02）

首个正式版本。
