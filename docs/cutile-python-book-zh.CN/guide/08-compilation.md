# 编译与导出

当一个用 `@ct.kernel` 标记的 kernel 函数通过 `ct.launch()` 启动时，
它会针对具体的启动参数**即时（JIT）特化并编译**。
此外也可以**提前（AOT）编译**一个 kernel，并把它导出为 CUDA 二进制（cubin）文件，
或导出为 TileIR bytecode 文件。

即时编译提供了「自动特化 kernel」的便利；而提前编译则要求用户**精确描述**
kernel 是为何种参数而编译的 —— 包括参数的类型，以及对参数取值施加的额外约束（假设）。

提前编译的主要 API 入口是 `cuda.tile.compilation.export_kernel()`：

`cuda.tile.compilation.export_kernel(kernel, signatures, output_file, *, gpu_code=None, output_format, bytecode_version=None)`

编译并导出 kernel。

参数：

- **kernel**（*cuda.tile.kernel*）—— 要导出的 kernel 函数。
- **signatures**（*Sequence[cuda.tile.compilation.KernelSignature]*）——
  用于编译该 kernel 的签名列表，不能为空。
- **output_file**（*IO | str | bytes | os.PathLike*）——
  输出文件名，或一个可写的二进制文件类对象。若要把结果留在内存中，
  可以传一个标准库 `io.BytesIO` 的实例。
- **gpu_code**（*str | None*）—— 为该 kernel 编译的目标 GPU 名称（例如 `"sm_100"`）。
  当 `output_format` 为 `"cubin"` 时**必填**。
  当 `output_format` 为 `"tileir_bytecode"` 时可以设为 None，
  以导出与架构无关的 bytecode（要求 bytecode 版本为 `"13.3"` 或更高）。
- **output_format**（*str*）—— 设为 `"cubin"` 导出 CUDA 二进制文件，
  设为 `"tileir_bytecode"` 导出 TileIR bytecode 文件。
- **bytecode_version**（*str | None*）—— 设为 None 表示自动探测编译器支持的最新
  TileIR bytecode 版本（默认）。否则必须是形如 `"major.minor"` 的字符串，
  指定要使用的 TileIR bytecode 版本（例如 `"13.1"`）。

## Kernel 签名

构造 `KernelSignature` 对象有两种方式。

**推荐**的方式是显式构造：实例化 `KernelSignature`，并提供一个由手工构造的
`ParameterConstraint` 对象组成的列表。

另一种方式是 `KernelSignature.from_kernel_args()`，它给出的是「若 kernel 针对给定的
示例参数做即时编译，会用到的签名」。这种做法虽然方便，但可能对 kernel 参数产生**意料之外
的假设**。例如，如果某个示例数组参数的基地址恰好能被 16 整除，就可能产生「它永远如此」
的假设 —— 之后用不满足该假设的数组去启动导出的 kernel，行为就会是未定义的。
因此建议**只把这种方式用于测试或原型验证**。

`class cuda.tile.compilation.KernelSignature(parameters, calling_convention, symbol=None)`

已编译 kernel 的签名。

参数：

- **parameters**（*Sequence[ParameterConstraint | ConstantValue | tuple]*）——
  对 kernel 的 Python 函数中的每个参数，给出一个对应的 `ParameterConstraint` 实例。

  可用的约束类有：`ScalarConstraint`、`ArrayConstraint`、`ListConstraint`、
  `TupleConstraint`、`ConstantConstraint`、`DataclassConstraint`。

  一个裸的 `ConstantValue`（例如字面量 `10`）可以当作「包装该值的 `ConstantConstraint`」
  的简写；同样，一个裸的 `tuple` 是 `TupleConstraint` 的简写。

  每个约束都必须与对应 kernel 参数上的注解兼容。例如，若某参数标记为 `ct.Constant`，
  则对应约束必须是 `ConstantConstraint`，或嵌套其中的 `TupleConstraint`
  （或按上面的简写规则给一个裸值）。

- **calling_convention**（*CallingConvention*）—— 使用的调用约定。
- **symbol**（*str | None*）—— 导出 kernel 时使用的符号名。
  设为 None 时会根据 Python 函数名和本签名自动生成，
  使用所选调用约定定义的 name mangling 算法。

`with_mangled_symbol(function_name)`

返回自身的一个副本，其 `symbol` 属性被替换为 mangled 后的名字。

参数：**function_name**（*str*）—— 用作 mangled 符号名基础的函数名。

返回：`KernelSignature`；返回类型：*KernelSignature*

`with_symbol(symbol)`

返回自身的一个副本，其 `symbol` 属性被替换为给定值。

参数：**symbol**（*str | None*）—— 新的符号名。

返回：`KernelSignature`；返回类型：*KernelSignature*

`static from_kernel_args(kernel, kernel_args, calling_convention, *, symbol=None)`

返回「若 kernel 针对给定参数做即时编译，会用到的签名」。

> **警告**
>
> 建议只把本函数用于测试或原型验证。从示例参数推导 kernel 签名可能对 kernel 参数
> 产生意料之外的假设。
>
> 例如，如果某个示例数组参数的基地址恰好能被 16 整除，就可能产生「它永远如此」的假设。
> 之后用不满足该假设的数组去启动导出的 kernel，行为就会是未定义的。

参数：

- **kernel**（*cuda.tile.kernel*）—— 用 `@ct.kernel` 装饰的 kernel 函数。
- **kernel_args**（*Sequence[Any]*）—— kernel 参数的 tuple，就当作要传给 `ct.launch()`。
- **calling_convention**（*CallingConvention*）—— 使用的调用约定。
- **symbol**（*str | None*）—— 指定返回签名的 `symbol` 属性。
  设为 None 时会用所选调用约定定义的 name mangling 算法自动填充。

返回：`KernelSignature`；返回类型：*KernelSignature*

`ParameterConstraint` 这个类型别名用作 kernel 参数约束的类型提示：

`cuda.tile.compilation.ParameterConstraint`

是 `ScalarConstraint | ArrayConstraint | ListConstraint | TupleConstraint | DataclassConstraint | ConstantConstraint` 的别名。

`class cuda.tile.compilation.ScalarConstraint(dtype)`

描述一个标量 kernel 参数及其相关的编译期假设。

参数：**dtype**（*DType*）—— 该标量的数据类型。

`class cuda.tile.compilation.ArrayConstraint(dtype, ndim, *, index_dtype, stride_lower_bound_incl, alias_groups, may_alias_internally, stride_constant=None, shape_constant=None, stride_divisible_by=1, shape_divisible_by=1, base_addr_divisible_by=1)`

描述一个数组 kernel 参数及其相关的编译期假设。

参数：

- **dtype**（*DType*）—— 该数组的数据类型。
- **ndim**（*int*）—— 该数组的维数，即 rank。
- **index_dtype**（*DType*）—— 用于表示数组 shape、strides 和索引的数据类型。
  支持 `ct.int32` 和 `ct.int64`。用 `int64` 可以支持 shape 或 stride 超出
  32 位整数范围的数组。
- **stride_lower_bound_incl**（*Sequence[int | None] | int | None*）——
  对数组的每个维度，可选地给出其 stride 的**闭区间下界**。
  如果所有维度共用同一个下界，可以只传一个数而不传序列。例如传 `0` 表示所有 stride 非负。
- **alias_groups**（*Sequence[str]*）—— 设为空序列时，表示该数组不得与任何其它参数互为别名。
  否则必须是一个由任意字符串组成的序列，称为「别名组」。
  两个参数互为别名当且仅当它们共享至少一个别名组。
- **may_alias_internally**（*bool*）—— 表示是否允许两个不同的「界内索引」指向同一内存位置
  （例如数组存在 0 步长时就会如此）。对主流张量库产生的大多数数组，
  这一项可以假定为 false。设为 True 可能禁用针对该数组的某些 load/store 优化。
- **stride_constant**（*Sequence[int | None] | None*）—— 对数组的每个维度，
  可选地给出其 stride 的常量值。例如若已知数组是 C 连续（C-contiguous）布局，
  可以把最后一维的 stride 设为 1，这可能启用针对该数组的某些 load/store 优化。
  若没有任何维度有已知 stride，可设为 None（默认值）。
- **shape_constant**（*Sequence[int | None] | None*）—— 对数组的每个维度，
  可选地给出其 shape 的常量值。若没有任何维度有已知 shape，可设为 None。
  需要 `cutile_python_v2`。
  若设置了 `shape_constant[i]`，则 `shape_divisible_by[i]` 是多余的，
  且必须彼此兼容（即 `shape_constant[i]` 必须能被 `shape_divisible_by[i]` 整除）；
  此时后者被忽略。
- **stride_divisible_by**（*Sequence[int] | int*）—— 对数组的每个维度，
  假定其 stride 能被某个因子整除。取值以**数组元素**为单位，而不是字节。
  例如对 float16 数组取值 8，表示能被 16 字节整除（每个元素 2 字节）。
  取值 1 表示不对 stride 的整除性做任何假设（默认值）。
- **shape_divisible_by**（*Sequence[int] | int*）—— 对数组的每个维度，
  假定其长度能被某个因子整除。取值以数组元素为单位，而不是字节。
  例如对 float16 数组取值 8，表示能被 16 字节整除。
  取值 1 表示不对 shape 的整除性做任何假设（默认值）。
- **base_addr_divisible_by**（*int*）—— 假定数组基地址能被某个因子整除。
  取值 1 表示不对基地址的整除性做任何假设（默认值）。

`class cuda.tile.compilation.ListConstraint(element, *, alias_groups, elements_may_alias)`

描述一个列表 kernel 参数及其相关的编译期假设。

参数：

- **element**（*ArrayConstraint*）—— 描述该列表的元素。
  目前必须是 `ArrayConstraint`，因为只支持「数组的列表」作为 kernel 参数。
- **alias_groups**（*Sequence[str]*）—— 描述这个列表本身的存储允许与哪些其它参数互为别名。
  注意这与 `element.alias_groups` 不同，后者是对列表**元素**设置别名假设。
  设为空序列时，表示该列表不得与任何其它参数互为别名。否则必须是由任意字符串组成的序列，
  称为「别名组」。两个参数互为别名当且仅当它们共享至少一个别名组。
- **elements_may_alias**（*bool*）—— 表示该列表中两个不同的元素是否允许互为别名。

`class cuda.tile.compilation.TupleConstraint(items)`

描述一个 tuple kernel 参数。

参数：**items**（*tuple[ScalarConstraint | ArrayConstraint | ListConstraint | TupleConstraint | DataclassConstraint | ConstantConstraint, ...]*）—— 逐项约束。

`class cuda.tile.compilation.ConstantConstraint(value)`

指定某个标记为 `ct.Constant` 的 kernel 参数的常量值。

参数：**value**（*ConstantValue*）—— 该编译期常量的取值。

`cuda.tile.compilation.ConstantValue`

是 `bool | int | float` 的别名。

## 调用约定（Calling Conventions）

调用约定定义导出 kernel 所提供的二进制接口的三个方面：

- kernel 参数的二进制格式与顺序，例如传给 CUDA Driver API 函数 `cuLaunchKernel()` 的形式。
- 所支持的参数约束集合。
- 用于从 kernel 的函数名和签名自动推导符号名的 name mangling 算法。

目前有两种调用约定：`cutile_python_v1` 与 `cutile_python_v2`。
两者都按 Python kernel 函数中声明参数的顺序传递二进制 kernel 参数，
唯一的例外是标注了 `ct.Constant` 的参数会被省略。
`cutile_python_v2` 在 `cutile_python_v1` 的基础上增加了对 tuple 参数（`TupleConstraint`）
和静态形状（`ArrayConstraint` 的 `shape_constant` 字段）的支持。
**新代码应使用 `cutile_python_v2`。**

参数约束及其二进制格式：

| 约束类 | 参数的二进制格式 |
| --- | --- |
| `ScalarConstraint` | 作为对应类型的单个参数传递。例如约束 dtype 为 `ct.int32` 时，对应的 C 类型是 `int32_t`；`ct.float64` 对应 C 的 `double`，依此类推。 |
| `ArrayConstraint` | 传递为 `1 + 2n` 个参数，其中 n 是数组的维数（ndim）。第一个参数是指向数组数据起始处的设备指针；随后 n 个参数表示数组的 shape；最后 n 个参数表示数组的 strides。shape 与 stride 参数的类型由约束的 `index_dtype` 字段决定：`ct.int32` → `int32_t`，`ct.uint32` → `uint32_t`，`ct.int64` → `int64_t`。`shape_constant` 字段（v2）需要 `cutile_python_v2`。 |
| `ListConstraint`（元素为 `ArrayConstraint`） | 传递为两个参数：一个指向列表数据起始处的设备指针，以及一个表示列表长度的 `int32_t`。该基指针必须指向 GPU 全局内存中 8 字节对齐的连续缓冲区，由 `(1 + 2n) * L` 个 64 位字组成，其中 L 是列表长度、n 是元素数组约束的 ndim。列表中的每个元素数组在该缓冲区里由 `(1 + 2n)` 个字表示：第一个字存放指向数组起始处的设备指针；接下来 n 个有符号整数存放 shape；最后 n 个有符号整数存放 strides。尽管 shape 和 strides 一律使用 64 位整数，它们会被截断到元素约束的 `index_dtype`。 |
| `TupleConstraint`（v2） | tuple 的各元素作为连续参数传递，就像每个元素都是一个独立的顶层参数。每个元素的二进制格式遵循其对应约束类型（`ScalarConstraint` 或 `ArrayConstraint`）的规则。当该 tuple 参数被标注为 `ct.Constant[tuple]` 时，整个 tuple 会从启动参数中省略（与 `ConstantConstraint` 相同）；此时 `TupleConstraint` 的所有元素都必须是 `ConstantConstraint`。 |
| `ConstantConstraint` | 从启动参数中省略。 |

调用约定由 `CallingConvention` 类表示：

`class cuda.tile.compilation.CallingConvention`

`static cutile_python_v1()`

返回 `cutile_python_v1` 调用约定。

`static cutile_python_v2()`

返回 `cutile_python_v2` 调用约定。

`static cutile_python_v3()`

返回 `cutile_python_v3` 调用约定。
