# 执行模型

## 抽象机

一个 **tile kernel** 由逻辑上的线程块（thread block）执行，这些块组织成 1 维、2 维或 3 维的 **grid**。

每个 **block** 运行在 GPU 的一个子集上，具体由底层编译器的实现决定。
每个 block 执行 kernel 的函数体：
标量操作在单个线程上串行执行，而数组操作则在 block 的所有线程上协作并行执行。

tile 程序只表达 **block 级并行**，不暴露 block 内部的单个线程。

**不允许**在 block 内部做显式同步或通信，但允许在不同的 block 之间进行。

block 定义的是「**执行**的单位」，tile 定义的是「**数据**的单位」，两者不要混淆。
一个 block 可以操作多个 tile —— 形状不同、来自不同的全局数组。

## 执行空间（Execution Spaces）

cuTile 代码运行在一个或多个 **target** 上。target 是由其硬件资源和编程模型所定义的一种执行环境。

某个构造（construct）可用的 target 集合称为它的**执行空间**。cuTile 定义了三个执行空间：

- **Host code（宿主代码）** —— 所有 CPU target。
- **SIMT code（SIMT 代码）** —— 所有 CUDA SIMT target。
  （历史上叫 *device code*；为避免歧义，本文不使用那个术语。）
- **Tile code（Tile 代码）** —— 所有 CUDA tile target。

有些构造横跨多个执行空间。例如 `cdiv()` 在宿主代码和 tile 代码里都能用。

装饰器显式指定了执行空间的函数，称为**带注解的函数**（annotated function）。

## Tile 函数

`class cuda.tile.function(func=None, /, *, host=False, tile=True)`

**Tile 函数**是可在 tile 代码中使用的函数。

这个装饰器声明一个函数可以从哪些执行空间调用。不带参数时，表示这是一个**仅 tile** 的函数。

当某个 tile 函数调用了一个未加注解的函数时，`tile` 会被加到那个未注解函数的执行空间上。
这个过程是**递归**的，不需要显式注解。

tile 函数可接受的参数类型见数据模型（data model）。

参数：

- **host**（*bool, 可选*）—— 该函数是否可从宿主代码调用。默认 False。
- **tile**（*bool, 可选*）—— 该函数是否可从 tile 代码调用。默认 True。

## Tile Kernel

`class cuda.tile.kernel(function=None, /, **kwargs)`

**tile kernel** 是由 grid 中每个 block 各执行一次的函数。

带此装饰器的函数就是 kernel。

kernel 是 tile 代码的入口点。它们的执行空间只能是 tile 代码，**不能**从宿主代码调用。

kernel 不能被直接调用。要启动它，请用 `launch()` 把一个 kernel 排入 grid 上执行。

kernel 可接受的参数类型见数据模型。

参数：

- **num_ctas** —— CGA 中 CTA 的个数。必须是 1 到 16（含）之间的 2 的幂。默认：None（自动）。
- **occupancy** —— 期望的每 SM 活跃 CTA 数，范围 [1, 32]。默认：None（自动）。
- **opt_level** —— 优化级别 [0, 3]，默认 3。
- **num_worker_warps** —— warp 专用化（warp-specialized）kernel 中 CUDA core warp 组的 warp 数。
  编译器可能额外加入一些 warp（例如用于异步内存传输），它们**不计入**此值。
  这个值**不代表** warp 总数。
  当某个 warp 专用化 kernel 的寄存器压力很高、别的办法都解决不了时，值得调一调这个值。
  典型场景是「大 tile + 归一化（normalization）」这类 kernel。
  取值必须是 4 或 8。默认：None（自动）。
  自 CTK 13.3 起支持；更早版本会忽略并给出警告。

上面这些编译器选项可以按 target 分别取值，方式是传入一个 `ByTarget` 对象。

`replace_hints(**hints)`

返回一个带更新后编译器提示（hints）的新 kernel。

注意：

```
因为 hints 会影响编译，返回的对象拥有自己的 JIT 缓存。
```

示例：

```python
import cuda.tile as ct
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

@ct.kernel(occupancy=2)
def kernel():
    pass

# 编译
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
# 命中缓存
ct.launch(torch.cuda.current_stream(), (1,), kernel, ())

new_kernel = kernel.replace_hints(occupancy=4)

# 用新 hints 编译
ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())
# 命中缓存
ct.launch(torch.cuda.current_stream(), (1,), new_kernel, ())

torch.cuda.synchronize()
```

`cuda.tile.launch(stream, grid, kernel, kernel_args, /, *, programmatic_dependent_launch=False)`

启动一个 cuTile kernel。

参数：

- **stream** —— 执行该 kernel 所用的 CUDA stream。
- **grid** —— 最多 3 个维度的 tuple，表示 kernel 执行的 grid 尺寸。
- **kernel** —— 要执行的 kernel。
- **kernel_args** —— 传给该 kernel 的位置参数。

## Python 子集

Tile 代码支持 Python 语言的一个**子集**。tile 代码内部**没有 Python 运行时**。

只有本文档明确列出的 Python 特性才受支持。很多特性 —— 例如异常、协程 —— 目前还不支持。

### 对象模型与生命周期

在 tile 代码中创建的所有对象都是**不可变**的。
任何概念上会修改对象的操作，实际都会创建并返回一个新对象。
不能给对象动态添加属性。

全局数组是「能读写全局设备内存」的**视图**，但视图本身也是不可变的。

kernel 的调用者必须保证：

- 传给 kernel 的数组之间**不存在别名**（alias）关系。
- 所有数组在 kernel 执行结束前保持有效。

### 控制流

Python 的控制流语句（`if`、`for`、`while` 等）在 tile 代码中可用，且可以任意嵌套。

#### 目前的限制

Tile 代码对控制流有额外限制：

- `step` 必须严格为正。
  目前不支持负步长，例如 `range(10, 0, -1)`。
  通过变量间接传入负步长可能导致未定义行为。

## Tile 并行

当一个 block 执行某个以 tile 为参数的函数时，它可以把求值过程并行化到该 block 的执行资源上。

## 常量性（Constantness）

### 常量表达式与常量对象

有些设施要求参数在**编译期**就有已知的值。
**常量表达式**（constant expression）会产出适合这类参数的**常量对象**（constant object）。
常量表达式包括：

- 字面量对象。
- 所有操作数都是字面量对象的整数算术表达式。
- 由字面量对象或常量表达式赋值的局部对象或参数。
- 在编译或启动时已定义的全局对象。

默认情况下，数值常量是**宽松类型**（loosely typed）的：整数常量具有无限精度，
浮点常量以 IEEE 754 双精度存储 —— 直到被用在需要特定宽度类型的上下文中。

**严格类型**（strictly typed）的常量通过调用 dtype 构造函数创建，例如 `ct.int16(5)`。
把严格类型常量与宽松类型常量组合，得到严格类型常量：
`ct.int16(5) + 2` 会产出严格类型的 `int16` 常量 7。

两个严格类型常量组合也得到严格类型常量，遵循常规的类型提升规则。
例如 `ct.int16(5) + ct.int32(7)` 会产出严格类型的 `int32` 常量 12。

### 常量内嵌（Constant Embedding）

如果某个 kernel 参数是**常量内嵌**的，那么：

- 该参数的每一次使用都等价于被替换成它的字面量值。
- 参数的**每一个不同取值**都会生成一份不同的机器表示。
  注意：即使开启了 JIT 缓存，kernel 也是「每个不同取值编译一次」。

## 类型注解

kernel 参数的类型注解使用 `typing.Annotated` 的元数据来控制：
常量内嵌、数组的形状与索引元数据、标量参数的整数 dtype，以及列表元素类型。

### 常量注解

```python
import cuda.tile as ct
```

```python
def needs_constant(x: ct.Constant):
    pass


def needs_constant_int(x: ct.Constant[int]):
    pass
```

`class cuda.tile.ConstantAnnotation`

一个 `typing.Annotated` 元数据类，表示某个对象应当被**常量内嵌**。

如果把此类的对象作为元数据参数传给参数上的 `typing.Annotated` 类型提示，
那么该参数将被常量内嵌。

`cuda.tile.Constant`

一个类型提示，表示某个值应当被常量内嵌。
它既可以带底层类型提示使用（`Constant[int]`），也可以不带（`Constant`，表示任意类型的常量）。

是 `Annotated[T, ConstantAnnotation()]` 的别名。

### 数组注解

`class cuda.tile.ArrayAnnotation(*, index_dtype=<DType 'int32'>, static_shape_dims=(), static_stride_dims=())`

用于数组参数的 `typing.Annotated` 元数据类。

`index_dtype`

shape 和 stride 取值所用的数据类型。必须是 `int32` 或 `int64`。
用 `int64` 可以支持形状或 stride 超出 32 位整数范围的张量。

类型：`cuda.tile._datatype.DType`

`static_shape_dims`

要**特化（specialize）**为启动时取值的形状维度，使其在 kernel 内部成为编译期常量。

类型：`tuple[int, …]`

`static_stride_dims`

要特化为启动时取值的 stride 维度，使其在 kernel 内部成为编译期常量。
这是该数组**静态 stride 的唯一来源**：一旦列出了任一维度，调度器就不再为该数组的
*任何*维度推断 `stride == 1`。因此如果你想让连续维也是编译期常量，需要把它显式列出来。

类型：`tuple[int, …]`

示例：

```python
@ct.kernel
def my_kernel(
    x: Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(0,))],
    y: Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(0, -1))],
    z: Annotated[ct.Array, ct.ArrayAnnotation(static_stride_dims=(0, -1))],
):
    # x.shape[0] 是编译期常量
    # y.shape[0] 和 y.shape[-1] 是编译期常量
    # z.strides[0] 和 z.strides[-1] 是编译期常量
    ...

# 可复用的别名：int64 索引 + 一个静态维度
BigStaticDim0 = Annotated[
    ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=(0,))
]
```

`cuda.tile.IndexedWithInt64`

一个类型提示，表示某个数组的 shape 和 stride 使用 i64。

示例：

```python
@ct.kernel
def my_kernel(big: ct.IndexedWithInt64, small):
    # big.shape[i] 是 i64，small.shape[i] 是 i32
    ...
```

是 `Annotated[T, ArrayAnnotation(index_dtype=int64, static_shape_dims=(), static_stride_dims=())]` 的别名。

### 标量注解

`cuda.tile.ScalarInt64`

一个类型提示，表示某个标量整数参数使用 int64。

默认情况下，整数 kernel 参数被推断为 int32。用这个注解可以强制把需要更宽范围的参数
推断为 int64。

示例：

```python
@ct.kernel
def my_kernel(small_int, large_int: ct.ScalarInt64):
    # small_int 推断为 int32
    # large_int 推断为 int64
    ...
```

是 `Annotated[T, ScalarAnnotation(dtype=int64)]` 的别名。

### 列表注解

`class cuda.tile.ListAnnotation(*, element)`

用于列表参数的 `typing.Annotated` 元数据类。

`element`

列表元素类型的注解。必须是 `ArrayAnnotation`，
或者是元数据中含有 `ArrayAnnotation` 的 `Annotated` 类型（例如 `IndexedWithInt64`）。

类型：`Any`
