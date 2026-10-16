# 数据模型

cuTile 是一个**基于数组**的编程模型。
最基本的数据结构是多维数组，其元素共享同一个同质（homogeneous）类型。
cuTile Python 只暴露数组，**不暴露指针**。

选择基于数组的模型，原因有：

- 数组知道自己的边界，因此访问可以做安全检查与正确性检查。
- 基于数组的 load/store 操作能被高效地下降（lower）为「光速」硬件机制。
- Python 程序员已经熟悉 NumPy 这类基于数组的框架。
- 指针与 Python 并不天然契合。

在 tile 代码内部，只支持本节描述的类型。

## 全局数组（Global Arrays）

**全局数组**（或简称**数组**）是按逻辑多维空间排列的、具有特定 dtype 的元素的容器。

数组的 **shape** 是一个整数 tuple，每个元素表示对应维度的长度。
shape tuple 的长度等于数组的维数，其各值之积等于数组中逻辑元素的总数。

数组在全局内存中以**带 stride 的内存布局**存放：除了 shape，
每个数组还有一个长度相同的 **strides** tuple，把逻辑索引映射到物理内存位置。
例如一个 3 维 float32 数组，strides 为 (s1, s2, s3)，
则索引 (i1, i2, i3) 处元素的内存地址为：

```
base_addr + 4 * (s1 * i1 + s2 * i2 + s3 * i3),
```

其中 `base_addr` 是数组的基地址，`4` 是一个 `float32` 元素的字节数。

**新数组只能由宿主分配**，并作为参数传给 tile kernel。
tile 代码只能对已有数组创建新的**视图**，例如通过 `Array.slice()`。
和 Python 一样，把一个数组赋给另一个变量**不会复制**底层数据，
只是创建了指向同一个数组的另一个引用。

任何实现了 DLPack 接口或 CUDA Array Interface 的对象都可以作为 kernel 参数传入 ——
例如 CuPy array 和 PyTorch tensor。

如果传给 kernel 的两个或更多数组参数在内存上有重叠，行为是**未定义**的。

数组的 shape 可以用 `Array.shape` 属性查询，它返回一个由 int32 标量组成的 tuple。
这些是**非常量**的运行时值。用 int32 可以提升性能，代价是 shape 可表示的最大值被限制在
2,147,483,647 个元素。这个限制将来会取消。

参见

`cuda.tile.Array` 类文档

## Tile 与标量

**tile** 是特定 dtype 元素的**不可变**多维集合。

tile 的 **shape** 是一个整数 tuple，每个元素表示对应维度的长度。
shape tuple 的长度等于 tile 的维数，其各值之积等于 tile 中元素的总数。

tile 的 shape **必须在编译期已知**。tile 的每个维度都必须是 **2 的幂**。

tile 的 dtype 和 shape 可分别用 `dtype` 和 `shape` 属性查询。
例如 `x` 是 float32 的 tile，则 `x.dtype` 返回编译期常量 `cuda.tile.float32`。

零维的 tile 称为**标量**（scalar）。标量恰好只有一个元素，shape 是空 tuple ()。
像 `7` 或 `3.14` 这样的数值字面量被当作常量标量，即零维 tile。

因为标量就是 tile，它们与 Python 的 `int`/`float` 对象略有不同。
例如它们有 `dtype` 和 `shape` 属性：

```python
a = 0
# 下面这行在 cuTile 里会求值为 cuda.tile.int32，
# 但在 Python 里会抛 AttributeError：
a.dtype
```

tile 只能用在 tile 代码里，不能用在宿主代码里。
tile 的内容**不一定**在内存中有物理表示。
tile 通过 `cuda.tile.load()`、`cuda.tile.gather()` 这类函数从全局数组加载而创建，
也可以用 `cuda.tile.zeros()` 这类工厂函数创建。

tile 可以通过 `cuda.tile.store()`、`cuda.tile.scatter()` 这类函数存回全局数组。

标量常量默认是宽松类型的，例如字面量 `2`，
或 `Tile.ndim`、`Tile.shape`、`Array.ndim` 这类常量属性。

参见

`cuda.tile.Tile` 类文档

## 元素空间与 Tile 空间

数组的**元素空间**（element space）是其元素构成的多维空间，
按给定布局（行主序、列主序等）存放于内存中。

数组的 **tile 空间**是在该数组内、按给定 tile 形状切分出的 tile 所构成的多维空间。
形状为 `S` 的 tile 索引 `(i, j, ...)` 指向第 `(i+1)`、`(j+1)`、…… 个 tile 所含的元素。

通过 tile 索引访问数组元素时，使用的是数组的多维内存布局。
若要按不同布局访问 tile 空间，请使用 load/store 操作的 `order` 参数。

## Tiled View（平铺视图）

**tiled view** 表示某个全局数组的 tile 空间。

tiled view 的 **num_tiles** 是一个整数 tuple，每个元素表示对应维度上的 tile 个数。
*num_tiles* tuple 的长度等于 tile 空间的维数，其各值之积等于 tile 空间中 tile 的总数。

tiled view 中的某个 tile 可以用它的 tile 索引加载或存储。

默认情况下，各轴上的相邻 tile 彼此紧邻、既不重叠也不留空隙：
每个后继 tile 的原点沿轴 *i* 前进 `tile_shape[i]` 个元素。
给 `Array.tiled_view()` 指定 `traversal_steps`，可以把每一步的前进量改为
`traversal_steps[i]` —— 当 `traversal_steps[i] < tile_shape[i]` 时 tile 相互**重叠**；
当 `traversal_steps[i] > tile_shape[i]` 时 tile 之间出现**空隙**。

参见

`cuda.tile.TiledView` 类文档

`Array.tiled_view()`

## 形状广播（Shape Broadcasting）

**形状广播**允许形状不同的 tile 参与同一个算术运算。
当运算涉及形状不同的 tile 时，较小的 tile 会按以下规则自动扩展到与较大的那个匹配：

- tile 按**尾部维度**对齐。
- 如果对应维度的长度相同、或其中之一为 1，则二者兼容。
- 如果某个 tile 的维数较少，则在**左侧**用 1 补齐其形状。

广播遵循与 NumPy 相同的语义，在保持计算效率的同时让代码简洁可读。

## 数据类型

`class cuda.tile.DType`

**数据类型**（或 **dtype**）描述数组、tile 或运算对象的类型。

dtype 决定值如何存于内存，以及对这些值的运算如何执行。
dtype 是**不可变**的。

dtype 可用于宿主代码和 tile 代码，也可以作为 kernel 参数。

`property bitwidth`

该数据类型单个元素的位数。

`property name`

该数据类型的名称。

`cuda.tile.bool_`
8 位算术 dtype（`True` 或 `False`）。

`cuda.tile.uint8`
8 位无符号整数算术 dtype，取值范围 [0, +255]。

`cuda.tile.uint16`
16 位无符号整数算术 dtype，取值范围 [0, +65535]。

`cuda.tile.uint32`
32 位无符号整数算术 dtype，取值范围 [0, +4294967295]。

`cuda.tile.uint64`
64 位无符号整数算术 dtype，取值范围 [0, +18446744073709551615]。

`cuda.tile.int8`
8 位有符号整数算术 dtype，取值范围 [-128, +127]。

`cuda.tile.int16`
16 位有符号整数算术 dtype，取值范围 [-32768, +32767]。

`cuda.tile.int32`
32 位有符号整数算术 dtype，取值范围 [-2147483648, +2147483647]。

`cuda.tile.int64`
64 位有符号整数算术 dtype，取值范围 [-9223372036854775808, +9223372036854775807]。

`cuda.tile.float16`
IEEE 754 半精度（16 位）二进制浮点算术 dtype（见 IEEE 754-2019）。

`cuda.tile.float32`
IEEE 754 单精度（32 位）二进制浮点算术 dtype（见 IEEE 754-2019）。

`cuda.tile.float64`
IEEE 754 双精度（64 位）二进制浮点算术 dtype（见 IEEE 754-2019）。

`cuda.tile.bfloat16`
16 位浮点算术 dtype，1 个符号位、8 个指数位、7 个尾数位。

`cuda.tile.tfloat32`
32 位张量浮点数值 dtype，1 个符号位、8 个指数位、10 个尾数位（19 位表示，存在 32 位容器中）。

`cuda.tile.float8_e4m3fn`
8 位浮点数值 dtype，1 个符号位、4 个指数位、3 个尾数位。

`cuda.tile.float8_e5m2`
8 位浮点数值 dtype，1 个符号位、5 个指数位、2 个尾数位。

`cuda.tile.float8_e8m0fnu`
8 位浮点数值 dtype，**无符号位**、8 个指数位、0 个尾数位。

`cuda.tile.float4_e2m1fn`
4 位浮点数值 dtype，1 个符号位、2 个指数位、1 个尾数位。

## 数值与算术数据类型

**数值**（numeric）数据类型表示数字。**算术**（arithmetic）数据类型是支持加减乘除等
一般算术运算的数值数据类型。

## 算术提升（Arithmetic Promotion）

二元运算可以作用在具有不同数值 dtype 的两个 tile 或标量操作数上。

当两个操作数都是**宽松类型**的数值常量时，结果也是宽松类型常量：
`5 + 7` 是宽松类型的整型常量 12，`5 + 3.0` 是宽松类型的浮点常量 8.0。

如果任一操作数不是宽松类型数值常量，则两者会按下面的规则**提升**到共同的 dtype：

- 每个操作数归入三类之一：*boolean*、*integral* 或 *floating-point*。
  三类的顺序为：*boolean* < *integral* < *floating-point*。
- 如果任一操作数是宽松类型数值常量，先为它选定一个具体 dtype：
  整型常量按取值视为 int32、int64 或 uint64；浮点常量视为 float32。
- 如果两个操作数类别不同，取类别更高的那个的具体 dtype 作为共同 dtype。
- 如果两个操作数类别相同，但其中一个是宽松类型数值常量，
  则取另一个操作数的 dtype 作为共同 dtype。
- 其余情况按下表计算共同 dtype。

| | b1 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | tf32 | f8e4m3fn | f8e5m2 | f8e8m0fnu | f4e2m1fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1 | b1 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| u8 | u8 | u8 | u16 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| u16 | u16 | u16 | u16 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| u32 | u32 | u32 | u32 | u32 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| u64 | u64 | u64 | u64 | u64 | u64 | ERR | ERR | ERR | ERR | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| i8 | i8 | ERR | ERR | ERR | ERR | i8 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| i16 | i16 | ERR | ERR | ERR | ERR | i16 | i16 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| i32 | i32 | ERR | ERR | ERR | ERR | i32 | i32 | i32 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| i64 | i64 | ERR | ERR | ERR | ERR | i64 | i64 | i64 | i64 | f16 | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f16 | f32 | f64 | ERR | ERR | ERR | ERR | ERR | ERR |
| f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f32 | f64 | f32 | ERR | ERR | ERR | ERR | ERR |
| f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 | ERR | ERR | ERR | ERR | ERR |
| bf | bf | bf | bf | bf | bf | bf | bf | bf | bf | ERR | f32 | f64 | bf | ERR | ERR | ERR | ERR | ERR |
| tf32 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | tf32 | ERR | ERR | ERR | ERR |
| f8e4m3fn | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e4m3fn | ERR | ERR | ERR |
| f8e5m2 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e5m2 | ERR | ERR |
| f8e8m0fnu | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f8e8m0fnu | ERR |
| f4e2m1fn | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | f4e2m1fn |

图例：

- `b1`：`bool_`
- `u8`：`uint8`
- `u16`：`uint16`
- `u32`：`uint32`
- `u64`：`uint64`
- `i8`：`int8`
- `i16`：`int16`
- `i32`：`int32`
- `i64`：`int64`
- `f16`：`float16`
- `f32`：`float32`
- `f64`：`float64`
- `bf`：`bfloat16`
- `tf32`：`tfloat32`
- `f8e4m3fn`：`float8_e4m3fn`
- `f8e5m2`：`float8_e5m2`
- `f8e8m0fnu`：`float8_e8m0fnu`
- `f4e2m1fn`：`float4_e2m1fn`
- `ERR`：不支持这些类型之间的隐式提升

## Tuple

Tuple 可以用在 tile 代码中，**不能**作为 kernel 参数。

## 舍入模式（Rounding Modes）

`class cuda.tile.RoundingMode`

浮点运算的舍入模式。

`RN = 'nearest_even'`
舍入到最近（平局取偶）。

`RZ = 'zero'`
向零舍入（截断）。

`RM = 'negative_inf'`
向负无穷舍入。

`RP = 'positive_inf'`
向正无穷舍入。

`RA = 'nearest_away'`
舍入到最近（平局远离零）。

`FULL = 'full'`
全精度舍入模式。

`APPROX = 'approx'`
近似舍入模式。

`RZI = 'nearest_int_to_zero'`
向零舍入到最近的整数。

## Padding 模式

`class cuda.tile.PaddingMode`

load 操作的填充（padding）模式。

`UNDETERMINED = 'undetermined'`
填充值不确定。

`ZERO = 'zero'`
填充值为 0。

`NEG_ZERO = 'neg_zero'`
填充值为负零。

`NAN = 'nan'`
填充值为 NaN。

`POS_INF = 'pos_inf'`
填充值为正无穷。

`NEG_INF = 'neg_inf'`
填充值为负无穷。
