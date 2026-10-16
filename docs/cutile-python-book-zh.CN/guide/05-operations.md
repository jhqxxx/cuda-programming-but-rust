# 操作（Operations）

本页是 `cuda.tile` 下可用操作的**分类索引**，每条给出一句话说明。
每个函数的完整签名、参数与返回值在上游的**逐符号 API 页**里 —— 从本页表格点击函数名
即可跳转（这 105 页本译本尚未翻译，说明见 [README](../README.md)）。
上游入口：<https://docs.nvidia.com/cuda/cutile-python/operations.html>。

## Load / Store

| 函数 | 说明 |
| --- | --- |
| `bid` | 取当前 block 的索引。 |
| `num_blocks` | 取某个轴上 block 的个数。 |
| `num_tiles` | 取数组的 tile 空间在某个轴上 tile 的个数。 |
| `load` | 从被划分成 tile 空间的数组里加载一个 tile。 |
| `store` | 把 tile 值存入数组的某个 tile 空间索引处。 |
| `load_advanced_indexing` | 从数组的**非连续**切片加载一个 tile。 |
| `store_advanced_indexing` | 把一个 tile 存入数组的**非连续**切片。 |
| `gather` | 按给定的索引从数组元素中加载一个 tile。 |
| `scatter` | 按给定的索引把一个 tile 值存入数组元素。 |

## 工厂（Factory）

| 函数 | 说明 |
| --- | --- |
| `arange` | 创建长度为 `size` 的一维 tile，取值为 `start, start + step, ..., start + (size - 1) * step`。 |
| `astile` | 从一个值创建 tile。 |
| `full` | 创建用给定值填满的 tile。 |
| `ones` | 创建全 1 的 tile。 |
| `zeros` | 创建全 0 的 tile。 |

## 形状与 DType

| 函数 | 说明 |
| --- | --- |
| `cat` | 沿指定轴拼接两个 tile。 |
| `broadcast_to` | 按 NumPy 广播规则把 tile 广播到指定形状。 |
| `expand_dims` | 在指定位置插入一个长度为 1 的新轴，从而重塑 tile。 |
| `reshape` | 把 tile 重塑为指定形状。 |
| `permute` | 重排输入 tile 的各轴顺序。 |
| `transpose` | 转置输入 tile 的两个轴（要求至少 2 维）。 |
| `astype` | 把 tile 转换成指定的数据类型。 |
| `bitcast` | 把 tile 重新解释为指定的数据类型。 |
| `pack_to_bytes` | 把 tile 展平，并把其原始字节重新解释为 uint8 元素。 |
| `unpack_from_bytes` | 把一维 uint8 字节 tile 重新解释为指定数据类型的一维 tile。 |

## 归约（Reduction）

| 函数 | 说明 |
| --- | --- |
| `sum` | 沿指定轴对 tile 做求和归约。 |
| `max` | 沿指定轴对 tile 做最大值归约。 |
| `min` | 沿指定轴对 tile 做最小值归约。 |
| `prod` | 沿指定轴对 tile 做乘积归约。 |
| `argmax` | 沿指定轴对 tile 做 argmax 归约。 |
| `argmin` | 沿指定轴对 tile 做 argmin 归约。 |
| `reduce` | 沿指定轴施加自定义的归约函数。 |

## 前缀扫描（Scan）

| 函数 | 说明 |
| --- | --- |
| `cumsum` | 沿指定轴对 tile 做累加和。 |
| `cumprod` | 沿指定轴对 tile 做累乘积。 |
| `scan` | 沿指定轴施加自定义的 scan（含首元素的前缀）函数。 |

## 矩阵乘（Matmul）

| 函数 | 说明 |
| --- | --- |
| `mma` | 矩阵乘法累加（multiply-accumulate）。 |
| `mma_scaled` | 块缩放（block-scaled）的矩阵乘法累加。 |
| `matmul` | 对给定的 tile 做矩阵乘。 |

## 选取（Selection）

| 函数 | 说明 |
| --- | --- |
| `where` | 按 condition 从 x 或 y 中选取元素返回。 |
| `extract` | 从输入 tile 中取出一个更小的 tile。 |
| `insert` | 把一个更小的 tile 存入更大的 tile。 |

## 数学（Math）

| 函数 | 说明 |
| --- | --- |
| `add` | 两个 tile 逐元素相加。 |
| `sub` | 两个 tile 逐元素相减。 |
| `mul` | 两个 tile 逐元素相乘。 |
| `truediv` | 两个 tile 逐元素真除（true division）。 |
| `floordiv` | 两个 tile 逐元素地板除。 |
| `cdiv` | 计算 ceil(x / y)。 |
| `pow` | 两个 tile 逐元素求幂。 |
| `atan2` | 两个 tile 逐元素求 atan2。 |
| `mod` | 两个 tile 逐元素求模。 |
| `divmod` | 两个 tile 逐元素计算 `divmod()`。 |
| `minimum` | 两个 tile 逐元素取较小值。 |
| `maximum` | 两个 tile 逐元素取较大值。 |
| `negative` | 等价于 `-x`。 |
| `abs` | 对 tile 求绝对值。 |
| `isnan` | 对 tile 判断是否为 NaN。 |
| `exp` | 对 tile 求 exp。 |
| `exp2` | 对 tile 求 exp2。 |
| `log` | 对 tile 求 log。 |
| `log2` | 对 tile 求 log2。 |
| `sqrt` | 对 tile 求 sqrt。 |
| `rsqrt` | 对 tile 求 rsqrt（平方根的倒数）。 |
| `sin` | 对 tile 求 sin。 |
| `cos` | 对 tile 求 cos。 |
| `tan` | 对 tile 求 tan。 |
| `sinh` | 对 tile 求 sinh。 |
| `cosh` | 对 tile 求 cosh。 |
| `tanh` | 对 tile 求 tanh。 |
| `floor` | 对 tile 向下取整。 |
| `ceil` | 对 tile 向上取整。 |

## 位运算（Bitwise）

| 函数 | 说明 |
| --- | --- |
| `bitwise_and` | 两个 tile 逐元素按位与。 |
| `bitwise_or` | 两个 tile 逐元素按位或。 |
| `bitwise_xor` | 两个 tile 逐元素按位异或。 |
| `bitwise_lshift` | 两个 tile 逐元素左移。 |
| `bitwise_rshift` | 两个 tile 逐元素右移。 |
| `bitwise_not` | 对 tile 逐元素按位取反。 |

## 比较（Comparison）

| 函数 | 说明 |
| --- | --- |
| `greater` | 两个 tile 逐元素比较 `>`。 |
| `greater_equal` | 两个 tile 逐元素比较 `>=`。 |
| `less` | 两个 tile 逐元素比较 `<`。 |
| `less_equal` | 两个 tile 逐元素比较 `<=`。 |
| `equal` | 两个 tile 逐元素比较 `==`。 |
| `not_equal` | 两个 tile 逐元素比较 `!=`。 |

## 原子操作（Atomic）

| 函数 | 说明 |
| --- | --- |
| `atomic_cas` | 对给定索引处的数组元素做**批量**原子比较并交换。 |
| `atomic_xchg` | 对给定索引处的数组元素做批量原子交换。 |
| `atomic_add` | 对给定索引处的数组元素做批量原子自增（后置加）。 |
| `atomic_max` | 对给定索引处的数组元素做批量原子取最大值赋值。 |
| `atomic_min` | 对给定索引处的数组元素做批量原子取最小值赋值。 |
| `atomic_and` | 对给定索引处的数组元素做批量原子 AND。 |
| `atomic_or` | 对给定索引处的数组元素做批量原子 OR。 |
| `atomic_xor` | 对给定索引处的数组元素做批量原子 XOR。 |

## 工具（Utility）

| 函数 | 说明 |
| --- | --- |
| `printf` | 在运行时从设备端打印值。 |
| `print` | 用 Python 风格的语法在运行时从设备端打印值。 |
| `assert_` | 断言给定 tile 的所有元素都为 True。 |
| `assume_divisible_by` | 声明 `x` 能被 `divisor` 整除。 |
| `grid_dependency_control_launch_dependents` | 允许程序化依赖（PDL）的后继 kernel 开始调度。 |
| `grid_dependency_control_wait` | 等待程序化依赖启动中前一个 kernel 完成。 |

## 元编程支持（Metaprogramming Support）

| 函数 | 说明 |
| --- | --- |
| `ensure_constant` | 断言参数是编译期常量并返回它。 |
| `static_assert` | 在编译期断言某个条件为真。 |
| `static_eval` | 在编译期求值给定的 Python 表达式。 |
| `static_iter` | 在编译期迭代。 |

## 类（Classes）

| 名称 | 说明 |
| --- | --- |
| `Array` | 全局数组对象对应的类。 |
| `TiledView` | 平铺视图（tiled view）对象对应的类。 |
| `Slice` | 数组维度上的一个「起点 + 长度」索引。 |

## 枚举（Enums）

| 名称 | 说明 |
| --- | --- |
| `RoundingMode` | 浮点运算的舍入模式。 |
| `PaddingMode` | load 操作的填充模式。 |

## 自动调优（Autotuning）

| 名称 | 说明 |
| --- | --- |
| `tune.exhaustive_search` | 搜索整个搜索空间并返回最优配置。 |
| `tune.TuningResult` | 保存每个配置的测量结果。 |
| `tune.Measurement` | 保存一个配置及其计时结果。 |
| `kernel.replace_hints` | 返回一个带更新后编译器提示（hints）的新 kernel。 |
| `compiler_timeout` | 临时设置编译器超时的上下文管理器。 |

## JAX FFI

| 名称 | 说明 |
| --- | --- |
| `jax.cutile_call` | 从 JAX 追踪出的图中启动一个 cuTile kernel。 |
| `jax.OutputPlaceholder` | 表示传给 `cutile_call` 的一个输出 buffer。 |
| `jax.InputOutput` | 包装一个输入 buffer，使其与 `cutile_call` 要返回的输出 buffer 互为别名。 |
