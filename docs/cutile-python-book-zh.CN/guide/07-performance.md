# 性能调优

cuTile 提供几种性能调优手段：

- 按 GPU 架构分别取值的配置，用 `ByTarget`；
- load/store 的性能提示，例如 `latency` 和 `allow_tma`；
- 整除性提示，用 `assume_divisible_by()`。

## 按架构区分配置

`class cuda.tile.ByTarget(*, default=UNSPECIFIED, **value_by_target)`

用于指定「依赖目标 GPU 架构」的取值的类型。

参数：

- **default** —— 当目标 GPU 架构没有显式列在 `value_by_target` 里时使用的兜底值。
- **value_by_target** —— 从 GPU 架构名到取值的映射。键必须是形如 `"sm_<major><minor>"` 的字符串，
  例如 `"sm_100"` 或 `"sm_120"`。

示例

所有架构都用同一个 `num_ctas`：

```python
import cuda.tile as ct
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

from cuda.tile import ByTarget

@ct.kernel(num_ctas=8)
def kernel_fn(x):
    ...

torch.cuda.synchronize()
```

给特定架构用不同的 `num_ctas`，其余架构用兜底值：

```python
import cuda.tile as ct
import torch

torch.cuda.init()
stream = torch.cuda.current_stream()

from cuda.tile import ByTarget

@ct.kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def kernel_fn(x):
    ...

torch.cuda.synchronize()
```

`num_ctas`、`occupancy`、`opt_level` 这些 kernel 配置参数的完整说明见
[Tile Kernel](02-execution.md) 一节。这些选项**都可以**传 `ByTarget` 值，
以便针对不同 GPU 架构分别特化。

## load/store 的性能提示

`load()` 和 `store()` 接受若干可选关键字参数，它们会影响显存流量如何被调度和下降：

- `latency`（`int` 或 `None`）—— 提示该操作的 DRAM 流量有多重。
  取值是 1（低）到 10（高）之间的整数。DRAM 流量大的场景通常适合给较大的值，
  这会倾向于让该内存操作采用更大的预取深度。
- `allow_tma`（`bool` 或 `None`）—— 为 `True` 时，若目标架构支持，
  该 load/store 可以被下降为使用 TMA（Tensor Memory Accelerator）。
  为 `False` 时该操作不使用 TMA。默认为允许 TMA。

这些提示是可选的：不指定也能编译运行，但对特定的访存模式给出提示，
能帮助编译器做出更好的代码生成决策。

### 示例

```python
import cuda.tile as ct

TILE_SIZE = 16


@ct.kernel
def load_store_with_hints_kernel(x, y):
    bid = ct.bid(0)
    tx = ct.load(
        x,
        index=(bid,),
        shape=(TILE_SIZE,),
        latency=8,        # 高延迟的 DRAM 读
    )
    ct.store(
        y,
        index=(bid,),
        tile=tx,
        latency=2,        # 写更便宜
        allow_tma=False,  # 禁用 TMA
    )
```

## 整除性提示

`assume_divisible_by()` 是一个编译器提示，声明某个整数标量能被一个编译期常量整除。
运行时**不做任何检查**：

```python
n = ct.assume_divisible_by(n, 16)
```

编译器会把整除性元数据在算术运算中传播下去 —— 于是派生出来的索引和指针偏移
也继承这个事实。当某个运行时标量被用来计算动态数组切片时，这一点最重要：

```python
@ct.kernel
def kernel(x, dim_offset: int, dim_size: int):
    dim_offset = ct.assume_divisible_by(dim_offset, 16)
    dim_size   = ct.assume_divisible_by(dim_size,   16)
    start = ct.bid(0) * dim_offset
    sub_x = x.slice(axis=0, start=start, stop=start + dim_size)
    tile  = ct.load(sub_x, index=(0,), shape=(128,))
    ct.store(sub_x, index=(0,), tile=tile)
```

没有这些提示时，编译器把 `dim_offset` 和 `dim_size` 视为完全未知，
无法为派生出的视图证明对齐。有了提示，它就能一路推断到视图的基地址和形状，
从而启用更宽的内存操作。

这个提示是**程序员的声明，不是强制约束**。如果运行时 `x` 实际上并不能被 `divisor` 整除，
行为是未定义的。

## 自动调优（Autotuning）

`tune.exhaustive_search()` 提供了一种便捷方式：在一组有限的配置上测量 kernel 性能，
并返回最优配置。

`cuda.tile.tune.exhaustive_search(search_space, stream, grid_fn, kernel, args_fn, hints_fn=None, *, quiet=False, single_run_timeout_sec=None)`

搜索整个搜索空间并返回最优配置。

参数：

- **search_space**（*Sequence[T]*）—— 待评估的配置序列。
- **stream** —— 执行 kernel 所用的 CUDA stream。
- **grid_fn**（*Callable[[T], tuple[int, ...]]*）—— 把配置映射到 grid 维度。
- **kernel**（*kernel | Callable[[T], kernel]*）—— 要调优的 kernel，
  或者一个把配置映射到「待调优 kernel」的函数。
  使用函数时，若多个配置映射到同一个 kernel，请返回**同一个 kernel 对象**以获得最佳性能。
- **args_fn**（*Callable[[T], tuple[Any, ...]]*）—— 把配置映射到用于计时的 kernel 参数。
- **hints_fn**（*Callable[[T], dict[str, Any]] | None*）—— 把配置映射到编译器提示。默认：无提示。
- **quiet**（*bool*）—— 为 true 时不打印任何进度或结果。
- **single_run_timeout_sec**（*float | None*）—— 每次 kernel 启动的墙钟超时（秒），
  通过在子进程中运行基准测试来实施。为 None（默认）时禁用超时，
  kernel 直接在本地运行、不启动子进程。

返回：`TuningResult`，包含最优配置及其耗时（微秒）。

返回类型：`TuningResult[T]`

示例：

```python
# 定义 kernel

@ct.kernel
def matmul(X, Y, Out,
           tm: ct.Constant[int],
           tn: ct.Constant[int],
           tk: ct.Constant[int]):

    i, j =  ct.bid(0), ct.bid(1)

    x_view = X.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
    y_view = Y.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((tm, tn), ct.float32)
    for k in range(x_view.num_tiles(1)):
        tx = x_view.load((i, k))
        ty = y_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    ct.store(Out, (i, j), acc.astype(Out.dtype))

# 调优这个 kernel

from itertools import product
from cuda.tile import ByTarget

def tune(x, y, out) -> ct.tune.TuningResult:
    keys = ("tm", "tn", "tk", "num_ctas")
    search_space = [dict(zip(keys, vals))
                    for vals in product(
                    (64, 128),
                    (64, 128),
                    (32, 64),
                    (1, 2))]
    grid = lambda cfg: (ct.cdiv(M, cfg['tm']), ct.cdiv(N, cfg['tn']))
    args = lambda cfg: (x, y, out.clone(), cfg['tm'], cfg['tn'], cfg['tk'])
    hints = lambda cfg: {'num_ctas': ByTarget(sm_100=cfg['num_ctas'])}
    stream = torch.cuda.current_stream()
    tuning_result = ct.tune.exhaustive_search(search_space,
                                              stream,
                                              grid,
                                              matmul,
                                              args,
                                              hints)
    return tuning_result

M, N, K = 1024, 256, 512
x = torch.rand((M, K), dtype=torch.float16, device='cuda:0')
y = torch.rand((K, N), dtype=torch.float16, device='cuda:0')
out = torch.zeros((M, N), dtype=torch.float16, device='cuda:0')

result = tune(x, y, out)
print(f"Best config: {result.best.config} ({result.best.mean_us:.1f}us)")

# 用调优结果启动 kernel

tm, tn, tk, num_ctas = result.best.config.values()
kernel = matmul.replace_hints(num_ctas=ByTarget(sm_100=num_ctas))
ct.launch(torch.cuda.current_stream(),
          (ct.cdiv(M, tm), ct.cdiv(N, tn)),
          kernel,
          (x, y, out, tm, tn, tk))

torch.testing.assert_close(out, x @ y)
```
