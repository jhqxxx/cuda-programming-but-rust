# 调试

## 异常类型

| 名称 | 说明 |
| --- | --- |
| `cuda.tile.TileSyntaxError` | `UnsupportedSyntaxError` 的别名 |
| `cuda.tile.TileTypeError` | `TypeCheckingError` 的别名 |
| `cuda.tile.TileValueError` | `InvalidValueError` 的别名 |
| `cuda.tile.TileUnsupportedFeatureError` | `UnsupportedFeatureError` 的别名 |
| `cuda.tile.TileCompilerExecutionError` | `CompilerExecutionError` 的别名 |
| `cuda.tile.TileCompilerTimeoutError` | `CompilerTimeoutError` 的别名 |

## 编译超时

`cuda.tile.compiler_timeout(timeout_sec)`

临时设置编译器超时的上下文管理器。

> **注意**
>
> 本函数**不是线程安全的**：它修改的是所有线程共享的全局状态。

示例：

```python
with ct.compiler_timeout(10):
    ct.launch(stream, grid, kernel, args)
```

## 环境变量

在 kernel 开发过程中遇到上面这些异常时，下面这些环境变量会派上用场。

设置 `CUDA_TILE_ENABLE_CRASH_DUMP=1`，可在遇到 `TileCompilerExecutionError`
或 `TileCompilerTimeoutError` 时导出一个包含 TileIR bytecode 的归档，
用来提交 bug 报告。

设置 `CUDA_TILE_COMPILER_TIMEOUT_SEC` 限制 TileIR 编译器 `tileiras` 可用的时间。

设置 `CUDA_TILE_LOGS=CUTILEIR` 把编译过程中的 cuTile Python IR 打印到 stderr。
调试 `TileTypeError` 时很有用。

设置 `CUDA_TILE_TEMP_DIR` 指定临时文件的存放目录。

设置 `CUDA_TILE_CACHE_DIR` 指定 bytecode→cubin 磁盘缓存的目录。
编译好的 cubin 缓存在这里，以避免未改动的 kernel 被重复编译。
设为 `0`、`off`、`none` 或空字符串可关闭缓存。默认为 `~/.cache/cutile-python`。

设置 `CUDA_TILE_CACHE_SIZE` 指定磁盘缓存的最大字节数，超出上限时淘汰最旧的条目。
默认 2 GB（2147483648）。

## 查看编译缓存

用 `cutile-cache log` 命令，可以按「最近访问优先」的顺序翻看缓存中的编译记录。
每条记录会显示 kernel 的 mangled 名、编译器版本、编译日期与耗时、CUBIN 大小，
以及优化备注（optimization remarks）：

```bash
cutile-cache log
```

使用 `tileiras` 13.4 或更新版本时，编译备注以 YAML 格式记录。
由更早的 cuTile Python 或编译器版本产生的条目仍然可读，但部分元数据可能显示为不可用。
