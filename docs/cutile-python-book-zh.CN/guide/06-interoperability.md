# 互操作

## 机器表示（Machine Representation）

cuTile 把 Python tile 代码翻译成一种能被 CUDA 设备执行的**机器表示**，从而在 NVIDIA GPU
上运行它。函数、类型和对象都有各自的机器表示。

机器表示是相对对应的 CUDA C++ 实体来定义的。
例如：`cuda.tile.float16` 与 CUDA C++ 中的 `__half` 具有相同的机器表示。

## 与 SIMT 的互操作

### Kernel 之间（Inter-Kernel）

Kernel 之间的互操作指的是所有**不跨越 kernel 边界**的互操作问题 ——
也就是除了「在同一个 kernel 里混用 tile 代码和 SIMT 代码」之外的一切。包括：

- 在同一个源文件里写 tile kernel 和 SIMT kernel。
- 把 tile kernel 和 SIMT kernel 链接进同一个二进制。
- 把同一种 array 同时传给 tile kernel 和 SIMT kernel。

Kernel 内部（intra-kernel）的互操作将在未来支持。

## JAX FFI

cuTile kernel 可以从 JAX 追踪出来的图中启动，方式是 `cuda.tile.jax.cutile_call()`。
它会把 buffer、标量和 tuple 参数穿过 JAX FFI 的调用点，
使 kernel 在 `jax.jit` 编译出的图里表现为一个普通的 op。

完整的参数约定见 `cuda.tile.jax.cutile_call()`；
声明输出和就地更新则见 `cuda.tile.jax.OutputPlaceholder` 与
`cuda.tile.jax.InputOutput`。
