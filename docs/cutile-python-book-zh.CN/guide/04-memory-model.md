# 内存模型

cuTile 的内存模型允许编译器和硬件出于性能考虑重排操作。
在没有显式同步的情况下，**跨 block** 的内存访问顺序是没有保证的。

为了协调 block 之间的内存访问，cuTile 给原子操作提供了两个属性：

- **Memory Order（内存序）** —— 定义原子操作的排序语义。
- **Memory Scope（内存作用域）** —— 定义参与排序的 block 集合。

同步以**逐元素**为粒度：数组中的每个元素独立地参与内存模型。

更多细节见 Tile IR 文档的 Memory Model 一节。

## 内存序（Memory Order）

`class cuda.tile.MemoryOrder`

内存操作的内存排序语义。

`WEAK = 'weak'`

Weak（非原子）排序。load/store 操作的默认值。

`RELAXED = 'relaxed'`

不提供任何顺序保证。不能用于线程间同步。

`ACQUIRE = 'acquire'`

Acquire 语义。当本次读取读到了由某个 release 写入的值时，
执行 release 的那个线程此前写入的内容变得可见。
同一 block 内后续的读/写不能被重排到本次操作之前。

`RELEASE = 'release'`

Release 语义。当某个 acquire 读到了由本次写入的值时，
本线程此前写入的内容对执行 acquire 的线程变得可见。
同一 block 内此前的读/写不能被重排到本次操作之后。

`ACQ_REL = 'acq_rel'`

acquire 与 release 的组合语义。

## 内存作用域（Memory Scope）

`class cuda.tile.MemoryScope`

参与内存排序的线程范围。

`NONE = 'none'`

无内存作用域。用于采用 WEAK 内存序的 load/store 操作。

`BLOCK = 'block'`

顺序保证作用于同一 block 内的线程。

`CLUSTER = 'cluster'`

顺序保证作用于同一 thread-block cluster 内的所有线程。仅 `cuda.lang`。

`DEVICE = 'device'`

顺序保证作用于同一块 GPU 上的所有线程。

`SYS = 'sys'`

顺序保证作用于整个系统中的所有线程，包括多块 GPU 和主机。
