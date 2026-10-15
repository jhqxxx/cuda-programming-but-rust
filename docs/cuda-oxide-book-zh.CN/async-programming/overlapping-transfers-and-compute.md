# 传输与计算的重叠

可分页的宿主内存用起来方便，但在 DMA 传输之前，CUDA 可能会先通过内部的页锁定内存对它做一次中转。这种额外的中转工作会限制异步拷贝。`PinnedHostBuffer<T>` 为 CUDA 提供一个稳定的宿主地址，以便直接传输。真正的拷贝/计算重叠还要求在非默认流上使用异步拷贝。

用下面的命令运行完整演示：

```bash
cargo oxide run pinned_overlap
```

该示例使用一个开销很小的原地自增 kernel，因此它的耗时主要由数据搬运而不是计算决定。

## 选择拷贝 API

安全的辅助函数会在返回之前同步流。当拷贝在途时宿主不需要做别的事情，就用它：

```rust
device.copy_to_pinned_host(&stream, &mut host)?;
// `host` 可以安全读取，因为辅助函数已经同步了 `stream`。
```

异步辅助函数在拷贝完成之前就返回。同步和缓冲区生命周期由调用者负责：

```rust
// SAFETY：`host` 和 `device` 会保持存活，且在 `done` 触发之前不会被复用。
unsafe { device.copy_to_pinned_host_async(&stream, &mut host)? };
let done = stream.record_event(None)?;
```

调用者必须让两个缓冲区都保持存活，不得读取或修改在途的宿主缓冲区，并且必须在丢弃或复用它之前等待流上的工作完成。同样的规则也适用于 `copy_from_pinned_host_async`。请预分配设备缓冲区并反复填充它们；`from_pinned_host` 在入队其拷贝之前仍然会执行一次同步的设备分配。

## 轮转锁页中转缓冲区

操作在单个流内部保持有序，而在支持并发拷贝与计算的硬件上，相互独立的流可以并发推进：

```text
stream 0: upload 0 -> kernel 0 -> download 0 -> upload 3 -> kernel 3
stream 1: upload 1 -> kernel 1 -> download 1 -> upload 4 -> kernel 4
stream 2: upload 2 -> kernel 2 -> download 2 -> upload 5 -> kernel 5
```

该示例为每个流提供一个常驻的 `DeviceBuffer`、一个锁页中转缓冲区，以及一个完成事件。上传、kernel 和下载本来就已经被流排好了序，所以这个事件只用于保护宿主侧的 slot 复用。复用某个 slot 之前，宿主要等待它的事件，把已完成的结果拷出来，然后重新填充中转缓冲区。最后的流 join 会排空所有工作，之后才会释放 CUDA 拥有的内存。

下面是示例循环中核心的入队序列；周边的初始化与结果处理已省略：

```rust
// SAFETY：这些 slot 缓冲区会保持存活，且在完成之前不会被复用。
unsafe {
    devices[slot].copy_from_pinned_host_async(&streams[slot], &stagers[slot])?;
    module.increment(&streams[slot], config, &mut devices[slot])?;
    devices[slot].copy_to_pinned_host_async(&streams[slot], &mut stagers[slot])?;
}
completions[slot] = Some(streams[slot].record_event(None)?);
```

完整的初始化、slot 复用等待和最终排空都在 [`run_overlapped` 实现](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap/src/main.rs)中。

## 实测影响

带宽基准测试预热三次，记录十个 CUDA 事件采样，并报告它们的中位数。流水线基准测试测量八个 32 MiB 数据块的宿主墙钟时间。串行路径为每个块分配并同步一个可分页的设备缓冲区。重叠路径预分配三个设备缓冲区和锁页中转缓冲区，并把宿主中转拷贝、事件等待和结果拷贝都计入它的墙钟时间。因此这个结果是端到端的路径对比，而不是纯粹的拷贝引擎测量。

在参考的 RTX 4090（`sm_89`）上运行时：

| 流水线 | 时间 |
|---|---:|
| 串行可分页 | 174.996 ms |
| 重叠锁页 | 99.332 ms |
| 加速比 | 1.76x |

完整的传输数据表在[示例 README](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap)中。结果会随 GPU、PCIe 拓扑、驱动和系统负载而变化。

## 实际权衡

锁页会占用有限的宿主内存，而把可分页输入拷贝进中转缓冲区也不是免费的。请复用一个小型的缓冲区环，并在增加流数量之前先对拷贝引擎做性能分析。

关于更底层的流模型，参见[调度与流](scheduling-and-streams.md)。完整的可运行实现参见 [`pinned_overlap` 示例](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap)。
