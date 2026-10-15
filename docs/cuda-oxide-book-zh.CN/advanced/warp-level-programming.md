# Warp 级编程

**warp** 是 CUDA 最基本的调度单位：32 个在同一 SM 上以锁步（lockstep）方式执行的线程。由于 32 个线程共享同一个指令指针，它们可以通过 **warp shuffle** 指令直接交换数据 —— 这是寄存器到寄存器（register-to-register）的传输，代价大约一个周期，既不需要共享内存，也不需要屏障或同步。

cuda-oxide 通过 `cuda_device::warp` 暴露了完整的 warp 内建指令集。本章介绍 shuffle、vote，以及它们带来的模式：warp 归约、广播、扫描和基于 ballot 的过滤。

> **参见**
>
> [CUDA Programming Guide — Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/#warp-shuffle-functions)，其中有 PTX 编码细节以及全部 width 变体。

---

## Lane 与 warp 标识

块中的每个线程在其所属 warp 内都有一个 **lane ID**（0–31）：

```rust
use cuda_device::warp;

let lane = warp::lane_id();     // 0..31，硬件寄存器 %laneid
let warp = warp::warp_id();     // threadIdx.x / 32
```

`warp_id()` 由 `threadIdx.x / 32` 推导而来。对于多维 block，它只考虑 x 维度 —— 这通常没问题，因为大多数关心 lane 标识的 kernel 都使用一维 block。

---

## Shuffle：寄存器到寄存器的数据交换

四种 shuffle 变体让一个线程能够读取另一个线程的寄存器，而不必经过内存：

| 函数                    | 作用                | 源 lane       |
| :-------------------------- | :---------------------------| :-----------------|
| `shuffle(val, src)`         | 从指定的 lane 读取   | `src`             |
| `shuffle_xor(val, mask)`    | 从 `lane_id ^ mask` 读取  | `lane_id ^ mask`  |
| `shuffle_down(val, delta)`  | 从 `lane_id + delta` 读取 | `lane_id + delta` |
| `shuffle_up(val, delta)`    | 从 `lane_id - delta` 读取 | `lane_id - delta` |

每种变体都有 `u32` 和 `f32` 两个版本：

```rust
let partner_val = warp::shuffle_xor_f32(my_val, 1);
let broadcast   = warp::shuffle_f32(my_val, 0);     // lane 0 的值发给所有 lane
let neighbor    = warp::shuffle_down_f32(my_val, 1); // 下一个 lane 的值
```

所有 shuffle 都是 **warp 同步**的 —— 它们隐式地同步整个 warp。不需要 `sync_threads()`；事实上，在 shuffle 模式内部调用 `sync_threads()` 既没必要又浪费。

---

## Warp 归约

最常见的 shuffle 模式是**蝶形归约**：在 ⌈log₂(32)⌉ = 5 步之内，每个 lane 都累加出全部 32 个值的和（或最小值、最大值等）。没有共享内存，没有屏障，只有五次 shuffle。

![使用 `shuffle_xor` 的蝶形归约。每一步中，lane 与它的 XOR 伙伴交换数值并相加。经过 5 步（掩码 16、8、4、2、1）之后，lane 0 持有全部 32 个值之和。](images/warp-shuffle-reduction.svg)

*使用 `shuffle_xor` 的蝶形归约。每一步中，lane 与它的 XOR 伙伴交换数值并相加。经过 5 步（掩码 16、8、4、2、1）之后，lane 0 持有全部 32 个值之和。*

```rust
use cuda_device::warp;

fn warp_reduce_sum(mut val: f32) -> f32 {
    val += warp::shuffle_xor_f32(val, 16);
    val += warp::shuffle_xor_f32(val, 8);
    val += warp::shuffle_xor_f32(val, 4);
    val += warp::shuffle_xor_f32(val, 2);
    val += warp::shuffle_xor_f32(val, 1);
    val
}
```

归约之后，**全部 32 个 lane** 都持有这个和（因为 XOR 是对称的 —— 双方都会累加）。如果你只需要 lane 0 上的结果，可以改用 `shuffle_down`：

```rust
fn warp_reduce_sum_lane0(mut val: f32) -> f32 {
    val += warp::shuffle_down_f32(val, 16);
    val += warp::shuffle_down_f32(val, 8);
    val += warp::shuffle_down_f32(val, 4);
    val += warp::shuffle_down_f32(val, 2);
    val += warp::shuffle_down_f32(val, 1);
    val
}
```

使用 `shuffle_down` 时，只有 lane 0 持有正确结果 —— 其他 lane 持有的是部分和。当只有 lane 0 写输出时，这样做没有问题。

> **提示**
>
> 需要块级归约？先用 shuffle 在每个 warp 内归约，把 32 个 warp 结果写入共享内存，`sync_threads()`，再用最后一次 warp 归约来合并这些 warp 级结果。这种混合方法比纯共享内存树更快，因为它省掉了 5 层屏障。

### Ampere 上的整数：一条指令取代十条

上面每一种归约 —— 手写的蝶形归约，以及通用的 `cooperative_groups::warp_reduce` —— 都要付出五次 shuffle 加五次合并的代价。对于 `sm_80` 及更新架构上的**整数**归约，有一条硬件指令 `redux.sync`，每个操作对应一个函数：

| 函数 | 归约内容 |
|:---------|:--------|
| `warp::redux_sync_add(mask, v)` | 回绕求和，`u32` |
| `warp::redux_sync_min_u32` / `_min_i32` | 最小值 |
| `warp::redux_sync_max_u32` / `_max_i32` | 最大值 |
| `warp::redux_sync_and` / `_or` / `_xor` | 按位 |

```rust
use cuda_device::warp;

const FULL_MASK: u32 = 0xffff_ffff;

// 整个归约。一条指令，每个 lane 都持有结果。
let total = warp::redux_sync_add(FULL_MASK, my_value);
```

在 A10G（`sm_86`）上实测，100 万个线程各自执行 64 次全 warp 的 `u32` 求和归约：蝶形形式每次启动耗时 159.4 µs，`redux.sync` 形式耗时 33.6 µs —— **4.74 倍**，而且结果逐位相同。`u32` 的回绕加法对 2³² 取模满足结合律，min/max/and/or/xor 都是精确的，所以归约顺序不会改变结果。

这个数字是把这个原语单独测出来的：探针 kernel 除了归约几乎什么都不做。对于一个在受内存带宽限制的遍历结束时才归约一次的 kernel，收益会非常小。只有当归约位于内层循环中时，才值得用它。

三条限制：

- **整数形式仅限 Ampere 及更新架构。** 有一个 `f32` 形式，但需要 Blackwell；见下一小节。`f64` 完全没有 `redux.sync` 形式，所以 `f64` 归约在任何架构上都继续用蝶形归约。
- **需要完整、已收敛的 warp。** 全部 32 个 lane 都必须活跃，并且以相同的 mask 到达同一个调用点 —— 这是 shuffle 归约本来就要求的同一份契约。子 warp tile 和短尾 warp 继续走 shuffle 路径（尾部情况参见 `reduce_sum_f32_partial` 之类的函数）。
- **在 `sm_80` 以下无法汇编。** 像 `redux_sum` 示例那样，用 `ctx.compute_capability()` 来给这个调用加门控，或者把构建限制在 Ampere 及更新架构上。

`warp_reduce` 目前不会替你选择这种形式；要让它自动这么做，需要让设备代码有办法知道自己的目标架构，这正是
[#811](https://github.com/NVlabs/cuda-oxide/issues/811) 中的未决问题。

### Blackwell 上的浮点：针对 `f32` 的 `redux.sync`

`redux.sync` 增加了 `f32` 的最小/最大值形式，对应八个函数。每个函数接收 `(mask: u32, value: f32)` 并返回 `f32`，与整数版本一样：

| 函数 | 归约内容 |
|:---------|:--------|
| `warp::redux_sync_min_f32` / `warp::redux_sync_max_f32` | 最小值 / 最大值，忽略 NaN 输入 |
| `warp::redux_sync_min_abs_f32` / `warp::redux_sync_max_abs_f32` | 同上，但对绝对值操作（`.abs`） |
| `warp::redux_sync_min_nan_f32` / `warp::redux_sync_max_nan_f32` | 传播 NaN（`.NaN`） |
| `warp::redux_sync_min_abs_nan_f32` / `warp::redux_sync_max_abs_nan_f32` | 同时使用两个修饰符 |

所以与整数家族不同，这里不是一个操作对应一个函数：`.abs` 和 `.NaN` 是彼此独立的指令修饰符，每种组合都有自己的入口点。

对于不能丢失 NaN 的归约来说，`.NaN` 形式很重要。普通形式和 `.abs` 形式会*忽略* NaN 输入；`.NaN` 形式则在任何 lane 贡献 NaN 时传播 NaN。要有意识地选择。两者只在某个 lane 持有 NaN 时才有差别，而恰恰这时选择才重要。

假设某个 lane 持有 `NaN`，而 warp 中最小的普通值是 `-1.0`：

- `redux_sync_min_f32(mask, v)` 返回 `-1.0`（NaN 被忽略）
- `redux_sync_min_nan_f32(mask, v)` 返回 `NaN`（NaN 被传播）

**这仅限 Blackwell，而且比"sm_100+"还要窄。** f32 形式需要 PTX 8.6，且只存在于 `sm_100a`、`sm_100f`、`sm_103a` 和 `sm_103f` 上。它们在 `sm_110a` 和 `sm_120a` 上都不可用，所以消费级 Blackwell 用不了。请针对这四个目标之一构建，否则编译会失败。

`redux_f32` 示例是可用的参考实现。它为 `sm_100a` 构建，所以 CI 会编译它并检查生成的 PTX，而不是运行它。除非设备报告的就是 `sm_100` 而不带其他后缀，否则它的 `main` 会直接跳过。

---

## 广播

把 lane 0 的值广播给所有 lane 只需一次 shuffle：

```rust
let leader_val = warp::shuffle_f32(my_val, 0);
```

任何 lane 都可以作为源。这取代了"lane 0 写入共享内存、同步、所有 lane 读取"的共享内存模式 —— 一条指令取代三步操作。

---

## 包含式前缀和（scan）

**包含式扫描**计算的是运行中的累计值：lane `i` 持有从 lane 0 到 lane `i` 的所有值之和。该模式使用 `shuffle_up`：

```rust
fn warp_inclusive_scan(mut val: f32) -> f32 {
    let mut offset = 1u32;
    while offset < 32 {
        let n = warp::shuffle_up_f32(val, offset);
        if warp::lane_id() >= offset {
            val += n;
        }
        offset *= 2;
    }
    val
}
```

经过 5 步之后，每个 lane 都持有包含自身值在内的前缀和。这是流压缩（stream compaction）、直方图构建和并行扫描算法的基本构件。

---

## Vote：warp 级谓词

Vote 操作让整个 warp 共同评估一个布尔条件：

| 函数              | 返回值                                                              |
| :-------------------- | :------------------------------------------------------------------- |
| `warp::all(pred)`     | 若*每个*活跃 lane 的 `pred` 都为 `true`，则为 `true`                     |
| `warp::any(pred)`     | 若*任一*活跃 lane 的 `pred` 为 `true`，则为 `true`                       |
| `warp::ballot(pred)`  | 一个 `u32` 位掩码 —— 若 lane `i` 的 `pred` 为 `true`，则第 `i` 位置位      |
| `warp::popc(pred)`    | 总体计数：有多少个活跃 lane 的 `pred == true`          |

### 用 ballot 过滤

一个常见的模式是压缩数组，只保留通过谓词的元素。`ballot` + `popc` 可以同时给出计数和每个 lane 的写入偏移：

```rust
use cuda_device::{DisjointSlice, kernel, thread, warp};

#[kernel]
pub fn compact_positive(
    input: &[f32],
    mut output: DisjointSlice<f32>,
    mut count: DisjointSlice<u32>,
) {
    let idx = thread::index_1d();
    let val = input[idx.get()];
    let is_positive = val > 0.0;

    let mask = warp::ballot(is_positive);
    let lane = warp::lane_id();

    // 统计该 lane 之下置位的比特，得到写入位置
    let offset = (mask & ((1u32 << lane) - 1)).count_ones();

    if is_positive {
        unsafe {
            *output.get_unchecked_mut(offset as usize) = val;
        }
    }

    // lane 0 记录该 warp 的总计数
    if lane == 0 {
        unsafe {
            *count.get_unchecked_mut(warp::warp_id() as usize) = mask.count_ones();
        }
    }
}
```

`ballot` 掩码用单个寄存器编码了整个 warp 的谓词结果。没有通信，没有共享内存 —— 硬件在一个周期内就算出来了。

---

## 何时用 warp 原语、何时用共享内存

| 任务                        | Warp shuffle                 | 共享内存                  |
| :-------------------------- | :--------------------------- | :----------------------------- |
| 归约 32 个值            | 5 次 shuffle，约 5 周期        | 5 次加载 + 5 次同步，约 50 周期  |
| 归约 256 个值           | shuffle + 1 次同步 + shuffle   | 树形归约，约 10 次同步      |
| Stencil（邻居访问）   | `shuffle_up`/`shuffle_down`  | 适合二维 stencil           |
| 数据对其他 warp 可见 | 做不到                 | 必需                       |
| 随机访问模式      | 不支持                | 可自由索引                   |
| 跨 warp 持续存在    | 不适用                   | 在块的生命周期内持续存在    |

经验法则：如果数据能放进单个 warp（32 个元素），而且访问模式规整，shuffle 更快。如果你需要跨 warp 通信、更大的数据集或随机访问，共享内存才是合适的工具。

---

## 一个完整示例：warp 级点积

把 shuffle 和 vote 结合起来，下面这个 kernel 用 warp 归约计算两个向量的点积：

```rust
use cuda_device::{DisjointSlice, kernel, thread, warp};

#[kernel]
pub fn warp_dot_product(
    a: &[f32],
    b: &[f32],
    n: u32,
    mut result: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();

    // 每个线程计算逐元素乘积中的一个元素
    let product = if idx.get() < n as usize {
        a[idx.get()] * b[idx.get()]
    } else {
        0.0f32
    };

    // Warp 级归约
    let mut sum = product;
    sum += warp::shuffle_xor_f32(sum, 16);
    sum += warp::shuffle_xor_f32(sum, 8);
    sum += warp::shuffle_xor_f32(sum, 4);
    sum += warp::shuffle_xor_f32(sum, 2);
    sum += warp::shuffle_xor_f32(sum, 1);

    // 每个 warp 的 lane 0 写入它的部分和
    if warp::lane_id() == 0 {
        unsafe {
            *result.get_unchecked_mut(warp::warp_id() as usize) = sum;
        }
    }
}
```

要得到完整的点积，再启动一趟处理来归约各 warp 的结果 —— 可以用另一个 warp kernel，也可以用原子操作。第一趟只用寄存器 shuffle 就消除了绝大部分工作。

> **参见**
>
> - [共享内存与同步](shared-memory-and-synchronization.md) —— 处理大于 warp 规模操作时的块级对应方案
> - [Tensor Memory Accelerator](tensor-memory-accelerator.md) —— 加速这些模式所依赖的全局→共享数据搬运的硬件
