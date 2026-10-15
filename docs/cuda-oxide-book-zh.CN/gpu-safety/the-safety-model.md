# 安全模型

一个 GPU kernel 会同时运行数千个线程，它们看到的是同一块内存。在 CPU 上，Rust 通过所有权和借用来防止数据竞争 —— 一个可变引用、不允许别名，并在编译期强制执行。在 GPU 上，每个 SM 有 2048 个线程，它们全都由同一个函数启动，全都指向同一个输出缓冲区。借用检查器并不是为此设计的。

cuda-oxide 分层解决这个问题。常见情形 —— 一个线程在受检查的启动契约下写入一个元素 —— 由构造本身保证安全。不常见的情形 —— 共享内存、warp shuffle、硬件 intrinsic —— 需要 `unsafe`，并附带成文的契约。而前沿情形 —— TMA、tensor core、cluster 级通信 —— 完全手动，其复杂度与其所控制的硬件相匹配。

本章解释这一模型，逐层走过每一层，并明确告诉你何时需要 `unsafe`，以及为什么。

---

## 三个层级

cuda-oxide 根据编译器能验证多少，把 kernel 的安全性组织为三个层级：

| 层级       | 描述                                                    | 是否需要 `unsafe`？ |
|:-----------|:--------------------------------------------------------|:-------------------|
| **Tier 1** | 安全的 kernel 主体，外加受检查的 `PreparedLaunch`        | 否                 |
| **Tier 2** | 显式的 `unsafe`，并带有清晰的安全契约                    | 是，作用域受限      |
| **Tier 3** | 原始硬件 intrinsic —— 完全由用户负责                     | 是，随处可见        |

大多数应用层 kernel 完全位于 Tier 1，或者横跨 Tier 1 与 Tier 2。Tier 3 面向的是在 CUTLASS 或 Triton IR 这一层面工作的性能工程师。如果你写的是一个 vecadd、一个 GEMM 或一个归约，你很少会离开 Tier 2。

---

## Tier 1：默认安全

### 核心思想：`DisjointSlice<T>` + `ThreadIndex`

设备侧主要的安全抽象是一对类型，它们共同支持无竞争的并行写入：

- **`ThreadIndex<'kernel, IndexSpace>`** —— 围绕一个 `usize` 的不透明见证（witness），没有公开的构造函数。获得它的唯一途径是通过受信任的函数（`index_1d`、`index_2d::<S>`），这些函数从**硬件内建变量**（`threadIdx`、`blockIdx`、`blockDim`）推导出该值 —— 这些是运行时在启动时填入的只读特殊寄存器。`ThreadIndex` 是 `!Send + !Sync + !Copy + !Clone` 的，并且 `'kernel` 生命周期把它绑定到 kernel 主体内一个栈局部的作用域，因此它无法通过共享内存被偷渡到其他线程，也无法比 kernel 活得更久。
- **`DisjointSlice<T, IndexSpace>`** —— 一个类似切片的类型，其 `get_mut()` 方法只接受 `IndexSpace` 与自身匹配的 `ThreadIndex`。返回 `Option<&mut T>` —— 越界的索引得到 `None`。

把两者放在一起，你就得到一个零 `unsafe` 的 kernel 主体：

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

`get_mut_indexed` 是一次调用即可的形式：它一次性地铸造出每线程的见证，并把它解析为 `&mut T`。当你需要该索引来对多个切片做并行的算术运算时，也可以使用显式的两步形式：

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

安全性来自五个事实：

1. `index_1d()` 只使用 X 坐标。当 block 与 grid 的 Y/Z 维度都为 1 时，它的取值是唯一的。`domain = 1` 的 prepared 启动证明了这一点；没有那个带类型的证明时，设备会检查 Y/Z，并在任一轴处于活动状态时把见证标记为无效。此时安全的切片访问会返回 `None`。
2. `get_mut()` 是带边界检查的 —— 超出范围的线程得到 `None`。
3. `IndexSpace` 参数把每个见证与其布局绑定：`DisjointSlice<T, Index2D<128>>` 不会接受 `ThreadIndex<'_, Index2D<256>>` —— 混用 stride 是编译错误。
4. 见证是 `!Send + !Sync + !Copy + !Clone` 且以 `'kernel` 为作用域的，所以线程无法通过共享内存洗白彼此的索引。
5. 受检查的 `PreparedLaunch<K>` 把宿主几何配置与确切的 kernel 绑定在一起。

借用检查器看到的仍然是每线程一个 `&mut T`。硬件提供了坐标；启动证明让它们的线性化互不相交。类型系统则把设备侧的证明与宿主侧的证明绑在一起。

```text
1D prepared 启动：Y=Z=1 -> index_1d 的值唯一 -> 安全的 get_mut
raw 2D 启动：       Y/Z 活动 -> 见证无效 -> 安全的 get_mut 返回 None
```

原始启动（raw launch）仍然是 `unsafe` 的：即使这个秩检查在设备上以失败关闭（fail closed）的方式处理，宿主仍必须履行它其余的访存与启动义务。

### 受信任的索引函数

`ThreadIndex` 的可信程度完全取决于创建它的函数。下面是 cuda-oxide 提供的构造函数：

| 函数                          | 公式                                    | 返回类型                                      | 备注                                             |
|:------------------------------|:----------------------------------------|:----------------------------------------------|:-------------------------------------------------|
| `index_1d()`                  | `blockIdx.x * blockDim.x + threadIdx.x` | `ThreadIndex<'kernel, Index1D>`               | 无类型启动存在活动的 Y/Z 时无效                   |
| `index_2d::<S>()`             | `row * S + col`                         | `Option<ThreadIndex<'kernel, Index2D<S>>>`    | 常量 stride；混用 stride 是编译错误               |
| `index_2d_runtime(&slice)`    | `row * width + col`                     | `Option<ThreadIndex<'kernel, Runtime2DIndex>>`| 行宽从切片中读取，由宿主绑定                      |
| `index_2d_row()`              | `blockIdx.y * blockDim.y + threadIdx.y` | `usize`                                       | 分量访问器，不是见证构造函数                      |
| `index_2d_col()`              | `blockIdx.x * blockDim.x + threadIdx.x` | `usize`                                       | 分量访问器，不是见证构造函数                      |

`index_2d_row()` 与 `index_2d_col()` 返回的是普通的 `usize` —— 它们给你用于算术运算的分量，但不能用来索引 `DisjointSlice`。只有线性化之后的那个结果才能换来一个 `ThreadIndex`。同理，2D 的唯一性假设本次启动没有活动的 Z 维度。`domain = 2` 的 prepared 启动证明了这一点。没有带类型的证明时，设备会检查 Z，并在它处于活动状态时返回 `None`。

<a id="check-a-tile-once"></a>
### 一次检查一整块 tile

对于小的固定 tile，一次边界检查就能覆盖该 tile 中的每一个元素：

```rust
use cuda_device::{DisjointSlice, LinearTiles, kernel, launch_contract, thread};

#[kernel(launch_context = launch_context)]
#[launch_contract(
    domain = 1,
    coordinates = u32,
    block = (64, 1, 1),
)]
pub fn add_one(mut values: DisjointSlice<u32, LinearTiles<4>>) {
    let thread_index = thread::index_1d_u32(launch_context);

    if let Some(mut tile) = values.tile_thread32(thread_index) {
        let first = tile.at_const::<0>().read();
        tile.at_const::<0>().write(first + 1);

        let second = tile.at_const::<1>().read();
        tile.at_const::<1>().write(second + 1);
    }
}
```

`LinearTiles<4>` 把元素 0--3 给线程 0，把元素 4--7 给线程 1，依此类推。`tile_thread32` 对完整的四元素 tile 检查一次。如果只有一个 tile 的一部分落在范围内，它返回 `None`。像 `at_const::<0>()` 这样的常量位置不需要再做运行时的边界检查；非法的常量会导致编译失败。

这两条标注提供了那些受检查的事实：

- `#[kernel(launch_context = launch_context)]` 创建一个局部的设备侧取值，它携带关于本次启动的事实。它不是 kernel 参数。
- `domain = 1` 表示只有 X 轴是活动的。
- `coordinates = u32` 允许使用 `thread::index_1d_u32(launch_context)`，它把本线程在完整 1-D 启动中的唯一索引以 `u32` 返回。
- `block = (64, 1, 1)` 要求每个 block 恰好有 64 个线程。本次启动仍然可以包含很多个 block。

确切的 `block` 会被检查两次，在两个相互独立失败的地方。准备工作会在宿主上拒绝不匹配的启动配置，且发生在任何驱动调用之前。这个形状也会到达设备端编译器，由它发出 `.reqntid 64, 1, 1`，因此 CUDA 驱动会拒绝任何在任一轴上 block 形状不同的启动。第二次检查覆盖了第一次看不到的路径：`unsafe` 的原始启动，以及加载的 PTX 并非生成的 Rust 标记所描述的那份产物的模块。

对于带契约的 kernel，`.reqntid` 取代 `.maxntid`。ptxas 会拒绝同时声明两者的入口，而一个确切的形状本来就已经限定了线程数。只有 `#[launch_bounds]` 的 kernel 会保留 `.maxntid`，它约束的是乘积 `x * y * z`，因此接受任何能落在该乘积之内的形状。这一差异正是线程数上限无法替代 block 形状的原因：在 `.maxntid 64, 1, 1` 之下，驱动会接受 `(8, 8, 1)` 和 `(4, 4, 4)`，与 `(64, 1, 1)` 并列。

对于 2-D tile，请使用 `thread::coord_2d_u32(launch_context)` 配合 `RowMajorTiles<ROWS, COLS, ROW_STRIDE>`。`ROW_STRIDE` 是以元素为单位的逻辑行宽，必须与缓冲区的布局匹配。缓冲区长度不必是行宽的整数倍；只会返回完整的 tile。

可运行的 1-D 与 2-D 版本见 [`proof_carrying_views`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/proof_carrying_views) 示例。

### `index_2d` 如何做到类型安全

`index_2d::<S>()` 在行 stride 上是常量泛型的。见证以 `ThreadIndex<'kernel, Index2D<S>>` 的形式返回，而 `DisjointSlice<T, Index2D<S>>` 只接受那个确切的 `S`。两个线程无法铸造出 stride 不同却都被喂进同一个切片的见证 —— 类型系统会拒绝它：

```rust
#[kernel]
pub fn ok(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<128>() {  // 匹配
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}

#[kernel]
pub fn rejected(mut out: DisjointSlice<u32, Index2D<128>>) {
    if let Some(idx) = thread::index_2d::<256>() { // ⛔ Index2D<256> != Index2D<128>
        if let Some(slot) = out.get_mut(idx) { *slot = 1; }
    }
}
```

该见证同样是 `!Send + !Sync + !Copy + !Clone` 的，并且它的 `'kernel` 生命周期借用自宏注入的某个栈局部作用域 —— 因此线程无法把自己的 `ThreadIndex` 存放在共享内存里等着邻居稍后取走，见证也无法比 kernel 主体活得更久。

#### 真正的运行时 stride：`index_2d_runtime`

有些 kernel 确实是在启动时才拿到 stride 的（例如矩阵维度只有在宿主上才知道）。对于这些情况，有一个对应的见证 `Runtime2DIndex`，它从索引将要寻址的那个切片中铸造：

```rust
let idx = thread::index_2d_runtime(&c)?;
```

不需要 `unsafe`，因为这里不存在可以互相不一致的每次调用 stride。行宽随切片一起传递，由宿主写入启动包一次，因此每一个索引 `c` 的线程按构造都使用同一个行宽。把 stride 单独传进来做不到这一点：在不同运行时 stride 下产生的两个 `ThreadIndex<'_, Runtime2DIndex>` 取值具有相同的类型，所以类型系统无法把它们区分开。如果你能在编译期固定 stride，请优先使用 `index_2d::<S>()`。

有一个推论塑造了这个见证本身。同一个 kernel 中的两个 `Runtime2DIndex` 切片可以携带不同的行宽，而它们的见证共享同一个类型，因此安全代码可以从每个切片各铸造一个见证，并把任意一个交给任意一个切片。于是该见证存储的是线程原始的 `(row, col)` 坐标，而不是扁平索引，`get_mut` 会针对**被寻址**的那个切片自己的行宽来解析它们（`row * width + col`，并要求 `col < width`）。不同的线程持有不同的坐标，而每个线程从它所寻址的切片读到的都是同一个由宿主绑定的行宽，所以无论见证是由哪个切片铸造的，映射在每个切片上都是单射的。反过来，如果在铸造时就把扁平索引烙进去，就会把铸造者所在切片的 grid 一起偷渡过来：宽度 5 下的 `(1, 0)` 与宽度 100 下的 `(0, 5)` 都被扁平化为 5，指向元素 5。

#### 运行时行宽存放在哪里

*启动*确立了“每个线程看到的值都相同”这一点：宿主把一个字编组进启动包，每个 block 的每个线程读到的都是同样的那些字节。kernel 主体没有任何可比的方式，就它自己的某个局部值确立同样的结论。

所以，一个索引空间携带运行时行宽的切片，也以同样的方式获得那个宽度。在宿主上，由 `RowWidth` 绑定它：

```rust
module.write_c(&stream, &prepared, cuda_host::RowWidth::new(&mut c, n))?;
```

而在设备上，访问器不带自己的参数就能把它读回来：

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    // 不需要 `unsafe`：行宽来自宿主，并绑定到了 `c`。
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

为什么这样就够了：在每个切片只有一种行宽的前提下，`last_col < stride` 的检查会让每个 tile 都留在同一个逻辑行内，这使坐标到偏移的映射成为单射，于是不同的线程拥有互不相交的元素。两个线程对宽度看法不一致，恰恰就是破坏这一点的情形。在 1x1 tile 下，stride 为 5 时线程 `(1, 0)` 解析出扁平索引 5，而 stride 为 100 时线程 `(0, 5)` 也解析出 5：同一个元素，两个 `&mut`。

每次调用单独传入的宽度无法排除这一点，即使每个候选值本身都是启动一致的 —— 因为安全代码可以在一个随线程而变的条件下在两个候选值之间做选择，或者在同一个切片的两个调用点上传入不同的值。把绑定放在宿主边界上，就消除了这种选择。

要让 kernel 算出预期结果，被绑定的值仍然必须就是缓冲区实际使用的行宽。这是正确性上的要求，而不是安全性上的要求；一个错误描述布局的宽度会在切片内部给出错误答案，而绝不会造成别名或越界访问。

#### 启动一致的标量：`Uniform<T>`

同样的推理适用于任何需要在线程间保持一致的标量，而不仅限于行宽。`Uniform<T>` 就是这个事实的类型化表示。只要声明这个参数，该义务就由 kernel ABI 履行：

```rust
#[kernel(launch_context = lc)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
pub fn write_c(rows: Uniform<u32>, mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
    let coord = thread::coord_2d_u32(lc);
    if coord.row() >= rows.get() {
        return;
    }
    if let Some(mut cell) = c.tile_2d32_rt(coord) {
        cell.at_const::<0, 0>().write(1.0);
    }
}
```

宿主侧的方法仍然接受一个普通的 `u32`；`#[cuda_module]` 会把它包起来，而 `#[repr(transparent)]` 让启动包保持逐字节相同。它没有构造函数，所以设备代码无法把一个每线程取值提升为见证。`Uniform::new_unchecked` 仍保留给那些通过其他方式被证明为一致的取值，而 `block_dim_x()` 及其同类函数直接返回见证，因为一次启动只有一个 block 形状。

一致性在“所有操作数都是一致的”算术运算下是封闭的，因此派生出来的取值也会保留该见证。它在选择运算下**不**封闭：在一个随线程而变的条件下于两个一致值之间做选择，得到的取值会在线程之间不同，这就是为什么这类见证无法携带一个必须横跨整个切片成立的证明。

### GEMM 模式

对于常量 stride 的 2D kernel（常见情形 —— 分块 GEMM、stencil kernel、固定通道数的图像 kernel），常量泛型形式是安全的默认选择：

```rust
const STRIDE: usize = 1024;     // C 是 M x STRIDE

#[kernel]
pub fn gemm(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Index2D<STRIDE>>, m: u32) {
    let row = thread::index_2d_row();
    if let Some((c_elem, _)) = c.get_mut_indexed() {
        // `Some` 已经保证了 col < STRIDE —— 无需手动检查
        if row < m as usize {
            // ... 把点积计算进一个本地累加器 `sum` ...
            *c_elem = alpha * sum + beta * (*c_elem);
        }
    }
}
```

对于运行时 stride，同样的形状也能用，而且依然安全；行宽来自输出切片而不是来自常量泛型：

```rust
#[kernel]
pub fn gemm_runtime(a: &[f32], b: &[f32], mut c: DisjointSlice<f32, Runtime2DIndex>, m: u32, n: u32) {
    let row = thread::index_2d_row();

    // 行宽来自 `c`，由宿主绑定一次。不需要 `unsafe`。
    if let Some(c_idx) = thread::index_2d_runtime(&c) {
        if row < m as usize {
            // ... 计算点积 ...
            if let Some(c_elem) = c.get_mut(c_idx) {
                *c_elem = alpha * sum + beta * (*c_elem);
            }
        }
    }
}
```

来自 `index_2d_runtime` 的 `if let Some` 取代了你在 CUDA C++ 中需要手写的 `col < n` 保护。`row < m` 检查仍然保留，因为它防止从输入矩阵中读到垃圾数据。

以上覆盖的是输出的写入。GEMM 的*输入*一侧 —— 以及每次 `a[...]` 读取上边界检查的代价 —— 有它自己的一章：[边界检查：一次付清，或者选择退出](bounds-checks.md)。

### 什么样的 kernel 属于 Tier 1

一个 kernel 及其启动在以下条件全部满足时是完全安全的 —— 即 Tier 1：

1. 所有可变的输出访问都经过 `DisjointSlice::get_mut(ThreadIndex)`
2. 所有输入都是共享的不可变引用（`&[T]`）
3. 没有共享内存、没有裸指针、除了线程索引之外没有别的 intrinsic
4. 宿主使用匹配的、受检查的 `PreparedLaunch<K>`

这一层的例子包括 `vecadd`、`helper_fn`、`generic`、`host_closure`，以及 `gemm` 与 `async_mlp` 示例中那些通过匹配契约启动的朴素 GEMM kernel。它们走原始 `LaunchConfig` 的路径仍然是不安全的。

---

## Tier 2：作用域受限的 `unsafe`

并非每个 kernel 都符合“一个线程、一个输出元素”的模式。当线程需要协作时 —— 通过快速的片上内存共享数据、在一个 warp 内的各 lane 之间通信，或者执行原子更新 —— 你就需要 `unsafe`。Tier 2 的关键性质在于，这些 `unsafe` 是*作用域受限*且*可审计*的：每个块都有成文的安全契约，而 kernel 的其余部分仍然是安全的。

### 共享内存

共享内存很快、位于片上，并且对 block 中的每个线程都可见。最后这条性质恰恰是它需要 `unsafe` 的原因 —— 借用检查器无法推理 256 个线程写入同一个 `static mut` 数组的情况：

```rust
static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

unsafe { TILE[ty * TILE_SIZE + tx] = value; }

thread::sync_threads();

let neighbor = unsafe { TILE[other_idx] };
```

契约是：确保并发线程之间在没有同步的情况下不会发生冲突写入。`sync_threads()` barrier 就是让这套做法得以成立的工具 —— 它保证所有线程在任何一个线程读取之前都已完成写入。

| API                       | 安全义务                                                                                            |
|:--------------------------|:---------------------------------------------------------------------------------------------------|
| `SharedArray<T, N>`       | 通过 `static mut` 访问。没有同步就不允许冲突写入。                                                  |
| `DynamicSharedArray<T>`   | 规则相同，但大小在启动时通过 `LaunchConfig::shared_mem_bytes` 设定。                                |

> **参见**：[共享内存与同步](../advanced/shared-memory-and-synchronization.md)，其中有完整的讲解：分块（tiling）、bank 冲突、动态分配以及双缓冲流水线。

### Warp intrinsic

warp 级原语让一个 warp 内的线程完全不碰内存就能交换数据 —— 寄存器到寄存器的传输，由硬件协调。它们是 `unsafe` 的，因为硬件不会检查线程收敛：如果你传入的掩码包含一个已经发散的线程，你会得到未定义行为（通常是静默挂起，这比崩溃更糟）。

| API                                                                | 安全义务                                              |
|:-------------------------------------------------------------------|:-----------------------------------------------------|
| `shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync`     | 源 lane 必须是活动的；掩码必须包含发起调用的线程        |
| `ballot_sync`, `any_sync`, `all_sync`                              | 掩码中的所有线程必须处于收敛状态                      |
| `activemask`                                                       | 结果只在调用的那个点上才有意义                         |

> **参见**：[Warp 级编程](../advanced/warp-level-programming.md)，其中有 shuffle 模式、归约，以及使用 warp intrinsic 的前缀和。

### Barrier 与生命周期

`ManagedBarrier` 的 typestate API 把 barrier 的生命周期（`Uninit` -> `Ready` -> `Invalidated`）编码进类型系统，因此你无法等待一个从未初始化过的 barrier，也无法使用一个已经被置为无效的 barrier。`init()` 与 `inval()` 这两个状态迁移仍然需要 `unsafe`，因为它们要与硬件交互，但这些类型状态在编译期阻止了最常见的错误。

| API                                    | 安全义务                                                                       |
|:---------------------------------------|:--------------------------------------------------------------------------------|
| `mbarrier_init`                        | 必须恰好由一个线程调用；barrier 必须位于共享内存中                              |
| `mbarrier_arrive` / `mbarrier_wait`    | barrier 必须已初始化；token 必须匹配                                            |
| `ManagedBarrier`（typestate）          | `init()` 与 `inval()` 需要 `unsafe`；状态机在编译期强制执行                      |

### 原子操作

一旦你有了有效的原子引用，原子操作*调用*起来就是安全的。`unsafe` 的暴露面在构造处 —— 从裸指针创建 `DeviceAtomicU32` 要求调用者保证该指针有效且对齐正确：

```rust
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};

// `from_ptr` 是不安全的构造函数：它重新解释已有的内存。
// `DeviceAtomicU32::new(value)` 是安全的那个，用于有所有权的 static。
let atom = unsafe { DeviceAtomicU32::from_ptr(ptr) };
atom.fetch_add(1, AtomicOrdering::Relaxed);  // 安全的调用
```

### 不受检查的切片访问

当“一个线程、一个元素”的模型不适用时 —— 例如在 warp 级归约中只有 lane 0 写入结果 —— `DisjointSlice::get_unchecked_mut(usize)` 提供了一个逃生通道：

```rust
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    // SAFETY：每个 warp 只有 lane 0 写入；warp 索引是唯一的
    unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
}
```

安全义务与 `ThreadIndex` 系统自动强制的那一套相同：索引在范围内，并且没有两个线程共享同一个索引。区别在于，这一回是你自己去证明它，而不是让类型系统替你完成。

这是按每次访问来算的。要想一次性去掉整个 kernel 中普通 `a[i]` 索引背后的检查，请参见[边界检查：一次付清，或者选择退出](bounds-checks.md)中的 `unchecked_indexing` 标志。

---

## Tier 3：原始硬件

栈的最底层是原始硬件 intrinsic —— 那些直接与特定 GPU 架构上特定功能单元对话的 API。每一次调用都是 `unsafe`，安全契约复杂且依赖架构，而文档更多存在于 PTX ISA 手册里，而不是 Rust 的文档注释中。

| 特性                                 | 关键 API                                                      | 架构                   |
|:-------------------------------------|:--------------------------------------------------------------|:-----------------------|
| **TMA**（Tensor Memory Accelerator） | `tma_load_2d`, `tma_store_2d`, `TmaDescriptor`                | sm_90+（Hopper）       |
| **tcgen05**（Tensor Core 第 5 代）   | `tcgen05_mma`, `tcgen05_commit`, `TensorMemoryHandle`         | sm_100a（Blackwell DC）|
| **WGMMA**（Warpgroup MMA）           | `wgmma_mma_async`, `wgmma_commit_group`, `wgmma_wait_group`   | sm_90+（Hopper）       |
| **Cluster**                          | `cluster_rank`, `map_shared_rank`, `cluster_barrier_arrive`   | sm_90+（Hopper）       |
| **CLC**（Cluster Launch Control）    | `clc_prefetch`, `clc_query_channel`                           | sm_100+（Blackwell）   |
| **TMEM**（Tensor Memory）            | `TmemGuard`（typestate）, `tmem_alloc`, `tmem_dealloc`        | sm_100a（Blackwell DC）|

如果你在写应用层的 kernel，你应当不需要 Tier 3 的 API。它们是为那些在构建下一个 CUTLASS 的人准备的 —— 而对那些人来说，cuda-oxide 提供了与 CUDA C++ 中内联 PTX 相同的硬件访问能力，并且还额外有 Rust 的类型系统作为护栏（但不是强制的）。

> **参见**：[Tensor Memory Accelerator](../advanced/tensor-memory-accelerator.md)、[矩阵乘法加速器](../advanced/matrix-multiply-accelerators.md)以及 [Cluster 编程](../advanced/cluster-programming.md)，其中有 Tier 3 特性的详细覆盖。

---

## 借用检查器为你提供了什么

cuda-oxide 不是一个 DSL，也不是一个宏系统 —— 它在你的 kernel 代码上运行真正的 `rustc` 前端。这意味着 Rust 在 CPU 上提供的每一项安全保证，在 GPU 上同样被强制执行：

| 保证                                    | 工作方式                                                                                                                                                     |
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **所有权与借用**                       | 生命周期错误、释放后使用以及别名违规都在编译期被捕获                                                                                                        |
| **安全的并行写入**                     | `DisjointSlice<T>` + `ThreadIndex` + 匹配的 prepared 几何配置证明写入不会竞争                                                                               |
| **显式的 `unsafe` 作用域**             | 裸指针访问需要 `unsafe`，使相关义务可见且可审计                                                                                                              |
| **收敛属性的强制**                     | 同步原语（barrier、fence、shuffle）在 IR 中被标记为 `convergent`，防止优化器跨控制流移动或复制它们                                                            |

前三条是标准 Rust。第四条是 GPU 特有的：CUDA 的 `bar.sync`、fence 以及 warp shuffle 指令绝不能被编译器复制或重排。cuda-oxide 在 IR 中把它们标记为 `convergent`，好让 LLVM 的优化 pass 不去动它们。

---

## 难题

Rust 的借用检查器是为单线程所有权设计的，并借助 `Send`/`Sync` 处理 CPU 上的并发。SIMT 执行引入了借用检查器从未被教导去推理的模式。下面诚实地列出 cuda-oxide 今天*尚未*强制的内容 —— 以及为什么这些问题是可以解决的。

### 线程发散的控制流

rustc 的 JumpThreading MIR 优化会把函数调用复制到 if 语句的两个分支中 —— 这在 CPU 上是一个完全健全的优化，但它破坏了 GPU 的 barrier 语义，因为 block 中的所有线程必须汇聚在同一条 `bar.sync` 指令上。cuda-oxide 目前对设备代码禁用了 JumpThreading（`-Z mir-enable-passes=-JumpThreading`）。一个妥善的解决方案是让编译器了解收敛性要求，从而能够围绕这些要求进行优化，而不是完全禁用该 pass。

### 共享内存的访问模式

借用检查器无法推理线程 0 写 `smem[0]` 与线程 1 写 `smem[1]` 是否安全 —— 它看到的是 `&mut smem`，然后拒绝。`DisjointSlice` 解决了唯一索引写入这一模式，但没有解决协作式模式，例如归约、扫描，或者生产者/消费者流水线，在这些模式中，多个线程会有意地访问相互重叠的区域，并在各个阶段之间进行同步。

### warp 级收敛

像 `shfl_sync` 与 `ballot_sync` 这样的操作要求参与掩码中点名的所有线程在调用点上确实处于收敛状态。类型系统今天还无法强制这一点。如果线程已经发散，而你传入了完整的掩码，你会得到一个静默挂起 —— 这是最糟糕的一类 bug，因为没有崩溃也没有错误信息，只有一个永远跑不完的 kernel。

### 内存空间感知

GPU 内存有不同的地址空间 —— global、shared、local、TMEM。指向共享内存的 `&mut` 对 block 中的每个线程都可见；指向本地内存的 `&mut` 则是一个线程私有的。借用检查器把它们一视同仁。这是保守的（它会拒绝一些本来安全的程序），但从不导致不健全（它不会接受不安全的程序）。尽管如此，一个能感知内存空间的借用检查器可以接受更多无需 `unsafe` 的程序。

### 为什么这些问题是可解的

这些构件块在 Rust 的类型系统中已经存在。它们需要被扩展，而不是被重新发明：

| 想法                                  | 它能解决什么                                                                                                                                                      |
|:--------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **感知执行资源的类型**                 | 用执行层级（grid / block / warp / thread）标注函数。位于发散分支内部的 barrier 调用变成编译期错误。                                                              |
| **内存视图**                          | 泛化的并行访问模式 —— 类似 `DisjointSlice`，但覆盖分块、跨步、转置以及组合式布局。在大规模上以类型检查保证无竞争。                                                |
| **面向同步的扩展借用检查**             | 静态地强制 barrier 不会被遗忘、不会出现在发散的控制流中、也不会被优化器复制。让收敛性进入类型系统。                                                              |

内存视图这一想法的第一批成果已经落地：面向运行时尺寸矩阵的行视图与列视图，以及运行时宽度的输出 tile。[边界检查：一次付清，或者选择退出](bounds-checks.md)介绍了它们。

所有这些都是编译期分析。生成的 PTX 与你手工写出来的完全相同 —— 安全网在代码生成时就消失了。零运行时开销。

cuda-oxide 有能力逐步交付这些能力。真正的 `rustc` 借用检查器已经在设备代码上运行。IR 基础设施（pliron 方言）支持 GPU 感知的分析 pass。从 MIR 到 PTX 的完整编译流水线都在我们的掌控之中。而且每一项新的安全检查都是增量的 —— 已有的 kernel 继续能编译，同时新的 kernel 获得更强的保证。

---

## 编写安全的 kernel：速查表

### 默认路径

对大多数 kernel，从这里开始：

```rust
#[kernel]
pub fn my_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
    if let Some((out, idx)) = output.get_mut_indexed() {
        *out = transform(input[idx.get()]);
    }
}
```

对于常量 stride 的 2D 情形，给切片加上参数，并索取常量索引：

```rust
#[kernel]
pub fn tile_kernel(mut output: DisjointSlice<f32, Index2D<1024>>) {
    if let Some((out, _idx)) = output.get_mut_indexed() {
        *out = ...;
    }
}
```

规则如下：

- 对所有可变输出使用 `DisjointSlice`。
- 对所有只读输入使用 `&[T]`。
- 对 1D grid，声明 `launch_contract(domain = 1, ...)` 并使用 prepared 启动。在 kernel 内部，默认使用 `get_mut_indexed()`。如果你需要该索引来对多个切片做算术运算，就退回到显式的成对形式：`let idx = thread::index_1d(); slice.get_mut(idx)`。
- 对常量 stride 的 2D grid，把切片参数化为 `DisjointSlice<T, Index2D<S>>`，并使用 `get_mut_indexed()` 或 `thread::index_2d::<S>()`。stride 不匹配是编译错误。
- 对运行时 stride，改用 `thread::index_2d_runtime(&slice)`，配合带 `Runtime2DIndex` 标记的切片。行宽随切片一起传递，由宿主绑定一次，因此不需要 `unsafe`，也不存在可以互相不一致的每次调用 stride。
- 始终通过 `get_mut()` / `get_mut_indexed()` 做边界检查（两者都返回 `Option`）。

如果 kernel 主体在不用 `unsafe` 的情况下编译通过，它就满足了证明中设备侧的那一部分。无竞争还要求有一个匹配的 prepared 启动，或者对原始配置给出一个显式的 unsafe 证明。

### 何时需要 `unsafe`

| 模式                  | 原因                                          | 缓解措施                                                            |
|:---------------------|:----------------------------------------------|:--------------------------------------------------------------------|
| 共享内存              | 多个线程访问同一个 `static mut`               | 在跨线程读取之前用 `sync_threads()` 同步                            |
| warp shuffle          | 线程收敛不由编译器检查                        | 整 warp 操作用 `FULL_MASK`；为部分掩码写文档                        |
| 原子操作              | 从裸指针构造                                  | 包进一个辅助函数；原子操作本身是安全的                              |
| 非一致写入            | 并非每个线程都写入自己的索引                  | 使用 `get_unchecked_mut`，并附上成文的唯一性论证                    |
| 硬件 intrinsic        | 复杂且依赖架构的契约                          | 遵循 PTX ISA 文档；在目标硬件上测试                                 |

### `SAFETY` 注释

对每一个 `unsafe` 块，都要记录*为什么*这些不变式成立。不是记录代码做了什么 —— 代码本身就说明了 —— 而是记录为什么这一处特定的用法是安全的：

```rust
// SAFETY：每个 warp 只有 lane 0 执行这个分支。
// warp 索引（gid / 32）在各 warp 之间是唯一的，因此没有两个
// 线程会写入同一个输出元素。
if warp::lane_id() == 0 {
    let warp_idx = gid.get() / 32;
    unsafe { *partial_sums.get_unchecked_mut(warp_idx) = warp_sum; }
}
```

这不是形式主义。当某个 kernel 在凌晨两点发生数据竞争，而你正盯着 `compute-sanitizer` 的日志时，过去的你写下的安全注释是通往 bug 的最快路径。

> **提示**：如果你无法为某个 `unsafe` 块写出一个有说服力的 `SAFETY` 注释，那说明这个不变式实际上并没有被维持住。重构代码直到论证变得显然，或者改用安全的 API。

---

## 小结

| 性质                                                            | 状态                                                 |
|:----------------------------------------------------------------|:-----------------------------------------------------|
| 设备代码上的借用检查器                                          | 已强制（真正的 `rustc` 前端）                        |
| 安全的 1D 并行写入（`DisjointSlice + index_1d`）                | 通过 `domain = 1` 的 prepared 启动强制                |
| 安全的 2D 并行写入 —— 常量 stride                               | 已强制（`Index2D<S>` 不匹配是编译错误）              |
| 安全的 2D 并行写入 —— 运行时 stride                             | 已强制（宿主绑定行宽，按切片解析）                   |
| `ThreadIndex` 不可跨线程转移（共享内存洗白）                     | 已强制（`!Send + !Sync + !Copy + !Clone + 'kernel`） |
| `&mut [T]` 作为 kernel 参数                                     | 未强制 —— 把任何 `&mut` 参数都当作 `unsafe`          |
| 共享内存、intrinsic 需要显式 `unsafe`                           | 已强制（Rust 语言规则）                              |
| 同步原语上的收敛属性                                            | 已强制（IR 层面的 `convergent` 标记）                |
| warp 操作的线程收敛                                             | 未强制（运行时义务）                                 |
| 内存空间感知（共享 vs 全局）                                    | 未强制（未来工作）                                   |

`&mut [T]` 作为 kernel 参数是下一个尚未填补的缺口：宏今天接受这个类型，但运行时布局（每个线程看到的都是同一个底层指针）意味着从两个不同线程经 `&mut data[i]` 写入，正是 `DisjointSlice` 存在所要防范的那类别名。在宏直接拒绝 `&mut [T]`（或把它改写为 `DisjointSlice`）之前，请把任何接受它的 kernel 当作其中每一行都是 `unsafe` 来对待。

安全模型的设计目标是：通过 kernel 主体加上 prepared 启动，让常见情形是安全的，同时为其他一切提供显式的逃生通道。只有在你要提供编译器与启动契约都不携带的证明时，才使用 `unsafe`。
