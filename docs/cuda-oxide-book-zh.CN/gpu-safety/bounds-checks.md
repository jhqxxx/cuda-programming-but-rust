# 边界检查：一次付清，或者选择退出

kernel 中的每一个 `a[i]` 背后都藏着一个安全检查。编译器把它变成“`i` 是否落在切片内？如果不在，就停止这个 kernel” —— 每次访问、每个线程都要做一次比较和一次分支。在 CPU 上你很少会注意到。在 GPU 的热循环里你会注意到：`gemm` 示例中的朴素 GEMM 把内层循环的大部分时间花在每次乘加的两个这样的检查上，在 RTX 5090 上大约达到 2,940 GFLOPS。同一个 kernel 去掉这些检查后能跑到 7,160 GFLOPS。

本章讨论如何在不放弃安全性的前提下缩小这个差距。它建立在[安全模型](the-safety-model.md)之上：你已经知道 `DisjointSlice` 让并行写入无竞争，而受检查的 `PreparedLaunch` 在宿主上证明启动几何配置。同样的思路 —— 在正确的地方把一个事实检查一次，并把证明携带在类型里 —— 可以扩展到读取以及缓冲区大小上。

从最安全到最粗鲁的三件工具：

| 工具                              | 作用                                                     | 安全性            |
|:----------------------------------|:---------------------------------------------------------|:------------------|
| 读视图                            | 把一整行或一整列检查一次，之后读取无需检查                | 完全安全          |
| 启动契约上的 `requires`           | 每次启动在 CPU 上把缓冲区大小检查一次                     | 完全安全          |
| `#[kernel(unchecked_indexing)]`   | 删除索引检查并信任你                                      | 若你判断错误则是 UB|

---

## 边界检查的代价

A 的一行与 B 的一列做点积看起来无害：

```rust
let mut sum = 0.0f32;
let mut i = 0;
while i < k {
    sum += a[row * k + i] * b[i * n + col];
    i += 1;
}
```

数一数每次迭代实际执行了什么：

```text
i < k              循环自身的退出测试            （必需）
row*k + i < a.len  边界检查，失败则分支到 trap   （开销）
load a
i*n + col < b.len  边界检查，失败则分支到 trap   （开销）
load b
fma
```

三个比较中有两个是开销，而且它们还把循环体拆成了多个基本块，从而阻碍了展开。编译器自己无法删除它们：它把 `a.len()` 看成一个不透明的数字，没有办法知道 `row * k + i` 会保持在其之下。*你*知道 —— 因为缓冲区是你确定大小的 —— 但这个事实存在于另一个程序里。本章余下的部分讲的就是如何把这个事实交给编译器。

---

## 读视图：一整条带只检查一次

矩阵在 GPU 内存中就是一个长数组，一行接一行。**行宽**（API 中称为 `stride`）就是一行占据多少元素。元素 `(row, col)` 位于扁平索引 `row * stride + col`：

```text
 3 x 4 矩阵，stride = 4：

  col:      0   1   2   3
  row 0:  [ 0   1   2   3 ]     row(1) 覆盖扁平索引 4, 5, 6, 7
  row 1:  [ 4   5   6   7 ]     col(2) 覆盖扁平索引 2, 6, 10
  row 2:  [ 8   9  10  11 ]              （每步跳 4）
```

`MatrixView32` 把一个普通的 `&[f32]` 参数与它的行宽包在一起。向它索取一行或一列只执行**一次**检查 —— “整条带是否都落在切片内？” —— 并返回一个视图，通过它读取无需再做检查：

```rust
let a_mat = MatrixView32::new(a, k);   // A 有 m 行，每行宽 k
let b_mat = MatrixView32::new(b, n);   // B 有 k 行，每行宽 n

if let Some(a_row) = a_mat.row(row, k)      // 一次检查：A 的整行
    && let Some(b_col) = b_mat.col(col, k)  // 一次检查：B 的整列
{
    // 通过 a_row / b_col 的每一次读取现在都不带检查
}
```

如果检查失败，你得到 `None`，而且任何指针都不会被创建。这里没有任何 `unsafe`：读取不可能产生竞争，因此任意数量的线程都可以持有相互重叠的读视图。

### 融合点积：`zip_exact`

迭代单个视图本身就已经不带检查了：迭代器的“还有下一个元素吗？”测试*就是*循环的退出条件，而即便是裸指针的循环也需要它。但点积要迭代两个视图，而把两个迭代器 zip 起来通常会在每一步检查两侧。`zip_exact` 只检查**一次**，确认行与列长度相同，然后用单个计数器同时驱动两者：

```rust
if let Some(pair) = a_row.zip_exact(b_col) {
    let mut sum = 0.0f32;
    for (x, y) in pair {
        sum += x * y;
    }
}
```

最终生成出来的循环是：

```text
load a, load b, fma, advance, compare, branch
```

—— 与手写的裸指针循环形状完全一致。[`gemm_views`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_views) 示例字面上证明了这一点：它把每个安全 kernel 与一个 `unsafe` 的裸指针孪生版本并排编译，并验证 PTX 具有相同的分支数、相同的访存指令、循环中没有 trap，以及在 GPU 上逐位相同的结果。在 1024x1024x1024 上测得：

| Kernel               | GFLOPS | 备注                        |
|:---------------------|-------:|:-------------------------------|
| `gemm`（普通 `a[i]`）|  2,942 | 每次乘加两个检查               |
| 朴素视图（安全）      |  7,159 | 2.4 倍 —— 检查只付一次         |
| 朴素裸指针（`unsafe`）|  7,161 | 安全性的代价：约 0.1%          |

### 视图与 barrier

kernel 中使用 `sync_threads()` 时有一条规则：每个线程都必须到达 barrier，所以某个视图检查失败的线程不能提前返回。改为回退到一个空视图，让读取退化为填充值：

```rust
let a_row = a_mat.row(row, k).unwrap_or(RowView32::empty());
// ... 稍后，在搬运循环中：
tile[idx] = a_row.get(i).unwrap_or(0.0);   // 一次比较，边缘处补零
```

`get` 的代价只是一次比较 —— 它是给边缘用的访问器，不是给热循环用的。`gemm_views` 中的分块 kernel 用的正是这个模式。

---

## 写入结果：行宽为运行时的 tile

*写入*的安全方式仍然是 `DisjointSlice`，如[安全模型](the-safety-model.md)所述。新增的是 `tile_2d32_rt`：一种行宽取运行时值（例如 `n`）的 2D tile，用于尺寸在编译期未知的输出矩阵：

```rust
// `c` 携带自己的行宽，由宿主绑定到同一个 `n`，
// 所以这次调用是安全的：没有每线程的宽度跨过这次调用。
if let Some(mut cell) = c.tile_2d32_rt(coord) {
    let previous = cell.at_const::<0, 0>().read();
    cell.at_const::<0, 0>().write(alpha * sum + beta * previous);
}
```

为什么宽度来自切片而不是来自一个参数？tile 互不相交的论证 —— “不同的线程坐标拥有互不重叠的矩形” —— 悄悄假设了所有线程对行宽的看法一致。两个看法不一致的线程描述的是覆盖同一块内存的两个不同 grid：在 1x1 tile 下，宽度为 5 时线程 `(1, 0)` 解析出扁平索引 `1*5 + 0 = 5`，而宽度为 100 时线程 `(0, 5)` 解析出 `0*100 + 5 = 5`。同一个元素，两个 `&mut`，一次数据竞争。

早期的一个设计把宽度作为 `unsafe` 的每次调用参数传入，但没有任何调用点义务能堵上这个漏洞：安全代码可以在一个随线程而变的条件下在两个“一致”的宽度之间做选择。所以行宽改放在切片里。宿主通过 `cuda_host::RowWidth` 在本次启动中把它绑定一次，设备代码无法选择它，而每个寻址 `c` 的线程按构造读到的都是同一个宽度。当宽度*确实*在编译期已知时，改用 `tile_2d32`，不匹配就会变成类型错误。

---

## `requires`：在启动时承诺大小

视图之所以要在运行时针对 `a.len()` 检查条带，是因为没有别的东西保证缓冲区足够大。但宿主*知道*这些大小 —— 缓冲区就是它分配的。启动契约上的 `requires` 键把这份知识写下来，就写在 kernel 旁边：

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
    requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
pub fn sgemm_naive_views(m: u32, n: u32, k: u32, alpha: f32,
    a: &[f32], b: &[f32], beta: f32,
    mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>) {
```

生成出来的启动器会在每次启动前、在任何东西触碰 GPU 之前，在 CPU 上把每一组关系求值一次。违反某个关系的启动不会在 kernel 运行到一半时出错 —— 它会返回一个点名了那条承诺的错误：

```text
sgemm_naive_views: size requirement `a.len() >= m * k` violated:
left-hand side is 8192, right-hand side is 16384
```

这些关系被刻意设计成*不是*任意 Rust。每一条都是由切片参数写成的 `name.len()`、无符号整数标量参数、整数字面量、括号以及 `+ - *` 构成的单个比较。有两个原因：

1. 每个标识符都会对照 kernel 实际的参数列表做校验，所以打错字是编译错误，而不是拿错误的值去做检查。
2. 每个 `+`、`-`、`*` 都编译成受检查的 `u64` 算术，所以一个巨大的 `m * k` 无法回绕成一个小数字而错误地通过。

`_unchecked` 系列的启动器会跳过这些检查，正如它们跳过几何配置检查一样；维护这些关系就成了它们 `SAFETY` 契约的一部分。

---

## 逃生通道：`unchecked_indexing`

有时候没有哪种视图能匹配这种访问模式，而你已经验证过这个 kernel 是正确的。针对这种情况，有一件粗鲁的工具：

```rust
#[kernel(unchecked_indexing)]
pub fn hot_loop(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d().get();
    if let Some(out) = c.get_mut(thread::index_1d()) {
        *out = a[idx] + b[idx];   // 没有检查，没有 trap
    }
}
```

这个标志会删除该 kernel 中每一个 `a[i]` 背后的检查。契约与 `get_unchecked` 相同：你承诺每个索引都在范围内，而如果你错了，结果就是未定义行为 —— 没有 trap，可能静默地破坏无关的内存。它不要求 `unsafe` 关键字，但请把它当作要求了来对待。

还有一个面向实验的整体构建开关：

```console
$ CUDA_OXIDE_UNCHECKED_INDEXING=1 cargo oxide build my_example
$ cargo oxide build my_example --unchecked-indexing   # 同样的效果
```

精确地界定它的作用范围：

- 只有索引检查（`a[i]`）会被移除。算术溢出、除以零以及所有其他安全检查仍然会 trap。范围索引（`&a[i..j]`）也仍然会 trap。
- 每 kernel 的标志覆盖该 kernel 的主体，包括内联进它的函数。它从不泄漏：一个 kernel 调用了别的被标记 kernel 的辅助函数，它自己仍然保留检查。
- 只在带检查的构建通过之后再打开它，最好是在 `compute-sanitizer` 之下。 [`unchecked_indexing`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/unchecked_indexing) 示例展示了带与不带该标志的 PTX。

---

## 在它们之间做选择

```text
视图能否表达这种访问模式（行、列、tile、点积）？
├── 能 -> 使用视图 + 一个 `requires` 契约。安全，而且实测与裸指针一样快。
└── 不能 -> 这个 kernel 是否已被验证为正确（sanitizer、测试）？
    ├── 是 -> #[kernel(unchecked_indexing)]，像对待 unsafe 代码那样为它写文档。
    └── 否 -> 保留这些检查。带 trap 的 2,900 GFLOPS 胜过
              7,000 GFLOPS 的垃圾结果。
```

## 小结

- `a[i]` 每次访问都要付出一次比较和一次分支的代价，而在热循环中这就是主要开销 —— 在朴素 GEMM 上是 2.4 倍。
- 读视图（`MatrixView32::row` / `col`、`zip_exact`）把一整条带检查一次，并让其中的每次读取都不带检查。完全安全，而且 PTX 与手写的 unsafe 代码一致。
- `requires = (...)` 把缓冲区大小检查移到宿主上：每次启动得到一条带类型的错误，而不是 kernel 运行到一半时 trap。
- `#[kernel(unchecked_indexing)]` 直接删除索引检查。契约与 `get_unchecked` 相同；越界就是未定义行为。
