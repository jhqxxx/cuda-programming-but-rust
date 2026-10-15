# 闭包与泛型

Rust 的零开销抽象 —— 泛型、闭包与 trait 约束 —— 在 GPU 上同样可用。这是 cuda-oxide
最具特色的能力之一：你可以编写一个能在任意数值类型上工作的泛型 kernel，或者从宿主传入
一个闭包来自定义 GPU 行为，而这一切都没有运行时开销。

## 泛型 kernel

kernel 可以像任何 Rust 函数一样，对类型、const 值和 trait 约束进行泛型化。编译器会把
每个特化版本单态化为一个独立的 PTX 入口点：

```rust
use core::ops::Mul;
use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(
    factor: T,
    input: &[T],
    mut out: DisjointSlice<T>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[i] * factor;
    }
}
```

### PTX 命名

每个单态化都会产生一个独立的 PTX 入口点。非泛型 kernel 保留其普通的函数名。泛型 kernel
（包括闭包泛型 kernel）会获得一个 `_TID_<hex32>` 后缀，其中 `<hex32>` 是 rustc 对具体
生成的 kernel 函数项的稳定 type-id 哈希，以 32 个小写十六进制字符表示：

| 实例化                    | PTX 入口点名称       |
|:--------------------------|:---------------------|
| `vecadd`（非泛型）        | `vecadd`             |
| `scale::<f32>`            | `scale_TID_<hex32>`  |
| `scale::<MyType>`         | `scale_TID_<hex32>`  |
| `tile::<4>`               | `tile_TID_<hex32>`   |
| `tile::<8>`               | `tile_TID_<hex32>`   |
| `map::<f32, _>`（闭包）   | `map_TID_<hex32>`    |

宿主启动器和设备后端会在同一次统一构建中，向固定版本的 rustc 工具链询问同一个 `FnDef`
哈希，因此这些字符串逐字节一致。函数项包含其定义以及所有有序的类型参数和 const 参数。
因此，无论泛型元数（arity）是多少，线上名称（on-wire name）都保持定长。借用生命周期在
哈希之前已被擦除，所以它们不会产生重复的 GPU 代码。

请把该后缀视为构建产物，而不是永久 ABI：宿主代码和 PTX 必须与 cuda-oxide 固定版本的
rustc 工具链一起构建。

### 启动泛型 kernel

启动时，在生成的带类型方法上指定类型参数。这会强制进行具体的实例化，并让加载器查找
匹配的 PTX 入口点：

```rust
use cuda_core::simt::LaunchConfig;

// 安全性：config 是一维的，且两个缓冲区都覆盖每个被启动的索引。
unsafe {
    module.scale::<f32>(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        2.0f32,
        &input_dev,
        &mut output_dev,
    )
}
.expect("Launch failed");
```

生成的方法会强制对 `scale::<f32>` 进行单态化，因此即使该实例化从未在 CPU 上被直接调用，
它也会出现在编译出的 PTX 中。

### 编译期 kernel 策略

策略类型把某个泛型 kernel 的调优选择归拢在一起：

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::LaunchConfig1D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};

trait TransformPolicy {
    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum Small {}
impl TransformPolicy for Small {
    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
    const UNROLL: u32 = 2;
}

enum Wide {}
impl TransformPolicy for Wide {
    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
    const UNROLL: u32 = 4;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub fn transform<P: TransformPolicy>(mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        if let Some(value) = output.get_mut(index) {
            let mut step = 0;
            #[unroll(P::UNROLL)]
            while step < 8 {
                *value = value.rotate_left(1);
                step += 1;
            }
        }
    }
}

let launch = module
    .prepare_transform::<Small>(LaunchConfig1D::new(1, 64, 0))
    .expect("valid Small launch");
module
    .transform::<Small>(&stream, &launch, &mut output)
    .expect("launch Small kernel");

let wide_launch = module
    .prepare_transform::<Wide>(LaunchConfig1D::new(1, 256, 0))
    .expect("valid Wide launch");
module
    .transform::<Wide>(&stream, &wide_launch, &mut output)
    .expect("launch Wide kernel");
```

`Small` 和 `Wide` 会产生各自独立的 PTX kernel。策略不是 kernel 参数，GPU 也不会在运行时
选择策略。

策略还可以携带仅含元数据的描述：

```rust
use cuda_device::config::{Atom, AtomKind, Block, Global, RowMajor, Shape1, Thread, Tile};

type BlockTile = Tile<Shape1<1024>, RowMajor, Global, Block>;

enum Rotate {}
impl AtomKind for Rotate {}
type ElementOp = Atom<Rotate, Shape1<1>, Thread>;
```

`BlockTile` 描述由 block 处理的 1,024 个行主序全局内存元素。`ElementOp` 命名由 thread
处理的一元素操作；kernel 库决定 `Rotate` 做什么。这两个类型都不会分配内存、计算地址或
执行操作。

`MAX_THREADS` 是最大值，不是精确的 block 大小。为 `Small` 准备的启动配置每 block 至多
接受 64 个 thread；`Wide` 至多接受 256 个。只有在形状必须精确时，才在 `#[launch_contract]`
中使用 `block = (x, y, z)`。`MIN_BLOCKS` 是给编译器的占用率提示，不是所请求的 grid 大小。

`UNROLL = 4` 把四次循环迭代分组。它不会在四次迭代之后停止循环，剩余的迭代仍然会被处理。

泛型策略表达式目前需要 Rust 的 `generic_const_exprs` feature。在带 contract 的
`launch_bounds` 最大值中使用的其他具名常量必须在模块作用域可见。

### const 泛型 kernel

const 参数可以用于 kernel 和设备函数入口点：

```rust
#[kernel]
pub fn add_value<const VALUE: u32>(mut output: DisjointSlice<u32>) {
    let index = thread::index_1d();
    if let Some(element) = output.get_mut(index) {
        *element += VALUE;
    }
}
```

```text
add_value::<4> -> add_value_TID_<hash A>
add_value::<8> -> add_value_TID_<hash B>
```

在生成的启动方法上使用普通的 Rust turbofish 语法：

```rust
// 安全性：config 与各 kernel 的一维索引及输出边界相匹配。
unsafe {
    module.add_value::<4>(&stream, config, &mut output)?;
    module.add_value::<{ Config::VALUE }>(&stream, config, &mut output)?;
}
```

较旧的 `#[kernel(f32, i32)]` 便捷形式只支持一个类型参数。对于 const 泛型或混合泛型，
请使用裸 `#[kernel]`，并在启动处进行特化。

## 作为 kernel 参数的宿主闭包

cuda-oxide 支持把闭包从宿主传递到 GPU。这实现了强大的 `map` 风格模式，其中 kernel 的
行为由函数参数化：

```rust
#[kernel]
pub fn map<F: Fn(i32) -> i32>(f: F, input: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[i]);
    }
}
```

用闭包启动：

```rust
let factor = 3i32;
// 安全性：config 是一维的，且两个缓冲区都覆盖每个被启动的索引。
unsafe {
    module.map::<_>(&stream, config, move |x| x * factor, &input_dev, &mut output_dev)
}
.expect("Launch failed");
```

### 闭包参数如何传递

闭包作为单个值经过启动过程传递 —— 而不是作为一列捕获字段。启动器只推入一个驱动参数
（整个闭包结构体，包括其捕获），kernel 则把它作为一个 byval `.param` 接收：

```text
宿主          factor = 3i32; cl = move |x| x * factor
              推入一个驱动参数 ─► 闭包结构体 { factor: i32 }

GPU kernel    .entry map_TID_<hex>(
                .param .align 4 .b8 f[4],    ; 一个 byval 闭包
                .param .u64 input_ptr,        ; slice 仍是 (ptr, len)
                .param .u64 input_len,
                ...
              )
```

Slice 保持其 `(ptr, len)` 展平形式，因为这种形状由宿主启动辅助函数和 PTX 入口点布局
共同使用。只有按值传递的聚合参数（闭包和按值传递的用户结构体）会在 kernel 边界上作为
单个 byval 落地。

没有捕获的闭包是零大小类型 —— 后端会完全去掉该 `.param`，而宿主启动器知道要跳过它，
从而保持数据包对齐。

### 闭包的 PTX 命名

闭包泛型 kernel 会获得与其他泛型 kernel 相同的 `_TID_<hex32>` 后缀。闭包的匿名类型是
具体函数项的一部分，因此两个不同的闭包字面量 —— 即使它们具有相同的 `Fn` 签名 —— 也会
产生两个不同的入口点：

| 闭包                                    | PTX 入口点        |
|:----------------------------------------|:------------------|
| `move \|x\| x * factor`（一个捕获）     | `map_TID_<hex_a>` |
| `move \|x\| x + offset`（一个捕获）     | `map_TID_<hex_b>` |

## move 闭包与引用闭包

`move` 关键字决定捕获如何被转移到 GPU：

### move 闭包（推荐的默认选择）

```rust
let factor = 3i32;
move |x| x * factor   // `factor` 被复制进闭包结构体
```

- 闭包结构体按值持有捕获（`{ factor: i32 }`）。
- kernel 把 `factor` 作为 byval 闭包的普通字段读取。
- 宿主变量可以在启动之后被丢弃。
- 在所有系统上都能工作 —— 不需要特殊硬件支持。

### 引用闭包（HMM）

```rust
let factor = 3i32;
|x| x * factor   // 闭包捕获 &factor
```

- 闭包结构体包含一个指向 `factor` 的**宿主指针**
  （`{ factor: &i32 }`）。
- 整个闭包仍作为一个 byval 参数传递；kernel 通过**硬件管理内存
  （Hardware-Managed Memory, HMM）**解引用那个宿主指针，HMM 会在访问时迁移宿主页。
- 在 kernel 完成之前，宿主变量**必须保持存活**。
- 需要 HMM 支持（Turing 及以上 GPU、Linux 6.1.24+、CUDA 12.2+）。

### 何时用哪种

| 场景                                          | 使用                                                              |
|:----------------------------------------------|:------------------------------------------------------------------|
| 小型标量捕获（数值、布尔值）                  | `move`（零复制开销）                                              |
| 大型结构体捕获                                | 若 kernel 会多次读取，用 `move`；若很少访问，用 HMM               |
| 原型开发                                      | 两者都行；`move` 可移植性更好                                     |
| 宿主与设备之间的共享可变状态                  | 引用（HMM）—— 但要注意同步                                        |

> **提示**
> 拿不准时，就用 `move` 闭包。它们更容易推理、到处都能工作，并且避免了共享宿主/设备内存
> 的同步风险。

## kernel 内闭包

完全在设备代码中定义并调用的闭包按普通 Rust 语义工作 —— 不涉及 host/device ABI，因为
一切都已经在 GPU 上：

```rust
#[kernel]
pub fn apply_transform(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    let transform = |x: f32| -> f32 {
        let clamped = if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };
        clamped * clamped
    };

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = transform(input[idx.get()]);
    }
}
```

kernel 内闭包会被编译器内联，零开销。它们适合在 kernel 内部把逻辑提取出来，而不必引入
单独的设备函数。

## 跨 crate 的 kernel

kernel 可以定义在库 crate 中，并从二进制 crate 启动：

```rust
// 在 lib crate `my_kernels` 中：
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
```

```rust
// 在二进制 crate 中：
use my_kernels::kernels;

let module = kernels::load(&ctx)?;
// 安全性：config 是一维的，且三个缓冲区都覆盖每个被启动的索引。
unsafe { module.vecadd(&stream, config, &a, &b, &mut c) }
    .expect("Launch failed");
```

编译器通过 `#[kernel]` 生成的标记 trait 处理跨 crate 的 kernel 发现。带类型的模块在
编译期解析 PTX 名称，并缓存已加载的函数句柄。

> **提示**
> 对于泛型的跨 crate kernel，单态化发生在**调用方** crate（那里知道具体类型），因此
> PTX 会作为该二进制文件编译的一部分生成。
