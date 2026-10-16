# 11. 使用 NVFP4/MXFP8 做推理

NVFP4 是 NVIDIA 的一种低精度推理格式，它用 4 位 E2M1 浮点数据加上独立的缩放 tensor 来表示模型取值。FP4 取值每字节打包两个，因此模型加载层需要交给 cuTile 两类信息：

- 打包后的 FP4 字节，以及它们的逻辑矩阵形状
- 缩放 tensor，以及它们的布局

cuTile 不要求特定的模型加载工具。分界是一个带类型的 Rust 值：打包的 FP4 字节成为 `f4e2m1fnx2`，逻辑 FP4 计算使用 `f4e2m1fn`，缩放 tensor 使用 `f8e4m3fn` 或 `f8e8m0fnu` 这样的 FP8 缩放类型。

示例使用一个线性层，并用 `mmaf_scaled` 做矩阵乘法。同一个操作也支持使用 FP8 数据和 E8M0 块缩放值的 MXFP8。

---

## 为什么要块缩放？

FP4 取值很紧凑，但自身的表示范围非常有限。块缩放让 FP4 载荷保持很小，同时给每一组固定大小的逻辑 FP4 取值配一个独立的缩放因子。在本教程使用的 NVFP4 E4M3 缩放布局中，这个比例是沿 K 方向每 16 个逻辑 FP4 值对应一个 FP8 缩放值。矩阵乘法同时消费这两部分：打包的 FP4 数据承载低精度取值，缩放 tensor 则恢复出有用的计算所需的表示范围。

16 个 `f16` 取值占 32 字节。在这个 NVFP4 布局下，同样 16 个逻辑取值只需要 8 个打包的 FP4 字节加上 1 个 FP8 缩放字节，合计 9 字节。这大约是 `f16` 操作数存储的 28%，也就是在还没计算输出流量的情况下，操作数带宽减少约 3.6 倍。

对于一个 `16 x 16 x 16` 的 GEMM，两个 `f16` 操作数 tile 需要 1024 字节：

```text
A[16, 16] f16 + B[16, 16] f16 = 512 bytes + 512 bytes
```

而对应的 NVFP4 操作数加 E4M3 缩放值只需要 288 字节：

```text
A packed FP4: 128 bytes + A scale values: 16 bytes
B packed FP4: 128 bytes + B scale values: 16 bytes
```

这 3.6 倍的缩减适用于输入操作数。输出和累加器仍然使用 `f32`。当推理需要反复读取大块激活和权重矩阵时，收益就体现出来了。

从概念上说，`mmaf_scaled` 计算的矩阵乘法中，每个操作数取值都要乘以它所属 K 组的缩放值：

```text
out[i, j] = acc[i, j] +
    sum over k:
        fp4_lhs[i, k] * lhs_scale[i, k / V] *
        fp4_rhs[k, j] * rhs_scale[k / V, j]
```

`V` 是沿 K 方向的缩放组大小。这里 `V = 16`，所以一个 `[BM, BK] @ [BK, BN]` 的 tile 使用 `[BM, BK_SCALES]` 个左侧缩放值和 `[BK_SCALES, BN]` 个右侧缩放值，其中 `BK_SCALES = BK / 16`。当 `BK = 64` 时，意味着沿 K 方向有四个缩放值。调用的形状是：

- `lhs`：`Tile<f4e2m1fn, [BM, BK]>`
- `rhs`：`Tile<f4e2m1fn, [BK, BN]>`
- `lhs_scale`：`Tile<f8e4m3fn, [BM, BK_SCALES]>`
- `rhs_scale`：`Tile<f8e4m3fn, [BK_SCALES, BN]>`
- `acc`：`Tile<f32, [BM, BN]>`

```rust
let out = mmaf_scaled(lhs, rhs, acc, lhs_scale, rhs_scale);
```

---

## 存储类型与计算类型

cuTile Rust 把打包存储与逻辑计算取值区分开来：

| Rust 类型 | 出现位置 | 含义 |
|---|---|---|
| `f4e2m1fnx2` | `Tensor<f4e2m1fnx2, ...>` 与 `Tile<f4e2m1fnx2, ...>` | 一个字节，内含两个 FP4 E2M1 取值 |
| `f4e2m1fn` | `Tile<f4e2m1fn, ...>` | 解包之后的一个逻辑 FP4 E2M1 取值 |
| `f8e4m3fn` / `f8e8m0fnu` | `Tensor` 或 `Tile` 的缩放取值 | 缩放式低精度 MMA 所使用的 FP8 块缩放格式 |

在 tensor 边界上，打包存储请用 `f4e2m1fnx2`；`f4e2m1fn` 出现在 kernel 内部解包之后。每个 `f4e2m1fnx2` 把第一个逻辑 FP4 值存放在低半字节，第二个逻辑 FP4 值存放在高半字节。半字节（nibble）就是一个字节的两个 4 位半区之一。

构造函数把布局写明白了：

```rust
use cutile::cuda_core::{f4e2m1fn, f4e2m1fnx2};

let byte = f4e2m1fnx2::from_bits(0xAB);
assert_eq!(byte.low(), f4e2m1fn(0x0B));
assert_eq!(byte.high(), f4e2m1fn(0x0A));

let pair = f4e2m1fnx2::from_nibbles(0x3, 0xC);
assert_eq!(pair.to_bits(), 0xC3);
```

---

## 推理数据布局

对于本教程中的这个简单线性层，推理数据这样安排：

| 逻辑量 | 存储类型 | 存储形状 |
|---|---|---|
| 激活 `[M, K]` | `f4e2m1fnx2` | `[M, K / 2]` |
| 权重 `[N, K]` | `f4e2m1fnx2` | `[N, K / 2]` |
| 激活缩放 | `f8e4m3fn` 或 `f8e8m0fnu` | 按本文所用布局为 `[M, K / 16]` |
| 权重缩放 | `f8e4m3fn` 或 `f8e8m0fnu` | 按本文所用布局为 `[N, K / 16]` |
| 输出 | `f32` | `[M, N]` |

cuTile 把 `f4e2m1fnx2` 取值当作已经打包好的字节来读取。模型加载器必须按正确的半字节顺序放置每一对取值。

本教程使用逻辑上按行主序的缩放 tensor，组大小 `V = 16` 如上所述。如果你的模型加载层使用不同的缩放 tensor 布局，请在启动之前把字节调整成你的 kernel 所期望的布局。

对于稠密线性层，逻辑操作是：

```text
output[M, N] = alpha * activations[M, K] @ weights[N, K].T
```

`alpha` 是普通的 `f32` 标量。如果你的模型数据带有全局的输入缩放或权重缩放，请在启动之前把它们合并进 `alpha`。

---

## 转换宿主字节

下面的代码从纯字节向量创建带类型的设备 tensor。

```rust
use cutile::api::{self, DeviceOpReshape};
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use std::sync::Arc;

fn packed_fp4(bytes: Vec<u8>) -> Arc<Vec<f4e2m1fnx2>> {
    Arc::new(bytes.into_iter().map(f4e2m1fnx2::from_bits).collect())
}

fn fp8_e4m3(bytes: Vec<u8>) -> Arc<Vec<f8e4m3fn>> {
    Arc::new(bytes.into_iter().map(f8e4m3fn).collect())
}

let m = 128usize;
let n = 256usize;
let k = 512usize;

// 自包含的示例数据。在真实模型里，这些字节来自模型加载路径
// 按上面的布局整理好数据之后的结果。
let activation_bytes = vec![0x22u8; m * k / 2];
let weight_bytes = vec![0x22u8; n * k / 2];
let activation_scale_bytes = vec![0x38u8; m * k / 16];
let weight_scale_bytes = vec![0x38u8; n * k / 16];

let x = api::copy_host_vec_to_device(&packed_fp4(activation_bytes))
    .reshape(&[m, k / 2]);
let y = api::copy_host_vec_to_device(&packed_fp4(weight_bytes))
    .reshape(&[n, k / 2]);
let x_scales = api::copy_host_vec_to_device(&fp8_e4m3(activation_scale_bytes))
    .reshape(&[m, k / 16]);
let y_scales = api::copy_host_vec_to_device(&fp8_e4m3(weight_scale_bytes))
    .reshape(&[n, k / 16]);

let alpha = 1.0f32;
```

把每个字节转换成打包 FP4 存储：

```rust
u8 -> f4e2m1fnx2::from_bits(byte)
```

这次转换只改变类型，不改变内存布局。若要自行打包 FP4 值，请使用 `f4e2m1fnx2::from_nibbles(low, high)`。

### Tensor<u8> 逃生通道

打包 FP4 输入请优先使用 `Tensor<f4e2m1fnx2, ...>`，因为这样 kernel 签名本身就说明了数据契约。如果某个互操作边界已经持有一个 `Tensor<u8>`、改类型不方便，cuTile 仍然支持底层路径：把字节当作 `u8` 加载，再用泛型的 `unpack` 生成逻辑 FP4 值。

```rust
#[cutile::entry()]
fn unpack_packed_fp4_bytes(
    output: &mut Tensor<f4e2m1fnx2, { [32] }>,
    input: &Tensor<u8, { [-1] }>,
) {
    let bytes = load_tile(input, shape![32], [0]);
    let f4s: Tile<f4e2m1fn, { [64] }> = unpack(bytes);
    let packed = f4s.pack(shape![32]);
    output.store(packed);
}
```

每一种 `u8` 位模式都是合法的字节存储，所以这种解包是安全的。

---

## kernel 模式

kernel 先加载打包字节，把它们解包成逻辑 FP4 矩阵 tile，再调用 `mmaf_scaled`：

```rust
#[cutile::module]
mod nvfp4_linear {
    use cutile::core::*;

    #[cutile::entry()]
    fn linear_tile<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        y: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        y_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid_m = program_id(0);
        let pid_n = program_id(1);

        let part_x = x.partition(shape![BM, BK_PACKED]);
        let part_y = y.partition(shape![BN, BK_PACKED]);
        let part_x_scales = x_scales.partition(shape![BM, BK_SCALES]);
        let part_y_scales = y_scales.partition(shape![BN, BK_SCALES]);

        let mut tile_z = constant(0.0f32, shape![BM, BN]);

        for k_tile in 0i32..num_tiles(&part_x, 1) {
            let tile_x_packed = part_x.load([pid_m, k_tile]);
            let tile_y_packed = part_y.load([pid_n, k_tile]);

            let tile_x = tile_x_packed.unpack(shape![BM, BK]);
            let tile_y = tile_y_packed.unpack(shape![BN, BK]).transpose();

            let tile_x_scales = part_x_scales.load([pid_m, k_tile]);
            let tile_y_scales = part_y_scales.load([pid_n, k_tile]).transpose();

            tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
        }

        let alpha_tile = broadcast_scalar(alpha, z.shape());
        z.store(tile_z * alpha_tile);
    }
}
```

解包会让元素数量翻倍：

```text
Tile<f4e2m1fnx2, [BM, BK_PACKED]>  // BK_PACKED = BK / 2
    unpack(shape![BM, BK]) -> Tile<f4e2m1fn, [BM, BK]>
```

不要把 `Tile<f4e2m1fnx2, ...>` 直接传给 `mmaf_scaled`。MMA 的操作数是解包后的逻辑 `Tile<f4e2m1fn, ...>`。该方法会在内部发出所需的秩为 1 的 Tile IR `unpack`，并把结果 reshape 回你所请求的 tile 形状。

---

## 启动形状

对于上面这个 kernel，`BK` 是逻辑上的 K tile 宽度。kernel 用 `num_tiles(&part_x, 1)` 遍历打包激活分区的各个 K tile；宿主侧传入 `BK`，以及静态 tile 形状中用到的、由它派生出的打包宽度和缩放宽度：

```rust
use cutile::api;
use cutile::tile_kernel::TileKernel;
use cutile::tensor::IntoPartition;

let bm = 16usize;
let bn = 16usize;
let bk = 64usize;
let bk_packed = bk / 2;
let bk_scales = bk / 16;
let z = api::zeros::<f32>(&[m, n]).partition([bm, bn]);

let (z, ..) = nvfp4_linear::linear_tile(
    z,
    x,
    y,
    x_scales,
    y_scales,
    alpha,
)
.generics(vec![
    bm.to_string(),
    bn.to_string(),
    bk.to_string(),
    bk_packed.to_string(),
    bk_scales.to_string(),
])
.sync()?;
```

---

## 架构取舍

打包 FP4 值可以在数据抵达 kernel 之前就减少操作数字节数。只要模型以打包形式存储和搬运，这就能降低访存流量。更大的加速来自目标 GPU 同时具备对块缩放 FP4 矩阵乘法的硬件支持 —— 此时编译器能把作用于逻辑 `f4e2m1fn` tile 的 `mmaf_scaled` 下降为 NVFP4 专用的 Tensor Core 指令。

CUDA Tile IR 13.3 总体支持 `sm_80` 及更新的目标，但这并不意味着每一种 Tile IR 类型或 MMA 操作在每个目标上都是原生的。带缩放浮点输入的 `mmaf_scaled` 是 `sm_100` 及更新的操作，而 `f4E2M1FN` 是 Blackwell 才支持的 Tile IR 类型。

| 目标系列 | 原生 `f4E2M1FN` 类型 | 专用的缩放 FP4 MMA |
|---|---|---|
| `sm_80` / Ampere | 否 | 否 |
| `sm_90` / Hopper | 否 | 否 |
| `sm_100` / Blackwell | 是 | 是，通过 `mmaf_scaled` 下降为 NVFP4 Tensor Core 指令 |
| `sm_120` / Blackwell | 是 | 是，通过 `mmaf_scaled` 下降为 NVFP4 Tensor Core 指令 |

没有原生支持的目标通常需要在 GEMM 之前解包或转换 FP4。这部分工作可能抵消带宽上的收益。请针对目标 GPU 把这条回退路径与成熟实现做基准对比。

仓库中可运行的示例是 `cutile-rs-examples/examples/nvfp4.rs`。在原生支持 NVFP4 的目标上，它会用全 1 输入的情形来校验 kernel 的输出。

---

## MXFP8 风格的 FP8

FP8 版本使用同一个 `mmaf_scaled` 操作，但不需要亚字节级的打包与解包。数据 tensor 每字节存一个 FP8 值，缩放 tensor 则沿 K 方向每 32 个 FP8 值配一个 E8M0 缩放值：

| 逻辑量 | 存储类型 | 存储形状 |
|---|---|---|
| 激活 `[M, K]` | `f8e4m3fn` | `[M, K]` |
| 权重 `[N, K]` | `f8e4m3fn` | `[N, K]` |
| 激活缩放 | `f8e8m0fnu` | `[M, K / 32]` |
| 权重缩放 | `f8e8m0fnu` | `[N, K / 32]` |
| 输出 | `f32` | `[M, N]` |

在 kernel 内部，直接加载 FP8 tile、加载 E8M0 缩放 tile，然后调用 `mmaf_scaled`：

```rust
let tile_x = part_x.load([pid_m, k_tile]);
let tile_y = part_y.load([pid_n, k_tile]).transpose();
let tile_x_scales = part_x_scales.load([pid_m, k_tile]);
let tile_y_scales = part_y_scales.load([pid_n, k_tile]).transpose();

tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
```

与 NVFP4 相比，主要的形状差异是：

```text
BK_PACKED 不再需要
BK_SCALES = BK / 32

lhs:       Tile<f8e4m3fn,  [BM, BK]>
rhs:       Tile<f8e4m3fn,  [BK, BN]>
lhs_scale: Tile<f8e8m0fnu, [BM, BK_SCALES]>
rhs_scale: Tile<f8e8m0fnu, [BK_SCALES, BN]>
```

仓库中可运行的示例是 `cutile-rs-examples/examples/mxfp8.rs`。在原生支持 MXFP8 的目标上，它会用全 1 输入的情形来校验 kernel 的输出。

这一模式背后的操作级参考，见[矩阵乘法](../reference/dsl-api.md#matrix-multiply)。
