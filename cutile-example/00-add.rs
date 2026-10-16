//! cuTile Rust 学习示例：向量加法 z = x + y
//!
//! 运行：`cargo run --example add`
//!
//! 本文件的注释按位置记录了学习过程中的疑问与解答，涉及：
//!   - `const B: i32` 调用时为什么不写
//!   - `{[B]}` / `{[-1]}` 是什么
//!   - `partition([128])` 与 grid 的关系
//!   - 函数定义没写返回值，为什么调用时能解构出元组
//!   - 惰性 DeviceOp 与 `IntoDeviceOp`

use anyhow::Result;
use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    /// 逐元素加法。
    ///
    /// # 这里是 kernel，不是普通函数
    ///
    /// 函数体描述的是「**一个 tile program（= 一个 thread block）该干什么**」，
    /// 一次启动会在 grid 个 block 上并行执行 grid 份（本例 8 份）。
    ///
    /// # `const B: i32` 调用时为什么不写？
    ///
    /// 因为实际被调用的是宏生成的 **launcher**，不是你写的这个函数。launcher 的
    /// 形参变成了 `_A0: IntoDeviceOp<_Q0>` 这样的泛型，`B` 已不在其泛型列表里，
    /// 所以既不用写、也没法用 turbofish 指定。
    ///
    /// `B` 由 launcher 在调用点从实参推断（`cutile-macro` 文档：
    /// "Const scalars (`const N: i32`) - Inferred from tensor shapes"）。
    /// 本例 `z` 是 `partition([128])`，所以 **B = 128**。
    ///
    /// # `{[B]}` / `{[-1]}` 是什么？
    ///
    /// 是 `Tensor` 的第二个类型参数 —— **形状**，类型为 `const D: [i32; N]`：
    ///
    ///   - 数组的**长度** = 维度数（rank）
    ///   - 数组的**元素** = 该维的长度
    ///   - `-1` = 动态维（源码注释：`-1` in `D` marks a dynamic dim）
    ///
    /// 所以 `{[B]}` 是「1 维、长度 B」，`{[-1]}` 是「1 维、长度运行时定」，
    /// 二维则写作 `{[32, 32]}`。
    ///
    /// 别和宿主侧的 `cutile::tensor::Tensor<T>` 混淆：那个只有**一个**类型参数，
    /// 形状是运行时字段 `shape: Vec<i32>`。
    ///
    /// # 为什么 `z` 用 `B`，`x`/`y` 用 `-1`
    ///
    /// `z` 被 partition 过，tile 形状是编译期常量（设备代码要用它算偏移和 TMA
    /// 长度），必须具名；`x`/`y` 是整块传入的输入，长度在设备侧不是静态常量。
    #[cutile::entry]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        // `load_like(z)` = "load a tile **like** z"。实现里做的事（源码 _core.rs）：
        //   1. pid = get_tile_block_id()   —— 当前 block 的 id（即 blockIdx）
        //   2. tile_shape = z.shape()      —— tile 大小**从 z 借**（= [128]）
        //   3. 把整块 x 临时包成 tile 尺寸 128 的 partition 视图
        //   4. 取第 pid 块  →  x[pid*128 .. pid*128+128]
        //
        // 所以定位信息只有两个：block id（第几块）+ z 的 tile 形状（每块多大）。
        // x 自己的 `-1` 不参与定位，它只表示「这是长度未静态确定的整块输入」。
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        // store 与 load 对称：写到**当前 tile-block 负责的那一块**，即 z[pid*128 .. +128]。
        z.store(tx + ty);
    }
}

fn main() -> Result<()> {
    // `api::*` 返回的都是**惰性** DeviceOp（本质是 future），不驱动就没有 GPU 工作。
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);

    // `.sync()?` 先物化成具体的 `Tensor<f32>`（1024 个 0）；
    // `.partition([128])` 只是打上分区元信息（纯元数据，不产生 GPU 工作），
    // 返回具体的 `Partition`，所以下面可以直接 `.grid()`。
    //
    // 若省掉 `.sync()?`，得到的是 `DeviceOperationPartition` —— 它是个 DeviceOp，
    // 而 `grid()` 定义在具体 `Partition` 上，所以那样写会报 E0599。
    let z = api::zeros::<f32>(&[1024]).sync()?.partition([128]);

    // grid 由 partition 推导：每轴 = ceil(张量该轴 / partition 该轴)，本例 1024/128 = 8。
    // 返回三元组 (x, y, z)，未用到的轴填 1。
    //
    // 注意 `partition([128])` 里的 128 是「**每个 block 处理多少数据**」，
    // 不是 CUDA 的 `blockDim`（线程数）—— 线程/向量化/TMA 由后端决定，
    // 你只能通过 `num_worker_warps_per_cta` 这类 hint 表达偏好。
    //
    // 另外 partition 的条目数 = 张量的秩（每轴一个数）；二维要写成 `[bm, bn]`。
    assert_eq!(z.grid()?, (8, 1, 1));

    // `kernel::add` 是 launcher（宏生成），一次调用 = **一次 kernel 启动**，
    // 在 grid=8 个 block 上并行执行 8 份函数体 —— 不是 Rust 层面循环调用 8 次。
    //
    // 你写的 `add` 没写返回值，为什么这里能解构？
    // 因为 launcher 的返回类型是 `MyKernel<...>`（一个 DeviceOp），产出的是
    // 「实参原样收回」拼成的元组，生命周期是：
    //     输入 → prepare() → Stored → 执行 → recover() → Returned
    // 而 `Returned` 就是你传进去的那个类型（见 `KernelOutput` / `KernelInput`
    // 的文档表格）。元组顺序 = **形参声明顺序**，按位置对应，不按名字。
    //
    // 传 `Partition` 或传 Op 都可以：launcher 形参的约束是 `_A0: IntoDeviceOp<_Q0>`，
    // 而 `IntoDeviceOp` 有一个 blanket impl（所有 DeviceOp 原样透传）+ 若干为具体值
    // 准备的显式 impl（用 `Value` 包一层）。两条路都收敛成
    // `DeviceOp<Output = Partition<Tensor<f32>>>`，所以都能收。
    let (z, x, y) = kernel::add(z, x, y).sync()?;

    // device 张量住在显存里，不能直接 println!，要先拷回主机。
    // `to_host_vec()` 返回的是惰性的 DeviceOp（和 api::* 一样），要用 sync() 驱动。
    let x_host: Vec<f32> = x.to_host_vec().sync()?;
    let y_host: Vec<f32> = y.to_host_vec().sync()?;
    // `z` 回来时仍是 `Partition`（因为传进去的就是 Partition），
    // 先 unpartition() 还原成整块，再拷回主机。
    let z_host: Vec<f32> = z.unpartition().to_host_vec().sync()?;

    // 1024 个元素全打出来太长，先看前 8 个。
    println!("x[:8] = {:?}", &x_host[..8]);
    println!("y[:8] = {:?}", &y_host[..8]);
    println!("z[:8] = {:?}", &z_host[..8]);
    println!(
        "长度: x={} y={} z={}",
        x_host.len(),
        y_host.len(),
        z_host.len()
    );
    Ok(())
}
