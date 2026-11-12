use std::sync::Arc;

use anyhow::Result;
use cuda_async::device_operation::DeviceOp;
// 0.5.0：能 new_stream 的是 Context（执行上下文），不是 Device（物理设备身份）
use cuda_core::Context;
use cutile::api::{ones, zeros};
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};

#[cutile::module]
mod vec_add_module {
    use cutile::core::*;

    #[cutile::entry]
    fn add<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,        // 输出： 具有静态形状S
        x: &Tensor<f32, { [-1, -1] }>, // 输入： 动态形状 完整tensor
        y: &Tensor<f32, { [-1, -1] }>, // 输入： 动态形状 完整tensor
    ) {
        // load_like便捷函数会按与z相同的形状和坐标对x和y进行分区，返回一个tile view
        let tile_x = x.load_like(z); // 按 z 的分区方式从 x 中加载一个tile
        let tile_y = y.load_like(z); // 按 z 的分区方式从 y 中加载一个tile
        z.store(tile_x + tile_y); // 相加并存储结果
    }

    #[cutile::entry]
    fn add2<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
        w: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        let tile_w = w.load_like(z);
        z.store(tile_x + tile_y + tile_w);
    }
}

fn main() -> Result<()> {
    let ctx = Context::new(0)?;
    let stream = ctx.new_stream()?;
    // 分区把tensor切成一格一格的子区域，每一格由一个tile block处理
    // 只读输入在宿主侧以 &Tensor<T>、Tensor<T>、Arc<Tensor<T>> 或 &TensorView<T> 的形式传入
    //      不需要在宿主侧分区
    //      多个tile block 可以安全的读取同一块或相互重叠的区域
    //      没有独占访问的要求
    //      只读tensor在kernel内部进行分区
    let x: Arc<Tensor<f32>> = ones::<f32>(&[32, 32]).sync_on(&stream)?.into();
    let y: Arc<Tensor<f32>> = ones::<f32>(&[32, 32]).sync_on(&stream)?.into();
    // 可变 tensor 必须在 kernel 启动之前于宿主侧完成分区。
    //      这保证了每个 tile block 写入的都是一块互不相同的子区域
    //      从而满足 Rust 对可变内存的独占访问要求。
    // 创建一个初始化为全零的 32×32 输出 tensor。
    // 把它切成 4×4 的子 tensor。
    // 形成一个 8×8 的网格（每个维度上 32÷4 = 8）
    let z = zeros::<f32>(&[32, 32]).sync_on(&stream)?.partition([4, 4]);
    // 两路相加
    let (z, _x, _y) = vec_add_module::add(z, x.clone(), y.clone()).sync_on(&stream)?;
    let z_host = z.unpartition().to_host_vec().sync_on(&stream)?;
    assert_eq!(z_host.len(), 32 * 32);
    assert!(z_host.iter().all(|&v| v == 2.0), "有元素不等于 2.0");
    println!("z[0] = {:?} (expected 2.0)", z_host[0]);

    let w: Arc<Tensor<f32>> = ones::<f32>(&[32, 32]).sync_on(&stream)?.into();
    let z2 = zeros::<f32>(&[32, 32]).sync_on(&stream)?.partition([4, 4]);
    // 三路相加
    let (z2, _x2, _y2, _w) =
        vec_add_module::add2(z2, x.clone(), y.clone(), w.clone()).sync_on(&stream)?;
    let z2_host = z2.unpartition().to_host_vec().sync_on(&stream)?;
    assert_eq!(z2_host.len(), 32 * 32);
    assert!(z2_host.iter().all(|&v| v == 3.0), "有元素不等于 3.0");
    println!("z2[0] = {:?} (expected 3.0)", z2_host[0]);

    // 非正方形分区
    let z3 = zeros::<f32>(&[32, 32]).sync_on(&stream)?.partition([8, 4]);
    let (z3, _x3, _y3, _w3) =
        vec_add_module::add2(z3, x.clone(), y.clone(), w.clone()).sync_on(&stream)?;
    let z3_host = z3.unpartition().to_host_vec().sync_on(&stream)?;
    assert_eq!(z3_host.len(), 32 * 32);
    assert!(z3_host.iter().all(|&v| v == 3.0), "有元素不等于 3.0");
    println!("z3[0] = {:?} (expected 3.0)", z3_host[0]);
    Ok(())
}
