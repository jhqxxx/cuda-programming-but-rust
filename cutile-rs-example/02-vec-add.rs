use std::sync::Arc;

use anyhow::Result;
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
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
        let tile_x = x.load_like(z); // 按 z 的分区方式从 x 中加载一个tile
        let tile_y = y.load_like(z); // 按 z 的分区方式从 y 中加载一个tile
        z.store(tile_x + tile_y); // 相加并存储结果
    }
}

fn main() -> Result<()> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let x: Arc<Tensor<f32>> = ones::<f32>(&[32, 32]).sync_on(&stream)?.into();
    let y: Arc<Tensor<f32>> = ones::<f32>(&[32, 32]).sync_on(&stream)?.into();
    let z = zeros::<f32>(&[32, 32]).sync_on(&stream)?.partition([4, 4]);
    let (z, _x, _y) = vec_add_module::add(z, x, y).sync_on(&stream)?;
    let z_host = z.unpartition().to_host_vec().sync_on(&stream)?;
    assert_eq!(z_host.len(), 32 * 32);
    assert!(z_host.iter().all(|&v| v == 2.0), "有元素不等于 2.0");
    println!("z[0] = {:?} (expected 2.0)", z_host[0]);
    Ok(())
}
