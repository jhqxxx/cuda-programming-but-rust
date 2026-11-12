use anyhow::Result;
use cutile::prelude::*;

use crate::hello_world_module::hello_world_kernel;

// #[cutile::module] 把一个模块标记为包含GPU代码
#[cutile::module]
mod hello_world_module {
    use cutile::core::*;

    // #[cutile::entry] 把一个函数标记为kernel入口点
    #[cutile::entry]
    fn hello_world_kernel() {
        let _pid0 = program_id(0); // 本程序在网格轴 0 上的索引
        let _pid1 = program_id(1); // 本程序在网格轴 1 上的索引
        let _pid2 = program_id(2); // 本程序在网格轴 2 上的索引
        let n0 = num_programs(0); // grid 第 0 轴上程序的总数
        let n1 = num_programs(1); // grid 第 1 轴上程序的总数
        let n2 = num_programs(2); // grid 第 2 轴上程序的总数

        let _total = n0 * n1 * n2;
        cuda_tile_print!(
            "Hello from program <{}, {}, {}> in a grid of <{}, {}, {}> programs! {} total programs\n",
            _pid0,
            _pid1,
            _pid2,
            n0,
            n1,
            n2,
            _total
        );
    }
}

fn main() -> Result<()> {
    // 宿主侧代码负责准备 GPU、指定 kernel 的启动网格，并启动 kernel。
    //
    // cutile 0.5.0（cuda-core 0.5）把 0.4.0 的 Device 拆成了两个类型：
    //   Device  = 物理设备身份，不持有 CUDA context，不能建 stream
    //   Context = 执行上下文，持有 primary context，new_stream / default_stream 在它上面
    // 所以旧版那句 Device::new(0)?.new_stream() 现在对应的是 Context::new(0)?。
    let ctx = Context::new(0)?; // 连接 GPU 并保留其 primary context
    let stream = ctx.new_stream()?; // 创建一条工作队列
    let launcher = hello_world_kernel(); // 拿到kernel启动器
    launcher.grid((2, 2, 1)).sync_on(&stream)?; // 启动 2x2x1 = 4 个 tile program
    // let launcher = hello_world_kernel();
    // launcher.grid((3, 3, 1)).sync_on(&stream)?;  // 启动 3x3x1 = 9 个 tile program
    // let launcher = hello_world_kernel();
    // launcher.grid((2, 2, 2)).sync_on(&stream)?;  // 启动 2x2x2 = 8 个 tile program
    Ok(())
}
