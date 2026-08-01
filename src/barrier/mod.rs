use cuda_host::cuda_module;

/// 本模块演示 CUDA 中 mbarrier（内存屏障）的两种典型用法：
///
/// - [`barrier_sync`]：最简单的同步屏障，仅确保所有线程到达同一执行点。
/// - [`barrier_shared_data`]：结合共享内存，展示屏障保护下的跨线程数据交换。
///
/// 两个 kernel 都使用 `static mut` 在共享内存中声明 `Barrier`，
/// 并通过 `&raw` 裸指针绕开 Rust 借用规则来操作它。
#[cuda_module]
pub mod barrier_mod {

    use cuda_device::{
        Barrier, DisjointSlice, SharedArray,
        barrier::{mbarrier_arrive, mbarrier_init, mbarrier_inval, mbarrier_test_wait},
        kernel, thread,
    };

    /// 一个基于 mbarrier 的同步屏障 kernel。
    ///
    /// 启动配置：1 个 block × N 个线程（N = 256），无动态共享内存。
    ///
    /// 流程：
    /// 1. 线程 0 初始化 `BAR`，期望到达线程数为 `block_size`；
    /// 2. 所有线程调用 `mbarrier_arrive` 报告到达；
    /// 3. 每个线程自旋等待，直到全部 `block_size` 个线程都到达；
    /// 4. 将输出缓冲区 `out` 的对应位置写为 `1`。
    ///
    /// # 参数
    /// - `out`：全局输出缓冲区，每个线程写入 `1`，用于验证同步成功。
    #[kernel]
    pub fn barrier_sync(mut out: DisjointSlice<u32>) {
        // 静态声明Barrier， 所有线程共用，UNINIT表示尚未初始化
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // 线程0初始化BAR，设置期望的到达线程数为block_size
        // 意味着需要所有256个线程都到达，barrier才会释放
        if tid == 0 {
            unsafe {
                // &raw mut BAR: 裸指针，
                // 绕过 Rust 的借用规则，直接从内存位置创建裸指针
                mbarrier_init(&raw mut BAR, block_size);
            }
        }

        // 确保所有线程都看到了BAR的初始化
        thread::sync_threads();

        // 每个线程都调用arrive
        let token = unsafe { mbarrier_arrive(&raw const BAR) };

        // 每个线程等待，直到block_size个线程都arrive
        // 所有线程必须全部到达此处，才能继续往下执行
        unsafe { while !mbarrier_test_wait(&raw const BAR, token) {} }

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1u32;
        }

        // 线程0使BAR失效，清理资源
        if tid == 0 {
            unsafe {
                mbarrier_inval(&raw mut BAR);
            }
        }
    }

    /// 使用 mbarrier 保护共享内存数据交换的 kernel。
    ///
    /// 启动配置：1 个 block × N 个线程（N = 256），无动态共享内存。
    ///
    /// 流程：
    /// 1. 线程 0 初始化 `BAR`，期望到达线程数为 `block_size`；
    /// 2. 每个线程把自己的 `tid` 写入共享内存 `DATA[tid]`；
    /// 3. 所有线程 arrive 后自旋等待，确保共享内存写入全部完成；
    /// 4. 每个线程读取**环形右邻居** `(tid + 1) % block_size` 的值并写入输出。
    ///
    /// # 参数
    /// - `out`：全局输出缓冲区，`out[i]` 被写为 `(i + 1) % N`，
    ///   用于验证屏障保护下的跨线程数据交换正确。
    #[kernel]
    pub fn barrier_shared_data(mut out: DisjointSlice<u32>) {
        // 静态声明Barrier和共享数据数组，所有线程共用，UNINIT表示尚未初始化
        static mut BAR: Barrier = Barrier::UNINIT;
        static mut DATA: SharedArray<u32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // 线程0初始化BAR，设置期望的到达线程数为block_size
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
            }
        }

        // 确保所有线程都看到了BAR的初始化
        thread::sync_threads();

        // 每个线程把自己的线程ID写入共享内存 DATA[tid] = tid
        unsafe {
            DATA[tid as usize] = tid;
        }

        // 每个线程都调用arrive，到达barrier
        let token = unsafe { mbarrier_arrive(&raw const BAR) };

        // 自旋等待，直到所有block_size个线程都写入DATA并到达
        // 此时所有线程都能安全地读取其他线程写入的DATA
        unsafe { while !mbarrier_test_wait(&raw const BAR, token) {} }

        // 读取环形右邻居（tid+1）写入的共享内存值
        // 因为有barrier保护，这里保证邻居的数据已经写入完成
        let neighbor_idx = ((tid + 1) % block_size) as usize;
        let neighbor_val = unsafe { DATA[neighbor_idx] };

        // 把邻居的值写入全局输出
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = neighbor_val;
        }

        // 线程0使BAR失效，清理资源
        if tid == 0 {
            unsafe {
                mbarrier_inval(&raw mut BAR);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    #[test]
    pub fn test_barrier_sync() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = barrier_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        unsafe { module.barrier_sync(&stream, cfg, &mut out_dev) }.expect("Kernel launch failed");

        stream.synchronize()?;
        let out_result = out_dev.to_host_vec(&stream)?;
        let all_ones = out_result.iter().all(|&x| x == 1);

        assert!(all_ones, "✗ Barrier sync failed! ");
        Ok(())
    }

    #[test]
    pub fn test_barrier_smem_sync() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = barrier_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

        unsafe { module.barrier_shared_data(&stream, cfg, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize()?;
        let out_result = out_dev.to_host_vec(&stream)?;
        let correect = out_result
            .iter()
            .enumerate()
            .all(|(i, &x)| x == ((i + 1) % N) as u32);

        assert!(correect, "✗ Pattern mismatch!");
        Ok(())
    }
}
