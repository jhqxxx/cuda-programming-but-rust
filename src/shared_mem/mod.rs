use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

/// 共享内存 CUDA 模块
///
/// 演示 CUDA 共享内存的多种使用方式：
/// - `shared_single` — 单个 `SharedArray` 静态共享内存
/// - `shared_dual` — 两个 `SharedArray` 静态共享内存
/// - `dynamic_smem_basic` — 动态共享内存（默认 16 字节对齐）
/// - `dynamic_smem_partition` — 动态共享内存分区
/// - `dynamic_smem_explicit_align` — 显式 128 字节对齐的动态共享内存
/// - `dynamic_smem_mixed_align` — 混合对齐的动态共享内存
#[cuda_module]
pub mod shared_mem_mod {
    use cuda_device::{DynamicSharedArray, gpu_printf};

    use super::*;

    /// 单个共享内存数组 kernel
    ///
    /// 声明一个编译期大小固定的 `SharedArray<f32, 256>`。
    /// 每个线程从全局内存读入一个元素到共享内存，同步后读取右邻居的值输出。
    /// 演示最基本的共享内存使用模式：加载 → 同步 → 读取。
    #[kernel]
    pub fn shared_single(data: &[f32], mut out: DisjointSlice<f32>) {
        // 声明一个共享内存数组
        // SharedArray: 编译时需要知道大小
        static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize; // 当前线程在block 内的索引 [0..255]
        let gid = thread::index_1d().get(); // 全局线程索引 [0..N-1]
        unsafe {
            TILE[tid] = data[gid]; // 每个线程从全局内存中读一个元素，写入共享内存
        }

        thread::sync_threads(); // 同步，保证所有共享内存写入完成
        unsafe {
            let neighbor_idx = (tid + 1) % 256; // 右邻居（环形）
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = TILE[neighbor_idx]; // 共享内存：块内共享，读取邻居值
            }
        }
    }

    /// 双共享内存数组 kernel
    ///
    /// 同时使用两个 `SharedArray`，分别加载输入 a 和 b。
    /// 演示如何在一个 kernel 中声明并使用多块静态共享内存。
    #[kernel]
    pub fn shared_dual(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        // 声明两个共享内存数组
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();
        unsafe {
            TILE_A[tid] = a[gid];
            TILE_B[tid] = b[tid];
        }
        thread::sync_threads();

        unsafe {
            let neighbor_idx = (tid + 1) % 256;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = TILE_A[neighbor_idx] + TILE_B[neighbor_idx];
            }
        }
    }

    /// 动态共享内存基础用法
    ///
    /// 使用 `DynamicSharedArray::<f32>::get()` 获取动态共享内存基地址指针。
    /// 大小在 launch 时通过 `LaunchConfig::shared_mem_bytes` 指定。
    /// 默认 16 字节对齐（与 nvcc 一致）。
    #[kernel]
    pub fn dynamic_smem_basic(data: &[f32], mut out: DisjointSlice<f32>) {
        let smem: *mut f32 = DynamicSharedArray::<f32>::get();
        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();
        if tid == 0 {
            gpu_printf!("[basic] smem addr: {:x} (align 16)", smem as u64);
        }

        unsafe {
            *smem.add(tid) = data[gid];
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem.add(neighbor_idx)
            }
        }
    }

    /// 动态共享内存分区
    ///
    /// 对同一块动态共享内存进行手动分区：`smem_a` 从偏移 0 开始，
    /// `smem_b` 从偏移 1024 字节（256 个 f32）开始。
    /// 演示如何通过 `offset()` 方法将一块动态共享内存划分为多个逻辑数组。
    #[kernel]
    pub fn dynamic_smem_partition(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let smem_a: *mut f32 = DynamicSharedArray::<f32>::get();
        let smem_b: *mut f32 = DynamicSharedArray::<f32>::offset(1024);

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();
        if tid == 0 {
            gpu_printf!(
                "[partition] smem_a: {:x}, smem_b: {:x} (diff={})\n",
                smem_a as u64,
                smem_b as u64,
                (smem_b as u64) - (smem_a as u64)
            );
        }

        unsafe {
            *smem_a.add(tid) = a[gid];
            *smem_b.add(tid) = b[gid];
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem_a.add(neighbor_idx) + *smem_b.add(neighbor_idx);
            }
        }
    }

    /// 显式 128 字节对齐的动态共享内存
    ///
    /// 使用 `DynamicSharedArray::<f32, 128>` 指定 128 字节对齐。
    /// 128 字节对齐是 TMA（Tensor Memory Accelerator）操作的硬性要求。
    /// 同时演示如何在 device 端通过 `gpu_printf!` 打印共享内存地址以验证对齐。
    #[kernel]
    pub fn dynamic_smem_explicit_align(data: &[f32], mut out: DisjointSlice<f32>) {
        let smem: *mut f32 = DynamicSharedArray::<f32, 128>::get();

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        if tid == 0 {
            let addr = smem as u64;
            gpu_printf!(
                "[explicit] smem addr: {:x} (align 128, mod128={})\n",
                addr,
                addr % 128
            );
        }

        unsafe {
            *smem.add(tid) = data[gid] * 2.0;
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbot_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem.add(neighbot_idx);
            }
        }
    }

    /// 混合对齐的动态共享内存
    ///
    /// 在一段动态共享内存中分区出三个数组，分别使用不同的对齐要求：
    /// - `smem_a` — 默认 16 字节对齐，偏移 0
    /// - `smem_b` — 128 字节对齐，偏移 1024 字节
    /// - `smem_c` — 256 字节对齐，偏移 2048 字节
    ///
    /// 演示在实际应用中为不同数据指定不同对齐需求的分区模式。
    #[kernel]
    pub fn dynamic_smem_mixed_align(a: &[f32], b: &[f32], c: &[f32], mut out: DisjointSlice<f32>) {
        let smem_a: *mut f32 = DynamicSharedArray::<f32>::get(); // 16字节对齐
        let smem_b: *mut f32 = DynamicSharedArray::<f32, 128>::offset(1024); // 128字节对齐
        let smem_c: *mut f32 = DynamicSharedArray::<f32, 256>::offset(2048); // 256字节对齐

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        if tid == 0 {
            let addr_a = smem_a as u64;
            let addr_b = smem_b as u64;
            let addr_c = smem_c as u64;
            gpu_printf!("[mixed] smem_a: {:x} (mod256={})\n", addr_a, addr_a % 256);
            gpu_printf!(
                "[mixed] smem_b: {:x} (offset={})\n",
                addr_b,
                addr_b - addr_a
            );
            gpu_printf!(
                "[mixed] smem_c: {:x} (offset={})\n",
                addr_c,
                addr_c - addr_a
            );
        }

        unsafe {
            *smem_a.add(tid) = a[gid];
            *smem_b.add(tid) = b[gid];
            *smem_c.add(tid) = c[gid];
        }

        thread::sync_threads();

        unsafe {
            let block_size = thread::blockDim_x() as usize;
            let neighbor_idx = (tid + 1) % block_size;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = *smem_a.add(neighbor_idx)
                    + *smem_b.add(neighbor_idx)
                    + *smem_c.add(neighbor_idx);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    /// 测试单个静态共享内存 kernel
    ///
    /// 1 个 block × 256 线程，无动态共享内存。
    /// 验证 `shared_single` 的环形邻居读取正确。
    #[test]
    pub fn test_shared_single() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = shared_mem_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let data_dev = DeviceBuffer::from_host(&stream, &data_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe { module.shared_single((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = data_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                errors += 1;
            }
        }
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试双静态共享内存 kernel
    ///
    /// 1 个 block × 256 线程，无动态共享内存。
    /// 验证两个 `SharedArray` 同时使用的正确性。
    #[test]
    pub fn test_shared_dual() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = shared_mem_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i + 100) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe { module.shared_dual(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = a_host[neighbor_idx] + b_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                errors += 1;
            }
        }
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试动态共享内存基础用法
    ///
    /// 1 个 block × 256 线程，动态共享内存大小为 N × sizeof(f32) = 1024 字节。
    /// 验证 `dynamic_smem_basic` 的正确性。
    #[test]
    pub fn test_dynamic_smem_basic() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;
        const BLOCK_SIZE: u32 = 256;

        let module = shared_mem_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (N * core::mem::size_of::<f32>()) as u32,
        };

        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let data_dev = DeviceBuffer::from_host(&stream, &data_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe { module.dynamic_smem_basic(&stream, cfg, &data_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = data_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                errors += 1;
            }
        }
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试动态共享内存分区
    ///
    /// 1 个 block × 256 线程，动态共享内存大小为 2 × N × sizeof(f32) = 2048 字节，
    /// 分别分配给 smem_a（1024 字节）和 smem_b（1024 字节）。
    /// 验证分区后两个数组的正确读写。
    #[test]
    pub fn test_dynamic_smem_partition() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;
        const BLOCK_SIZE: u32 = 256;

        let module = shared_mem_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (2 * N * core::mem::size_of::<f32>()) as u32,
        };

        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i + 100) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe { module.dynamic_smem_partition(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = a_host[neighbor_idx] + b_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                errors += 1;
            }
        }
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试 128 字节对齐的动态共享内存
    ///
    /// 1 个 block × 256 线程，动态共享内存大小为 N × sizeof(f32) = 1024 字节。
    /// 验证 `dynamic_smem_explicit_align` 的正确性。
    #[test]
    pub fn test_dynamic_smem_explicit_align() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;
        const BLOCK_SIZE: u32 = 256;

        let module = shared_mem_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (N * core::mem::size_of::<f32>()) as u32,
        };

        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

        let data_dev = DeviceBuffer::from_host(&stream, &data_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe { module.dynamic_smem_explicit_align(&stream, cfg, &data_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = data_host[neighbor_idx] * 2.0;
            if (out_result[i] - expected).abs() > 1e-5 {
                errors += 1;
            }
        }
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试混合对齐的动态共享内存
    ///
    /// 1 个 block × 256 线程，动态共享内存大小为 3 × N × sizeof(f32) = 3072 字节。
    /// 三块区域分别使用 16、128、256 字节对齐，验证混合对齐分区的正确性。
    #[test]
    pub fn test_dynamic_smem_mixed_align() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;
        const BLOCK_SIZE: u32 = 256;

        let module = shared_mem_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: (3 * N * core::mem::size_of::<f32>()) as u32,
        };

        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
        let c_host: Vec<f32> = (0..N).map(|i| (i * 3) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let c_dev = DeviceBuffer::from_host(&stream, &c_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe {
            module.dynamic_smem_mixed_align(&stream, cfg, &a_dev, &b_dev, &c_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = a_host[neighbor_idx] + b_host[neighbor_idx] + c_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                errors += 1;
            }
        }
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }
}
