use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

/// 设备端辅助函数：逐元素加法 a[i] + b[i] -> c[i]
///
/// 可被多个 kernel 复用，避免重复编写相同的加法逻辑。
/// `#[device]` 属性标记此函数为 GPU 设备端函数，只能从 kernel 或其他 device 函数中调用。
#[device]
pub fn vecadd_device(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[i] + b[i];
    }
}

/// 逐元素加法 CUDA 模块
///
/// 包含多种 kernel 变体，演示不同的实现策略：
/// - `vecadd_with_helper` — 调用 `#[device]` 辅助函数
/// - `elementwise_add_f32_kernel` — 基础 f32 逐元素加法
/// - `elementwise_add_f32x4_kernel` — 4 元向量化 f32 加法（手动展开）
/// - `elementwise_add_f16_kernel` — f16 半精度加法
#[cuda_module]
pub mod elementwise_module {

    use super::*;

    /// 调用 `vecadd_device` 辅助函数的包装 kernel
    ///
    /// 演示将公共逻辑提取到 `#[device]` 函数中复用的模式。
    /// 注意：c 是 `DisjointSlice<f32>` 而非 `&mut`，因为 device 函数内部处理了可变性。
    #[kernel]
    pub fn vecadd_with_helper(a: &[f32], b: &[f32], c: DisjointSlice<f32>) {
        vecadd_device(a, b, c);
    }

    /// 逐元素加法内核，f32 类型
    #[kernel]
    pub fn elementwise_add_f32_kernel(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }

    /// 逐元素加法内核，f32 类型，4 元向量化版本
    ///
    /// 每个线程一次处理 4 个元素（`i = idx * 4`），通过手动展开循环减少指令数，
    /// 利用 GPU 的 128 位访存指令提高带宽利用率。
    /// 边界处理：当剩余元素不足 4 个时，回退到逐元素处理。
    #[kernel]
    pub fn elementwise_add_f32x4_kernel(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get() * 4;
        let n = a.len();
        if i + 3 < n {
            let c_ptr = c.as_mut_ptr();
            unsafe {
                *c_ptr.add(i) = a[i] + b[i];
                *c_ptr.add(i + 1) = a[i + 1] + b[i + 1];
                *c_ptr.add(i + 2) = a[i + 2] + b[i + 2];
                *c_ptr.add(i + 3) = a[i + 3] + b[i + 3];
            }
        } else if i < n {
            let c_ptr = c.as_mut_ptr();
            unsafe {
                for offset in 0..(n - i) {
                    let index = i + offset;
                    *c_ptr.add(index) = a[index] + b[index];
                }
            }
        }
    }

    // #[kernel]
    // pub fn elementwise_add_f32x4_kernel_CuSimd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    //     let idx = thread::index_1d();
    //     let i = idx.get() * 4;
    //     let n = a.len();
    //     unsafe {
    //         if i + 3 < n {
    //             // read_unaligned/write_unaligned
    //             // 避免: a, b, c 的地址不是 16 字节对齐的情况时访问越界
    //             let reg_a = ptr::read_unaligned(a.as_ptr().add(i).cast::<Float4>());
    //             let reg_b = ptr::read_unaligned(b.as_ptr().add(i).cast::<Float4>());
    //             let reg_c = Float4::new([
    //                 reg_a.get::<0>() + reg_b.get::<0>(),
    //                 reg_a.get::<1>() + reg_b.get::<1>(),
    //                 reg_a.get::<2>() + reg_b.get::<2>(),
    //                 reg_a.get::<3>() + reg_b.get::<3>(),
    //             ]);
    //             ptr::write_unaligned(c.as_mut_ptr().add(i).cast::<Float4>(), reg_c);
    //         } else if i < n {
    //             let a_ptr = a.as_ptr();
    //             let b_ptr = b.as_ptr();
    //             let c_ptr = c.as_mut_ptr();
    //             for offset in 0..(n - i) {
    //                 let index = i + offset;
    //                 *c_ptr.add(index) = *a_ptr.add(index) + *b_ptr.add(index);
    //             }
    //         }
    //     }
    // }

    /// 逐元素加法内核，f16（半精度浮点）类型
    ///
    /// 与 f32 版本逻辑相同，但操作数为 `f16` 类型，节省显存和带宽。
    /// 适用于对精度要求不高的场景（如某些 AI 推理任务）。
    #[kernel]
    pub fn elementwise_add_f16_kernel(a: &[f16], b: &[f16], mut c: DisjointSlice<f16>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    use super::*;
    use crate::utils::get_launch_config_1d;

    /// 测试通过 device 辅助函数调用的逐元素加法
    ///
    /// 加载 CUDA 模块，在 GPU 上执行 `vecadd_with_helper`，验证结果正确性。
    #[test]
    fn test_vecadd_helper() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            elementwise_module.vecadd_with_helper(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )?;
        }

        let c_host = c_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试基础 f32 逐元素加法 kernel
    #[test]
    fn test_elementwise_add_f32() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            elementwise_module.elementwise_add_f32_kernel(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )?;
        }

        let c_host = c_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试 4 元向量化 f32 逐元素加法 kernel
    ///
    /// 使用 `get_launch_config_1d(N, 256/4)` 确保 block 数量正确匹配向量化后的线程数。
    #[test]
    fn test_elementwise_add_f32x4() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");

        unsafe {
            elementwise_module.elementwise_add_f32x4_kernel(
                &stream,
                get_launch_config_1d(N as u32, 256 / 4),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )?;
        }
        let c_host = c_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试 f16 半精度逐元素加法 kernel
    ///
    /// 注意：由于 f16 精度有限，比较时先通过 f32 做加法再转回 f16，
    /// 匹配 GPU 上的半精度计算行为。
    #[test]
    fn test_elementwise_add_f16() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f16> = (0..N).map(|i| i as f16).collect();
        let b_host: Vec<f16> = (0..N).map(|i| (i * 2) as f16).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f16>::zeroed(&stream, N)?;
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            elementwise_module.elementwise_add_f16_kernel(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )?;
        }
        let c_host = c_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| {
                // Round through f16 to match GPU precision
                let expected = (a_host[i] as f32 + b_host[i] as f32) as f16;
                (c_host[i] as f32 - expected as f32).abs() > 1e-5
            })
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }
}
