use cuda_device::{ConstantMemory, DisjointSlice, constant, kernel, thread};
use cuda_host::cuda_module;

/// 常量内存模块
///
/// 演示 CUDA 常量内存的用法。常量内存具有缓存加速、只读、线程安全的特点，
/// 适合存储所有线程共用的只读参数（如缩放因子、偏置、多项式系数等）。
#[cuda_module]
pub mod constant_mod {
    use super::*;

    /// 缩放因子常量，f32 类型
    #[constant]
    static SCALE: ConstantMemory<f32> = ConstantMemory::UNINIT;

    /// 偏置常量，f32 类型
    #[constant]
    static BIAS: ConstantMemory<f32> = ConstantMemory::UNINIT;

    /// 多项式系数常量，存储 4 个 f32 系数的数组
    #[constant]
    static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;

    /// 线性变换 kernel：out[i] = i * SCALE + BIAS
    ///
    /// 演示单一标量常量从常量内存读取的使用方式。
    #[kernel]
    pub fn multiply(mut out: DisjointSlice<f32>) {
        let s = SCALE.get();
        let b = BIAS.get();
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(e) = out.get_mut(idx) {
            *e = (i as f32) * s + b;
        }
    }

    /// 三次多项式 kernel：out[i] = c0 + c1*x + c2*x² + c3*x³，其中 x = i
    ///
    /// 演示数组类型常量从常量内存读取的使用方式。
    #[kernel]
    pub fn apply_coeffs(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let [c0, c1, c2, c3] = COEFFS.get();
            let x = i as f32;
            *out_elem = c0 + c1 * x + c2 * x * x + c3 * x * x * x;
        }
    }
}

/// 主机端辅助函数：计算多项式 c0 + c1*x + c2*x² + c3*x³
///
/// 用于与 GPU kernel 的结果进行比对验证。
#[allow(unused)]
fn poly(x: f32, coeffs: &[f32; 4]) -> f32 {
    coeffs[0] + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    /// 测试线性变换 kernel：设置 SCALE 和 BIAS 常量，验证 GPU 计算结果正确
    #[test]
    fn test_multiply() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = constant_mod::load(&ctx)?;
        // 通过生成的 setter 设置常量内存值
        module.set_scale(&stream, &3.0)?;
        module.set_bias(&stream, &1.0)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, 8)?;

        unsafe {
            module.multiply(&stream, LaunchConfig::for_num_elems(8), &mut out)?;
        }

        let out_host = out.to_host_vec(&stream)?;

        // 预期：out[i] = i * 3.0 + 1.0
        let expected = vec![1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0];
        assert_eq!(
            out_host, expected,
            "constant_memory_simple: kernel output mismatch"
        );
        Ok(())
    }

    /// 测试多项式 kernel：设置 COEFFS 常量，验证 GPU 计算结果与主机端 poly() 一致
    #[test]
    fn test_apply_coeffs() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = constant_mod::load(&ctx)?;
        // 设置多项式系数：1.0 + 2.0*x + 0.5*x² - 0.25*x³
        let coeffs = [1.0f32, 2.0, 0.5, -0.25];
        module.set_coeffs(&stream, &coeffs)?;
        const N: usize = 256;
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        unsafe {
            module.apply_coeffs(&stream, LaunchConfig::for_num_elems(N as u32), &mut out)?;
        }

        let out_host = out.to_host_vec(&stream)?;

        let errors = (0..N)
            .filter(|&i| (out_host[i] - poly(i as f32, &coeffs)).abs() > 1e-5)
            .count();
        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }
}
