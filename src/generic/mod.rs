use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::ops::{Add, Mul};

/// 泛型 CUDA 模块
///
/// 演示 Rust 泛型在 GPU kernel 中的应用，包括：
/// - `scale` — 泛型逐元素缩放（需要实现 `Mul` trait）
/// - `add` — 泛型逐元素加法（需要实现 `Add` trait）
/// - `closure_capture` — 闭包捕获外部变量实现线性变换
/// - `apply_unary` — 将闭包作为高阶函数参数传递
#[allow(dead_code)]
#[cuda_module]
pub mod generic_module {

    use super::*;

    /// 逐元素缩放，泛型类型 T
    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[i] * factor;
        }
    }

    /// 逐元素加法，泛型类型 T
    #[kernel]
    pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }

    /// 对输入值应用一元闭包转换
    ///
    /// 将闭包 `f` 应用到值 `x` 上。通过 `#[inline(never)]` 确保闭包调用
    /// 不会被内联，从而在 GPU 代码中产生明确的函数调用边界。
    #[inline(never)]
    fn apply_unary<T, F>(f: F, x: T) -> T
    where
        F: FnOnce(T) -> T,
    {
        f(x)
    }

    /// 使用闭包捕获进行逐元素线性变换：`out[i] = input[i] * scale + bias`
    ///
    /// 演示如何在 GPU kernel 中通过闭包捕获外部变量（`scale` 和 `bias`），
    /// 并将闭包作为参数传递给辅助函数 `apply_unary`。
    #[kernel]
    pub fn closure_capture<T>(bias: T, scale: T, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let transform = move |x: T| x * scale + bias;
            *out_elem = apply_unary(transform, input[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    /// 测试 f32 类型的泛型缩放 kernel
    ///
    /// 使用 factor=2.0，验证 `out[i] = input[i] * 2.0`。
    #[test]
    fn test_scale_f32() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let factor: f32 = 2.0;
        unsafe {
            module.scale::<f32>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )?;
        }

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (factor * input_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试 f16 类型的泛型缩放 kernel
    ///
    /// 验证 `scale::<f16>` 对不同浮点类型的泛型支持。
    #[test]
    fn test_scale_f16() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f16> = (0..N).map(|i| i as f16).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
        let mut output_dev = DeviceBuffer::<f16>::zeroed(&stream, N)?;
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let factor: f16 = 2.0;
        unsafe {
            module.scale::<f16>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut output_dev,
            )?;
        }

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (factor * input_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试 f32 类型的泛型加法 kernel
    ///
    /// 验证 `out[i] = a[i] + b[i]` 的泛型实现。
    #[test]
    fn test_add_f32() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let module = generic_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            module.add(
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

    /// 测试 f16 类型的泛型加法 kernel
    ///
    /// 验证 `add::<f16>` 在半精度下的正确性。由于 f16 精度有限，
    /// 预期值通过 f32 加法后转 f16 舍入得到，以对齐 GPU 计算行为。
    #[test]
    fn test_add_f16() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f16> = (0..N).map(|i| i as f16).collect();
        let b_host: Vec<f16> = (0..N).map(|i| (i * 2) as f16).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f16>::zeroed(&stream, N)?;
        let module = generic_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            module.add(
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

    /// 测试 f32 类型的闭包捕获 kernel
    ///
    /// 验证 `closure_capture::<f32>` 通过闭包捕获 `scale` 和 `bias`，
    /// 实现 `out[i] = input[i] * 3.0 + 2.0`。
    #[test]
    fn test_closure_f32() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let bias: f32 = 2.0;
        let scale: f32 = 3.0;
        unsafe {
            module.closure_capture::<f32>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                bias,
                scale,
                &input_dev,
                &mut output_dev,
            )?;
        }

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (scale * input_host[i] + bias)).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }

    /// 测试 f16 类型的闭包捕获 kernel
    ///
    /// 验证 `closure_capture::<f16>` 在半精度下的闭包捕获功能。
    /// 输入数据缩小为 `i * 0.01` 以避免 f16 精度范围的溢出。
    #[test]
    fn test_closure_f16() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f16> = (0..N).map(|i| i as f16 * 0.01).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
        let mut output_dev = DeviceBuffer::<f16>::zeroed(&stream, N)?;
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let bias: f16 = 2.0;
        let scale: f16 = 3.0;
        unsafe {
            module.closure_capture::<f16>(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                bias,
                scale,
                &input_dev,
                &mut output_dev,
            )?;
        }

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| {
                // 通过 f16 舍入预期值，对齐 GPU 精度
                let expected = (scale as f32 * input_host[i] as f32 + bias as f32) as f16;
                (output_host[i] as f32 - expected as f32).abs() > 1e-5
            })
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs");
        Ok(())
    }
}
