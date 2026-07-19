use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::ops::{Add, Mul};

#[allow(dead_code)]
#[cuda_module]
pub mod generic_module {

    use super::*;

    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[i] * factor;
        }
    }

    #[kernel]
    pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }

    #[inline(never)]
    fn apply_unary<T, F>(f: F, x: T) -> T
    where
        F: FnOnce(T) -> T,
    {
        f(x)
    }

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
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    #[test]
    fn test_scale_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let factor: f32 = 2.0;
        unsafe {
            module
                .scale::<f32>(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    factor,
                    &input_dev,
                    &mut output_dev,
                )
                .unwrap();
        }

        let output_host = output_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (factor * input_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }

    #[test]
    fn test_scale_f16() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f16> = (0..N).map(|i| i as f16).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
        let mut output_dev = DeviceBuffer::<f16>::zeroed(&stream, N).unwrap();
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let factor: f16 = 2.0;
        unsafe {
            module
                .scale::<f16>(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    factor,
                    &input_dev,
                    &mut output_dev,
                )
                .unwrap();
        }

        let output_host = output_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (factor * input_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }

    #[test]
    fn test_add_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        let module = generic_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            module
                .add(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &a_dev,
                    &b_dev,
                    &mut c_dev,
                )
                .unwrap();
        }

        let c_host = c_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }

    #[test]
    fn test_add_f16() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f16> = (0..N).map(|i| i as f16).collect();
        let b_host: Vec<f16> = (0..N).map(|i| (i * 2) as f16).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut c_dev = DeviceBuffer::<f16>::zeroed(&stream, N).unwrap();
        let module = generic_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            module
                .add(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    &a_dev,
                    &b_dev,
                    &mut c_dev,
                )
                .unwrap();
        }

        let c_host = c_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| {
                // Round through f16 to match GPU precision
                let expected = (a_host[i] as f32 + b_host[i] as f32) as f16;
                (c_host[i] as f32 - expected as f32).abs() > 1e-5
            })
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }

    #[test]
    fn test_closure_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let bias: f32 = 2.0;
        let scale: f32 = 3.0;
        unsafe {
            module
                .closure_capture::<f32>(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    bias,
                    scale,
                    &input_dev,
                    &mut output_dev,
                )
                .unwrap();
        }

        let output_host = output_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (scale * input_host[i] + bias)).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }

    #[test]
    fn test_closure_f16() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let input_host: Vec<f16> = (0..N).map(|i| i as f16 * 0.01).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
        let mut output_dev = DeviceBuffer::<f16>::zeroed(&stream, N).unwrap();
        let module = generic_module::load(&ctx).expect("failed to load ctx");

        let bias: f16 = 2.0;
        let scale: f16 = 3.0;
        unsafe {
            module
                .closure_capture::<f16>(
                    &stream,
                    LaunchConfig::for_num_elems(N as u32),
                    bias,
                    scale,
                    &input_dev,
                    &mut output_dev,
                )
                .unwrap();
        }

        let output_host = output_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| {
                // 通过 f16 舍入预期值，对齐 GPU 精度
                let expected = (scale as f32 * input_host[i] as f32 + bias as f32) as f16;
                (output_host[i] as f32 - expected as f32).abs() > 1e-5
            })
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }
}
