use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod elementwise_module {

    // use std::ptr;

    // use cuda_device::Float4;

    use super::*;

    #[kernel]
    pub fn elementwise_add_f32_kernel(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }

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
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    use super::*;
    use crate::utils::get_launch_config_1d;

    #[test]
    fn elementwise_add_f32() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            elementwise_module
                .elementwise_add_f32_kernel(
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
    fn elementwise_add_f32x4() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");

        unsafe {
            elementwise_module
                .elementwise_add_f32x4_kernel(
                    &stream,
                    get_launch_config_1d(N as u32, 256 / 4),
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
    fn elementwise_add_f16() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        const N: usize = 1024;
        let a_host: Vec<f16> = (0..N).map(|i| i as f16).collect();
        let b_host: Vec<f16> = (0..N).map(|i| (i * 2) as f16).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut c_dev = DeviceBuffer::<f16>::zeroed(&stream, N).unwrap();
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");
        unsafe {
            elementwise_module
                .elementwise_add_f16_kernel(
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
}
