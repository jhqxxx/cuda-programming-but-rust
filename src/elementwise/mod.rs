use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod elementwise_module {

    use super::*;

    #[kernel]
    pub fn elementwise_add_f32_kernel(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
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
        let elementwise_module = elementwise_module::load(&ctx).expect("load cuda module failed");
        elementwise_module
            .elementwise_add_f32_kernel(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &a_dev,
                &b_dev,
                &mut c_dev,
            )
            .unwrap();
        let c_host = c_dev.to_host_vec(&stream).unwrap();
        let errors = (0..N)
            .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
            .count();

        assert!(errors == 0, "FAILED {errors} wrongs")
    }
}
