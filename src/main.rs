use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// 普通辅助函数，无需注解
/// 编译器通过"vecadd"的调用关系自动发现它。
fn add(a: f32, b: f32) -> f32 {
    a + b
}

fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

fn max(x: f32, y: f32) -> f32 {
    if x >= y { x } else { y }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = add(a[i], b[i]);
        }
    }

    #[kernel]
    pub fn apply_clamp(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = clamp(input[i], 0.0, 1.0);
        }
    }

    #[device]
    pub fn magnitude(x: f32, y: f32) -> f32 {
        (x * x + y * y).sqrt()
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("failed to load ctx");
    module
        .vecadd(
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
    if errors == 0 {
        println!("PASSED: all {} params right", N);
    } else {
        eprintln!("FAILED: {} error", errors);
        std::process::exit(1);
    }
}
