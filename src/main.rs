#![feature(f16)]
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
mod elementwise;
mod utils;
use crate::elementwise::elementwise_module;

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let module = elementwise_module::load(&ctx).expect("failed to load ctx");

    unsafe {
        module
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
    if errors == 0 {
        println!("PASSED: all {} params right", N);
    } else {
        eprintln!("FAILED: {} error", errors);
        std::process::exit(1);
    }
}
