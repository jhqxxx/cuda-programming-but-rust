use cuda_device::cuda_module;

/// 普通辅助函数，无需注解
/// 编译器通过"vecadd"的调用关系自动发现它。
fn add(a: f32, b: f32) -> f32 {
    a + b
}

// #[cuda_module]
// mod kernels {

// }

fn main() {
    
}
