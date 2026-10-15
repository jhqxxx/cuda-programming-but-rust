//! 示例 01：向量加法 —— c[i] = a[i] + b[i]
//!
//! 运行：`cargo oxide run 01-vector-add`（在 `cuda-oxide-example/` 目录下执行）
//!
//! # 这个文件想说明什么
//!
//! cuda-oxide 与 CUDA C++ 最大的形态差别：**主机代码和 device 代码写在同一个文件里，
//! 一次性编译，不需要 `__global__` / `__host__` 之类的声明，也不需要拆成 .cu + .cpp**。
//!
//! 编译期发生的事（由 cargo oxide 驱动的自定义 codegen 后端 rustc-codegen-cuda 完成）：
//!
//! 1. rustc 先按常规流程解析整个文件，为所有函数生成 MIR；
//! 2. codegen 阶段被后端拦截：
//!    - 被 `#[kernel]` 标记的函数走 mir-importer 管线 → 生成 PTX → 嵌入最终二进制；
//!    - `main` 及其它主机代码走标准 LLVM 后端 → 生成 x86_64 机器码；
//! 3. 产物是**一个二进制**，里面同时包含主机机器码和嵌入式 PTX 模块。
//!
//! 所以下面对 `kernels::vecadd` 的调用，本质是运行时从嵌入的 PTX 里加载出 CUDA
//! module 再启动，而不是一次普通的函数调用。

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// =============================================================================
// Device 侧：这段代码会被编译成 PTX
// =============================================================================

/// 内核写在 `#[cuda_module]` 标记的模块里。
///
/// 被 `#[kernel]` 标记的函数同时存在于两个世界：
/// - 主机侧：函数体会被类型检查，但永远不会被真正调用；
/// - Device 侧：被编译成 PTX。
#[cuda_module]
mod kernels {
    use super::*;

    /// `c[i] = a[i] + b[i]`
    ///
    /// 几个要点：
    /// - `a: &[f32]` / `b: &[f32]` 是只读输入；`c: DisjointSlice<f32>` 是可写输出。
    ///   可变输出必须用 `DisjointSlice`，它向编译器承诺「每个线程只写自己那一个元素」，
    ///   这样 Rust 的别名规则才能在上面成立（不需要 unsafe）。
    /// - `thread::index_1d()` 取当前线程的全局一维下标，等价于 CUDA 里的
    ///   `blockIdx.x * blockDim.x + threadIdx.x`。
    /// - 越界判断用 `c.get_mut(idx)` 返回的 `Option` 完成：长度是 N，而网格长度是
    ///   线程块大小的整数倍，所以最后一个块里往往有线程拿不到元素。
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

// =============================================================================
// 主机侧：这段代码会被编译成 x86_64 机器码
// =============================================================================

fn main() {
    println!("=== 01 向量加法：c[i] = a[i] + b[i] ===\n");

    // 初始化 CUDA 上下文（0 号设备）并取其默认流。
    let ctx = CudaContext::new(0).expect("无法创建 CUDA 上下文");
    let stream = ctx.default_stream();

    // ---- 准备数据 ----------------------------------------------------------
    // 长度刻意取 1000，不是 256 的整数倍，用来验证内核里的越界判断确实生效。
    const N: usize = 1000;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("输入（前 5 个元素）：");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // ---- 传到设备 ----------------------------------------------------------
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).expect("拷贝 a 到设备失败");
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).expect("拷贝 b 到设备失败");
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).expect("分配 c 失败");

    // ---- 从嵌入的 PTX 加载模块并用类型化 API 启动 ----------------------------
    let module = kernels::load(&ctx).expect("加载嵌入的 CUDA 模块失败");

    // LaunchConfig::for_num_elems 会按元素个数自动算出网格/线程块尺寸。
    //
    // SAFETY：启动配置与内核的资源需求匹配，且三个缓冲区都覆盖了该内核的访问范围。
    unsafe {
        module.vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }
    .expect("内核启动失败");

    // ---- 取回结果并校验 ----------------------------------------------------
    let c_host = c_dev.to_host_vec(&stream).expect("拷贝 c 回主机失败");

    println!("\n输出（前 5 个元素）：");
    println!("  c = {:?}", &c_host[0..5]);

    let mut errors = 0usize;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if (c_host[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!("  下标 {} 处错误：期望 {}，实际 {}", i, expected, c_host[i]);
            }
            errors += 1;
        }
    }

    assert_eq!(errors, 0, "有 {} 个元素结果不正确", errors);
    println!("\n✓ 全部 {} 个元素校验通过", N);
}
