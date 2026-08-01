//! 本模块演示**设备全局变量（device global）**的使用：
//!
//! - `device_global` — 跨 kernel 调用共享的可变设备全局状态
//! - `nonzero_static_table` — 静态常量表（非零值）的读取
//! - `static_initializer_edges` — 静态初始化器的边界情况（NaN 位模式、结构体字段打包）
//!
//! 设备全局变量位于 `.global` 内存，kernel 之间共享；
//! 而 `static` 常量则以原始位模式存放在设备内存中，按位读写。

use cuda_device::kernel;
use cuda_host::cuda_module;

// —— 设备全局状态 ——
// 可变静态变量：在 kernel 间共享，跨多次启动累积状态
static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MACKER: u32 = 0;

// —— 设备端常量数据 ——
// 静态常量表：4 组 (权重对)，从设备内存直接读取
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];
// 带 payload 的 quiet NaN（0x7fc0_1234）
// 通过 &'static 引用读取，避免 LLVM 常量折叠将 payload 规范化为 0
static STATIC_NAN: f32 = f32::from_bits(0x7fc0_1234);

// 带填充的结构体常量：测试静态初始化器的字段布局与字节打包
#[repr(C)]
struct PaddedStatic {
    tag: u8,
    value: u32,
}

static PADDED_STATIC: PaddedStatic = PaddedStatic {
    tag: 0xab,
    value: 0x1234_5678,
};

/// 返回静态权重表的引用。
///
/// `#[inline(never)]` 防止内联，确保在设备端通过内存加载真实访问该静态变量。
#[inline(never)]
fn get_static_weights() -> &'static [[f32; 2]; 4] {
    &STATIC_WEIGHTS
}

/// 返回静态 NaN 的引用。
///
/// 关键点：**按引用返回**使 NaN 作为 `.global` 内存中的数据按位读取，
/// 而非作为编译期浮点常量参与 LLVM 常量折叠——这样 payload `0x1234` 得以保留。
#[inline(never)]
fn get_static_nan() -> &'static f32 {
    &STATIC_NAN
}

/// 返回带填充的结构体常量的引用。
#[inline(never)]
fn get_padded_static() -> &'static PaddedStatic {
    &PADDED_STATIC
}

/// 以 `[f32; 2]` 为单位加载第 `i_pair` 个浮点对。
///
/// 演示如何将 f32 指针按 `[f32; 2]` 数组步长进行偏移读取。
#[inline(always)]
unsafe fn load_pair(ptr: *const f32, i_pair: usize) -> [f32; 2] {
    unsafe { *(ptr as *const [f32; 2]).add(i_pair) }
}

#[cuda_module]
mod device_global_mod {
    use super::*;

    /// 使用可变设备全局变量的 kernel。
    ///
    /// 每次启动累加 `DEVICE_COUNTER`（跨 launch 共享状态），写入 `DEVICE_MACKER`，
    /// 然后将二者异或后输出。
    ///
    /// # 参数
    /// - `out`：输出缓冲区（1 个 u64），写入 `DEVICE_COUNTER ^ DEVICE_MACKER`。
    #[kernel]
    pub unsafe fn device_global(out: *mut u64) {
        unsafe {
            DEVICE_COUNTER += 1; // 跨 launch 累加的设备全局计数器
            DEVICE_MACKER = 0x00C0_FFEE; // 写入固定标记值
            *out = DEVICE_COUNTER ^ (DEVICE_MACKER as u64);
        }
    }

    /// 从静态常量表读取数据的 kernel。
    ///
    /// 通过 `get_static_weights()` 引用静态权重表，用 `load_pair` 按
    /// `[f32; 2]` 步长加载第 2 组（索引 2）：`[4.0, 8.0]`，输出二者之和 `12.0`。
    ///
    /// # 参数
    /// - `out`：输出缓冲区（1 个 f32），写入 `4.0 + 8.0 = 12.0`。
    #[kernel]
    pub unsafe fn nonzero_static_table(out: *mut f32) {
        let weights = get_static_weights();
        let pair = unsafe { load_pair(&weights[0][0], 2) };
        unsafe {
            *out = pair[0] + pair[1];
        }
    }

    /// 测试静态初始化器边界情况的 kernel。
    ///
    /// 验证两类静态初始化器边缘场景：
    /// 1. **NaN 位模式**：`STATIC_NAN` 的 payload `0x1234` 需通过按引用读取完整保留；
    /// 2. **结构体字段打包**：`PaddedStatic` 的 `tag` 和 `value` 字段按位重组输出。
    ///
    /// # 参数
    /// - `nan_out`：输出缓冲区（1 个 f32），写入 `STATIC_NAN` 的原始位模式。
    /// - `padded_out`：输出缓冲区（1 个 u64），写入 `(value << 8) | tag`。
    #[kernel]
    pub unsafe fn static_initializer_edges(nan_out: *mut f32, padded_out: *mut u64) {
        let padded = get_padded_static();
        unsafe {
            *nan_out = *get_static_nan(); // 按引用读取，保留 NaN payload
            *padded_out = ((padded.value as u64) << 8) | (padded.tag as u64);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    #[test]
    pub fn test_device_global() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        let out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        // Test size - must match TILE size (256 elements)

        let module = device_global_mod::load(&ctx).expect("load cuda module failed");
        for launch_idx in 1..2 {
            unsafe {
                module.device_global(
                    &stream,
                    LaunchConfig::for_num_elems(1),
                    out_dev.cu_deviceptr() as *mut u64,
                )
            }?;
            let result = out_dev.to_host_vec(&stream)?[0];
            let expected = launch_idx as u64 ^ 0x00C0_FFEE;
            assert_eq!(
                result, expected,
                "FAILED: expected {expected:#x}, got {result:#x}"
            );
        }
        Ok(())
    }

    #[test]
    pub fn test_nonzero_static_table() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        let out_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
        // Test size - must match TILE size (256 elements)

        let module = device_global_mod::load(&ctx).expect("load cuda module failed");
        unsafe {
            module.nonzero_static_table(
                &stream,
                LaunchConfig::for_num_elems(1),
                out_dev.cu_deviceptr() as *mut f32,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?[0];
        assert!(
            (result - 12.0f32).abs() < f32::EPSILON,
            "FAILED: expected 12.0, got {result}"
        );
        Ok(())
    }

    #[test]
    pub fn test_static_initializer_edges() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        let nan_out_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
        let padded_out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        let module = device_global_mod::load(&ctx).expect("load cuda module failed");
        unsafe {
            module.static_initializer_edges(
                &stream,
                LaunchConfig::for_num_elems(1),
                nan_out_dev.cu_deviceptr() as *mut f32,
                padded_out_dev.cu_deviceptr() as *mut u64,
            )
        }?;
        let nan_bits = nan_out_dev.to_host_vec(&stream)?[0].to_bits();
        let padded_result = padded_out_dev.to_host_vec(&stream)?[0];
        assert_eq!(
            nan_bits, 0x7fc0_1234,
            "FAILED: expected NaN bits 0x7fc01234, got {nan_bits:#x}"
        );
        let padded_expected = ((0x1234_5678u64) << 8) | 0xab;
        assert_eq!(
            padded_result, padded_expected,
            "FAILED: expected {padded_expected:#x}, got {padded_result:#x}"
        );
        Ok(())
    }
}
