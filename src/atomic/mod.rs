//! 本模块演示 CUDA **原子操作（atomic）** 的多种用法：
//!
//! - 设备作用域原子（`DeviceAtomic*`，PTX `.gpu` 作用域）：**跨 block 保证原子性**
//! - 块作用域原子（`BlockAtomicU32`，PTX `.cta` 作用域）：仅块内保证原子性
//! - 多种操作：`fetch_add/sub`、`compare_exchange`、`swap`、`fetch_or/and/xor`、`fetch_min/max`、`store/load`
//! - 多种内存序：`Relaxed`、`Acquire`、`AcqRel`、`SeqCst`
//! - 覆盖类型：`u32`、`i32`、`u64`、`i64`、`f32`、`f64`

use cuda_device::atomic::{
    AtomicOrdering, BlockAtomicU32, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicI32,
    DeviceAtomicI64, DeviceAtomicU32, DeviceAtomicU64,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod atomic_mod {

    use super::*;

    /// 基础的设备作用域原子自增 kernel（`fetch_add`）。
    ///
    /// 每个线程对全局计数器原子地 `+1`（`Relaxed` 序），并把**旧值**写入输出。
    /// 256 个线程全部累加后，计数器应为 256，且每个线程拿到的旧值 0..255 互不重复。
    ///
    /// # 参数
    /// - `counter`：全局计数器（1 个 u32），被所有线程原子累加。
    /// - `out`：输出缓冲区，`out[i]` 写入第 i 个线程拿到的旧值。
    #[kernel]
    pub fn atomic_fetch_add_u32(counter: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        // 把 u32 切片指针 reinterpret 为设备作用域原子类型（零成本包装）
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }

    /// 原子 store/load + 块内同步 kernel。
    ///
    /// 线程 0 原子写入 `42`，`sync_threads()` 保证所有线程看到写入后，
    /// 所有线程用 `Acquire` 序原子读取该值并写入输出。
    ///
    /// # 参数
    /// - `flat`：全局标志位（1 个 u32），演示原子写后读。
    /// - `out`：输出缓冲区，所有元素应为 `42`。
    #[kernel]
    pub fn atomic_load_store_u32(flat: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_flag = unsafe { &*(flat.as_ptr() as *const DeviceAtomicU32) };
        if tid == 0 {
            atomic_flag.store(42, AtomicOrdering::Relaxed);
        }
        thread::sync_threads(); // 确保所有线程看到线程0的写入

        let val = atomic_flag.load(AtomicOrdering::Acquire);
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = val;
        }
    }

    /// 原子 CAS（compare_exchange）选举唯一获胜者 kernel。
    ///
    /// 所有线程竞争将计数器从 `0` 原子交换为 `tid + 1`。
    /// 只有**一个**线程能成功（`Ok`），其余全部失败（`Err`），
    /// 从而选出唯一的获胜者，其编号写入计数器。
    ///
    /// # 参数
    /// - `winner`：全局 CAS 目标（1 个 u32），最终保存获胜者的 `tid + 1`。
    /// - `out`：输出缓冲区，获胜者位置写 `1`，其余写 `0`。
    #[kernel]
    pub fn atomic_cas_u32(winner: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_winner = unsafe { &*(winner.as_ptr() as *const DeviceAtomicU32) };

        let result = atomic_winner.compare_exchange(
            0,
            tid + 1,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        );

        if let Some(out_elem) = out.get_mut(gid) {
            match result {
                Ok(_) => {
                    *out_elem = 1; // 竞争成功
                }
                Err(_) => {
                    *out_elem = 0; // 竞争失败
                }
            }
        }
    }

    /// 使用 `AcqRel`（获取-释放）内存序的原子自增 kernel。
    ///
    /// 与 `atomic_fetch_add_u32` 行为相同，仅内存序改为 `AcqRel`，
    /// 用于演示不同内存序对原子计数正确性的影响。
    ///
    /// # 参数
    /// - `counter`：全局计数器（1 个 u32）。
    /// - `out`：输出缓冲区，`out[i]` 写入旧值。
    #[kernel]
    pub fn atomic_fetch_add_acqrel_u32(counter: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::AcqRel);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }

    /// 使用 `SeqCst`（顺序一致）内存序的原子自增 kernel。
    ///
    /// 与 `atomic_fetch_add_u32` 行为相同，仅内存序改为 `SeqCst`（最强的保证）。
    ///
    /// # 参数
    /// - `counter`：全局计数器（1 个 u32）。
    /// - `out`：输出缓冲区，`out[i]` 写入旧值。
    #[kernel]
    pub fn atomic_fetch_add_seqcst_u32(counter: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::SeqCst);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }

    /// `i32` 类型的原子自增 + CAS kernel。
    ///
    /// 所有线程先对 `counter` 原子 `+1`；随后线程 0 通过 CAS 把 `cas_target`
    /// 从 `0` 交换为 `-42`，其他线程同步后读取最终值（应为 `-42`）。
    ///
    /// # 参数
    /// - `counter`：i32 全局计数器，验证 `fetch_add` 的 i32 版本。
    /// - `cas_target`：i32 CAS 目标，最终为 `-42`。
    /// - `out`：输出缓冲区，`out[0]` 反映 CAS 是否成功（1/0），其余为读取的 `-42`。
    #[kernel]
    pub fn atomic_i32(counter: &[i32], cas_target: &[i32], mut out: DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicI32) };
        let atomic_cas = unsafe { &*(cas_target.as_ptr() as *const DeviceAtomicI32) };

        let _old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let result = atomic_cas.compare_exchange(
                    0,
                    -42,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                );
                match result {
                    Ok(_) => {
                        *out_elem = 1;
                    }
                    Err(_) => {
                        *out_elem = 0;
                    }
                }
            } else {
                thread::sync_threads(); // 等待线程0完成CAS
                let val = atomic_cas.load(AtomicOrdering::Acquire);
                *out_elem = val;
            }
        }
    }

    /// `u64` 类型的原子自增 kernel。
    ///
    /// 演示 64 位原子操作（`DeviceAtomicU64`）。
    ///
    /// # 参数
    /// - `counter`：u64 全局计数器（1 个 u64）。
    /// - `out`：输出缓冲区，`out[i]` 写入旧值。
    #[kernel]
    pub fn atomic_fetch_add_u64(counter: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU64) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }

    /// `i64` 类型的原子自增 + CAS kernel。
    ///
    /// 与 `atomic_i32` 结构相同，使用 64 位版本，CAS 交换值为 `-100`。
    ///
    /// # 参数
    /// - `counter`：i64 全局计数器。
    /// - `cas_target`：i64 CAS 目标，最终为 `-100`。
    /// - `out`：输出缓冲区，`out[0]` 反映 CAS 是否成功，其余为 `-100`。
    #[kernel]
    pub fn atomic_i64(counter: &[i64], cas_target: &[i64], mut out: DisjointSlice<i64>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicI64) };
        let atomic_cas = unsafe { &*(cas_target.as_ptr() as *const DeviceAtomicI64) };

        let _old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let result = atomic_cas.compare_exchange(
                    0,
                    -100,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                );
                match result {
                    Ok(_) => {
                        *out_elem = 1;
                    }
                    Err(_) => {
                        *out_elem = 0;
                    }
                }
            } else {
                thread::sync_threads();
                let val = atomic_cas.load(AtomicOrdering::Acquire);
                *out_elem = val;
            }
        }
    }

    /// 原子自减 kernel（`fetch_sub`）。
    ///
    /// 计数器初始为 `N`，每个线程原子地 `-1` 并取回旧值。
    /// 256 个线程执行后计数器归零。
    ///
    /// # 参数
    /// - `counter`：全局计数器（1 个 u32），初始为 `N`。
    /// - `out`：输出缓冲区，`out[i]` 写入旧值。
    #[kernel]
    pub fn atomic_fetch_sub_u32(counter: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        let old = atomic_counter.fetch_sub(1, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }

    /// 原子位运算 kernel（`fetch_or` / `fetch_and` / `fetch_xor`）。
    ///
    /// 三个独立的原子累加器演示不同位操作：
    /// - `or`：每个线程按 `tid % 32` 置位，256 线程后低 32 位全 1（`0xFFFF_FFFF`）
    /// - `and`：线程 0 与 `0x0000_FFFF`，结果保留低 16 位
    /// - `xor`：每个线程异或 `1`，偶数次操作后归零
    ///
    /// # 参数
    /// - `or_acc` / `and_acc` / `xor_acc`：三个位运算累加器。
    /// - `out`：输出缓冲区，写入 `tid`（占位验证）。
    #[kernel]
    pub fn atomic_bitwise_u32(
        or_acc: &[u32],
        and_acc: &[u32],
        xor_acc: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_or = unsafe { &*(or_acc.as_ptr() as *const DeviceAtomicU32) };
        let atomic_and = unsafe { &*(and_acc.as_ptr() as *const DeviceAtomicU32) };
        let atomic_xor = unsafe { &*(xor_acc.as_ptr() as *const DeviceAtomicU32) };

        let bit = 1u32 << (tid % 32);
        atomic_or.fetch_or(bit, AtomicOrdering::Relaxed); // 置位

        if tid == 0 {
            atomic_and.fetch_and(0x0000FFFF, AtomicOrdering::Relaxed); // 保留低16位
        }

        atomic_xor.fetch_xor(1, AtomicOrdering::Relaxed); // 翻转最低位
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = tid;
        }
    }

    /// 原子交换 kernel（`swap`）。
    ///
    /// 线程 0 把目标从 `0` 原子交换为 `0xDEADBEEF` 并取回旧值 `0`；
    /// 其他线程同步后读取最终值（应为 `0xDEADBEEF`）。
    ///
    /// # 参数
    /// - `target`：全局交换目标（1 个 u32），最终为 `0xDEADBEEF`。
    /// - `out`：输出缓冲区，`out[0]` 为旧值 `0`，其余为 `0xDEADBEEF`。
    #[kernel]
    pub fn atomic_swap_u32(target: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_target = unsafe { &*(target.as_ptr() as *const DeviceAtomicU32) };

        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let old = atomic_target.swap(0xDEADBEEF, AtomicOrdering::AcqRel);
                *out_elem = old;
            } else {
                thread::sync_threads();
                let val = atomic_target.load(AtomicOrdering::Acquire);
                *out_elem = val;
            }
        }
    }

    /// 原子最小值/最大值 kernel（`fetch_min` / `fetch_max`，`i32`）。
    ///
    /// 线程写入值 `tid - 128`，范围 `[-128, 127]`。
    /// 累加器初始为 `i32::MAX` / `i32::MIN`，最终 `min = -128`、`max = 127`。
    ///
    /// # 参数
    /// - `min_acc` / `max_acc`：min/max 累加器（各 1 个 i32）。
    /// - `out`：输出缓冲区，写入 `tid - 128`（占位验证）。
    #[kernel]
    pub fn atomic_minmax_i32(min_acc: &[i32], max_acc: &[i32], mut out: DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_min = unsafe { &*(min_acc.as_ptr() as *const DeviceAtomicI32) };
        let atomic_max = unsafe { &*(max_acc.as_ptr() as *const DeviceAtomicI32) };

        let value = (tid as i32) - 128;

        atomic_min.fetch_min(value, AtomicOrdering::Relaxed);
        atomic_max.fetch_max(value, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = value;
        }
    }

    /// `f32` 类型的原子自增 kernel。
    ///
    /// 演示浮点原子累加（`DeviceAtomicF32`），每线程加 `1.0`。
    ///
    /// # 参数
    /// - `counter`：f32 全局计数器（1 个 f32），最终为 `N`。
    /// - `out`：输出缓冲区（u32），全部写 `1`（占位验证）。
    #[kernel]
    pub fn atomic_fetch_add_f32(counter: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicF32) };

        let _old = atomic_counter.fetch_add(1.0, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = 1;
        }
    }

    /// `f64` 类型的原子自增 kernel。
    ///
    /// 演示双精度浮点原子累加（`DeviceAtomicF64`），每线程加 `1.0`。
    ///
    /// # 参数
    /// - `counter`：f64 全局计数器（1 个 f64），最终为 `N`。
    /// - `out`：输出缓冲区（u32），全部写 `1`（占位验证）。
    #[kernel]
    pub fn atomic_fetch_add_f64(counter: &[f64], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicF64) };

        let _old = atomic_counter.fetch_add(1.0, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = 1;
        }
    }

    /// `f32` 原子交换 kernel（`swap`）。
    ///
    /// 线程 0 把目标从 `0.0` 交换为 `3.14`；其他线程同步后读取 `3.14`。
    /// 输出用布尔标志验证交换/读取结果。
    ///
    /// # 参数
    /// - `target`：f32 全局交换目标，最终为 `3.14`。
    /// - `out`：输出缓冲区，成功标志写 `1`。
    #[kernel]
    pub fn atomic_swap_f32(target: &[f32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_target = unsafe { &*(target.as_ptr() as *const DeviceAtomicF32) };

        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let old = atomic_target.swap(3.14, AtomicOrdering::AcqRel);
                if old < 0.01 {
                    *out_elem = 1;
                } else {
                    *out_elem = 0;
                }
            } else {
                thread::sync_threads();
                let val = atomic_target.load(AtomicOrdering::Acquire);
                if val > 3.0 {
                    *out_elem = 1;
                } else {
                    *out_elem = 0;
                }
            }
        }
    }

    /// 无符号 `u32` 原子最小值/最大值 kernel。
    ///
    /// 累加器初始为 `u32::MAX` / `0`，每线程 `fetch_min(tid)` / `fetch_max(tid)`，
    /// 最终 `min = 0`、`max = N-1`。
    ///
    /// # 参数
    /// - `min_acc` / `max_acc`：min/max 累加器（各 1 个 u32）。
    /// - `out`：输出缓冲区，写入 `tid`（占位验证）。
    #[kernel]
    pub fn atomic_unsigned_minmax_u32(
        min_acc: &[u32],
        max_acc: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_min = unsafe { &*(min_acc.as_ptr() as *const DeviceAtomicU32) };
        let atomic_max = unsafe { &*(max_acc.as_ptr() as *const DeviceAtomicU32) };

        atomic_min.fetch_min(tid, AtomicOrdering::Relaxed);
        atomic_max.fetch_max(tid, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = tid;
        }
    }

    /// **块作用域**原子自增 kernel（`BlockAtomicU32`，PTX `.cta` 作用域）。
    ///
    /// 与 `atomic_fetch_add_u32` 结构相同，但使用块作用域原子。
    ///
    /// ⚠️ 注意：`.cta` 作用域**只保证块内**的原子性。
    /// 若多个 block 对**同一个全局计数器**自增，跨 block 的原子性**不由规范保证**，
    /// 结果正确依赖硬件 L2 原子单元的串行化（实现细节，非模型保证）。
    /// 多 block 共享计数器的正确写法应使用 `DeviceAtomicU32`（`.gpu` 作用域）。
    ///
    /// # 参数
    /// - `counter`：全局计数器（1 个 u32）。
    /// - `out`：输出缓冲区，`out[i]` 写入旧值。
    #[kernel]
    pub fn atomic_block_scope_fetch_add_u32(counter: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const BlockAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }

    /// **每块独立计数器**的块作用域原子自增 kernel（`BlockAtomicU32`，`.cta` 作用域）。
    ///
    /// 与 `atomic_block_scope_fetch_add_u32` 的关键区别：每个 block 操作**自己专属的计数器**
    /// `counters[bid]`，而不是所有 block 共享同一个计数器。
    ///
    /// 这样 `.cta` 作用域在语义上就**足够且正确**：每个计数器只被一个 block 访问，
    /// 不依赖硬件 L2 原子串行化也能保证正确。
    ///
    /// # 参数
    /// - `counters`：每块一个计数器的数组（长度 = gridDim.x），`counters[bid]` 被块 `bid` 独占累加。
    /// - `out`：输出缓冲区，`out[i]` 写入该线程拿到的旧值。
    #[kernel]
    pub fn atomic_block_scope_per_block_counter(counters: &[u32], mut out: DisjointSlice<u32>) {
        let bid = thread::blockIdx_x() as usize; // 当前 block 的索引
        let idx = thread::index_1d();
        // 取本 block 专属的计数器地址，reinterpret 为块作用域原子类型
        let atomic_counter = unsafe { &*(counters.as_ptr().add(bid) as *const BlockAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = old;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    /// 验证 `atomic_fetch_add_u32`：1 个 block × 256 线程，计数器应为 256，
    /// 且每个线程拿到的旧值 0..255 互不重复。
    #[test]
    pub fn test_atomic_fetch_add_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_u32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as u32,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        let mut sorted = out_vals.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), N, "  FAIL: Output values are not unique");
        Ok(())
    }

    /// 验证 `atomic_load_store_u32`：所有线程同步后读到的值都应为 `42`。
    #[test]
    pub fn test_atomic_load_store_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let flag_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_load_store_u32(&stream, cfg, &flag_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let out_vals = out_dev.to_host_vec(&stream)?;
        let all_42 = out_vals.iter().all(|&x| x == 42);
        assert!(all_42, "  FAIL: Not all output values are 42");

        Ok(())
    }

    /// 验证 `atomic_cas_u32`：CAS 应恰好选出一个获胜者，
    /// 且获胜者编号落在 `[1, N]` 内。
    #[test]
    pub fn test_atomic_cas_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let winner_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_cas_u32(&stream, cfg, &winner_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let winner_val = winner_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        let num_winners: usize = out_vals.iter().filter(|&&x| x == 1).count();
        assert_eq!(num_winners, 1, "  FAIL: There should be exactly one winner");
        assert!(
            winner_val >= 1 && winner_val <= N as u32,
            "  FAIL: Winner value {} is out of expected range [1, {}]",
            winner_val,
            N
        );
        Ok(())
    }

    /// 验证 `AcqRel` 内存序下的原子自增：计数器应为 256，旧值唯一。
    #[test]
    pub fn test_atomic_fetch_add_acqrel_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_acqrel_u32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as u32,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        let mut sorted = out_vals.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            N,
            "  Counter = {} with AcqRel ordering, all {} values unique",
            N,
            N
        );
        Ok(())
    }

    /// 验证 `SeqCst` 内存序下的原子自增：计数器应为 256，旧值唯一。
    #[test]
    pub fn test_atomic_fetch_add_seqcst_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_seqcst_u32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as u32,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        let mut sorted = out_vals.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            N,
            "  Counter = {} with SeqCst ordering, all {} values unique",
            N,
            N
        );
        Ok(())
    }

    /// 验证 `atomic_i32`：i32 计数器应为 256，CAS 目标应为 `-42`。
    #[test]
    pub fn test_atomic_i32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<i32>::zeroed(&stream, 1)?;
        let cas_target_dev = DeviceBuffer::<i32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_i32(&stream, cfg, &counter_dev, &cas_target_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let cas_target_val = cas_target_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as i32,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        assert_eq!(
            cas_target_val, -42,
            "  FAIL: CAS target = {} (expected {})",
            cas_target_val, -42
        );
        assert_eq!(
            out_vals[0], 1,
            "  FAIL: Out value = {} (expected {})",
            out_vals[0], 1
        );
        Ok(())
    }

    /// 验证多 block 场景下的 `DeviceAtomicU32`（`.gpu` 作用域）：
    /// 4 个 block × 64 线程累加同一计数器，应正确得到 256。
    #[test]
    pub fn test_atomic_multiblock_fetch_add_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (4, 1, 1),
            block_dim: (N as u32 / 4, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_u32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as u32,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        let mut sorted = out_vals.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), N, "  FAIL: Output values are not unique");
        Ok(())
    }

    /// 验证 `u64` 原子自增：计数器应为 256，旧值唯一。
    #[test]
    pub fn test_atomic_fetch_add_u64() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_u64(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as u64,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        let mut sorted = out_vals.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), N, "  FAIL: Output values are not unique");
        Ok(())
    }

    /// 验证 `i64` 原子自增 + CAS：计数器应为 256，CAS 目标应为 `-100`。
    #[test]
    pub fn test_atomic_i64() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<i64>::zeroed(&stream, 1)?;
        let cas_target_dev = DeviceBuffer::<i64>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<i64>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_i64(&stream, cfg, &counter_dev, &cas_target_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let cas_target_val = cas_target_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, N as i64,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        assert_eq!(
            cas_target_val, -100,
            "  FAIL: CAS target = {} (expected {})",
            cas_target_val, -100
        );
        assert_eq!(
            out_vals[0], 1,
            "  FAIL: Out value = {} (expected {})",
            out_vals[0], 1
        );
        Ok(())
    }

    /// 验证原子自减：计数器从 256 减到 0，旧值唯一。
    #[test]
    pub fn test_atomic_fetch_sub_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let counter_host = vec![N as u32];
        let counter_dev = DeviceBuffer::from_host(&stream, &counter_host)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_sub_u32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            counter_val, 0,
            "  FAIL: Counter = {} (expected {})",
            counter_val, N
        );
        let mut sorted = out_vals.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), N, "  FAIL: Output values are not unique");
        Ok(())
    }

    /// 验证原子位运算：OR 得 `0xFFFF_FFFF`、AND 保留低 16 位、XOR 归零。
    #[test]
    pub fn test_atomic_bitwise_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let or_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let and_dev = DeviceBuffer::<u32>::from_host(&stream, &[0xFFFF_FFFFu32])?;
        let xor_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_bitwise_u32(&stream, cfg, &or_dev, &and_dev, &xor_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let or_val = or_dev.to_host_vec(&stream)?[0];
        let and_val = and_dev.to_host_vec(&stream)?[0];
        let xor_val = xor_dev.to_host_vec(&stream)?[0];
        assert_eq!(
            or_val, 0xFFFF_FFFFu32,
            "  FAIL: OR value = {} (expected {})",
            or_val, 0xFFFF_FFFFu32
        );
        assert_eq!(
            and_val, 0x0000_FFFFu32,
            "  FAIL: AND value = {} (expected {})",
            and_val, 0x0000_FFFFu32
        );
        assert_eq!(
            xor_val, 0,
            "  FAIL: XOR value = {} (expected {})",
            xor_val, 0
        );
        Ok(())
    }

    /// 验证原子交换：目标变为 `0xDEADBEEF`，旧值为 `0`。
    #[test]
    pub fn test_atomic_swap_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let target_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_swap_u32(&stream, cfg, &target_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let target_val = target_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert_eq!(
            target_val, 0xDEADBEEFu32,
            "  FAIL: Target = {} (expected {})",
            target_val, 0xDEADBEEFu32
        );
        assert_eq!(out_vals[0], 0, "  FAIL: old=0x{:08X}", out_vals[0]);
        Ok(())
    }

    /// 验证 `i32` 原子 min/max：应得 `min = -128`、`max = 127`。
    #[test]
    pub fn test_atomic_minmax_i32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let min_dev = DeviceBuffer::from_host(&stream, &[i32::MAX])?;
        let max_dev = DeviceBuffer::from_host(&stream, &[i32::MIN])?;
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_minmax_i32(&stream, cfg, &min_dev, &max_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let min_val = min_dev.to_host_vec(&stream)?[0];
        let max_val = max_dev.to_host_vec(&stream)?[0];
        assert_eq!(
            min_val, -128,
            "  FAIL: Min = {} (expected {})",
            min_val, -128
        );
        assert_eq!(max_val, 127, "  FAIL: Max = {} (expected {})", max_val, 127);
        Ok(())
    }

    /// 验证 `f32` 原子自增：计数器应为 256.0，输出全部为 1。
    #[test]
    pub fn test_atomic_fetch_add_f32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_f32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert!(
            (counter_val - N as f32).abs() < 1e-6,
            "  FAIL: Counter = {} (expected {})",
            counter_val,
            N
        );
        let all_one = out_vals.iter().all(|&x| x == 1);
        assert!(all_one, "  FAIL: Not all output values are 1");
        Ok(())
    }

    /// 验证 `f64` 原子自增：计数器应为 256.0，输出全部为 1。
    #[test]
    pub fn test_atomic_fetch_add_f64() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<f64>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_fetch_add_f64(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert!(
            (counter_val - N as f64).abs() < 1e-6,
            "  FAIL: Counter = {} (expected {})",
            counter_val,
            N
        );
        let all_one = out_vals.iter().all(|&x| x == 1);
        assert!(all_one, "  FAIL: Not all output values are 1");
        Ok(())
    }

    /// 验证 `f32` 原子交换：目标变为 `3.14`，交换成功标志为 1。
    #[test]
    pub fn test_atomic_swap_f32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let target_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_swap_f32(&stream, cfg, &target_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let target_val = target_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        assert!(
            (target_val - 3.14).abs() < 1e-6,
            "  FAIL: Target = {} (expected {})",
            target_val,
            3.14
        );
        assert!(out_vals[0] == 1, "  FAIL: old=1 (expected 1)");
        Ok(())
    }

    /// 验证无符号 `u32` 原子 min/max：应得 `min = 0`、`max = N-1`。
    #[test]
    pub fn test_atomic_minmax_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let min_dev = DeviceBuffer::from_host(&stream, &[u32::MAX])?;
        let max_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_unsigned_minmax_u32(&stream, cfg, &min_dev, &max_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let min_val = min_dev.to_host_vec(&stream)?[0];
        let max_val = max_dev.to_host_vec(&stream)?[0];
        assert_eq!(min_val, 0, "  FAIL: Min = {} (expected {})", min_val, 0);
        assert_eq!(
            max_val,
            (N as u32 - 1),
            "  FAIL: Max = {} (expected {})",
            max_val,
            (N as u32 - 1)
        );
        Ok(())
    }

    /// 验证块作用域原子（`BlockAtomicU32`，`.cta` 作用域）的跨 block 行为。
    ///
    /// 启动 2 个 block × 128 线程，对同一全局计数器自增。
    /// ⚠️ `.cta` 作用域**不保证**跨 block 原子性，结果正确依赖硬件 L2 原子串行化；
    /// 此处仅打印观察，不做强断言（多 block 共享计数器的正确写法应改用 `DeviceAtomicU32`）。
    #[test]
    pub fn test_atomic_block_scope_fetch_add_u32() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config for shared memory kernels
        let cfg = LaunchConfig {
            grid_dim: (2, 1, 1),
            block_dim: (N as u32 / 2, 1, 1),
            shared_mem_bytes: 0,
        };

        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_block_scope_fetch_add_u32(&stream, cfg, &counter_dev, &mut out_dev)?;
        }
        stream.synchronize()?;
        let counter_val = counter_dev.to_host_vec(&stream)?[0];
        let out_vals = out_dev.to_host_vec(&stream)?;
        println!("Counter value: {}", counter_val);
        println!("Output values: {:?}", out_vals);
        Ok(())
    }

    /// 验证**每块独立计数器**的块作用域原子（`.cta` 作用域）。
    ///
    /// 启动 2 个 block × 128 线程，每块原子累加**自己的**计数器。
    /// 因为每个计数器只被一个 block 访问，`.cta` 作用域在语义上就足够正确：
    /// 每个计数器应恰好为 128，且每个 block 内的旧值 0..127 互不重复。
    #[test]
    pub fn test_atomic_block_scope_per_block_counter() -> Result<()> {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();
        // Test size - must match TILE size (256 elements)
        const N: usize = 256;
        const BLOCKS: u32 = 2;

        let module = atomic_mod::load(&ctx).expect("load cuda module failed");
        // Launch config: 2 blocks, each with N/2 = 128 threads
        let cfg = LaunchConfig {
            grid_dim: (BLOCKS, 1, 1),
            block_dim: (N as u32 / BLOCKS, 1, 1),
            shared_mem_bytes: 0,
        };

        // 每块一个计数器，共 BLOCKS 个（每个计数器只被对应的 block 独占访问）
        let counters_dev = DeviceBuffer::<u32>::zeroed(&stream, BLOCKS as usize)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            module.atomic_block_scope_per_block_counter(
                &stream,
                cfg,
                &counters_dev,
                &mut out_dev,
            )?;
        }
        stream.synchronize()?;
        let counters = counters_dev.to_host_vec(&stream)?;
        let out_vals = out_dev.to_host_vec(&stream)?;

        let per_block = (N / BLOCKS as usize) as u32; // 每个 block 的线程数 128

        // 每个计数器应被其独占的 block 恰好累加 per_block 次
        for (b, &c) in counters.iter().enumerate() {
            assert_eq!(
                c, per_block,
                "  FAIL: counters[{b}] = {c} (expected {per_block})"
            );
        }

        // 每个 block 内拿到的旧值应互不重复（0..per_block-1）
        for b in 0..BLOCKS as usize {
            let start = b * per_block as usize;
            let mut block_vals = out_vals[start..start + per_block as usize].to_vec();
            block_vals.sort();
            block_vals.dedup();
            assert_eq!(
                block_vals.len(),
                per_block as usize,
                "  FAIL: block {b} output values are not unique"
            );
        }
        Ok(())
    }
}
