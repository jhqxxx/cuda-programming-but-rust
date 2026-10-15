# 调试与性能分析

调试从小的、确定性的输入开始。把结果回读到宿主，与 CPU 参考实现比对，等正确性确立之后，再去检查生成的 Tile IR，或者对 GPU 做性能分析。

## 打印与断言

`cuda_tile_print!` 可以从 GPU kernel 内部打印：

```rust
#[cutile::entry()]
fn debug_kernel<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    let pid0 = program_id(0);
    let pid1 = program_id(1);
    let tile = x.load_like(z);

    cuda_tile_print!("Program ({}, {}): loaded tile\n", pid0, pid1);
    z.store(tile);
}
```

GPU 打印很慢，并且会让 tile block 的执行串行化。只在小网格上使用，并在测量性能之前把它移除。

`cuda_tile_assert!` 在 kernel 内部检查条件：

```rust
let n: i32 = x.shape()[0];
cuda_tile_assert!(n > 0, "expected a non-empty input");
```

## 宿主回读

宿主回读本身是一个 `DeviceOp`；在读取宿主 vector 之前先执行它：

```rust
let z_host: Vec<f32> = z
    .unpartition()
    .to_host_vec()
    .sync_on(&stream)?;

assert!(!z_host.iter().any(|x| x.is_nan()));
assert!(!z_host.iter().any(|x| x.is_infinite()));
```

如果某个融合 kernel 结果不对，就把它拆成几个阶段，逐段回读中间结果。每个阶段都应当在小输入上与一个简单的 CPU 实现一致。

## 正确性测试

先用最小的输入：

```rust
#[test]
fn small_add_matches_cpu() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let expected = vec![11.0, 22.0, 33.0, 44.0];

    let result = run_add_kernel(&a, &b);
    assert_eq!(result, expected);
}
```

然后用更大的随机输入与 CPU 参考实现按合适的容差比对：

```rust
for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
    assert!((cpu - gpu).abs() < 1e-5, "CPU={cpu}, GPU={gpu}");
}
```

对于数值敏感的 kernel，要测试边界情况：全零、大正值、大负值、（在支持的前提下）不能被整除的形状，以及已知容易溢出的输入。

## 检查 Tile IR

`print_ir = true` 会在 JIT 编译期间打印生成的入口点包装与 Tile IR 文本：

```rust
#[cutile::entry(print_ir = true)]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

`dump_mlir_dir` 会把编译后的 Tile IR 文本写入文件：

```rust
#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

模块级的转储也可以通过环境变量获得。它们会被写入 stderr，每个已编译模块一次：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `CUTILE_DUMP` | 逗号分隔的待转储阶段：`ir`（Tile IR 模块文本）与 `bytecode`（别名 `bc`；把编码后的字节码解码回文本），或者 `all`。`ast`、`resolved`、`typed`、`instantiated` 这些名字会被接受，但目前没有任何代码路径会输出它们 | 未设置 |
| `CUTILE_DUMP_FILTER` | 逗号分隔的 `module::function` 路径；转储是按模块进行的，所以只会匹配 `module` 部分。裸函数名不会排除任何模块 | 未设置 |

## 错误与崩溃

大多数 cuTile Rust 错误在 kernel 运行之前就会被捕获：

| 错误 | 原因 | 修复 |
|---|---|---|
| 形状不匹配 | tile 形状不兼容 | 对齐形状，或使用 `reshape` / `broadcast` |
| 元素类型不匹配 | 同一个操作里混用了不同的元素类型 | 加显式的 `convert_tile()` |
| 非法的归约轴 | 轴超出了 tile 的秩 | 使用 `0..rank` 范围内的轴 |
| 不支持的 MMA 形状或 dtype | 该组合没有下降实现 | 改用受支持的形状与元素类型 |
| 缺少入口 | 函数没有标记 `#[cutile::entry()]` | 加上入口属性 |

运行时错误通常来自越界访问、工具包问题，或者非法的裸指针用法：

| 错误 | 原因 | 修复 |
|---|---|---|
| CUDA error: no kernel image | GPU 架构不对，或 cubin 过期 | 清缓存、重新构建、确认目标 SM |
| Failed to load kernel | CUDA 工具包或驱动问题 | 检查 `nvidia-smi` 与工具包版本 |
| 内存不足 | tensor 分配或 JIT 内存压力 | 减小分配尺寸或特化数量 |
| 运行时形状不匹配 | tensor 尺寸与分区不兼容 | 确保预期的整除性或边界处理 |

CPU 段错误通常意味着故障发生在宿主侧 FFI、JIT 编译，或者裸指针的生命周期管理上，而不是发生在普通的安全 tile 代码里。先拿到调用栈：

```bash
RUST_BACKTRACE=1 cargo run
RUST_BACKTRACE=full cargo run

gdb --args ./target/debug/my_program
(gdb) run
(gdb) bt
```

检查 CUDA 驱动、CUDA 工具包路径、裸指针生命周期、spawn 出来的任务的生命周期，以及首次启动编译期间的宿主内存占用。

## 调试构建与 Sanitizer

设备调试信息默认跟随 Cargo 的 profile `debug` 设置。使用标准 profile 时，`cargo build` 会启用设备调试模式，`cargo build --release` 则禁用它。这与 `debug-assertions` 无关。

例如，通过工作区的 `Cargo.toml` 在开发构建中关闭设备调试：

```toml
[profile.dev]
debug = false
```

Cargo 只向构建脚本提供 `DEBUG=true|false`，不提供确切的调试级别。目前任何被启用的级别都会选择设备调试模式，包括 `debug = "line-tables-only"` 和 `"limited"`。除非显式覆盖，它还会选择设备优化级别 0 —— 即使在 `release` profile 下配了 `debug = true` 也是如此。自动的"仅行号"映射尚未实现。要做优化后的性能分析，请使用下面显式的 `line` 覆盖。参见 Cargo 的 [profile 设置][cargo-profiles]与[构建脚本环境变量][cargo-build-env]。

`CUDA_RUST_DEBUG` 会在构建应用时覆盖由 Cargo 推导出的默认值。不需要在每次启动时调用 `.compile_options()`：

```bash
# 优化后的设备代码，但保留源码行号以便性能分析。
CUDA_RUST_DEBUG=line cargo build --release

# 未优化的设备代码，带源码位置与内联帧。
CUDA_RUST_DEBUG=full cargo build
cuda-gdb --args ./target/debug/my_program
```

可接受的取值是 `none`、`line` 和 `full`。未设置时跟随 Cargo；其它取值会让构建失败。Cargo 会跟踪变化，因此不需要 `cargo clean`。该设置是在构建目标 `cutile-compiler` 库时捕获的，而不是在构建 proc macro 时。针对 `cutile-compiler` 的包级 profile 覆盖会影响这个共享默认值；只针对某个 kernel 所在 crate 的覆盖则不会。仅在运行已经构建好的二进制时设置该环境变量，不会改变设备编译。

`CompileOptions::new()` 与 `CompileOptions::default()` 会继承这个构建默认值。用 `debug_info()` 可以为单次启动替换它。它会同时设置两个调试标志，因此也能关掉构建时的 `full` 默认值：

```rust
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};

my_kernel(args)
    .compile_options(CompileOptions::new().debug_info(DebugInfoLevel::None))
    .sync()?;
```

每个选项都是 JIT 缓存键的一部分，所以这些模式不会共享同一个已编译 kernel 的缓存。各个单独的标志仍然可用，sanitizer 插桩也一样：

```rust
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};

// cuda-gdb：带调试信息，不优化。
my_kernel(args)
    .compile_options(CompileOptions::new().debug_info(DebugInfoLevel::Full))
    .sync()?;

// 性能分析器关联：只要行号信息，完全优化。
my_kernel(args)
    .compile_options(CompileOptions::new().debug_info(DebugInfoLevel::Line))
    .sync()?;

// Compute Sanitizer：访存插桩。
my_kernel(args).compile_options(CompileOptions::new().sanitize_memcheck(true)).sync()?;
```

- `debug_info(level)` 会替换两个调试标志。而单独的 `device_debug(bool)` 与 `lineinfo(bool)` setter 只改自己那一个标志，保留其它继承来的设置。
- `device_debug(true)` 会向设备编译器传 `--device-debug`，并隐含优化级别 0。显式的 `opt_level` 优先级更高，但 `tileiras` 目前在优化级别下会拒绝完整的调试信息。前端也会停止把边界检查提升出循环。它已证明多余、或已移到启动时执行的检查不受影响；无论哪种模式，它们都不会进入设备代码。
- `lineinfo(true)` 会传 `--lineinfo`：为 Nsight Compute 和 Nsight Systems 提供源码行关联，而不改变代码生成。这是用于分析优化后 kernel 的选项。
- `sanitize_memcheck(true)` 会传 `--sanitize=memcheck`，配合 `compute-sanitizer --tool memcheck` 使用。
- `opt_level(n)` 直接选择 `--opt-level`；默认是 3。

完整调试模式描述的是源码位置与内联帧，而**不是** Rust 变量或取值。Tile IR 目前支持对未优化代码做控制流调试，但不支持查看用户变量；参见 [Tile IR 调试信息文档][tile-debug-info]。

用户辅助函数和方法在被内联时会保留其定义所在的源文件与行号，跨模块也一样。核心操作会被归属到用户的调用点。多条生成的指令仍然可能映射到同一行 Rust 源码；这不是一一对应的映射，在开启优化时尤其如此。

## 性能分析

针对单个 kernel，使用 NVIDIA **Nsight Compute**（`ncu`，用 `ncu-ui` 打开报告）：吞吐、占用率、寄存器使用以及停顿。**Nsight Systems**（`nsys`、`nsys-ui`）展示应用时间线：CPU/GPU 调度、传输、同步以及启动间隙。

要做源码关联，请用 `CUDA_RUST_DEBUG=line` 构建。并确保在你查看报告的那台机器上 Rust 源码仍然可用。要分析构建出来的可执行文件，而不是 Cargo 构建过程本身：

```bash
CUDA_RUST_DEBUG=line cargo build --release
ncu --target-processes all ./my_cutile_program
ncu --set full -o profile_report ./my_cutile_program
ncu-ui profile_report.ncu-rep
```

把 `./my_cutile_program` 换成你的二进制路径，例如 `./target/release/my_program`。出现 `ERR_NVGPUCTRPERM` 报告，意味着这台机器的 GPU 性能计数器访问权限需要由管理员开启；改行号信息设置并不能解决它。

用 Nsight Systems 看 CPU/GPU 调度：

```bash
nsys profile ./my_cutile_program
nsys-ui report.nsys-rep
```

关注启动间隙、不必要的同步、内存传输的重叠情况，以及相互独立的 kernel 是否真的在不同的流上重叠。

Tile IR 显示与 Rust 源码行关联是两件事。NVIDIA 把 Tile IR Source 视图列为 [Nsight Compute 2026.3][ncu-tile-ir] 的新特性。如果看不到它，请先记录 Nsight Compute 与 CUDA Toolkit 版本、编译选项、cubin 和报告，再去诊断元数据问题。上面那些独立的 Tile IR 转储与该视图无关，照样有用。

## 回归检查

仓库中有一些不需要 GPU 的测试，用来验证源码来源信息与编译选项的缓存隔离：

```bash
cargo test -p cutile --test debug_info
bash scripts/test_debug_defaults.sh
```

第二条命令需要 `jq`。它会检查 Cargo profile、调试断言与构建依赖设置的相互独立、显式覆盖，以及"仅行号"映射的限制。它还会检查修改运行时环境不会改变构建出的默认值。额外的检查是 opt-in 的：

```bash
# 需要 tileiras、nvdisasm 和 readelf，但不需要 GPU。
cargo test -p cutile --test debug_info cubin_has_rust_lines_and_inline_frames -- --ignored

# 需要受支持的 GPU 和 tileiras。
cargo test -p cutile --test debug_info kernel_executes_in_all_debug_modes -- --ignored

# 还需要 cuda-gdb 和 timeout；会在某个辅助函数的源码行停下，
# 并检查它嵌套的内联调用栈，然后才继续完成 kernel。
cargo test -p cutile --test debug_info cuda_gdb_stops_in_cross_file_helper -- --ignored
```

[tile-debug-info]: https://docs.nvidia.com/cuda/tile-ir/latest/sections/debug_info.html
[ncu-tile-ir]: https://developer.nvidia.com/nsight-compute-2026_3-new-features
[cargo-profiles]: https://doc.rust-lang.org/cargo/reference/profiles.html#debug
[cargo-build-env]: https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts

## 调试检查清单

- 形状与操作及启动分区相匹配。
- tensor 尺寸与分区形状兼容。
- 元素类型一致，或者已显式转换。
- 小输入与 CPU 参考实现一致。
- 数值敏感的代码处理了上溢和下溢。
- 裸指针的生命周期长于所有使用它的 GPU 工作。
- `print_ir` 显示了预期的 Tile IR 操作。
- 在正确性检查通过之后才采集性能剖析数据。

---

性能优化策略见[性能](performance.md)，自定义 CUDA kernel 见[互操作](interoperability.md)。
