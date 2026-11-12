# CUDA 与 Tile IR 兼容性

一个 kernel 既需要受支持的 Tile IR 版本，也需要兼容的目标 GPU。下面的表格来自同一份需求登记表（requirements registry），编译器、字节码写出器、启动守卫以及示例/测试的前置条件都在使用它。

这里的版本是**实际输出的 Tile IR 版本**，不是 `cuDriverGetVersion` 报告的版本。我们会询问选定的 `tileiras` 接受哪些格式，并选择我们的写出器也支持的最新一个。对于不支持 `--list-versions` 的较旧汇编器，则通过编译一个小型代表性 kernel 来探测。显式设置的 `CUTILE_BYTECODE_VERSION` 必须是受支持的；我们不会把它降级。

`CUTILE_TILEIRAS_PATH` 的优先级高于工具包目录。更新的驱动并不会让旧的汇编器读懂更新的字节码。

调用 tile 库函数（[Tile 库](../guide/tile-libraries.md)）要求选定的版本是 13.5。

<!-- BEGIN TILE IR REQUIREMENTS -->
| 特性 | 13.2 | 13.3 | 13.4 | 13.5 | 目标要求 |
|---|---|---|---|---|---|
| baseline Tile IR | Yes | Yes | Yes | Yes | `sm_80+` |
| alloca | — | Yes | Yes | Yes | 任意受支持目标 |
| pack/unpack | — | Yes | Yes | Yes | 任意受支持目标 |
| block-scaled MMA | — | Yes | Yes | Yes | `sm_100+` |
| gather_scatter_view | — | Yes | Yes | Yes | 任意受支持目标 |
| strided_view | — | Yes | Yes | Yes | 任意受支持目标 |
| atomic_red_view_tko | — | Yes | Yes | Yes | 任意受支持目标 |
| insert | — | — | Yes | Yes | 任意受支持目标 |
| fpowi | — | — | Yes | Yes | 任意受支持目标 |
| GDC tokens | — | — | Yes | Yes | 任意受支持目标 |
| memory_fence_alias_tko | — | — | Yes | Yes | 任意受支持目标 |
| i4 | — | Yes | Yes | Yes | 任意受支持目标 |
| f8E4M3FN/f8E5M2 | Yes | Yes | Yes | Yes | `sm_90+` |
| f8E8M0FNU | Yes | Yes | Yes | Yes | `sm_100+` |
| f4E2M1FN | — | Yes | Yes | Yes | `sm_100+` |
| f8E5M3FNU | — | — | Yes | Yes | `sm_107+` |
| cuda_tile.mmaf_scaled with f8E5M3FNU scales | — | — | Yes | Yes | 仅 `sm_107` |
| ptr_attr | — | — | Yes | Yes | 任意受支持目标 |
| unicast/multicast pointer | — | — | — | Yes | `sm_100` / `sm_103` / `sm_107` |
| multimem view operations | — | — | — | Yes | `sm_100` / `sm_103` / `sm_107` |
| atomic_red_view_tko.inbounds=true | — | — | — | Yes | 任意受支持目标 |
| if/select optimization hints | — | — | — | Yes | 任意受支持目标 |
| min_ctas_per_sm | — | — | — | Yes | 任意受支持目标 |
| argument attributes | — | — | — | Yes | 任意受支持目标 |
| rounding_mode=nearest_away | — | — | Yes | Yes | 任意受支持目标 |
| ftof additional rounding modes (type-dependent) | — | — | Yes | Yes | 任意受支持目标 |
| ftoi.saturating | — | — | Yes | Yes | 任意受支持目标 |
| inbounds=true | — | — | Yes | Yes | 任意受支持目标 |
| mmaf.fast_acc | — | Yes | Yes | Yes | 任意受支持目标 |
| exp.rounding_mode (non-approximate) | — | Yes | Yes | Yes | 任意受支持目标 |
| module.producer | — | Yes | Yes | Yes | 任意受支持目标 |
| global.constant | — | Yes | Yes | Yes | 任意受支持目标 |
| global.symbol_visibility | — | Yes | Yes | Yes | 任意受支持目标 |
| num_worker_warps_per_cta | — | Yes | Yes | Yes | 任意受支持目标 |
| num_cta_in_cga greater than 1 | Yes | Yes | Yes | Yes | `sm_90+` |
| bf16 atomic add | — | Yes | Yes | Yes | `sm_90+` |
| return inside loop | — | — | Yes | Yes | 任意受支持目标 |
| target sm_90 | — | Yes | Yes | Yes | 仅 `sm_90` |
| target sm_107 | — | — | Yes | Yes | 仅 `sm_107` |
| programmatic dependent launch | — | — | Yes | Yes | `sm_90+` |
| preview$call (foreign tile-library call) | — | — | — | Yes | 任意受支持目标 |

| 目标 | 最低输出 Tile IR 版本 |
|---|---|
| `sm_80` | 13.2 |
| `sm_86` | 13.2 |
| `sm_87` | 13.2 |
| `sm_88` | 13.2 |
| `sm_89` | 13.2 |
| `sm_90` | 13.3 |
| `sm_100` | 13.2 |
| `sm_103` | 13.2 |
| `sm_107` | 13.4 |
| `sm_110` | 13.2 |
| `sm_120` | 13.2 |
| `sm_121` | 13.2 |
<!-- END TILE IR REQUIREMENTS -->

if/select 优化提示那一行描述的是前端 IR 里的接受情况。这些提示在输出 13.5 字节码时会被丢弃，对后端性能没有影响。

各项要求是叠加的。例如 FP4 打包既需要 `pack/unpack` 操作，也需要 FP4 元素类型。Hopper 目标要求 Tile IR 13.3，即使那些操作本身在更早的版本里就已存在。

架构编号并不是一条特性等级链：`sm_120` 支持 NVFP4，但不支持使用 `f8E5M3FNU` 缩放值的缩放式 MMA。操作数类型、形状和属性取值仍然必须满足操作的校验器。特别是，`num_worker_warps_per_cta` 在 13.3 与 13.5 中接受 1、2、4、8、16 或 32，而在 13.4 中只接受 4 或 8。

PDL 还需要驱动的 `cuLaunchKernelEx` 入口点。可选的宿主 API 是在运行时解析的；它们的可用性与设备 IR 的需求是两回事。见 [PDL 启动契约](host-api.md#programmatic-dependent-launch)。

驱动/工具包的组合必须满足 [NVIDIA 的 CUDA 兼容性规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。我们不会一律强制"驱动版本不低于工具包版本"这种检查。

## CUDA 13.5 新增特性

13.5 这层 API 包含单播/多播指针分类、`multimem_load_reduce_view_tko`、`multimem_reduce_view_tko`、`multimem_store_view_tko`、原子视图的 inbounds 承诺、最小驻留提示，以及函数参数属性。具名的 `simt` 预览与这层稳定 API 是分开的。cuTile Rust 通过那个预览支持[外部 tile 库调用](../guide/tile-libraries.md)；受支持的操作数与当前限制请见该指南。

裸 Rust API 以 unsafe 函数的形式暴露在 `cutile::tileir` 中，定义在 `cutile/src/_tileir.rs`：`ptr_with_attr_unicast`、`ptr_with_attr_multicast`、三个 `multimem_*_view_tko` 操作，以及 `atomic_red_view_tko_with_inbounds`。`multimem` 模块提供 Add、Min、Max、And、Or、Xor、Umin 和 Umax 归约标记。这些 unsafe API 要求映射是存活的且分类正确，并且要求显式的内存序协议。一个 Tile IR token 并不会让彼此独立的 GPU 之间同步。

单播与多播的契约、可用的同步原语以及可运行的双 GPU 示例，见[不安全的 GPU 通信](unsafe-comms.md)。

Likelihood 使用安全的 `Likely` 扩展 trait，由 `use cutile::core::*` 导入。接收者必须是一个标量 `bool` 或一个布尔 `Tile`，该方法会保持它的类型与取值：

```rust
if condition.likely::<true>() {
    // 普通控制流；已有的 DSL 限制仍然适用。
}
let result = select(mask.likely::<false>(), a, b);
```

这个预测是建议性的：提示给错永远不会影响正确性。拷贝和内联辅助函数会保留 IR 提示；布尔运算和控制流汇合点可能会丢弃它。请把它应用在最终的那个条件上。

**Tile IR 13.5 在输出字节码时会丢弃 likelihood 提示。** 这些方法目前要求 13.5，但在该目标上对后端性能没有影响。普通的 `if` 与 `select` 使用标准的 13.5 编码，不带优化提示字段。

入口标注可以接受 `optimization_hints = (sm_100 = (min_ctas_per_sm = 2,),)` 以及一个 `default` 组。`CompileOptions::min_ctas_per_sm` 会覆盖当前 GPU 对应的入口提示，并且参与 JIT 缓存标识。最小驻留的取值范围是 0–32；0 表示禁用该约束。提示不会改变程序语义。函数参数属性可以通过底层 IR builder 使用。`DISubprogram::is_declaration` 暴露 13.5 的调试声明标志；普通的定义保持该标志为清除状态，而较旧的写出器会拒绝声明。

具名的 `simt` 预览有它自己的 feature 门，并不是一个带版本的 CTK 13.5 API；13.6 的 MMA 溢出属性同样不在这层带版本的 API 之内。

多播操作需要支持 NVLink SHARP 的 fabric，以及受支持的 `sm_100`、`sm_103` 或 `sm_107` 家族。全为 false 的原子 inbounds 数组仍沿用较旧的版本下限。

### 选择工具包

为宿主架构安装一套 CUDA 13.5 工具包，并在默认环境使用较旧工具包时显式选择它的汇编器：

```bash
export CUDA_TOOLKIT_PATH=/path/to/cuda-13.5
export CUTILE_TILEIRAS_PATH="$CUDA_TOOLKIT_PATH/bin/tileiras"
export CUTILE_BYTECODE_VERSION=13.5
```

跨项目统一使用同一套工具包安装。汇编器及其编译器后端必须支持所请求的字节码版本；只改运行时库的路径并不会升级汇编器。tile 库必须针对链接它的那套工具包来构建。

### 验证

在 `cutile-rs/` 下运行。常规的 CPU runner 已经包含新的前端、字节码、校验和 GPU kernel 编译测试。当有配套的 CUDA Tile IR 转换器可用时，下面这条还会用该转换器独立的 C++ 字节码读取器与校验器来检查输出的镜像：

```bash
CUTILE_REFERENCE_TRANSLATE=/path/to/cuda-tile-translate \
  cargo test -p cutile --test tile_ir_13_5 --test tile_ir_13_5_multicast
```

配合上面那套共享的 13.5 环境，可以在没有 GPU 的情况下为目标 `sm_100` 汇编特性内核与多播执行内核：

```bash
cargo test -p cutile --test tile_ir_13_5 --test tile_ir_13_5_multicast \
  assemble_ -- --ignored
```

然后在带有 NVLink SHARP 的双卡（或更多）B200 实例上：

```bash
cargo test -p cutile --test tile_ir_13_5_multicast \
  multicast_load_reduce_and_store_execute_from_each_gpu -- --ignored --test-threads=1
bash scripts/run_gpu_tests.sh
```

这条专门的多播测试会让每张 GPU 分别作为发起方跑一遍，检查归约加载的输出以及归约/存储之后两块设备上的内容；如果所需的设备或 fabric 缺失就会失败。它默认被 ignore，因为单卡机器跑不了。汇编器与硬件执行仍与通过 CPU 测试套件是两件独立的事。

## 让表格保持最新

需求定义在 `cutile-ir/src/requirements.rs`，依据的是[带版本的 Tile IR 规范](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html)。修改它们之后，用下面的命令打印 README 与参考文档中的表格：

```bash
cargo run -p cutile-ir --example feature_matrix
```

`documented_matrices_match_the_registry` 测试会校验这两张已提交入库的表格。CPU 测试不需要驱动就能验证这些需求；汇编器的 CI 覆盖 13.2、13.3 和 13.4。示例会把不受支持的前置条件报告为 `SKIP`；而发现、编译和执行阶段的错误仍然算作失败。
