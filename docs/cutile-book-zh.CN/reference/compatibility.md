# CUDA 与 Tile IR 兼容性

一个 kernel 既需要受支持的 Tile IR 版本，也需要兼容的目标 GPU。下面的表格来自同一份需求登记表（requirements registry），编译器、字节码写出器、启动守卫以及示例/测试的前置条件都在使用它。

这里的版本是**实际输出的 Tile IR 版本**，不是 `cuDriverGetVersion` 报告的版本。我们会询问选定的 `tileiras` 接受哪些格式，并选择我们的写出器也支持的最新一个。对于不支持 `--list-versions` 的较旧汇编器，则通过编译一个小型代表性 kernel 来探测。显式设置的 `CUTILE_BYTECODE_VERSION` 必须是受支持的；我们不会把它降级。

`CUTILE_TILEIRAS_PATH` 的优先级高于工具包目录。更新的驱动并不会让旧的汇编器读懂更新的字节码。

<!-- BEGIN TILE IR REQUIREMENTS -->
| 特性 | 13.2 | 13.3 | 13.4 | 目标要求 |
|---|---|---|---|---|
| baseline Tile IR | Yes | Yes | Yes | `sm_80+` |
| alloca | — | Yes | Yes | 任意受支持目标 |
| pack/unpack | — | Yes | Yes | 任意受支持目标 |
| block-scaled MMA | — | Yes | Yes | `sm_100+` |
| gather_scatter_view | — | Yes | Yes | 任意受支持目标 |
| strided_view | — | Yes | Yes | 任意受支持目标 |
| atomic_red_view_tko | — | Yes | Yes | 任意受支持目标 |
| insert | — | — | Yes | 任意受支持目标 |
| fpowi | — | — | Yes | 任意受支持目标 |
| GDC tokens | — | — | Yes | 任意受支持目标 |
| memory_fence_alias_tko | — | — | Yes | 任意受支持目标 |
| i4 | — | Yes | Yes | 任意受支持目标 |
| f8E4M3FN/f8E5M2 | Yes | Yes | Yes | `sm_90+` |
| f8E8M0FNU | Yes | Yes | Yes | `sm_100+` |
| f4E2M1FN | — | Yes | Yes | `sm_100+` |
| f8E5M3FNU | — | — | Yes | `sm_107+` |
| cuda_tile.mmaf_scaled with f8E5M3FNU scales | — | — | Yes | 仅 `sm_107` |
| ptr_attr | — | — | Yes | 任意受支持目标 |
| rounding_mode=nearest_away | — | — | Yes | 任意受支持目标 |
| ftof additional rounding modes (type-dependent) | — | — | Yes | 任意受支持目标 |
| ftoi.saturating | — | — | Yes | 任意受支持目标 |
| inbounds=true | — | — | Yes | 任意受支持目标 |
| mmaf.fast_acc | — | Yes | Yes | 任意受支持目标 |
| exp.rounding_mode (non-approximate) | — | Yes | Yes | 任意受支持目标 |
| module.producer | — | Yes | Yes | 任意受支持目标 |
| global.constant | — | Yes | Yes | 任意受支持目标 |
| global.symbol_visibility | — | Yes | Yes | 任意受支持目标 |
| num_worker_warps_per_cta | — | Yes | Yes | 任意受支持目标 |
| num_cta_in_cga greater than 1 | Yes | Yes | Yes | `sm_90+` |
| num_worker_warps_per_cta restricted to 4 or 8 | — | — | Yes | 任意受支持目标 |
| bf16 atomic add | — | Yes | Yes | `sm_90+` |
| return inside loop | — | — | Yes | 任意受支持目标 |
| target sm_90 | — | Yes | Yes | 仅 `sm_90` |
| target sm_107 | — | — | Yes | 仅 `sm_107` |
| programmatic dependent launch | — | — | Yes | `sm_90+` |

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

各项要求是叠加的。例如 FP4 打包既需要 `pack/unpack` 操作，也需要 FP4 元素类型。Hopper 目标要求 Tile IR 13.3，即使那些操作本身在更早的版本里就已存在。

架构编号并不是一条特性等级链：`sm_120` 支持 NVFP4，但不支持使用 `f8E5M3FNU` 缩放值的缩放式 MMA。操作数类型、形状和属性取值仍然必须满足操作的校验器。特别是，`num_worker_warps_per_cta` 在 13.3 中接受 1、2、4、8、16 或 32，而在 13.4 中只接受 4 或 8。

PDL 还需要驱动的 `cuLaunchKernelEx` 入口点。可选的宿主 API 是在运行时解析的；它们的可用性与设备 IR 的需求是两回事。见 [PDL 启动契约](host-api.md#programmatic-dependent-launch)。

驱动/工具包的组合必须满足 [NVIDIA 的 CUDA 兼容性规则](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。我们不会一律强制"驱动版本不低于工具包版本"这种检查。

## 让表格保持最新

需求定义在 `cutile-ir/src/requirements.rs`，依据的是[带版本的 Tile IR 规范](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html)。修改它们之后，用下面的命令打印 README 与参考文档中的表格：

```bash
cargo run -p cutile-ir --example feature_matrix
```

`documented_matrices_match_the_registry` 测试会校验这两张已提交入库的表格。CPU 测试不需要驱动就能验证这些需求；汇编器的 CI 覆盖 13.2 和 13.3，并且在对应镜像配置好时会启用一条 13.4 的流水线。示例会把不受支持的前置条件报告为 `SKIP`；而发现、编译和执行阶段的错误仍然算作失败。
