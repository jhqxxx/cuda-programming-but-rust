# 边界检查的放置

安全的 partition API 默认会做访问检查。即使 kernel 里检查数量不多，热点循环内部的一个检查也可能打断加载与计算之间的软件流水。编译器会移除它能证明多余的检查，并在可能时把另一些检查移出循环。

每个被检查的维度有三种结果之一：

- **已消除（Discharged）：** kernel 中不再残留任何检查。这包括通过 `iter_indices()` 的存储、使用分区自身坐标的加载、对静态形状的常量索引，以及被声明的 `preconditions` 或在启动时校验过的跨 tensor 事实所覆盖的访问。
- **提升（Hoisted）：** 检查在循环之前只运行一次；如果编译器能证明移到外层循环是安全的，则会继续提升到更外层循环之前。
- **原地（In place）：** 检查在每次迭代的访问处运行。

在入口上设置 `deny_in_kernel_checks = true`，可以拒绝任何仍然留在 kernel 内的检查。诊断信息会指出具体的访问位置，并建议如何重构它。

## 检查何时会被提升

循环内某个访问的检查，在以下条件**全部**满足时（逐索引坐标判断）会被提升到循环之前：

| 索引坐标的形式 | 放置位置 |
|---|---|
| 在循环之前计算出的值 | 提升 |
| 编译期常量，或者边界已知的常量值 | 提升（或针对静态形状被消除） |
| `for j in lo..hi` 循环的循环变量 `j` | 提升，在 `hi - 1` 处检查 |
| 常量 `a`、`b` 对应的 `a * j + b` | 当能证明该极端迭代适合 `i32` 时，提升到极端迭代处；否则原地 |
| `(lo..hi).step_by(k)` 循环的循环变量 | 当 `lo`、`hi`、`k` 都是编译期常量时，提升并在*最后到达*的索引 `lo + k * floor((hi - 1 - lo) / k)` 处检查；否则原地 |
| 在循环体**内部**计算出的值（上面几种形式除外） | 原地 |
| 写在循环内 `if`/`else` 里的任何访问 | 原地 |
| 位于包含 `continue` 的循环体中的任何访问 | 原地 |

此外，一个已提升的检查会沿着"直接嵌套、且循环次数在静态上非零、且循环体不含 `continue`"的循环继续向外攀升，直到遇到第一个它所依赖其边界的循环为止。提升后的检查都带有保护，因此执行零次的循环绝不会触发陷阱。提升确实会改变违规**何时**被报告：提升后的检查在循环开始之前就触发陷阱，因此排在出错那次迭代之前的那些迭代 —— 以及它们本会执行的存储 —— 都不会执行。它绝不会改变 kernel **是否**触发陷阱：提升后的检查只测试循环真正到达过的索引值，这也是为什么含 `continue` 的循环体（部分迭代会跳过该访问）或者步长在运行时决定的阶梯循环（到达集合未知）会保留原地检查。

要帮助编译器把检查移出热点循环：

- **把索引算术提到最内层循环之上。** 写在 K/V 循环之前的 `let kv_head = q_head / group;` 能让所有使用它的检查都被提升；同样的表达式写在循环体内则不行（编译器目前不会追踪循环体内的不变算术 —— 它按位置来证明不变性）。
- **用循环变量本身、或它的仿射表达式**来索引热点循环中的访问，并把循环写成 `for j in lo..hi`；只有当 `lo`、`hi`、`k` 都是编译期常量时，`step_by(k)` 循环的检查才会被提升。
- **让热点循环中的访问保持无条件。** `if` 之下的加载可能在任何一次迭代上都不执行，所以它的检查无法移动；要么把条件提到循环之外，要么接受原地检查。含 `continue` 的循环体同理：它之后的每个访问都是有条件的。
- **让索引算术不发生回绕。** 只有当一个操作能被证明不会使 `i32` 溢出时，范围事实才能穿过该操作存活；可能回绕的表达式会丧失它的事实（即使后续的 `max`/`%` 步骤把数学上的取值范围拉回界内），该访问就要对实际运行时取值承担一次原地检查。

```rust
for index in out.iter_indices() {
    let (q_tile, q_head, _) = index.components();
    let kv_head = q_head / GROUP;          // 在循环之上：会提升
    for j in 0i32..kv_tiles {
        let k = k_part.load_pipelined::<L>([kv_head, j, 0i32]);
        //                                  ^ 已提升  ^ 已提升（在 kv_tiles - 1 处检查）
        // ...
    }
}
```

## 完全离开 kernel 的检查

如果某个检查的操作数在启动时就已知，它就可以在生成的宿主启动器里运行。tensor 的范围（extent）符合条件；循环变量和已加载的值不符合。启动器会检查实际形状，并以一条指明该检查的错误拒绝非法启动。这样不需要任何设备寄存器或指令。

例如，来自 `for j in 0..num_tiles(&p, a)` 的索引，或者映射分区 `iter_indices()` 的索引，都携带了来自该 tensor 的边界。用它去索引另一个 tensor，就会产生一个针对 tile 数量的启动检查，例如 `ceil(dim(x, 1)/BK) <= ceil(dim(y, 0)/BK)`。

`cutile-examples/examples/persistent_gemm.rs` 里的持久化 GEMM 就使用了这些派生检查。它的映射分量索引 `x` 和 `y`，它的 K 循环遍历 `num_tiles`。所有形状检查都在启动时运行，因此它可以在不加任何标注、不加额外 const 泛型的情况下，以 `deny_in_kernel_checks = true` 构建。

声明的 `preconditions` 更进一步：启动器本身已经用真实形状校验了每一个声明的事实，因此编译器可以直接假定它，并在编译期消除对应的检查。一个声明了 `dim(x, 1) % 64 == 0` 的 kernel，对于匹配的绑定在任何地方都不会生成东西；而 `dim(a, i) == dim(b, j)` 与派生形式一样，精确地关联两个 tensor 的轴。当你希望把契约显式体现在签名中，或者需要一条遍历无法派生的事实时才声明 precondition；注意声明式的相等比派生式的比较更严格（对范围用 `==`，对 tile 数量用 `<=`），因此它会拒绝一些派生形式本可接受的启动。

`with_bounds`/`Dim` 这一族标注已废弃：它所能证明的一切都已被上面的派生事实与声明的 precondition 涵盖，而且检查落在启动时而不是可能落在 kernel 内。

## 读懂编译器的决定

`CUTILE_JIT_TIMING=1` 会在每条编译行上报告逐 kernel 的统计：

```text
CUTILE_JIT_TIMING module=kernels function=fmha_prefill ... \
    checks_discharged=3 checks_hoisted=4 checks_in_place=2
```

`CUTILE_JIT_LOG=1` 会解释每一个仍留在循环体内的检查：

```text
[cutile::jit] bounds check for dim 1 stays in the loop body: index is
computed inside the loop body
```

把生成的 Tile IR 与未检查版本对比，就能看到它对调度的影响。在入口上设置 `unchecked_accesses = true`，把 kernel 标为 `unsafe`，并用 `unsafe { ... }` 包住启动。把两个变体都转储到 stderr 再比较：

```bash
dump() { CUTILE_DUMP=ir CUTILE_DUMP_FILTER=kernels::fmha_prefill ./my_app 2>&1 \
         | sed -n '/=== CUTILE DUMP: ir/,/^}/p'; }
dump > safe.ir
# flip the kernel to its unsafe twin, then:
dump > raw.ir
diff safe.ir raw.ir
```

持久化 GEMM 示例为它的安全版本与 unsafe 版本生成了完全相同的 IR：103 个操作，循环体一致。它的检查全部在启动器中运行。

在其它 kernel 中，出现 `assert` 就意味着设备上仍然残留了一个检查。如果你期望所有检查都离开 kernel，`grep -c assert safe.ir` 应当返回零；`deny_in_kernel_checks = true` 会在编译期间强制这一点。循环内位于加载与 `mma` 操作之间的额外比较、选择或分支会打断流水。而循环之前的一个已提升检查没有这种代价。

如果 IR 相同但性能不同，就比较寄存器数量。当所有检查都离开 kernel 时，两者应当一致。

有两个消融开关让你无需重新构建就能用同一个二进制测量检查的放置：`CUTILE_DISABLE_CHECK_HOISTING=1` 把每一个残余检查钉在它的访问点上，而 `CUTILE_FORCE_DEVICE_CHECKS=1` 进一步压制所有证明，对每次访问按实际取值做双侧检查 —— 这就是测试套件用来对比检查放置的参照语义。

## 什么时候该动用 `unsafe`

先测量 `unchecked_accesses = true` 的效果。在用于开发检查提升的那个 flash-attention prefill kernel 上（RTX 5090，`checks_in_place=2`），完全检查的 kernel 每次调用 55.0 µs，而关闭全部检查的下限是 53.6 µs —— 约 2.5%，且这 2.5% 全部来自那两个针对调度派生坐标的原地检查，它们每个持久化索引只执行一次，而不是每次内层循环迭代都执行。同一个 kernel 的 unsafe 孪生版本跑 56.7 µs；检查版本反而更快。

当检查仍留在热点循环里时，未检查访问可能有帮助：阶梯循环的索引、依赖数据的索引（从内存加载的值）、有条件的访问，以及编译器无法证明不回绕的索引算术。如果 `CUTILE_JIT_LOG` 显示你的内层循环里还有原地检查，而重构又移不走它们，那就测一测 `unchecked_accesses = true` 是否有帮助。
