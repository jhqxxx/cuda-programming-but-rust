# 模糊测试与差异测试

cuda-oxide 的顺利路径很容易用示例来测试：`vecadd`、`gemm`、`sharedmem` 之类
证明了已知程序仍然可用。但编译器不只在已知程序上失败。它们在那些没人想过要
手写的古怪小角落里失败 —— 分支之后的类型转换、调用终结符里的元组、那个没人
邀请来参加派对的整数宽度。

这就是模糊测试登场的地方。

cuda-oxide 使用一个小型的、基于 rustlantis 的测试框架来生成随机 custom MIR
程序，让它们分别经过普通的 Rust CPU 后端和 cuda-oxide 的 GPU 后端，并比较中间
值的紧凑轨迹。目的不是证明整个 Rust 在 GPU 上都是正确的。目的更朴素、也更有用：
用并非我们手写的程序来压测 MIR 导入器、下降流水线、LLVM 导出、PTX 生成以及
运行时执行。

---

## 我们在比较什么

乍一看，"比较 CPU 与 GPU 的执行"听起来很可疑。CPU 和 GPU 有不同的执行模型：
一个标量线程对比成千上万个 SIMT lane、发散的控制流、不同的内存空间、不同的
同步规则。笼统的 CPU 与 GPU 语义比较会是一种非常花哨的自欺欺人方式。

所以测试框架刻意避免了这一点。

它把生成的 MIR 作为标量 GPU 程序运行：

```text
<<<1, 1>>>
```

一个 block。一个线程。没有线程间通信。没有调度上的戏码。GPU 被当作同一份标量
MIR 的第二个代码生成目标，而不是一种并行编程模型。

比较的内容是：

```text
same generated MIR
  -> normal rustc/LLVM CPU execution
  -> cuda-oxide -> LLVM IR -> PTX -> CUDA execution
  -> compare trace hash
```

如果哈希匹配，那么生成的程序在两条路径上观察到了相同的中间值序列。如果不同，
那么 cuda-oxide 路径中的某个地方值得关注。

> **注意**
>
> 轨迹刻意保持紧凑。每次 `dump_var(...)` 调用不是把每个中间值都拷回宿主，而是
> 用逐字节哈希把值折叠进一个 `u64`。这让生成的测试运行成本低、易于比较。

---

## 各个活动的部件

模糊测试装置分为四个部分：

| 部件                                  | 角色                                           |
| :--------------------------------------| :--------------------------------------------- |
| `cuda-oxide/crates/fuzzer`                        | 共享轨迹 API 与内置的 rustlantis               |
| `cuda-oxide/crates/fuzzer/tools/mir_generator.py` | 种子到 `generated_case.rs` 的适配器            |
| `cuda-oxide/crates/fuzzer/tools/run_seed.py`      | 批量运行器和产物记录器                          |
| `rustlantis-smoke`                     | 稳定的 CPU/GPU 执行测试框架                     |

`cuda-oxide/crates/fuzzer` 是一个普通的工作区 crate，但它的库接口是 `no_std`
的。设备代码从那里导入 `trace_reset`、`trace_finish` 以及泛型 `dump_var`。
实际的 rustlantis 源码内置在 `cuda-oxide/crates/fuzzer/rustlantis` 下；它作为
外部生成器被调用，而不是作为 Rust 库依赖使用。

`rustlantis-smoke` 示例位于
`cuda-oxide/crates/rustc-codegen-cuda/examples/` 下。它拥有宿主/GPU 启动逻辑
和一个小型手工编写的 sanity 测试，然后包含一个生成的文件：

```text
cuda-oxide/crates/rustc-codegen-cuda/examples/rustlantis-smoke/src/generated_case.rs
```

模糊测试工具为每个种子重写该文件。示例中的其他一切保持稳定。这让测试框架易于
评审：如果某个种子失败，生成的 MIR 被隔离在一个地方。

---

## 种子流水线

一个种子走这样的路径：

1. **生成。** rustlantis 接收一个数字种子和一个很小的配置。相同的种子加相同的
   配置意味着相同的 custom MIR 程序。
2. **提取。** `mir_generator.py` 取出第一个生成的 `#[custom_mir]` 函数。
3. **适配。** rustlantis 发出像 `dump_var(a, b, c)` 这样的调用。适配器把它们
   重写为元组局部变量和一次泛型 `fuzzer::dump_var(...)` 调用，因为 custom MIR
   的调用操作数对元组表达式很挑剔。
4. **注入。** 适配后的函数和一个小包装器被写入 `src/generated_case.rs`。
5. **运行。** `cargo oxide run rustlantis-smoke` 执行 CPU 参考实现和 GPU kernel。
6. **分类。** `run_seed.py` 记录该种子是通过、不匹配、编译失败，还是超出了
   适配器当前的支持范围。

已检入的生成用例目前使用种子 `19`，因为它展示了我们希望从测试框架得到的重要
性质：多个中间转储，而不只是一个最终值。

```rust
__rl_dump0 = (Move(_1), Move(_2), Move(_3), Move(_4));
Call(_9 = dump_var(Move(__rl_dump0)), ReturnTo(bb4), UnwindUnreachable())

__rl_dump1 = (Move(_6),);
Call(_9 = dump_var(Move(__rl_dump1)), ReturnTo(bb5), UnwindUnreachable())
```

这意味着最终的轨迹哈希包含来自程序中某一处的若干个值，以及来自更靠后位置的
另一个值。它仍然紧凑，但不再仅仅是"返回值匹配了吗？"

---

## 运行它

运行一个种子：

```bash
python3 cuda-oxide/crates/fuzzer/tools/run_seed.py --seed 192
```

运行一批：

```bash
python3 cuda-oxide/crates/fuzzer/tools/run_seed.py --start 0 --count 20 --keep-going
```

有用的标志：

- `--keep-going`：在一个种子失败后继续。
- `--keep-logs`：为通过的种子也写出日志。
- `--no-build`：复用已经构建好的 rustlantis 生成器。
- `--append-summary`：追加到现有摘要，而不是替换它。

默认情况下，`summary.jsonl` 在每次运行开始时被替换。这使它回答那个显而易见
的问题："我刚才那次运行发生了什么？"如果你想要历史记录，就用
`--append-summary` 显式选择。

---

## 解读结果

运行器为每个种子打印一行，然后打印完整摘要：

```text
results:
  seed 0: UNSUPPORTED [adapter] unsupported return type for return-value tracing: *const i8 (...)
  seed 1: COMPILE_FAIL [backend] Unsupported construct: Type translation not yet implemented for: RigidTy(Char) (...)
summary: COMPILE_FAIL=1, UNSUPPORTED=1
```

各状态的含义：

| 状态                     | 含义                                                   |
| :----------------------- | :------------------------------------------------------|
| `PASS`                   | CPU 与 GPU 轨迹匹配                                    |
| `MISMATCH`               | CPU 与 GPU 轨迹不同                                    |
| `COMPILE_FAIL [backend]` | 适配器产生了一个用例，但 cuda-oxide 让它失败了          |
| `UNSUPPORTED [adapter]`  | rustlantis 生成了 MIR，但适配器拒绝了它                 |

`MISMATCH` 是最需要严肃对待的结果。两条路径都编译并运行了，但观察到了不同的值。
那是一个潜在的后端正确性 bug。

`COMPILE_FAIL [backend]` 意味着生成的用例通过了适配器并进入了 cuda-oxide。该
失败可能仍是预期的 —— 例如一个目前不支持的 MIR 类型 —— 但拒绝它的组件是后端。

`UNSUPPORTED [adapter]` 意味着 rustlantis 产生了一个程序，但我们的适配器拒绝
把它变成一个 smoke 用例。例如：

```text
unsupported return type for return-value tracing: *const i8
```

这个拒绝来自函数边界，而不是某个转储点。rustlantis 只转储它能哈希的值，而适配器
接受所有这些值。适配器仍然可能拒绝的是函数本身：一个轨迹 API 无法哈希的返回
类型，或者一个它没有字面量的参数类型。轨迹会哈希这些标量：

```text
bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, char,
f32, f64
```

它也会哈希该列表中任何内容的数组和元组，嵌套多深都行：轨迹只是把内部的每个标量
都哈希一遍。元组最多支持 5 个元素，与 `TraceDump` 的实现相匹配。

```text
[u128; 4]                  hashes fine
(u32, char)                hashes fine
(i8, i8, i8, i8, i8, i8)   rejected: six elements
*const i8                  rejected: pointer
```

被排除的内容：裸指针和引用，因为 CPU 和 GPU 对同一对象持有不同的地址，所以
指针的哈希永远不可能匹配；以及元素多于 5 个的元组。因此许多适配器级别的不支持
情形既不是"糟糕的 MIR"也不是 cuda-oxide 的 bug。指针拒绝在设计上是永久的；
再放宽多少也无法让那些哈希一致。元组限制只是一个空白。测试框架还需要一个
`TraceDump` 实现，也许再来点零食，才能哈希六元素元组。

---

## 产物

每个种子的日志位于：

```text
cuda-oxide/crates/fuzzer/artifacts/
```

失败日志包括：

- 种子
- 状态和阶段
- 原因
- 返回码
- 命令
- 完整的命令输出
- 生成的 `generated_case.rs` 快照（如果存在的话）

生成的快照很重要。如果某个后端失败出现在 CI 或一次长时间批量运行中，日志足以
让你看到触发它的那个确切的 MIR 形状程序。种子让你能重新生成它，但快照省去了你
一次往返。

---

## 当前限制

当前配置刻意很小。它让第一阶段聚焦于标量 custom MIR 和后端管道，而不是一次性
覆盖每一种 Rust 构造。这就是为什么许多早期种子被分类为 `UNSUPPORTED [adapter]`。

放宽计划是渐进的，其中两步已经落地：适配器现在接受轨迹 API 能哈希的每一个标量
（f32 和 f64 最后到来，见 [#484](https://github.com/NVlabs/cuda-oxide/pull/484)），
以及由它们构建的数组和元组
（[#792](https://github.com/NVlabs/cuda-oxide/pull/792)）。剩下的：

1. 决定是否以及如何在 cuda-oxide 的类型翻译中支持 `char`。
2. 扩展控制流和类型转换的覆盖范围。
3. 添加结构体和枚举；轨迹目前还无法窥视它们内部。
4. 为失败的种子添加收缩。

这个顺序是刻意的。一个第一天就生成一切的模糊测试器，主要生成的是噪音。一个每次
只在一个维度上成长的模糊测试器会告诉你什么坏了、以及为什么。友好得多。
