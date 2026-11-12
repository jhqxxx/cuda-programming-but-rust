# 不安全的 GPU 通信

CUDA 13.5 的通信面提供了对等内存访问与多播操作。kernel 用 `cutile::core::*` 里的访存操作与排序类型，用 `cutile::tileir::*` 里的新指针分类与多播操作。分配、映射、访问权限以及生命周期都由宿主负责。

这些是底层的 unsafe API。它们提供构建通信协议所需的同步原语，但**不会**自动插入 GPU 之间的屏障。

## 单播：访问某一张 GPU 的内存

`unsafe ptr_with_attr_unicast(pointer)` 把指针分类为 NVLink 对等指针。它保持地址不变，既不分配内存，也不开启对等访问或授予权限。reshape、broadcast 与 offset 操作会保留这一分类。

把这个已分类的指针与已有的 unsafe 访存操作配合使用：

| 操作 | 用途 | 可选排序 |
|---|---|---|
| `load_ptr_tko` | 读取对端元素 | Weak、Relaxed、Acquire |
| `store_ptr_tko` | 写入对端元素 | Weak、Relaxed、Release |
| `atomic_rmw_tko` | 原子更新，并返回旧值 | Relaxed、Acquire、Release、AcqRel |
| `atomic_cas_tko` | 原子比较并交换 | Relaxed、Acquire、Release、AcqRel |

建立在单播指针之上的视图会保留它的分类，并支持普通视图的访存操作，但要受各自的类型与形状限制。并发单播示例验证了指针的 load、store 与整数原子加；它并不声称覆盖了所有原子模式或类型。

要协调不同 GPU 之间的操作，请选 `scope::System`。设备作用域只覆盖发起操作的那张设备，不足以完成 GPU 之间的 release/acquire 握手。原子操作要求对端原生支持原子。指针的目标必须在整个启动期间都对发起方 GPU 可访问。

## 多播：访问多张 GPU 上对应的内存

`unsafe ptr_with_attr_multicast(pointer)` 要求背后是真实的 CUDA 多播映射。普通的分配不能仅靠加上这个分类就变成多播分配。普通的 load、store 和原子操作会拒绝多播指针；请改为构造分区视图并使用下面这些操作：

| 操作 | 效果 | 可选排序 |
|---|---|---|
| `multimem_load_reduce_view_tko` | 从所有参与者读取对应元素并返回它们的归约结果 | Weak、Relaxed、Acquire |
| `multimem_reduce_view_tko` | 把给定的 tile 原子地合并进每个参与者的底层内存 | Relaxed、Release |
| `multimem_store_view_tko` | 把给定的 tile 写入每个参与者的底层内存 | Weak、Relaxed、Release |

Weak 访问假定被寻址的位置上没有并发访问，并且不带编码的内存作用域。Relaxed/acquire/release 访问则要求指定作用域。这里描述的 GPU 间场景请用系统作用域。原子归约是**逐元素**原子的；一个完整 tile 或所有参与者并不构成一次原子事务。

| 归约 | load-reduce 类型 | atomic-reduce 类型 |
|---|---|---|
| `multimem::Add` | i32、i64、u64、f16、bf16、f32、f64 | 同上 |
| `multimem::Min`、`Max` | i32、i64、f16、bf16 | i32、i64 |
| `multimem::Umin`、`Umax` | i32、i64、u64，按无符号解释 | 同上 |
| `multimem::And`、`Or`、`Xor` | i32、i64、u64 | 同上 |

Rust 的 `u64` 使用与 Rust `i64` 相同的 Tile IR `i64` 表示；64 位加法对两者产生相同的结果位，并按 2^64 取模回绕。整数的 Min/Max 一律使用有符号比较，与 Rust 类型无关；无符号比较请用 Umin/Umax。多播 store 支持 i32、i64、u64、f16、bf16、f32 和 f64。浮点 tile 的字节数必须是 4 的倍数；f16 与 bf16 视图还额外要求恰好有一个静态的单位步长维度。

每一个为 true 的 `inbounds` 条目都是调用方给出的承诺：对应的 tile 坐标对每个程序实例都有效。

## 同步契约

原语本身已经具备：

| 原语 | 它能确立什么 |
|---|---|
| 使用系统作用域的 release store 或原子操作，与一次 acquire load 或原子操作配对 | 把之前的访问发布给观察到该信号的对端 |
| `Token`、`Some(token)`、`join_tokens` | 同一个 Tile 程序里各次访问之间的依赖关系 |
| `unsafe memory_fence_alias_tko(token)` | 通过不同虚拟别名访问同一物理存储时的别名代理排序 |

单靠一个 token 并不会通知另一张 GPU。单靠一次别名 fence 也不会等待另一张 GPU。用于轮询的 load 必须具备正确的 acquire 排序，且它返回的 token 必须喂给后续的依赖读取。同时请把生产者的数据操作链到它的 release 信号上；仅靠源码顺序是不够的。

举例来说，每张 GPU 都可以写入自己的贡献，把一个序号 release-store 到对端的收件箱，然后 acquire-load 自己的收件箱直到期望的序号到达，再读取各方的贡献。读取之后再交换一次，可以避免任一生产者过早复用缓冲区。单调递增的序号则用于区分同一收件箱的重复使用。

示例在普通映射与多播映射之间的切换处显式插入了别名 fence。两个 kernel 都在宿主等待其中任何一个之前就已启动。如果改成串行地连续做两次阻塞式启动，第一个 kernel 会等待一个尚未启动的对端，从而死锁。

## 宿主侧映射

示例使用已有的 `cuda_core::vmm` 包装：

| 包装 / 函数 | 职责 |
|---|---|
| `PhysicalAllocation::new` | 在选定的 GPU 上分配底层内存 |
| `VirtualReservation::new`、`Mapping::new` | 预留并映射一个普通的虚拟地址 |
| `set_access` | 授予发起访问的设备读/写权限 |
| `MulticastObject::new`、`add_device`、`bind_mem` | 创建多播组，并绑定每个参与者的底层存储 |
| `Mapping::new_multicast` | 把多播对象映射到一段地址范围 |

要查询并遵守分配/多播的对齐粒度。绑定底层内存之前，先把每个多播参与者都加进去。映射、绑定与缓冲区要保留到所有已提交的 kernel 完成为止。释放关联的虚拟地址预留之前，先解除映射。示例给两张设备都授予了对每个普通映射的访问权限；多播模式还会额外映射多播对象。单播模式既不需要、也不会创建多播对象。

## 支持情况与可运行示例

当前的 Rust 分支在 `sm_100`、`sm_103` 和 `sm_107` 上、配合 Tile IR 13.5，同时启用了这两类分类。单播要求有 NVLink 通路；多播还额外要求支持 NVLink SHARP 的 NVSwitch。所选的 `tileiras` 必须使用匹配的 13.5 后端。见 [CUDA 与 Tile IR 兼容性](compatibility.md)。

在 `cutile-rs/` 下、激活该工具链之后：

```bash
cargo run -p cutile --example unsafe_comms -- unicast
cargo run -p cutile --example unsafe_comms -- multicast
```

源码在 `cutile/examples/unsafe_comms.rs`。可选的第二个参数用来设置轮数，例如 `-- unicast 1024`。这是一个正确性示例，每张 GPU 只用一个 CTA，不是吞吐基准，也不是可复用的集合通信抽象。

每种模式默认跑 256 轮，并把启动顺序反过来再重复一遍：

- **单播：** 两张 GPU 都读取对端的输入、写入对端的输出，并原子地加到本地与对端内存中。宿主会在两张设备上检查每一个阶段。
- **多播：** 两张 GPU 都对各自的贡献做归约加载，并原子地归约进两块底层分配；负责广播的生产者每轮交替。宿主会检查每一个阶段以及两块最终的底层分配。

宿主会预编译两次启动、初始化收件箱，并把输出的快照置为毒值。从第一轮到最后一轮之间，所有协调都发生在 kernel 里。轮询循环是有界的，如果对端没能给出信号就会断言失败。不受支持的硬件与执行失败都算错误，而不是跳过。

硬件测试直接调用示例里的实现：

```bash
cargo test -p cutile --test tile_ir_13_5_comms \
  concurrent_signaling_and_buffer_reuse -- --ignored --test-threads=1 --nocapture
```

CPU 侧的验证照常运行；汇编器验证是另一回事，且不需要 GPU：

```bash
cargo test -p cutile --test tile_ir_13_5_comms
CUTILE_BYTECODE_VERSION=13.5 cargo test -p cutile --test tile_ir_13_5_comms \
  assemble_concurrent -- --ignored
```
