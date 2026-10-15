# kernel 家族

GPU kernel 常常针对不同的问题形状、硬件能力、资源限制或调优决策做特化。因此一个逻辑操作可能会拥有多个提前编译好的入口点。

`KernelFamily` 表示这样一组已编译的变体，并提供了一套经过校验的宿主侧选择机制。

它把两个彼此独立的决策分开：

```text
Eligibility：这个变体能安全地执行这个问题吗？
Preference： 应该由哪个合格变体来执行？
```

这个区分很重要。性能策略绝不能选中一个违反了自身启动、形状、硬件或资源要求的变体。

`KernelFamily` 完全在 CPU 上完成选择。它不编译 kernel、不查询 CUDA 设备、不对入口做基准测试、不创建线程，也不访问文件系统。应用需要提供所有相关的问题事实与硬件事实。

## 何时使用 kernel 家族

当一个操作拥有一小组固定的、已经编译好的实现时，使用 `KernelFamily`，例如：

* 不同的输出 tile 大小
* 不同的 block 维度
* 针对特定计算能力特化的 tensor-core kernel
* 小输入与大输入的实现
* 共享内存需求不同的 kernel
* 为不同问题形状区间调优的 kernel
* 一个通用回退实现，外加一个或多个优化变体

kernel 家族不是运行时注册表，也不是 JIT 编译器。它的变体数量是一个 const 泛型，因此是 Rust 类型的一部分：

```rust
KernelFamily<Id, Entry, Meta, const N: usize>
```

这个家族恰好拥有 `N` 个变体。

当可用的 kernel 在构建时确定、并且应当保持显式且可审计时，这种有界的表示很有用。

## 选择模型

选择流程有两种模式：

```text
Force(id) -> 解析变体 -> 校验 -------------------------------> Override

Auto      -> 查缓存 -> 解析 -> 校验 -------------------------> Cache
          -> 过滤合格变体
          -> selector 选出一个 ID
          -> 校验 selector 的输出
          -> 缓存选中的 ID --------------------------------> Selector
```

`SelectionMode::Force(id)` 是一种经过校验的手动覆盖。它绕过缓存与 selector，但它不会绕过合格性校验。

`SelectionMode::Auto` 首先尝试使用缓存中那个稳定的变体 ID。未知的或刚刚变得不合格的缓存条目都会被当作未命中。随后 selector 只会看到合格的变体。

每次成功的选择都会报告它的来源：

```rust
pub enum SelectionSource {
    Override,
    Cache,
    Selector,
}
```

这让应用可以在日志与诊断中区分运算符覆盖、缓存下来的决策，以及 selector 新做出的决策。

## 家族的结构

一个家族包含一个稳定的家族标识和一个变体数组：

```rust
pub struct KernelFamily<Id, Entry, Meta, const N: usize> {
    // 私有字段
}
```

每个变体包含三个由调用者定义的值：

```rust
pub struct KernelVariant<Id, Entry, Meta> {
    // 私有字段
}
```

| 组成部分 | 用途                                                |
| :-------- | :----------------------------------------------------- |
| `Id`      | 该变体稳定的语义标识               |
| `Entry`   | 与该变体关联的可调用入口或启动器 |
| `Meta`    | 合格性与偏好元数据                    |
| `N`       | 编译期的变体数量                        |

`Entry` 有意保持泛型。它可以是一个函数指针、一个带类型的启动包装器、一个描述所生成方法的枚举，或者其它由应用定义的值。

`Meta` 同样由应用定义。它通常保存要求与调优事实，例如：

* 最低计算能力
* tile 维度
* 对齐要求
* 共享内存用量
* 支持的数据类型
* 偏好的问题规模区间
* 启动几何配置
* 调优评分或优先级

## 定义稳定的变体 ID

每个变体都需要一个显式的 ID：

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    Scalar,
    Chunked,
}
```

ID 会被以下各方使用：

* 手动覆盖
* 缓存
* selector
* 诊断
* 配置文件
* 环境变量

一个 ID 必须表示某个已编译变体的语义标识。

好的 ID 包括枚举值，或含义由应用掌控的稳定字符串：

```rust
enum GemmVariantId {
    M256xN256,
    M512xN256,
}
```

不要使用那些构建之间或进程之间可能改变其标识的取值：

* 数组下标
* 指针
* `TypeId`
* 格式化后的调试输出
* 所生成函数的地址
* 算法或输入不稳定的哈希

数组位置不能替代稳定的 ID。声明顺序可以影响选择策略，但它不能定义缓存标识。

## 构建一个家族

用 `KernelVariant::new` 创建变体：

```rust
let variant = KernelVariant::new(
    VariantId::Chunked,
    chunked_entry,
    VariantMetadata {
        min_len: 1024,
        chunk_width: 256,
    },
);
```

用 `KernelFamily::try_new` 创建家族：

```rust
let family = KernelFamily::try_new(
    "vector-copy",
    1,
    [
        scalar_variant,
        chunked_variant,
    ],
)?;
```

各参数为：

```rust
KernelFamily::try_new(
    name: &'static str,
    revision: u32,
    variants: [KernelVariant<Id, Entry, Meta>; N],
)
```

构建过程会校验三条结构性不变式：

1. 家族至少包含一个变体。
2. 家族名字不是空的，也不是只含空白字符。
3. 变体 ID 在家族内唯一。

构建过程可能返回：

```rust
pub enum KernelFamilyBuildError {
    EmptyFamily,
    EmptyFamilyName,
    DuplicateVariantId {
        first_index: usize,
        duplicate_index: usize,
    },
}
```

## 家族标识与修订

家族名字与修订共同构成它的选择与缓存命名空间：

```rust
let id = family.id();

println!("{}", id.name());
println!("{}", id.revision());
println!("{id}"); // vector-copy@1
```

持久化缓存或内存缓存都必须把这两个值纳入它的键。

只要某个已有的缓存决策可能不再表示同样的含义，就要递增修订号。

在修改以下内容时，必须更新修订号：

* 变体成员
* 某个变体 ID 的含义
* 合格性规则
* selector 或偏好策略
* 调优方法
* 选择所使用的元数据
* 声明顺序，除非选择与调优被显式地定义为顺序无关

例如：

```rust
KernelFamily::try_new("vector-copy", 1, variants_v1)?;
KernelFamily::try_new("vector-copy", 2, variants_v2)?;
```

改变修订号可以避免在第一份契约下做出的决策被按照第二份契约来解释。

家族名字应当描述逻辑操作与特化轴：

```text
image-filter/radius
gemm/output-tile
reduction/block-size
attention/head-dimension
```

避免使用基于临时实现位置或进程特有状态的名字。

## 用 `KernelProblem` 表达合格性

合格性回答的是某个具体变体能否安全地处理某个问题。

实现：

```rust
pub trait KernelProblem<Variant> {
    type Rejection: std::error::Error + Send + Sync + 'static;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection>;
}
```

问题类型应当包含校验这次启动所需的全部事实。

例如：

```rust
#[derive(Clone, Copy, Debug)]
struct CopyProblem {
    len: usize,
}
```

校验可以考虑：

* 问题的维度
* 整除性要求
* 缓冲区对齐
* 数据类型
* 设备的计算能力
* 共享内存容量
* grid 与 block 限制
* cooperative 启动支持
* cluster 启动支持
* 整数转换与分配溢出
* 应用特定的语义约束

实现应当是确定性的，并且没有副作用。

合格性是一条安全与正确性的边界。它不应只编码性能偏好。

例如，一个要求长度能被 256 整除的变体，应当拒绝不兼容的问题，即使另一层预计会避免选中它。

## 用 `KernelSelector` 表达偏好

selector 在已经通过合格性校验的变体之间做选择：

```rust
pub trait KernelSelector<Problem, Variant, Id> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn select(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        eligible: &[&Variant],
    ) -> Result<Id, Self::Error>;
}
```

`eligible` 切片有两条保证：

1. 它不为空。
2. 它保持家族的声明顺序。

selector 返回的是一个稳定的变体 ID，而不是数组下标或入口引用。

selector 可以实现诸如以下的策略：

* 选择问题所能支持的最大 tile
* 在低于某个阈值时选择小输入 kernel
* 依据离线基准测试结果来选择
* 依据硬件代次来选择
* 选择第一个合格的变体
* 从应用配置中选择
* 调用外部的调优服务

selector 必须从所提供的合格切片中返回一个 ID。`KernelFamily` 会检查这一要求，并拒绝无效的 selector 输出。

当闭包的签名匹配时，闭包也可以充当 selector。请为每个闭包参数标注类型。没有这些标注时，闭包的生命周期推导可能把这些借用固定到某一次具体调用上，rustc 会以“implementation of `FnMut` is not general enough”拒绝该闭包。一个不会失败的 selector 可以用 `std::convert::Infallible` 作为它的错误类型：

```rust
let mut selector =
    |_: KernelFamilyId, _: &CopyProblem, eligible: &[&Variant]| {
        Ok::<_, Infallible>(*eligible[0].id())
    };
```

## 自动选择

自动选择使用：

```rust
SelectionMode::Auto
```

完整的调用是：

```rust
let selected = family.select(
    &problem,
    SelectionMode::Auto,
    &mut selector,
    &mut cache,
)?;
```

自动路径会执行以下操作：

1. 向缓存询问一个稳定的变体 ID。
2. 在家族内解析这个缓存下来的 ID。
3. 用当前的问题重新校验这个缓存下来的变体。
4. 如果是一次有效的缓存命中，立即返回。
5. 否则，校验每一个家族成员。
6. 按声明顺序保留合格的变体。
7. 如果没有变体合格则失败。
8. 让 selector 从这个合格切片中选择。
9. 验证 selector 返回的是这个家族的一个成员。
10. 验证返回的那个成员确实在合格切片中。
11. 把校验过的选中 ID 存进缓存。
12. 返回选中的变体，来源为 `SelectionSource::Selector`。

缓存下来的 ID 是提示，不是权威。

一个缓存条目可能因为以下原因而失效：

* 这条缓存是由更旧版本的应用写入的
* 合格性依赖于硬件
* 问题约束发生了变化
* 持久化缓存被损坏
* 某个变体被移除了
* 之前的某个策略错误地使用了另一个修订

出于这些原因，每一条缓存下来的 ID 在使用前都会被解析并校验。

## 强制选择

当调用者显式要求某个变体时，使用：

```rust
SelectionMode::Force(id)
```

典型的来源包括：

* 命令行选项
* 环境变量
* 测试配置
* 调试开关
* 性能实验
* 运算符策略

示例：

```rust
let selected = family.select(
    &problem,
    SelectionMode::Force(VariantId::Scalar),
    &mut selector,
    &mut cache,
)?;
```

强制选择会：

* 绕过缓存查找
* 绕过 selector 调用
* 不更新缓存
* 验证所请求的 ID 属于这个家族
* 校验所请求的变体是合格的

一个未知的强制 ID 会返回 `UnknownForcedVariant`。

一个已知但不兼容的变体返回 `IneligibleForcedVariant`。

因此，手动覆盖无法静默地绕过启动要求。

## 选择结果

一次成功的调用返回一个借用的 `SelectedVariant`：

```rust
pub struct SelectedVariant<'family, Id, Entry, Meta> {
    // 私有字段
}
```

用以下方式访问变体与来源：

```rust
let variant = selected.variant();
let source = selected.source();

println!("selected {:?} from {}", variant.id(), source);
```

选中的值借用自这个家族。它不会复制或移动变体条目与元数据。

变体提供：

```rust
variant.id();
variant.entry();
variant.metadata();
```

`SelectionSource` 实现了 `Display`：

```text
override
cache
selector
```

## 完整示例

下面的示例使用普通的 CPU 函数作为入口。当入口是包裹已编译 CUDA kernel 的带类型包装器时，选择结构完全相同。

```rust
use cuda_host::{
    KernelFamily, KernelFamilyBuildError, KernelFamilyId, KernelProblem, KernelSelector,
    KernelVariant, NoKernelSelectionCache, SelectionMode, SelectionSource,
};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    Scalar,
    Chunked,
}

type Entry = fn(&[f32], &mut [f32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VariantMetadata {
    min_len: usize,
    chunk_width: usize,
}

type Variant = KernelVariant<VariantId, Entry, VariantMetadata>;
type CopyFamily = KernelFamily<VariantId, Entry, VariantMetadata, 2>;

fn scalar_copy(input: &[f32], output: &mut [f32]) {
    output.copy_from_slice(input);
}

fn chunked_copy(input: &[f32], output: &mut [f32]) {
    const WIDTH: usize = 256;

    for (input_chunk, output_chunk) in input
        .chunks_exact(WIDTH)
        .zip(output.chunks_exact_mut(WIDTH))
    {
        output_chunk.copy_from_slice(input_chunk);
    }
}

fn copy_family() -> Result<CopyFamily, KernelFamilyBuildError> {
    KernelFamily::try_new(
        "book/vector-copy",
        1,
        [
            KernelVariant::new(
                VariantId::Scalar,
                scalar_copy as Entry,
                VariantMetadata {
                    min_len: 0,
                    chunk_width: 1,
                },
            ),
            KernelVariant::new(
                VariantId::Chunked,
                chunked_copy as Entry,
                VariantMetadata {
                    min_len: 1024,
                    chunk_width: 256,
                },
            ),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
struct CopyProblem {
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyVariantRejection {
    variant: VariantId,
    required_min_len: usize,
    required_multiple: usize,
    actual_len: usize,
}

impl fmt::Display for CopyVariantRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} requires len >= {} and len divisible by {}, but len is {}",
            self.variant,
            self.required_min_len,
            self.required_multiple,
            self.actual_len,
        )
    }
}

impl Error for CopyVariantRejection {}

impl KernelProblem<Variant> for CopyProblem {
    type Rejection = CopyVariantRejection;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection> {
        let metadata = variant.metadata();

        if self.len < metadata.min_len
            || !self.len.is_multiple_of(metadata.chunk_width)
        {
            return Err(CopyVariantRejection {
                variant: *variant.id(),
                required_min_len: metadata.min_len,
                required_multiple: metadata.chunk_width,
                actual_len: self.len,
            });
        }

        Ok(())
    }
}

struct CopySelector;

impl KernelSelector<CopyProblem, Variant, VariantId> for CopySelector {
    type Error = Infallible;

    fn select(
        &mut self,
        _family: KernelFamilyId,
        _problem: &CopyProblem,
        eligible: &[&Variant],
    ) -> Result<VariantId, Self::Error> {
        let preferred = eligible
            .iter()
            .copied()
            .max_by_key(|variant| variant.metadata().chunk_width)
            .expect("eligible is guaranteed to be non-empty");

        Ok(*preferred.id())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let family = copy_family()?;

    let input = vec![1.0f32; 4096];
    let mut output = vec![0.0f32; input.len()];
    let problem = CopyProblem { len: input.len() };

    let mut selector = CopySelector;
    let mut cache = NoKernelSelectionCache;

    let selected = family.select(
        &problem,
        SelectionMode::Auto,
        &mut selector,
        &mut cache,
    )?;

    println!(
        "selected {:?} from {}",
        selected.variant().id(),
        selected.source(),
    );

    assert_eq!(selected.variant().id(), &VariantId::Chunked);
    assert_eq!(selected.source(), SelectionSource::Selector);

    let entry = *selected.variant().entry();
    entry(&input, &mut output);

    assert_eq!(input, output);

    let forced = family.select(
        &problem,
        SelectionMode::Force(VariantId::Scalar),
        &mut selector,
        &mut cache,
    )?;

    assert_eq!(forced.variant().id(), &VariantId::Scalar);
    assert_eq!(forced.source(), SelectionSource::Override);

    Ok(())
}
```

对于长度 4096 的输入，两个变体都合格。selector 偏好 `chunk_width` 更大的那个变体。

对于长度 1000 的输入，只有 `Scalar` 合格，因为 `Chunked` 至少需要 1024 个元素，并要求长度能被 256 整除。

这套选择机制并不知道 `chunk_width` 是什么意思。合格性与偏好的语义完全属于应用。

## 加入缓存

当每次都必须重新执行选择时，使用 `NoKernelSelectionCache`：

```rust
let mut cache = NoKernelSelectionCache;
```

对于反复出现的问题类别，实现 `KernelSelectionCache`：

```rust
pub trait KernelSelectionCache<Problem, Id> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
    ) -> Result<Option<Id>, Self::Error>;

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        variant: &Id,
    ) -> Result<(), Self::Error>;
}
```

针对前面示例的一个简单内存缓存可以把问题长度作为它键的一部分：

```rust
use cuda_host::{KernelFamilyId, KernelSelectionCache};
use std::collections::HashMap;
use std::convert::Infallible;

#[derive(Default)]
struct MemoryCache {
    values: HashMap<(KernelFamilyId, usize), VariantId>,
}

impl KernelSelectionCache<CopyProblem, VariantId> for MemoryCache {
    type Error = Infallible;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &CopyProblem,
    ) -> Result<Option<VariantId>, Self::Error> {
        Ok(self.values.get(&(family, problem.len)).copied())
    }

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &CopyProblem,
        variant: &VariantId,
    ) -> Result<(), Self::Error> {
        self.values.insert((family, problem.len), *variant);
        Ok(())
    }
}
```

由应用决定哪些问题属性定义了缓存的等价性。

例如，一个 GEMM 的缓存键可能包含：

```text
家族名字
家族修订
计算能力
M
N
K
输入类型
输出类型
布局
对齐类别
```

当某个属性可能改变合格性或偏好时，不要漏掉它。

一份缓存实现可以是：

* 一张内存中的映射表
* 一个进程级缓存
* 一个数据库
* 一个配置文件
* 一个基准测试结果仓库
* 一个远程调优服务

缓存错误是显式的。如果 `lookup` 或 `store` 返回错误，选择会停止，并返回 `CacheLookupFailed` 或 `CacheStoreFailed`。

一个尽力而为（best-effort）的缓存可以在内部把存储失败转换为未命中，或转换为空操作式的写入，只要那种策略是合适的。

## 错误处理

`KernelFamily::select` 可以报告来自四个层的错误：

1. 强制选择
2. 合格性校验
3. selector 的执行或输出校验
4. 缓存操作

主要的变体有：

| 错误                               | 含义                                                     |
| :---------------------------------- | :---------------------------------------------------------- |
| `UnknownForcedVariant`              | 被强制指定的 ID 不是这个家族的成员                          |
| `IneligibleForcedVariant`           | 被强制指定的家族成员无法处理该问题            |
| `NoEligibleVariants`                | 所有家族成员都未通过校验                       |
| `SelectorFailed`                    | selector 返回了它自己的策略错误                  |
| `SelectorReturnedUnknownVariant`    | selector 返回的 ID 不在这个家族内              |
| `SelectorReturnedIneligibleVariant` | selector 返回了一个不合格的家族成员 |
| `CacheLookupFailed`                 | 缓存查找返回了错误                              |
| `CacheStoreFailed`                  | 缓存写入返回了错误                             |

未知的或不合格的缓存 ID 不算选择错误。它们会被当作缓存未命中，因为缓存内容是不可信的提示。

selector 的失误则算错误，因为 selector 在当次选择调用中拿到的是权威的合格切片。

## 声明顺序

家族的变体保留声明顺序：

```rust
KernelFamily::try_new(
    "operation",
    1,
    [
        fallback,
        medium,
        aggressive,
    ],
)?;
```

selector 接收到的合格变体也保持同样的相对顺序。

这支持确定性的策略，例如：

```rust
Ok(*eligible[0].id())
```

或者通过声明顺序编码优先级。

如果声明顺序会影响选择或调优，那么改变顺序就需要递增家族修订号。

当策略比简单的有序回退更复杂时，优先使用显式的元数据。元数据让策略更容易检查、测试与演进。

## 测试一个家族

kernel 家族的策略是宿主侧逻辑，无需 CUDA 设备即可测试。

测试至少应当验证：

* 空家族被拒绝
* 空白名字被拒绝
* 重复 ID 被拒绝
* 强制选择绕过 selector 与缓存
* 强制选择仍然会校验合格性
* 有效的缓存命中会绕过 selector
* 未知的缓存 ID 会回退到 selector
* 刚刚变得不合格的缓存 ID 会回退到 selector
* 陈旧的缓存条目会被替换
* selector 只会收到合格的变体
* 声明顺序被保留
* 未知的 selector 输出被拒绝
* 不合格的 selector 输出被拒绝
* 没有合格变体时会产生错误
* 缓存查找失败会向上传播
* 缓存写入失败会向上传播
* 家族修订能隔离缓存下来的决策

仓库里的集成测试提供了可执行的示例：

```text
cuda-oxide/crates/cuda-host/tests/kernel_family.rs
```

用以下命令运行它们：

```text
cargo test -p cuda-host --test kernel_family
```

## 使用生成的 CUDA 启动器

在真实应用中，`Entry` 通常表示一个具体的带类型启动器。

例如：

```rust
type KernelLauncher = unsafe fn(
    &kernels::LoadedModule,
    &CudaStream,
    LaunchConfig,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
) -> Result<(), cuda_core::DriverError>;
```

于是这个家族为每一种已编译的特化各保存一个启动器：

```rust
type Family =
    KernelFamily<VariantId, KernelLauncher, VariantMetadata, 2>;
```

选择返回启动器以及它的元数据：

```rust
let selected = family.select(
    &problem,
    mode,
    &mut selector,
    &mut cache,
)?;

let launch = *selected.variant().entry();
let metadata = selected.variant().metadata();

println!(
    "launching {:?} from {}",
    selected.variant().id(),
    selected.source(),
);

unsafe {
    launch(
        &module,
        &stream,
        launch_config,
        &input,
        &mut output,
    )?;
}
```

`KernelFamily` 校验的是变体的选择。它不会让原本不安全的裸启动变得安全。

启动几何配置、指针有效性、缓冲区大小、上下文标识，以及 kernel 的 ABI 义务仍然属于启动 API。当 kernel 的几何配置可以静态表达时，优先使用 prepared 启动契约。参见[启动 kernel](launching-kernels.md)。

## 真实世界的示例

`gemm_sol_final` 示例为两个 Blackwell GEMM 输出 tile 特化使用了一个含两个条目的家族：

```text
M256xN256
M512xN256
```

它的家族元数据包含：

* kernel 名字
* 输出 tile
* M tile 大小
* N tile 大小
* K tile 的倍数
* 问题规模的偏好阈值

合格性检查会校验矩阵维度、tile 整除性、整数范围、分配大小的溢出以及 CUDA grid 限制。

selector 会选择阈值与问题兼容的最大偏好 tile。

`GEMM_SOL_VARIANT` 环境变量暴露了：

```text
auto
m256xn256
m512xn256
```

被强制指定的取值仍然只是经过校验的覆盖。它们无法选中一个编译形状与矩阵维度不兼容的 tile。

参见：

* [`gemm_sol_final`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol_final)
* [`KernelFamily` 集成测试](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/cuda-host/tests/kernel_family.rs)
* [`cuda-host` 的 kernel-family 实现](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/cuda-host/src/kernel_family.rs)
* [`cuda-host` README](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/cuda-host/README.md)

## 设计准则

在引入一个新的家族时，请遵循以下规则。

### 保持家族有界

优先使用一小组有意义的已编译变体。

一个庞大的、动态变化的集合通常意味着：注册表、生成的派发表或调优数据库是更好的抽象。

### 让 ID 具备语义

ID 应当描述被选中了哪份编译契约，而不是它碰巧出现在数组中的哪个位置。

### 把正确性放进合格性

任何为安全或正确执行所必需的条件都属于 `KernelProblem::validate`。

不要依赖 selector 来避开不安全的变体。

### 把性能放进偏好

selector 应当只在已经有效的选项中做选择。

这样可以让调优策略与安全校验保持独立。

### 把缓存条目当作不可信

始终让 `KernelFamily` 去解析并重新校验缓存下来的 ID。

不要在没有经过家族校验的情况下直接用缓存 ID 去启动。

### 为策略变更做版本管理

当某个缓存下来的选择在新代码下可能具有不同含义时，递增家族修订号。

### 保持可观测性

在诊断中记录 `SelectionSource`、家族标识、选中的 ID 以及相关的问题事实。

在比较运算符覆盖、缓存下来的调优结果与当前 selector 行为时，这一点尤其有用。

## 小结

`KernelFamily` 为提前编译的 kernel 特化提供了一个有界且经过校验的派发层。

它的核心性质是：

* 固定的家族成员
* 稳定的语义化变体 ID
* 由调用者定义的入口与元数据
* 显式的合格性校验
* 彼此独立的偏好选择
* 经过校验的手动覆盖
* 会被重新校验的缓存条目
* 显式的家族修订
* 可观测的选择来源
* 纯 CPU、与设备无关的策略逻辑

这个抽象不决定 kernel 应当如何调优。它提供的是那些不变式，让调优策略变得显式、可测试，并且可以安全地与已编译的 CUDA 启动器集成。
