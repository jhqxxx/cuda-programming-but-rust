# tile 示例（CUDA Tile C++）

编程单位是**一个 tile**（一整块数据），不是一根线程 —— 没有 `threadIdx`，
边界走 `load_masked` / `store_masked`，块内线程数由编译器决定。

```cpp
#include "cuda_tile.h"

__tile_global__ void vecAddTile(float* __restrict__ a, float* __restrict__ b,
                                float* __restrict__ out, int n) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView   = ct::partition_view{ct::tensor_span{a,   ct::extents{n}}, ct::shape{128_ic}};
    auto bView   = ct::partition_view{ct::tensor_span{b,   ct::extents{n}}, ct::shape{128_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{128_ic}};

    int bx = ct::bid().x;                 // 只有块索引，没有 threadIdx
    outView.store_masked(aView.load_masked(bx) + bView.load_masked(bx), bx);
}

vecAddTile<<<num_blocks, 1>>>(d_a, d_b, d_out, N);   // 第二格必须是 1
```

## 示例

| 文件 | 主题 | 目标名 |
| --- | --- | --- |
| [`01-hello-world.cu`](./01-hello-world.cu) | 块索引与 grid 尺寸（`ct::bid` / `ct::num_blocks`） | `tile_01-hello-world` |

上面那段 `vecAddTile` 是 **tile 模型**的写法示意 —— 它要处理数组，所以有视图、
`load_masked`、`ct::shape{128_ic}` 那一整套。`01-hello-world.cu` 是**没有数组**的
最小起步，和 `cutile-rs-example/01-hello-world.rs`、
`python/examples/tile/01-hello-world.py` 是同一个主题。

## 编译：两条硬要求

```bash
nvcc --enable-tile -std=c++20 -arch=sm_86 -O2 01-hello-world.cu -o tile_01-hello-world
```

两条都不能少：

| 参数 | 不加会怎样 |
| --- | --- |
| `--enable-tile` | nvcc 的默认等于 `--simt-only`。只会给一句警告 `parsing for Tile constructs is not enabled. Tile annotations (including __tile__ and __tile_global__) will be ignored`，然后转成一堆莫名其妙的 `calling a __tile__ function from a __host__ function is not allowed` |
| `-std=c++20` | `crt/cuda_tile.h` 第 55 行直接 `#error "This file needs C++20 features. Please compile with c++20 or later dialect"` |

**不需要额外装库，也不需要自己编 LLVM/MLIR。** tile 支持在 nvcc 编译器内部；
运行期一个 `*tile*` 库都不需要 —— 这也是当初误判的原因之一。

nvcc 13.3 的 tile 相关开关：

| 开关 | 作用 |
| --- | --- |
| `--enable-tile` | 打开 tile（默认 `--simt-only`） |
| `--simt-only` / `--tile-only` | 只编 SIMT / 只编 tile。同一个 `.cu` 里两种可以共存 |
| `--tilebc` | 产出 Tile IR 字节码 |
| `--tilecubin` / `--tilefatbin` | 只产设备端 cubin / fatbin |
| `-Xtileiras` | 把参数透传给 Tile IR 汇编器 `tileiras` |

## 本机实测（RTX 3080 / sm_86 / CUDA 13.3）

| 用例 | 结果 |
| --- | --- |
| `vec_add`（[指南 2.4.6.1.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html) 的 partition_view 版，128 元素） | ✅ 128/128 |
| `ct::mma` 单次调用（16×16，f16 → f32） | ✅ 正确 |
| gemm（`ct::mma` + masked + 循环，32×32×16） | ✅ 正确 |
| `ct::irange` + `partition_view`（指南 2.4.7.1 的 `tile_sum`，`int` 边界） | ✅ 正确 |
| `ct::irange` 用 **`std::size_t` 边界** | ❌ **死循环挂死** |

`sm_86` 是官方支持目标：[cuTile Rust 的兼容性表](../../../docs/cutile-rs-book-zh.CN/reference/compatibility.md)
里 `baseline Tile IR` 要求 `sm_80+`。FP8 / block-scaled MMA / `num_cta_in_cga > 1`
这些要 `sm_90+` 或 `sm_100+`，在 3080 上不可用。

### ⚠️ 已知坑：`ct::irange` 用 `std::size_t` 边界会死循环

**官方指南 2.4.9.1 的 `gemm` 示例在这套工具链上会挂死**，就因为它这么写：

```cpp
std::size_t num_k = (K + tk - 1) / tk;
for (auto k : ct::irange(std::size_t{0}, num_k)) { ... }   // ← 死循环
```

其余代码逐字相同、只把边界换成 `int` 就正常通过，已隔离验证。**写 tile 循环时把
`ct::irange` 的边界转成 `int`。** 这看起来是 nvcc 13.3 tile 后端的 bug。

另一个小坑：`32_ic` 和 `16_ic` 是**不同类型**，不能共用一条 `auto` 声明 ——
`constexpr auto tm = 32_ic, tk = 16_ic;` 会报
`"auto" type is ... but was previously implied to be ...`。一行一个。

### IntelliSense 报 `namespace "cuda::tiles" has no member "assume_aligned"`

**nvcc 编得过，红波浪线却在 `assume_aligned` 上** —— 原因是 include 顺序。

`crt/cuda_tile.h` **自己不包含** `host_defines.h`，它假定调用方先包含过
`<cuda_runtime.h>`。而 `__tile__` / `__tile_global__` / `__tile_builtin__` 全都是这个形式：

```c
// crt/host_defines.h
#define __tile_builtin__  __location__(tile_builtin)   // → __attribute__((tile_builtin)) 之类
```

`__location__` 只在 `host_defines.h` 里定义。**nvcc 把 `__location__` 当编译器内建属性，
所以顺序反了也照样编过**；但 IntelliSense 之类的非 nvcc 解析器会把这行当成语法错误
（`__location__(tile_builtin) T assume_aligned(...)` 是两个类型名连着），于是**跳过这条
声明**，于是报「这个命名空间里没有这个成员」。

修法就是让 `cuda_utils.h`（它带来 `<cuda_runtime.h>`）排在 `cuda_tile.h` 前面：

```cpp
#include "cuda_utils.h"   // 必须先：带来 <cuda_runtime.h> → host_defines.h
#include "cuda_tile.h"
```

`01-hello-world.cu` 里已经这么排了。验法：用任意非 nvcc 编译器试解析即可 ——

```bash
g++ -std=c++20 -fsyntax-only \
    -Icpp/common -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl \
    -x c++ cpp/examples/tile/01-hello-world.cu
```

顺序反了会喷几千行错；顺序对了只剩 `<<<>>>` 那两行（非 CUDA 解析器不认启动语法，
而 cpptools 对 `.cu` 原生支持，所以它不会报）。

另外仓库根的 [`.vscode/c_cpp_properties.json`](../../../.vscode/c_cpp_properties.json)
把 IntelliSense 设成 C++20 并补了两个 include 路径：

| 设置 | 为什么 |
| --- | --- |
| `cppStandard: c++20` | `crt/cuda_tile.h:55` 是 `#error "This file needs C++20 features"` |
| `/usr/local/cuda/include/cccl` | `cuda_utils.h` 用了 `<cuda/cmath>`（CCCL），头文件在 `include/cccl/` 下 |

**别把 `compilerPath` 指向 nvcc，也别手动定义 `__CUDACC__`** —— 实测那样反而有 27 个错，
全是 CCCL 的 `cmath` `constexpr` 重声明冲突。cpptools 用 linux-gcc 模式（自带
`__GNUC__`）才是干净的那条路。

## 和 SIMT 的差别

| | SIMT（[`../simt/`](../simt/)） | tile（本目录） |
| --- | --- | --- |
| 你描述的是 | **一个线程**做什么 | **一个 thread block** 处理哪一块数据 |
| 索引 | 自己算 `threadIdx + blockIdx * blockDim` | `ct::bid()`，只有块索引 |
| 线程数 | 你显式指定 `blockDim` | 后端决定；chevron 第二格必须是 `1` |
| 边界 | 自己 `if (idx < n)` | `load_masked` / `store_masked` |
| 每块暂存 | 自己管 `__shared__` | 编译器管（tile 有值语义，不用自己分配） |
| 张量核 | 手写 WMMA / mma PTX | `ct::mma(a, b, acc)` |
| 启动 | `kernel<<<grid, block>>>` | `kernel<<<grid, 1>>>` |

不止是语法不同 —— **是两种不同的抽象**。所以只和同模型的实现对照。

同一个 `.cu` 文件里 `__tile_global__` 和 `__global__` kernel 可以共存，一个 host
程序可以启动两者。但 13.3 有个限制：`__tile__` 函数不能从 `__global__` / `__device__`
调用，反向也不行。

## 三条语言线的对照

| 语言 | tile 模型入口 | 状态 |
| --- | --- | --- |
| Rust | [`cutile-rs-example/`](../../../cutile-rs-example/) —— cuTile | ✅ 可跑 |
| Python | [`python/examples/tile/`](../../../python/examples/tile/) —— cuTile Python（`cuda.tile`） | ✅ 可跑 |
| **C++** | **本目录 —— CUDA Tile C++（`cuda::tiles`）** | ✅ 可跑 |

**这是本仓库唯一真正的「同模型跨语言」对照** —— 三个绑定共享同一个编译器后端
（CUDA Tile IR → `tileiras`），所以语义一致、API 一一对应。

但**命名一一对应地说，其实各叫各的** —— 比如取块索引 Rust 是
`program_id(0)`、C++ 是 `ct::bid().x`、Python 是 `ct.bid(0)`。完整的对照表和实测踩过的
坑统一放在 [`docs/cross-language/tile/`](../../../docs/cross-language/tile/)：

| 主题 | 文件 |
| --- | --- |
| 块索引、grid 尺寸、设备端打印、启动 | [`01-block-index.md`](../../../docs/cross-language/tile/01-block-index.md) |

（那张表原先是写在这里的，已经挪走 —— 免得 C++ 和 Python 两份 README 各维护一遍。
索引见 [`docs/cross-language/README.md`](../../../docs/cross-language/README.md)。）

（Triton 是**另一套模型 + 另一套编译器**，不在这个对照里。）

## 约定

- 文件名 = 两位编号 + slug，如 `01-hello-world.cu`
- **CMake 目标名带模型前缀**：`tile_01-hello-world`（不加前缀会和 `../simt/` 下的同名文件撞车）
- CMake 里 tile 目标单独抬到 C++20 并加 `-enable-tile`，见 [`../../CMakeLists.txt`](../../CMakeLists.txt)
- 每个示例自带 `main` 并**自检结果**，别只打印不验证
- 公共工具用 `#include "cuda_utils.h"`（`../../common` 已在 include 路径上）
- **`ct::irange` 的边界用 `int`**，不要用 `std::size_t`（原因见上文已知坑）
- **`ct::shape` 只能用 `_ic` 字面量**，如 `ct::shape{128_ic}`。传普通的 `constexpr int`
  会报 `cannot deduce class template arguments` —— 它要的是编译期值
  （`ct::integral_constant`）。host 侧算网格的 tile 大小只能另写一个 `constexpr int`，
  两边手动对齐
- 张量核类示例记得标注 sm_86 上哪些特性不可用

## 参考

- CUDA Programming Guide 2.4 编写 Tile kernel
  （仓库内中译：`docs/CUDA编程指南-V13.3-zh.CN/chapter-2-CUDA GPU编程/2.4-编写 Tile kernel.md`）
- [CUDA Tile C++ API Reference](https://docs.nvidia.com/cuda/cuda-tile-cpp-api-reference/api_reference.html)
- [CUDA Tile 官方入口](https://developer.nvidia.com/cuda/tile)
- [CUDA Tile IR 规范](https://docs.nvidia.com/cuda/tile-ir/latest/)（编译器层，和本目录是两回事）
