# 块索引与 grid 尺寸

> 对应示例：[Rust `01-hello-world.rs`](../../../cutile-rs-example/01-hello-world.rs) ·
> [C++ `01-hello-world.cu`](../../../cpp/examples/tile/01-hello-world.cu) ·
> [Python `01-hello-world.py`](../../../python/examples/tile/01-hello-world.py)

**这是 tile 模型里最容易上手的第一个主题**，也是最容易踩「命名不一致」坑的一个 ——
因为它只用块索引，不碰数组，所以三种语言的差异全都裸露在外面。

## 一句话对照

| | Rust | C++ | Python |
| --- | --- | --- | --- |
| 取块索引 | `program_id(0)` | `ct::bid().x` | `ct.bid(0)` |
| grid 尺寸 | `num_programs(0)` | `ct::num_blocks().x` | `ct.num_blocks(0)` |
| 返回值形态 | 3 个标量，一个轴一个 | `uint3` / `dim3`，一次拿三个 | 一个轴一个，返回标量 |

三行里有**两个**是反直觉的：

- **Python 没有 `program_id` / `num_programs`** —— 那是 Rust 的拼法，Python 叫
  `bid` / `num_blocks`。
- **Rust 和 Python 是按轴逐个取**（`(0)`、`(1)`、`(2)`），**C++ 是一次取三个**（`.x .y .z`）。

## 三个语言的完整 kernel

### Rust

```rust
#[cutile::module]
mod hello_world_module {
    use cutile::core::*;

    #[cutile::entry]
    fn hello_world_kernel() {
        let pid0 = program_id(0);   // 本 program 在轴 0 上的索引
        let pid1 = program_id(1);
        let pid2 = program_id(2);
        let n0 = num_programs(0);   // grid 轴 0 上 program 的总数
        let n1 = num_programs(1);
        let n2 = num_programs(2);

        let total = n0 * n1 * n2;
        cuda_tile_print!(
            "Hello from program <{}, {}, {}> in a grid of <{}, {}, {}> programs! {} total programs\n",
            pid0, pid1, pid2, n0, n1, n2, total
        );
    }
}
```

### C++

```cpp
__tile_global__ void helloWorldKernel()
{
    unsigned bid_x = ct::bid().x;          // uint3：三个轴一次拿全
    unsigned bid_y = ct::bid().y;
    unsigned bid_z = ct::bid().z;

    unsigned num_x = ct::num_blocks().x;   // dim3
    unsigned num_y = ct::num_blocks().y;
    unsigned num_z = ct::num_blocks().z;

    printf("Hello from block <%u, %u, %u> in a grid of <%u, %u, %u> blocks! %u total blocks\n",
           bid_x, bid_y, bid_z, num_x, num_y, num_z, num_x * num_y * num_z);
}
```

### Python

```python
@ct.kernel
def hello_world_kernel():
    pid0 = ct.bid(0)          # 按轴逐个取
    pid1 = ct.bid(1)
    pid2 = ct.bid(2)
    n0 = ct.num_blocks(0)
    n1 = ct.num_blocks(1)
    n2 = ct.num_blocks(2)

    total = n0 * n1 * n2
    ct.print(f"Hello from block <{pid0}, {pid1}, {pid2}> "
             f"in a grid of <{n0}, {n1}, {n2}> blocks! {total} total blocks")
```

## 启动方式对照

| | Rust | C++ | Python |
| --- | --- | --- | --- |
| 写法 | `kernel().grid((2,2,1)).sync_on(&stream)?` | `kernel<<<dim3(2,2,1), 1>>>()` | `ct.launch(stream, (2,2,1), kernel, ())` |
| grid 参数 | tuple | `dim3` | tuple |
| 流从哪来 | `Device::new(0)?` + `device.new_stream()?` | **不用给**，传统默认流 | 必须显式传，且 **context 得先存在** |
| 同步 | `.sync_on()` 自带 | `cudaDeviceSynchronize()` | `stream.synchronize()` |
| 坑 | — | **第二个 chevron 参数必须是 `1`** | `launch` 要 **4 个位置参数**；context 不自动创建 |

## 实测输出（三个语言一致）

```
grid = (2, 2, 1) → 4 个 block，四个组合各打印一行：

Rust   : Hello from program <0, 0, 0> in a grid of <2, 2, 1> programs! 4 total programs
         ...（共 4 行）
Python : Hello from block   <0, 0, 0> in a grid of <2, 2, 1> blocks!   4 total blocks
         ...（共 4 行）
C++    : Hello from block   <0, 0, 0> in a grid of <2, 2, 1> blocks!   4 total blocks
         ...（共 4 行）
```

## ⚠️ 踩过的坑

### 1. `program_id` 在 Python 里不存在

这是最先撞上的。Rust 的 `program_id` / `num_programs` 在 `cuda.tile` 里**根本没有**：

```python
import cuda.tile as ct
hasattr(ct, 'program_id')    # False
hasattr(ct, 'num_programs')  # False
hasattr(ct, 'bid')           # True
hasattr(ct, 'num_blocks')    # True
```

同一套 TileIR 后端，但两个前端**各叫各的** —— 不要以为 ABI 一样命名就一样。

### 2. 设备端打印：三种语言三种写法

| | 写法 | 说明 |
| --- | --- | --- |
| Rust | `cuda_tile_print!("…{}…", …)` | `{}` 占位，**不是 `%d`** |
| C++ | `printf("…%u…", …)` | `crt/cuda_tile.h` 里**没有**任何 print 原语，只能用 `<cstdio>` 的 |
| Python | `ct.print(f"…")` | 支持 f-string，也能当普通 `print()` 用 |

Python 里还有个近亲 `ct.printf`，但它的文档明写 *"Only tile input is supported"* ——
**标量传不进去**，`ct.bid(0)` 这种不能塞给它。要打标量就用 `ct.print`。

（TileIR 层其实有 `cuda_tile.print_tko`，但 C++ 头文件没暴露出来，Python 包成 `ct.print`。）

### 3. C++ 用 `%u` 而不是 `%d`

`ct::bid()` 返回 `uint3`、`ct::num_blocks()` 返回 `dim3`，成员都是 `unsigned`。
写成 `int bid_x = ct::bid().x;` 配 `%u` 能编过，但会有 `-Wformat` 告警。

### 4. Python 的 CUDA context 不会自动建

这是**本主题独有**的坑 —— 因为 hello-world 一个数组参数都没有，没人替它把 context 建起来：

| 写法 | 结果 |
| --- | --- |
| 只 `cp.cuda.get_current_stream()` | ✗ `RuntimeError: invalid device context` |
| `cp.empty(0)` 之后取流 | ✗ 仍然失败 —— **0 字节不算分配** |
| `cp.empty(1)` 之后取流 | ✓ |
| **`cp.cuda.Device(0).use()`** 之后取流 | ✓ ← 最干净 |
| `torch.cuda.init()` 之后取流 | ✓ |

**平时不会遇到这个**：只要 kernel 有数组参数，造数组那一步就已经把 context 建好了。
所以只有 hello-world 这种零数组参数的例子需要显式初始化。

### 5. 同一个东西，Rust 叫 program、C++/Python 叫 block

看上面的输出就能发现 —— Rust 打印的是 "from **program**"，C++/Python 是 "from **block**"。
指的是同一个东西（一个 tile block）。读 Rust 文档时把 "program" 当 "block" 翻译即可。

## 相关

- [跨语言对照索引](../README.md)
- 三种语言的编译 / 运行方式：各线自己的 README
  （[Rust](../../../cutile-rs-example/README.md) ·
  [C++](../../../cpp/examples/tile/README.md) ·
  [Python](../../../python/examples/tile/README.md)）
