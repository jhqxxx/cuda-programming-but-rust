"""Hello world —— 每个 block 从设备端打印自己的坐标。

对照实现（同一 tile 模型、同一套 TileIR 编译器）：
    Rust    cutile-rs-example/01-hello-world.rs
    Python  本文件

运行：
    cd python
    uv run python examples/tile/01-hello-world.py
"""

import cupy as cp

import cuda.tile as ct

# grid 尺寸（3 个轴）。可以改着试：(2, 2, 1) / (3, 3, 1) / (2, 2, 2)
GRID = (2, 2, 1)


@ct.kernel
def hello_world_kernel():
    # ⚠️ 命名和 Rust 侧不一样：Python 是 bid / num_blocks，
    # Rust 才叫 program_id / num_programs —— 那两个名字在 cuda.tile 里不存在。
    pid0 = ct.bid(0)        # 本 block 在轴 0 上的下标
    pid1 = ct.bid(1)        # 轴 1
    pid2 = ct.bid(2)        # 轴 2
    n0 = ct.num_blocks(0)   # 轴 0 上一共有多少个 block
    n1 = ct.num_blocks(1)
    n2 = ct.num_blocks(2)

    total = n0 * n1 * n2
    ct.print(f"Hello from block <{pid0}, {pid1}, {pid2}> "
             f"in a grid of <{n0}, {n1}, {n2}> blocks! {total} total blocks")


def main():
    # cuTile 自己**不提供** stream / device / context 的构造方法 —— cuda.tile 是个
    # 「编译器 + 启动器」，上下文和流都借用你已有的运行时（cupy / torch / ...）。
    # 所以先让 cupy 把 CUDA context 建起来，再问它要当前的流：
    #   cupy  : cp.cuda.Device(0).use()  →  cp.cuda.get_current_stream()
    #   torch : torch.cuda.init()        →  torch.cuda.current_stream()
    #
    # 注意：只有 hello-world 这种「一个数组参数都没有」的 kernel 才会踩到这个坑。
    # 平时你总得先造数组，造数组那一步就已经把 context 建好了。
    
    cp.cuda.Device(0).use()
    stream = cp.cuda.get_current_stream()

    # launch 的签名是 (stream, grid, kernel, kernel_args) —— 4 个都是位置参数。
    # 这个 kernel 没有参数，所以 kernel_args 给空 tuple。
    # stream 也可以直接给裸句柄：ct.launch(0, ...) 里的 0 就是默认流。
    ct.launch(stream, GRID, hello_world_kernel, ())
    # ct.launch(0, GRID, hello_world_kernel, ())

    # 设备端的 print 要等同步之后才会落到 stdout
    stream.synchronize()


if __name__ == "__main__":
    main()