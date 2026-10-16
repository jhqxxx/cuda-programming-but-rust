// cpp/examples/tile/01-hello-world.cu
//
// 每个 tile block 从设备端打印自己的坐标 —— 建立「块索引」这个概念。
// 注意没有 threadIdx：一个 block 就是一个逻辑线程，所以每个 block 只打印一行。
//
// 对照实现（同一 tile 模型、同一套 TileIR 编译器）：
//   Rust    cutile-rs-example/01-hello-world.rs
//   Python  python/examples/tile/01-hello-world.py
//
// 编译要求见 ../README.md：nvcc --enable-tile -std=c++20
#include <cstdio>

// ⚠️ 顺序不能反：cuda_utils.h 带来 <cuda_runtime.h>（其中定义了 __location__ 那族宏），
// cuda_tile.h 依赖它。原因见 ../README.md 的 IntelliSense 一节。
#include "cuda_utils.h"
#include "cuda_tile.h"

namespace ct = cuda::tiles;

// grid 尺寸（3 个轴）。可以改着试：(2, 2, 1) / (3, 3, 1) / (2, 2, 2)
inline constexpr unsigned GRID_X = 2;
inline constexpr unsigned GRID_Y = 2;
inline constexpr unsigned GRID_Z = 1;

__tile_global__ void helloWorldKernel()
{
    // ct::bid() 返回 uint3：本 block 在三个轴上的下标
    unsigned bid_x = ct::bid().x;
    unsigned bid_y = ct::bid().y;
    unsigned bid_z = ct::bid().z;

    // ct::num_blocks() 返回 dim3：三个轴上各有多少个 block
    unsigned num_x = ct::num_blocks().x;
    unsigned num_y = ct::num_blocks().y;
    unsigned num_z = ct::num_blocks().z;

    unsigned total = num_x * num_y * num_z;

    // 设备端打印就用标准的 printf —— cuda_tile.h 里**没有** print 原语
    // （TileIR 那边是 cuda_tile.print_tko，但 C++ 头文件没暴露出来）。
    // 用 %u 而不是 %d：上面两个返回的都是无符号类型。
    printf("Hello from block <%u, %u, %u> in a grid of <%u, %u, %u> blocks! %u total blocks\n",
           bid_x, bid_y, bid_z, num_x, num_y, num_z, total);
}

int main() 
{
    // 第二个 chevron 参数必须是 1 —— 块内线程数由编译器决定
    helloWorldKernel<<<dim3(GRID_X, GRID_Y, GRID_Z), 1>>>();
    // 设备端 printf 要等同步之后才会落到 stdout
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}