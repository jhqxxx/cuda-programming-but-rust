// cpp/examples/tile/00-add.cu
//
// tile 模型的向量加法：z = x + y
//
// 编程单位是**一个 tile**（一整块数据），不是一根线程 —— 没有 threadIdx，
// 边界由 load_masked / store_masked 处理，块内的线程由编译器分配。
//
// 对照实现：
//   Rust   cutile-rs-example/00-add.rs        （cuTile，同一模型）
//   Python python/examples/tile/              （cuTile Python，同一模型）
//   不要拿 ../simt/ 比 —— SIMT 是另一种抽象
//
// 编译要求（两条都不能少，见 ../README.md）：
//   nvcc --enable-tile -std=c++20 -arch=sm_86
#include <vector>

// ⚠️ 顺序不能反：cuda_utils.h 带来 <cuda_runtime.h>，其中定义了 __location__
// 这一族宏。而 cuda_tile.h（准确说是它 include 的 crt/cuda_tile.h）**不自己
// 包含 host_defines.h**，它假定调用方已经包含过；__tile__ / __tile_global__ /
// __tile_builtin__ 全都展开成 __location__(...)。
//
// nvcc 把 __location__ 当编译器内建属性，所以顺序反了也照样编过；但
// IntelliSense 之类的非 nvcc 解析器会把它当成语法错误，进而跳过那几条声明，
// 报出 “namespace "cuda::tiles" has no member "assume_aligned"”。所以
// **cuda_utils.h 必须排在 cuda_tile.h 前面**。
#include "cuda_utils.h"
#include "cuda_tile.h"

// host 侧算网格用的 tile 大小。
// 必须和 kernel 里的 `128_ic` 一致：`ct::shape` 要的是**编译期值**
// （`ct::integral_constant`），普通的 `constexpr int` 传不进去 —— 写
// `ct::shape{TILE}` 会报 “cannot deduce class template arguments”。
inline constexpr int TILE = 128;

__tile_global__ void vecAddTile(float *__restrict__ a, float *__restrict__ b,
                                float *__restrict__ out, int n)
{
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    // tile 的 partition_view 要降到 TMA 必须有 16 字节对齐的指针
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    // ct::extents{n} 用运行期维度（数组长度只有启动时才知道）；
    // 128_ic 保持编译期 —— 形状必须编译期已知，才能生成专用机器码。
    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape{128_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape{128_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{128_ic}};

    // 只有块索引。整套 threadIdx 的活编译器干了。
    int bx = ct::bid().x;

    // n 不是 TILE 的整数倍，最后一块会越界，所以走 masked 变体：
    //   load_masked  越界 lane 补 0（加法单位元，结果正确）
    //   store_masked 越界写入直接丢弃
    auto aTile = aView.load_masked(bx);
    auto bTile = bView.load_masked(bx);
    outView.store_masked(aTile + bTile, bx);
}

int main()
{
    constexpr int N = 1000;
    constexpr int num_blocks = (N + TILE - 1) / TILE; // 上取整，8 块

    std::vector<float> h_a(N), h_b(N), h_out(N), h_ref(N);
    initArray(h_a.data(), N);
    initArray(h_b.data(), N);
    for (int i = 0; i < N; ++i)
    {
        h_ref[i] = h_a[i] + h_b[i];
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    // 第二个 chevron 参数必须是 1 —— 块内线程数由编译器决定
    vecAddTile<<<num_blocks, 1>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    const bool ok = vectorApproximatelyEqual(h_out.data(), h_ref.data(), N);
    printf("tile 00-add: N=%d, TILE=%d, %d 个 tile block\n", N, TILE, num_blocks);
    printf("z[:4] = %.4f %.4f %.4f %.4f\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    printf("%s\n", ok ? "✓ 全部元素正确" : "✗ 结果不匹配");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    return ok ? 0 : 1;
}
