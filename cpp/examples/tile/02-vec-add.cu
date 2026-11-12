#include "cuda_utils.h"
#include "cuda_tile.h"
#include <vector>
#include <cstdio>

namespace ct = cuda::tiles;
using namespace ct::literals;

// 1维
__tile_global__ void vec_add(float *__restrict__ a, float *__restrict__ b, float *__restrict__ out, int n)
{
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{n}}, ct::shape{4_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{n}}, ct::shape{4_ic}};
    auto oView = ct::partition_view{ct::tensor_span{out, ct::extents{n}}, ct::shape{4_ic}};

    unsigned bx = ct::bid().x;
    auto aTile = aView.load(bx);
    auto bTile = bView.load(bx);
    oView.store(aTile + bTile, bx);
}

// 2维
__tile_global__ void vec_add2(float *__restrict__ a, float *__restrict__ b, float *__restrict__ out, int rows, int cols)
{
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{rows, cols}}, ct::shape{4_ic, 4_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{rows, cols}}, ct::shape{4_ic, 4_ic}};
    auto oView = ct::partition_view{ct::tensor_span{out, ct::extents{rows, cols}}, ct::shape{4_ic, 4_ic}};

    unsigned bx = ct::bid().x;
    unsigned by = ct::bid().y;
    auto aTile = aView.load(bx, by);
    auto bTile = bView.load(bx, by);
    oView.store(aTile + bTile, bx, by);
}

// 2维, 非正方形分区
__tile_global__ void vec_add2_rect(float *__restrict__ a, float *__restrict__ b, float *__restrict__ out, int rows, int cols)
{
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView = ct::partition_view{ct::tensor_span{a, ct::extents{rows, cols}}, ct::shape{4_ic, 8_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b, ct::extents{rows, cols}}, ct::shape{4_ic, 8_ic}};
    auto oView = ct::partition_view{ct::tensor_span{out, ct::extents{rows, cols}}, ct::shape{4_ic, 8_ic}};

    unsigned bx = ct::bid().x;
    unsigned by = ct::bid().y;
    auto aTile = aView.load(bx, by);
    auto bTile = bView.load(bx, by);
    oView.store(aTile + bTile, bx, by);
}

namespace
{
    // 跑一个向量加法用例：准备 → 分配/上传 → 启动 → 回读 → 校验 → 释放。
    //
    // 三段 host 代码里**只有「启动 kernel」这一步不同**（三个 kernel 的参数个数和
    // grid 都不一样），所以用 lambda 把它当参数传进来：helper 的接口保持统一，
    // 每个用例各自适配自己的 kernel。
    //
    // 最重要的：`len` 一个参数同时决定 4 件事 ——
    //   数组大小、malloc 字节数、memcpy 长度、比较长度。
    template <typename Launch>
    bool run_case(const char *name, int len, Launch &&launch)
    {
        // 1. host 侧准备：输入全 1.0，期望值 = a + b
        std::vector<float> ha(len, 1.0f), hb(len, 1.0f), hout(len, 0.0f), href(len, 0.0f);
        for (int i = 0; i < len; ++i)
        {
            href[i] = ha[i] + hb[i];
        }

        // 2. device 侧：分配 + 上传
        float *da = nullptr, *db = nullptr, *dout = nullptr;
        CUDA_CHECK(cudaMalloc(&da, len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db, len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dout, len * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(da, ha.data(), len * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb.data(), len * sizeof(float), cudaMemcpyHostToDevice));

        // 3. 唯一因用例而异的一步
        launch(da, db, dout);

        // 4. 同步 + 回读
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hout.data(), dout, len * sizeof(float), cudaMemcpyDeviceToHost));

        // 5. 校验（vectorApproximatelyEqual 在不匹配时会打印 Index N mismatch）
        const bool ok = vectorApproximatelyEqual(hout.data(), href.data(), len);

        // 6. 释放 —— 放在算出 ok 之后，这样将来往校验里加东西也不会用到已释放的指针
        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(dout));

        printf("%-28s %s\n", name, ok ? "✓ 全部正确" : "✗ 结果不对");
        return ok;
    }
} // namespace

int main()
{
    constexpr int vector_size = 32;
    constexpr int tile_size = 4;
    constexpr int blocks = vector_size / tile_size; // 1 维网格，也是 2 维方块的每轴块数
    constexpr int len2d = vector_size * vector_size;

    // ⚠️ ct::shape 里只能写字面量（4_ic / 8_ic），host 侧算 grid 只能另写一份。
    //    改下面这两个值时必须同步改 vec_add2_rect 里的 ct::shape{4_ic, 8_ic}。
    constexpr int rect_r = 4;
    constexpr int rect_c = 8;

    bool ok = true;

    // 1 维：grid = (32/4, 1) = (8, 1)
    ok &= run_case("vec_add (1 维 tile=4)", vector_size,
                   [&](float *a, float *b, float *out)
                   {
                       vec_add<<<blocks, 1>>>(a, b, out, vector_size);
                   });

    // 2 维方块：ct::shape{4,4} → grid = (32/4, 32/4) = (8, 8)
    ok &= run_case("vec_add2 (2 维 tile=4x4)", len2d,
                   [&](float *a, float *b, float *out)
                   {
                       vec_add2<<<dim3(blocks, blocks), 1>>>(a, b, out, vector_size, vector_size);
                   });

    // 2 维非方块：ct::shape{4,8} → grid = (32/4, 32/8) = (8, 4)
    ok &= run_case("vec_add2_rect (2 维 tile=4x8)", len2d,
                   [&](float *a, float *b, float *out)
                   {
                       vec_add2_rect<<<dim3(vector_size / rect_r, vector_size / rect_c), 1>>>(
                           a, b, out, vector_size, vector_size);
                   });

    printf("\n%s\n", ok ? "✓ 全部用例通过" : "✗ 有用例失败");
    return ok ? 0 : 1;
}