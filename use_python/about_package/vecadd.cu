#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

__global__ void vecadd_kernel(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

void vecadd(torch::Tensor a, torch::Tensor b, torch::Tensor c, int N)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "all tensor need on GPU");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "tensor need be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32 && c.dtype() == torch::kFloat32, "only support float32 type");
    TORCH_CHECK(a.numel() == N && b.numel() == N && c.numel() == N, "a,b,c size need be same");
    float* a_ptr = a.data_ptr<float>();
    float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    const int threads = 256;
    const int blocks = cuda::ceil_div(N, threads);
    vecadd_kernel<<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vecadd", &vecadd, "Vector add");
}