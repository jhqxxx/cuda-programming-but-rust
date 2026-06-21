#include "cuda_utils.h"
__constant__ float coeffs[2];

__global__ void compute(float *out)
{
    int idx = threadIdx.x;
    out[idx] = coeffs[0] * idx + coeffs[1];
}

int main(int argc, char **argv)
{
    int size = 10;
    float h_coeffs[2] = {1.0f, 2.0f};
    CUDA_CHECK(cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs)));
    float *out = nullptr;
    float *dev_out = nullptr;

    CUDA_CHECK(cudaMallocHost(&out, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_out, size * sizeof(float)));
    compute<<<1, 10>>>(dev_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, dev_out, size * sizeof(float), cudaMemcpyDefault));
    for (int i = 0; i < size; i++)
    {
        printf("out i: %d, num: %f \n", i, out[i]);
    }

    CUDA_CHECK(cudaFree(dev_out));
    CUDA_CHECK(cudaFreeHost(out));
    return 0;
}