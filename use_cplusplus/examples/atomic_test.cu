#include "cuda_utils.h"
#include <cuda/atomic>

__global__ void sumReduction(int n, float *array, float *result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(*result);
    result_ref.fetch_add(array[tid]);
}

int main(int argc, char **argv)
{
    float A[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float *B = nullptr;
    CUDA_CHECK(cudaMallocHost(&B, 1 * sizeof(float)));
    float *devA = nullptr;
    float *devB = nullptr;
    CUDA_CHECK(cudaMalloc(&devA, 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, 1 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(devA, A, 5 * sizeof(float), cudaMemcpyDefault));
    sumReduction<<<1, 5>>>(5, devA, devB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(B, devB, 1 * sizeof(float), cudaMemcpyDefault));
    printf("sum: %f", *B);
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    return 0;
}