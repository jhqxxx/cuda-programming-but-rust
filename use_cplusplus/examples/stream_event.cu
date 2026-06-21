#include "cuda_utils.h"
#include "vector_ops.cu"

__global__ void init(float *A, int m)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < m)
    {
        A[idx] = 2.0 * idx + 1.0;
    }
}
int main(int argc, char **argv)
{
    int vectorLength = 1024;
    float *A = nullptr;
    CUDA_CHECK(cudaMallocHost(&A, vectorLength * sizeof(float)));

    float *devA = nullptr;
    CUDA_CHECK(cudaMalloc(&devA, vectorLength * sizeof(float)));
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    init<<<blocks, threads, 0, stream>>>(devA, vectorLength);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyAsync(A, devA, vectorLength * sizeof(float), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t status = cudaStreamQuery(stream);
    switch (status)
    {
    case cudaSuccess:
        printf("The stream is empty \n");
        break;
    case cudaErrorNotReady:
        printf("This stream is not empty \n");
        break;
    default:
        break;
    }
    cudaStreamDestroy(stream);
    for (int i = 0; i < 10; i++)
    {
        printf("i: %d, value: %f", i, A[i]);
    }
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFreeHost(A));
    return 0;
}