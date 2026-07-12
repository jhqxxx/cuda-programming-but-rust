#include "cuda_utils.h"

__global__ void example_syncthreads(int *input_data, int *output_data)
{
    // 此为静态共享内存
    // 静态共享内存在编译时必须具有确定大小
    // 所以这里的数值不能用 blockDim.x或函数参数传进来
    __shared__ int shared_data[128];
    shared_data[threadIdx.x] = input_data[blockDim.x * blockIdx.x + threadIdx.x];
    __syncthreads(); // 线程块内线程同步

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++)
        {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}

__global__ void example_syncthreads_dynamic(int *input_data, int *output_data)
{
    // 此为动态共享内存
    // 大小通过函数调用的三括号传进来
    // 一个kernel只能有一个动态分配的共享数组
    extern __shared__ int shared_data[];
    shared_data[threadIdx.x] = input_data[blockDim.x * blockIdx.x + threadIdx.x];
    __syncthreads(); // 线程块内线程同步

    if (threadIdx.x == 0)
    {
        float sum = 0;
        for (int i = 0; i < blockDim.x; i++)
        {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}

void hostSum(int *A, int *B, int vectorLength, int threads, int blocks)
{
    for (int i = 0; i < blocks; i++)
    {
        int sum = 0;
        for (int j = 0; j < threads; j++)
        {
            sum += A[i * threads + j];
        }
        B[i] = sum;
    }
}

int main(int argc, char **argv)
{
    int vectorLength = 1024;
    if (argc >= 2)
    {
        vectorLength = std::atoi(argv[1]);
    }

    int *A = nullptr;
    int *B = nullptr;

    int *devA = nullptr;
    int *devB = nullptr;
    int threads = 128;
    int blocks = cuda::ceil_div(vectorLength, threads);
    CUDA_CHECK(cudaMallocHost(&A, vectorLength * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&B, blocks * sizeof(int)));
    initArray(A, vectorLength);

    CUDA_CHECK(cudaMalloc(&devA, vectorLength * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&devB, blocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(devA, A, vectorLength * sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(devB, 0, blocks * sizeof(int)));

    // example_syncthreads<<<blocks, threads>>>(devA, devB);
    example_syncthreads_dynamic<<<blocks, threads, blocks*sizeof(int)>>>(devA, devB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(B, devB, blocks * sizeof(int), cudaMemcpyDefault));
    int *compareResult = (int *)malloc(blocks * sizeof(int));
    hostSum(A, compareResult, vectorLength, threads, blocks);
    if (vectorApproximatelyEqual(B, compareResult, blocks))
    {
        printf("Explicit Memory: CPU and GPU answers match \n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match \n");
    }
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    free(compareResult);
    return 0;
}