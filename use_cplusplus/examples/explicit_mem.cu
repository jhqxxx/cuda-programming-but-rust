// examples/explicit.cu
#include "../common/cuda_utils.h"
#include "../common/vector_ops.cu"

int main(int argc, char **argv)
{
    int vectorLength = 1024;
    if (argc >= 2)
    {
        vectorLength = std::atoi(argv[1]);
    }

    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    float *comparisonResult = (float *)malloc(vectorLength * sizeof(float));

    float *devA = nullptr;
    float *devB = nullptr;
    float *devC = nullptr;

    // cudaMallocHost分配主机内存，页锁定内存，可提高复制效率
    // 如果系统上页锁定的主机内存过多，性能可能会下降
    // 最佳实践是仅页锁定将用于向 GPU 发送数据或从 GPU 接收数据的缓冲区
    CUDA_CHECK(cudaMallocHost(&A, vectorLength * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, vectorLength * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C, vectorLength * sizeof(float)));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // cudaMalloc分配设备内存
    CUDA_CHECK(cudaMalloc(&devA, vectorLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength * sizeof(float)));

    // cudaMemcpy将src中的数据复制到dst中，它执行的是复制操作，src中的数据保持不变
    // 即可从主机端复制到设备端
    // 也可从设备端复制到主机端
    CUDA_CHECK(cudaMemcpy(devA, A, vectorLength * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(devB, B, vectorLength * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(devC, 0, vectorLength * sizeof(float)));

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, devC, vectorLength * sizeof(float), cudaMemcpyDefault));
    serialVecAdd(A, B, comparisonResult, vectorLength);
    if (vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match \n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers do not match \n");
    }
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFreeHost(C));
    free(comparisonResult);

    return 0;
}