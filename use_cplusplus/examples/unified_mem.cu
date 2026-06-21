// examples/unified_mem.cu
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
    // cudaMallocManaged 分配统一虚拟地址空间，初始时没有物理内存
    CUDA_CHECK(cudaMallocManaged(&A, vectorLength * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, vectorLength * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, vectorLength * sizeof(float)));
    // CPU访问 AB->触发缺页，物理页分配在主机内存中
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    // GPU访问 ABC-> 触发缺页，
    //   1. 硬件将AB的页面通过PCIe迁移到GPU显存中，此时，主机端的原始物理页被释放或标记为无效，不再保留有效数据
    //   2. C缺页，物理页直接分配在GPU显存中
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CPU访问 AB-> 触发缺页，通过PCIe将AB从当前页GPU显存中，迁移回主机内存中，设备端的物理页被释放
    serialVecAdd(A, B, comparisonResult, vectorLength);
    // CPU访问 C-> 触发缺页，通过PCIe将C从当前页GPU显存中，迁移回主机内存中，设备端的物理页被释放
    if (vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match \n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match \n");
    }
    // cudaMallocManaged分配的内存，必须用cudaFree来释放，无论它当前时是在CPU上还是GPU上
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    free(comparisonResult);

    return 0;
}