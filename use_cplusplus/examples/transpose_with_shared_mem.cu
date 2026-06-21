#include "cuda_utils.h"

#define THREADS_PER_BLOCK_X 2
#define THREADS_PER_BLOCK_Y 2

__global__ void smemTranspose(int m, float *A, float *B)
{
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];
    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if (myCol < m && myRow < m)
    {
        smemArray[threadIdx.x][threadIdx.y] = A[INDX(tileX + threadIdx.x, tileY + threadIdx.y, m)];
    }
    __syncthreads();

    if (myRow < m && myCol < m)
    {
        // 需要考虑grid和block两个层面的索引
        B[INDX(tileY + threadIdx.x, tileX + threadIdx.y, m)] = smemArray[threadIdx.y][threadIdx.x];
    }
}

int main(int argc, char **argv)
{
    int m = 4;
    if (argc >= 2)
    {
        m = std::atoi(argv[1]);
    }
    float *A = nullptr;
    float *B = nullptr;

    float *devA = nullptr;
    float *devB = nullptr;

    CUDA_CHECK(cudaMallocHost(&A, (m * m) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, (m * m) * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&devA, (m * m) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, (m * m) * sizeof(float)));

    initArray(A, m * m);
    for (int i = 0; i < m; i++)
    {
        printf("row: %d", i);
        for (int j = 0; j < m; j++)
        {
            printf(" %f ", A[i * m + j]);
        }
        printf("\n");
    }
    CUDA_CHECK(cudaMemcpy(devA, A, (m * m) * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(devB, 0, (m * m) * sizeof(float)));
    dim3 blockDim(2, 2);
    dim3 grid(2, 2);
    smemTranspose<<<grid, blockDim>>>(m, devA, devB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(B, devB, (m * m) * sizeof(float), cudaMemcpyDefault));
    for (int i = 0; i < m; i++)
    {
        printf("row: %d", i);
        for (int j = 0; j < m; j++)
        {
            printf(" %f ", B[i * m + j]);
        }
        printf("\n");
    }
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    return 0;
}