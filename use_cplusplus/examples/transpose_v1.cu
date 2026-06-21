#include "cuda_utils.h"

__global__ void cudaTranspose(int m, float *A, float *B)
{
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    if (myCol < m && myRow < m)
    {
        B[INDX(myCol, myRow, m)] = A[INDX(myRow, myCol, m)];
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
#define INDX(row, col, ld) (((row) * (ld)) + (col))
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
    dim3 blockDim(m, m);
    cudaTranspose<<<1, blockDim>>>(m, devA, devB);
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