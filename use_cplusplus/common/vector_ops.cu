// common/vector_ops.cu
#include "cuda_utils.h"

__global__ void vecAdd(float *A, float *B, float *C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void serialVecAdd(float *A, float *B, float *res, int vectorLength)
{
    for (int i = 0; i < vectorLength; i++)
    {
        res[i] = A[i] + B[i];
    }
}
