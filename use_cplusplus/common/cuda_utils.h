// common/cuda_utils.h
#pragma once
#include <stdio.h>
#include <ctime>
#include <cuda_runtime.h> 
#include <cuda/cmath>
#include <random>      
#include <type_traits> 

#define INDX(row, col, ld) (((row) * (ld)) + (col))

#define CUDA_CHECK(expr_to_check)                          \
    do                                                     \
    {                                                      \
        cudaError_t result = expr_to_check;                \
        if (result != cudaSuccess)                         \
        {                                                  \
            fprintf(stderr,                                \
                    "CUDA Runtime Error: %s:%i:%d = %s\n", \
                    __FILE__,                              \
                    __LINE__,                              \
                    result,                                \
                    cudaGetErrorString(result));           \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0)

template <typename T>
void initArray(T *arr, int length)
{
    static thread_local std::mt19937 generator(std::time(nullptr));

    if constexpr (std::is_floating_point<T>::value)
    {
        std::uniform_real_distribution<T> distribution(0.0, 1.0);
        for (int i = 0; i < length; ++i)
        {
            arr[i] = distribution(generator);
        }
    }
    else if constexpr (std::is_integral<T>::value)
    {
        std::uniform_int_distribution<T> distribution(0, 100);
        for (int i = 0; i < length; ++i)
        {
            arr[i] = distribution(generator);
        }
    }
    else
    {
        static_assert(std::is_arithmetic<T>::value, "initArray only supports arithmetic types");
    }
}

template <typename T>
bool vectorApproximatelyEqual(T *A, T *B, int length)
{
    for (int i = 0; i < length; i++)
    {
        T dif = fabs(A[i] - B[i]);
        if constexpr (std::is_floating_point<T>::value)
        {
            if (dif > 0.00001)
            {
                printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
                return false;
            }
        }
        else if constexpr (std::is_integral<T>::value)
        {
            if (dif != 0)
            {
                printf("Index %d mismatch: %d != %d", i, A[i], B[i]);
                return false;
            }
        }
        else
        {
            static_assert(std::is_arithmetic<T>::value, "initArray only supports arithmetic types");
        }
    }
    return true;
}