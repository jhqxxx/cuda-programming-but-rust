from numba import cuda
import cupy as cp
import numpy as np

@cuda.jit
def vecadd(a, b, c):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    c[idx] = a[idx] + b[idx]

vector_size = 2**24 + 11

device = cp.cuda.Device()
a = cp.random.uniform(-1, 1, vector_size)
b = cp.random.uniform(-1, 1, vector_size)
c = cp.zeros_like(a)

block_size = 256
grid_size = int(np.ceil(vector_size/block_size))
vecadd[grid_size, block_size](a, b, c)

device.synchronize()

a_np = cp.asnumpy(a)
b_np = cp.asnumpy(b)
c_np = cp.asnumpy(c)

expected = a_np + b_np
np.testing.assert_array_almost_equal(c_np, expected)

print("Test succeeded")