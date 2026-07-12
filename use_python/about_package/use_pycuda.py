import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# CUDA C 内核代码
mod = SourceModule("""
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
""")
add_kernel = mod.get_function("add")

# 准备数据
a = np.random.randn(1024).astype(np.float32)
b = np.random.randn(1024).astype(np.float32)
c = np.zeros_like(a)

# 启动内核
add_kernel(
    cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(len(a)),
    block=(256, 1, 1), grid=(4, 1, 1)
)

expected = a + b
np.testing.assert_array_almost_equal(c, expected)

print("Test succeeded")