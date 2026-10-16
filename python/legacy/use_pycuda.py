import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# CUDA 13 起 cuModuleLoadData 不再自动创建上下文，必须先有当前 context。
# 缺这一行时 SourceModule 会失败，而且 pycuda 报出的错误很有误导性
# （可能是 UnicodeDecodeError，也可能是 "initialization error"）。
import pycuda.autoinit  # noqa: F401  建好并托管当前线程的 context

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