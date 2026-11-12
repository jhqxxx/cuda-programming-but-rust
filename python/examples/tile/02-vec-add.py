import cuda.tile as ct
import cupy as cp
import numpy as np

# 一维
@ct.kernel
def vec_add1(z, x, y, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    # 通过块下标 pid 计算 tile 的起始索引，tile_size 是每个 tile 的长度
    a_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(y, index=(pid,), shape=(tile_size,))
    ct.store(z, index=(pid,), tile=a_tile + b_tile)

# 二维
@ct.kernel
def vec_add2(z, x, y, t1: ct.Constant[int], t2: ct.Constant[int]):
    pid0 = ct.bid(0)
    pid1 = ct.bid(1)
    a_tile = ct.load(x, index=(pid0, pid1), shape=(t1, t2))
    b_tile = ct.load(y, index=(pid0, pid1), shape=(t1, t2))
    ct.store(z, index=(pid0, pid1), tile=a_tile+b_tile)

def main():
    vector_size = 32
    tile_size = 4
    # 1维
    grid1 = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = cp.ones((vector_size,), dtype=cp.float32)
    b = cp.ones((vector_size,), dtype=cp.float32)
    c = cp.zeros((vector_size,), dtype=cp.float32)

    ct.launch(cp.cuda.get_current_stream(),
              grid1,
              vec_add1,
              (c, a, b, tile_size))

    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)
    print("vec_add1 Test passed!")

    # 2维
    grid2 = (ct.cdiv(vector_size, tile_size), ct.cdiv(vector_size, tile_size), 1)
    a2 = cp.ones((vector_size, vector_size), dtype=cp.float32)
    b2 = cp.ones((vector_size, vector_size), dtype=cp.float32)
    c2 = cp.zeros((vector_size, vector_size), dtype=cp.float32)
    ct.launch(cp.cuda.get_current_stream(),
              grid2,
              vec_add2,
              (c2, a2, b2, tile_size, tile_size))
    a2_np = cp.asnumpy(a2)
    b2_np = cp.asnumpy(b2)
    c2_np = cp.asnumpy(c2)

    expected2 = a2_np + b2_np
    np.testing.assert_array_almost_equal(c2_np, expected2)
    print("vec_add2 Test passed!")

    # 非正方形分区
    # tile_size = (4, 8)
    # grid3 = (8, 4, 1)
    grid3 = (ct.cdiv(vector_size, tile_size), tile_size, 1)
    a3 = cp.ones((vector_size, vector_size), dtype=cp.float32)
    b3 = cp.ones((vector_size, vector_size), dtype=cp.float32)
    c3 = cp.zeros((vector_size, vector_size), dtype=cp.float32)
    ct.launch(cp.cuda.get_current_stream(),
              grid3,
              vec_add2,
              (c3, a3, b3, tile_size, ct.cdiv(vector_size, tile_size))) # 
    a3_np = cp.asnumpy(a3)
    b3_np = cp.asnumpy(b3)
    c3_np = cp.asnumpy(c3)  
    expected3 = a3_np + b3_np
    np.testing.assert_array_almost_equal(c3_np, expected3)
    print("vec_add2 with non-square partition Test passed!")

if __name__ == "__main__":
    main()