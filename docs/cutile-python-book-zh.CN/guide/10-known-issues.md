# 已知问题

1. **FP8 的 Torch Tensor 需要 `torch>=2.10`。** 更早版本的 PyTorch 不支持通过 dlpack
   协议转换 fp8 数据类型，并且在转换为 dlpack tensor 失败时会**泄漏内存**。
