# CUDA Programming But Rust 🦀⚡

> **中文版（主要）** → [README.md](./README.md)
>
> **English (primary mirror)** ← You are here

A learning project for **cuda-oxide** ([NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)) — writing CUDA GPU kernels in pure Rust, no `.cu` files needed! **Work in progress 🚧**

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [📘 cuda-oxide Book v0.1.0 (Chinese Translation)](./cuda-oxide-zh.CN/README.md) | Chinese translation of the official cuda-oxide book |
| [📙 CUDA Programming Guide v13.3 (Chinese Translation)](./CUDA编程指南-V13.3-zh.CN/README.md) | Chinese translation of NVIDIA's CUDA Programming Guide |

---

## ✨ Features

### 🦀 Rust CUDA Kernels (`src/`)

| Module | File | Contents |
|--------|------|----------|
| **elementwise** | [`src/elementwise/`](./src/elementwise/mod.rs) | Element-wise add: f32, f32x4 vectorized, f16 half-precision, `#[device]` helper function |
| **generic** | [`src/generic/`](./src/generic/mod.rs) | Generic kernels: `scale<T>`, `add<T>`, closure capture & higher-order functions |
| **constant_memory** | [`src/constant_memory/`](./src/constant_memory/mod.rs) | CUDA constant memory: scalar/array constants, host-side setters |
| **shared_mem** | [`src/shared_mem/`](./src/shared_mem/mod.rs) | Shared memory: `SharedArray` static, dynamic shared memory, alignment (16/128/256 bytes), mixed-alignment partitioning |
| **barrier** | [`src/barrier/`](./src/barrier/mod.rs) | Thread barriers: mbarrier synchronization, shared-memory data exchange under a barrier |
| **device_global** | [`src/device_global/`](./src/device_global/mod.rs) | Device globals: cross-kernel shared state, static constant tables, static initializer edges (NaN bit patterns) |
| **atomic** | [`src/atomic/`](./src/atomic/mod.rs) | Atomic operations: 19 tests covering multiple ops/types/scopes/memory orderings |
| **printf** | [`src/printf/`](./src/printf/mod.rs) | GPU `gpu_printf!`: integers, floats, booleans, formatted output |
| **utils** | [`src/utils/`](./src/utils/mod.rs) | Helper: `get_launch_config_1d` shorthand for launch configs |

### 🔬 C++ CUDA Reference (`use_cpp/`)

Traditional CUDA C++ examples for side-by-side comparison:

| Example | File | Description |
|---------|------|-------------|
| Device Attributes | [`get_device_attr.cu`](./use_cpp/examples/get_device_attr.cu) | Query GPU device properties |
| Atomic Ops | [`atomic_test.cu`](./use_cpp/examples/atomic_test.cu) | CUDA atomic operations |
| Constant Memory | [`const_mem.cu`](./use_cpp/examples/const_mem.cu) | Constant memory usage |
| Explicit Memory | [`explicit_mem.cu`](./use_cpp/examples/explicit_mem.cu) | `cudaMalloc`/`cudaMemcpy` etc. |
| Shared Memory | [`shared_mem.cu`](./use_cpp/examples/shared_mem.cu) | Shared memory tile loading |
| Unified Memory | [`unified_mem.cu`](./use_cpp/examples/unified_mem.cu) | `cudaMallocManaged` |
| Stream + Event | [`stream_event.cu`](./use_cpp/examples/stream_event.cu) | CUDA streams and events |
| Transpose v1 | [`transpose_v1.cu`](./use_cpp/examples/transpose_v1.cu) | Naive matrix transpose |
| Transpose (shared mem) | [`transpose_with_shared_mem.cu`](./use_cpp/examples/transpose_with_shared_mem.cu) | Shared-memory optimized transpose |

Build with CMake:
```bash
cd use_cpp && cmake -B build && make -C build
```

### 🐍 Python CUDA Reference (`use_python/`)

| File | Description |
|------|-------------|
| [`about_package/vecadd.cu`](./use_python/about_package/vecadd.cu) | CUDA C++ template kernel |
| [`about_package/use_numba.py`](./use_python/about_package/use_numba.py) | Numba CUDA example |
| [`about_package/use_pycuda.py`](./use_python/about_package/use_pycuda.py) | PyCUDA example |
| [`about_package/use_pytorch.py`](./use_python/about_package/use_pytorch.py) | PyTorch CUDA example |
| [`kernels/elementwise.py`](./use_python/kernels/elementwise.py) | Python element-wise kernel |

---

## 🚀 Quick Start

### Prerequisites

| Item | Requirement | Notes |
|------|-------------|-------|
| OS | Linux | Ubuntu 24.04 tested; other distros may work |
| NVIDIA GPU | Ampere+ (sm_80+) | Driver 545+ recommended |
| CUDA Toolkit | 12.x+ | `nvcc` and `cuda.h` must be available |
| LLVM | 21+ | Must include the NVPTX backend |
| Clang | 21+ | `clang-21` — needed by bindgen for host `cuda-bindings` |
| Rust | Nightly (pinned) | Pinned in `rust-toolchain.toml` |

> ⚠️ cuda-oxide currently targets Linux only. Windows is not supported.

### Installation Notes

See the [official cuda-oxide installation guide](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html) for full instructions. Key points:

- **Dev Container** or **Nix flake** are the fastest way to get a reproducible environment
- **Manual setup** requires: CUDA Toolkit → LLVM 21+ (with NVPTX backend) → Clang 21+ → Rust nightly (`rust-src` + `rustc-dev` components) → `cargo-oxide`
- Run `cargo oxide doctor` for one-shot environment diagnostics
- If using the Dev Container, you can skip manual CUDA Toolkit, LLVM, Clang, and Rust setup

### Run Tests in This Project

```bash
git clone https://github.com/jhqxxx/cuda-programming-but-rust.git
cd cuda-programming-but-rust

# Run all tests
cargo oxide test

# Run tests for a specific module
cargo oxide test -- elementwise
cargo oxide test -- shared_mem
cargo oxide test -- constant_memory
cargo oxide test -- generic
cargo oxide test -- printf
```

---

## 🧪 Test Results

**43 tests, all passing ✅**

| Module | Tests | Coverage |
|--------|-------|----------|
| `elementwise` | 4 | f32 add, f32x4 vectorized, f16, device helper |
| `generic` | 6 | scale(f32/f16), add(f32/f16), closure(f32/f16) |
| `constant_memory` | 2 | linear transform, polynomial |
| `shared_mem` | 6 | single/dual static, dynamic basic/partition/128-align/mixed-align |
| `barrier` | 2 | mbarrier sync barrier, shared-memory data exchange under a barrier |
| `device_global` | 3 | device globals, static constant tables, static initializer edges (NaN etc.) |
| `atomic` | 19 | multiple atomic ops/types, `.gpu` vs `.cta` scopes, multiple memory orderings |
| `printf` | 1 | integers/floats/formatting/booleans/return value |

---

## 📦 Project Structure

```
cuda-programming-but-rust/
├── Cargo.toml                 # Dependencies (cuda-device, cuda-host, cuda-core)
├── rust-toolchain.toml        # Nightly toolchain config
├── README.zh-CN.md            # Chinese README (primary)
├── README.md                  # English README (mirror)
├── src/                       # Rust CUDA source code
│   ├── lib.rs                 # Library entry
│   ├── main.rs                # Binary entry
│   ├── elementwise/           # Element-wise kernels
│   ├── generic/               # Generic kernels
│   ├── constant_memory/       # Constant memory kernels
│   ├── shared_mem/            # Shared memory kernels
│   ├── barrier/               # Thread barrier kernels (mbarrier)
│   ├── device_global/         # Device global kernels
│   ├── atomic/                # Atomic operation kernels
│   ├── printf/                # GPU printf kernels
│   └── utils/                 # Utility functions
├── use_cpp/             # Reference: C++ CUDA implementations
│   ├── CMakeLists.txt
│   ├── common/                # Shared headers and CUDA utilities
│   └── examples/              # C++ CUDA examples
├── use_python/                # Reference: Python CUDA implementations
│   ├── about_package/         # Numba/PyCUDA/PyTorch examples
│   └── kernels/               # Python kernels
├── cuda-oxide-zh.CN/          # cuda-oxide Book Chinese translation
└── CUDA编程指南-V13.3-zh.CN/  # CUDA Programming Guide Chinese translation
```

---

## 🧠 Learning Path

1. **Getting Started** → [cuda-oxide Book Ch.1](./cuda-oxide-zh.CN/01-入门/01-安装.md)
2. **Element-wise** → [`elementwise`](./src/elementwise/mod.rs) — Simplest kernels
3. **Generics & Closures** → [`generic`](./src/generic/mod.rs) — Rust generics on GPU
4. **Constant Memory** → [`constant_memory`](./src/constant_memory/mod.rs) — Read-only parameter caching
5. **Shared Memory** → [`shared_mem`](./src/shared_mem/mod.rs) — Intra-block collaboration
6. **Thread Barriers** → [`barrier`](./src/barrier/mod.rs) — mbarrier & block synchronization
7. **Device Globals** → [`device_global`](./src/device_global/mod.rs) — Cross-kernel shared state
8. **Atomics** → [`atomic`](./src/atomic/mod.rs) — Atomicity & memory ordering
9. **Debug Output** → [`printf`](./src/printf/mod.rs) — GPU-side debugging

---

## 🔧 Tech Stack

- **[cuda-oxide](https://github.com/NVlabs/cuda-oxide)** — NVIDIA's open-source Rust CUDA compiler backend
- **Rust nightly** — Custom codegen backend via `rustc_codegen_cuda`
- **CUDA 12.x** — Low-level CUDA driver API (wrapped by `cuda-core` crate)
- **CMake** — Build system for C++ CUDA examples
- **Python** — PyCUDA / Numba / PyTorch CUDA reference implementations

---

## 📚 References

- [cuda-oxide Official Docs](https://nvlabs.github.io/cuda-oxide/index.html)
- [NVlabs/cuda-oxide GitHub](https://github.com/NVlabs/cuda-oxide)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## 📄 License

[Apache-2.0](./LICENSE)

## ⚠️ Notes

- This project is a **work in progress** — content will be updated continuously
- cuda-oxide is in early stages; APIs may change
- Documentation translations are AI-assisted; corrections are welcome


