# CUDA Programming But Rust 🦀⚡

> **中文版（主要）** → [README.md](./README.md)

<p align="center">
  <b>Write GPU kernels in Rust — no <code>.cu</code> files.</b>
</p>

A CUDA learning repository. The focus is **writing GPU kernels in Rust**, plus Chinese
translations of three official books. It also keeps a set of earlier CUDA C++ and Python
examples — these are essentially frozen and rarely updated, kept for reference only.

---

## Two Rust paths

There are two usable Rust GPU programming paths today; this repo has a practice area for each:

| | [cuTile Rust](./cutile-example/) | [cuda-oxide](./cuda-oxide-example/) |
| --- | --- | --- |
| Approach | High-level tile-oriented programming model | Custom rustc codegen backend, compiles kernels straight to PTX |
| Toolchain | **stable** (root `1.99.0`) | **nightly-2026-10-11** (needs `rustc-dev`) |
| How to run | `cargo run --example <name>` | `cargo oxide run` |
| Examples live in | `cutile-example/*.rs`, registered as `[[example]]` in the root `Cargo.toml` | `cuda-oxide-example/<name>/`, each a standalone project |

Their toolchain requirements are mutually exclusive (stable vs nightly), so they are
**physically isolated**: `cuda-oxide-example/` ships its own `rust-toolchain.toml` and is
excluded from the root workspace. See each area's README for details and pitfalls.

## Reference examples (C++ / Python)

These are earlier examples and are essentially frozen — they no longer evolve with the
Rust side and are kept for reference:

- **CUDA C++** — `use_cpp/examples/`, 9 examples built with CMake: device attributes,
  atomics, constant memory, explicit memory, unified memory, shared memory, streams and
  events, matrix transpose (naive and shared-memory optimized)
- **Python** — `use_python/about_package/`, using Numba, PyCUDA and PyTorch bindings

## Layout

```
.
├── cutile-example/             # cuTile Rust examples (stable)
├── cuda-oxide-example/         # cuda-oxide examples (nightly)
│   └── 01-vector-add/          # vector add: c[i] = a[i] + b[i]
├── use_cpp/                    # CUDA C++ reference examples
├── use_python/                 # Python reference examples
├── docs/                       # Chinese documentation
│   ├── cuda-oxide-book-zh.CN/        # cuda-oxide book (42 pages)
│   ├── cutile-book-zh.CN/            # cuTile book (29 pages)
│   ├── CUDA编程指南-V13.3-zh.CN/      # CUDA Programming Guide (15 pages)
│   ├── atomic.md                     # atomics and memory ordering
│   └── barrier.md                    # mbarrier explained
└── assets/img/
```

## Quick start

```bash
# cuda-oxide example (needs nightly; the codegen backend is built on first use)
cd cuda-oxide-example/01-vector-add
cargo oxide doctor   # check toolchain, nvcc, GPU, backend cache
cargo oxide run      # compile and run on the GPU

# CUDA C++ example
cd use_cpp && mkdir -p build && cd build && cmake .. && make
./transpose_with_shared_mem
```

## Documentation

Chinese translations of three official books live under `docs/`:

| Document | Description |
| --- | --- |
| [cuda-oxide book](./docs/cuda-oxide-book-zh.CN/README.md) | Official cuda-oxide book |
| [cuTile book](./docs/cutile-book-zh.CN/README.md) | Official cuTile Rust book |
| [CUDA Programming Guide v13.3](./docs/CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA C++ Programming Guide |

## Environment (as tested)

| Item | Version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 (compute capability 8.6) |
| Driver | 595.99.02 |
| CUDA Toolkit | 13.3 (`/usr/local/cuda`) |
| Rust (main workspace) | 1.99.0 stable |
| Rust (cuda-oxide) | nightly-2026-10-11 |
| cargo-oxide | 0.2.1, built from `NVIDIA/cuda-rust` at `bc46b1ff` |

## Status

🚧 Work in progress. The cuda-oxide side has a working `01-vector-add`; the cuTile side has
its conventions and template in place.

## References

- cuda-rust: <https://github.com/NVIDIA/cuda-rust>
- CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html/>

