# CUDA Programming But Rust 🦀⚡

> **中文版（主要）** → [README.md](./README.md)

<p align="center">
  <b>Three languages × two programming models — learn GPU programming side by side.</b>
</p>

A CUDA learning repository: three parallel tracks — Rust, C++ and Python — each covering
both the **tile** and the **SIMT** programming model, plus Chinese translations of three
official books.

Rust is where this started and remains the repository's distinctive angle, but C++ and
Python are equally unavoidable for GPU programming.

---

## Two programming models

The real dividing line in GPU programming isn't the language — it's **which level of
abstraction you use to express parallelism**:

| | tile model | SIMT model |
| --- | --- | --- |
| Unit of programming | One block of data (a tile) ≈ one thread block | One thread |
| What you describe | "which slice does my block own" | "which index does my thread compute" |
| Where the index comes from | The framework gives it (`program_id`) | You compute it (`threadIdx + blockIdx * blockDim`) |
| Thread count | Backend decides; you only pick the tile size | You specify `blockDim` explicitly |
| Typical entry points | cuTile Rust / Triton | CUDA C++ / cuda-oxide / Numba / PyCUDA |

These are **two different abstractions, not two syntaxes for the same thing** — so examples
are only ever compared against implementations in the same model.

## Three languages × two models

| | Rust | C++ | Python |
| --- | --- | --- | --- |
| **tile** | ✅ [`cutile-rs-example/`](./cutile-rs-example/) — cuTile | ⚠️ [`cpp/examples/tile/`](./cpp/examples/tile/) — no user-facing C++ entry point | ⬜ [`python/examples/tile/`](./python/examples/tile/) — Triton / cuda-tile IR |
| **SIMT** | ✅ [`cuda-oxide-example/`](./cuda-oxide-example/) — cuda-oxide | ⬜ [`cpp/examples/simt/`](./cpp/examples/simt/) — CUDA C++ | ⬜ [`python/examples/simt/`](./python/examples/simt/) — Numba / PyCUDA / PyTorch |

The C++ cell is ⚠️: the `cuda_tile.h` shipping with CUDA 13.3 is merely a device-runtime
forwarding header (`crt/cuda_tile.h`), and the only real tile entry point sits at the MLIR
layer (CUDA Tile IR). See [`cpp/examples/tile/README.md`](./cpp/examples/tile/README.md)
for the investigation.

## Two Rust paths

There are two usable Rust GPU programming paths today, matching the two models above:

| | [cuTile Rust](./cutile-rs-example/) | [cuda-oxide](./cuda-oxide-example/) |
| --- | --- | --- |
| Approach | High-level tile-oriented programming model | Custom rustc codegen backend, compiles kernels straight to PTX |
| Toolchain | **stable** (root `1.99.0`) | **nightly-2026-10-11** (needs `rustc-dev`) |
| How to run | `cargo run --example <name>` | `cargo oxide run` |
| Examples live in | `cutile-rs-example/*.rs`, registered as `[[example]]` in the root `Cargo.toml` | `cuda-oxide-example/<name>/`, each a standalone project |

Their toolchain requirements are mutually exclusive (stable vs nightly), so they are
**physically isolated**: `cuda-oxide-example/` ships its own `rust-toolchain.toml` and is
excluded from the root workspace. See each area's README for details and pitfalls.

## The C++ and Python tracks

Beyond Rust there are two more actively maintained tracks; their examples are split by
model in the same way:

- **CUDA C++** — `cpp/`, built with CMake. New SIMT examples go in `examples/simt/`;
  `examples/tile/` is deliberately empty (see the previous section); `legacy/` keeps the 9 earlier
  examples: device attributes, atomics, constant memory, explicit memory, unified memory,
  shared memory, streams and events, matrix transpose (naive and shared-memory optimized)
- **Python** — `python/`. `examples/simt/` (Numba / PyCUDA / PyTorch extension) and
  `examples/tile/` (Triton / cuda-tile Python IR bindings); `legacy/` keeps the 4 earlier
  files

## Layout

```
.
├── cutile-rs-example/             # Rust · tile   — cuTile (stable)
├── cuda-oxide-example/         # Rust · SIMT   — cuda-oxide (nightly)
│   └── 01-vector-add/          # vector add: c[i] = a[i] + b[i]
├── cpp/                        # C++ track
│   ├── common/                 # cuda_utils.h / vector_ops.cu
│   ├── examples/
│   │   ├── simt/               # thread-level CUDA C++
│   │   └── tile/               # deliberately empty: no user-facing C++ tile API
│   └── legacy/                 # 9 earlier examples, read-only
├── python/                     # Python track
│   ├── examples/
│   │   ├── simt/               # Numba / PyCUDA / PyTorch
│   │   └── tile/               # Triton / cuda-tile IR bindings
│   └── legacy/                 # 4 earlier files, read-only
├── docs/                       # Chinese documentation
│   ├── cuda-oxide-book-zh.CN/        # cuda-oxide book (42 pages)
│   ├── cutile-rs-book-zh.CN/            # cuTile book (29 pages)
│   ├── CUDA编程指南-V13.3-zh.CN/      # CUDA Programming Guide (15 pages)
│   ├── atomic.md                     # atomics and memory ordering
│   └── barrier.md                    # mbarrier explained
└── assets/img/
```

## Quick start

```bash
# Rust · tile (uses the repo-root stable toolchain)
cargo run --example 00-add

# Rust · SIMT (needs nightly; the codegen backend is built on first use)
cd cuda-oxide-example/01-vector-add
cargo oxide doctor   # check toolchain, nvcc, GPU, backend cache
cargo oxide run      # compile and run on the GPU

# C++ track (only legacy has content so far)
cd cpp && mkdir -p build && cd build && cmake .. && make
./legacy_transpose_with_shared_mem
```

## Documentation

Chinese translations of three official books live under `docs/`:

| Document | Description |
| --- | --- |
| [cuda-oxide book](./docs/cuda-oxide-book-zh.CN/README.md) | Official cuda-oxide book |
| [cuTile book](./docs/cutile-rs-book-zh.CN/README.md) | Official cuTile Rust book |
| [CUDA Programming Guide v13.3](./docs/CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA C++ Programming Guide |

## Environment (as tested)

| Item | Version |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 (compute capability 8.6) |
| Driver | 615.71.09 (R615 = CUDA 13.4) |
| CUDA Toolkit | 13.3 (`/usr/local/cuda`) |
| Rust (main workspace) | 1.99.0 stable |
| Rust (cuda-oxide) | nightly-2026-10-11 |
| cargo-oxide | 0.2.1, built from `NVIDIA/cuda-rust` at `bc46b1ff` |

## Status

🚧 Work in progress.

| Track | Status |
| --- | --- |
| Rust · tile (cuTile) | ✅ `00-add` and `01-hello-world` run; conventions and template in place |
| Rust · SIMT (cuda-oxide) | ✅ `01-vector-add` runs; codegen backend cache is warm |
| C++ · SIMT | ⬜ Layout and CMake ready, examples to be written |
| C++ · tile | ⚠️ No user-facing entry point; see `cpp/examples/tile/README.md` |
| Python · SIMT / tile | ⬜ Layout ready; environment not set up yet, see `python/README.md` |

## References

- cuda-rust: <https://github.com/NVIDIA/cuda-rust>
- CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide>

