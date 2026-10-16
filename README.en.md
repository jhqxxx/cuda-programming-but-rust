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
| Typical entry points | cuTile (Rust / Python / C++), Triton | CUDA C++ / cuda-oxide / Numba / PyCUDA |

These are **two different abstractions, not two syntaxes for the same thing** — so examples
are only ever compared against implementations in the same model.

## Three languages × two models

| | Rust | C++ | Python |
| --- | --- | --- | --- |
| **tile** | ✅ [`cutile-rs-example/`](./cutile-rs-example/) — cuTile Rust | ✅ [`cpp/examples/tile/`](./cpp/examples/tile/) — CUDA Tile C++ | ⬜ [`python/examples/tile/`](./python/examples/tile/) — cuTile Python (env verified, examples pending) |
| **SIMT** | ✅ [`cuda-oxide-example/`](./cuda-oxide-example/) — cuda-oxide | ⬜ [`cpp/examples/simt/`](./cpp/examples/simt/) — CUDA C++ | ⬜ [`python/examples/simt/`](./python/examples/simt/) — Numba / PyCUDA / PyTorch |

The three tile entries **share a single compiler backend** (CUDA Tile IR → `tileiras`),
so they are a genuine same-model cross-language comparison with a 1:1 API mapping. The C++
path has been available since **CUDA Toolkit 13.3**: `nvcc --enable-tile -std=c++20`.

> This cell previously claimed "no user-facing C++ entry point", which was **wrong** — it
> missed the `crt/cuda_tile.h` that `cuda_tile.h` forwards to (4064 lines / 177 KB) and
> never checked `nvcc --help`. See
> [`cpp/examples/tile/README.md`](./cpp/examples/tile/README.md) for the correction log.

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

- **CUDA C++** — `cpp/`, built with CMake. `examples/simt/` holds SIMT examples (to be
  written); `examples/tile/` holds CUDA Tile C++ examples (`00-add` runs; CMake adds
  `--enable-tile` and `-std=c++20` to those targets automatically); `legacy/` keeps the 9
  earlier examples: device attributes, atomics, constant memory, explicit memory, unified
  memory, shared memory, streams and events, matrix transpose (naive and shared-memory
  optimized)
- **Python** — `python/`, managed with uv. `examples/simt/` (Numba / PyCUDA / PyTorch
  extension) and `examples/tile/` (cuTile Python); `legacy/` keeps the 4 earlier files

## Layout

```
.
├── cutile-rs-example/          # Rust · tile   — cuTile (stable)
├── cuda-oxide-example/         # Rust · SIMT   — cuda-oxide (nightly)
│   └── 01-vector-add/          # vector add: c[i] = a[i] + b[i]
├── cpp/                        # C++ track
│   ├── common/                 # cuda_utils.h / vector_ops.cu
│   ├── examples/
│   │   ├── simt/               # thread-level CUDA C++
│   │   └── tile/               # block-level CUDA Tile C++ (--enable-tile + C++20)
│   └── legacy/                 # 9 earlier examples, read-only
├── python/                     # Python track
│   ├── examples/
│   │   ├── simt/               # Numba / PyCUDA / PyTorch
│   │   └── tile/               # cuTile Python (cuda-tile)
│   └── legacy/                 # 4 earlier files, read-only
├── docs/                       # Chinese documentation
│   ├── cuda-oxide-book-zh.CN/         # cuda-oxide book (42 pages)
│   ├── cutile-rs-book-zh.CN/          # cuTile Rust book (29 pages)
│   ├── cutile-python-book-zh.CN/      # cuTile Python book (13 pages)
│   ├── CUDA编程指南-V13.3-zh.CN/      # CUDA Programming Guide (15 pages)
│   ├── atomic.md                      # atomics and memory ordering
│   └── barrier.md                     # mbarrier explained
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

# C++ track
cd cpp && mkdir -p build && cd build && cmake .. && make
./tile_01-hello-world                # tile model (CMake adds --enable-tile + C++20)
./legacy_transpose_with_shared_mem   # SIMT, earlier examples
```

## Documentation

Chinese translations of four official books live under `docs/`:

| Document | Description |
| --- | --- |
| [cuda-oxide book](./docs/cuda-oxide-book-zh.CN/README.md) | Official cuda-oxide book |
| [cuTile Rust book](./docs/cutile-rs-book-zh.CN/README.md) | Official cuTile Rust book |
| [cuTile Python book](./docs/cutile-python-book-zh.CN/README.md) | Official cuTile Python (`cuda.tile`) documentation |
| [CUDA Programming Guide v13.3](./docs/CUDA编程指南-V13.3-zh.CN/README.md) | NVIDIA CUDA C++ Programming Guide |

`docs/atomic.md` and `docs/barrier.md` are two topic notes written for this repo.

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
| C++ · tile | ✅ `01-hello-world` runs (CUDA Tile C++, `--enable-tile` + C++20) |
| Python · SIMT | ⬜ uv environment ready; `examples/` to be written, all 3 `legacy/` examples pass |
| Python · tile | ⬜ cuTile Python verified in the environment; `examples/tile/` to be written |

## References

- cuda-rust: <https://github.com/NVIDIA/cuda-rust>
- CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide>

