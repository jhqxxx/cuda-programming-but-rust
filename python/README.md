# Python 示例

Python 侧和 Rust 侧一样，有**两套并行的编程模型**，所以 `examples/` 下面按模型分目录。

## 目录

```
python/
├── pyproject.toml     # 依赖声明（uv 项目根）
├── uv.lock            # 锁文件，已提交 → 克隆后能复现完全一致的环境
├── .python-version    # 3.12
├── examples/
│   ├── simt/          # 线程级：Numba / PyCUDA / PyTorch 扩展
│   └── tile/          # 块级：cuTile Python（cuda-tile）
└── legacy/            # 早期示例，保留原样备查
    ├── use_numba.py
    ├── use_pycuda.py
    ├── use_pytorch.py
    └── vecadd.cu
```

| 目录 | 模型 | 编程单位 | 库 |
| --- | --- | --- | --- |
| `examples/simt/` | SIMT | 一个线程 | PyTorch C++ 扩展（`torch.utils.cpp_extension`） |
| `examples/tile/` | tile | 一个块（tile） | cuTile Python（PyPI 包名 `cuda-tile`） |
| `legacy/` | SIMT | 一个线程 | 早期写的三条路线对照：Numba / PyCUDA / PyTorch 扩展 |

两个目录各自的取舍见 `examples/simt/README.md` 和 `examples/tile/README.md`。

## 环境（uv 管理）

本目录是一个 [uv](https://docs.astral.sh/uv/) 项目：依赖写在 `pyproject.toml`，
锁文件 `uv.lock` 已提交 —— 克隆仓库后一条命令就能复现环境。

```bash
cd python
uv sync          # 建 .venv 并装入基础依赖
```

Python 版本由 `.python-version` 指定（3.12）；本机没有时 uv 会自动下载，
**不需要手动装 Python**。

### 依赖分层

基础依赖放 `examples/` 新例子要用的东西，外加两条线共用的 **cupy**（设备数组 + 主机/设备
拷贝，`legacy/use_numba.py` 也靠它）；`legacy/` 里各自独有的库才是 extra，按需装：

| 命令 | 装上什么 | 能跑哪些 |
| --- | --- | --- |
| `uv sync` | cuda-tile、cupy、numpy、torch、triton | **`examples/` 下的新例子**（SIMT = PyTorch 扩展，tile = cuTile Python） |
| `uv sync --extra numba` | + numba | 只想试 Numba 那一路 |
| `uv sync --extra pycuda` | + pycuda | `legacy/use_pycuda.py` |
| `uv sync --extra legacy` | 上面两个 | `legacy/` 全部三份 ← **一般就用这个** |

### 运行

```bash
uv run python examples/tile/00-add/add.py    # 不用手动 activate
```

`uv run` 会自动使用本目录的 `.venv`。想要交互式环境就 `uv run python`，
或 `source .venv/bin/activate`。

### 不用 uv 的话

仓库里不放 `requirements.txt` —— 它必然和 `uv.lock` 不同步。需要时自己生成：

```bash
uv export --format requirements-txt --no-hashes > requirements.txt
```

### 本机实测

| 项 | 版本 |
| --- | --- |
| uv | 0.8.18 |
| Python | 3.12.9（uv 管理） |
| 驱动 | 615.71.09（R615 分支 = CUDA 13.4） |
| CUDA Toolkit | 13.3（`/usr/local/cuda`） |
| GPU | NVIDIA GeForce RTX 3080（sm_86） |
| numba / pycuda / triton | 0.68.0 / 2026.1 / 3.8.0 |
| numpy | 2.5.4 |
| torch | 2.14.1+cu130 |
| cupy-cuda13x | 14.2.0 |

### 驱动版本要求：R610 及以上（踩过坑）

**症状**：numba 报

```
LinkerError: [222] Call to cuLinkAddData results in CUDA_ERROR_UNSUPPORTED_PTX_VERSION
ptxas application ptx input, line 9; fatal: Unsupported .version 9.3; current version is '9.2'
```

**原因**：这是**驱动的 PTX 版本上限**不够。系统工具链是 CUDA 13.3，它产出的 PTX 是
**9.3**；驱动如果属于更早的分支，JIT 链接器只认到 9.2，就报这个错。

驱动分支与 CUDA 的对应（NVIDIA 官方发行说明）：

| CUDA | 驱动分支 |
| --- | --- |
| 13.4 | R615 |
| 13.3 | R610 |
| 13.2 | R595 |
| 13.1 | R590 |

判断方法：`nvidia-smi` 头部的 `CUDA UMD Version` 就是驱动支持的 CUDA 上限，
**PTX 版本与它同号**。本机原来是 `595.99.02`（R595 = CUDA 13.2）配 13.3 工具链，
所以必然冲突。

**解决**：升驱动。本机已升到 `615.71.09`（R615 = CUDA 13.4），numba 直接可用。

```bash
cd python
uv sync --extra numba
uv run python -c "from numba.cuda.cuda_paths import get_cuda_paths as g; print(g()['nvvm'].info)"
# → /usr/local/cuda/nvvm/lib64/libnvvm.so.4.0.0   （自动用系统那份，无需配置）
uv run python legacy/use_numba.py     # → Test succeeded
```

> 网上流传过一套绕法：钉 `nvidia-nvvm==13.2.86`，再用 `CUDA_HOME` 把 numba 指到旧的
> libnvvm（还要补 `nvvm/lib64` 的目录布局，因为 numba 只会在 conda 环境里自动发现 pip
> 装的 `nvidia-nvvm`）。在 R595 驱动下它确实能让 numba 跑起来，本仓库一度就这么干；
> 升到 R615 后整套都删了。**所以遇到这个错先看驱动版本，别急着折腾 libnvvm。**
> 另外 `NUMBA_CUDA_LOW_PTX_VERSION=1` 这个环境变量并不存在，实测无效。

### PyCUDA 需要显式初始化上下文

`legacy/use_pycuda.py` 曾经报 `UnicodeDecodeError`，换驱动后又变成
`cuModuleLoadDataEx failed: initialization error` —— **两个报错都是误导性的**。

真正原因：**CUDA 13 起 `cuModuleLoadData` 不再自动创建上下文**。`SourceModule` 一构造
就要加载 cubin，所以当前线程必须先有 context；老版本 CUDA 会自动兜底，新版本不会，
于是失败 —— 而 pycuda 把失败原因报成了「解码错误」。

**修法**：加一行 `import pycuda.autoinit`（已改好）。这也解释了为什么 `pycuda.driver`
单独用一直正常 —— 它不加载 module。

排查时验证过：`compile()` 返回的是完好的 ELF cubin（`b'\x7fELF...'`，2856 字节），
所以编译环节没问题，问题只在加载时缺 context。

### 跑 legacy/ 里的示例

```bash
cd python
uv sync --extra legacy                   # 补上 numba / pycuda（cupy 已在基础依赖里）
uv run python legacy/use_numba.py        # → Test succeeded
uv run python legacy/use_pycuda.py       # → Test succeeded
uv run python legacy/use_pytorch.py      # → Test succeeded
```

三份都能从任意目录直接运行（`use_pytorch.py` 已改成用 `__file__` 定位 `vecadd.cu`）。

## 与 Rust / C++ 侧的关系

只和**同模型**的实现对比：

| 模型 | Rust | C++ | Python |
| --- | --- | --- | --- |
| SIMT | `cuda-oxide-example/01-vector-add/` | `cpp/examples/simt/` | `examples/simt/` |
| tile | `cutile-rs-example/00-add.rs` | `cpp/examples/tile/01-hello-world.cu` | `examples/tile/01-hello-world.py`（cuTile Python） |

三语言的 tile 实现共享同一个编译器后端（CUDA Tile IR → `tileiras`），
所以是同模型的 1:1 对照，不是「相近的东西」。

跨模型对照会误导 —— tile 和 SIMT 是两种不同的抽象，不是同一写法的两种语法。

## 状态

✅ `examples/tile/01-hello-world.py` 已跑通（`uv run python examples/tile/01-hello-world.py`）；
⬜ `examples/simt/` 待写。`legacy/` 三份参考实现全部跑通（`use_pycuda.py` 补了一行
上下文初始化，见上）。
