# skyfix-cuda

NVIDIA CUDA acceleration for skyfix. Sits on top of [cudarc] 0.19 and the NVIDIA CUDA Toolkit.

## What's implemented

| Operation                                  | Phase | Wins vs CPU                                          |
|--------------------------------------------|-------|------------------------------------------------------|
| `CudaGdopSweep2D::compute_grid`            | 6a    | Embarrassingly parallel; grid cells fully independent |
| `CudaPfRanges2D::predict` / `::update`     | 6b    | Per-particle parallelism; biggest win at large K     |

Next: device-side cuRAND for higher PF throughput, parallel-scan resampling, MUSIC/ESPRIT eigendecomposition via cuSOLVER.

## Empirical wall times (RTX 5090, CUDA 12.0)

`cargo run -p skyfix-cuda --release --example gdop_grid_benchmark` produces:

| Grid    | Cells  | CPU (loop over `CrlbBuilder`) | GPU (`CudaGdopSweep2D`) | Speedup |
|---------|--------|-------------------------------|-------------------------|---------|
| 10×10   |    100 |  1.88 µs                      | 54.83 µs                | 0.0×    |
| 25×25   |    625 | 10.62 µs                      | 20.86 µs                | 0.5×    |
| 50×50   |  2 500 | 39.85 µs                      | 24.17 µs                | **1.6×** |
| 100×100 | 10 000 | 158.50 µs                     | 36.98 µs                | 4.3×    |
| 200×200 | 40 000 | 621.71 µs                     | 104.32 µs               | 6.0×    |

Crossover is around 50×50 = 2 500 cells. Below that, kernel-launch overhead dominates; above that, GPU parallelism wins linearly with grid size. All cells agree with the CPU `CrlbBuilder` within 1e-3 relative tolerance.

## NVIDIA toolkit usage

This crate goes through the official NVIDIA Toolkit at every layer:

- **`nvcc`** (Toolkit compiler) builds `.cu` kernels under `kernels/` into PTX at build time.
- **`libcudart` / `libcuda`** (Toolkit runtime) is what `cudarc` calls at runtime.
- **`cudarc`** is purely the Rust↔C FFI shim — the actual compute happens in the Toolkit components.

## Building

Requires the CUDA Toolkit. On Ubuntu 24.04:

```sh
sudo apt install nvidia-cuda-toolkit
```

(Ships CUDA 12.0; the CUDA 13 driver runs it via forward compatibility.) Verify with `nvcc --version`.

For the official NVIDIA-provided toolkit (matches driver version exactly), add the CUDA apt repo:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-13-0
```

Then update `Cargo.toml` to use the `cuda-13000` cudarc feature.

## Architecture target

PTX is compiled for `compute_70` (Volta) — virtual architecture, JIT'd by the driver to the actual SASS at module load. Single binary runs from Volta through Blackwell (sm_70 … sm_120, including the RTX 5090).

## Running

```sh
cargo build -p skyfix-cuda
cargo test -p skyfix-cuda
```

`skyfix-cuda` is **not** in workspace `default-members`, so plain `cargo build` doesn't require the CUDA Toolkit. Opt in explicitly with `-p`.

[cudarc]: https://crates.io/crates/cudarc
