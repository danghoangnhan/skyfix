# GPU acceleration with NVIDIA CUDA

`skyfix-cuda` is a separate workspace crate sitting on top of [cudarc] 0.19 and the NVIDIA CUDA Toolkit. It ports operations that benefit from massive parallelism — *not* every algorithm in `skyfix-core`. The rule of thumb: if the workload is "one target, one fix," stay on the CPU; if it's "ten thousand candidate target positions" or "thousands of particles," consider the GPU.

This chapter walks through the two shipped operations, the kernel-build pipeline, and the architectural choices that keep `skyfix-core` GPU-agnostic.

## When GPU is worth it

| Workload | Where the work lives |
|---|---|
| Single closed-form fix (trilateration / Chan / Foy) | **CPU** — microseconds, kernel-launch overhead alone is larger |
| Recursive filter step (EKF / UKF / small-K PF) | **CPU** — single-target dimensionality stays tiny |
| GDOP sweep over a heatmap grid (≥ 50×50 cells) | **GPU** (`CudaGdopSweep2D`) — cells fully independent |
| Particle filter at large K (≥ ~1024 particles) | **GPU** (`CudaPfRanges2D`) — per-particle update parallel |
| MUSIC / ESPRIT eigendecomposition | **GPU** (deferred — needs cuSOLVER) |

The current two shipped ops fit the parallelism profile cleanly. Future ops will land on the same threshold logic: if the CPU version is already microseconds, leave it alone; if it's milliseconds or worse on a realistic workload, port it.

## The dependency stack

`Cargo.toml` declares cudarc with a deliberate four-feature set:

```toml
cudarc = { version = "0.19", default-features = false, features = [
    "std",
    "driver",
    "nvrtc",
    "cuda-12000",
    "dynamic-linking",
] }
```

Each one earns its place:

- **`std`** — `skyfix-cuda` is host-side by definition; `no_std` doesn't apply here the way it does to the algorithmic core.
- **`driver`** — the cudarc submodule that wraps the CUDA driver API (`CudaContext`, `CudaStream`, `CudaSlice`, `launch_builder`). This is the actual surface skyfix-cuda calls.
- **`nvrtc`** — required even though we ship pre-compiled PTX, because `CudaContext::load_module()` and the `Ptx` type live behind this feature.
- **`cuda-12000`** — matches Ubuntu 24.04's `nvidia-cuda-toolkit` package (CUDA 12.0.140). The CUDA 13 driver runs CUDA 12 code via forward compatibility, so the same binary works on the dev box's RTX 5090 (driver 580 / CUDA 13).
- **`dynamic-linking`** — cudarc links against `libcudart.so` at build time rather than `dlopen`-ing it. Slightly faster startup; requires the toolkit installed at compile time.

## The kernel-build pipeline

Kernels live in `kernels/*.cu` as plain CUDA C++. The `build.rs` invokes `nvcc -ptx --gpu-architecture=compute_70` and exposes the resulting PTX paths through environment variables (`PTX_GDOP_2D`, `PTX_PF_KERNELS_2D`). At Rust compile time, each PTX file is embedded via `include_str!`:

```rust,no_run
const PTX_GDOP_2D: &str = include_str!(env!("PTX_GDOP_2D"));
```

At runtime, the driver JIT-compiles PTX to actual SASS for whichever GPU is connected:

```text
    skyfix-cuda binary  ──includes──►  PTX (compute_70 virtual arch)
                                          │
                                          ▼
                                       driver JIT
                                          │
                                          ▼
                               SASS for sm_70 (Volta) … sm_120 (Blackwell)
```

`compute_70` is a virtual architecture — JIT compilation targets the actual hardware on first module load. A single binary therefore runs on every NVIDIA GPU from Volta through Blackwell, including the RTX 5090 (sm_120) and Jetson Orin (sm_87) used as the dev / deployment targets.

## `CudaGdopSweep2D`

The first shipped op: batched 2D GDOP over a grid of `(x, y)` candidate target positions, given a fixed anchor set. This is the GPU port of the CPU loop sketched at the end of [chapter 3](./chapter_03_crlb.md).

```rust,no_run
use skyfix_cuda::{Anchor2D, CudaGdopSweep2D};

let anchors = [
    Anchor2D::new( 0.0,  0.0, 0.04),  // x, y, variance (m²)
    Anchor2D::new(10.0,  0.0, 0.04),
    Anchor2D::new( 0.0, 10.0, 0.04),
    Anchor2D::new(10.0, 10.0, 0.04),
];

let sweep = CudaGdopSweep2D::new(&anchors)?;

let xs: Vec<f32> = (0..200).map(|i| i as f32 * 0.05).collect();
let ys: Vec<f32> = (0..200).map(|i| i as f32 * 0.05).collect();

let gdops = sweep.compute_grid(&xs, &ys)?;
// row-major: gdops[iy * xs.len() + ix] = GDOP at (xs[ix], ys[iy])

# Ok::<_, skyfix_cuda::CudaError>(())
```

The kernel (`kernels/gdop_2d.cu`) accumulates the 2×2 FIM with `(a11, a12, a22)` register triple and uses the closed-form 2×2 inverse — `det = a11·a22 − a12²`, `GDOP = √((a11 + a22)/det)`. Singular cells (`det ≤ 0`) are filled with `f32::INFINITY`. Each block processes a row-major slab of cells; `LaunchConfig::for_num_elems` sizes the grid.

### Empirical crossover

`cargo run -p skyfix-cuda --release --example gdop_grid_benchmark` on an RTX 5090:

| Grid    | Cells   | CPU (`CrlbBuilder` loop) | GPU (`CudaGdopSweep2D`) | Speedup |
|---------|---------|--------------------------|-------------------------|---------|
| 10×10   |     100 |   1.88 µs                | 54.83 µs                | 0.0×    |
| 25×25   |     625 |  10.62 µs                | 20.86 µs                | 0.5×    |
| 50×50   |   2 500 |  39.85 µs                | 24.17 µs                | **1.6×** |
| 100×100 |  10 000 | 158.50 µs                | 36.98 µs                | 4.3×    |
| 200×200 |  40 000 | 621.71 µs                | 104.32 µs               | 6.0×    |

Crossover ≈ 2 500 cells (50×50). Below that, kernel-launch overhead dominates; above it, GPU parallelism wins linearly. Plan accordingly: a one-shot 32×32 heatmap should stay CPU-side; a continuous-replan workload that updates an HDOP map every 100 ms at 256×256 should be on the GPU.

All cells agree with the CPU reference (`crates/skyfix-core/tests/crlb.rs` patterns) within 1e-3 relative tolerance. The integration test in `crates/skyfix-cuda/tests/gdop_cross_validation.rs` exercises this every time CI runs the `cuda` workflow.

## `CudaPfRanges2D`

The second shipped op: a 2D particle filter with range-to-anchor updates, with the particle state and log-weights resident on the device across `predict` / `update` calls. Only `mean()` / `effective_sample_size()` / `download()` transfer back to the host.

The pipeline:

```text
    ┌─────────────────────┐     ┌─────────────────────────────┐
    │ host: rng.normal()  │ ──► │ device: pf_predict_2d       │
    │ × 2K samples / step │     │   x_i ← x_i + L · z_i        │
    └─────────────────────┘     └─────────────────────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────┐
                                │ device: pf_update_range_2d  │
                                │   log_w_i −= ½ · innov²/var │
                                └─────────────────────────────┘
                                               │
                                               ▼
                              host: mean() / ESS() / download()
```

Host-side noise sampling is **deliberate**, not a limitation. It lets us reproduce the CPU `Pf` exactly bit-for-bit (modulo f32 ↔ f64): feed both the CPU filter and the CUDA filter the same `ChaCha8Rng`-derived normal samples and the particle ensembles agree on every step. The cross-validation test in `crates/skyfix-cuda/tests/pf_cross_validation.rs` exploits this.

A future revision can swap in device-side cuRAND for higher throughput at large K — listed as Phase 6b in the roadmap, gated on getting cuRAND's `cudarc` feature wired up.

## The architectural rule: separate crate, not feature flag

`skyfix-cuda` is **a separate workspace member**, never a `cuda` feature on `skyfix-core`. Three reasons:

1. **Embedded builds can't tolerate a CUDA feature**. Even a no-op `cuda = []` Cargo feature shows up in dep-graph resolution; if anyone later attaches it to `cudarc`, every Cortex-M / RISC-V build breaks until they explicitly disable it.
2. **`skyfix-core` stays the reference implementation**. The CPU path is what every cross-validation test compares against. Mixing GPU code in would create a "which one is canonical?" question that doesn't have a good answer.
3. **The toolchain dependency is opt-in**. `cargo build --workspace` works on any machine; `cargo build -p skyfix-cuda` adds the CUDA-toolkit requirement only when you actually want GPU code. `skyfix-cuda` is excluded from `default-members` in the workspace `Cargo.toml` to enforce this.

To make sure rule #1 doesn't degrade silently, CI has a **no-cuda-leak guard** that fails the build if anything CUDA-flavored shows up in the embedded dep graph:

```yaml
# .github/workflows/ci.yml (no-cuda-leak job)
- run: |
    tree=$(cargo tree -p skyfix-core --target thumbv7em-none-eabihf \
                       --no-default-features --features libm 2>&1)
    if echo "$tree" | grep -iE 'cuda|cudarc|libloading|nvrtc'; then
      echo "::error::CUDA-related crate leaked into skyfix-core embedded build"
      exit 1
    fi
```

The grep is over-inclusive on purpose: `libloading` is dragged in by any `dlopen`-style FFI shim, and we'd rather a false positive force a discussion than a real leak slip through.

## Workflow split

There are two GitHub Actions workflows:

- **`ci.yml`** — the main matrix (fmt / clippy / build / test) runs on `ubuntu-latest`, which has no CUDA toolkit. Every job there passes `--exclude skyfix-cuda`.
- **`cuda.yml`** — a separate workflow uses `Jimver/cuda-toolkit` to install the toolkit, then builds and tests `skyfix-cuda` specifically. This is build-only for v0.1; cross-validation against the CPU paths is run manually on the dev box before each release tag (no self-hosted GPU runner yet).

## What's deferred

Listed in the CLAUDE.md roadmap as Phase 6 follow-ups:

- **Phase 6b**: batched PF resampling on CUDA via Thrust-style parallel-scan, plus device-side cuRAND for higher throughput. Removes the host-side RNG bottleneck at large K.
- **Phase 6c**: MUSIC / ESPRIT eigendecomposition via cuSOLVER. Needs the `cusolver` cudarc feature added to the dependency set; right now we only need `driver + nvrtc`.

Both will land as additions to `skyfix-cuda`, not as new feature flags on `skyfix-core` — the architectural rule above is stable.

## Next

[Chapter 5](./chapter_05_embedded.md) — back to the other end of the platform spectrum: running the same `skyfix-core` on Cortex-M and RISC-V MCUs, with no allocator and no executor.

[cudarc]: https://crates.io/crates/cudarc
