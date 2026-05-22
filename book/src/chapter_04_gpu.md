# GPU acceleration with NVIDIA CUDA

*(Coming next turn.)*

This chapter will cover `skyfix-cuda` — when GPU acceleration of localization workloads is worth it, and when it isn't. Topics:

- Building skyfix-cuda (CUDA Toolkit setup, the `cudarc` feature soup, why `dynamic-linking + nvrtc + std + cuda-12000`)
- The kernel build pipeline (`.cu` → `nvcc -ptx --gpu-architecture=compute_70` → embedded via `include_str!` → JIT'd by the driver at module load)
- `CudaGdopSweep2D` — empirical CPU↔GPU crossover at ~50×50 grid on RTX 5090
- `CudaPfRanges2D` — particle filter on GPU with host-side RNG for cross-validation
- Architecture rationale: why skyfix-cuda is a *separate crate*, not a feature flag on skyfix-core (and the `cargo tree` CI guard that enforces it)

Until this chapter lands, see `crates/skyfix-cuda/README.md` for benchmarks and the layered architecture overview.
