# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 7a (skyfix-uwb adapter)

- New `skyfix-uwb` workspace crate — hardware-agnostic data layer for UWB ranging. `#![no_std]` by default; opt into `std`/`libm` features for desktop / embedded targets respectively.
- `UwbRange<N>` — anchor EUI-64 + position + range_m + optional timestamp. Designed to be the natural output of a DW3000 ranging exchange.
- `UwbRange::to_toa()` — convert to `skyfix_core::ToaMeasurement` for trilateration / NLS / Bayesian filters.
- `pair_to_tdoa(reference, other)` and `ranges_to_tdoa_batch(&ranges, &mut output)` — TDoA conversion helpers. The batch variant takes a caller-supplied output buffer for `no_std` usage (stack array or `heapless::Vec`).
- 5 integration tests demonstrating typical pipelines: ToA → Trilateration, TDoA batch → ChanLinear2D, sign-correctness of pair_to_tdoa, 3D timestamp roundtrip.
- Phase 7b (actual dw3000-ng driver integration with SPI/IRQ/calibration) deferred until the project has an embedded target wired up.

### Phase 6 polish (CPU↔GPU GDOP benchmark)

- `cargo run -p skyfix-cuda --release --example gdop_grid_benchmark` — measures CPU `CrlbBuilder` loop vs GPU `CudaGdopSweep2D` batched kernel across grid sizes from 10×10 to 200×200, cross-validates every cell within 1e-3 relative tolerance, prints a speedup table.
- Empirical crossover on the RTX 5090: ~50×50 grid (2 500 cells). At 200×200 (40 000 cells) the GPU is 6× faster; further linear scaling expected at larger grids.

### Phase 6b (CUDA particle filter)

- `skyfix-cuda::CudaPfRanges2D` — GPU-resident 2D particle filter with range-to-anchor updates. State + log-weights live on device across calls; only `mean()` / `effective_sample_size()` / `download()` transfer back to host.
- Two new CUDA kernels in `kernels/pf_kernels_2d.cu`:
  - `pf_predict_2d` — applies `x_i ← x_i + L · z_i` per particle, with host-supplied standard-normal samples and Cholesky factor of `Q`.
  - `pf_update_range_2d` — log-likelihood update `log_w_i −= ½ · (z − ‖x_i − anchor‖)² / variance`.
- Host-side RNG (caller-supplied noise samples) — deliberate choice that makes deterministic cross-validation against the CPU `Pf` trivial. Future revision can swap in device-side cuRAND for throughput.
- 3 GPU tests in `tests/pf_cross_validation.rs`: mean matches CPU to f32 precision after zero-noise steps, ESS matches within 5% after a skewed update, input-validation rejects bad sizes.
- `CudaError` gained `SizeMismatch { expected, got }` and `NoParticles` variants.
- `skyfix-cuda` split into modules: `lib.rs` (re-exports + error + GDOP), `pf.rs` (PF). Existing `CudaGdopSweep2D` API unchanged.

### Phase 6a (skyfix-cuda)

- New `skyfix-cuda` workspace crate. Excluded from `default-members` so `cargo build` doesn't require the CUDA toolkit.
- `CudaGdopSweep2D` — batched 2D GDOP analyzer running on the GPU. Wraps a hand-written CUDA C++ kernel (`kernels/gdop_2d.cu`) compiled to PTX by `build.rs` (`nvcc -ptx --gpu-architecture=compute_70`).
- `cudarc 0.19.7` with `std + driver + nvrtc + cuda-12000 + dynamic-linking` features. Targets Ubuntu 24.04's `nvidia-cuda-toolkit` (CUDA 12.0.140); CUDA 13 driver runs it via forward compatibility.
- 4 GPU cross-validation tests, all passing on the RTX 5090.
- `cargo tree` verifies `cudarc` does not leak into `skyfix-core` embedded dep graphs.
- MSRV bumped 1.87 → 1.88 (`libloading 0.9.0`, pulled in by cudarc, requires 1.88).
- CI workflow updated to `--exclude skyfix-cuda` from `clippy` / `build` / `test` jobs (a separate `cuda.yml` workflow with `Jimver/cuda-toolkit` lands in Phase 8a release polish).

### Phase 5b-1 (multi-filter comparison demo)

- New example `cargo run -p skyfix-sim --release --example multi_filter_comparison` — runs Trilateration / EKF / UKF / PF (K=512) on a single shared scenario (identical truth + identical noisy measurements), prints RMSE / max-error / wall-time table, writes wide-format `multi_comparison.csv` for plotting.
- The showcase demo: makes the practical accuracy/cost trade-offs across all four estimation strategies legible at a glance. Empirically on a 100-step circular scenario at σ=0.2 m: Trilateration 0.32 m RMSE / 2.78 µs; EKF 0.22 m / 12 µs; UKF 0.22 m / 23 µs; PF (K=512) 0.24 m / 2.23 ms.

### Phase 5a (skyfix-sim binary + examples)

- `skyfix-sim` crate now has a runnable binary (`cargo run -p skyfix-sim`) that simulates 100 steps of circular-trajectory tracking with 4 corner anchors and σ=0.2 m range noise. Writes CSV output (truth, estimate, anchors) for downstream plotting.
- Library primitives in `skyfix-sim::lib`: `Anchor2D`, `Trajectory2D` trait + `CircularTrajectory` impl, `ToASimulator2D` (noisy measurement generator), `StepRecord` + `rmse` helpers.
- Example `cargo run -p skyfix-sim --example trilateration_to_ekf` demonstrates the idiomatic trilateration → EKF bootstrap pattern (closed-form fix as initial state, EKF for refinement).
- `plotters` deliberately omitted (system `fontconfig` dep friction). CSV output is the plotting boundary; users plot with their preferred tool.

### Phase 4 (CRLB / FIM / GDOP analyzer)

- `skyfix-core::CrlbBuilder<T, N>` — Fisher-Information accumulator with `.add_toa()`, `.add_tdoa()`, and (2D-only) `.add_aoa()` methods.
- `skyfix-core::CrlbAnalysis<T, N>` — frozen CRLB analysis with `.covariance()` (=FIM⁻¹), `.gdop()`, `.hdop()` (2D + 3D), `.vdop()` (3D).
- Integration tests in `tests/crlb.rs` (7 tests).

### Phase 3e (particle filter)

- `skyfix-core::Pf<T, N, K>` — Sequential-Importance-Resampling particle filter. Static-K (compile-time const generic), `no_std`-compatible (stack-allocated `SMatrix<T, N, K>` ensemble).
- RNG abstraction via `FnMut() -> T` closures: one for standard-normal samples, one for uniform `[0, 1)` samples. No `rand` dep in the core crate.
- Log-weights with log-sum-exp normalization for numerical stability.
- Systematic resampling with `resample_if_needed(threshold_frac, uniform)` triggered by effective sample size.
- Graceful handling of singular process-noise covariance (e.g. zero noise → deterministic transition).
- Integration tests use `rand_chacha::ChaCha8Rng` as a deterministic seedable RNG (dev-dep only).

### Phase 3c (hybrid TDoA + AoA)

- `skyfix-core::HybridTdoaAoa2D` — 2D Gauss-Newton solver fusing TDoA range-difference and AoA bearing measurements in a single iteration. Uses the proper bearing-residual NLS form (Jacobian `[-Δy/r², Δx/r²]`) with angle wrap to `[-π, π]`.
- Integration tests in `tests/hybrid.rs` (4 tests).

### Phase 3a-b (Bayesian filter trio)

- `skyfix-core::filter::{TransitionModel, ObservationModel}` traits — the model abstraction for recursive Bayesian filters.
- `skyfix-core::filter::{IdentityTransition, RangeAnchor}` — built-in models for the common cases (stationary state, range-to-anchor measurement).
- `skyfix-core::Ekf<T, N>` — Extended Kalman Filter with Joseph-form covariance update. Generic over state dim `N` and measurement dim `M` (via `update` method).
- `skyfix-core::Ukf<T, N, SIGMAS>` — Unscented Kalman Filter with classic-UT defaults. Type aliases `Ukf1D` / `Ukf2D` / `Ukf3D` / `Ukf4D` / `Ukf6D` enforce `SIGMAS = 2N+1`.
- `numeric::invert_square` — square matrix inverse via per-column Gauss elimination.
- `numeric::cholesky` — `A = L L^T` decomposition used by UKF sigma-point generation.
- Integration tests: `tests/ekf.rs` (6 tests including the idiomatic trilateration-seeded workflow), `tests/ukf.rs` (4 tests including cold-start outperforming EKF bias floor).

### Phase 2 (TDoA / AoA / RSSI / ToaNls + numeric helpers)

- `skyfix-core::numeric` module exposing `solve_linear_system`, `solve_normal_equations`, and `gauss_newton` helpers (promoted from Phase 1's `toa.rs` so future estimators don't duplicate dense-solver code).
- `skyfix-core::ToaNls<T, N>` — overdetermined Gauss-Newton ToA estimator. `Estimator::estimate` defaults to the anchor centroid as initial guess; `.iterate(initial, …)` accepts an explicit initial (e.g. from `ToaTrilateration`).
- `skyfix-core::ChanLinear2D<T>` and `ChanLinear3D<T>` — Chan's closed-form linear-stage TDoA solver. Chan stage 2 (ML refinement) deferred pending a measurement-covariance API.
- `skyfix-core::FoyTdoa<T, N>` — Foy's Taylor-series WLS for TDoA, generic over `N`. Iterative; pair with Chan for the natural Chan-into-Foy pipeline.
- `skyfix-core::StansfieldAoa<T>` — 2D bearings-only triangulation via Stansfield's LSQ. 3D AoA (azimuth + elevation) deferred.
- `skyfix-core::RssiPathLoss<T>` — log-distance path-loss calibration with `range_from_rssi` / `rssi_at_range`. Feeds estimated ranges into the ToA estimators.
- New measurement types: `TdoaMeasurement<T, N>`, `AoaBearing<T>`, `RssiSample<T, N>`.
- Integration test files: `tdoa.rs`, `toa_nls.rs`, `aoa.rs`, `rssi.rs`.

### Phase 1 (ToA trilateration + estimator surface)

- `skyfix-core::Estimator` trait — single-shot position estimator API generic over `RealField` and dimension.
- `skyfix-core::ToaMeasurement<T, N>` — anchor + range measurement type.
- `skyfix-core::ToaTrilateration<T, N>` — closed-form `N`-dimensional ToA trilateration for `≥ N+1` anchors via linearized LU solve.
- `skyfix-core::EstimationError` — non-exhaustive error enum (`InsufficientMeasurements`, `SingularSystem`, `NonFinite`, `DidNotConverge`) with `core::error::Error`.
- CI: added `embedded` matrix job that cross-compiles `skyfix-core` to `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, and `riscv32imac-unknown-none-elf` with `--no-default-features --features libm`.

### Added

- Workspace skeleton with `skyfix-core`, `skyfix-sim`, `skyfix-fixtures` crates.
- MSRV pinned to 1.87 via `rust-toolchain.toml` (nalgebra 0.34.2 hard floor).
- CI: fmt, clippy, build, test, `cargo deny`.
- Dual MIT / Apache-2.0 license.
