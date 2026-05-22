# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
