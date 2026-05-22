# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
