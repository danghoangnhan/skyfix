# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Phases 0–6a + 6b + Phase 3e + Phase 5a + Phase 7a + Phase 8a complete** (as of 2026-05-22). v0.1.0-alpha is **release-ready** pending git commits + crates.io Trusted Publisher setup. **61 tests passing** (54 CPU + 7 GPU).

The GPU now has two end-to-end ops cross-validated against the CPU reference: `CudaGdopSweep2D` (batched 2D GDOP) and `CudaPfRanges2D` (2D particle filter with range-anchor updates, host-side RNG for deterministic cross-checks). Both run on the RTX 5090 in our local environment.

The skyfix-core CPU surface ships the full Bayesian filter trio (EKF, UKF, PF) plus the CRLB analyzer. The first end-to-end CUDA op (`CudaGdopSweep2D`) runs on the RTX 5090. The `skyfix-sim` demo binary runs the whole library end-to-end on a 100-step circular-trajectory scenario. CI now has three workflows (ci, cuda, release), the no-cuda-leak guard, and a Trusted Publishing-OIDC scaffold. **53 tests passing.**

## Tagging v0.1.0-alpha.0 (when ready)

1. Commit the workspace (Phase 0–8a) — blocked on git identity at the time of writing.
2. Configure Trusted Publisher for `skyfix-core` at https://crates.io/me/tokens once the user (`danghoangnhan`) creates the empty crate name. Set repo = `danghoangnhan/skyfix`, workflow file = `release.yml`, environment = (empty), then save.
3. Optionally bump `workspace.package.version` in root `Cargo.toml` from `0.1.0-alpha.0` → `0.1.0-alpha.1` (the release.yml workflow verifies the git tag matches this version).
4. `git tag v0.1.0-alpha.0 && git push origin v0.1.0-alpha.0` → release.yml fires, runs `cargo publish -p skyfix-core` with the OIDC token.

## Running the demo

```sh
cargo run -p skyfix-sim --release
# → prints RMSE + final error, writes truth.csv / estimate.csv / anchors.csv

cargo run -p skyfix-sim --release --example trilateration_to_ekf
# → trilateration cold start → 30 EKF cycles, prints estimate trajectory

cargo run -p skyfix-sim --release --example multi_filter_comparison
# → side-by-side Trilateration / EKF / UKF / PF table + wide-format CSV
```

The multi-filter comparison is the marketing demo — runs all four estimators on identical noisy measurements and prints RMSE / max-error / wall-time per method. Empirically on the 100-step circular scenario: Trilateration 0.32 m RMSE / 2.78 µs total; EKF 0.22 m / 12 µs; UKF 0.22 m / 23 µs; PF (K=512) 0.24 m / 2.23 ms.

The default demo uses 4 anchors at the corners of a 10×10 m room, σ=0.2 m range noise, EKF seeded from a `ToaTrilateration` cold-start fix. Empirically: RMSE ~0.22 m, final error ~0.27 m over 10 s of simulated flight.

CSV output is the demo's plotting boundary — `plotters` was dropped because its TTF font backend pulls in `fontconfig-sys` (system dep). Plot from CSV with gnuplot / matplotlib / Excel; a feature-gated PNG output is a Phase 5b follow-up once font handling is sorted.

Shipped in `skyfix-core`:

| Modality      | Estimator                           | Type                              | Phase |
|---------------|-------------------------------------|-----------------------------------|-------|
| (core)        | `Estimator<T, N>` trait             | single-shot API surface           | 1     |
| (core)        | `numeric::{solve_linear_system, solve_normal_equations, gauss_newton, invert_square, cholesky}` | hand-rolled dense solvers | 1→3 |
| (core)        | `filter::{TransitionModel, ObservationModel}` traits + `IdentityTransition`, `RangeAnchor` built-in models | filter API surface | 3a |
| ToA           | `ToaTrilateration<T, N>`            | closed-form (N+1)                 | 1     |
| ToA           | `ToaNls<T, N>`                      | Gauss-Newton (≥ N)                | 2     |
| TDoA          | `ChanLinear2D<T>`, `ChanLinear3D<T>`| closed-form linear                | 2     |
| TDoA          | `FoyTdoa<T, N>`                     | Taylor-series WLS                 | 2     |
| AoA (2D)      | `StansfieldAoa<T>`                  | bearings-only LSQ                 | 2     |
| RSSI          | `RssiPathLoss<T>`                   | log-distance calib.               | 2     |
| Bayesian      | `Ekf<T, N>`                         | EKF + Joseph form                 | 3a    |
| Bayesian      | `Ukf<T, N, SIGMAS>` (+ `Ukf2D/3D/4D/6D` aliases) | classic UT (α=1, β=2, κ=3−N) | 3b    |

`EstimationError` includes `DidNotConverge` for iterative estimators.

**Test count**: 33 tests passing.

**UKF design notes**:
- `SIGMAS` is a separate const generic that must equal `2N+1`; the type aliases enforce the right pairing.
- Defaults are classic UT (`alpha=1, beta=2, kappa=3-N`) giving `c = N + λ = 3` and well-conditioned weights for all N. The Wan/Van der Merwe scaled variant (`alpha=1e-3`) is numerically unstable for moderate covariances (weights scale as `O(1/α²) ≈ 10⁶`); avoid it.
- Cholesky is implemented in `numeric::cholesky` to spread sigma points.

**Filter convergence regimes (empirically verified)**:
- EKF warm start: ~1e-4 absolute error in 20 iterations × 4 anchors.
- EKF cold start: bias floor at ~5e-2 (Jacobian linearization at the wrong prior).
- UKF warm start: ~2.4e-4 (slightly worse than EKF when prior is good; expected — sigma-point reconstruction has finite-difference precision).
- UKF cold start: < 2e-2 (much better than EKF's bias floor — UKF samples the curvature).
- **Idiomatic pattern**: seed an EKF/UKF with a closed-form estimate from `ToaTrilateration` or `ChanLinear*`. The `ekf_seeded_from_trilateration_recovers_tight_estimate` test demonstrates this lands at machine precision.

CI cross-compiles `skyfix-core` to `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `riscv32imac-unknown-none-elf` with `--no-default-features --features libm`.

## skyfix-cuda

Lives at `crates/skyfix-cuda/`. **Not** in workspace `default-members` — opt in with `cargo build -p skyfix-cuda`. Excluded from the main CI `clippy`/`build`/`test` jobs; needs its own workflow with `Jimver/cuda-toolkit` (added in Phase 6b).

**Dependency stack**:
- `cudarc 0.19.7` with features `std + driver + nvrtc + cuda-12000 + dynamic-linking`.
- `nvrtc` feature is required because cudarc's `CudaContext::load_module()` is gated behind it (the `Ptx` type lives in the nvrtc module even when only loading pre-compiled PTX).
- `dynamic-linking` mode: cudarc links against `libcudart.so` at build time.
- `cuda-12000` matches Ubuntu 24.04's `nvidia-cuda-toolkit` package (CUDA 12.0.140); CUDA 13 driver runs CUDA 12 code via forward compatibility.

**Kernel pipeline**:
1. `kernels/*.cu` written in CUDA C++.
2. `build.rs` invokes `nvcc -ptx --gpu-architecture=compute_70` to produce PTX (virtual arch; driver JIT's to actual SASS at module load).
3. PTX is embedded via `include_str!(env!("PTX_*"))` into the Rust binary.
4. At runtime: `CudaContext::new(0)` → `ctx.load_module(Ptx::from_src(...))` → `module.load_function("name")` → `stream.launch_builder(&func).arg(...)...launch(cfg)`.

**Shipped ops**:
- `CudaGdopSweep2D` — batched 2D GDOP over a grid of (x, y) targets, given a fixed anchor set. Closed-form 2×2 FIM inverse per cell, written in CUDA C++. CPU↔GPU crossover at ~50×50 grid; 6× speedup at 200×200.
- `CudaPfRanges2D` — 2D particle filter with range-to-anchor updates. State + log-weights live on device; host-side RNG for deterministic cross-validation against CPU `Pf`.

Together: **7 GPU tests** + 2 runnable example demos (`multi_filter_comparison` for CPU side, `gdop_grid_benchmark` for CUDA crossover).

**No-CUDA-leak guard**: `cargo tree -p skyfix-core --target thumbv7em-none-eabihf --no-default-features --features libm` produces zero matches for `cuda|cudarc|libloading|nvrtc`. The separate-crate architecture (not feature flag) keeps the embedded path clean.

## Particle filter (Phase 3e)

`skyfix-core::Pf<T, N, K>` — Sequential-Importance-Resampling PF. Static-K design: particle ensemble is a stack-allocated `SMatrix<T, N, K>` so it runs in pure `no_std`. K is a compile-time const generic (typical 64–512 for UAV tracking).

- **RNG abstraction**: `FnMut() -> T` closures for standard-normal samples (predict / from_gaussian) and uniform `[0, 1)` samples (resample). No `rand` dep in the core; user plugs in any RNG. Tests use `rand_chacha::ChaCha8Rng` via dev-dep.
- **Log-weights internally** for numerical stability; `normalized_weights()` uses log-sum-exp before exponentiating.
- **Systematic resampling** — variance-minimal among single-sample schemes. `resample_if_needed(threshold_frac, uniform)` triggers when ESS drops below `threshold_frac × K`.
- **Singular Q gracefully handled**: if `model.noise(dt)` is singular (e.g. zero process noise), the predict step uses the deterministic transition only — `normal` is not called. Documented in the `predict` docstring.

## Phase 3 / 6 deferred (not blockers for v0.1)
- **Phase 3d**: ESKF for IMU integration (pair with `skyfix-imu` driver crate)
- **Phase 3e-alloc**: Dynamic-K PF behind the `alloc` feature for very large ensembles
- **Phase 6b**: Batched PF on CUDA via Thrust-style parallel-scan resampling (needs cuRAND device-side sampling)
- **Phase 6c**: MUSIC/ESPRIT eigendecomposition via cuSOLVER (add the `cusolver` cudarc feature)

The design below remains the authoritative roadmap; update it from the code as implementation lands.

## What skyfix is

A Rust UAV (drone) localization & triangulation library targeting edge devices — both `no_std` microcontrollers (Cortex-M, RISC-V, ESP32) and `std` companion computers (Jetson, Raspberry Pi, x86).

It exists because no current Rust crate covers AoA + TDoA + ToA + RSSI + hybrid estimators + Bayesian filters + CRLB tooling in one place. The closest analogues are language-specific (`UCNLNav` in C#/MATLAB, `pylocus` in Python). The wedge is "all of localization, in Rust, embedded-first."

## Architecture (planned)

Cargo workspace, dual-licensed **MIT OR Apache-2.0**. Crates split by responsibility so embedded targets pull in only the `no_std` core:

- `skyfix-core` — `#![no_std]` algorithmic core. No executor, no allocator by default. AoA (bearings, MUSIC, ESPRIT), ToA (trilateration, NLS), TDoA (Chan closed-form, Foy Taylor-series WLS), RSSI (log-distance), hybrid estimators, EKF/UKF/PF, ESKF, CRLB/FIM/GDOP analyzer, ECEF/ENU/LLA geo conversions.
- `skyfix-sim` — `std`-only desktop simulator, CSV replay, `plotters` visualizations.
- `skyfix-uwb` — DW3000/DW1000 ranging → position. Wraps `dw3000-ng`.
- `skyfix-gnss` — u-blox + NMEA. Wraps `ublox` + `nmea`.
- `skyfix-imu` — BNO055 / ICM-20948 AHRS adapters feeding the EKF.
- `skyfix-mavlink` — MAVLink `POSITION_TARGET_*` integration via `rust-mavlink`.
- `skyfix-ros2` — ROS 2 node wrappers (feature-gated; `r2r` or `safe_drive` backend). **Defer to v0.2** — `rclrs` is missing timers and is single-threaded.
- `skyfix-py` — PyO3 bindings, built with `maturin`.
- `skyfix-c` — `cbindgen`-generated C ABI.
- `skyfix-wasm` — `wasm-bindgen` for in-browser visualization.

Bindings crates (`-py`, `-c`, `-wasm`) live in separate workspace members so `std`/`alloc` don't bleed into the core. Keep PyO3 / cbindgen / wasm-bindgen behind feature flags or in their own crates — never mix `#[pyclass]` into the algorithmic core.

## Key dependency decisions

- **`nalgebra = { version = "0.34", default-features = false, features = ["libm"] }`** — the linear algebra spine. Statically-sized `SMatrix<T, R, C>` only in the core; `DVector`/`DMatrix` gated behind an `alloc` feature.
- **Pattern after `adskalman`** for Kalman/RTS API shape — it's the cleanest `no_std` Kalman idiom in Rust and is production-tested.
- **`dw3000-ng` 1.0+** for UWB (ESP32 + Embassy tested). Legacy `dw1000` only if a target board requires it.
- **`microfft`** for MCU FFT (AoA/MUSIC); `rustfft` + `realfft` on SoC targets.
- **No hard executor dependency** in the core. Provide `embedded-hal-async` adapters so users can plug into Embassy or RTIC v2.
- **Errors**: `thiserror` on `std` builds; plain `enum` errors with optional `defmt::Format` in `no_std`. Never `anyhow` in library code.

## Feature flag strategy (on `skyfix-core`)

- `default = ["f32"]`
- `f32`, `f64` — mutually exclusive numeric type. `f32` is the default because Cortex-M4F/M7/M33 with single-precision FPU runs `f32` 2–10× faster than emulated `f64`. Use `f64` for desktop sim, accuracy benchmarking, and CRLB where numerical conditioning matters.
- `libm` — forwards to `nalgebra/libm` for `no_std` transcendentals.
- `alloc` — enables `DVector` and dynamic-particle PF.
- `std` — enables serde-by-default, plotting hooks, `std::error::Error`.
- `serde`, `defmt`, `log` — instrumentation.

Algorithms are generic over `T: nalgebra::RealField + Copy` so the same code compiles for `f32` and `f64`.

## v0.1.0 MVP scope

Algorithms in `skyfix-core`:
1. ToA trilateration (closed-form 3-anchor) + multilateration (Gauss–Newton NLS for N>3).
2. TDoA: Chan's closed-form 2D/3D, Foy's Taylor-series WLS, robust WLS baseline.
3. AoA / bearings-only triangulation (2D LSQ, Stansfield or ML option).
4. RSSI log-distance path-loss with calibration interface; weighted centroid fallback.
5. Hybrid TDoA + AoA via weighted Gauss–Newton.
6. EKF and UKF generic over state/measurement sizes via const generics.
7. Particle filter — `alloc`-gated for dynamic count; statically-sized for true `no_std`.
8. **CRLB / FIM / GDOP analyzer** — the killer-demo feature for anchor placement planning. Unique among Rust localization crates; ship in v0.1.

Drivers (separate crates):
- One UWB path: `skyfix-uwb` over `dw3000-ng` with DS-TWR → ToA/TDoA.
- One GNSS path: `skyfix-gnss` over `ublox` + `nmea`.
- One IMU path: `skyfix-imu` over `bno055` or `icm20948-async`.

Tooling:
- `skyfix-sim` desktop simulator loading CSV trajectories + anchor layouts.
- Three runnable examples: (a) Linux sim, (b) `thumbv7em-none-eabihf` STM32F4 EKF demo, (c) ESP32 + DW3000 + Embassy ranging-to-position.
- mdBook tutorial reproducing `UCNLNav` / `pylocus` reference vectors.

**Out of scope for v0.1:** visual triangulation (camera+IMU), SLAM, factor-graph optimization (no Ceres/GTSAM equivalent), PX4/ROS 2 integration. These belong in 0.2/0.3.

## Build & test (planned commands)

These are the expected workflows once the workspace exists:

```bash
# Workspace build
cargo build --workspace

# Verify true no_std on the core
cargo build -p skyfix-core --no-default-features --features libm

# Cross-compile to embedded targets
cargo build -p skyfix-core --target thumbv7em-none-eabihf --no-default-features --features libm
cargo build -p skyfix-core --target thumbv8m.main-none-eabihf --no-default-features --features libm
cargo build -p skyfix-core --target riscv32imac-unknown-none-elf --no-default-features --features libm

# ESP32 / Xtensa requires the espup toolchain
cargo +esp build -p skyfix-uwb --target xtensa-esp32-none-elf

# Full workspace tests (std side)
cargo test --workspace

# Single test, e.g. Chan's TDoA closed-form
cargo test -p skyfix-core --test tdoa_chan

# Benchmarks (Criterion)
cargo bench -p skyfix-core

# On-target benchmarks via QEMU or embedded-test
# (per-example; see examples/<name>/README)

# Docs
cargo doc --workspace --no-deps --open
```

CI matrix should cover: `stable` / `beta` / MSRV (locked to **1.87** — `nalgebra` 0.34.2 requires Rust 1.87; the research report's 1.81 floor was based on older microfft/spectrum-analyzer bounds that may no longer bind); targets `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `riscv32imac-unknown-none-elf`, `xtensa-esp32-none-elf` (via `espup`), `wasm32-unknown-unknown`. Add `cargo deny` (licenses) and `cargo audit` (security).

## Cross-validation expectation

Estimator outputs (trilateration, Chan, Foy, EKF) should be cross-checked against `UCNLNav` (C#/MATLAB/Rust, `ucnl/UCNLNav` on GitHub) and `cliansang/positioning-algorithms-for-uwb-matlab` reference vectors. Check fixtures into `crates/skyfix-core/tests/fixtures/`. Property tests with `proptest` for invariants like "noise-free estimates land inside the convex hull of anchors."

## Publishing

- Verify the chosen crate name (working assumption: `skyfix`, falling back to `edgeloc` / `uavloc` if conflicts arise) on crates.io before publishing.
- Use crates.io **Trusted Publishing** from GitHub Actions from day one — no long-lived API tokens.
- Stay on `0.x` until MSRV is documented, the public API has gone two minor releases without breaking change, and there are ≥3 named external users.

## Dev host

Daniel's dev host as of 2026-05-22: Ubuntu 24.04 x86_64, **RTX 5090 (32 GB VRAM, compute capability 12.0)**, NVIDIA driver 580.126.09, CUDA 13.0. Toolkit lives at `/usr/local/nvidia/toolkit` (no `/usr/local/cuda` symlink — set `CUDA_PATH=/usr/local/nvidia/toolkit` in env for builds). `nvcc` not on PATH yet. Deployment target is **Jetson Orin native** (aarch64); the RTX 5090 is the dev/test/CI box for the CUDA path. Cross-compilation from x86 → Jetson is deferred to v0.2 per the Plan agent's risk register.

CUDA runner strategy for v0.1: **build-only on GitHub Actions** (no self-hosted GPU runner). Cross-validation against CPU paths is run manually on this dev box before each release tag.

## Production-adoption context (calibrates honesty in docs / READMEs)

The strong, on-the-record Rust-on-UAV datapoints are **Tweede Golf** (UWB-TDoA realtime localization, plus GAMA Alpha satellite firmware launched 2023-01-03), **Tangram Vision** (perception SDK, Rust-as-default), and **Fusion Engineering** (drone flight controllers on 64-bit ARM Cortex-A, 2019). Anduril publishes a Rust gRPC SDK for Lattice C2 but has **not** publicly confirmed Rust onboard for navigation. PX4 remains C/C++ on NuttX. Don't overclaim "primes use Rust onboard" in marketing copy.
