# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Phase 0 skeleton in place** (as of 2026-05-22). Workspace exists with three crate shells: `skyfix-core`, `skyfix-sim`, `skyfix-fixtures`. No estimators implemented yet — Phase 1 is next. The design below remains the authoritative roadmap; update it from the code as implementation lands.

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
