# skyfix

Rust UAV localization & triangulation for the edge.

A Cargo workspace for UAV position estimation: range / bearing / RSSI measurements → estimator → state. Targets both **`no_std` microcontrollers** (Cortex-M, RISC-V, ESP32) and **`std` companion computers** (Jetson Orin, x86 + dGPU).

**Status: pre-alpha**, not on crates.io yet. The CPU algorithmic surface is feature-complete; the first GPU-accelerated operation runs on RTX 5090; integrations (driver crates, embedded examples) are next.

## At a glance

| Modality | Estimator | Type | `no_std` |
|---|---|---|---|
| ToA | `ToaTrilateration<T, N>` | closed-form, `N+1` anchors | ✓ |
| ToA | `ToaNls<T, N>` | Gauss-Newton, ≥ `N` anchors | ✓ |
| TDoA | `ChanLinear2D<T>`, `ChanLinear3D<T>` | Chan stage-1 closed-form | ✓ |
| TDoA | `FoyTdoa<T, N>` | Taylor-series weighted LSQ | ✓ |
| AoA (2D) | `StansfieldAoa<T>` | bearings-only LSQ | ✓ |
| RSSI | `RssiPathLoss<T>` | log-distance calibration | ✓ |
| Hybrid | `HybridTdoaAoa2D<T>` | TDoA + AoA Gauss-Newton | ✓ |
| Bayesian | `Ekf<T, N>` | EKF + Joseph-form covariance | ✓ |
| Bayesian | `Ukf<T, N, SIGMAS>` | classic Unscented Transform | ✓ |
| Bayesian | `Pf<T, N, K>` | SIR + systematic resampling | ✓ |
| Bayesian | `Eskf<T, N>` + `Imu2DStrapdown` | Error-State KF for IMU + range fusion | ✓ |
| Analysis | `CrlbBuilder` + `CrlbAnalysis` | Fisher Information / CRLB / GDOP | ✓ |
| GPU | `CudaGdopSweep2D` | batched 2D GDOP via NVIDIA CUDA | (std) |

68 tests passing (61 CPU + 7 GPU) across 14 test binaries. Demo binary (`cargo run -p skyfix-sim`) tracks a moving 2D target with RMSE 0.22 m at σ = 0.2 m range noise.

## Quick start

```sh
# Run the EKF tracking demo
cargo run -p skyfix-sim --release

# Run the trilateration → EKF bootstrap example
cargo run -p skyfix-sim --release --example trilateration_to_ekf

# Cross-compile the core to a Cortex-M target
cargo build -p skyfix-core \
    --no-default-features --features libm \
    --target thumbv7em-none-eabihf

# Build the CUDA acceleration crate (requires NVIDIA CUDA Toolkit ≥ 12)
cargo build -p skyfix-cuda
```

## Workspace layout

```
crates/
├── skyfix-core/       no_std-by-default algorithmic core
├── skyfix-sim/        desktop demo binary + examples
├── skyfix-cuda/       NVIDIA CUDA acceleration (cudarc 0.19 + nvcc kernels)
├── skyfix-uwb/        UWB ranging adapter (DW3000-compatible, no_std)
└── skyfix-fixtures/   shared reference test vectors
```

See [CLAUDE.md](./CLAUDE.md) for the full architecture rationale, the v0.1 roadmap, and the gotchas-worth-remembering log (nalgebra `ToTypenum` constraint, UKF Wan/Van der Merwe numerical pitfalls, cudarc feature soup, etc.).

## Architecture decisions

- **No executor dependency in the core.** Use Embassy, RTIC, Tokio, or none — `skyfix-core` is executor-agnostic.
- **`f32` and `f64` both supported** via generic `T: nalgebra::RealField + Copy`.
- **`nalgebra` 0.34** for statically-sized matrices; `libm` for `no_std` transcendentals.
- **Hand-rolled solvers** (`numeric::{solve_linear_system, solve_normal_equations, gauss_newton, invert_square, cholesky}`) — nalgebra's `.lu()` / `.qr()` etc. require `Const<N>: ToTypenum` and don't compile over generic const `N`.
- **CUDA in a separate crate**, never a feature flag on `skyfix-core`. CI enforces this via a `cargo tree` grep guard on the Cortex-M dep graph.
- **MSRV 1.88** — bumped from 1.87 because cudarc 0.19 pulls in `libloading` 0.9 which requires 1.88.

## Filter convergence (empirically observed)

| Scenario | Estimator | Behavior |
|---|---|---|
| Warm-start tracking | EKF | ~1e-4 m error in 20 iter × 4 anchors |
| Cold-start tracking | EKF | ~5 cm bias floor (Jacobian linearization at the wrong prior) |
| Cold-start tracking | UKF | < 2 cm (sigma points sample the curvature) |
| Closed-form seed + EKF refinement | both | machine precision |

The idiomatic skyfix pattern is **closed-form bootstrap → recursive filter**: use `ToaTrilateration` (or `ChanLinear2D`) for the first fix, then hand off to `Ekf` / `Ukf` / `Pf` for tracking.

## Production-adoption context

Rust has graduated from "hobbyist" to "credible production embedded." The strongest on-the-record UAV / robotics Rust datapoints come from specialist consultancies, not the headline primes:

- **Tweede Golf** — Rust UWB-TDoA realtime localization (20–50 cm indoor accuracy), plus the GAMA Alpha satellite firmware (SpaceX Falcon 9, 2023-01-03).
- **Tangram Vision** — Rust-as-default for full-stack robotics / perception SDK.
- **Fusion Engineering** — drone flight controllers in Rust on 64-bit ARM Cortex-A (per founder Mara Bos, HN 2019-04-06).
- **Tier IV / `safe_drive`** — formally-verified ROS 2 Rust binding used in Autoware production AV stacks.

PX4 remains C/C++ on NuttX. Anduril publishes a Rust gRPC SDK for the Lattice C2 platform but has not publicly confirmed Rust onboard for navigation. Don't read more into "primes use Rust onboard" than the public record supports.

## License

Dual-licensed under either of:

- MIT License ([LICENSE-MIT](./LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](./LICENSE-APACHE))

at your option. Contributions are accepted under the same dual license unless explicitly stated otherwise.

## Contributing

Run the full gate set locally before opening a PR:

```sh
cargo fmt --all --check
cargo clippy --workspace --exclude skyfix-cuda --all-targets -- -Dwarnings
cargo test --workspace --exclude skyfix-cuda
cargo build -p skyfix-core --no-default-features --features libm --target thumbv7em-none-eabihf
# (if you have a CUDA toolkit installed)
cargo build -p skyfix-cuda
cargo test -p skyfix-cuda     # requires a GPU
```

CI runs all of these plus a `no-cuda-leak` guard that verifies the Cortex-M dep graph stays free of any `cuda*` / `cudarc` / `libloading` / `nvrtc` references.
