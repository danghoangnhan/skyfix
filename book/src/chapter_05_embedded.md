# Embedded targets

`skyfix-core` is `#![no_std]` by default. The same crate that powers the desktop demo also cross-compiles to Cortex-M4/M7/M33 (ARMv7E-M, ARMv8-M), RISC-V `rv32imac`, and (with the espup toolchain) Xtensa ESP32 — without an allocator, an executor, or any heap-backed math.

This chapter is the field guide for the embedded path: how to build it, what to feature-gate, what to plug it into, and what the verified CI targets are.

## The `no_std` philosophy

Every estimator in `skyfix-core` is generic over a const `N` and stores its state matrices as `nalgebra::SMatrix<T, R, C>` — stack-allocated, statically sized, no heap. The trade-off:

- ✅ No allocator dependency. The crate builds for any target rustc supports, including bare-metal MCUs without `liballoc`.
- ✅ Predictable stack usage. The memory footprint of an `Ekf<f32, 6>` is `6 × 4 + 6 × 6 × 4 = 168` bytes plus the model-specific Jacobian buffers; budget that against the executor's stack budget per task.
- ⚠️ State dimension is a compile-time const. You can't decide `N` at runtime; pick the dimension you need (2D vs 3D, position-only vs position+velocity) and the same binary will run that scenario.
- ⚠️ Particle count K is also compile-time const. The `Pf<T, N, K>` ensemble is `SMatrix<T, N, K>` — a 64-particle 2D PF is `64 × 2 × 4 = 512` bytes; a 512-particle 2D PF is 4 KB. Plan stack budget accordingly.

## Cross-compilation per target

The three targets the CI matrix verifies are:

```sh
# Cortex-M4F / M7 (STM32F4, STM32H7, nRF52840 with FPU, etc.)
rustup target add thumbv7em-none-eabihf
cargo build -p skyfix-core --target thumbv7em-none-eabihf \
    --no-default-features --features libm

# Cortex-M33 (nRF5340, STM32L5, RP2350 secure core, etc.)
rustup target add thumbv8m.main-none-eabihf
cargo build -p skyfix-core --target thumbv8m.main-none-eabihf \
    --no-default-features --features libm

# RISC-V rv32imac (ESP32-C3/C6 standalone, GD32V, RP2350 RISC-V core, etc.)
rustup target add riscv32imac-unknown-none-elf
cargo build -p skyfix-core --target riscv32imac-unknown-none-elf \
    --no-default-features --features libm
```

The `--no-default-features --features libm` incantation:

- `--no-default-features` drops the `f32` workspace default that quietly pulls in `std`-touching parts.
- `--features libm` forwards to `nalgebra/libm`, which routes transcendentals (`sin`, `cos`, `sqrt`, `atan2`, `exp`, `ln`) through Rust's `libm` crate instead of `core::f32::*` intrinsics that aren't bare-metal-safe everywhere.

For Xtensa ESP32 (original ESP32 / ESP32-S2 / ESP32-S3, not the RISC-V ESP32-C3) you need the espup toolchain:

```sh
cargo install espup
espup install
cargo +esp build -p skyfix-uwb --target xtensa-esp32-none-elf \
    --no-default-features --features libm
```

This path is verified manually on the dev box; the GitHub CI matrix doesn't yet include Xtensa because espup setup is heavier than the dtolnay/rust-toolchain action. Slated for v0.2 once a usable GitHub action exists.

## Why skyfix-core ships its own dense solvers

A footgun worth knowing about: nalgebra's `.lu()`, `.qr()`, `.svd()`, etc. require `Const<N>: ToTypenum`, a trait that only has impls for `N ≤ 127`. The const is fine for everyday work but means a generic function

```rust,ignore
fn solve<T: RealField + Copy, const N: usize>(a: SMatrix<T, N, N>, b: SVector<T, N>) -> SVector<T, N> {
    a.lu().solve(&b).unwrap()  // ← compile error: Const<N>: ToTypenum not satisfied
}
```

fails to compile for arbitrary `N`, because the type system can't prove `Const<N>` always lands in the `ToTypenum` impl range. The standard workaround in nalgebra ecosystem crates is to instantiate at concrete dimensions, but that defeats the purpose of generic code.

Instead, `skyfix-core::numeric` ships hand-rolled solvers that operate directly on `SMatrix<T, N, N>`:

- `solve_linear_system(a, b)` — partial-pivot Gauss elimination, used by `ToaTrilateration` and the EKF Joseph form.
- `solve_normal_equations(j, r)` — for least-squares (Foy WLS, Gauss-Newton).
- `gauss_newton(...)` — generic NLS iteration.
- `invert_square(m)` — full inverse for FIM → CRLB.
- `cholesky(m)` — used by `Ukf` to spread sigma points.

All five are generic over `N` and live in `crates/skyfix-core/src/numeric.rs`. The cost is reinventing well-trodden algorithms; the benefit is the algorithmic surface compiles cleanly for any `N` you need.

If you find yourself reaching for a decomposition not in `numeric`, prefer adding it there rather than calling into nalgebra's decompositions module from the algorithmic core — the `Const<N>: ToTypenum` trait bound will leak through and break embedded builds.

## Pairing with an executor

`skyfix-core` has **no executor dependency**. The estimators are plain structs with `&mut self` methods; you call them from whatever execution context fits your hardware:

- **[Embassy](https://embassy.dev)** — async-first, ergonomic for sensor-driven workflows where measurements arrive via interrupts feeding `Channel`s.
- **[RTIC v2](https://rtic.rs)** — priority-based hard real-time, ergonomic when filter updates need bounded worst-case latency.
- **Bare metal** — direct from the main loop with a hardware timer for the integration tick.

A typical Embassy task wrapping an EKF predict/update cycle:

```rust,ignore
// pseudocode — replace the I/O with your DW3000 / GNSS / IMU driver
#[embassy_executor::task]
async fn tracker_task(mut ekf: Ekf<f32, 4>) {
    let mut ticker = Ticker::every(Duration::from_millis(100));
    loop {
        ticker.next().await;
        let dt = 0.1_f32;
        ekf.predict(&transition, dt);

        if let Some(measurement) = range_channel.try_receive() {
            let model = RangeAnchor::new(measurement.anchor, 0.04);
            let z = SVector::from([measurement.range]);
            ekf.update(&model, &z).ok();
        }
    }
}
```

The Ekf struct itself stays portable; only the executor harness changes between Embassy / RTIC / bare-metal use.

## The skyfix-uwb integration sketch

`skyfix-uwb` is the adapter crate between a UWB ranging stack and the `skyfix-core` estimators. It's `#![no_std]` with no I/O — just the data types and conversion helpers — so the full conversion pipeline can be exercised from desktop tests today and dropped onto the MCU without changes tomorrow.

```rust,no_run
use nalgebra::{SVector, Vector2};
use skyfix_core::{Ekf, IdentityTransition, RangeAnchor, ToaTrilateration, Estimator};
use skyfix_uwb::UwbRange;

// One DW3000 ranging exchange produces a UwbRange.
let r0 = UwbRange::<2>::new(0xA001, Vector2::new( 0.0,  0.0), 4.95);
let r1 = UwbRange::<2>::new(0xA002, Vector2::new(10.0,  0.0), 7.10);
let r2 = UwbRange::<2>::new(0xA003, Vector2::new( 0.0, 10.0), 7.10);

// Convert to skyfix-core's ToaMeasurement and cold-start with trilateration.
let toas = [r0.to_toa(), r1.to_toa(), r2.to_toa()];
let seed = ToaTrilateration::<f64, 2>::new().estimate(&toas).expect("ok");

// Hand off to an EKF for tracking.
let mut ekf = Ekf::<f64, 2>::new(seed, nalgebra::SMatrix::<f64, 2, 2>::identity() * 0.5);
# let _ = (&ekf,);
```

The same pattern works on the MCU — `UwbRange` is `Copy + no_std`, and `Ekf` is allocator-free. Wire your DW3000 driver to produce `UwbRange` values and feed the trilateration + EKF pipeline from a loop or task. The `dw3000-ng` 1.0+ driver is the recommended SPI-side stack; the SPI / interrupt boilerplate is what would live in `skyfix-uwb`'s eventual Phase 7b expansion.

For TDoA rather than ToA, `skyfix_uwb::pair_to_tdoa` and `ranges_to_tdoa_batch` turn pairs of `UwbRange` into `TdoaMeasurement` values that feed `ChanLinear2D` / `FoyTdoa`.

## Stack-budgeting the particle filter

The static-K design makes the PF buildable for MCU, but K is bounded by stack budget. Approximate footprint of `Pf<f32, N, K>`:

- Particles: `N × K × 4` bytes
- Log-weights: `K × 4` bytes
- Working buffers (resample indices, normalized weights): another `K × 4` bytes

For a 2D PF: `(2N + 2) × K × 4 ≈ 24K` bytes. K = 64 → ~1.5 KB; K = 256 → ~6 KB; K = 1024 → ~24 KB. Cortex-M task stacks are typically 4-16 KB, so K in the 64–256 range is what fits comfortably. For larger ensembles, either bump the stack (Embassy `pool_size` / RTIC task stack), accept lower K with a tighter prior, or punt to the desktop path.

## The no-cuda-leak guard in CI

Whenever you're working on the algorithmic core, the embedded-target builds in CI keep you honest:

```yaml
# .github/workflows/ci.yml — embedded job
- run: cargo build -p skyfix-core --no-default-features --features libm \
       --target ${{ matrix.target }}
```

…paired with the `no-cuda-leak` job described in [chapter 4](./chapter_04_gpu.md) that fails the build if anything `cuda|cudarc|libloading|nvrtc`-flavored shows up in the embedded dep graph. Together these enforce the architectural invariant: nothing in `skyfix-core`'s dep graph ever depends, even transitively, on host-only crates.

If you add a feature that breaks an embedded target, the CI matrix tells you which target on which line. The most common offenders historically:

- Pulling `std::f32` instead of going through nalgebra (`use nalgebra::ComplexField` for `sqrt`, `sin`, etc.).
- Adding a `thiserror` use that isn't `#[cfg(feature = "std")]`-gated.
- Importing from a crate that doesn't expose a `no_std` build.

## Verified targets (as of this writing)

| Target | Architecture | Verified by |
|---|---|---|
| `thumbv7em-none-eabihf` | Cortex-M4F / M7 | `embedded` CI matrix |
| `thumbv8m.main-none-eabihf` | Cortex-M33 | `embedded` CI matrix |
| `riscv32imac-unknown-none-elf` | RISC-V (ESP32-C3, etc.) | `embedded` CI matrix |
| `xtensa-esp32-none-elf` | Xtensa ESP32 | manual via espup |

The deployment target for the v0.1 milestones is **Jetson Orin native** (aarch64) for the GPU path and Cortex-M / RISC-V for the embedded path. Cross-compilation x86 → Jetson is deferred to v0.2.

## Where next

You've now seen all of skyfix — closed-form fixes, recursive filters, CRLB-driven anchor planning, GPU acceleration, and embedded deployment. The natural next step is to start building:

1. Clone, run `cargo run -p skyfix-sim --release --example multi_filter_comparison`, and verify the numbers in [chapter 2](./chapter_02_filters.md) reproduce.
2. Read `crates/skyfix-core/tests/` — the test suite is the most accurate API documentation in the repo.
3. Pick an MCU board, wire up a DW3000, and feed `skyfix_uwb::UwbRange` values into a trilateration + EKF pipeline. Open an issue on the repo if anything along that path is rough — the embedded-side ergonomics are the project's least-exercised surface and feedback is wanted.
