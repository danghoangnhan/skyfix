# skyfix-core

`no_std` algorithmic core for UAV localization and triangulation.

## What's implemented (so far)

| Modality | Estimator                           | Type                 | Phase |
|----------|-------------------------------------|----------------------|-------|
| ToA      | `ToaTrilateration<T, N>`            | closed-form (N+1)    | 1     |
| ToA      | `ToaNls<T, N>`                      | Gauss-Newton (≥ N)   | 2     |
| TDoA     | `ChanLinear2D<T>`, `ChanLinear3D<T>`| closed-form linear   | 2     |
| TDoA     | `FoyTdoa<T, N>`                     | Taylor-series WLS    | 2     |
| AoA (2D) | `StansfieldAoa<T>`                  | bearings-only LSQ    | 2     |
| RSSI     | `RssiPathLoss<T>` (calibration)     | log-distance model   | 2     |

Bayesian filters (EKF/UKF/PF) and the CRLB/GDOP analyzer land in Phases 3 and 4.

## Features

| Feature   | Default | What it does                                                                 |
|-----------|---------|------------------------------------------------------------------------------|
| `std`     | yes     | Enables `std`, `nalgebra/std`. Pulls in `alloc`.                             |
| `alloc`   | (impl.) | `alloc`-only support without `std`.                                          |
| `libm`    | no      | `no_std` transcendentals via `libm`. Use for embedded targets.               |
| `serde`   | no      | `Serialize` / `Deserialize` via `nalgebra/serde-serialize`.                  |
| `defmt`   | no      | `defmt::Format` for embedded logging.                                        |

## Embedded build

```sh
cargo build -p skyfix-core \
    --no-default-features --features libm \
    --target thumbv7em-none-eabihf
```

CI verifies this for `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, and `riscv32imac-unknown-none-elf`.

## Numeric note

`nalgebra` decompositions (`.lu()`, `.qr()`, `.svd()`) require `Const<N>: ToTypenum`, which doesn't hold for arbitrary `const N`. The crate ships its own [`numeric::solve_linear_system`], [`numeric::solve_normal_equations`], and [`numeric::gauss_newton`] helpers built on plain Gauss elimination with partial pivoting. They're public so out-of-tree estimators can reuse them.

See the [project README](../../README.md) for the wider picture.
