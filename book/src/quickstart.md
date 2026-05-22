# Quickstart

The fastest path from "let me see what this does" to a working tracking demo.

## Prerequisites

- **Rust 1.88+** — `rustup install 1.88 && rustup default 1.88` (the MSRV is pinned in `rust-toolchain.toml`; if you have rustup, the right version installs automatically when you `cd` into the repo).
- **A C linker** (`gcc` or `clang`) — needed by Rust's build scripts. On Ubuntu: `sudo apt install build-essential`.
- **(Optional) NVIDIA CUDA Toolkit 12+** — only if you want to build `skyfix-cuda`. On Ubuntu: `sudo apt install nvidia-cuda-toolkit`.

## Get the code

```sh
git clone https://github.com/danghoangnhan/skyfix.git
cd skyfix
```

## Run the EKF tracking demo

```sh
cargo run -p skyfix-sim --release
```

You should see something like:

```
skyfix-sim EKF tracking demo
 steps      = 100
 dt         = 0.1 s
 anchors    = 4 at room corners (10×10 m)
 range σ²   = 0.04 m² (σ = 0.2 m)
 RMSE       = 0.2177 m
 final err  = 0.2745 m
 wrote      = truth.csv, estimate.csv, anchors.csv
```

That's a moving 2D target tracked through 100 noisy range measurements over 10 seconds, using an EKF seeded from a closed-form trilateration fix.

## Plot the result (gnuplot)

```gnuplot
set datafile separator ","
plot 'truth.csv' using 2:3 with lines title 'truth' lw 2 lc rgb 'blue', \
     'estimate.csv' using 2:3 with lines title 'EKF' lw 1 lc rgb 'red', \
     'anchors.csv' using 1:2 with points pt 7 ps 2 title 'anchors' lc rgb 'black'
```

## Run the multi-filter comparison

```sh
cargo run -p skyfix-sim --release --example multi_filter_comparison
```

This is the marketing demo. It runs Trilateration / EKF / UKF / PF on identical noisy measurements and prints a side-by-side table:

```
 method              RMSE (m)     max (m)     wall time
 ────────────────  ──────────  ──────────  ────────────
 Trilateration         0.3198      0.8337      2.78µs
 EKF (seeded)          0.2169      0.3847     11.92µs
 UKF (warm)            0.2167      0.3849     23.44µs
 PF (K=512)            0.2421      0.4408      2.23ms
```

Read [chapter 2](./chapter_02_filters.md) for what those numbers mean.

## Run the GPU benchmark (requires CUDA toolkit)

```sh
cargo run -p skyfix-cuda --release --example gdop_grid_benchmark
```

Shows where the CUDA GDOP sweep wins over the CPU loop:

```
    grid    cells          CPU          GPU    speedup
 ─────── ──────── ──────────── ──────────── ──────────
   10×10      100       1.88µs      54.83µs       0.0×
   25×25      625      10.62µs      20.86µs       0.5×
   50×50     2500      39.85µs      24.17µs       1.6×
 100×100    10000     158.50µs      36.98µs       4.3×
 200×200    40000     621.71µs     104.32µs       6.0×
```

Crossover around 2500 cells; see [chapter 4](./chapter_04_gpu.md) for why.

## Cross-compile to an embedded target

```sh
rustup target add thumbv7em-none-eabihf
cargo build -p skyfix-core --no-default-features --features libm --target thumbv7em-none-eabihf
```

`skyfix-core` builds clean for Cortex-M4/M7, Cortex-M33, and RISC-V `rv32imac` (the three targets the CI matrix runs). No allocator required; covariance matrices live on the stack via `nalgebra::SMatrix<T, N, N>`.

## Where to go next

- [Chapter 1](./chapter_01_trilateration.md) for the simplest possible position fix (closed-form trilateration).
- [Chapter 2](./chapter_02_filters.md) for recursive filters and the EKF/UKF/PF comparison.
- The [`skyfix-sim` README](https://github.com/danghoangnhan/skyfix/tree/main/crates/skyfix-sim) for more examples.
