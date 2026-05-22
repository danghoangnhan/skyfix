# skyfix-sim

Desktop simulator and runnable examples for skyfix.

## What's here

| Target                                       | Command                                                                |
|----------------------------------------------|------------------------------------------------------------------------|
| Default demo (EKF + CSV output)              | `cargo run -p skyfix-sim --release`                                    |
| Bootstrap example (trilateration → EKF)      | `cargo run -p skyfix-sim --example trilateration_to_ekf`               |
| **Multi-filter comparison (showcase)**       | `cargo run -p skyfix-sim --release --example multi_filter_comparison`  |

The default demo runs a 100-step circular-trajectory simulation with 4 anchors at the corners of a 10 × 10 m room. Noisy ToA measurements (σ = 0.2 m) feed into an EKF that is bootstrapped from a `ToaTrilateration` cold-start fix on the first measurement set. The output is three CSVs (`truth.csv`, `estimate.csv`, `anchors.csv`) suitable for plotting in gnuplot, matplotlib, Excel, etc.

The **multi-filter comparison** runs the same scenario through Trilateration / EKF / UKF / PF (K=512) — all sharing identical noisy measurements — and prints a side-by-side RMSE / max-error / wall-time table, plus a wide-format `multi_comparison.csv`. Sample output:

```
 method              RMSE (m)     max (m)     wall time
 ────────────────  ──────────  ──────────  ────────────
 Trilateration         0.3198      0.8337      2.78µs
 EKF (seeded)          0.2169      0.3847     11.92µs
 UKF (warm)            0.2167      0.3849     23.44µs
 PF (K=512)            0.2421      0.4408      2.23ms
```

Trilateration is ~430× cheaper than EKF but 50% worse RMSE; EKF and UKF land within rounding of each other on this warm-start scenario; PF (K=512) trades 190× the wall time for a slightly worse mean (Monte-Carlo variance dominates at K=512 for noise-free linearization).

Quick gnuplot recipe:

```gnuplot
set datafile separator ","
plot 'truth.csv' using 2:3 with lines title 'truth', \
     'estimate.csv' using 2:3 with lines title 'EKF', \
     'anchors.csv' using 1:2 with points pt 7 ps 2 title 'anchors'
```

## Library primitives (for your own scenarios)

- `Anchor2D` — position + range variance pair.
- `Trajectory2D` trait + `CircularTrajectory` impl.
- `ToASimulator2D` — generates noisy `ToaMeasurement<f64, 2>` from any target position.
- `StepRecord` + `rmse` — per-step records and RMSE helper for comparisons.

## Why CSV instead of a built-in plot?

Plotters with TTF text rendering pulls in `fontconfig` as a system dep, which isn't worth the install friction for a default demo. CSV output has no system deps and plays cleanly with any visualization tool the user already has. A plot feature (gated behind an optional `plot` feature) can land in a follow-up once font handling is sorted.

## Deferred to next phases

- STM32 / ESP32 embedded examples (Phase 7 — needs the driver crates and embedded HAL setup).
- mdBook tutorial reproducing UCNLNav reference vectors.
- CSV trajectory replay (`CsvTrajectory` impl of `Trajectory2D`).
