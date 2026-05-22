# Introduction

**skyfix** is a Rust workspace for UAV localization & triangulation, targeting both `no_std` microcontrollers and `std` companion computers. This book is its practical guide — it walks from "I have noisy range measurements" all the way to "my filter is tracking a moving target on actual hardware."

## What's in the workspace

```
crates/
├── skyfix-core/       no_std-by-default algorithmic core
├── skyfix-sim/        desktop demo binary + examples
├── skyfix-cuda/       NVIDIA CUDA acceleration (cudarc 0.19 + nvcc kernels)
├── skyfix-uwb/        UWB ranging adapter (DW3000-compatible, no_std)
└── skyfix-fixtures/   shared reference test vectors
```

## What it covers (algorithm matrix)

| Modality | Estimator | Type | `no_std` |
|---|---|---|---|
| ToA | `ToaTrilateration<T, N>` | closed-form, `N+1` anchors | ✓ |
| ToA | `ToaNls<T, N>` | Gauss-Newton, ≥ `N` anchors | ✓ |
| TDoA | `ChanLinear2D<T>`, `ChanLinear3D<T>` | Chan stage-1 closed-form | ✓ |
| TDoA | `FoyTdoa<T, N>` | Taylor-series WLS | ✓ |
| AoA (2D) | `StansfieldAoa<T>` | bearings-only LSQ | ✓ |
| RSSI | `RssiPathLoss<T>` | log-distance calibration | ✓ |
| Hybrid | `HybridTdoaAoa2D<T>` | TDoA + AoA Gauss-Newton | ✓ |
| Bayesian | `Ekf<T, N>` | Joseph-form EKF | ✓ |
| Bayesian | `Ukf<T, N, SIGMAS>` | classic Unscented Transform | ✓ |
| Bayesian | `Pf<T, N, K>` | SIR + systematic resampling | ✓ |
| Analysis | `CrlbBuilder` / `CrlbAnalysis` | Fisher Information / CRLB / GDOP | ✓ |
| GPU | `CudaGdopSweep2D`, `CudaPfRanges2D` | batched ops via NVIDIA CUDA | (std) |

## How to read this book

- **In order**, if you're new. Each chapter builds on the previous one.
- **By topic**, if you've used a Kalman-style filter before. Jump straight to the algorithm chapter.
- **By example**, if you just want to copy-paste. Every code block in this book runs as-is on a Rust 1.88+ toolchain — most are extracted from the demo programs in `crates/skyfix-sim/examples/`.

## Status note

skyfix is **pre-alpha**, not yet on crates.io. The CPU algorithm surface is feature-complete; integrations (driver crates, embedded examples) are next. For now, pin a git SHA when vendoring.

The empirical numbers quoted in this book come from a Linux x86_64 dev box with an NVIDIA RTX 5090, Ubuntu 24.04, CUDA 12.0.140, Rust 1.88. Your hardware will differ; the *relative* trade-offs should not.
