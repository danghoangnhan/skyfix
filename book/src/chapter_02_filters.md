# Bayesian filters: EKF, UKF, PF

Trilateration gives you a single-shot fix. Recursive filters maintain *state across time* — both a position estimate AND its uncertainty, updated incrementally as new measurements arrive. skyfix ships three: **EKF** (linearized analytic), **UKF** (sigma-point), **PF** (Monte Carlo).

This chapter says when to pick which.

## The model interface

All three filters share the same trait surface:

```rust,no_run
pub trait TransitionModel<T: RealField + Copy, const N: usize> {
    fn transition(&self, x: &SVector<T, N>, dt: T) -> SVector<T, N>;
    fn jacobian(&self, x: &SVector<T, N>, dt: T) -> SMatrix<T, N, N>;
    fn noise(&self, dt: T) -> SMatrix<T, N, N>;
}

pub trait ObservationModel<T: RealField + Copy, const N: usize, const M: usize> {
    fn predict(&self, x: &SVector<T, N>) -> SVector<T, M>;
    fn jacobian(&self, x: &SVector<T, N>) -> SMatrix<T, M, N>;
    fn noise(&self) -> SMatrix<T, M, M>;
}
```

Implement these once for your dynamics + sensor and you can swap any of EKF/UKF/PF underneath without touching the model code. skyfix-core ships two built-in implementations to get you started: `IdentityTransition` (state unchanged, additive process noise — for stationary or slow-moving targets) and `RangeAnchor` (one range measurement to a fixed anchor).

## The empirical comparison

From `cargo run -p skyfix-sim --release --example multi_filter_comparison` on a 100-step circular trajectory through a 10×10 m room, 4 anchors at corners, σ = 0.2 m range noise:

| Method | RMSE (m) | max err (m) | wall time |
|---|---|---|---|
| Trilateration | 0.3198 | 0.8337 | 2.78 µs |
| **EKF (seeded)** | **0.2169** | **0.3847** | **11.92 µs** |
| UKF (warm) | 0.2167 | 0.3849 | 23.44 µs |
| PF (K=512) | 0.2421 | 0.4408 | 2.23 ms |

Several things stand out:

1. **EKF and UKF tie**. On this warm-started range-tracking scenario, sigma-point sampling buys nothing — the measurement function is smooth enough that the analytic Jacobian is accurate at the operating point.
2. **PF is 190× slower than EKF** for marginally worse accuracy. PF earns its keep on *non-Gaussian* posteriors (e.g. multimodal distributions from ambiguous bearing measurements, or hard nonlinearities like reflection); a tame range-tracking problem isn't its sweet spot.
3. **EKF is 4× the cost of trilateration** for 33% better accuracy. This is the empirical sweet spot for most UAV indoor tracking.

## When to pick which

| Pick | When |
|---|---|
| `ToaTrilateration` | Single-shot fix, no time correlation, stationary or sparse-anchor scenarios |
| `Ekf` | Smooth nonlinearities, good prior, you want the cheapest filter that tracks. The default. |
| `Ukf` | Significant linearization error from the prior (cold start) or strongly nonlinear `h(x)` (e.g. AoA bearings at long range, RSSI with steep path-loss exponent) |
| `Pf` | Non-Gaussian posteriors (multimodal distributions, hard ambiguity, particle dynamics that don't admit a Gaussian approximation) |

In practice for UAV indoor tracking: **seed an EKF from trilateration, then forget the others exist** unless the empirical RMSE doesn't satisfy you.

## EKF cold start vs warm start

EKF's Achilles heel is **linearization at the prior**. If your prior is wrong, the Jacobian is computed at the wrong state, the update direction is wrong, and the filter can converge to a *biased* fixed point — even with noise-free measurements.

The skyfix test suite measures this explicitly:

```
ekf_2d_warm_start_converges_tightly        : 1e-4 m error
ekf_2d_cold_start_converges_within_bias_floor : ~5 cm error (bias floor)
ekf_seeded_from_trilateration_recovers_tight_estimate : machine precision
```

The first test starts the EKF at (1.5, 2.5) with truth at (2.0, 3.0) — a 0.7 m initial error. It converges to within 1e-4 m. The second test starts at (0, 0) with truth at (2, 3) — a 3.6 m initial error. It converges to ~5 cm and *stays there* because the Jacobians evaluated during the initial steps were systematically biased. The third test uses the trilateration-seeded pattern (the cold start fix from chapter 1) and gets machine precision.

**Always seed your EKF.** Either:
- Use `ToaTrilateration` / `ChanLinear*` for a closed-form fix on the first measurement set, or
- Use a UKF for the first ~20 steps until you're close to truth, then switch to EKF.

## UKF parameters

skyfix's `Ukf` defaults to the **classic Unscented Transform**: `alpha = 1, beta = 2, kappa = 3 - N`. This gives `c = N + λ = 3` and well-conditioned weights for any N.

You may see the **Wan & van der Merwe scaled UT** (`alpha = 1e-3, beta = 2, kappa = 0`) recommended in tutorials. **Don't use it on moderate-magnitude covariances.** It produces weights of order `1/alpha² = 10⁶`, which destroys precision on anything but extremely small covariance matrices. The skyfix README documents this gotcha; the test suite has the cold-start case that exposed it during development.

The const generic `SIGMAS` must equal `2N + 1`. To avoid getting this wrong, use the type aliases:

```rust,no_run
use skyfix_core::{Ukf2D, Ukf3D, Ukf4D, Ukf6D};
// Equivalent to Ukf<T, 2, 5>, Ukf<T, 3, 7>, etc.
```

## PF design

skyfix's `Pf<T, N, K>` is a **Sequential Importance Resampling** filter. Static-K (compile-time const generic), `no_std`-compatible — the particle ensemble is a stack-allocated `SMatrix<T, N, K>`, so the filter runs on a Cortex-M4 without an allocator.

RNG enters via `FnMut() -> T` closures, not a crate dep. You plug in any sampler: a hardware TRNG on an MCU, ChaCha20 on a host, a deterministic seeded PRNG for tests. The skyfix-core tests use `rand_chacha::ChaCha8Rng` as a dev-dep.

```rust,no_run
# use nalgebra::{SMatrix, Vector2};
# use rand::SeedableRng;
# use rand_chacha::ChaCha8Rng;
# use rand_distr::{Distribution, StandardNormal};
# use skyfix_core::Pf;
let mut rng = ChaCha8Rng::seed_from_u64(42);
let pf = Pf::<f64, 2, 512>::from_gaussian(
    Vector2::new(0.0, 0.0),
    SMatrix::<f64, 2, 2>::identity(),
    || StandardNormal.sample(&mut rng),
).expect("ok");
```

`resample_if_needed(0.5, uniform_sampler)` triggers a systematic resampling when the effective sample size drops below 50% of K. ESS uses log-sum-exp normalization for numerical stability.

## Next

[Chapter 3](./chapter_03_crlb.md) (coming soon) — the **CRLB / GDOP analyzer** for anchor placement. The question isn't "how do I track better?" — it's "where should I put the anchors so that *any* estimator can track well?"
