# Trilateration: the first fix

The simplest UAV localization problem: **given M anchors at known positions and a noisy range measurement from each anchor to the UAV, where is the UAV?**

This chapter walks through `ToaTrilateration` — the closed-form solver. It's `O(N³)` for an N-dimensional space, runs in microseconds, requires no prior estimate, and is the natural way to seed every other estimator in the library.

## The math, in 30 seconds

A ToA range measurement from anchor `a_i` to target `x` gives

$$
\|x - a_i\|^2 = r_i^2
$$

Squaring out the norm: \\(x^Tx - 2 a_i^T x + \|a_i\|^2 = r_i^2\\). For each pair of measurements, the \\(x^T x\\) term subtracts away:

$$
2 (a_0 - a_i)^T \, x = r_i^2 - r_0^2 - \|a_i\|^2 + \|a_0\|^2
$$

In N dimensions, picking one measurement as a reference and forming N such equations against the remaining ones gives a square \\(N \times N\\) linear system. Gauss elimination solves it in `O(N³)`.

`ToaTrilateration` requires at least **N + 1 measurements** (3 in 2D, 4 in 3D). With more measurements supplied, only the first N+1 participate — use `ToaNls` for the overdetermined least-squares variant.

## The Rust code

```rust,no_run
use nalgebra::Vector2;
use skyfix_core::{Estimator, ToaMeasurement, ToaTrilateration};

let measurements = [
    ToaMeasurement::new(Vector2::new(0.0_f64, 0.0), 5.0),
    ToaMeasurement::new(Vector2::new(5.0, 0.0), 5.0),
    ToaMeasurement::new(Vector2::new(0.0, 5.0), 5.0),
];

let estimate = ToaTrilateration::<f64, 2>::new()
    .estimate(&measurements)
    .expect("non-degenerate anchor geometry");

println!("target at ({:.3}, {:.3})", estimate.x, estimate.y);
```

`ToaTrilateration` is generic over `T: nalgebra::RealField + Copy` and `const N: usize` — the same struct handles 2D, 3D, and any other N where you have at least N+1 well-conditioned anchors.

## When trilateration fails

Three failure modes, all surfaced as `EstimationError`:

| Variant | When |
|---|---|
| `InsufficientMeasurements { needed, got }` | Fewer than N+1 measurements |
| `SingularSystem` | Anchors degenerate (colinear in 2D, coplanar in 3D) |
| `NonFinite` | LU produced NaN or ±∞ (shouldn't happen on finite inputs but the check is free) |

The `SingularSystem` case is real and important — if your anchors all sit on a line, no algorithm in this library can disambiguate left from right of that line from range alone. The [CRLB chapter](./chapter_03_crlb.md) shows how to detect this *before* deploying anchors.

## How fast is it?

From the multi-filter comparison demo (`cargo run -p skyfix-sim --release --example multi_filter_comparison`), on an Ubuntu 24.04 x86_64 box:

| Method | RMSE (m) | wall time |
|---|---|---|
| **Trilateration** | **0.3198** | **2.78 µs** |
| EKF (seeded) | 0.2169 | 11.92 µs |
| UKF (warm) | 0.2167 | 23.44 µs |
| PF (K=512) | 0.2421 | 2.23 ms |

100 position fixes in **2.78 µs total** — that's 36 million fixes per second per core. The accuracy is worse than the recursive filters by ~50% (no temporal averaging) but the cost gap is *4 orders of magnitude*, so trilateration shines as:

1. **The seed for a recursive filter** — bootstrap an EKF / UKF / PF with one closed-form fix on the first measurement set, then track from there.
2. **The lone estimator for stationary scenarios** with high sensor SNR.
3. **The fallback when temporal correlation isn't available** (event-driven sensing, sparse anchor visits, etc.).

## The idiomatic skyfix pattern

```rust,no_run
# use nalgebra::{SMatrix, SVector, Vector2};
# use skyfix_core::{Ekf, Estimator, IdentityTransition, RangeAnchor, ToaMeasurement, ToaTrilateration};
// 1. Cold start with closed-form trilateration on the first measurement set.
let measurements: Vec<ToaMeasurement<f64, 2>> = /* from your sensor stack */
#   vec![ToaMeasurement::new(Vector2::new(0.0, 0.0), 5.0)];
let seed = ToaTrilateration::<f64, 2>::new()
    .estimate(&measurements)
    .expect("trilateration ok");

// 2. Hand off to an EKF for refinement + tracking.
let mut ekf = Ekf::<f64, 2>::new(seed, SMatrix::<f64, 2, 2>::identity() * 0.5);
let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.01);

// 3. Process subsequent measurements through the filter:
let dt = 0.1;
ekf.predict(&transition, dt);
for m in &measurements {
    let model = RangeAnchor::<f64, 2>::new(m.anchor, 0.04);
    let mut z = SVector::<f64, 1>::zeros();
    z[0] = m.range;
    ekf.update(&model, &z).expect("ekf update ok");
}

let position = ekf.state();  // sub-cm precision once the filter converges
```

The test `ekf_seeded_from_trilateration_recovers_tight_estimate` in `crates/skyfix-core/tests/ekf.rs` validates this pattern lands at machine precision for noise-free inputs.

## Next

[Chapter 2](./chapter_02_filters.md) is the deep dive on recursive filters — when EKF beats UKF, when PF is worth its cost, and what "warm start vs cold start" means quantitatively.
