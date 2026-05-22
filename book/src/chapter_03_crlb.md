# CRLB & anchor placement

The first two chapters answered *"given measurements, where is the UAV?"* This chapter answers a different question: *"given a coverage region, where should I put the anchors so that any estimator has a chance?"*

That's the Cramér-Rao Lower Bound (CRLB) — the **achievable lower bound** on the covariance of any unbiased estimator, derived from the measurement geometry and noise model alone. If the CRLB is large at your operating point, no filter — not EKF, UKF, PF, or anything published in the future — will track tightly there. The fix is geometric: move the anchors.

## The math, in 30 seconds

For a target at position \\(x\\) with independent measurements whose likelihoods have Jacobians \\(J_i\\) and noise variances \\(\sigma_i^2\\), the Fisher Information Matrix is

$$
\mathrm{FIM} = \sum_i \frac{1}{\sigma_i^2}\, J_i^{\top} J_i
$$

The **CRLB on position covariance** is \\(\mathrm{FIM}^{-1}\\): any unbiased estimator's covariance is bounded below by this matrix in the positive-semidefinite sense. The scalar summary is the **Geometric Dilution of Precision**:

$$
\mathrm{GDOP} = \sqrt{\mathrm{tr}(\mathrm{FIM}^{-1})}
$$

GDOP is dimensionless multiplier: at a target position with GDOP = 2.0, an isotropic σ = 0.1 m of measurement noise yields a *position-error* standard deviation lower-bounded by 0.2 m, regardless of estimator. Halve GDOP and you halve the floor.

Per-modality Jacobian rows (1D measurement rows of \\(J_i\\)):

| Modality | Row of \\(J_i\\) | Variance units |
|---|---|---|
| ToA range | \\(u_i^{\top} = (x - a_i)^{\top} / \\|x - a_i\\|\\) | m² |
| TDoA range diff | \\(u_i^{\top} - u_{\mathrm{ref}}^{\top}\\) | m² |
| 2D AoA bearing | \\([-\Delta y / r^2,\; \Delta x / r^2]\\) | rad² |

All three are implemented in `crlb.rs`.

## The Rust code

```rust,no_run
use nalgebra::Vector2;
use skyfix_core::CrlbBuilder;

let target = Vector2::new(0.0_f64, 0.0);
let anchors = [
    Vector2::new( 1.0,  1.0),
    Vector2::new(-1.0,  1.0),
    Vector2::new( 1.0, -1.0),
    Vector2::new(-1.0, -1.0),
];

let mut builder = CrlbBuilder::<f64, 2>::new();
for &a in &anchors {
    builder.add_toa(target, a, /* variance */ 1.0);
}
let analysis = builder.finish();

println!("GDOP = {:.3}", analysis.gdop().unwrap());     // 1.000
println!("HDOP = {:.3}", analysis.hdop().unwrap());     // 1.000 (≡ GDOP in 2D)
println!("CRLB = {:?}",  analysis.covariance().unwrap()); // diag(0.5, 0.5)
```

The 4-corner anchor layout is the canonical sanity check. The FIM evaluates to \\(2 \cdot I\\), so CRLB is \\(0.5 \cdot I\\) and \\(\mathrm{GDOP} = \sqrt{0.5 + 0.5} = 1\\) — geometrically optimal for unit-variance ToA. This is the test `crlb_2d_symmetric_anchors_at_unit_corners_gives_gdop_1` in `crates/skyfix-core/tests/crlb.rs`.

## Detecting bad geometries before you deploy

The reason CRLB earns a chapter is that it catches geometry pathologies *before* you bolt the anchors to the wall. Three colinear anchors:

```rust,no_run
# use nalgebra::Vector2;
# use skyfix_core::CrlbBuilder;
let target = Vector2::new(0.0_f64, 0.0);
let anchors = [
    Vector2::new(1.0_f64, 0.0),
    Vector2::new(2.0,     0.0),
    Vector2::new(3.0,     0.0),
];

let mut b = CrlbBuilder::<f64, 2>::new();
for &a in &anchors {
    b.add_toa(target, a, 1.0);
}

assert!(b.finish().covariance().is_err()); // SingularSystem
```

The FIM has rank 1 — every Jacobian row is parallel to the x-axis, so the cross-track dimension carries zero information. No estimator can disambiguate left from right of the anchor line; this surfaces as `EstimationError::SingularSystem` from `covariance()` / `gdop()`. Test: `crlb_2d_three_colinear_anchors_singular_fim`.

## Variance scales linearly into GDOP

A useful sanity check from `crlb_2d_smaller_variance_yields_smaller_gdop`:

```rust,no_run
# use nalgebra::Vector2;
# use skyfix_core::CrlbBuilder;
# let target = Vector2::new(0.0_f64, 0.0);
# let anchors = [Vector2::new(1.0, 1.0), Vector2::new(-1.0, 1.0), Vector2::new(1.0, -1.0), Vector2::new(-1.0, -1.0)];
let gdop_at = |variance: f64| {
    let mut b = CrlbBuilder::<f64, 2>::new();
    for &a in &anchors { b.add_toa(target, a, variance); }
    b.finish().gdop().unwrap()
};

let loud  = gdop_at(1.0);   // σ² = 1 m²
let quiet = gdop_at(0.01);  // σ² = 0.01 m²

assert!((quiet / loud - 0.1).abs() < 1e-12);
```

Since the FIM scales as \\(1/\sigma^2\\), CRLB scales as \\(\sigma^2\\), and GDOP scales as \\(\sigma\\). A 10× tightening of range noise gives a 10× tightening of the achievable position-error floor — but only up to the point that geometry, not noise, becomes the binding constraint.

## Hybrid sensors stack additively

The FIM is **additive across independent measurements**, including across modalities. From `crlb_2d_hybrid_toa_plus_aoa_outperforms_either_alone`:

```rust,no_run
# use nalgebra::Vector2;
# use skyfix_core::CrlbBuilder;
# let target = Vector2::new(0.0_f64, 0.0);
# let anchors = [Vector2::new(5.0, 0.0), Vector2::new(0.0, 5.0), Vector2::new(-5.0, 0.0)];
// ToA-only baseline.
let mut b = CrlbBuilder::<f64, 2>::new();
for &a in &anchors { b.add_toa(target, a, 0.01); }
let toa_only = b.finish().gdop().unwrap();

// Hybrid: same anchors, but each is *both* a ToA ranger and an AoA bearing source.
let mut b = CrlbBuilder::<f64, 2>::new();
for &a in &anchors {
    b.add_toa(target, a, 0.01);
    b.add_aoa(target, a, 0.001);
}
let hybrid = b.finish().gdop().unwrap();

assert!(hybrid < toa_only);
```

Empirically the test shows `hybrid < toa_only` and `hybrid < aoa_only` — fusing two modalities at the same anchors always tightens the CRLB, even when each modality alone is geometrically fine. The math is just \\(\mathrm{FIM}_{\mathrm{hybrid}} = \mathrm{FIM}_{\mathrm{toa}} + \mathrm{FIM}_{\mathrm{aoa}}\\).

This is the design tool for sensor-suite planning: try every combination of (anchor positions × modalities at each anchor), pick the layout that minimizes the worst-case GDOP across your operating region.

## 3D: HDOP vs VDOP

In 3D, the trace splits cleanly into horizontal and vertical components:

```rust,no_run
use nalgebra::Vector3;
use skyfix_core::CrlbBuilder;

let target = Vector3::new(0.0_f64, 0.0, 0.0);
let anchors = [
    Vector3::new( 1.0,  1.0,  1.0),
    Vector3::new(-1.0, -1.0,  1.0),
    Vector3::new(-1.0,  1.0, -1.0),
    Vector3::new( 1.0, -1.0, -1.0),
];

let mut b = CrlbBuilder::<f64, 3>::new();
for &a in &anchors { b.add_toa(target, a, 1.0); }
let analysis = b.finish();

let gdop = analysis.gdop().unwrap();
let hdop = analysis.hdop().unwrap();   // √(P_xx + P_yy)
let vdop = analysis.vdop().unwrap();   // √P_zz

assert!((hdop * hdop + vdop * vdop - gdop * gdop).abs() < 1e-12);
```

This is the standard GNSS decomposition: a constellation with anchors clustered overhead has tight HDOP but loose VDOP, which matters for UAVs where altitude error is often the binding constraint on landing accuracy.

## Sweeping GDOP across a region

A single GDOP evaluation answers "how good is this target position?" The real planning question is "how does GDOP vary across my whole coverage region?" — a heatmap.

The naïve implementation is a CPU loop over a grid:

```rust,no_run
# use nalgebra::Vector2;
# use skyfix_core::CrlbBuilder;
# let anchors: [Vector2<f64>; 4] = [Vector2::new(0.0, 0.0); 4];
let nx = 200;
let ny = 200;
let mut heatmap = vec![0.0_f64; nx * ny];

for iy in 0..ny {
    for ix in 0..nx {
        let target = Vector2::new(ix as f64 / 20.0, iy as f64 / 20.0);
        let mut b = CrlbBuilder::<f64, 2>::new();
        for &a in &anchors { b.add_toa(target, a, 0.04); }
        heatmap[iy * nx + ix] = b.finish().gdop().unwrap_or(f64::INFINITY);
    }
}
```

For a 200×200 grid that's 40 000 independent CRLB evaluations — embarrassingly parallel and a textbook GPU workload. That's the wedge for [chapter 4](./chapter_04_gpu.md): `CudaGdopSweep2D` runs the same loop on the GPU and crosses over the CPU around 50×50, delivering a 6× speedup at 200×200.

## Next

[Chapter 4](./chapter_04_gpu.md) — putting the GDOP sweep on the GPU, the cudarc 0.19 feature soup, and the architectural rule that keeps CUDA out of the embedded build graph.
