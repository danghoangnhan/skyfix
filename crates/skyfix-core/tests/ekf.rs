//! Integration tests for the Extended Kalman Filter.

use approx::assert_relative_eq;
use nalgebra::{SMatrix, SVector, Vector2, Vector3};
use skyfix_core::{Ekf, IdentityTransition, RangeAnchor};

/// EKF tracking from a *close* prior (warm start) — its strong suit.
/// Should converge to within ~1e-3 of truth after a handful of iterations.
#[test]
fn ekf_2d_warm_start_converges_tightly() {
    let truth = Vector2::new(2.0_f64, 3.0);
    let prior_mean = Vector2::new(1.5_f64, 2.5); // close to truth
    let prior_cov = SMatrix::<f64, 2, 2>::identity() * 1.0;

    let mut ekf = Ekf::<f64, 2>::new(prior_mean, prior_cov);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-9);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    for _step in 0..20 {
        ekf.predict(&transition, 1.0);
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 2>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ekf.update(&model, &z).expect("update ok");
        }
    }

    assert_relative_eq!(ekf.state(), truth, epsilon = 1e-4);
}

/// EKF tracking from a *close* 3D prior. Tight tolerance.
#[test]
fn ekf_3d_warm_start_converges_tightly() {
    let truth = Vector3::new(1.5_f64, -2.0, 4.2);
    let prior_mean = Vector3::new(1.0_f64, -1.5, 4.0); // close to truth
    let prior_cov = SMatrix::<f64, 3, 3>::identity() * 1.0;

    let mut ekf = Ekf::<f64, 3>::new(prior_mean, prior_cov);
    let transition = IdentityTransition::<f64, 3>::with_uniform_noise(1e-9);
    let anchors = [
        Vector3::new(0.0_f64, 0.0, 0.0),
        Vector3::new(10.0, 0.0, 0.0),
        Vector3::new(0.0, 10.0, 0.0),
        Vector3::new(0.0, 0.0, 10.0),
        Vector3::new(10.0, 10.0, 10.0),
    ];

    for _step in 0..20 {
        ekf.predict(&transition, 1.0);
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 3>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ekf.update(&model, &z).expect("update ok");
        }
    }

    assert_relative_eq!(ekf.state(), truth, epsilon = 1e-3);
}

/// EKF tracking from a *far* prior (cold start) only converges to the bias
/// floor set by the Jacobian linearization at the bad initial state. Confirm
/// it gets close-ish to truth without diverging — the well-known EKF limit.
/// For cold-start scenarios use UKF (Phase 3.5) or seed with a closed-form
/// estimate from `ToaTrilateration` / `ChanLinear*`.
#[test]
fn ekf_2d_cold_start_converges_within_bias_floor() {
    let truth = Vector2::new(2.0_f64, 3.0);
    let prior_mean = Vector2::new(0.0_f64, 0.0);
    let prior_cov = SMatrix::<f64, 2, 2>::identity() * 100.0;

    let mut ekf = Ekf::<f64, 2>::new(prior_mean, prior_cov);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-6);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    for _step in 0..30 {
        ekf.predict(&transition, 1.0);
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 2>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ekf.update(&model, &z).expect("update ok");
        }
    }

    // EKF cold start: residual error is bounded but non-trivial. 5 cm here.
    let err = (ekf.state() - truth).norm();
    assert!(err < 0.05, "cold-start error too large: {err}");
}

#[test]
fn ekf_predict_with_identity_and_zero_noise_is_no_op_on_state() {
    let state = Vector2::new(1.0_f64, 2.0);
    let cov = SMatrix::<f64, 2, 2>::identity();
    let mut ekf = Ekf::<f64, 2>::new(state, cov);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.0);

    ekf.predict(&transition, 0.1);
    assert_relative_eq!(ekf.state(), state, epsilon = 1e-12);
    assert_relative_eq!(ekf.covariance(), cov, epsilon = 1e-12);
}

#[test]
fn ekf_single_range_update_shrinks_covariance_along_anchor_direction() {
    // Prior at origin with isotropic covariance. Anchor on +x axis. After
    // one update, the variance along x should be smaller than along y —
    // a range measurement constrains the radial direction.
    let mut ekf = Ekf::<f64, 2>::new(
        Vector2::new(0.0_f64, 0.0),
        SMatrix::<f64, 2, 2>::identity() * 10.0,
    );
    let anchor = Vector2::new(5.0_f64, 0.0);
    let model = RangeAnchor::<f64, 2>::new(anchor, 0.01_f64);
    let mut z = SVector::<f64, 1>::zeros();
    z[0] = 5.0;

    ekf.update(&model, &z).expect("update ok");
    let cov = ekf.covariance();
    assert!(
        cov[(0, 0)] < cov[(1, 1)],
        "x-variance {} should be smaller than y-variance {}",
        cov[(0, 0)],
        cov[(1, 1)]
    );
}

/// Idiomatic cold-start workflow: use a closed-form estimator (here,
/// `ToaTrilateration` from Phase 1) to seed the EKF, then track. Should give
/// a tight estimate even from a "no prior" starting point.
#[test]
fn ekf_seeded_from_trilateration_recovers_tight_estimate() {
    use skyfix_core::{Estimator, ToaMeasurement, ToaTrilateration};

    let truth = Vector2::new(2.0_f64, 3.0);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    // Seed the EKF with a closed-form trilateration solution.
    let trilat_measurements: Vec<_> = anchors
        .iter()
        .map(|&a| ToaMeasurement::new(a, (truth - a).norm()))
        .collect();
    let seed = ToaTrilateration::<f64, 2>::new()
        .estimate(&trilat_measurements)
        .expect("trilateration seed ok");

    let mut ekf = Ekf::<f64, 2>::new(seed, SMatrix::<f64, 2, 2>::identity() * 0.1);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-9);
    for _ in 0..10 {
        ekf.predict(&transition, 1.0);
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 2>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ekf.update(&model, &z).expect("update ok");
        }
    }

    assert_relative_eq!(ekf.state(), truth, epsilon = 1e-9);
}
