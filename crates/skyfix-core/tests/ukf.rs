//! Integration tests for the Unscented Kalman Filter.

use approx::assert_relative_eq;
use nalgebra::{SMatrix, SVector, Vector2, Vector3};
use skyfix_core::{IdentityTransition, RangeAnchor, Ukf2D, Ukf3D};

#[test]
fn ukf_2d_warm_start_converges_tightly() {
    let truth = Vector2::new(2.0_f64, 3.0);
    let prior_mean = Vector2::new(1.5_f64, 2.5);
    let prior_cov = SMatrix::<f64, 2, 2>::identity() * 1.0;

    let mut ukf = Ukf2D::<f64>::with_defaults(prior_mean, prior_cov);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-9);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    for _step in 0..20 {
        ukf.predict(&transition, 1.0).expect("predict ok");
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 2>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ukf.update(&model, &z).expect("update ok");
        }
    }

    assert_relative_eq!(ukf.state(), truth, epsilon = 1e-3);
}

#[test]
fn ukf_3d_warm_start_converges_tightly() {
    let truth = Vector3::new(1.5_f64, -2.0, 4.2);
    let prior_mean = Vector3::new(1.0_f64, -1.5, 4.0);
    let prior_cov = SMatrix::<f64, 3, 3>::identity() * 1.0;

    let mut ukf = Ukf3D::<f64>::with_defaults(prior_mean, prior_cov);
    let transition = IdentityTransition::<f64, 3>::with_uniform_noise(1e-9);
    let anchors = [
        Vector3::new(0.0_f64, 0.0, 0.0),
        Vector3::new(10.0, 0.0, 0.0),
        Vector3::new(0.0, 10.0, 0.0),
        Vector3::new(0.0, 0.0, 10.0),
        Vector3::new(10.0, 10.0, 10.0),
    ];

    for _step in 0..20 {
        ukf.predict(&transition, 1.0).expect("predict ok");
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 3>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ukf.update(&model, &z).expect("update ok");
        }
    }

    assert_relative_eq!(ukf.state(), truth, epsilon = 5e-3);
}

/// UKF cold start should outperform EKF cold start because sigma points
/// sample the curvature of `h` rather than relying on a single linearization.
#[test]
fn ukf_2d_cold_start_converges_better_than_ekf_bias_floor() {
    let truth = Vector2::new(2.0_f64, 3.0);
    let prior_mean = Vector2::new(0.0_f64, 0.0);
    let prior_cov = SMatrix::<f64, 2, 2>::identity() * 100.0;

    let mut ukf = Ukf2D::<f64>::with_defaults(prior_mean, prior_cov);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-6);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    for _step in 0..30 {
        ukf.predict(&transition, 1.0).expect("predict ok");
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 2>::new(anchor, 0.01_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            ukf.update(&model, &z).expect("update ok");
        }
    }

    let err = (ukf.state() - truth).norm();
    // EKF cold start was ~0.04. UKF should do better than 0.02.
    assert!(err < 0.02, "UKF cold-start error: {err}");
}

#[test]
fn ukf_predict_with_identity_and_zero_noise_preserves_state() {
    let state = Vector2::new(1.0_f64, 2.0);
    let cov = SMatrix::<f64, 2, 2>::identity();
    let mut ukf = Ukf2D::<f64>::with_defaults(state, cov);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.0);

    ukf.predict(&transition, 0.1).expect("predict ok");
    assert_relative_eq!(ukf.state(), state, epsilon = 1e-10);
    assert_relative_eq!(ukf.covariance(), cov, epsilon = 1e-10);
}
