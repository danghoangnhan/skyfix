//! Integration tests for closed-form ToA trilateration.

use approx::assert_relative_eq;
use nalgebra::{Vector2, Vector3};
use proptest::prelude::*;
use skyfix_core::{Estimator, ToaMeasurement, ToaTrilateration};

#[test]
fn trilateration_2d_equilateral_centroid() {
    let a1 = Vector2::new(0.0_f64, 0.0);
    let a2 = Vector2::new(1.0, 0.0);
    let a3 = Vector2::new(0.5, (3.0_f64).sqrt() / 2.0);
    let target = Vector2::new(0.5, (3.0_f64).sqrt() / 6.0);

    let estimator = ToaTrilateration::<f64, 2>::new();
    let result = estimator
        .estimate(&[
            ToaMeasurement::new(a1, (target - a1).norm()),
            ToaMeasurement::new(a2, (target - a2).norm()),
            ToaMeasurement::new(a3, (target - a3).norm()),
        ])
        .expect("estimation succeeded");

    assert_relative_eq!(result, target, epsilon = 1e-10);
}

#[test]
fn trilateration_3d_unit_tetrahedron_center() {
    let a1 = Vector3::new(0.0_f64, 0.0, 0.0);
    let a2 = Vector3::new(1.0, 0.0, 0.0);
    let a3 = Vector3::new(0.0, 1.0, 0.0);
    let a4 = Vector3::new(0.0, 0.0, 1.0);
    let target = Vector3::new(0.25_f64, 0.25, 0.25);

    let r = |a: Vector3<f64>| (target - a).norm();
    let estimator = ToaTrilateration::<f64, 3>::new();
    let result = estimator
        .estimate(&[
            ToaMeasurement::new(a1, r(a1)),
            ToaMeasurement::new(a2, r(a2)),
            ToaMeasurement::new(a3, r(a3)),
            ToaMeasurement::new(a4, r(a4)),
        ])
        .expect("estimation succeeded");

    assert_relative_eq!(result, target, epsilon = 1e-10);
}

#[test]
fn trilateration_2d_with_extra_measurements_ignores_them() {
    // Currently only N+1 measurements participate; extras should not break the solver.
    let a1 = Vector2::new(0.0_f64, 0.0);
    let a2 = Vector2::new(1.0, 0.0);
    let a3 = Vector2::new(0.5, (3.0_f64).sqrt() / 2.0);
    let a4 = Vector2::new(-1.0, 1.0);
    let target = Vector2::new(0.3_f64, 0.4);

    let r = |a: Vector2<f64>| (target - a).norm();
    let estimator = ToaTrilateration::<f64, 2>::new();
    let result = estimator
        .estimate(&[
            ToaMeasurement::new(a1, r(a1)),
            ToaMeasurement::new(a2, r(a2)),
            ToaMeasurement::new(a3, r(a3)),
            ToaMeasurement::new(a4, r(a4)),
        ])
        .expect("estimation succeeded");

    assert_relative_eq!(result, target, epsilon = 1e-10);
}

// Invariant: noise-free ToA trilateration must recover any target position
// when given a well-conditioned anchor set. We use a fixed equilateral
// triangle as the anchor configuration and let the target roam.
proptest! {
    #[test]
    fn trilateration_2d_recovers_noise_free_target(
        tx in -10.0_f64..10.0,
        ty in -10.0_f64..10.0,
    ) {
        let a1 = Vector2::new(0.0_f64, 0.0);
        let a2 = Vector2::new(1.0, 0.0);
        let a3 = Vector2::new(0.5, (3.0_f64).sqrt() / 2.0);
        let target = Vector2::new(tx, ty);

        let m = [
            ToaMeasurement::new(a1, (target - a1).norm()),
            ToaMeasurement::new(a2, (target - a2).norm()),
            ToaMeasurement::new(a3, (target - a3).norm()),
        ];

        let result = ToaTrilateration::<f64, 2>::new()
            .estimate(&m)
            .expect("estimation succeeded");
        prop_assert!((result - target).norm() < 1e-8);
    }

    #[test]
    fn trilateration_3d_recovers_noise_free_target(
        tx in -5.0_f64..5.0,
        ty in -5.0_f64..5.0,
        tz in -5.0_f64..5.0,
    ) {
        let a1 = Vector3::new(0.0_f64, 0.0, 0.0);
        let a2 = Vector3::new(1.0, 0.0, 0.0);
        let a3 = Vector3::new(0.0, 1.0, 0.0);
        let a4 = Vector3::new(0.0, 0.0, 1.0);
        let target = Vector3::new(tx, ty, tz);

        let r = |a: Vector3<f64>| (target - a).norm();
        let m = [
            ToaMeasurement::new(a1, r(a1)),
            ToaMeasurement::new(a2, r(a2)),
            ToaMeasurement::new(a3, r(a3)),
            ToaMeasurement::new(a4, r(a4)),
        ];

        let result = ToaTrilateration::<f64, 3>::new()
            .estimate(&m)
            .expect("estimation succeeded");
        prop_assert!((result - target).norm() < 1e-8);
    }
}
