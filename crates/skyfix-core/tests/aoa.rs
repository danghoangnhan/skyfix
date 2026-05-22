//! Integration tests for Stansfield 2D bearings-only triangulation.

use approx::assert_relative_eq;
use nalgebra::Vector2;
use skyfix_core::{AoaBearing, Estimator, StansfieldAoa};

fn bearing(anchor: Vector2<f64>, target: Vector2<f64>) -> f64 {
    let d = target - anchor;
    d.y.atan2(d.x)
}

#[test]
fn stansfield_2d_two_anchors_exact_intersection() {
    let target = Vector2::new(1.0_f64, 1.0);
    let a1 = Vector2::new(0.0_f64, 0.0);
    let a2 = Vector2::new(2.0, 0.0);
    let m = [
        AoaBearing::new(a1, bearing(a1, target)),
        AoaBearing::new(a2, bearing(a2, target)),
    ];
    let result = StansfieldAoa::<f64>::new().estimate(&m).expect("ok");
    assert_relative_eq!(result, target, epsilon = 1e-10);
}

#[test]
fn stansfield_2d_overdetermined_recovers_target() {
    let target = Vector2::new(-0.5_f64, 2.3);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(3.0, 0.0),
        Vector2::new(0.0, 3.0),
        Vector2::new(3.0, 3.0),
        Vector2::new(1.0, -2.0),
    ];
    let m: Vec<_> = anchors
        .iter()
        .map(|&a| AoaBearing::new(a, bearing(a, target)))
        .collect();
    let result = StansfieldAoa::<f64>::new().estimate(&m).expect("ok");
    assert_relative_eq!(result, target, epsilon = 1e-9);
}

#[test]
fn stansfield_2d_rejects_single_bearing() {
    let m = [AoaBearing::new(Vector2::new(0.0_f64, 0.0), 0.5)];
    let err = StansfieldAoa::<f64>::new().estimate(&m).unwrap_err();
    assert!(matches!(
        err,
        skyfix_core::EstimationError::InsufficientMeasurements { needed: 2, got: 1 }
    ));
}
