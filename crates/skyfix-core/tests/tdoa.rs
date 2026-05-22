//! Integration tests for Chan linear-stage and Foy Taylor-series TDoA.

use approx::assert_relative_eq;
use nalgebra::{Vector2, Vector3};
use skyfix_core::{ChanLinear2D, ChanLinear3D, Estimator, FoyTdoa, TdoaMeasurement};

fn tdoa_2d_set(
    target: Vector2<f64>,
    a_ref: Vector2<f64>,
    anchors: &[Vector2<f64>],
) -> Vec<TdoaMeasurement<f64, 2>> {
    let r_ref = (target - a_ref).norm();
    anchors
        .iter()
        .map(|&a| TdoaMeasurement::new(a, a_ref, (target - a).norm() - r_ref))
        .collect()
}

fn tdoa_3d_set(
    target: Vector3<f64>,
    a_ref: Vector3<f64>,
    anchors: &[Vector3<f64>],
) -> Vec<TdoaMeasurement<f64, 3>> {
    let r_ref = (target - a_ref).norm();
    anchors
        .iter()
        .map(|&a| TdoaMeasurement::new(a, a_ref, (target - a).norm() - r_ref))
        .collect()
}

#[test]
fn chan_2d_recovers_target() {
    let target = Vector2::new(1.0_f64, 1.5);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(5.0_f64, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];
    let m = tdoa_2d_set(target, a_ref, &anchors);
    let result = ChanLinear2D::<f64>::new().estimate(&m).expect("chan ok");
    assert_relative_eq!(result, target, epsilon = 1e-9);
}

#[test]
fn chan_3d_recovers_target() {
    let target = Vector3::new(0.7_f64, 1.0, 1.3);
    let a_ref = Vector3::new(0.0_f64, 0.0, 0.0);
    let anchors = [
        Vector3::new(5.0_f64, 0.0, 0.0),
        Vector3::new(0.0, 5.0, 0.0),
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(5.0, 5.0, 5.0),
    ];
    let m = tdoa_3d_set(target, a_ref, &anchors);
    let result = ChanLinear3D::<f64>::new().estimate(&m).expect("chan ok");
    assert_relative_eq!(result, target, epsilon = 1e-9);
}

#[test]
fn chan_2d_rejects_insufficient_measurements() {
    let target = Vector2::new(1.0_f64, 1.0);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [Vector2::new(5.0_f64, 0.0)];
    let m = tdoa_2d_set(target, a_ref, &anchors);
    let err = ChanLinear2D::<f64>::new().estimate(&m).unwrap_err();
    assert!(matches!(
        err,
        skyfix_core::EstimationError::InsufficientMeasurements { needed: 3, .. }
    ));
}

#[test]
fn foy_refines_chan_initial_estimate() {
    let target = Vector2::new(1.5_f64, 2.0);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(5.0_f64, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
        Vector2::new(-2.0, 3.0),
    ];
    let m = tdoa_2d_set(target, a_ref, &anchors);

    let initial = ChanLinear2D::<f64>::new()
        .estimate(&m)
        .expect("chan initial ok");
    let foy = FoyTdoa::<f64, 2>::new(100, 1e-12);
    let result = foy.iterate(initial, &m).expect("foy ok");
    assert_relative_eq!(result, target, epsilon = 1e-9);
}

#[test]
fn foy_default_estimate_from_centroid_initial() {
    let target = Vector2::new(0.5_f64, 0.8);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(2.0_f64, 0.0),
        Vector2::new(0.0, 2.0),
        Vector2::new(2.0, 2.0),
        Vector2::new(-1.0, 1.0),
    ];
    let m = tdoa_2d_set(target, a_ref, &anchors);
    let result = FoyTdoa::<f64, 2>::default().estimate(&m).expect("foy ok");
    assert_relative_eq!(result, target, epsilon = 1e-4);
}
