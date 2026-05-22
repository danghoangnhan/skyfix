//! Integration tests for ToaNls (overdetermined Gauss-Newton ToA).

use approx::assert_relative_eq;
use nalgebra::{Vector2, Vector3};
use skyfix_core::{Estimator, ToaMeasurement, ToaNls, ToaTrilateration};

#[test]
fn toa_nls_recovers_2d_target_from_centroid_initial() {
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(2.0, 0.0),
        Vector2::new(1.0, 2.0),
        Vector2::new(-1.0, 1.0),
        Vector2::new(0.5, -1.5),
    ];
    let target = Vector2::new(0.7_f64, 0.4);
    let m: Vec<_> = anchors
        .iter()
        .map(|&p| ToaMeasurement::new(p, (target - p).norm()))
        .collect();

    let nls = ToaNls::<f64, 2>::new(100, 1e-12);
    let result = nls.estimate(&m).expect("nls estimate ok");
    assert_relative_eq!(result, target, epsilon = 1e-8);
}

#[test]
fn toa_nls_pairs_with_trilateration_for_3d_overdetermined() {
    let anchors = [
        Vector3::new(0.0_f64, 0.0, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
        Vector3::new(0.0, 2.0, 0.0),
        Vector3::new(0.0, 0.0, 2.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector3::new(-1.0, 0.5, 2.0),
    ];
    let target = Vector3::new(0.5_f64, 0.3, 0.7);
    let m: Vec<_> = anchors
        .iter()
        .map(|&p| ToaMeasurement::new(p, (target - p).norm()))
        .collect();

    let initial = ToaTrilateration::<f64, 3>::new()
        .estimate(&m[..4])
        .expect("trilateration ok");
    let nls = ToaNls::<f64, 3>::new(100, 1e-12);
    let result = nls.iterate(initial, &m).expect("nls iterate ok");
    assert_relative_eq!(result, target, epsilon = 1e-9);
}

#[test]
fn toa_nls_handles_noise_free_with_default() {
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];
    let target = Vector2::new(2.0_f64, 3.0);
    let m: Vec<_> = anchors
        .iter()
        .map(|&p| ToaMeasurement::new(p, (target - p).norm()))
        .collect();

    let nls = ToaNls::<f64, 2>::default();
    let result = nls.estimate(&m).expect("default nls ok");
    assert_relative_eq!(result, target, epsilon = 1e-4);
}
