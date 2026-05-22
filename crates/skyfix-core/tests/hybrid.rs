//! Integration tests for HybridTdoaAoa2D.

use approx::assert_relative_eq;
use nalgebra::Vector2;
use skyfix_core::{AoaBearing, HybridTdoaAoa2D, TdoaMeasurement};

#[test]
fn hybrid_2d_fuses_tdoa_and_aoa_recovers_target() {
    let truth = Vector2::new(2.0_f64, 3.0);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let tdoa_anchors = [Vector2::new(5.0_f64, 0.0), Vector2::new(0.0, 5.0)];
    let aoa_anchor = Vector2::new(5.0_f64, 5.0);

    let r_ref = (truth - a_ref).norm();
    let tdoas: Vec<_> = tdoa_anchors
        .iter()
        .map(|&a| TdoaMeasurement::new(a, a_ref, (truth - a).norm() - r_ref))
        .collect();
    let aoa_diff = truth - aoa_anchor;
    let aoas = [AoaBearing::new(aoa_anchor, aoa_diff.y.atan2(aoa_diff.x))];

    let hybrid = HybridTdoaAoa2D::<f64>::new(100, 1e-12);
    let initial = Vector2::new(1.5_f64, 2.5);
    let result = hybrid.iterate(initial, &tdoas, &aoas).expect("hybrid ok");
    assert_relative_eq!(result, truth, epsilon = 1e-9);
}

#[test]
fn hybrid_2d_with_only_tdoas_matches_foy_result() {
    use skyfix_core::FoyTdoa;
    let truth = Vector2::new(1.5_f64, 2.0);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(5.0_f64, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];
    let r_ref = (truth - a_ref).norm();
    let tdoas: Vec<_> = anchors
        .iter()
        .map(|&a| TdoaMeasurement::new(a, a_ref, (truth - a).norm() - r_ref))
        .collect();

    let hybrid = HybridTdoaAoa2D::<f64>::new(100, 1e-12);
    let foy = FoyTdoa::<f64, 2>::new(100, 1e-12);
    let initial = Vector2::new(1.0_f64, 1.0);
    let h_result = hybrid.iterate(initial, &tdoas, &[]).expect("hybrid ok");
    let f_result = foy.iterate(initial, &tdoas).expect("foy ok");
    assert_relative_eq!(h_result, f_result, epsilon = 1e-12);
    assert_relative_eq!(h_result, truth, epsilon = 1e-9);
}

#[test]
fn hybrid_2d_with_only_aoas_recovers_target() {
    let truth = Vector2::new(-0.5_f64, 2.3);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(3.0, 0.0),
        Vector2::new(0.0, 3.0),
    ];
    let aoas: Vec<_> = anchors
        .iter()
        .map(|&a| {
            let d = truth - a;
            AoaBearing::new(a, d.y.atan2(d.x))
        })
        .collect();

    let hybrid = HybridTdoaAoa2D::<f64>::new(100, 1e-12);
    let initial = Vector2::new(0.0_f64, 1.0);
    let result = hybrid.iterate(initial, &[], &aoas).expect("hybrid ok");
    assert_relative_eq!(result, truth, epsilon = 1e-9);
}

#[test]
fn hybrid_2d_rejects_insufficient_measurements() {
    let hybrid = HybridTdoaAoa2D::<f64>::new(10, 1e-6);
    let aoa = [AoaBearing::new(Vector2::new(0.0_f64, 0.0), 0.5)];
    let err = hybrid
        .iterate(Vector2::new(1.0, 1.0), &[], &aoa)
        .unwrap_err();
    assert!(matches!(
        err,
        skyfix_core::EstimationError::InsufficientMeasurements { needed: 2, got: 1 }
    ));
}
