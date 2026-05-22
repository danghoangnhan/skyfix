//! Integration tests for the log-distance RSSI path-loss model.

use approx::assert_relative_eq;
use nalgebra::Vector2;
use skyfix_core::{Estimator, RssiPathLoss, ToaMeasurement, ToaTrilateration};

#[test]
fn rssi_at_reference_distance_returns_d0() {
    let model = RssiPathLoss::<f64>::new(1.0, -40.0, 2.0);
    assert_relative_eq!(model.range_from_rssi(-40.0), 1.0, epsilon = 1e-12);
}

#[test]
fn rssi_free_space_doubling_distance_loses_6db() {
    // For n=2, 10·2·log10(2) ≈ 6.02 dB. At RSSI = -46 dBm with reference
    // -40 dBm at 1 m, the predicted range should be ~2 m.
    let model = RssiPathLoss::<f64>::new(1.0, -40.0, 2.0);
    assert_relative_eq!(model.range_from_rssi(-46.0), 2.0, epsilon = 1e-2);
}

#[test]
fn rssi_round_trip_for_several_ranges_and_exponents() {
    for n in &[1.8_f64, 2.0, 2.7, 3.5, 4.0] {
        let model = RssiPathLoss::<f64>::new(1.0, -40.0, *n);
        for &d in &[0.5_f64, 1.0, 2.5, 10.0, 50.0] {
            let r = model.rssi_at_range(d);
            let back = model.range_from_rssi(r);
            assert_relative_eq!(d, back, epsilon = 1e-9);
        }
    }
}

#[test]
fn rssi_pipeline_with_trilateration_recovers_target() {
    // Synthetic: place anchors, compute true ranges, encode to RSSI via the
    // model, decode back to ranges, run trilateration — should be exact.
    let model = RssiPathLoss::<f64>::new(1.0, -40.0, 2.5);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(4.0, 0.0),
        Vector2::new(2.0, 4.0),
    ];
    let target = Vector2::new(1.5_f64, 2.0);
    let measurements: Vec<_> = anchors
        .iter()
        .map(|&a| {
            let true_range = (target - a).norm();
            let rssi = model.rssi_at_range(true_range);
            let estimated_range = model.range_from_rssi(rssi);
            ToaMeasurement::new(a, estimated_range)
        })
        .collect();

    let result = ToaTrilateration::<f64, 2>::new()
        .estimate(&measurements)
        .expect("trilateration ok");
    assert_relative_eq!(result, target, epsilon = 1e-9);
}
