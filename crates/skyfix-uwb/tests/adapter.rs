//! Integration tests for skyfix-uwb adapter types.

use approx::assert_relative_eq;
use nalgebra::{Vector2, Vector3};
use skyfix_core::{ChanLinear2D, Estimator, ToaTrilateration};
use skyfix_uwb::{pair_to_tdoa, ranges_to_tdoa_batch, UwbRange};

/// A typical hardware flow: collect 4 DW3000 DSTwR ranges from 4 anchors,
/// convert to ToA, hand off to the trilateration solver. Validates against a
/// known target position.
#[test]
fn uwb_ranges_drive_toa_trilateration_2d() {
    let target = Vector2::new(2.0_f64, 3.0);
    let raw = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    let ranges: [UwbRange<2>; 4] = [
        UwbRange::new(0x1111_1111_1111_1111, raw[0], (target - raw[0]).norm()),
        UwbRange::new(0x2222_2222_2222_2222, raw[1], (target - raw[1]).norm()),
        UwbRange::new(0x3333_3333_3333_3333, raw[2], (target - raw[2]).norm()),
        UwbRange::new(0x4444_4444_4444_4444, raw[3], (target - raw[3]).norm()),
    ];

    let toa: Vec<_> = ranges.iter().map(UwbRange::to_toa).collect();
    let estimate = ToaTrilateration::<f64, 2>::new()
        .estimate(&toa)
        .expect("trilateration ok");
    assert_relative_eq!(estimate, target, epsilon = 1e-9);
}

#[test]
fn ranges_to_tdoa_batch_drives_chan_2d() {
    let target = Vector2::new(1.5_f64, 2.0);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(5.0_f64, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    let ranges = [
        UwbRange::new(0xAAAA, a_ref, (target - a_ref).norm()),
        UwbRange::new(0xBBBB, anchors[0], (target - anchors[0]).norm()),
        UwbRange::new(0xCCCC, anchors[1], (target - anchors[1]).norm()),
        UwbRange::new(0xDDDD, anchors[2], (target - anchors[2]).norm()),
    ];

    // Stack-allocated output buffer — no_std-friendly.
    let zero_anchor = Vector2::new(0.0_f64, 0.0);
    let mut tdoa_buf: [skyfix_core::TdoaMeasurement<f64, 2>; 3] =
        core::array::from_fn(|_| skyfix_core::TdoaMeasurement::new(zero_anchor, zero_anchor, 0.0));
    let tdoa = ranges_to_tdoa_batch(&ranges, &mut tdoa_buf).expect("batch ok");
    assert_eq!(tdoa.len(), 3);

    let estimate = ChanLinear2D::<f64>::new().estimate(tdoa).expect("chan ok");
    assert_relative_eq!(estimate, target, epsilon = 1e-9);
}

#[test]
fn pair_to_tdoa_signs_correctly() {
    // Reference closer to target → other range is larger → range_diff > 0.
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let a_other = Vector2::new(10.0, 0.0);
    let target = Vector2::new(1.0, 0.0);

    let ref_r = UwbRange::new(0x01, a_ref, (target - a_ref).norm());
    let other_r = UwbRange::new(0x02, a_other, (target - a_other).norm());

    let tdoa = pair_to_tdoa(&ref_r, &other_r);
    assert!(
        tdoa.range_diff > 0.0,
        "expected positive range diff, got {}",
        tdoa.range_diff
    );
    assert_relative_eq!(tdoa.range_diff, 9.0 - 1.0, epsilon = 1e-12);
}

#[test]
fn ranges_to_tdoa_batch_rejects_too_few_ranges() {
    let r = UwbRange::new(0xAA, Vector2::new(0.0_f64, 0.0), 1.0);
    let zero = Vector2::new(0.0_f64, 0.0);
    let mut buf: [skyfix_core::TdoaMeasurement<f64, 2>; 3] =
        core::array::from_fn(|_| skyfix_core::TdoaMeasurement::new(zero, zero, 0.0));
    assert!(ranges_to_tdoa_batch::<2>(&[r], &mut buf).is_none());
    assert!(ranges_to_tdoa_batch::<2>(&[], &mut buf).is_none());
}

#[test]
fn uwb_range_3d_with_timestamp_roundtrips() {
    let r = UwbRange::new(0xBEEF, Vector3::new(1.0_f64, 2.0, 3.0), 4.5).with_timestamp(123_456);
    assert_eq!(r.anchor, 0xBEEF);
    assert_eq!(r.timestamp_us, Some(123_456));
    assert_relative_eq!(r.range_m, 4.5, epsilon = 1e-12);

    let toa = r.to_toa();
    assert_relative_eq!(toa.range, 4.5, epsilon = 1e-12);
    assert_relative_eq!(toa.anchor, Vector3::new(1.0, 2.0, 3.0), epsilon = 1e-12);
}
