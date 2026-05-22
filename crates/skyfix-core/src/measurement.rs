//! Measurement types consumed by skyfix-core estimators.

use nalgebra::{RealField, SVector, Vector2};

/// Time-of-Arrival (ToA) measurement: an anchor position paired with the
/// measured range from that anchor to the target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToaMeasurement<T: RealField + Copy, const N: usize> {
    /// Position of the anchor in `N`-dimensional space.
    pub anchor: SVector<T, N>,
    /// Measured Euclidean range from the anchor to the target.
    pub range: T,
}

impl<T: RealField + Copy, const N: usize> ToaMeasurement<T, N> {
    /// Construct a new ToA measurement.
    pub fn new(anchor: SVector<T, N>, range: T) -> Self {
        Self { anchor, range }
    }
}

/// Time-Difference-of-Arrival (TDoA) measurement.
///
/// For a target at position `x`,
/// `range_diff = ||x - anchor|| - ||x - anchor_ref||`.
/// Chan's closed-form algorithm assumes a single `anchor_ref` shared across
/// the whole measurement batch; the iterative `FoyTdoa` does not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdoaMeasurement<T: RealField + Copy, const N: usize> {
    /// Secondary anchor.
    pub anchor: SVector<T, N>,
    /// Reference anchor.
    pub anchor_ref: SVector<T, N>,
    /// `||x - anchor|| - ||x - anchor_ref||` in meters.
    pub range_diff: T,
}

impl<T: RealField + Copy, const N: usize> TdoaMeasurement<T, N> {
    /// Construct a new TDoA measurement.
    pub fn new(anchor: SVector<T, N>, anchor_ref: SVector<T, N>, range_diff: T) -> Self {
        Self {
            anchor,
            anchor_ref,
            range_diff,
        }
    }
}

/// 2D Angle-of-Arrival bearing measurement.
///
/// `bearing` is the angle in radians from the anchor toward the target,
/// measured counter-clockwise from the positive x-axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoaBearing<T: RealField + Copy> {
    /// Anchor position in 2D.
    pub anchor: Vector2<T>,
    /// Bearing in radians.
    pub bearing: T,
}

impl<T: RealField + Copy> AoaBearing<T> {
    /// Construct a new AoA bearing measurement.
    pub fn new(anchor: Vector2<T>, bearing: T) -> Self {
        Self { anchor, bearing }
    }
}

/// Received Signal Strength Indicator (RSSI) sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RssiSample<T: RealField + Copy, const N: usize> {
    /// Anchor position in `N`-dimensional space.
    pub anchor: SVector<T, N>,
    /// Received power in dBm.
    pub rssi_dbm: T,
}

impl<T: RealField + Copy, const N: usize> RssiSample<T, N> {
    /// Construct a new RSSI sample.
    pub fn new(anchor: SVector<T, N>, rssi_dbm: T) -> Self {
        Self { anchor, rssi_dbm }
    }
}
