//! Received Signal Strength Indicator (RSSI) path-loss model.
//!
//! The log-distance model predicts received power as
//!
//! ```text
//!     RSSI(d) = RSSI(d₀) − 10 n log₁₀(d / d₀)
//! ```
//!
//! Inverting yields a range estimate from a measured RSSI value, which can
//! then be fed into a ToA estimator ([`crate::ToaTrilateration`] or
//! [`crate::toa::ToaNls`]) for the position fix.

use nalgebra::RealField;

/// Calibration for the log-distance path-loss model.
///
/// Distances are in meters; RSSI values are in dBm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RssiPathLoss<T: RealField + Copy> {
    /// Reference distance (typically 1 m).
    pub d0: T,
    /// Measured RSSI at the reference distance, in dBm.
    pub rssi_at_d0: T,
    /// Path-loss exponent. Free space: 2.0; indoor LOS: 1.6–1.8; indoor NLOS: 2.7–4.3.
    pub n: T,
}

impl<T: RealField + Copy> RssiPathLoss<T> {
    /// Construct a new path-loss calibration.
    pub const fn new(d0: T, rssi_at_d0: T, n: T) -> Self {
        Self { d0, rssi_at_d0, n }
    }

    /// Convert a measured RSSI in dBm into an estimated range in meters.
    pub fn range_from_rssi(&self, rssi_dbm: T) -> T {
        let ten: T = nalgebra::convert(10.0);
        let exp_val = (self.rssi_at_d0 - rssi_dbm) / (ten * self.n);
        self.d0 * ten.powf(exp_val)
    }

    /// Inverse: predict the RSSI that would be observed at a given range.
    pub fn rssi_at_range(&self, range: T) -> T {
        let ten: T = nalgebra::convert(10.0);
        let ratio = range / self.d0;
        self.rssi_at_d0 - ten * self.n * ratio.log(ten)
    }
}
