//! Hardware-agnostic UWB ranging adapter for skyfix.
//!
//! This crate provides the **data layer** between a UWB ranging stack
//! (typically [`dw3000-ng`] driving a Decawave DW3000-family chip) and
//! [`skyfix-core`]'s position estimators:
//!
//! - [`UwbRange`] — a single anchor-to-target range measurement with the
//!   anchor's EUI-64 address, its position, and the estimated distance.
//! - Conversion helpers from [`UwbRange`] to
//!   [`ToaMeasurement`](skyfix_core::ToaMeasurement) and from
//!   pairs of `UwbRange` to [`TdoaMeasurement`](skyfix_core::TdoaMeasurement).
//!
//! # What's *not* here yet
//!
//! Actual driving of a DW3000 chip (SPI setup, message exchange, calibration,
//! interrupt handling) lands in Phase 7b once the project has an embedded
//! target wired up. The split keeps this crate compiling on every target —
//! it's `#![no_std]` with no I/O — so the conversion logic can be exercised
//! from desktop tests today.
//!
//! [`dw3000-ng`]: https://docs.rs/dw3000-ng

#![no_std]

use nalgebra::SVector;
use skyfix_core::{TdoaMeasurement, ToaMeasurement};

/// Anchor identifier — typically the IEEE 802.15.4 EUI-64 burned into the
/// anchor's DW3000 OTP. Opaque to skyfix; used only for bookkeeping and
/// the `Display` impl.
pub type AnchorAddr = u64;

/// One anchor-to-target range observation, the raw output of a UWB ranging
/// exchange (single-sided or double-sided TWR).
///
/// `N` is the spatial dimension of the anchor coordinate system (2 or 3).
/// The range is in **meters** in the same frame as `anchor_position`; the
/// caller is responsible for any antenna-delay or speed-of-light calibration
/// before constructing this value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UwbRange<const N: usize> {
    /// Anchor EUI-64.
    pub anchor: AnchorAddr,
    /// Anchor position in the target's coordinate frame.
    pub anchor_position: SVector<f64, N>,
    /// Estimated range from anchor to target in meters.
    pub range_m: f64,
    /// Optional timestamp of the ranging event in microseconds, in whatever
    /// epoch the caller maintains (typically a monotonic counter from the
    /// MCU's RTC or DW3000 system time).
    pub timestamp_us: Option<u64>,
}

impl<const N: usize> UwbRange<N> {
    /// Construct a new range observation. `timestamp_us` defaults to `None`;
    /// add one with [`Self::with_timestamp`].
    pub const fn new(anchor: AnchorAddr, anchor_position: SVector<f64, N>, range_m: f64) -> Self {
        Self {
            anchor,
            anchor_position,
            range_m,
            timestamp_us: None,
        }
    }

    /// Attach a timestamp.
    pub const fn with_timestamp(mut self, t_us: u64) -> Self {
        self.timestamp_us = Some(t_us);
        self
    }

    /// Convert to a [`skyfix_core::ToaMeasurement`]. Drops the anchor ID and
    /// any timestamp — both are unused by the closed-form / NLS / Bayesian
    /// estimators in `skyfix-core`.
    pub fn to_toa(&self) -> ToaMeasurement<f64, N> {
        ToaMeasurement::new(self.anchor_position, self.range_m)
    }
}

/// Convert a pair of UWB ranges from different anchors into a TDoA
/// measurement. The `reference` range provides `anchor_ref` and its range is
/// subtracted from the `other` range to form the range difference Chan and
/// Foy estimators consume.
///
/// All TDoA measurements that feed [`skyfix_core::ChanLinear2D`] /
/// [`skyfix_core::ChanLinear3D`] in a single solve **must** share the same
/// `reference` — Chan's algorithm encodes that assumption directly.
pub fn pair_to_tdoa<const N: usize>(
    reference: &UwbRange<N>,
    other: &UwbRange<N>,
) -> TdoaMeasurement<f64, N> {
    TdoaMeasurement::new(
        other.anchor_position,
        reference.anchor_position,
        other.range_m - reference.range_m,
    )
}

/// Convert a slice of UWB ranges sharing a single reference anchor into a
/// batch of TDoA measurements suitable for Chan / Foy. The first range in
/// `ranges` is taken as the reference; the rest become TDoA pairs against it.
///
/// `output` must have capacity `ranges.len() - 1`. Returns the number of
/// measurements written, or `None` if `ranges.len() < 2`.
///
/// Designed for `no_std` usage: caller supplies the output buffer (e.g. a
/// stack-allocated array or a `heapless::Vec`).
pub fn ranges_to_tdoa_batch<'o, const N: usize>(
    ranges: &[UwbRange<N>],
    output: &'o mut [TdoaMeasurement<f64, N>],
) -> Option<&'o [TdoaMeasurement<f64, N>]> {
    if ranges.len() < 2 {
        return None;
    }
    let needed = ranges.len() - 1;
    if output.len() < needed {
        return None;
    }
    let reference = &ranges[0];
    for (i, other) in ranges[1..].iter().enumerate() {
        output[i] = pair_to_tdoa(reference, other);
    }
    Some(&output[..needed])
}
