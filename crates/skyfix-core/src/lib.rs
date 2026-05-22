#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod aoa;
pub mod error;
pub mod estimator;
pub mod measurement;
pub mod numeric;
pub mod rssi;
pub mod tdoa;
pub mod toa;

pub use error::EstimationError;
pub use estimator::Estimator;
pub use measurement::{AoaBearing, RssiSample, TdoaMeasurement, ToaMeasurement};

pub use aoa::StansfieldAoa;
pub use rssi::RssiPathLoss;
pub use tdoa::{ChanLinear2D, ChanLinear3D, FoyTdoa};
pub use toa::{ToaNls, ToaTrilateration};
