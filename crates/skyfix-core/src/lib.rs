#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod aoa;
pub mod crlb;
pub mod ekf;
pub mod error;
pub mod estimator;
pub mod filter;
pub mod hybrid;
pub mod measurement;
pub mod numeric;
pub mod pf;
pub mod rssi;
pub mod tdoa;
pub mod toa;
pub mod ukf;

pub use error::EstimationError;
pub use estimator::Estimator;
pub use measurement::{AoaBearing, RssiSample, TdoaMeasurement, ToaMeasurement};

pub use aoa::StansfieldAoa;
pub use crlb::{CrlbAnalysis, CrlbBuilder};
pub use ekf::Ekf;
pub use filter::{IdentityTransition, ObservationModel, RangeAnchor, TransitionModel};
pub use hybrid::HybridTdoaAoa2D;
pub use pf::Pf;
pub use rssi::RssiPathLoss;
pub use tdoa::{ChanLinear2D, ChanLinear3D, FoyTdoa};
pub use toa::{ToaNls, ToaTrilateration};
pub use ukf::{Ukf, Ukf1D, Ukf2D, Ukf3D, Ukf4D, Ukf6D};
