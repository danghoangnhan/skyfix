//! Single-shot position estimator trait.

use crate::error::EstimationError;
use nalgebra::{RealField, SVector};

/// A single-shot estimator that turns a batch of measurements into a position.
///
/// Implementors choose their own measurement type — typically one of the
/// per-modality structs in [`crate::measurement`] (e.g. [`crate::ToaMeasurement`]).
///
/// Recursive estimators (Kalman variants, particle filters) live behind the
/// separate `Filter` trait (added in Phase 3).
pub trait Estimator<T: RealField + Copy, const N: usize> {
    /// Measurement type consumed by this estimator.
    type Measurement;

    /// Estimate the target position from the supplied measurements.
    ///
    /// # Errors
    /// Returns [`EstimationError`] when measurements are insufficient or the
    /// numerical system is degenerate.
    fn estimate(
        &self,
        measurements: &[Self::Measurement],
    ) -> Result<SVector<T, N>, EstimationError>;
}
