//! Angle-of-Arrival (AoA) estimators.

use crate::error::EstimationError;
use crate::estimator::Estimator;
use crate::measurement::AoaBearing;
use crate::numeric::solve_normal_equations;
use core::marker::PhantomData;
use nalgebra::{RealField, Vector2};

/// 2D bearings-only triangulation via Stansfield's least-squares method.
///
/// Each bearing defines a line passing through its anchor; the estimator
/// returns the LSQ intersection. Requires at least 2 non-parallel bearings.
///
/// Reference: Stansfield, R. G. (1947). "Statistical theory of DF fixing."
/// Journal of the IEE — Part IIIA: Radiocommunication, 94(15), 762-770.
pub struct StansfieldAoa<T: RealField + Copy> {
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy> StansfieldAoa<T> {
    /// Construct a new Stansfield solver.
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy> Default for StansfieldAoa<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> Estimator<T, 2> for StansfieldAoa<T> {
    type Measurement = AoaBearing<T>;

    fn estimate(&self, m: &[Self::Measurement]) -> Result<Vector2<T>, EstimationError> {
        if m.len() < 2 {
            return Err(EstimationError::InsufficientMeasurements {
                needed: 2,
                got: m.len(),
            });
        }
        let rows = m.iter().map(|meas| {
            let s = meas.bearing.sin();
            let c = meas.bearing.cos();
            let jac = Vector2::new(s, -c);
            let rhs = meas.anchor[0] * s - meas.anchor[1] * c;
            (jac, T::one(), rhs)
        });
        solve_normal_equations(rows)
    }
}
