//! Time-of-Arrival (ToA) estimators.

use crate::error::EstimationError;
use crate::estimator::Estimator;
use crate::measurement::ToaMeasurement;
use crate::numeric::{gauss_newton, solve_linear_system};
use core::marker::PhantomData;
use nalgebra::{ComplexField, RealField, SMatrix, SVector};

// ============================================================================
// Closed-form trilateration (Phase 1)
// ============================================================================

/// Closed-form ToA trilateration solver for `N`-dimensional space.
///
/// Requires at least `N + 1` measurements — 3 anchors in 2D, 4 in 3D. When
/// more measurements are supplied, only the first `N + 1` participate; use
/// [`ToaNls`] for the overdetermined least-squares variant.
///
/// # Algorithm
///
/// The range equations `||x - a_i||² = r_i²` are quadratic. Subtracting the
/// equation for `i = 0` from each of the others eliminates `||x||²` and
/// produces a square `N × N` linear system solved by Gauss elimination
/// with partial pivoting. Singular when chosen anchors are linearly
/// dependent (colinear in 2D, coplanar in 3D).
pub struct ToaTrilateration<T: RealField + Copy, const N: usize> {
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy, const N: usize> ToaTrilateration<T, N> {
    /// Construct a new trilateration solver.
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy, const N: usize> Default for ToaTrilateration<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy, const N: usize> Estimator<T, N> for ToaTrilateration<T, N> {
    type Measurement = ToaMeasurement<T, N>;

    fn estimate(&self, m: &[Self::Measurement]) -> Result<SVector<T, N>, EstimationError> {
        let needed = N + 1;
        if m.len() < needed {
            return Err(EstimationError::InsufficientMeasurements {
                needed,
                got: m.len(),
            });
        }

        let two = T::one() + T::one();
        let a0 = m[0].anchor;
        let r0_sq = m[0].range * m[0].range;
        let a0_sq = a0.dot(&a0);

        let mut a_mat = SMatrix::<T, N, N>::zeros();
        let mut b_vec = SVector::<T, N>::zeros();

        for i in 0..N {
            let mi = &m[i + 1];
            let row = (a0 - mi.anchor) * two;
            for j in 0..N {
                a_mat[(i, j)] = row[j];
            }
            b_vec[i] = mi.range * mi.range - r0_sq - mi.anchor.dot(&mi.anchor) + a0_sq;
        }

        let x = solve_linear_system(a_mat, b_vec)?;

        if !x.iter().all(|v| ComplexField::is_finite(v)) {
            return Err(EstimationError::NonFinite);
        }

        Ok(x)
    }
}

// ============================================================================
// Overdetermined Gauss-Newton (Phase 2)
// ============================================================================

/// Iterative Gauss-Newton ToA solver for overdetermined anchor sets.
///
/// Requires at least `N` measurements. The default [`Estimator::estimate`]
/// impl uses the anchor centroid as the initial estimate; [`Self::iterate`]
/// accepts an explicit initial — typically the result of running
/// [`ToaTrilateration`] on the first `N + 1` measurements first.
pub struct ToaNls<T: RealField + Copy, const N: usize> {
    max_iters: usize,
    tol: T,
}

impl<T: RealField + Copy, const N: usize> ToaNls<T, N> {
    /// Construct an NLS solver with the given iteration cap and convergence tolerance.
    pub const fn new(max_iters: usize, tol: T) -> Self {
        Self { max_iters, tol }
    }

    /// Run Gauss-Newton starting from an explicit initial position guess.
    pub fn iterate(
        &self,
        initial: SVector<T, N>,
        measurements: &[ToaMeasurement<T, N>],
    ) -> Result<SVector<T, N>, EstimationError> {
        if measurements.len() < N {
            return Err(EstimationError::InsufficientMeasurements {
                needed: N,
                got: measurements.len(),
            });
        }
        gauss_newton(initial, measurements, self.max_iters, self.tol, |x, m| {
            let diff = x - m.anchor;
            let r = diff.norm();
            if r == T::zero() {
                (SVector::zeros(), T::zero())
            } else {
                (diff / r, m.range - r)
            }
        })
    }
}

impl<T: RealField + Copy, const N: usize> Default for ToaNls<T, N> {
    fn default() -> Self {
        Self::new(50, nalgebra::convert(1e-6))
    }
}

impl<T: RealField + Copy, const N: usize> Estimator<T, N> for ToaNls<T, N> {
    type Measurement = ToaMeasurement<T, N>;

    fn estimate(&self, m: &[Self::Measurement]) -> Result<SVector<T, N>, EstimationError> {
        if m.is_empty() {
            return Err(EstimationError::InsufficientMeasurements { needed: 1, got: 0 });
        }
        let mut centroid = SVector::<T, N>::zeros();
        for k in m {
            centroid += k.anchor;
        }
        let n: T = nalgebra::convert(m.len() as f64);
        let initial = centroid / n;
        self.iterate(initial, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn trilateration_rejects_insufficient_measurements() {
        let estimator = ToaTrilateration::<f64, 2>::new();
        let m = [
            ToaMeasurement::new(Vector2::new(0.0, 0.0), 1.0),
            ToaMeasurement::new(Vector2::new(1.0, 0.0), 1.0),
        ];
        assert_eq!(
            estimator.estimate(&m).unwrap_err(),
            EstimationError::InsufficientMeasurements { needed: 3, got: 2 }
        );
    }

    #[test]
    fn trilateration_rejects_colinear_anchors() {
        let estimator = ToaTrilateration::<f64, 2>::new();
        let m = [
            ToaMeasurement::new(Vector2::new(0.0, 0.0), 1.0),
            ToaMeasurement::new(Vector2::new(1.0, 0.0), 1.0),
            ToaMeasurement::new(Vector2::new(2.0, 0.0), 1.0),
        ];
        assert_eq!(
            estimator.estimate(&m).unwrap_err(),
            EstimationError::SingularSystem
        );
    }

    #[test]
    fn nls_rejects_empty_measurements() {
        let nls = ToaNls::<f64, 2>::default();
        assert_eq!(
            nls.estimate(&[]).unwrap_err(),
            EstimationError::InsufficientMeasurements { needed: 1, got: 0 }
        );
    }
}
