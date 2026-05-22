//! Time-Difference-of-Arrival (TDoA) estimators.

use crate::error::EstimationError;
use crate::estimator::Estimator;
use crate::measurement::TdoaMeasurement;
use crate::numeric::{gauss_newton, solve_normal_equations};
use core::marker::PhantomData;
use nalgebra::{ComplexField, RealField, SVector, Vector2, Vector3};

// ============================================================================
// Chan's closed-form (linear stage)
// ============================================================================

/// Chan's linear-stage TDoA solver for 2D positioning.
///
/// All measurements MUST share the same `anchor_ref` (the algorithm
/// introduces `R_0 = ||x - anchor_ref||` as a single extra unknown).
/// Requires at least 3 TDoA measurements.
///
/// Reference: Chan, Y. T., & Ho, K. C. (1994). "A simple and efficient
/// estimator for hyperbolic location." IEEE Transactions on Signal
/// Processing, 42(8), 1905-1915.
///
/// This is Chan's *stage 1*. The stage-2 maximum-likelihood refinement
/// needs a measurement-noise covariance matrix and lands in a later phase.
pub struct ChanLinear2D<T: RealField + Copy> {
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy> ChanLinear2D<T> {
    /// Construct a new Chan 2D solver.
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy> Default for ChanLinear2D<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> Estimator<T, 2> for ChanLinear2D<T> {
    type Measurement = TdoaMeasurement<T, 2>;

    fn estimate(&self, m: &[Self::Measurement]) -> Result<Vector2<T>, EstimationError> {
        chan_linear_solve::<T, 2, 3>(m)
    }
}

/// Chan's linear-stage TDoA solver for 3D positioning. See [`ChanLinear2D`]
/// for the full algorithm description. Requires at least 4 TDoA measurements.
pub struct ChanLinear3D<T: RealField + Copy> {
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy> ChanLinear3D<T> {
    /// Construct a new Chan 3D solver.
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: RealField + Copy> Default for ChanLinear3D<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> Estimator<T, 3> for ChanLinear3D<T> {
    type Measurement = TdoaMeasurement<T, 3>;

    fn estimate(&self, m: &[Self::Measurement]) -> Result<Vector3<T>, EstimationError> {
        chan_linear_solve::<T, 3, 4>(m)
    }
}

/// Build and solve the augmented `(N + 1)`-dim linear system for Chan's stage 1.
///
/// `NAUG` must equal `N + 1`. Use the [`ChanLinear2D`] / [`ChanLinear3D`]
/// wrappers — they enforce the right pairing.
fn chan_linear_solve<T: RealField + Copy, const N: usize, const NAUG: usize>(
    m: &[TdoaMeasurement<T, N>],
) -> Result<SVector<T, N>, EstimationError> {
    if m.len() < NAUG {
        return Err(EstimationError::InsufficientMeasurements {
            needed: NAUG,
            got: m.len(),
        });
    }
    let a_ref = m[0].anchor_ref;
    let a_ref_sq = a_ref.dot(&a_ref);
    let two = T::one() + T::one();

    let rows = m.iter().map(|meas| {
        let mut row = SVector::<T, NAUG>::zeros();
        for j in 0..N {
            row[j] = two * (a_ref[j] - meas.anchor[j]);
        }
        row[N] = -two * meas.range_diff;
        let rhs = meas.range_diff * meas.range_diff + a_ref_sq - meas.anchor.dot(&meas.anchor);
        (row, T::one(), rhs)
    });
    let aug = solve_normal_equations::<T, NAUG, _>(rows)?;

    if !aug.iter().all(|v| ComplexField::is_finite(v)) {
        return Err(EstimationError::NonFinite);
    }

    let mut pos = SVector::<T, N>::zeros();
    for i in 0..N {
        pos[i] = aug[i];
    }
    Ok(pos)
}

// ============================================================================
// Foy Taylor-series iterative TDoA (Gauss-Newton on range-diff residuals)
// ============================================================================

/// Foy's Taylor-series WLS for TDoA positioning.
///
/// Iterative; needs an initial estimate. Pair with [`ChanLinear2D`] /
/// [`ChanLinear3D`] for the natural Chan-into-Foy pipeline. The default
/// [`Estimator::estimate`] uses the centroid of all anchors as initial.
///
/// Reference: Foy, W. H. (1976). "Position-location solutions by Taylor-series
/// estimation." IEEE Transactions on Aerospace and Electronic Systems,
/// AES-12(2), 187-194.
pub struct FoyTdoa<T: RealField + Copy, const N: usize> {
    max_iters: usize,
    tol: T,
}

impl<T: RealField + Copy, const N: usize> FoyTdoa<T, N> {
    /// Construct a new Foy solver with the given iteration cap and tolerance.
    pub const fn new(max_iters: usize, tol: T) -> Self {
        Self { max_iters, tol }
    }

    /// Run Gauss-Newton starting from an explicit initial estimate.
    pub fn iterate(
        &self,
        initial: SVector<T, N>,
        measurements: &[TdoaMeasurement<T, N>],
    ) -> Result<SVector<T, N>, EstimationError> {
        if measurements.len() < N {
            return Err(EstimationError::InsufficientMeasurements {
                needed: N,
                got: measurements.len(),
            });
        }
        gauss_newton(
            initial,
            measurements,
            self.max_iters,
            self.tol,
            |x, meas| {
                let d_anchor = x - meas.anchor;
                let d_ref = x - meas.anchor_ref;
                let r_anchor = d_anchor.norm();
                let r_ref = d_ref.norm();
                if r_anchor == T::zero() || r_ref == T::zero() {
                    (SVector::zeros(), T::zero())
                } else {
                    let jac = d_anchor / r_anchor - d_ref / r_ref;
                    let predicted = r_anchor - r_ref;
                    (jac, meas.range_diff - predicted)
                }
            },
        )
    }
}

impl<T: RealField + Copy, const N: usize> Default for FoyTdoa<T, N> {
    fn default() -> Self {
        Self::new(50, nalgebra::convert(1e-6))
    }
}

impl<T: RealField + Copy, const N: usize> Estimator<T, N> for FoyTdoa<T, N> {
    type Measurement = TdoaMeasurement<T, N>;

    fn estimate(&self, m: &[Self::Measurement]) -> Result<SVector<T, N>, EstimationError> {
        if m.is_empty() {
            return Err(EstimationError::InsufficientMeasurements { needed: 1, got: 0 });
        }
        let mut sum = SVector::<T, N>::zeros();
        let mut count: usize = 0;
        for k in m {
            sum += k.anchor;
            sum += k.anchor_ref;
            count += 2;
        }
        let n: T = nalgebra::convert(count as f64);
        let initial = sum / n;
        self.iterate(initial, m)
    }
}
