//! Recursive filter traits plus a few built-in transition/observation models.
//!
//! Concrete filter implementations live in [`crate::ekf`] (Phase 3). UKF /
//! ESKF / particle filter land in subsequent phases.

use nalgebra::{RealField, SMatrix, SVector};

/// State-transition model for a recursive filter.
///
/// - `transition(x, dt)` returns the predicted next state `f(x, dt)`.
/// - `jacobian(x, dt)` returns `∂f/∂x` evaluated at `x` (= `F` for linear models).
/// - `noise(dt)` returns the additive process-noise covariance `Q`.
pub trait TransitionModel<T: RealField + Copy, const N: usize> {
    /// Predicted next state.
    fn transition(&self, x: &SVector<T, N>, dt: T) -> SVector<T, N>;
    /// Jacobian `∂f/∂x` at `x`.
    fn jacobian(&self, x: &SVector<T, N>, dt: T) -> SMatrix<T, N, N>;
    /// Additive process noise covariance.
    fn noise(&self, dt: T) -> SMatrix<T, N, N>;
}

/// Measurement model mapping state `x ∈ R^N` to expected measurement `z ∈ R^M`.
pub trait ObservationModel<T: RealField + Copy, const N: usize, const M: usize> {
    /// Predicted measurement `h(x)`.
    fn predict(&self, x: &SVector<T, N>) -> SVector<T, M>;
    /// Jacobian `∂h/∂x` evaluated at `x`.
    fn jacobian(&self, x: &SVector<T, N>) -> SMatrix<T, M, N>;
    /// Measurement noise covariance `R`.
    fn noise(&self) -> SMatrix<T, M, M>;
}

// ============================================================================
// Built-in transition models
// ============================================================================

/// Trivial transition: state is unchanged across the predict step but accrues
/// additive process noise. Useful when ingesting measurements without a
/// dynamics model (e.g. stationary or slow-moving targets sampled densely).
pub struct IdentityTransition<T: RealField + Copy, const N: usize> {
    /// Additive process noise covariance, applied at every predict step.
    pub process_noise: SMatrix<T, N, N>,
}

impl<T: RealField + Copy, const N: usize> IdentityTransition<T, N> {
    /// Construct an identity transition with the supplied process-noise matrix.
    pub const fn new(process_noise: SMatrix<T, N, N>) -> Self {
        Self { process_noise }
    }

    /// Convenience constructor: process noise = `sigma · I`.
    pub fn with_uniform_noise(sigma: T) -> Self {
        let mut q = SMatrix::<T, N, N>::zeros();
        for i in 0..N {
            q[(i, i)] = sigma;
        }
        Self { process_noise: q }
    }
}

impl<T: RealField + Copy, const N: usize> TransitionModel<T, N> for IdentityTransition<T, N> {
    fn transition(&self, x: &SVector<T, N>, _dt: T) -> SVector<T, N> {
        *x
    }
    fn jacobian(&self, _x: &SVector<T, N>, _dt: T) -> SMatrix<T, N, N> {
        let mut i = SMatrix::<T, N, N>::zeros();
        for k in 0..N {
            i[(k, k)] = T::one();
        }
        i
    }
    fn noise(&self, _dt: T) -> SMatrix<T, N, N> {
        self.process_noise
    }
}

// ============================================================================
// Built-in observation models
// ============================================================================

/// Range measurement to a fixed anchor.
///
/// Assumes the state vector **is** the position (no velocity tracking). For
/// state representations that include velocity / orientation, write your own
/// `ObservationModel` that projects the position components out of the state.
pub struct RangeAnchor<T: RealField + Copy, const N: usize> {
    /// Anchor position.
    pub anchor: SVector<T, N>,
    /// Range-measurement variance (units of distance²).
    pub variance: T,
}

impl<T: RealField + Copy, const N: usize> RangeAnchor<T, N> {
    /// Construct a new range-anchor model.
    pub const fn new(anchor: SVector<T, N>, variance: T) -> Self {
        Self { anchor, variance }
    }
}

impl<T: RealField + Copy, const N: usize> ObservationModel<T, N, 1> for RangeAnchor<T, N> {
    fn predict(&self, x: &SVector<T, N>) -> SVector<T, 1> {
        let mut z = SVector::<T, 1>::zeros();
        z[0] = (*x - self.anchor).norm();
        z
    }

    fn jacobian(&self, x: &SVector<T, N>) -> SMatrix<T, 1, N> {
        let diff = *x - self.anchor;
        let r = diff.norm();
        let mut h = SMatrix::<T, 1, N>::zeros();
        if r != T::zero() {
            for j in 0..N {
                h[(0, j)] = diff[j] / r;
            }
        }
        h
    }

    fn noise(&self) -> SMatrix<T, 1, 1> {
        let mut r = SMatrix::<T, 1, 1>::zeros();
        r[(0, 0)] = self.variance;
        r
    }
}
