//! Extended Kalman Filter generic over state dimension `N`.

use crate::error::EstimationError;
use crate::filter::{ObservationModel, TransitionModel};
use crate::numeric::invert_square;
use nalgebra::{RealField, SMatrix, SVector};

/// Extended Kalman Filter for an `N`-dimensional state.
///
/// Stores the posterior mean and covariance. Each call to [`Self::predict`]
/// advances the filter by `dt` using the supplied [`TransitionModel`]; each
/// call to [`Self::update`] ingests one measurement of dimension `M` via the
/// supplied [`ObservationModel`]. Multiple measurements are folded in by
/// repeated `update` calls — order doesn't matter for the steady-state
/// estimate provided the linearization is good at the operating point.
///
/// The covariance update uses **Joseph's form**:
///
/// ```text
///     P ← (I − K H) P (I − K H)^T + K R K^T
/// ```
///
/// which preserves symmetry and is numerically robust against ill-conditioned
/// gains, at the cost of two extra `N × N` matrix multiplies compared to the
/// shorter `P ← (I − K H) P` form.
pub struct Ekf<T: RealField + Copy, const N: usize> {
    state: SVector<T, N>,
    covariance: SMatrix<T, N, N>,
}

impl<T: RealField + Copy, const N: usize> Ekf<T, N> {
    /// Construct an EKF from prior mean and covariance.
    pub const fn new(state: SVector<T, N>, covariance: SMatrix<T, N, N>) -> Self {
        Self { state, covariance }
    }

    /// Current posterior mean.
    pub fn state(&self) -> SVector<T, N> {
        self.state
    }

    /// Current posterior covariance.
    pub fn covariance(&self) -> SMatrix<T, N, N> {
        self.covariance
    }

    /// Advance the filter by `dt` using the supplied transition model.
    pub fn predict<TM: TransitionModel<T, N>>(&mut self, model: &TM, dt: T) {
        let f = model.jacobian(&self.state, dt);
        let q = model.noise(dt);
        self.state = model.transition(&self.state, dt);
        self.covariance = f * self.covariance * f.transpose() + q;
    }

    /// Ingest one measurement of dimension `M`.
    pub fn update<const M: usize, OM>(
        &mut self,
        model: &OM,
        z: &SVector<T, M>,
    ) -> Result<(), EstimationError>
    where
        OM: ObservationModel<T, N, M>,
    {
        let h = model.jacobian(&self.state);
        let r = model.noise();
        let innovation = z - model.predict(&self.state);
        let s = h * self.covariance * h.transpose() + r;
        let s_inv = invert_square(s)?;
        let k = self.covariance * h.transpose() * s_inv;
        self.state += k * innovation;

        // Joseph form: P = (I − K H) P (I − K H)^T + K R K^T
        let mut i_n = SMatrix::<T, N, N>::zeros();
        for j in 0..N {
            i_n[(j, j)] = T::one();
        }
        let factor = i_n - k * h;
        self.covariance = factor * self.covariance * factor.transpose() + k * r * k.transpose();
        Ok(())
    }
}
