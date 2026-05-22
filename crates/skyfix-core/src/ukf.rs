//! Unscented Kalman Filter — sigma-point based nonlinear filtering without
//! Jacobian linearization.

use crate::error::EstimationError;
use crate::filter::{ObservationModel, TransitionModel};
use crate::numeric::{cholesky, invert_square};
use nalgebra::{RealField, SMatrix, SVector};

/// Unscented Kalman Filter for an `N`-dimensional state.
///
/// `SIGMAS` **must** equal `2 * N + 1` (debug-asserted at construction).
/// Convenience type aliases [`Ukf2D`], [`Ukf3D`] etc. enforce the right
/// pairing.
///
/// The UKF avoids the EKF's Jacobian linearization by propagating a small
/// deterministic set of sigma points through the nonlinear `f` and `h`
/// functions and reconstructing mean and covariance. This typically gives
/// better accuracy than EKF when measurement / transition functions have
/// significant curvature in the regime of operation — in particular,
/// cold-start estimation when the prior is far from truth.
///
/// Defaults (`with_defaults`) use the **classic Unscented Transform**:
/// `alpha = 1`, `beta = 2`, `kappa = 3 − N` (giving `c = N + λ = 3` and
/// well-conditioned weights regardless of `N`). The scaled Van der Merwe
/// variant with small `alpha` is numerically unstable for moderate-magnitude
/// covariances because it produces weights of magnitude `O(1/alpha²)`; use
/// it only when you actually need tight sigma spread.
pub struct Ukf<T: RealField + Copy, const N: usize, const SIGMAS: usize> {
    state: SVector<T, N>,
    covariance: SMatrix<T, N, N>,
    alpha: T,
    beta: T,
    kappa: T,
}

/// `Ukf` for 1D state — 3 sigma points.
pub type Ukf1D<T> = Ukf<T, 1, 3>;
/// `Ukf` for 2D state — 5 sigma points.
pub type Ukf2D<T> = Ukf<T, 2, 5>;
/// `Ukf` for 3D state — 7 sigma points.
pub type Ukf3D<T> = Ukf<T, 3, 7>;
/// `Ukf` for 4D state (e.g. 2D position + 2D velocity) — 9 sigma points.
pub type Ukf4D<T> = Ukf<T, 4, 9>;
/// `Ukf` for 6D state (e.g. 3D position + 3D velocity) — 13 sigma points.
pub type Ukf6D<T> = Ukf<T, 6, 13>;

impl<T: RealField + Copy, const N: usize, const SIGMAS: usize> Ukf<T, N, SIGMAS> {
    /// Construct a UKF with explicit Van der Merwe scaling parameters.
    pub fn new(
        state: SVector<T, N>,
        covariance: SMatrix<T, N, N>,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Self {
        debug_assert_eq!(SIGMAS, 2 * N + 1, "SIGMAS must equal 2 * N + 1");
        Self {
            state,
            covariance,
            alpha,
            beta,
            kappa,
        }
    }

    /// Construct a UKF with classic UT defaults: `alpha = 1`, `beta = 2`,
    /// `kappa = 3 − N`. Yields `c = 3` regardless of state dimension.
    pub fn with_defaults(state: SVector<T, N>, covariance: SMatrix<T, N, N>) -> Self {
        let n_t: T = nalgebra::convert(N as f64);
        let three: T = nalgebra::convert(3.0);
        Self::new(
            state,
            covariance,
            T::one(),
            nalgebra::convert(2.0),
            three - n_t,
        )
    }

    /// Current posterior mean.
    pub fn state(&self) -> SVector<T, N> {
        self.state
    }

    /// Current posterior covariance.
    pub fn covariance(&self) -> SMatrix<T, N, N> {
        self.covariance
    }

    fn lambda(&self) -> T {
        let n_t: T = nalgebra::convert(N as f64);
        self.alpha * self.alpha * (n_t + self.kappa) - n_t
    }

    /// Returns `(w_m, w_c)` — mean and covariance weights, length `SIGMAS`.
    fn weights(&self) -> (SVector<T, SIGMAS>, SVector<T, SIGMAS>) {
        let n_t: T = nalgebra::convert(N as f64);
        let lambda = self.lambda();
        let c = n_t + lambda;
        let mut w_m = SVector::<T, SIGMAS>::zeros();
        let mut w_c = SVector::<T, SIGMAS>::zeros();
        w_m[0] = lambda / c;
        w_c[0] = w_m[0] + (T::one() - self.alpha * self.alpha + self.beta);
        let two: T = nalgebra::convert(2.0);
        let w = T::one() / (two * c);
        for i in 1..SIGMAS {
            w_m[i] = w;
            w_c[i] = w;
        }
        (w_m, w_c)
    }

    /// Generate the `SIGMAS` sigma points around `mean` with spread set by `cov`.
    fn sigma_points(
        &self,
        mean: SVector<T, N>,
        cov: SMatrix<T, N, N>,
    ) -> Result<SMatrix<T, N, SIGMAS>, EstimationError> {
        let n_t: T = nalgebra::convert(N as f64);
        let c = n_t + self.lambda();
        let l = cholesky(cov * c)?;

        let mut sigmas = SMatrix::<T, N, SIGMAS>::zeros();
        for j in 0..N {
            sigmas[(j, 0)] = mean[j];
        }
        for i in 0..N {
            for j in 0..N {
                let offset = l[(j, i)];
                sigmas[(j, i + 1)] = mean[j] + offset;
                sigmas[(j, N + i + 1)] = mean[j] - offset;
            }
        }
        Ok(sigmas)
    }

    /// Advance the filter by `dt` using the supplied transition model.
    pub fn predict<TM: TransitionModel<T, N>>(
        &mut self,
        model: &TM,
        dt: T,
    ) -> Result<(), EstimationError> {
        let (w_m, w_c) = self.weights();
        let sigmas = self.sigma_points(self.state, self.covariance)?;

        // Propagate each sigma point through f.
        let mut prop = SMatrix::<T, N, SIGMAS>::zeros();
        for i in 0..SIGMAS {
            let mut sigma = SVector::<T, N>::zeros();
            for j in 0..N {
                sigma[j] = sigmas[(j, i)];
            }
            let propagated = model.transition(&sigma, dt);
            for j in 0..N {
                prop[(j, i)] = propagated[j];
            }
        }

        // Reconstruct mean.
        let mut new_mean = SVector::<T, N>::zeros();
        for i in 0..SIGMAS {
            for j in 0..N {
                new_mean[j] += w_m[i] * prop[(j, i)];
            }
        }

        // Reconstruct covariance with process noise.
        let mut new_cov = SMatrix::<T, N, N>::zeros();
        for i in 0..SIGMAS {
            for r in 0..N {
                let dr = prop[(r, i)] - new_mean[r];
                for c in 0..N {
                    let dc = prop[(c, i)] - new_mean[c];
                    new_cov[(r, c)] += w_c[i] * dr * dc;
                }
            }
        }
        new_cov += model.noise(dt);

        self.state = new_mean;
        self.covariance = new_cov;
        Ok(())
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
        let (w_m, w_c) = self.weights();
        let sigmas = self.sigma_points(self.state, self.covariance)?;

        // Project each sigma into measurement space.
        let mut z_sigmas = SMatrix::<T, M, SIGMAS>::zeros();
        for i in 0..SIGMAS {
            let mut sigma = SVector::<T, N>::zeros();
            for j in 0..N {
                sigma[j] = sigmas[(j, i)];
            }
            let z_i = model.predict(&sigma);
            for j in 0..M {
                z_sigmas[(j, i)] = z_i[j];
            }
        }

        // Predicted measurement mean.
        let mut z_mean = SVector::<T, M>::zeros();
        for i in 0..SIGMAS {
            for j in 0..M {
                z_mean[j] += w_m[i] * z_sigmas[(j, i)];
            }
        }

        // Innovation covariance Pzz = sum w_c · (z_i - z̄)(z_i - z̄)^T + R.
        let mut pzz = SMatrix::<T, M, M>::zeros();
        for i in 0..SIGMAS {
            for r in 0..M {
                let dr = z_sigmas[(r, i)] - z_mean[r];
                for c in 0..M {
                    let dc = z_sigmas[(c, i)] - z_mean[c];
                    pzz[(r, c)] += w_c[i] * dr * dc;
                }
            }
        }
        pzz += model.noise();

        // Cross-covariance Pxz = sum w_c · (x_i - x̄)(z_i - z̄)^T.
        let mut pxz = SMatrix::<T, N, M>::zeros();
        for i in 0..SIGMAS {
            for r in 0..N {
                let dr = sigmas[(r, i)] - self.state[r];
                for c in 0..M {
                    let dc = z_sigmas[(c, i)] - z_mean[c];
                    pxz[(r, c)] += w_c[i] * dr * dc;
                }
            }
        }

        let pzz_inv = invert_square(pzz)?;
        let k = pxz * pzz_inv;
        self.state += k * (z - z_mean);
        self.covariance -= k * pzz * k.transpose();
        Ok(())
    }
}
