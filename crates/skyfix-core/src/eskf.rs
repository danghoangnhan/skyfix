//! Error-State Kalman Filter (ESKF) for IMU-driven dead-reckoning.
//!
//! The ESKF splits the state into a *nominal* trajectory (propagated by the
//! IMU's strapdown integration, possibly non-linear) and a small *error
//! state* (estimated by a linearized Kalman update). After each measurement
//! update the error state is "reset" by folding the correction into the
//! nominal state — for a Euclidean state vector that's just an addition; for
//! manifold states (e.g. quaternion attitude) it's a manifold-specific
//! retraction. This crate currently ships the Euclidean variant; future
//! revisions can add a 3D variant where heading lives on `SO(3)`.
//!
//! The headline scenario for v0.1 is **2D IMU + UWB range**: an aerial
//! target with body-frame accelerometer + yaw-rate gyro, ranging against a
//! fixed UWB anchor set. The companion [`Imu2DStrapdown`] integrator wraps
//! that standard pipeline.

use crate::error::EstimationError;
use crate::filter::ObservationModel;
use crate::numeric::invert_square;
use nalgebra::{ComplexField, RealField, SMatrix, SVector};

/// Strapdown integration step driven by a `U`-dimensional control input
/// (typically `U = 3` for 2D accel + gyro, `U = 6` for 3D accel + gyro).
///
/// - `integrate(nominal, control, dt)` advances the nominal state.
/// - `error_jacobian(nominal, control, dt)` returns the error-state
///   transition matrix `F`, i.e. `∂δx_{k+1}/∂δx_k` evaluated at the
///   nominal state.
/// - `noise(dt)` returns the additive process-noise covariance `Q` applied
///   to the error state at each predict step.
pub trait ImuIntegrator<T: RealField + Copy, const N: usize, const U: usize> {
    /// Predicted next nominal state.
    fn integrate(&self, nominal: &SVector<T, N>, control: &SVector<T, U>, dt: T) -> SVector<T, N>;
    /// Error-state Jacobian `F = ∂δx_{k+1}/∂δx_k`.
    fn error_jacobian(
        &self,
        nominal: &SVector<T, N>,
        control: &SVector<T, U>,
        dt: T,
    ) -> SMatrix<T, N, N>;
    /// Additive process noise covariance on the error state per step.
    fn noise(&self, dt: T) -> SMatrix<T, N, N>;
}

/// Error-State Kalman Filter for an `N`-dimensional Euclidean state.
///
/// Stores the nominal state and the error-state covariance. The mean of the
/// error state is held at zero by construction (it's reset back to zero
/// after every `update` by injecting the correction into the nominal).
///
/// Predict and update each take their model as an argument so the same
/// filter can ingest different sensor streams without reconfiguration.
pub struct Eskf<T: RealField + Copy, const N: usize> {
    nominal: SVector<T, N>,
    covariance: SMatrix<T, N, N>,
}

impl<T: RealField + Copy, const N: usize> Eskf<T, N> {
    /// Construct an ESKF from prior nominal state and error-state covariance.
    pub const fn new(nominal: SVector<T, N>, covariance: SMatrix<T, N, N>) -> Self {
        Self {
            nominal,
            covariance,
        }
    }

    /// Current nominal state.
    pub fn nominal(&self) -> SVector<T, N> {
        self.nominal
    }

    /// Current error-state covariance.
    pub fn covariance(&self) -> SMatrix<T, N, N> {
        self.covariance
    }

    /// Override the nominal — useful for re-seeding from a closed-form fix.
    pub fn set_nominal(&mut self, nominal: SVector<T, N>) {
        self.nominal = nominal;
    }

    /// Advance the filter by `dt` using the supplied IMU integrator and
    /// control input.
    pub fn predict<I, const U: usize>(&mut self, integrator: &I, control: &SVector<T, U>, dt: T)
    where
        I: ImuIntegrator<T, N, U>,
    {
        let f = integrator.error_jacobian(&self.nominal, control, dt);
        let q = integrator.noise(dt);
        self.nominal = integrator.integrate(&self.nominal, control, dt);
        self.covariance = f * self.covariance * f.transpose() + q;
    }

    /// Ingest one measurement of dimension `M` using the supplied
    /// observation model evaluated at the current nominal state.
    ///
    /// The measurement-Jacobian convention is `H = ∂h/∂x` at `nominal`. For
    /// a Euclidean error state this is the same `H` an EKF would use; for a
    /// manifold error state, write your own [`ObservationModel`] that
    /// projects through the manifold's tangent space.
    pub fn update<const M: usize, OM>(
        &mut self,
        model: &OM,
        z: &SVector<T, M>,
    ) -> Result<(), EstimationError>
    where
        OM: ObservationModel<T, N, M>,
    {
        let h = model.jacobian(&self.nominal);
        let r = model.noise();
        let innovation = z - model.predict(&self.nominal);
        let s = h * self.covariance * h.transpose() + r;
        let s_inv = invert_square(s)?;
        let k = self.covariance * h.transpose() * s_inv;
        let delta_x = k * innovation;

        // Inject the error-state correction into the nominal. For Euclidean
        // states this is just addition; for manifold states an override
        // point would go here.
        self.nominal += delta_x;

        // Joseph form covariance update for numerical robustness.
        let mut i_n = SMatrix::<T, N, N>::zeros();
        for j in 0..N {
            i_n[(j, j)] = T::one();
        }
        let factor = i_n - k * h;
        self.covariance = factor * self.covariance * factor.transpose() + k * r * k.transpose();
        Ok(())
    }
}

// ============================================================================
// Built-in: 2D position + velocity + heading strapdown integrator
// ============================================================================

/// 2D IMU strapdown integrator for a 5-dimensional Euclidean state
/// `[px, py, vx, vy, heading θ]` and 3-dimensional control input
/// `[a_body_x, a_body_y, gyro_z]`.
///
/// The accelerometer measures body-frame acceleration (after gravity has
/// been removed — for a level 2D scenario gravity is purely vertical and
/// already projects out). Body acceleration is rotated into the navigation
/// frame using the current heading, integrated to update velocity, and
/// integrated again to update position. The gyro updates heading directly.
///
/// Process-noise inputs `accel_sigma` and `gyro_sigma` are the
/// continuous-time standard deviations of the accelerometer and gyro noise,
/// in `m/s²` and `rad/s` respectively. The discrete-time covariance is
/// `σ² · dt²` (a first-order approximation suitable for the small `dt`
/// typical of UAV IMU sampling).
#[derive(Debug, Clone, Copy)]
pub struct Imu2DStrapdown<T: RealField + Copy> {
    /// Accelerometer standard deviation (m/s², per axis).
    pub accel_sigma: T,
    /// Gyro standard deviation (rad/s).
    pub gyro_sigma: T,
}

impl<T: RealField + Copy> Imu2DStrapdown<T> {
    /// Construct an integrator with the given IMU noise parameters.
    pub const fn new(accel_sigma: T, gyro_sigma: T) -> Self {
        Self {
            accel_sigma,
            gyro_sigma,
        }
    }
}

impl<T: RealField + Copy> ImuIntegrator<T, 5, 3> for Imu2DStrapdown<T> {
    fn integrate(&self, nominal: &SVector<T, 5>, control: &SVector<T, 3>, dt: T) -> SVector<T, 5> {
        let px = nominal[0];
        let py = nominal[1];
        let vx = nominal[2];
        let vy = nominal[3];
        let theta = nominal[4];

        let ax_body = control[0];
        let ay_body = control[1];
        let omega = control[2];

        let half = T::one() / (T::one() + T::one());
        // Midpoint heading for the integration step.
        let theta_mid = theta + omega * dt * half;
        let cos_m = ComplexField::cos(theta_mid);
        let sin_m = ComplexField::sin(theta_mid);

        // Rotate body-frame accel into world-frame.
        let ax_world = cos_m * ax_body - sin_m * ay_body;
        let ay_world = sin_m * ax_body + cos_m * ay_body;

        let px_new = px + vx * dt + ax_world * dt * dt * half;
        let py_new = py + vy * dt + ay_world * dt * dt * half;
        let vx_new = vx + ax_world * dt;
        let vy_new = vy + ay_world * dt;
        let theta_new = theta + omega * dt;

        let mut out = SVector::<T, 5>::zeros();
        out[0] = px_new;
        out[1] = py_new;
        out[2] = vx_new;
        out[3] = vy_new;
        out[4] = theta_new;
        out
    }

    fn error_jacobian(
        &self,
        nominal: &SVector<T, 5>,
        control: &SVector<T, 3>,
        dt: T,
    ) -> SMatrix<T, 5, 5> {
        let theta = nominal[4];
        let ax_body = control[0];
        let ay_body = control[1];
        let omega = control[2];

        let half = T::one() / (T::one() + T::one());
        let theta_mid = theta + omega * dt * half;
        let cos_m = ComplexField::cos(theta_mid);
        let sin_m = ComplexField::sin(theta_mid);

        // ∂R/∂θ · a_body = [-sin θ_mid, -cos θ_mid; cos θ_mid, -sin θ_mid] · a_body
        let drot_ax = -sin_m * ax_body - cos_m * ay_body;
        let drot_ay = cos_m * ax_body - sin_m * ay_body;

        let mut f = SMatrix::<T, 5, 5>::zeros();
        // ∂p_new/∂p = I
        f[(0, 0)] = T::one();
        f[(1, 1)] = T::one();
        // ∂p_new/∂v = dt · I
        f[(0, 2)] = dt;
        f[(1, 3)] = dt;
        // ∂p_new/∂θ = 0.5 · (∂R/∂θ · a_body) · dt²
        f[(0, 4)] = drot_ax * dt * dt * half;
        f[(1, 4)] = drot_ay * dt * dt * half;
        // ∂v_new/∂v = I
        f[(2, 2)] = T::one();
        f[(3, 3)] = T::one();
        // ∂v_new/∂θ = (∂R/∂θ · a_body) · dt
        f[(2, 4)] = drot_ax * dt;
        f[(3, 4)] = drot_ay * dt;
        // ∂θ_new/∂θ = 1
        f[(4, 4)] = T::one();
        f
    }

    fn noise(&self, dt: T) -> SMatrix<T, 5, 5> {
        let mut q = SMatrix::<T, 5, 5>::zeros();
        let accel_var = self.accel_sigma * self.accel_sigma * dt * dt;
        let gyro_var = self.gyro_sigma * self.gyro_sigma * dt * dt;
        // Position-process-noise: derived from velocity-channel noise via
        // p = v · dt, so σ_p ≈ σ_v · dt and var_p ≈ var_v · dt² — much
        // smaller than the direct velocity-channel variance. For short dt
        // we treat it as 0 and rely on the velocity channel; see CHANGELOG.
        q[(2, 2)] = accel_var;
        q[(3, 3)] = accel_var;
        q[(4, 4)] = gyro_var;
        q
    }
}
