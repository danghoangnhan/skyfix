//! Cramér-Rao Lower Bound (CRLB), Fisher Information Matrix (FIM), and
//! Geometric Dilution of Precision (GDOP) analysis.
//!
//! The CRLB lower-bounds the achievable covariance of any *unbiased*
//! estimator of a position fix given a measurement geometry and noise
//! model. It is the right tool for anchor-placement planning: pick anchor
//! geometries that minimize GDOP at the operational region of interest.
//!
//! For independent measurements with Jacobians `J_i` and noise variances
//! `σ²_i` the Fisher Information Matrix is
//!
//! ```text
//!     FIM = Σ_i (1 / σ²_i) · J_i^T J_i
//! ```
//!
//! and the CRLB on position covariance is `FIM⁻¹`. GDOP is `√tr(FIM⁻¹)`.

use crate::error::EstimationError;
use crate::numeric::invert_square;
use core::marker::PhantomData;
use nalgebra::{ComplexField, RealField, SMatrix, SVector, Vector2};

/// Accumulator for measurement contributions to the Fisher Information.
///
/// Add per-modality contributions via [`Self::add_toa`], [`Self::add_tdoa`],
/// and (in 2D) [`Self::add_aoa`]. Call [`Self::finish`] to produce a
/// [`CrlbAnalysis`].
pub struct CrlbBuilder<T: RealField + Copy, const N: usize> {
    fim: SMatrix<T, N, N>,
    _marker: PhantomData<T>,
}

impl<T: RealField + Copy, const N: usize> CrlbBuilder<T, N> {
    /// New empty FIM accumulator.
    pub fn new() -> Self {
        Self {
            fim: SMatrix::<T, N, N>::zeros(),
            _marker: PhantomData,
        }
    }

    /// Add a ToA range-to-anchor measurement contribution.
    ///
    /// Jacobian row: `u_i^T = (x − a_i)^T / ‖x − a_i‖`.
    pub fn add_toa(
        &mut self,
        target: SVector<T, N>,
        anchor: SVector<T, N>,
        variance: T,
    ) -> &mut Self {
        let diff = target - anchor;
        let r = diff.norm();
        if r == T::zero() || variance == T::zero() {
            return self;
        }
        let u = diff / r;
        let inv_var = T::one() / variance;
        for row in 0..N {
            for col in 0..N {
                self.fim[(row, col)] += inv_var * u[row] * u[col];
            }
        }
        self
    }

    /// Add a TDoA range-difference contribution `‖x−a‖ − ‖x−a_ref‖`.
    ///
    /// Jacobian row: `(u_i − u_0)^T` where `u_k = (x − a_k) / ‖x − a_k‖`.
    /// NB: this treats different TDoA measurements as independent — for
    /// the correlated case (all TDoAs share one reference) the off-diagonal
    /// terms of the measurement-noise covariance matter and a future API
    /// will accept the full `R` matrix.
    pub fn add_tdoa(
        &mut self,
        target: SVector<T, N>,
        anchor: SVector<T, N>,
        anchor_ref: SVector<T, N>,
        variance: T,
    ) -> &mut Self {
        let d_anchor = target - anchor;
        let d_ref = target - anchor_ref;
        let r_anchor = d_anchor.norm();
        let r_ref = d_ref.norm();
        if r_anchor == T::zero() || r_ref == T::zero() || variance == T::zero() {
            return self;
        }
        let j = d_anchor / r_anchor - d_ref / r_ref;
        let inv_var = T::one() / variance;
        for row in 0..N {
            for col in 0..N {
                self.fim[(row, col)] += inv_var * j[row] * j[col];
            }
        }
        self
    }

    /// Finalize the builder.
    pub fn finish(self) -> CrlbAnalysis<T, N> {
        CrlbAnalysis { fim: self.fim }
    }

    /// Direct access to the accumulated FIM (rarely needed; prefer the
    /// analysis API).
    pub fn fim(&self) -> SMatrix<T, N, N> {
        self.fim
    }
}

impl<T: RealField + Copy, const N: usize> Default for CrlbBuilder<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// 2D-only AoA contribution (azimuth bearing).
impl<T: RealField + Copy> CrlbBuilder<T, 2> {
    /// Add a 2D AoA bearing contribution.
    ///
    /// Jacobian row: `[−Δy/r², Δx/r²]` where `(Δx, Δy) = target − anchor`.
    /// `variance` is the angular noise variance (radians²).
    pub fn add_aoa(&mut self, target: Vector2<T>, anchor: Vector2<T>, variance: T) -> &mut Self {
        let diff = target - anchor;
        let r_sq = diff.dot(&diff);
        if r_sq == T::zero() || variance == T::zero() {
            return self;
        }
        let j = Vector2::new(-diff.y / r_sq, diff.x / r_sq);
        let inv_var = T::one() / variance;
        for row in 0..2 {
            for col in 0..2 {
                self.fim[(row, col)] += inv_var * j[row] * j[col];
            }
        }
        self
    }
}

/// Frozen CRLB / GDOP analysis for a given geometry.
pub struct CrlbAnalysis<T: RealField + Copy, const N: usize> {
    fim: SMatrix<T, N, N>,
}

impl<T: RealField + Copy, const N: usize> CrlbAnalysis<T, N> {
    /// Direct access to the Fisher Information Matrix.
    pub fn fim(&self) -> SMatrix<T, N, N> {
        self.fim
    }

    /// CRLB on position covariance = `FIM⁻¹`.
    ///
    /// Returns [`EstimationError::SingularSystem`] for degenerate geometries
    /// (e.g. colinear anchors in 2D, coplanar anchors in 3D).
    pub fn covariance(&self) -> Result<SMatrix<T, N, N>, EstimationError> {
        invert_square(self.fim)
    }

    /// Geometric Dilution of Precision = `√tr(FIM⁻¹)`.
    ///
    /// Units are meters (or whatever the anchor positions use) per standard
    /// deviation of unit measurement noise. Smaller is better; a doubling of
    /// GDOP doubles the position-error standard deviation.
    pub fn gdop(&self) -> Result<T, EstimationError> {
        let cov = self.covariance()?;
        let mut trace = T::zero();
        for i in 0..N {
            trace += cov[(i, i)];
        }
        Ok(ComplexField::sqrt(trace))
    }
}

// 2D conveniences.
impl<T: RealField + Copy> CrlbAnalysis<T, 2> {
    /// Horizontal DOP = `√(P_xx + P_yy)`. For 2D this equals GDOP.
    pub fn hdop(&self) -> Result<T, EstimationError> {
        let cov = self.covariance()?;
        Ok(ComplexField::sqrt(cov[(0, 0)] + cov[(1, 1)]))
    }
}

// 3D conveniences.
impl<T: RealField + Copy> CrlbAnalysis<T, 3> {
    /// Horizontal DOP = `√(P_xx + P_yy)`. Excludes the vertical component.
    pub fn hdop(&self) -> Result<T, EstimationError> {
        let cov = self.covariance()?;
        Ok(ComplexField::sqrt(cov[(0, 0)] + cov[(1, 1)]))
    }

    /// Vertical DOP = `√P_zz`.
    pub fn vdop(&self) -> Result<T, EstimationError> {
        let cov = self.covariance()?;
        Ok(ComplexField::sqrt(cov[(2, 2)]))
    }
}
