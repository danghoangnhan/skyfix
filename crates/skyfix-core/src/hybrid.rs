//! Hybrid estimators combining multiple measurement modalities.
//!
//! Phase 3c ships a 2D Gauss-Newton solver that fuses TDoA range-difference
//! and AoA bearing measurements in a single iteration. Per-iteration cost is
//! the sum of contributions from both modalities, giving a tighter estimate
//! than any single-modality solver when both data streams are available.

use crate::error::EstimationError;
use crate::measurement::{AoaBearing, TdoaMeasurement};
use crate::numeric::solve_linear_system;
use nalgebra::{RealField, SMatrix, SVector, Vector2};

/// Iterative Gauss-Newton solver fusing TDoA + AoA in 2D.
///
/// Each iteration accumulates Jacobian × residual contributions from every
/// supplied measurement (TDoA and AoA), then solves the 2×2 normal equations.
/// Requires an initial guess. Natural pairing: bootstrap from
/// [`crate::ChanLinear2D`] when TDoA is present, otherwise from
/// [`crate::StansfieldAoa`].
///
/// AoA contribution uses the proper bearing-residual NLS form
/// (Jacobian `[-Δy/r², Δx/r²]`), not Stansfield's linear surrogate.
/// Angles are wrapped to `[-π, π]` before forming the residual.
pub struct HybridTdoaAoa2D<T: RealField + Copy> {
    max_iters: usize,
    tol: T,
}

impl<T: RealField + Copy> HybridTdoaAoa2D<T> {
    /// Construct a hybrid solver with the given iteration cap and `‖Δx‖` tolerance.
    pub const fn new(max_iters: usize, tol: T) -> Self {
        Self { max_iters, tol }
    }

    /// Run Gauss-Newton from an explicit initial estimate.
    ///
    /// At least 2 total measurements (counting TDoA + AoA together) are
    /// required for the 2×2 normal equations to have a unique solution.
    pub fn iterate(
        &self,
        initial: Vector2<T>,
        tdoas: &[TdoaMeasurement<T, 2>],
        aoas: &[AoaBearing<T>],
    ) -> Result<Vector2<T>, EstimationError> {
        let mut x = initial;

        for _ in 0..self.max_iters {
            let mut ata = SMatrix::<T, 2, 2>::zeros();
            let mut atb = SVector::<T, 2>::zeros();
            let mut total_rows: usize = 0;

            for m in tdoas {
                let d_anchor = x - m.anchor;
                let d_ref = x - m.anchor_ref;
                let r_anchor = d_anchor.norm();
                let r_ref = d_ref.norm();
                if r_anchor == T::zero() || r_ref == T::zero() {
                    continue;
                }
                let jac = d_anchor / r_anchor - d_ref / r_ref;
                let predicted = r_anchor - r_ref;
                let residual = m.range_diff - predicted;
                accumulate_row(&mut ata, &mut atb, &jac, residual);
                total_rows += 1;
            }

            for m in aoas {
                let diff = x - m.anchor;
                let r_sq = diff.dot(&diff);
                if r_sq == T::zero() {
                    continue;
                }
                let predicted = diff.y.atan2(diff.x);
                let residual = wrap_pi(m.bearing - predicted);
                let jac = Vector2::new(-diff.y / r_sq, diff.x / r_sq);
                accumulate_row(&mut ata, &mut atb, &jac, residual);
                total_rows += 1;
            }

            if total_rows < 2 {
                return Err(EstimationError::InsufficientMeasurements {
                    needed: 2,
                    got: total_rows,
                });
            }

            let dx = solve_linear_system(ata, atb)?;
            x += dx;
            if dx.norm() < self.tol {
                return Ok(x);
            }
        }
        Err(EstimationError::DidNotConverge)
    }
}

fn accumulate_row<T: RealField + Copy>(
    ata: &mut SMatrix<T, 2, 2>,
    atb: &mut SVector<T, 2>,
    jac: &Vector2<T>,
    residual: T,
) {
    for r in 0..2 {
        for c in 0..2 {
            ata[(r, c)] += jac[r] * jac[c];
        }
        atb[r] += jac[r] * residual;
    }
}

/// Wrap an angle to `[-π, π]`. Used to form bearing residuals that respect
/// the circular topology of the angle space.
fn wrap_pi<T: RealField + Copy>(a: T) -> T {
    let pi: T = nalgebra::convert(core::f64::consts::PI);
    let two_pi = pi + pi;
    let mut w = a;
    while w > pi {
        w -= two_pi;
    }
    let neg_pi = -pi;
    while w < neg_pi {
        w += two_pi;
    }
    w
}
