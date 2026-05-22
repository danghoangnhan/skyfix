//! Shared dense numerical routines for skyfix-core estimators.
//!
//! Hand-rolled to avoid `nalgebra`'s decomposition methods (`.lu()`, `.qr()`,
//! `.svd()`, …) which require `Const<N>: ToTypenum` and therefore don't work
//! over generic `const N`.

use crate::error::EstimationError;
use nalgebra::{ComplexField, RealField, SMatrix, SVector};

/// Solve the square `N × N` linear system `A x = b` via Gauss elimination
/// with partial pivoting.
///
/// Returns [`EstimationError::SingularSystem`] when any pivot is exactly zero.
pub fn solve_linear_system<T: RealField + Copy, const N: usize>(
    mut a: SMatrix<T, N, N>,
    mut b: SVector<T, N>,
) -> Result<SVector<T, N>, EstimationError> {
    let zero = T::zero();

    for k in 0..N {
        let mut max_abs = ComplexField::abs(a[(k, k)]);
        let mut pivot_row = k;
        for i in (k + 1)..N {
            let v = ComplexField::abs(a[(i, k)]);
            if v > max_abs {
                max_abs = v;
                pivot_row = i;
            }
        }
        if max_abs == zero {
            return Err(EstimationError::SingularSystem);
        }
        if pivot_row != k {
            a.swap_rows(k, pivot_row);
            b.swap_rows(k, pivot_row);
        }
        let pivot = a[(k, k)];
        for i in (k + 1)..N {
            let factor = a[(i, k)] / pivot;
            a[(i, k)] = zero;
            for j in (k + 1)..N {
                let above = a[(k, j)];
                a[(i, j)] -= factor * above;
            }
            let b_above = b[k];
            b[i] -= factor * b_above;
        }
    }

    let mut x = SVector::<T, N>::zeros();
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum -= a[(i, j)] * x[j];
        }
        let diag = a[(i, i)];
        if diag == zero {
            return Err(EstimationError::SingularSystem);
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

/// Solve a weighted linear least-squares problem `J x ≈ y` by forming the
/// normal equations `(J^T W J) x = J^T W y` and calling [`solve_linear_system`].
///
/// `rows` yields one `(jacobian_row, weight, observation)` triple per
/// measurement. Returns [`EstimationError::InsufficientMeasurements`] when
/// fewer than `N` rows are supplied.
pub fn solve_normal_equations<T, const N: usize, I>(
    rows: I,
) -> Result<SVector<T, N>, EstimationError>
where
    T: RealField + Copy,
    I: IntoIterator<Item = (SVector<T, N>, T, T)>,
{
    let mut ata = SMatrix::<T, N, N>::zeros();
    let mut atb = SVector::<T, N>::zeros();
    let mut count: usize = 0;
    for (j, w, y) in rows {
        for r in 0..N {
            let wj = w * j[r];
            for c in 0..N {
                ata[(r, c)] += wj * j[c];
            }
            atb[r] += wj * y;
        }
        count += 1;
    }
    if count < N {
        return Err(EstimationError::InsufficientMeasurements {
            needed: N,
            got: count,
        });
    }
    solve_linear_system(ata, atb)
}

/// Invert the square `N × N` matrix `A` by solving `A x = e_j` for each
/// canonical basis vector `e_j` and stacking the resulting columns.
///
/// Returns [`EstimationError::SingularSystem`] if any column solve fails.
/// `O(N⁴)` in the worst case; intended for the small `N` values typical of
/// EKF/UKF measurement and state spaces (≤ 12).
pub fn invert_square<T: RealField + Copy, const N: usize>(
    a: SMatrix<T, N, N>,
) -> Result<SMatrix<T, N, N>, EstimationError> {
    let mut inv = SMatrix::<T, N, N>::zeros();
    for j in 0..N {
        let mut e = SVector::<T, N>::zeros();
        e[j] = T::one();
        let col = solve_linear_system(a, e)?;
        for i in 0..N {
            inv[(i, j)] = col[i];
        }
    }
    Ok(inv)
}

/// Cholesky decomposition `A = L L^T` of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor `L`. Returns
/// [`EstimationError::SingularSystem`] when `A` is not positive definite
/// (a non-positive diagonal sum is encountered). Used by the UKF to spread
/// sigma points around the current covariance.
pub fn cholesky<T: RealField + Copy, const N: usize>(
    a: SMatrix<T, N, N>,
) -> Result<SMatrix<T, N, N>, EstimationError> {
    let zero = T::zero();
    let mut l = SMatrix::<T, N, N>::zeros();
    for i in 0..N {
        let mut sum = a[(i, i)];
        for k in 0..i {
            sum -= l[(i, k)] * l[(i, k)];
        }
        if sum <= zero {
            return Err(EstimationError::SingularSystem);
        }
        l[(i, i)] = ComplexField::sqrt(sum);
        for j in (i + 1)..N {
            let mut sum2 = a[(j, i)];
            for k in 0..i {
                sum2 -= l[(j, k)] * l[(i, k)];
            }
            l[(j, i)] = sum2 / l[(i, i)];
        }
    }
    Ok(l)
}

/// Gauss-Newton iteration for nonlinear least squares.
///
/// At each step, the caller's `linearize(x, m)` returns a Jacobian row
/// `∂f/∂x` and a residual `r = y − f(x; m)` for each measurement `m`.
/// The unweighted normal equations `(J^T J) Δx = J^T r` are solved and
/// `x ← x + Δx` is applied until `‖Δx‖ < tol` or `max_iters` is reached.
///
/// Returns [`EstimationError::DidNotConverge`] when the iteration cap is hit.
pub fn gauss_newton<T, M, const N: usize, F>(
    initial: SVector<T, N>,
    measurements: &[M],
    max_iters: usize,
    tol: T,
    mut linearize: F,
) -> Result<SVector<T, N>, EstimationError>
where
    T: RealField + Copy,
    F: FnMut(SVector<T, N>, &M) -> (SVector<T, N>, T),
{
    let mut x = initial;
    for _ in 0..max_iters {
        let rows = measurements.iter().map(|m| {
            let (j, r) = linearize(x, m);
            (j, T::one(), r)
        });
        let dx = solve_normal_equations::<T, N, _>(rows)?;
        x += dx;
        if dx.norm() < tol {
            return Ok(x);
        }
    }
    Err(EstimationError::DidNotConverge)
}
