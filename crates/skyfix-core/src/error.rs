//! Error types shared across skyfix-core estimators.

use core::fmt;

/// Errors that can arise during position estimation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationError {
    /// Fewer measurements supplied than the chosen estimator requires.
    InsufficientMeasurements {
        /// Minimum number of measurements the estimator needs.
        needed: usize,
        /// Number of measurements actually supplied.
        got: usize,
    },
    /// The underlying linear system has no unique solution (e.g. colinear
    /// anchors in 2D, coplanar anchors in 3D).
    SingularSystem,
    /// The estimator produced a non-finite value (NaN or infinity).
    NonFinite,
    /// An iterative estimator (Gauss-Newton, Foy WLS) exhausted `max_iters`
    /// without converging within `tol`.
    DidNotConverge,
}

impl fmt::Display for EstimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientMeasurements { needed, got } => write!(
                f,
                "insufficient measurements: need at least {needed}, got {got}"
            ),
            Self::SingularSystem => f.write_str("singular linear system (anchors degenerate)"),
            Self::NonFinite => f.write_str("estimator produced a non-finite value"),
            Self::DidNotConverge => {
                f.write_str("iterative estimator did not converge within max_iters")
            }
        }
    }
}

impl core::error::Error for EstimationError {}
