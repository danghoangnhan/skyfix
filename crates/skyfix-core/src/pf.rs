//! Sequential Importance Resampling (SIR) particle filter.
//!
//! Static-K design: the particle ensemble is a stack-allocated
//! `SMatrix<T, N, K>`, so the filter runs in pure `no_std` without an
//! allocator. K is a compile-time const generic — choose it once at the
//! call site. For dynamically-sized ensembles, gate behind the `alloc`
//! feature in a future phase.
//!
//! # RNG abstraction
//!
//! [`Pf`] does not depend on `rand`. Sampling enters via `FnMut() -> T`
//! closures: one for standard-normal samples (used in `predict` and
//! `from_gaussian`), one for uniform `[0, 1)` samples (used in `resample`).
//! Plug in any RNG you like — hardware TRNG, ChaCha20, a deterministic
//! seeded PRNG for tests.

use crate::error::EstimationError;
use crate::filter::{ObservationModel, TransitionModel};
use crate::numeric::{cholesky, invert_square};
use nalgebra::{ComplexField, RealField, SMatrix, SVector};

/// Sequential-Importance-Resampling particle filter.
///
/// `N` is the state dimension, `K` is the particle count (typically 64–512
/// for skyfix's UAV tracking workloads).
pub struct Pf<T: RealField + Copy, const N: usize, const K: usize> {
    particles: SMatrix<T, N, K>,
    log_weights: SVector<T, K>,
}

impl<T: RealField + Copy, const N: usize, const K: usize> Pf<T, N, K> {
    /// Construct a particle filter from an explicit particle ensemble.
    /// Weights are initialized to uniform (`1/K` each).
    pub fn from_particles(particles: SMatrix<T, N, K>) -> Self {
        let k_t: T = nalgebra::convert(K as f64);
        let log_uniform = -k_t.ln();
        let mut log_weights = SVector::<T, K>::zeros();
        for i in 0..K {
            log_weights[i] = log_uniform;
        }
        Self {
            particles,
            log_weights,
        }
    }

    /// Construct a particle filter by sampling `K` particles from a Gaussian
    /// `N(mean, cov)` prior. `normal` must yield standard-normal samples.
    pub fn from_gaussian<F: FnMut() -> T>(
        mean: SVector<T, N>,
        cov: SMatrix<T, N, N>,
        mut normal: F,
    ) -> Result<Self, EstimationError> {
        let l = cholesky(cov)?;
        let mut particles = SMatrix::<T, N, K>::zeros();
        for k in 0..K {
            let mut z = SVector::<T, N>::zeros();
            for n in 0..N {
                z[n] = normal();
            }
            let offset = l * z;
            for n in 0..N {
                particles[(n, k)] = mean[n] + offset[n];
            }
        }
        Ok(Self::from_particles(particles))
    }

    /// Weighted mean of the current ensemble (the canonical state estimate).
    pub fn state(&self) -> SVector<T, N> {
        let w = self.normalized_weights();
        let mut m = SVector::<T, N>::zeros();
        for k in 0..K {
            for n in 0..N {
                m[n] += w[k] * self.particles[(n, k)];
            }
        }
        m
    }

    /// Weighted covariance of the current ensemble.
    pub fn covariance(&self) -> SMatrix<T, N, N> {
        let m = self.state();
        let w = self.normalized_weights();
        let mut c = SMatrix::<T, N, N>::zeros();
        for k in 0..K {
            for r in 0..N {
                let dr = self.particles[(r, k)] - m[r];
                for col in 0..N {
                    let dc = self.particles[(col, k)] - m[col];
                    c[(r, col)] += w[k] * dr * dc;
                }
            }
        }
        c
    }

    /// Effective sample size (Kong et al. 1992): `1 / Σ w_i²`. Equal to `K`
    /// when weights are perfectly uniform; collapses toward 1 when one
    /// particle dominates.
    pub fn effective_sample_size(&self) -> T {
        let w = self.normalized_weights();
        let mut s = T::zero();
        for k in 0..K {
            s += w[k] * w[k];
        }
        T::one() / s
    }

    /// Direct read access to the particle ensemble (columns are particles).
    pub fn particles(&self) -> &SMatrix<T, N, K> {
        &self.particles
    }

    /// Normalized weights, length `K`, summing to 1.
    pub fn weights(&self) -> SVector<T, K> {
        self.normalized_weights()
    }

    /// Predict step: propagate every particle through the transition model
    /// and add a sample of Gaussian process noise drawn via `normal`.
    ///
    /// If the process-noise covariance returned by `model.noise(dt)` is
    /// singular (zero or PSD), each particle gets the deterministic
    /// transition with no noise added — `normal` is *not* called. This
    /// matches the common idiom of feeding `IdentityTransition::with_uniform_noise(0.0)`
    /// for deterministic-dynamics scenarios.
    pub fn predict<TM, F>(
        &mut self,
        model: &TM,
        dt: T,
        mut normal: F,
    ) -> Result<(), EstimationError>
    where
        TM: TransitionModel<T, N>,
        F: FnMut() -> T,
    {
        let q = model.noise(dt);
        let l_opt = cholesky(q).ok();
        for k in 0..K {
            let mut x = SVector::<T, N>::zeros();
            for n in 0..N {
                x[n] = self.particles[(n, k)];
            }
            let mean_next = model.transition(&x, dt);
            let new_x = if let Some(l) = &l_opt {
                let mut z = SVector::<T, N>::zeros();
                for n in 0..N {
                    z[n] = normal();
                }
                mean_next + l * z
            } else {
                mean_next
            };
            for n in 0..N {
                self.particles[(n, k)] = new_x[n];
            }
        }
        Ok(())
    }

    /// Update step: incorporate one measurement by adjusting log-weights
    /// according to the Gaussian likelihood `p(z | x_i)`.
    ///
    /// Drops the `−(M/2) log(2π) − ½ log|R|` constant — only relative
    /// log-likelihoods matter for resampling and weighted mean/cov.
    pub fn update<const M: usize, OM>(
        &mut self,
        model: &OM,
        z: &SVector<T, M>,
    ) -> Result<(), EstimationError>
    where
        OM: ObservationModel<T, N, M>,
    {
        let r = model.noise();
        let r_inv = invert_square(r)?;
        let half: T = nalgebra::convert(0.5);

        for k in 0..K {
            let mut x = SVector::<T, N>::zeros();
            for n in 0..N {
                x[n] = self.particles[(n, k)];
            }
            let predicted = model.predict(&x);
            let innovation = *z - predicted;

            let mut quadratic = T::zero();
            for r_idx in 0..M {
                for c_idx in 0..M {
                    quadratic += innovation[r_idx] * r_inv[(r_idx, c_idx)] * innovation[c_idx];
                }
            }
            self.log_weights[k] -= half * quadratic;
        }
        Ok(())
    }

    /// Systematic resampling. Pulls one uniform `[0, 1)` sample from
    /// `uniform`, then deterministically partitions the weighted CDF into
    /// `K` strata — variance-minimal among single-sample resampling schemes.
    /// After this call, weights are reset to uniform.
    pub fn resample<F: FnMut() -> T>(&mut self, mut uniform: F) {
        let weights = self.normalized_weights();

        // Cumulative distribution function.
        let mut cdf = SVector::<T, K>::zeros();
        cdf[0] = weights[0];
        for i in 1..K {
            cdf[i] = cdf[i - 1] + weights[i];
        }

        let k_t: T = nalgebra::convert(K as f64);
        let inv_k = T::one() / k_t;
        let u0 = uniform() * inv_k;

        let mut new_particles = SMatrix::<T, N, K>::zeros();
        let mut j: usize = 0;
        for i in 0..K {
            let i_t: T = nalgebra::convert(i as f64);
            let u = u0 + i_t * inv_k;
            while j + 1 < K && cdf[j] < u {
                j += 1;
            }
            for n in 0..N {
                new_particles[(n, i)] = self.particles[(n, j)];
            }
        }
        self.particles = new_particles;

        let log_uniform = -k_t.ln();
        for i in 0..K {
            self.log_weights[i] = log_uniform;
        }
    }

    /// Resample if `ESS < threshold_frac * K`. Returns whether resampling
    /// happened. `threshold_frac = 0.5` is a common default.
    pub fn resample_if_needed<F: FnMut() -> T>(&mut self, threshold_frac: T, uniform: F) -> bool {
        let k_t: T = nalgebra::convert(K as f64);
        if self.effective_sample_size() < threshold_frac * k_t {
            self.resample(uniform);
            true
        } else {
            false
        }
    }

    /// Log-sum-exp-stable normalization of log weights → linear weights.
    fn normalized_weights(&self) -> SVector<T, K> {
        let mut max_log = self.log_weights[0];
        for i in 1..K {
            if self.log_weights[i] > max_log {
                max_log = self.log_weights[i];
            }
        }
        let mut w = SVector::<T, K>::zeros();
        let mut sum = T::zero();
        for i in 0..K {
            let e = ComplexField::exp(self.log_weights[i] - max_log);
            w[i] = e;
            sum += e;
        }
        for i in 0..K {
            w[i] /= sum;
        }
        w
    }
}
