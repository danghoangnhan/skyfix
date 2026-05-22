//! 2D particle filter on the GPU.
//!
//! State and weights live on the device across `predict` / `update` calls;
//! only `mean()` / `effective_sample_size()` / `download()` transfer back to
//! the host. The dynamics model is hardcoded to identity (no deterministic
//! drift) plus Cholesky-correlated process noise sampled on the host — same
//! interface contract as [`skyfix_core::IdentityTransition`] + the `normal`
//! sampler argument to [`skyfix_core::Pf::predict`].
//!
//! Host-side noise sampling is deliberate: it makes cross-validation against
//! the CPU `Pf` straightforward (feed both filters the same noise sequence
//! → bit-identical particle motion, modulo f32↔f64). A future revision can
//! swap in device-side cuRAND for higher throughput at large K.
//!
//! # Pipeline
//!
//! ```text
//!     ┌─────────────────────┐     ┌─────────────────────────────┐
//!     │ host: rng.normal()  │ ──► │ device: pf_predict_2d       │
//!     │ × 2K samples / step │     │   x_i ← x_i + L · z_i        │
//!     └─────────────────────┘     └─────────────────────────────┘
//!                                                │
//!                                                ▼
//!                                 ┌─────────────────────────────┐
//!                                 │ device: pf_update_range_2d  │
//!                                 │   log_w_i −= ½ · innov²/var  │
//!                                 └─────────────────────────────┘
//!                                                │
//!                                                ▼
//!                              host: mean() / ESS() / download()
//! ```

use crate::CudaError;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const PTX_PF_KERNELS_2D: &str = include_str!(env!("PTX_PF_KERNELS_2D"));

/// GPU-resident 2D particle filter for range-to-anchor measurements.
pub struct CudaPfRanges2D {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    predict_fn: CudaFunction,
    update_fn: CudaFunction,
    particles_dev: CudaSlice<f32>,
    log_weights_dev: CudaSlice<f32>,
    k: usize,
}

impl CudaPfRanges2D {
    /// Initialize a particle filter from a flat `2K` array of `(x, y)`
    /// particle coordinates. Log-weights start uniform at `−ln K` (so that
    /// `exp(log_w)` already sums to 1 before any update).
    ///
    /// # Errors
    /// - [`CudaError::NoParticles`] if `initial_particles` is empty.
    /// - [`CudaError::SizeMismatch`] if length is not a multiple of 2.
    pub fn new(initial_particles: &[f32]) -> Result<Self, CudaError> {
        if initial_particles.is_empty() {
            return Err(CudaError::NoParticles);
        }
        if initial_particles.len() % 2 != 0 {
            return Err(CudaError::SizeMismatch {
                expected: initial_particles.len() + 1,
                got: initial_particles.len(),
            });
        }
        let k = initial_particles.len() / 2;

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX_PF_KERNELS_2D))?;
        let predict_fn = module.load_function("pf_predict_2d")?;
        let update_fn = module.load_function("pf_update_range_2d")?;

        let particles_dev = stream.clone_htod(initial_particles)?;
        let log_uniform = -((k as f32).ln());
        let log_weights_init = vec![log_uniform; k];
        let log_weights_dev = stream.clone_htod(&log_weights_init)?;

        Ok(Self {
            _ctx: ctx,
            stream,
            _module: module,
            predict_fn,
            update_fn,
            particles_dev,
            log_weights_dev,
            k,
        })
    }

    /// Particle count `K`.
    pub fn particle_count(&self) -> usize {
        self.k
    }

    /// Predict step: apply `x_i ← x_i + L · z_i` for every particle, where
    /// `L` is the lower-triangular Cholesky factor of the 2×2 process-noise
    /// covariance `Q` (so `L · Lᵀ = Q`) and `z_i` is a 2-vector of
    /// standard-normal samples from `noise`.
    ///
    /// `noise` must be `2 * K` long, with `(noise[2i], noise[2i+1])` paired
    /// per particle `i`.
    pub fn predict(&mut self, cholesky_l: [[f32; 2]; 2], noise: &[f32]) -> Result<(), CudaError> {
        let expected = 2 * self.k;
        if noise.len() != expected {
            return Err(CudaError::SizeMismatch {
                expected,
                got: noise.len(),
            });
        }
        let noise_dev = self.stream.clone_htod(noise)?;
        let cfg = LaunchConfig::for_num_elems(self.k as u32);
        let k_i = self.k as i32;
        let l00 = cholesky_l[0][0];
        let l10 = cholesky_l[1][0];
        let l11 = cholesky_l[1][1];

        unsafe {
            let mut launcher = self.stream.launch_builder(&self.predict_fn);
            launcher
                .arg(&mut self.particles_dev)
                .arg(&noise_dev)
                .arg(&k_i)
                .arg(&l00)
                .arg(&l10)
                .arg(&l11);
            launcher.launch(cfg)?;
        }
        Ok(())
    }

    /// Update step: ingest one range measurement `z` from `anchor` with
    /// Gaussian noise variance `variance`. Every particle's log-weight is
    /// decremented by `½ · (z − ‖x_i − anchor‖)² / variance`.
    pub fn update(
        &mut self,
        anchor: [f32; 2],
        range_z: f32,
        variance: f32,
    ) -> Result<(), CudaError> {
        let cfg = LaunchConfig::for_num_elems(self.k as u32);
        let k_i = self.k as i32;
        let ax = anchor[0];
        let ay = anchor[1];

        unsafe {
            let mut launcher = self.stream.launch_builder(&self.update_fn);
            launcher
                .arg(&self.particles_dev)
                .arg(&mut self.log_weights_dev)
                .arg(&k_i)
                .arg(&ax)
                .arg(&ay)
                .arg(&range_z)
                .arg(&variance);
            launcher.launch(cfg)?;
        }
        Ok(())
    }

    /// Download particles (`2K` floats) and log-weights (`K` floats) to host.
    pub fn download(&self) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let particles = self.stream.clone_dtoh(&self.particles_dev)?;
        let log_weights = self.stream.clone_dtoh(&self.log_weights_dev)?;
        Ok((particles, log_weights))
    }

    /// Weighted mean of the ensemble, computed on the host via log-sum-exp.
    pub fn mean(&self) -> Result<[f32; 2], CudaError> {
        let (particles, weights) = self.normalized_weights()?;
        let mut mean = [0.0_f32; 2];
        for i in 0..self.k {
            mean[0] += weights[i] * particles[2 * i];
            mean[1] += weights[i] * particles[2 * i + 1];
        }
        Ok(mean)
    }

    /// Effective sample size `1 / Σ wᵢ²` for the current weights.
    pub fn effective_sample_size(&self) -> Result<f32, CudaError> {
        let (_, weights) = self.normalized_weights()?;
        let sum_sq: f32 = weights.iter().map(|w| w * w).sum();
        Ok(1.0 / sum_sq)
    }

    /// Returns `(particles, normalized_weights)` — `weights` sum to 1.
    fn normalized_weights(&self) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let (particles, log_weights) = self.download()?;
        let max_log = log_weights
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights = vec![0.0_f32; self.k];
        let mut sum = 0.0_f32;
        for i in 0..self.k {
            weights[i] = (log_weights[i] - max_log).exp();
            sum += weights[i];
        }
        for w in &mut weights {
            *w /= sum;
        }
        Ok((particles, weights))
    }
}
