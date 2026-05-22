//! NVIDIA CUDA acceleration for skyfix.
//!
//! This crate sits on top of the NVIDIA CUDA Toolkit via [cudarc] and provides
//! GPU implementations of operations where parallel evaluation is worth the
//! kernel-launch overhead. The current op surface is the batched 2D GDOP
//! sweep — embarrassingly parallel over a grid of candidate target positions,
//! the canonical workload for anchor-placement planning.
//!
//! # Architectural placement
//!
//! `skyfix-cuda` is a **separate workspace crate**, never a feature flag on
//! [`skyfix_core`]. This is deliberate: putting CUDA behind a `skyfix_core`
//! feature would force CUDA-toolkit awareness into every embedded build's
//! dependency resolution. Algorithms that benefit from GPU acceleration are
//! ported here; the CPU surface in `skyfix_core` is unchanged and remains the
//! reference implementation for cross-validation.
//!
//! # Layering
//!
//! ```text
//!     skyfix-cuda  ─────────────────────►  cudarc 0.19 (driver + cuda-12000)
//!                                              │
//!                                              ▼
//!                                          libcuda.so / libcudart.so
//!                                              │
//!                                              ▼
//!                                          NVIDIA driver + GPU
//! ```
//!
//! Kernels are written in CUDA C++ under `kernels/` and compiled to PTX by
//! `build.rs` (`nvcc -ptx --gpu-architecture=compute_70`). PTX is embedded
//! via `include_str!` and JIT-compiled at module load by the driver, giving
//! a single binary that runs across compute capabilities 7.0 (Volta) through
//! 12.0 (Blackwell, including the RTX 5090).

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

mod pf;
pub use pf::CudaPfRanges2D;

const PTX_GDOP_2D: &str = include_str!(env!("PTX_GDOP_2D"));

/// Error type for skyfix-cuda operations.
#[derive(Debug)]
pub enum CudaError {
    /// Underlying CUDA driver failure (launch, allocation, copy, etc.).
    Driver(DriverError),
    /// Empty grid supplied to a sweep.
    EmptyGrid,
    /// Empty anchor list.
    NoAnchors,
    /// A particle-filter input slice has the wrong length for the configured K.
    SizeMismatch { expected: usize, got: usize },
    /// The particle filter was initialized with zero particles.
    NoParticles,
}

impl From<DriverError> for CudaError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::EmptyGrid => f.write_str("grid must contain at least one cell"),
            Self::NoAnchors => f.write_str("at least one anchor is required"),
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected}, got {got}")
            }
            Self::NoParticles => f.write_str("particle filter initialized with zero particles"),
        }
    }
}

impl std::error::Error for CudaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(e) => Some(e),
            _ => None,
        }
    }
}

/// Single-anchor record packed for transfer to the GPU.
#[derive(Debug, Clone, Copy)]
pub struct Anchor2D {
    pub x: f32,
    pub y: f32,
    /// Range-measurement variance (m²).
    pub variance: f32,
}

impl Anchor2D {
    pub const fn new(x: f32, y: f32, variance: f32) -> Self {
        Self { x, y, variance }
    }
}

/// GPU-resident batched 2D GDOP sweep.
///
/// Construct once with an anchor set, then call [`Self::compute_grid`] with
/// any `(xs, ys)` pair to obtain GDOP values for every cell of the
/// `xs.len() × ys.len()` grid. Output is row-major with `y` outer:
///
/// ```text
///     output[iy * xs.len() + ix] = GDOP at (xs[ix], ys[iy])
/// ```
pub struct CudaGdopSweep2D {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    func: CudaFunction,
    anchors_dev: cudarc::driver::CudaSlice<f32>,
    n_anchors: usize,
}

impl CudaGdopSweep2D {
    /// Initialize a sweep with the supplied anchor set on GPU 0.
    pub fn new(anchors: &[Anchor2D]) -> Result<Self, CudaError> {
        if anchors.is_empty() {
            return Err(CudaError::NoAnchors);
        }

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(PTX_GDOP_2D))?;
        let func = module.load_function("gdop_2d")?;

        // Pack (x, y, variance) for transfer.
        let mut flat = Vec::with_capacity(anchors.len() * 3);
        for a in anchors {
            flat.push(a.x);
            flat.push(a.y);
            flat.push(a.variance);
        }
        let anchors_dev = stream.clone_htod(&flat)?;

        Ok(Self {
            _ctx: ctx,
            stream,
            _module: module,
            func,
            anchors_dev,
            n_anchors: anchors.len(),
        })
    }

    /// Compute GDOP at every cell of the cartesian product `xs × ys`.
    ///
    /// Returns a row-major `Vec<f32>` of length `xs.len() * ys.len()`.
    /// Cells whose Fisher Information matrix is singular (zero or negative
    /// determinant) are filled with `f32::INFINITY`.
    pub fn compute_grid(&self, xs: &[f32], ys: &[f32]) -> Result<Vec<f32>, CudaError> {
        if xs.is_empty() || ys.is_empty() {
            return Err(CudaError::EmptyGrid);
        }
        let n_cells = xs.len() * ys.len();

        let xs_dev = self.stream.clone_htod(xs)?;
        let ys_dev = self.stream.clone_htod(ys)?;
        let mut output_dev = self.stream.alloc_zeros::<f32>(n_cells)?;

        let cfg = LaunchConfig::for_num_elems(n_cells as u32);
        let n_anchors = self.n_anchors as i32;
        let n_xs = xs.len() as i32;
        let n_ys = ys.len() as i32;

        unsafe {
            let mut launcher = self.stream.launch_builder(&self.func);
            launcher
                .arg(&self.anchors_dev)
                .arg(&n_anchors)
                .arg(&xs_dev)
                .arg(&n_xs)
                .arg(&ys_dev)
                .arg(&n_ys)
                .arg(&mut output_dev);
            launcher.launch(cfg)?;
        }

        Ok(self.stream.clone_dtoh(&output_dev)?)
    }
}
