//! Desktop simulator primitives for skyfix.
//!
//! Provides anchor layouts, target trajectories, and noise-injecting
//! measurement simulators that feed into the estimators in `skyfix-core`.

use nalgebra::Vector2;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use skyfix_core::ToaMeasurement;

/// A range-measuring anchor with known position and noise model.
#[derive(Debug, Clone, Copy)]
pub struct Anchor2D {
    /// Anchor position in 2D.
    pub position: Vector2<f64>,
    /// Range-measurement variance (m²).
    pub range_variance: f64,
}

impl Anchor2D {
    pub const fn new(position: Vector2<f64>, range_variance: f64) -> Self {
        Self {
            position,
            range_variance,
        }
    }
}

/// Trait for a target's 2D motion. Pure function of time.
pub trait Trajectory2D {
    fn position_at(&self, t: f64) -> Vector2<f64>;
}

/// Constant-speed circular trajectory, useful as a default demo target.
#[derive(Debug, Clone, Copy)]
pub struct CircularTrajectory {
    pub center: Vector2<f64>,
    pub radius: f64,
    /// Angular speed in radians per second.
    pub angular_speed: f64,
    /// Initial phase (radians).
    pub phase0: f64,
}

impl CircularTrajectory {
    pub const fn new(center: Vector2<f64>, radius: f64, angular_speed: f64) -> Self {
        Self {
            center,
            radius,
            angular_speed,
            phase0: 0.0,
        }
    }
}

impl Trajectory2D for CircularTrajectory {
    fn position_at(&self, t: f64) -> Vector2<f64> {
        let theta = self.phase0 + self.angular_speed * t;
        Vector2::new(
            self.center.x + self.radius * theta.cos(),
            self.center.y + self.radius * theta.sin(),
        )
    }
}

/// Generates noisy ToA range measurements from a given target position to a
/// fixed anchor set, using Gaussian noise per anchor's `range_variance`.
pub struct ToASimulator2D {
    pub anchors: Vec<Anchor2D>,
}

impl ToASimulator2D {
    pub fn new(anchors: Vec<Anchor2D>) -> Self {
        Self { anchors }
    }

    pub fn measure<R: Rng>(
        &self,
        target: Vector2<f64>,
        rng: &mut R,
    ) -> Vec<ToaMeasurement<f64, 2>> {
        self.anchors
            .iter()
            .map(|a| {
                let true_range = (target - a.position).norm();
                let sigma = a.range_variance.sqrt();
                let normal = Normal::new(0.0, sigma).unwrap();
                let noise = normal.sample(rng);
                ToaMeasurement::new(a.position, (true_range + noise).max(0.0))
            })
            .collect()
    }
}

/// Per-step record of a simulation run, useful for plotting / CSV export.
#[derive(Debug, Clone, Copy)]
pub struct StepRecord {
    pub time: f64,
    pub truth: Vector2<f64>,
    pub estimate: Vector2<f64>,
}

impl StepRecord {
    pub fn error(&self) -> f64 {
        (self.truth - self.estimate).norm()
    }
}

/// Root-mean-squared error across a series of step records.
pub fn rmse(records: &[StepRecord]) -> f64 {
    if records.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = records.iter().map(|r| r.error().powi(2)).sum();
    (sum_sq / records.len() as f64).sqrt()
}
