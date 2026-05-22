//! Idiomatic skyfix pipeline: bootstrap the EKF from a closed-form
//! `ToaTrilateration` fix on the first measurement set, then track with the
//! EKF on subsequent steps. Validates against a single static-target run.
//!
//! Run with: `cargo run -p skyfix-sim --example trilateration_to_ekf`

use nalgebra::{SMatrix, SVector, Vector2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use skyfix_core::{Ekf, Estimator, IdentityTransition, RangeAnchor, ToaTrilateration};
use skyfix_sim::{Anchor2D, ToASimulator2D};

fn main() {
    let truth = Vector2::new(2.0_f64, 3.0);
    let anchors = vec![
        Anchor2D::new(Vector2::new(0.0_f64, 0.0), 0.01),
        Anchor2D::new(Vector2::new(5.0, 0.0), 0.01),
        Anchor2D::new(Vector2::new(0.0, 5.0), 0.01),
        Anchor2D::new(Vector2::new(5.0, 5.0), 0.01),
    ];
    let sim = ToASimulator2D::new(anchors);
    let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);

    // ── 1. Cold start: trilateration on first measurement set ─────────────
    let first = sim.measure(truth, &mut rng);
    let initial = ToaTrilateration::<f64, 2>::new()
        .estimate(&first)
        .expect("trilateration ok");
    println!("After trilateration bootstrap:");
    println!(" truth     = ({:.4}, {:.4})", truth.x, truth.y);
    println!(" estimate  = ({:.4}, {:.4})", initial.x, initial.y);
    println!(" error     = {:.4} m", (truth - initial).norm());

    // ── 2. EKF refinement: 30 update cycles tighten the estimate ──────────
    let mut ekf = Ekf::<f64, 2>::new(initial, SMatrix::<f64, 2, 2>::identity() * 0.1);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-6);
    for _ in 0..30 {
        ekf.predict(&transition, 1.0);
        let measurements = sim.measure(truth, &mut rng);
        for m in &measurements {
            let model = RangeAnchor::<f64, 2>::new(m.anchor, 0.01);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = m.range;
            ekf.update(&model, &z).expect("update ok");
        }
    }

    println!();
    println!("After 30 EKF cycles:");
    println!(" estimate  = ({:.6}, {:.6})", ekf.state().x, ekf.state().y);
    println!(" error     = {:.6} m", (truth - ekf.state()).norm());
    println!(" cov[(0,0)] = {:.6}", ekf.covariance()[(0, 0)]);
    println!(" cov[(1,1)] = {:.6}", ekf.covariance()[(1, 1)]);
}
