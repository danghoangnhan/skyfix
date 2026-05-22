//! skyfix-sim: desktop demo of skyfix EKF tracking under noisy ToA ranges.
//!
//! Runs a 100-step circular-trajectory simulation with 4 anchors at the
//! corners of a 10×10 m room, feeds noisy ranges into an EKF seeded from
//! a `ToaTrilateration` cold start, prints aggregate stats, and writes
//! three CSV files for downstream plotting (gnuplot, matplotlib, Excel, …).
//!
//! ```sh
//! cargo run -p skyfix-sim --release
//! # → writes truth.csv, estimate.csv, anchors.csv to the current directory
//! ```

use nalgebra::{SMatrix, SVector, Vector2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use skyfix_core::{Ekf, Estimator, IdentityTransition, RangeAnchor, ToaTrilateration};
use skyfix_sim::{rmse, Anchor2D, CircularTrajectory, StepRecord, ToASimulator2D, Trajectory2D};
use std::fs::File;
use std::io::Write;

const STEPS: usize = 100;
const DT: f64 = 0.1;
const ROOM_SIZE: f64 = 10.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Scenario ─────────────────────────────────────────────────────────
    let anchors = vec![
        Anchor2D::new(Vector2::new(0.0, 0.0), 0.04),
        Anchor2D::new(Vector2::new(ROOM_SIZE, 0.0), 0.04),
        Anchor2D::new(Vector2::new(ROOM_SIZE, ROOM_SIZE), 0.04),
        Anchor2D::new(Vector2::new(0.0, ROOM_SIZE), 0.04),
    ];
    let sim = ToASimulator2D::new(anchors.clone());
    let trajectory = CircularTrajectory::new(
        Vector2::new(ROOM_SIZE / 2.0, ROOM_SIZE / 2.0),
        3.0,
        2.0 * std::f64::consts::PI / (STEPS as f64 * DT),
    );

    // ── EKF setup ────────────────────────────────────────────────────────
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    // Cold start: bootstrap the EKF with a trilateration fix on the first
    // noisy measurement set. Idiomatic skyfix pattern from CLAUDE.md.
    let initial_truth = trajectory.position_at(0.0);
    let bootstrap = sim.measure(initial_truth, &mut rng);
    let initial_state = ToaTrilateration::<f64, 2>::new()
        .estimate(&bootstrap)
        .expect("trilateration bootstrap ok");
    let initial_cov = SMatrix::<f64, 2, 2>::identity() * 0.5;
    let mut ekf = Ekf::<f64, 2>::new(initial_state, initial_cov);

    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.01);

    // ── Simulation loop ──────────────────────────────────────────────────
    let mut records: Vec<StepRecord> = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let t = step as f64 * DT;
        let truth = trajectory.position_at(t);
        let measurements = sim.measure(truth, &mut rng);

        ekf.predict(&transition, DT);
        for m in &measurements {
            let model = RangeAnchor::<f64, 2>::new(m.anchor, 0.04);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = m.range;
            ekf.update(&model, &z)?;
        }

        records.push(StepRecord {
            time: t,
            truth,
            estimate: ekf.state(),
        });
    }

    // ── Stats ────────────────────────────────────────────────────────────
    let err = rmse(&records);
    let final_err = records.last().unwrap().error();
    println!("skyfix-sim EKF tracking demo");
    println!(" steps      = {STEPS}");
    println!(" dt         = {DT} s");
    println!(
        " anchors    = {} at room corners ({}×{} m)",
        anchors.len(),
        ROOM_SIZE,
        ROOM_SIZE
    );
    println!(" range σ²   = 0.04 m² (σ = 0.2 m)");
    println!(" RMSE       = {err:.4} m");
    println!(" final err  = {final_err:.4} m");

    // ── CSV output ───────────────────────────────────────────────────────
    write_csv(&records, &anchors)?;
    println!(" wrote      = truth.csv, estimate.csv, anchors.csv");
    Ok(())
}

fn write_csv(records: &[StepRecord], anchors: &[Anchor2D]) -> std::io::Result<()> {
    let mut truth = File::create("truth.csv")?;
    writeln!(truth, "t,x,y")?;
    for r in records {
        writeln!(truth, "{},{},{}", r.time, r.truth.x, r.truth.y)?;
    }

    let mut estimate = File::create("estimate.csv")?;
    writeln!(estimate, "t,x,y,err")?;
    for r in records {
        writeln!(
            estimate,
            "{},{},{},{}",
            r.time,
            r.estimate.x,
            r.estimate.y,
            r.error()
        )?;
    }

    let mut anchor_file = File::create("anchors.csv")?;
    writeln!(anchor_file, "x,y,variance")?;
    for a in anchors {
        writeln!(
            anchor_file,
            "{},{},{}",
            a.position.x, a.position.y, a.range_variance
        )?;
    }
    Ok(())
}
