//! Multi-filter comparison demo: runs Trilateration / EKF / UKF / PF on the
//! *same* scenario (identical truth trajectory, identical noisy
//! measurements per step) and prints a per-method RMSE / max-error / wall-
//! time table. Writes a wide-format CSV for plotting.
//!
//! ```sh
//! cargo run -p skyfix-sim --release --example multi_filter_comparison
//! # → prints summary table, writes multi_comparison.csv
//! ```
//!
//! This is the showcase demo: it makes the practical trade-offs across the
//! four estimation strategies legible at a glance.

use nalgebra::{SMatrix, SVector, Vector2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use skyfix_core::{
    Ekf, Estimator, IdentityTransition, Pf, RangeAnchor, ToaMeasurement, ToaTrilateration, Ukf2D,
};
use skyfix_sim::{rmse, Anchor2D, CircularTrajectory, StepRecord, ToASimulator2D, Trajectory2D};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const STEPS: usize = 100;
const DT: f64 = 0.1;
const ROOM_SIZE: f64 = 10.0;
const RANGE_VARIANCE: f64 = 0.04; // σ = 0.2 m
const PF_PARTICLES: usize = 512;

/// Frozen scenario: truth at every step + pre-rolled noisy measurements per
/// step. All methods see *exactly* the same observations.
struct Scenario {
    truth: Vec<Vector2<f64>>,
    measurements: Vec<Vec<ToaMeasurement<f64, 2>>>,
}

impl Scenario {
    fn generate(seed: u64) -> Self {
        let anchors = vec![
            Anchor2D::new(Vector2::new(0.0, 0.0), RANGE_VARIANCE),
            Anchor2D::new(Vector2::new(ROOM_SIZE, 0.0), RANGE_VARIANCE),
            Anchor2D::new(Vector2::new(ROOM_SIZE, ROOM_SIZE), RANGE_VARIANCE),
            Anchor2D::new(Vector2::new(0.0, ROOM_SIZE), RANGE_VARIANCE),
        ];
        let sim = ToASimulator2D::new(anchors);
        let trajectory = CircularTrajectory::new(
            Vector2::new(ROOM_SIZE / 2.0, ROOM_SIZE / 2.0),
            3.0,
            2.0 * std::f64::consts::PI / (STEPS as f64 * DT),
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut truth = Vec::with_capacity(STEPS);
        let mut measurements = Vec::with_capacity(STEPS);
        for step in 0..STEPS {
            let t = step as f64 * DT;
            let pos = trajectory.position_at(t);
            measurements.push(sim.measure(pos, &mut rng));
            truth.push(pos);
        }
        Self {
            truth,
            measurements,
        }
    }
}

struct Run {
    name: &'static str,
    records: Vec<StepRecord>,
    elapsed: Duration,
}

impl Run {
    fn rmse(&self) -> f64 {
        rmse(&self.records)
    }
    fn max_err(&self) -> f64 {
        self.records
            .iter()
            .map(|r| r.error())
            .fold(0.0_f64, f64::max)
    }
}

fn run_trilateration(s: &Scenario) -> Run {
    let estimator = ToaTrilateration::<f64, 2>::new();
    let start = Instant::now();
    let records: Vec<StepRecord> = s
        .truth
        .iter()
        .zip(&s.measurements)
        .enumerate()
        .map(|(i, (&truth, m))| StepRecord {
            time: i as f64 * DT,
            truth,
            estimate: estimator.estimate(m).expect("trilateration ok"),
        })
        .collect();
    Run {
        name: "Trilateration",
        records,
        elapsed: start.elapsed(),
    }
}

fn run_ekf(s: &Scenario) -> Run {
    let start = Instant::now();
    let trilat = ToaTrilateration::<f64, 2>::new();
    let seed = trilat.estimate(&s.measurements[0]).expect("seed ok");
    let mut ekf = Ekf::<f64, 2>::new(seed, SMatrix::<f64, 2, 2>::identity() * 0.5);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.01);

    let mut records = Vec::with_capacity(STEPS);
    for (i, (&truth, m)) in s.truth.iter().zip(&s.measurements).enumerate() {
        ekf.predict(&transition, DT);
        for meas in m {
            let model = RangeAnchor::<f64, 2>::new(meas.anchor, RANGE_VARIANCE);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = meas.range;
            ekf.update(&model, &z).expect("ekf update ok");
        }
        records.push(StepRecord {
            time: i as f64 * DT,
            truth,
            estimate: ekf.state(),
        });
    }
    Run {
        name: "EKF (seeded)",
        records,
        elapsed: start.elapsed(),
    }
}

fn run_ukf(s: &Scenario) -> Run {
    let start = Instant::now();
    let trilat = ToaTrilateration::<f64, 2>::new();
    let seed = trilat.estimate(&s.measurements[0]).expect("seed ok");
    let mut ukf = Ukf2D::<f64>::with_defaults(seed, SMatrix::<f64, 2, 2>::identity() * 0.5);
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.01);

    let mut records = Vec::with_capacity(STEPS);
    for (i, (&truth, m)) in s.truth.iter().zip(&s.measurements).enumerate() {
        ukf.predict(&transition, DT).expect("ukf predict ok");
        for meas in m {
            let model = RangeAnchor::<f64, 2>::new(meas.anchor, RANGE_VARIANCE);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = meas.range;
            ukf.update(&model, &z).expect("ukf update ok");
        }
        records.push(StepRecord {
            time: i as f64 * DT,
            truth,
            estimate: ukf.state(),
        });
    }
    Run {
        name: "UKF (warm)",
        records,
        elapsed: start.elapsed(),
    }
}

fn run_pf(s: &Scenario) -> Run {
    let mut rng = ChaCha8Rng::seed_from_u64(0xF11E2);
    let start = Instant::now();
    let trilat = ToaTrilateration::<f64, 2>::new();
    let seed = trilat.estimate(&s.measurements[0]).expect("seed ok");
    let mut pf = Pf::<f64, 2, PF_PARTICLES>::from_gaussian(
        seed,
        SMatrix::<f64, 2, 2>::identity() * 0.25,
        || StandardNormal.sample(&mut rng),
    )
    .expect("pf init ok");

    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.01);

    let mut records = Vec::with_capacity(STEPS);
    for (i, (&truth, m)) in s.truth.iter().zip(&s.measurements).enumerate() {
        pf.predict(&transition, DT, || StandardNormal.sample(&mut rng))
            .expect("pf predict ok");
        for meas in m {
            let model = RangeAnchor::<f64, 2>::new(meas.anchor, RANGE_VARIANCE);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = meas.range;
            pf.update(&model, &z).expect("pf update ok");
        }
        pf.resample_if_needed(0.5, || {
            use rand::Rng;
            rng.random::<f64>()
        });
        records.push(StepRecord {
            time: i as f64 * DT,
            truth,
            estimate: pf.state(),
        });
    }
    Run {
        name: "PF (K=512)",
        records,
        elapsed: start.elapsed(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scenario = Scenario::generate(0xC0FFEE);

    let runs = [
        run_trilateration(&scenario),
        run_ekf(&scenario),
        run_ukf(&scenario),
        run_pf(&scenario),
    ];

    // ── Summary table ─────────────────────────────────────────────────────
    println!("skyfix multi-filter comparison");
    println!(" scenario : {STEPS} steps × {DT} s, {ROOM_SIZE}×{ROOM_SIZE} m room");
    let sigma = RANGE_VARIANCE.sqrt();
    println!(" anchors  : 4 at corners, σ = {sigma:.2} m");
    println!(" trajectory : circular, radius 3 m, full revolution");
    println!();
    println!(
        " {:<16}  {:>10}  {:>10}  {:>12}",
        "method", "RMSE (m)", "max (m)", "wall time"
    );
    println!(" {:─<16}  {:─>10}  {:─>10}  {:─>12}", "", "", "", "");
    for r in &runs {
        println!(
            " {:<16}  {:>10.4}  {:>10.4}  {:>10.2?}",
            r.name,
            r.rmse(),
            r.max_err(),
            r.elapsed,
        );
    }

    // ── Wide-format CSV ───────────────────────────────────────────────────
    let path = "multi_comparison.csv";
    let mut f = File::create(path)?;
    let header_methods: Vec<String> = runs
        .iter()
        .flat_map(|r| {
            let prefix = r.name.split_whitespace().next().unwrap().to_lowercase();
            [
                format!("{prefix}_x"),
                format!("{prefix}_y"),
                format!("{prefix}_err"),
            ]
        })
        .collect();
    writeln!(f, "t,truth_x,truth_y,{}", header_methods.join(","))?;
    for i in 0..STEPS {
        let truth = scenario.truth[i];
        let mut row = format!("{},{},{}", i as f64 * DT, truth.x, truth.y);
        for r in &runs {
            let rec = &r.records[i];
            row.push_str(&format!(
                ",{},{},{}",
                rec.estimate.x,
                rec.estimate.y,
                rec.error()
            ));
        }
        writeln!(f, "{row}")?;
    }
    println!();
    println!(" wrote    : {path}");
    Ok(())
}
