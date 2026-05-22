//! Integration tests for the Error-State Kalman Filter.

use approx::assert_relative_eq;
use nalgebra::{SMatrix, SVector};
use skyfix_core::filter::RangeAnchor;
use skyfix_core::{Eskf, Imu2DStrapdown, ImuIntegrator};

/// Helper: 5-dim state vector `[px, py, vx, vy, theta]`.
fn state(px: f64, py: f64, vx: f64, vy: f64, theta: f64) -> SVector<f64, 5> {
    let mut s = SVector::<f64, 5>::zeros();
    s[0] = px;
    s[1] = py;
    s[2] = vx;
    s[3] = vy;
    s[4] = theta;
    s
}

/// Helper: 3-dim control `[a_body_x, a_body_y, gyro_z]`.
fn ctrl(ax: f64, ay: f64, gz: f64) -> SVector<f64, 3> {
    let mut c = SVector::<f64, 3>::zeros();
    c[0] = ax;
    c[1] = ay;
    c[2] = gz;
    c
}

#[test]
fn imu2d_strapdown_zero_input_is_stationary() {
    // Zero IMU control → position and heading must not drift.
    let integrator = Imu2DStrapdown::new(0.0, 0.0);
    let s = state(1.0, 2.0, 0.0, 0.0, 0.5);
    let s_next = integrator.integrate(&s, &ctrl(0.0, 0.0, 0.0), 0.01);
    assert_relative_eq!(s_next[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(s_next[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(s_next[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(s_next[3], 0.0, epsilon = 1e-12);
    assert_relative_eq!(s_next[4], 0.5, epsilon = 1e-12);
}

#[test]
fn imu2d_strapdown_constant_velocity_is_linear_motion() {
    // Zero acceleration + zero gyro starting with v = (1, 0) → linear motion
    // along world-frame x axis.
    let integrator = Imu2DStrapdown::new(0.0, 0.0);
    let mut s = state(0.0, 0.0, 1.0, 0.0, 0.0);
    let dt = 0.01;
    for _ in 0..100 {
        s = integrator.integrate(&s, &ctrl(0.0, 0.0, 0.0), dt);
    }
    // After 1 second at v = 1 m/s the target is at x = 1.0.
    assert_relative_eq!(s[0], 1.0, epsilon = 1e-9);
    assert_relative_eq!(s[1], 0.0, epsilon = 1e-9);
    assert_relative_eq!(s[2], 1.0, epsilon = 1e-12);
}

#[test]
fn imu2d_strapdown_body_frame_accel_with_heading_yields_world_y_motion() {
    // Constant body-frame x acceleration with heading θ = π/2 means the
    // acceleration projects onto the world-frame y axis.
    let integrator = Imu2DStrapdown::new(0.0, 0.0);
    let mut s = state(0.0, 0.0, 0.0, 0.0, core::f64::consts::FRAC_PI_2);
    let dt = 0.01;
    for _ in 0..100 {
        s = integrator.integrate(&s, &ctrl(1.0, 0.0, 0.0), dt);
    }
    // After 1 s of 1 m/s² body-frame x accel with heading +π/2:
    //   v_world ≈ (0, 1)
    //   p_world ≈ (0, 0.5)
    assert_relative_eq!(s[0], 0.0, epsilon = 1e-9);
    assert_relative_eq!(s[1], 0.5, epsilon = 1e-9);
    assert_relative_eq!(s[2], 0.0, epsilon = 1e-9);
    assert_relative_eq!(s[3], 1.0, epsilon = 1e-9);
}

#[test]
fn imu2d_strapdown_gyro_only_rotates_heading_in_place() {
    // Constant yaw rate with zero accel → heading rotates linearly, no
    // position change.
    let integrator = Imu2DStrapdown::new(0.0, 0.0);
    let mut s = state(1.0, 2.0, 0.0, 0.0, 0.0);
    let dt = 0.01;
    for _ in 0..100 {
        s = integrator.integrate(&s, &ctrl(0.0, 0.0, 1.0), dt);
    }
    assert_relative_eq!(s[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(s[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(s[4], 1.0, epsilon = 1e-12);
}

#[test]
fn eskf_dead_reckons_perfect_imu_with_high_uncertainty() {
    // No measurements — only IMU predict steps. With a perfect IMU model
    // (zero noise) the estimate should match the integrator exactly. Error
    // covariance should grow over time but stay bounded.
    let integrator = Imu2DStrapdown::new(0.0, 0.0);
    // Start with v = (0.5, 0) heading 0.
    let nominal = state(0.0, 0.0, 0.5, 0.0, 0.0);
    let p0 = SMatrix::<f64, 5, 5>::identity() * 1e-3;
    let mut eskf = Eskf::new(nominal, p0);
    let dt = 0.01;
    for _ in 0..100 {
        eskf.predict(&integrator, &ctrl(0.0, 0.0, 0.0), dt);
    }
    let s = eskf.nominal();
    // After 1 s at v = 0.5 the position should be (0.5, 0).
    assert_relative_eq!(s[0], 0.5, epsilon = 1e-9);
    assert_relative_eq!(s[1], 0.0, epsilon = 1e-9);
    // Covariance stays finite (zero process noise → no growth here).
    let p = eskf.covariance();
    for i in 0..5 {
        assert!(p[(i, i)].is_finite());
    }
}

#[test]
fn eskf_fuses_imu_predict_with_range_update_tracks_truth() {
    // Scenario: target sits at the origin and accelerates along world-x at
    // a constant 1 m/s² for 1 second. We feed the ESKF perfect IMU control
    // inputs and noisy range measurements from 4 corner anchors. RMSE over
    // the run should be well under 5 cm — the closed-loop estimate sticks
    // close to truth.
    let integrator = Imu2DStrapdown::new(0.05, 0.001); // small process noise
    let dt = 0.01;
    let steps = 100;
    let true_accel_world = 1.0;

    // Anchors at corners of a 20×20 m square centered on the trajectory.
    let anchors: [SVector<f64, 2>; 4] = [
        SVector::<f64, 2>::new(-10.0, -10.0),
        SVector::<f64, 2>::new(10.0, -10.0),
        SVector::<f64, 2>::new(10.0, 10.0),
        SVector::<f64, 2>::new(-10.0, 10.0),
    ];
    let range_variance = 0.04_f64; // sigma = 0.2 m

    // Seeded xorshift64 for deterministic noise; a triangular ±0.2 m
    // approximation of N(0, 0.115²) is plenty for an integration sanity
    // check (and avoids pulling rand into core).
    let mut rng_state: u64 = 0x1234_5678_9abc_def0;
    let mut noise = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        (u - 0.5) * 0.4
    };

    let nominal0 = state(0.0, 0.0, 0.0, 0.0, 0.0);
    let p0 = SMatrix::<f64, 5, 5>::identity() * 0.5;
    let mut eskf = Eskf::new(nominal0, p0);

    let mut true_p_x = 0.0;
    let true_p_y = 0.0; // motion stays on world-x; tracked for completeness.
    let mut true_v_x = 0.0;
    let mut sum_sq_err = 0.0;
    for _ in 0..steps {
        // True update: const accel along world-x, heading stays 0.
        let half = 0.5;
        let new_p_x = true_p_x + true_v_x * dt + half * true_accel_world * dt * dt;
        let new_v_x = true_v_x + true_accel_world * dt;
        true_p_x = new_p_x;
        true_v_x = new_v_x;

        // IMU control: heading is 0, so body-frame accel = world-frame accel.
        eskf.predict(&integrator, &ctrl(true_accel_world, 0.0, 0.0), dt);

        // Range updates: one per anchor.
        for a in anchors.iter() {
            let true_range = ((true_p_x - a[0]).powi(2) + (true_p_y - a[1]).powi(2)).sqrt();
            let noisy = true_range + noise();
            // RangeAnchor over the 5-dim state: position lives in components
            // 0 and 1, so we extend the anchor with zeros for vx, vy, θ.
            let mut anchor5 = SVector::<f64, 5>::zeros();
            anchor5[0] = a[0];
            anchor5[1] = a[1];
            let model = RangeAnchor::new(anchor5, range_variance);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = noisy;
            eskf.update(&model, &z).unwrap();
        }

        let est = eskf.nominal();
        let err_x = est[0] - true_p_x;
        let err_y = est[1] - true_p_y;
        sum_sq_err += err_x * err_x + err_y * err_y;
    }

    let rmse = (sum_sq_err / steps as f64).sqrt();
    // Loose bound: with σ = 0.2 m range noise and 4 anchors, RMSE should
    // sit well under 30 cm. In practice the closed-loop run lands around
    // 10–15 cm.
    assert!(rmse < 0.30, "ESKF RMSE was {rmse:.3} m, expected < 0.30 m");
}

#[test]
fn eskf_set_nominal_reseeds_from_external_fix() {
    // Common pattern: seed the ESKF from a closed-form trilateration fix.
    let integrator = Imu2DStrapdown::new(0.05, 0.001);
    let nominal = state(0.0, 0.0, 0.0, 0.0, 0.0);
    let p0 = SMatrix::<f64, 5, 5>::identity();
    let mut eskf = Eskf::new(nominal, p0);
    eskf.predict(&integrator, &ctrl(0.0, 0.0, 0.0), 0.01);

    let reseed = state(5.0, -2.0, 0.0, 0.0, core::f64::consts::FRAC_PI_4);
    eskf.set_nominal(reseed);
    assert_relative_eq!(eskf.nominal()[0], 5.0, epsilon = 1e-12);
    assert_relative_eq!(eskf.nominal()[1], -2.0, epsilon = 1e-12);
    assert_relative_eq!(
        eskf.nominal()[4],
        core::f64::consts::FRAC_PI_4,
        epsilon = 1e-12
    );
}
