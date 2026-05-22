//! Integration tests for the particle filter.

use approx::assert_relative_eq;
use nalgebra::{SMatrix, SVector, Vector2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use skyfix_core::{IdentityTransition, Pf, RangeAnchor};

/// Pull one f64 from the chosen ChaCha8 stream for either distribution.
fn make_normal<'a>(rng: &'a mut ChaCha8Rng) -> impl FnMut() -> f64 + 'a {
    move || StandardNormal.sample(rng)
}

fn make_uniform<'a>(rng: &'a mut ChaCha8Rng) -> impl FnMut() -> f64 + 'a {
    use rand::Rng;
    move || rng.random::<f64>()
}

#[test]
fn pf_from_gaussian_recovers_prior_mean_in_expectation() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
    let prior_mean = Vector2::new(1.0_f64, 2.0);
    let prior_cov = SMatrix::<f64, 2, 2>::identity() * 0.25;

    let pf = Pf::<f64, 2, 1024>::from_gaussian(prior_mean, prior_cov, make_normal(&mut rng))
        .expect("from_gaussian ok");

    // Sample mean of 1024 particles should be close to the prior mean.
    assert_relative_eq!(pf.state(), prior_mean, epsilon = 0.05);
    // Diagonal of weighted covariance should approximate the prior diagonal.
    let cov = pf.covariance();
    assert_relative_eq!(cov[(0, 0)], 0.25, epsilon = 0.05);
    assert_relative_eq!(cov[(1, 1)], 0.25, epsilon = 0.05);
}

#[test]
fn pf_initial_ess_equals_k_with_uniform_weights() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let pf = Pf::<f64, 2, 256>::from_gaussian(
        Vector2::new(0.0, 0.0),
        SMatrix::<f64, 2, 2>::identity(),
        make_normal(&mut rng),
    )
    .expect("ok");
    assert_relative_eq!(pf.effective_sample_size(), 256.0, epsilon = 1e-9);
}

#[test]
fn pf_tracks_stationary_target_with_range_measurements() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let truth = Vector2::new(2.0_f64, 3.0);
    let prior_cov = SMatrix::<f64, 2, 2>::identity() * 4.0;
    let mut pf = Pf::<f64, 2, 512>::from_gaussian(
        Vector2::new(1.0_f64, 1.0),
        prior_cov,
        make_normal(&mut rng),
    )
    .expect("ok");

    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(1e-4);
    let anchors = [
        Vector2::new(0.0_f64, 0.0),
        Vector2::new(5.0, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];

    for _step in 0..30 {
        pf.predict(&transition, 1.0, make_normal(&mut rng))
            .expect("predict ok");
        for &anchor in &anchors {
            let model = RangeAnchor::<f64, 2>::new(anchor, 0.04_f64);
            let mut z = SVector::<f64, 1>::zeros();
            z[0] = (truth - anchor).norm();
            pf.update(&model, &z).expect("update ok");
        }
        pf.resample_if_needed(0.5, make_uniform(&mut rng));
    }

    let err = (pf.state() - truth).norm();
    assert!(err < 0.2, "PF tracking error too large: {err}");
}

#[test]
fn pf_resample_after_skewed_weights_restores_ess_to_k() {
    // After an extreme observation, ESS collapses; resampling resets it to K.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut pf = Pf::<f64, 2, 256>::from_gaussian(
        Vector2::new(0.0_f64, 0.0),
        SMatrix::<f64, 2, 2>::identity(),
        make_normal(&mut rng),
    )
    .expect("ok");

    let anchor = Vector2::new(5.0_f64, 0.0);
    let model = RangeAnchor::<f64, 2>::new(anchor, 0.001_f64); // very tight noise
    let mut z = SVector::<f64, 1>::zeros();
    z[0] = 5.0;
    pf.update(&model, &z).expect("update ok");

    let ess_before = pf.effective_sample_size();
    assert!(
        ess_before < 256.0,
        "expected ESS collapse, got {ess_before}"
    );

    pf.resample(make_uniform(&mut rng));
    assert_relative_eq!(pf.effective_sample_size(), 256.0, epsilon = 1e-9);
}

#[test]
fn pf_predict_with_zero_noise_preserves_state_estimate() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let initial = Vector2::new(3.0_f64, 4.0);
    let mut pf = Pf::<f64, 2, 256>::from_gaussian(
        initial,
        SMatrix::<f64, 2, 2>::identity() * 0.01,
        make_normal(&mut rng),
    )
    .expect("ok");

    let state_before = pf.state();
    // Zero-noise identity transition: each predict is a no-op for the mean.
    let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.0);
    pf.predict(&transition, 1.0, make_normal(&mut rng))
        .expect("ok");
    let state_after = pf.state();
    assert_relative_eq!(state_before, state_after, epsilon = 1e-12);
}
