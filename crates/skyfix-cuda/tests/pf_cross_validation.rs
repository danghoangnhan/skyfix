//! Cross-validate `CudaPfRanges2D` against the CPU `skyfix_core::Pf` by
//! feeding both filters identical initial particles and identical noise
//! sequences. After several predict / update cycles, the weighted means
//! must agree to f32 precision.
//!
//! This is the "is the kernel doing the right math" test — not a
//! generative-quality test (that's covered by `tests/pf.rs` in skyfix-core).

use nalgebra::{SMatrix, SVector, Vector2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use skyfix_core::{IdentityTransition, Pf, RangeAnchor};
use skyfix_cuda::CudaPfRanges2D;

const K: usize = 256;

fn make_normal<'a>(rng: &'a mut ChaCha8Rng) -> impl FnMut() -> f64 + 'a {
    move || StandardNormal.sample(rng)
}

#[test]
fn cuda_pf_mean_matches_cpu_pf_zero_noise() {
    let initial_mean = Vector2::new(1.0_f64, 2.0);
    let initial_cov = SMatrix::<f64, 2, 2>::identity() * 0.25;

    // CPU PF: seed RNG, draw K initial particles from N(mean, cov).
    let mut rng_cpu = ChaCha8Rng::seed_from_u64(0xDEAD);
    let mut cpu =
        Pf::<f64, 2, K>::from_gaussian(initial_mean, initial_cov, make_normal(&mut rng_cpu))
            .expect("cpu from_gaussian");

    // GPU PF: start with exactly the same particles (downloaded from CPU).
    let cpu_particles = cpu.particles();
    let mut initial_f32 = Vec::with_capacity(2 * K);
    for k in 0..K {
        initial_f32.push(cpu_particles[(0, k)] as f32);

        initial_f32.push(cpu_particles[(1, k)] as f32);
    }
    let mut gpu = CudaPfRanges2D::new(&initial_f32).expect("cuda init");

    // Five identity-transition (no-noise) predict + range-update steps.
    let anchor = Vector2::new(5.0_f64, 5.0);
    let truth = Vector2::new(1.5_f64, 2.5);
    let z = (truth - anchor).norm();
    let variance = 0.04_f64;

    let zero_noise = vec![0.0_f32; 2 * K];
    let zero_chol = [[0.0_f32, 0.0], [0.0, 0.0]];

    for _ in 0..5 {
        // CPU: zero-noise predict + range update
        let transition = IdentityTransition::<f64, 2>::with_uniform_noise(0.0);
        cpu.predict(&transition, 1.0, make_normal(&mut rng_cpu))
            .expect("cpu predict");
        let model = RangeAnchor::<f64, 2>::new(anchor, variance);
        let mut z_vec = SVector::<f64, 1>::zeros();
        z_vec[0] = z;
        cpu.update(&model, &z_vec).expect("cpu update");

        // GPU: same scenario
        gpu.predict(zero_chol, &zero_noise).expect("gpu predict");
        gpu.update(
            [anchor.x as f32, anchor.y as f32],
            z as f32,
            variance as f32,
        )
        .expect("gpu update");
    }

    let cpu_mean = cpu.state();
    let gpu_mean = gpu.mean().expect("gpu mean");

    let dx = (cpu_mean.x as f32 - gpu_mean[0]).abs();
    let dy = (cpu_mean.y as f32 - gpu_mean[1]).abs();
    assert!(
        dx < 1e-3,
        "x: cpu={} gpu={} diff={dx}",
        cpu_mean.x,
        gpu_mean[0]
    );
    assert!(
        dy < 1e-3,
        "y: cpu={} gpu={} diff={dy}",
        cpu_mean.y,
        gpu_mean[1]
    );
}

#[test]
fn cuda_pf_ess_matches_cpu_after_skewed_update() {
    let initial_mean = Vector2::new(0.0_f64, 0.0);
    let initial_cov = SMatrix::<f64, 2, 2>::identity();

    let mut rng_cpu = ChaCha8Rng::seed_from_u64(7);
    let mut cpu =
        Pf::<f64, 2, K>::from_gaussian(initial_mean, initial_cov, make_normal(&mut rng_cpu))
            .expect("cpu from_gaussian");

    let cpu_particles = cpu.particles();
    let mut initial_f32 = Vec::with_capacity(2 * K);
    for k in 0..K {
        initial_f32.push(cpu_particles[(0, k)] as f32);

        initial_f32.push(cpu_particles[(1, k)] as f32);
    }
    let mut gpu = CudaPfRanges2D::new(&initial_f32).expect("cuda init");

    // Single very-tight-noise update — both filters should see the same ESS
    // collapse (one particle ends up much more likely than the rest).
    let anchor = Vector2::new(5.0_f64, 0.0);
    let z = 5.0;
    let variance = 0.001;

    let model = RangeAnchor::<f64, 2>::new(anchor, variance);
    let mut z_vec = SVector::<f64, 1>::zeros();
    z_vec[0] = z;
    cpu.update(&model, &z_vec).expect("cpu update");
    gpu.update(
        [anchor.x as f32, anchor.y as f32],
        z as f32,
        variance as f32,
    )
    .expect("gpu update");

    let cpu_ess = cpu.effective_sample_size() as f32;
    let gpu_ess = gpu.effective_sample_size().expect("gpu ess");

    // ESS depends on relative weight magnitudes — f32↔f64 quadratic terms
    // can drift, so allow 5% relative tolerance.
    let rel = (cpu_ess - gpu_ess).abs() / cpu_ess.max(1.0);
    assert!(
        rel < 0.05,
        "ESS mismatch: cpu={cpu_ess} gpu={gpu_ess} relative={rel}"
    );
}

#[test]
fn cuda_pf_rejects_bad_size() {
    use skyfix_cuda::CudaError;
    assert!(matches!(
        CudaPfRanges2D::new(&[]),
        Err(CudaError::NoParticles)
    ));
    assert!(matches!(
        CudaPfRanges2D::new(&[1.0, 2.0, 3.0]),
        Err(CudaError::SizeMismatch { .. })
    ));

    let particles = vec![0.0_f32; 2 * 8];
    let mut pf = CudaPfRanges2D::new(&particles).expect("ok");
    let wrong_noise = vec![0.0_f32; 5];
    assert!(matches!(
        pf.predict([[1.0, 0.0], [0.0, 1.0]], &wrong_noise),
        Err(CudaError::SizeMismatch {
            expected: 16,
            got: 5
        })
    ));
}
