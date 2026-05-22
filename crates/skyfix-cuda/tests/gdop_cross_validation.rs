//! Cross-validate GPU GDOP sweep against the CPU `CrlbBuilder`.
//!
//! Both implementations should produce the same GDOP at every grid cell
//! (within f32 rounding). This test confirms the kernel math matches the
//! analytic formula used by `skyfix-core`.

use approx::assert_relative_eq;
use nalgebra::Vector2;
use skyfix_core::CrlbBuilder;
use skyfix_cuda::{Anchor2D, CudaGdopSweep2D};

fn cpu_gdop(anchors: &[Anchor2D], target: Vector2<f32>) -> f32 {
    let mut b = CrlbBuilder::<f32, 2>::new();
    for a in anchors {
        b.add_toa(target, Vector2::new(a.x, a.y), a.variance);
    }
    match b.finish().gdop() {
        Ok(g) => g,
        Err(_) => f32::INFINITY,
    }
}

#[test]
fn gpu_gdop_matches_cpu_on_symmetric_grid() {
    let anchors = [
        Anchor2D::new(0.0, 0.0, 0.1),
        Anchor2D::new(10.0, 0.0, 0.1),
        Anchor2D::new(0.0, 10.0, 0.1),
        Anchor2D::new(10.0, 10.0, 0.1),
    ];
    let sweep = CudaGdopSweep2D::new(&anchors).expect("cuda init ok");

    let xs: Vec<f32> = (0..11).map(|i| i as f32).collect();
    let ys: Vec<f32> = (0..11).map(|i| i as f32).collect();
    let gpu = sweep.compute_grid(&xs, &ys).expect("compute ok");

    for (iy, &y) in ys.iter().enumerate() {
        for (ix, &x) in xs.iter().enumerate() {
            let cpu = cpu_gdop(&anchors, Vector2::new(x, y));
            let g = gpu[iy * xs.len() + ix];
            if cpu.is_infinite() {
                assert!(g.is_infinite(), "cell ({x}, {y}) cpu=inf gpu={g}");
            } else {
                assert_relative_eq!(g, cpu, max_relative = 1e-4);
            }
        }
    }
}

#[test]
fn gpu_gdop_handles_target_on_anchor_gracefully() {
    let anchors = [
        Anchor2D::new(0.0, 0.0, 0.1),
        Anchor2D::new(5.0, 0.0, 0.1),
        Anchor2D::new(0.0, 5.0, 0.1),
    ];
    let sweep = CudaGdopSweep2D::new(&anchors).expect("cuda init ok");
    // Single-cell grid at exactly anchor 0.
    let gpu = sweep.compute_grid(&[0.0], &[0.0]).expect("compute ok");
    // The on-anchor unit vector is undefined; kernel skips that anchor.
    // With the remaining 2 anchors the FIM has rank 2 and GDOP is finite.
    assert!(gpu[0].is_finite(), "expected finite GDOP, got {}", gpu[0]);
}

#[test]
fn gpu_gdop_target_on_anchor_line_is_singular() {
    // Target ON the anchor line: all u_i point in the same direction, so the
    // FIM is rank-1 and singular. (Anchors merely colinear *off-target* are
    // fine — that's how multilateration works.)
    let anchors = [
        Anchor2D::new(1.0, 0.0, 0.1),
        Anchor2D::new(2.0, 0.0, 0.1),
        Anchor2D::new(3.0, 0.0, 0.1),
    ];
    let sweep = CudaGdopSweep2D::new(&anchors).expect("cuda init ok");
    // Target at (0, 0) is on the x-axis along with all anchors.
    let gpu = sweep.compute_grid(&[0.0], &[0.0]).expect("compute ok");
    assert!(gpu[0].is_infinite(), "expected inf, got {}", gpu[0]);
}

#[test]
fn gpu_gdop_rejects_empty_inputs() {
    use skyfix_cuda::CudaError;
    let anchors = [Anchor2D::new(0.0, 0.0, 0.1)];
    let sweep = CudaGdopSweep2D::new(&anchors).expect("cuda init ok");
    assert!(matches!(
        sweep.compute_grid(&[], &[1.0]),
        Err(CudaError::EmptyGrid)
    ));
    assert!(matches!(
        CudaGdopSweep2D::new(&[]),
        Err(CudaError::NoAnchors)
    ));
}
