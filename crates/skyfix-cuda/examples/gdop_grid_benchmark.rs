//! GDOP grid sweep: CPU vs GPU wall-clock comparison.
//!
//! Both paths compute the same scalar (GDOP at each grid cell, given a fixed
//! anchor set) and we check they agree to f32 rounding. The point is to see
//! the cross-over: at small grid sizes the GPU loses to kernel-launch
//! overhead; at large grids it wins by the number of CUDA cores.
//!
//! ```sh
//! cargo run -p skyfix-cuda --release --example gdop_grid_benchmark
//! ```

use nalgebra::Vector2;
use skyfix_core::CrlbBuilder;
use skyfix_cuda::{Anchor2D, CudaGdopSweep2D};
use std::time::Instant;

const ROOM_SIZE: f32 = 10.0;
const RANGE_VARIANCE: f32 = 0.01;
const GRID_SIZES: &[usize] = &[10, 25, 50, 100, 200];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let anchors_gpu = vec![
        Anchor2D::new(0.0, 0.0, RANGE_VARIANCE),
        Anchor2D::new(ROOM_SIZE, 0.0, RANGE_VARIANCE),
        Anchor2D::new(ROOM_SIZE, ROOM_SIZE, RANGE_VARIANCE),
        Anchor2D::new(0.0, ROOM_SIZE, RANGE_VARIANCE),
    ];
    let anchors_cpu: Vec<Vector2<f32>> =
        anchors_gpu.iter().map(|a| Vector2::new(a.x, a.y)).collect();

    // Initialize the GPU sweep once — outside the timing loop, since anchor
    // upload is a one-time cost amortized over many compute_grid() calls.
    let sweep = CudaGdopSweep2D::new(&anchors_gpu)?;

    println!("skyfix GDOP grid sweep — CPU vs GPU (RTX-class)");
    let sigma = RANGE_VARIANCE.sqrt();
    println!(" anchors  : 4 at corners of {ROOM_SIZE}×{ROOM_SIZE} m room (σ = {sigma:.2} m)");
    println!();
    println!(
        " {:>7} {:>8} {:>12} {:>12} {:>10}",
        "grid", "cells", "CPU", "GPU", "speedup"
    );
    let bar = |n: usize| "─".repeat(n);
    println!(
        " {:>7} {:>8} {:>12} {:>12} {:>10}",
        bar(7),
        bar(8),
        bar(12),
        bar(12),
        bar(10)
    );

    for &n in GRID_SIZES {
        let step = ROOM_SIZE / (n.saturating_sub(1).max(1)) as f32;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 * step).collect();
        let ys: Vec<f32> = (0..n).map(|i| i as f32 * step).collect();
        let n_cells = n * n;

        // ── CPU path ─────────────────────────────────────────────────────
        let cpu_start = Instant::now();
        let mut cpu_grid = vec![0.0_f32; n_cells];
        for (iy, &y) in ys.iter().enumerate() {
            for (ix, &x) in xs.iter().enumerate() {
                let target = Vector2::new(x, y);
                let mut builder = CrlbBuilder::<f32, 2>::new();
                for a in &anchors_cpu {
                    builder.add_toa(target, *a, RANGE_VARIANCE);
                }
                cpu_grid[iy * n + ix] = builder.finish().gdop().unwrap_or(f32::INFINITY);
            }
        }
        let cpu_time = cpu_start.elapsed();

        // ── GPU path ─────────────────────────────────────────────────────
        let gpu_start = Instant::now();
        let gpu_grid = sweep.compute_grid(&xs, &ys)?;
        let gpu_time = gpu_start.elapsed();

        // ── Cross-validate ───────────────────────────────────────────────
        let mut max_rel = 0.0_f32;
        for i in 0..n_cells {
            if cpu_grid[i].is_finite() && gpu_grid[i].is_finite() {
                let rel = (cpu_grid[i] - gpu_grid[i]).abs() / cpu_grid[i].max(1e-6);
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        assert!(
            max_rel < 1e-3,
            "CPU↔GPU mismatch at grid {n}: max relative diff = {max_rel}"
        );

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        let dim_str = format!("{n}×{n}");
        println!(" {dim_str:>7} {n_cells:>8} {cpu_time:>12.2?} {gpu_time:>12.2?} {speedup:>9.1}×");
    }

    println!();
    println!(" note: GPU time excludes the one-shot anchor upload (constructor).");
    println!(" all CPU↔GPU GDOP values agreed within 1e-3 relative tolerance.");
    Ok(())
}
