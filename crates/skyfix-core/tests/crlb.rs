//! Integration tests for the CRLB / FIM / GDOP analyzer.

use approx::assert_relative_eq;
use nalgebra::{Vector2, Vector3};
use skyfix_core::CrlbBuilder;

#[test]
fn crlb_2d_symmetric_anchors_at_unit_corners_gives_gdop_1() {
    // Four ToA anchors at (±1, ±1) around origin target, variance = 1.
    // FIM = Σ uᵢ uᵢᵀ = 2·I, so CRLB = (1/2)·I and GDOP = √(1/2 + 1/2) = 1.
    let target = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(1.0_f64, 1.0),
        Vector2::new(-1.0, 1.0),
        Vector2::new(1.0, -1.0),
        Vector2::new(-1.0, -1.0),
    ];
    let mut builder = CrlbBuilder::<f64, 2>::new();
    for &a in &anchors {
        builder.add_toa(target, a, 1.0);
    }
    let analysis = builder.finish();
    let cov = analysis.covariance().expect("FIM is invertible");
    assert_relative_eq!(cov[(0, 0)], 0.5, epsilon = 1e-12);
    assert_relative_eq!(cov[(1, 1)], 0.5, epsilon = 1e-12);
    assert_relative_eq!(cov[(0, 1)], 0.0, epsilon = 1e-12);
    assert_relative_eq!(analysis.gdop().unwrap(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(analysis.hdop().unwrap(), 1.0, epsilon = 1e-12);
}

#[test]
fn crlb_2d_three_colinear_anchors_singular_fim() {
    let target = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(1.0_f64, 0.0),
        Vector2::new(2.0, 0.0),
        Vector2::new(3.0, 0.0),
    ];
    let mut builder = CrlbBuilder::<f64, 2>::new();
    for &a in &anchors {
        builder.add_toa(target, a, 1.0);
    }
    let analysis = builder.finish();
    assert!(
        analysis.covariance().is_err(),
        "colinear anchors → singular FIM"
    );
}

#[test]
fn crlb_2d_smaller_variance_yields_smaller_gdop() {
    let target = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(1.0_f64, 1.0),
        Vector2::new(-1.0, 1.0),
        Vector2::new(1.0, -1.0),
        Vector2::new(-1.0, -1.0),
    ];

    let gdop_at = |variance: f64| {
        let mut b = CrlbBuilder::<f64, 2>::new();
        for &a in &anchors {
            b.add_toa(target, a, variance);
        }
        b.finish().gdop().unwrap()
    };

    let g_loud = gdop_at(1.0); // σ² = 1 m²
    let g_quiet = gdop_at(0.01); // σ² = 0.01 m²
                                 // FIM scales as 1/σ², so CRLB scales as σ² and GDOP scales as σ.
                                 // Tightening σ by 10× should tighten GDOP by 10× too.
    assert_relative_eq!(g_quiet / g_loud, 0.1, epsilon = 1e-12);
}

#[test]
fn crlb_2d_aoa_only_two_anchors_finite_gdop() {
    let target = Vector2::new(0.0_f64, 0.0);
    let anchors = [Vector2::new(5.0_f64, 0.0), Vector2::new(0.0, 5.0)];
    let mut builder = CrlbBuilder::<f64, 2>::new();
    for &a in &anchors {
        builder.add_aoa(target, a, 0.01_f64); // (0.1 rad)² angular noise
    }
    let analysis = builder.finish();
    let gdop = analysis.gdop().expect("non-singular");
    assert!(gdop.is_finite() && gdop > 0.0, "AoA-only GDOP: {gdop}");
}

#[test]
fn crlb_2d_hybrid_toa_plus_aoa_outperforms_either_alone() {
    let target = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(5.0_f64, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(-5.0, 0.0),
    ];

    let toa_only = {
        let mut b = CrlbBuilder::<f64, 2>::new();
        for &a in &anchors {
            b.add_toa(target, a, 0.01);
        }
        b.finish().gdop().unwrap()
    };
    let aoa_only = {
        let mut b = CrlbBuilder::<f64, 2>::new();
        for &a in &anchors {
            b.add_aoa(target, a, 0.001);
        }
        b.finish().gdop().unwrap()
    };
    let hybrid = {
        let mut b = CrlbBuilder::<f64, 2>::new();
        for &a in &anchors {
            b.add_toa(target, a, 0.01);
            b.add_aoa(target, a, 0.001);
        }
        b.finish().gdop().unwrap()
    };
    assert!(
        hybrid < toa_only && hybrid < aoa_only,
        "hybrid GDOP {hybrid} should beat ToA-only {toa_only} and AoA-only {aoa_only}"
    );
}

#[test]
fn crlb_3d_tetrahedral_anchors_exposes_hdop_vdop() {
    let target = Vector3::new(0.0_f64, 0.0, 0.0);
    let anchors = [
        Vector3::new(1.0_f64, 1.0, 1.0),
        Vector3::new(-1.0, -1.0, 1.0),
        Vector3::new(-1.0, 1.0, -1.0),
        Vector3::new(1.0, -1.0, -1.0),
    ];
    let mut b = CrlbBuilder::<f64, 3>::new();
    for &a in &anchors {
        b.add_toa(target, a, 1.0);
    }
    let analysis = b.finish();
    let gdop = analysis.gdop().expect("ok");
    let hdop = analysis.hdop().expect("ok");
    let vdop = analysis.vdop().expect("ok");
    // Sanity: HDOP and VDOP are partials of GDOP, so HDOP² + VDOP² = GDOP².
    assert_relative_eq!(hdop * hdop + vdop * vdop, gdop * gdop, epsilon = 1e-12);
}

#[test]
fn crlb_2d_tdoa_chan_anchors_recovers_finite_gdop() {
    let target = Vector2::new(1.0_f64, 1.5);
    let a_ref = Vector2::new(0.0_f64, 0.0);
    let anchors = [
        Vector2::new(5.0_f64, 0.0),
        Vector2::new(0.0, 5.0),
        Vector2::new(5.0, 5.0),
    ];
    let mut b = CrlbBuilder::<f64, 2>::new();
    for &a in &anchors {
        b.add_tdoa(target, a, a_ref, 0.01);
    }
    let analysis = b.finish();
    let gdop = analysis.gdop().expect("non-singular");
    assert!(gdop.is_finite() && gdop > 0.0, "TDoA GDOP: {gdop}");
}
