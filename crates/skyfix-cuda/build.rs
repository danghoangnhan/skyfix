//! Build-time compilation of CUDA kernels via `nvcc`.
//!
//! Kernels live under `kernels/` as `.cu` files. For each kernel, `nvcc -ptx`
//! produces a PTX assembly file in `OUT_DIR`; the path is exposed as an
//! environment variable so `src/lib.rs` can embed the PTX via `include_str!`.
//!
//! The PTX is *virtual* architecture (`compute_70`), so the device JIT
//! compiles to the actual SASS at module-load time. This keeps a single
//! binary portable across GPUs from Volta (sm_70) through Blackwell (sm_120).

use std::env;
use std::path::PathBuf;
use std::process::Command;

const KERNELS: &[(&str, &str)] = &[
    ("gdop_2d", "kernels/gdop_2d.cu"),
    ("pf_kernels_2d", "kernels/pf_kernels_2d.cu"),
];

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    for (name, src) in KERNELS {
        println!("cargo:rerun-if-changed={src}");
        let ptx_path = out_dir.join(format!("{name}.ptx"));

        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg(src)
            .arg("-o")
            .arg(&ptx_path)
            .arg("-O3")
            .arg("--gpu-architecture=compute_70")
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "failed to invoke nvcc: {e}\n\
                     hint: install CUDA Toolkit (apt: nvidia-cuda-toolkit) and ensure nvcc is on PATH"
                )
            });

        if !status.success() {
            panic!("nvcc failed to compile {src} → {ptx_path:?}");
        }

        let env_name = format!("PTX_{}", name.to_uppercase());
        println!("cargo:rustc-env={env_name}={}", ptx_path.display());
    }
}
