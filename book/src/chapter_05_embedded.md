# Embedded targets

*(Coming next turn.)*

This chapter will cover the `no_std` path — running skyfix estimators on Cortex-M / RISC-V / ESP32 MCUs. Topics:

- Cross-compilation setup per target (`thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `riscv32imac-unknown-none-elf`, `xtensa-esp32-none-elf` via espup)
- Feature gating (`--no-default-features --features libm`)
- Why the algorithmic core ships its own dense solvers (the nalgebra `Const<N>: ToTypenum` constraint)
- Pairing with an executor — Embassy for async, RTIC for priority-based hard real-time
- Integration sketches: DW3000 UWB → `skyfix-uwb::UwbRange` → `ToaTrilateration` → `Ekf`
- Stack-size budgeting for the static-K particle filter

Until this chapter lands, see the CI matrix in `.github/workflows/ci.yml` for the verified target list, the `skyfix-uwb` crate for the adapter pattern, and the cargo tree guard in the `no-cuda-leak` job for evidence that the embedded dep graph stays CUDA-free.
