# skyfix

Rust UAV localization & triangulation for the edge.

**Status: pre-alpha.** No public API yet. See [CLAUDE.md](./CLAUDE.md) for the design intent and v0.1.0 roadmap.

## What it is (planned)

A Cargo workspace covering UAV localization across:

- **AoA** — bearings-only LSQ, MUSIC, ESPRIT
- **ToA** — closed-form trilateration, Gauss–Newton NLS
- **TDoA** — Chan's closed-form, Foy Taylor-series WLS
- **RSSI** — log-distance path-loss, weighted centroid
- **Hybrid** — TDoA + AoA joint Gauss–Newton
- **Bayesian filters** — EKF, UKF, ESKF, particle filter
- **CRLB / FIM / GDOP analyzer** — for anchor placement planning

Targets both `no_std` microcontrollers (Cortex-M, RISC-V, ESP32) and `std` companion computers (Jetson Orin, x86 + dGPU). The algorithmic core never depends on an executor or allocator by default.

## Layout

```
crates/
├── skyfix-core/       no_std algorithmic core
├── skyfix-sim/        std-only desktop simulator
└── skyfix-fixtures/   shared reference test vectors
```

Driver and binding crates (`skyfix-uwb`, `skyfix-gnss`, `skyfix-imu`, `skyfix-cuda`, `skyfix-py`, `skyfix-c`, `skyfix-wasm`) land in later phases.

## License

Dual-licensed under MIT ([LICENSE-MIT](./LICENSE-MIT)) or Apache 2.0 ([LICENSE-APACHE](./LICENSE-APACHE)), at your option.

Contributions are accepted under the same dual license unless explicitly stated otherwise.
