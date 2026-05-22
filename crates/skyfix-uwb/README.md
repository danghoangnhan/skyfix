# skyfix-uwb

Hardware-agnostic UWB ranging adapter for skyfix. Sits between a UWB ranging stack (typically [dw3000-ng] driving a Decawave DW3000-family chip) and [`skyfix-core`]'s position estimators.

## What's here (Phase 7a)

- `UwbRange<N>` — one anchor-to-target range observation: anchor EUI-64 + position + range in meters + optional timestamp.
- `UwbRange::to_toa()` — convert to `skyfix_core::ToaMeasurement<f64, N>` for trilateration / NLS / Bayesian filters.
- `pair_to_tdoa(reference, other)` — convert two ranges to a `TdoaMeasurement<f64, N>` for Chan / Foy.
- `ranges_to_tdoa_batch(&ranges, &mut output)` — `no_std`-friendly batch conversion (caller provides output buffer).

## What's *not* here yet (Phase 7b)

Actual driving of a DW3000 chip — SPI setup, double-sided two-way ranging exchange, antenna-delay calibration, interrupt handling. That lands once the project has an embedded target wired up; it'll live behind a `dw3000` feature flag so the adapter layer above stays usable from desktop / `no_std` contexts without pulling in `dw3000-ng`'s embedded-hal dep chain.

## Typical embedded usage (sketch)

```rust,no_run
# use nalgebra::Vector3;
# use skyfix_uwb::UwbRange;
# use skyfix_core::{Estimator, ToaTrilateration};
// Whatever your DW3000 stack hands you per ranging exchange:
let raw_ranges: [(u64, Vector3<f64>, f64); 4] = /* SPI ranging exchange */
#    [
#        (0x1, Vector3::new(0.0, 0.0, 0.0), 3.0),
#        (0x2, Vector3::new(1.0, 0.0, 0.0), 3.0),
#        (0x3, Vector3::new(0.0, 1.0, 0.0), 3.0),
#        (0x4, Vector3::new(0.0, 0.0, 1.0), 3.0),
#    ];

let ranges: [UwbRange<3>; 4] = core::array::from_fn(|i| {
    let (addr, pos, range) = raw_ranges[i];
    UwbRange::new(addr, pos, range)
});

// Convert to ToaMeasurement, run trilateration, get a 3D position fix.
let toa: [_; 4] = core::array::from_fn(|i| ranges[i].to_toa());
let position = ToaTrilateration::<f64, 3>::new().estimate(&toa).unwrap();
```

For TDoA mode (only one reference anchor responds to ranging requests, others sniff and report differential timing), use `ranges_to_tdoa_batch` to convert the slice and feed `ChanLinear3D` for the closed-form fix.

[dw3000-ng]: https://crates.io/crates/dw3000-ng
[`skyfix-core`]: ../skyfix-core
