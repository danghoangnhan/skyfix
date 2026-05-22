# CRLB & anchor placement

*(Coming next turn.)*

This chapter will cover the Cramér-Rao Lower Bound — the **achievable lower bound** on the covariance of any unbiased estimator, given a fixed anchor geometry and noise model. CRLB is the right tool to answer:

- Where should I place my anchors to minimize position uncertainty in my coverage region?
- How does GDOP vary across the room?
- Will my proposed anchor layout produce a singular FIM at any target position?
- How does adding a fifth anchor compare to halving the noise variance?

Until this chapter lands, the API is documented in `crates/skyfix-core/src/crlb.rs` and exercised by `crates/skyfix-core/tests/crlb.rs`. The GPU version (`CudaGdopSweep2D`) lives in `crates/skyfix-cuda/src/lib.rs` with a benchmark example showing the CPU↔GPU crossover.
