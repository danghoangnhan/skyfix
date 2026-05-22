// Particle-filter kernels for a 2D state, range-anchor measurement model.
//
// Two kernels share this translation unit because they pair on every PF step
// and bundling them keeps the cudarc module load to one call.
//
// Particle layout: `particles[2*i + 0]` = x of particle i, `particles[2*i + 1]` = y.
// Log-weights live in a parallel `log_weights[i]` array.

// ──────────────────────────────────────────────────────────────────────────
// predict: x_i ← x_i + L · z_i, where z_i is a 2-vector of standard-normal
// samples supplied by the host. L is the lower-triangular Cholesky factor of
// the process-noise covariance Q (passed as 3 scalars since L is 2×2 lower-
// triangular: l00, l10, l11). The deterministic part of the transition is
// identity (state unchanged), matching the IdentityTransition built-in.
// ──────────────────────────────────────────────────────────────────────────
extern "C" __global__ void pf_predict_2d(
    float* __restrict__ particles,
    const float* __restrict__ noise,
    int k,
    float l00,
    float l10,
    float l11
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    float z0 = noise[2 * i + 0];
    float z1 = noise[2 * i + 1];

    // L · z, where L = [[l00, 0], [l10, l11]]
    float dx = l00 * z0;
    float dy = l10 * z0 + l11 * z1;

    particles[2 * i + 0] += dx;
    particles[2 * i + 1] += dy;
}

// ──────────────────────────────────────────────────────────────────────────
// update_range: log_w_i ← log_w_i − 0.5 · innovation² / variance, where
// innovation = z − ‖x_i − anchor‖. Drops the (constant) − ½ log(2πR) term —
// only relative log-likelihoods matter for downstream normalization, ESS, and
// resampling.
// ──────────────────────────────────────────────────────────────────────────
extern "C" __global__ void pf_update_range_2d(
    const float* __restrict__ particles,
    float* __restrict__ log_weights,
    int k,
    float ax,
    float ay,
    float z,
    float variance
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    float dx = particles[2 * i + 0] - ax;
    float dy = particles[2 * i + 1] - ay;
    float predicted = sqrtf(dx * dx + dy * dy);
    float innovation = z - predicted;

    log_weights[i] -= 0.5f * innovation * innovation / variance;
}
