// Batched 2D GDOP sweep over a grid of candidate target positions.
//
// For each grid cell (x, y), compute the Fisher Information Matrix from a
// shared set of ToA anchors, then derive GDOP = sqrt(trace(FIM^-1)). The 2x2
// FIM has a closed-form inverse, so each cell is a fixed-cost computation
// proportional to the anchor count.
//
// Layout:
//   anchors[3*i + 0..2] = (ax, ay)
//   anchors[3*i + 2]    = variance
//   output[iy * n_xs + ix] = GDOP

extern "C" __global__ void gdop_2d(
    const float* __restrict__ anchors,
    int n_anchors,
    const float* __restrict__ xs,
    int n_xs,
    const float* __restrict__ ys,
    int n_ys,
    float* __restrict__ output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_cells = n_xs * n_ys;
    if (idx >= n_cells) return;

    int iy = idx / n_xs;
    int ix = idx % n_xs;
    float x = xs[ix];
    float y = ys[iy];

    float a11 = 0.0f, a12 = 0.0f, a22 = 0.0f;

    for (int i = 0; i < n_anchors; ++i) {
        float ax = anchors[3 * i + 0];
        float ay = anchors[3 * i + 1];
        float var = anchors[3 * i + 2];
        float dx = x - ax;
        float dy = y - ay;
        float r = sqrtf(dx * dx + dy * dy);
        if (r == 0.0f || var == 0.0f) continue;
        float ux = dx / r;
        float uy = dy / r;
        float inv_var = 1.0f / var;
        a11 += inv_var * ux * ux;
        a12 += inv_var * ux * uy;
        a22 += inv_var * uy * uy;
    }

    // FIM = [[a11, a12], [a12, a22]]; inverse = (1/det) * [[a22, -a12], [-a12, a11]]
    // trace(inverse) = (a11 + a22) / det; GDOP = sqrt(trace).
    float det = a11 * a22 - a12 * a12;
    if (det <= 0.0f) {
        output[idx] = INFINITY;
        return;
    }
    output[idx] = sqrtf((a11 + a22) / det);
}
