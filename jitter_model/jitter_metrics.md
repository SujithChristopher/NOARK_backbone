# Jitter and movement-smoothness metrics

The movement metrics compare the raw camera trajectory against the aligned mocap
trajectory. No temporal smoothing is applied to either trajectory.

Only valid, consecutive frame pairs are considered. A pair is classified as
moving when the mocap displacement between the two frames is at least 3 mm.

## Jerk error

Jerk is the rate of change of acceleration, or the third derivative of
position. Position is measured in metres and time in seconds.

For consecutive position samples, velocity is calculated as:

$$
\mathbf{v}_i =
\frac{\mathbf{p}_{i+1} - \mathbf{p}_i}
     {t_{i+1} - t_i}
$$

Acceleration is calculated from consecutive velocity samples:

$$
\mathbf{a}_i =
\frac{\mathbf{v}_{i+1} - \mathbf{v}_i}
     {t^{v}_{i+1} - t^{v}_i}
$$

Jerk is calculated from consecutive acceleration samples:

$$
\mathbf{j}_i =
\frac{\mathbf{a}_{i+1} - \mathbf{a}_i}
     {t^{a}_{i+1} - t^{a}_i}
$$

The velocity and acceleration timestamps are the midpoints of the timestamps
used to calculate them. These calculations are performed independently for the
camera and mocap trajectories using identical timestamps.

The jerk error vector is:

$$
\mathbf{e}^{j}_i =
\mathbf{j}^{\mathrm{camera}}_i - \mathbf{j}^{\mathrm{mocap}}_i
$$

The reported and plotted jerk metric is the three-dimensional vector RMSE:

$$
\mathrm{JerkErrorRMSE} =
\sqrt{
    \frac{1}{N}
    \sum_{i=1}^{N}
    \left\|\mathbf{e}^{j}_i\right\|_2^2
}
$$

Equivalently, without relying on rendered mathematics:

```text
jerk error RMSE = sqrt(mean(error_x^2 + error_y^2 + error_z^2))
```

The unit is **m/s³**, and lower is better. A value of zero would mean that the
camera trajectory's changes in acceleration exactly match mocap.

This is a **jerk tracking error**, rather than simply the raw jerk of the camera.
The output CSV also includes:

- `camera_jerk_rms_m_s3`: RMS magnitude of camera jerk.
- `mocap_jerk_rms_m_s3`: RMS magnitude of mocap jerk.
- `jerk_ratio_camera_over_mocap`: camera jerk RMS divided by mocap jerk RMS.

Because jerk requires three finite-difference operations, it strongly amplifies
position noise and timestamp errors. Camera desynchronization can therefore
produce a large jerk error even when the position trajectory looks reasonable.

## Path-length ratio

For every valid moving frame pair, the three-dimensional distance travelled is:

$$
d_i = \left\|\mathbf{p}_{i+1} - \mathbf{p}_i\right\|_2
$$

The total camera and mocap path lengths are:

$$
L_{\mathrm{camera}} =
\sum_i
\left\|
\mathbf{p}^{\mathrm{camera}}_{i+1} -
\mathbf{p}^{\mathrm{camera}}_i
\right\|_2
$$

$$
L_{\mathrm{mocap}} =
\sum_i
\left\|
\mathbf{p}^{\mathrm{mocap}}_{i+1} -
\mathbf{p}^{\mathrm{mocap}}_i
\right\|_2
$$

The reported ratio is:

$$
\mathrm{PathLengthRatio} =
\frac{L_{\mathrm{camera}}}{L_{\mathrm{mocap}}}
$$

Plain-text equivalent:

```text
path-length ratio = total camera path length / total mocap path length
```

Interpretation:

- **1.0**: camera and mocap travelled the same total distance.
- **Greater than 1.0**: the camera path is longer, which can indicate noise,
  zig-zagging, overshoot, or timing problems.
- **Less than 1.0**: the camera path is shorter, which can indicate missed
  movement, underestimation, or excessive smoothing.

For example, a path-length ratio of `1.233` means that the camera trajectory is
23.3% longer than the mocap trajectory over the evaluated movement intervals.

Path-length ratio does not measure absolute position accuracy, direction error,
or temporal alignment. It is a relative measure of accumulated travel and is
used here primarily as a trajectory-roughness indicator.

## Depth-binned plots

The same calculations are used for each 50 mm optical-axis depth bin in
`metrics_vs_depth.csv`. Only pairs that remain consecutive in the original
recording are accepted. A missing value or a gap in the static-jitter plot means
that the bin did not contain enough valid pairs; values are not interpolated.

## Implementation

The calculations are implemented in `movement_smoothness_metrics()` in
[`03_jitter_model.py`](03_jitter_model.py).
