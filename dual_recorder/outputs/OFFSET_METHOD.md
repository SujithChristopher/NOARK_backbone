# How the inter-camera offset was measured and removed

Reference for the two figures in this folder, `camera_delay_raw.png` and
`camera_delay_skew.png`, both produced by `plot_delay.py` from a
`capture_flicker.py` take.

**One thing up front, because the word "resolve" points two different ways
here:** this take was recorded with the sensors *free-running on purpose*, and
the offset was removed **arithmetically, after the fact**. Nothing was done to
the cameras to bring them into phase. The hardware mechanism that actually
fixes the offset exists in `rcam` and is used by `dual_recorder_rcam.py`; it was
switched off here so the plots would show the uncorrected physics. Both routes
are written up below - post-hoc removal first, since that is what these figures
did.

---

## 1. Where the offset comes from

The two OV9281s self-clock off separate 24 MHz crystals. There is no FSIN wiring
between them, so nothing tells either sensor when the other starts a frame. Each
begins exposing whenever it was told to stream and free-runs from there, which
leaves two independent error terms:

| Term | Cause | Size on this take |
|---|---|---|
| **Offset** | arbitrary start phase, uniform over one frame period | **-14.41 ms** (43% of a 33.34 ms frame) |
| **Skew** | the crystals' rate difference, so the offset walks | **+5.67 ± 1.70 ppm** |
| **Jitter** | per-frame timestamp noise, irreducible | **~85 µs** sd |

The offset is uniform over `[-T/2, +T/2]`, so a cold start can land anywhere from
simultaneous to half a frame apart. This one landed at 43% of a frame - close to
the worst case, and consistent with the -14.9 ms that `dual_recorder_rcam.py`'s
docstring records for cold free-running sensors.

## 2. The timestamps the measurement is built on

Everything below depends on using the right clock, so this is the part to get
right first.

Each frame carries a `sensor_ns` from `VIDIOC_DQBUF`'s `v4l2_buffer.timestamp`
(`ts-monotonic`, `ts-src-eof`), stamped by CAMSS **in its frame-done
interrupt**. It is the same `CLOCK_MONOTONIC` as Python's `time.monotonic_ns()`,
but taken in the kernel:

| Clock | Where taken | Jitter |
|---|---|---|
| `sensor_ns` (used) | CAMSS frame-done IRQ | **~100 µs** |
| `time.monotonic_ns()` | Python, after the frame is unpacked | 1.6-1.9 ms |

A userspace arrival time carries GIL hand-off and scheduler latency, which is
20x the effect being measured. Using it would have buried the whole result. The
capture threads in `capture_flicker.py` do read `sensor_ns` from a Python thread,
but the *value* was stamped in the kernel, so the thread's own scheduling does
not enter it.

`capture_flicker.py` also stores the driver's `sequence` counter per frame. A gap
in it means the sensor produced a frame that never reached us, which a gap in
timestamps alone cannot distinguish from a capture thread being descheduled.
Both cameras reported **0 lost frames** on this take, so no gap-filling was
needed.

## 3. Pairing the frames

Two free-running cameras do not produce frames in lockstep, and they did not even
produce the same *number* of frames here (304 vs 301). Frames are paired by
**nearest timestamp**, never by index:

```python
idx = np.clip(np.searchsorted(ts_b, ts_a), 1, len(ts_b) - 1)
left, right = ts_b[idx - 1], ts_b[idx]
idx = np.where(np.abs(ts_a - left) <= np.abs(ts_a - right), idx - 1, idx)
```

Index pairing would have been silently wrong: the moment either sensor drops a
frame, every later pair slides by a whole frame period and the "offset" picks up
a 33 ms step that is pure bookkeeping.

**Validity mask.** Frames at either end of the take can have no partner at all -
one camera starts or stops while the other is still streaming. Their nearest
neighbour is then frame periods away, so anything more than half a frame period
off is dropped rather than let through:

```python
period_ns = np.median(np.diff(ts_a))
valid = np.abs(ts_b[idx] - ts_a) <= period_ns / 2
```

This mattered. Before the mask existed the four unpaired edge frames produced
delays of up to 133 ms - four times a frame period - which dragged the skew fit
to **-659 ppm**, two orders of magnitude off. With the mask, 4 of 300 pairs are
dropped and the fit lands at +5.67 ppm.

## 4. Computing the offset

With valid pairs in hand the delay is a subtraction, and the offset is its mean:

```python
delay_us = (ts1[j] - ts0) / 1e3          # per-pair delay, microseconds
offset   = delay_us.mean()               # -14405.4 us
```

**Why a plain arithmetic mean, and when it would be wrong.** Delay lives on a
circle of circumference `T`: a pair 16.6 ms apart is equally describable as
-16.7 ms with the pairing off by one. `rcam.sync.phase_from_timestamps` therefore
wraps every offset into `(-T/2, +T/2]` and takes a *circular* mean, averaging the
values as unit vectors - which is the generally correct thing to do.

Here the arithmetic mean is safe, because nearest-neighbour pairing already
bounds every delay to `±T/2`, and this take stays clear of the -16.67 ms wrap
point: the mean by 2.27 ms (26.7 σ) and even the single closest frame by 1.98 ms
(23.4 σ), against 85 µs of jitter. **If a future
take lands near half a frame, the delays will straddle the wrap point and the
arithmetic mean will collapse toward zero.** Switch to the circular mean from
`rcam.sync` if that happens; the symptom is a mean near 0 with a bimodal box plot
piled at both `±T/2`.

## 5. Removing offset and skew

Two stages, each subtracting one term, fitted by ordinary least squares:

```python
slope, intercept = np.polyfit(t, delay_us, 1)   # t in seconds
offset_removed = delay_us - offset                        # constant term gone
skew_removed   = delay_us - (intercept + slope * t)       # linear term gone too
```

`slope` comes out in µs per second, which **is** ppm exactly (1 µs/s = 1e-6), so
the fitted number needs no conversion.

The residual after both is the jitter floor of §1 - the part no correction can
touch, because it is the timestamps' own noise.

### Reporting the fit honestly

The slope is quoted with its standard error, and this is not decoration:

```python
slope_se = fixed.std(ddof=2) / np.sqrt(np.sum((t - t.mean()) ** 2))
```

Over 10 s at ~5.7 ppm the skew ramp spans only ~57 µs, sitting *underneath* ~85 µs
of jitter. The lever arm is too short to separate them: the fit reads
**+5.67 ± 1.70 ppm**, and an earlier take of the same setup gave
**+2.13 ± 2.02 ppm**, i.e. consistent with zero. The plot prints the ± and says
"not resolved" whenever `|slope| < 2·se`, so the number is never read as a
precise measurement. **Use `--duration 60` or longer if the skew term itself is
what you care about.**

This is also why the two figures are separate. The offset is ~14 ms against a
~0.08 ms residual; on one shared axis both corrected series flatten onto the zero
line and the comparison shows nothing.

## 6. Fixing the offset in hardware instead

`plot_delay.py` removes the offset from *numbers*. To remove it from the
*recording*, `rcam.FrameSync` retimes a sensor, and `dual_recorder_rcam.py` runs
it by default - `--no-frame-sync` is what disables it, and is effectively what
this take did.

The OV9281 has no phase control, so the trick is indirect. Frame period is

```
period = (height + vertical_blanking) x line_time
```

so `Camera.nudge_phase()` temporarily inflates `vertical_blanking` over the
sensor's subdev, sleeps about half a frame so the sensor latches the stretched
period into at least one frame, then restores the original value. That one frame
runs long; every frame after it lands later by the same amount, at the original
rate. Nothing is dropped and the other camera is never touched.

Only *delays* are possible - a sensor can be slowed with blanking but not
hurried - so a camera running early is corrected the long way round:

```python
delay_us = (-phase_us) % self.period_us
```

The correction is open-loop imprecise (the stretch is held by a `sleep`, so it
may cover one frame or two), so `FrameSync.align()` closes the loop: measure 45
frames, nudge, re-measure, repeat until within `--phase-tol` (200 µs default) or
6 iterations. Two or three rounds normally reach the ~100 µs measurement floor -
the same floor as the jitter in §1, and the reason the tolerance is not tighter.

Drift is handled separately during the take. `resync_if_needed()` re-reads the
phase from timestamps the capture threads have already written - taking no frames
of its own - and nudges only once drift exceeds `--resync-threshold` (1 ms
default). The threshold is a real trade: every nudge puts one long frame interval
into the recording, and setting it near the ~150 µs measurement jitter makes the
loop chase noise and wander further than leaving it alone.

## 7. Results from this take

```
296 frame pairs over 9.84 s (CAM2 -> CAM3)
  frame period       33.341 ms
  mean delay      -14405.4 us (43.2% of a frame)
  fitted skew         +5.67 +/- 1.70 ppm
  raw (no correction)    sd   84.79 us   p2p   499.00 us   |max|  14687.00 us
  offset removed         sd   84.79 us   p2p   499.00 us   |max|    281.65 us
  offset + skew removed  sd   83.24 us   p2p   455.85 us   |max|    257.93 us
```

`sd` barely moves across the three rows, and that is the point: the correction
removes *bias*, not *noise*. The column that moves is `|max|`, the worst-case
pairing error, which falls from 14.7 ms to 282 µs - because with the offset left
in, the worst case simply *is* the offset.

## 8. Caveats

- **Skew is not resolved at 10 s.** §5. Quote the ± or record longer.
- **The arithmetic mean assumes the delay is clear of `±T/2`.** §4. True here by
  26 σ; check it before trusting the number on a new take.
- **4 of 300 pairs were dropped** as unpairable edge frames. §3.
- **These numbers describe one cold start.** The offset is uniform over a frame
  period, so a re-run gives a different value - it is not a property of the rig.
  The two takes recorded while building this agreed on offset (-14.41 vs
  -14.55 ms) but not on skew, which is §5's point.
- **The take is 10 s of one scene** at 5000 µs exposure, i.e. the flicker
  capture. Nothing about the delay depends on exposure, but the frame period does
  depend on `--fps`.

## Reproducing

```bash
uv run python dual_recorder/outputs/capture_flicker.py     # 10 s, free-running
uv run python dual_recorder/outputs/plot_delay.py          # both figures
uv run python dual_recorder/outputs/capture_flicker.py --duration 60
uv run python dual_recorder/outputs/plot_delay.py --duration 60   # skew resolves
```
