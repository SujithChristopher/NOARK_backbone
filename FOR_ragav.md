# FOR RAGAV — What We Actually Did Today (And Why It Matters)

---

## The Big Picture

Today we did three things:
1. Built a force validation analysis notebook from scratch
2. Debugged why the data looked wrong at every step
3. Sped up the GUI logging by separating display from computation

Each one taught a different lesson. Let's go through all of them.

---

## PART 1 — The Analysis Notebook

### Step 1: What approach did we take, and why?

The goal was simple: compare what force the robot *commanded* vs what the load cell *measured*. 

We had two CSV files:
- `gui.csv` — what the robot was *told* to do (commanded Fx, Fz, magnitude, direction)
- `loadcell.csv` — what the load cell *actually felt* (measured Fx, Fy)

The approach was a pipeline: **clean → filter → resample → merge → analyse → plot**.

Why this order? Because each step depends on the previous one being correct. You can't merge dirty data. You can't analyse unfiltered noise. Think of it like cooking — you wash the vegetables before you chop them, and you chop them before you cook them. Doing it out of order creates problems downstream.

### Step 2: What other approaches did we consider but abandon?

**Using sweep_targets.csv for comparison** — rejected. The `gui.csv` already contains every commanded value (Fx, Fz, magnitude, direction) at every timestamp. `sweep_targets.csv` only records one row per step. Using gui.csv gives us the full time-series, not just step boundaries. More data = better analysis.

**Using nearest-neighbour resampling** (`reindex` with `method="nearest"`) — we started with this. The problem: if the nearest sample is 8ms away and the tolerance is 10ms, you get the right value. But if it's 11ms away, you get NaN. It's fragile. `resample().mean()` and `resample().first()` are more robust because they *aggregate* all samples in a bin rather than picking one.

**Mean resampling for gui** — tried this, caused the worst bug of the day (see Step 6). Rejected in favour of `first()` + `ffill()`.

### Step 3: How do the parts connect?

```
Cell 1: Load raw data (350Hz loadcell, 136Hz gui)
    ↓
Cell 2: Median filter loadcell (remove electrical spikes)
    ↓
Cell 3: Resample both to 100Hz (common time grid)
    ↓
Cell 4: Merge on timestamp + compute measured magnitude/direction + cap steps to HOLD_TIME
    ↓
Cell 5: Plot Fx and Fz (commanded vs measured)
    ↓
Cell 6: Plot magnitude and direction (commanded vs measured)
    ↓
Cell 7: Print error table per step (stable window only)
```

Each cell produces something the next cell needs. If you skip Cell 2, the noise corrupts the error table. If you skip Cell 3, the merge produces misaligned timestamps. The order is load-bearing.

### Step 4: What tools and methods did we use?

**`pd.merge_asof`** — joins two DataFrames on a timestamp column using nearest-match logic. Perfect when two streams are at different rates. Alternative would be `pd.merge` on exact timestamp, but that would drop almost every row since no two streams sample at exactly the same millisecond.

**Rolling median filter** — slides a window of 20 samples, takes the middle value. Used instead of a rolling mean because one spike (e.g., 50N when the true value is 5N) cannot affect the median — it just moves to the end of the sorted window and gets ignored.

**`resample().first().ffill()`** for gui — `first()` picks the actual commanded value (no averaging). `ffill()` fills empty bins by carrying the last known value forward. This is correct because commanded values are step-wise — the command doesn't change between logging ticks.

**`resample().mean()`** for loadcell — averaging 3-4 sensor readings within a 10ms bin reduces noise. Perfectly valid for a continuous physical signal.

**Step detection via `diff().abs() > threshold`** — finds where magnitude or direction jumps by more than a threshold value. This tells us when the sweep moved to the next step.

### Step 5: What tradeoffs did we make?

| Decision | What we gained | What we sacrificed |
|----------|---------------|-------------------|
| 100Hz resample | Clean common grid for both streams | Lost ~70% of loadcell samples (350→100Hz) |
| Median filter window=20 | Strong spike removal | 57ms of phase lag in the signal |
| `ffill()` for gui | No NaN gaps in commanded values | If the GUI crashes mid-step, the last value gets repeated indefinitely |
| Stable window = last 50% of step | Only settled data in error calculation | Fewer samples per step (~200 instead of ~400) |
| Cap steps to HOLD_TIME | Removes post-sweep noise tail | If firmware runs slightly long, you might clip the last real sample |

### Step 6: Mistakes, dead ends, and wrong turns

**Bug 1: Red lines everywhere in the angle-separation plot.**
We tried to draw one vertical line per angle change. Used `direction.round(1) != direction.shift(1).round(1)` — but after resampling with `.mean()`, the direction column had tiny floating-point differences between consecutive rows even within the same step. So *every row* triggered a "change" and the plot turned solid red. Fix: only trigger when `abs(diff) > 1 degree`.

**Bug 2: 12.5N, 7.5N, 19.5N appearing as fake commanded magnitudes.**
When two adjacent rows in a 10ms bin had different magnitudes (e.g., step changing from 10N to 15N), `resample().mean()` averaged them to 12.5N. A magnitude that was *never commanded* appeared in the data and corrupted step detection. Fix: change gui resampling to `.first()` so you always get a real commanded value, never an average of two different commands.

**Bug 3: -76.1° 5N step missing from the error table.**
Even after fixing Bug 2, the 5N step at -76.1° angle kept disappearing. Root cause: the step detection was still using `cumsum()` on data that included transition bins (intermediate angle values created during resampling). These fake steps broke the step numbering. Fix: filter to only rows matching known commanded magnitudes (`MAG_LIST ± 1N`) *before* running `cumsum()` for step detection.

**Bug 4: Last step distorting the error table.**
When recording stops, the motors zero out but logging continues. This means the last step's data has a noisy tail. We first tried dropping the last step entirely — but then 24N at the final angle disappeared from the results (it was the last sweep step). Fix: cap every step to exactly `HOLD_TIME` seconds from its start. The post-sweep noise gets cut off automatically, and the actual last step is fully preserved.

**Bug 5: `KeyError: 'given_Fx'` in the plot cell.**
The summary DataFrame was grouped by `cmd_dir` and `cmd_mag`, but `cmd_Fx` and `cmd_Fz` weren't carried through the `groupby().agg()`. They existed in the step-level data but not in the summary. Fix: add `cmd_Fx=("cmd_Fx", "first")` to the `agg()` call.

### Step 7: Pitfalls to watch out for

**Never use `mean()` on step-wise commanded values.** Commanded values are discrete — they jump from one value to another. Averaging across a jump creates a value that never existed in reality. Always use `first()` or `last()` for commanded signals, `mean()` only for sensor readings.

**Always filter transition rows before step detection.** When you resample data that contains step transitions, you will always get 1-2 rows with intermediate (averaged) values at each boundary. These corrupt your step detection. Filter them out first.

**`cumsum()` step numbering breaks if you filter afterwards.** If you detect steps (cumsum) and *then* filter rows, the step numbers become non-consecutive and confusing. Always filter first, then detect steps.

**The last step is always suspect.** In any experiment where logging continues after the protocol ends, the last step's tail is contaminated. Always handle it explicitly — either drop it or cap it.

**`ffill()` assumes the last known value is still valid.** If you use forward-fill and the source stream has crashed or gone silent, you'll silently propagate stale data. Always combine `ffill()` with a staleness check if you care about data validity.

### Step 8: What an expert would notice

An expert would immediately ask: *"What's your reference frame?"*

When the load cell reports `Fx` and `Fy`, these are in the **load cell's own frame**. When the GUI commands `Fx` and `Fz`, these are in the **robot's table frame**. If those two frames are not aligned — if the load cell is rotated even 5 degrees relative to the table — every single error measurement in the notebook is wrong. Not slightly wrong. Completely wrong.

The 14-43% systematic errors we saw are suspiciously large and suspiciously consistent across angles. That pattern suggests a calibration or frame mismatch problem, not a control problem. An expert would verify the axis mapping *before* spending time optimising the controller.

**The second expert observation:** The errors are not random — they're structured. For -129° they're consistently negative (under-delivery), for -49° they're consistently positive (over-delivery). This is a **direction-dependent bias**, which points to either cable geometry errors (the pulley positions in the model are wrong) or load cell mounting angle.

### Step 9: Lessons for completely different projects

**Lesson 1: Clean data at the source, not in analysis.**
Every bug we fixed in the notebook was caused by something messy in the recording — gui at 136Hz instead of 200Hz, resampling averaging step boundaries, logging continuing after the experiment ended. The best analysis pipeline is one that barely needs to exist because the data is already clean.

**Lesson 2: Step detection is hard. Threshold-based approaches always need tuning.**
Whether you're detecting steps in force data, peaks in audio, or events in network logs — the threshold is never obvious. Too sensitive and you get false positives (the solid red line bug). Too loose and you miss real transitions. Always visualise before you trust your detector.

**Lesson 3: Two clocks are dangerous.**
When gui logs at 136Hz and loadcell at 350Hz, you have two independent clocks that will never be perfectly synchronised. Any time you join two async streams, you need a strategy: nearest-neighbour, interpolation, or resampling to a common grid. We chose resampling. Each strategy has different failure modes — know yours.

**Lesson 4: The last item in any sequence is always special.**
Whether it's the last database transaction before a crash, the last packet in a network stream, or the last sweep step before recording stops — the last one is always incomplete or contaminated. Design your analysis to handle it explicitly rather than assuming all items are equal.

**Lesson 5: Separate mechanism from display.**
A speedometer in a car doesn't measure your speed — it *displays* it. The speed sensor runs independently. In software, your data-collection logic should never be coupled to your display logic. If updating the screen slows down your sensor read loop, you're mixing mechanism and display. Always separate them.

---

## PART 2 — The GUI Speed Fix

### What was the problem?

The GUI timer fired every 5ms (200Hz target). But the actual log rate was only 136Hz. Why? Because `_refresh()` was doing too much work in each tick:
- Calling `solve_tensions()` (matrix algebra)
- Updating 15+ Qt text widgets (each one triggers a repaint)
- Computing errors and colours

On a Raspberry Pi, these 15 widget updates alone can take 7-8ms. So a 5ms timer was actually running at ~7ms per tick = 136Hz.

### The fix

Separate the work into two categories:

**Must run every tick (computation):**
- `solve_tensions()` — result cached for the motor driver
- Measured force calculation — kept in shared state

**Can run every 4th tick (display at ~50Hz):**
- All `setText()` calls
- All `setStyleSheet()` calls  
- `workspace.update()` (canvas redraw)

The `_display_tick` counter counts ticks and only does widget updates when `tick % 4 == 0`. Human eyes can't see above ~60Hz anyway, so 50Hz display is indistinguishable from 200Hz display.

### The analogy

Think of a restaurant kitchen. The chef (computation) works at full speed — chopping, cooking, plating. The waiter (display) delivers plates to customers every few minutes. If the chef had to walk to every table after every knife cut, the kitchen would grind to a halt. Separating "doing the work" from "showing the result" is the same principle.

### Why this matters beyond this project

Any time you have a fast data stream feeding a slow display:
- Scientific instruments displaying live measurements
- Trading terminals showing market prices  
- Game engines rendering physics at 120Hz to a 60Hz monitor

The pattern is always the same: **run the critical path at full speed, throttle the display to what humans can perceive.** This is called *decoupling the update rate from the display rate* and it's one of the most fundamental performance patterns in real-time systems.

---

## Summary: The Three Rules You Should Remember

1. **Sensor data → mean(). Commanded data → first().** They're different in nature. One is continuous. The other is discrete. Treat them differently.

2. **Filter before you detect structure.** Noise and transition artefacts will always corrupt step detection, peak finding, and event segmentation. Remove them first, then look for patterns.

3. **Separate computation from display.** Your logging rate and your screen refresh rate should be independent. One is about data integrity. The other is about human perception. They have different requirements.

---

*Written after the force validation analysis session — 17 June 2026*
