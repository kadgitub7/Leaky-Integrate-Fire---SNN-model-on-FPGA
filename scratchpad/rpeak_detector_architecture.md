# Analog R-Peak Detector — Architecture Specification

## Summary

A fully analog-feasible R-peak detector validated on all 43 MIT-BIH Arrhythmia Database records.

| Metric | Value |
|--------|-------|
| Sensitivity | 99.50% |
| PPV (Precision) | 99.14% |
| F1 Score | 0.9932 |
| Timing MAE | 2.83 ms |
| Timing P95 | 1.44 ms |
| Problem records (sens<95% or PPV<80%) | 0 / 43 |
| Records with F1 ≥ 0.95 | 41 / 43 |
| Records with F1 ≥ 0.99 | 37 / 43 |

Target technology: SKY130 130nm CMOS, VDD = 0.9V, subthreshold operation.

## Signal Flow

```
ECG ──→ BPF(5-15Hz) ──→ d/dt ──→ |x| ──┬──→ Peak_Det(100ms) ──→ Smooth(30ms) ──→ [env_fast]
                                          │
                                          └──→ RC_LP(1.5s) ───────────────────────→ [env_avg]

Detection Logic:
  find_peaks(env_fast, min_distance=80ms)
  for each peak:
    if env_fast > threshold1  AND  env_fast > 3.0 × env_avg:
      → T-wave rejection → refractory check → DETECT (map to raw ECG max)
    if gap > 1.66 × avg_RR:
      → search-back with relaxed noise gate (1.8×)
```

## Block Descriptions

### 1. Bandpass Filter (5–15 Hz)
**Function**: Isolate QRS complex energy, reject baseline wander and high-frequency noise.
**Analog**: 2nd-order gm-C bandpass filter. Two OTA-C integrators in feedback.
**Parameters**: f_low = 5 Hz, f_high = 15 Hz, Q ≈ 1.
**Components**: 2 OTAs (subthreshold gm ≈ 20–200 nA/V), 2 integration caps (~10–50 pF).

### 2. Differentiator
**Function**: Emphasize sharp QRS slopes over slower P/T waves.
**Analog**: Capacitor (Cd ≈ 1 pF) in series with a resistive element (weak-inversion FET, R ≈ 1–10 MΩ). Transfer: s·Cd / (1 + s·Cd·R).
**Components**: 1 MIM cap, 1 NFET (long-channel for high resistance).

### 3. Full-Wave Rectifier
**Function**: Convert biphasic derivative to unipolar signal.
**Analog**: Two diode-connected NFETs (one for positive, one for negative half). The negative path inverts via a current mirror before summing.
**Components**: 4–6 MOSFETs. Subthreshold diode-connected FETs have soft turn-on (≈60 mV/decade), which provides natural smoothing.

### 4. Fast Envelope Follower (τ = 100 ms)
**Function**: Track peaks of rectified signal with 100 ms exponential decay.
**Analog**: Leaky peak detector.
- **Charge path**: Diode-connected NFET (fast attack, charges hold cap when input exceeds held voltage).
- **Discharge path**: Weak NFET (vbias-controlled, L=8 W=0.42) slowly discharges cap.
- **Hold cap**: ~10 pF MIM. Time constant τ = C / I_leak. For τ=100ms with C=10pF: I_leak = 100 pA.
**Components**: 2 MOSFETs + 1 MIM cap.

### 5. Envelope Smoother (30 ms window)
**Function**: Remove noise spikes from the fast envelope.
**Analog**: Simple RC low-pass filter with τ ≈ 30 ms.
- R ≈ 30 MΩ (long-channel NFET in triode), C ≈ 1 pF.
**Components**: 1 MOSFET + 1 MIM cap.

### 6. Slow Average Tracker (τ = 1.5 s)
**Function**: Track the average energy level of the rectified signal. This is the noise floor reference.
**Analog**: RC low-pass filter with τ = 1.5 s.
- R ≈ 150 MΩ (very long-channel NFET, L=16+ W=0.42), C ≈ 10 pF.
- Alternatively: OTA with very low gm driving a 10 pF cap. gm = C/τ = 6.67 nS.
**Components**: 1 MOSFET + 1 MIM cap (or 1 OTA + 1 cap).
**Critical**: This block is the key innovation. By tracking the AVERAGE (not peak), it distinguishes QRS peaks (which spike above average) from sustained noise (which raises both peak and average equally).

### 7. Noise Gate Comparator
**Function**: Compare fast envelope peak to 3× slow average. Only pass detections where the QRS peak clearly exceeds the noise floor.
**Analog**: Comparator with built-in 3:1 gain ratio.
- **Option A**: Current-mirror comparator with 3:1 mirror ratio. env_fast drives a 1× transistor, env_avg drives a 3× transistor. The comparator trips when I_fast > I_avg×3, i.e., when gm×env_fast > 3×gm×env_avg.
- **Option B**: Voltage scaling. Use a resistive divider (3 matched FETs in series) to scale env_fast by 1/3, then compare with env_avg using a standard differential pair.
**Parameters**: Noise ratio = 3.0 (primary path), 1.8 (search-back path).
**Components**: 6–10 MOSFETs.

### 8. EWMA Adaptive Threshold
**Function**: Track signal and noise levels using exponential weighted moving average. Threshold = noise + 0.25 × (signal − noise).
**Analog**: Two peak-hold + leaky-integrator circuits:
- **Signal level (spki)**: Updated with α=0.125 on each detection. Leaky integrator with τ = 8 beats.
- **Noise level (npki)**: Updated with α=0.125 on each non-detection peak. Same circuit topology.
- **Threshold generation**: Analog weighted sum. npki + 0.25×(spki − npki) = 0.75×npki + 0.25×spki. Implemented with a 1:3 current mirror ratio.
**Components**: ~10 MOSFETs + 2 hold caps.

### 9. Refractory Timer (200 ms)
**Function**: Blanking period after each detection to prevent double-triggering.
**Analog**: Existing `rpeak_hold.spice` circuit — monostable with 200 ms hold time. Detection pulse charges a cap; slow discharge defines the hold period. Comparator on the cap voltage produces the refractory gate signal.
**Components**: Reuse existing circuit.

### 10. Search-Back Logic
**Function**: When a beat gap exceeds 166% of the average RR interval, look for a missed beat in the gap using a relaxed threshold (50% of primary) and relaxed noise gate (1.8× instead of 3.0×).
**Analog implementation**: This is the most complex block to implement in pure analog.
- **RR average**: EWMA of recent RR intervals. Already available from the existing `rr_timing.spice` circuit.
- **166% comparator**: Compare current timer voltage against 1.66× the EWMA-predicted RR. Implementable with a scaled comparator (similar to noise gate).
- **Relaxed threshold**: A second threshold comparator at 50% of the primary.
- **Relaxed noise gate**: A second noise comparator with 1.8× ratio instead of 3.0×.
- **Peak selection in gap**: The largest envelope peak in the gap that passes both relaxed criteria.
**Simplification for analog**: In practice, the search-back fires rarely (<1% of beats in clean records). It primarily helps during transient rhythm changes. An analog implementation could use a simpler version: if the gap timer exceeds 166% RR, temporarily lower the threshold and noise gate for the next candidate peak.

### 11. T-Wave Rejection
**Function**: Reject T-waves that appear within 360 ms of the last QRS. A T-wave has lower slope than the preceding QRS.
**Analog**: Compare the current peak's derivative magnitude to 50% of the last detection's derivative. This requires storing the last QRS slope.
- **Slope capture**: Sample-and-hold on the derivative at detection time.
- **50% scaling**: Current mirror with 1:2 ratio.
- **Comparator**: Standard diff pair comparing current slope to half the stored slope.
**Components**: ~8 MOSFETs + 1 hold cap.

### 12. Raw ECG Peak Locator
**Function**: Map from the envelope peak back to the actual R-peak in the raw ECG signal. The envelope peak lags the raw ECG peak by ~27 ms (75 ms search window backward, 150 ms forward). Find the maximum of the raw ECG within this window.
**Analog**: The existing `true_peak_detector.spice` can serve this function. It tracks the raw ECG peak during the QRS window and fires when the ECG drops below the tracked peak.
**Components**: Reuse existing circuit.

## Component Budget

| Block | MOSFETs | Caps | Notes |
|-------|---------|------|-------|
| BPF (gm-C) | 8–12 | 2 | Two OTA-C integrators |
| Differentiator | 2 | 1 | Cap + FET resistor |
| Rectifier | 4–6 | 0 | Diode-connected + mirror |
| Fast envelope | 2 | 1 | Peak det + discharge |
| Smoother | 1 | 1 | RC low-pass |
| Slow average | 1–2 | 1 | RC low-pass |
| Noise gate | 6–10 | 0 | Scaled comparator |
| EWMA threshold | 8–12 | 2 | Signal/noise track + sum |
| Refractory | 4–6 | 1 | Monostable (existing) |
| T-wave rejection | 6–8 | 1 | Slope compare |
| Peak locator | 10–14 | 1 | True peak det (existing) |
| **Total** | **~55–85** | **~11** | |

This is comparable to the existing frontend (~200 FETs for all feature extraction + trigger chain). The R-peak detector adds ~60–80 FETs.

## Key Design Parameters

| Parameter | Value | Analog mapping |
|-----------|-------|----------------|
| BPF passband | 5–15 Hz | gm-C: gm₁ = C₁×2π×5, gm₂ = C₂×2π×15 |
| Fast envelope τ | 100 ms | C_hold / I_discharge |
| Smoother τ | 30 ms | R × C |
| Slow average τ | 1.5 s | R × C (or C/gm for OTA-C) |
| Noise ratio | 3.0× | Current mirror ratio or voltage divider |
| Threshold fraction | 0.25 | 1:3 current mirror ratio |
| EWMA α | 0.125 | Charge injection ratio |
| Refractory period | 200 ms | Hold cap discharge time |
| T-wave window | 360 ms | Timer comparator |
| Search-back trigger | 1.66 × RR_avg | Scaled comparator on RR timer |
| Search-back NR | 1.8× | Relaxed noise gate ratio |
| Min peak distance | 80 ms | Refractory-like blanking |

## Performance on Hardest Records

| Record | Beats | Sens | PPV | F1 | Notes |
|--------|-------|------|-----|----|----|
| 207 | 1858 | 95.9% | 85.0% | 0.901 | Ventricular tachycardia episodes |
| 108 | 1761 | 97.8% | 90.5% | 0.941 | Very low amplitude, noisy |
| 203 | 2978 | 95.6% | 97.4% | 0.965 | PVCs with noise |
| 114 | 1877 | 99.7% | 97.1% | 0.984 | Noise regions, inverted R-peaks |
| 107 | 2135 | 98.4% | 98.5% | 0.984 | Paced rhythm |

Previously (V4 without noise gate), records 108/114/232/219 had PPV of 50–70% due to massive false positives in noisy regions. The dual-envelope noise gate eliminated this problem entirely.

## Comparison to Previous Versions

| Version | Sens | PPV | F1 | Problem Records | Key Change |
|---------|------|-----|----|----|---|
| V1 (basic) | 99.24% | 89.43% | 0.9408 | 7 | Baseline |
| V4 (improved adaptive) | 99.25% | 92.52% | 0.9577 | 5 | Better threshold |
| V9 (smoothed) | 99.26% | 93.58% | 0.9634 | 5 | Envelope smoothing |
| **V15/Final (dual-envelope)** | **99.50%** | **99.14%** | **0.9932** | **0** | **Noise gate** |

The dual-envelope noise gate is the single most impactful improvement, raising PPV by 6 percentage points while also improving sensitivity.

## Software Reference Implementation

File: `scratchpad/analog_detector_final.py`

The `detect_final()` function is the reference. It uses only operations that map directly to analog circuits:
- `butter() + sosfiltfilt()` → gm-C bandpass
- `np.diff()` → capacitor differentiator
- `np.abs()` → diode-connected FET rectifier
- `envelope_follower()` → leaky peak detector (diode + RC)
- `moving_average()` → RC low-pass filter
- `rc_lowpass()` → RC low-pass filter
- `find_peaks()` → comparator on smoothed envelope
- Threshold comparison → comparator
- Noise ratio comparison → scaled comparator
- EWMA update (α=0.125) → charge/discharge ratio on hold cap
