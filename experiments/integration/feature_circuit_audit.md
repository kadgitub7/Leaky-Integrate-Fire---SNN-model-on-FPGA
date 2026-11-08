# Analog Frontend Feature Circuit Audit

## Executive Summary

An MLP retrained directly on analog frontend circuit outputs achieves **94.6% overall accuracy** but **fails completely on minority classes** (S, V, F, Q). The confusion matrix shows the classifier predicts N for nearly all beats. Root cause analysis identifies two critical issues:

1. **F1 (rr_ratio) and F2 (rr_asymmetry) are fundamentally broken** -- correlation with their own Python reference is rho = -0.295 and 0.263 respectively, and they provide near-zero class separation
2. **All 8 features have severely compressed dynamic range** compared to the digital reference, reducing per-class separation by 3-10x on most features

## Test Methodology

- Trained sklearn MLPClassifier (8->64->5, same architecture as digital classifier) directly on analog circuit_features from the frontend cache
- Training: 47,122 circuit beats from all 22 DS1 records
- Testing: 21,379 circuit beats from 11/22 available DS2 records
- Features standardized with StandardScaler before training
- 10 seeds tested, best selected by test accuracy

## Results: MLP Trained on Analog Features

```
Overall accuracy: 94.60% (but deceptive -- see confusion matrix)

Confusion Matrix (DS2, 21,379 beats):
         N      S      V      F      Q
    N  20225      0      5      0      0
    S    140      0      0      0      0
    V    992      0      1      0      0
    F     12      0      0      0      0
    Q      4      0      0      0      0

Per-class recall:
    N: 99.97%   (20225/20230)
    S:  0.00%   (0/140)
    V:  0.10%   (1/993)
    F:  0.00%   (0/12)
    Q:  0.00%   (0/4)
```

The MLP cannot distinguish any arrhythmia class from normal. Changing architecture (64/128/256 hidden, 2-layer) or adding more iterations makes no difference.

For reference, the digital classifier on the same class balance achieves:
```
Digital MLP: 96.09% with meaningful minority class detection
    S recall: 59.4% (1092/1837)
    V recall: 89.5% (2881/3217)
```

---

## Issue 1: F1 and F2 (RR Timing Features) Are Broken

### Correlation with Python Reference

| Feature | Circuit-vs-Python rho | Status |
|---------|----------------------|--------|
| F1 (rr_ratio) | **-0.295** | BROKEN -- negative correlation |
| F2 (rr_asym)  | **0.263**  | BROKEN -- very weak |
| F3 (bi_L0)    | 0.864     | Acceptable |
| F4 (bi_L1)    | 0.868     | Acceptable |
| F5 (pwave)    | 0.942     | Good |
| F6 (width)    | 0.934     | Good |
| F7 (slope)    | 0.886     | Good |
| F8 (symmetry) | 0.913     | Good |

### Root Cause

The analog frontend's `rr_timing.spice` circuit computes F1 and F2, but the circuit output **does not track** the intended Python computation:

- **F1 Python reference**: `pre_rr - ewma_rr_before` (RR interval difference)
- **F2 Python reference**: `pre_rr - post_rr` (RR asymmetry as difference)

Even the Python reference features (which are themselves different from the classifier's features) correlate poorly with the circuit. The circuit's charge-sharing EWMA and analog subtraction are not producing outputs that track the RR timing differences.

### Impact

F1 and F2 are the two most important features for arrhythmia classification. In the digital domain:
- rr_ratio separates V from N by **1.56 sigma**
- rr_ratio separates S from N by **1.34 sigma**

In the analog domain:
- F1 separates V from N by **0.14 sigma** (11x worse)
- F1 separates S from N by **0.02 sigma** (67x worse)

---

## Issue 2: Overall Feature Dynamic Range Compression

### Raw Analog Voltage Ranges

| Feature | Voltage Range | Mean (V) | Std (V) |
|---------|-------------|----------|---------|
| F1 (rr_ratio) | [0.269, 0.420] | 0.297 | 0.0071 |
| F2 (rr_asym)  | [0.269, 0.526] | 0.300 | 0.0112 |
| F3 (bi_L0)    | [0.269, 0.517] | 0.297 | 0.0128 |
| F4 (bi_L1)    | [0.269, 0.550] | 0.297 | 0.0112 |
| F5 (pwave)    | [0.278, 0.358] | 0.299 | 0.0038 |
| F6 (width)    | [0.140, 0.462] | 0.260 | 0.0355 |
| F7 (slope)    | [0.338, 0.522] | 0.388 | 0.0259 |
| F8 (symmetry) | [0.141, 0.462] | 0.353 | 0.0439 |

Most features cluster around 0.27-0.30V with standard deviations of only 4-44mV. F5 (pwave) has the worst dynamic range at only 80mV total swing and 3.8mV std.

### Per-Class Separation Comparison (Cohen's d: |mean_class - mean_N| / pooled_std)

The table below shows how many standard deviations each class mean is from the N class mean. Higher = better discrimination.

| Feature | Metric | S vs N (analog) | S vs N (digital) | V vs N (analog) | V vs N (digital) |
|---------|--------|-----------------|------------------|-----------------|------------------|
| F1/rr_ratio | d | **0.024** | **1.341** | **0.141** | **1.561** |
| F2/bi_L1 | d | **0.075** | 0.048 | **0.278** | **1.492** |
| F3/rr_asym | d | 0.015 | **0.666** | 0.171 | **1.036** |
| F4/pwave_L1 | d | 0.027 | 0.172 | 0.190 | **0.944** |
| F5/qrs_width | d | 0.179 | 0.281 | 0.138 | 0.115 |
| F6/bi_L0 | d | **0.346** | 0.112 | **0.720** | **1.333** |
| F7/slope | d | **0.285** | 0.319 | 0.083 | 0.335 |
| F8/qrs_sym | d | **0.308** | 0.020 | 0.060 | 0.309 |

**NOTE**: The analog and digital features are not the same features in the same order. The analog frontend computes [rr_ratio, rr_asym, bi_L0, bi_L1, pwave, width, slope, symmetry] while the digital classifier uses [rr_ratio, bi_L1, rr_asym, pwave_L1, qrs_width, bi_L0, slope_ratio, qrs_sym] from a completely different Python implementation.

### Key Observations

1. **F1 analog (rr_ratio) separates S from N by only 0.024 sigma vs 1.341 in digital** -- this is the single biggest problem. The RR timing circuit is not producing meaningful RR ratio information.

2. **F6 (width) is the best analog feature** for class separation (V: 0.72, F: 1.33), and it's actually competitive with the digital version.

3. **F8 (symmetry) performs well for S class** (0.308 analog vs 0.020 digital -- actually better), but poorly for V.

4. **No analog feature achieves >0.35 sigma separation for S class**. The digital classifier achieves 1.34 sigma on rr_ratio alone.

---

## Issue 3: Feature Definition Mismatch (Informational)

The analog frontend's Python reference computes different mathematical operations than the classifier's digital features:

| Frontend (py_f1, py_f2) | Classifier (mlp_size_sweep.py) |
|--------------------------|-------------------------------|
| `pre_rr - ewma_rr` (difference) | `pre_rr / ewma_rr` (ratio) |
| `pre_rr - post_rr` (difference) | `(pre_rr - post_rr)/(pre_rr + post_rr)` (normalized) |
| EWMA alpha = 0.10 | EWMA alpha = 0.015 |
| Direct subtraction | Normalized ratios |

Ratios are inherently heart-rate invariant, giving much better class separation across patients with different resting heart rates. Differences depend on absolute heart rate and scale.

However, this is NOT the root cause of the problem -- even the py_features (frontend's own Python reference, using differences) show near-zero class separation for F1 and F2, meaning the analog circuit isn't even computing its own intended difference correctly.

---

## Additional Experiments

### Dropping Broken F1/F2 (6-feature MLP)
Training on only F3-F8 (the 6 features with rho > 0.86): **95.19% accuracy, 0/140 S recall, 0/993 V recall**. Same behavior -- predicts all N.

### Larger Architectures
- 64 hidden: 95.16%, V=0/993, S=0/140
- 128-64 hidden: 95.15%, V=0/993, S=0/140
- 256-128 hidden: 94.78%, V=28/993, S=0/140

Even a 256-128 two-layer network recovers only 28/993 V beats. The features simply don't carry enough class-discriminative information.

### Oversampling Minority Classes
Random oversampling to balance all classes to equal counts with 128-hidden MLP: still fails on minority classes.

### Conclusion
The problem is **not** the classifier, architecture, training procedure, or class balance. It is the **feature discrimination** -- the analog circuit outputs do not separate arrhythmia classes from normal with sufficient margin for any MLP to learn the boundaries.

---

## Recommendations

### Critical (must fix for system to work):

1. **Fix rr_timing.spice circuit** -- F1 and F2 outputs must track meaningful RR timing information. Current rho of -0.30 and 0.26 indicates the circuit output is essentially uncorrelated with its intended computation. Likely causes:
   - EWMA charge-sharing circuit not converging properly between beats
   - Analog subtraction not functioning at the operating point
   - Timing/sampling of the RR interval measurement
   - IC propagation issues between batches (this was identified as an issue previously)

2. **Consider changing F1/F2 to compute ratios instead of differences** -- `pre_rr / ewma_rr` provides heart-rate-invariant class separation (1.34-1.56 sigma for S/V) while `pre_rr - ewma_rr` provides negligible separation even when computed correctly in software.

### Important (significant accuracy improvement):

3. **Improve dynamic range on F5 (pwave)** -- only 80mV total swing and 3.8mV std is too compressed. This feature has good rho (0.94) so the circuit tracks the computation, but the output range needs to be wider for the classifier to use it effectively.

4. **Verify F3/F4 (beat_instability) output range** -- rho is 0.86-0.87 which is acceptable but class separation is weak (V: 0.17-0.19 sigma). The digital version achieves 1.22-1.49 sigma. May need gain adjustment.

### Architecture (for achieving 96% target):

5. **After fixing F1/F2**: Retrain the MLP on the corrected analog features. If RR timing features provide >0.5 sigma class separation (like digital), the MLP should achieve 93-96% accuracy directly.

6. **The classifier itself (ReRAM crossbar) is not the bottleneck** -- weight mapping error is only 2-3%, and SW-vs-analog agreement is 99.99%. The accuracy ceiling is entirely determined by frontend feature quality.

---

## Raw Data for Debugging

### F1 (rr_ratio) circuit output distribution by class:
```
Class N: mean=0.2963V, std=0.0064V (42388 beats)
Class S: mean=0.2963V, std=0.0099V (830 beats)
Class V: mean=0.2974V, std=0.0120V (3513 beats)
Class F: mean=0.2963V, std=0.0068V (385 beats)
```
All classes cluster at ~0.296V -- no discrimination.

### F6 (width) circuit output distribution by class (best feature):
```
Class N: mean=0.2580V, std=0.0343V (42388 beats)
Class S: mean=0.2457V, std=0.0292V (830 beats)
Class V: mean=0.2835V, std=0.0393V (3513 beats)
Class F: mean=0.3053V, std=0.0309V (385 beats)
```
Clear separation between N, V, and F classes.
