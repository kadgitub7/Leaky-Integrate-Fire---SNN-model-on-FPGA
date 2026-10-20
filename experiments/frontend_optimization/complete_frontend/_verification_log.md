# Verification Log — Analog Frontend

## Phase 0: Baseline (Feature Circuits with Ideal PWL Sources)

**Date**: 2026-10-15
**Data source**: Cached results from full_accuracy_test.py across 16 DS1 records (29,875 circuit-matched beats)

### Per-Record |rho| Results

| Rec | Beats | F1:rr  | F2:asym | F3:bi0 | F4:bi1 | F5:pw  | F6:wid | F7:slp | F8:sym |
|-----|-------|--------|---------|--------|--------|--------|--------|--------|--------|
| 101 | 1581  | 0.390  | 0.356   | 0.992  | 0.987  | 0.989  | 0.999  | 0.997  | 0.997  |
| 106 | 1719  | 0.891  | 0.843   | 0.977  | 0.988  | 0.995  | 0.998  | 0.992  | 1.000  |
| 108 | 1496  | 0.444  | 0.411   | 0.976  | 0.992  | 0.988  | 0.993  | 0.999  | 1.000  |
| 112 | 2153  | 0.568  | 0.657   | 0.986  | 0.987  | 0.985  | 0.998  | 0.993  | 0.998  |
| 114 | 1595  | 0.390  | 0.404   | 0.992  | 0.866  | 0.988  | 0.996  | 0.998  | 1.000  |
| 115 | 1657  | 0.499  | 0.482   | 0.990  | 0.992  | 0.987  | 0.999  | 1.000  | 0.998  |
| 116 | 2044  | 0.496  | 0.610   | 0.993  | 0.991  | 0.968  | 0.998  | 1.000  | 1.000  |
| 118 | 1934  | 0.443  | 0.455   | 0.991  | 0.989  | 0.995  | 0.999  | 0.999  | 0.998  |
| 119 | 1685  | 0.822  | 0.826   | 0.986  | 0.977  | 0.999  | 1.000  | 1.000  | 0.999  |
| 122 | 2102  | 0.414  | 0.467   | 0.991  | 0.991  | 0.982  | 0.996  | 0.994  | 0.999  |
| 124 | 1371  | 0.291  | 0.314   | 0.976  | 0.988  | 0.952  | 1.000  | 0.999  | 1.000  |
| 201 | 1666  | 0.696  | 0.684   | 0.989  | 0.981  | 0.988  | 0.999  | 0.997  | 0.999  |
| 203 | 2528  | 0.851  | 0.809   | 0.983  | 0.982  | 0.979  | 0.999  | 1.000  | 0.999  |
| 205 | 2252  | 0.457  | 0.506   | 0.941  | 0.980  | 0.988  | 0.998  | 0.996  | 0.995  |
| 207 | 1542  | 0.461  | 0.467   | 0.987  | 0.987  | 0.981  | 1.000  | 0.995  | 1.000  |
| 209 | 2550  | 0.647  | 0.584   | 0.984  | 0.990  | 0.985  | 0.999  | 0.998  | 0.997  |

### Pass Rate per Feature (threshold from v3)

| Feature | Threshold | Pass/Total | Notes |
|---------|-----------|------------|-------|
| F1:rr_ratio | >= 0.970 | 0/16 | Max |rho|=0.891 (rec 106). Circuit dynamic range only ~42mV. |
| F2:rr_asym  | >= 0.995 | 0/16 | Max |rho|=0.843 (rec 106). Same dynamic range issue. |
| F3:bi_L0    | >= 0.980 | 12/16 | Solid; failures marginal (0.941-0.977). |
| F4:bi_L1    | >= 0.986 | 11/16 | Mostly passing; rec 114 outlier at 0.866. |
| F5:pwave    | >= 0.995 | 1/16 | Only rec 119 passes (0.999). Most are 0.98-0.99. |
| F6:width    | >= 0.998 | 11/16 | Very strong; failures at 0.993-0.996. |
| F7:slope    | >= 0.995 | 12/16 | Very strong; failures at 0.992-0.994. |
| F8:symmetry | >= 0.995 | 16/16 | Perfect — all records pass. |

### NOTE: Above results are PRE-FIX (no IC propagation, 0.5ms capture, F2 measured during capture overwrite)

### F1/F2 Fixes Applied (2026-10-19)
1. IC propagation between simulation batches (ewma, esamp, prev nodes)
2. 55ms capture pulse (was 0.5ms) for source follower settling
3. F2 measurement moved to rpeak-65ms (before capture window opens)

### Post-Fix Baseline (16-record all-8 test, IN PROGRESS)
| Rec | F1     | F2     | F3     | F4     | F5     | F6     | F7     | F8     |
|-----|--------|--------|--------|--------|--------|--------|--------|--------|
| 100 | 0.995  | 0.999  | 0.988  | 0.994  | 0.996  | 0.998  | 0.998  | 0.996  |
(Remaining 15 records still running)

---

## Phase 2: Supplementary Circuit Isolation (2026-10-19)

### bias_gen
- [PASS] vbias=0.2915V (spec 0.29±0.01V), 67pA supply, stable by 1ms

### rpeak_detector
- [KNOWN OFFSET] Fires 1-5ms early on QRS upslope at 90% avg peak threshold
- Mean offset=-2.7ms, std=1.4ms across 5 beats (record 100)
- Accepted: systematic offset preserves Spearman ranking

### pulse_gen
- [FIX → PASS] Bleeder NFET W: 0.42→1.5 (3.6× more discharge current)
- Result: 0.41ms pulse (spec 0.5±0.2ms), 0.90V amplitude, clean quiescent

### delay_timer (5 triggers, 4 iterations of threshold calibration)
- [PASS] All within ±0.5ms of target
- Thresholds: vth_10=0.2905V, vth_14_fast=0.2100V, vth_83=0.4714V, vth_69=0.4496V, vth_300=0.3323V
- Final: rr_copy=-0.003ms, s14=+0.206ms, wid=-0.33ms, s69=+0.46ms, bi_copy=-0.18ms

### pwave_predictor
- [FUNCTIONAL] pw_base: 207.7ms before next R (target 206ms, err -1.7ms)
- pw_peak: 134.7ms before next R (target 131ms, err -3.7ms)
- rr_cap: 30ms before next R (target 5ms) — larger offset required for comparator to resolve before timer reset
- Thresholds for RR=800ms: vth_base=0.1902V, vth_peak=0.1580V, ewma_offset=+30mV
