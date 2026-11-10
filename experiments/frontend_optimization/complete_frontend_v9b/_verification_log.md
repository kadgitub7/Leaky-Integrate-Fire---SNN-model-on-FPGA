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

### Post-Fix Baseline (16-record all-8 test, COMPLETE 2026-10-20)
| Rec | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | Pass |
|-----|------|------|------|------|------|------|------|------|------|
| 100 | 0.995 | 0.999 | 0.988 | 0.994 | 0.996 | 0.998 | 0.998 | 0.996 | 8/8 |
| 103 | 0.992 | 0.999 | 0.989 | 0.981 | 0.994 | 0.999 | 0.998 | 0.998 | 6/8 |
| 107 | 0.996 | 0.999 | 0.965 | 0.991 | 0.990 | 1.000 | 1.000 | 1.000 | 6/8 |
| 109 | 0.993 | 0.999 | 0.971 | 0.961 | 0.998 | 0.999 | 0.999 | 1.000 | 6/8 |
| 112 | 0.998 | 0.998 | 0.985 | 0.987 | 0.985 | 0.998 | 0.993 | 0.998 | 6/8 |
| 117 | 0.992 | 0.999 | 0.993 | 0.998 | 0.991 | 0.999 | 0.982 | 0.999 | 6/8 |
| 119 | 0.999 | 1.000 | 0.986 | 0.978 | 0.999 | 1.000 | 1.000 | 0.999 | 7/8 |
| 123 | 0.980 | 0.999 | 0.993 | 0.994 | 0.990 | 0.997 | 0.998 | 0.997 | 6/8 |
| 200 | 1.000 | 1.000 | 0.987 | 0.985 | 0.989 | 0.999 | 0.996 | 1.000 | 6/8 |
| 205 | 0.984 | 0.994 | 0.943 | 0.980 | 0.988 | 0.998 | 0.994 | 0.996 | 3/8 |
| 208 | 0.996 | 1.000 | 0.993 | 0.991 | 0.989 | 0.999 | 0.998 | 0.996 | 7/8 |
| 212 | 0.997 | 0.998 | 0.988 | 0.992 | 0.985 | 0.999 | 0.998 | 0.999 | 7/8 |
| 217 | 0.998 | 1.000 | 0.987 | 0.976 | 0.985 | 0.999 | 0.999 | 1.000 | 6/8 |
| 221 | 1.000 | 1.000 | 0.986 | 0.971 | 0.985 | 0.999 | 0.981 | 0.999 | 5/8 |
| 228 | 1.000 | 1.000 | 0.987 | 0.991 | 0.993 | 0.999 | 0.998 | 1.000 | 7/8 |
| 232 | 0.992 | 0.999 | 0.985 | 0.981 | 0.989 | 0.999 | 0.997 | 1.000 | 6/8 |

Summary: F1 0/16 fail, F2 1/16 fail, F3 3/16, F4 8/16, F5 13/16, F6 2/16, F7 4/16, F8 0/16

---

## Phase 2: Supplementary Circuit Isolation — ±0.5% Tolerance (2026-10-20)

### bias_gen (XMn2 L=8→L=12)
- [PASS ±0.5%] vbias=0.30031V (target 0.300V, error +0.10%), 67pA supply, stable by 1ms

### pulse_gen (XMbleed W=0.42→W=1.34)
- [PASS ±0.5%] Pulse width=0.498ms (target 0.500ms, error -0.38%), 0.90V amplitude, clean quiescent
- Test condition: 0.4ms input ramp (matching comparator output rise time)

### delay_timer — recalibrated thresholds for real bias_gen (vbias=0.30031V)
- [PASS ±0.5%] All 5 triggers within ±0.3%
- Thresholds: vth_10=0.2892V, vth_14_fast=0.2117V, vth_83=0.47093V, vth_69=0.44934V, vth_300=0.3303V
- Results: rr_copy=9.978ms(-0.22%), s14=13.903ms(+0.02%), wid=27.778ms(-0.08%), s69=69.381ms(-0.03%), bi_copy=300.915ms(+0.30%)

### rpeak_detector
- [MARGINAL] Fires 1-5ms early on QRS upslope (mean -2.7ms, std 1.4ms)
- At RR=800ms: mean error -0.34%, worst case -0.51% (at boundary)
- Systematic offset cancels for most features (triggers are relative to detected R-peak)
- Needs evaluation in Phase 3 integration

### pwave_predictor — calibrated thresholds for real bias_gen
- [PASS ±0.5%] pw_base: 206.57ms before next R (target 206ms, +0.28%)
  - vth_base = 0.1874V
- [PASS ±0.5%] pw_peak: 130.55ms before next R (target 131ms, -0.34%)
  - vth_peak = 0.1538V
- [FUNDAMENTAL ISSUE] rr_cap: fires 34.5ms before next R (target 5ms)
  - SF rate at 800ms is only 0.47 mV/ms → comparator cannot resolve within 5ms
  - Consistent offset should preserve Spearman ranking; to be verified in Phase 3

### Comprehensive Integration Test v1 (2026-10-20)
- [FAIL] 3-beat test (rpeak at 10/810/1610ms): pulse_gen fails for slow-transitioning signals
- Root cause: Main timer comparator output transitions ~0.4ms; coupling current (0.49nA) ≈ bleeder current (0.48nA) → pulse_raw never reaches inverter threshold
- Affected: wid, s69, bi_copy (main timer), pw_base, pw_peak, rr_cap (pwave predictor)
- Not affected: s14, rr_copy (fast timer — edges naturally fast enough)

### edge_buffer.spice — Fix for slow comparator transitions
- Two cascaded inverter_sub instances sharpen ~0.4ms transition to ~10ns
- Applied between delay_timer main timer outputs / pwave outputs and pulse_gen inputs
- New file: `complete_frontend/edge_buffer.spice`

### Comprehensive Integration Test v2 (2026-10-20)
- [PASS] ALL 8 pulse generators fire correctly (0.9V amplitude, 0.6-0.7ms width)
- Beat 1 timing: rr_copy -0.39%*, s14 -0.24%, wid -0.15%, s69 -0.07%, bi_copy +0.28%, pw_base -0.13%, pw_peak -0.02%
- Beat 2 timing: rr_copy 0%, s14 +0.29%, wid +0.18%, s69 +0.07%, bi_copy +0.32%
- Pwave levels beat 2 consistent with beat 1 (pw_base -0.11%, pw_peak -0.006% from rpeak 3)
- Bias stable: 0.30031V across full 2.5s simulation
- (*rr_copy beat 1: 0.1ms timestep gives -0.85%, but 0.01ms timestep gives -0.39%; measurement resolution artifact)
- Pulse widths: 0.6-0.7ms (slightly wider than 0.5ms target due to sharper buffered edge; acceptable for S&H settling)

---

## Phase 4: Full Analog Integration (2026-10-20)

### Round 1: All triggers use standard pulse_gen (0.7ms pulse)
- [PARTIAL] F1,F2,F5,F6,F7,F8: all within ±0.005 of ideal baseline
- [FAIL] F3:bi_L0 degraded: 0.895 vs 0.956 ideal (-0.061)
- [FAIL] F4:bi_L1 degraded: 0.870 vs 0.988 ideal (-0.118)
- Root cause isolated: analog bi_copy trigger (0.7ms pulse from edge_buffer+pulse_gen)
- Isolation test (3 modes on rec 100, 36 beats):
  - ideal: F3=0.956, F4=0.988
  - all analog: F3=0.895, F4=0.870
  - analog+ideal_bi: F3=0.956, F4=0.988 (matches ideal exactly)

### long_pulse_gen.spice — Fix for bi_copy EWMA charge-sharing
- Same topology as pulse_gen but bleeder W=0.41 (vs W=1.34) for ~2ms pulse width
- [PASS ±0.5%] Pulse width=1.990ms (target 2.000ms, error -0.5%), 0.90V amplitude
- New file: `complete_frontend/long_pulse_gen.spice`

### Round 2: bi_copy uses long_pulse_gen (1.99ms pulse), all others use pulse_gen (0.7ms)
- [PASS] Record 100 (36 beats): F3=0.9560, F4=0.9884 — exact match with ideal baseline
- All 8 features within ±0.005 of ideal (F1 actually improved: 0.857 vs 0.829)
- Full 5-record test COMPLETE (9887 beats, ~9 hours):

| Rec | Beats | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|-----|-------|------|------|------|------|------|------|------|------|
| 100 | 1929 | 0.991 | 0.999 | 0.985 | 0.992 | 0.996 | 0.998 | 0.997 | 0.996 |
| 119 | 1683 | 0.996 | 1.000 | 0.983 | 0.974 | 0.999 | 1.000 | 0.994 | 0.999 |
| 208 | 2509 | 0.994 | 1.000 | 0.982 | 0.985 | 0.989 | 0.999 | 0.998 | 0.996 |
| 232 | 1511 | 0.982 | 0.999 | 0.980 | 0.982 | 0.989 | 0.999 | 0.994 | 1.000 |
| 205 | 2255 | 0.980 | 0.994 | 0.942 | 0.979 | 0.989 | 0.998 | 0.990 | 0.996 |

- Comparison vs ideal baseline (delta from post-fix Phase 0):
  - Most features within ±0.005 of ideal
  - Largest deltas: rec 208 F3 (-0.011), rec 232 F1 (-0.010)
  - Record 205 F3=0.942 matches ideal baseline (0.943) — inherent to record
  - F3/F4 degradation from Round 1 (0.7ms bi_copy) fully resolved by long_pulse_gen
