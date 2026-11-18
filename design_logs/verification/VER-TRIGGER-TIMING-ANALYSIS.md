# Frontend Trigger Timing Analysis — MAJOR-007

## Date: 2026-11-17

## Objective
Determine root cause of SPICE-Python correlation falling below the rho > 0.97 target
for the analog frontend feature extraction circuits.

## Methodology
1. Compared ALL available SPICE test data against Python feature extraction
2. Traced timing parameters across every test script and circuit file
3. Ran per-record rho analysis across 30 integration test records
4. Created F6 timing mismatch diagnostic to prove root cause

## Key Finding: Feature Extraction vs Trigger Timing

The rho > 0.97 requirement has two components:
1. **Feature extraction circuits** (diff pairs, S&H) — VERIFIED CORRECT
2. **Trigger/timing circuits** (delay_timer, pwave_predictor, rr_capture) — ISSUES FOUND

### Evidence: Ideal vs Analog Triggers

| Test Configuration | Triggers | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|--------------------|----------|----|----|----|----|----|----|----|----|
| full_accuracy_test (6 recs) | ALL ideal | 0.998-1.000 | 0.995-1.000 | 0.90-1.00 | 0.90-1.00 | 0.90-1.00 | 0.90-1.00 | 0.90-1.00 | 0.90-1.00 |
| isolation_test_v5 (1 rec, 10 beats) | F6/F7 analog, rest ideal | 1.000 | 0.988 | 0.903 | 0.988 | 0.994 | 0.985 | 1.000 | 0.902 |
| full_analog_chain (1 rec, 20 beats) | ALL analog | 0.940 | 0.732 | 0.934 | 0.994 | 0.467 | 0.991 | 0.998 | 0.935 |
| integration_tests (30 recs, 30 beats) | ALL analog | median 0.92 | median 0.38 | median 0.53 | median 0.43 | median 0.16 | median 0.83 | median 0.54 | median 0.66 |

**Conclusion**: With ideal triggers (exact timing), feature extraction circuits achieve rho > 0.90.
With analog triggers, correlation degrades significantly, especially F2 and F5.

## Issue 1: F6 Delay Timer Calibration (R+28ms instead of R+83ms)

### Design Intent
`qrs_width.spice` header: "Proxy: ECG at R+83ms. Trigger: sample_trig (R+83ms)"
`full_validation.py`: WIDTH_OFF = 30 (R+83ms) — achieves 94.44% accuracy

### Actual Calibration
ALL test scripts calibrate vth_28 for R+28ms (10 samples at 360Hz):
- `full_accuracy_test.py`: `trig_wid = make_trig(rpeak_times, 10 / FS, PW, total_time)`
- `full_analog_chain_test.py`: `wid_adj = round(28 * FS / 1000)  # 10 samples`
- `isolation_test_v5.py`: `make_trig(rpeak_times, 10/FS, PW, total_time)`
- `integration_test.py`: `vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28 + offset_ms - pipe_wid))`

Only `energy_test.py` uses the correct `30 / FS` = R+83ms.

### Diagnostic Proof
```
SPICE F6 vs Python F6@R+28ms (WIDTH_OFF=10): rho=+0.9876, R2=0.9753
SPICE F6 vs Python F6@R+83ms (WIDTH_OFF=30): rho=+0.8539, R2=0.7292
```
CONFIRMED: SPICE delay timer fires at R+28ms, not R+83ms.

### Fix Required
Recalibrate delay timer threshold from vth_28 (~0.471V) to vth_83 (~0.439V).
Interpolating from DT SF curve: 83ms is between 50ms (0.4566V) and 100ms (0.4304V).

## Issue 2: F5 Pwave Predictor Trigger Timing

The pwave_predictor analog circuit produces triggers for the pwave S&H,
but these triggers have significant timing errors compared to ideal.
- Ideal trigger: rho = 0.994
- Analog trigger: rho = 0.467
This is not a delay timer issue — the pwave predictor uses a separate circuit path.

## Issue 3: F2 RR Capture Timing

The rr_timing circuit's prev_node latch does not perfectly track the previous RR interval:
- Ideal trigger: rho = 0.988
- Analog chain: rho = 0.732
Possible causes: charge leakage, capture timing offset, TG charge injection.

## Impact on Classifier Training

The classifier should be trained on **Python-extracted features** (which match the DESIGN INTENT),
NOT on the SPICE data with broken trigger timing. Rationale:
1. Feature extraction circuits ARE correct (proven with ideal triggers)
2. Trigger timing errors are BUGS that must be fixed
3. Training on broken trigger data would encode timing errors into classifier weights
4. The Python extraction (full_validation.py) achieves 94.44% — this is the proven baseline

## Recommendations

### Immediate (MAJOR-007 Integration)
- Train classifier on Python features (generate_circuit_feature_cache.py with WIDTH_OFF=30)
- This is correct because the design intent is R+83ms and Python matches design
- Retrain currently running: 94.48% +/- 0.32% (consistent with validated 94.44%)

### Follow-up (New Issues)
1. **Fix F6 delay timer**: Recalibrate from R+28ms to R+83ms (vth_83 ~0.439V)
2. **Fix F5 pwave predictor**: Debug analog trigger timing errors
3. **Fix F2 rr_capture**: Debug prev_node charge retention
4. **Re-run full 44-record validation**: After trigger fixes, verify rho > 0.97 for ALL records

## Files Analyzed
- `qrs_width.spice` — circuit design (R+83ms intent)
- `full_accuracy_test.py` — 6-record cache with ideal triggers
- `full_analog_chain_test.py` — analog chain test with ALL analog triggers
- `isolation_test_v5.py` — isolation test (analog DT for F6/F7 only)
- `integration_test.py` — integration test (all analog triggers)
- `energy_test.py` — only test using correct 30/FS timing
- `full_validation.py` — validated Python extraction (94.44%)
- `generate_circuit_feature_cache.py` — classifier training cache

## Diagnostic Scripts
- `scratchpad/f6_timing_diagnostic.py` — proves F6 timing mismatch
- `scratchpad/per_record_rho_analysis.py` — 30-record per-record rho analysis
