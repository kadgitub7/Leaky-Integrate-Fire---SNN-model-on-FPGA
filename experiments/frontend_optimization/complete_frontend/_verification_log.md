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

### Finding: F1/F2 thresholds not achievable with current rr_timing.spice

The rr_timing circuit's charge-sharing RR ratio computation produces output voltages in a very narrow range (~0.28-0.33V). For records with regular rhythm (e.g., 101), the RR variations are so small that the circuit output is nearly constant (CV=1.5%), making rank correlation unreliable. Records with more arrhythmia (106, 119, 203) show higher |rho| (0.82-0.89) but still below the 0.97 threshold.

This is inherent to the circuit design, not a regression. The v3 threshold values for F1/F2 may need to be revisited.

### Conclusion

- **F6, F7, F8**: BASELINE CEMENTED — consistently meet or nearly meet thresholds across all records
- **F3, F4**: Strong but threshold may be slightly aggressive for some records
- **F5**: Good (0.95-0.99) but threshold of 0.995 is only met by 1/16 records
- **F1, F2**: Threshold not achievable with current circuit — needs investigation
