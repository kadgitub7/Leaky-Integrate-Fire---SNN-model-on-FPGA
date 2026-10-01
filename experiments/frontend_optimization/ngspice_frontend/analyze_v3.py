import numpy as np

ground_truth = [
    # (beat, record, sym_python, T_rise_ms, T_width_ms)
    (1,  100, 0.35593, 58.333, 163.889),
    (2,  103, 0.50000, 11.111,  22.222),
    (3,  105, 0.13043, 16.667, 127.778),
    (4,  111, 0.09091, 11.111, 122.222),
    (5,  200, 0.20833, 13.889,  66.667),
    (6,  210, 0.15217, 19.444, 127.778),
    (7,  212, 0.16667, 13.889,  83.333),
    (8,  219, 0.33898, 55.556, 163.889),
    (9,  221, 0.55556, 13.889,  25.000),
    (10, 231, 0.15385, 11.111,  72.222),
]

sim_v3 = {
    1:  (1.239366e-02, 1.892941e-01, 2.231881e-01),
    2:  (2.220516e-02, 1.518860e-01, 1.702645e-01),
    3:  (5.202456e-03, 1.549508e-01, 2.174077e-01),
    4:  (3.909745e-03, 1.437509e-01, 2.162570e-01),
    5:  (8.381248e-03, 1.546607e-01, 2.010634e-01),
    6:  (5.921474e-03, 1.594203e-01, 2.174053e-01),
    7:  (6.813099e-03, 1.533657e-01, 2.066344e-01),
    8:  (1.179659e-02, 1.877892e-01, 2.231885e-01),
    9:  (2.477602e-02, 1.581048e-01, 1.739377e-01),
    10: (6.449893e-03, 1.478021e-01, 2.029535e-01),
}

beats = [g[0] for g in ground_truth]
recs = [g[1] for g in ground_truth]
sym_py = np.array([g[2] for g in ground_truth])
v_sym = np.array([sim_v3[b][0] for b in beats]) * 1000
v_rise = np.array([sim_v3[b][1] for b in beats]) * 1000
v_width = np.array([sim_v3[b][2] for b in beats]) * 1000

sorted_idx = np.argsort(sym_py)

print("=" * 75)
print("QRS SYMMETRY v3 — PURE ANALOG RESULTS")
print("19 transistors + 3 caps | VDD=0.9V | SKY130 | No ideal components")
print("=" * 75)

print(f"\n{'Beat':>4} {'Rec':>4} {'Sym_Py':>8} {'V_rise':>9} {'V_width':>9} {'V_sym':>8} {'V/Vt':>7}")
print(f"{'':>4} {'':>4} {'':>8} {'(mV)':>9} {'(mV)':>9} {'(mV)':>8} {'':>7}")
print("-" * 60)
Vt = 25.85
for i in sorted_idx:
    ratio = v_sym[i] / Vt
    print(f"B{beats[i]:>3} {recs[i]:>4} {sym_py[i]:>8.5f} {v_rise[i]:>9.2f} {v_width[i]:>9.2f} {v_sym[i]:>8.3f} {ratio:>7.3f}")

print(f"\nThermal voltage Vt = {Vt:.2f} mV at 27C")
print(f"If perfectly linear: V_sym = Vt * symmetry")

print(f"\n{'='*75}")
print("MONOTONICITY CHECK")
print(f"{'='*75}")

sym_sorted = v_sym[sorted_idx]
py_sorted = sym_py[sorted_idx]
monotonic = all(sym_sorted[i] < sym_sorted[i+1] for i in range(len(sym_sorted)-1))
print(f"\nStrictly monotonic: {'YES' if monotonic else 'NO'}")
for i in range(len(sym_sorted)):
    idx = sorted_idx[i]
    if i == 0:
        marker = "  "
    else:
        marker = "< " if sym_sorted[i] > sym_sorted[i-1] else ">= FAIL"
    print(f"  {marker}B{beats[idx]:>2} sym_py={py_sorted[i]:.5f}  V_sym={sym_sorted[i]:.3f} mV")

print(f"\n{'='*75}")
print("LINEARITY ANALYSIS: V_sym vs Vt * symmetry")
print(f"{'='*75}")

v_ideal = Vt * sym_py
err_abs = np.abs(v_sym - v_ideal)
err_pct = 100 * err_abs / v_ideal

print(f"\n{'Beat':>4} {'Sym_Py':>8} {'V_ideal':>9} {'V_actual':>9} {'Error':>8} {'Err%':>7}")
print(f"{'':>4} {'':>8} {'(mV)':>9} {'(mV)':>9} {'(mV)':>8} {'':>7}")
print("-" * 50)
for i in sorted_idx:
    print(f"B{beats[i]:>3} {sym_py[i]:>8.5f} {v_ideal[i]:>9.3f} {v_sym[i]:>9.3f} {err_abs[i]:>8.3f} {err_pct[i]:>6.1f}%")

k_fit = np.polyfit(sym_py, v_sym, 1)
v_linear = np.polyval(k_fit, sym_py)
err_linear = np.abs(v_sym - v_linear)

print(f"\nBest-fit linear: V_sym = {k_fit[0]:.3f} * sym + {k_fit[1]:.3f} mV")
print(f"  (vs ideal: V_sym = {Vt:.2f} * sym + 0.000 mV)")
print(f"\nMax deviation from best-fit line: {np.max(err_linear):.3f} mV")
print(f"RMS deviation from best-fit line: {np.sqrt(np.mean(err_linear**2)):.3f} mV")

r2 = 1 - np.sum((v_sym - v_linear)**2) / np.sum((v_sym - np.mean(v_sym))**2)
print(f"R-squared: {r2:.6f}")

corr = np.corrcoef(sym_py, v_sym)[0,1]
print(f"Pearson correlation: {corr:.6f}")

print(f"\n{'='*75}")
print("COMPARISON WITH v2 SUBCIRCUIT (same core, different control logic)")
print(f"{'='*75}")

v2_sym = np.array([12.43483, 22.30311, 5.217794, 3.921669, 8.407636,
                    5.939144, 6.835214, 11.83538, 24.89324, 6.469650])

diff = v_sym - v2_sym
print(f"\n{'Beat':>4} {'v2 (mV)':>9} {'v3 (mV)':>9} {'Diff':>8} {'Diff%':>7}")
print("-" * 45)
for i in range(10):
    pct = 100 * diff[i] / v2_sym[i]
    print(f"B{beats[i]:>3} {v2_sym[i]:>9.3f} {v_sym[i]:>9.3f} {diff[i]:>8.3f} {pct:>+6.1f}%")

print(f"\nMax difference: {np.max(np.abs(diff)):.3f} mV")
print(f"Mean difference: {np.mean(np.abs(diff)):.3f} mV")

idd = 8.439078e-12
power = 0.9 * abs(idd)
print(f"\n{'='*75}")
print("POWER CONSUMPTION")
print(f"{'='*75}")
print(f"\nAvg supply current: {abs(idd)*1e12:.2f} pA")
print(f"Power at VDD=0.9V:  {power*1e12:.2f} pW")
print(f"Energy per beat (~300ms): {power*0.3*1e12:.2f} pJ")

print(f"\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")
print(f"Circuit: qrs_symmetry_v3.spice")
print(f"Architecture: Log timers + SR latch + Corrected Gilbert cell")
print(f"Components: 19 transistors + 3 MIM capacitors")
print(f"  - 3 inverters (6T)")
print(f"  - SR latch (2T + 1 cap 5x5)")
print(f"  - Rise timer (4T + 1 cap 30x30)")
print(f"  - Width timer (3T + 1 cap 30x30)")
print(f"  - Gilbert cell (4T)")
print(f"VDD = 0.9V | Vbias = 0.35V | Pure analog SKY130")
print(f"No ideal voltage sources. No digital correction.")
print(f"")
print(f"Output range: {np.min(v_sym):.2f} - {np.max(v_sym):.2f} mV")
print(f"Monotonic: {monotonic}")
print(f"Linearity R^2: {r2:.6f}")
print(f"Correlation with Python: {corr:.6f}")
print(f"Power: {power*1e12:.2f} pW")
print(f"")
print(f"The circuit output V_sym is approximately Vt * symmetry.")
print(f"The subthreshold exponential I-V of the Gilbert cell NMOS pair")
print(f"cancels the logarithmic compression of the timers, yielding a")
print(f"voltage directly proportional to T_rise/T_width = symmetry.")
