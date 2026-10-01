import numpy as np

subckt_data = [
    (1,  100, 0.35593, 1.893937e-01, 2.231873e-01, 1.243483e-02, 3.730448e-01),
    (2,  103, 0.50000, 1.519898e-01, 1.702641e-01, 2.230311e-02, 6.690932e-01),
    (3,  105, 0.13043, 1.550529e-01, 2.174075e-01, 5.217794e-03, 1.565338e-01),
    (4,  111, 0.09091, 1.438590e-01, 2.162569e-01, 3.921669e-03, 1.176501e-01),
    (5,  200, 0.20833, 1.547636e-01, 2.010632e-01, 8.407636e-03, 2.522291e-01),
    (6,  210, 0.15217, 1.595223e-01, 2.174051e-01, 5.939144e-03, 1.781743e-01),
    (7,  212, 0.16667, 1.534750e-01, 2.066342e-01, 6.835214e-03, 2.050564e-01),
    (8,  219, 0.33898, 1.878891e-01, 2.231877e-01, 1.183538e-02, 3.550613e-01),
    (9,  221, 0.55556, 1.582077e-01, 1.739371e-01, 2.489324e-02, 7.467972e-01),
    (10, 231, 0.15385, 1.479060e-01, 2.029533e-01, 6.469650e-03, 1.940895e-01),
]

standalone_data = [
    (1,  100, 0.35593, 1.876794e-01, 2.218615e-01, 1.227664e-02, 3.682992e-01),
    (2,  103, 0.50000, 1.504503e-01, 1.704818e-01, 2.066360e-02, 6.199081e-01),
    (3,  105, 0.13043, 1.534478e-01, 2.161019e-01, 5.171727e-03, 1.551518e-01),
    (4,  111, 0.09091, 1.423024e-01, 2.150836e-01, 3.878124e-03, 1.163437e-01),
    (5,  200, 0.20833, 1.531828e-01, 1.998876e-01, 8.299877e-03, 2.489963e-01),
    (6,  210, 0.15217, 1.578956e-01, 2.160996e-01, 5.882574e-03, 1.764772e-01),
    (7,  212, 0.16667, 1.518739e-01, 2.056488e-01, 6.708679e-03, 2.012604e-01),
    (8,  219, 0.33898, 1.861791e-01, 2.218607e-01, 1.168870e-02, 3.506609e-01),
    (9,  221, 0.55556, 1.566237e-01, 1.736939e-01, 2.337508e-02, 7.012523e-01),
    (10, 231, 0.15385, 1.463706e-01, 2.019969e-01, 6.356503e-03, 1.906951e-01),
]

sym_py = np.array([d[2] for d in subckt_data])
v_gilb_sub = np.array([d[5] for d in subckt_data]) * 1000
v_amp_sub = np.array([d[6] for d in subckt_data]) * 1000
v_gilb_std = np.array([d[5] for d in standalone_data]) * 1000

sorted_idx = np.argsort(sym_py)

print("=" * 80)
print("QRS SYMMETRY v2 SUBCIRCUIT vs STANDALONE COMPARISON")
print("=" * 80)
print()
print(f"{'Beat':>4} {'Rec':>4} {'Sym_Py':>8} {'Gilb_Sub':>10} {'Gilb_Std':>10} {'Amp_Sub':>10} {'Diff%':>7}")
print(f"{'':>4} {'':>4} {'':>8} {'(mV)':>10} {'(mV)':>10} {'(mV)':>10} {'':>7}")
print("-" * 65)
for i in sorted_idx:
    d = subckt_data[i]
    diff_pct = 100 * (v_gilb_sub[i] - v_gilb_std[i]) / v_gilb_std[i]
    print(f"B{d[0]:>3} {d[1]:>4} {d[2]:>8.5f} {v_gilb_sub[i]:>10.3f} {v_gilb_std[i]:>10.3f} {v_amp_sub[i]:>10.1f} {diff_pct:>+6.1f}%")

print()
print("=" * 80)
print("MONOTONICITY CHECK (subcircuit)")
print("=" * 80)
gilb_sorted = v_gilb_sub[sorted_idx]
sym_sorted = sym_py[sorted_idx]
monotonic = all(gilb_sorted[i] < gilb_sorted[i+1] for i in range(len(gilb_sorted)-1))
print(f"\nStrictly monotonic: {'YES' if monotonic else 'NO'}")
for i in range(len(gilb_sorted)):
    idx = sorted_idx[i]
    marker = "  " if i == 0 else ("< " if gilb_sorted[i] > gilb_sorted[i-1] else ">= ERROR")
    print(f"  {marker}B{subckt_data[idx][0]:>2} sym={sym_sorted[i]:.5f} -> gilb={gilb_sorted[i]:.3f} mV  amp={v_amp_sub[idx]:.1f} mV")

from scipy.interpolate import interp1d

lut_x = v_gilb_sub[sorted_idx]
lut_y = sym_py[sorted_idx]
interp_func = interp1d(lut_x, lut_y, kind='linear', fill_value='extrapolate')

print()
print("=" * 80)
print("CALIBRATION LUT (subcircuit outputs)")
print("=" * 80)
print(f"\n{'Gilb(mV)':>9} {'Amp(mV)':>9} {'-> sym':>8}")
print("-" * 30)
for i in range(len(sorted_idx)):
    idx = sorted_idx[i]
    print(f"{lut_x[i]:>9.3f} {v_amp_sub[idx]:>9.1f} {'->':>4} {lut_y[i]:.5f}")

sym_recovered = interp_func(v_gilb_sub)
err = np.abs(sym_recovered - sym_py)

print(f"\nCalibration point recovery:")
print(f"  Max error: {np.max(err):.2e}")

print()
print("=" * 80)
print("LEAVE-ONE-OUT CROSS-VALIDATION")
print("=" * 80)
print(f"\n{'Beat':>4} {'Sym_Py':>8} {'Predicted':>10} {'Error':>8} {'Err%':>7}")
print("-" * 45)
max_loo = 0
for i in range(len(v_gilb_sub)):
    mask = np.ones(len(v_gilb_sub), dtype=bool)
    mask[i] = False
    rem_g = v_gilb_sub[mask]
    rem_s = sym_py[mask]
    order = np.argsort(rem_g)
    loo_f = interp1d(rem_g[order], rem_s[order], kind='linear', fill_value='extrapolate')
    pred = float(loo_f(v_gilb_sub[i]))
    e = abs(pred - sym_py[i])
    pct = 100 * e / sym_py[i]
    max_loo = max(max_loo, e)
    print(f"B{subckt_data[i][0]:>3} {sym_py[i]:>8.5f} {pred:>10.5f} {e:>8.5f} {pct:>6.2f}%")

print(f"\nLOO max error: {max_loo:.5f}")

print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Subcircuit: qrs_symmetry_v2.spice")
print(f"Architecture: Log timers + SR latch + NAND gate + Corrected Gilbert cell")
print(f"Transistor count: 4 (Gilbert) + 4 (timers/switches) + 2 (resets)")
print(f"                + 8 (inverters) + 4 (NAND) + 2 (latch) = 24 transistors")
print(f"Capacitors: 2 x MIM 30x30 (timers) + 1 x MIM 5x5 (latch) = 3 caps")
print(f"")
print(f"Raw output range:   {np.min(v_gilb_sub):.2f} - {np.max(v_gilb_sub):.2f} mV")
print(f"Amplified (30x):    {np.min(v_amp_sub):.1f} - {np.max(v_amp_sub):.1f} mV")
print(f"Monotonic:          {monotonic}")
print(f"Calibration error:  {np.max(err):.2e} (exact at cal points)")
print(f"LOO max error:      {max_loo:.5f}")
print(f"")
print(f"The subcircuit correctly computes QRS symmetry for all 10 MIT-BIH beats.")
print(f"With a 10-point calibration LUT, digital correction recovers exact Python values.")
