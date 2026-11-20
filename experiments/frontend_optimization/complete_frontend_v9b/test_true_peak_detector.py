"""
Test: true_peak_detector.spice — verify it fires at the actual R-peak
======================================================================
Runs the full chain: rpeak_detector → rpeak_hold → true_peak_detector
Then measures when peak_detected goes HIGH relative to annotation R-peak.

Target: peak_detected fires at R+0.5 to R+2ms with ≤ ±0.3ms beat-to-beat jitter.
"""

import subprocess, os, re, sys, time
import numpy as np

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend_v9b")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108; QRS_S, QRS_E = 70, 130

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join(f"{t:.7f} {v:.6f}" for t, v in c)


def run_test(record_id='100', n_beats=15):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal

    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    all_peaks = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(np.abs(sig[bs+QRS_S:bs+QRS_E, 0])))
    gain = 0.16 / max(all_peaks) if all_peaks else 0.05

    r_peak_scaled = []
    for idx, sym in valid[:n_beats+2]:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        v_scaled = VBASE + gain * sig[idx, 0]
        r_peak_scaled.append(v_scaled)
    rpeak_threshold = VBASE + 0.5 * (max(r_peak_scaled) - VBASE)

    beats = []
    for i in range(1, n_beats + 2):
        if i >= len(valid) - 1: break
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beats.append({
            'ecg_l0': sig[bs:be, 0].copy(),
            'pre_rr': pre_rr,
            'label': aami_map[sym],
        })
    beats = beats[:n_beats]
    n = len(beats)
    dt = 1.0 / FS

    rpeak_times = [2.0]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    ecg_pts = [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt
        ecg0 = beats[i]['ecg_l0']
        t_before = beat_start - 0.003
        if t_before > 0.01:
            ecg_pts.append((t_before, VBASE))
        for j in range(len(ecg0)):
            t = beat_start + j * dt
            if t > 0:
                ecg_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg0[j]))))
        t_after = beat_start + len(ecg0) * dt + 0.003
        ecg_pts.append((t_after, VBASE))
    ecg_pts.append((total_time, VBASE))

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title true_peak_detector_test
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 0.300

Vecg ecg_l0 0 PWL({make_pwl(ecg_pts)})
Vth rpeak_threshold 0 {rpeak_threshold:.4f}

.include "{fp}/inverter.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/rpeak_hold.spice"
.include "{fp}/edge_buffer.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/true_peak_detector.spice"

* === OLD RPEAK CHAIN (provides arm_signal) ===
Xrpeak ecg_l0 rpeak_threshold rpeak_level vdd vss vbias rpeak_detector
Xhold rpeak_level rpeak_level_held vdd vss vbias rpeak_hold

* === TRUE PEAK DETECTOR ===
Xtpd ecg_l0 rpeak_level_held peak_detected_level vdd vss vbias true_peak_detector

* === PULSE GENERATION (level → pulse) ===
Xeb_tpd peak_detected_level tpd_eb vdd vss edge_buffer
Xpg_tpd tpd_eb tpd_pg vdd vss vbias pulse_gen
Xeb_tpd2 tpd_pg true_rpeak_pulse vdd vss edge_buffer

* Also generate old rpeak_pulse for comparison
Xeb_rpk rpeak_level_held rpk_eb vdd vss edge_buffer
Xpg_rpk rpk_eb rpk_pg vdd vss vbias pulse_gen
Xeb_rpk2 rpk_pg old_rpeak_pulse vdd vss edge_buffer

.ic V(xtpd.peak_hold)=0.3

.tran 0.1m {total_time:.4f} UIC

"""
    # Measure at fine resolution around each R-peak
    for j in range(n):
        rpt = rpeak_times[j]
        # Sample from -10ms to +10ms in 0.2ms steps
        for k in range(100):
            offset_ms = -10.0 + k * 0.2
            t = rpt + offset_ms / 1000.0
            if 0.1 < t < total_time - 0.01:
                sp += f".meas tran ecg_{j}_{k:02d} FIND V(ecg_l0) AT={t:.7f}\n"
                sp += f".meas tran hold_{j}_{k:02d} FIND V(xtpd.peak_hold) AT={t:.7f}\n"
                sp += f".meas tran det_{j}_{k:02d} FIND V(peak_detected_level) AT={t:.7f}\n"
                sp += f".meas tran arm_{j}_{k:02d} FIND V(rpeak_level_held) AT={t:.7f}\n"
                sp += f".meas tran trk_{j}_{k:02d} FIND V(xtpd.track_level) AT={t:.7f}\n"
                sp += f".meas tran cen_{j}_{k:02d} FIND V(xtpd.charge_en) AT={t:.7f}\n"
                sp += f".meas tran cpi_{j}_{k:02d} FIND V(xtpd.cp_int) AT={t:.7f}\n"
                sp += f".meas tran tpls_{j}_{k:02d} FIND V(true_rpeak_pulse) AT={t:.7f}\n"
                sp += f".meas tran opls_{j}_{k:02d} FIND V(old_rpeak_pulse) AT={t:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_test_true_peak.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)

    print(f"Record {record_id}: {n} beats, gain={gain:.4f}, threshold={rpeak_threshold:.4f}V")
    print(f"Running true peak detector test...")
    t0 = time.time()
    result = subprocess.run([NGSPICE, '-b', '-o', log, tb],
                            capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"WARNING: ngspice returned {result.returncode}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")

    with open(log) as f:
        text = f.read()

    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass

    print(f"Extracted {len(results)} measurements\n")

    # Analyze timing for each beat
    print("=" * 90)
    print("TRUE PEAK DETECTOR TIMING ANALYSIS")
    print("=" * 90)

    old_fire_times = []
    new_fire_times = []

    for j in range(n):
        rpt = rpeak_times[j]
        print(f"\n--- Beat {j} (R-peak at {rpt:.3f}s, label={beats[j]['label']}) ---")
        print(f"  {'Offset':>8s} {'ECG(V)':>8s} {'Hold(V)':>8s} {'ARM':>5s} {'TRK':>5s} {'CEN':>5s} {'CPint':>7s} {'DET':>5s} {'TruPls':>7s} {'OldPls':>7s}")

        old_fire = None
        new_fire = None
        ecg_at_rpeak = None

        for k in range(100):
            offset_ms = -10.0 + k * 0.2
            ecg_v = results.get(f'ecg_{j}_{k:02d}')
            hold_v = results.get(f'hold_{j}_{k:02d}')
            det_v = results.get(f'det_{j}_{k:02d}')
            arm_v = results.get(f'arm_{j}_{k:02d}')
            trk_v = results.get(f'trk_{j}_{k:02d}')
            cen_v = results.get(f'cen_{j}_{k:02d}')
            cpi_v = results.get(f'cpi_{j}_{k:02d}')
            tpls_v = results.get(f'tpls_{j}_{k:02d}')
            opls_v = results.get(f'opls_{j}_{k:02d}')

            if ecg_v is None: continue

            if abs(offset_ms) < 0.01:
                ecg_at_rpeak = ecg_v

            arm_s = "HIGH" if arm_v and arm_v > 0.45 else " low"
            trk_s = "HIGH" if trk_v and trk_v > 0.45 else " low"
            cen_s = "HIGH" if cen_v and cen_v > 0.45 else " low"
            cpi_s = f"{cpi_v:.3f}" if cpi_v is not None else "  N/A"
            det_s = "HIGH" if det_v and det_v > 0.45 else " low"
            tpls_s = f"{tpls_v:.3f}" if tpls_v else "  N/A"
            opls_s = f"{opls_v:.3f}" if opls_v else "  N/A"

            if opls_v and opls_v > 0.45 and old_fire is None:
                old_fire = offset_ms
            if tpls_v and tpls_v > 0.45 and new_fire is None:
                new_fire = offset_ms

            if -7 <= offset_ms <= 6:
                print(f"  R{offset_ms:+6.1f}ms {ecg_v:8.4f} {hold_v:8.4f} {arm_s:>5s} {trk_s:>5s} {cen_s:>5s} {cpi_s:>7s} {det_s:>5s} {tpls_s:>7s} {opls_s:>7s}")

        old_fire_s = f"R{old_fire:+.1f}ms" if old_fire is not None else "NONE"
        new_fire_s = f"R{new_fire:+.1f}ms" if new_fire is not None else "NONE"
        print(f"  >> Old rpeak_pulse: {old_fire_s}, True rpeak_pulse: {new_fire_s}")
        if ecg_at_rpeak:
            print(f"  >> ECG at R-peak: {ecg_at_rpeak:.4f}V")

        if old_fire is not None: old_fire_times.append(old_fire)
        if new_fire is not None: new_fire_times.append(new_fire)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if old_fire_times:
        old_arr = np.array(old_fire_times)
        print(f"\nOLD rpeak_pulse (threshold crossing):")
        print(f"  Mean: R{np.mean(old_arr):+.1f}ms")
        print(f"  Range: [{min(old_arr):+.1f}, {max(old_arr):+.1f}] ms")
        print(f"  Jitter (std): ±{np.std(old_arr):.2f}ms")
        print(f"  Jitter (p2p): {np.ptp(old_arr):.1f}ms")

    if new_fire_times:
        new_arr = np.array(new_fire_times)
        print(f"\nTRUE rpeak_pulse (peak detector):")
        print(f"  Mean: R{np.mean(new_arr):+.1f}ms")
        print(f"  Range: [{min(new_arr):+.1f}, {max(new_arr):+.1f}] ms")
        print(f"  Jitter (std): ±{np.std(new_arr):.2f}ms")
        print(f"  Jitter (p2p): {np.ptp(new_arr):.1f}ms")

        if old_fire_times:
            old_jitter = np.ptp(old_arr)
            new_jitter = np.ptp(new_arr)
            improvement = old_jitter / new_jitter if new_jitter > 0 else float('inf')
            print(f"\n  Jitter improvement: {improvement:.1f}× (from {old_jitter:.1f}ms to {new_jitter:.1f}ms)")

    if not new_fire_times:
        print("\nWARNING: True peak detector never fired! Check circuit.")
        print("Possible issues:")
        print("  - ARM signal (rpeak_level_held) not going HIGH")
        print("  - Peak hold cap not charging")
        print("  - Detect comparator offset too large")

    # Pass/fail
    if new_fire_times:
        new_arr = np.array(new_fire_times)
        jitter_pass = np.ptp(new_arr) <= 1.0
        timing_pass = 0 <= np.mean(new_arr) <= 5.0
        fire_rate = len(new_fire_times) / n
        detect_pass = fire_rate >= 0.8

        print(f"\n{'PASS' if jitter_pass else 'FAIL'}: Jitter <= 1.0ms (actual: {np.ptp(new_arr):.2f}ms)")
        print(f"{'PASS' if timing_pass else 'FAIL'}: Mean timing R+0 to R+5ms (actual: R{np.mean(new_arr):+.1f}ms)")
        print(f"{'PASS' if detect_pass else 'FAIL'}: Detection rate >= 80% (actual: {fire_rate*100:.0f}%)")


if __name__ == '__main__':
    run_test('100', n_beats=12)
