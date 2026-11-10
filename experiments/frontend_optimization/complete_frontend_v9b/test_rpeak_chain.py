"""
Modular test: rpeak_detector + pulse_gen chain
===============================================
Tests R-peak detection timing in isolation.

The rpeak_detector produces rpeak_raw (HIGH during QRS).
pulse_gen converts the rising edge of rpeak_raw to a ~0.5ms pulse.

Key questions:
1. When does rpeak_raw go HIGH relative to R-peak? (QRS onset)
2. When does rpeak_raw drop LOW? (QRS end = timer start)
3. When does rpeak_pulse fire relative to R-peak?
4. Are these timings consistent across beats?

Target: rpeak_pulse should fire at ~R+0ms (same as PWL trig_rpeak_sh)
"""

import subprocess, os, re, sys, time
import numpy as np

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108; QRS_S, QRS_E = 70, 130; RPEAK_IDX = 20

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


def run_test(record_id='100', n_beats=10):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal

    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    # Compute gain
    all_peaks = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(np.abs(sig[bs+QRS_S:bs+QRS_E, 0])))
    gain = 0.16 / max(all_peaks) if all_peaks else 0.05

    # Compute threshold (same as full_analog_chain_test.py)
    r_peak_scaled = []
    for idx, sym in valid[:n_beats+2]:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        v_scaled = VBASE + gain * sig[idx, 0]
        r_peak_scaled.append(v_scaled)
    rpeak_threshold = VBASE + 0.5 * (max(r_peak_scaled) - VBASE)

    # Load beats and compute R-peak times
    beats = []
    for i in range(1, n_beats + 2):
        if i >= len(valid) - 1: break
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]
        beats.append({
            'ecg_l0': beat[:, 0].copy(),
            'pre_rr': pre_rr,
            'label': aami_map[sym],
        })
    beats = beats[:n_beats]
    n = len(beats)

    # R-peak times in simulation
    dt = 1.0 / FS
    rpeak_times = [2.0]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    # ECG PWL
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

    sp = f""".title rpeak_chain_test
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 0.300

Vecg ecg_l0 0 PWL({make_pwl(ecg_pts)})
Vth rpeak_threshold 0 {rpeak_threshold:.4f}

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/pulse_gen.spice"

Xbias vbias_gen vref vdd vss bias_gen
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

.ic V(xbias.iref)=0.3

.tran 0.05m {total_time:.4f}

"""
    # Measure rpeak_raw and rpeak_pulse timing for each beat
    for j in range(n):
        rpt = rpeak_times[j]
        # Measure around R-peak: -10ms to +30ms in 0.5ms steps
        for offset_ms in range(-10, 31):
            t = rpt + offset_ms / 1000.0
            if 0.1 < t < total_time - 0.01:
                sp += f".meas tran raw_{j}_{offset_ms+10:02d} FIND V(rpeak_raw) AT={t:.7f}\n"
                sp += f".meas tran pls_{j}_{offset_ms+10:02d} FIND V(rpeak_pulse) AT={t:.7f}\n"
                sp += f".meas tran ecg_{j}_{offset_ms+10:02d} FIND V(ecg_l0) AT={t:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_test_rpeak_chain.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)

    print(f"Record {record_id}: {n} beats, gain={gain:.4f}, threshold={rpeak_threshold:.4f}V")
    print(f"Running rpeak chain test...")
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

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
    print("=" * 70)
    print("R-PEAK DETECTION TIMING")
    print("=" * 70)
    print(f"{'Beat':>4s} {'Label':>5s} {'rpk_raw_rise':>14s} {'rpk_raw_drop':>14s} {'QRS_width':>10s} {'rpk_pulse':>14s} {'Pulse_width':>12s}")
    print("-" * 75)

    timing_data = []
    for j in range(n):
        # Find rpeak_raw rise and drop
        raw_rise = None
        raw_drop = None
        pulse_start = None
        pulse_end = None

        for offset_ms in range(-10, 31):
            idx = offset_ms + 10
            v_raw = results.get(f'raw_{j}_{idx:02d}')
            v_pls = results.get(f'pls_{j}_{idx:02d}')
            if v_raw is not None:
                if raw_rise is None and v_raw > 0.45:
                    raw_rise = offset_ms
                if raw_rise is not None and raw_drop is None and v_raw < 0.45:
                    raw_drop = offset_ms
            if v_pls is not None:
                if pulse_start is None and v_pls > 0.45:
                    pulse_start = offset_ms
                elif pulse_start is not None and pulse_end is None and v_pls < 0.45:
                    pulse_end = offset_ms

        qrs_w = raw_drop - raw_rise if raw_rise is not None and raw_drop is not None else None
        pulse_w = pulse_end - pulse_start if pulse_start is not None and pulse_end is not None else None

        timing_data.append({
            'raw_rise': raw_rise, 'raw_drop': raw_drop,
            'pulse_start': pulse_start, 'pulse_end': pulse_end,
            'qrs_w': qrs_w, 'pulse_w': pulse_w,
        })

        raw_rise_s = f"R{raw_rise:+d}ms" if raw_rise is not None else "N/A"
        raw_drop_s = f"R{raw_drop:+d}ms" if raw_drop is not None else "N/A"
        qrs_s = f"{qrs_w}ms" if qrs_w is not None else "N/A"
        pulse_s = f"R{pulse_start:+d}ms" if pulse_start is not None else "N/A"
        pw_s = f"{pulse_w}ms" if pulse_w is not None else "N/A"

        print(f"{j:4d} {beats[j]['label']:>5s} {raw_rise_s:>14s} {raw_drop_s:>14s} {qrs_s:>10s} {pulse_s:>14s} {pw_s:>12s}")

    # Summary statistics
    rises = [t['raw_rise'] for t in timing_data if t['raw_rise'] is not None]
    drops = [t['raw_drop'] for t in timing_data if t['raw_drop'] is not None]
    pulses = [t['pulse_start'] for t in timing_data if t['pulse_start'] is not None]

    print(f"\n{'SUMMARY':}")
    if rises:
        print(f"  rpeak_raw rise:  mean={np.mean(rises):.1f}ms, range=[{min(rises)},{max(rises)}]ms relative to R-peak")
    if drops:
        print(f"  rpeak_raw drop:  mean={np.mean(drops):.1f}ms, range=[{min(drops)},{max(drops)}]ms relative to R-peak")
    if pulses:
        print(f"  rpeak_pulse:     mean={np.mean(pulses):.1f}ms, range=[{min(pulses)},{max(pulses)}]ms relative to R-peak")

    print(f"\n  Target: rpeak_pulse at R+0ms (for F3/F4/F8 capture)")
    if pulses:
        mean_err = np.mean(pulses)
        print(f"  Actual:  rpeak_pulse at R{mean_err:+.1f}ms (error: {mean_err:+.1f}ms)")
    if drops:
        mean_drop = np.mean(drops)
        print(f"  Timer starts at: R{mean_drop:+.1f}ms (rpeak_raw drop)")

    # Show detailed trace for beat 2
    print(f"\n  Detailed trace (beat 2):")
    print(f"  {'Offset':>8s} {'ECG (V)':>8s} {'rpk_raw':>8s} {'rpk_pls':>8s}")
    for offset_ms in range(-10, 31):
        idx = offset_ms + 10
        ecg_v = results.get(f'ecg_2_{idx:02d}')
        raw_v = results.get(f'raw_2_{idx:02d}')
        pls_v = results.get(f'pls_2_{idx:02d}')
        if ecg_v is not None:
            print(f"  R{offset_ms:+4d}ms {ecg_v:8.4f} {raw_v:8.4f} {pls_v:8.4f}")


if __name__ == '__main__':
    run_test('100', n_beats=10)
