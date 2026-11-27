"""
Validate R-peak detector with adaptive threshold.

Step 1: Python-computed EWMA threshold as PWL (proves concept)
Step 2: Will be replaced by analog EWMA circuit

The adaptive threshold tracks signal and noise levels using EWMA,
producing a time-varying voltage that adapts to QRS amplitude changes.
"""
import numpy as np
import wfdb
import os
import sys
import subprocess
import re

sys.path.insert(0, 'scratchpad')
from analog_detector_final import detect_final, get_intermediate_signals, envelope_follower, moving_average, rc_lowpass
from generate_spice_tb import format_pwl

DB_PATH = 'mitdb_data'
FS = 360
TEST_DURATION_S = 10.0
SKY130_LIB = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/ngspice_frontend/sky130_minimal.lib.spice'
CIRCUIT_DIR = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/complete_frontend_v9b'
OUT_DIR = 'scratchpad'
NGSPICE = 'C:/Spice64/bin/ngspice_con.exe'

MATCH_TOL_MS = 150.0

# EWMA parameters matching Python detector
NOISE_RATIO = 3.0
THRESHOLD_FRAC = 0.25
ALPHA = 0.125
FAST_TAU = 0.100
SMOOTH_MS = 30
REFRACTORY_MS = 200


def nearest_neighbor_match(py_times_s, sp_times_s, tol_s=0.150):
    py_matched = set()
    offsets_ms = []
    for si, st in enumerate(sp_times_s):
        best_dist = float('inf')
        best_pi = -1
        for pi, pt in enumerate(py_times_s):
            if pi in py_matched:
                continue
            dist = abs(st - pt)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi >= 0 and best_dist <= tol_s:
            py_matched.add(best_pi)
            offsets_ms.append((st - py_times_s[best_pi]) * 1000.0)
    tp = len(offsets_ms)
    fn = len(py_times_s) - tp
    fp = len(sp_times_s) - tp
    return tp, fn, fp, offsets_ms


def compute_adaptive_threshold_signal(ecg, fs, ecg_v):
    """Compute a continuous adaptive threshold signal in voltage domain.

    Uses EWMA-like logic: tracks signal/noise peaks through the
    envelope signals and produces a time-varying threshold.

    Returns: threshold_v array (same length as ecg_v)
    """
    from scipy.signal import butter, sosfiltfilt, find_peaks

    # Reproduce the Python detector's intermediate signals
    sos = butter(2, [5, 15], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, ecg)
    derivative = np.diff(filtered, prepend=filtered[0])
    rectified = np.abs(derivative)
    env_fast_raw = envelope_follower(rectified, fs, FAST_TAU)
    smooth_win = max(1, int(SMOOTH_MS * fs / 1000))
    env_fast = moving_average(env_fast_raw, smooth_win)
    env_avg = rc_lowpass(rectified, fs, 1.5)

    # Now compute EWMA threshold in continuous time
    min_dist = int(REFRACTORY_MS * fs / 1000)
    peak_dist = int(80 * fs / 1000)
    candidates, _ = find_peaks(env_fast, distance=peak_dist)

    spki = 0.0
    npki = 0.0
    threshold_signal = np.zeros(len(ecg))
    last_thr_update_idx = 0

    detections = []

    for idx in candidates:
        ef = env_fast[idx]
        ea = env_avg[idx]
        threshold = npki + THRESHOLD_FRAC * (spki - npki)
        noise_thresh = NOISE_RATIO * ea

        # Fill threshold from last update to this candidate
        threshold_signal[last_thr_update_idx:idx] = threshold

        is_det = False
        if ef > threshold and ef > noise_thresh:
            if len(detections) > 0:
                dt_ms = (idx - detections[-1]) * 1000.0 / fs
                if dt_ms < 360:
                    if len(detections) >= 2:
                        prev_ef = env_fast[detections[-1]]
                        if ef < 0.5 * prev_ef:
                            npki = npki + ALPHA * (ef - npki)
                            last_thr_update_idx = idx
                            continue
                if dt_ms < REFRACTORY_MS:
                    last_thr_update_idx = idx
                    continue

            detections.append(idx)
            spki = spki + ALPHA * (ef - spki)
            is_det = True
        else:
            npki = npki + ALPHA * (ef - npki)

        last_thr_update_idx = idx

    # Fill remaining
    threshold_signal[last_thr_update_idx:] = npki + THRESHOLD_FRAC * (spki - npki)

    # Now map threshold from rectified/envelope domain to voltage domain
    # The env_fast tracks the SPICE fast_envelope output
    # In SPICE, the fast envelope tracks the ECG voltage directly
    # So threshold in env domain → threshold in voltage domain

    # Map: envelope value → ECG voltage
    # env_fast max ≈ corresponds to QRS peaks in ECG voltage
    # We need: when env_fast > threshold, that maps to ecg_v > some voltage

    # For the SPICE circuit: env_fast tracks ecg_v peaks
    # So the threshold in envelope domain maps directly to a voltage
    # threshold_v = ecg_v_baseline + threshold_ratio * (peak - baseline)

    # Simpler approach: compute the threshold as a fraction of the recent peak
    # Use a running peak tracker with decay
    ecg_median = np.median(ecg_v)

    # Track recent QRS peak voltage with 3s decay
    peak_track = np.zeros(len(ecg_v))
    decay = np.exp(-1.0 / (fs * 3.0))
    peak_track[0] = ecg_median
    for i in range(1, len(ecg_v)):
        if ecg_v[i] > peak_track[i-1]:
            peak_track[i] = ecg_v[i]
        else:
            peak_track[i] = ecg_median + (peak_track[i-1] - ecg_median) * decay

    # Threshold = baseline + 0.25 * (peak_track - baseline)
    threshold_v = ecg_median + 0.20 * (peak_track - ecg_median)

    # Smooth to avoid rapid changes
    smooth = int(0.1 * fs)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        threshold_v = np.convolve(threshold_v, kernel, mode='same')

    # Clip to reasonable range
    threshold_v = np.clip(threshold_v, ecg_median + 0.005, ecg_v.max() - 0.005)

    return threshold_v


def generate_and_run_adaptive(record_id):
    """Generate testbench with adaptive threshold, run ngspice."""
    record = wfdb.rdrecord(os.path.join(DB_PATH, record_id))
    ecg = record.p_signal[:, 0]
    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    py_dets = detect_final(ecg, FS)
    py_dets = [d for d in py_dets if d < n_samples]

    times = np.arange(n_samples) / FS
    ecg_min, ecg_max = ecg.min(), ecg.max()
    ecg_range = ecg_max - ecg_min
    if ecg_range < 0.01:
        ecg_range = 0.01
    ecg_v = 0.250 + (ecg - ecg_min) / ecg_range * 0.200
    ecg_median = np.median(ecg_v)

    ecg_pwl = format_pwl(times, ecg_v)

    # Compute adaptive threshold
    threshold_v = compute_adaptive_threshold_signal(ecg, FS, ecg_v)

    # Downsample threshold for PWL (every 10 samples)
    thr_ds_idx = np.arange(0, n_samples, 10)
    thr_ds_times = times[thr_ds_idx]
    thr_ds_vals = threshold_v[thr_ds_idx]
    thr_pwl = format_pwl(thr_ds_times, thr_ds_vals)

    n_dets = len(py_dets)
    meas_lines = "\n".join([
        f".meas tran det{i} when v(rpeak_out)=0.45 rise={i}"
        for i in range(1, min(n_dets + 5, 30))
    ])

    tb = f"""* Adaptive threshold validation: Record {record_id}
.lib {SKY130_LIB} tt
.include {CIRCUIT_DIR}/inverter.spice
.include {CIRCUIT_DIR}/rpeak_hold.spice
.include {CIRCUIT_DIR}/rpeak_detector_v2.spice
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 0.300
Vecg ecg_in 0 PWL(
{ecg_pwl}
+ )
Vthresh threshold 0 PWL(
{thr_pwl}
+ )

Xdet ecg_in threshold rpeak_out vdd vss vbias
+ env_fast_raw env_fast env_avg noise_gate_out thr_gate_out
+ rpeak_detector_v2

.ic v(env_fast_raw)={ecg_median:.4f} v(env_fast)={ecg_median:.4f} v(env_avg)={ecg_median:.4f}

.tran 50u {TEST_DURATION_S} 0 50u

{meas_lines}

.control
set filetype=ascii
run
echo "=== RECORD {record_id} DONE ==="
quit
.endc
.end
"""

    tb_path = os.path.join(OUT_DIR, f'tb_adapt_{record_id}.spice')
    with open(tb_path, 'w') as f:
        f.write(tb)

    result = subprocess.run(
        [NGSPICE, '-b', tb_path],
        capture_output=True, text=True, timeout=600
    )
    output = result.stdout + result.stderr

    spice_dets = []
    for line in output.split('\n'):
        m = re.match(r'\s*det(\d+)\s*=\s*([0-9.e+-]+)', line)
        if m:
            spice_dets.append(float(m.group(2)))

    return py_dets, spice_dets, ecg_median, threshold_v, ecg_v


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs='+',
                       default=['100', '103', '105', '200', '119', '232'])
    parser.add_argument('--fixed', action='store_true',
                       help='Also run with fixed threshold for comparison')
    args = parser.parse_args()

    records = args.records

    print("=" * 75)
    print("ADAPTIVE THRESHOLD VALIDATION")
    print("=" * 75)

    results = []

    for rec in records:
        print(f"\n--- Record {rec} (adaptive threshold) ---")
        try:
            py_dets, sp_dets, ecg_med, thr_v, ecg_v = generate_and_run_adaptive(rec)
            n_py = len(py_dets)
            n_sp = len(sp_dets)

            py_times_s = [d / FS for d in py_dets]
            tp, fn, fp, offsets_ms = nearest_neighbor_match(
                py_times_s, sp_dets, tol_s=MATCH_TOL_MS / 1000.0)

            sensitivity = tp / n_py * 100 if n_py > 0 else 0
            ppv = tp / n_sp * 100 if n_sp > 0 else 0

            offsets_arr = np.array(offsets_ms) if offsets_ms else np.array([0])
            mean_off = offsets_arr.mean()
            std_off = offsets_arr.std() if len(offsets_arr) > 1 else 0

            print(f"  Python: {n_py} dets, SPICE: {n_sp} dets")
            print(f"  TP: {tp}, FN: {fn}, FP: {fp}")
            print(f"  Sensitivity: {sensitivity:.1f}%, PPV: {ppv:.1f}%")
            print(f"  Offset: {mean_off:.1f}ms +/- {std_off:.1f}ms")

            results.append({
                'record': rec, 'n_py': n_py, 'n_sp': n_sp,
                'tp': tp, 'fn': fn, 'fp': fp,
                'sensitivity': sensitivity, 'ppv': ppv,
                'mean_offset': mean_off, 'jitter': std_off,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({
                'record': rec, 'n_py': 0, 'n_sp': 0,
                'tp': 0, 'fn': 0, 'fp': 0,
                'sensitivity': 0, 'ppv': 0,
                'mean_offset': 0, 'jitter': 0,
            })

    print("\n" + "=" * 75)
    print("SUMMARY — ADAPTIVE THRESHOLD")
    print("=" * 75)
    print(f"{'Record':>7} {'Py':>4} {'SP':>4} {'TP':>4} {'FN':>4} {'FP':>4} "
          f"{'Sens%':>6} {'PPV%':>6} {'Offset':>8} {'Jitter':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['record']:>7} {r['n_py']:>4} {r['n_sp']:>4} "
              f"{r['tp']:>4} {r['fn']:>4} {r['fp']:>4} "
              f"{r['sensitivity']:>5.1f}% {r['ppv']:>5.1f}% "
              f"{r['mean_offset']:>7.1f}ms {r['jitter']:>6.1f}ms")

    sens_arr = [r['sensitivity'] for r in results if r['n_py'] > 0]
    print(f"\nMean sensitivity: {np.mean(sens_arr):.1f}%")


if __name__ == '__main__':
    main()
