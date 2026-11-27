"""
Multi-record validation of the analog R-peak detector.
Generates SPICE testbenches for multiple MIT-BIH records,
runs them, and compares detection times against Python reference.

Uses nearest-neighbor matching for proper offset/jitter computation
even when beats are missed.
"""
import numpy as np
import wfdb
import os
import sys
import subprocess
import re

sys.path.insert(0, 'scratchpad')
from analog_detector_final import detect_final
from generate_spice_tb import format_pwl

DB_PATH = 'mitdb_data'
FS = 360
TEST_DURATION_S = 10.0
VBASE = 0.300
SKY130_LIB = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/ngspice_frontend/sky130_minimal.lib.spice'
CIRCUIT_DIR = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/complete_frontend_v9b'
OUT_DIR = 'scratchpad'
NGSPICE = 'C:/Spice64/bin/ngspice_con.exe'

RECORDS = ['100', '101', '103', '105', '112']

MATCH_TOL_MS = 150.0


def nearest_neighbor_match(py_times_s, sp_times_s, tol_s=0.150):
    """Match SPICE detections to Python detections using nearest-neighbor.

    Returns:
        tp: true positives (matched pairs)
        fn: false negatives (Python dets with no SPICE match)
        fp: false positives (SPICE dets with no Python match)
        offsets_ms: list of (spice - python) offsets in ms for matched pairs
    """
    py_matched = set()
    sp_matched = set()
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
            sp_matched.add(si)
            offsets_ms.append((st - py_times_s[best_pi]) * 1000.0)

    tp = len(offsets_ms)
    fn = len(py_times_s) - tp
    fp = len(sp_times_s) - tp
    return tp, fn, fp, offsets_ms


def generate_and_run(record_id):
    """Generate testbench, run ngspice, return (python_dets, spice_dets)."""
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

    threshold_v = ecg_median + 0.25 * (ecg_v.max() - ecg_median)

    n_dets = len(py_dets)
    meas_lines = "\n".join([
        f".meas tran det{i} when v(rpeak_out)=0.45 rise={i}"
        for i in range(1, min(n_dets + 5, 30))
    ])

    tb = f"""* Multi-record validation: Record {record_id}
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
Vthresh threshold 0 {threshold_v}

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

    tb_path = os.path.join(OUT_DIR, f'tb_rec{record_id}.spice')
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


def analyze_record_python(record_id):
    """Analyze a record with just the Python detector to understand signal characteristics."""
    record = wfdb.rdrecord(os.path.join(DB_PATH, record_id))
    ecg = record.p_signal[:, 0]
    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    py_dets = detect_final(ecg, FS)
    py_dets = [d for d in py_dets if d < n_samples]

    ecg_min, ecg_max = ecg.min(), ecg.max()
    ecg_range = ecg_max - ecg_min if (ecg_max - ecg_min) > 0.01 else 0.01
    ecg_v = 0.250 + (ecg - ecg_min) / ecg_range * 0.200
    ecg_median = np.median(ecg_v)
    threshold_v = ecg_median + 0.25 * (ecg_v.max() - ecg_median)

    peak_voltages = [ecg_v[d] for d in py_dets]
    rr_intervals = [(py_dets[i+1] - py_dets[i]) / FS * 1000 for i in range(len(py_dets)-1)]

    below_thr = sum(1 for pv in peak_voltages if pv < threshold_v)

    return {
        'record': record_id,
        'n_dets': len(py_dets),
        'ecg_median': ecg_median,
        'threshold_v': threshold_v,
        'peak_min': min(peak_voltages) if peak_voltages else 0,
        'peak_max': max(peak_voltages) if peak_voltages else 0,
        'peak_mean': np.mean(peak_voltages) if peak_voltages else 0,
        'below_threshold': below_thr,
        'rr_min': min(rr_intervals) if rr_intervals else 0,
        'rr_max': max(rr_intervals) if rr_intervals else 0,
        'rr_mean': np.mean(rr_intervals) if rr_intervals else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--python-only', action='store_true',
                       help='Only run Python analysis, skip SPICE')
    parser.add_argument('--records', nargs='+', default=RECORDS,
                       help='Records to test')
    args = parser.parse_args()

    records = args.records

    if args.python_only:
        print("=" * 70)
        print("PYTHON-ONLY ANALYSIS: Signal Characteristics per Record")
        print("=" * 70)
        print(f"{'Rec':>5} {'Dets':>4} {'Median':>7} {'Thresh':>7} "
              f"{'PkMin':>7} {'PkMax':>7} {'<Thr':>4} {'RR_min':>7} {'RR_max':>7}")
        print("-" * 70)
        for rec in records:
            try:
                info = analyze_record_python(rec)
                print(f"{info['record']:>5} {info['n_dets']:>4} "
                      f"{info['ecg_median']:.4f} {info['threshold_v']:.4f} "
                      f"{info['peak_min']:.4f} {info['peak_max']:.4f} "
                      f"{info['below_threshold']:>4} "
                      f"{info['rr_min']:>6.0f}ms {info['rr_max']:>6.0f}ms")
            except Exception as e:
                print(f"{rec:>5} ERROR: {e}")
        return

    print("=" * 70)
    print("MULTI-RECORD VALIDATION: Analog R-Peak Detector")
    print("=" * 70)

    results = []

    for rec in records:
        print(f"\n--- Record {rec} ---")
        try:
            py_dets, sp_dets, ecg_med, thr, ecg_v = generate_and_run(rec)
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

            peak_voltages = [ecg_v[d] for d in py_dets]
            below_thr = sum(1 for pv in peak_voltages if pv < thr)

            print(f"  Python: {n_py} dets, SPICE: {n_sp} dets")
            print(f"  TP: {tp}, FN: {fn}, FP: {fp}")
            print(f"  Sensitivity: {sensitivity:.1f}%, PPV: {ppv:.1f}%")
            print(f"  Offset: {mean_off:.1f}ms +/- {std_off:.1f}ms")
            print(f"  ECG median: {ecg_med:.4f}V, Threshold: {thr:.4f}V")
            print(f"  Peaks below fixed threshold: {below_thr}/{n_py}")

            results.append({
                'record': rec, 'n_py': n_py, 'n_sp': n_sp,
                'tp': tp, 'fn': fn, 'fp': fp,
                'sensitivity': sensitivity, 'ppv': ppv,
                'mean_offset': mean_off, 'jitter': std_off,
                'below_thr': below_thr
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({
                'record': rec, 'n_py': 0, 'n_sp': 0,
                'tp': 0, 'fn': 0, 'fp': 0,
                'sensitivity': 0, 'ppv': 0,
                'mean_offset': 0, 'jitter': 0,
                'below_thr': 0
            })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Record':>7} {'Py':>4} {'SP':>4} {'TP':>4} {'FN':>4} {'FP':>4} "
          f"{'Sens%':>6} {'PPV%':>6} {'Offset':>8} {'Jitter':>7} {'<Thr':>4}")
    print("-" * 75)
    for r in results:
        print(f"{r['record']:>7} {r['n_py']:>4} {r['n_sp']:>4} "
              f"{r['tp']:>4} {r['fn']:>4} {r['fp']:>4} "
              f"{r['sensitivity']:>5.1f}% {r['ppv']:>5.1f}% "
              f"{r['mean_offset']:>7.1f}ms {r['jitter']:>6.1f}ms "
              f"{r['below_thr']:>4}")

    sens_arr = [r['sensitivity'] for r in results if r['n_py'] > 0]
    print(f"\nMean sensitivity: {np.mean(sens_arr):.1f}%")
    print(f"Records with >=95% sensitivity: {sum(1 for s in sens_arr if s >= 95)}/{len(sens_arr)}")


if __name__ == '__main__':
    main()
