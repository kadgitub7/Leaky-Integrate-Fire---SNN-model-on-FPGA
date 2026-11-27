"""
Validate rpeak_detector_v3 (adaptive threshold) across MIT-BIH records.
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
SKY130_LIB = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/ngspice_frontend/sky130_minimal.lib.spice'
CIRCUIT_DIR = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/complete_frontend_v9b'
OUT_DIR = 'scratchpad'
NGSPICE = 'C:/Spice64/bin/ngspice_con.exe'

MATCH_TOL_MS = 150.0


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


def generate_and_run_v3(record_id):
    """Generate testbench for rpeak_detector_v3, run ngspice."""
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

    n_dets = len(py_dets)
    meas_lines = "\n".join([
        f".meas tran det{i} when v(rpeak_out)=0.45 rise={i}"
        for i in range(1, min(n_dets + 5, 30))
    ])

    tb = f"""* V3 adaptive threshold validation: Record {record_id}
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

* V3 detector — no external threshold needed
Xdet ecg_in rpeak_out vdd vss vbias
+ env_fast_raw env_fast env_avg peak_level adapt_thr
+ slope_gate_out thr_gate_out
+ rpeak_detector_v3

.ic v(env_fast_raw)={ecg_median:.4f} v(env_fast)={ecg_median:.4f}
+ v(env_avg)={ecg_median:.4f} v(peak_level)={ecg_median:.4f}
+ v(adapt_thr)={ecg_median:.4f}
+ v(xdet.xslope.delayed)={ecg_median:.4f}

.tran 50u {TEST_DURATION_S} 0 50u

{meas_lines}

.control
set filetype=ascii
run
echo "=== RECORD {record_id} V3 DONE ==="
quit
.endc
.end
"""

    tb_path = os.path.join(OUT_DIR, f'tb_v3_{record_id}.spice')
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

    # Check for errors
    errors = [l for l in output.split('\n') if 'error' in l.lower() or 'fatal' in l.lower()]
    if errors and not spice_dets:
        print(f"  SPICE errors: {errors[:3]}")

    return py_dets, spice_dets, ecg_median, ecg_v


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs='+',
                       default=['100', '103', '105'])
    args = parser.parse_args()

    records = args.records

    print("=" * 75)
    print("V3 DETECTOR VALIDATION (Analog Adaptive Threshold)")
    print("=" * 75)

    results = []

    for rec in records:
        print(f"\n--- Record {rec} ---")
        try:
            py_dets, sp_dets, ecg_med, ecg_v = generate_and_run_v3(rec)
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
    print("SUMMARY — V3 ADAPTIVE THRESHOLD DETECTOR")
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
    ppv_arr = [r['ppv'] for r in results if r['n_sp'] > 0]
    print(f"\nMean sensitivity: {np.mean(sens_arr):.1f}%")
    print(f"Mean PPV: {np.mean(ppv_arr):.1f}%")


if __name__ == '__main__':
    main()
