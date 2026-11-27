"""
Diagnostic testbench for v3 detector on a specific record.
Saves all intermediate waveforms for analysis.
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


def run_diagnostic(record_id):
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

    tb = f"""* Diagnostic: Record {record_id} with v3 detector
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

.save v(ecg_in) v(env_fast) v(env_avg) v(peak_level) v(adapt_thr)
+ v(slope_gate_out) v(thr_gate_out) v(rpeak_out)

.control
set filetype=ascii
run
echo "=== DIAG RECORD {record_id} ==="
wrdata {OUT_DIR}/diag_{record_id}.txt v(ecg_in) v(env_fast) v(env_avg) v(peak_level) v(adapt_thr) v(slope_gate_out) v(thr_gate_out) v(rpeak_out)
quit
.endc
.end
"""

    tb_path = os.path.join(OUT_DIR, f'tb_diag_{record_id}.spice')
    with open(tb_path, 'w') as f:
        f.write(tb)

    print(f"Running SPICE for Record {record_id}...")
    result = subprocess.run(
        [NGSPICE, '-b', tb_path],
        capture_output=True, text=True, timeout=300
    )
    output = result.stdout + result.stderr

    spice_dets = []
    for line in output.split('\n'):
        m = re.match(r'\s*det(\d+)\s*=\s*([0-9.e+-]+)', line)
        if m:
            spice_dets.append(float(m.group(2)))

    print(f"Python dets: {len(py_dets)}, SPICE dets: {len(spice_dets)}")
    print(f"Python det times (s): {[f'{d/FS:.3f}' for d in py_dets]}")
    print(f"SPICE det times (s):  {[f'{t:.3f}' for t in spice_dets]}")

    # Read and analyze the waveform data
    diag_file = os.path.join(OUT_DIR, f'diag_{record_id}.txt')
    if os.path.exists(diag_file):
        data = np.loadtxt(diag_file)
        t = data[:, 0]
        ecg_sp = data[:, 1]
        env_fast_sp = data[:, 3]
        env_avg_sp = data[:, 5]
        peak_lvl_sp = data[:, 7]
        adapt_thr_sp = data[:, 9]
        ng_out = data[:, 11]
        thr_out = data[:, 13]
        rpeak_sp = data[:, 15]

        # Sample at beat times
        for i, d in enumerate(py_dets):
            t_beat = d / FS
            idx = np.argmin(np.abs(t - t_beat))
            print(f"  Beat {i+1} at {t_beat:.3f}s: "
                  f"ecg={ecg_sp[idx]:.4f} env_fast={env_fast_sp[idx]:.4f} "
                  f"env_avg={env_avg_sp[idx]:.4f} peak_lvl={peak_lvl_sp[idx]:.4f} "
                  f"adapt_thr={adapt_thr_sp[idx]:.4f} "
                  f"ng={ng_out[idx]:.2f} thr={thr_out[idx]:.2f} rpk={rpeak_sp[idx]:.2f}")

        # Check env_fast range
        print(f"\nenv_fast range: {env_fast_sp.min():.4f} - {env_fast_sp.max():.4f}")
        print(f"env_avg range: {env_avg_sp.min():.4f} - {env_avg_sp.max():.4f}")
        print(f"peak_level range: {peak_lvl_sp.min():.4f} - {peak_lvl_sp.max():.4f}")
        print(f"adapt_thr range: {adapt_thr_sp.min():.4f} - {adapt_thr_sp.max():.4f}")

    # ECG peak voltages
    print(f"\nECG peak voltages at Python beat times:")
    for i, d in enumerate(py_dets):
        print(f"  Beat {i+1}: ecg_v={ecg_v[d]:.4f}V")

    print(f"\nECG median: {ecg_median:.4f}V")
    print(f"ECG range: {ecg_v.min():.4f} - {ecg_v.max():.4f}V")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('record', default='112')
    args = parser.parse_args()
    run_diagnostic(args.record)
