"""
Test rpeak_detector_v4 — the Python-faithful BPF+derivative detector.

Generates SPICE testbench, runs ngspice, and compares detections
with the Python reference detector.

Usage:
  python scratchpad/test_v4_detector.py --record 100
  python scratchpad/test_v4_detector.py --records 100 101 108
  python scratchpad/test_v4_detector.py --synth  (synthetic ramp test)
"""
import numpy as np
import subprocess
import os
import sys
import argparse
import re

sys.path.insert(0, 'scratchpad')
from analog_detector_final import detect_final

DB_PATH = 'mitdb_data'
FS = 360
TEST_DURATION_S = 10.0
MATCH_TOL_S = 0.150

NGSPICE = r'C:\Spice64\bin\ngspice_con.exe'
SKY130_LIB = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/ngspice_frontend/sky130_minimal.lib.spice'
CIRCUIT_DIR = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/complete_frontend_v9b'
OUT_DIR = 'scratchpad'

VBIAS = 0.40
VBASE = 0.30


def format_pwl(times, voltages, max_per_line=4):
    lines = []
    buf = []
    for t, v in zip(times, voltages):
        buf.append(f"{t:.7f} {v:.6f}")
        if len(buf) >= max_per_line:
            lines.append("+ " + " ".join(buf))
            buf = []
    if buf:
        lines.append("+ " + " ".join(buf))
    return "\n".join(lines)


def ecg_to_pwl(ecg, fs, vbase=VBASE, vscale=0.200):
    """Convert ECG to PWL voltage source centered around vbase."""
    ecg_min, ecg_max = ecg.min(), ecg.max()
    ecg_range = ecg_max - ecg_min
    if ecg_range < 1e-6:
        ecg_range = 1.0
    ecg_norm = (ecg - ecg_min) / ecg_range
    ecg_v = vbase + ecg_norm * vscale
    times = np.arange(len(ecg)) / fs
    return times, ecg_v


def generate_v4_testbench(record_id, ecg, fs=360, out_path=None):
    """Generate SPICE testbench for rpeak_detector_v4."""
    import wfdb

    n_samples = int(TEST_DURATION_S * fs)
    ecg = ecg[:n_samples]
    times, ecg_v = ecg_to_pwl(ecg, fs)

    # Downsample PWL: every sample at 360Hz (2.78ms spacing is fine for SPICE)
    pwl_str = format_pwl(times, ecg_v)

    ecg_median = np.median(ecg_v)
    sim_time = times[-1] + 1.0/fs

    if out_path is None:
        out_path = os.path.join(OUT_DIR, f'_tb_v4_{record_id}.spice')

    data_file = os.path.join(OUT_DIR, f'_v4_data_{record_id}.txt')
    data_file_fwd = data_file.replace('\\', '/')

    tb = f"""* V4 Detector Testbench — Record {record_id}
* BPF(5-15Hz) + derivative + half-wave rect + dual envelope

.lib {SKY130_LIB} tt
.include {CIRCUIT_DIR}/rpeak_detector_v4.spice
.include {CIRCUIT_DIR}/rpeak_hold.spice
.include {CIRCUIT_DIR}/inverter.spice
.include {CIRCUIT_DIR}/rpeak_detector_v2.spice
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

* Power supplies
Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 {VBIAS}

* ECG input
Vecg ecg_in 0 pwl(
{pwl_str}
+ )

* Instantiate v4 detector
Xdet ecg_in rpeak_out vdd vss vbias
+ v_hpf v_bpf rect_out env_fast_raw env_fast env_avg
+ peak_level adapt_thr noise_gate_out thr_gate_out
+ rpeak_detector_v4

* Initial conditions — pin nodes at top level
.ic v(v_hpf)={ecg_median:.4f}
.ic v(v_bpf)={ecg_median:.4f}
.ic v(xdet.xdirect.diff_node)={VBIAS:.4f}
.ic v(rect_out)={VBIAS:.4f}
.ic v(env_fast_raw)={VBASE:.4f}
.ic v(env_fast)={VBASE:.4f}
.ic v(env_avg)={VBASE:.4f}
.ic v(peak_level)={VBASE:.4f}
.ic v(adapt_thr)={VBASE:.4f}

* Save key signals
.save v(ecg_in) v(v_hpf) v(v_bpf)
+ v(xdet.xdirect.diff_node) v(rect_out)
+ v(env_fast_raw) v(env_fast) v(env_avg)
+ v(peak_level) v(adapt_thr)
+ v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

.tran 0.1m {sim_time:.4f} uic

* Write waveform data
.control
run
wrdata {data_file_fwd} v(ecg_in) v(v_hpf) v(v_bpf) v(xdet.xdirect.diff_node) v(rect_out) v(env_fast_raw) v(env_fast) v(env_avg) v(peak_level) v(adapt_thr) v(noise_gate_out) v(thr_gate_out) v(rpeak_out)
quit
.endc

.end
"""
    with open(out_path, 'w') as f:
        f.write(tb)

    return out_path, data_file


def run_ngspice(tb_path, timeout=600):
    """Run ngspice and return success/failure."""
    try:
        abs_tb = os.path.abspath(tb_path)
        result = subprocess.run(
            [NGSPICE, '-b', abs_tb],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            print(f"  ngspice FAILED (exit {result.returncode})")
            # Print last 20 lines of stderr
            for line in result.stderr.strip().split('\n')[-20:]:
                print(f"    {line}")
            return False
        # Check for convergence errors
        if 'singular matrix' in result.stdout.lower() or 'timestep too small' in result.stdout.lower():
            print("  WARNING: convergence issues detected")
        return True
    except subprocess.TimeoutExpired:
        print(f"  ngspice TIMED OUT after {timeout}s")
        return False


def parse_wrdata(data_file):
    """Parse ngspice wrdata output file."""
    if not os.path.exists(data_file):
        print(f"  Data file not found: {data_file}")
        return None

    data = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('#'):
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
                data.append(vals)
            except ValueError:
                continue

    if not data:
        print("  No data parsed from file")
        return None

    arr = np.array(data)
    print(f"  Parsed {arr.shape[0]} samples, {arr.shape[1]} columns")
    return arr


def detect_rpeak_from_spice(data, rpeak_col=13, threshold=0.45):
    """Extract R-peak detection times from SPICE rpeak_out signal."""
    times = data[:, 0]
    rpeak = data[:, rpeak_col]

    # Find rising edges
    detections = []
    was_high = rpeak[0] > threshold
    for i in range(1, len(rpeak)):
        is_high = rpeak[i] > threshold
        if is_high and not was_high:
            detections.append(times[i])
        was_high = is_high

    return detections


def nearest_neighbor_match(ref_times, det_times, tol_s=MATCH_TOL_S):
    """Match detections to reference beats."""
    ref_matched = set()
    offsets_ms = []
    for dt in det_times:
        best_dist = float('inf')
        best_ri = -1
        for ri, rt in enumerate(ref_times):
            if ri in ref_matched:
                continue
            dist = abs(dt - rt)
            if dist < best_dist:
                best_dist = dist
                best_ri = ri
        if best_ri >= 0 and best_dist <= tol_s:
            ref_matched.add(best_ri)
            offsets_ms.append((dt - ref_times[best_ri]) * 1000.0)
    tp = len(offsets_ms)
    fn = len(ref_times) - tp
    fp = len(det_times) - tp
    return tp, fn, fp, offsets_ms


def test_record(record_id):
    """Test v4 detector on one record."""
    import wfdb

    print(f"\n--- Record {record_id} ---")

    # Load ECG
    record = wfdb.rdrecord(os.path.join(DB_PATH, record_id))
    ecg = record.p_signal[:, 0]
    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    # Python reference detections
    py_dets = detect_final(ecg, FS)
    py_dets = [d for d in py_dets if d < n_samples]
    py_times = [d / FS for d in py_dets]
    print(f"  Python: {len(py_dets)} detections")

    # Generate and run SPICE testbench
    tb_path, data_file = generate_v4_testbench(record_id, ecg)
    print(f"  Running ngspice...")

    if not run_ngspice(tb_path, timeout=600):
        return None

    # Parse results
    data = parse_wrdata(data_file)
    if data is None:
        return None

    # Column mapping from wrdata:
    # 0=time, 1=ecg_in, 2=v_hpf, 3=v_bpf, 4=diff_node, 5=rect_out,
    # 6=env_fast_raw, 7=env_fast, 8=env_avg, 9=peak_level, 10=adapt_thr,
    # 11=noise_gate_out, 12=thr_gate_out, 13=rpeak_out

    # Print signal ranges for diagnostics
    col_names = ['time', 'ecg_in', 'v_hpf', 'v_bpf', 'diff_node', 'rect_out',
                 'env_fast_raw', 'env_fast', 'env_avg', 'peak_level', 'adapt_thr',
                 'noise_gate_out', 'thr_gate_out', 'rpeak_out']

    for i, name in enumerate(col_names[1:], 1):
        if i < data.shape[1]:
            vmin, vmax = data[:, i].min(), data[:, i].max()
            print(f"    {name:16s}: {vmin:.4f}V - {vmax:.4f}V (range={vmax-vmin:.4f}V)")

    # Detect R-peaks from SPICE output
    if data.shape[1] > 13:
        spice_times = detect_rpeak_from_spice(data)
        print(f"  SPICE: {len(spice_times)} detections")

        if len(py_times) > 0 and len(spice_times) > 0:
            tp, fn, fp, offsets = nearest_neighbor_match(py_times, spice_times)
            sens = tp / len(py_times) * 100
            ppv = tp / len(spice_times) * 100 if len(spice_times) > 0 else 0
            off_arr = np.array(offsets) if offsets else np.array([0])
            print(f"  TP={tp}, FN={fn}, FP={fp}")
            print(f"  Sensitivity: {sens:.1f}%, PPV: {ppv:.1f}%")
            print(f"  Offset: {off_arr.mean():.1f}ms +/- {off_arr.std():.1f}ms")

            return {
                'record': record_id, 'n_py': len(py_times), 'n_sp': len(spice_times),
                'tp': tp, 'fn': fn, 'fp': fp,
                'sens': sens, 'ppv': ppv,
                'offset': off_arr.mean(), 'jitter': off_arr.std()
            }
    else:
        print(f"  WARNING: only {data.shape[1]} columns (expected 14)")

    return None


def test_synthetic():
    """Test BPF + differentiator with synthetic ramp input."""
    print("\n=== SYNTHETIC RAMP TEST ===")
    print("Testing HPF + LPF + differentiator response to known slopes")

    # Create a synthetic ECG-like signal: flat + ramp up + flat + ramp down
    duration = 2.0  # seconds
    n = int(duration * FS)
    times = np.arange(n) / FS

    ecg = np.ones(n) * 0.0  # baseline
    # Three "beats": sharp up-down ramps at 0.5s, 1.0s, 1.5s
    for t_beat in [0.5, 1.0, 1.5]:
        for i in range(n):
            t = times[i]
            dt = t - t_beat
            if -0.015 <= dt < 0:   # 15ms upslope
                ecg[i] += (dt + 0.015) / 0.015 * 0.200  # 200mV peak, slope = 13.3 V/s
            elif 0 <= dt < 0.020:  # 20ms downslope
                ecg[i] += (1 - dt/0.020) * 0.200

    # Scale to voltage range
    ecg_v = VBASE + ecg
    pwl_str = format_pwl(times, ecg_v)

    tb_path = os.path.join(OUT_DIR, '_tb_v4_synth.spice')
    data_file = os.path.join(OUT_DIR, '_v4_data_synth.txt')
    data_file_fwd = data_file.replace('\\', '/')

    tb = f"""* V4 Synthetic Ramp Test — BPF + Differentiator
.lib {SKY130_LIB} tt
.include {CIRCUIT_DIR}/rpeak_detector_v4.spice
.include {CIRCUIT_DIR}/rpeak_hold.spice
.include {CIRCUIT_DIR}/inverter.spice
.include {CIRCUIT_DIR}/rpeak_detector_v2.spice
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 {VBIAS}

Vecg ecg_in 0 pwl(
{pwl_str}
+ )

Xdet ecg_in rpeak_out vdd vss vbias
+ v_hpf v_bpf rect_out env_fast_raw env_fast env_avg
+ peak_level adapt_thr noise_gate_out thr_gate_out
+ rpeak_detector_v4

.ic v(v_hpf)={VBASE:.4f}
.ic v(v_bpf)={VBASE:.4f}
.ic v(xdet.xdirect.diff_node)={VBIAS:.4f}
.ic v(rect_out)={VBIAS:.4f}
.ic v(env_fast_raw)={VBASE:.4f}
.ic v(env_fast)={VBASE:.4f}
.ic v(env_avg)={VBASE:.4f}
.ic v(peak_level)={VBASE:.4f}
.ic v(adapt_thr)={VBASE:.4f}

.save v(ecg_in) v(v_hpf) v(v_bpf)
+ v(xdet.xdirect.diff_node) v(rect_out)
+ v(env_fast_raw) v(env_fast) v(env_avg)
+ v(peak_level) v(adapt_thr)
+ v(noise_gate_out) v(thr_gate_out) v(rpeak_out)

.tran 0.05m {duration:.4f} uic

.control
run
wrdata {data_file_fwd} v(ecg_in) v(v_hpf) v(v_bpf) v(xdet.xdirect.diff_node) v(rect_out) v(env_fast_raw) v(env_fast) v(env_avg) v(peak_level) v(adapt_thr) v(noise_gate_out) v(thr_gate_out) v(rpeak_out)
quit
.endc

.end
"""
    with open(tb_path, 'w') as f:
        f.write(tb)

    print(f"  Running ngspice...")
    if not run_ngspice(tb_path, timeout=120):
        return

    data = parse_wrdata(data_file)
    if data is None:
        return

    col_names = ['time', 'ecg_in', 'v_hpf', 'v_bpf', 'diff_node', 'rect_out',
                 'env_fast_raw', 'env_fast', 'env_avg', 'peak_level', 'adapt_thr',
                 'noise_gate_out', 'thr_gate_out', 'rpeak_out']

    for i, name in enumerate(col_names[1:], 1):
        if i < data.shape[1]:
            vmin, vmax = data[:, i].min(), data[:, i].max()
            print(f"    {name:16s}: {vmin:.4f}V - {vmax:.4f}V (range={vmax-vmin:.4f}V)")

    # Check that BPF passes the ramp signals
    if data.shape[1] > 3:
        bpf_range = data[:, 3].max() - data[:, 3].min()
        print(f"\n  BPF output range: {bpf_range*1000:.1f}mV (expect >50mV)")

    # Check that differentiator produces peaks at ramp transitions
    if data.shape[1] > 4:
        diff_range = data[:, 4].max() - data[:, 4].min()
        print(f"  Diff node range:  {diff_range*1000:.1f}mV (expect >10mV)")

    # Check for detections
    if data.shape[1] > 13:
        spice_times = detect_rpeak_from_spice(data)
        print(f"\n  Detected {len(spice_times)} peaks (expected 3)")
        for t in spice_times:
            print(f"    t={t*1000:.1f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', default='100')
    parser.add_argument('--records', nargs='+')
    parser.add_argument('--synth', action='store_true')
    args = parser.parse_args()

    if args.synth:
        test_synthetic()
    elif args.records:
        print("=" * 70)
        print("V4 DETECTOR VALIDATION (BPF + Derivative)")
        print("=" * 70)
        results = []
        for rec in args.records:
            r = test_record(rec)
            if r:
                results.append(r)
        if results:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"{'Rec':>4} {'Py':>3} {'SP':>3} {'TP':>3} {'FN':>3} {'FP':>3} {'Sens':>6} {'PPV':>6} {'Offset':>8} {'Jitter':>7}")
            for r in results:
                print(f"{r['record']:>4} {r['n_py']:>3} {r['n_sp']:>3} "
                      f"{r['tp']:>3} {r['fn']:>3} {r['fp']:>3} "
                      f"{r['sens']:>5.1f}% {r['ppv']:>5.1f}% "
                      f"{r['offset']:>7.1f}ms {r['jitter']:>6.1f}ms")
            sens_arr = [r['sens'] for r in results]
            print(f"\nMean sensitivity: {np.mean(sens_arr):.1f}%")
    else:
        print("=" * 70)
        print("V4 DETECTOR TEST — Single Record")
        print("=" * 70)
        test_record(args.record)
