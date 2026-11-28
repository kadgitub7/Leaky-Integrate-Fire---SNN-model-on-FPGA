"""
Test rpeak_detector_v5 on MIT-BIH records.

Usage:
  python scratchpad/test_v5_mitbih.py --record 100
  python scratchpad/test_v5_mitbih.py --records 100 108 200 207
"""
import numpy as np
import subprocess
import os
import sys
import argparse

sys.path.insert(0, 'scratchpad')

DB_PATH = 'mitdb_data'
FS = 360
TEST_DURATION_S = 10.0
MATCH_TOL_S = 0.150

NGSPICE = r'C:\Spice64\bin\ngspice.exe'
SKY130_LIB = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/ngspice_frontend/sky130_minimal.lib.spice'
CIRCUIT_FILE = 'C:/Users/kadhi/OneDrive/Desktop/amux/verilogLearning/Leaky-Integrate-Fire---SNN-model-on-FPGA/experiments/frontend_optimization/complete_frontend_v9b/rpeak_detector_v5.spice'
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
    ecg_min, ecg_max = ecg.min(), ecg.max()
    ecg_range = ecg_max - ecg_min
    if ecg_range < 1e-6:
        ecg_range = 1.0
    ecg_norm = (ecg - ecg_min) / ecg_range
    ecg_v = vbase + ecg_norm * vscale
    times = np.arange(len(ecg)) / fs
    return times, ecg_v


def generate_testbench(record_id, ecg, fs=360):
    n_samples = int(TEST_DURATION_S * fs)
    ecg = ecg[:n_samples]
    times, ecg_v = ecg_to_pwl(ecg, fs)
    pwl_str = format_pwl(times, ecg_v)
    sim_time = times[-1] + 1.0/fs

    tb_path = os.path.join(OUT_DIR, f'_tb_v5_{record_id}.spice')
    data_file = os.path.join(OUT_DIR, f'_v5_data_{record_id}.txt')
    data_file_fwd = data_file.replace('\\', '/')

    tb = f"""* V5 Detector Test — MIT-BIH Record {record_id}
.lib {SKY130_LIB} tt
.include {CIRCUIT_FILE}
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15

Vdd vdd 0 0.9
Vss vss 0 0.0
Vbias vbias 0 {VBIAS}

Vecg ecg_in 0 pwl(
{pwl_str}
+ )

Xdet ecg_in rpeak_out vdd vss vbias
+ v_hpf v_bpf diff_out mwi_out slow_avg noise_gate_out
+ rpeak_detector_v5

.ic v(v_hpf)=0.40
.ic v(v_bpf)=0.40
.ic v(diff_out)=0.40
.ic v(mwi_out)=0.40
.ic v(slow_avg)=0.40

.save v(ecg_in) v(rpeak_out) v(mwi_out) v(slow_avg)
+ v(noise_gate_out) v(diff_out)

.tran 0.1m {sim_time:.4f} uic

.control
run
wrdata {data_file_fwd} v(ecg_in) v(rpeak_out) v(mwi_out) v(slow_avg) v(noise_gate_out) v(diff_out)
quit
.endc

.end
"""
    with open(tb_path, 'w') as f:
        f.write(tb)
    return tb_path, data_file


def run_ngspice(tb_path, timeout=1200):
    try:
        result = subprocess.run(
            [NGSPICE, '-b', os.path.abspath(tb_path)],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ngspice TIMED OUT after {timeout}s")
        return False


def parse_wrdata(data_file):
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
        return None
    return np.array(data)


def detect_from_spice(data, threshold=0.45):
    time = data[:, 0]
    rpeak_col = 2*1 + 1  # rpeak_out = signal index 1 -> col 3
    if rpeak_col >= data.shape[1]:
        print(f"  Not enough columns: {data.shape[1]}")
        return [], time
    rpeak = data[:, rpeak_col]
    detections = []
    was_high = rpeak[0] > threshold
    for i in range(1, len(rpeak)):
        is_high = rpeak[i] > threshold
        if is_high and not was_high:
            detections.append(time[i])
        was_high = is_high
    return detections, time


def match_detections(ref_times, det_times, tol_s=MATCH_TOL_S):
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
                best_dist, best_ri = dist, ri
        if best_ri >= 0 and best_dist <= tol_s:
            ref_matched.add(best_ri)
            offsets_ms.append((dt - ref_times[best_ri]) * 1000.0)
    tp = len(offsets_ms)
    fn = len(ref_times) - tp
    fp = len(det_times) - tp
    return tp, fn, fp, offsets_ms


def test_record(record_id, duration_s=TEST_DURATION_S):
    import wfdb

    print(f"\n{'='*60}")
    print(f"Record {record_id} ({duration_s:.0f}s)")
    print(f"{'='*60}")

    record = wfdb.rdrecord(os.path.join(DB_PATH, record_id))
    ann = wfdb.rdann(os.path.join(DB_PATH, record_id), 'atr')
    ecg = record.p_signal[:, 0]
    n_samples = int(duration_s * FS)
    ecg = ecg[:n_samples]

    beat_types = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
                  'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    ann_samples = [s for s, sym in zip(ann.sample, ann.symbol)
                   if s < n_samples and sym in beat_types]
    ann_times = [s / FS for s in ann_samples]
    print(f"  WFDB annotations: {len(ann_times)} beats")

    print(f"  Generating testbench...")
    tb_path, data_file = generate_testbench(record_id, ecg)
    print(f"  Running ngspice ({duration_s:.0f}s sim)...")

    if not run_ngspice(tb_path, timeout=1200):
        print("  FAILED: ngspice error")
        return None

    data = parse_wrdata(data_file)
    if data is None:
        print("  FAILED: no data")
        return None

    print(f"  Parsed {data.shape[0]} samples, {data.shape[1]} columns")

    sig_names = ['ecg_in', 'rpeak_out', 'mwi_out', 'slow_avg',
                 'noise_gate_out', 'diff_out']
    time = data[:, 0]
    print(f"\n  {'Signal':<18} {'Min':>8} {'Max':>8} {'Range(mV)':>10}")
    print(f"  {'-'*48}")
    for i, name in enumerate(sig_names):
        col = 2*i + 1
        if col < data.shape[1]:
            v = data[:, col]
            print(f"  {name:<18} {v.min():>8.4f} {v.max():>8.4f} {(v.max()-v.min())*1000:>10.1f}")

    spice_times, _ = detect_from_spice(data)
    spice_times = [t for t in spice_times if t > 2.0]
    ann_times_filt = [t for t in ann_times if t > 2.0]

    print(f"\n  SPICE detections (after 2s settling): {len(spice_times)}")

    tp, fn, fp, offsets = match_detections(ann_times_filt, spice_times)
    n_ref = len(ann_times_filt)
    sens = tp / n_ref * 100 if n_ref > 0 else 0
    ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    off_arr = np.array(offsets) if offsets else np.array([0])

    print(f"\n  vs WFDB ({n_ref} beats after 2s):")
    print(f"    TP={tp}, FN={fn}, FP={fp}")
    print(f"    Sensitivity: {sens:.1f}%")
    print(f"    PPV: {ppv:.1f}%")
    if len(offsets) > 0:
        print(f"    Timing: {off_arr.mean():.1f} +/- {off_arr.std():.1f} ms")

    # Detail: list all detections vs annotations
    print(f"\n  Detection detail:")
    ref_matched = set()
    for dt in spice_times:
        best_dist, best_ri = float('inf'), -1
        for ri, rt in enumerate(ann_times_filt):
            if ri in ref_matched:
                continue
            d = abs(dt - rt)
            if d < best_dist:
                best_dist, best_ri = d, ri
        if best_ri >= 0 and best_dist <= MATCH_TOL_S:
            ref_matched.add(best_ri)
            # Find MWI value
            idx = np.argmin(np.abs(time - dt))
            mwi_val = data[idx, 5] if 5 < data.shape[1] else 0  # mwi_out
            savg_val = data[idx, 7] if 7 < data.shape[1] else 0  # slow_avg
            print(f"    TP  t={dt:.3f}s (ann={ann_times_filt[best_ri]:.3f}s, dt={best_dist*1000:.0f}ms) MWI-base={(mwi_val-savg_val)*1000:.1f}mV")
        else:
            idx = np.argmin(np.abs(time - dt))
            mwi_val = data[idx, 5] if 5 < data.shape[1] else 0
            savg_val = data[idx, 7] if 7 < data.shape[1] else 0
            print(f"    FP  t={dt:.3f}s MWI-base={(mwi_val-savg_val)*1000:.1f}mV")

    # Missed beats
    for ri, rt in enumerate(ann_times_filt):
        if ri not in ref_matched:
            print(f"    FN  ann={rt:.3f}s (missed)")

    result = {
        'record': record_id, 'n_wfdb': n_ref, 'n_spice': len(spice_times),
        'tp': tp, 'fn': fn, 'fp': fp,
        'sens': sens, 'ppv': ppv,
    }

    status = "PASS" if sens >= 99.5 and ppv >= 99.5 else "NEEDS WORK" if sens >= 90 else "FAIL"
    print(f"\n  >>> {status}: Sens={sens:.1f}%, PPV={ppv:.1f}% <<<")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=str, help='Single record')
    parser.add_argument('--records', nargs='+', help='Multiple records')
    parser.add_argument('--duration', type=float, default=TEST_DURATION_S)
    args = parser.parse_args()

    if args.duration:
        TEST_DURATION_S = args.duration

    records = []
    if args.record:
        records = [args.record]
    elif args.records:
        records = args.records
    else:
        records = ['100']

    results = []
    for rec in records:
        r = test_record(rec, TEST_DURATION_S)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Record':>8} {'Beats':>6} {'Dets':>6} {'TP':>5} {'FN':>5} {'FP':>5} {'Sens%':>7} {'PPV%':>7}")
        for r in results:
            print(f"{r['record']:>8} {r['n_wfdb']:>6} {r['n_spice']:>6} {r['tp']:>5} {r['fn']:>5} {r['fp']:>5} {r['sens']:>7.1f} {r['ppv']:>7.1f}")
        avg_sens = np.mean([r['sens'] for r in results])
        avg_ppv = np.mean([r['ppv'] for r in results])
        print(f"{'AVG':>8} {'':>6} {'':>6} {'':>5} {'':>5} {'':>5} {avg_sens:>7.1f} {avg_ppv:>7.1f}")
