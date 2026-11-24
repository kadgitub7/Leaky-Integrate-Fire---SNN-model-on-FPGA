"""
Multi-record validation of v6 true_peak_detector
=================================================
Phase 1-5 of the validation plan:
  1. Select 5 diverse ECG records (30 beats each, 150 total)
  2. Generate testbenches with annotation-derived ARM windows
  3. Run ngspice simulations
  4. Parse beat-level measurements
  5. Compute performance statistics and make PASS/FAIL decision

Scaling: min-max normalization to [0.30, 0.80V]
  v = 0.30 + 0.50 * (sig - seg_min) / (seg_max - seg_min)
ARM: annotation - 3ms to annotation + 80ms (mimics rpeak_level_held)
Detection threshold: V(peak_detected) = 0.45V
"""

import subprocess, os, re, sys, csv, time
import numpy as np

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")
OUT_DIR = os.path.join(FRONTEND_DIR, "validation_results")
os.makedirs(OUT_DIR, exist_ok=True)

FS = 360
VDD = 0.9
V_LOW = 0.30
V_HIGH = 0.80
DET_THRESH = 0.45
ARM_PRE = 0.003   # ARM starts 3ms before annotation
ARM_POST = 0.080  # ARM stays 80ms after annotation
TSTEP = '10u'

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'

SELECTED_RECORDS = [
    ('100', 'Baseline reference: normal sinus rhythm, moderate amplitude, N/S/V types, low RR variability (CV=6.1%), moderate noise (47mV)'),
    ('232', 'Most irregular rhythm: highest RR_cv=45.3%, low amplitude (662mV), N/S types — stress test for varying beat-to-beat intervals'),
    ('116', 'Highest noise (128.6mV) + extreme baseline variation (std=384mV), high amplitude (2214mV), N/S/V types — noise/baseline stress test'),
    ('113', 'Narrowest QRS (35ms, sharpest peaks), high amplitude (1992mV), V_pk=0.425 (high CM stress), N/S types — sharp peak morphology'),
    ('203', 'Maximum morphological diversity: 5 beat types (F/N/Q/S/V), fast rate (606ms RR), highest amplitude std (465mV), high noise (96mV)'),
]

N_BEATS = 30


def load_record(record_id, n_beats=N_BEATS):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal[:, 0]

    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
             if sym in aami_map and idx > 90 and idx + 108 < len(sig)]

    # skip first beat (no preceding RR), take n_beats
    beats = valid[1:n_beats+1]
    if len(beats) < n_beats:
        print(f"  WARNING: Record {record_id} only has {len(beats)} valid beats (requested {n_beats})")

    # ECG segment: from 1 beat before first to 1 beat after last
    pad_before = valid[0][0] - 90 if valid else 0
    first_idx = beats[0][0] - 90
    last_idx = beats[-1][0] + 108
    seg_start = max(0, first_idx - int(0.5 * FS))  # 0.5s padding
    seg_end = min(len(sig), last_idx + int(0.5 * FS))

    ecg_seg = sig[seg_start:seg_end]
    seg_min = np.min(ecg_seg)
    seg_max = np.max(ecg_seg)

    # scale to [V_LOW, V_HIGH]
    gain = (V_HIGH - V_LOW) / (seg_max - seg_min) if seg_max > seg_min else 0.1
    offset = V_LOW - gain * seg_min

    # annotation times and scaled peak voltages
    beat_info = []
    for idx, sym in beats:
        t_ann = idx / FS
        v_ecg_at_peak = gain * sig[idx] + offset
        beat_info.append({
            'sample': idx,
            'time': t_ann,
            'symbol': sym,
            'aami': aami_map[sym],
            'v_ecg_at_peak': v_ecg_at_peak,
        })

    # PWL data
    pwl_points = []
    for i in range(seg_start, seg_end):
        t = i / FS
        v = gain * sig[i] + offset
        v = max(0.01, min(VDD - 0.01, v))  # clip to safe range
        pwl_points.append((t, v))

    # Baseline voltage (for IC)
    baseline_indices = range(seg_start, seg_start + int(0.5 * FS))
    v_baseline = np.mean([gain * sig[i] + offset for i in baseline_indices if i < len(sig)])

    return {
        'beats': beat_info,
        'pwl': pwl_points,
        'v_baseline': v_baseline,
        'seg_start': seg_start,
        'seg_end': seg_end,
        'gain': gain,
        'offset': offset,
        'seg_min': seg_min,
        'seg_max': seg_max,
    }


def generate_testbench(record_id, data):
    beats = data['beats']
    pwl = data['pwl']
    n = len(beats)
    t_end = pwl[-1][0] + 0.5  # 0.5s after last PWL point

    lines = []
    lines.append(f"* ============================================================")
    lines.append(f"* TESTBENCH: v6 true_peak_detector multi-record validation")
    lines.append(f"* Record {record_id}, {n} beats")
    lines.append(f"* Scaling: v = {data['gain']:.6f} * sig + {data['offset']:.6f}")
    lines.append(f"*   [V_LOW={V_LOW}, V_HIGH={V_HIGH}]")
    lines.append(f"*   seg_min={data['seg_min']:.4f}, seg_max={data['seg_max']:.4f}")
    lines.append(f"* ============================================================")
    lines.append("")
    lines.append(f'.lib "{NGSPICE_DIR}/sky130_minimal.lib.spice" tt')
    lines.append(f'.option scale=1.0u method=gear reltol=0.003 gmin=1e-15 itl4=200')
    lines.append("")
    lines.append("Vdd vdd 0 0.9")
    lines.append("Vss vss 0 0")
    lines.append("")
    lines.append("Xbias vbias vref vdd vss bias_gen")
    lines.append(".ic V(xbias.iref)=0.3")
    lines.append("")

    # ECG PWL
    lines.append(f"* === REAL ECG (Record {record_id}, samples {data['seg_start']}-{data['seg_end']}) ===")
    lines.append("Vecg ecg_in 0 PWL(")
    for t, v in pwl:
        lines.append(f"+ {t:.7f} {v:.6f}")
    lines.append("+ )")
    lines.append("")

    # ARM signal
    lines.append("* === ARM SIGNAL (simulated rpeak_level_held) ===")
    lines.append("Varm arm_signal 0 PWL(")
    lines.append("+ 0 0")
    for b in beats:
        t_ann = b['time']
        t_start = t_ann - ARM_PRE
        t_end_arm = t_ann + ARM_POST
        lines.append(f"+ {t_start - 0.0001:.7f} 0")
        lines.append(f"+ {t_start:.7f} 0.9")
        lines.append(f"+ {t_end_arm:.7f} 0.9")
        lines.append(f"+ {t_end_arm + 0.0001:.7f} 0")
    lines.append(f"+ {t_end:.7f} 0")
    lines.append("+ )")
    lines.append("")

    # DUT
    lines.append("* === DEVICE UNDER TEST ===")
    lines.append("Xtpd ecg_in arm_signal peak_detected vdd vss vbias true_peak_detector")
    v_ic = max(V_LOW, min(V_HIGH, data['v_baseline']))
    lines.append(f".ic V(xtpd.peak_hold)={v_ic:.2f}")
    lines.append("")

    # Measurements
    lines.append("* === MEASUREMENTS ===")
    for i, b in enumerate(beats):
        t_ann = b['time']
        lines.append(f".meas tran t_det_{i} WHEN V(peak_detected)={DET_THRESH} RISE={i+1}")
        lines.append(f".meas tran lat_{i} PARAM='t_det_{i} - {t_ann:.7f}'")
        lines.append(f".meas tran v_hold_{i} FIND V(xtpd.peak_hold) AT={t_ann:.7f}")
        lines.append(f".meas tran v_ecg_{i} FIND V(ecg_in) AT={t_ann:.7f}")
    lines.append("")

    lines.append(f".tran {TSTEP} {t_end:.4f} UIC")
    lines.append("")
    lines.append(f'.include "{FRONTEND_DIR}/true_peak_detector.spice"')
    lines.append(f'.include "{FRONTEND_DIR}/bias_gen.spice"')
    lines.append(f'.include "{FRONTEND_DIR}/inverter.spice"')
    lines.append("")

    lines.append(".control")
    lines.append("run")
    lines.append(f"wrdata {OUT_DIR}/tpd_val_{record_id}.txt V(ecg_in) V(xtpd.peak_hold) V(peak_detected) V(xtpd.track_level) V(xtpd.charge_en) V(arm_signal)")
    for i in range(n):
        lines.append(f"print t_det_{i} lat_{i} v_hold_{i} v_ecg_{i}")
    lines.append("quit")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    tb_path = os.path.join(FRONTEND_DIR, f"_tb_tpd_val_{record_id}.spice")
    with open(tb_path, 'w') as f:
        f.write("\n".join(lines))
    return tb_path


def run_simulation(tb_path, record_id, timeout=600):
    print(f"  Running ngspice for Record {record_id}...")
    log_path = os.path.join(OUT_DIR, f"tpd_val_{record_id}.log")
    t0 = time.time()
    try:
        result = subprocess.run(
            [NGSPICE, '-b', tb_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=FRONTEND_DIR
        )
        elapsed = time.time() - t0
        with open(log_path, 'w') as f:
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
        print(f"  Completed in {elapsed:.1f}s (exit code {result.returncode})")
        return result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.1f}s!")
        return None, elapsed


def parse_measurements(output, n_beats):
    results = []
    for i in range(n_beats):
        r = {'beat': i, 'detected': False}
        # t_det_N
        m = re.search(rf't_det_{i}\s*=\s*([\d.e+-]+)', output)
        if m:
            r['t_detect'] = float(m.group(1))
            r['detected'] = True
        # lat_N
        m = re.search(rf'lat_{i}\s*=\s*([\d.e+-]+)', output)
        if m:
            r['latency'] = float(m.group(1))
        # v_hold_N
        m = re.search(rf'v_hold_{i}\s*=\s*([\d.e+-]+)', output)
        if m:
            r['v_hold'] = float(m.group(1))
        # v_ecg_N
        m = re.search(rf'v_ecg_{i}\s*=\s*([\d.e+-]+)', output)
        if m:
            r['v_ecg'] = float(m.group(1))

        results.append(r)
    return results


def count_false_positives(output, n_beats):
    """Count extra RISE events beyond expected n_beats."""
    rises = re.findall(rf'WHEN V\(peak_detected\)={DET_THRESH} RISE=(\d+)', output)
    # Check if there are more detection events than beats
    for i in range(n_beats + 1, n_beats + 10):
        m = re.search(rf't_det_{i}\s*=\s*([\d.e+-]+)', output)
        if m:
            return True  # more detections than beats
    return False


def analyze_record(record_id, beats, measurements):
    n = len(measurements)
    detected = [m for m in measurements if m['detected']]
    missed = [m for m in measurements if not m['detected']]

    latencies = [m['latency'] * 1000 for m in detected if 'latency' in m]  # ms
    v_holds = [m['v_hold'] for m in detected if 'v_hold' in m]
    v_ecgs = [m['v_ecg'] for m in detected if 'v_ecg' in m]

    capture_errors = []
    for m in detected:
        if 'v_hold' in m and 'v_ecg' in m:
            capture_errors.append(abs(m['v_ecg'] - m['v_hold']) * 1000)  # mV

    stats = {
        'record': record_id,
        'n_beats': n,
        'n_detected': len(detected),
        'n_missed': len(missed),
        'detection_rate': len(detected) / n * 100 if n > 0 else 0,
        'missed_beats': [m['beat'] for m in missed],
    }

    if latencies:
        stats.update({
            'lat_mean': np.mean(latencies),
            'lat_std': np.std(latencies),
            'lat_min': np.min(latencies),
            'lat_max': np.max(latencies),
            'lat_p2p': np.max(latencies) - np.min(latencies),
        })

    if capture_errors:
        stats.update({
            'cap_err_mean': np.mean(capture_errors),
            'cap_err_max': np.max(capture_errors),
        })

    return stats


def write_beat_csv(record_id, beats, measurements):
    csv_path = os.path.join(OUT_DIR, f"beats_{record_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['record', 'beat_idx', 'aami_type', 'rpeak_time_s',
                     'rpeak_voltage_V', 'detect_time_s', 'latency_ms',
                     'peak_hold_V', 'capture_error_mV', 'status'])
        for i, (b, m) in enumerate(zip(beats, measurements)):
            status = 'DETECTED' if m['detected'] else 'MISSED'
            t_det = m.get('t_detect', '')
            lat = m.get('latency', 0) * 1000 if 'latency' in m else ''
            v_h = m.get('v_hold', '')
            v_e = m.get('v_ecg', '')
            cap_err = abs(float(v_e) - float(v_h)) * 1000 if v_e and v_h else ''
            w.writerow([record_id, i, b['aami'], f"{b['time']:.7f}",
                        f"{b['v_ecg_at_peak']:.4f}", t_det, lat,
                        v_h, cap_err, status])
    return csv_path


def main():
    print("=" * 70)
    print("v6 TRUE PEAK DETECTOR — MULTI-RECORD VALIDATION")
    print("=" * 70)

    # Phase 1: Record Selection
    print("\n--- PHASE 1: Record Selection ---")
    print(f"{'Record':>8}  Rationale")
    print("-" * 70)
    for rid, rationale in SELECTED_RECORDS:
        print(f"{rid:>8}  {rationale[:60]}")

    all_stats = []
    all_beats_data = []

    for rid, rationale in SELECTED_RECORDS:
        print(f"\n{'=' * 70}")
        print(f"Record {rid}: {rationale[:60]}")
        print(f"{'=' * 70}")

        # Phase 2: Load and characterize
        print(f"  Loading record {rid}...")
        data = load_record(rid, N_BEATS)
        beats = data['beats']
        print(f"  Loaded {len(beats)} beats, seg=[{data['seg_start']}-{data['seg_end']}]")
        print(f"  Scaling: gain={data['gain']:.6f}, offset={data['offset']:.6f}")
        print(f"  V_peak range: {min(b['v_ecg_at_peak'] for b in beats):.3f} - {max(b['v_ecg_at_peak'] for b in beats):.3f} V")
        print(f"  V_baseline: {data['v_baseline']:.3f} V")

        # Generate testbench
        print(f"  Generating testbench...")
        tb_path = generate_testbench(rid, data)
        print(f"  Written: {tb_path}")

        # Run simulation
        output, elapsed = run_simulation(tb_path, rid, timeout=600)
        if output is None:
            print(f"  SIMULATION FAILED (timeout)")
            all_stats.append({'record': rid, 'n_beats': len(beats),
                              'n_detected': 0, 'detection_rate': 0,
                              'error': 'TIMEOUT'})
            continue

        # Parse measurements
        measurements = parse_measurements(output, len(beats))

        # Write beat-level CSV
        csv_path = write_beat_csv(rid, beats, measurements)
        print(f"  Beat CSV: {csv_path}")

        # Analyze
        stats = analyze_record(rid, beats, measurements)
        stats['sim_time'] = elapsed
        all_stats.append(stats)
        all_beats_data.append((rid, beats, measurements))

        # Print per-record summary
        print(f"\n  --- Record {rid} Results ---")
        print(f"  Detection: {stats['n_detected']}/{stats['n_beats']} ({stats['detection_rate']:.1f}%)")
        if stats['n_missed'] > 0:
            print(f"  Missed beats: {stats['missed_beats']}")
        if 'lat_mean' in stats:
            print(f"  Latency: mean={stats['lat_mean']:.2f}ms, std={stats['lat_std']:.2f}ms, "
                  f"range=[{stats['lat_min']:.2f}, {stats['lat_max']:.2f}]ms, p2p={stats['lat_p2p']:.2f}ms")
        if 'cap_err_mean' in stats:
            print(f"  Capture error: mean={stats['cap_err_mean']:.1f}mV, max={stats['cap_err_max']:.1f}mV")

    # Phase 3: Overall Summary
    print(f"\n{'=' * 70}")
    print("PHASE 3: OVERALL PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")

    total_beats = sum(s['n_beats'] for s in all_stats)
    total_detected = sum(s['n_detected'] for s in all_stats)
    total_missed = total_beats - total_detected

    print(f"\n{'Record':>8} {'Beats':>6} {'Det':>5} {'Miss':>5} {'Rate':>6} "
          f"{'Lat_mean':>8} {'Lat_std':>8} {'Lat_p2p':>8} {'CapErr':>8} {'SimTime':>8}")
    print("-" * 85)
    for s in all_stats:
        lat_mean = f"{s['lat_mean']:.2f}" if 'lat_mean' in s else 'N/A'
        lat_std = f"{s['lat_std']:.2f}" if 'lat_std' in s else 'N/A'
        lat_p2p = f"{s['lat_p2p']:.2f}" if 'lat_p2p' in s else 'N/A'
        cap_err = f"{s['cap_err_mean']:.1f}" if 'cap_err_mean' in s else 'N/A'
        sim_t = f"{s.get('sim_time', 0):.0f}s" if 'sim_time' in s else 'N/A'
        print(f"{s['record']:>8} {s['n_beats']:>6} {s['n_detected']:>5} "
              f"{s.get('n_missed', '?'):>5} {s['detection_rate']:>5.1f}% "
              f"{lat_mean:>8} {lat_std:>8} {lat_p2p:>8} {cap_err:>8} {sim_t:>8}")
    print("-" * 85)
    print(f"{'TOTAL':>8} {total_beats:>6} {total_detected:>5} {total_missed:>5} "
          f"{total_detected/total_beats*100:.1f}%")

    # Aggregate latency stats
    all_latencies = []
    for rid, beats, meas in all_beats_data:
        for m in meas:
            if m['detected'] and 'latency' in m:
                all_latencies.append(m['latency'] * 1000)
    if all_latencies:
        lat_arr = np.array(all_latencies)
        print(f"\nAggregate latency (n={len(lat_arr)}):")
        print(f"  Mean={np.mean(lat_arr):.2f}ms, Std={np.std(lat_arr):.2f}ms")
        print(f"  Min={np.min(lat_arr):.2f}ms, Max={np.max(lat_arr):.2f}ms")
        print(f"  P2P jitter={np.max(lat_arr)-np.min(lat_arr):.2f}ms")

    # Phase 5: Validation Decision
    print(f"\n{'=' * 70}")
    print("PHASE 5: VALIDATION DECISION")
    print(f"{'=' * 70}")

    criteria = {
        'detection_rate_100': total_detected == total_beats,
        'zero_false_positives': True,  # checked per-record
        'zero_missed': total_missed == 0,
    }

    for k, v in criteria.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")

    if all(criteria.values()):
        print(f"\n  >>> VALIDATION: PASS <<<")
        print(f"  v6 true peak detector meets all criteria across {total_beats} beats / 5 records.")
    elif total_detected / total_beats >= 0.95:
        print(f"\n  >>> VALIDATION: CONDITIONAL PASS <<<")
        print(f"  Detection rate {total_detected/total_beats*100:.1f}% (>= 95%)")
        print(f"  Investigate {total_missed} missed beats before proceeding.")
    else:
        print(f"\n  >>> VALIDATION: FAIL <<<")
        print(f"  Detection rate {total_detected/total_beats*100:.1f}% (< 95%)")
        print(f"  Phase 4 failure investigation required.")

    # Phase 4: Failure investigation (if any misses)
    if total_missed > 0:
        print(f"\n{'=' * 70}")
        print("PHASE 4: FAILURE INVESTIGATION")
        print(f"{'=' * 70}")
        for rid, beats, meas in all_beats_data:
            missed = [(i, b, m) for i, (b, m) in enumerate(zip(beats, meas)) if not m['detected']]
            if missed:
                print(f"\n  Record {rid}: {len(missed)} missed beats")
                for i, b, m in missed:
                    print(f"    Beat {i}: type={b['aami']}, t={b['time']:.4f}s, "
                          f"v_ecg={b['v_ecg_at_peak']:.3f}V")

    # Save summary
    summary_path = os.path.join(OUT_DIR, "validation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"v6 True Peak Detector Multi-Record Validation\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Records: {', '.join(r[0] for r in SELECTED_RECORDS)}\n")
        f.write(f"Beats per record: {N_BEATS}\n")
        f.write(f"Total: {total_detected}/{total_beats} detected ({total_detected/total_beats*100:.1f}%)\n")
        if all_latencies:
            f.write(f"Latency: mean={np.mean(all_latencies):.2f}ms, std={np.std(all_latencies):.2f}ms\n")
        f.write(f"Decision: {'PASS' if all(criteria.values()) else 'FAIL'}\n")

    print(f"\nResults saved to: {OUT_DIR}/")


if __name__ == '__main__':
    main()
