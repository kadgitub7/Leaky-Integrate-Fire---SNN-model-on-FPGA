"""
Analyze Python detector intermediate signals across records.
Quantify voltage ranges and slopes after each processing stage
to set analog circuit design targets.
"""
import numpy as np
import wfdb
import os
import sys

sys.path.insert(0, 'scratchpad')
from analog_detector_final import get_intermediate_signals, detect_final

DB_PATH = 'mitdb_data'
FS = 360
TEST_DURATION_S = 10.0

RECORDS = ['100', '101', '102', '103', '104', '105', '106',
           '107', '108', '109', '112', '113', '114', '115']


def analyze_record(record_id):
    record = wfdb.rdrecord(os.path.join(DB_PATH, record_id))
    ecg = record.p_signal[:, 0]
    n_samples = int(TEST_DURATION_S * FS)
    ecg = ecg[:n_samples]

    sigs = get_intermediate_signals(ecg, FS)
    dets = detect_final(ecg, FS)
    dets = [d for d in dets if d < n_samples]

    filt = sigs['filtered']
    deriv = sigs['derivative']
    rect = sigs['rectified']
    env_fast = sigs['env_fast']
    env_avg = sigs['env_avg']

    # Measure at detection times
    det_ef = [env_fast[d] for d in dets]
    det_ea = [env_avg[d] for d in dets]
    det_rect = [rect[d] for d in dets]
    det_ratio = [env_fast[d] / env_avg[d] if env_avg[d] > 0 else 0 for d in dets]

    # Measure rectified signal stats
    rect_max = np.max(rect)
    rect_mean = np.mean(rect)
    rect_at_beats = np.mean(det_rect) if det_rect else 0

    # Measure env_fast range
    ef_min, ef_max = np.min(env_fast), np.max(env_fast)
    ea_min, ea_max = np.min(env_avg), np.max(env_avg)

    # QRS vs non-QRS discrimination in rectified domain
    # QRS peaks in rectified: highest values near detections
    qrs_rect_peaks = []
    for d in dets:
        win_start = max(0, d - 18)  # 50ms window
        win_end = min(len(rect), d + 18)
        qrs_rect_peaks.append(np.max(rect[win_start:win_end]))

    # Non-QRS max (between beats)
    non_qrs_max = []
    for i in range(len(dets) - 1):
        gap_start = dets[i] + 36  # 100ms after beat
        gap_end = dets[i+1] - 36  # 100ms before next beat
        if gap_start < gap_end:
            non_qrs_max.append(np.max(rect[gap_start:gap_end]))

    return {
        'record': record_id,
        'n_dets': len(dets),
        'ecg_range': (ecg.min(), ecg.max()),
        'filt_range': (filt.min(), filt.max()),
        'deriv_range': (deriv.min(), deriv.max()),
        'rect_max': rect_max,
        'rect_mean': rect_mean,
        'rect_at_beats': rect_at_beats,
        'ef_range': (ef_min, ef_max),
        'ea_range': (ea_min, ea_max),
        'mean_ratio': np.mean(det_ratio) if det_ratio else 0,
        'min_ratio': np.min(det_ratio) if det_ratio else 0,
        'qrs_rect_peak': np.mean(qrs_rect_peaks) if qrs_rect_peaks else 0,
        'qrs_rect_min': np.min(qrs_rect_peaks) if qrs_rect_peaks else 0,
        'non_qrs_max': np.mean(non_qrs_max) if non_qrs_max else 0,
        'non_qrs_p95': np.percentile(non_qrs_max, 95) if non_qrs_max else 0,
        'discrimination': (np.min(qrs_rect_peaks) / np.percentile(non_qrs_max, 95)
                          if non_qrs_max and qrs_rect_peaks else 0),
    }


if __name__ == '__main__':
    print("=" * 100)
    print("PYTHON DETECTOR INTERMEDIATE SIGNAL ANALYSIS")
    print("=" * 100)

    results = []
    for rec in RECORDS:
        try:
            r = analyze_record(rec)
            results.append(r)
        except Exception as e:
            print(f"Record {rec}: ERROR - {e}")
            continue

    print(f"\n{'Rec':>4} {'Det':>3} {'ECG range':>14} {'Filt range':>14} "
          f"{'Rect max':>9} {'EF range':>14} {'EA range':>14} "
          f"{'Ratio':>6} {'QRS pk':>7} {'NonQRS':>7} {'Discr':>6}")
    print("-" * 120)

    for r in results:
        print(f"{r['record']:>4} {r['n_dets']:>3} "
              f"[{r['ecg_range'][0]:+.3f},{r['ecg_range'][1]:+.3f}] "
              f"[{r['filt_range'][0]:+.3f},{r['filt_range'][1]:+.3f}] "
              f"{r['rect_max']:>8.4f} "
              f"[{r['ef_range'][0]:.4f},{r['ef_range'][1]:.4f}] "
              f"[{r['ea_range'][0]:.4f},{r['ea_range'][1]:.4f}] "
              f"{r['mean_ratio']:>5.1f}x "
              f"{r['qrs_rect_min']:>6.4f} "
              f"{r['non_qrs_p95']:>6.4f} "
              f"{r['discrimination']:>5.1f}x")

    print("\n--- Key Design Targets ---")
    all_rect_max = [r['rect_max'] for r in results]
    all_qrs_min = [r['qrs_rect_min'] for r in results]
    all_nonqrs = [r['non_qrs_p95'] for r in results]
    all_discr = [r['discrimination'] for r in results if r['discrimination'] > 0]

    print(f"Rectified signal max across records: {min(all_rect_max):.4f} - {max(all_rect_max):.4f}")
    print(f"Smallest QRS peak in rectified: {min(all_qrs_min):.4f}")
    print(f"Largest non-QRS in rectified (p95): {max(all_nonqrs):.4f}")
    print(f"Worst-case QRS/non-QRS discrimination: {min(all_discr):.1f}x")
    print(f"Median discrimination: {np.median(all_discr):.1f}x")
