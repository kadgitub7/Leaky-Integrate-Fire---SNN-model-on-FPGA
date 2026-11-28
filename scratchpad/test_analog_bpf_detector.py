"""
Test an analog-realizable detector that faithfully replicates the Python model.
Uses causal 1st-order filters (not sosfiltfilt which is zero-phase/non-causal).

Architecture matching Python model:
  1. 1st-order HPF at 5Hz (AC coupling + OTA bias restore)
  2. 1st-order LPF at 20Hz (OTA follower)
  3. Capacitor differentiator (series cap → TIA)
  4. Half-wave rectification (positive half only, like diode)
  5. Fast envelope (100ms leaky peak detector) - existing circuit
  6. 30ms smoothing - existing circuit
  7. Slow average (1.5s RC on rectified) - existing circuit
  8. EWMA adaptive threshold + noise gate
"""
import numpy as np
import wfdb
import os
import sys
from scipy.signal import butter, sosfilt

sys.path.insert(0, 'scratchpad')
from analog_detector_final import detect_final, envelope_follower, rc_lowpass, moving_average

DB_PATH = 'mitdb_data'
FS = 360
TEST_DURATION_S = 10.0
MATCH_TOL_S = 0.150

# Analog-realizable parameters
HPF_FREQ = 5.0    # Hz, 1st-order HPF (AC coupling)
LPF_FREQ = 20.0   # Hz, 1st-order LPF
FAST_TAU = 0.100   # 100ms envelope
SMOOTH_MS = 30
SLOW_TAU = 1.5     # 1.5s average
REFRACTORY_MS = 200
NOISE_RATIO = 3.0
THRESHOLD_FRAC = 0.25
TWAVE_MS = 360

RECORDS = ['100', '101', '102', '103', '104', '105', '106',
           '107', '108', '109', '112', '113', '114', '115']


def causal_1st_order_hpf(signal, fs, fc):
    """1st-order causal HPF (RC high-pass)."""
    rc = 1.0 / (2 * np.pi * fc)
    alpha = rc / (rc + 1.0/fs)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * (out[i-1] + signal[i] - signal[i-1])
    return out


def causal_1st_order_lpf(signal, fs, fc):
    """1st-order causal LPF (RC low-pass)."""
    rc = 1.0 / (2 * np.pi * fc)
    alpha = (1.0/fs) / (rc + 1.0/fs)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = out[i-1] + alpha * (signal[i] - out[i-1])
    return out


def analog_derivative(signal, fs):
    """Causal first-difference (capacitor differentiator)."""
    deriv = np.zeros_like(signal)
    for i in range(1, len(signal)):
        deriv[i] = (signal[i] - signal[i-1]) * fs
    return deriv


def half_wave_rectify(signal):
    """Half-wave rectification (diode-connected FET)."""
    return np.maximum(signal, 0.0)


def full_wave_rectify(signal):
    """Full-wave rectification."""
    return np.abs(signal)


def detect_analog_v4(ecg, fs=360, use_derivative=True, use_fullwave=True):
    """
    Analog-realizable detector.
    Returns detected R-peak sample indices.
    """
    # 1. HPF at 5Hz (AC coupling + OTA bias restore)
    hpf = causal_1st_order_hpf(ecg, fs, HPF_FREQ)

    # 2. LPF at 20Hz (OTA follower)
    bpf = causal_1st_order_lpf(hpf, fs, LPF_FREQ)

    if use_derivative:
        # 3. Derivative (capacitor differentiator)
        deriv = analog_derivative(bpf, fs)

        # 4. Rectification
        if use_fullwave:
            rect = full_wave_rectify(deriv)
        else:
            rect = half_wave_rectify(deriv)
    else:
        # Skip derivative, just rectify the BPF output
        if use_fullwave:
            rect = full_wave_rectify(bpf)
        else:
            rect = half_wave_rectify(bpf)

    # 5. Fast envelope (100ms leaky peak detector)
    env_fast_raw = envelope_follower(rect, fs, FAST_TAU)

    # 6. 30ms smoothing
    smooth_win = max(1, int(SMOOTH_MS * fs / 1000))
    env_fast = moving_average(env_fast_raw, smooth_win)

    # 7. Slow average (1.5s RC on rectified signal)
    env_avg = rc_lowpass(rect, fs, SLOW_TAU)

    # 8. Find candidate peaks
    from scipy.signal import find_peaks
    peak_dist = int(80 * fs / 1000)
    candidates, _ = find_peaks(env_fast, distance=peak_dist)

    # 9. EWMA adaptive threshold
    spki = 0.0
    npki = 0.0
    alpha = 0.125
    detections = []
    rr_intervals = []
    min_dist = int(REFRACTORY_MS * fs / 1000)

    for idx in candidates:
        ef = env_fast[idx]
        ea = env_avg[idx]

        threshold = npki + THRESHOLD_FRAC * (spki - npki)
        noise_thresh = NOISE_RATIO * ea

        if ef > threshold and ef > noise_thresh:
            # T-wave rejection
            if len(detections) > 0:
                dt_ms = (idx - detections[-1]) * 1000.0 / fs
                if dt_ms < TWAVE_MS:
                    if len(detections) >= 2:
                        prev_ef = env_fast[detections[-1]]
                        if ef < 0.5 * prev_ef:
                            npki = npki + alpha * (ef - npki)
                            continue

            # Refractory check
            if len(detections) > 0:
                dt_ms = (idx - detections[-1]) * 1000.0 / fs
                if dt_ms < REFRACTORY_MS:
                    continue

            detections.append(idx)
            spki = spki + alpha * (ef - spki)

            if len(detections) >= 2:
                rr = detections[-1] - detections[-2]
                rr_intervals.append(rr)
        else:
            npki = npki + alpha * (ef - npki)

    # 10. Search-back
    if len(rr_intervals) > 2:
        avg_rr = np.mean(rr_intervals[-8:]) if len(rr_intervals) >= 8 else np.mean(rr_intervals)
        gap_thresh = 1.66 * avg_rr
        relaxed_nr = NOISE_RATIO * 0.6

        new_dets = list(detections)
        for i in range(1, len(new_dets)):
            gap = new_dets[i] - new_dets[i-1]
            if gap > gap_thresh:
                start = new_dets[i-1] + min_dist
                end = new_dets[i] - min_dist
                if start < end:
                    segment = env_fast[start:end]
                    if len(segment) > 0:
                        sub_peaks, _ = find_peaks(segment, distance=peak_dist)
                        for sp in sub_peaks:
                            sp_idx = sp + start
                            ef = env_fast[sp_idx]
                            ea = env_avg[sp_idx]
                            thr = npki + 0.5 * THRESHOLD_FRAC * (spki - npki)
                            if ef > thr and ef > relaxed_nr * ea:
                                new_dets.append(sp_idx)
        detections = sorted(set(new_dets))

    # 11. Map to raw ECG peaks
    final = []
    search_win = int(75 * fs / 1000)
    for d in detections:
        start = max(0, d - search_win)
        end = min(len(ecg), d + search_win)
        peak_idx = start + np.argmax(np.abs(ecg[start:end] - np.median(ecg)))
        final.append(peak_idx)

    return final


def nearest_neighbor_match(ref_times, det_times, tol_s=0.150):
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', nargs='+', default=RECORDS)
    parser.add_argument('--no-deriv', action='store_true')
    parser.add_argument('--half-wave', action='store_true')
    args = parser.parse_args()

    use_deriv = not args.no_deriv
    use_fullwave = not args.half_wave

    mode = f"BPF(5-20Hz)"
    if use_deriv:
        mode += " + derivative"
    if use_fullwave:
        mode += " + fullwave rect"
    else:
        mode += " + halfwave rect"

    print("=" * 80)
    print(f"ANALOG-REALIZABLE DETECTOR TEST: {mode}")
    print("=" * 80)

    results = []
    for rec in args.records:
        try:
            record = wfdb.rdrecord(os.path.join(DB_PATH, rec))
            ecg = record.p_signal[:, 0]
            n_samples = int(TEST_DURATION_S * FS)
            ecg = ecg[:n_samples]

            # Reference: Python detector
            py_dets = detect_final(ecg, FS)
            py_dets = [d for d in py_dets if d < n_samples]

            # Analog-realizable detector
            analog_dets = detect_analog_v4(ecg, FS, use_derivative=use_deriv,
                                           use_fullwave=use_fullwave)
            analog_dets = [d for d in analog_dets if d < n_samples]

            py_times = [d / FS for d in py_dets]
            an_times = [d / FS for d in analog_dets]

            tp, fn, fp, offsets = nearest_neighbor_match(py_times, an_times)
            n_py = len(py_dets)
            n_an = len(analog_dets)
            sens = tp / n_py * 100 if n_py > 0 else 0
            ppv = tp / n_an * 100 if n_an > 0 else 0

            off_arr = np.array(offsets) if offsets else np.array([0])

            print(f"  Record {rec}: Py={n_py} An={n_an} TP={tp} FN={fn} FP={fp} "
                  f"Sens={sens:.1f}% PPV={ppv:.1f}% "
                  f"Off={off_arr.mean():.1f}±{off_arr.std():.1f}ms")

            results.append({
                'record': rec, 'n_py': n_py, 'n_an': n_an,
                'tp': tp, 'fn': fn, 'fp': fp,
                'sens': sens, 'ppv': ppv,
            })
        except Exception as e:
            print(f"  Record {rec}: ERROR - {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    sens_arr = [r['sens'] for r in results]
    ppv_arr = [r['ppv'] for r in results]
    print(f"Mean sensitivity: {np.mean(sens_arr):.1f}%")
    print(f"Mean PPV: {np.mean(ppv_arr):.1f}%")
    print(f"Min sensitivity: {np.min(sens_arr):.1f}% (Record {results[np.argmin(sens_arr)]['record']})")
    print(f"Records >=95% sens: {sum(1 for s in sens_arr if s >= 95)}/{len(sens_arr)}")
    print(f"Records =100% sens: {sum(1 for s in sens_arr if s >= 99.9)}/{len(sens_arr)}")
