"""Generate QRS symmetry ground truth from 10 real MIT-BIH beats.

Computes exact Python symmetry values AND the timing parameters
(onset_to_peak_ms, qrs_width_ms) needed to drive an isolated
ngspice testbench with controlled PWL inputs.

Output: qrs_sym_ground_truth.json
"""

import os, sys, json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MITDB_DIR = os.path.join(REPO_ROOT, 'mitdb_data')

sys.path.insert(0, REPO_ROOT)

try:
    import wfdb
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wfdb', '-q'])
    import wfdb

FS = 360
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130

RECORDS = ['100', '103', '105', '111', '200', '210', '212', '219', '221', '231']


def compute_symmetry_for_beat(qrs_segment, fs=360):
    """Exact replica of the Python golden model symmetry computation."""
    abs_qrs = np.abs(qrs_segment)
    peak = np.max(abs_qrs)

    if peak <= 1e-6:
        return 0.5, None

    peak_idx = np.argmax(abs_qrs)
    threshold = 0.3 * peak
    above = abs_qrs > threshold

    if not np.any(above):
        return 0.5, None

    onset = np.argmax(above)
    last = len(above) - 1 - np.argmax(above[::-1])
    width_samples = last - onset
    width_ms = width_samples * 1000.0 / fs

    time_to_peak_samples = peak_idx - onset
    time_to_peak_ms = time_to_peak_samples * 1000.0 / fs

    qrs_w = max(width_ms, 1e-6) * fs / 1000.0  # back to samples
    symmetry = time_to_peak_samples / (qrs_w + 1e-8)
    symmetry = float(np.clip(symmetry, 0.0, 1.0))

    fall_time_samples = last - peak_idx
    fall_time_ms = fall_time_samples * 1000.0 / fs

    return symmetry, {
        'peak_idx_in_qrs': int(peak_idx),
        'onset_idx_in_qrs': int(onset),
        'offset_idx_in_qrs': int(last),
        'width_samples': int(width_samples),
        'width_ms': float(width_ms),
        'time_to_peak_samples': int(time_to_peak_samples),
        'time_to_peak_ms': float(time_to_peak_ms),
        'fall_time_samples': int(fall_time_samples),
        'fall_time_ms': float(fall_time_ms),
        'peak_amplitude_mV': float(peak * 1000),
        'threshold_mV': float(threshold * 1000),
    }


def main():
    results = []
    beat_count = 0

    for rec_id in RECORDS:
        record_path = os.path.join(MITDB_DIR, rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"  Skip {rec_id}: {e}")
            continue

        signals = record.p_signal
        valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
                 if idx - WIN_L >= 0 and idx + WIN_R < len(signals)]

        # Take beat index 10 from each record (past warm-up)
        for bi in [10]:
            if bi >= len(valid):
                continue

            idx, sym = valid[bi]
            beat = signals[idx - WIN_L: idx + WIN_R, :]
            qrs_L0 = beat[QRS_S:QRS_E, 0]

            symmetry, details = compute_symmetry_for_beat(qrs_L0, FS)

            if details is None:
                continue

            entry = {
                'record': rec_id,
                'beat_index': bi,
                'rpeak_sample': int(idx),
                'annotation': sym,
                'symmetry_python': symmetry,
                **details,
            }
            results.append(entry)
            beat_count += 1

            print(f"Record {rec_id} beat {bi}: symmetry={symmetry:.4f}  "
                  f"rise={details['time_to_peak_ms']:.1f}ms  "
                  f"fall={details['fall_time_ms']:.1f}ms  "
                  f"width={details['width_ms']:.1f}ms")

    out_path = os.path.join(os.path.dirname(__file__), 'qrs_sym_ground_truth.json')
    with open(out_path, 'w') as f:
        json.dump({'n_beats': beat_count, 'beats': results}, f, indent=2)

    print(f"\nSaved {beat_count} beats to {out_path}")
    print(f"\nSummary:")
    syms = [r['symmetry_python'] for r in results]
    print(f"  Min symmetry: {min(syms):.4f}")
    print(f"  Max symmetry: {max(syms):.4f}")
    print(f"  Mean symmetry: {np.mean(syms):.4f}")
    print(f"  Std symmetry: {np.std(syms):.4f}")


if __name__ == '__main__':
    main()
