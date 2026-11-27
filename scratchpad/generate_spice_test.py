"""
Generate SPICE test data from the Python R-peak detector.

Phase 1: Generate PWL files for the intermediate signals so we can
test the analog envelope/detection path in isolation.

Creates:
  - rectified_pwl.txt  : |d/dt(BPF(ecg))| signal, scaled to 0.3V baseline + signal
  - ecg_pwl.txt        : raw ECG scaled to IA output range (0.30-0.46V)
  - detection_times.txt : Python detection sample indices for comparison
  - env_fast_ref.txt    : Python env_fast for comparison
  - env_avg_ref.txt     : Python env_avg for comparison
"""

import numpy as np
import wfdb
import os
import sys

sys.path.insert(0, 'scratchpad')
from analog_detector_final import detect_final, get_intermediate_signals

DB_PATH = 'mitdb_data'
RECORD = '100'
VBASE = 0.300    # ECG baseline voltage (V)
VDD = 0.900
FS = 360         # MIT-BIH sample rate

# How many seconds of data to use for SPICE test
# Full record is ~30min; use 30s for initial testing
TEST_DURATION_S = 30.0

# Output directory
OUT_DIR = 'scratchpad'


def scale_to_voltage(signal, v_center, v_range):
    """Scale signal to voltage range centered at v_center."""
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max - sig_min < 1e-15:
        return np.full_like(signal, v_center)
    normalized = (signal - sig_min) / (sig_max - sig_min)  # 0 to 1
    return v_center - v_range/2 + normalized * v_range


def write_pwl(filename, times, voltages, comment=""):
    """Write a SPICE PWL file (no comment lines — ngspice doesn't allow them)."""
    with open(filename, 'w') as f:
        for t, v in zip(times, voltages):
            f.write(f"{t:.9f} {v:.6f}\n")
    print(f"  Written: {filename} ({len(times)} points, {comment})")


def main():
    # Load record
    record = wfdb.rdrecord(os.path.join(DB_PATH, RECORD))
    ann = wfdb.rdann(os.path.join(DB_PATH, RECORD), 'atr')
    ecg = record.p_signal[:, 0]
    fs = record.fs

    # Trim to test duration
    n_samples = int(TEST_DURATION_S * fs)
    ecg = ecg[:n_samples]

    print(f"Record {RECORD}: {n_samples} samples ({TEST_DURATION_S}s at {fs}Hz)")

    # Get Python intermediate signals
    sigs = get_intermediate_signals(ecg, fs)
    detections = detect_final(ecg, fs)
    # Keep only detections within test window
    detections = [d for d in detections if d < n_samples]

    print(f"Python detections in window: {len(detections)}")

    # Time vector
    times = np.arange(n_samples) / fs

    # --- ECG PWL (scaled to IA output range) ---
    # IA output: 0.30V baseline, peak at ~0.46V (160mV range)
    ecg_v = scale_to_voltage(ecg, v_center=0.38, v_range=0.16)
    write_pwl(os.path.join(OUT_DIR, 'ecg_pwl.txt'), times, ecg_v,
              f"ECG Record {RECORD} scaled to IA output range")

    # --- Rectified signal PWL ---
    # The rectified signal (|d/dt(BPF)|) needs to be scaled to a
    # reasonable voltage range for the analog envelope detectors.
    # In the analog circuit, this would be the output of the
    # diff+rect stage, sitting at some DC bias with signal on top.
    #
    # Scale so: baseline ≈ 0.30V, QRS peaks reach ~0.45V
    # The rectified signal is always ≥ 0
    rect = sigs['rectified']
    rect_v = VBASE + rect / (rect.max() + 1e-15) * 0.15
    write_pwl(os.path.join(OUT_DIR, 'rectified_pwl.txt'), times, rect_v,
              f"Rectified derivative, Record {RECORD}")

    # --- Reference envelope signals (for SPICE comparison) ---
    # Scale env_fast and env_avg to same voltage range as rectified
    env_fast = sigs['env_fast']
    env_avg = sigs['env_avg']

    env_fast_v = VBASE + env_fast / (rect.max() + 1e-15) * 0.15
    env_avg_v = VBASE + env_avg / (rect.max() + 1e-15) * 0.15

    write_pwl(os.path.join(OUT_DIR, 'env_fast_ref.txt'), times, env_fast_v,
              "Python env_fast (reference)")
    write_pwl(os.path.join(OUT_DIR, 'env_avg_ref.txt'), times, env_avg_v,
              "Python env_avg (reference)")

    # --- Detection times ---
    with open(os.path.join(OUT_DIR, 'detection_times.txt'), 'w') as f:
        f.write(f"* Python detections for Record {RECORD} ({len(detections)} beats)\n")
        f.write(f"* Format: sample_index time_seconds\n")
        for d in detections:
            f.write(f"{d} {d/fs:.6f}\n")
    print(f"  Written: detection_times.txt ({len(detections)} detections)")

    # --- Print some stats for SPICE sizing ---
    print(f"\nSignal statistics for SPICE sizing:")
    print(f"  Rectified: min={rect.min():.6f}, max={rect.max():.6f}, mean={rect.mean():.6f}")
    print(f"  Env_fast:  min={env_fast.min():.6f}, max={env_fast.max():.6f}")
    print(f"  Env_avg:   min={env_avg.min():.6f}, max={env_avg.max():.6f}")
    print(f"  Noise ratio at QRS peaks: ", end="")

    # Check ratio at detection times
    ratios = []
    for d in detections:
        if d < len(env_fast) and env_avg[d] > 0:
            ratios.append(env_fast[d] / env_avg[d])
    if ratios:
        print(f"mean={np.mean(ratios):.1f}, min={np.min(ratios):.1f}, max={np.max(ratios):.1f}")

    # Print voltage ranges for SPICE
    print(f"\nVoltage ranges (after scaling):")
    print(f"  ECG:       {ecg_v.min():.3f}V - {ecg_v.max():.3f}V")
    print(f"  Rectified: {rect_v.min():.3f}V - {rect_v.max():.3f}V")
    print(f"  Env_fast:  {env_fast_v.min():.3f}V - {env_fast_v.max():.3f}V")
    print(f"  Env_avg:   {env_avg_v.min():.3f}V - {env_avg_v.max():.3f}V")

    # Print env_fast/env_avg voltage difference at peaks (for noise gate sizing)
    diffs = []
    for d in detections:
        if d < len(env_fast_v):
            diffs.append(env_fast_v[d] - env_avg_v[d])
    if diffs:
        print(f"\n  Env_fast - Env_avg at QRS peaks:")
        print(f"    mean={np.mean(diffs)*1000:.1f}mV, min={np.min(diffs)*1000:.1f}mV, max={np.max(diffs)*1000:.1f}mV")


if __name__ == '__main__':
    main()
