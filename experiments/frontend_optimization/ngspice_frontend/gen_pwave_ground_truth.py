"""
Generate P-wave energy ground truth and ngspice PWL stimulus.

Uses the same 10 MIT-BIH beats as QRS symmetry:
  B1=rec100, B2=rec103, B3=rec105, B4=rec111, B5=rec200,
  B6=rec210, B7=rec212, B8=rec219, B9=rec221, B10=rec231

For each record, picks the first valid normal beat (same as qrs_sym tests).

Computes:
  pwave_energy = sum(bpf_segment^2) / fs
  where bpf_segment = 3-8Hz bandpass of ECG, samples PW_S:PW_E of beat window

Generates:
  1. pwave_ground_truth.json - ground truth values
  2. pwave_ecg_pwl.inc - PWL source for ngspice (P-wave BPF segment)
"""

import numpy as np
import json
import os
import sys

# Add project root for wfdb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import wfdb
from scipy.signal import butter, filtfilt

# Constants (matching frontend_optimization.py)
FS = 360
WIN_L = 90
WIN_R = 108
QRS_S = 70
PW_S = 16
PW_E = QRS_S  # = 70

AAMI = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
        'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
        'V': 'V', 'E': 'V',
        'F': 'F',
        '/': 'Q', 'f': 'Q', 'Q': 'Q'}

RECORDS = ['100', '103', '105', '111', '200', '210', '212', '219', '221', '231']
MITDB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mitdb_data')

def design_pwave_filter(fs):
    nyq = fs / 2
    low = 3.0 / nyq
    high = 8.0 / nyq
    b, a = butter(2, [low, high], btype='band')
    return b, a

PW_B, PW_A = design_pwave_filter(FS)

results = []

for beat_num, rec_id in enumerate(RECORDS, 1):
    record_path = os.path.join(MITDB_PATH, rec_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    signals = record.p_signal
    fs = record.fs

    # Bandpass filter entire signal
    pwave_filtered = np.zeros_like(signals)
    for lead in range(min(signals.shape[1], 2)):
        pwave_filtered[:, lead] = filtfilt(PW_B, PW_A, signals[:, lead])

    # Find first valid normal beat
    for idx, sym in zip(annotation.sample, annotation.symbol):
        if idx - WIN_L >= 0 and idx + WIN_R < len(signals) and sym in AAMI:
            if AAMI[sym] == 'N':
                beat = signals[idx - WIN_L : idx + WIN_R, :]
                beat_pw = pwave_filtered[idx - WIN_L : idx + WIN_R, :]

                # Use lead 1 (index 1) for pwave_energy_L1
                lead = 1 if signals.shape[1] > 1 else 0

                pw_segment = beat_pw[PW_S:PW_E, lead]
                pwave_energy = np.sum(pw_segment**2) / fs

                # Also get the raw ECG P-wave segment (for PWL)
                raw_pw_segment = beat[PW_S:PW_E, lead]
                # And the full beat BPF output for context
                full_bpf = beat_pw[:, lead]

                # Time axis for P-wave window (relative to beat start)
                # PW_S=16 to PW_E=70, at 360 Hz
                pw_times = np.arange(PW_S, PW_E) / fs  # seconds

                results.append({
                    'beat': beat_num,
                    'record': rec_id,
                    'r_peak_sample': int(idx),
                    'pwave_energy_L1': float(pwave_energy),
                    'pw_segment_bpf': pw_segment.tolist(),
                    'pw_segment_raw': raw_pw_segment.tolist(),
                    'full_bpf_L1': full_bpf.tolist(),
                    'pw_peak_bpf': float(np.max(np.abs(pw_segment))),
                    'pw_rms_bpf': float(np.sqrt(np.mean(pw_segment**2))),
                    'pw_times_s': pw_times.tolist(),
                })
                break

# Save ground truth
gt_file = os.path.join(os.path.dirname(__file__), 'pwave_ground_truth.json')
with open(gt_file, 'w') as f:
    json.dump({
        'description': 'P-wave energy ground truth for 10 MIT-BIH beats (lead 1)',
        'feature': 'pwave_energy_L1',
        'formula': 'sum(bpf_segment^2) / fs, where bpf = 3-8Hz Butterworth order 2',
        'pwave_window': f'samples {PW_S}-{PW_E} in beat window (WIN_L={WIN_L})',
        'fs': FS,
        'beats': results,
    }, f, indent=2)

# Print summary
print("=" * 75)
print("P-WAVE ENERGY GROUND TRUTH — 10 MIT-BIH BEATS (Lead 1)")
print("=" * 75)
print(f"\n{'Beat':>4} {'Rec':>4} {'PW_Energy':>12} {'Peak_BPF':>10} {'RMS_BPF':>10}")
print(f"{'':>4} {'':>4} {'(V^2/Hz)':>12} {'(mV)':>10} {'(mV)':>10}")
print("-" * 50)

sorted_idx = np.argsort([r['pwave_energy_L1'] for r in results])
for i in sorted_idx:
    r = results[i]
    print(f"B{r['beat']:>3} {r['record']:>4} {r['pwave_energy_L1']:>12.6e} "
          f"{r['pw_peak_bpf']*1000:>10.3f} {r['pw_rms_bpf']*1000:>10.3f}")

# Check correlation between peak and energy
energies = np.array([r['pwave_energy_L1'] for r in results])
peaks = np.array([r['pw_peak_bpf'] for r in results])
rms_vals = np.array([r['pw_rms_bpf'] for r in results])

corr_peak = np.corrcoef(energies, peaks)[0,1]
corr_rms = np.corrcoef(energies, rms_vals)[0,1]

print(f"\nCorrelation energy vs peak(|BPF|): {corr_peak:.4f}")
print(f"Correlation energy vs RMS(BPF):    {corr_rms:.4f}")
print(f"Energy range: {np.min(energies):.6e} to {np.max(energies):.6e}")
print(f"Ratio max/min: {np.max(energies)/np.min(energies):.1f}x")

# Generate PWL files for ngspice
# We'll feed the full conditioned ECG signal (not just P-wave window) because
# the BPF needs time to settle. Use 500ms of signal per beat.
#
# For each beat, extract 500ms of ECG centered on the R-peak.
# The P-wave is ~250ms before R-peak (within the beat window).
# With 500ms total (250ms before, 250ms after), the BPF can settle.

print(f"\n{'='*75}")
print("GENERATING PWL STIMULUS FOR NGSPICE")
print(f"{'='*75}")

BEAT_DURATION = 2.0  # 2s per beat slot (enough BPF settling)
SETTLE_TIME = 1.0    # 1s settle before first beat

# Scale ECG to 0.9V range centered at vref=0.45V
# MIT-BIH signals are in mV, typical P-wave amplitude ~0.1-0.3 mV
# Scale: V_ngspice = 0.45 + signal * gain
# Need signal range to fit in [0.1, 0.8] V for the OTA to work
# Max signal across all beats:
all_ecg = []
all_bpf = []
for r in results:
    all_ecg.extend(r['pw_segment_raw'])
    all_bpf.extend(r['pw_segment_bpf'])

ecg_max = max(abs(min(all_ecg)), abs(max(all_ecg)))
bpf_max = max(abs(min(all_bpf)), abs(max(all_bpf)))

print(f"Raw ECG range: {min(all_ecg)*1000:.2f} to {max(all_ecg)*1000:.2f} mV")
print(f"BPF range: {min(all_bpf)*1000:.2f} to {max(all_bpf)*1000:.2f} mV")

# Extract ECG data per beat — exactly BEAT_DURATION to avoid PWL overlap
ECG_BEFORE = 360   # 1.0s before R-peak (8.5τ BPF settling before P-wave)
ECG_AFTER = int(BEAT_DURATION * FS) - ECG_BEFORE  # = 360 samples = 1.0s
ECG_CONTEXT = ECG_BEFORE + ECG_AFTER  # = 720 samples = 2.0s

pwl_lines = []
t_offset = SETTLE_TIME
beat_timing = []

for r in results:
    rec_id = r['record']
    idx = r['r_peak_sample']

    record_path = os.path.join(MITDB_PATH, rec_id)
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal

    lead = 1 if signals.shape[1] > 1 else 0

    # Extract 1 second of ECG
    start = max(0, idx - ECG_BEFORE)
    end = min(len(signals), idx + ECG_AFTER)
    ecg_segment = signals[start:end, lead]

    # Scale to [0.1, 0.8]V centered at 0.45V
    # ECG is in mV (wfdb returns physical units)
    # Typical MIT-BIH range: ±2 mV
    gain = 0.15  # 0.15 V per mV → ±2mV maps to ±0.3V around 0.45V
    vref = 0.45

    # Generate PWL points
    for j, sample in enumerate(ecg_segment):
        t = t_offset + j / FS
        v = vref + sample * gain
        v = max(0.05, min(0.85, v))  # clamp
        pwl_lines.append(f"+ {t:.6f} {v:.6f}")

    # Timing for this beat's signals
    # P-wave window: PW_S to PW_E relative to beat start (idx - WIN_L)
    # In the ECG context: the R-peak is at t_offset + ECG_BEFORE/FS
    rpeak_t = t_offset + (idx - start) / FS
    # Beat window start: rpeak_t - WIN_L/FS
    beat_start = rpeak_t - WIN_L / FS
    # P-wave window in real time:
    pw_start = beat_start + PW_S / FS
    pw_end = beat_start + PW_E / FS

    beat_timing.append({
        'beat': r['beat'],
        'record': rec_id,
        't_offset': t_offset,
        'rpeak_t': rpeak_t,
        'pw_start': pw_start,
        'pw_end': pw_end,
        'pwave_energy': r['pwave_energy_L1'],
    })

    print(f"B{r['beat']:>2} rec{rec_id}: t={t_offset:.3f}-{t_offset + len(ecg_segment)/FS:.3f}s, "
          f"pw=[{pw_start:.4f},{pw_end:.4f}]s, rpeak={rpeak_t:.4f}s")

    t_offset += BEAT_DURATION

total_time = t_offset + 0.1
print(f"\nTotal simulation time: {total_time:.2f}s")

# Write PWL include file
pwl_file = os.path.join(os.path.dirname(__file__), 'pwave_ecg_pwl.inc')
with open(pwl_file, 'w') as f:
    f.write(f"* ECG Lead 1 PWL source for P-wave energy test\n")
    f.write(f"* 10 beats from MIT-BIH, gain={gain} V/mV, vref={vref}V\n")
    f.write(f"* Total time: {total_time:.2f}s\n")
    f.write(f"Vecg ecg_in 0 PWL(\n")
    f.write(f"+ 0 {vref:.6f}\n")
    for line in pwl_lines:
        f.write(line + "\n")
    f.write(f"+ {total_time:.6f} {vref:.6f})\n")

# Write pwave_gate PWL (HIGH during P-wave window for each beat)
gate_file = os.path.join(os.path.dirname(__file__), 'pwave_gate_pwl.inc')
with open(gate_file, 'w') as f:
    f.write(f"* P-wave gate signal — HIGH during P-wave window\n")
    f.write(f"Vpwgate pwave_gate 0 PWL(\n")
    f.write(f"+ 0 0\n")
    for bt in beat_timing:
        s = bt['pw_start']
        e = bt['pw_end']
        f.write(f"+ {s-0.0001:.6f} 0 {s:.6f} 0.9 {e:.6f} 0.9 {e+0.0001:.6f} 0\n")
    f.write(f"+ {total_time:.6f} 0)\n")

# Write rpeak_trigger PWL (2ms pulse at each R-peak)
rpeak_file = os.path.join(os.path.dirname(__file__), 'pwave_rpeak_pwl.inc')
with open(rpeak_file, 'w') as f:
    f.write(f"* R-peak trigger — 2ms pulse at each R-peak\n")
    f.write(f"Vrpeak rpeak_trigger 0 PWL(\n")
    f.write(f"+ 0 0\n")
    for bt in beat_timing:
        t = bt['rpeak_t']
        f.write(f"+ {t-0.0001:.6f} 0 {t:.6f} 0.9 {t+0.002:.6f} 0.9 {t+0.0021:.6f} 0\n")
    f.write(f"+ {total_time:.6f} 0)\n")

# Save timing info for the test bench
timing_file = os.path.join(os.path.dirname(__file__), 'pwave_timing.json')
with open(timing_file, 'w') as f:
    json.dump({
        'total_time': total_time,
        'vref': vref,
        'gain': gain,
        'beats': beat_timing,
    }, f, indent=2)

print(f"\nFiles written:")
print(f"  pwave_ground_truth.json")
print(f"  pwave_ecg_pwl.inc")
print(f"  pwave_gate_pwl.inc")
print(f"  pwave_rpeak_pwl.inc")
print(f"  pwave_timing.json")
