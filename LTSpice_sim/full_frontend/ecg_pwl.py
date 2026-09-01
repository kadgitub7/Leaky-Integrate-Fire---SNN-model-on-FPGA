import numpy as np
import wfdb
import os
from scipy.signal import lfilter

# ---------- parameters (must match your circuit) ----------
fs = 360            # MIT-BIH sample rate (Hz)
win_left = 90       # samples before R-peak
win_right = 108     # samples after R-peak
beat_len = win_left + win_right  # 198 samples

Vref = 0.9           # circuit midpoint (V)
VDD = 1.88           # supply voltage (V)
center_mV = -0.002500
scale = 0.136786     # fixed gain from extract_norm_params.py (global min/max)

downsample = 8       # same as Python feature code

# ---------- load one beat ----------
record_path = os.path.join('./mitdb_data', '100')
record = wfdb.rdrecord(record_path)
annotation = wfdb.rdann(record_path, 'atr')

signals = record.p_signal  # shape: (num_samples, num_leads), in mV

# find first valid beat
for idx in annotation.sample:
    if idx - win_left >= 0 and idx + win_right < len(signals):
        beat_mV = signals[idx - win_left : idx + win_right, 0]  # lead 0
        break

print(f"Beat: {beat_len} samples, range [{beat_mV.min():.3f}, {beat_mV.max():.3f}] mV")

# ---------- map to circuit voltage (fixed gain) ----------
# Uses global min/max from ALL MIT-BIH beats (extract_norm_params.py)
# so the mapping is identical to what was used for training statistics.
beat_V = Vref + (beat_mV - center_mV) * scale

print(f"Circuit voltage range: [{beat_V.min():.3f}, {beat_V.max():.3f}] V")

# ---------- write PWL file ----------
T_sample = 1.0 / fs  # 2.778 ms

pwl_path = os.path.join('LTSpice_sim', 'full_frontend', 'ecg_beat.txt')
with open(pwl_path, 'w') as f:
    for i, v in enumerate(beat_V):
        t = i * T_sample
        f.write(f"{t:.6e} {v:.6f}\n")

print(f"PWL written to {pwl_path} ({len(beat_V)} points)")

# ---------- compute reference features ----------
def rc_lowpass(signal_1d, alpha):
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, signal_1d)

def rc_highpass(signal_1d, alpha):
    return signal_1d - rc_lowpass(signal_1d, alpha)

# Use the CIRCUIT voltages (beat_V) as input, since that's what LTSpice sees
sig = beat_V

# Path 1: raw downsampled
path1 = sig[::downsample]
print(f"\nPath 1 (raw downsampled): {len(path1)} values")
print(f"  Values: {np.array2string(path1, precision=4, max_line_width=120)}")

# Path 2: fast highpass (alpha=0.8), downsampled
fast_hp = rc_highpass(sig, 0.8)
path2 = fast_hp[::downsample]
print(f"\nPath 2 (HP a=0.8): {len(path2)} values")
print(f"  Values: {np.array2string(path2, precision=4, max_line_width=120)}")

# Path 3: slow highpass (alpha=0.15), downsampled
slow_hp = rc_highpass(sig, 0.15)
path3 = slow_hp[::downsample]
print(f"\nPath 3 (HP a=0.15): {len(path3)} values")
print(f"  Values: {np.array2string(path3, precision=4, max_line_width=120)}")

# Path 4: fast lowpass (alpha=0.5), downsampled
smooth_fast = rc_lowpass(sig, 0.5)
path4 = smooth_fast[::downsample]
print(f"\nPath 4 (LP a=0.5): {len(path4)} values")
print(f"  Values: {np.array2string(path4, precision=4, max_line_width=120)}")

# Path 5: slow lowpass (alpha=0.08), downsampled
smooth_slow = rc_lowpass(sig, 0.08)
path5 = smooth_slow[::downsample]
print(f"\nPath 5 (LP a=0.08): {len(path5)} values")
print(f"  Values: {np.array2string(path5, precision=4, max_line_width=120)}")

# Path 6: bandpass (LP_0.5 - LP_0.08), downsampled
bandpass = smooth_fast - smooth_slow
path6 = bandpass[::downsample]
print(f"\nPath 6 (bandpass): {len(path6)} values")
print(f"  Values: {np.array2string(path6, precision=4, max_line_width=120)}")

# Path 7: rectified fast HP, downsampled
path7 = np.abs(fast_hp[::downsample])
print(f"\nPath 7 (|HP a=0.8|): {len(path7)} values")
print(f"  Values: {np.array2string(path7, precision=4, max_line_width=120)}")

# Path 8: second derivative (HP_0.7 cascaded twice), downsampled
second_deriv = rc_highpass(rc_highpass(sig, 0.7), 0.7)
path8 = second_deriv[::downsample]
print(f"\nPath 8 (2nd deriv): {len(path8)} values")
print(f"  Values: {np.array2string(path8, precision=4, max_line_width=120)}")

# Path 9: gated energy integrals
abs_sig = np.abs(sig)
energy_qrs = np.sum(abs_sig[80:120])
energy_pre = np.sum(abs_sig[40:80])
energy_post = np.sum(abs_sig[120:170])
print(f"\nPath 9 (gated energy):")
print(f"  QRS (80-120): {energy_qrs:.4f}")
print(f"  PRE (40-80):  {energy_pre:.4f}")
print(f"  POST (120-170): {energy_post:.4f}")

# Path 10: gated peak/valley/amplitude (QRS window only)
qrs_max = np.max(sig[80:120])
qrs_min = np.min(sig[80:120])
qrs_amp = qrs_max - qrs_min
print(f"\nPath 10 (gated peak/valley):")
print(f"  Max (peak):  {qrs_max:.4f} V")
print(f"  Min (valley): {qrs_min:.4f} V")
print(f"  Amplitude:   {qrs_amp:.4f} V")

# ---------- also save reference values to a file ----------
ref_path = os.path.join('LTSpice_sim', 'full_frontend', 'python_reference.txt')
with open(ref_path, 'w') as f:
    f.write("# Python reference features for ECG beat from record 100\n")
    f.write(f"# Input voltage range: [{beat_V.min():.4f}, {beat_V.max():.4f}] V\n\n")
    
    for name, values in [("path1_raw", path1), ("path2_hp08", path2), 
                          ("path3_hp015", path3), ("path4_lp05", path4),
                          ("path5_lp008", path5), ("path6_bandpass", path6),
                          ("path7_rect_hp", path7), ("path8_2nd_deriv", path8)]:
        f.write(f"{name}: {np.array2string(values, precision=6, separator=', ', max_line_width=200)}\n")
    
    f.write(f"\nenergy_qrs: {energy_qrs:.6f}\n")
    f.write(f"energy_pre: {energy_pre:.6f}\n")
    f.write(f"energy_post: {energy_post:.6f}\n")
    f.write(f"qrs_max: {qrs_max:.6f}\n")
    f.write(f"qrs_min: {qrs_min:.6f}\n")
    f.write(f"qrs_amp: {qrs_amp:.6f}\n")

print(f"\nReference values saved to {ref_path}")

