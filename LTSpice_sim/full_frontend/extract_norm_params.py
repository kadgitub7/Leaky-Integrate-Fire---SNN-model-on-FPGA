import numpy as np
import wfdb
import os
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import time

# ================================================================
# This script computes the normalization parameters (mean, std)
# for each feature group, in circuit-voltage domain.
#
# It loads ALL beats from MIT-BIH, maps them to circuit voltage
# using a FIXED gain (not per-beat adaptive), computes analog
# features, does the same train/test split, and exports the
# training mean/std grouped by feature type.
# ================================================================

# ---------- circuit parameters ----------
Vref = 0.9
VDD = 1.88
swing = 0.7          # ±0.7V around Vref
downsample = 8

# ---------- dataset parameters (must match snn_analog_behavorial.py) ----------
fs = 360
win_left = 90
win_right = 108
beat_len = win_left + win_right  # 198

all_record_names = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]

N = ['N', 'L', 'R', 'e', 'j']
S = ['A', 'a', 'J', 'S']
V = ['V', 'E']
F = ['F']
Q = ['/', 'f', 'Q']

aami_map = {}
for sym in N: aami_map[sym] = 0
for sym in S: aami_map[sym] = 1
for sym in V: aami_map[sym] = 2
for sym in F: aami_map[sym] = 3
for sym in Q: aami_map[sym] = 4

# ---------- filter functions ----------
def rc_lowpass_batch(signals, alpha):
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, signals, axis=1)

def rc_highpass_batch(signals, alpha):
    return signals - rc_lowpass_batch(signals, alpha)

# ---------- load all beats ----------
def extract_beats_from_records(rec_list):
    beats, labels = [], []
    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            for idx, label in zip(annotation.sample, annotation.symbol):
                if idx - win_left >= 0 and idx + win_right < len(signals) and label in aami_map:
                    beat_segment = signals[idx - win_left : idx + win_right, :]
                    beats.append(beat_segment)
                    labels.append(label)
        except Exception as e:
            print(f"Skipping {rec_id}: {e}")
    return np.array(beats), labels

t0 = time.time()
print("Loading all beats from MIT-BIH...")
all_beats_mV, all_raw_labels = extract_beats_from_records(all_record_names)
print(f"  {len(all_beats_mV)} beats, {all_beats_mV.shape[2]} leads ({time.time()-t0:.1f}s)")

num_leads = all_beats_mV.shape[2]

# ---------- map to circuit voltage with FIXED gain ----------
# Find global min/max across ALL beats and leads to set a fixed gain
global_min = all_beats_mV.min()
global_max = all_beats_mV.max()
print(f"  Global ECG range: [{global_min:.3f}, {global_max:.3f}] mV")

# Fixed linear mapping: V_circuit = Vref + (V_mV - center) * scale
ecg_center = (global_max + global_min) / 2.0
ecg_range = global_max - global_min
scale = 2.0 * swing / ecg_range  # maps full range to ±swing

print(f"  Fixed gain: {scale:.4f} V/mV (center: {ecg_center:.3f} mV)")
print(f"  Circuit range: [{Vref - swing:.3f}, {Vref + swing:.3f}] V")

# Apply mapping to all beats
all_beats_V = Vref + (all_beats_mV - ecg_center) * scale

# Verify range
print(f"  Mapped range: [{all_beats_V.min():.3f}, {all_beats_V.max():.3f}] V")

# ---------- compute features on circuit voltages ----------
def compute_analog_features_circuit(beats_V, num_leads):
    num_beats, length, _ = beats_V.shape
    all_features = []

    for lead in range(num_leads):
        signals = beats_V[:, :, lead]

        # Path 1: raw downsampled (25)
        all_features.append(signals[:, ::downsample])

        # Path 2: fast HP (25)
        fast_hp = rc_highpass_batch(signals, 0.8)
        all_features.append(fast_hp[:, ::downsample])

        # Path 3: slow HP (25)
        slow_hp = rc_highpass_batch(signals, 0.15)
        all_features.append(slow_hp[:, ::downsample])

        # Path 4: fast LP (25)
        smooth_fast = rc_lowpass_batch(signals, 0.5)
        all_features.append(smooth_fast[:, ::downsample])

        # Path 5: slow LP (25)
        smooth_slow = rc_lowpass_batch(signals, 0.08)
        all_features.append(smooth_slow[:, ::downsample])

        # Path 6: bandpass (25)
        bandpass = smooth_fast - smooth_slow
        all_features.append(bandpass[:, ::downsample])

        # Path 7: rectified HP (25)
        all_features.append(np.abs(fast_hp[:, ::downsample]))

        # Path 8: 2nd derivative (25)
        second_deriv = rc_highpass_batch(rc_highpass_batch(signals, 0.7), 0.7)
        all_features.append(second_deriv[:, ::downsample])

        # Path 9a-c: gated energy (1 each)
        abs_signals = np.abs(signals)
        all_features.append(np.sum(abs_signals[:, 80:120], axis=1, keepdims=True))
        all_features.append(np.sum(abs_signals[:, 40:80], axis=1, keepdims=True))
        all_features.append(np.sum(abs_signals[:, 120:170], axis=1, keepdims=True))

        # Path 10a-c: gated peak/valley/amp (1 each)
        qrs_max = np.max(signals[:, 80:120], axis=1, keepdims=True)
        qrs_min = np.min(signals[:, 80:120], axis=1, keepdims=True)
        all_features.append(qrs_max)
        all_features.append(qrs_min)
        all_features.append(qrs_max - qrs_min)

    return np.hstack(all_features)

t1 = time.time()
print("\nComputing analog features on circuit voltages...")
all_features = compute_analog_features_circuit(all_beats_V, num_leads)
print(f"  Done ({time.time()-t1:.1f}s)")
print(f"  Feature matrix: {all_features.shape} ({all_features.shape[1]} features)")

# ---------- train/test split (same as snn_analog_behavorial.py) ----------
all_numeric_labels = np.array([aami_map[l] for l in all_raw_labels])

train_idx, test_idx = train_test_split(
    np.arange(len(all_features)), test_size=0.2, random_state=42, stratify=all_numeric_labels
)

train_features = all_features[train_idx]
print(f"\n  Train: {len(train_features)}, Test: {len(test_idx)}")

# ---------- compute train mean and std ----------
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)

# ---------- group by feature type ----------
# Per lead, the feature order is:
#   Path 1: raw (25), Path 2: hp08 (25), Path 3: hp015 (25),
#   Path 4: lp05 (25), Path 5: lp008 (25), Path 6: bandpass (25),
#   Path 7: rect_hp (25), Path 8: deriv2 (25),
#   Path 9a: e_qrs (1), Path 9b: e_pre (1), Path 9c: e_post (1),
#   Path 10a: qrs_max (1), Path 10b: qrs_min (1), Path 10c: qrs_amp (1)
# Total per lead: 8*25 + 6 = 206

features_per_lead = 206

group_defs = [
    ("raw",        0,   25),
    ("hp08",      25,   50),
    ("hp015",     50,   75),
    ("lp05",      75,  100),
    ("lp008",    100,  125),
    ("bandpass",  125, 150),
    ("rect_hp",  150,  175),
    ("deriv2",   175,  200),
    ("energy_qrs", 200, 201),
    ("energy_pre", 201, 202),
    ("energy_post", 202, 203),
    ("qrs_max",   203, 204),
    ("qrs_min",   204, 205),
    ("qrs_amp",   205, 206),
]

print("\n" + "=" * 70)
print("NORMALIZATION PARAMETERS (circuit voltage domain)")
print("=" * 70)
print(f"{'Group':<14} {'Mean (V)':>10} {'Std (V)':>10} {'Gain=0.2/std':>14} {'R_fb (MΩ)':>12}")
print("-" * 70)

results = []

for lead in range(num_leads):
    if num_leads > 1:
        print(f"\n--- Lead {lead} ---")
    offset = lead * features_per_lead
    for name, start, end in group_defs:
        idxs = list(range(offset + start, offset + end))
        group_mean = train_mean[idxs].mean()
        group_std = train_std[idxs].mean()

        # Gain to map 3*std to 0.6V swing: G = 0.2 / std
        if group_std > 1e-6:
            gain = 0.2 / group_std
        else:
            gain = 1.0

        R_in = 1.0  # MΩ (fixed)
        R_fb = gain * R_in

        print(f"  {name:<14} {group_mean:>10.6f} {group_std:>10.6f} {gain:>14.4f} {R_fb:>12.4f}")

        results.append({
            'lead': lead,
            'name': name,
            'mean': group_mean,
            'std': group_std,
            'gain': gain,
            'R_in_MOhm': R_in,
            'R_fb_MOhm': R_fb,
        })

# ---------- export to file ----------
out_path = os.path.join('LTSpice_sim', 'full_frontend', 'norm_params.csv')
with open(out_path, 'w') as f:
    f.write("lead,group,mean_V,std_V,gain,R_in_MOhm,R_fb_MOhm\n")
    for r in results:
        f.write(f"{r['lead']},{r['name']},{r['mean']:.6f},{r['std']:.6f},"
                f"{r['gain']:.6f},{r['R_in_MOhm']:.4f},{r['R_fb_MOhm']:.4f}\n")

print(f"\nParameters saved to {out_path}")

# ---------- also export the fixed voltage mapping for PWL regeneration ----------
print(f"\n--- Fixed voltage mapping ---")
print(f"V_circuit = {Vref} + (V_mV - {ecg_center:.6f}) * {scale:.6f}")
print(f"To regenerate ecg_beat.txt with this fixed mapping,")
print(f"replace the per-beat normalization in ecg_pwl.py with:")
print(f"  beat_V = {Vref} + (beat_mV - {ecg_center:.6f}) * {scale:.6f}")

# Save mapping constants
map_path = os.path.join('LTSpice_sim', 'full_frontend', 'voltage_mapping.txt')
with open(map_path, 'w') as f:
    f.write(f"# Fixed voltage mapping: V_circuit = Vref + (V_mV - center) * scale\n")
    f.write(f"Vref = {Vref}\n")
    f.write(f"center_mV = {ecg_center:.6f}\n")
    f.write(f"scale = {scale:.6f}\n")
    f.write(f"global_min_mV = {global_min:.6f}\n")
    f.write(f"global_max_mV = {global_max:.6f}\n")

print(f"Voltage mapping saved to {map_path}")
