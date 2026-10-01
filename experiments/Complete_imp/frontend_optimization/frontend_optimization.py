"""
FRONTEND OPTIMIZATION: Comprehensive feature evaluation for analog ECG classification
======================================================================================
Evaluates CURRENT features vs PROPOSED novel features on real MIT-BIH data.

Novel features under test:
  A. P-wave energy (bandpass energy in 150ms pre-QRS window)
  B. ST-T discordance (sign(QRS_integral) * ST_integral)
  C. QRS symmetry ratio (time_to_peak / qrs_width)
  D. Inter-lead morphology divergence
  E. Beat-to-beat amplitude instability

Metrics computed:
  1. ANOVA F-score (linear separability)
  2. Mutual information (non-linear discriminative power)
  3. Random Forest importance (interaction-aware)
  4. Per-class distribution stats (means, stds, overlap)
  5. Pairwise correlation (redundancy detection)
  6. Subset ablation (accuracy vs feature count tradeoff)
  7. Per-class confusion analysis (which features fix which errors)

Run: python experiments/frontend_optimization.py
"""

import numpy as np
import wfdb
import os
from collections import Counter
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONSTANTS
# ================================================================

aami_map = {}
for sym in ['N', 'L', 'R', 'e', 'j']: aami_map[sym] = 0
for sym in ['A', 'a', 'J', 'S']: aami_map[sym] = 1
for sym in ['V', 'E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/', 'f', 'Q']: aami_map[sym] = 4

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']

FS = 360
EWMA_ALPHA = 0.015
N_TEMPLATE = 6
EWMA_GATE = 0.85
EWMA_INIT = 10

# Beat window: 90 samples left, 108 samples right of R-peak
WIN_L = 90
WIN_R = 108

# QRS region within the beat window (centered on R-peak)
QRS_S = 70   # QRS start index within beat
QRS_E = 130  # QRS end index within beat

# P-wave region: 150ms before QRS onset
# At 360 Hz, 150ms = 54 samples. QRS onset is at index QRS_S=70 in beat window.
# P-wave window: beat[16:70] (54 samples before QRS)
PW_S = 16
PW_E = QRS_S

# ST segment: starts ~40ms after QRS end, lasts ~80ms
# QRS end = index 130, +14 samples (40ms) = 144, +29 samples (80ms) = 173
# But beat window is only 198 samples (90+108), so ST region: beat[144:173]
ST_S = 144
ST_E = min(173, WIN_L + WIN_R)

# Bandpass filter for P-wave extraction (3-8 Hz)
def design_pwave_filter(fs=360):
    nyq = fs / 2
    low = 3.0 / nyq
    high = 8.0 / nyq
    b, a = butter(2, [low, high], btype='band')
    return b, a

PW_B, PW_A = design_pwave_filter(FS)


# ================================================================
# ANALOG COST MODEL
# ================================================================
# Cost units: relative analog circuit complexity
# 1 unit = 1 comparator or 1 timer
# Based on OTA-C filter costs, integrator costs, etc.

ANALOG_COST = {
    # Timing features
    'pre_rr': 1,              # timer
    'post_rr': 1,             # timer (same circuit, different readout)
    'rr_ratio': 3,            # timer + EWMA + divider
    'rr_asymmetry': 2,        # two timers + divider
    'compensatory_ratio': 3,  # two timers + EWMA + divider
    'rr_std_10': 5,           # requires variance computation over window
    'rr_std_20': 5,

    # Raw morphology (per lead)
    'qrs_width_L0': 2, 'qrs_width_L1': 2,    # comparator + timer
    'qrs_area_L0': 2, 'qrs_area_L1': 2,      # gated integrator
    'max_slope_L0': 2, 'max_slope_L1': 2,     # differentiator + peak detector
    'peak_amp_L0': 1, 'peak_amp_L1': 1,       # peak detector
    'qrs_energy_L0': 2, 'qrs_energy_L1': 2,   # squarer + integrator
    'slope_ratio_L0': 4, 'slope_ratio_L1': 4,  # two peak detectors + divider

    # Patient-adaptive (per lead) — need EWMA + storage
    'rel_area_L0': 5, 'rel_area_L1': 5,       # integrator + EWMA + divider
    'rel_width_L0': 5, 'rel_width_L1': 5,
    'rel_peak_L0': 5, 'rel_peak_L1': 5,
    'templ_corr_L0': 12, 'templ_corr_L1': 12, # 6-pt template in ReRAM + dot product

    # Cross-lead
    'area_ratio_L0L1': 3,     # two integrators (shared) + divider
    'width_diff_L0L1': 3,
    'corr_L0L1': 6,           # full QRS correlation between leads

    # NOVEL features
    'pwave_energy_L0': 4,     # BPF (OTA-C) + gated integrator
    'pwave_energy_L1': 4,
    'st_discordance_L0': 4,   # two gated integrators + sign detection
    'st_discordance_L1': 4,
    'qrs_symmetry_L0': 3,     # comparator + two timers
    'qrs_symmetry_L1': 3,
    'interlead_divergence': 3, # two integrators (shared) + divider
    'beat_instability_L0': 3,  # peak detector + S&H + subtractor + divider
    'beat_instability_L1': 3,
}


# ================================================================
# FEATURE EXTRACTION — ALL FEATURES (current + novel)
# ================================================================

ALL_FEATURE_NAMES = [
    # Timing (7)
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio',
    'rr_std_10', 'rr_std_20',
    # Raw morphology per lead (12)
    'qrs_width_L0', 'qrs_width_L1',
    'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1',
    'peak_amp_L0', 'peak_amp_L1',
    'qrs_energy_L0', 'qrs_energy_L1',
    'slope_ratio_L0', 'slope_ratio_L1',
    # Patient-adaptive per lead (8)
    'rel_area_L0', 'rel_area_L1',
    'rel_width_L0', 'rel_width_L1',
    'rel_peak_L0', 'rel_peak_L1',
    'templ_corr_L0', 'templ_corr_L1',
    # Cross-lead (3)
    'area_ratio_L0L1', 'width_diff_L0L1', 'corr_L0L1',
    # NOVEL: P-wave energy (2)
    'pwave_energy_L0', 'pwave_energy_L1',
    # NOVEL: ST-T discordance (2)
    'st_discordance_L0', 'st_discordance_L1',
    # NOVEL: QRS symmetry (2)
    'qrs_symmetry_L0', 'qrs_symmetry_L1',
    # NOVEL: Inter-lead divergence (1)
    'interlead_divergence',
    # NOVEL: Beat-to-beat instability (2)
    'beat_instability_L0', 'beat_instability_L1',
]

N_FEAT = len(ALL_FEATURE_NAMES)


def extract_all_features(rec_list):
    """Extract all current + novel features from MIT-BIH records."""
    all_labels = []
    all_features = []
    all_record_ids = []

    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            fs = record.fs
            num_leads = min(signals.shape[1], 2)

            # Pre-filter entire signal for P-wave extraction
            pwave_filtered = np.zeros_like(signals)
            for lead in range(num_leads):
                try:
                    pwave_filtered[:, lead] = filtfilt(PW_B, PW_A, signals[:, lead])
                except Exception:
                    pass

            valid = []
            for idx, sym in zip(annotation.sample, annotation.symbol):
                if idx - WIN_L >= 0 and idx + WIN_R < len(signals) and sym in aami_map:
                    valid.append((idx, sym))

            # EWMA state per lead
            ewma_area = [None] * num_leads
            ewma_width = [None] * num_leads
            ewma_peak = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_widths = [[] for _ in range(num_leads)]
            init_peaks = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []
            prev_peak = [None] * num_leads

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - WIN_L : idx + WIN_R, :]
                beat_pw = pwave_filtered[idx - WIN_L : idx + WIN_R, :]
                label = aami_map[sym]

                # --- TIMING ---
                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - 20), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833

                rr_history.append(pre_rr)
                if len(rr_history) > 20:
                    rr_history.pop(0)

                feat = np.zeros(N_FEAT, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history[-10:]) if len(rr_history) >= 2 else 0.0
                feat[6] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                # --- PER-LEAD FEATURES ---
                qrs_signals = []
                widths = []
                areas = []
                peaks_amp = []

                for lead in range(num_leads):
                    qrs = beat[QRS_S:QRS_E, lead]
                    abs_qrs = np.abs(qrs)
                    peak = np.max(abs_qrs) if len(abs_qrs) > 0 else 0.0
                    area = np.sum(abs_qrs) / fs

                    # QRS width (30% threshold crossing)
                    width = 0.0
                    if peak > 1e-6:
                        threshold = 0.3 * peak
                        above = abs_qrs > threshold
                        if np.any(above):
                            first = np.argmax(above)
                            last = len(above) - 1 - np.argmax(above[::-1])
                            width = (last - first) * 1000.0 / fs

                    dqrs = np.diff(qrs) * fs
                    max_slope = np.max(np.abs(dqrs)) if len(dqrs) > 0 else 0.0
                    energy = np.sum(qrs**2) / fs

                    if len(dqrs) > 0:
                        up = max(np.max(dqrs), 1e-8)
                        down = max(abs(np.min(dqrs)), 1e-8)
                        slope_ratio = up / (down + 1e-8)
                    else:
                        slope_ratio = 1.0

                    # Template correlation (EWMA)
                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]

                    if i < EWMA_INIT:
                        init_areas[lead].append(max(area, 1e-6))
                        init_widths[lead].append(max(width, 1e-6))
                        init_peaks[lead].append(max(peak, 1e-6))
                        init_templates[lead].append(qrs_ds.copy())

                    if ewma_area[lead] is None:
                        if i >= EWMA_INIT - 1 and len(init_areas[lead]) >= EWMA_INIT:
                            ewma_area[lead] = np.median(init_areas[lead])
                            ewma_width[lead] = np.median(init_widths[lead])
                            ewma_peak[lead] = np.median(init_peaks[lead])
                            ewma_template[lead] = np.median(init_templates[lead], axis=0)
                        else:
                            ewma_area[lead] = max(area, 1e-6)
                            ewma_width[lead] = max(width, 1e-6)
                            ewma_peak[lead] = max(peak, 1e-6)
                            ewma_template[lead] = qrs_ds.copy()

                    norm_c = np.linalg.norm(qrs_ds)
                    norm_t = np.linalg.norm(ewma_template[lead])
                    tc = np.dot(qrs_ds, ewma_template[lead]) / (norm_c * norm_t + 1e-8) if norm_c > 1e-8 and norm_t > 1e-8 else 1.0

                    if tc > EWMA_GATE:
                        a = EWMA_ALPHA
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_width[lead] = a * max(width, 1e-6) + (1 - a) * ewma_width[lead]
                        ewma_peak[lead] = a * max(peak, 1e-6) + (1 - a) * ewma_peak[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

                    # Store raw morphology
                    feat[7 + lead] = width
                    feat[9 + lead] = area
                    feat[11 + lead] = max_slope
                    feat[13 + lead] = peak
                    feat[15 + lead] = energy
                    feat[17 + lead] = slope_ratio

                    # Patient-adaptive
                    feat[19 + lead] = area / (ewma_area[lead] + 1e-8)
                    feat[21 + lead] = width / (ewma_width[lead] + 1e-8)
                    feat[23 + lead] = peak / (ewma_peak[lead] + 1e-8)
                    feat[25 + lead] = tc

                    qrs_signals.append(qrs)
                    widths.append(width)
                    areas.append(area)
                    peaks_amp.append(peak)

                    # ---- NOVEL: P-wave energy ----
                    pw_segment = beat_pw[PW_S:PW_E, lead]
                    pwave_energy = np.sum(pw_segment**2) / fs
                    feat[30 + lead] = pwave_energy

                    # ---- NOVEL: ST-T discordance ----
                    qrs_integral = np.sum(qrs) / fs  # signed integral
                    st_end = min(ST_E, beat.shape[0])
                    st_segment = beat[ST_S:st_end, lead]
                    st_integral = np.sum(st_segment) / fs if len(st_segment) > 0 else 0.0
                    qrs_sign = 1.0 if qrs_integral >= 0 else -1.0
                    feat[32 + lead] = qrs_sign * st_integral

                    # ---- NOVEL: QRS symmetry ----
                    if peak > 1e-6:
                        peak_idx = np.argmax(abs_qrs)
                        threshold = 0.3 * peak
                        above = abs_qrs > threshold
                        if np.any(above):
                            onset = np.argmax(above)
                            qrs_w = max(width, 1e-6) * fs / 1000.0  # back to samples
                            time_to_peak = peak_idx - onset
                            symmetry = time_to_peak / (qrs_w + 1e-8)
                            symmetry = np.clip(symmetry, 0.0, 1.0)
                        else:
                            symmetry = 0.5
                    else:
                        symmetry = 0.5
                    feat[34 + lead] = symmetry

                    # ---- NOVEL: Beat-to-beat instability ----
                    if prev_peak[lead] is not None and prev_peak[lead] > 1e-6:
                        instability = abs(peak - prev_peak[lead]) / (prev_peak[lead] + 1e-8)
                    else:
                        instability = 0.0
                    feat[37 + lead] = instability
                    prev_peak[lead] = peak

                # --- CROSS-LEAD FEATURES ---
                if num_leads == 2:
                    feat[27] = areas[0] / (areas[1] + 1e-8)
                    feat[28] = abs(widths[0] - widths[1])
                    n0 = np.linalg.norm(qrs_signals[0])
                    n1 = np.linalg.norm(qrs_signals[1])
                    feat[29] = np.dot(qrs_signals[0], qrs_signals[1]) / (n0 * n1 + 1e-8) if n0 > 1e-8 and n1 > 1e-8 else 0.0

                    # NOVEL: Inter-lead divergence
                    a0, a1 = areas[0], areas[1]
                    feat[36] = (a0 - a1)**2 / (a0**2 + a1**2 + 1e-8)

                all_labels.append(label)
                all_features.append(feat)
                all_record_ids.append(rec_id)

        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features), all_record_ids


# ================================================================
# EXTRACT DATA
# ================================================================

print("=" * 75)
print("FRONTEND OPTIMIZATION: Comprehensive Feature Evaluation")
print("=" * 75)
print(f"\nExtracting {N_FEAT} features (current + novel) from MIT-BIH...")

train_labels, train_features, train_recs = extract_all_features(DS1)
test_labels, test_features, test_recs = extract_all_features(DS2)

print(f"  DS1 (train): {len(train_labels)} beats")
print(f"  DS2 (test):  {len(test_labels)} beats")
print(f"  Features: {N_FEAT}")
print(f"  Train class dist: {dict(sorted(Counter(train_labels).items()))}")
print(f"  Test class dist:  {dict(sorted(Counter(test_labels).items()))}")

# Normalize
tr_mean = train_features.mean(axis=0)
tr_std = train_features.std(axis=0)
dead_features = tr_std < 1e-10
if np.any(dead_features):
    dead_names = [ALL_FEATURE_NAMES[i] for i in np.where(dead_features)[0]]
    print(f"\n  WARNING: Near-zero variance features: {dead_names}")
train_norm = (train_features - tr_mean) / (tr_std + 1e-8)
test_norm = (test_features - tr_mean) / (tr_std + 1e-8)


# ================================================================
# 1. ANOVA F-SCORES
# ================================================================

print(f"\n{'=' * 75}")
print("1. ANOVA F-SCORES (higher = more discriminative between classes)")
print(f"{'=' * 75}")
f_scores, p_values = f_classif(train_norm, train_labels)
anova_ranked = sorted(zip(ALL_FEATURE_NAMES, f_scores, p_values), key=lambda x: -x[1])
for i, (name, f, p) in enumerate(anova_ranked):
    bar = '#' * min(int(f / (max(f_scores) + 1e-8) * 40), 40)
    novel = " *NOVEL*" if name in ['pwave_energy_L0','pwave_energy_L1','st_discordance_L0','st_discordance_L1','qrs_symmetry_L0','qrs_symmetry_L1','interlead_divergence','beat_instability_L0','beat_instability_L1'] else ""
    cost = ANALOG_COST.get(name, '?')
    print(f"  {i+1:2d}. {name:25s} F={f:8.1f}  cost={cost:2}  {bar}{novel}")


# ================================================================
# 2. MUTUAL INFORMATION
# ================================================================

print(f"\n{'=' * 75}")
print("2. MUTUAL INFORMATION (non-linear discriminative power)")
print(f"{'=' * 75}")
mi = mutual_info_classif(train_norm, train_labels, random_state=42, n_neighbors=5)
mi_ranked = sorted(zip(ALL_FEATURE_NAMES, mi), key=lambda x: -x[1])
for i, (name, m) in enumerate(mi_ranked):
    bar = '#' * min(int(m / (max(mi) + 1e-8) * 40), 40)
    novel = " *NOVEL*" if 'pwave' in name or 'st_disc' in name or 'symmetry' in name or 'divergence' in name or 'instability' in name else ""
    cost = ANALOG_COST.get(name, '?')
    print(f"  {i+1:2d}. {name:25s} MI={m:.4f}  cost={cost:2}  {bar}{novel}")


# ================================================================
# 3. RANDOM FOREST IMPORTANCE
# ================================================================

print(f"\n{'=' * 75}")
print("3. RANDOM FOREST IMPORTANCE (inter-patient, interaction-aware)")
print(f"{'=' * 75}")
rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1,
                            class_weight='balanced')
rf.fit(train_norm, train_labels)
rf_preds = rf.predict(test_norm)
rf_acc = np.mean(rf_preds == test_labels) * 100
print(f"  RF accuracy (all {N_FEAT} features, inter-patient): {rf_acc:.2f}%")
print(classification_report(test_labels, rf_preds, target_names=CLASS_NAMES))

importances = rf.feature_importances_
rf_ranked = sorted(zip(ALL_FEATURE_NAMES, importances), key=lambda x: -x[1])
print(f"  Feature importance ranking:")
for i, (name, imp) in enumerate(rf_ranked):
    bar = '#' * min(int(imp / (max(importances) + 1e-8) * 40), 40)
    novel = " *NOVEL*" if 'pwave' in name or 'st_disc' in name or 'symmetry' in name or 'divergence' in name or 'instability' in name else ""
    cost = ANALOG_COST.get(name, '?')
    print(f"  {i+1:2d}. {name:25s} imp={imp:.4f}  cost={cost:2}  {bar}{novel}")


# ================================================================
# 4. PER-CLASS DISTRIBUTION ANALYSIS
# ================================================================

print(f"\n{'=' * 75}")
print("4. PER-CLASS FEATURE DISTRIBUTIONS (normalized means and separation)")
print(f"{'=' * 75}")

# Show top 20 features by RF importance
top_20_rf = [name for name, _ in rf_ranked[:25]]
top_20_idx = [ALL_FEATURE_NAMES.index(name) for name in top_20_rf]

print(f"\n  {'Feature':25s} {'N':>8} {'S':>8} {'V':>8} {'F':>8} {'Q':>8}  {'Best sep':20s} {'Cost':>4}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20} {'-'*4}")
for name, idx in zip(top_20_rf, top_20_idx):
    means = []
    for cls in range(5):
        mask = train_labels == cls
        if mask.sum() > 0:
            means.append(train_norm[mask, idx].mean())
        else:
            means.append(0.0)
    means = np.array(means)
    max_sep = 0
    best_pair = ""
    for a in range(5):
        for b in range(a+1, 5):
            sep = abs(means[a] - means[b])
            if sep > max_sep:
                max_sep = sep
                best_pair = f"{CLASS_NAMES[a]} vs {CLASS_NAMES[b]}"
    cost = ANALOG_COST.get(name, '?')
    novel = "*" if 'pwave' in name or 'st_disc' in name or 'symmetry' in name or 'divergence' in name or 'instability' in name else " "
    print(f" {novel}{name:25s} {means[0]:8.3f} {means[1]:8.3f} {means[2]:8.3f} {means[3]:8.3f} {means[4]:8.3f}  {best_pair:20s} {cost:>4}")


# ================================================================
# 5. CORRELATION MATRIX (redundancy)
# ================================================================

print(f"\n{'=' * 75}")
print("5. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.80)")
print(f"{'=' * 75}")
corr = np.corrcoef(train_norm.T)
pairs = []
for i in range(N_FEAT):
    for j in range(i+1, N_FEAT):
        r = corr[i, j]
        if abs(r) > 0.80:
            pairs.append((ALL_FEATURE_NAMES[i], ALL_FEATURE_NAMES[j], r))
pairs.sort(key=lambda x: -abs(x[2]))
for n1, n2, r in pairs:
    print(f"  {n1:25s} <-> {n2:25s}  r={r:+.3f}")
if not pairs:
    print("  No highly correlated pairs found.")


# ================================================================
# 6. FEATURE VALUE-PER-COST RANKING
# ================================================================

print(f"\n{'=' * 75}")
print("6. VALUE-PER-COST RANKING (RF importance / analog cost)")
print(f"{'=' * 75}")
value_per_cost = []
for name, imp in rf_ranked:
    cost = ANALOG_COST.get(name, 10)
    vpc = imp / cost
    value_per_cost.append((name, imp, cost, vpc))
value_per_cost.sort(key=lambda x: -x[3])
print(f"  {'Rank':>4} {'Feature':25s} {'RF imp':>8} {'Cost':>5} {'Value/Cost':>10}")
print(f"  {'-'*4} {'-'*25} {'-'*8} {'-'*5} {'-'*10}")
for i, (name, imp, cost, vpc) in enumerate(value_per_cost):
    novel = "*" if 'pwave' in name or 'st_disc' in name or 'symmetry' in name or 'divergence' in name or 'instability' in name else " "
    print(f"  {i+1:4d} {novel}{name:24s} {imp:8.4f} {cost:5d} {vpc:10.5f}")


# ================================================================
# 7. SUBSET ABLATION — RF accuracy with different feature sets
# ================================================================

print(f"\n{'=' * 75}")
print("7. FEATURE SUBSET ABLATION (RF inter-patient accuracy)")
print(f"{'=' * 75}")

def rf_eval(feature_names, label, detailed=False):
    indices = [ALL_FEATURE_NAMES.index(f) for f in feature_names]
    _rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1,
                                 class_weight='balanced')
    _rf.fit(train_norm[:, indices], train_labels)
    preds = _rf.predict(test_norm[:, indices])
    acc = np.mean(preds == test_labels) * 100
    f1s = f1_score(test_labels, preds, average=None, zero_division=0)
    total_cost = sum(ANALOG_COST.get(f, 10) for f in feature_names)
    print(f"  {label:50s} {len(feature_names):2d}f  cost={total_cost:3d}  acc={acc:.2f}%  F1: N={f1s[0]:.2f} S={f1s[1]:.2f} V={f1s[2]:.2f} F={f1s[3]:.2f} Q={f1s[4]:.2f}")
    if detailed:
        cm = confusion_matrix(test_labels, preds)
        print(f"    Confusion matrix:")
        print(f"    {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
        for ii, lbl in enumerate(CLASS_NAMES):
            print(f"    {lbl:>5} {' '.join(f'{v:>6}' for v in cm[ii])}")
    return acc, f1s, total_cost

print("\n  --- Reference sets ---")
# Current 16 features (from snn_best_condensed.py)
current_16 = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1', 'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1', 'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
]
rf_eval(current_16, "Current 16 (production)", detailed=True)

# Current 14 features (from shared_data.py)
current_14 = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'rr_std_10',
    'qrs_area_L0', 'max_slope_L0', 'slope_ratio_L0',
    'rel_area_L0', 'templ_corr_L0', 'rel_peak_L0',
    'rel_area_L1', 'templ_corr_L1',
    'corr_L0L1',
]
rf_eval(current_14, "Current 14 (shared_data.py)", detailed=True)

# All features
rf_eval(ALL_FEATURE_NAMES, "All features (kitchen sink)")

print("\n  --- Proposed 7-feature set (from plan) ---")
proposed_7 = [
    'rr_ratio',            # RR prematurity
    'compensatory_ratio',  # compensatory index
    'qrs_width_L0',        # QRS width
    'pwave_energy_L0',     # P-wave energy
    'st_discordance_L0',   # ST-T discordance
    'qrs_symmetry_L0',     # QRS symmetry
    'interlead_divergence', # inter-lead divergence
]
rf_eval(proposed_7, "Proposed 7 (from plan)", detailed=True)

print("\n  --- Proposed 7 + augmented variants ---")
proposed_7_plus_lead1 = proposed_7 + ['pwave_energy_L1', 'st_discordance_L1', 'qrs_symmetry_L1']
rf_eval(proposed_7_plus_lead1, "Proposed 7 + lead1 novel (10)")

proposed_7_plus_instab = proposed_7 + ['beat_instability_L0', 'beat_instability_L1']
rf_eval(proposed_7_plus_instab, "Proposed 7 + instability (9)")

proposed_full_novel = proposed_7 + ['pwave_energy_L1', 'st_discordance_L1', 'qrs_symmetry_L1',
                                     'beat_instability_L0', 'beat_instability_L1']
rf_eval(proposed_full_novel, "Proposed 7 + all novel extras (12)")

print("\n  --- Hybrid: best current + best novel ---")
# Take the top current features and add novel ones
hybrid_1 = ['pre_rr', 'rr_ratio', 'rr_asymmetry',
            'templ_corr_L0', 'templ_corr_L1', 'rel_area_L0',
            'pwave_energy_L0', 'st_discordance_L0', 'qrs_symmetry_L0']
rf_eval(hybrid_1, "Hybrid A: top3 timing + templ + novel (9)")

hybrid_2 = ['rr_ratio', 'rr_asymmetry', 'compensatory_ratio',
            'qrs_width_L0', 'rel_area_L0', 'templ_corr_L0',
            'pwave_energy_L0', 'st_discordance_L0', 'interlead_divergence']
rf_eval(hybrid_2, "Hybrid B: timing + width + adaptive + novel (9)")

hybrid_3 = ['rr_ratio', 'compensatory_ratio',
            'qrs_width_L0', 'templ_corr_L0',
            'pwave_energy_L0', 'st_discordance_L0',
            'qrs_symmetry_L0', 'interlead_divergence',
            'beat_instability_L0']
rf_eval(hybrid_3, "Hybrid C: proposed7 + templ_corr + instab (9)")

print("\n  --- Novel-only (no current features) ---")
novel_only = ['pwave_energy_L0', 'pwave_energy_L1',
              'st_discordance_L0', 'st_discordance_L1',
              'qrs_symmetry_L0', 'qrs_symmetry_L1',
              'interlead_divergence',
              'beat_instability_L0', 'beat_instability_L1']
rf_eval(novel_only, "Novel features only (9)")

print("\n  --- Timing-only baselines ---")
timing_2 = ['pre_rr', 'post_rr']
rf_eval(timing_2, "Timing only (2)")
timing_3 = ['rr_ratio', 'rr_asymmetry', 'compensatory_ratio']
rf_eval(timing_3, "Timing ratios only (3)")

print("\n  --- Greedy forward selection (value-per-cost) ---")
# Greedy selection: add feature that gives best acc/cost improvement
remaining_vpc = [name for name, _, _, _ in value_per_cost]
selected_greedy = []
for name in remaining_vpc:
    idx = ALL_FEATURE_NAMES.index(name)
    redundant = False
    for sel in selected_greedy:
        sel_idx = ALL_FEATURE_NAMES.index(sel)
        if abs(corr[idx, sel_idx]) > 0.85:
            redundant = True
            break
    if not redundant:
        selected_greedy.append(name)
        if len(selected_greedy) <= 15:
            rf_eval(selected_greedy, f"VPC greedy top {len(selected_greedy)}: +{name}")


# ================================================================
# 8. PER-CLASS ERROR ANALYSIS
# ================================================================

print(f"\n{'=' * 75}")
print("8. PER-CLASS ERROR ANALYSIS — Which novel features fix which errors?")
print(f"{'=' * 75}")

# Train RF with current-16, get misclassified beats
idx_c16 = [ALL_FEATURE_NAMES.index(f) for f in current_16]
rf_c16 = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1,
                                class_weight='balanced')
rf_c16.fit(train_norm[:, idx_c16], train_labels)
preds_c16 = rf_c16.predict(test_norm[:, idx_c16])

# For each class, show feature values of correctly vs incorrectly classified beats
novel_feat_names = ['pwave_energy_L0', 'st_discordance_L0', 'qrs_symmetry_L0',
                    'interlead_divergence', 'beat_instability_L0']
novel_indices = [ALL_FEATURE_NAMES.index(f) for f in novel_feat_names]

for cls in range(5):
    cls_mask = test_labels == cls
    if cls_mask.sum() == 0:
        continue
    correct_mask = cls_mask & (preds_c16 == cls)
    wrong_mask = cls_mask & (preds_c16 != cls)

    n_correct = correct_mask.sum()
    n_wrong = wrong_mask.sum()
    if n_wrong == 0:
        print(f"\n  {CLASS_NAMES[cls]}-class: {n_correct} correct, 0 wrong — nothing to fix")
        continue

    print(f"\n  {CLASS_NAMES[cls]}-class: {n_correct} correct, {n_wrong} wrong ({n_wrong/(n_correct+n_wrong)*100:.1f}% error rate)")
    print(f"    {'Novel feature':25s} {'Correct mean':>12} {'Wrong mean':>12} {'Separation':>10}")
    for fname, fidx in zip(novel_feat_names, novel_indices):
        if correct_mask.sum() > 0:
            cm = test_norm[correct_mask, fidx].mean()
        else:
            cm = 0
        wm = test_norm[wrong_mask, fidx].mean()
        sep = abs(cm - wm)
        marker = " <<<" if sep > 0.3 else ""
        print(f"    {fname:25s} {cm:12.3f} {wm:12.3f} {sep:10.3f}{marker}")

    # Show what classes they're confused with
    wrong_preds = preds_c16[wrong_mask]
    confusion = Counter(wrong_preds)
    print(f"    Misclassified as: {dict(zip([CLASS_NAMES[k] for k in confusion.keys()], confusion.values()))}")


# ================================================================
# 9. BUNDLE BRANCH BLOCK ANALYSIS
# ================================================================

print(f"\n{'=' * 75}")
print("9. BUNDLE BRANCH BLOCK TRAP — LBBB/RBBB in N-class vs PVC")
print(f"{'=' * 75}")

# Records known to have LBBB: 109, 111, 207, 214
# Records known to have RBBB: 118, 124, 212, 231
bbb_records_train = ['109', '207']  # in DS1
bbb_records_test = ['111', '214', '212', '231']  # in DS2

print("\n  Comparing feature values: Normal-conduction N vs BBB N vs V-class")
print(f"  {'Feature':25s} {'Normal N':>10} {'BBB N':>10} {'V-class':>10} {'BBB-V sep':>10}")

# Get indices for BBB records in test set
bbb_test_mask = np.array([r in bbb_records_test for r in test_recs])
n_class_mask = test_labels == 0
v_class_mask = test_labels == 2

bbb_n_mask = bbb_test_mask & n_class_mask
normal_n_mask = ~bbb_test_mask & n_class_mask

key_features = ['qrs_width_L0', 'st_discordance_L0', 'qrs_symmetry_L0',
                'interlead_divergence', 'pwave_energy_L0',
                'templ_corr_L0', 'beat_instability_L0',
                'rr_ratio', 'compensatory_ratio']
for fname in key_features:
    fidx = ALL_FEATURE_NAMES.index(fname)
    nm = test_norm[normal_n_mask, fidx].mean() if normal_n_mask.sum() > 0 else 0
    bm = test_norm[bbb_n_mask, fidx].mean() if bbb_n_mask.sum() > 0 else 0
    vm = test_norm[v_class_mask, fidx].mean() if v_class_mask.sum() > 0 else 0
    bbb_v_sep = abs(bm - vm)
    novel = "*" if fname in novel_feat_names else " "
    print(f" {novel}{fname:25s} {nm:10.3f} {bm:10.3f} {vm:10.3f} {bbb_v_sep:10.3f}")

print(f"\n  BBB N-class beats in test: {bbb_n_mask.sum()}")
print(f"  Normal N-class beats in test: {normal_n_mask.sum()}")
print(f"  V-class beats in test: {v_class_mask.sum()}")


# ================================================================
# 10. OPTIMAL FEATURE SET SEARCH
# ================================================================

print(f"\n{'=' * 75}")
print("10. OPTIMAL FEATURE SET SEARCH (greedy forward by accuracy, max 12 features)")
print(f"{'=' * 75}")

candidates = [name for name, _ in rf_ranked]
selected_opt = []
best_acc_so_far = 0.0

for step in range(min(12, len(candidates))):
    best_next = None
    best_next_acc = 0.0
    best_next_f1s = None

    for cand in candidates:
        if cand in selected_opt:
            continue
        trial = selected_opt + [cand]
        idx_trial = [ALL_FEATURE_NAMES.index(f) for f in trial]
        _rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
                                     class_weight='balanced')
        _rf.fit(train_norm[:, idx_trial], train_labels)
        preds = _rf.predict(test_norm[:, idx_trial])
        acc = np.mean(preds == test_labels) * 100
        if acc > best_next_acc:
            best_next_acc = acc
            best_next = cand
            best_next_f1s = f1_score(test_labels, preds, average=None, zero_division=0)

    if best_next is None:
        break
    selected_opt.append(best_next)
    cost_so_far = sum(ANALOG_COST.get(f, 10) for f in selected_opt)
    novel = "*" if best_next in novel_feat_names else " "
    print(f"  {step+1:2d}. +{novel}{best_next:24s} -> {best_next_acc:.2f}%  cost={cost_so_far:3d}  F1: N={best_next_f1s[0]:.2f} S={best_next_f1s[1]:.2f} V={best_next_f1s[2]:.2f} F={best_next_f1s[3]:.2f} Q={best_next_f1s[4]:.2f}")

print(f"\n  Optimal set ({len(selected_opt)} features, cost={sum(ANALOG_COST.get(f,10) for f in selected_opt)}):")
for f in selected_opt:
    novel = " *NOVEL*" if f in novel_feat_names else ""
    print(f"    - {f} (cost={ANALOG_COST.get(f, '?')}){novel}")


# ================================================================
# 11. SUMMARY TABLE
# ================================================================

print(f"\n{'=' * 75}")
print("11. SUMMARY: FEATURE SET COMPARISON")
print(f"{'=' * 75}")

print(f"\n  {'Set':50s} {'#F':>3} {'Cost':>5} {'Acc%':>6} {'S-F1':>5} {'V-F1':>5}")
print(f"  {'-'*50} {'-'*3} {'-'*5} {'-'*6} {'-'*5} {'-'*5}")

sets_to_compare = [
    ("Current 16 (production)", current_16),
    ("Current 14 (shared_data)", current_14),
    ("Proposed 7 (plan)", proposed_7),
    ("Proposed 7 + instability (9)", proposed_7_plus_instab),
    ("Proposed 7 + lead1 novel (10)", proposed_7_plus_lead1),
    ("Hybrid A: timing+templ+novel (9)", hybrid_1),
    ("Hybrid B: timing+width+adaptive+novel (9)", hybrid_2),
    ("Hybrid C: proposed7+templ+instab (9)", hybrid_3),
    ("Novel only (9)", novel_only),
    ("All features (kitchen sink)", ALL_FEATURE_NAMES),
]

for label, fset in sets_to_compare:
    indices = [ALL_FEATURE_NAMES.index(f) for f in fset]
    _rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1,
                                 class_weight='balanced')
    _rf.fit(train_norm[:, indices], train_labels)
    preds = _rf.predict(test_norm[:, indices])
    acc = np.mean(preds == test_labels) * 100
    f1s = f1_score(test_labels, preds, average=None, zero_division=0)
    cost = sum(ANALOG_COST.get(f, 10) for f in fset)
    print(f"  {label:50s} {len(fset):3d} {cost:5d} {acc:6.2f} {f1s[1]:5.2f} {f1s[2]:5.2f}")

print(f"\n{'=' * 75}")
print("DONE — Use these results to select the optimal feature set for analog implementation.")
print(f"{'=' * 75}")
