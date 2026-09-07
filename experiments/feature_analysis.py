"""
FEATURE ANALYSIS: Measure actual discriminative power of each feature
=====================================================================
Computes per-feature class separability metrics from the real data:
  1. ANOVA F-score (between-class vs within-class variance)
  2. Mutual information with class labels
  3. Per-class distribution overlap analysis
  4. Correlation matrix (find redundancy)
  5. Random Forest feature importance (non-linear discriminative power)

Also tests EXPANDED feature set to find features we might be missing.

Run: python experiments/feature_analysis.py
"""

import numpy as np
import wfdb
import os
from collections import Counter
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

EWMA_ALPHA_MORPH = 0.015
N_TEMPLATE = 6
EWMA_GATE_THRESH = 0.85
EWMA_INIT_BEATS = 10
QRS_START = 70
QRS_END = 130
win_left = 90
win_right = 108

aami_map = {}
for sym in ['N', 'L', 'R', 'e', 'j']: aami_map[sym] = 0
for sym in ['A', 'a', 'J', 'S']: aami_map[sym] = 1
for sym in ['V', 'E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/', 'f', 'Q']: aami_map[sym] = 4

DS1_records = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]
DS2_records = [
    '100', '103', '105', '111', '113', '117', '121', '200', '202', '210',
    '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]

# EXPANDED feature set — everything we could extract from the same analog circuits
FEATURE_NAMES = [
    # Timing (7)
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio',
    'rr_std_10', 'rr_std_20',
    # Morphology per lead (x2 = 14)
    'qrs_width_L0', 'qrs_width_L1',
    'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1',
    'peak_amp_L0', 'peak_amp_L1',
    'qrs_energy_L0', 'qrs_energy_L1',          # sum of squares (power)
    'slope_ratio_L0', 'slope_ratio_L1',         # max_upslope / max_downslope
    # Patient-adaptive per lead (x2 = 8)
    'rel_area_L0', 'rel_area_L1',
    'rel_width_L0', 'rel_width_L1',
    'rel_peak_L0', 'rel_peak_L1',
    'templ_corr_L0', 'templ_corr_L1',
    # Cross-lead (3)
    'area_ratio_L0L1',                          # area_L0 / area_L1
    'width_diff_L0L1',                          # abs(width_L0 - width_L1)
    'corr_L0L1',                                # correlation between lead QRS shapes
]


def extract_expanded_features(rec_list):
    all_labels = []
    all_features = []
    n_feat = len(FEATURE_NAMES)

    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            fs = record.fs
            num_leads = min(signals.shape[1], 2)

            valid = []
            for idx, sym in zip(annotation.sample, annotation.symbol):
                if (idx - win_left >= 0 and idx + win_right < len(signals) and sym in aami_map):
                    valid.append((idx, sym))

            ewma_area = [None] * num_leads
            ewma_width = [None] * num_leads
            ewma_peak = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_widths = [[] for _ in range(num_leads)]
            init_peaks = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - win_left : idx + win_right, :]
                all_labels.append(aami_map[sym])

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - 20), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833

                rr_history.append(pre_rr)
                if len(rr_history) > 20:
                    rr_history.pop(0)

                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history[-10:]) if len(rr_history) >= 2 else 0.0
                feat[6] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                qrs_signals = []
                widths = []
                areas = []
                peaks = []

                for lead in range(num_leads):
                    qrs = beat[QRS_START:QRS_END, lead]
                    abs_qrs = np.abs(qrs)
                    peak = np.max(abs_qrs) if len(abs_qrs) > 0 else 0.0

                    width = 0.0
                    if peak > 1e-6:
                        threshold = 0.3 * peak
                        above = abs_qrs > threshold
                        if np.any(above):
                            first = np.argmax(above)
                            last = len(above) - 1 - np.argmax(above[::-1])
                            width = (last - first) * 1000.0 / fs

                    area = np.sum(abs_qrs) / fs
                    dqrs = np.diff(qrs) * fs
                    max_slope = np.max(np.abs(dqrs)) if len(dqrs) > 0 else 0.0
                    energy = np.sum(qrs**2) / fs

                    # Slope ratio: max upslope / max downslope
                    if len(dqrs) > 0:
                        up = np.max(dqrs) if np.max(dqrs) > 0 else 1e-8
                        down = np.abs(np.min(dqrs)) if np.min(dqrs) < 0 else 1e-8
                        slope_ratio = up / (down + 1e-8)
                    else:
                        slope_ratio = 1.0

                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]

                    feat[7 + lead] = width
                    feat[9 + lead] = area
                    feat[11 + lead] = max_slope
                    feat[13 + lead] = peak
                    feat[15 + lead] = energy
                    feat[17 + lead] = slope_ratio

                    qrs_signals.append(qrs)
                    widths.append(width)
                    areas.append(area)
                    peaks.append(peak)

                    # EWMA init from median
                    if i < EWMA_INIT_BEATS:
                        init_areas[lead].append(max(area, 1e-6))
                        init_widths[lead].append(max(width, 1e-6))
                        init_peaks[lead].append(max(peak, 1e-6))
                        init_templates[lead].append(qrs_ds.copy())

                    if ewma_area[lead] is None:
                        if i >= EWMA_INIT_BEATS - 1 and len(init_areas[lead]) >= EWMA_INIT_BEATS:
                            ewma_area[lead] = np.median(init_areas[lead])
                            ewma_width[lead] = np.median(init_widths[lead])
                            ewma_peak[lead] = np.median(init_peaks[lead])
                            ewma_template[lead] = np.median(init_templates[lead], axis=0)
                        else:
                            ewma_area[lead] = max(area, 1e-6)
                            ewma_width[lead] = max(width, 1e-6)
                            ewma_peak[lead] = max(peak, 1e-6)
                            ewma_template[lead] = qrs_ds.copy()

                    feat[19 + lead] = area / (ewma_area[lead] + 1e-8)
                    feat[21 + lead] = width / (ewma_width[lead] + 1e-8)
                    feat[23 + lead] = peak / (ewma_peak[lead] + 1e-8)

                    norm_curr = np.linalg.norm(qrs_ds)
                    norm_tmpl = np.linalg.norm(ewma_template[lead])
                    if norm_curr > 1e-8 and norm_tmpl > 1e-8:
                        templ_corr = np.dot(qrs_ds, ewma_template[lead]) / (norm_curr * norm_tmpl)
                    else:
                        templ_corr = 1.0
                    feat[25 + lead] = templ_corr

                    if templ_corr > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_width[lead] = a * max(width, 1e-6) + (1 - a) * ewma_width[lead]
                        ewma_peak[lead] = a * max(peak, 1e-6) + (1 - a) * ewma_peak[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

                # Cross-lead features
                if num_leads == 2:
                    feat[27] = areas[0] / (areas[1] + 1e-8)
                    feat[28] = abs(widths[0] - widths[1])
                    if len(qrs_signals[0]) == len(qrs_signals[1]):
                        n0 = np.linalg.norm(qrs_signals[0])
                        n1 = np.linalg.norm(qrs_signals[1])
                        if n0 > 1e-8 and n1 > 1e-8:
                            feat[29] = np.dot(qrs_signals[0], qrs_signals[1]) / (n0 * n1)
                        else:
                            feat[29] = 0.0

                all_features.append(feat)

        except Exception as e:
            print(f"Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features)


# ================================================================
# EXTRACT DATA
# ================================================================

print("Extracting expanded features from DS1 (train) and DS2 (test)...")
train_labels, train_features = extract_expanded_features(DS1_records)
test_labels, test_features = extract_expanded_features(DS2_records)
print(f"  DS1: {len(train_labels)} beats, DS2: {len(test_labels)} beats")
print(f"  Features: {len(FEATURE_NAMES)}")
print(f"  Class distribution (train): {dict(Counter(train_labels))}")
print(f"  Class distribution (test):  {dict(Counter(test_labels))}")

# Normalize
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_norm = (train_features - train_mean) / (train_std + 1e-8)
test_norm = (test_features - train_mean) / (train_std + 1e-8)


# ================================================================
# 1. ANOVA F-SCORES
# ================================================================

print(f"\n{'='*70}")
print("1. ANOVA F-SCORES (higher = more discriminative between classes)")
print(f"{'='*70}")
f_scores, p_values = f_classif(train_norm, train_labels)
ranked = sorted(zip(FEATURE_NAMES, f_scores, p_values), key=lambda x: -x[1])
for i, (name, f, p) in enumerate(ranked):
    bar = '#' * min(int(f / max(f_scores) * 40), 40)
    print(f"  {i+1:2d}. {name:20s} F={f:8.1f}  p={p:.2e}  {bar}")


# ================================================================
# 2. MUTUAL INFORMATION
# ================================================================

print(f"\n{'='*70}")
print("2. MUTUAL INFORMATION (captures non-linear relationships)")
print(f"{'='*70}")
mi = mutual_info_classif(train_norm, train_labels, random_state=42, n_neighbors=5)
ranked_mi = sorted(zip(FEATURE_NAMES, mi), key=lambda x: -x[1])
for i, (name, m) in enumerate(ranked_mi):
    bar = '#' * min(int(m / max(mi) * 40), 40)
    print(f"  {i+1:2d}. {name:20s} MI={m:.4f}  {bar}")


# ================================================================
# 3. RANDOM FOREST IMPORTANCE (non-linear, interaction-aware)
# ================================================================

print(f"\n{'='*70}")
print("3. RANDOM FOREST IMPORTANCE (inter-patient)")
print(f"{'='*70}")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
                            class_weight='balanced')
rf.fit(train_norm, train_labels)
rf_acc = rf.score(test_norm, test_labels) * 100
rf_preds = rf.predict(test_norm)
print(f"  RF accuracy (inter-patient): {rf_acc:.2f}%")
print(classification_report(test_labels, rf_preds, target_names=['N', 'S', 'V', 'F', 'Q']))

importances = rf.feature_importances_
ranked_rf = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
for i, (name, imp) in enumerate(ranked_rf):
    bar = '#' * min(int(imp / max(importances) * 40), 40)
    print(f"  {i+1:2d}. {name:20s} imp={imp:.4f}  {bar}")


# ================================================================
# 4. CORRELATION MATRIX (redundancy detection)
# ================================================================

print(f"\n{'='*70}")
print("4. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.8 = redundant)")
print(f"{'='*70}")
corr = np.corrcoef(train_norm.T)
pairs = []
for i in range(len(FEATURE_NAMES)):
    for j in range(i+1, len(FEATURE_NAMES)):
        r = corr[i, j]
        if abs(r) > 0.8:
            pairs.append((FEATURE_NAMES[i], FEATURE_NAMES[j], r))
pairs.sort(key=lambda x: -abs(x[2]))
for n1, n2, r in pairs:
    print(f"  {n1:20s} <-> {n2:20s}  r={r:+.3f}")
if not pairs:
    print("  No highly correlated pairs found.")


# ================================================================
# 5. PER-CLASS SEPARABILITY
# ================================================================

print(f"\n{'='*70}")
print("5. PER-CLASS FEATURE MEANS (which features separate which classes)")
print(f"{'='*70}")
class_names = ['N', 'S', 'V', 'F', 'Q']
# Show top features and their per-class z-scored means
top_features = [name for name, _ in ranked_rf[:20]]
top_indices = [FEATURE_NAMES.index(name) for name in top_features]

print(f"  {'Feature':20s} {'N':>8} {'S':>8} {'V':>8} {'F':>8} {'Q':>8}  Best separator")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
for name, idx in zip(top_features, top_indices):
    means = []
    for cls in range(5):
        cls_mask = train_labels == cls
        means.append(train_norm[cls_mask, idx].mean())
    means = np.array(means)
    # Which pair has max separation?
    max_sep = 0
    best_pair = ""
    for a in range(5):
        for b in range(a+1, 5):
            sep = abs(means[a] - means[b])
            if sep > max_sep:
                max_sep = sep
                best_pair = f"{class_names[a]} vs {class_names[b]}"
    print(f"  {name:20s} {means[0]:8.3f} {means[1]:8.3f} {means[2]:8.3f} {means[3]:8.3f} {means[4]:8.3f}  {best_pair} ({max_sep:.2f})")


# ================================================================
# 6. ABLATION: RF accuracy with feature subsets
# ================================================================

print(f"\n{'='*70}")
print("6. FEATURE SUBSET ABLATION (RF inter-patient accuracy)")
print(f"{'='*70}")

# Test our current 16 features
current_16 = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1', 'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1', 'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
]

def rf_test_subset(feature_names, label):
    indices = [FEATURE_NAMES.index(f) for f in feature_names]
    rf_sub = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1,
                                    class_weight='balanced')
    rf_sub.fit(train_norm[:, indices], train_labels)
    acc = rf_sub.score(test_norm[:, indices], test_labels) * 100
    preds = rf_sub.predict(test_norm[:, indices])
    from sklearn.metrics import f1_score
    f1s = f1_score(test_labels, preds, average=None)
    print(f"  {label:45s} {len(feature_names):2d} feat -> {acc:.2f}%  F1: N={f1s[0]:.2f} S={f1s[1]:.2f} V={f1s[2]:.2f} F={f1s[3]:.2f} Q={f1s[4]:.2f}")
    return acc

# All 30 features
rf_test_subset(FEATURE_NAMES, "All expanded (30 features)")

# Current 16
rf_test_subset(current_16, "Current 16 features")

# Top-N by RF importance
for n in [8, 10, 12, 14, 16, 18, 20]:
    top_n = [name for name, _ in ranked_rf[:n]]
    rf_test_subset(top_n, f"Top {n} by RF importance")

# Timing only
timing = ['pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10', 'rr_std_20']
rf_test_subset(timing, "Timing only (7)")

# Morphology only (no patient-adaptive)
morph = ['qrs_width_L0', 'qrs_width_L1', 'qrs_area_L0', 'qrs_area_L1',
         'max_slope_L0', 'max_slope_L1', 'peak_amp_L0', 'peak_amp_L1',
         'qrs_energy_L0', 'qrs_energy_L1', 'slope_ratio_L0', 'slope_ratio_L1']
rf_test_subset(morph, "Morphology only (12)")

# Patient-adaptive only
adaptive = ['rel_area_L0', 'rel_area_L1', 'rel_width_L0', 'rel_width_L1',
            'rel_peak_L0', 'rel_peak_L1', 'templ_corr_L0', 'templ_corr_L1']
rf_test_subset(adaptive, "Patient-adaptive only (8)")

# Timing + patient-adaptive (no raw morphology)
rf_test_subset(timing + adaptive, "Timing + patient-adaptive (15)")

# Best combo: top RF features that aren't redundant
# Remove features with |r| > 0.9 correlation, keeping higher-ranked one
print(f"\n  --- Optimal subset (greedy forward selection) ---")
remaining = [name for name, _ in ranked_rf]
selected = []
for name in remaining:
    idx = FEATURE_NAMES.index(name)
    redundant = False
    for sel in selected:
        sel_idx = FEATURE_NAMES.index(sel)
        r = corr[idx, sel_idx]
        if abs(r) > 0.85:
            redundant = True
            break
    if not redundant:
        selected.append(name)
        if len(selected) <= 20:
            rf_test_subset(selected, f"Greedy top {len(selected)}: +{name}")

print(f"\n  Optimal feature set ({len(selected)} features):")
for name in selected:
    print(f"    - {name}")
