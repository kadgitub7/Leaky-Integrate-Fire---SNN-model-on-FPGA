"""
SHARED DATA MODULE — Optimized 14-feature extraction for all architecture experiments.

Feature set rationale (every feature data-proven):
  KEPT (top RF importance + ANOVA):
    1. pre_rr           RF#2, ANOVA#5   — absolute timing
    2. post_rr          RF#7, ANOVA#8   — forward timing
    3. rr_ratio         RF#3, ANOVA#3   — relative timing (vs local mean)
    4. rr_asymmetry     RF#5, ANOVA#4   — pre/post ratio
    5. rr_std_10        RF#19           — rhythm stability (10-beat window)
    6. qrs_area_L0      RF#17           — absolute morphology
    7. max_slope_L0     RF#6, ANOVA#6   — depolarization rate
    8. rel_area_L0      RF#9, ANOVA#2   — patient-normalized area
    9. templ_corr_L0    RF#1, ANOVA#1   — #1 most important feature
   10. rel_area_L1      RF#14           — cross-lead patient-adaptive
   11. templ_corr_L1    RF#15           — cross-lead template match

  ADDED (data-proven, not in previous set):
   12. slope_ratio_L0   RF#16           — up/down slope asymmetry, separates F-class
   13. rel_peak_L0      RF#10, MI#8     — was dropped from Exp B, proven important
   14. corr_L0L1        RF#4            — cross-lead QRS correlation, #4 overall!

  DROPPED (data-proven useless or redundant):
    - compensatory_ratio  r=0.873 with post_rr — redundant
    - qrs_width_L0        RF#22 — low importance
    - max_slope_L1        RF#28, ANOVA#30 — dead last
    - qrs_width_L1        RF#29 — near-last
"""

import numpy as np
import wfdb
import os
import torch
from collections import Counter
from sklearn.model_selection import train_test_split

NUM_FEATURES = 14
NUM_CLASSES = 5
BATCH_SIZE = 128
EWMA_ALPHA_MORPH = 0.015
N_TEMPLATE = 6
EWMA_GATE_THRESH = 0.85
EWMA_INIT_BEATS = 10
QRS_START = 70
QRS_END = 130
WIN_LEFT = 90
WIN_RIGHT = 108

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'rr_std_10',
    'qrs_area_L0', 'max_slope_L0', 'slope_ratio_L0',
    'rel_area_L0', 'templ_corr_L0', 'rel_peak_L0',
    'rel_area_L1', 'templ_corr_L1',
    'corr_L0L1',
]

aami_map = {}
for sym in ['N', 'L', 'R', 'e', 'j']: aami_map[sym] = 0
for sym in ['A', 'a', 'J', 'S']: aami_map[sym] = 1
for sym in ['V', 'E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/', 'f', 'Q']: aami_map[sym] = 4

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']


def _extract_records(rec_list, return_sequences=False, seq_len=3):
    """Extract features from records.

    If return_sequences=True, also returns sequences of consecutive beats
    for temporal models (each sample = seq_len consecutive beats' features).
    """
    all_labels = []
    all_features = []
    all_seq_features = [] if return_sequences else None
    all_seq_labels = [] if return_sequences else None

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
                if idx - WIN_LEFT >= 0 and idx + WIN_RIGHT < len(signals) and sym in aami_map:
                    valid.append((idx, sym))

            ewma_area = [None] * num_leads
            ewma_peak = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_peaks = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []

            rec_features = []
            rec_labels = []

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - WIN_LEFT : idx + WIN_RIGHT, :]
                label = aami_map[sym]

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - 20), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833

                rr_history.append(pre_rr)
                if len(rr_history) > 10:
                    rr_history.pop(0)

                feat = np.zeros(NUM_FEATURES, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                # Lead 0 morphology
                qrs0 = beat[QRS_START:QRS_END, 0]
                abs_qrs0 = np.abs(qrs0)
                peak0 = np.max(abs_qrs0) if len(abs_qrs0) > 0 else 0.0
                area0 = np.sum(abs_qrs0) / fs
                dqrs0 = np.diff(qrs0) * fs
                max_slope0 = np.max(np.abs(dqrs0)) if len(dqrs0) > 0 else 0.0

                if len(dqrs0) > 0:
                    up = max(np.max(dqrs0), 1e-8)
                    down = max(abs(np.min(dqrs0)), 1e-8)
                    slope_ratio0 = up / (down + 1e-8)
                else:
                    slope_ratio0 = 1.0

                feat[5] = area0
                feat[6] = max_slope0
                feat[7] = slope_ratio0

                # Lead 0 EWMA
                tmpl_idx = np.linspace(0, len(qrs0) - 1, N_TEMPLATE).astype(int)
                qrs_ds0 = qrs0[tmpl_idx]

                if i < EWMA_INIT_BEATS:
                    init_areas[0].append(max(area0, 1e-6))
                    init_peaks[0].append(max(peak0, 1e-6))
                    init_templates[0].append(qrs_ds0.copy())

                if ewma_area[0] is None:
                    if i >= EWMA_INIT_BEATS - 1 and len(init_areas[0]) >= EWMA_INIT_BEATS:
                        ewma_area[0] = np.median(init_areas[0])
                        ewma_peak[0] = np.median(init_peaks[0])
                        ewma_template[0] = np.median(init_templates[0], axis=0)
                    else:
                        ewma_area[0] = max(area0, 1e-6)
                        ewma_peak[0] = max(peak0, 1e-6)
                        ewma_template[0] = qrs_ds0.copy()

                feat[8] = area0 / (ewma_area[0] + 1e-8)
                norm_c = np.linalg.norm(qrs_ds0)
                norm_t = np.linalg.norm(ewma_template[0])
                tc0 = np.dot(qrs_ds0, ewma_template[0]) / (norm_c * norm_t + 1e-8) if norm_c > 1e-8 and norm_t > 1e-8 else 1.0
                feat[9] = tc0
                feat[10] = peak0 / (ewma_peak[0] + 1e-8)

                if tc0 > EWMA_GATE_THRESH:
                    a = EWMA_ALPHA_MORPH
                    ewma_area[0] = a * max(area0, 1e-6) + (1 - a) * ewma_area[0]
                    ewma_peak[0] = a * max(peak0, 1e-6) + (1 - a) * ewma_peak[0]
                    ewma_template[0] = a * qrs_ds0 + (1 - a) * ewma_template[0]

                # Lead 1
                if num_leads >= 2:
                    qrs1 = beat[QRS_START:QRS_END, 1]
                    abs_qrs1 = np.abs(qrs1)
                    area1 = np.sum(abs_qrs1) / fs
                    tmpl_idx1 = np.linspace(0, len(qrs1) - 1, N_TEMPLATE).astype(int)
                    qrs_ds1 = qrs1[tmpl_idx1]

                    if i < EWMA_INIT_BEATS:
                        init_areas[1].append(max(area1, 1e-6))
                        init_templates[1].append(qrs_ds1.copy())

                    if ewma_area[1] is None:
                        if i >= EWMA_INIT_BEATS - 1 and len(init_areas[1]) >= EWMA_INIT_BEATS:
                            ewma_area[1] = np.median(init_areas[1])
                            ewma_template[1] = np.median(init_templates[1], axis=0)
                        else:
                            ewma_area[1] = max(area1, 1e-6)
                            ewma_template[1] = qrs_ds1.copy()

                    feat[11] = area1 / (ewma_area[1] + 1e-8)
                    norm_c1 = np.linalg.norm(qrs_ds1)
                    norm_t1 = np.linalg.norm(ewma_template[1])
                    tc1 = np.dot(qrs_ds1, ewma_template[1]) / (norm_c1 * norm_t1 + 1e-8) if norm_c1 > 1e-8 and norm_t1 > 1e-8 else 1.0
                    feat[12] = tc1

                    if tc1 > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[1] = a * max(area1, 1e-6) + (1 - a) * ewma_area[1]
                        ewma_template[1] = a * qrs_ds1 + (1 - a) * ewma_template[1]

                    n0 = np.linalg.norm(qrs0)
                    n1 = np.linalg.norm(qrs1)
                    feat[13] = np.dot(qrs0, qrs1) / (n0 * n1 + 1e-8) if n0 > 1e-8 and n1 > 1e-8 else 0.0

                all_labels.append(label)
                all_features.append(feat)
                rec_features.append(feat.copy())
                rec_labels.append(label)

            # Build sequences from this record's beats
            if return_sequences and len(rec_features) >= seq_len:
                for k in range(seq_len - 1, len(rec_features)):
                    seq = np.stack(rec_features[k - seq_len + 1 : k + 1])  # (seq_len, n_feat)
                    all_seq_features.append(seq)
                    all_seq_labels.append(rec_labels[k])

        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")

    result = (np.array(all_labels), np.array(all_features))
    if return_sequences:
        result = result + (np.array(all_seq_labels), np.array(all_seq_features))
    return result


def smote_oversample(features, labels, target_ratio=0.33):
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    new_f, new_l = list(features), list(labels)
    for cls in range(NUM_CLASSES):
        idx = np.where(labels == cls)[0]
        if len(idx) >= target_count: continue
        n_syn = target_count - len(idx)
        cf = features[idx]
        if len(cf) < 2:
            for _ in range(n_syn):
                new_f.append(cf[0] + np.random.randn(features.shape[1]) * 0.05)
                new_l.append(cls)
            continue
        for _ in range(n_syn):
            a, b = np.random.randint(len(cf), size=2)
            while b == a: b = np.random.randint(len(cf))
            new_f.append(cf[a] + np.random.random() * (cf[b] - cf[a]))
            new_l.append(cls)
    return np.array(new_f), np.array(new_l)


def smote_oversample_seq(features, labels, target_ratio=0.33):
    """SMOTE for sequential data (features shape: N x seq_len x n_feat)."""
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    new_f, new_l = list(features), list(labels)
    for cls in range(NUM_CLASSES):
        idx = np.where(labels == cls)[0]
        if len(idx) >= target_count: continue
        n_syn = target_count - len(idx)
        cf = features[idx]
        if len(cf) < 2:
            for _ in range(n_syn):
                new_f.append(cf[0] + np.random.randn(*cf[0].shape) * 0.05)
                new_l.append(cls)
            continue
        for _ in range(n_syn):
            a, b = np.random.randint(len(cf), size=2)
            while b == a: b = np.random.randint(len(cf))
            new_f.append(cf[a] + np.random.random() * (cf[b] - cf[a]))
            new_l.append(cls)
    return np.array(new_f), np.array(new_l)


def load_data(return_sequences=False, seq_len=3):
    """Load and preprocess MIT-BIH data with optimized 14-feature set."""
    print("Loading MIT-BIH data with optimized 14-feature set...")
    print(f"  Features: {FEATURE_NAMES}")

    if return_sequences:
        tr_labels, tr_feat, tr_seq_labels, tr_seq_feat = _extract_records(DS1, True, seq_len)
        te_labels, te_feat, te_seq_labels, te_seq_feat = _extract_records(DS2, True, seq_len)
    else:
        tr_labels, tr_feat = _extract_records(DS1)
        te_labels, te_feat = _extract_records(DS2)

    print(f"  DS1: {len(tr_labels)} beats | DS2: {len(te_labels)} beats")
    print(f"  Class dist: {dict(Counter(tr_labels))}")

    # SMOTE
    tr_feat_s, tr_labels_s = smote_oversample(tr_feat, tr_labels, 0.33)
    print(f"  After SMOTE: {dict(Counter(tr_labels_s))}")

    # Normalize
    mu, sd = tr_feat_s.mean(0), tr_feat_s.std(0)
    tr_feat_s = (tr_feat_s - mu) / (sd + 1e-8)
    te_feat_n = (te_feat - mu) / (sd + 1e-8)

    # Train/val split
    tr_idx, val_idx = train_test_split(
        np.arange(len(tr_labels_s)), test_size=0.1, random_state=42, stratify=tr_labels_s)

    result = {
        'train_feat': tr_feat_s[tr_idx], 'train_labels': tr_labels_s[tr_idx],
        'val_feat': tr_feat_s[val_idx], 'val_labels': tr_labels_s[val_idx],
        'test_feat': te_feat_n, 'test_labels': te_labels,
        'mu': mu, 'sd': sd,
    }

    if return_sequences:
        tr_seq_s, tr_seq_l_s = smote_oversample_seq(tr_seq_feat, tr_seq_labels, 0.33)
        # Normalize each timestep
        for t in range(seq_len):
            tr_seq_s[:, t, :] = (tr_seq_s[:, t, :] - mu) / (sd + 1e-8)
            te_seq_feat[:, t, :] = (te_seq_feat[:, t, :] - mu) / (sd + 1e-8)
        seq_tr_idx, seq_val_idx = train_test_split(
            np.arange(len(tr_seq_l_s)), test_size=0.1, random_state=42, stratify=tr_seq_l_s)
        result['seq_train_feat'] = tr_seq_s[seq_tr_idx]
        result['seq_train_labels'] = tr_seq_l_s[seq_tr_idx]
        result['seq_val_feat'] = tr_seq_s[seq_val_idx]
        result['seq_val_labels'] = tr_seq_l_s[seq_val_idx]
        result['seq_test_feat'] = te_seq_feat
        result['seq_test_labels'] = te_seq_labels

    return result


class BeatDS(torch.utils.data.Dataset):
    def __init__(self, f, l):
        self.data = torch.tensor(f, dtype=torch.float32)
        self.targets = torch.tensor(l, dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.targets[i]


def make_loaders(data, seq=False):
    """Create train/val/test DataLoaders with weighted sampling."""
    prefix = 'seq_' if seq else ''
    train_ds = BeatDS(data[f'{prefix}train_feat'], data[f'{prefix}train_labels'])
    val_ds = BeatDS(data[f'{prefix}val_feat'], data[f'{prefix}val_labels'])
    test_ds = BeatDS(data[f'{prefix}test_feat'], data[f'{prefix}test_labels'])

    tr_counts = np.bincount(data[f'{prefix}train_labels'], minlength=5)
    sw = 1.0 / (tr_counts ** 0.65 + 1e-8)
    sample_w = [sw[l] for l in data[f'{prefix}train_labels']]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    cw = 1.0 / (tr_counts.astype(np.float64) ** 0.65 + 1e-8)
    cw = cw / cw.sum() * 5

    return (
        torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True),
        torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        torch.tensor(cw, dtype=torch.float32),
    )
