"""
Shared data loading using PROVEN 16-feature set (91.45% inter-patient).
Supports optional 17th feature (corr_L0L1, RF importance #4).
Supports both inter-patient (DS1/DS2) and intra-patient (stratified 80/20).
"""
import numpy as np
import wfdb
import os
import torch
from collections import Counter
from sklearn.model_selection import train_test_split

EWMA_ALPHA_MORPH = 0.015
N_TEMPLATE = 6
EWMA_GATE_THRESH = 0.85
EWMA_INIT_BEATS = 10
QRS_START = 70
QRS_END = 130
WIN_LEFT = 90
WIN_RIGHT = 108
NUM_CLASSES = 5
BATCH_SIZE = 128

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
ALL_RECORDS = sorted(set(DS1 + DS2))


def extract_features(rec_list, include_cross_lead=False):
    all_labels, all_features = [], []
    n_feat = 17 if include_cross_lead else 16
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
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []
            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - WIN_LEFT : idx + WIN_RIGHT, :]
                all_labels.append(aami_map[sym])
                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833
                local_rrs = []
                for j in range(max(1, i - 10), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833
                rr_history.append(pre_rr)
                if len(rr_history) > 10: rr_history.pop(0)
                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history) if len(rr_history) >= 2 else 0.0
                qrs_data = {}
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
                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]
                    feat[6 + lead] = width
                    feat[8 + lead] = area
                    feat[10 + lead] = max_slope
                    if i < EWMA_INIT_BEATS:
                        init_areas[lead].append(max(area, 1e-6))
                        init_templates[lead].append(qrs_ds.copy())
                    if ewma_area[lead] is None:
                        if i >= EWMA_INIT_BEATS - 1 and len(init_areas[lead]) >= EWMA_INIT_BEATS:
                            ewma_area[lead] = np.median(init_areas[lead])
                            ewma_template[lead] = np.median(init_templates[lead], axis=0)
                        else:
                            ewma_area[lead] = max(area, 1e-6)
                            ewma_template[lead] = qrs_ds.copy()
                    feat[12 + lead] = area / (ewma_area[lead] + 1e-8)
                    norm_c = np.linalg.norm(qrs_ds)
                    norm_t = np.linalg.norm(ewma_template[lead])
                    tc = np.dot(qrs_ds, ewma_template[lead]) / (norm_c * norm_t + 1e-8) if norm_c > 1e-8 and norm_t > 1e-8 else 1.0
                    feat[14 + lead] = tc
                    if tc > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]
                    qrs_data[lead] = qrs
                if include_cross_lead and num_leads >= 2:
                    n0 = np.linalg.norm(qrs_data[0])
                    n1 = np.linalg.norm(qrs_data[1])
                    feat[16] = np.dot(qrs_data[0], qrs_data[1]) / (n0 * n1 + 1e-8) if n0 > 1e-8 and n1 > 1e-8 else 0.0
                all_features.append(feat)
        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")
    return np.array(all_labels), np.array(all_features)


def smote_oversample(features, labels, target_ratio=0.33, exclude_classes=None):
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    exclude = set(exclude_classes) if exclude_classes else set()
    new_f, new_l = list(features), list(labels)
    for cls in range(NUM_CLASSES):
        if cls in exclude:
            continue
        idx = np.where(labels == cls)[0]
        if len(idx) >= target_count:
            continue
        n_syn = target_count - len(idx)
        cf = features[idx]
        for _ in range(n_syn):
            i = np.random.randint(len(cf))
            j = np.random.randint(len(cf))
            while j == i and len(cf) > 1:
                j = np.random.randint(len(cf))
            new_f.append(cf[i] + np.random.random() * (cf[j] - cf[i]))
            new_l.append(cls)
    return np.array(new_f), np.array(new_l)


def load_split(split='inter', include_cross_lead=False, smote_ratio=0.33,
               smote_exclude_classes=None, cls_power=0.65):
    n_feat_label = 17 if include_cross_lead else 16
    print(f"\n{'='*60}")
    print(f"Loading {split}-patient data ({n_feat_label} features, SMOTE {smote_ratio})")
    print(f"{'='*60}")

    if split == 'inter':
        train_labels, train_features = extract_features(DS1, include_cross_lead)
        test_labels, test_features = extract_features(DS2, include_cross_lead)
        print(f"  DS1 train: {len(train_labels)} | DS2 test: {len(test_labels)}")
    else:
        all_labels, all_features = extract_features(ALL_RECORDS, include_cross_lead)
        train_idx, test_idx = train_test_split(
            np.arange(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels)
        train_features, test_features = all_features[train_idx], all_features[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]
        print(f"  Train: {len(train_labels)} | Test: {len(test_labels)}")

    n_features = train_features.shape[1]
    print(f"  Class dist: {dict(Counter(train_labels))}")

    train_features, train_labels = smote_oversample(
        train_features, train_labels, smote_ratio, exclude_classes=smote_exclude_classes)
    print(f"  After SMOTE: {dict(Counter(train_labels))}")

    mu, sd = train_features.mean(0), train_features.std(0)
    train_features = (train_features - mu) / (sd + 1e-8)
    test_features = (test_features - mu) / (sd + 1e-8)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    tr_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    sw = 1.0 / (tr_counts ** cls_power)
    sample_w = [sw[l] for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        DS(train_features, train_labels), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        DS(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw = 1.0 / (tr_counts.astype(np.float64) ** cls_power)
    cw = cw / cw.sum() * NUM_CLASSES
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    return train_loader, test_loader, cw_tensor, device, n_features
