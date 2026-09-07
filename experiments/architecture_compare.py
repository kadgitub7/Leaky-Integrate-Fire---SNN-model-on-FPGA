"""
ARCHITECTURE COMPARISON: ANN vs SNN vs Hybrid on optimal features
=================================================================
Tests whether the SNN temporal dynamics help or hurt on static features.

  1. Pure ANN:    features -> FC(48,ReLU) -> FC(24,ReLU) -> FC(5)
  2. Pure SNN:    features -> FC(48) -> RLeaky -> FC(24) -> RLeaky -> FC(5) -> Leaky
  3. Hybrid:      features -> FC(32,ReLU) -> FC(16) -> RLeaky -> FC(5) -> Leaky
  4. ANN-deep:    features -> FC(64,ReLU) -> FC(32,ReLU) -> FC(16,ReLU) -> FC(5)

All with:
  - Optimized 16-feature set (swap useless L1 features for high-value ones)
  - QAT 4-bit, focal loss, SMOTE 0.33, validation-based checkpointing
  - Inter-patient evaluation

Run: python experiments/architecture_compare.py
"""

import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
import wfdb
import os
import copy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

torch.set_num_threads(2)
batch_size = 128
num_class = 5
t0 = time.time()

# ================================================================
# OPTIMIZED FEATURE SET (data-driven selection)
# ================================================================
# Swapped out: max_slope_L1 (#30 ANOVA), qrs_width_L1 (#29)
# Swapped in:  corr_L0L1 (#4 RF importance), slope_ratio_L0 (#16 RF)

FEATURE_NAMES = [
    # Timing (6) — all proven top-tier
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    # Lead 0 morphology (4) — all top-12 in RF importance
    'qrs_width_L0', 'qrs_area_L0', 'max_slope_L0', 'slope_ratio_L0',
    # Lead 0 patient-adaptive (3) — all proven
    'rel_area_L0', 'templ_corr_L0', 'rel_peak_L0',
    # Lead 1 patient-adaptive (2) — cross-patient normalization helps
    'rel_area_L1', 'templ_corr_L1',
    # Cross-lead (1) — #4 in RF importance, separates F-class
    'corr_L0L1',
]

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


def extract_features(rec_list):
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
                if idx - win_left >= 0 and idx + win_right < len(signals) and sym in aami_map:
                    valid.append((idx, sym))

            ewma_area = [None] * num_leads
            ewma_peak = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
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
                if len(rr_history) > 10:
                    rr_history.pop(0)

                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                # Lead 0 morphology
                qrs0 = beat[QRS_START:QRS_END, 0]
                abs_qrs0 = np.abs(qrs0)
                peak0 = np.max(abs_qrs0) if len(abs_qrs0) > 0 else 0.0

                width0 = 0.0
                if peak0 > 1e-6:
                    threshold = 0.3 * peak0
                    above = abs_qrs0 > threshold
                    if np.any(above):
                        first = np.argmax(above)
                        last = len(above) - 1 - np.argmax(above[::-1])
                        width0 = (last - first) * 1000.0 / fs

                area0 = np.sum(abs_qrs0) / fs
                dqrs0 = np.diff(qrs0) * fs
                max_slope0 = np.max(np.abs(dqrs0)) if len(dqrs0) > 0 else 0.0

                if len(dqrs0) > 0:
                    up = max(np.max(dqrs0), 1e-8)
                    down = max(abs(np.min(dqrs0)), 1e-8)
                    slope_ratio0 = up / (down + 1e-8)
                else:
                    slope_ratio0 = 1.0

                feat[6] = width0
                feat[7] = area0
                feat[8] = max_slope0
                feat[9] = slope_ratio0

                tmpl_idx = np.linspace(0, len(qrs0) - 1, N_TEMPLATE).astype(int)
                qrs_ds0 = qrs0[tmpl_idx]

                # Lead 0 EWMA
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

                feat[10] = area0 / (ewma_area[0] + 1e-8)

                norm_c0 = np.linalg.norm(qrs_ds0)
                norm_t0 = np.linalg.norm(ewma_template[0])
                templ_corr0 = np.dot(qrs_ds0, ewma_template[0]) / (norm_c0 * norm_t0 + 1e-8) if norm_c0 > 1e-8 and norm_t0 > 1e-8 else 1.0
                feat[11] = templ_corr0
                feat[12] = peak0 / (ewma_peak[0] + 1e-8)

                if templ_corr0 > EWMA_GATE_THRESH:
                    a = EWMA_ALPHA_MORPH
                    ewma_area[0] = a * max(area0, 1e-6) + (1 - a) * ewma_area[0]
                    ewma_peak[0] = a * max(peak0, 1e-6) + (1 - a) * ewma_peak[0]
                    ewma_template[0] = a * qrs_ds0 + (1 - a) * ewma_template[0]

                # Lead 1 patient-adaptive + cross-lead
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

                    feat[13] = area1 / (ewma_area[1] + 1e-8)

                    norm_c1 = np.linalg.norm(qrs_ds1)
                    norm_t1 = np.linalg.norm(ewma_template[1])
                    templ_corr1 = np.dot(qrs_ds1, ewma_template[1]) / (norm_c1 * norm_t1 + 1e-8) if norm_c1 > 1e-8 and norm_t1 > 1e-8 else 1.0
                    feat[14] = templ_corr1

                    if templ_corr1 > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[1] = a * max(area1, 1e-6) + (1 - a) * ewma_area[1]
                        ewma_template[1] = a * qrs_ds1 + (1 - a) * ewma_template[1]

                    # Cross-lead correlation
                    n0 = np.linalg.norm(qrs0)
                    n1 = np.linalg.norm(qrs1)
                    feat[15] = np.dot(qrs0, qrs1) / (n0 * n1 + 1e-8) if n0 > 1e-8 and n1 > 1e-8 else 0.0

                all_features.append(feat)

        except Exception as e:
            print(f"Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features)


def smote_oversample(features, labels, target_ratio=0.33):
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    new_features = list(features)
    new_labels = list(labels)
    for cls in range(num_class):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) >= target_count:
            continue
        n_syn = target_count - len(cls_idx)
        cls_feat = features[cls_idx]
        if len(cls_feat) < 2:
            for _ in range(n_syn):
                new_features.append(cls_feat[0] + np.random.randn(features.shape[1]) * 0.05)
                new_labels.append(cls)
            continue
        for _ in range(n_syn):
            i, j = np.random.randint(len(cls_feat)), np.random.randint(len(cls_feat))
            while j == i: j = np.random.randint(len(cls_feat))
            new_features.append(cls_feat[i] + np.random.random() * (cls_feat[j] - cls_feat[i]))
            new_labels.append(cls)
    return np.array(new_features), np.array(new_labels)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, input, target):
        ce = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def quantize_tensor(x, num_bits):
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    scale = torch.clamp((x.max() - x.min()) / (qmax - qmin), min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


# ================================================================
# MODEL ARCHITECTURES
# ================================================================

class PureANN(nn.Module):
    """Standard feedforward ANN — analog weighted sums + nonlinearity"""
    def __init__(self, n_features, hidden, n_classes, dropout=0.1):
        super().__init__()
        h1, h2 = hidden, max(hidden // 2, 8)
        self.bn = nn.BatchNorm1d(n_features)
        self.net = nn.Sequential(
            nn.Linear(n_features, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )
        self.h1, self.h2 = h1, h2

    def forward(self, x):
        return self.net(self.bn(x))


class DeepANN(nn.Module):
    """Deeper ANN with 3 hidden layers"""
    def __init__(self, n_features, hidden, n_classes, dropout=0.1):
        super().__init__()
        h1, h2, h3 = hidden, max(hidden // 2, 8), max(hidden // 4, 8)
        self.bn = nn.BatchNorm1d(n_features)
        self.net = nn.Sequential(
            nn.Linear(n_features, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h3, n_classes),
        )
        self.h1, self.h2 = h1, h2

    def forward(self, x):
        return self.net(self.bn(x))


class PureSNN(nn.Module):
    """Current architecture — two recurrent LIF layers"""
    def __init__(self, n_features, hidden, n_classes, beta=0.9, dropout=0.1):
        super().__init__()
        h1, h2 = hidden, max(hidden // 2, 8)
        self.bn = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, h1)
        self.drop1 = nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc_mid = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(dropout)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(h2, n_classes)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2

    def forward(self, x, num_steps):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        mem_out = self.lif_out.init_leaky()
        mem_out_rec, spk1_rec = [], []
        x = self.bn(x)
        fc1_out = self.drop1(self.fc1(x))
        for step in range(num_steps):
            spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
            spk1_rec.append(spk1)
            mid = self.drop2(self.fc_mid(spk1))
            spk2, mem2 = self.rlif2(mid, spk2, mem2)
            _, mem_out = self.lif_out(self.fc2(spk2), mem_out)
            mem_out_rec.append(mem_out)
        return torch.stack(mem_out_rec), torch.stack(spk1_rec)


class HybridANNSNN(nn.Module):
    """ANN feature encoder -> SNN temporal classifier"""
    def __init__(self, n_features, hidden, n_classes, beta=0.9, dropout=0.1):
        super().__init__()
        ann_h = max(hidden // 2, 16)
        snn_h = max(hidden // 2, 16)
        self.bn = nn.BatchNorm1d(n_features)
        # ANN encoder: non-linear feature combination
        self.encoder = nn.Sequential(
            nn.Linear(n_features, ann_h), nn.ReLU(), nn.Dropout(dropout),
        )
        # SNN classifier: spike-based temporal integration
        self.fc_snn = nn.Linear(ann_h, snn_h)
        self.drop = nn.Dropout(dropout)
        self.rlif = snn.RLeaky(beta=beta, linear_features=snn_h, learn_beta=True, learn_threshold=True)
        self.fc_out = nn.Linear(snn_h, n_classes)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = ann_h, snn_h

    def forward(self, x, num_steps):
        spk, mem = self.rlif.init_rleaky()
        mem_out = self.lif_out.init_leaky()
        mem_out_rec, spk_rec = [], []
        x = self.encoder(self.bn(x))
        fc_out = self.drop(self.fc_snn(x))
        for step in range(num_steps):
            spk, mem = self.rlif(fc_out, spk, mem)
            spk_rec.append(spk)
            _, mem_out = self.lif_out(self.fc_out(spk), mem_out)
            mem_out_rec.append(mem_out)
        return torch.stack(mem_out_rec), torch.stack(spk_rec)


# ================================================================
# TRAINING + EVALUATION
# ================================================================

def train_and_eval(model_name, model, train_loader, val_loader, test_loader, class_weights_tensor, device, is_snn=False, num_steps=20):
    loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 300
    warmup = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup)
    best_val_acc = 0.0
    best_state = None
    patience = 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{model_name}] Parameters: {n_params:,}")

    for epoch in range(num_epochs):
        if epoch < warmup:
            for pg in optimizer.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup

        model.train()
        epoch_loss = 0.0
        batches = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            if is_snn:
                saved = []
                with torch.no_grad():
                    for p in model.parameters():
                        saved.append(p.data.clone())
                        p.data.copy_(quantize_tensor(p.data, 4))
                mem_out, spk_hidden = model(data, num_steps)
                ce = sum(loss_fn(mem_out[s], targets) for s in range(num_steps))
                fire = spk_hidden.mean()
                total_loss = ce + 1.0 * torch.clamp(fire - 0.15, min=0.0)
                optimizer.zero_grad()
                total_loss.backward()
                with torch.no_grad():
                    for p, s in zip(model.parameters(), saved): p.data.copy_(s)
            else:
                saved = []
                with torch.no_grad():
                    for p in model.parameters():
                        saved.append(p.data.clone())
                        p.data.copy_(quantize_tensor(p.data, 4))
                out = model(data)
                ce = loss_fn(out, targets)
                total_loss = ce
                optimizer.zero_grad()
                total_loss.backward()
                with torch.no_grad():
                    for p, s in zip(model.parameters(), saved): p.data.copy_(s)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += ce.item()
            batches += 1

        if epoch >= warmup: scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                if is_snn:
                    mem_out, _ = model(data, num_steps)
                    _, pred = mem_out[-1].max(1)
                else:
                    _, pred = model(data).max(1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
        val_acc = correct / total * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{num_epochs} | CE: {epoch_loss/batches:.3f} | val: {val_acc:.2f}% (best: {best_val_acc:.2f}%) | {time.time()-t0:.0f}s")

        if patience >= 40 and epoch >= 50:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best and quantize
    model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters():
            p.data.copy_(quantize_tensor(p.data, 4))

    # Test
    model.eval()
    correct = total = 0
    all_preds, all_targets = [], []
    total_spikes = total_possible = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            if is_snn:
                mem_out, spk_hidden = model(data, num_steps)
                _, pred = mem_out[-1].max(1)
                total_spikes += spk_hidden.sum().item()
                total_possible += spk_hidden.numel()
            else:
                _, pred = model(data).max(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = correct / total * 100
    fire_rate = total_spikes / total_possible if total_possible > 0 else 0.0

    print(f"\n  [{model_name}] RESULT: {acc:.2f}% inter-patient | Params: {n_params:,} | Fire: {fire_rate:.3f}")
    print(classification_report(all_targets, all_preds, target_names=['N', 'S', 'V', 'F', 'Q']))

    cm = confusion_matrix(all_targets, all_preds)
    print(f"  Confusion matrix:")
    print(f"    {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
        print(f"    {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

    # Energy estimate
    h1, h2 = model.h1, model.h2
    if is_snn:
        fc_macs = 16 * h1
        rec_macs = h1 * h1 * num_steps * fire_rate if hasattr(model, 'rlif1') else h2 * h2 * num_steps * fire_rate
        mid_macs = h1 * h2 * num_steps * fire_rate if hasattr(model, 'fc_mid') else 0
        out_macs = h2 * 5 * num_steps * fire_rate
        total_macs = fc_macs + rec_macs + mid_macs + out_macs
    else:
        total_macs = 16 * h1 + h1 * h2 + h2 * 5
        if hasattr(model.net, '6'):  # deep ANN has extra layer
            total_macs += h2 * (h2 // 2) + (h2 // 2) * 5

    cls_nJ = total_macs * 2.0 / 1000
    frontend_nJ = 13.7
    print(f"  Energy: {total_macs:,.0f} MACs, classifier={cls_nJ:.1f}nJ, frontend={frontend_nJ:.1f}nJ, total={cls_nJ+frontend_nJ:.1f}nJ")

    return acc, n_params


# ================================================================
# DATA LOADING
# ================================================================

print("Loading data...")
train_labels, train_features = extract_features(DS1_records)
test_labels, test_features = extract_features(DS2_records)
print(f"  DS1: {len(train_labels)} beats, DS2: {len(test_labels)} beats")
print(f"  Features: {len(FEATURE_NAMES)}")
print(f"  Class dist (train): {dict(Counter(train_labels))}")

# SMOTE
train_features, train_labels = smote_oversample(train_features, train_labels, 0.33)
print(f"  After SMOTE: {dict(Counter(train_labels))}")

# Normalize
tr_mean, tr_std = train_features.mean(0), train_features.std(0)
train_features = (train_features - tr_mean) / (tr_std + 1e-8)
test_features = (test_features - tr_mean) / (tr_std + 1e-8)

# Train/val split
tr_idx, val_idx = train_test_split(np.arange(len(train_labels)), test_size=0.1, random_state=42, stratify=train_labels)

class DS(torch.utils.data.Dataset):
    def __init__(self, f, l):
        self.data = torch.tensor(f, dtype=torch.float32)
        self.targets = torch.tensor(l, dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.targets[i]

train_ds = DS(train_features[tr_idx], train_labels[tr_idx])
val_ds = DS(train_features[val_idx], train_labels[val_idx])
test_ds = DS(test_features, test_labels)

tr_counts = np.bincount(train_labels[tr_idx], minlength=5)
sw = 1.0 / (tr_counts ** 0.65)
sample_w = [sw[l] for l in train_labels[tr_idx]]
sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, sampler=sampler, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

cw = 1.0 / (tr_counts.astype(np.float64) ** 0.65)
cw = cw / cw.sum() * 5
cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

num_features = train_features.shape[1]


# ================================================================
# RUN ALL ARCHITECTURES
# ================================================================

print(f"\n{'='*70}")
print(f"ARCHITECTURE COMPARISON (inter-patient, {num_features} features, QAT-4bit)")
print(f"{'='*70}")

results = {}

# 1. Pure ANN (48->24->5)
torch.manual_seed(42); np.random.seed(42)
m = PureANN(num_features, 48, 5).to(device)
acc, params = train_and_eval("ANN-48-24", m, train_loader, val_loader, test_loader, cw_tensor, device, is_snn=False)
results["ANN (48->24->5)"] = (acc, params)

# 2. Pure SNN (48->24->5, current architecture)
torch.manual_seed(42); np.random.seed(42)
m = PureSNN(num_features, 48, 5).to(device)
acc, params = train_and_eval("SNN-48-24", m, train_loader, val_loader, test_loader, cw_tensor, device, is_snn=True, num_steps=20)
results["SNN (48->24->5)"] = (acc, params)

# 3. Hybrid ANN+SNN (24 ANN -> 24 SNN -> 5)
torch.manual_seed(42); np.random.seed(42)
m = HybridANNSNN(num_features, 48, 5).to(device)
acc, params = train_and_eval("Hybrid-24-24", m, train_loader, val_loader, test_loader, cw_tensor, device, is_snn=True, num_steps=20)
results["Hybrid (ANN24->SNN24->5)"] = (acc, params)

# 4. Deep ANN (48->24->12->5)
torch.manual_seed(42); np.random.seed(42)
m = DeepANN(num_features, 48, 5).to(device)
acc, params = train_and_eval("ANN-Deep-48-24-12", m, train_loader, val_loader, test_loader, cw_tensor, device, is_snn=False)
results["Deep ANN (48->24->12->5)"] = (acc, params)

# 5. Wider SNN (64->32->5)
torch.manual_seed(42); np.random.seed(42)
m = PureSNN(num_features, 64, 5).to(device)
acc, params = train_and_eval("SNN-64-32", m, train_loader, val_loader, test_loader, cw_tensor, device, is_snn=True, num_steps=20)
results["SNN (64->32->5)"] = (acc, params)

# 6. ANN with same param count as SNN-48 (~5100 params)
torch.manual_seed(42); np.random.seed(42)
m = PureANN(num_features, 64, 5).to(device)
acc, params = train_and_eval("ANN-64-32", m, train_loader, val_loader, test_loader, cw_tensor, device, is_snn=False)
results["ANN (64->32->5)"] = (acc, params)

# Final summary
print(f"\n{'='*70}")
print(f"ARCHITECTURE COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"  {'Architecture':35s} {'Accuracy':>10} {'Params':>10}")
print(f"  {'-'*35} {'-'*10} {'-'*10}")
for name, (acc, params) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  {name:35s} {acc:9.2f}% {params:10,}")
print(f"\n  Random Forest baseline: 93.61%")
print(f"\n  Total time: {time.time()-t0:.0f}s")
