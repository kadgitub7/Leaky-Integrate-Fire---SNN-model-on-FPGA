"""
EXPERIMENT A: Training Pipeline Improvements Only
==================================================
Same 19 features and architecture as baseline, but with:
  1. Focal loss (gamma=2) instead of CrossEntropyLoss
  2. SMOTE on feature space for minority classes
  3. AdaBN at test time (update BN stats on first 50 beats per patient)
  4. Gated EWMA (only update when templ_corr > 0.85)
  5. Split EWMA alphas (0.03 timing, 0.015 morphology)
  6. EWMA init from median of first 10 beats
  7. Local RR window expanded from 5 to 10 beats

Goal: Measure how much training pipeline alone improves over baseline.

Run: python experiments/exp_a_training_improvements.py --split inter
"""

import argparse
import snntorch as snn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import wfdb
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

parser = argparse.ArgumentParser(description='Exp A: Training improvements')
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--split', choices=['intra', 'inter'], default='inter')
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
dtype = torch.float
num_class = 5
t0 = time.time()

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio',
    'peak_amp_L0', 'peak_amp_L1',
    'qrs_width_L0', 'qrs_width_L1',
    'qrs_area_L0', 'qrs_area_L1',
    'polarity_L0', 'polarity_L1',
    'rel_peak_L0', 'rel_peak_L1',
    'rel_width_L0', 'rel_width_L1',
    'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
]

EWMA_ALPHA_TIMING = 0.03
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

all_record_names = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]
DS1_records = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]
DS2_records = [
    '100', '103', '105', '111', '113', '117', '121', '200', '202', '210',
    '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]


def extract_beats_and_features(rec_list):
    all_labels = []
    all_features = []

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
                if (idx - win_left >= 0 and
                    idx + win_right < len(signals) and
                    sym in aami_map):
                    valid.append((idx, sym))

            ewma_peak = [None] * num_leads
            ewma_width = [None] * num_leads
            ewma_area = [None] * num_leads
            ewma_template = [None] * num_leads

            init_peaks = [[] for _ in range(num_leads)]
            init_widths = [[] for _ in range(num_leads)]
            init_areas = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - win_left : idx + win_right, :]
                all_labels.append(aami_map[sym])

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                # Expanded local RR window: 10 beats instead of 5
                local_rrs = []
                for j in range(max(1, i - 10), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833
                rr_ratio = pre_rr / (local_rr + 1e-8)

                feat = np.zeros(19, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = rr_ratio

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

                    polarity = 0.0
                    if peak > 1e-6:
                        max_idx = np.argmax(abs_qrs)
                        polarity = 1.0 if qrs[max_idx] >= 0 else -1.0

                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]

                    feat[3 + lead] = peak
                    feat[5 + lead] = width
                    feat[7 + lead] = area
                    feat[9 + lead] = polarity

                    # EWMA init from median of first EWMA_INIT_BEATS
                    if i < EWMA_INIT_BEATS:
                        init_peaks[lead].append(max(peak, 1e-6))
                        init_widths[lead].append(max(width, 1e-6))
                        init_areas[lead].append(max(area, 1e-6))
                        init_templates[lead].append(qrs_ds.copy())

                    if ewma_peak[lead] is None:
                        if i >= EWMA_INIT_BEATS - 1 and len(init_peaks[lead]) >= EWMA_INIT_BEATS:
                            ewma_peak[lead] = np.median(init_peaks[lead])
                            ewma_width[lead] = np.median(init_widths[lead])
                            ewma_area[lead] = np.median(init_areas[lead])
                            ewma_template[lead] = np.median(init_templates[lead], axis=0)
                        else:
                            ewma_peak[lead] = max(peak, 1e-6)
                            ewma_width[lead] = max(width, 1e-6)
                            ewma_area[lead] = max(area, 1e-6)
                            ewma_template[lead] = qrs_ds.copy()

                    feat[11 + lead] = peak / (ewma_peak[lead] + 1e-8)
                    feat[13 + lead] = width / (ewma_width[lead] + 1e-8)
                    feat[15 + lead] = area / (ewma_area[lead] + 1e-8)

                    norm_curr = np.linalg.norm(qrs_ds)
                    norm_tmpl = np.linalg.norm(ewma_template[lead])
                    if norm_curr > 1e-8 and norm_tmpl > 1e-8:
                        templ_corr = np.dot(qrs_ds, ewma_template[lead]) / (norm_curr * norm_tmpl)
                    else:
                        templ_corr = 1.0
                    feat[17 + lead] = templ_corr

                    # Gated EWMA: only update if beat looks normal
                    if templ_corr > EWMA_GATE_THRESH:
                        a_morph = EWMA_ALPHA_MORPH
                        ewma_peak[lead] = a_morph * max(peak, 1e-6) + (1 - a_morph) * ewma_peak[lead]
                        ewma_width[lead] = a_morph * max(width, 1e-6) + (1 - a_morph) * ewma_width[lead]
                        ewma_area[lead] = a_morph * max(area, 1e-6) + (1 - a_morph) * ewma_area[lead]
                        ewma_template[lead] = a_morph * qrs_ds + (1 - a_morph) * ewma_template[lead]

                all_features.append(feat)

        except Exception as e:
            print(f"Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features)


# ================================================================
# SMOTE on feature space
# ================================================================

def smote_oversample(features, labels, target_ratio=0.33):
    from collections import Counter
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)

    new_features = list(features)
    new_labels = list(labels)

    for cls in range(num_class):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) >= target_count:
            continue
        n_synthetic = target_count - len(cls_idx)
        cls_features = features[cls_idx]

        for _ in range(n_synthetic):
            i = np.random.randint(len(cls_features))
            j = np.random.randint(len(cls_features))
            while j == i and len(cls_features) > 1:
                j = np.random.randint(len(cls_features))
            lam = np.random.random()
            synthetic = cls_features[i] + lam * (cls_features[j] - cls_features[i])
            new_features.append(synthetic)
            new_labels.append(cls)

    return np.array(new_features), np.array(new_labels)


# ================================================================
# FOCAL LOSS
# ================================================================

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce = torch.nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ================================================================
# LOAD DATA
# ================================================================

print("Loading data and extracting features...")
print("  Improvements: focal loss, SMOTE, AdaBN, gated EWMA, split alphas, median init, RR window=10")

if args.split == 'inter':
    print("  Mode: INTER-PATIENT (AAMI DS1 train / DS2 test)")
    train_labels, train_features = extract_beats_and_features(DS1_records)
    test_labels, test_features = extract_beats_and_features(DS2_records)
    print(f"  DS1 (train): {len(train_labels)} beats")
    print(f"  DS2 (test):  {len(test_labels)} beats")
else:
    print("  Mode: INTRA-PATIENT (80/20 random split)")
    all_labels, all_features = extract_beats_and_features(all_record_names)
    print(f"  Total: {len(all_labels)} beats")
    train_idx, test_idx = train_test_split(
        np.arange(len(all_labels)), test_size=0.2, random_state=42,
        stratify=all_labels)
    train_features, test_features = all_features[train_idx], all_features[test_idx]
    train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

num_features = train_features.shape[1]

# SMOTE oversample minority classes
print(f"\n  Class distribution before SMOTE: {np.bincount(train_labels, minlength=5)}")
train_features, train_labels = smote_oversample(train_features, train_labels, target_ratio=0.33)
print(f"  Class distribution after SMOTE:  {np.bincount(train_labels, minlength=5)}")

print(f"  Features: {num_features}")
print(f"  ({time.time()-t0:.1f}s)")

train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_features = (train_features - train_mean) / (train_std + 1e-8)
test_features = (test_features - train_mean) / (train_std + 1e-8)


# ================================================================
# DATA LOADERS
# ================================================================

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

train_dataset = FeatureDataset(train_features, train_labels)
test_dataset = FeatureDataset(test_features, test_labels)

train_label_counts = np.bincount(train_labels, minlength=num_class)
class_sample_weights = 1.0 / (train_label_counts ** 0.65)
sample_weights = [class_sample_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"Device: {device}")

class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** 0.65)
class_weights = class_weights / class_weights.sum() * num_class
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


# ================================================================
# SNN (same architecture as baseline)
# ================================================================

class MinimalSNN(torch.nn.Module):
    def __init__(self, n_features, hidden, n_classes, beta=0.9, dropout=0.1):
        super().__init__()
        h1 = hidden
        h2 = max(hidden // 2, 8)
        self.bn = torch.nn.BatchNorm1d(n_features)
        self.fc1 = torch.nn.Linear(n_features, h1)
        self.drop1 = torch.nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1,
                                learn_beta=True, learn_threshold=True)
        self.fc_mid = torch.nn.Linear(h1, h2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2,
                                learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(h2, n_classes)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1 = h1
        self.h2 = h2

    def forward(self, x, num_steps):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        mem_out = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk1_rec = [], [], []

        x = self.bn(x)
        fc1_out = self.drop1(self.fc1(x))
        for step in range(num_steps):
            spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
            spk1_rec.append(spk1)
            mid = self.drop2(self.fc_mid(spk1))
            spk2, mem2 = self.rlif2(mid, spk2, mem2)
            cur_out = self.fc2(spk2)
            spk_o, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_o)
            mem_out_rec.append(mem_out)

        return (torch.stack(spk_out_rec, dim=0),
                torch.stack(mem_out_rec, dim=0),
                torch.stack(spk1_rec, dim=0))


def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


# ================================================================
# TRAINING with Focal Loss
# ================================================================

def train_model(net, num_epochs, num_steps, lambda_sparse, label=""):
    loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs)

    best_ce = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup_epochs

        net.train()
        epoch_loss = 0.0
        epoch_rate = 0.0
        batches = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            saved = []
            with torch.no_grad():
                for p in net.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize_tensor(p.data, 4))

            spk_out, mem_out, spk_hidden = net(data, num_steps)

            ce = torch.zeros(1, dtype=dtype, device=device)
            for s in range(num_steps):
                ce += loss_fn(mem_out[s], targets)

            firing_rate = spk_hidden.mean()
            sparsity = lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)
            total_loss = ce + sparsity

            optimizer.zero_grad()
            total_loss.backward()

            with torch.no_grad():
                for p, s in zip(net.parameters(), saved):
                    p.data.copy_(s)

            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce.item()
            epoch_rate += firing_rate.item()
            batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_ce = epoch_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"CE: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f} | "
                  f"{time.time()-t0:.0f}s")

    if best_state is not None:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))
        print(f"  [{label}] Restored best (CE={best_ce:.3f}), quantized to 4-bit")

    return net


# ================================================================
# EVALUATION with AdaBN
# ================================================================

def evaluate_with_adabn(net, test_features_raw, test_labels_raw, num_steps, train_mean, train_std):
    """Run AdaBN: update BN stats per-patient using first 50 beats, then evaluate."""
    if args.split != 'inter':
        return evaluate(net, test_loader, num_steps)

    # For inter-patient: process per-record to apply AdaBN
    all_preds = []
    all_targets = []
    total_spikes = 0
    total_possible = 0

    # We need per-record features. Re-extract to get record boundaries.
    record_starts = []
    offset = 0
    for rec_id in DS2_records:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            record = wfdb.rdrecord(record_path)
            count = sum(1 for idx, sym in zip(annotation.sample, annotation.symbol)
                       if idx - win_left >= 0 and idx + win_right < len(record.p_signal) and sym in aami_map)
            record_starts.append((offset, offset + count))
            offset += count
        except:
            pass

    for start, end in record_starts:
        if start >= len(test_labels_raw) or end > len(test_labels_raw):
            continue
        rec_features = test_features_raw[start:end]
        rec_labels = test_labels_raw[start:end]
        if len(rec_features) == 0:
            continue

        rec_norm = (rec_features - train_mean) / (train_std + 1e-8)

        # AdaBN: run first 50 beats through BN in train mode to adapt stats
        adapt_net = copy.deepcopy(net)
        n_adapt = min(50, len(rec_norm))
        adapt_data = torch.tensor(rec_norm[:n_adapt], dtype=torch.float32).to(device)
        adapt_net.train()
        with torch.no_grad():
            for _ in range(3):  # multiple passes for stable stats
                adapt_net(adapt_data, num_steps)

        # Now evaluate in eval mode
        adapt_net.eval()
        rec_data = torch.tensor(rec_norm, dtype=torch.float32).to(device)
        rec_targets = torch.tensor(rec_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            spk_out, _, spk_hidden = adapt_net(rec_data, num_steps)
            _, pred = spk_out.sum(dim=0).max(1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(rec_targets.cpu().numpy())
            total_spikes += spk_hidden.sum().item()
            total_possible += spk_hidden.numel()

    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets) * 100
    fire_rate = total_spikes / total_possible if total_possible > 0 else 0
    return acc, all_preds, all_targets, fire_rate


def evaluate(net, loader, num_steps):
    total = correct = 0
    all_preds, all_targets = [], []
    total_spikes = total_possible = 0
    with torch.no_grad():
        net.eval()
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, _, spk_hidden = net(data, num_steps)
            _, pred = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_spikes += spk_hidden.sum().item()
            total_possible += spk_hidden.numel()
    acc = correct / total * 100
    fire_rate = total_spikes / total_possible
    return acc, all_preds, all_targets, fire_rate


# ================================================================
# MAIN
# ================================================================

hidden = args.hidden
num_steps = args.steps
num_epochs = args.epochs
lambda_sparse = args.lam
label = f"expA_h{hidden}_s{num_steps}"

net = MinimalSNN(num_features, hidden, num_class).to(device)
h1, h2 = net.h1, net.h2
n_params = sum(p.numel() for p in net.parameters())

print(f"\n{'='*65}")
print(f"EXPERIMENT A: Training Improvements")
print(f"  Architecture: {num_features} -> {h1} RLeaky -> {h2} RLeaky -> 5 Leaky")
print(f"  Parameters:   {n_params:,}")
print(f"  Split:        {args.split}")
print(f"  Improvements: Focal loss, SMOTE, gated EWMA, split alphas, median init")
print(f"{'='*65}")

net = train_model(net, num_epochs, num_steps, lambda_sparse, label=label)

# Standard evaluation
acc, preds, targets, fire_rate = evaluate(net, test_loader, num_steps)
print(f"\n{'='*65}")
print(f"RESULTS (standard eval): {acc:.2f}%")
print(classification_report(targets, preds, target_names=['N', 'S', 'V', 'F', 'Q']))
cm = confusion_matrix(targets, preds)
print(f"Confusion matrix:")
print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
    print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

# AdaBN evaluation
if args.split == 'inter':
    # Re-extract raw (unnormalized) test features for AdaBN
    test_labels_raw, test_features_raw = extract_beats_and_features(DS2_records)
    acc_abn, preds_abn, targets_abn, fire_abn = evaluate_with_adabn(
        net, test_features_raw, test_labels_raw, num_steps, train_mean, train_std)
    print(f"\n{'='*65}")
    print(f"RESULTS (AdaBN eval): {acc_abn:.2f}%")
    print(classification_report(targets_abn, preds_abn, target_names=['N', 'S', 'V', 'F', 'Q']))
    cm_abn = confusion_matrix(targets_abn, preds_abn)
    print(f"Confusion matrix (AdaBN):")
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm_abn[i])}")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
