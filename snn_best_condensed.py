"""
CONDENSED BEST MODEL v3: Fix overfitting, revert harmful changes
================================================================
Reverts to proven Exp B configuration (91.45% inter) and adds principled fixes:

  REVERTED (v2 mistakes):
    - Dropped post_rr_ratio (redundant, derivable from rr_ratio + compensatory_ratio)
    - Back to SMOTE 0.33 (0.5 was worse — too many synthetic minority samples)
    - Back to 16 features from ~11 circuits

  NEW (principled fixes):
    - Validation-based early stopping (10% train holdout for checkpoint selection)
    - Longer RR baseline window (20 beats instead of 10) for more stable local_rr
    - Patience-based stopping (stop after 30 epochs of no validation improvement)
    - Multi-seed ensemble option (majority vote, no new hardware)

  Architecture: 16 -> BN -> 48 RLeaky -> 24 RLeaky -> 5 Leaky
  Training: Focal(g=2) + SMOTE(0.33) + QAT-4bit + val-based checkpointing

Usage:
  python snn_best_condensed.py --split inter
  python snn_best_condensed.py --split intra
  python snn_best_condensed.py --split inter --ensemble 3    # 3-model majority vote
"""

import argparse
import snntorch as snn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import wfdb
import os
import copy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

parser = argparse.ArgumentParser(description='Best condensed SNN model v3')
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=40, help='Early stopping patience on val accuracy')
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--split', choices=['intra', 'inter', 'both'], default='inter')
parser.add_argument('--smote_ratio', type=float, default=0.33)
parser.add_argument('--rr_window', type=int, default=20, help='RR baseline averaging window')
parser.add_argument('--ensemble', type=int, default=1, help='Number of models for majority vote')
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
num_class = 5
t0 = time.time()

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1',
    'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1',
    'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
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
    n_feat = len(FEATURE_NAMES)
    rr_win = args.rr_window

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

            ewma_area = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - win_left : idx + win_right, :]
                all_labels.append(aami_map[sym])

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - rr_win), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833
                rr_ratio = pre_rr / (local_rr + 1e-8)

                rr_asymmetry = pre_rr / (post_rr + 1e-8)
                compensatory_ratio = (pre_rr + post_rr) / (2 * local_rr + 1e-8)

                rr_history.append(pre_rr)
                if len(rr_history) > rr_win:
                    rr_history.pop(0)
                rr_std_10 = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = rr_ratio
                feat[3] = rr_asymmetry
                feat[4] = compensatory_ratio
                feat[5] = rr_std_10

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

                    norm_curr = np.linalg.norm(qrs_ds)
                    norm_tmpl = np.linalg.norm(ewma_template[lead])
                    if norm_curr > 1e-8 and norm_tmpl > 1e-8:
                        templ_corr = np.dot(qrs_ds, ewma_template[lead]) / (norm_curr * norm_tmpl)
                    else:
                        templ_corr = 1.0
                    feat[14 + lead] = templ_corr

                    if templ_corr > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

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
        n_synthetic = target_count - len(cls_idx)
        cls_features = features[cls_idx]
        if len(cls_features) < 2:
            for _ in range(n_synthetic):
                noise = np.random.randn(features.shape[1]) * 0.05
                new_features.append(cls_features[0] + noise)
                new_labels.append(cls)
            continue
        for _ in range(n_synthetic):
            i = np.random.randint(len(cls_features))
            j = np.random.randint(len(cls_features))
            while j == i:
                j = np.random.randint(len(cls_features))
            lam = np.random.random()
            synthetic = cls_features[i] + lam * (cls_features[j] - cls_features[i])
            new_features.append(synthetic)
            new_labels.append(cls)
    return np.array(new_features), np.array(new_labels)


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, input, target):
        ce = torch.nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class MinimalSNN(torch.nn.Module):
    def __init__(self, n_features, hidden, n_classes, beta=0.9, dropout=0.1):
        super().__init__()
        h1 = hidden
        h2 = max(hidden // 2, 8)
        self.bn = torch.nn.BatchNorm1d(n_features)
        self.fc1 = torch.nn.Linear(n_features, h1)
        self.drop1 = torch.nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc_mid = torch.nn.Linear(h1, h2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2, learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(h2, n_classes)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2

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


def evaluate(net, loader, num_steps, device):
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
    return correct / total * 100, all_preds, all_targets, total_spikes / total_possible


def train_model(net, train_loader, val_loader, class_weights_tensor, num_epochs, num_steps, lambda_sparse, device, label=""):
    loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup_epochs
        net.train()
        epoch_loss = epoch_rate = 0.0
        batches = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            saved = []
            with torch.no_grad():
                for p in net.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize_tensor(p.data, 4))
            spk_out, mem_out, spk_hidden = net(data, num_steps)
            ce = sum(loss_fn(mem_out[s], targets) for s in range(num_steps))
            firing_rate = spk_hidden.mean()
            total_loss = ce + lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)
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
        val_acc, _, _, _ = evaluate(net, val_loader, num_steps, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(net.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f} | val: {val_acc:.2f}% (best: {best_val_acc:.2f}%) | {time.time()-t0:.0f}s")

        if patience_counter >= args.patience and epoch >= 50:
            print(f"  [{label}] Early stopping at epoch {epoch+1} (patience={args.patience}, best val={best_val_acc:.2f}%)")
            break

    if best_state:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))
    return net


def compute_energy(num_features, h1, h2, num_steps, fire_rate):
    fc1_macs = num_features * h1
    rec1_macs = h1 * h1 * num_steps * fire_rate
    mid_macs = h1 * h2 * num_steps * fire_rate
    rec2_macs = h2 * h2 * num_steps * fire_rate
    fc2_macs = h2 * num_class * num_steps * fire_rate
    total_macs = fc1_macs + rec1_macs + mid_macs + rec2_macs + fc2_macs
    frontend_nJ = 5 * 0.833 + 95 * 0.100
    cls_nJ = total_macs * 2.0 / 1000
    total_nJ = cls_nJ + frontend_nJ
    return total_macs, cls_nJ, frontend_nJ, total_nJ


def hardware_robustness_sweep(net, test_loader, num_steps, device):
    print("\n  Hardware Robustness Sweep:")
    for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
        accs = []
        for trial in range(3):
            net_copy = copy.deepcopy(net)
            if sigma > 0:
                with torch.no_grad():
                    for p in net_copy.parameters():
                        noise = torch.randn_like(p) * sigma * p.abs().mean()
                        p.add_(noise)
            acc, _, _, _ = evaluate(net_copy, test_loader, num_steps, device)
            accs.append(acc)
        print(f"    sigma={sigma:.2f}: {np.mean(accs):.2f}% (+/- {np.std(accs):.2f}%)")


def run_single_model(train_features, train_labels, test_features, test_labels, seed, split_name):
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_features = train_features.shape[1]

    tr_feat_smote, tr_lbl_smote = smote_oversample(train_features, train_labels, target_ratio=args.smote_ratio)
    print(f"  After SMOTE (ratio={args.smote_ratio}): {np.bincount(tr_lbl_smote, minlength=5)}")

    tr_mean = tr_feat_smote.mean(axis=0)
    tr_std = tr_feat_smote.std(axis=0)
    tr_feat_norm = (tr_feat_smote - tr_mean) / (tr_std + 1e-8)
    te_feat_norm = (test_features - tr_mean) / (tr_std + 1e-8)

    # Split 10% of training for validation-based checkpointing
    tr_idx, val_idx = train_test_split(
        np.arange(len(tr_lbl_smote)), test_size=0.1, random_state=seed, stratify=tr_lbl_smote)

    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.data = torch.tensor(features, dtype=torch.float32)
            self.targets = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx], self.targets[idx]

    train_dataset = FeatureDataset(tr_feat_norm[tr_idx], tr_lbl_smote[tr_idx])
    val_dataset = FeatureDataset(tr_feat_norm[val_idx], tr_lbl_smote[val_idx])
    test_dataset = FeatureDataset(te_feat_norm, test_labels)

    train_label_counts = np.bincount(tr_lbl_smote[tr_idx], minlength=num_class)
    class_sample_weights = 1.0 / (train_label_counts ** 0.65)
    sample_weights = [class_sample_weights[l] for l in tr_lbl_smote[tr_idx]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** 0.65)
    class_weights = class_weights / class_weights.sum() * num_class
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    hidden = args.hidden
    num_steps = args.steps
    net = MinimalSNN(num_features, hidden, num_class).to(device)
    h1, h2 = net.h1, net.h2
    n_params = sum(p.numel() for p in net.parameters())

    label = f"{split_name}_seed{seed}"
    net = train_model(net, train_loader, val_loader, class_weights_tensor, args.epochs, num_steps, args.lam, device, label=label)

    return net, test_loader, device, n_params, h1, h2, num_features


def run_split(split_name):
    print(f"\n{'#'*65}")
    print(f"# {split_name.upper()}-PATIENT EVALUATION")
    print(f"{'#'*65}")

    print("Loading data and extracting features...")
    if split_name == 'inter':
        train_labels, train_features = extract_beats_and_features(DS1_records)
        test_labels, test_features = extract_beats_and_features(DS2_records)
        print(f"  DS1 (train): {len(train_labels)} beats")
        print(f"  DS2 (test):  {len(test_labels)} beats")
    else:
        all_labels, all_features = extract_beats_and_features(all_record_names)
        train_idx, test_idx = train_test_split(
            np.arange(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels)
        train_features, test_features = all_features[train_idx], all_features[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

    print(f"  Before SMOTE: {np.bincount(train_labels, minlength=5)}")
    print(f"  RR baseline window: {args.rr_window} beats")

    n_models = args.ensemble
    if n_models == 1:
        net, test_loader, device, n_params, h1, h2, num_features = run_single_model(
            train_features, train_labels, test_features, test_labels, seed=42, split_name=split_name)

        print(f"\n  Architecture: {num_features} -> BN -> {h1} RLeaky -> {h2} RLeaky -> 5 Leaky")
        print(f"  Parameters:   {n_params:,}")

        acc, preds, targets, fire_rate = evaluate(net, test_loader, args.steps, device)

        print(f"\n{'='*65}")
        print(f"RESULTS: {acc:.2f}% ({split_name}-patient)")
        print(f"  Parameters: {n_params:,} | Firing rate: {fire_rate:.3f}")
        print(classification_report(targets, preds, target_names=['N', 'S', 'V', 'F', 'Q']))
        cm = confusion_matrix(targets, preds)
        print(f"Confusion matrix:")
        print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
        for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
            print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

        total_macs, cls_nJ, frontend_nJ, total_nJ = compute_energy(num_features, h1, h2, args.steps, fire_rate)
        print(f"\n  Energy: {total_macs:,.0f} MACs, classifier={cls_nJ:.1f}nJ @2pJ/MAC, frontend={frontend_nJ:.1f}nJ, total={total_nJ:.1f}nJ")

        hardware_robustness_sweep(net, test_loader, args.steps, device)
        return acc, n_params, total_nJ

    else:
        print(f"\n  ENSEMBLE: Training {n_models} models for majority vote...")
        models = []
        for i in range(n_models):
            seed = 42 + i * 13
            print(f"\n  --- Model {i+1}/{n_models} (seed={seed}) ---")
            net, test_loader, device, n_params, h1, h2, num_features = run_single_model(
                train_features, train_labels, test_features, test_labels, seed=seed, split_name=split_name)
            models.append(net)

        # Majority vote
        all_model_preds = []
        all_targets = []
        total_fire = 0.0
        for net in models:
            _, preds, targets, fire_rate = evaluate(net, test_loader, args.steps, device)
            all_model_preds.append(preds)
            total_fire += fire_rate
            if not all_targets:
                all_targets = targets
        avg_fire = total_fire / n_models

        ensemble_preds = []
        for j in range(len(all_targets)):
            votes = [all_model_preds[m][j] for m in range(n_models)]
            ensemble_preds.append(max(set(votes), key=votes.count))

        acc = sum(p == t for p, t in zip(ensemble_preds, all_targets)) / len(all_targets) * 100

        # Individual model accuracies
        for i in range(n_models):
            ind_acc = sum(p == t for p, t in zip(all_model_preds[i], all_targets)) / len(all_targets) * 100
            print(f"  Model {i+1} individual: {ind_acc:.2f}%")

        print(f"\n{'='*65}")
        print(f"ENSEMBLE RESULTS: {acc:.2f}% ({split_name}-patient, {n_models}-model majority vote)")
        print(f"  Parameters per model: {n_params:,} | Avg firing rate: {avg_fire:.3f}")
        print(classification_report(all_targets, ensemble_preds, target_names=['N', 'S', 'V', 'F', 'Q']))
        cm = confusion_matrix(all_targets, ensemble_preds)
        print(f"Confusion matrix:")
        print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
        for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
            print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

        total_macs, cls_nJ, frontend_nJ, total_nJ = compute_energy(num_features, h1, h2, args.steps, avg_fire)
        print(f"\n  Energy (per model): {total_macs:,.0f} MACs, classifier={cls_nJ:.1f}nJ, frontend={frontend_nJ:.1f}nJ, total={total_nJ:.1f}nJ")

        return acc, n_params, total_nJ


# ================================================================
# MAIN
# ================================================================

print(f"{'='*65}")
print(f"CONDENSED BEST MODEL v3")
print(f"  Features:     16 from ~11 circuits")
print(f"  Architecture: 16 -> BN -> {args.hidden} RLeaky -> {args.hidden//2} RLeaky -> 5")
print(f"  Training:     Focal(g=2) + SMOTE(r={args.smote_ratio}) + QAT-4bit")
print(f"  Checkpoint:   Validation-based (10% holdout) + patience={args.patience}")
print(f"  RR window:    {args.rr_window} beats")
print(f"  Ensemble:     {args.ensemble} model(s)")
print(f"{'='*65}")

splits_to_run = ['inter', 'intra'] if args.split == 'both' else [args.split]
results = {}
energy_nJ = 0.0

for split in splits_to_run:
    acc, n_params, total_nJ = run_split(split)
    results[split] = acc
    energy_nJ = total_nJ

print(f"\n{'='*65}")
print(f"FINAL SUMMARY")
print(f"{'='*65}")
for split, acc in results.items():
    print(f"  {split:>5}-patient: {acc:.2f}%")
print(f"  Parameters:   {n_params:,}")
print(f"  Features:     16 from ~11 analog circuits")
print(f"  Energy:       {energy_nJ:.1f} nJ total per classification")
print(f"  RR window:    {args.rr_window} beats")
print(f"  SMOTE ratio:  {args.smote_ratio}")
print(f"  Ensemble:     {args.ensemble} model(s)")
print(f"  Total time:   {time.time()-t0:.0f}s")
print(f"{'='*65}")
