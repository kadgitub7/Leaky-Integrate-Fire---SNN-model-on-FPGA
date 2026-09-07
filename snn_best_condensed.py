"""
CONDENSED BEST MODEL v2: Parameterized for A/B comparison
==========================================================
Run conservative vs aggressive to find the accuracy sweet spot:

  CONSERVATIVE (safe, ~45 nJ):
    python snn_best_condensed.py --split inter --hidden 48 --gamma 2.0 --cls_power 0.65 --label_smooth 0.0

  AGGRESSIVE (push minority classes, ~68 nJ):
    python snn_best_condensed.py --split inter --hidden 64 --gamma 3.0 --cls_power 0.5 --label_smooth 0.05

  Both use: 17 features, 400 epochs, SMOTE 0.5, QAT-4bit, focal loss, gated EWMA

  17 features from ~11 circuits (~130 nW):
    Timing (7):  pre_rr, post_rr, rr_ratio, post_rr_ratio, rr_asymmetry,
                 compensatory_ratio, rr_std_10
    Morph (10):  qrs_width x2, qrs_area x2, max_slope x2, rel_area x2, templ_corr x2
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

parser = argparse.ArgumentParser(description='Best condensed SNN model')
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--split', choices=['intra', 'inter', 'both'], default='inter')
parser.add_argument('--smote_ratio', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
parser.add_argument('--cls_power', type=float, default=0.65, help='Class weight power (lower=more aggressive)')
parser.add_argument('--label_smooth', type=float, default=0.0, help='Label smoothing factor')
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
num_class = 5
t0 = time.time()

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio', 'post_rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
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
                for j in range(max(1, i - 10), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833
                rr_ratio = pre_rr / (local_rr + 1e-8)

                post_rr_ratio = post_rr / (local_rr + 1e-8)
                rr_asymmetry = pre_rr / (post_rr + 1e-8)
                compensatory_ratio = (pre_rr + post_rr) / (2 * local_rr + 1e-8)

                rr_history.append(pre_rr)
                if len(rr_history) > 10:
                    rr_history.pop(0)
                rr_std_10 = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = rr_ratio
                feat[3] = post_rr_ratio
                feat[4] = rr_asymmetry
                feat[5] = compensatory_ratio
                feat[6] = rr_std_10

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

                    feat[7 + lead] = width
                    feat[9 + lead] = area
                    feat[11 + lead] = max_slope

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

                    feat[13 + lead] = area / (ewma_area[lead] + 1e-8)

                    norm_curr = np.linalg.norm(qrs_ds)
                    norm_tmpl = np.linalg.norm(ewma_template[lead])
                    if norm_curr > 1e-8 and norm_tmpl > 1e-8:
                        templ_corr = np.dot(qrs_ds, ewma_template[lead]) / (norm_curr * norm_tmpl)
                    else:
                        templ_corr = 1.0
                    feat[15 + lead] = templ_corr

                    if templ_corr > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

                all_features.append(feat)

        except Exception as e:
            print(f"Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features)


def smote_oversample(features, labels, target_ratio=0.5):
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
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
    def forward(self, input, target):
        ce = torch.nn.functional.cross_entropy(
            input, target, weight=self.weight, reduction='none',
            label_smoothing=self.label_smoothing)
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


def train_model(net, train_loader, class_weights_tensor, num_epochs, num_steps, lambda_sparse, device, label=""):
    loss_fn = FocalLoss(weight=class_weights_tensor, gamma=args.gamma, label_smoothing=args.label_smooth)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    best_ce = float('inf')
    best_state = None

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
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f} | {time.time()-t0:.0f}s")
    if best_state:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))
    return net


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


def print_results(acc, preds, targets, fire_rate, n_params, num_features, h1, h2, num_steps, split_name):
    print(f"\n{'='*65}")
    print(f"RESULTS: {acc:.2f}% ({split_name}-patient)")
    print(f"  Parameters: {n_params:,} | Firing rate: {fire_rate:.3f}")
    print(classification_report(targets, preds, target_names=['N', 'S', 'V', 'F', 'Q']))
    cm = confusion_matrix(targets, preds)
    print(f"Confusion matrix:")
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

    total_macs, cls_nJ, frontend_nJ, total_nJ = compute_energy(num_features, h1, h2, num_steps, fire_rate)
    print(f"\n  Energy: {total_macs:,.0f} MACs, classifier={cls_nJ:.1f}nJ @2pJ/MAC, frontend={frontend_nJ:.1f}nJ, total={total_nJ:.1f}nJ")


def run_split(split_name, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'#'*65}")
    print(f"# {split_name.upper()}-PATIENT EVALUATION (seed={seed})")
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
            np.arange(len(all_labels)), test_size=0.2, random_state=seed, stratify=all_labels)
        train_features, test_features = all_features[train_idx], all_features[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

    num_features = train_features.shape[1]

    print(f"  Before SMOTE: {np.bincount(train_labels, minlength=5)}")
    train_features, train_labels = smote_oversample(train_features, train_labels, target_ratio=args.smote_ratio)
    print(f"  After SMOTE (ratio={args.smote_ratio}): {np.bincount(train_labels, minlength=5)}")

    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_features = (train_features - train_mean) / (train_std + 1e-8)
    test_features = (test_features - train_mean) / (train_std + 1e-8)

    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.data = torch.tensor(features, dtype=torch.float32)
            self.targets = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx], self.targets[idx]

    train_dataset = FeatureDataset(train_features, train_labels)
    test_dataset = FeatureDataset(test_features, test_labels)
    train_label_counts = np.bincount(train_labels, minlength=num_class)
    class_sample_weights = 1.0 / (train_label_counts ** args.cls_power)
    sample_weights = [class_sample_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** args.cls_power)
    class_weights = class_weights / class_weights.sum() * num_class
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    hidden = args.hidden
    num_steps = args.steps
    net = MinimalSNN(num_features, hidden, num_class).to(device)
    h1, h2 = net.h1, net.h2
    n_params = sum(p.numel() for p in net.parameters())

    print(f"\n  Architecture: {num_features} -> BN -> {h1} RLeaky -> {h2} RLeaky -> 5 Leaky")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Hyperparams:  gamma={args.gamma}, cls_power={args.cls_power}, label_smooth={args.label_smooth}")

    label = f"{split_name}_h{hidden}_g{args.gamma}"
    net = train_model(net, train_loader, class_weights_tensor, args.epochs, num_steps, args.lam, device, label=label)
    acc, preds, targets, fire_rate = evaluate(net, test_loader, num_steps, device)
    print_results(acc, preds, targets, fire_rate, n_params, num_features, h1, h2, num_steps, split_name)
    hardware_robustness_sweep(net, test_loader, num_steps, device)

    _, _, _, total_nJ = compute_energy(num_features, h1, h2, num_steps, fire_rate)
    return acc, n_params, fire_rate, total_nJ


# ================================================================
# MAIN
# ================================================================

print(f"{'='*65}")
print(f"CONDENSED BEST MODEL v2")
print(f"  Features:     17 from ~11 circuits")
print(f"  Architecture: 17 -> BN -> {args.hidden} RLeaky -> {args.hidden//2} RLeaky -> 5")
print(f"  Training:     Focal(g={args.gamma}) + SMOTE(r={args.smote_ratio}) + QAT-4bit")
print(f"  Class weight: power={args.cls_power} | Label smooth: {args.label_smooth}")
print(f"  Epochs:       {args.epochs}")
print(f"{'='*65}")

splits_to_run = ['inter', 'intra'] if args.split == 'both' else [args.split]
results = {}
energy_nJ = 0.0

for split in splits_to_run:
    if args.runs > 1:
        accs = []
        for run in range(args.runs):
            seed = 42 + run * 7
            acc, n_params, fire_rate, total_nJ = run_split(split, seed=seed)
            accs.append(acc)
        print(f"\n{'='*65}")
        print(f"  {split.upper()}-PATIENT: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}% over {args.runs} runs")
        print(f"  Best: {max(accs):.2f}% | Worst: {min(accs):.2f}%")
        results[split] = np.mean(accs)
        energy_nJ = total_nJ
    else:
        acc, n_params, fire_rate, total_nJ = run_split(split)
        results[split] = acc
        energy_nJ = total_nJ

print(f"\n{'='*65}")
print(f"FINAL SUMMARY")
print(f"{'='*65}")
for split, acc in results.items():
    print(f"  {split:>5}-patient: {acc:.2f}%")
print(f"  Parameters:   {n_params:,}")
print(f"  Features:     17 from ~11 analog circuits")
print(f"  Energy:       {energy_nJ:.1f} nJ total per classification")
print(f"  Config:       hidden={args.hidden}, gamma={args.gamma}, cls_power={args.cls_power}, label_smooth={args.label_smooth}")
print(f"  Total time:   {time.time()-t0:.0f}s")
print(f"{'='*65}")
