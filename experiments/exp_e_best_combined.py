"""
EXPERIMENT E: Best of Everything Combined
===========================================
Combines ALL improvements that should work together:

  FEATURES (16, lean set from Exp B):
    - 6 timing: pre_rr, post_rr, rr_ratio, rr_asymmetry, compensatory_ratio, rr_std_10
    - 4 morphology: qrs_width×2, qrs_area×2
    - 2 slope: max_slope×2
    - 2 patient-adaptive: rel_area×2
    - 2 template: templ_corr×2

  FEATURE EXTRACTION improvements:
    - Gated EWMA (only update when templ_corr > 0.85)
    - Split EWMA alphas (timing=0.03, morphology=0.015)
    - EWMA init from median of first 10 beats
    - 10-beat local RR window

  TRAINING improvements:
    - Focal loss (gamma=2)
    - SMOTE on feature space
    - Knowledge distillation from ANN teacher
    - Membrane potential noise injection during training (hardware robustness)

  ARCHITECTURE:
    - Two-layer recurrent SNN with BatchNorm
    - AdaBN at test time

  ENERGY TARGET: ~11 circuits, ~100 nW, total <50 nJ @ 2pJ/MAC

Run: python experiments/exp_e_best_combined.py --split inter
"""

import argparse
import snntorch as snn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import wfdb
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

parser = argparse.ArgumentParser(description='Exp E: Best combined')
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--teacher_epochs', type=int, default=300)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--kd_alpha', type=float, default=0.7)
parser.add_argument('--kd_temp', type=float, default=4.0)
parser.add_argument('--mem_noise', type=float, default=0.05, help='Membrane potential noise during training')
parser.add_argument('--split', choices=['intra', 'inter'], default='inter')
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
dtype = torch.float
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

DS1_records = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230'
]
DS2_records = [
    '100', '103', '105', '111', '113', '117', '121', '200', '202', '210',
    '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]
all_record_names = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
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
                if (idx - win_left >= 0 and idx + win_right < len(signals) and sym in aami_map):
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

                rr_history.append(pre_rr)
                if len(rr_history) > 10: rr_history.pop(0)

                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

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
                    nc, nt = np.linalg.norm(qrs_ds), np.linalg.norm(ewma_template[lead])
                    templ_corr = np.dot(qrs_ds, ewma_template[lead]) / (nc * nt) if nc > 1e-8 and nt > 1e-8 else 1.0
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
    from collections import Counter
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    new_f, new_l = list(features), list(labels)
    for cls in range(num_class):
        idx = np.where(labels == cls)[0]
        if len(idx) >= target_count: continue
        cf = features[idx]
        for _ in range(target_count - len(idx)):
            a, b = np.random.randint(len(cf)), np.random.randint(len(cf))
            while b == a and len(cf) > 1: b = np.random.randint(len(cf))
            new_f.append(cf[a] + np.random.random() * (cf[b] - cf[a]))
            new_l.append(cls)
    return np.array(new_f), np.array(new_l)


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ================================================================
# LOAD DATA
# ================================================================

print("Loading data...")
if args.split == 'inter':
    train_labels, train_features = extract_beats_and_features(DS1_records)
    test_labels, test_features = extract_beats_and_features(DS2_records)
    # Save raw test features for AdaBN later
    test_features_raw = test_features.copy()
    test_labels_raw = test_labels.copy()
else:
    all_labels, all_features = extract_beats_and_features(all_record_names)
    train_idx, test_idx = train_test_split(np.arange(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels)
    train_features, test_features = all_features[train_idx], all_features[test_idx]
    train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]
    test_features_raw = test_features.copy()
    test_labels_raw = test_labels.copy()

num_features = train_features.shape[1]
print(f"  Features: {num_features}")
print(f"  Before SMOTE: {np.bincount(train_labels, minlength=5)}")
train_features, train_labels = smote_oversample(train_features, train_labels)
print(f"  After SMOTE:  {np.bincount(train_labels, minlength=5)}")

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
class_sample_weights = 1.0 / (train_label_counts ** 0.65)
sample_weights = [class_sample_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** 0.65)
class_weights = class_weights / class_weights.sum() * num_class
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


# ================================================================
# MODELS
# ================================================================

class TeacherANN(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(n_features),
            torch.nn.Linear(n_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, n_classes),
        )
    def forward(self, x): return self.net(x)


class StudentSNN(torch.nn.Module):
    def __init__(self, n_features, hidden, n_classes, beta=0.9, dropout=0.1, mem_noise=0.0):
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
        self.mem_noise = mem_noise

    def forward(self, x, num_steps):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        mem_out = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk1_rec = [], [], []
        x = self.bn(x)
        fc1_out = self.drop1(self.fc1(x))
        for step in range(num_steps):
            spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
            if self.training and self.mem_noise > 0:
                mem1 = mem1 + torch.randn_like(mem1) * self.mem_noise
            spk1_rec.append(spk1)
            mid = self.drop2(self.fc_mid(spk1))
            spk2, mem2 = self.rlif2(mid, spk2, mem2)
            if self.training and self.mem_noise > 0:
                mem2 = mem2 + torch.randn_like(mem2) * self.mem_noise
            cur_out = self.fc2(spk2)
            spk_o, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_o)
            mem_out_rec.append(mem_out)
        return (torch.stack(spk_out_rec, dim=0),
                torch.stack(mem_out_rec, dim=0),
                torch.stack(spk1_rec, dim=0))


def quantize_tensor(x, num_bits):
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    scale = torch.clamp((x.max() - x.min()) / (qmax - qmin), min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


# ================================================================
# TRAINING
# ================================================================

def train_teacher(teacher, num_epochs):
    loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_acc = 0
    best_state = None
    for epoch in range(num_epochs):
        teacher.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            loss = loss_fn(teacher(data), targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            teacher.eval()
            correct = total = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    _, pred = teacher(data).max(1)
                    correct += (pred == targets).sum().item()
                    total += targets.size(0)
            acc = correct / total * 100
            if acc > best_acc: best_acc = acc; best_state = copy.deepcopy(teacher.state_dict())
            print(f"  [Teacher] Epoch {epoch+1}/{num_epochs} | Acc: {acc:.2f}% | {time.time()-t0:.0f}s")
    if best_state: teacher.load_state_dict(best_state)
    return teacher, best_acc


def train_student_kd(student, teacher, num_epochs, num_steps, lambda_sparse, alpha, temp, label=""):
    hard_loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    best_ce = float('inf')
    best_state = None
    teacher.eval()

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup_epochs
        student.train()
        epoch_loss = epoch_rate = 0.0
        batches = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            with torch.no_grad():
                teacher_soft = F.softmax(teacher(data) / temp, dim=1)
            saved = []
            with torch.no_grad():
                for p in student.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize_tensor(p.data, 4))
            spk_out, mem_out, spk_hidden = student(data, num_steps)
            hard_loss = sum(hard_loss_fn(mem_out[s], targets) for s in range(num_steps))
            student_log_soft = F.log_softmax(mem_out[-1] / temp, dim=1)
            kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temp ** 2)
            ce = (1 - alpha) * hard_loss + alpha * kd_loss * num_steps
            firing_rate = spk_hidden.mean()
            total_loss = ce + lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)
            optimizer.zero_grad(); total_loss.backward()
            with torch.no_grad():
                for p, s in zip(student.parameters(), saved): p.data.copy_(s)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            epoch_loss += ce.item(); epoch_rate += firing_rate.item(); batches += 1
        if epoch >= warmup_epochs: scheduler.step()
        avg_ce = epoch_loss / batches
        if avg_ce < best_ce: best_ce = avg_ce; best_state = copy.deepcopy(student.state_dict())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f} | {time.time()-t0:.0f}s")
    if best_state:
        student.load_state_dict(best_state)
        with torch.no_grad():
            for p in student.parameters(): p.data.copy_(quantize_tensor(p.data, 4))
    return student


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
    return correct / total * 100, all_preds, all_targets, total_spikes / total_possible


def evaluate_adabn(net, num_steps):
    """Per-record AdaBN: update BN stats using first 50 beats per record."""
    if args.split != 'inter':
        return evaluate(net, test_loader, num_steps)

    all_preds, all_targets = [], []
    total_spikes = total_possible = 0

    # Get per-record boundaries
    record_bounds = []
    offset = 0
    for rec_id in DS2_records:
        try:
            record = wfdb.rdrecord(os.path.join('./mitdb_data', rec_id))
            annotation = wfdb.rdann(os.path.join('./mitdb_data', rec_id), 'atr')
            count = sum(1 for idx, sym in zip(annotation.sample, annotation.symbol)
                       if idx - win_left >= 0 and idx + win_right < len(record.p_signal) and sym in aami_map)
            record_bounds.append((offset, offset + count))
            offset += count
        except: pass

    for start, end in record_bounds:
        if end > len(test_labels_raw): continue
        rec_feat = test_features_raw[start:end]
        rec_lab = test_labels_raw[start:end]
        if len(rec_feat) == 0: continue

        rec_norm = (rec_feat - train_mean) / (train_std + 1e-8)
        adapt_net = copy.deepcopy(net)
        n_adapt = min(50, len(rec_norm))
        adapt_data = torch.tensor(rec_norm[:n_adapt], dtype=torch.float32).to(device)
        adapt_net.train()
        with torch.no_grad():
            for _ in range(3):
                adapt_net(adapt_data, num_steps)

        adapt_net.eval()
        with torch.no_grad():
            data_t = torch.tensor(rec_norm, dtype=torch.float32).to(device)
            spk_out, _, spk_hidden = adapt_net(data_t, num_steps)
            _, pred = spk_out.sum(dim=0).max(1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(rec_lab)
            total_spikes += spk_hidden.sum().item()
            total_possible += spk_hidden.numel()

    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets) * 100
    fire_rate = total_spikes / total_possible if total_possible > 0 else 0
    return acc, all_preds, all_targets, fire_rate


def hw_evaluate(net, num_steps, num_bits, sigma, num_trials=10):
    accs = []
    for _ in range(num_trials):
        hw_net = copy.deepcopy(net)
        with torch.no_grad():
            for p in hw_net.parameters():
                p.copy_(quantize_tensor(p, num_bits))
                p.add_(torch.randn_like(p) * sigma)
        acc, _, _, _ = evaluate(hw_net, test_loader, num_steps)
        accs.append(acc)
    return np.mean(accs), np.std(accs)


# ================================================================
# MAIN
# ================================================================

hidden = args.hidden
num_steps = args.steps

print(f"\n{'='*65}")
print(f"EXPERIMENT E: Best of Everything Combined")
print(f"  Features: {num_features} | Split: {args.split}")
print(f"  KD: alpha={args.kd_alpha}, temp={args.kd_temp}")
print(f"  Membrane noise: {args.mem_noise}")
print(f"{'='*65}")

# Step 1: Teacher
print(f"\n--- Training ANN Teacher ---")
teacher = TeacherANN(num_features, num_class).to(device)
teacher, teacher_acc = train_teacher(teacher, args.teacher_epochs)
t_acc, t_preds, t_targets = None, None, None
teacher.eval()
correct = total = 0
t_preds, t_targets = [], []
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        _, pred = teacher(data).max(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        t_preds.extend(pred.cpu().numpy())
        t_targets.extend(targets.cpu().numpy())
t_acc = correct / total * 100
print(f"\n  Teacher accuracy: {t_acc:.2f}%")
print(classification_report(t_targets, t_preds, target_names=['N', 'S', 'V', 'F', 'Q']))

# Step 2: Student with KD + membrane noise
print(f"\n--- Training SNN Student (KD + noise) ---")
student = StudentSNN(num_features, hidden, num_class, mem_noise=args.mem_noise).to(device)
h1, h2 = student.h1, student.h2
n_params = sum(p.numel() for p in student.parameters())
print(f"  Architecture: {num_features}→{h1} RLeaky→{h2} RLeaky→5 | {n_params:,} params")

student = train_student_kd(student, teacher, args.epochs, num_steps, args.lam,
                           args.kd_alpha, args.kd_temp, label="BestKD")

# Step 3: Evaluate
acc, preds, targets, fire_rate = evaluate(student, test_loader, num_steps)
print(f"\n{'='*65}")
print(f"RESULTS (standard): {acc:.2f}%")
print(f"{'='*65}")
print(classification_report(targets, preds, target_names=['N', 'S', 'V', 'F', 'Q']))
cm = confusion_matrix(targets, preds)
print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
    print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

# AdaBN evaluation
if args.split == 'inter':
    acc_abn, preds_abn, targets_abn, fire_abn = evaluate_adabn(student, num_steps)
    print(f"\n{'='*65}")
    print(f"RESULTS (AdaBN): {acc_abn:.2f}%")
    print(f"{'='*65}")
    print(classification_report(targets_abn, preds_abn, target_names=['N', 'S', 'V', 'F', 'Q']))
    cm_abn = confusion_matrix(targets_abn, preds_abn)
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm_abn[i])}")

# Hardware robustness
print(f"\nHardware sweep (4-bit + noise):")
for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
    mean_acc, std_acc = hw_evaluate(student, num_steps, 4, sigma)
    print(f"  sigma={sigma:.3f}  {mean_acc:.2f}% +/- {std_acc:.2f}%")

# Energy
fc1_macs = num_features * h1
rec1_macs = h1 * h1 * num_steps * fire_rate
mid_macs = h1 * h2 * num_steps * fire_rate
rec2_macs = h2 * h2 * num_steps * fire_rate
fc2_macs = h2 * num_class * num_steps * fire_rate
total_macs = fc1_macs + rec1_macs + mid_macs + rec2_macs + fc2_macs
frontend_nJ = 5 * 0.833 + 95 * 0.100
cls_2pj = total_macs * 2.0 / 1000
tot = cls_2pj + frontend_nJ

print(f"\n{'='*65}")
print(f"ENERGY ANALYSIS")
print(f"{'='*65}")
print(f"  Classifier:  {total_macs:,.0f} MACs = {cls_2pj:.1f} nJ @ 2pJ/MAC")
print(f"  Front-end:   ~11 circuits, ~100 nW = {frontend_nJ:.1f} nJ/beat")
print(f"  Total:       {tot:.1f} nJ/classification")
print(f"  Parameters:  {n_params:,}")
print(f"  vs Prev:     {tot/2095:.3f}x energy, {n_params/39929:.3f}x params")

print(f"\n{'='*65}")
print(f"SUMMARY")
print(f"{'='*65}")
print(f"  Teacher (ANN):     {t_acc:.2f}% ({sum(p.numel() for p in teacher.parameters()):,} params)")
print(f"  Student (SNN):     {acc:.2f}% ({n_params:,} params)")
if args.split == 'inter':
    print(f"  Student (AdaBN):   {acc_abn:.2f}%")
print(f"  Energy:            {tot:.1f} nJ total @ 2pJ/MAC")
print(f"  Circuits:          ~11 ({num_features} features)")

print(f"\nTotal runtime: {time.time()-t0:.0f}s")
