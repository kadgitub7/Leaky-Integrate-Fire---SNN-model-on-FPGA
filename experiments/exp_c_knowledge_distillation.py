"""
EXPERIMENT C: Knowledge Distillation from ANN Teacher
======================================================
Research identified this as the single biggest accuracy lever.

  1. Train a large ANN teacher (256→128→64) on same features
  2. Distill into the lean SNN student using soft targets
  3. Uses the lean 16-feature set from Exp B + all training improvements

The teacher is a standard MLP — no hardware constraints.
The student is the same memristive-crossbar-compatible SNN.

Run: python experiments/exp_c_knowledge_distillation.py --split inter
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

parser = argparse.ArgumentParser(description='Exp C: Knowledge distillation')
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--teacher_epochs', type=int, default=300)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--kd_alpha', type=float, default=0.7, help='Distillation weight (vs hard label)')
parser.add_argument('--kd_temp', type=float, default=4.0, help='Distillation temperature')
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
                    norm_c = np.linalg.norm(qrs_ds)
                    norm_t = np.linalg.norm(ewma_template[lead])
                    templ_corr = np.dot(qrs_ds, ewma_template[lead]) / (norm_c * norm_t) if norm_c > 1e-8 and norm_t > 1e-8 else 1.0
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
else:
    all_labels, all_features = extract_beats_and_features(all_record_names)
    train_idx, test_idx = train_test_split(np.arange(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels)
    train_features, test_features = all_features[train_idx], all_features[test_idx]
    train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

num_features = train_features.shape[1]
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
# TEACHER: Large ANN (no hardware constraints)
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
    def forward(self, x):
        return self.net(x)


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
            logits = teacher(data)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(teacher.state_dict())
            print(f"  [Teacher] Epoch {epoch+1}/{num_epochs} | Acc: {acc:.2f}% (best: {best_acc:.2f}%) | {time.time()-t0:.0f}s")

    if best_state:
        teacher.load_state_dict(best_state)
    return teacher, best_acc


# ================================================================
# STUDENT SNN
# ================================================================

class StudentSNN(torch.nn.Module):
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


def train_student_kd(student, teacher, num_epochs, num_steps, lambda_sparse, alpha, temp, label=""):
    """Train student SNN with knowledge distillation from ANN teacher."""
    hard_loss_fn = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    best_ce = float('inf')
    best_state = None

    teacher.eval()

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup_epochs

        student.train()
        epoch_loss = epoch_rate = 0.0
        batches = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # Teacher soft targets
            with torch.no_grad():
                teacher_logits = teacher(data)
                teacher_soft = F.softmax(teacher_logits / temp, dim=1)

            # QAT forward
            saved = []
            with torch.no_grad():
                for p in student.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize_tensor(p.data, 4))

            spk_out, mem_out, spk_hidden = student(data, num_steps)

            # Hard label loss (focal) on all timesteps
            hard_loss = sum(hard_loss_fn(mem_out[s], targets) for s in range(num_steps))

            # Soft distillation loss on final membrane potential
            student_log_soft = F.log_softmax(mem_out[-1] / temp, dim=1)
            kd_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temp ** 2)

            # Combined loss
            ce = (1 - alpha) * hard_loss + alpha * kd_loss * num_steps

            firing_rate = spk_hidden.mean()
            total_loss = ce + lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)

            optimizer.zero_grad()
            total_loss.backward()

            with torch.no_grad():
                for p, s in zip(student.parameters(), saved):
                    p.data.copy_(s)

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            epoch_loss += ce.item()
            epoch_rate += firing_rate.item()
            batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_ce = epoch_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(student.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f} | {time.time()-t0:.0f}s")

    if best_state:
        student.load_state_dict(best_state)
        with torch.no_grad():
            for p in student.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))
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


def evaluate_ann(net, loader):
    total = correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        net.eval()
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            _, pred = net(data).max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return correct / total * 100, all_preds, all_targets


# ================================================================
# MAIN
# ================================================================

hidden = args.hidden
num_steps = args.steps

# Step 1: Train teacher
print(f"\n{'='*65}")
print(f"STEP 1: Training ANN Teacher (256→128→64)")
print(f"{'='*65}")
teacher = TeacherANN(num_features, num_class).to(device)
teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"  Teacher parameters: {teacher_params:,}")
teacher, teacher_acc = train_teacher(teacher, args.teacher_epochs)

teacher_acc, teacher_preds, teacher_targets = evaluate_ann(teacher, test_loader)
print(f"\n  Teacher accuracy: {teacher_acc:.2f}%")
print(classification_report(teacher_targets, teacher_preds, target_names=['N', 'S', 'V', 'F', 'Q']))

# Step 2: Train student with KD
print(f"\n{'='*65}")
print(f"STEP 2: Training SNN Student with Knowledge Distillation")
print(f"  alpha={args.kd_alpha}, temperature={args.kd_temp}")
print(f"{'='*65}")

student = StudentSNN(num_features, hidden, num_class).to(device)
h1, h2 = student.h1, student.h2
n_params = sum(p.numel() for p in student.parameters())
print(f"  Student: {num_features}→{h1} RLeaky→{h2} RLeaky→5, {n_params:,} params")

student = train_student_kd(student, teacher, args.epochs, num_steps, args.lam,
                           args.kd_alpha, args.kd_temp, label="KD")

# Step 3: Also train a baseline student WITHOUT KD for comparison
print(f"\n{'='*65}")
print(f"STEP 3: Training SNN Student WITHOUT KD (baseline comparison)")
print(f"{'='*65}")

baseline = StudentSNN(num_features, hidden, num_class).to(device)
baseline = train_student_kd(baseline, teacher, args.epochs, num_steps, args.lam,
                            alpha=0.0, temp=1.0, label="NoKD")

# Evaluate both
kd_acc, kd_preds, kd_targets, kd_fire = evaluate(student, test_loader, num_steps)
bl_acc, bl_preds, bl_targets, bl_fire = evaluate(baseline, test_loader, num_steps)

print(f"\n{'='*65}")
print(f"RESULTS COMPARISON")
print(f"{'='*65}")
print(f"  Teacher (ANN, {teacher_params:,} params): {teacher_acc:.2f}%")
print(f"  Student w/ KD  ({n_params:,} params):     {kd_acc:.2f}%")
print(f"  Student no KD  ({n_params:,} params):     {bl_acc:.2f}%")
print(f"  KD improvement: {kd_acc - bl_acc:+.2f}%")

print(f"\n--- Student with KD ---")
print(classification_report(kd_targets, kd_preds, target_names=['N', 'S', 'V', 'F', 'Q']))
cm = confusion_matrix(kd_targets, kd_preds)
print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
    print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

print(f"\n--- Student without KD ---")
print(classification_report(bl_targets, bl_preds, target_names=['N', 'S', 'V', 'F', 'Q']))

print(f"\nTotal runtime: {time.time()-t0:.0f}s")
