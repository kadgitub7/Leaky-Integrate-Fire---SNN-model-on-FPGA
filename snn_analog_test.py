import sys
import argparse
import snntorch as snn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import wfdb
import os
import copy
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import time

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=None)
parser.add_argument('--lam', type=float, default=None)
args = parser.parse_args()

torch.set_num_threads(2)

batch_size = 128
dtype = torch.float

# ================================================================
# ANALOG FEATURE EXTRACTION (proven 412-feature bank)
# ================================================================
# Keep the full filter bank that achieves 97%+ accuracy.
# The power optimization happens in the PROCESSOR (smaller hidden,
# sparsity penalty) not the front-end. Benchmarks like Chu et al.
# report processor-only energy, so we compare apples to apples.

def rc_lowpass_batch(signals, alpha):
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, signals, axis=1)

def rc_highpass_batch(signals, alpha):
    return signals - rc_lowpass_batch(signals, alpha)

def compute_analog_features(beats, num_leads):
    num_beats, length, _ = beats.shape
    downsample = 8
    all_features = []

    for lead in range(num_leads):
        signals = beats[:, :, lead]
        all_features.append(signals[:, ::downsample])

        fast_hp = rc_highpass_batch(signals, 0.8)
        all_features.append(fast_hp[:, ::downsample])

        slow_hp = rc_highpass_batch(signals, 0.15)
        all_features.append(slow_hp[:, ::downsample])

        smooth_fast = rc_lowpass_batch(signals, 0.5)
        all_features.append(smooth_fast[:, ::downsample])

        smooth_slow = rc_lowpass_batch(signals, 0.08)
        all_features.append(smooth_slow[:, ::downsample])

        bandpass = smooth_fast - smooth_slow
        all_features.append(bandpass[:, ::downsample])

        all_features.append(np.abs(fast_hp[:, ::downsample]))

        second_deriv = rc_highpass_batch(rc_highpass_batch(signals, 0.7), 0.7)
        all_features.append(second_deriv[:, ::downsample])

        abs_signals = np.abs(signals)
        all_features.append(np.sum(abs_signals[:, 80:120], axis=1, keepdims=True))
        all_features.append(np.sum(abs_signals[:, 40:80], axis=1, keepdims=True))
        all_features.append(np.sum(abs_signals[:, 120:170], axis=1, keepdims=True))

        qrs_max = np.max(signals[:, 80:120], axis=1, keepdims=True)
        qrs_min = np.min(signals[:, 80:120], axis=1, keepdims=True)
        all_features.append(qrs_max)
        all_features.append(qrs_min)
        all_features.append(qrs_max - qrs_min)

    return np.hstack(all_features)


# ================================================================
# Data loading
# ================================================================

all_record_names = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]

N = ['N', 'L', 'R', 'e', 'j']
S = ['A', 'a', 'J', 'S']
V = ['V', 'E']
F = ['F']
Q = ['/', 'f', 'Q']

aami_map = {}
for sym in N: aami_map[sym] = 0
for sym in S: aami_map[sym] = 1
for sym in V: aami_map[sym] = 2
for sym in F: aami_map[sym] = 3
for sym in Q: aami_map[sym] = 4

win_left = 90
win_right = 108

def extract_beats_from_records(rec_list):
    beats, labels = [], []
    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            for idx, label in zip(annotation.sample, annotation.symbol):
                if idx - win_left >= 0 and idx + win_right < len(signals) and label in aami_map:
                    beat_segment = signals[idx - win_left : idx + win_right, :]
                    beats.append(beat_segment)
                    labels.append(label)
        except Exception as e:
            print(f"Skipping {rec_id}: {e}")
    return np.array(beats), labels

t0 = time.time()
print("Extracting all patients (intra-patient split)...")
all_beats, all_raw_labels = extract_beats_from_records(all_record_names)
print(f"  {len(all_beats)} beats ({time.time()-t0:.1f}s)")

num_leads = all_beats.shape[2]

t1 = time.time()
print("Computing analog features...")
all_features = compute_analog_features(all_beats, num_leads)
print(f"  Done ({time.time()-t1:.1f}s)")

num_features = all_features.shape[1]
print(f"Feature bank: {num_features} features, {num_leads} leads")

num_class = 5
all_numeric_labels = np.array([aami_map[l] for l in all_raw_labels])

train_idx, test_idx = train_test_split(
    np.arange(len(all_features)), test_size=0.2, random_state=42, stratify=all_numeric_labels
)

train_features = all_features[train_idx]
test_features = all_features[test_idx]
train_numeric_labels = all_numeric_labels[train_idx].tolist()
test_numeric_labels = all_numeric_labels[test_idx].tolist()

print(f"  Train: {len(train_features)}, Test: {len(test_features)} (80/20 stratified random split)")

train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_features = (train_features - train_mean) / (train_std + 1e-8)
test_features = (test_features - train_mean) / (train_std + 1e-8)

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

train_dataset = FeatureDataset(train_features, train_numeric_labels)
test_dataset = FeatureDataset(test_features, test_numeric_labels)

train_label_counts = np.bincount(train_numeric_labels, minlength=num_class)
class_sample_weights = 1.0 / (train_label_counts ** 0.65)
sample_weights = [class_sample_weights[l] for l in train_numeric_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** 0.65)
class_weights = class_weights / class_weights.sum() * num_class
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

beta = 0.9
num_steps = 40


# ================================================================
# Recurrent SNN with sparsity-aware training
# ================================================================

class RecurrentNet(torch.nn.Module):
    def __init__(self, n_features, hidden, n_classes, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, hidden)
        self.drop1 = torch.nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=hidden, learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        spk1_rec = []
        fc1_out = self.drop1(self.fc1(x))
        for step in range(num_steps):
            spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
            spk1_rec.append(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return (torch.stack(spk2_rec, dim=0),
                torch.stack(mem2_rec, dim=0),
                torch.stack(spk1_rec, dim=0))


# ================================================================
# Training with sparsity penalty
# ================================================================

def train_model(net, num_epochs, lambda_sparse, label=""):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    warmup_epochs = 5
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        if epoch < warmup_epochs:
            warmup_lr = 1e-3 * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        net.train()
        epoch_loss = 0.0
        epoch_sparsity = 0.0
        batches = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_out, mem_out, spk_hidden = net(data)

            ce_loss = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                ce_loss += loss_fn(mem_out[step], targets)

            firing_rate = spk_hidden.mean()
            sparsity_loss = lambda_sparse * firing_rate
            total_loss = ce_loss + sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce_loss.item()
            epoch_sparsity += firing_rate.item()
            batches += 1

        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        avg_rate = epoch_sparsity / batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {epoch_loss/batches:.2f} | "
                  f"fire_rate: {avg_rate:.3f} | {time.time()-epoch_start:.1f}s")

    return net


def evaluate_model(net):
    total = 0
    correct = 0
    all_preds = []
    all_targets = []
    total_spikes = 0
    total_possible = 0

    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_out, _, spk_hidden = net(data)
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_spikes += spk_hidden.sum().item()
            total_possible += spk_hidden.numel()

    acc = correct / total * 100
    avg_firing_rate = total_spikes / total_possible
    return acc, all_preds, all_targets, avg_firing_rate


# ================================================================
# Hardware simulation helpers
# ================================================================

def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x_q


def combined_hw_evaluate(net, num_bits, sigma, num_trials=10):
    accs = []
    frs = []
    for _ in range(num_trials):
        hw_net = copy.deepcopy(net)
        with torch.no_grad():
            for param in hw_net.parameters():
                param.copy_(quantize_tensor(param, num_bits))
                param.add_(torch.randn_like(param) * sigma)
        acc, _, _, fr = evaluate_model(hw_net)
        accs.append(acc)
        frs.append(fr)
    return np.mean(accs), np.std(accs), np.mean(frs)


# ================================================================
# Config selection: CLI args or full sweep
# ================================================================
from sklearn.metrics import classification_report, confusion_matrix

num_epochs = 250

if args.hidden is not None and args.lam is not None:
    configs = [(args.hidden, args.lam, f"h{args.hidden}_s{args.lam}")]
else:
    configs = [
        (64, 0.0,  "h64_s0"),
        (64, 1.0,  "h64_s1"),
    ]

pJ_per_MAC = 5.0

for hidden, lam, label in configs:
    print(f"\n{'='*60}")
    print(f"CONFIG: {label} (hidden={hidden}, lambda={lam})")
    print(f"{'='*60}")

    net = RecurrentNet(num_features, hidden, num_class).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")

    net = train_model(net, num_epochs, lambda_sparse=lam, label=label)
    acc, preds, targets, fire_rate = evaluate_model(net)

    w1_macs = num_features * hidden
    w_rec_macs = hidden * hidden * num_steps * fire_rate
    w2_macs = hidden * num_class * num_steps * fire_rate
    total_macs = w1_macs + w_rec_macs + w2_macs
    processor_nJ = total_macs * pJ_per_MAC / 1000

    print(f"\nRESULT: {label}")
    print(f"  Accuracy:    {acc:.2f}%")
    print(f"  Firing rate: {fire_rate:.3f} ({fire_rate*100:.1f}%)")
    print(f"  Parameters:  {n_params:,}")
    print(f"  MACs:        {total_macs:,.0f}")
    print(f"  Processor:   {processor_nJ:.0f} nJ (Chu: 750 nJ, ratio: {processor_nJ/750:.2f}x)")

    print(classification_report(targets, preds, target_names=['N','S','V','F','Q']))

    print(f"  Hardware sweep (4-bit + noise):")
    for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
        mean_acc, std_acc, fr = combined_hw_evaluate(net, num_bits=4, sigma=sigma)
        print(f"    sigma={sigma:.3f}  {mean_acc:.2f}% +/- {std_acc:.2f}%")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
