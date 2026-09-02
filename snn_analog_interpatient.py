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
from sklearn.metrics import classification_report, confusion_matrix
import time

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=80)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--split', type=str, default='inter', choices=['intra', 'inter'],
                    help='intra = 80/20 random split, inter = AAMI DS1/DS2 patient split')
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

torch.set_num_threads(2)

batch_size = 128
dtype = torch.float

# ================================================================
# ANALOG FEATURE EXTRACTION (identical to snn_analog_behavorial.py)
# ================================================================

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
# AAMI DS1/DS2 inter-patient split (de Chazal et al. 2004)
# This is the standard used by Chu et al. 2022 and most
# inter-patient evaluation papers.
# ================================================================

DS1_records = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]

DS2_records = [
    '100', '103', '105', '111', '113', '117', '121',
    '200', '202', '210', '212', '213', '214', '219',
    '221', '222', '228', '231', '232', '233', '234'
]

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

if args.split == 'inter':
    print("=" * 60)
    print("INTER-PATIENT evaluation (AAMI DS1/DS2 split)")
    print("  DS1 (train): 22 records")
    print("  DS2 (test):  21 records")
    print("  No patient overlap between train and test")
    print("=" * 60)

    print("\nExtracting DS1 (training) beats...")
    train_beats, train_raw_labels = extract_beats_from_records(DS1_records)
    print(f"  DS1: {len(train_beats)} beats")

    print("Extracting DS2 (testing) beats...")
    test_beats, test_raw_labels = extract_beats_from_records(DS2_records)
    print(f"  DS2: {len(test_beats)} beats")

    num_leads = train_beats.shape[2]

    print("Computing analog features (DS1)...")
    train_features = compute_analog_features(train_beats, num_leads)
    print("Computing analog features (DS2)...")
    test_features = compute_analog_features(test_beats, num_leads)

    train_numeric_labels = np.array([aami_map[l] for l in train_raw_labels]).tolist()
    test_numeric_labels = np.array([aami_map[l] for l in test_raw_labels]).tolist()

else:
    print("=" * 60)
    print("INTRA-PATIENT evaluation (80/20 stratified random split)")
    print("=" * 60)

    print("\nExtracting all patients...")
    all_beats, all_raw_labels = extract_beats_from_records(all_record_names)
    print(f"  {len(all_beats)} beats")

    num_leads = all_beats.shape[2]

    print("Computing analog features...")
    all_features_data = compute_analog_features(all_beats, num_leads)

    all_numeric_labels = np.array([aami_map[l] for l in all_raw_labels])

    train_idx, test_idx = train_test_split(
        np.arange(len(all_features_data)), test_size=0.2, random_state=42,
        stratify=all_numeric_labels
    )

    train_features = all_features_data[train_idx]
    test_features = all_features_data[test_idx]
    train_numeric_labels = all_numeric_labels[train_idx].tolist()
    test_numeric_labels = all_numeric_labels[test_idx].tolist()

num_features = train_features.shape[1]
print(f"\nFeature bank: {num_features} features")
print(f"  Train: {len(train_features)}, Test: {len(test_features)}")

train_counts = np.bincount(train_numeric_labels, minlength=5)
test_counts = np.bincount(test_numeric_labels, minlength=5)
print(f"  Train class dist: N={train_counts[0]}, S={train_counts[1]}, V={train_counts[2]}, F={train_counts[3]}, Q={train_counts[4]}")
print(f"  Test  class dist: N={test_counts[0]}, S={test_counts[1]}, V={test_counts[2]}, F={test_counts[3]}, Q={test_counts[4]}")

num_class = 5

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
# Recurrent SNN (identical architecture)
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
        spk2_rec, mem2_rec, spk1_rec = [], [], []
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
# QAT Training with target-rate sparsity + best checkpoint
# ================================================================

def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x_q


def train_model(net, num_epochs, lambda_sparse, label=""):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    warmup_epochs = 5
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    best_ce = float('inf')
    best_state = None

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

            saved_weights = []
            with torch.no_grad():
                for param in net.parameters():
                    saved_weights.append(param.data.clone())
                    param.data.copy_(quantize_tensor(param.data, 4))

            spk_out, mem_out, spk_hidden = net(data)

            ce_loss = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                ce_loss += loss_fn(mem_out[step], targets)

            firing_rate = spk_hidden.mean()
            target_rate = 0.15
            sparsity_loss = lambda_sparse * torch.clamp(firing_rate - target_rate, min=0.0)
            total_loss = ce_loss + sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()

            with torch.no_grad():
                for param, saved in zip(net.parameters(), saved_weights):
                    param.data.copy_(saved)

            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce_loss.item()
            epoch_sparsity += firing_rate.item()
            batches += 1

        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        avg_ce = epoch_loss / batches
        avg_rate = epoch_sparsity / batches

        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {avg_ce:.2f} | "
                  f"fire_rate: {avg_rate:.3f} | {time.time()-epoch_start:.1f}s")

    if best_state is not None:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for param in net.parameters():
                param.data.copy_(quantize_tensor(param.data, 4))
        print(f"  [{label}] Restored best model (CE={best_ce:.3f}), weights quantized to 4-bit")

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
# Multi-point energy analysis
# ================================================================

def energy_analysis(total_macs, fire_rate, hidden, num_features_in):
    w1_macs = num_features_in * hidden
    w_rec_macs = hidden * hidden * num_steps * fire_rate
    w2_macs = hidden * num_class * num_steps * fire_rate
    total = w1_macs + w_rec_macs + w2_macs

    print(f"\n{'='*60}")
    print(f"ENERGY ANALYSIS: Multi-point pJ/MAC sweep")
    print(f"{'='*60}")
    print(f"  Total MACs: {total:,.0f}")
    print(f"  Breakdown: W1={w1_macs:,.0f} (once), W_rec={w_rec_macs:,.0f} (spike-driven), W2={w2_macs:,.0f} (spike-driven)")
    print(f"  Firing rate: {fire_rate:.3f} ({fire_rate*100:.1f}%)")
    print()

    energy_points = [
        (0.1,  "Ideal memristive CIM (projected, sub-nm)"),
        (0.5,  "Optimistic RRAM crossbar (advanced node)"),
        (1.0,  "High-efficiency RRAM CIM (65nm, recent demos)"),
        (2.0,  "Good memristive crossbar (65-90nm)"),
        (3.0,  "Typical RRAM crossbar (90-130nm)"),
        (5.0,  "Conservative memristive estimate (our baseline)"),
        (7.0,  "RRAM with peripheral overhead"),
        (10.0, "Including ADC/DAC peripheral circuits"),
        (15.0, "Pessimistic analog CIM estimate"),
        (20.0, "Worst-case with full peripheral overhead"),
    ]

    chu_nJ = 750.0

    print(f"  {'pJ/MAC':>8} | {'Energy (nJ)':>12} | {'vs Chu':>8} | Source assumption")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*40}")
    for pj, desc in energy_points:
        nJ = total * pj / 1000
        ratio = nJ / chu_nJ
        print(f"  {pj:>7.1f}  | {nJ:>11.1f}  | {ratio:>7.2f}x | {desc}")

    print()
    print(f"  Reference: Chu et al. 2022 = {chu_nJ:.0f} nJ (40nm digital SNN)")
    print(f"  Note: Even at 10 pJ/MAC (pessimistic), energy = {total * 10 / 1000:.0f} nJ = {total * 10 / 1000 / chu_nJ:.2f}x Chu")

    return total


def total_system_energy_estimate(total_macs, fire_rate, hidden):
    print(f"\n{'='*60}")
    print(f"TOTAL SYSTEM ENERGY ESTIMATE (front-end + classifier)")
    print(f"{'='*60}")

    processor_nJ_5pj = total_macs * 5.0 / 1000

    print(f"\n  1) CLASSIFIER (memristive crossbar)")
    print(f"     MACs: {total_macs:,.0f}")
    print(f"     Energy @ 5 pJ/MAC: {processor_nJ_5pj:.0f} nJ")

    print(f"\n  2) ANALOG FRONT-END")
    print(f"     Components per lead (x2 leads):")

    n_ota_filters = 5 * 2
    ota_current_nA = 100
    ota_vdd = 1.0
    ota_power_nW = ota_current_nA * ota_vdd
    ota_total_power_nW = n_ota_filters * ota_power_nW

    n_rectifiers = 1 * 2
    rect_current_nA = 50
    rect_power_nW = rect_current_nA * ota_vdd * n_rectifiers

    n_peak_det = 2 * 2
    peak_current_nA = 50
    peak_power_nW = peak_current_nA * ota_vdd * n_peak_det

    n_integrators = 3 * 2
    integ_current_nA = 100
    integ_power_nW = integ_current_nA * ota_vdd * n_integrators

    n_comparators = 3 * 2
    comp_current_nA = 50
    comp_power_nW = comp_current_nA * ota_vdd * n_comparators

    frontend_total_nW = ota_total_power_nW + rect_power_nW + peak_power_nW + integ_power_nW + comp_power_nW

    heartbeat_period_ms = 833
    frontend_nJ_per_beat = frontend_total_nW * heartbeat_period_ms / 1e6

    print(f"     a) OTA-C filters: {n_ota_filters} OTAs x {ota_current_nA} nA = {ota_total_power_nW:.0f} nW")
    print(f"     b) Rectifiers: {n_rectifiers} x {rect_current_nA} nA = {rect_power_nW:.0f} nW")
    print(f"     c) Peak detectors: {n_peak_det} x {peak_current_nA} nA = {peak_power_nW:.0f} nW")
    print(f"     d) Charge integrators: {n_integrators} x {integ_current_nA} nA = {integ_power_nW:.0f} nW")
    print(f"     e) Comparators/S&H: {n_comparators} x {comp_current_nA} nA = {comp_power_nW:.0f} nW")
    print(f"     Front-end total power: {frontend_total_nW:.0f} nW = {frontend_total_nW/1000:.2f} uW")
    print(f"     Front-end energy/beat ({heartbeat_period_ms}ms): {frontend_nJ_per_beat:.1f} nJ")

    print(f"\n  3) DIGITAL SYSTEM COMPARISON (what we eliminate)")
    print(f"     Typical 10-bit SAR ADC (180nm): 50-200 nW at 360 Hz")
    print(f"     Digital feature extraction: 100-500 nJ (FFT, wavelet, etc.)")
    print(f"     Total digital front-end: 150-700 nJ per beat")
    print(f"     Our analog front-end: {frontend_nJ_per_beat:.1f} nJ (always-on, no ADC)")

    total_system_nJ = processor_nJ_5pj + frontend_nJ_per_beat

    print(f"\n  4) TOTAL SYSTEM ENERGY")
    print(f"     Classifier:  {processor_nJ_5pj:.0f} nJ")
    print(f"     Front-end:   {frontend_nJ_per_beat:.1f} nJ")
    print(f"     TOTAL:       {total_system_nJ:.0f} nJ per classification")
    print(f"     System power: {total_system_nJ * 1.2 / 1000:.2f} uW @ 72 BPM")

    print(f"\n  5) COMPARISON TABLE (total system)")
    print(f"     {'System':<30} | {'Classifier':>12} | {'Front-end':>12} | {'Total':>10}")
    print(f"     {'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    print(f"     {'This work (analog, 5pJ/MAC)':<30} | {processor_nJ_5pj:>10.0f} nJ | {frontend_nJ_per_beat:>10.1f} nJ | {total_system_nJ:>8.0f} nJ")
    print(f"     {'This work (analog, 2pJ/MAC)':<30} | {total_macs*2/1000:>10.0f} nJ | {frontend_nJ_per_beat:>10.1f} nJ | {total_macs*2/1000+frontend_nJ_per_beat:>8.0f} nJ")
    print(f"     {'This work (analog, 1pJ/MAC)':<30} | {total_macs*1/1000:>10.0f} nJ | {frontend_nJ_per_beat:>10.1f} nJ | {total_macs*1/1000+frontend_nJ_per_beat:>8.0f} nJ")
    print(f"     {'Chu 2022 (40nm digital)':<30} | {'750':>10} nJ | {'~200':>10} nJ | {'~950':>8} nJ")
    print(f"     {'EKGNet 2023 (analog)':<30} | {'—':>10}    | {'—':>10}    | {'~9130':>8} nJ")


# ================================================================
# Main execution
# ================================================================

hidden = args.hidden
lam = args.lam
num_epochs = args.epochs
label = f"h{hidden}_s{lam}_{args.split}"

print(f"\n{'='*60}")
print(f"CONFIG: {label} (hidden={hidden}, lambda={lam}, split={args.split})")
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
processor_nJ = total_macs * 5.0 / 1000

print(f"\nRESULT: {label}")
print(f"  Split:       {args.split}-patient")
print(f"  Accuracy:    {acc:.2f}%")
print(f"  Firing rate: {fire_rate:.3f} ({fire_rate*100:.1f}%)")
print(f"  Parameters:  {n_params:,}")
print(f"  MACs:        {total_macs:,.0f}")
print(f"  Processor:   {processor_nJ:.0f} nJ @ 5 pJ/MAC (Chu: 750 nJ, ratio: {processor_nJ/750:.2f}x)")

print(classification_report(targets, preds, target_names=['N','S','V','F','Q']))

print(f"Confusion matrix:")
cm = confusion_matrix(targets, preds)
labels_cm = ['N', 'S', 'V', 'F', 'Q']
print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
for i, row_label in enumerate(labels_cm):
    row_str = ' '.join(f'{v:>6}' for v in cm[i])
    print(f"  {row_label:>5} {row_str}")

print(f"\nHardware sweep (4-bit + noise):")
for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
    mean_acc, std_acc, fr = combined_hw_evaluate(net, num_bits=4, sigma=sigma)
    print(f"  sigma={sigma:.3f}  {mean_acc:.2f}% +/- {std_acc:.2f}%")

energy_analysis(total_macs, fire_rate, hidden, num_features)
total_system_energy_estimate(total_macs, fire_rate, hidden)

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
