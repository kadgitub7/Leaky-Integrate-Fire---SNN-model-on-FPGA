import sys
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

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--thresholds', type=int, default=2, choices=[1, 2, 3],
                    help='Number of delta levels: 1=simple, 2=dual, 3=fine')
parser.add_argument('--downsample', type=int, default=2,
                    help='Temporal downsample factor before delta encoding')
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
dtype = torch.float
num_class = 5

# ================================================================
# DELTA MODULATION ENCODER
# ================================================================
# Replaces the entire 14-filter OTA-C front-end with 3 simple
# channels per lead: raw ECG + 1st derivative + 2nd derivative,
# each converted to spike trains via delta modulation.
#
# Hardware: one comparator with hysteresis per channel per threshold.
# No OTAs, no continuous bias currents beyond the comparators.
# RC differentiators for derivatives are passive (zero power).

DELTA_VALUES = {
    1: [0.3],
    2: [0.2, 0.6],
    3: [0.15, 0.4, 0.8],
}

CHANNEL_NAMES = ['raw', '1st_deriv', '2nd_deriv']


def delta_modulate(signal, delta):
    n = len(signal)
    up = np.zeros(n, dtype=np.float32)
    down = np.zeros(n, dtype=np.float32)
    ref = signal[0]
    for i in range(1, n):
        if signal[i] - ref >= delta:
            up[i] = 1.0
            ref += delta
        elif ref - signal[i] >= delta:
            down[i] = 1.0
            ref -= delta
    return up, down


def encode_beat(beat, num_leads, thresholds, downsample=1):
    if downsample > 1:
        beat = beat[::downsample, :]
    num_steps = beat.shape[0]
    channels = []

    for lead in range(num_leads):
        raw = beat[:, lead]
        d1 = np.concatenate([[0], np.diff(raw)])
        d2 = np.concatenate([[0, 0], np.diff(raw, n=2)])

        for signal in [raw, d1, d2]:
            scale = np.max(np.abs(signal))
            if scale < 1e-8:
                normed = np.zeros_like(signal)
            else:
                normed = signal / scale

            for delta in thresholds:
                up, down = delta_modulate(normed, delta)
                channels.append(up)
                channels.append(down)

    return np.stack(channels, axis=1)


# ================================================================
# Data loading (identical to previous scripts)
# ================================================================

all_record_names = [
    '100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
    '114', '115', '116', '117', '118', '119', '121', '122', '124',
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
    '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
    '231', '232', '233', '234'
]

aami_map = {}
for sym in ['N', 'L', 'R', 'e', 'j']: aami_map[sym] = 0
for sym in ['A', 'a', 'J', 'S']: aami_map[sym] = 1
for sym in ['V', 'E']: aami_map[sym] = 2
for sym in ['F']: aami_map[sym] = 3
for sym in ['/', 'f', 'Q']: aami_map[sym] = 4

win_left = 90
win_right = 108

def extract_beats_from_records(rec_list):
    beats, labels, rr_feats = [], [], []
    for rec_id in rec_list:
        record_path = os.path.join('./mitdb_data', rec_id)
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signals = record.p_signal
            fs = record.fs

            valid = []
            for idx, sym in zip(annotation.sample, annotation.symbol):
                if idx - win_left >= 0 and idx + win_right < len(signals) and sym in aami_map:
                    valid.append((idx, sym))

            for i, (idx, sym) in enumerate(valid):
                beats.append(signals[idx - win_left : idx + win_right, :])
                labels.append(sym)

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                pre_pre_rr = (valid[i-1][0] - valid[i-2][0]) / fs if i > 1 else pre_rr

                local_rrs = []
                for j in range(max(1, i-5), i+1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833

                rr_feats.append([pre_rr, pre_pre_rr, pre_rr / (local_rr + 1e-8)])

        except Exception as e:
            print(f"Skipping {rec_id}: {e}")
    return np.array(beats), labels, np.array(rr_feats, dtype=np.float32)


t0 = time.time()
print("Loading beats...")
all_beats, all_raw_labels, all_rr = extract_beats_from_records(all_record_names)
print(f"  {len(all_beats)} beats ({time.time()-t0:.1f}s)")

num_leads = all_beats.shape[2]
beat_length = all_beats.shape[1]
all_numeric_labels = np.array([aami_map[l] for l in all_raw_labels])

thresholds = DELTA_VALUES[args.thresholds]
num_steps = beat_length // args.downsample
n_delta = len(thresholds) * 3 * num_leads * 2

print(f"\nDelta modulation encoding:")
print(f"  Beat length: {beat_length} samples -> {num_steps} steps (downsample={args.downsample})")
print(f"  Thresholds: {thresholds}")
print(f"  Delta channels: {len(thresholds)} levels x 3 types x {num_leads} leads x 2 dirs = {n_delta}")

t1 = time.time()
print("Encoding all beats...")
all_spikes = []
for i in range(len(all_beats)):
    spike_train = encode_beat(all_beats[i], num_leads, thresholds, args.downsample)
    all_spikes.append(spike_train)
all_spikes = np.array(all_spikes)
print(f"  Done ({time.time()-t1:.1f}s)")

# ================================================================
# R-R interval features (spike-burst encoded)
# ================================================================
# In hardware: R-R comes from QRS detection timer (already needed).
# Encoded as spike bursts at start of window — energy = N_spikes x SOPs.
# Zero additional front-end circuits required.

n_rr = 3
max_burst = 10

pre_rr_norm = np.clip((all_rr[:, 0] - 0.2) / 1.8, 0, 1)
pre_pre_rr_norm = np.clip((all_rr[:, 1] - 0.2) / 1.8, 0, 1)
rr_ratio_norm = np.clip((all_rr[:, 2] - 0.3) / 2.7, 0, 1)
rr_normed = np.stack([pre_rr_norm, pre_pre_rr_norm, rr_ratio_norm], axis=1)

rr_spikes = np.zeros((len(all_beats), num_steps, n_rr), dtype=np.float32)
for f in range(n_rr):
    n_spk = np.round(rr_normed[:, f] * max_burst).astype(int)
    for i in range(len(all_beats)):
        rr_spikes[i, :min(n_spk[i], num_steps), f] = 1.0

all_input = np.concatenate([all_spikes, rr_spikes], axis=2)
n_channels = all_input.shape[2]

print(f"  R-R channels: {n_rr} (pre_rr, pre_pre_rr, rr_ratio) burst-encoded")
print(f"  Total channels: {n_delta} delta + {n_rr} R-R = {n_channels}")
print(f"  Input shape: {all_input.shape}")

avg_delta_spk = all_spikes.sum(axis=(1,2)).mean()
avg_rr_spk = rr_spikes.sum(axis=(1,2)).mean()
avg_rate = all_input.mean()
print(f"  Avg delta spikes/beat: {avg_delta_spk:.1f}")
print(f"  Avg R-R spikes/beat: {avg_rr_spk:.1f}")
print(f"  Overall spike rate: {avg_rate:.3f} ({avg_rate*100:.1f}%)")

# ================================================================
# Train/test split
# ================================================================

train_idx, test_idx = train_test_split(
    np.arange(len(all_input)), test_size=0.2, random_state=42,
    stratify=all_numeric_labels
)

train_data = all_input[train_idx]
test_data = all_input[test_idx]
train_labels = all_numeric_labels[train_idx].tolist()
test_labels = all_numeric_labels[test_idx].tolist()

print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

class SpikeDataset(torch.utils.data.Dataset):
    def __init__(self, spikes, labels):
        self.data = torch.tensor(spikes, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

train_dataset = SpikeDataset(train_data, train_labels)
test_dataset = SpikeDataset(test_data, test_labels)

train_label_counts = np.bincount(train_labels, minlength=num_class)
class_sample_weights = 1.0 / (train_label_counts ** 0.65)
sample_weights = [class_sample_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** 0.65)
class_weights = class_weights / class_weights.sum() * num_class
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


# ================================================================
# Feedforward LIF SNN (no recurrence)
# ================================================================

class DeltaSNN(torch.nn.Module):
    def __init__(self, n_inputs, hidden, n_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_inputs, hidden)
        self.lif1 = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_rec, spk2_rec, mem2_rec = [], [], []
        T = x.shape[1]
        for step in range(T):
            cur1 = self.fc1(x[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_rec.append(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return (torch.stack(spk2_rec, dim=0),
                torch.stack(mem2_rec, dim=0),
                torch.stack(spk1_rec, dim=0))


# ================================================================
# Quantization
# ================================================================

def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


# ================================================================
# Training — full precision, loss on final membrane potential
# ================================================================
# Three fixes from failed v1:
#   1. No QAT during training (969-param model can't absorb 4-bit noise)
#   2. Loss on FINAL membrane potential only (not all 99 timesteps)
#   3. Membrane potential readout (not spike count sum)
# Quantization applied post-training at evaluation only.

def train_model(net, num_epochs, label=""):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 2e-3 * (epoch + 1) / warmup_epochs

        net.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_rate = 0.0
        batches = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            spk_out, mem_out, spk_hidden = net(data)

            loss = loss_fn(mem_out[-1], targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, pred = mem_out[-1].max(1)
            epoch_correct += (pred == targets).sum().item()
            epoch_total += targets.size(0)
            epoch_rate += spk_hidden.mean().item()
            batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / batches
        train_acc = epoch_correct / epoch_total * 100

        if train_acc > best_acc:
            best_acc = train_acc
            best_state = copy.deepcopy(net.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"loss: {avg_loss:.3f} | acc: {train_acc:.1f}% | "
                  f"fire: {epoch_rate/batches:.3f} | {time.time()-t0:.0f}s")

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"  [{label}] Restored best (train_acc={best_acc:.1f}%)")

    return net


# ================================================================
# Evaluation
# ================================================================

def evaluate(net, loader):
    total = correct = 0
    all_preds, all_targets = [], []
    total_input_spikes = 0
    total_hidden_spikes = 0
    total_possible_hidden = 0

    with torch.no_grad():
        net.eval()
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, mem_out, spk_hidden = net(data)
            _, pred = mem_out[-1].max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_input_spikes += data.sum().item()
            total_hidden_spikes += spk_hidden.sum().item()
            total_possible_hidden += spk_hidden.numel()

    acc = correct / total * 100
    avg_input_spikes = total_input_spikes / total
    avg_hidden_spikes = total_hidden_spikes / total
    hidden_rate = total_hidden_spikes / total_possible_hidden
    return acc, all_preds, all_targets, avg_input_spikes, avg_hidden_spikes, hidden_rate


def hw_evaluate(net, num_bits, sigma, num_trials=10):
    accs = []
    for _ in range(num_trials):
        hw_net = copy.deepcopy(net)
        with torch.no_grad():
            for p in hw_net.parameters():
                if p.dim() >= 2:
                    p.copy_(quantize_tensor(p, num_bits))
                    if sigma > 0:
                        p.add_(torch.randn_like(p) * sigma * p.abs().mean())
        acc, _, _, _, _, _ = evaluate(hw_net, test_loader)
        accs.append(acc)
    return np.mean(accs), np.std(accs)


# ================================================================
# Main
# ================================================================

hidden = args.hidden
num_epochs = args.epochs
label = f"delta_h{hidden}_t{args.thresholds}"

print(f"\n{'='*65}")
print(f"DELTA-SNN: {label}")
print(f"  Architecture: {n_channels} -> {hidden} LIF -> 5 LIF (feedforward)")
print(f"  Timesteps: {num_steps}")
print(f"  No recurrence, no filter bank")
print(f"{'='*65}")

net = DeltaSNN(n_channels, hidden, num_class).to(device)
n_params = sum(p.numel() for p in net.parameters())
print(f"  Parameters: {n_params:,}")
print(f"    fc1: {n_channels}x{hidden} = {n_channels*hidden:,} weights")
print(f"    fc2: {hidden}x5 = {hidden*5:,} weights")

net = train_model(net, num_epochs, label=label)

acc, preds, targets, avg_in_spk, avg_hid_spk, hid_rate = evaluate(net, test_loader)

print(f"\n{'='*65}")
print(f"RESULTS: {label}")
print(f"{'='*65}")
print(f"  Accuracy:          {acc:.2f}%")
print(f"  Parameters:        {n_params:,}")
print(f"  Hidden fire rate:  {hid_rate:.3f} ({hid_rate*100:.1f}%)")
print(f"  Avg input spikes:  {avg_in_spk:.1f} per beat")
print(f"  Avg hidden spikes: {avg_hid_spk:.1f} per beat")

print(classification_report(targets, preds, target_names=['N','S','V','F','Q']))

cm = confusion_matrix(targets, preds)
print(f"Confusion matrix:")
print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
    print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

print(f"\nHardware sweep (4-bit + noise):")
for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
    mean_acc, std_acc = hw_evaluate(net, num_bits=4, sigma=sigma)
    print(f"  sigma={sigma:.3f}  {mean_acc:.2f}% +/- {std_acc:.2f}%")


# ================================================================
# ENERGY ANALYSIS
# ================================================================

print(f"\n{'='*65}")
print(f"ENERGY ANALYSIS")
print(f"{'='*65}")

input_sops = avg_in_spk * hidden
hidden_sops = avg_hid_spk * num_class
total_sops = input_sops + hidden_sops

print(f"\n  CLASSIFIER (spike-driven synaptic operations)")
print(f"    Input spikes/beat:     {avg_in_spk:.1f}")
print(f"    -> fc1 SOPs:           {avg_in_spk:.1f} x {hidden} = {input_sops:,.0f}")
print(f"    Hidden spikes/beat:    {avg_hid_spk:.1f}")
print(f"    -> fc2 SOPs:           {avg_hid_spk:.1f} x {num_class} = {hidden_sops:,.0f}")
print(f"    Total SOPs/beat:       {total_sops:,.0f}")

print(f"\n  FRONT-END (delta modulators + R-R timer)")
n_comparators = len(thresholds) * 3 * num_leads
bias_nW_per_comp = 10
timer_nW = 5
frontend_nW = n_comparators * bias_nW_per_comp + timer_nW
beat_period_s = 0.833
frontend_nJ = frontend_nW * beat_period_s

print(f"    Delta comparators:     {n_comparators} (subthreshold, {bias_nW_per_comp} nW each)")
print(f"    RC differentiators:    {2 * num_leads} (passive, ~0 nW)")
print(f"    R-R timer:             1 ({timer_nW} nW)")
print(f"    Front-end power:       {frontend_nW} nW")
print(f"    Front-end energy/beat: {frontend_nJ:.1f} nJ")

print(f"\n  MULTI-POINT ENERGY SWEEP")
print(f"  {'pJ/SOP':>8} | {'Classifier':>12} | {'Front-end':>10} | {'Total':>10} | {'vs Chu':>8} | Note")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*30}")

sop_points = [
    (0.35, "180nm memristive (measured)"),
    (1.0,  "Good RRAM crossbar"),
    (2.0,  "Typical RRAM (90-130nm)"),
    (5.0,  "Conservative estimate"),
    (10.0, "With peripheral overhead"),
]

for pj, note in sop_points:
    cls_nJ = total_sops * pj / 1000
    tot_nJ = cls_nJ + frontend_nJ
    ratio = tot_nJ / 750
    print(f"  {pj:>7.2f}  | {cls_nJ:>10.1f} nJ | {frontend_nJ:>8.1f} nJ | {tot_nJ:>8.1f} nJ | {ratio:>7.3f}x | {note}")


print(f"\n  COMPARISON WITH PREVIOUS DESIGN AND LITERATURE")
print(f"  {'System':<40} | {'Processor':>10} | {'Front-end':>10} | {'Total':>10}")
print(f"  {'-'*40}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

cls_035 = total_sops * 0.35 / 1000
cls_5 = total_sops * 5.0 / 1000
print(f"  {'Delta-SNN (this, 0.35 pJ/SOP)':<40} | {cls_035:>8.1f} nJ | {frontend_nJ:>8.1f} nJ | {cls_035+frontend_nJ:>8.1f} nJ")
print(f"  {'Delta-SNN (this, 5 pJ/SOP)':<40} | {cls_5:>8.1f} nJ | {frontend_nJ:>8.1f} nJ | {cls_5+frontend_nJ:>8.1f} nJ")
print(f"  {'Previous design (OTA bank, 5 pJ/MAC)':<40} | {'262':>8} nJ | {'1833':>8} nJ | {'2095':>8} nJ")
print(f"  {'Chu 2022 (40nm digital SNN)':<40} | {'750':>8} nJ | {'~200':>8} nJ | {'~950':>8} nJ")
print(f"  {'SparrowSNN 2024 (22nm digital)':<40} | {'11.8':>8} nJ | {'~150':>8} nJ | {'~162':>8} nJ")
print(f"  {'EKGNet 2023 (fully analog)':<40} | {'—':>8}    | {'—':>8}    | {'~9130':>8} nJ")

print(f"\n  KEY METRICS")
print(f"    Parameters:       {n_params:,} (vs 39,929 previous = {n_params/39929:.1f}x)")
print(f"    Input channels:   {n_channels} (vs 412 previous)")
print(f"    Front-end power:  {frontend_nW} nW (vs ~2,200 nW previous = {frontend_nW/2200:.2f}x)")
print(f"    Front-end circuits: {n_comparators} comparators + {2*num_leads} RC")
print(f"                        (vs 28+ OTAs + rectifiers + integrators)")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
