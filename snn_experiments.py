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
from sklearn.metrics import classification_report
import time

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='all',
                    choices=['ablation', 'timestep', 'spike', 'all'])
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
dtype = torch.float

# ================================================================
# Feature extraction with selectable feature types
# ================================================================

FEATURE_NAMES = {
    0: "Raw signal",
    1: "Fast HP (α=0.8)",
    2: "Slow HP (α=0.15)",
    3: "Smooth fast LP (α=0.5)",
    4: "Smooth slow LP (α=0.08)",
    5: "Bandpass",
    6: "Rectified fast HP",
    7: "Second derivative",
    8: "QRS area",
    9: "Pre-QRS area",
    10: "Post-QRS area",
    11: "QRS max",
    12: "QRS min",
    13: "QRS range",
}

def rc_lowpass_batch(signals, alpha):
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, signals, axis=1)

def rc_highpass_batch(signals, alpha):
    return signals - rc_lowpass_batch(signals, alpha)

def compute_analog_features(beats, num_leads, exclude_types=None):
    if exclude_types is None:
        exclude_types = set()
    num_beats, length, _ = beats.shape
    downsample = 8
    all_features = []

    for lead in range(num_leads):
        signals = beats[:, :, lead]

        if 0 not in exclude_types:
            all_features.append(signals[:, ::downsample])

        fast_hp = rc_highpass_batch(signals, 0.8)
        if 1 not in exclude_types:
            all_features.append(fast_hp[:, ::downsample])

        if 2 not in exclude_types:
            slow_hp = rc_highpass_batch(signals, 0.15)
            all_features.append(slow_hp[:, ::downsample])

        smooth_fast = rc_lowpass_batch(signals, 0.5)
        if 3 not in exclude_types:
            all_features.append(smooth_fast[:, ::downsample])

        smooth_slow = rc_lowpass_batch(signals, 0.08)
        if 4 not in exclude_types:
            all_features.append(smooth_slow[:, ::downsample])

        if 5 not in exclude_types:
            if 3 in exclude_types:
                smooth_fast = rc_lowpass_batch(signals, 0.5)
            if 4 in exclude_types:
                smooth_slow = rc_lowpass_batch(signals, 0.08)
            bandpass = smooth_fast - smooth_slow
            all_features.append(bandpass[:, ::downsample])

        if 6 not in exclude_types:
            all_features.append(np.abs(fast_hp[:, ::downsample]))

        if 7 not in exclude_types:
            second_deriv = rc_highpass_batch(rc_highpass_batch(signals, 0.7), 0.7)
            all_features.append(second_deriv[:, ::downsample])

        abs_signals = np.abs(signals)
        if 8 not in exclude_types:
            all_features.append(np.sum(abs_signals[:, 80:120], axis=1, keepdims=True))
        if 9 not in exclude_types:
            all_features.append(np.sum(abs_signals[:, 40:80], axis=1, keepdims=True))
        if 10 not in exclude_types:
            all_features.append(np.sum(abs_signals[:, 120:170], axis=1, keepdims=True))

        qrs_max = np.max(signals[:, 80:120], axis=1, keepdims=True)
        qrs_min = np.min(signals[:, 80:120], axis=1, keepdims=True)
        if 11 not in exclude_types:
            all_features.append(qrs_max)
        if 12 not in exclude_types:
            all_features.append(qrs_min)
        if 13 not in exclude_types:
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
print("Loading data...")
all_beats, all_raw_labels = extract_beats_from_records(all_record_names)
print(f"  {len(all_beats)} beats ({time.time()-t0:.1f}s)")
num_leads = all_beats.shape[2]
num_class = 5
all_numeric_labels = np.array([aami_map[l] for l in all_raw_labels])

train_idx, test_idx = train_test_split(
    np.arange(len(all_beats)), test_size=0.2, random_state=42,
    stratify=all_numeric_labels
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")


# ================================================================
# Helper: prepare data loaders for a given feature set
# ================================================================

def prepare_data(exclude_types=None):
    features = compute_analog_features(all_beats, num_leads, exclude_types=exclude_types)
    train_features = features[train_idx]
    test_features = features[test_idx]
    train_labels = all_numeric_labels[train_idx].tolist()
    test_labels = all_numeric_labels[test_idx].tolist()

    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0)
    train_features = (train_features - train_mean) / (train_std + 1e-8)
    test_features = (test_features - train_mean) / (train_std + 1e-8)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    train_ds = DS(train_features, train_labels)
    test_ds = DS(test_features, test_labels)

    train_counts = np.bincount(train_labels, minlength=num_class)
    csw = 1.0 / (train_counts ** 0.65)
    sw = [csw[l] for l in train_labels]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    cw = 1.0 / (np.array(train_counts, dtype=np.float64) ** 0.65)
    cw = cw / cw.sum() * num_class
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    return train_loader, test_loader, cw_tensor, features.shape[1]


# ================================================================
# Standard RecurrentNet
# ================================================================

def make_net(n_features, hidden, n_classes, num_steps, beta=0.9, dropout=0.1):
    class RecurrentNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(n_features, hidden)
            self.drop1 = torch.nn.Dropout(dropout)
            self.rlif1 = snn.RLeaky(beta=beta, linear_features=hidden,
                                     learn_beta=True, learn_threshold=True)
            self.fc2 = torch.nn.Linear(hidden, n_classes)
            self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
            self.num_steps = num_steps

        def forward(self, x):
            spk1, mem1 = self.rlif1.init_rleaky()
            mem2 = self.lif2.init_leaky()
            spk2_rec, mem2_rec, spk1_rec = [], [], []
            fc1_out = self.drop1(self.fc1(x))
            for step in range(self.num_steps):
                spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
                spk1_rec.append(spk1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)
            return (torch.stack(spk2_rec, dim=0),
                    torch.stack(mem2_rec, dim=0),
                    torch.stack(spk1_rec, dim=0))
    return RecurrentNet()


# ================================================================
# Spike-encoded input network
# ================================================================

def make_spike_net(n_features, hidden, n_classes, num_steps, beta=0.9, dropout=0.1):
    class SpikeEncodedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = snn.Leaky(beta=0.85, learn_beta=True, learn_threshold=True)
            self.fc1 = torch.nn.Linear(n_features, hidden)
            self.drop1 = torch.nn.Dropout(dropout)
            self.rlif1 = snn.RLeaky(beta=beta, linear_features=hidden,
                                     learn_beta=True, learn_threshold=True)
            self.fc2 = torch.nn.Linear(hidden, n_classes)
            self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
            self.num_steps = num_steps

        def forward(self, x):
            mem_enc = self.encoder.init_leaky()
            spk1, mem1 = self.rlif1.init_rleaky()
            mem2 = self.lif2.init_leaky()
            spk2_rec, mem2_rec, spk1_rec, enc_spk_rec = [], [], [], []

            for step in range(self.num_steps):
                spk_enc, mem_enc = self.encoder(x, mem_enc)
                enc_spk_rec.append(spk_enc)
                fc1_out = self.drop1(self.fc1(spk_enc))
                spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
                spk1_rec.append(spk1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)

            return (torch.stack(spk2_rec, dim=0),
                    torch.stack(mem2_rec, dim=0),
                    torch.stack(spk1_rec, dim=0),
                    torch.stack(enc_spk_rec, dim=0))
    return SpikeEncodedNet()


# ================================================================
# Quantize helper
# ================================================================

def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


# ================================================================
# Training: simplified (no QAT) for fast ablation
# ================================================================

def train_simple(net, train_loader, test_loader, cw_tensor, num_epochs,
                 lambda_sparse, label="", is_spike_net=False):
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup)

    for epoch in range(num_epochs):
        if epoch < warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup

        net.train()
        epoch_loss = 0.0
        epoch_rate = 0.0
        batches = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = net(data)
            if is_spike_net:
                spk_out, mem_out, spk_hidden, _ = outputs
            else:
                spk_out, mem_out, spk_hidden = outputs

            ce = torch.zeros(1, dtype=dtype, device=device)
            ns = net.num_steps
            for s in range(ns):
                ce += loss_fn(mem_out[s], targets)

            fr = spk_hidden.mean()
            sp_loss = lambda_sparse * torch.clamp(fr - 0.15, min=0.0)
            loss = ce + sp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce.item()
            epoch_rate += fr.item()
            batches += 1

        if epoch >= warmup:
            scheduler.step()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"CE: {epoch_loss/batches:.2f} | fire: {epoch_rate/batches:.3f}")

    return net


# ================================================================
# Training: with QAT for hardware-realistic experiments
# ================================================================

def train_qat(net, train_loader, test_loader, cw_tensor, num_epochs,
              lambda_sparse, label="", is_spike_net=False):
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup)
    best_ce = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        if epoch < warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup

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

            outputs = net(data)
            if is_spike_net:
                spk_out, mem_out, spk_hidden, _ = outputs
            else:
                spk_out, mem_out, spk_hidden = outputs

            ce = torch.zeros(1, dtype=dtype, device=device)
            ns = net.num_steps
            for s in range(ns):
                ce += loss_fn(mem_out[s], targets)

            fr = spk_hidden.mean()
            sp_loss = lambda_sparse * torch.clamp(fr - 0.15, min=0.0)
            loss = ce + sp_loss

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for p, s in zip(net.parameters(), saved):
                    p.data.copy_(s)

            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += ce.item()
            epoch_rate += fr.item()
            batches += 1

        if epoch >= warmup:
            scheduler.step()

        avg_ce = epoch_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"CE: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f}")

    if best_state is not None:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))
        print(f"  [{label}] Restored best (CE={best_ce:.3f}), quantized to 4-bit")

    return net


# ================================================================
# Evaluation
# ================================================================

def evaluate(net, test_loader, is_spike_net=False):
    total = correct = 0
    total_spikes = total_possible = 0
    input_spikes = input_possible = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = net(data)
            if is_spike_net:
                spk_out, _, spk_hidden, spk_input = outputs
                input_spikes += spk_input.sum().item()
                input_possible += spk_input.numel()
            else:
                spk_out, _, spk_hidden = outputs

            _, pred = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_spikes += spk_hidden.sum().item()
            total_possible += spk_hidden.numel()

    acc = correct / total * 100
    hidden_fr = total_spikes / total_possible
    input_fr = input_spikes / input_possible if input_possible > 0 else 0
    return acc, all_preds, all_targets, hidden_fr, input_fr


# ================================================================
# EXPERIMENT 1: Feature ablation (leave-one-type-out)
# ================================================================

def run_ablation():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: FEATURE ABLATION (leave-one-type-out)")
    print("  Training: 50 epochs, no QAT (relative comparison)")
    print("  Hidden: 80, Lambda: 1.0, Steps: 40")
    print("=" * 70)

    hidden = 80
    num_steps_val = 40
    num_epochs = 50
    lam = 1.0

    print("\n--- Baseline (all 14 feature types) ---")
    train_loader, test_loader, cw, nf = prepare_data(exclude_types=None)
    net = make_net(nf, hidden, num_class, num_steps_val).to(device)
    net = train_simple(net, train_loader, test_loader, cw, num_epochs, lam, "baseline")
    base_acc, _, _, base_fr, _ = evaluate(net, test_loader)
    print(f"  Baseline: {base_acc:.2f}% accuracy, {nf} features, fire_rate={base_fr:.3f}")

    results = []

    for ftype in range(14):
        name = FEATURE_NAMES[ftype]
        print(f"\n--- Without: {ftype} ({name}) ---")
        train_loader, test_loader, cw, nf = prepare_data(exclude_types={ftype})
        net = make_net(nf, hidden, num_class, num_steps_val).to(device)
        net = train_simple(net, train_loader, test_loader, cw, num_epochs, lam, f"no_{ftype}")
        acc, _, _, fr, _ = evaluate(net, test_loader)
        drop = base_acc - acc
        results.append((ftype, name, nf, acc, drop, fr))
        print(f"  Result: {acc:.2f}% ({drop:+.2f}% vs baseline), {nf} features")

    print(f"\n--- Group ablations ---")

    print(f"\n--- Without ALL scalar features (8-13) ---")
    train_loader, test_loader, cw, nf = prepare_data(exclude_types={8,9,10,11,12,13})
    net = make_net(nf, hidden, num_class, num_steps_val).to(device)
    net = train_simple(net, train_loader, test_loader, cw, num_epochs, lam, "no_scalars")
    acc, _, _, fr, _ = evaluate(net, test_loader)
    print(f"  Result: {acc:.2f}% ({base_acc-acc:+.2f}% vs baseline), {nf} features")

    print(f"\n--- Without ALL temporal waveforms (1-7), keep raw + scalars ---")
    train_loader, test_loader, cw, nf = prepare_data(exclude_types={1,2,3,4,5,6,7})
    net = make_net(nf, hidden, num_class, num_steps_val).to(device)
    net = train_simple(net, train_loader, test_loader, cw, num_epochs, lam, "raw+scalars")
    acc, _, _, fr, _ = evaluate(net, test_loader)
    print(f"  Result: {acc:.2f}% ({base_acc-acc:+.2f}% vs baseline), {nf} features")

    print(f"\n--- Only raw signal + fast HP + bandpass + scalars (minimal set) ---")
    train_loader, test_loader, cw, nf = prepare_data(exclude_types={2,3,4,6,7})
    net = make_net(nf, hidden, num_class, num_steps_val).to(device)
    net = train_simple(net, train_loader, test_loader, cw, num_epochs, lam, "minimal")
    acc, _, _, fr, _ = evaluate(net, test_loader)
    print(f"  Result: {acc:.2f}% ({base_acc-acc:+.2f}% vs baseline), {nf} features")

    print(f"\n--- Only raw signal + fast HP + second deriv + scalars ---")
    train_loader, test_loader, cw, nf = prepare_data(exclude_types={2,3,4,5,6})
    net = make_net(nf, hidden, num_class, num_steps_val).to(device)
    net = train_simple(net, train_loader, test_loader, cw, num_epochs, lam, "alt_minimal")
    acc, _, _, fr, _ = evaluate(net, test_loader)
    print(f"  Result: {acc:.2f}% ({base_acc-acc:+.2f}% vs baseline), {nf} features")

    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY (sorted by accuracy drop)")
    print(f"{'='*70}")
    print(f"  Baseline: {base_acc:.2f}% with {412} features")
    print(f"  {'Type':<5} {'Name':<25} {'Features':>8} {'Acc':>8} {'Drop':>8} {'Verdict'}")
    print(f"  {'-'*5} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")

    results.sort(key=lambda x: x[4], reverse=True)
    for ftype, name, nf, acc, drop, fr in results:
        verdict = "KEEP" if drop > 0.5 else "REMOVE OK" if drop < -0.1 else "MARGINAL"
        print(f"  {ftype:<5} {name:<25} {nf:>8} {acc:>7.2f}% {drop:>+7.2f}% {verdict}")


# ================================================================
# EXPERIMENT 2: Timestep sweep
# ================================================================

def run_timestep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TIMESTEP SWEEP")
    print("  Training: 100 epochs with QAT")
    print("  Hidden: 80, Lambda: 1.0")
    print("=" * 70)

    hidden = 80
    lam = 1.0
    num_epochs = 100

    train_loader, test_loader, cw, nf = prepare_data()

    timestep_results = []

    for steps in [10, 20, 30, 40]:
        print(f"\n--- num_steps = {steps} ---")
        net = make_net(nf, hidden, num_class, steps).to(device)
        net = train_qat(net, train_loader, test_loader, cw, num_epochs, lam, f"steps_{steps}")
        acc, _, targets, fr, _ = evaluate(net, test_loader)

        w1 = nf * hidden
        w_rec = hidden * hidden * steps * fr
        w2 = hidden * num_class * steps * fr
        total_macs = w1 + w_rec + w2
        nJ_5 = total_macs * 5.0 / 1000
        nJ_1 = total_macs * 1.0 / 1000

        timestep_results.append((steps, acc, fr, total_macs, nJ_5, nJ_1))
        print(f"  Result: {acc:.2f}%, fire_rate={fr:.3f}, MACs={total_macs:,.0f}, "
              f"Energy={nJ_5:.0f} nJ @5pJ, {nJ_1:.0f} nJ @1pJ")

    print(f"\n\n{'='*70}")
    print("TIMESTEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Steps':>5} | {'Acc':>8} | {'Fire Rate':>9} | {'MACs':>10} | {'@5pJ/MAC':>10} | {'@1pJ/MAC':>10} | vs Chu")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for steps, acc, fr, macs, nj5, nj1 in timestep_results:
        print(f"  {steps:>5} | {acc:>7.2f}% | {fr:>8.3f} | {macs:>10,.0f} | {nj5:>8.0f} nJ | {nj1:>8.0f} nJ | {nj5/750:.2f}x")


# ================================================================
# EXPERIMENT 3: Spike-encoded input
# ================================================================

def run_spike():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SPIKE-ENCODED INPUT")
    print("  All inputs encoded as spikes via LIF encoder layer")
    print("  W1 becomes spike-driven (no input DACs needed)")
    print("  Training: 150 epochs with QAT")
    print("=" * 70)

    hidden = 80
    lam = 1.0
    num_epochs = 150
    num_steps_val = 40

    train_loader, test_loader, cw, nf = prepare_data()

    print(f"\n--- Standard model (baseline) ---")
    net_std = make_net(nf, hidden, num_class, num_steps_val).to(device)
    net_std = train_qat(net_std, train_loader, test_loader, cw, num_epochs, lam, "standard")
    acc_std, _, _, fr_std, _ = evaluate(net_std, test_loader)

    w1_std = nf * hidden
    w_rec_std = hidden * hidden * num_steps_val * fr_std
    w2_std = hidden * num_class * num_steps_val * fr_std
    macs_std = w1_std + w_rec_std + w2_std

    print(f"  Standard: {acc_std:.2f}%, fire_rate={fr_std:.3f}, MACs={macs_std:,.0f}")

    print(f"\n--- Spike-encoded input model ---")
    net_spk = make_spike_net(nf, hidden, num_class, num_steps_val).to(device)
    net_spk = train_qat(net_spk, train_loader, test_loader, cw, num_epochs, lam,
                        "spike_enc", is_spike_net=True)
    acc_spk, _, _, fr_spk, input_fr = evaluate(net_spk, test_loader, is_spike_net=True)

    w1_spk = nf * hidden * num_steps_val * input_fr
    w_rec_spk = hidden * hidden * num_steps_val * fr_spk
    w2_spk = hidden * num_class * num_steps_val * fr_spk
    ops_spk = w1_spk + w_rec_spk + w2_spk

    print(f"  Spike-enc: {acc_spk:.2f}%, hidden_fr={fr_spk:.3f}, input_fr={input_fr:.3f}")
    print(f"  Spike ops: {ops_spk:,.0f} (accumulates, not full MACs)")

    print(f"\n\n{'='*70}")
    print("SPIKE-ENCODED INPUT SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} | {'Standard':>15} | {'Spike-encoded':>15}")
    print(f"  {'-'*30}-+-{'-'*15}-+-{'-'*15}")
    print(f"  {'Accuracy':<30} | {acc_std:>14.2f}% | {acc_spk:>14.2f}%")
    print(f"  {'Hidden firing rate':<30} | {fr_std:>14.3f} | {fr_spk:>14.3f}")
    print(f"  {'Input firing rate':<30} | {'N/A (continuous)':>15} | {input_fr:>14.3f}")
    print(f"  {'W1 operations':<30} | {w1_std:>12,.0f} MAC | {w1_spk:>10,.0f} acc")
    print(f"  {'W_rec operations':<30} | {w_rec_std:>12,.0f} MAC | {w_rec_spk:>10,.0f} acc")
    print(f"  {'W2 operations':<30} | {w2_std:>12,.0f} MAC | {w2_spk:>10,.0f} acc")
    print(f"  {'Total operations':<30} | {macs_std:>12,.0f} MAC | {ops_spk:>10,.0f} acc")
    print(f"  {'Energy @5pJ/MAC, 1pJ/acc':<30} | {macs_std*5/1000:>10.0f} nJ   | {ops_spk*1/1000:>10.0f} nJ")
    print(f"  {'Energy @2pJ/MAC, 0.5pJ/acc':<30} | {macs_std*2/1000:>10.0f} nJ   | {ops_spk*0.5/1000:>10.0f} nJ")
    print(f"  {'Input DACs required':<30} | {'412 (multi-bit)':>15} | {'0 (binary)':>15}")
    print(f"  {'Hardware complexity':<30} | {'Higher':>15} | {'Lower':>15}")


# ================================================================
# Main
# ================================================================

if args.experiment in ('ablation', 'all'):
    run_ablation()

if args.experiment in ('timestep', 'all'):
    run_timestep()

if args.experiment in ('spike', 'all'):
    run_spike()

total_time = time.time() - t0
print(f"\nTotal experiment time: {total_time:.0f}s ({total_time/60:.1f}min)")
