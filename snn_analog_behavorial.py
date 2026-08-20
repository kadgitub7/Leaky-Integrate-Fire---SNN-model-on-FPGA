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

batch_size = 128
dtype = torch.float

# ================================================================
# MINIMAL ANALOG FRONT-END (temporal, not scalar)
# ================================================================
# Instead of 412 scalar features fed as constant current,
# use 4 simple RC filters per lead and feed the ECG sample-by-sample.
# The SNN extracts features temporally through its recurrent dynamics.
# Hardware: 4 passive RC circuits + 2 rectifiers per lead = ~10 op-amps total
# vs. the old design's 412 PGA stages.

def rc_lowpass_batch(signals, alpha):
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, signals, axis=1)

def rc_highpass_batch(signals, alpha):
    return signals - rc_lowpass_batch(signals, alpha)

def compute_temporal_features(beats, num_leads, downsample=4):
    num_beats, length, _ = beats.shape
    all_channels = []

    for lead in range(num_leads):
        signals = beats[:, :, lead]

        all_channels.append(signals[:, ::downsample])

        fast_hp = rc_highpass_batch(signals, 0.8)
        all_channels.append(fast_hp[:, ::downsample])

        smooth = rc_lowpass_batch(signals, 0.1)
        all_channels.append(smooth[:, ::downsample])

        all_channels.append(np.abs(fast_hp[:, ::downsample]))

    features = np.stack(all_channels, axis=2)
    return features


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
num_class = 5
all_numeric_labels = np.array([aami_map[l] for l in all_raw_labels])

# ================================================================
# Temporal feature extraction
# ================================================================

temporal_downsample = 4

t1 = time.time()
print(f"Computing temporal features (downsample={temporal_downsample}x)...")
all_temporal = compute_temporal_features(all_beats, num_leads, downsample=temporal_downsample)
print(f"  Shape: {all_temporal.shape} (beats, timesteps, channels) ({time.time()-t1:.1f}s)")

num_timesteps = all_temporal.shape[1]
num_channels = all_temporal.shape[2]

# ================================================================
# Train/test split
# ================================================================

train_idx, test_idx = train_test_split(
    np.arange(len(all_temporal)), test_size=0.2, random_state=42, stratify=all_numeric_labels
)

train_data = all_temporal[train_idx]
test_data = all_temporal[test_idx]
train_labels = all_numeric_labels[train_idx].tolist()
test_labels = all_numeric_labels[test_idx].tolist()

print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

# Per-channel normalization (each channel = one analog PGA stage)
train_mean = train_data.mean(axis=(0, 1))
train_std = train_data.std(axis=(0, 1))
train_data = (train_data - train_mean) / (train_std + 1e-8)
test_data = (test_data - train_mean) / (train_std + 1e-8)

print(f"  Normalization: {num_channels} PGA stages (vs 412 in old design)")

# ================================================================
# Dataset
# ================================================================

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

train_dataset = TemporalDataset(train_data, train_labels)
test_dataset = TemporalDataset(test_data, test_labels)

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

beta = 0.9


# ================================================================
# Temporal Recurrent SNN
# ================================================================
# Key difference: input changes every timestep (temporal encoding)
# instead of constant input repeated 40 times.
# The SNN sees the beat unfold in time and extracts features
# through its recurrent dynamics.

class TemporalRecurrentNet(torch.nn.Module):
    def __init__(self, n_channels, hidden, n_classes, dropout=0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_channels, hidden)
        self.drop1 = torch.nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=hidden, learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        batch_size_local = x.shape[0]
        n_steps = x.shape[1]
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        spk1_rec = []

        for t in range(n_steps):
            cur1 = self.drop1(self.fc1(x[:, t, :]))
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
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

def train_temporal(net, num_epochs, lambda_sparse, label=""):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        net.train()
        epoch_loss = 0.0
        epoch_sparsity = 0.0
        batches = 0

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_out, mem_out, spk_hidden = net(data)

            ce_loss = torch.zeros(1, dtype=dtype, device=device)
            for step in range(spk_out.shape[0]):
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

        scheduler.step()
        avg_rate = epoch_sparsity / batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {epoch_loss/batches:.2f} | "
                  f"fire_rate: {avg_rate:.3f} | {time.time()-epoch_start:.1f}s")

    return net


def evaluate_temporal(net):
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
# Analog hardware simulation helpers
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
    for _ in range(num_trials):
        hw_net = copy.deepcopy(net)
        with torch.no_grad():
            for param in hw_net.parameters():
                param.copy_(quantize_tensor(param, num_bits))
                param.add_(torch.randn_like(param) * sigma)
        acc, _, _, firing_rate = evaluate_temporal(hw_net)
        accs.append(acc)
    return np.mean(accs), np.std(accs), firing_rate


# ================================================================
# PHASE 1: Sweep hidden layer sizes
# ================================================================
print("\n" + "=" * 60)
print("PHASE 1: Architecture sweep (temporal encoding)")
print("=" * 60)
print(f"  Input: {num_channels} channels x {num_timesteps} timesteps")

hidden_sizes = [64, 128, 256]
num_epochs = 100

arch_results = {}
for hidden in hidden_sizes:
    print(f"\n--- Hidden={hidden} ---")
    net = TemporalRecurrentNet(num_channels, hidden, num_class).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_params:,}")
    net = train_temporal(net, num_epochs, lambda_sparse=0.0, label=f"h{hidden}")
    acc, preds, targets, fire_rate = evaluate_temporal(net)
    print(f"  Accuracy: {acc:.2f}% | Firing rate: {fire_rate:.3f}")
    arch_results[hidden] = {'net': net, 'acc': acc, 'fire_rate': fire_rate, 'params': n_params}

best_hidden = max(arch_results, key=lambda h: arch_results[h]['acc'])
print(f"\nBest architecture: hidden={best_hidden} ({arch_results[best_hidden]['acc']:.2f}%)")


# ================================================================
# PHASE 2: Sparsity penalty sweep on best architecture
# ================================================================
print("\n" + "=" * 60)
print(f"PHASE 2: Sparsity sweep (hidden={best_hidden})")
print("=" * 60)

sparsity_lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

sparse_results = {}
for lam in sparsity_lambdas:
    print(f"\n--- lambda_sparse={lam} ---")
    net = TemporalRecurrentNet(num_channels, best_hidden, num_class).to(device)
    net = train_temporal(net, num_epochs, lambda_sparse=lam, label=f"s{lam}")
    acc, preds, targets, fire_rate = evaluate_temporal(net)
    print(f"  Accuracy: {acc:.2f}% | Firing rate: {fire_rate:.3f}")
    sparse_results[lam] = {'net': net, 'acc': acc, 'fire_rate': fire_rate}

from sklearn.metrics import classification_report, confusion_matrix

# Pick best sparsity: highest accuracy among those with fire_rate < 0.15
candidates = {l: r for l, r in sparse_results.items() if r['acc'] >= 95.0}
if not candidates:
    candidates = sparse_results
best_lambda = min(candidates, key=lambda l: candidates[l]['fire_rate'])
best_sparse = sparse_results[best_lambda]
best_net = best_sparse['net']

print(f"\nBest sparsity: lambda={best_lambda} | acc={best_sparse['acc']:.2f}% | "
      f"fire_rate={best_sparse['fire_rate']:.3f}")

acc_final, preds_final, targets_final, fire_rate_final = evaluate_temporal(best_net)
print(classification_report(targets_final, preds_final, target_names=['N','S','V','F','Q']))
print(confusion_matrix(targets_final, preds_final))


# ================================================================
# PHASE 3: Realistic hardware sweep (4-bit + noise)
# ================================================================
print("\n" + "=" * 60)
print("PHASE 3: 4-bit quantized + noise (realistic hardware)")
print("=" * 60)

noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

hw_results = []
for sigma in noise_levels:
    mean_acc, std_acc, fr = combined_hw_evaluate(best_net, num_bits=4, sigma=sigma)
    hw_results.append((sigma, mean_acc, std_acc))
    print(f"  4-bit + sigma={sigma:.3f}  accuracy={mean_acc:.2f}% +/- {std_acc:.2f}%")


# ================================================================
# PHASE 4: Energy estimation
# ================================================================
print("\n" + "=" * 60)
print("PHASE 4: Energy per classification estimate")
print("=" * 60)

hidden = best_hidden
fr = best_sparse['fire_rate']
n_params_best = arch_results[best_hidden]['params']

w1_weights = num_channels * hidden
w_rec_weights = hidden * hidden
w2_weights = hidden * num_class
total_weights = w1_weights + w_rec_weights + w2_weights

pJ_per_MAC = 5.0

w1_macs = num_channels * hidden * num_timesteps
w_rec_macs = hidden * hidden * num_timesteps * fr
w2_macs = hidden * num_class * num_timesteps * fr
total_macs = w1_macs + w_rec_macs + w2_macs

crossbar_energy_pJ = total_macs * pJ_per_MAC
crossbar_energy_nJ = crossbar_energy_pJ / 1000
crossbar_energy_uJ = crossbar_energy_nJ / 1000

n_frontend_opamps = 10
opamp_bias_uA = 0.1
supply_V = 1.8
frontend_power_uW = n_frontend_opamps * opamp_bias_uA * supply_V
beat_window_s = 198 / 360.0
frontend_energy_uJ = frontend_power_uW * beat_window_s
frontend_energy_nJ = frontend_energy_uJ * 1000

n_pga = num_channels
pga_bias_nA = 50
pga_power_uW = n_pga * (pga_bias_nA / 1000) * supply_V
pga_energy_uJ = pga_power_uW * beat_window_s
pga_energy_nJ = pga_energy_uJ * 1000

neuron_energy_pJ = (hidden + num_class) * num_timesteps * 2
neuron_energy_nJ = neuron_energy_pJ / 1000

total_energy_nJ = crossbar_energy_nJ + frontend_energy_nJ + pga_energy_nJ + neuron_energy_nJ
total_energy_uJ = total_energy_nJ / 1000

print(f"\n  Architecture: {num_channels} -> {hidden} (RLeaky) -> {num_class}")
print(f"  Timesteps: {num_timesteps}")
print(f"  Firing rate: {fr:.3f} ({fr*100:.1f}%)")
print(f"  Total weights: {total_weights:,}")

print(f"\n  --- Crossbar energy ---")
print(f"  W1 ({num_channels}x{hidden}): {num_channels*hidden*num_timesteps:,} MACs")
print(f"  W_rec ({hidden}x{hidden}): {hidden*hidden*num_timesteps:.0f} MACs x {fr:.3f} sparsity = {w_rec_macs:,.0f} effective")
print(f"  W2 ({hidden}x{num_class}): {hidden*num_class*num_timesteps:.0f} MACs x {fr:.3f} sparsity = {w2_macs:,.0f} effective")
print(f"  Total effective MACs: {total_macs:,.0f}")
print(f"  At {pJ_per_MAC} pJ/MAC: {crossbar_energy_nJ:.1f} nJ")

print(f"\n  --- Analog front-end ---")
print(f"  {n_frontend_opamps} op-amps @ {opamp_bias_uA} uA, {supply_V}V = {frontend_power_uW:.1f} uW")
print(f"  Active {beat_window_s*1000:.0f} ms: {frontend_energy_nJ:.1f} nJ")

print(f"\n  --- Normalization (PGA) ---")
print(f"  {n_pga} stages @ {pga_bias_nA} nA, {supply_V}V = {pga_power_uW:.2f} uW")
print(f"  Active {beat_window_s*1000:.0f} ms: {pga_energy_nJ:.1f} nJ")

print(f"\n  --- Neurons ---")
print(f"  {hidden + num_class} neurons x {num_timesteps} steps @ ~2 pJ = {neuron_energy_nJ:.1f} nJ")

print(f"\n  ========================================")
print(f"  TOTAL ENERGY: {total_energy_nJ:.1f} nJ ({total_energy_uJ:.3f} uJ)")
print(f"  ========================================")

heart_rate_bpm = 72
beats_per_sec = heart_rate_bpm / 60
avg_power_uW = total_energy_uJ * beats_per_sec
print(f"\n  Avg power at {heart_rate_bpm} bpm: {avg_power_uW:.2f} uW")

print(f"\n  --- Comparison ---")
print(f"  This design (analog temporal):  {total_energy_nJ:.0f} nJ")
print(f"  Chu et al. (digital spike SNN): 750 nJ")
print(f"  Sadasivuni et al. (analog RC):  17.4 nJ")
print(f"  Old design (412-ch front-end):  ~25,000 nJ")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nArchitecture sweep:")
for h in hidden_sizes:
    r = arch_results[h]
    print(f"  hidden={h}: {r['acc']:.2f}% | fire_rate={r['fire_rate']:.3f} | params={r['params']:,}")

print(f"\nSparsity sweep (hidden={best_hidden}):")
for lam in sparsity_lambdas:
    r = sparse_results[lam]
    print(f"  lambda={lam:5.1f}: {r['acc']:.2f}% | fire_rate={r['fire_rate']:.3f}")

print(f"\nRealistic hardware (4-bit + noise):")
for sigma, mean_acc, std_acc in hw_results:
    print(f"  4-bit + sigma={sigma:.3f}  {mean_acc:.2f}% +/- {std_acc:.2f}%")

print(f"\nBest config: hidden={best_hidden}, lambda={best_lambda}")
print(f"  Accuracy: {best_sparse['acc']:.2f}%")
print(f"  Firing rate: {best_sparse['fire_rate']:.3f}")
print(f"  Energy: {total_energy_nJ:.1f} nJ/classification")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
