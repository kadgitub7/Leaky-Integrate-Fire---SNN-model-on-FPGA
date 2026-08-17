import snntorch as snn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import wfdb
import os
import copy
import math
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
import time

batch_size = 128
dtype = torch.float

# ================================================================
# ANALOG FEATURE EXTRACTION (same as snn_rMLP.py, vectorized)
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
class_sample_weights = 1.0 / (train_label_counts ** 0.5)
sample_weights = [class_sample_weights[l] for l in train_numeric_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

beta = 0.9
hidden_layer = 256
output_layer = num_class
num_steps = 25


# ================================================================
# MODEL 1: Recurrent SNN (analog crossbar)
# ================================================================

class RecurrentNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_features, hidden_layer)
        self.drop1 = torch.nn.Dropout(0.3)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=hidden_layer, learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(hidden_layer, output_layer)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        mem2_rec = []
        spk2_rec = []
        fc1_out = self.drop1(self.fc1(x))
        for step in range(num_steps):
            spk1, mem1 = self.rlif1(fc1_out, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2_rec.append(mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# ================================================================
# MODEL 2: Brain-Like Network (all-to-all + pruning)
# ================================================================

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem_minus_thr):
        spk = (mem_minus_thr > 0).float()
        ctx.save_for_backward(mem_minus_thr)
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        mem_minus_thr, = ctx.saved_tensors
        grad = 1.0 / (1.0 + (math.pi * mem_minus_thr).pow(2))
        return grad_output * grad

spike_fn = SurrogateSpike.apply


class BrainLikeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hid = hidden_layer
        self.n_out = output_layer
        n_spiking = self.n_hid + self.n_out
        self.fc_in = torch.nn.Linear(num_features, n_spiking)
        self.drop_in = torch.nn.Dropout(0.3)

        self.W = torch.nn.Parameter(torch.empty(n_spiking, n_spiking))
        torch.nn.init.normal_(self.W, 0, 1.0 / math.sqrt(n_spiking))
        self.mask_logits = torch.nn.Parameter(torch.full((n_spiking, n_spiking), 3.0))

        self.beta_h = torch.nn.Parameter(torch.tensor(0.9))
        self.beta_o = torch.nn.Parameter(torch.tensor(0.9))
        self.thr_h = torch.nn.Parameter(torch.tensor(1.0))
        self.thr_o = torch.nn.Parameter(torch.tensor(1.0))

    def get_mask(self):
        return torch.sigmoid(self.mask_logits)

    def forward(self, x):
        input_current = self.drop_in(self.fc_in(x))
        batch = x.size(0)
        dev = x.device

        mem_h = torch.zeros(batch, self.n_hid, device=dev)
        mem_o = torch.zeros(batch, self.n_out, device=dev)
        spk_h = torch.zeros(batch, self.n_hid, device=dev)
        spk_o = torch.zeros(batch, self.n_out, device=dev)

        mask = self.get_mask()
        W_eff = self.W * mask

        spk_o_rec = []
        mem_o_rec = []

        for step in range(num_steps):
            all_spk = torch.cat([spk_h, spk_o], dim=1)
            recurrent = torch.mm(all_spk, W_eff)
            cur_h = input_current[:, :self.n_hid] + recurrent[:, :self.n_hid]
            cur_o = input_current[:, self.n_hid:] + recurrent[:, self.n_hid:]

            mem_h = self.beta_h * mem_h + cur_h
            spk_h = spike_fn(mem_h - self.thr_h)
            mem_h = mem_h * (1 - spk_h.detach())

            mem_o = self.beta_o * mem_o + cur_o
            spk_o = spike_fn(mem_o - self.thr_o)
            mem_o = mem_o * (1 - spk_o.detach())

            spk_o_rec.append(spk_o)
            mem_o_rec.append(mem_o)

        return torch.stack(spk_o_rec, dim=0), torch.stack(mem_o_rec, dim=0)

    def connection_report(self):
        mask = self.get_mask().detach().cpu()
        alive = (mask > 0.01)
        regions = {
            'hidden->hidden': alive[:self.n_hid, :self.n_hid],
            'hidden->output': alive[:self.n_hid, self.n_hid:],
            'output->hidden': alive[self.n_hid:, :self.n_hid],
            'output->output': alive[self.n_hid:, self.n_hid:],
        }
        print("\n  Connection survival after pruning:")
        total_alive = 0
        total_possible = 0
        for name, region in regions.items():
            a = region.sum().item()
            t = region.numel()
            total_alive += a
            total_possible += t
            print(f"    {name:20s}: {int(a):>6d} / {t:>6d} alive ({a/t*100:5.1f}%)")
        print(f"    {'TOTAL':20s}: {int(total_alive):>6d} / {total_possible:>6d} alive ({total_alive/total_possible*100:5.1f}%)")


# ---- Training functions ----

def train_baseline(net, num_epochs=50):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in range(num_epochs):
        epoch_start = time.time()
        net.train()
        epoch_loss = 0.0
        batches = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, mem_rec = net(data)
            loss_val = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)
            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_val.item()
            batches += 1
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | loss: {epoch_loss/batches:.2f} | {time.time()-epoch_start:.1f}s")
    return net


def train_brain_like(net, num_epochs=50, l1_lambda=1e-5, prune_start=20, prune_every=5, prune_frac=0.2):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        net.train()
        epoch_loss = 0.0
        batches = 0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, mem_rec = net(data)
            loss_val = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)
            mask = net.get_mask()
            l1_penalty = l1_lambda * (net.W.abs() * mask).sum()
            total_loss = loss_val + l1_penalty
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_val.item()
            batches += 1
        scheduler.step()

        pruned_this_epoch = False
        if epoch >= prune_start and (epoch - prune_start) % prune_every == 0:
            with torch.no_grad():
                mask = net.get_mask()
                active = mask > 0.01
                active_vals = mask[active]
                if len(active_vals) > 0:
                    k = max(1, int(prune_frac * len(active_vals)))
                    cutoff = active_vals.flatten().kthvalue(k).values.item()
                    to_prune = (mask <= cutoff) & active
                    net.mask_logits.data[to_prune] = -10.0
            n_alive = (net.get_mask() > 0.01).sum().item()
            n_total = net.mask_logits.numel()
            pruned_this_epoch = True

        if (epoch + 1) % 5 == 0 or epoch == 0 or pruned_this_epoch:
            msg = f"  Epoch {epoch+1:3d}/{num_epochs} | loss: {epoch_loss/batches:.2f} | {time.time()-epoch_start:.1f}s"
            if pruned_this_epoch:
                msg += f" | PRUNED -> {int(n_alive)}/{n_total} ({n_alive/n_total*100:.0f}%)"
            print(msg)

    return net


# ---- Evaluation ----

def evaluate_model(net):
    total = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            test_spk, _ = net(data)
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    acc = correct / total * 100
    return acc, all_preds, all_targets


# ---- Analog hardware simulation helpers ----

def inject_noise_and_evaluate(net, sigma, num_trials=10):
    accs = []
    for _ in range(num_trials):
        noisy_net = copy.deepcopy(net)
        with torch.no_grad():
            for param in noisy_net.parameters():
                param.add_(torch.randn_like(param) * sigma)
        acc, _, _ = evaluate_model(noisy_net)
        accs.append(acc)
    return np.mean(accs), np.std(accs)


def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x_q


def quantize_and_evaluate(net, num_bits):
    q_net = copy.deepcopy(net)
    with torch.no_grad():
        for param in q_net.parameters():
            param.copy_(quantize_tensor(param, num_bits))
    acc, _, _ = evaluate_model(q_net)
    return acc


# ================================================================
# PHASE 1: Train Recurrent SNN baseline
# ================================================================
print("\n" + "=" * 60)
print("PHASE 1: Training Recurrent SNN (analog crossbar)")
print("=" * 60)

phase1_start = time.time()
baseline_net = RecurrentNet().to(device)
baseline_net = train_baseline(baseline_net, num_epochs=50)
baseline_acc, baseline_preds, baseline_targets = evaluate_model(baseline_net)
print(f"\nBaseline accuracy: {baseline_acc:.2f}% ({time.time()-phase1_start:.0f}s)")

from sklearn.metrics import classification_report
print(classification_report(baseline_targets, baseline_preds, target_names=['N','S','V','F','Q']))


# ================================================================
# PHASE 2: Train Brain-Like Network
# ================================================================
print("=" * 60)
print("PHASE 2: Training Brain-Like Network")
print("=" * 60)

phase2_start = time.time()
brain_net = BrainLikeNet().to(device)
n_total_conn = brain_net.mask_logits.numel()
print(f"  Total spiking connections: {n_total_conn}")

brain_net = train_brain_like(brain_net, num_epochs=50)
brain_acc, brain_preds, brain_targets = evaluate_model(brain_net)
print(f"\nBrain-Like accuracy: {brain_acc:.2f}% ({time.time()-phase2_start:.0f}s)")
print(classification_report(brain_targets, brain_preds, target_names=['N','S','V','F','Q']))
brain_net.connection_report()


# ================================================================
# PHASE 3: Noise injection sweep
# ================================================================
print("\n" + "=" * 60)
print("PHASE 3: Post-training noise injection sweep")
print("=" * 60)

noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

print("\n  --- Recurrent SNN ---")
baseline_noise_results = []
for sigma in noise_levels:
    mean_acc, std_acc = inject_noise_and_evaluate(baseline_net, sigma)
    baseline_noise_results.append((sigma, mean_acc, std_acc))
    print(f"  sigma={sigma:.3f}  accuracy={mean_acc:.2f}% +/- {std_acc:.2f}%")

print("\n  --- Brain-Like Network ---")
brain_noise_results = []
for sigma in noise_levels:
    mean_acc, std_acc = inject_noise_and_evaluate(brain_net, sigma)
    brain_noise_results.append((sigma, mean_acc, std_acc))
    print(f"  sigma={sigma:.3f}  accuracy={mean_acc:.2f}% +/- {std_acc:.2f}%")


# ================================================================
# PHASE 4: Quantization sweep
# ================================================================
print("\n" + "=" * 60)
print("PHASE 4: Post-training quantization sweep")
print("=" * 60)

bit_widths = [8, 6, 5, 4, 3, 2]

print("\n  --- Recurrent SNN ---")
baseline_quant_results = []
for bits in bit_widths:
    acc = quantize_and_evaluate(baseline_net, bits)
    baseline_quant_results.append((bits, acc))
    print(f"  {bits}-bit: {acc:.2f}%")

print("\n  --- Brain-Like Network ---")
brain_quant_results = []
for bits in bit_widths:
    acc = quantize_and_evaluate(brain_net, bits)
    brain_quant_results.append((bits, acc))
    print(f"  {bits}-bit: {acc:.2f}%")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY: Recurrent SNN vs Brain-Like (both fully analog)")
print("=" * 60)

print(f"\nClean accuracy:")
print(f"  Recurrent SNN:      {baseline_acc:.2f}%")
print(f"  Brain-Like Network: {brain_acc:.2f}%")

print(f"\nNoise robustness:")
print(f"  {'sigma':>8s}  {'Recurrent':>12s}  {'Brain-Like':>12s}")
for i, sigma in enumerate(noise_levels):
    b_acc = baseline_noise_results[i][1]
    r_acc = brain_noise_results[i][1]
    print(f"  {sigma:8.3f}  {b_acc:11.2f}%  {r_acc:11.2f}%")

print(f"\nQuantization robustness:")
print(f"  {'bits':>5s}  {'Recurrent':>12s}  {'Brain-Like':>12s}")
for i, bits in enumerate(bit_widths):
    b_acc = baseline_quant_results[i][1]
    r_acc = brain_quant_results[i][1]
    print(f"  {bits:5d}  {b_acc:11.2f}%  {r_acc:11.2f}%")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
