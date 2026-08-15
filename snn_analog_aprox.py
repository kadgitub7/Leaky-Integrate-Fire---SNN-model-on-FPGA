import snntorch as snn
import torch
from snntorch import spikegen
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import wfdb
import os
import copy
import math

batch_size = 128
dtype = torch.float

class MITBIHDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        num_beats = data.shape[0]
        flattened_data = data.reshape(num_beats, -1)
        self.data = torch.tensor(flattened_data, dtype=torch.float32)
        mins = self.data.min(dim=1, keepdim=True).values
        maxs = self.data.max(dim=1, keepdim=True).values
        self.data = (self.data - mins) / (maxs - mins + 1e-8)
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

train_record_names = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
]
test_record_names = [
    '100', '103', '105', '111', '113', '117', '121', '202', '210', '212',
    '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
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

print("Extracting Training Patients...")
train_beats, train_raw_labels = extract_beats_from_records(train_record_names)
print("Extracting Testing Patients...")
test_beats, test_raw_labels = extract_beats_from_records(test_record_names)

num_class = 5
train_numeric_labels = [aami_map[l] for l in train_raw_labels]
test_numeric_labels = [aami_map[l] for l in test_raw_labels]

train_dataset = MITBIHDataset(train_beats, train_numeric_labels)
test_dataset = MITBIHDataset(test_beats, test_numeric_labels)

train_label_counts = np.bincount(train_numeric_labels, minlength=num_class)
class_sample_weights = 1.0 / train_label_counts
sample_weights = [class_sample_weights[l] for l in train_numeric_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

num_leads = train_beats.shape[2]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

beta = 0.9
input_layer = 198 * num_leads
hidden_layer = 256
output_layer = num_class
num_steps = 50
gain = 0.7


# ================================================================
# MODEL 1: Baseline RecurrentNet (same as snn_rMLP.py)
# ================================================================

class RecurrentNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_layer, hidden_layer)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=hidden_layer, learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(hidden_layer, output_layer)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        mem2_rec = []
        spk2_rec = []
        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            mem2_rec.append(mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# ================================================================
# MODEL 2: Brain-Like Network (overproduction + pruning)
# ================================================================
# Architecture: every neuron can connect to every non-input neuron.
# This means we have connections:
#   input  -> hidden  (feedforward)
#   input  -> output  (skip connection)
#   hidden -> hidden  (recurrent / lateral)
#   hidden -> output  (feedforward)
#   output -> hidden  (feedback)
#   output -> output  (recurrent / lateral)
#
# Each connection has a learnable soft mask (sigmoid of a logit).
# During training, L1 penalty pressures unused connections toward 0.
# On a schedule, the weakest connections are hard-pruned (set to 0).
# This mimics: overproduction -> activity testing -> pruning -> optimization.

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
        self.n_in = input_layer
        self.n_hid = hidden_layer
        self.n_out = output_layer
        n_sources = self.n_in + self.n_hid + self.n_out   # 657: all neurons can send
        n_targets = self.n_hid + self.n_out                # 261: hidden + output receive

        # All-to-all weight matrix with fan-in scaled initialization
        self.W = torch.nn.Parameter(torch.empty(n_sources, n_targets))
        torch.nn.init.normal_(self.W, 0, 1.0 / math.sqrt(n_sources))

        # Soft mask logits: start at +3.0 so sigmoid ~ 0.95 (nearly all connections on)
        self.mask_logits = torch.nn.Parameter(torch.full((n_sources, n_targets), 3.0))

        # Separate learnable LIF parameters for hidden vs output populations
        self.beta_h = torch.nn.Parameter(torch.tensor(0.9))
        self.beta_o = torch.nn.Parameter(torch.tensor(0.9))
        self.thr_h = torch.nn.Parameter(torch.tensor(1.0))
        self.thr_o = torch.nn.Parameter(torch.tensor(1.0))

    def get_mask(self):
        return torch.sigmoid(self.mask_logits)

    def forward(self, x):
        batch = x.shape[1]
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
            # Concatenate ALL spike sources: input spikes + hidden spikes + output spikes
            all_spk = torch.cat([x[step], spk_h, spk_o], dim=1)

            # Single matmul: every source -> every target
            current = torch.mm(all_spk, W_eff)

            # Split current into hidden and output portions
            cur_h = current[:, :self.n_hid]
            cur_o = current[:, self.n_hid:]

            # LIF dynamics for hidden neurons
            mem_h = self.beta_h * mem_h + cur_h
            spk_h = spike_fn(mem_h - self.thr_h)
            mem_h = mem_h * (1 - spk_h.detach())

            # LIF dynamics for output neurons
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
            'input->hidden':  alive[:self.n_in, :self.n_hid],
            'input->output':  alive[:self.n_in, self.n_hid:],
            'hidden->hidden': alive[self.n_in:self.n_in+self.n_hid, :self.n_hid],
            'hidden->output': alive[self.n_in:self.n_in+self.n_hid, self.n_hid:],
            'output->hidden': alive[self.n_in+self.n_hid:, :self.n_hid],
            'output->output': alive[self.n_in+self.n_hid:, self.n_hid:],
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

def train_baseline(net, num_epochs=30):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    for epoch in range(num_epochs):
        net.train()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spike_data = spikegen.rate(data, num_steps=num_steps, gain=gain)
            spk_rec, mem_rec = net(spike_data)
            loss_val = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} complete")
    return net


def train_brain_like(net, num_epochs=40, l1_lambda=1e-5, prune_start=15, prune_every=5, prune_frac=0.2):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    for epoch in range(num_epochs):
        net.train()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spike_data = spikegen.rate(data, num_steps=num_steps, gain=gain)
            spk_rec, mem_rec = net(spike_data)

            # Classification loss
            loss_val = torch.zeros(1, dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)

            # L1 sparsity penalty on effective weights (encourages unused connections to die)
            mask = net.get_mask()
            l1_penalty = l1_lambda * (net.W.abs() * mask).sum()
            total_loss = loss_val + l1_penalty

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Pruning schedule: after prune_start epochs, prune every prune_every epochs
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
            print(f"  Epoch {epoch+1}: pruned -> {int(n_alive)}/{n_total} connections alive ({n_alive/n_total*100:.1f}%)")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} complete")

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
            test_spike_data = spikegen.rate(data, num_steps=num_steps, gain=gain)
            test_spk, _ = net(test_spike_data)
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
# PHASE 1: Train baseline RecurrentNet
# ================================================================
print("=" * 60)
print("PHASE 1: Training baseline RecurrentNet")
print("=" * 60)

baseline_net = RecurrentNet().to(device)
baseline_net = train_baseline(baseline_net, num_epochs=30)
baseline_acc, baseline_preds, baseline_targets = evaluate_model(baseline_net)
print(f"\nBaseline accuracy: {baseline_acc:.2f}%")

from sklearn.metrics import classification_report
print(classification_report(baseline_targets, baseline_preds, target_names=['N','S','V','F','Q']))


# ================================================================
# PHASE 2: Train Brain-Like Network (overproduction + pruning)
# ================================================================
print("=" * 60)
print("PHASE 2: Training Brain-Like Network")
print("  Starting with ALL possible connections (overproduction)")
print("  Pruning weakest 20% of active connections every 5 epochs after epoch 15")
print("=" * 60)

brain_net = BrainLikeNet().to(device)

n_total_conn = brain_net.mask_logits.numel()
print(f"  Total possible connections: {n_total_conn}")

brain_net = train_brain_like(brain_net, num_epochs=40, l1_lambda=1e-5, prune_start=15, prune_every=5, prune_frac=0.2)
brain_acc, brain_preds, brain_targets = evaluate_model(brain_net)
print(f"\nBrain-Like accuracy: {brain_acc:.2f}%")
print(classification_report(brain_targets, brain_preds, target_names=['N','S','V','F','Q']))

brain_net.connection_report()


# ================================================================
# PHASE 3: Noise injection sweep on BOTH models
# ================================================================
print("\n" + "=" * 60)
print("PHASE 3: Post-training noise injection sweep")
print("=" * 60)

noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

print("\n  --- Baseline RecurrentNet ---")
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
# PHASE 4: Quantization sweep on BOTH models
# ================================================================
print("\n" + "=" * 60)
print("PHASE 4: Post-training quantization sweep")
print("=" * 60)

bit_widths = [8, 6, 5, 4, 3, 2]

print("\n  --- Baseline RecurrentNet ---")
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
print("SUMMARY: Baseline vs Brain-Like")
print("=" * 60)

print(f"\nClean accuracy:")
print(f"  Baseline RecurrentNet: {baseline_acc:.2f}%")
print(f"  Brain-Like Network:    {brain_acc:.2f}%")

print(f"\nNoise robustness (accuracy at each sigma):")
print(f"  {'sigma':>8s}  {'Baseline':>12s}  {'Brain-Like':>12s}")
for i, sigma in enumerate(noise_levels):
    b_acc = baseline_noise_results[i][1]
    r_acc = brain_noise_results[i][1]
    print(f"  {sigma:8.3f}  {b_acc:11.2f}%  {r_acc:11.2f}%")

print(f"\nQuantization robustness:")
print(f"  {'bits':>5s}  {'Baseline':>12s}  {'Brain-Like':>12s}")
for i, bits in enumerate(bit_widths):
    b_acc = baseline_quant_results[i][1]
    r_acc = brain_quant_results[i][1]
    print(f"  {bits:5d}  {b_acc:11.2f}%  {r_acc:11.2f}%")
