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

# ================================================================
# CONFIGURATION
# ================================================================

parser = argparse.ArgumentParser(
    description='Minimal analog front-end SNN for ECG arrhythmia classification')
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lam', type=float, default=1.0, help='Sparsity penalty weight')
parser.add_argument('--split', choices=['intra', 'inter'], default='intra')
args = parser.parse_args()

torch.set_num_threads(2)
batch_size = 128
dtype = torch.float
num_class = 5
t0 = time.time()


# ================================================================
# ANALOG FRONT-END: 7 circuits → 11 features
# ================================================================
#
# Each feature maps to a specific analog circuit:
#
# ┌──────────────────┬───────────────────────┬────────┬─────────────────────────┐
# │ Feature          │ Analog circuit        │ Power  │ What it captures        │
# ├──────────────────┼───────────────────────┼────────┼─────────────────────────┤
# │ pre_rr           │ Timer (counter)       │ ~5 nW  │ Interval to prev QRS    │
# │ post_rr          │ Timer (counter)       │  same  │ Interval to next QRS    │
# │ rr_ratio         │ Capacitor ratio       │ ~10 nW │ pre_rr / local avg      │
# │ peak_amp × 2     │ Peak-and-hold × 2     │ ~20 nW │ Max |V| in QRS          │
# │ qrs_width × 2    │ Comparator+timer × 2  │ ~20 nW │ QRS duration            │
# │ qrs_area × 2     │ Gated integrator × 2  │ ~30 nW │ ∫|V| during QRS         │
# │ polarity × 2     │ Sign comparator × 2   │ ~10 nW │ QRS deflection sign     │
# │ rel_peak × 2     │ Divider + EWMA RC × 2 │ ~15 nW │ peak / running_baseline │
# │ rel_width × 2    │ Divider + EWMA RC × 2 │ ~15 nW │ width / running_baseline│
# │ rel_area × 2     │ Divider + EWMA RC × 2 │ ~15 nW │ area / running_baseline │
# │ templ_corr × 2   │ Mini crossbar (1×6)×2 │ ~20 nW │ Cosine sim to template  │
# ├──────────────────┼───────────────────────┼────────┼─────────────────────────┤
# │ Total            │ ~15 circuits          │~160 nW │ 19 features             │
# └──────────────────┴───────────────────────┴────────┴─────────────────────────┘
#
# Patient-adaptive: EWMA (RC low-pass) tracks patient baseline over ~50 beats.
# Relative features = current/baseline → patient-independent, transfers across patients.
# Template correlation = cosine similarity with running QRS shape → catches fusion beats.
#
# Previous front-end: 28+ OTAs + rectifiers + integrators → 412 features, ~2200 nW
# This front-end:     ~15 simple circuits → 19 features, ~160 nW (14× less power)

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'rr_ratio',
    'peak_amp_L0', 'peak_amp_L1',
    'qrs_width_L0', 'qrs_width_L1',
    'qrs_area_L0', 'qrs_area_L1',
    'polarity_L0', 'polarity_L1',
    'rel_peak_L0', 'rel_peak_L1',
    'rel_width_L0', 'rel_width_L1',
    'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
]

EWMA_ALPHA = 0.02
N_TEMPLATE = 6

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


# ================================================================
# DATA LOADING + FEATURE EXTRACTION
# ================================================================

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

            ewma_peak = [None] * num_leads
            ewma_width = [None] * num_leads
            ewma_area = [None] * num_leads
            ewma_template = [None] * num_leads

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - win_left : idx + win_right, :]
                all_labels.append(aami_map[sym])

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - 5), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833
                rr_ratio = pre_rr / (local_rr + 1e-8)

                feat = np.zeros(19, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = rr_ratio

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

                    polarity = 0.0
                    if peak > 1e-6:
                        max_idx = np.argmax(abs_qrs)
                        polarity = 1.0 if qrs[max_idx] >= 0 else -1.0

                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]

                    feat[3 + lead] = peak
                    feat[5 + lead] = width
                    feat[7 + lead] = area
                    feat[9 + lead] = polarity

                    if ewma_peak[lead] is None:
                        ewma_peak[lead] = max(peak, 1e-6)
                        ewma_width[lead] = max(width, 1e-6)
                        ewma_area[lead] = max(area, 1e-6)
                        ewma_template[lead] = qrs_ds.copy()

                    feat[11 + lead] = peak / (ewma_peak[lead] + 1e-8)
                    feat[13 + lead] = width / (ewma_width[lead] + 1e-8)
                    feat[15 + lead] = area / (ewma_area[lead] + 1e-8)

                    norm_curr = np.linalg.norm(qrs_ds)
                    norm_tmpl = np.linalg.norm(ewma_template[lead])
                    if norm_curr > 1e-8 and norm_tmpl > 1e-8:
                        feat[17 + lead] = np.dot(qrs_ds, ewma_template[lead]) / (norm_curr * norm_tmpl)
                    else:
                        feat[17 + lead] = 1.0

                    a = EWMA_ALPHA
                    ewma_peak[lead] = a * max(peak, 1e-6) + (1 - a) * ewma_peak[lead]
                    ewma_width[lead] = a * max(width, 1e-6) + (1 - a) * ewma_width[lead]
                    ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                    ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

                all_features.append(feat)

        except Exception as e:
            print(f"Skipping {rec_id}: {e}")

    return None, np.array(all_labels), np.array(all_features)


# ================================================================
# LOAD DATA
# ================================================================

print("Loading data and extracting features...")

if args.split == 'inter':
    print("  Mode: INTER-PATIENT (AAMI DS1 train / DS2 test)")
    _, train_labels, train_features = extract_beats_and_features(DS1_records)
    _, test_labels, test_features = extract_beats_and_features(DS2_records)
    print(f"  DS1 (train): {len(train_labels)} beats")
    print(f"  DS2 (test):  {len(test_labels)} beats")
else:
    print("  Mode: INTRA-PATIENT (80/20 random split)")
    _, all_labels, all_features = extract_beats_and_features(all_record_names)
    print(f"  Total: {len(all_labels)} beats")

    train_idx, test_idx = train_test_split(
        np.arange(len(all_labels)), test_size=0.2, random_state=42,
        stratify=all_labels
    )
    train_features = all_features[train_idx]
    test_features = all_features[test_idx]
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]
    print(f"  Train: {len(train_labels)}, Test: {len(test_labels)}")

num_features = train_features.shape[1]
print(f"  Features: {num_features} ({', '.join(FEATURE_NAMES)})")
print(f"  ({time.time()-t0:.1f}s)")

# Normalize using training statistics
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
train_features = (train_features - train_mean) / (train_std + 1e-8)
test_features = (test_features - train_mean) / (train_std + 1e-8)

# Print feature statistics
print(f"\n  Feature ranges (normalized training set):")
for i, name in enumerate(FEATURE_NAMES):
    raw_mean = train_mean[i]
    raw_std = train_std[i]
    print(f"    {name:>14}: raw mean={raw_mean:>8.4f}, std={raw_std:>8.4f}")


# ================================================================
# DATA LOADERS
# ================================================================

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

train_dataset = FeatureDataset(train_features, train_labels)
test_dataset = FeatureDataset(test_features, test_labels)

train_label_counts = np.bincount(train_labels, minlength=num_class)
class_sample_weights = 1.0 / (train_label_counts ** 0.65)
sample_weights = [class_sample_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                          drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False)

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"Device: {device}")

class_weights = 1.0 / (np.array(train_label_counts, dtype=np.float64) ** 0.65)
class_weights = class_weights / class_weights.sum() * num_class
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


# ================================================================
# RECURRENT SNN CLASSIFIER
# ================================================================
# Two-layer recurrent SNN with input normalization:
#   BatchNorm → fc1 → Dropout → RLeaky1 → fc_mid → Dropout → RLeaky2 → fc2 → Leaky
# Input is 19 patient-adaptive features presented as constant current for N steps.
# Two stacked recurrent layers: more expressive with fewer parameters than one wide layer.

class MinimalSNN(torch.nn.Module):
    def __init__(self, n_features, hidden, n_classes, beta=0.9, dropout=0.1):
        super().__init__()
        h1 = hidden
        h2 = max(hidden // 2, 8)
        self.bn = torch.nn.BatchNorm1d(n_features)
        self.fc1 = torch.nn.Linear(n_features, h1)
        self.drop1 = torch.nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1,
                                learn_beta=True, learn_threshold=True)
        self.fc_mid = torch.nn.Linear(h1, h2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2,
                                learn_beta=True, learn_threshold=True)
        self.fc2 = torch.nn.Linear(h2, n_classes)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1 = h1
        self.h2 = h2

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


# ================================================================
# QUANTIZATION (4-bit for memristive crossbar)
# ================================================================

def quantize_tensor(x, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


# ================================================================
# TRAINING: QAT + sparsity penalty + best checkpoint
# ================================================================

def train_model(net, num_epochs, num_steps, lambda_sparse, label=""):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs)

    best_ce = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup_epochs

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

            spk_out, mem_out, spk_hidden = net(data, num_steps)

            ce = torch.zeros(1, dtype=dtype, device=device)
            for s in range(num_steps):
                ce += loss_fn(mem_out[s], targets)

            firing_rate = spk_hidden.mean()
            sparsity = lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)
            total_loss = ce + sparsity

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"CE: {avg_ce:.2f} | fire: {epoch_rate/batches:.3f} | "
                  f"{time.time()-t0:.0f}s")

    if best_state is not None:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))
        print(f"  [{label}] Restored best (CE={best_ce:.3f}), quantized to 4-bit")

    return net


# ================================================================
# EVALUATION
# ================================================================

def evaluate(net, loader, num_steps):
    total = correct = 0
    all_preds, all_targets = [], []
    total_spikes = 0
    total_possible = 0

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

    acc = correct / total * 100
    fire_rate = total_spikes / total_possible
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
num_epochs = args.epochs
lambda_sparse = args.lam
label = f"min_h{hidden}_s{num_steps}"

net = MinimalSNN(num_features, hidden, num_class).to(device)
h1, h2 = net.h1, net.h2

print(f"\n{'='*65}")
print(f"MINIMAL FRONT-END SNN: {label}")
print(f"  Architecture: {num_features} -> {h1} RLeaky -> {h2} RLeaky -> 5 Leaky")
print(f"  Timesteps:    {num_steps}")
print(f"  Split:        {args.split}")
print(f"  Sparsity λ:   {lambda_sparse}")
print(f"{'='*65}")

n_params = sum(p.numel() for p in net.parameters())
print(f"  Parameters:   {n_params:,}")
print(f"    bn:         {num_features}×2 = {num_features*2}")
print(f"    fc1:        {num_features}×{h1} = {num_features*h1:,}")
print(f"    rec1:       {h1}×{h1} = {h1*h1:,}")
print(f"    fc_mid:     {h1}×{h2} = {h1*h2:,}")
print(f"    rec2:       {h2}×{h2} = {h2*h2:,}")
print(f"    fc2:        {h2}×5 = {h2*5}")

net = train_model(net, num_epochs, num_steps, lambda_sparse, label=label)
acc, preds, targets, fire_rate = evaluate(net, test_loader, num_steps)

print(f"\n{'='*65}")
print(f"RESULTS: {label} ({args.split}-patient)")
print(f"{'='*65}")
print(f"  Accuracy:     {acc:.2f}%")
print(f"  Parameters:   {n_params:,}")
print(f"  Firing rate:  {fire_rate:.3f} ({fire_rate*100:.1f}%)")

print(classification_report(targets, preds, target_names=['N', 'S', 'V', 'F', 'Q']))

cm = confusion_matrix(targets, preds)
print(f"Confusion matrix:")
print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
for i, lbl in enumerate(['N', 'S', 'V', 'F', 'Q']):
    print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

print(f"\nHardware sweep (4-bit quantization + analog noise):")
for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
    mean_acc, std_acc = hw_evaluate(net, num_steps, num_bits=4, sigma=sigma)
    print(f"  sigma={sigma:.3f}  {mean_acc:.2f}% +/- {std_acc:.2f}%")


# ================================================================
# ENERGY ANALYSIS
# ================================================================

print(f"\n{'='*65}")
print(f"ENERGY ANALYSIS")
print(f"{'='*65}")

fc1_macs = num_features * h1
rec1_macs = h1 * h1 * num_steps * fire_rate
mid_macs = h1 * h2 * num_steps * fire_rate
rec2_macs = h2 * h2 * num_steps * fire_rate
fc2_macs = h2 * num_class * num_steps * fire_rate
total_macs = fc1_macs + rec1_macs + mid_macs + rec2_macs + fc2_macs

print(f"\n  CLASSIFIER (memristive crossbar)")
print(f"    fc1:          {num_features} x {h1} = {fc1_macs:,} MACs (one-shot)")
print(f"    rec1:         {h1} x {h1} x {num_steps} x {fire_rate:.3f} = {rec1_macs:,.0f} MACs")
print(f"    fc_mid:       {h1} x {h2} x {num_steps} x {fire_rate:.3f} = {mid_macs:,.0f} MACs")
print(f"    rec2:         {h2} x {h2} x {num_steps} x {fire_rate:.3f} = {rec2_macs:,.0f} MACs")
print(f"    fc2:          {h2} x {num_class} x {num_steps} x {fire_rate:.3f} = {fc2_macs:,.0f} MACs")
print(f"    Total MACs:   {total_macs:,.0f}")

print(f"\n  FRONT-END (minimal analog circuits)")
n_circuits = 15
frontend_nW = 160
beat_period_s = 0.833
qrs_active_s = 0.100
timer_nW = 5

frontend_always_on_nJ = timer_nW * beat_period_s
frontend_qrs_active_nJ = (frontend_nW - timer_nW) * qrs_active_s
frontend_total_nJ = frontend_always_on_nJ + frontend_qrs_active_nJ

print(f"    Circuits:        {n_circuits} (comparators, peak-hold, integrators, RC-LPF, mini-crossbar)")
print(f"    Always-on:       R-R timer @ {timer_nW} nW x {beat_period_s}s = {frontend_always_on_nJ:.1f} nJ")
print(f"    QRS-gated:       {frontend_nW - timer_nW} nW x {qrs_active_s}s = {frontend_qrs_active_nJ:.1f} nJ")
print(f"    Front-end total: {frontend_total_nJ:.1f} nJ/beat")

print(f"\n  MULTI-POINT ENERGY SWEEP")
print(f"  {'pJ/MAC':>8} | {'Classifier':>12} | {'Front-end':>10} | {'Total':>10} | Note")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")

mac_points = [
    (0.35, "180nm RRAM cell (measured)"),
    (1.0,  "130nm RRAM crossbar"),
    (2.0,  "180nm full chip (realistic)"),
    (5.0,  "Conservative w/ peripherals"),
    (10.0, "Worst case"),
]

for pj, note in mac_points:
    cls_nJ = total_macs * pj / 1000
    tot_nJ = cls_nJ + frontend_total_nJ
    print(f"  {pj:>7.2f}  | {cls_nJ:>10.1f} nJ | {frontend_total_nJ:>8.1f} nJ | "
          f"{tot_nJ:>8.1f} nJ | {note}")

print(f"\n  COMPARISON TABLE")
print(f"  {'System':<45} | {'Proc.':>8} | {'Front':>8} | {'Total':>8} | {'Params':>8}")
print(f"  {'-'*45}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

cls_2pj = total_macs * 2.0 / 1000
cls_5pj = total_macs * 5.0 / 1000
tot_2pj = cls_2pj + frontend_total_nJ
tot_5pj = cls_5pj + frontend_total_nJ

rows = [
    (f"This work (2 pJ/MAC, {acc:.1f}%)",
     f"{cls_2pj:.1f} nJ", f"{frontend_total_nJ:.1f} nJ", f"{tot_2pj:.1f} nJ", f"{n_params:,}"),
    (f"This work (5 pJ/MAC, {acc:.1f}%)",
     f"{cls_5pj:.1f} nJ", f"{frontend_total_nJ:.1f} nJ", f"{tot_5pj:.1f} nJ", f"{n_params:,}"),
    ("Prev. design (OTA bank, 97.6%)",
     "262 nJ", "1833 nJ", "2095 nJ", "39,929"),
    ("Chu 2022 (40nm digital, 98.5%)",
     "750 nJ", "~200 nJ", "~950 nJ", "~10k"),
    ("SparrowSNN 2024 (22nm, 97.6%)",
     "11.8 nJ", "~150 nJ", "~162 nJ", "~5k"),
    ("EKGNet 2023 (analog, 95.0%)",
     "—", "—", "~9130 nJ", "~4k"),
]
for name, proc, fe, total, params in rows:
    print(f"  {name:<45} | {proc:>8} | {fe:>8} | {total:>8} | {params:>8}")

print(f"\n  KEY ADVANTAGES")
print(f"    Front-end:     {n_circuits} circuits, {frontend_nW} nW (vs 28+ OTAs, 2200 nW)")
print(f"    Parameters:    {n_params:,} (vs 39,929 = {n_params/39929:.2f}x)")
print(f"    Features:      {num_features} (vs 412 = {num_features/412:.2f}x)")
print(f"    Total energy:  {tot_2pj:.1f} nJ @ 2pJ (vs 2095 nJ = {tot_2pj/2095:.3f}x)")
if tot_2pj < 162:
    print(f"    vs SparrowSNN: {tot_2pj/162:.2f}x total system energy")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
