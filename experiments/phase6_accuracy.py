"""
Phase 6: Accuracy Push — parallel-runnable variants to reach 93% inter / 96% intra / <40nJ
============================================================================================
Each variant runs independently via --variant flag for parallel execution.

Variants:
  A  — Baseline single model (reference point)
  B  — 5-seed ensemble (majority vote)
  C  — 7-seed ensemble (majority vote)
  E  — Cosine warm restarts + snapshot ensemble
  F  — Temporal attention (learned per-timestep weights)
  G  — R-Drop regularization (alpha=0.5)
  H  — Full combo: R-Drop + temporal attn + 5-seed ensemble

Each variant also runs per-class threshold optimization on TRAINING data
predictions (not test data — avoids data leakage).

Usage (run all in separate terminals):
  python experiments/phase6_accuracy.py --variant A --split inter
  python experiments/phase6_accuracy.py --variant B --split inter
  python experiments/phase6_accuracy.py --variant C --split inter
  python experiments/phase6_accuracy.py --variant E --split inter
  python experiments/phase6_accuracy.py --variant F --split inter
  python experiments/phase6_accuracy.py --variant G --split inter
  python experiments/phase6_accuracy.py --variant H --split inter
  (repeat with --split intra)
"""

import argparse, sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from data_proven import load_split, NUM_CLASSES, BATCH_SIZE

parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True, choices=['A','B','C','E','F','G','H'])
parser.add_argument('--split', default='inter', choices=['inter','intra'])
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

torch.set_num_threads(2)
t0 = time.time()


# ===================================================================
# Model definitions
# ===================================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def quantize_tensor(x, num_bits=4):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


class BaselineSNN(nn.Module):
    def __init__(self, n_features, hidden=48, n_classes=5, beta=0.9, dropout=0.05):
        super().__init__()
        h1, h2 = hidden, max(hidden // 2, 8)
        self.bn = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(n_features, h1)
        self.drop1 = nn.Dropout(dropout)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc_mid = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(dropout)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(h2, n_classes)
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


class TemporalAttnSNN(BaselineSNN):
    """Same architecture but with learnable per-timestep attention weights."""
    def __init__(self, n_features, hidden=48, n_classes=5, beta=0.9, dropout=0.05, num_steps=20):
        super().__init__(n_features, hidden, n_classes, beta, dropout)
        self.temporal_logits = nn.Parameter(torch.zeros(num_steps))

    def weighted_spike_count(self, spk_out):
        attn = F.softmax(self.temporal_logits[:spk_out.shape[0]], dim=0)
        return (spk_out * attn.view(-1, 1, 1)).sum(dim=0)


# ===================================================================
# Training
# ===================================================================

def train_model(net, train_loader, class_weights, device, num_epochs, num_steps,
                lambda_sparse=1.0, label="", use_rdrop=False, rdrop_alpha=0.5,
                use_warm_restarts=False, collect_snapshots=False):
    loss_fn = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5

    if use_warm_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs)

    best_ce = float('inf')
    best_state = None
    snapshots = []

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / warmup_epochs
        net.train()
        epoch_loss = epoch_rate = 0.0
        batches = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            saved = []
            with torch.no_grad():
                for p in net.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize_tensor(p.data, 4))

            spk_out, mem_out, spk_hidden = net(data, num_steps)
            ce = sum(loss_fn(mem_out[s], targets) for s in range(num_steps))
            firing_rate = spk_hidden.mean()
            total_loss = ce + lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)

            if use_rdrop:
                spk_out2, mem_out2, _ = net(data, num_steps)
                ce2 = sum(loss_fn(mem_out2[s], targets) for s in range(num_steps))
                total_loss = (ce + ce2) / 2
                total_loss += lambda_sparse * torch.clamp(firing_rate - 0.15, min=0.0)
                p1 = F.log_softmax(mem_out[-1], dim=1)
                p2 = F.softmax(mem_out2[-1], dim=1)
                q1 = F.softmax(mem_out[-1], dim=1)
                q2 = F.log_softmax(mem_out2[-1], dim=1)
                kl = (F.kl_div(p1, p2, reduction='batchmean') +
                      F.kl_div(q2, q1, reduction='batchmean')) / 2
                total_loss += rdrop_alpha * kl

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
            if use_warm_restarts:
                scheduler.step(epoch - warmup_epochs)
            else:
                scheduler.step()

        avg_ce = epoch_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())

        if collect_snapshots and epoch >= warmup_epochs:
            cycle_epoch = epoch - warmup_epochs
            if cycle_epoch > 0 and (cycle_epoch + 1) % 50 == 0:
                snap = copy.deepcopy(net.state_dict())
                with torch.no_grad():
                    for k in snap:
                        if snap[k].is_floating_point():
                            snap[k] = quantize_tensor(snap[k], 4)
                snapshots.append((epoch, snap))
                print(f"    [{label}] Snapshot saved at epoch {epoch+1}")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {avg_ce:.2f} | "
                  f"fire: {epoch_rate/batches:.3f} | {time.time()-t0:.0f}s")

    if best_state:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))

    return net, snapshots


# ===================================================================
# Evaluation & threshold optimization
# ===================================================================

def evaluate(net, loader, num_steps, device, use_temporal_attn=False):
    total = correct = 0
    all_preds, all_targets, all_spike_counts = [], [], []
    total_spikes = total_possible = 0
    with torch.no_grad():
        net.eval()
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, _, spk_hidden = net(data, num_steps)
            if use_temporal_attn and hasattr(net, 'weighted_spike_count'):
                counts = net.weighted_spike_count(spk_out)
            else:
                counts = spk_out.sum(dim=0)
            _, pred = counts.max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_spike_counts.append(counts.cpu().numpy())
            total_spikes += spk_hidden.sum().item()
            total_possible += spk_hidden.numel()
    acc = correct / total * 100
    fire_rate = total_spikes / total_possible
    spike_counts = np.concatenate(all_spike_counts, axis=0)
    return acc, all_preds, all_targets, fire_rate, spike_counts


def optimize_thresholds(spike_counts, targets, n_classes=5):
    """Optimize per-class bias via coordinate-descent grid search (robust, no scale).
    Bias-only avoids the instability of scale parameters on step-function objectives."""
    targets = np.array(targets)
    best_bias = np.zeros(n_classes)
    base_preds = spike_counts.argmax(axis=1)
    best_acc = np.mean(base_preds == targets)

    for iteration in range(5):
        improved = False
        for c in range(n_classes):
            for delta in np.linspace(-3, 3, 61):
                trial_bias = best_bias.copy()
                trial_bias[c] = delta
                adjusted = spike_counts + trial_bias
                acc = np.mean(adjusted.argmax(axis=1) == targets)
                if acc > best_acc:
                    best_acc = acc
                    best_bias = trial_bias.copy()
                    improved = True
        if not improved:
            break
    return best_bias


def apply_thresholds(spike_counts, bias):
    return (spike_counts + bias).argmax(axis=1)


def compute_energy(n_features, h1, h2, num_steps, fire_rate):
    fc1_macs = n_features * h1
    rec1_macs = h1 * h1 * num_steps * fire_rate
    mid_macs = h1 * h2 * num_steps * fire_rate
    rec2_macs = h2 * h2 * num_steps * fire_rate
    out_macs = h2 * 5 * num_steps * fire_rate
    total_macs = fc1_macs + rec1_macs + mid_macs + rec2_macs + out_macs
    cls_nJ = total_macs * 2 / 1000
    frontend_nJ = 5 * 0.833 + 95 * 0.100
    return total_macs, cls_nJ, frontend_nJ, cls_nJ + frontend_nJ


def report(label, acc, preds, targets, fire_rate, n_features, h1, h2, num_steps):
    print(f"\n{'='*65}")
    print(f"  {label}: {acc:.2f}%")
    print(f"  Fire rate: {fire_rate:.3f}")
    total_macs, cls_nJ, fe_nJ, total_nJ = compute_energy(n_features, h1, h2, num_steps, fire_rate)
    print(f"  Energy: {total_macs:,.0f} MACs | classifier={cls_nJ:.1f}nJ | "
          f"frontend={fe_nJ:.1f}nJ | total={total_nJ:.1f}nJ")
    print(classification_report(targets, preds, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(targets, preds)
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N','S','V','F','Q']):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")
    return total_nJ


def train_and_eval_single(train_loader, test_loader, class_weights, device,
                          n_features, seed, num_steps, num_epochs, label,
                          use_temporal_attn=False, use_rdrop=False,
                          use_warm_restarts=False, collect_snapshots=False):
    """Train one model end-to-end. Returns (net, acc, preds, targets, fire, counts, snapshots)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_temporal_attn:
        net = TemporalAttnSNN(n_features, 48, NUM_CLASSES, dropout=0.05,
                              num_steps=num_steps).to(device)
    else:
        net = BaselineSNN(n_features, 48, NUM_CLASSES, dropout=0.05).to(device)

    actual_epochs = 250 if use_warm_restarts else num_epochs
    net, snapshots = train_model(net, train_loader, class_weights, device,
                                 num_epochs=actual_epochs, num_steps=num_steps,
                                 label=label, use_rdrop=use_rdrop,
                                 rdrop_alpha=0.5, use_warm_restarts=use_warm_restarts,
                                 collect_snapshots=collect_snapshots)
    acc, preds, targets, fire, counts = evaluate(
        net, test_loader, num_steps, device, use_temporal_attn=use_temporal_attn)
    return net, acc, preds, targets, fire, counts, snapshots


def get_train_spike_counts(net, train_loader, num_steps, device, use_temporal_attn=False):
    """Get spike counts on TRAINING data for threshold optimization (no leakage)."""
    all_counts, all_targets = [], []
    with torch.no_grad():
        net.eval()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, _, _ = net(data, num_steps)
            if use_temporal_attn and hasattr(net, 'weighted_spike_count'):
                counts = net.weighted_spike_count(spk_out)
            else:
                counts = spk_out.sum(dim=0)
            all_counts.append(counts.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return np.concatenate(all_counts, axis=0), all_targets


# ===================================================================
# Variant implementations
# ===================================================================

def run_A(train_loader, test_loader, class_weights, device, n_features):
    """Baseline single model + threshold optimization."""
    num_steps, h1, h2 = args.steps, 48, 24
    split_name = args.split.upper()

    print(f"\n{'-'*65}")
    print(f"  [A] Baseline — single model, proven pipeline ({num_steps} steps)")
    print(f"{'-'*65}")

    net, acc, preds, targets, fire, test_counts, _ = train_and_eval_single(
        train_loader, test_loader, class_weights, device, n_features,
        seed=42, num_steps=num_steps, num_epochs=args.epochs, label="A-baseline")
    report(f"A. Baseline ({split_name})", acc, preds, targets, fire,
           n_features, h1, h2, num_steps)

    # Threshold optimization on TRAINING data (no leakage)
    train_counts, train_targets = get_train_spike_counts(net, train_loader, num_steps, device)
    bias = optimize_thresholds(train_counts, train_targets)
    thresh_preds = apply_thresholds(test_counts, bias)
    thresh_acc = np.mean(thresh_preds == np.array(targets)) * 100
    print(f"\n  Threshold optimization (trained on training predictions):")
    print(f"  Bias: {np.round(bias, 4)}")
    report(f"A+thresh ({split_name})", thresh_acc, thresh_preds.tolist(), targets, fire,
           n_features, h1, h2, num_steps)

    print(f"\n  SUMMARY: baseline={acc:.2f}% | +thresh={thresh_acc:.2f}%")


def run_ensemble(train_loader, test_loader, class_weights, device, n_features, n_seeds):
    """N-seed ensemble with majority vote + threshold optimization."""
    num_steps, h1, h2 = args.steps, 48, 24
    split_name = args.split.upper()
    variant_label = 'B' if n_seeds == 5 else 'C'
    seeds = [42 + i * 13 for i in range(n_seeds)]

    print(f"\n{'-'*65}")
    print(f"  [{variant_label}] {n_seeds}-seed ensemble ({num_steps} steps)")
    print(f"{'-'*65}")

    all_preds, all_counts, all_fires = [], [], []
    all_train_counts = []
    targets = None
    train_targets_ens = None

    for i, seed in enumerate(seeds):
        net, acc_i, preds_i, targets, fire_i, counts_i, _ = train_and_eval_single(
            train_loader, test_loader, class_weights, device, n_features,
            seed=seed, num_steps=num_steps, num_epochs=args.epochs,
            label=f"{variant_label}-s{seed}")
        all_preds.append(preds_i)
        all_counts.append(counts_i)
        all_fires.append(fire_i)
        print(f"    Model {i+1}/{n_seeds} (seed={seed}): {acc_i:.2f}%")

        # Get training spike counts for threshold optimization
        tr_counts, tr_tgts = get_train_spike_counts(net, train_loader, num_steps, device)
        all_train_counts.append(tr_counts)
        if train_targets_ens is None:
            train_targets_ens = tr_tgts
        del net

    avg_fire = np.mean(all_fires)

    # Hard majority vote
    ens_preds = []
    for j in range(len(targets)):
        votes = [all_preds[m][j] for m in range(n_seeds)]
        ens_preds.append(max(set(votes), key=votes.count))
    ens_acc = np.mean(np.array(ens_preds) == np.array(targets)) * 100
    report(f"{variant_label}. {n_seeds}-seed ensemble ({split_name})", ens_acc,
           ens_preds, targets, avg_fire, n_features, h1, h2, num_steps)

    # Soft vote (average spike counts) + threshold optimization on training data
    avg_test_counts = np.mean(all_counts, axis=0)
    avg_train_counts = np.mean(all_train_counts, axis=0)
    bias = optimize_thresholds(avg_train_counts, train_targets_ens)
    soft_thresh_preds = apply_thresholds(avg_test_counts, bias)
    soft_thresh_acc = np.mean(soft_thresh_preds == np.array(targets)) * 100
    print(f"\n  Bias: {np.round(bias, 4)}")
    report(f"{variant_label}+thresh. soft vote+thresholds ({split_name})", soft_thresh_acc,
           soft_thresh_preds.tolist(), targets, avg_fire, n_features, h1, h2, num_steps)

    # Also report plain soft vote (argmax of averaged counts, no thresholds)
    soft_preds = avg_test_counts.argmax(axis=1)
    soft_acc = np.mean(soft_preds == np.array(targets)) * 100
    print(f"\n  Soft vote (no thresholds): {soft_acc:.2f}%")

    print(f"\n  SUMMARY: hard={ens_acc:.2f}% | soft={soft_acc:.2f}% | soft+thresh={soft_thresh_acc:.2f}%")


def run_E(train_loader, test_loader, class_weights, device, n_features):
    """Cosine warm restarts + snapshot ensemble."""
    num_steps, h1, h2 = args.steps, 48, 24
    split_name = args.split.upper()

    print(f"\n{'-'*65}")
    print(f"  [E] Cosine warm restarts + snapshot ensemble ({num_steps} steps)")
    print(f"{'-'*65}")

    net, acc, preds, targets, fire, test_counts, snapshots = train_and_eval_single(
        train_loader, test_loader, class_weights, device, n_features,
        seed=42, num_steps=num_steps, num_epochs=250, label="E-warmrestart",
        use_warm_restarts=True, collect_snapshots=True)
    report(f"E. Warm restart best-CE ({split_name})", acc, preds, targets, fire,
           n_features, h1, h2, num_steps)

    if snapshots:
        print(f"\n  Snapshot ensemble ({len(snapshots)} snapshots):")
        snap_preds_all, snap_counts_all = [], []
        for si, (ep, snap_state) in enumerate(snapshots):
            snap_net = BaselineSNN(n_features, 48, NUM_CLASSES, dropout=0.05).to(device)
            snap_net.load_state_dict(snap_state)
            acc_s, preds_s, _, fire_s, counts_s = evaluate(
                snap_net, test_loader, num_steps, device)
            snap_preds_all.append(preds_s)
            snap_counts_all.append(counts_s)
            print(f"    Snapshot {si+1} (epoch {ep+1}): {acc_s:.2f}%")
            del snap_net

        # Snapshot majority vote
        snap_ens_preds = []
        for j in range(len(targets)):
            votes = [snap_preds_all[m][j] for m in range(len(snapshots))]
            snap_ens_preds.append(max(set(votes), key=votes.count))
        snap_acc = np.mean(np.array(snap_ens_preds) == np.array(targets)) * 100
        report(f"E-snap. Snapshot ensemble ({split_name})", snap_acc,
               snap_ens_preds, targets, fire, n_features, h1, h2, num_steps)

        # Snapshot soft vote + thresholds (on training data)
        avg_snap_counts = np.mean(snap_counts_all, axis=0)
        train_counts, train_targets = get_train_spike_counts(
            net, train_loader, num_steps, device)
        bias = optimize_thresholds(train_counts, train_targets)
        snap_thresh_preds = apply_thresholds(avg_snap_counts, bias)
        snap_thresh_acc = np.mean(snap_thresh_preds == np.array(targets)) * 100

        print(f"\n  SUMMARY: best-CE={acc:.2f}% | snap-vote={snap_acc:.2f}% | "
              f"snap+thresh={snap_thresh_acc:.2f}%")
    else:
        print(f"\n  WARNING: No snapshots collected. Check epoch count vs cycle length.")
        print(f"  SUMMARY: best-CE={acc:.2f}%")


def run_F(train_loader, test_loader, class_weights, device, n_features):
    """Temporal attention (learned per-timestep weights)."""
    num_steps, h1, h2 = args.steps, 48, 24
    split_name = args.split.upper()

    print(f"\n{'-'*65}")
    print(f"  [F] Temporal attention ({num_steps} steps)")
    print(f"{'-'*65}")

    net, acc, preds, targets, fire, test_counts, _ = train_and_eval_single(
        train_loader, test_loader, class_weights, device, n_features,
        seed=42, num_steps=num_steps, num_epochs=args.epochs, label="F-tempattn",
        use_temporal_attn=True)
    report(f"F. Temporal attention ({split_name})", acc, preds, targets, fire,
           n_features, h1, h2, num_steps)

    if hasattr(net, 'temporal_logits'):
        attn_w = F.softmax(net.temporal_logits, dim=0).detach().cpu().numpy()
        print(f"  Learned temporal weights: {np.round(attn_w, 3)}")
        early_weight = attn_w[:num_steps//2].sum()
        late_weight = attn_w[num_steps//2:].sum()
        print(f"  Early half weight: {early_weight:.3f} | Late half weight: {late_weight:.3f}")

    # + thresholds
    train_counts, train_targets = get_train_spike_counts(
        net, train_loader, num_steps, device, use_temporal_attn=True)
    bias = optimize_thresholds(train_counts, train_targets)
    thresh_preds = apply_thresholds(test_counts, bias)
    thresh_acc = np.mean(thresh_preds == np.array(targets)) * 100
    report(f"F+thresh ({split_name})", thresh_acc, thresh_preds.tolist(), targets, fire,
           n_features, h1, h2, num_steps)

    print(f"\n  SUMMARY: temporal_attn={acc:.2f}% | +thresh={thresh_acc:.2f}%")


def run_G(train_loader, test_loader, class_weights, device, n_features):
    """R-Drop regularization."""
    num_steps, h1, h2 = args.steps, 48, 24
    split_name = args.split.upper()

    print(f"\n{'-'*65}")
    print(f"  [G] R-Drop regularization ({num_steps} steps)")
    print(f"{'-'*65}")

    net, acc, preds, targets, fire, test_counts, _ = train_and_eval_single(
        train_loader, test_loader, class_weights, device, n_features,
        seed=42, num_steps=num_steps, num_epochs=args.epochs, label="G-rdrop",
        use_rdrop=True)
    report(f"G. R-Drop ({split_name})", acc, preds, targets, fire,
           n_features, h1, h2, num_steps)

    # + thresholds
    train_counts, train_targets = get_train_spike_counts(net, train_loader, num_steps, device)
    bias = optimize_thresholds(train_counts, train_targets)
    thresh_preds = apply_thresholds(test_counts, bias)
    thresh_acc = np.mean(thresh_preds == np.array(targets)) * 100
    report(f"G+thresh ({split_name})", thresh_acc, thresh_preds.tolist(), targets, fire,
           n_features, h1, h2, num_steps)

    print(f"\n  SUMMARY: rdrop={acc:.2f}% | +thresh={thresh_acc:.2f}%")


def run_H(train_loader, test_loader, class_weights, device, n_features):
    """Full combo: R-Drop + temporal attn + warm restarts + 5-seed ensemble."""
    num_steps, h1, h2 = args.steps, 48, 24
    split_name = args.split.upper()
    seeds = [42, 55, 68, 81, 94]

    print(f"\n{'-'*65}")
    print(f"  [H] Full combo: R-Drop + temporal attn + warm restarts + 5-seed ({num_steps} steps)")
    print(f"{'-'*65}")

    all_preds, all_counts, all_fires = [], [], []
    all_train_counts = []
    targets = None
    train_targets_ens = None

    for i, seed in enumerate(seeds):
        net, acc_i, preds_i, targets, fire_i, counts_i, _ = train_and_eval_single(
            train_loader, test_loader, class_weights, device, n_features,
            seed=seed, num_steps=num_steps, num_epochs=250,
            label=f"H-s{seed}",
            use_temporal_attn=True, use_rdrop=True, use_warm_restarts=True)
        all_preds.append(preds_i)
        all_counts.append(counts_i)
        all_fires.append(fire_i)
        print(f"    Model {i+1}/5 (seed={seed}): {acc_i:.2f}%")

        tr_counts, tr_tgts = get_train_spike_counts(
            net, train_loader, num_steps, device, use_temporal_attn=True)
        all_train_counts.append(tr_counts)
        if train_targets_ens is None:
            train_targets_ens = tr_tgts
        del net

    avg_fire = np.mean(all_fires)

    # Hard vote
    hard_preds = []
    for j in range(len(targets)):
        votes = [all_preds[m][j] for m in range(5)]
        hard_preds.append(max(set(votes), key=votes.count))
    hard_acc = np.mean(np.array(hard_preds) == np.array(targets)) * 100
    report(f"H. Full combo hard vote ({split_name})", hard_acc,
           hard_preds, targets, avg_fire, n_features, h1, h2, num_steps)

    # Soft vote + thresholds (trained on training predictions)
    avg_test_counts = np.mean(all_counts, axis=0)
    avg_train_counts = np.mean(all_train_counts, axis=0)
    bias = optimize_thresholds(avg_train_counts, train_targets_ens)
    soft_thresh_preds = apply_thresholds(avg_test_counts, bias)
    soft_thresh_acc = np.mean(soft_thresh_preds == np.array(targets)) * 100
    report(f"H+thresh. Full combo + thresholds ({split_name})", soft_thresh_acc,
           soft_thresh_preds.tolist(), targets, avg_fire, n_features, h1, h2, num_steps)

    # Plain soft vote
    soft_preds = avg_test_counts.argmax(axis=1)
    soft_acc = np.mean(soft_preds == np.array(targets)) * 100

    print(f"\n  SUMMARY: hard={hard_acc:.2f}% | soft={soft_acc:.2f}% | soft+thresh={soft_thresh_acc:.2f}%")


# ===================================================================
# Main
# ===================================================================

if __name__ == '__main__':
    print(f"\nPhase 6 - Variant {args.variant} | {args.split}-patient | {args.steps} steps")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_loader, test_loader, class_weights, device, n_features = load_split(
        split=args.split, include_cross_lead=False, smote_ratio=0.33,
        smote_exclude_classes=None, cls_power=0.65)

    if args.variant == 'A':
        run_A(train_loader, test_loader, class_weights, device, n_features)
    elif args.variant == 'B':
        run_ensemble(train_loader, test_loader, class_weights, device, n_features, 5)
    elif args.variant == 'C':
        run_ensemble(train_loader, test_loader, class_weights, device, n_features, 7)
    elif args.variant == 'E':
        run_E(train_loader, test_loader, class_weights, device, n_features)
    elif args.variant == 'F':
        run_F(train_loader, test_loader, class_weights, device, n_features)
    elif args.variant == 'G':
        run_G(train_loader, test_loader, class_weights, device, n_features)
    elif args.variant == 'H':
        run_H(train_loader, test_loader, class_weights, device, n_features)

    print(f"\n  Total time: {time.time()-t0:.0f}s")
