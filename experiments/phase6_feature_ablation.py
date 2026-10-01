"""
Phase 6b: Feature Ablation — test dropping high-domain-shift morphology features
================================================================================
Timing features shift <0.02σ between DS1→DS2 (patient-invariant).
Morphology features shift 0.4-0.6σ (patient-specific, unreliable for inter).

Tests dropping combinations of the highest-shift features:
  - max_slope_L0, max_slope_L1 (highest patient variability)
  - rel_area_L0, rel_area_L1 (high shift, moderate importance)
  - Keep timing (6) + qrs_width (2) + qrs_area (2) + templ_corr (2) = 12 minimum

Also tests adding the RF #4 feature (corr_L0L1) while dropping slopes.

Usage:
  python experiments/phase6_feature_ablation.py
"""

import sys, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from data_proven import (extract_features, smote_oversample, DS1, DS2, ALL_RECORDS,
                         NUM_CLASSES, BATCH_SIZE)
from sklearn.model_selection import train_test_split

torch.set_num_threads(2)
t0 = time.time()

FEATURE_NAMES_16 = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1', 'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1', 'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1'
]

ABLATION_CONFIGS = {
    'baseline_16f': list(range(16)),
    'drop_slopes_14f': [i for i in range(16) if i not in [10, 11]],
    'drop_rel_area_14f': [i for i in range(16) if i not in [12, 13]],
    'drop_both_12f': [i for i in range(16) if i not in [10, 11, 12, 13]],
    'timing_only_6f': list(range(6)),
    'timing_templ_8f': list(range(6)) + [14, 15],
}


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


class AblationSNN(nn.Module):
    def __init__(self, n_features, hidden=48, n_classes=5, beta=0.9, dropout=0.05):
        super().__init__()
        h1 = hidden
        h2 = max(hidden // 2, 8)
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


def train_and_eval(train_features, train_labels, test_features, test_labels,
                   device, n_features, label="", num_epochs=200, num_steps=20):
    train_features_s, train_labels_s = smote_oversample(
        train_features, train_labels, target_ratio=0.33)

    mu, sd = train_features_s.mean(0), train_features_s.std(0)
    train_norm = (train_features_s - mu) / (sd + 1e-8)
    test_norm = (test_features - mu) / (sd + 1e-8)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    tr_counts = np.bincount(train_labels_s, minlength=NUM_CLASSES)
    sw = 1.0 / (tr_counts ** 0.65)
    sample_w = [sw[l] for l in train_labels_s]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        DS(train_norm, train_labels_s), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        DS(test_norm, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    cw = 1.0 / (tr_counts.astype(np.float64) ** 0.65)
    cw = cw / cw.sum() * NUM_CLASSES
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    torch.manual_seed(42)
    np.random.seed(42)
    net = AblationSNN(n_features, hidden=48, dropout=0.05).to(device)

    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 5)
    best_ce = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        if epoch < 5:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-3 * (epoch + 1) / 5
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
            total_loss = ce + 1.0 * torch.clamp(firing_rate - 0.15, min=0.0)
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
        if epoch >= 5:
            scheduler.step()
        avg_ce = epoch_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    [{label}] Epoch {epoch+1:3d}/{num_epochs} | CE: {avg_ce:.2f} | "
                  f"fire: {epoch_rate/batches:.3f} | {time.time()-t0:.0f}s")

    if best_state:
        net.load_state_dict(best_state)
        with torch.no_grad():
            for p in net.parameters():
                p.data.copy_(quantize_tensor(p.data, 4))

    total = correct = 0
    all_preds, all_targets = [], []
    total_spikes = total_possible = 0
    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
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


def compute_energy(n_features, h1=48, h2=24, num_steps=20, fire_rate=0.165):
    fc1 = n_features * h1
    rec1 = h1 * h1 * num_steps * fire_rate
    mid = h1 * h2 * num_steps * fire_rate
    rec2 = h2 * h2 * num_steps * fire_rate
    out = h2 * 5 * num_steps * fire_rate
    total_macs = fc1 + rec1 + mid + rec2 + out
    cls_nJ = total_macs * 2 / 1000
    n_circuits = n_features - 6 + 5
    frontend_nJ = 5 * 0.833 + n_circuits * 0.100
    return cls_nJ, frontend_nJ, cls_nJ + frontend_nJ


if __name__ == '__main__':
    print(f"\nPhase 6b: Feature Ablation Study")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    print(f"\nExtracting features from DS1 and DS2...")
    train_labels_all, train_features_all = extract_features(DS1, include_cross_lead=True)
    test_labels_all, test_features_all = extract_features(DS2, include_cross_lead=True)
    print(f"  DS1: {len(train_labels_all)} beats | DS2: {len(test_labels_all)} beats")

    results = {}

    for config_name, feat_indices in ABLATION_CONFIGS.items():
        feat_names = [FEATURE_NAMES_16[i] if i < 16 else 'corr_L0L1' for i in feat_indices]
        n_feat = len(feat_indices)

        print(f"\n{'-'*65}")
        print(f"  {config_name} ({n_feat} features)")
        print(f"  Features: {', '.join(feat_names)}")
        print(f"{'-'*65}")

        train_f = train_features_all[:, feat_indices]
        test_f = test_features_all[:, feat_indices]

        acc, preds, targets, fire_rate = train_and_eval(
            train_f, train_labels_all, test_f, test_labels_all,
            device, n_feat, label=config_name)

        cls_nJ, fe_nJ, total_nJ = compute_energy(n_feat, fire_rate=fire_rate)
        print(f"\n  {config_name}: {acc:.2f}% | fire={fire_rate:.3f} | "
              f"cls={cls_nJ:.1f}nJ | total={total_nJ:.1f}nJ")
        print(classification_report(targets, preds,
              target_names=['N','S','V','F','Q'], zero_division=0))
        cm = confusion_matrix(targets, preds)
        print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
        for i, lbl in enumerate(['N','S','V','F','Q']):
            print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

        results[config_name] = {'acc': acc, 'energy': total_nJ, 'n_feat': n_feat,
                                'fire': fire_rate}

    # Also test with corr_L0L1 replacing slopes
    print(f"\n{'-'*65}")
    print(f"  replace_slopes_with_corr (14f + corr_L0L1 = 15f)")
    print(f"{'-'*65}")
    idx_15 = [i for i in range(16) if i not in [10, 11]] + [16]
    feat_names_15 = [FEATURE_NAMES_16[i] if i < 16 else 'corr_L0L1' for i in idx_15]
    print(f"  Features: {', '.join(feat_names_15)}")
    train_f15 = train_features_all[:, idx_15]
    test_f15 = test_features_all[:, idx_15]
    acc_15, preds_15, tgts_15, fire_15 = train_and_eval(
        train_f15, train_labels_all, test_f15, test_labels_all,
        device, len(idx_15), label="replace_slopes")
    cls_15, fe_15, total_15 = compute_energy(len(idx_15), fire_rate=fire_15)
    print(f"\n  replace_slopes: {acc_15:.2f}% | fire={fire_15:.3f} | total={total_15:.1f}nJ")
    print(classification_report(tgts_15, preds_15,
          target_names=['N','S','V','F','Q'], zero_division=0))
    cm15 = confusion_matrix(tgts_15, preds_15)
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(['N','S','V','F','Q']):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm15[i])}")
    results['replace_slopes_15f'] = {'acc': acc_15, 'energy': total_15, 'n_feat': 15,
                                     'fire': fire_15}

    # Summary
    print(f"\n{'='*65}")
    print(f"  FEATURE ABLATION SUMMARY — INTER-PATIENT")
    print(f"{'='*65}")
    print(f"  {'Config':>25} {'#Feat':>6} {'Acc':>8} {'Energy':>8} {'Delta':>8}")
    baseline_acc = results['baseline_16f']['acc']
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        delta = r['acc'] - baseline_acc
        print(f"  {name:>25} {r['n_feat']:>6} {r['acc']:>7.2f}% {r['energy']:>7.1f}nJ {delta:>+7.2f}%")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
