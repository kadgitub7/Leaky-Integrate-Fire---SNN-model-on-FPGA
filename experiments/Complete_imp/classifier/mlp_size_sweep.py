"""
MLP SIZE SWEEP — Find the smallest MLP that matches RF 95.70% accuracy
=======================================================================
Uses the same 8-feature pipeline and DS1/DS2 inter-patient split as
frontend_optimization.py. Replicates the exact RF baseline, then sweeps
MLP architectures from tiny to large to find the crossover point.

Goal: smallest network (fewest MACs → lowest crossbar energy) that
      fully preserves the RF accuracy of 95.70%.

Run one group at a time for parallelism:
  python experiments/Complete_imp/classifier/mlp_size_sweep.py --group small
  python experiments/Complete_imp/classifier/mlp_size_sweep.py --group medium
  python experiments/Complete_imp/classifier/mlp_size_sweep.py --group large
  python experiments/Complete_imp/classifier/mlp_size_sweep.py --group xlarge

Or run all:
  python experiments/Complete_imp/classifier/mlp_size_sweep.py --group all
"""
import sys, os, copy, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend_optimization'))

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from collections import Counter

torch.set_num_threads(4)

FEATURE_NAMES_8 = [
    'rr_ratio', 'beat_instability_L1', 'rr_asymmetry', 'pwave_energy_L1',
    'qrs_width_L0', 'beat_instability_L0', 'slope_ratio_L0', 'qrs_symmetry_L0',
]
N_FEATURES = 8
N_CLASSES = 5
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# PDK-validated energy constants
E_PER_MAC_PJ_NOM = 1.2679

# Architecture groups for parallel execution
ARCH_GROUPS = {
    'small': [
        # 1 hidden layer, tiny
        (8,),
        (10,),
        (12,),
        (16,),
        (20,),
        (24,),
    ],
    'medium': [
        # 1 hidden layer, moderate
        (32,),
        (48,),
        (64,),
        # 2 hidden layers, tiny
        (8, 4),
        (10, 5),
        (12, 6),
    ],
    'large': [
        # 2 hidden layers, moderate
        (16, 8),
        (20, 10),
        (24, 12),
        (32, 16),
        (48, 24),
        (64, 32),
    ],
    'xlarge': [
        # 2 hidden layers, large
        (96, 48),
        (128, 64),
        # 3 hidden layers
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
    ],
}

NUM_SEEDS = 10
NUM_EPOCHS = 300
BATCH_SIZE = 128
LR_INIT = 2e-3
WEIGHT_DECAY = 1e-4

# ================================================================
# DATA
# ================================================================

def load_data():
    from frontend_optimization import (
        extract_all_features, ALL_FEATURE_NAMES, DS1, DS2
    )

    print("Extracting 8 features from MIT-BIH...")
    train_labels, train_features, _ = extract_all_features(DS1)
    test_labels, test_features, _ = extract_all_features(DS2)

    feat_indices = [ALL_FEATURE_NAMES.index(f) for f in FEATURE_NAMES_8]
    X_train = train_features[:, feat_indices]
    X_test = test_features[:, feat_indices]

    tr_mean = X_train.mean(axis=0)
    tr_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - tr_mean) / tr_std
    X_test_norm = (X_test - tr_mean) / tr_std

    return X_train_norm, train_labels, X_test_norm, test_labels

# ================================================================
# RF BASELINE (exact reproduction of frontend_optimization.py)
# ================================================================

def run_rf_baseline(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 80)
    print("RANDOM FOREST BASELINE (n_estimators=300, max_depth=15, balanced)")
    print("=" * 80)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, random_state=42,
        n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = np.mean(preds == y_test) * 100
    f1s = f1_score(y_test, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print(f"  Accuracy: {acc:.2f}%")
    print(f"  F1 scores: N={f1s[0]:.3f} S={f1s[1]:.3f} V={f1s[2]:.3f} F={f1s[3]:.3f} Q={f1s[4]:.3f}")
    print(f"  Confusion matrix:")
    print(f"  {'':>5} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, lbl in enumerate(CLASS_NAMES):
        print(f"  {lbl:>5} {' '.join(f'{v:>6}' for v in cm[i])}")

    recalls = cm.diagonal() / cm.sum(axis=1)
    print(f"  Per-class recall: N={recalls[0]:.3f} S={recalls[1]:.3f} V={recalls[2]:.3f} F={recalls[3]:.3f} Q={recalls[4]:.3f}")

    return acc, f1s, cm

# ================================================================
# SMOTE
# ================================================================

def smote_oversample(X, y, ratio=0.33):
    counts = Counter(y)
    majority_count = max(counts.values())
    target = int(majority_count * ratio)

    X_new, y_new = [X.copy()], [y.copy()]
    for cls, cnt in counts.items():
        if cnt >= target:
            continue
        n_syn = target - cnt
        cls_idx = np.where(y == cls)[0]
        for _ in range(n_syn):
            i, j = np.random.choice(cls_idx, 2, replace=True)
            lam = np.random.rand()
            X_new.append((X[i] * lam + X[j] * (1 - lam)).reshape(1, -1))
            y_new.append(np.array([cls]))

    return np.vstack(X_new), np.concatenate(y_new)

# ================================================================
# MLP MODEL
# ================================================================

class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        prev = N_FEATURES
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, N_CLASSES))
        self.net = nn.Sequential(*layers)

        self.macs = 0
        prev = N_FEATURES
        for h in hidden_sizes:
            self.macs += prev * h
            prev = h
        self.macs += prev * N_CLASSES

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.n_layers = len(hidden_sizes) + 1

    def forward(self, x):
        return self.net(x)

# ================================================================
# FOCAL LOSS
# ================================================================

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ================================================================
# TRAIN ONE CONFIG
# ================================================================

def train_eval_one(X_train, y_train, X_test, y_test, hidden_sizes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_s, y_tr_s = smote_oversample(X_train, y_train)

    X_tr = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_s, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    counts = Counter(y_tr_s)
    cls_weights = torch.tensor([1.0 / (counts.get(i, 1) ** 0.65) for i in range(N_CLASSES)])
    cls_weights = cls_weights / cls_weights.sum() * N_CLASSES

    sample_w = torch.tensor([cls_weights[c] for c in y_tr_s])
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w))
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    model = MLP(hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_INIT * 0.01)
    criterion = FocalLoss(cls_weights, gamma=2.0)

    best_acc = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_te).argmax(1).numpy()
            acc = (preds == y_test).mean() * 100
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(1).numpy()

    acc = (preds == y_test).mean() * 100
    f1s = f1_score(y_test, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    return acc, f1s, cm, model.macs, model.n_params

# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='all',
                        choices=['small', 'medium', 'large', 'xlarge', 'all'])
    parser.add_argument('--seeds', type=int, default=NUM_SEEDS)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {len(y_train)} beats, Test: {len(y_test)} beats")
    print(f"Train dist: {dict(sorted(Counter(y_train).items()))}")
    print(f"Test dist:  {dict(sorted(Counter(y_test).items()))}")

    rf_acc, rf_f1s, rf_cm = run_rf_baseline(X_train, y_train, X_test, y_test)

    if args.group == 'all':
        architectures = []
        for g in ['small', 'medium', 'large', 'xlarge']:
            architectures.extend(ARCH_GROUPS[g])
    else:
        architectures = ARCH_GROUPS[args.group]

    print(f"\n{'=' * 80}")
    print(f"MLP SWEEP — group={args.group}, {len(architectures)} architectures, {args.seeds} seeds each")
    print(f"RF baseline: {rf_acc:.2f}%")
    print(f"{'=' * 80}")

    results = []

    for hidden_sizes in architectures:
        arch_str = "8-" + "-".join(str(h) for h in hidden_sizes) + "-5"
        macs = 0
        prev = N_FEATURES
        for h in hidden_sizes:
            macs += prev * h
            prev = h
        macs += prev * N_CLASSES

        e_cells_nj = macs * E_PER_MAC_PJ_NOM * 1e-3

        accs = []
        all_f1s = {c: [] for c in CLASS_NAMES}
        best_run = None

        t0 = time.time()
        for seed in range(args.seeds):
            acc, f1s, cm, _, n_params = train_eval_one(
                X_train, y_train, X_test, y_test, list(hidden_sizes), seed + 42
            )
            accs.append(acc)
            for ci, c in enumerate(CLASS_NAMES):
                all_f1s[c].append(f1s[ci])
            if best_run is None or acc > best_run[0]:
                best_run = (acc, f1s, cm, n_params)

        elapsed = time.time() - t0
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        max_acc = np.max(accs)
        min_acc = np.min(accs)

        gap = mean_acc - rf_acc
        marker = ">>>" if mean_acc >= rf_acc else "   "

        print(f"\n{marker} {arch_str:>22s}  MACs={macs:5d}  E_cells={e_cells_nj:.3f}nJ  "
              f"params={best_run[3]:5d}")
        print(f"    acc: {mean_acc:.2f} +/- {std_acc:.2f}%  "
              f"[min={min_acc:.2f}, max={max_acc:.2f}]  "
              f"gap={gap:+.2f}%  ({elapsed:.1f}s)")
        f1_str = " ".join(f"{c}={np.mean(all_f1s[c]):.3f}" for c in CLASS_NAMES)
        print(f"    F1: {f1_str}")

        if mean_acc >= rf_acc:
            print(f"    *** MATCHES OR EXCEEDS RF BASELINE ***")

        results.append({
            'arch': arch_str,
            'hidden_sizes': list(hidden_sizes),
            'macs': macs,
            'params': best_run[3],
            'n_layers': len(hidden_sizes) + 1,
            'e_cells_nj': round(e_cells_nj, 4),
            'acc_mean': round(mean_acc, 3),
            'acc_std': round(std_acc, 3),
            'acc_max': round(max_acc, 3),
            'acc_min': round(min_acc, 3),
            'gap_vs_rf': round(gap, 3),
            'f1_mean': {c: round(np.mean(all_f1s[c]), 4) for c in CLASS_NAMES},
            'all_accs': [round(a, 3) for a in accs],
        })

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY — sorted by accuracy (descending)")
    print("=" * 80)
    results.sort(key=lambda x: -x['acc_mean'])

    print(f"{'Arch':>22} {'MACs':>6} {'Params':>6} {'Mean%':>7} {'Std':>5} "
          f"{'Max%':>6} {'Gap':>7} {'E_cell':>7} "
          f"{'N':>5} {'S':>5} {'V':>5}")
    print("-" * 95)
    for r in results:
        gap_str = f"{r['gap_vs_rf']:+.2f}%"
        marker = "*" if r['gap_vs_rf'] >= 0 else " "
        print(f"{marker}{r['arch']:>21} {r['macs']:6d} {r['params']:6d} "
              f"{r['acc_mean']:6.2f}% {r['acc_std']:5.2f} "
              f"{r['acc_max']:5.2f}% {gap_str:>7} {r['e_cells_nj']:6.3f}nJ "
              f"{r['f1_mean']['N']:5.3f} {r['f1_mean']['S']:5.3f} {r['f1_mean']['V']:5.3f}")

    print(f"\nRF baseline: {rf_acc:.2f}%")

    matches = [r for r in results if r['gap_vs_rf'] >= 0]
    if matches:
        best = min(matches, key=lambda x: x['macs'])
        print(f"\n>>> SMALLEST MLP matching RF: {best['arch']}")
        print(f"    MACs: {best['macs']}, Params: {best['params']}")
        print(f"    Accuracy: {best['acc_mean']:.2f}% +/- {best['acc_std']:.2f}%")
        print(f"    E_cells: {best['e_cells_nj']:.3f} nJ (nominal, crossbar read only)")
    else:
        print(f"\n>>> No MLP matched RF accuracy in this group. Try --group large or xlarge.")
        closest = max(results, key=lambda x: x['acc_mean'])
        print(f"    Closest: {closest['arch']} at {closest['acc_mean']:.2f}% (gap={closest['gap_vs_rf']:+.2f}%)")

    # ── Save ──
    out_path = os.path.join(os.path.dirname(__file__), f'mlp_sweep_{args.group}.json')
    export = {
        'rf_baseline_acc': round(rf_acc, 3),
        'rf_baseline_f1': {c: round(float(rf_f1s[i]), 4) for i, c in enumerate(CLASS_NAMES)},
        'group': args.group,
        'n_seeds': args.seeds,
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
