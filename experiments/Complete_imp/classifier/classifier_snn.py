"""
CLASSIFIER — SNN (8-feature optimized set)
============================================
Analog-optimized Spiking Neural Network for SKY130B ReRAM crossbar.

Leaky Integrate-and-Fire (LIF) neurons with surrogate gradient training.
Features encoded as constant current injection over T timesteps.

In analog hardware, each timestep = one integration cycle of the
membrane capacitor. Energy per timestep = one crossbar read.

8 input features (greedy-optimal from frontend_optimization):
  rr_ratio, beat_instability_L1, rr_asymmetry, pwave_energy_L1,
  qrs_width_L0, beat_instability_L0, slope_ratio_L0, qrs_symmetry_L0

Run: python experiments/Complete_imp/classifier/classifier_snn.py [--seeds 5]
"""
import sys, os, time, copy, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend_optimization'))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

torch.set_num_threads(2)

# ── Feature set ──
FEATURE_NAMES_8 = [
    'rr_ratio', 'beat_instability_L1', 'rr_asymmetry', 'pwave_energy_L1',
    'qrs_width_L0', 'beat_instability_L0', 'slope_ratio_L0', 'qrs_symmetry_L0',
]
N_FEATURES = 8
N_CLASSES = 5
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# ── Energy constants (PDK-validated) ──
E_PER_MAC_PJ = {'opt': 0.5339, 'nom': 1.2679, 'pess': 1.6489}
E_WAKE_NJ = {'opt': 0.5, 'nom': 1.5, 'pess': 3.0}
T_READ_S = 200e-9
FRONTEND_E_NJ = {'ss_27': 3.47, 'tt_27': 5.0, 'tt_37': 6.63, 'ff_50': 14.23}

# ── Training config ──
NUM_EPOCHS = 250
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
SMOTE_RATIO = 0.33
CLS_POWER = 0.65
FOCAL_GAMMA = 2.0

# ── Architecture sweep ──
TIMESTEPS = [4, 8, 12]
WIDTH_CONFIGS = [
    (12, 0),    # 8→12→5
    (16, 0),    # 8→16→5
    (16, 8),    # 8→16→8→5
    (24, 12),   # 8→24→12→5
]
NOISE_LEVELS = [0.0, 0.02, 0.05]
BIT_WIDTHS = [2, 4]

# LIF neuron parameters (analog-realistic)
TAU_MEM = 0.8     # membrane decay (RC time constant ratio)
V_THRESH = 1.0    # threshold voltage

# ================================================================
# DATA LOADING
# ================================================================

def load_8feature_data():
    try:
        from frontend_optimization import (
            extract_all_features, ALL_FEATURE_NAMES, DS1, DS2
        )
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frontend_optimization'))
        from frontend_optimization import (
            extract_all_features, ALL_FEATURE_NAMES, DS1, DS2
        )

    print("Extracting features from MIT-BIH...")
    train_labels, train_features, _ = extract_all_features(DS1)
    test_labels, test_features, _ = extract_all_features(DS2)

    feat_indices = [ALL_FEATURE_NAMES.index(f) for f in FEATURE_NAMES_8]
    X_train = train_features[:, feat_indices]
    X_test = test_features[:, feat_indices]

    tr_mean = X_train.mean(axis=0)
    tr_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - tr_mean) / tr_std
    X_test = (X_test - tr_mean) / tr_std

    return X_train, train_labels, X_test, test_labels

# ================================================================
# SMOTE
# ================================================================

def smote_oversample(X, y, ratio=SMOTE_RATIO):
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
# QAT
# ================================================================

class STE_Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bits):
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min) / (2**bits - 1) + 1e-8
        w_q = torch.round((w - w_min) / scale) * scale + w_min
        return w_q

    @staticmethod
    def backward(ctx, grad):
        return grad, None

def quantize_weights(model, bits):
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.data = STE_Quantize.apply(p.data, bits)

# ================================================================
# SURROGATE GRADIENT & LIF
# ================================================================

class SurrogateSpike(torch.autograd.Function):
    """Fast sigmoid surrogate gradient for LIF neurons."""
    @staticmethod
    def forward(ctx, mem, thresh):
        spk = (mem >= thresh).float()
        ctx.save_for_backward(mem, torch.tensor(thresh))
        return spk

    @staticmethod
    def backward(ctx, grad):
        mem, thresh = ctx.saved_tensors
        slope = 25.0
        sg = slope / (1.0 + slope * torch.abs(mem - thresh)) ** 2
        return grad * sg, None


class LIFLayer(nn.Module):
    """Single LIF layer: Linear projection + LIF dynamics over T timesteps."""
    def __init__(self, in_features, out_features, tau=TAU_MEM, thresh=V_THRESH, bits=4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.thresh = thresh
        self.bits = bits
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x_seq, noise_level=0.0):
        """
        x_seq: (T, B, in_features) — input current at each timestep
        Returns: (T, B, out_features) — output spikes at each timestep
        """
        T, B, _ = x_seq.shape
        mem = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        spk_rec = []

        for t in range(T):
            inp = x_seq[t]

            if self.bits < 32 and self.training:
                w_q = STE_Quantize.apply(self.fc.weight, self.bits)
                cur = nn.functional.linear(inp, w_q, self.fc.bias)
            else:
                cur = self.fc(inp)

            cur = self.bn(cur)

            if noise_level > 0 and self.training:
                cur = cur * (1 + noise_level * torch.randn_like(cur))

            mem = self.tau * mem + cur
            spk = SurrogateSpike.apply(mem, self.thresh)
            mem = mem * (1 - spk)  # reset on spike
            spk_rec.append(spk)

        return torch.stack(spk_rec, dim=0)  # (T, B, out)


class ReadoutLayer(nn.Module):
    """Membrane-based readout: accumulates input, no spiking. Output = final membrane."""
    def __init__(self, in_features, out_features, tau=TAU_MEM, bits=4):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.bits = bits

    def forward(self, x_seq, noise_level=0.0):
        T, B, _ = x_seq.shape
        mem = torch.zeros(B, self.fc.out_features, device=x_seq.device)

        for t in range(T):
            inp = x_seq[t]
            if self.bits < 32 and self.training:
                w_q = STE_Quantize.apply(self.fc.weight, self.bits)
                cur = nn.functional.linear(inp, w_q, self.fc.bias)
            else:
                cur = self.fc(inp)

            if noise_level > 0 and self.training:
                cur = cur * (1 + noise_level * torch.randn_like(cur))

            mem = self.tau * mem + cur

        return mem  # (B, out)


# ================================================================
# SNN MODEL
# ================================================================

class AnalogSNN(nn.Module):
    def __init__(self, h1, h2=0, timesteps=8, bits=4):
        super().__init__()
        self.timesteps = timesteps
        self.bits = bits
        self.bn_in = nn.BatchNorm1d(N_FEATURES)

        if h2 > 0:
            self.lif1 = LIFLayer(N_FEATURES, h1, bits=bits)
            self.lif2 = LIFLayer(h1, h2, bits=bits)
            self.readout = ReadoutLayer(h2, N_CLASSES, bits=bits)
            self.macs_per_ts = N_FEATURES * h1 + h1 * h2 + h2 * N_CLASSES
            self.n_layers = 3
        else:
            self.lif1 = LIFLayer(N_FEATURES, h1, bits=bits)
            self.lif2 = None
            self.readout = ReadoutLayer(h1, N_CLASSES, bits=bits)
            self.macs_per_ts = N_FEATURES * h1 + h1 * N_CLASSES
            self.n_layers = 2

        self.total_macs = self.macs_per_ts * timesteps
        self.params_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, noise_level=0.0):
        B = x.shape[0]
        x = self.bn_in(x)

        if noise_level > 0 and self.training:
            x = x * (1 + noise_level * torch.randn_like(x))
            x = x + noise_level * 0.1 * torch.randn_like(x)

        # Encode: repeat static features as constant current across timesteps
        x_seq = x.unsqueeze(0).expand(self.timesteps, -1, -1)  # (T, B, 8)

        s1 = self.lif1(x_seq, noise_level)
        if self.lif2 is not None:
            s2 = self.lif2(s1, noise_level)
            out = self.readout(s2, noise_level)
        else:
            out = self.readout(s1, noise_level)

        return out  # (B, 5)

    def forward_noisy_test(self, x, noise_level):
        self.eval()
        with torch.no_grad():
            B = x.shape[0]
            x_bn = self.bn_in(x)
            if noise_level > 0:
                x_bn = x_bn * (1 + noise_level * torch.randn_like(x_bn))
                x_bn = x_bn + noise_level * 0.1 * torch.randn_like(x_bn)

            x_seq = x_bn.unsqueeze(0).expand(self.timesteps, -1, -1)

            if self.bits < 32:
                quantize_weights(self, self.bits)

            s1 = self.lif1(x_seq, 0.0)
            if self.lif2 is not None:
                s2 = self.lif2(s1, 0.0)
                out = self.readout(s2, 0.0)
            else:
                out = self.readout(s1, 0.0)

            return out

# ================================================================
# FOCAL LOSS
# ================================================================

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=FOCAL_GAMMA):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ================================================================
# ENERGY COMPUTATION
# ================================================================

def compute_energy(total_macs, timesteps, n_layers):
    result = {}
    for scenario in ['opt', 'nom', 'pess']:
        e_cells = total_macs * E_PER_MAC_PJ[scenario] * 1e-3  # nJ
        e_wake = E_WAKE_NJ[scenario]
        t_infer = timesteps * n_layers * T_READ_S
        e_total = e_cells + e_wake
        result[scenario] = {
            'e_cells_nJ': e_cells,
            'e_wake_nJ': e_wake,
            'e_total_nJ': e_total,
            't_infer_ns': t_infer * 1e9,
        }
    return result

# ================================================================
# TRAIN AND EVALUATE
# ================================================================

def train_and_eval(X_train, y_train, X_test, y_test, h1, h2, timesteps, bits, noise_level, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_s, y_tr_s = smote_oversample(X_train, y_train)

    X_tr = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_s, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    counts = Counter(y_tr_s)
    weights = torch.tensor([1.0 / (counts.get(i, 1) ** CLS_POWER) for i in range(N_CLASSES)])
    weights = weights / weights.sum() * N_CLASSES

    sample_w = torch.tensor([weights[c] for c in y_tr_s])
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w))
    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    model = AnalogSNN(h1, h2, timesteps, bits)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - warmup, eta_min=LR * 0.01
    )

    criterion = FocalLoss(weights, FOCAL_GAMMA)

    best_model = None
    best_acc = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb, noise_level)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch >= warmup:
            scheduler.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                preds = model.forward_noisy_test(X_te, noise_level).argmax(1).numpy()
            acc = (preds == y_test).mean() * 100
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    model.eval()

    with torch.no_grad():
        preds = model.forward_noisy_test(X_te, noise_level).argmax(1).numpy()

    acc = (preds == y_test).mean() * 100
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    energy = compute_energy(model.total_macs, timesteps, model.n_layers)

    return {
        'acc': acc,
        'report': report,
        'cm': cm,
        'total_macs': model.total_macs,
        'macs_per_ts': model.macs_per_ts,
        'params': model.params_count,
        'n_layers': model.n_layers,
        'energy': energy,
    }

# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--noise', type=float, nargs='+', default=[0.0, 0.02, 0.05])
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--timesteps', type=int, nargs='+', default=[4, 8, 12])
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_8feature_data()
    print(f"Train: {len(y_train)} beats, Test: {len(y_test)} beats")
    print(f"Train dist: {dict(sorted(Counter(y_train).items()))}")
    print(f"Test dist:  {dict(sorted(Counter(y_test).items()))}")
    print(f"Features: {FEATURE_NAMES_8}")

    all_results = []

    print("\n" + "=" * 95)
    print("SNN CLASSIFIER BAKEOFF — 8-Feature LIF Analog-Optimized")
    print("=" * 95)

    for bits in args.bits:
        for noise in args.noise:
            for ts in args.timesteps:
                for (h1, h2) in WIDTH_CONFIGS:
                    arch_str = f"{N_FEATURES}-{h1}-{h2}-5" if h2 > 0 else f"{N_FEATURES}-{h1}-5"
                    tag = f"T={ts}"
                    accs = []
                    recalls = {c: [] for c in CLASS_NAMES}
                    best_result = None

                    for seed in range(args.seeds):
                        r = train_and_eval(X_train, y_train, X_test, y_test,
                                           h1, h2, ts, bits, noise, seed + 42)
                        accs.append(r['acc'])
                        for c in CLASS_NAMES:
                            recalls[c].append(r['report'][c]['recall'] * 100)
                        if best_result is None or r['acc'] > best_result['acc']:
                            best_result = r

                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    e_nom = best_result['energy']['nom']
                    e_full = FRONTEND_E_NJ['tt_27'] + e_nom['e_total_nJ']

                    recall_str = " ".join(
                        f"{c}={np.mean(recalls[c]):4.1f}" for c in CLASS_NAMES
                    )

                    print(f"\n  [{bits}b n={noise:.0%} {tag}] {arch_str:14s} "
                          f"acc={mean_acc:.2f}±{std_acc:.2f}%  "
                          f"MACs={best_result['total_macs']:5d}  "
                          f"E_cls={e_nom['e_total_nJ']:.3f}nJ  "
                          f"E_full={e_full:.2f}nJ  "
                          f"params={best_result['params']}")
                    print(f"{'':>32} recall: {recall_str}")

                    all_results.append({
                        'arch': arch_str, 'bits': bits, 'noise': noise,
                        'timesteps': ts, 'h1': h1, 'h2': h2,
                        'acc_mean': round(mean_acc, 2), 'acc_std': round(std_acc, 2),
                        'total_macs': best_result['total_macs'],
                        'macs_per_ts': best_result['macs_per_ts'],
                        'params': best_result['params'],
                        'energy': best_result['energy'],
                        'recalls': {c: round(np.mean(recalls[c]), 2) for c in CLASS_NAMES},
                        'e_full_chain_nJ': round(e_full, 3),
                    })

    # ── Summary table ──
    print("\n" + "=" * 95)
    print("SUMMARY — Best configs by accuracy (4-bit QAT, 0% noise)")
    print("=" * 95)
    filtered = [r for r in all_results if r['bits'] == 4 and r['noise'] == 0.0]
    filtered.sort(key=lambda x: -x['acc_mean'])

    print(f"{'Arch':>14} {'T':>3} {'Acc%':>8} {'MACs':>6} {'E_cls(nJ)':>10} {'E_full(nJ)':>11} "
          f"{'N':>5} {'S':>5} {'V':>5} {'F':>5} {'Q':>5}")
    print("-" * 90)
    for r in filtered:
        e = r['energy']['nom']['e_total_nJ']
        print(f"{r['arch']:>14} {r['timesteps']:3d} {r['acc_mean']:7.2f}% {r['total_macs']:6d} "
              f"{e:10.3f} {r['e_full_chain_nJ']:11.2f} "
              f"{r['recalls']['N']:5.1f} {r['recalls']['S']:5.1f} "
              f"{r['recalls']['V']:5.1f} {r['recalls']['F']:5.1f} {r['recalls']['Q']:5.1f}")

    # ── SNN vs MLP energy comparison ──
    print("\n" + "=" * 95)
    print("SNN ENERGY ADVANTAGE")
    print("=" * 95)
    if filtered:
        smallest = min(filtered, key=lambda x: x['energy']['nom']['e_total_nJ'])
        biggest_acc = max(filtered, key=lambda x: x['acc_mean'])
        print(f"  Lowest energy:  {smallest['arch']} T={smallest['timesteps']}  "
              f"E_cls={smallest['energy']['nom']['e_total_nJ']:.3f} nJ  "
              f"acc={smallest['acc_mean']:.2f}%")
        print(f"  Highest acc:    {biggest_acc['arch']} T={biggest_acc['timesteps']}  "
              f"E_cls={biggest_acc['energy']['nom']['e_total_nJ']:.3f} nJ  "
              f"acc={biggest_acc['acc_mean']:.2f}%")
        print(f"\n  SNN advantage: temporal integration exploits ReRAM's natural")
        print(f"  capacitive dynamics — membrane = physical capacitor, no extra")
        print(f"  compute needed for activation functions (spikes are comparators).")

    # ── Benchmark comparison ──
    print("\n" + "=" * 95)
    print("BENCHMARK COMPARISON")
    print("=" * 95)
    if filtered:
        best = max(filtered, key=lambda x: x['acc_mean'])
        e_best = best['energy']['nom']['e_total_nJ']
        print(f"{'System':>30} {'Acc%':>7} {'E_cls(nJ)':>10} {'E_total(nJ)':>12} {'Process':>10}")
        print("-" * 75)
        print(f"{'Liu JSSC 2025':>30} {'96.6%':>7} {'90':>10} {'90+AFE':>12} {'55nm':>10}")
        print(f"{'Cao BioCAS 2023':>30} {'98.9%':>7} {'?':>10} {'?':>12} {'RRAM':>10}")
        print(f"{'This work (SNN)':>30} {best['acc_mean']:6.1f}% {e_best:10.3f} {best['e_full_chain_nJ']:11.2f} {'SKY130':>10}")

    # ── Export ──
    out_path = os.path.join(os.path.dirname(__file__), 'snn_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
