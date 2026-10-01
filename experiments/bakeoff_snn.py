"""
ARCHITECTURE BAKEOFF — SNN
===========================
Controlled comparison: Recurrent SNN with lateral inhibition.
Tests width sweep, timestep sweep, with per-class metrics,
full-chain energy, feature noise injection, and multi-seed eval.

Same controlled variables as bakeoff_mlp.py:
  - 16 features (proven cardiologist set)
  - AAMI 5-class, de Chazal DS1/DS2 inter-patient
  - SMOTE 0.33, focal loss gamma=2.0, 4-bit QAT
  - 200 epochs, batch=128, Adam lr=1e-3, weight_decay=1e-4
  - Cosine annealing with 5-epoch warmup
  - Sparsity penalty lambda=1.0, target_fire=0.15
  - cls_power=0.65, dropout=0.05

Run: python experiments/bakeoff_snn.py [--seeds 5] [--noise 0.0] [--bits 4]
"""
import sys, os, time, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.data_proven import extract_features, DS1, DS2, NUM_CLASSES, BATCH_SIZE, smote_oversample
from collections import Counter
import json

torch.set_num_threads(2)

# ── Constants ──
NUM_EPOCHS = 200
FEATURE_NAMES = ['pre_rr','post_rr','rr_ratio','rr_asymmetry','compensatory_ratio',
                 'rr_std_10','qrs_width_L0','qrs_width_L1','qrs_area_L0','qrs_area_L1',
                 'max_slope_L0','max_slope_L1','rel_area_L0','rel_area_L1',
                 'templ_corr_L0','templ_corr_L1']
CLASS_NAMES = ['N','S','V','F','Q']

# Energy model parameters — PDK-VALIDATED (sky130_fd_pr_reram__reram_cell.va)
# E_per_MAC from sinh(V/V_ref) I-V model + NMOS R_on=1010 ohms, 4-cell column MAC
V_READ = 0.15   # validated: all voltages 0.05-0.30V are read-disturb safe
T_READ = 200e-9  # validated: 200ns read pulse
G_LRS_OPT = 30e-6    # Tfilament=4.571nm
G_LRS_NOM = 75e-6    # Tfilament=4.823nm
G_LRS_PESS = 99.3e-6  # Tfilament=4.900nm (PDK LRS max, 120uS not achievable)
E_PER_MAC_PJ_OPT = 0.5339   # pJ, PDK-validated
E_PER_MAC_PJ_NOM = 1.2679   # pJ, PDK-validated
E_PER_MAC_PJ_PESS = 1.6489  # pJ, PDK-validated

# Periphery energy (input buffers, neurons, WTA, bias generation)
# These are estimates pending circuit design.
E_WAKE_NJ_OPT = 0.5
E_WAKE_NJ_NOM = 1.5
E_WAKE_NJ_PESS = 3.0
P_PERIPH_NW_OPT = 20.0
P_PERIPH_NW_NOM = 50.0
P_PERIPH_NW_PESS = 100.0
T_INFER_WINDOW_S = 5e-6  # SNN: ~25 crossbar activations * 200ns = 5us

TIMING_POWER_NW = 5.0
TIMING_DURATION_S = 0.833
MORPH_POWER_NW = 95.0
MORPH_DURATION_S = 0.100
AFE_POWER_NW = 80.0
AFE_DURATION_S = 0.833
FRONTEND_NJ = TIMING_POWER_NW * TIMING_DURATION_S + MORPH_POWER_NW * MORPH_DURATION_S
AFE_NJ = AFE_POWER_NW * AFE_DURATION_S

# Single-lead feature indices (L0 only: timing + L0 morphology)
SINGLE_LEAD_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14]
# Single-lead frontend: 1 IA instead of 2, half morphology circuits
FRONTEND_NJ_SINGLE = TIMING_POWER_NW * TIMING_DURATION_S + (MORPH_POWER_NW / 2) * MORPH_DURATION_S

# Width configurations to sweep (80-40 excluded: fails <100 nJ without LI)
WIDTH_CONFIGS = [(10, 5), (20, 10), (40, 20), (48, 24)]
STEP_CONFIGS = [8, 12, 16]


def quantize(x, bits=4):
    qmin, qmax = -(2**(bits-1)), 2**(bits-1)-1
    s = torch.clamp((x.max() - x.min()) / (qmax - qmin), min=1e-8)
    return torch.clamp(torch.round(x / s), qmin, qmax) * s


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, x, t):
        ce = nn.functional.cross_entropy(x, t, weight=self.weight, reduction='none')
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class BakeoffSNN(nn.Module):
    def __init__(self, n_in, h1, h2, n_out=5, top_k=None):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h1)
        self.drop1 = nn.Dropout(0.05)
        self.rlif1 = snn.RLeaky(beta=0.9, linear_features=h1,
                                learn_beta=True, learn_threshold=True)
        self.fc_mid = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(0.05)
        self.rlif2 = snn.RLeaky(beta=0.9, linear_features=h2,
                                learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(h2, n_out)
        self.lif_out = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2
        self.top_k = top_k if top_k else h1

    def forward(self, x, num_steps):
        s1, m1 = self.rlif1.init_rleaky()
        s2, m2 = self.rlif2.init_rleaky()
        mo = self.lif_out.init_leaky()
        spk_out, spk1_all = [], []
        x = self.bn(x)
        fc1 = self.drop1(self.fc1(x))
        for _ in range(num_steps):
            s1, m1 = self.rlif1(fc1, s1, m1)
            if self.top_k < self.h1:
                vals = s1.abs()
                topk_vals, _ = vals.topk(self.top_k, dim=-1)
                threshold = topk_vals[..., -1:].detach()
                s1_out = s1 * (vals >= threshold).float()
            else:
                s1_out = s1
            spk1_all.append(s1_out)
            s2, m2 = self.rlif2(self.drop2(self.fc_mid(s1_out)), s2, m2)
            so, mo = self.lif_out(self.fc2(s2), mo)
            spk_out.append(so)
        return torch.stack(spk_out), torch.stack(spk1_all)


def compute_energy_snn(n_features, h1, h2, num_steps, fire_rate, top_k, single_lead=False):
    eff_fire = fire_rate * (top_k / h1)
    fc1_macs = n_features * h1
    rec1_macs = h1 * h1 * num_steps * eff_fire
    mid_macs = h1 * h2 * num_steps * eff_fire
    rec2_macs = h2 * h2 * num_steps * eff_fire
    out_macs = h2 * 5 * num_steps * eff_fire
    total_macs = fc1_macs + rec1_macs + mid_macs + rec2_macs + out_macs
    t_window = T_INFER_WINDOW_S * (num_steps / 12)
    frontend = FRONTEND_NJ_SINGLE if single_lead else FRONTEND_NJ
    results = {}
    for label, e_mac, e_wake, p_periph in [
        ('opt',  E_PER_MAC_PJ_OPT,  E_WAKE_NJ_OPT,  P_PERIPH_NW_OPT),
        ('nom',  E_PER_MAC_PJ_NOM,  E_WAKE_NJ_NOM,  P_PERIPH_NW_NOM),
        ('pess', E_PER_MAC_PJ_PESS, E_WAKE_NJ_PESS, P_PERIPH_NW_PESS),
    ]:
        e_cells = total_macs * e_mac / 1000
        e_periph = p_periph * t_window * 1e9 / 1e9  # nW * s = nJ
        cls_nJ = e_cells + e_wake + e_periph
        total_nJ = cls_nJ + frontend + AFE_NJ
        results[label] = {
            'e_cells': e_cells, 'e_wake': e_wake, 'e_periph': e_periph,
            'cls_nJ': cls_nJ, 'frontend_nJ': frontend, 'afe_nJ': AFE_NJ,
            'total_nJ': total_nJ, 'e_per_mac_pJ': e_mac,
        }
    return total_macs, results


def load_data(feature_noise=0.0, single_lead=False):
    train_labels, train_features = extract_features(DS1)
    test_labels, test_features = extract_features(DS2)
    if single_lead:
        train_features = train_features[:, SINGLE_LEAD_INDICES]
        test_features = test_features[:, SINGLE_LEAD_INDICES]
        print(f"  SINGLE-LEAD MODE: {len(SINGLE_LEAD_INDICES)} features (L0 only)")
    n_features = train_features.shape[1]
    print(f"  DS1: {len(train_labels)} | DS2: {len(test_labels)} | Features: {n_features}")
    train_features, train_labels = smote_oversample(train_features, train_labels, 0.33)
    mu, sd = train_features.mean(0), train_features.std(0)
    train_features = (train_features - mu) / (sd + 1e-8)
    test_features = (test_features - mu) / (sd + 1e-8)
    if feature_noise > 0:
        print(f"  Feature noise: {feature_noise*100:.1f}% (multiplicative + additive)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    cls_power = 0.65
    cw = 1.0 / (tr_counts.astype(np.float64) ** cls_power)
    cw = cw / cw.sum() * NUM_CLASSES
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l): self.data = torch.tensor(f, dtype=torch.float32); self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    sw = 1.0 / (tr_counts ** cls_power)
    sample_w = [sw[l] for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    train_loader = torch.utils.data.DataLoader(DS(train_features, train_labels), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(DS(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, cw_tensor, device, n_features


def inject_noise(x, noise_level):
    if noise_level <= 0:
        return x
    mult = 1.0 + noise_level * torch.randn_like(x)
    add = noise_level * 0.1 * torch.randn_like(x)
    return x * mult + add


def train_and_eval(h1, h2, num_steps, top_k, train_loader, test_loader,
                   cw_tensor, device, n_features, seed, noise_level=0.0, qat_bits=4,
                   single_lead=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = BakeoffSNN(n_features, h1, h2, top_k=top_k).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup_epochs)
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    best_ce, best_state = float('inf'), None

    for epoch in range(NUM_EPOCHS):
        net.train()
        epoch_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            data = inject_noise(data, noise_level)
            for p in net.parameters():
                if p.dim() >= 2:
                    p.data = quantize(p.data, bits=qat_bits)
            spk_out, spk1 = net(data, num_steps)
            out = spk_out.sum(dim=0)
            loss = loss_fn(out, targets)
            fire = spk1.detach().abs().mean()
            loss = loss + 1.0 * torch.relu(fire - 0.15)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        if epoch >= warmup_epochs:
            sched.step()
        avg_ce = epoch_loss / len(train_loader)
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    net.eval()
    all_p, all_t, tot_spk, tot_pos = [], [], 0, 0
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                p.data = quantize(p.data, bits=qat_bits)
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            data = inject_noise(data, noise_level)
            spk_out, spk1 = net(data, num_steps)
            preds = spk_out.sum(dim=0).argmax(dim=1)
            all_p.extend(preds.cpu().numpy())
            all_t.extend(targets.cpu().numpy())
            tot_spk += spk1.abs().sum().item()
            tot_pos += spk1.numel()
    fire_rate = tot_spk / tot_pos if tot_pos > 0 else 0
    acc = np.mean(np.array(all_p) == np.array(all_t)) * 100
    total_macs, energy = compute_energy_snn(n_features, h1, h2, num_steps, fire_rate, top_k, single_lead)
    cm = confusion_matrix(all_t, all_p, labels=list(range(NUM_CLASSES)))
    per_class_recall = {}
    for i, name in enumerate(CLASS_NAMES):
        row_sum = cm[i].sum()
        per_class_recall[name] = cm[i, i] / row_sum * 100 if row_sum > 0 else 0
    return {
        'acc': acc, 'fire_rate': fire_rate, 'n_params': n_params,
        'total_macs': total_macs, 'qat_bits': qat_bits,
        'energy_opt': energy['opt'], 'energy_nom': energy['nom'], 'energy_pess': energy['pess'],
        'cls_nJ_nom': energy['nom']['cls_nJ'], 'total_nJ_nom': energy['nom']['total_nJ'],
        'recall': per_class_recall, 'cm': cm.tolist(), 'seed': seed,
        'h1': h1, 'h2': h2, 'steps': num_steps, 'top_k': top_k,
        'single_lead': single_lead
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--bits', type=int, nargs='+', default=[4])
    parser.add_argument('--steps', type=int, nargs='+', default=None)
    parser.add_argument('--widths', type=str, default=None)
    parser.add_argument('--single-lead', action='store_true',
                        help='Use only L0 features (11 features, 1 IA)')
    args = parser.parse_args()

    step_list = args.steps if args.steps else STEP_CONFIGS
    if args.widths:
        width_list = [tuple(map(int, w.split(','))) for w in args.widths.split(';')]
    else:
        width_list = WIDTH_CONFIGS
    bit_widths = args.bits
    single_lead = args.single_lead

    t0 = time.time()
    lead_tag = "SINGLE-LEAD (L0)" if single_lead else "DUAL-LEAD"
    print(f"\n{'='*70}")
    print(f"  SNN ARCHITECTURE BAKEOFF ({lead_tag})")
    print(f"  Widths: {width_list}")
    print(f"  Steps: {step_list}")
    print(f"  QAT bits: {bit_widths}")
    print(f"  Seeds: {args.seeds} | Noise: {args.noise*100:.1f}%")
    print(f"  Energy model: V_read={V_READ}V, t_read={T_READ*1e9:.0f}ns")
    print(f"  E/MAC (opt/nom/pess): {E_PER_MAC_PJ_OPT:.3f}/{E_PER_MAC_PJ_NOM:.3f}/{E_PER_MAC_PJ_PESS:.3f} pJ")
    print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    train_loader, test_loader, cw_tensor, device, n_features = load_data(args.noise, single_lead)
    all_results = []

    for bits in bit_widths:
      for h1, h2 in width_list:
        for steps in step_list:
            for use_li in [False, True]:
                top_k = h1 // 2 if use_li else h1
                config_name = f"SNN_{h1}-{h2}_s{steps}_{bits}b" + (f"_LI{top_k}" if use_li else "")
                if single_lead:
                    config_name += "_1L"
                print(f"\n  -- {config_name} --")
                seed_results = []
                for s in range(args.seeds):
                    seed = 42 + s
                    r = train_and_eval(h1, h2, steps, top_k, train_loader, test_loader,
                                       cw_tensor, device, n_features, seed, args.noise, bits,
                                       single_lead)
                    seed_results.append(r)
                    e = r['energy_nom']
                    print(f"    seed={seed}: {r['acc']:.2f}% | fire={r['fire_rate']:.3f} | "
                          f"S={r['recall']['S']:.1f}% V={r['recall']['V']:.1f}% | "
                          f"cls={e['cls_nJ']:.1f}nJ total={e['total_nJ']:.1f}nJ "
                          f"[{r['energy_opt']['total_nJ']:.1f}-{r['energy_pess']['total_nJ']:.1f}]")

                accs = [r['acc'] for r in seed_results]
                s_recs = [r['recall']['S'] for r in seed_results]
                v_recs = [r['recall']['V'] for r in seed_results]
                e_opt = [r['energy_opt']['total_nJ'] for r in seed_results]
                e_nom = [r['energy_nom']['total_nJ'] for r in seed_results]
                e_pess = [r['energy_pess']['total_nJ'] for r in seed_results]

                summary = {
                    'config': config_name, 'type': 'SNN',
                    'h1': h1, 'h2': h2, 'steps': steps, 'top_k': top_k,
                    'use_li': use_li, 'qat_bits': bits, 'single_lead': single_lead,
                    'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
                    'S_recall_mean': np.mean(s_recs), 'S_recall_std': np.std(s_recs),
                    'V_recall_mean': np.mean(v_recs), 'V_recall_std': np.std(v_recs),
                    'energy_opt_mean': np.mean(e_opt),
                    'energy_nom_mean': np.mean(e_nom),
                    'energy_pess_mean': np.mean(e_pess),
                    'n_params': seed_results[0]['n_params'],
                    'noise': args.noise, 'seeds': seed_results
                }
                all_results.append(summary)
                print(f"  => {config_name}: {summary['acc_mean']:.2f}+-{summary['acc_std']:.2f}% | "
                      f"S={summary['S_recall_mean']:.1f}% V={summary['V_recall_mean']:.1f}% | "
                      f"E={summary['energy_nom_mean']:.1f}nJ [{summary['energy_opt_mean']:.1f}-{summary['energy_pess_mean']:.1f}] | "
                      f"params={summary['n_params']}")

    # Final summary table
    print(f"\n{'='*70}")
    print(f"  SNN BAKEOFF SUMMARY ({lead_tag}, noise={args.noise*100:.1f}%)")
    print(f"{'='*70}")
    print(f"  {'Config':<32} {'Acc%':>8} {'S_rec%':>8} {'V_rec%':>8} {'E_nom':>8} {'E_range':>14} {'Params':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*14} {'-'*8}")
    for r in sorted(all_results, key=lambda x: -x['acc_mean']):
        print(f"  {r['config']:<32} {r['acc_mean']:>7.2f}% {r['S_recall_mean']:>7.1f}% "
              f"{r['V_recall_mean']:>7.1f}% {r['energy_nom_mean']:>7.1f} "
              f"[{r['energy_opt_mean']:>5.1f}-{r['energy_pess_mean']:>5.1f}] {r['n_params']:>8}")

    # Benchmark comparison
    print(f"\n  {'='*70}")
    print(f"  BENCHMARK COMPARISON")
    print(f"  {'='*70}")
    print(f"  {'Reference':<36} {'Acc%':>8} {'Energy':>12} {'Scope':>16}")
    print(f"  {'-'*36} {'-'*8} {'-'*12} {'-'*16}")
    print(f"  {'Liu et al., JSSC 2025':<36} {'96.6%':>8} {'90 nJ':>12} {'cls+feat only':>16}")
    print(f"  {'Cao et al., BioCAS 2023':<36} {'98.9%':>8} {'N/A':>12} {'cls only':>16}")
    print(f"  {'Zhang et al., TBCAS 2024':<36} {'98.6%':>8} {'150 nJ':>12} {'cls only':>16}")
    print(f"  {'Chu et al., TBCAS 2022':<36} {'98.2%':>8} {'750 nJ':>12} {'cls only':>16}")
    best = sorted(all_results, key=lambda x: -x['acc_mean'])[0] if all_results else None
    if best:
        print(f"  {'This work (best SNN)':<36} {best['acc_mean']:>7.1f}% "
              f"{best['energy_nom_mean']:>8.1f} nJ {'full chain':>16}")
    print(f"\n  Note: Liu/Cao/Zhang/Chu energy excludes AFE. Our energy includes AFE ({AFE_NJ:.1f} nJ).")
    print(f"  Like-for-like (cls+feat only): subtract {AFE_NJ:.1f} nJ from our total.")

    # Save results
    sl_tag = "_1L" if single_lead else ""
    out_path = os.path.join(os.path.dirname(__file__), f'bakeoff_snn_results_n{int(args.noise*100)}_b{"_".join(map(str, bit_widths))}{sl_tag}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print(f"  Total runtime: {time.time()-t0:.0f}s")
