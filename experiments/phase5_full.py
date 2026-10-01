"""
PHASE 5 SCRIPT 4: Full Combined + SWA + AdaBN (Maximum Accuracy Push)
======================================================================
Takes the best combined architecture and adds:

  1. Everything from Script 3:
     - 18 features (16 base + 2 interactions)
     - 12 timesteps, per-layer beta, lateral inhibition
     - Hierarchical 2-stage output

  2. SWA (Stochastic Weight Averaging)
     Rationale: Averages weights from last 50 epochs (ep 150-200, every 5)
     to find wider minimum that generalizes better to DS2.

  3. AdaBN (Adaptive Batch Normalization) — inter-patient only
     Rationale: Per-patient BN adaptation at test time. Each DS2 patient
     gets BN stats recomputed on their own data. Addresses the 0.4-0.6σ
     morphology feature shift between DS1→DS2 patients.

Tests:
  A: Combined + SWA (inter + intra)
  B: Combined + SWA + AdaBN (inter only — AdaBN is for domain shift)
  C: Combined + AdaBN only, no SWA (inter only — ablation)

Run: python experiments/phase5_full.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.data_proven import (
    extract_features, DS1, DS2, ALL_RECORDS, NUM_CLASSES, BATCH_SIZE,
    smote_oversample
)
from collections import Counter
from sklearn.model_selection import train_test_split

torch.set_num_threads(2)
t0 = time.time()
NUM_STEPS = 12
NUM_EPOCHS = 200
SWA_START = 150
SWA_FREQ = 5
FRONTEND_NJ = 13.7


def add_interaction_features(features):
    rr_ratio = features[:, 2]
    templ_corr_L0 = features[:, 14]
    rr_asymmetry = features[:, 3]
    rel_area_L0 = features[:, 12]
    prematurity = (1.0 - np.clip(rr_ratio, 0, 2)) * np.clip(templ_corr_L0, 0, 1)
    timing_morph = np.clip(np.abs(1.0 - rr_asymmetry), 0, 3) * np.clip(np.abs(1.0 - rel_area_L0), 0, 3)
    return np.column_stack([features, prematurity, timing_morph])


def quantize(x, bits=4):
    qmin, qmax = -(2**(bits-1)), 2**(bits-1)-1
    s = torch.clamp((x.max()-x.min())/(qmax-qmin), min=1e-8)
    return torch.clamp(torch.round(x/s), qmin, qmax)*s


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma, self.weight = gamma, weight
    def forward(self, x, t):
        ce = nn.functional.cross_entropy(x, t, weight=self.weight, reduction='none')
        return (((1-torch.exp(-ce))**self.gamma)*ce).mean()


class FullSNN(nn.Module):
    """Full combined: per-layer beta + lateral inhibition + hierarchical output."""
    def __init__(self, n_in, h1=48, h2=24, beta1=0.85, beta2=0.95, top_k=24):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h1)
        self.drop1 = nn.Dropout(0.05)
        self.rlif1 = snn.RLeaky(beta=beta1, linear_features=h1,
                                learn_beta=True, learn_threshold=True)
        self.fc_mid = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(0.05)
        self.rlif2 = snn.RLeaky(beta=beta2, linear_features=h2,
                                learn_beta=True, learn_threshold=True)
        self.fc_binary = nn.Linear(h2, 2)
        self.fc_subtype = nn.Linear(h2, 4)
        self.lif_binary = snn.Leaky(beta=beta2, learn_beta=True, learn_threshold=True)
        self.lif_subtype = snn.Leaky(beta=beta2, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2
        self.top_k = top_k

    def lateral_inhibition(self, spk, k):
        vals = spk.abs()
        topk_vals, _ = vals.topk(k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        mask = (vals >= threshold).float()
        if self.training:
            return spk * (mask + spk.detach() * (1 - mask) * 0.1
                          - (spk.detach() * (1 - mask) * 0.1).detach())
        return spk * mask

    def forward(self, x, num_steps):
        s1, m1 = self.rlif1.init_rleaky()
        s2, m2 = self.rlif2.init_rleaky()
        mb = self.lif_binary.init_leaky()
        ms = self.lif_subtype.init_leaky()
        spk_bin_rec, mem_bin_rec = [], []
        spk_sub_rec, mem_sub_rec = [], []
        spk1_rec = []
        x = self.bn(x)
        fc1 = self.drop1(self.fc1(x))
        for _ in range(num_steps):
            s1, m1 = self.rlif1(fc1, s1, m1)
            s1_out = self.lateral_inhibition(s1, self.top_k)
            spk1_rec.append(s1_out)
            s2, m2 = self.rlif2(self.drop2(self.fc_mid(s1_out)), s2, m2)
            sb, mb = self.lif_binary(self.fc_binary(s2), mb)
            ss, ms = self.lif_subtype(self.fc_subtype(s2), ms)
            spk_bin_rec.append(sb); mem_bin_rec.append(mb)
            spk_sub_rec.append(ss); mem_sub_rec.append(ms)
        return (torch.stack(spk_bin_rec), torch.stack(mem_bin_rec),
                torch.stack(spk_sub_rec), torch.stack(mem_sub_rec),
                torch.stack(spk1_rec))


def load_data_raw(split):
    """Load raw data with interaction features, return arrays + normalization stats."""
    if split == 'inter':
        train_labels, train_features = extract_features(DS1)
        test_labels, test_features = extract_features(DS2)
    else:
        all_labels, all_features = extract_features(ALL_RECORDS)
        train_idx, test_idx = train_test_split(
            np.arange(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels)
        train_features, test_features = all_features[train_idx], all_features[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

    train_features = add_interaction_features(train_features)
    test_features = add_interaction_features(test_features)
    n_features = train_features.shape[1]
    print(f"\n  {split}-patient | {n_features} features | train={len(train_labels)} test={len(test_labels)}")

    train_features_aug, train_labels_aug = smote_oversample(train_features, train_labels, 0.33)
    mu, sd = train_features_aug.mean(0), train_features_aug.std(0)
    train_norm = (train_features_aug - mu) / (sd + 1e-8)
    test_norm = (test_features - mu) / (sd + 1e-8)

    return (train_norm, train_labels_aug, test_norm, test_labels,
            test_features, test_labels, mu, sd, n_features)


def make_loaders(train_features, train_labels, test_features, test_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    cls_power = 0.65
    cw = 1.0 / (tr_counts.astype(np.float64) ** cls_power)
    cw = cw / cw.sum() * NUM_CLASSES
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    sw = 1.0 / (tr_counts ** cls_power)
    sample_w = [sw[l] for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        DS(train_features, train_labels), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        DS(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, cw_tensor, device


def compute_energy(n_features, h1, h2, num_steps, fire, top_k):
    effective_fire = fire * (top_k / h1)
    total_macs = (n_features*h1 + h1*h1*num_steps*effective_fire +
                  h1*h2*num_steps*effective_fire + h2*h2*num_steps*effective_fire +
                  h2*6*num_steps*effective_fire)
    return total_macs * 2.0 / 1000


def train_with_swa(train_loader, test_loader, cw_tensor, device, n_features, label):
    """Train hierarchical combined model with SWA."""
    torch.manual_seed(42); np.random.seed(42)
    model = FullSNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] Params: {n_params:,} | SWA from ep {SWA_START}")

    bin_weight = torch.tensor([0.2, 1.0], dtype=torch.float32).to(device)
    sub_weight = torch.tensor([1.0, 0.3, 1.2, 1.0], dtype=torch.float32).to(device)
    focal_bin = FocalLoss(weight=bin_weight, gamma=2.0)
    focal_sub = FocalLoss(weight=sub_weight, gamma=2.0)

    # SWA: collect weight snapshots
    swa_states = []

    for epoch in range(NUM_EPOCHS):
        if epoch < warmup:
            for pg in opt.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup
        model.train()
        ep_loss, ep_rate, batches = 0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            bin_targets = (targets > 0).long()
            abnormal_mask = targets > 0
            sub_targets = targets[abnormal_mask] - 1

            saved = []
            with torch.no_grad():
                for p in model.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize(p.data))

            spk_bin, mem_bin, spk_sub, mem_sub, spk_h = model(data, NUM_STEPS)
            ce_bin = sum(focal_bin(mem_bin[s], bin_targets) for s in range(NUM_STEPS))
            ce_sub = torch.tensor(0.0, device=device)
            if abnormal_mask.sum() > 0:
                for s in range(NUM_STEPS):
                    ce_sub = ce_sub + focal_sub(mem_sub[s][abnormal_mask], sub_targets)
            fire = spk_h.mean()
            loss = 0.5 * ce_bin + 0.5 * ce_sub + 1.0 * torch.clamp(fire - 0.15, min=0)

            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                for p, s in zip(model.parameters(), saved): p.data.copy_(s)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += (ce_bin.item() + ce_sub.item()); ep_rate += fire.item(); batches += 1

        if epoch >= warmup: sched.step()
        avg_ce = ep_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce; best_state = copy.deepcopy(model.state_dict())

        # SWA: collect snapshots
        if epoch >= SWA_START and (epoch - SWA_START) % SWA_FREQ == 0:
            swa_states.append(copy.deepcopy(model.state_dict()))
            print(f"    SWA snapshot at epoch {epoch+1} ({len(swa_states)} collected)")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | CE {avg_ce:.3f} | fire {ep_rate/batches:.3f} | {time.time()-t0:.0f}s")

    # Apply SWA: average all collected snapshots
    if swa_states:
        avg_state = {}
        for key in swa_states[0]:
            avg_state[key] = sum(s[key].float() for s in swa_states) / len(swa_states)
            avg_state[key] = avg_state[key].to(swa_states[0][key].dtype)
        model.load_state_dict(avg_state)
        print(f"  SWA: averaged {len(swa_states)} snapshots")
    else:
        model.load_state_dict(best_state)

    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))

    # Update BN stats with SWA weights
    model.train()
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            model(data, NUM_STEPS)
    model.eval()

    return model


def evaluate_model(model, test_loader, device, n_features, label):
    model.eval()
    all_p, all_t, tot_spk, tot_pos = [], [], 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spk_bin, _, spk_sub, _, spk_h = model(data, NUM_STEPS)
            bin_counts = spk_bin.sum(dim=0)
            _, bin_pred = bin_counts.max(1)
            sub_counts = spk_sub.sum(dim=0)
            _, sub_pred = sub_counts.max(1)
            final_pred = torch.where(bin_pred == 0,
                                     torch.zeros_like(sub_pred),
                                     sub_pred + 1)
            tot_spk += spk_h.sum().item(); tot_pos += spk_h.numel()
            all_p.extend(final_pred.cpu().numpy()); all_t.extend(targets.cpu().numpy())

    fire = tot_spk / tot_pos if tot_pos > 0 else 0
    acc = np.mean(np.array(all_p) == np.array(all_t)) * 100
    cls_nJ = compute_energy(n_features, 48, 24, NUM_STEPS, fire, 24)
    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% | Fire: {fire:.3f} | Energy: {cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_t, all_p, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


def evaluate_with_adabn(model, device, n_features, test_features_raw, test_labels_raw,
                        mu, sd, label):
    """Per-patient AdaBN: reset BN stats per DS2 record, forward pass in train mode
    to update BN, then evaluate in eval mode."""
    # Group test data by DS2 record
    # We need per-record features. Re-extract per record.
    from experiments.data_proven import DS2 as ds2_records
    all_preds, all_targets = [], []

    for rec_id in ds2_records:
        rec_labels, rec_features = extract_features([rec_id])
        if len(rec_labels) == 0:
            continue
        rec_features = add_interaction_features(rec_features)
        rec_norm = (rec_features - mu) / (sd + 1e-8)
        rec_tensor = torch.tensor(rec_norm, dtype=torch.float32).to(device)
        rec_targets = torch.tensor(rec_labels, dtype=torch.long).to(device)

        # Reset BN running stats
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                m.num_batches_tracked.zero_()

        # Forward pass in train mode to update BN stats (no gradient)
        model.train()
        with torch.no_grad():
            # Multiple passes for better BN estimation
            for _ in range(3):
                model(rec_tensor, NUM_STEPS)

        # Evaluate in eval mode
        model.eval()
        with torch.no_grad():
            spk_bin, _, spk_sub, _, _ = model(rec_tensor, NUM_STEPS)
            bin_counts = spk_bin.sum(dim=0)
            _, bin_pred = bin_counts.max(1)
            sub_counts = spk_sub.sum(dim=0)
            _, sub_pred = sub_counts.max(1)
            final_pred = torch.where(bin_pred == 0,
                                     torch.zeros_like(sub_pred),
                                     sub_pred + 1)

        all_preds.extend(final_pred.cpu().numpy())
        all_targets.extend(rec_targets.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% (with per-patient AdaBN)")
    print(f"  {'='*60}")
    print(classification_report(all_targets, all_preds, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_targets, all_preds, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc


if __name__ == '__main__':
    results = {}

    # =====================================================
    # INTER-PATIENT
    # =====================================================
    print(f"\n{'='*60}")
    print(f"LOADING INTER-PATIENT DATA")
    (train_norm, train_labels, test_norm, test_labels,
     test_raw, test_labels_raw, mu, sd, n_feat) = load_data_raw('inter')
    train_loader, test_loader, cw_t, device = make_loaders(
        train_norm, train_labels, test_norm, test_labels)

    # A: Combined + SWA (inter)
    print(f"\n{'='*60}")
    print(f"VARIANT A: Combined + SWA (inter)")
    model_swa = train_with_swa(train_loader, test_loader, cw_t, device, n_feat, "swa-inter")
    acc, nJ = evaluate_model(model_swa, test_loader, device, n_feat, "swa-inter")
    results["swa-inter"] = (acc, nJ)

    # B: Combined + SWA + AdaBN (inter)
    print(f"\n{'='*60}")
    print(f"VARIANT B: Combined + SWA + AdaBN (inter)")
    model_adabn = copy.deepcopy(model_swa)
    acc_adabn = evaluate_with_adabn(model_adabn, device, n_feat, test_raw, test_labels_raw,
                                     mu, sd, "swa+adabn-inter")
    results["swa+adabn-inter"] = (acc_adabn, nJ)

    # C: Combined + AdaBN only (retrain without SWA)
    print(f"\n{'='*60}")
    print(f"VARIANT C: Combined + AdaBN only (inter)")
    # Train without SWA for ablation
    torch.manual_seed(42); np.random.seed(42)
    model_noswa = FullSNN(n_feat).to(device)
    opt = torch.optim.Adam(model_noswa.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    bin_weight = torch.tensor([0.2, 1.0], dtype=torch.float32).to(device)
    sub_weight = torch.tensor([1.0, 0.3, 1.2, 1.0], dtype=torch.float32).to(device)
    focal_bin = FocalLoss(weight=bin_weight, gamma=2.0)
    focal_sub = FocalLoss(weight=sub_weight, gamma=2.0)
    print(f"\n  [adabn-only-inter] Training without SWA...")

    for epoch in range(NUM_EPOCHS):
        if epoch < warmup:
            for pg in opt.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup
        model_noswa.train()
        ep_loss, batches = 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            bin_targets = (targets > 0).long()
            abnormal_mask = targets > 0
            sub_targets = targets[abnormal_mask] - 1
            saved = []
            with torch.no_grad():
                for p in model_noswa.parameters():
                    saved.append(p.data.clone()); p.data.copy_(quantize(p.data))
            spk_bin, mem_bin, spk_sub, mem_sub, spk_h = model_noswa(data, NUM_STEPS)
            ce_bin = sum(focal_bin(mem_bin[s], bin_targets) for s in range(NUM_STEPS))
            ce_sub = torch.tensor(0.0, device=device)
            if abnormal_mask.sum() > 0:
                for s in range(NUM_STEPS):
                    ce_sub = ce_sub + focal_sub(mem_sub[s][abnormal_mask], sub_targets)
            fire = spk_h.mean()
            loss = 0.5*ce_bin + 0.5*ce_sub + 1.0*torch.clamp(fire-0.15, min=0)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                for p, s in zip(model_noswa.parameters(), saved): p.data.copy_(s)
            nn.utils.clip_grad_norm_(model_noswa.parameters(), 1.0)
            opt.step()
            ep_loss += (ce_bin.item()+ce_sub.item()); batches += 1
        if epoch >= warmup: sched.step()
        avg_ce = ep_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce; best_state = copy.deepcopy(model_noswa.state_dict())
        if (epoch+1) % 50 == 0:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | CE {avg_ce:.3f} | {time.time()-t0:.0f}s")

    model_noswa.load_state_dict(best_state)
    with torch.no_grad():
        for p in model_noswa.parameters(): p.data.copy_(quantize(p.data))
    acc_noswa = evaluate_with_adabn(model_noswa, device, n_feat, test_raw, test_labels_raw,
                                     mu, sd, "adabn-only-inter")
    results["adabn-only-inter"] = (acc_noswa, nJ)

    # =====================================================
    # INTRA-PATIENT (SWA only, no AdaBN needed)
    # =====================================================
    print(f"\n{'='*60}")
    print(f"LOADING INTRA-PATIENT DATA")
    (train_norm, train_labels, test_norm, test_labels,
     _, _, _, _, n_feat) = load_data_raw('intra')
    train_loader, test_loader, cw_t, device = make_loaders(
        train_norm, train_labels, test_norm, test_labels)

    print(f"\n{'='*60}")
    print(f"VARIANT A: Combined + SWA (intra)")
    model_swa_intra = train_with_swa(train_loader, test_loader, cw_t, device, n_feat, "swa-intra")
    acc, nJ = evaluate_model(model_swa_intra, test_loader, device, n_feat, "swa-intra")
    results["swa-intra"] = (acc, nJ)

    # =====================================================
    # SUMMARY
    # =====================================================
    print(f"\n{'='*60}")
    print(f"PHASE 5 SCRIPT 4: FULL COMBINED + SWA + AdaBN SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':>30} {'Accuracy':>10} {'Energy':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'baseline-inter':>30} {'91.94':>9}% {'42.9':>8}nJ")
    for label, (acc, nJ) in results.items():
        nJ_str = f"{nJ:.1f}nJ" if nJ else "N/A"
        print(f"  {label:>30} {acc:>9.2f}% {nJ_str:>10}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
