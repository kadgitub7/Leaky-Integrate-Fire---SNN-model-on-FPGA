"""
PHASE 5 SCRIPT 3: Combined Optimized Architecture
===================================================
Stacks ALL compatible improvements from Scripts 1 & 2:

  1. Feature interactions: 2 pre-computed nonlinear features (18 total)
     - prematurity_index = (1 - rr_ratio) * templ_corr_L0
     - timing_morph_interaction = |1 - rr_asymmetry| * |1 - rel_area_L0|

  2. Reduced timesteps: 20 → 12 (40% energy reduction)

  3. Per-layer beta: beta1=0.85 (longer memory), beta2=0.95 (faster output)

  4. Lateral inhibition: top-k=24 of 48 hidden neurons (WTA sparsity)

  5. Hierarchical 2-stage output: Normal vs Abnormal + 4-class subtype

  6. dropout=0.05 (proven best)

Tests:
  A: All improvements, standard 5-class output
  B: All improvements, hierarchical 2-stage output
  C: All improvements, hierarchical + 17th feature (corr_L0L1)

Both inter and intra patient.
Run: python experiments/phase5_combined.py
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


class CombinedSNN(nn.Module):
    """Combined: per-layer beta + lateral inhibition + standard 5-class output."""
    def __init__(self, n_in, h1=48, h2=24, n_out=5,
                 beta1=0.85, beta2=0.95, top_k=24):
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
        self.fc2 = nn.Linear(h2, n_out)
        self.lif_out = snn.Leaky(beta=beta2, learn_beta=True, learn_threshold=True)
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
        mo = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk1_rec = [], [], []
        x = self.bn(x)
        fc1 = self.drop1(self.fc1(x))
        for _ in range(num_steps):
            s1, m1 = self.rlif1(fc1, s1, m1)
            s1_out = self.lateral_inhibition(s1, self.top_k)
            spk1_rec.append(s1_out)
            s2, m2 = self.rlif2(self.drop2(self.fc_mid(s1_out)), s2, m2)
            so, mo = self.lif_out(self.fc2(s2), mo)
            spk_out_rec.append(so)
            mem_out_rec.append(mo)
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), torch.stack(spk1_rec)


class CombinedHierarchicalSNN(nn.Module):
    """Combined: per-layer beta + lateral inhibition + hierarchical 2-stage output."""
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


def load_data(split, use_interactions=False, include_cross_lead=False):
    if split == 'inter':
        train_labels, train_features = extract_features(DS1, include_cross_lead)
        test_labels, test_features = extract_features(DS2, include_cross_lead)
    else:
        all_labels, all_features = extract_features(ALL_RECORDS, include_cross_lead)
        train_idx, test_idx = train_test_split(
            np.arange(len(all_labels)), test_size=0.2, random_state=42, stratify=all_labels)
        train_features, test_features = all_features[train_idx], all_features[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

    if use_interactions:
        train_features = add_interaction_features(train_features)
        test_features = add_interaction_features(test_features)

    n_features = train_features.shape[1]
    print(f"\n  {split}-patient | {n_features} features | train={len(train_labels)} test={len(test_labels)}")

    train_features, train_labels = smote_oversample(train_features, train_labels, 0.33)
    mu, sd = train_features.mean(0), train_features.std(0)
    train_features = (train_features - mu) / (sd + 1e-8)
    test_features = (test_features - mu) / (sd + 1e-8)

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

    return train_loader, test_loader, cw_tensor, device, n_features


def compute_energy(n_features, h1, h2, num_steps, fire, top_k, n_out_neurons):
    effective_fire = fire * (top_k / h1)
    total_macs = (n_features*h1 + h1*h1*num_steps*effective_fire +
                  h1*h2*num_steps*effective_fire + h2*h2*num_steps*effective_fire +
                  h2*n_out_neurons*num_steps*effective_fire)
    return total_macs * 2.0 / 1000


def train_standard(train_loader, test_loader, cw_tensor, device, n_features, label):
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    torch.manual_seed(42); np.random.seed(42)
    model = CombinedSNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] Params: {n_params:,} | steps={NUM_STEPS} | LI k=24 | beta1=0.85 beta2=0.95")

    for epoch in range(NUM_EPOCHS):
        if epoch < warmup:
            for pg in opt.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup
        model.train()
        ep_loss, ep_rate, batches = 0, 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            saved = []
            with torch.no_grad():
                for p in model.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize(p.data))
            spk_out, mem_out, spk_h = model(data, NUM_STEPS)
            ce = sum(loss_fn(mem_out[s], targets) for s in range(NUM_STEPS))
            fire = spk_h.mean()
            loss = ce + 1.0 * torch.clamp(fire - 0.15, min=0)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                for p, s in zip(model.parameters(), saved): p.data.copy_(s)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += ce.item(); ep_rate += fire.item(); batches += 1
        if epoch >= warmup: sched.step()
        avg_ce = ep_loss / batches
        if avg_ce < best_ce:
            best_ce = avg_ce; best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | CE {avg_ce:.3f} | fire {ep_rate/batches:.3f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))

    model.eval()
    all_p, all_t, tot_spk, tot_pos = [], [], 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, _, spk_h = model(data, NUM_STEPS)
            _, pred = spk_out.sum(dim=0).max(1)
            tot_spk += spk_h.sum().item(); tot_pos += spk_h.numel()
            all_p.extend(pred.cpu().numpy()); all_t.extend(targets.cpu().numpy())

    fire = tot_spk / tot_pos if tot_pos > 0 else 0
    acc = np.mean(np.array(all_p) == np.array(all_t)) * 100
    cls_nJ = compute_energy(n_features, 48, 24, NUM_STEPS, fire, 24, 5)
    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% | Fire: {fire:.3f} | Energy: {cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_t, all_p, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


def train_hierarchical(train_loader, test_loader, cw_tensor, device, n_features, label):
    torch.manual_seed(42); np.random.seed(42)
    model = CombinedHierarchicalSNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] Params: {n_params:,} | steps={NUM_STEPS} | hierarchical | LI k=24")

    bin_weight = torch.tensor([0.2, 1.0], dtype=torch.float32).to(device)
    sub_weight = torch.tensor([1.0, 0.3, 1.2, 1.0], dtype=torch.float32).to(device)
    focal_bin = FocalLoss(weight=bin_weight, gamma=2.0)
    focal_sub = FocalLoss(weight=sub_weight, gamma=2.0)

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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | CE {avg_ce:.3f} | fire {ep_rate/batches:.3f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))

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
    cls_nJ = compute_energy(n_features, 48, 24, NUM_STEPS, fire, 24, 6)
    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% | Fire: {fire:.3f} | Energy: {cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_t, all_p, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


if __name__ == '__main__':
    results = {}

    for split in ['inter', 'intra']:
        # A: All improvements + standard 5-class (18 features)
        print(f"\n{'='*60}")
        print(f"VARIANT A: Combined standard ({split})")
        train_loader, test_loader, cw_t, device, n_feat = load_data(split, use_interactions=True)
        acc, nJ = train_standard(train_loader, test_loader, cw_t, device, n_feat,
                                 f"combined-std-{split}")
        results[f"combined-std-{split}"] = (acc, nJ)

        # B: All improvements + hierarchical (18 features)
        print(f"\n{'='*60}")
        print(f"VARIANT B: Combined hierarchical ({split})")
        train_loader, test_loader, cw_t, device, n_feat = load_data(split, use_interactions=True)
        acc, nJ = train_hierarchical(train_loader, test_loader, cw_t, device, n_feat,
                                     f"combined-hier-{split}")
        results[f"combined-hier-{split}"] = (acc, nJ)

        # C: Combined + 17th feature (19 features total with interactions)
        print(f"\n{'='*60}")
        print(f"VARIANT C: Combined hierarchical + 17feat ({split})")
        train_loader, test_loader, cw_t, device, n_feat = load_data(
            split, use_interactions=True, include_cross_lead=True)
        acc, nJ = train_hierarchical(train_loader, test_loader, cw_t, device, n_feat,
                                     f"combined-hier-17f-{split}")
        results[f"combined-hier-17f-{split}"] = (acc, nJ)

    print(f"\n{'='*60}")
    print(f"PHASE 5 SCRIPT 3: COMBINED ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':>35} {'Accuracy':>10} {'Energy':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    print(f"  {'baseline-20steps-inter':>35} {'91.94':>9}% {'42.9':>8}nJ")
    for label, (acc, nJ) in results.items():
        print(f"  {label:>35} {acc:>9.2f}% {nJ:>8.1f}nJ")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
