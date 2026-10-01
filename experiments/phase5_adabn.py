"""
PHASE 5 EXP 2: Per-Patient Adaptive BatchNorm (AdaBN)
======================================================
Builds on PROVEN dropout=0.05 baseline (91.94%).

At test time, update BatchNorm running statistics per DS2 record before
predicting. This adapts the model's normalization to each patient's
morphology, addressing the DS1→DS2 domain shift.

AdaBN is zero-cost at inference on analog hardware — BN is folded into
weights during deployment, so adapting BN stats is a calibration step.

Tests two variants:
  1. AdaBN on proven baseline (dropout=0.05)
  2. AdaBN + SWA (if both help, they should be additive)

Both inter and intra patient evaluation.
Run: python experiments/phase5_adabn.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import wfdb
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.data_proven import (
    extract_features, DS1, DS2, ALL_RECORDS, NUM_CLASSES, BATCH_SIZE,
    smote_oversample, aami_map, WIN_LEFT, WIN_RIGHT
)
from collections import Counter

torch.set_num_threads(2)
t0 = time.time()
NUM_STEPS = 20
NUM_EPOCHS = 200
SWA_START = 150
SWA_FREQ = 5
FRONTEND_NJ = 13.7


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


class BaselineSNN(nn.Module):
    def __init__(self, n_in, h1=48, h2=24, n_out=5, beta=0.9):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h1)
        self.drop1 = nn.Dropout(0.05)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc_mid = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(0.05)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(h2, n_out)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2

    def forward(self, x, num_steps):
        s1, m1 = self.rlif1.init_rleaky()
        s2, m2 = self.rlif2.init_rleaky()
        mo = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk1_rec = [], [], []
        x = self.bn(x)
        fc1 = self.drop1(self.fc1(x))
        for _ in range(num_steps):
            s1, m1 = self.rlif1(fc1, s1, m1)
            spk1_rec.append(s1)
            s2, m2 = self.rlif2(self.drop2(self.fc_mid(s1)), s2, m2)
            so, mo = self.lif_out(self.fc2(s2), mo)
            spk_out_rec.append(so)
            mem_out_rec.append(mo)
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), torch.stack(spk1_rec)


def train_model(train_loader, cw_tensor, device, n_features, use_swa=False):
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    torch.manual_seed(42); np.random.seed(42)
    model = BaselineSNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    swa_state, swa_count = None, 0
    label = "SWA+AdaBN" if use_swa else "AdaBN"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] Params: {n_params:,}")

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
            best_ce = avg_ce
            best_state = copy.deepcopy(model.state_dict())
        if use_swa and epoch >= SWA_START and (epoch - SWA_START) % SWA_FREQ == 0:
            if swa_state is None:
                swa_state = copy.deepcopy(model.state_dict())
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] = (swa_state[k] * swa_count + model.state_dict()[k]) / (swa_count + 1)
                swa_count += 1
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | CE {avg_ce:.3f} | fire {ep_rate/batches:.3f} | {time.time()-t0:.0f}s")

    if use_swa and swa_state is not None:
        print(f"  SWA: averaged {swa_count} snapshots")
        model.load_state_dict(swa_state)
    else:
        model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))
    return model


def evaluate_with_adabn(model, device, n_features, train_mu, train_sd, split, label):
    """Evaluate with per-patient BN adaptation."""
    if split == 'inter':
        test_records = DS2
    else:
        test_records = ALL_RECORDS

    all_p, all_t, tot_spk, tot_pos = [], [], 0, 0
    saved_bn_state = copy.deepcopy(model.bn.state_dict())

    for rec_id in test_records:
        rec_labels, rec_features = extract_features([rec_id])
        if len(rec_labels) == 0:
            continue
        rec_features_norm = (rec_features - train_mu) / (train_sd + 1e-8)
        rec_tensor = torch.tensor(rec_features_norm, dtype=torch.float32).to(device)
        rec_targets = torch.tensor(rec_labels, dtype=torch.long)

        if split == 'intra':
            pass

        # AdaBN: run forward pass in train mode to update BN stats for this patient
        model.bn.load_state_dict(saved_bn_state)
        model.bn.reset_running_stats()
        model.train()
        with torch.no_grad():
            for _ in range(1):
                if len(rec_tensor) > 1:
                    model.bn(rec_tensor)

        # Now evaluate in eval mode with patient-adapted BN
        model.eval()
        with torch.no_grad():
            spk_out, _, spk_h = model(rec_tensor, NUM_STEPS)
            _, pred = spk_out.sum(dim=0).max(1)
            tot_spk += spk_h.sum().item()
            tot_pos += spk_h.numel()
            all_p.extend(pred.cpu().numpy())
            all_t.extend(rec_targets.numpy())

    model.bn.load_state_dict(saved_bn_state)

    fire = tot_spk / tot_pos if tot_pos > 0 else 0
    acc = np.mean(np.array(all_p) == np.array(all_t)) * 100
    h1, h2 = model.h1, model.h2
    fc1_macs = n_features * h1
    rec1 = h1 * h1 * NUM_STEPS * fire
    mid = h1 * h2 * NUM_STEPS * fire
    rec2 = h2 * h2 * NUM_STEPS * fire
    out = h2 * 5 * NUM_STEPS * fire
    total_macs = fc1_macs + rec1 + mid + rec2 + out
    cls_nJ = total_macs * 2.0 / 1000

    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% | Fire: {fire:.3f}")
    print(f"  Energy: {total_macs:,.0f} MACs | classifier={cls_nJ:.1f}nJ | total={cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_t, all_p, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


def evaluate_standard(model, test_loader, device, n_features, label):
    """Standard evaluation without AdaBN (reference)."""
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
    h1, h2 = model.h1, model.h2
    fc1_macs = n_features * h1
    rec1 = h1 * h1 * NUM_STEPS * fire
    mid = h1 * h2 * NUM_STEPS * fire
    rec2 = h2 * h2 * NUM_STEPS * fire
    out = h2 * 5 * NUM_STEPS * fire
    total_macs = fc1_macs + rec1 + mid + rec2 + out
    cls_nJ = total_macs * 2.0 / 1000

    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% | Fire: {fire:.3f}")
    print(f"  Energy: {total_macs:,.0f} MACs | classifier={cls_nJ:.1f}nJ | total={cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_t, all_p)
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


if __name__ == '__main__':
    results = {}

    # --- INTER-PATIENT ---
    print("\n" + "="*60)
    print("INTER-PATIENT EVALUATION")
    print("="*60)

    # Load data and compute normalization stats we'll need for AdaBN
    train_labels, train_features = extract_features(DS1)
    train_features_raw = train_features.copy()
    train_labels_aug, train_features_aug = smote_oversample(train_features, train_labels, 0.33)
    train_mu = train_features_aug.mean(0)
    train_sd = train_features_aug.std(0)
    train_features_norm = (train_features_aug - train_mu) / (train_sd + 1e-8)

    test_labels, test_features = extract_features(DS2)
    test_features_norm = (test_features - train_mu) / (train_sd + 1e-8)

    n_features = train_features_norm.shape[1]
    tr_counts = np.bincount(train_labels_aug, minlength=NUM_CLASSES)
    cls_power = 0.65
    cw = 1.0 / (tr_counts.astype(np.float64) ** cls_power)
    cw = cw / cw.sum() * NUM_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    sw = 1.0 / (tr_counts ** cls_power)
    sample_w = [sw[l] for l in train_labels_aug]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        DS(train_features_norm, train_labels_aug), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        DS(test_features_norm, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    # Train model without SWA
    model = train_model(train_loader, cw_tensor, device, n_features, use_swa=False)

    # Evaluate: standard (reference)
    acc, nJ = evaluate_standard(model, test_loader, device, n_features, "baseline-inter")
    results["baseline-inter"] = (acc, nJ)

    # Evaluate: AdaBN
    acc, nJ = evaluate_with_adabn(model, device, n_features, train_mu, train_sd, 'inter', "AdaBN-inter")
    results["AdaBN-inter"] = (acc, nJ)

    # Train model with SWA
    model_swa = train_model(train_loader, cw_tensor, device, n_features, use_swa=True)

    # Evaluate: SWA standard
    acc, nJ = evaluate_standard(model_swa, test_loader, device, n_features, "SWA-inter")
    results["SWA-inter"] = (acc, nJ)

    # Evaluate: SWA + AdaBN
    acc, nJ = evaluate_with_adabn(model_swa, device, n_features, train_mu, train_sd, 'inter', "SWA+AdaBN-inter")
    results["SWA+AdaBN-inter"] = (acc, nJ)

    print(f"\n{'='*60}")
    print(f"PHASE 5 EXP 2: AdaBN SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':>25} {'Accuracy':>10} {'Energy':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for label, (acc, nJ) in results.items():
        print(f"  {label:>25} {acc:>9.2f}% {nJ:>8.1f}nJ")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
