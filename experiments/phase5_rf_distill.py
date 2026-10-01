"""
PHASE 5 EXP 3: RF Knowledge Distillation
==========================================
The RF achieves 93.61% on the same 16 features. We train the SNN to match
the RF's soft probability predictions, transferring its learned decision
boundaries to the SNN without changing inference energy.

Loss = alpha * FocalLoss(snn, hard_labels) + (1-alpha) * KL(snn, rf_probs)

Also tests: adding corr_L0L1 (RF importance #4) as 17th feature.

Both inter and intra patient evaluation.
Run: python experiments/phase5_rf_distill.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from experiments.data_proven import load_split, extract_features, DS1, DS2, smote_oversample, NUM_CLASSES, BATCH_SIZE
from collections import Counter

torch.set_num_threads(2)
t0 = time.time()
NUM_STEPS = 20
NUM_EPOCHS = 200
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


def evaluate(model, test_loader, device, n_features, label):
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
    cm = confusion_matrix(all_t, all_p, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


def train_rf_teacher(train_features, train_labels):
    """Train RF and get soft probabilities for all training samples."""
    print(f"  Training RF teacher on {len(train_labels)} samples...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(train_features, train_labels)
    train_probs = rf.predict_proba(train_features)
    train_acc = np.mean(rf.predict(train_features) == train_labels) * 100
    print(f"  RF train accuracy: {train_acc:.1f}%")
    return rf, train_probs


def train_with_distillation(train_features_norm, train_labels, rf_probs,
                            cw_tensor, device, n_features, alpha=0.7, temperature=3.0):
    """Train SNN with RF knowledge distillation."""
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    torch.manual_seed(42); np.random.seed(42)
    model = BaselineSNN(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [RF-distill alpha={alpha} T={temperature}] Params: {n_params:,}")

    class DistillDS(torch.utils.data.Dataset):
        def __init__(self, f, l, p):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
            self.probs = torch.tensor(p, dtype=torch.float32)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i], self.probs[i]

    tr_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    cls_power = 0.65
    sw = 1.0 / (tr_counts ** cls_power)
    sample_w = [sw[l] for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        DistillDS(train_features_norm, train_labels, rf_probs),
        batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

    for epoch in range(NUM_EPOCHS):
        if epoch < warmup:
            for pg in opt.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup
        model.train()
        ep_loss, ep_rate, batches = 0, 0, 0
        for data, targets, teacher_probs in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            teacher_probs = teacher_probs.to(device)

            saved = []
            with torch.no_grad():
                for p in model.parameters():
                    saved.append(p.data.clone())
                    p.data.copy_(quantize(p.data))

            spk_out, mem_out, spk_h = model(data, NUM_STEPS)

            # Hard label loss (focal)
            ce = sum(loss_fn(mem_out[s], targets) for s in range(NUM_STEPS))

            # Soft label loss (KL divergence with temperature scaling)
            # Use average membrane potential across timesteps for distillation
            avg_mem = torch.stack([mem_out[s] for s in range(NUM_STEPS)]).mean(0)
            student_log_probs = nn.functional.log_softmax(avg_mem / temperature, dim=1)
            teacher_soft = nn.functional.softmax(
                torch.log(teacher_probs + 1e-8) / temperature, dim=1)
            kl = kl_loss(student_log_probs, teacher_soft) * (temperature ** 2)

            fire = spk_h.mean()
            loss = alpha * ce + (1 - alpha) * kl + 1.0 * torch.clamp(fire - 0.15, min=0)

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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{NUM_EPOCHS} | CE {avg_ce:.3f} | fire {ep_rate/batches:.3f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))
    return model


if __name__ == '__main__':
    results = {}

    # --- INTER-PATIENT with RF distillation ---
    print("\n" + "="*60)
    print("INTER-PATIENT: RF Knowledge Distillation (16 features)")
    print("="*60)

    # Extract raw features for RF
    train_labels, train_features = extract_features(DS1)
    train_features_aug, train_labels_aug = smote_oversample(train_features, train_labels, 0.33)
    train_mu = train_features_aug.mean(0)
    train_sd = train_features_aug.std(0)
    train_features_norm = (train_features_aug - train_mu) / (train_sd + 1e-8)

    test_labels, test_features = extract_features(DS2)
    test_features_norm = (test_features - train_mu) / (train_sd + 1e-8)

    n_features = train_features_norm.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_counts = np.bincount(train_labels_aug, minlength=NUM_CLASSES)
    cls_power = 0.65
    cw = 1.0 / (tr_counts.astype(np.float64) ** cls_power)
    cw = cw / cw.sum() * NUM_CLASSES
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    # Train RF teacher on augmented data
    rf, rf_probs = train_rf_teacher(train_features_aug, train_labels_aug)

    # Check RF test accuracy
    test_pred = rf.predict(test_features)
    rf_test_acc = np.mean(test_pred == test_labels) * 100
    print(f"  RF test accuracy (inter): {rf_test_acc:.2f}%")
    results["RF-teacher-inter"] = (rf_test_acc, 0)

    # Test loader for evaluation
    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    test_loader = torch.utils.data.DataLoader(
        DS(test_features_norm, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    # Train SNN with distillation — test multiple alpha values
    for alpha in [0.5, 0.7, 0.9]:
        model = train_with_distillation(
            train_features_norm, train_labels_aug, rf_probs,
            cw_tensor, device, n_features, alpha=alpha, temperature=3.0)
        acc, nJ = evaluate(model, test_loader, device, n_features,
                          f"distill-a{alpha}-inter")
        results[f"distill-a{alpha}-inter"] = (acc, nJ)

    # --- Also test with 17th feature (corr_L0L1) ---
    print("\n" + "="*60)
    print("INTER-PATIENT: RF Distillation + corr_L0L1 (17 features)")
    print("="*60)

    train_labels17, train_features17 = extract_features(DS1, include_cross_lead=True)
    train_features17_aug, train_labels17_aug = smote_oversample(train_features17, train_labels17, 0.33)
    train_mu17 = train_features17_aug.mean(0)
    train_sd17 = train_features17_aug.std(0)
    train_features17_norm = (train_features17_aug - train_mu17) / (train_sd17 + 1e-8)

    test_labels17, test_features17 = extract_features(DS2, include_cross_lead=True)
    test_features17_norm = (test_features17 - train_mu17) / (train_sd17 + 1e-8)

    n_features17 = train_features17_norm.shape[1]
    tr_counts17 = np.bincount(train_labels17_aug, minlength=NUM_CLASSES)
    cw17 = 1.0 / (tr_counts17.astype(np.float64) ** cls_power)
    cw17 = cw17 / cw17.sum() * NUM_CLASSES
    cw_tensor17 = torch.tensor(cw17, dtype=torch.float32).to(device)

    rf17, rf_probs17 = train_rf_teacher(train_features17_aug, train_labels17_aug)
    rf17_test_acc = np.mean(rf17.predict(test_features17) == test_labels17) * 100
    print(f"  RF test accuracy (17feat): {rf17_test_acc:.2f}%")
    results["RF-17feat-inter"] = (rf17_test_acc, 0)

    test_loader17 = torch.utils.data.DataLoader(
        DS(test_features17_norm, test_labels17), batch_size=BATCH_SIZE, shuffle=False)

    model17 = train_with_distillation(
        train_features17_norm, train_labels17_aug, rf_probs17,
        cw_tensor17, device, n_features17, alpha=0.7, temperature=3.0)
    acc, nJ = evaluate(model17, test_loader17, device, n_features17,
                      "distill-17feat-inter")
    results["distill-17feat-inter"] = (acc, nJ)

    print(f"\n{'='*60}")
    print(f"PHASE 5 EXP 3: RF DISTILLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':>25} {'Accuracy':>10} {'Energy':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'proven baseline':>25} {'91.94':>9}% {'42.9':>8}nJ")
    for label, (acc, nJ) in results.items():
        e_str = f"{nJ:>8.1f}nJ" if nJ > 0 else "    N/A "
        print(f"  {label:>25} {acc:>9.2f}% {e_str}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
