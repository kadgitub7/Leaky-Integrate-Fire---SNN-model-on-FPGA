"""
ARCHITECTURE COMPARISON — Built on the PROVEN Exp B foundation.
================================================================
Uses the exact training pipeline that achieved 91.45%:
  - 16 proven features (not the experimental 14)
  - Training-CE checkpoint (not val-based — DS1 val ≠ DS2 test)
  - Output spike COUNT prediction (not final membrane potential)
  - SMOTE 0.33, focal loss γ=2, cls_power=0.65, QAT-4bit

Then tests 6 architectures on top:
  1. SNN-Baseline:     16→48 RLeaky→24 RLeaky→5 Leaky (proven 91.45%)
  2. ANN-Direct:       16→48 ReLU→24 ReLU→5 (tests if SNN helps)
  3. Hybrid ANN→SNN:   16→32 ReLU→24 RLeaky→5 Leaky (learned combos)
  4. Skip-SNN:         16→32 RLeaky→16 RLeaky→5 (raw features bypass)
  5. Dual-Pathway:     Timing(5→16 fast) + Morph(11→24 slow)→16 merge→5
  6. SNN + corr_L0L1:  17 features (adds #4 RF feature to proven 16)

Run: python experiments/architecture_compare.py
"""

import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
import wfdb
import os
import copy
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import time

torch.set_num_threads(2)
batch_size = 128
num_class = 5
t0 = time.time()

# ================================================================
# PROVEN 16-FEATURE SET (Exp B, 91.45% inter-patient)
# ================================================================
FEATURE_NAMES_16 = [
    'pre_rr', 'post_rr', 'rr_ratio', 'rr_asymmetry', 'compensatory_ratio', 'rr_std_10',
    'qrs_width_L0', 'qrs_width_L1',
    'qrs_area_L0', 'qrs_area_L1',
    'max_slope_L0', 'max_slope_L1',
    'rel_area_L0', 'rel_area_L1',
    'templ_corr_L0', 'templ_corr_L1',
]

# 17-FEATURE SET: proven 16 + corr_L0L1 (RF importance #4)
FEATURE_NAMES_17 = FEATURE_NAMES_16 + ['corr_L0L1']

EWMA_ALPHA_MORPH = 0.015
N_TEMPLATE = 6
EWMA_GATE_THRESH = 0.85
EWMA_INIT_BEATS = 10
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

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','200','202','210',
       '212','213','214','219','221','222','228','231','232','233','234']


def extract_features(rec_list, include_cross_lead=False):
    """Extract features — matches Exp B exactly, with optional corr_L0L1."""
    all_labels, all_features = [], []
    n_feat = 17 if include_cross_lead else 16

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
                if idx - win_left >= 0 and idx + win_right < len(signals) and sym in aami_map:
                    valid.append((idx, sym))

            ewma_area = [None] * num_leads
            ewma_template = [None] * num_leads
            init_areas = [[] for _ in range(num_leads)]
            init_templates = [[] for _ in range(num_leads)]
            rr_history = []

            for i, (idx, sym) in enumerate(valid):
                beat = signals[idx - win_left : idx + win_right, :]
                all_labels.append(aami_map[sym])

                pre_rr = (idx - valid[i-1][0]) / fs if i > 0 else 0.833
                post_rr = (valid[i+1][0] - idx) / fs if i < len(valid) - 1 else 0.833

                local_rrs = []
                for j in range(max(1, i - 10), i + 1):
                    local_rrs.append((valid[j][0] - valid[j-1][0]) / fs)
                local_rr = np.mean(local_rrs) if local_rrs else 0.833

                rr_history.append(pre_rr)
                if len(rr_history) > 10: rr_history.pop(0)

                feat = np.zeros(n_feat, dtype=np.float32)
                feat[0] = pre_rr
                feat[1] = post_rr
                feat[2] = pre_rr / (local_rr + 1e-8)
                feat[3] = pre_rr / (post_rr + 1e-8)
                feat[4] = (pre_rr + post_rr) / (2 * local_rr + 1e-8)
                feat[5] = np.std(rr_history) if len(rr_history) >= 2 else 0.0

                qrs_data = {}
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
                    dqrs = np.diff(qrs) * fs
                    max_slope = np.max(np.abs(dqrs)) if len(dqrs) > 0 else 0.0

                    tmpl_idx = np.linspace(0, len(qrs) - 1, N_TEMPLATE).astype(int)
                    qrs_ds = qrs[tmpl_idx]

                    feat[6 + lead] = width
                    feat[8 + lead] = area
                    feat[10 + lead] = max_slope

                    if i < EWMA_INIT_BEATS:
                        init_areas[lead].append(max(area, 1e-6))
                        init_templates[lead].append(qrs_ds.copy())

                    if ewma_area[lead] is None:
                        if i >= EWMA_INIT_BEATS - 1 and len(init_areas[lead]) >= EWMA_INIT_BEATS:
                            ewma_area[lead] = np.median(init_areas[lead])
                            ewma_template[lead] = np.median(init_templates[lead], axis=0)
                        else:
                            ewma_area[lead] = max(area, 1e-6)
                            ewma_template[lead] = qrs_ds.copy()

                    feat[12 + lead] = area / (ewma_area[lead] + 1e-8)

                    norm_c = np.linalg.norm(qrs_ds)
                    norm_t = np.linalg.norm(ewma_template[lead])
                    tc = np.dot(qrs_ds, ewma_template[lead]) / (norm_c * norm_t + 1e-8) if norm_c > 1e-8 and norm_t > 1e-8 else 1.0
                    feat[14 + lead] = tc

                    if tc > EWMA_GATE_THRESH:
                        a = EWMA_ALPHA_MORPH
                        ewma_area[lead] = a * max(area, 1e-6) + (1 - a) * ewma_area[lead]
                        ewma_template[lead] = a * qrs_ds + (1 - a) * ewma_template[lead]

                    qrs_data[lead] = qrs

                # Cross-lead correlation (feature #17)
                if include_cross_lead and num_leads >= 2:
                    n0 = np.linalg.norm(qrs_data[0])
                    n1 = np.linalg.norm(qrs_data[1])
                    feat[16] = np.dot(qrs_data[0], qrs_data[1]) / (n0 * n1 + 1e-8) if n0 > 1e-8 and n1 > 1e-8 else 0.0

                all_features.append(feat)
        except Exception as e:
            print(f"  Skipping {rec_id}: {e}")

    return np.array(all_labels), np.array(all_features)


def smote_oversample(features, labels, target_ratio=0.33):
    counts = Counter(labels)
    max_count = max(counts.values())
    target_count = int(max_count * target_ratio)
    new_f, new_l = list(features), list(labels)
    for cls in range(num_class):
        idx = np.where(labels == cls)[0]
        if len(idx) >= target_count: continue
        n_syn = target_count - len(idx)
        cf = features[idx]
        for _ in range(n_syn):
            i = np.random.randint(len(cf))
            j = np.random.randint(len(cf))
            while j == i and len(cf) > 1: j = np.random.randint(len(cf))
            new_f.append(cf[i] + np.random.random() * (cf[j] - cf[i]))
            new_l.append(cls)
    return np.array(new_f), np.array(new_l)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma, self.weight = gamma, weight
    def forward(self, x, t):
        ce = nn.functional.cross_entropy(x, t, weight=self.weight, reduction='none')
        return (((1-torch.exp(-ce))**self.gamma)*ce).mean()


def quantize(x, bits=4):
    qmin, qmax = -(2**(bits-1)), 2**(bits-1)-1
    s = (x.max()-x.min()) / (qmax-qmin)
    s = torch.clamp(s, min=1e-8)
    return torch.clamp(torch.round(x/s), qmin, qmax)*s


# ================================================================
# ARCHITECTURES
# ================================================================

class BaselineSNN(nn.Module):
    """Exact Exp B architecture — the proven 91.45% model."""
    def __init__(self, n_in, h1=48, h2=24, n_out=5, beta=0.9):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h1)
        self.drop1 = nn.Dropout(0.1)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc_mid = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(0.1)
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


class PureANN(nn.Module):
    """Same topology but ReLU — tests if temporal dynamics help."""
    def __init__(self, n_in, h1=48, h2=24, n_out=5):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(h2, n_out),
        )
        self.h1, self.h2 = h1, h2
    def forward(self, x, num_steps=None):
        return self.net(self.bn(x))


class HybridANNSNN(nn.Module):
    """ANN encoder (16→32 ReLU) → SNN classifier (32→24 RLeaky → 5)."""
    def __init__(self, n_in, ann_h=32, snn_h=24, n_out=5, beta=0.9):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.encoder = nn.Sequential(nn.Linear(n_in, ann_h), nn.ReLU(), nn.Dropout(0.1))
        self.fc_snn = nn.Linear(ann_h, snn_h)
        self.drop = nn.Dropout(0.1)
        self.rlif = snn.RLeaky(beta=beta, linear_features=snn_h, learn_beta=True, learn_threshold=True)
        self.fc_out = nn.Linear(snn_h, n_out)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = ann_h, snn_h

    def forward(self, x, num_steps):
        s, m = self.rlif.init_rleaky()
        mo = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk_rec = [], [], []
        enc = self.encoder(self.bn(x))
        fc = self.drop(self.fc_snn(enc))
        for _ in range(num_steps):
            s, m = self.rlif(fc, s, m)
            spk_rec.append(s)
            so, mo = self.lif_out(self.fc_out(s), mo)
            spk_out_rec.append(so)
            mem_out_rec.append(mo)
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), torch.stack(spk_rec)


class SkipSNN(nn.Module):
    """Raw features bypass to every layer — free in analog (just wires)."""
    def __init__(self, n_in, h1=32, h2=16, n_out=5, beta=0.9):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h1)
        self.drop1 = nn.Dropout(0.1)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(n_in + h1, h2)
        self.drop2 = nn.Dropout(0.1)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2, learn_beta=True, learn_threshold=True)
        self.fc3 = nn.Linear(n_in + h2, n_out)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2
        self.n_in = n_in

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
            s2, m2 = self.rlif2(self.drop2(self.fc2(torch.cat([x, s1], 1))), s2, m2)
            so, mo = self.lif_out(self.fc3(torch.cat([x, s2], 1)), mo)
            spk_out_rec.append(so)
            mem_out_rec.append(mo)
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), torch.stack(spk1_rec)


TIMING_IDX = [0, 1, 2, 3, 4, 5]  # pre_rr, post_rr, rr_ratio, rr_asymmetry, comp_ratio, rr_std
MORPH_IDX  = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # widths, areas, slopes, rel_areas, templ_corrs

class DualPathwaySNN(nn.Module):
    """Timing(fast) + Morphology(slow) parallel pathways — mirrors cardiology."""
    def __init__(self, n_in=16, n_out=5):
        super().__init__()
        n_t, n_m = len(TIMING_IDX), len(MORPH_IDX)
        h_t, h_m, h_merge = 16, 24, 16
        self.bn_t = nn.BatchNorm1d(n_t)
        self.fc_t = nn.Linear(n_t, h_t)
        self.drop_t = nn.Dropout(0.1)
        self.rlif_t = snn.RLeaky(beta=0.7, linear_features=h_t, learn_beta=True, learn_threshold=True)
        self.bn_m = nn.BatchNorm1d(n_m)
        self.fc_m = nn.Linear(n_m, h_m)
        self.drop_m = nn.Dropout(0.1)
        self.rlif_m = snn.RLeaky(beta=0.95, linear_features=h_m, learn_beta=True, learn_threshold=True)
        self.fc_merge = nn.Linear(h_t + h_m, h_merge)
        self.drop_mrg = nn.Dropout(0.1)
        self.rlif_merge = snn.RLeaky(beta=0.9, linear_features=h_merge, learn_beta=True, learn_threshold=True)
        self.fc_out = nn.Linear(h_merge, n_out)
        self.lif_out = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h_t + h_m, h_merge

    def forward(self, x, num_steps):
        st, mt = self.rlif_t.init_rleaky()
        sm, mm = self.rlif_m.init_rleaky()
        s_mrg, m_mrg = self.rlif_merge.init_rleaky()
        mo = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk_rec = [], [], []
        x_t = self.drop_t(self.fc_t(self.bn_t(x[:, TIMING_IDX])))
        x_m = self.drop_m(self.fc_m(self.bn_m(x[:, MORPH_IDX])))
        for _ in range(num_steps):
            st, mt = self.rlif_t(x_t, st, mt)
            sm, mm = self.rlif_m(x_m, sm, mm)
            merged = torch.cat([st, sm], 1)
            spk_rec.append(merged)
            s_mrg, m_mrg = self.rlif_merge(self.drop_mrg(self.fc_merge(merged)), s_mrg, m_mrg)
            so, mo = self.lif_out(self.fc_out(s_mrg), mo)
            spk_out_rec.append(so)
            mem_out_rec.append(mo)
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), torch.stack(spk_rec)


# ================================================================
# TRAINING (exact Exp B method)
# ================================================================

def train_and_eval(name, model, train_loader, test_loader, cw_tensor, device,
                   is_snn=True, num_steps=20, num_epochs=200, n_features=16):
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs - warmup)

    # PROVEN: checkpoint by training CE (not val accuracy)
    best_ce = float('inf')
    best_state = None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{name}] Params: {n_params:,}")

    for epoch in range(num_epochs):
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

            if is_snn:
                spk_out, mem_out, spk_hidden = model(data, num_steps)
                ce = sum(loss_fn(mem_out[s], targets) for s in range(num_steps))
                fire = spk_hidden.mean()
                loss = ce + 1.0 * torch.clamp(fire - 0.15, min=0)
                ep_rate += fire.item()
            else:
                out = model(data)
                ce = loss_fn(out, targets)
                loss = ce

            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p, s in zip(model.parameters(), saved): p.data.copy_(s)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += ce.item()
            batches += 1

        if epoch >= warmup: sched.step()
        avg_ce = ep_loss / batches

        # PROVEN: checkpoint by training CE
        if avg_ce < best_ce:
            best_ce = avg_ce
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d}/{num_epochs} | CE {avg_ce:.3f} | fire {ep_rate/batches:.3f} | {time.time()-t0:.0f}s")

    # Load best and quantize
    model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))

    # PROVEN: evaluate using output spike COUNT (not final membrane)
    model.eval()
    all_p, all_t = [], []
    tot_spk, tot_pos = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            if is_snn:
                spk_out, _, spk_hidden = model(data, num_steps)
                _, pred = spk_out.sum(dim=0).max(1)  # SPIKE COUNT prediction
                tot_spk += spk_hidden.sum().item()
                tot_pos += spk_hidden.numel()
            else:
                pred = model(data).argmax(1)
            all_p.extend(pred.cpu().numpy())
            all_t.extend(targets.cpu().numpy())

    fire = tot_spk / tot_pos if tot_pos > 0 else 0
    acc = np.mean(np.array(all_p) == np.array(all_t)) * 100

    print(f"\n  [{name}] RESULT: {acc:.2f}% | Params: {n_params:,} | Fire: {fire:.3f}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q']))
    cm = confusion_matrix(all_t, all_p)
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")

    # Energy
    h1, h2 = model.h1, model.h2
    if is_snn:
        fc_macs = n_features * h1
        rec_macs = h1 * h1 * num_steps * fire
        mid_macs = h1 * h2 * num_steps * fire
        rec2_macs = h2 * h2 * num_steps * fire
        out_macs = h2 * 5 * num_steps * fire
        total_macs = fc_macs + rec_macs + mid_macs + rec2_macs + out_macs
    else:
        total_macs = n_features * h1 + h1 * h2 + h2 * 5
    cls_nJ = total_macs * 2.0 / 1000
    frontend_nJ = 13.7
    print(f"  Energy: {total_macs:,.0f} MACs | classifier={cls_nJ:.1f}nJ | total={cls_nJ+frontend_nJ:.1f}nJ")

    return acc, n_params, cls_nJ + frontend_nJ


# ================================================================
# DATA LOADING
# ================================================================

results = {}

for feat_mode, include_cross, n_feat_label in [('16feat', False, '16'), ('17feat', True, '17')]:
    print(f"\n{'='*70}")
    print(f"LOADING DATA: {n_feat_label} features")
    print(f"{'='*70}")

    train_labels, train_features = extract_features(DS1, include_cross_lead=include_cross)
    test_labels, test_features = extract_features(DS2, include_cross_lead=include_cross)
    n_features = train_features.shape[1]
    print(f"  DS1: {len(train_labels)} | DS2: {len(test_labels)} | Features: {n_features}")
    print(f"  Class dist: {dict(Counter(train_labels))}")

    train_features, train_labels = smote_oversample(train_features, train_labels, 0.33)
    print(f"  After SMOTE: {dict(Counter(train_labels))}")

    mu, sd = train_features.mean(0), train_features.std(0)
    train_features = (train_features - mu) / (sd + 1e-8)
    test_features = (test_features - mu) / (sd + 1e-8)

    class DS(torch.utils.data.Dataset):
        def __init__(self, f, l):
            self.data = torch.tensor(f, dtype=torch.float32)
            self.targets = torch.tensor(l, dtype=torch.long)
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i], self.targets[i]

    tr_counts = np.bincount(train_labels, minlength=5)
    sw = 1.0 / (tr_counts ** 0.65)
    sample_w = [sw[l] for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = torch.utils.data.DataLoader(DS(train_features, train_labels), batch_size=128, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(DS(test_features, test_labels), batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw = 1.0 / (tr_counts.astype(np.float64) ** 0.65)
    cw = cw / cw.sum() * 5
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    if feat_mode == '16feat':
        # Test all architectures with proven 16 features
        for arch_name, model_fn, is_snn in [
            ("1_SNN_Baseline_48-24", lambda: BaselineSNN(n_features), True),
            ("2_ANN_Direct_48-24", lambda: PureANN(n_features), False),
            ("3_Hybrid_ANN32-SNN24", lambda: HybridANNSNN(n_features), True),
            ("4_Skip_SNN_32-16", lambda: SkipSNN(n_features), True),
            ("5_DualPathway_16f24s-16m", lambda: DualPathwaySNN(n_features), True),
        ]:
            torch.manual_seed(42); np.random.seed(42)
            m = model_fn().to(device)
            acc, params, nJ = train_and_eval(arch_name, m, train_loader, test_loader, cw_t, device, is_snn=is_snn, n_features=n_features)
            results[arch_name] = (acc, params, nJ)

    else:
        # Test baseline with 17 features (16 + corr_L0L1)
        torch.manual_seed(42); np.random.seed(42)
        m = BaselineSNN(n_features).to(device)
        acc, params, nJ = train_and_eval("6_SNN_Baseline_17feat", m, train_loader, test_loader, cw_t, device, is_snn=True, n_features=n_features)
        results["6_SNN_Baseline_17feat"] = (acc, params, nJ)


# ================================================================
# FINAL SUMMARY
# ================================================================

print(f"\n{'='*70}")
print(f"ARCHITECTURE COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"  {'Architecture':40s} {'Acc':>8} {'Params':>8} {'Energy':>8}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
for name, (acc, params, nJ) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  {name:40s} {acc:7.2f}% {params:7,} {nJ:6.1f}nJ")
print(f"\n  Exp B baseline: 91.45% (training-CE checkpoint, spike count prediction)")
print(f"  RF ceiling: 93.61%")
print(f"\n  Total time: {time.time()-t0:.0f}s")
