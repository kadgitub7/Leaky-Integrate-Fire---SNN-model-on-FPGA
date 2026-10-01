"""
PHASE 5 SCRIPT 1: Energy-Optimized SNN
========================================
Changes from proven baseline (91.94% / 42.9nJ):

  1. Reduced timesteps: 20 → 12
     Rationale: ECG beat is ~167ms at 360Hz. 20 timesteps oversample the
     temporal dynamics. Data shows timing features dominate classification;
     12 steps provide sufficient integration. Reduces energy proportionally.

  2. Lateral inhibition (top-k sparsity) in hidden layer
     Rationale: Only top-k neurons fire per timestep. Forces sparse
     representations, reduces redundant spikes by 30-50%.
     Literature: WTA mechanisms cut energy with minimal accuracy loss.

  3. Per-layer beta initialization: beta1=0.85 (longer memory for timing
     integration), beta2=0.95 (faster response for output decisions)
     Rationale: learn_beta=True means these are just inits, but different
     starting points explore different regions of loss landscape.

  4. dropout=0.05 (proven best from Phase 4)

Tests: 3 variants
  A: 12 timesteps only (vs 20 baseline)
  B: 12 timesteps + lateral inhibition (k=24 of 48)
  C: 12 timesteps + lateral inhibition + per-layer beta

Both inter and intra patient.
Run: python experiments/phase5_energy.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.data_proven import load_split

torch.set_num_threads(2)
t0 = time.time()
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


class EnergySNN(nn.Module):
    """SNN with optional lateral inhibition and per-layer beta."""
    def __init__(self, n_in, h1=48, h2=24, n_out=5,
                 beta1=0.9, beta2=0.9, use_lateral_inhibition=False, top_k=24):
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
        self.use_li = use_lateral_inhibition
        self.top_k = top_k

    def lateral_inhibition(self, spk, k):
        """Keep only top-k spikes per timestep (WTA). During training,
        use soft masking via straight-through estimator to maintain gradients."""
        if not self.training:
            # Hard top-k during eval
            vals = spk.abs()
            topk_vals, _ = vals.topk(k, dim=-1)
            threshold = topk_vals[..., -1:].detach()
            mask = (vals >= threshold).float()
            return spk * mask
        else:
            # Soft approximation: scale down non-top-k spikes
            vals = spk.abs()
            topk_vals, _ = vals.topk(k, dim=-1)
            threshold = topk_vals[..., -1:].detach()
            mask = (vals >= threshold).float()
            # STE: forward uses hard mask, backward passes gradient through
            return spk * (mask + spk.detach() * (1 - mask) * 0.1 - (spk.detach() * (1 - mask) * 0.1).detach())

    def forward(self, x, num_steps):
        s1, m1 = self.rlif1.init_rleaky()
        s2, m2 = self.rlif2.init_rleaky()
        mo = self.lif_out.init_leaky()
        spk_out_rec, mem_out_rec, spk1_rec = [], [], []
        x = self.bn(x)
        fc1 = self.drop1(self.fc1(x))
        for _ in range(num_steps):
            s1, m1 = self.rlif1(fc1, s1, m1)
            s1_out = self.lateral_inhibition(s1, self.top_k) if self.use_li else s1
            spk1_rec.append(s1_out)
            s2, m2 = self.rlif2(self.drop2(self.fc_mid(s1_out)), s2, m2)
            so, mo = self.lif_out(self.fc2(s2), mo)
            spk_out_rec.append(so)
            mem_out_rec.append(mo)
        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), torch.stack(spk1_rec)


def train_and_eval(train_loader, test_loader, cw_tensor, device, n_features,
                   num_steps, beta1, beta2, use_li, top_k, label):
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    torch.manual_seed(42); np.random.seed(42)
    model = EnergySNN(n_features, beta1=beta1, beta2=beta2,
                      use_lateral_inhibition=use_li, top_k=top_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] Params: {n_params:,} | steps={num_steps} | LI={use_li}(k={top_k}) | beta1={beta1} beta2={beta2}")

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
            spk_out, mem_out, spk_h = model(data, num_steps)
            ce = sum(loss_fn(mem_out[s], targets) for s in range(num_steps))
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
            spk_out, _, spk_h = model(data, num_steps)
            _, pred = spk_out.sum(dim=0).max(1)
            tot_spk += spk_h.sum().item(); tot_pos += spk_h.numel()
            all_p.extend(pred.cpu().numpy()); all_t.extend(targets.cpu().numpy())

    fire = tot_spk / tot_pos if tot_pos > 0 else 0
    acc = np.mean(np.array(all_p) == np.array(all_t)) * 100
    h1, h2 = model.h1, model.h2
    effective_fire = fire * (top_k / h1) if use_li else fire
    fc1_macs = n_features * h1
    rec1 = h1 * h1 * num_steps * effective_fire
    mid = h1 * h2 * num_steps * effective_fire
    rec2 = h2 * h2 * num_steps * effective_fire
    out = h2 * 5 * num_steps * effective_fire
    total_macs = fc1_macs + rec1 + mid + rec2 + out
    cls_nJ = total_macs * 2.0 / 1000

    print(f"\n  {'='*60}")
    print(f"  [{label}] RESULT: {acc:.2f}% | Fire: {fire:.3f} (eff: {effective_fire:.3f})")
    print(f"  Energy: {total_macs:,.0f} MACs | classifier={cls_nJ:.1f}nJ | total={cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q'], zero_division=0))
    cm = confusion_matrix(all_t, all_p, labels=[0,1,2,3,4])
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, cls_nJ + FRONTEND_NJ


if __name__ == '__main__':
    results = {}

    configs = [
        # (steps, beta1, beta2, use_li, top_k, label)
        (12, 0.9, 0.9, False, 48, "12-steps"),
        (12, 0.9, 0.9, True,  24, "12-steps+LI-k24"),
        (12, 0.85, 0.95, True, 24, "12-steps+LI+perbeta"),
    ]

    for split in ['inter', 'intra']:
        train_loader, test_loader, cw_t, device, n_features = load_split(split=split)
        for steps, b1, b2, li, k, label in configs:
            full_label = f"{label}-{split}"
            acc, nJ = train_and_eval(train_loader, test_loader, cw_t, device,
                                     n_features, steps, b1, b2, li, k, full_label)
            results[full_label] = (acc, nJ)

    print(f"\n{'='*60}")
    print(f"PHASE 5 SCRIPT 1: ENERGY OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':>30} {'Accuracy':>10} {'Energy':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'baseline-20steps-inter':>30} {'91.94':>9}% {'42.9':>8}nJ")
    for label, (acc, nJ) in results.items():
        print(f"  {label:>30} {acc:>9.2f}% {nJ:>8.1f}nJ")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
