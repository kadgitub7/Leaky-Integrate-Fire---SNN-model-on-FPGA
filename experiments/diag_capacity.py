"""
DIAGNOSTIC 2: Model Capacity
==============================
Tests larger hidden sizes on the PROVEN baseline architecture.
If accuracy improves with more neurons, the model is capacity-limited.

  48-24: 5,107 params, ~43nJ (proven baseline for reference)
  52-26: 6,033 params, ~47nJ
  56-28: 6,993 params, ~51nJ (borderline energy)

Inter-patient only.
Run: python experiments/diag_capacity.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.data_proven import load_split

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


def train_and_eval(train_loader, test_loader, cw_tensor, device, n_features, h1, h2):
    loss_fn = FocalLoss(weight=cw_tensor, gamma=2.0)
    torch.manual_seed(42); np.random.seed(42)
    model = BaselineSNN(n_features, h1=h1, h2=h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS - warmup)
    best_ce, best_state = float('inf'), None
    n_params = sum(p.numel() for p in model.parameters())
    label = f"{h1}-{h2}"
    print(f"\n  [Capacity {label}] Params: {n_params:,}")

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
    fc1_macs = n_features * h1
    rec1 = h1 * h1 * NUM_STEPS * fire
    mid = h1 * h2 * NUM_STEPS * fire
    rec2 = h2 * h2 * NUM_STEPS * fire
    out = h2 * 5 * NUM_STEPS * fire
    total_macs = fc1_macs + rec1 + mid + rec2 + out
    cls_nJ = total_macs * 2.0 / 1000

    print(f"\n  {'='*60}")
    print(f"  [Capacity {label}] RESULT: {acc:.2f}% | Params: {n_params:,} | Fire: {fire:.3f}")
    print(f"  Energy: {total_macs:,.0f} MACs | classifier={cls_nJ:.1f}nJ | total={cls_nJ+FRONTEND_NJ:.1f}nJ")
    print(f"  {'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q']))
    cm = confusion_matrix(all_t, all_p)
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")
    return acc, n_params, cls_nJ + FRONTEND_NJ


if __name__ == '__main__':
    train_loader, test_loader, cw_t, device, n_features = load_split(split='inter')
    results = {}
    for h1, h2 in [(52, 26), (56, 28)]:
        acc, params, nJ = train_and_eval(train_loader, test_loader, cw_t, device, n_features, h1, h2)
        results[f"{h1}-{h2}"] = (acc, params, nJ)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC 2: MODEL CAPACITY SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':>10} {'Accuracy':>10} {'Params':>8} {'Energy':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    print(f"  {'48-24':>10} {'91.45':>9}% {'5,107':>8} {'42.7':>8}nJ  (proven baseline)")
    for label, (acc, params, nJ) in results.items():
        print(f"  {label:>10} {acc:>9.2f}% {params:>8,} {nJ:>8.1f}nJ")
    print(f"\n  Total time: {time.time()-t0:.0f}s")
