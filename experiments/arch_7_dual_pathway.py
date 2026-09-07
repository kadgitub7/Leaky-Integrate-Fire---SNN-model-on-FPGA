"""
ARCH 7: Dual-Pathway SNN — Mirrors how cardiologists process beats.

CLINICAL INSIGHT: Cardiologists use a TWO-STEP decision process:
  Path A: CHECK RHYTHM (timing features) → "Is it premature? Compensatory pause?"
  Path B: CHECK MORPHOLOGY (shape features) → "Is QRS wide? Aberrant? Cross-lead concordant?"

Then MERGE: rhythm context + morphology evidence → classification.

This architecture gives each pathway DIFFERENT neuron dynamics:
  - Timing pathway: FAST decay (beta=0.7) — rhythm is transient, act quickly
  - Morphology pathway: SLOW decay (beta=0.95) — shape evidence accumulates

Hardware: Two small crossbar arrays + merger. Same transistor count as one big array.
Energy: ~35 nJ total (two parallel small arrays cheaper than one large one)
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.shared_data import load_data, make_loaders, NUM_FEATURES, NUM_CLASSES

torch.set_num_threads(2)
t0 = time.time()

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

# Feature indices for each pathway
TIMING_IDX = [0, 1, 2, 3, 4]       # pre_rr, post_rr, rr_ratio, rr_asymmetry, rr_std_10
MORPH_IDX  = [5, 6, 7, 8, 9, 10, 11, 12, 13]  # qrs_area, max_slope, slope_ratio, rel_area, templ_corr, rel_peak (L0+L1), corr_L0L1

class DualPathwaySNN(nn.Module):
    """
    Two parallel pathways with different temporal dynamics:
      Timing:     5 → 16 RLeaky(beta=0.7,fast)  → merge
      Morphology: 9 → 24 RLeaky(beta=0.95,slow) → merge
      Merged:     40 → 16 RLeaky → 5 Leaky
    """
    def __init__(self, n_out=5):
        super().__init__()
        n_timing, n_morph = len(TIMING_IDX), len(MORPH_IDX)
        h_t, h_m, h_merge = 16, 24, 16

        # Pathway A: Timing (fast dynamics)
        self.bn_t = nn.BatchNorm1d(n_timing)
        self.fc_t = nn.Linear(n_timing, h_t)
        self.drop_t = nn.Dropout(0.1)
        self.rlif_t = snn.RLeaky(beta=0.7, linear_features=h_t, learn_beta=True, learn_threshold=True)

        # Pathway B: Morphology (slow dynamics)
        self.bn_m = nn.BatchNorm1d(n_morph)
        self.fc_m = nn.Linear(n_morph, h_m)
        self.drop_m = nn.Dropout(0.1)
        self.rlif_m = snn.RLeaky(beta=0.95, linear_features=h_m, learn_beta=True, learn_threshold=True)

        # Merger: combines both pathways
        self.fc_merge = nn.Linear(h_t + h_m, h_merge)
        self.drop_merge = nn.Dropout(0.1)
        self.rlif_merge = snn.RLeaky(beta=0.9, linear_features=h_merge, learn_beta=True, learn_threshold=True)
        self.fc_out = nn.Linear(h_merge, n_out)
        self.lif_out = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)

        self.h1 = h_t + h_m  # for energy calc
        self.h2 = h_merge

    def forward(self, x, steps=20):
        st, mt = self.rlif_t.init_rleaky()
        sm, mm = self.rlif_m.init_rleaky()
        s_mrg, m_mrg = self.rlif_merge.init_rleaky()
        mo = self.lif_out.init_leaky()
        mem_rec, spk_rec = [], []

        x_t = self.drop_t(self.fc_t(self.bn_t(x[:, TIMING_IDX])))
        x_m = self.drop_m(self.fc_m(self.bn_m(x[:, MORPH_IDX])))

        for _ in range(steps):
            st, mt = self.rlif_t(x_t, st, mt)
            sm, mm = self.rlif_m(x_m, sm, mm)
            merged = torch.cat([st, sm], dim=1)
            spk_rec.append(merged)
            s_mrg, m_mrg = self.rlif_merge(self.drop_merge(self.fc_merge(merged)), s_mrg, m_mrg)
            _, mo = self.lif_out(self.fc_out(s_mrg), mo)
            mem_rec.append(mo)

        return torch.stack(mem_rec), torch.stack(spk_rec)

def run():
    data = load_data()
    tr_loader, val_loader, te_loader, cw = make_loaders(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw = cw.to(device)

    torch.manual_seed(42); np.random.seed(42)
    model = DualPathwaySNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[ARCH 7: Dual-Pathway (Timing→16fast + Morph→24slow → 16merge)] Params: {n_params:,}")

    loss_fn = FocalLoss(weight=cw, gamma=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup, epochs, steps = 5, 300, 20
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs-warmup)

    best_val, best_state, patience = 0.0, None, 0
    for ep in range(epochs):
        if ep < warmup:
            for pg in opt.param_groups: pg['lr'] = 1e-3*(ep+1)/warmup
        model.train()
        ep_loss, batches = 0, 0
        for feat, tgt in tr_loader:
            feat, tgt = feat.to(device), tgt.to(device)
            saved = [p.data.clone() for p in model.parameters()]
            with torch.no_grad():
                for p in model.parameters(): p.data.copy_(quantize(p.data))
            mem, spk = model(feat, steps)
            ce = sum(loss_fn(mem[s], tgt) for s in range(steps))
            loss = ce + 1.0*torch.clamp(spk.mean()-0.15, min=0)
            opt.zero_grad(); loss.backward()
            with torch.no_grad():
                for p, s in zip(model.parameters(), saved): p.data.copy_(s)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += ce.item(); batches += 1
        if ep >= warmup: sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for f, t in val_loader:
                f, t = f.to(device), t.to(device)
                m, _ = model(f, steps)
                correct += (m[-1].argmax(1)==t).sum().item(); total += t.size(0)
        va = correct/total*100
        if va > best_val: best_val, best_state, patience = va, copy.deepcopy(model.state_dict()), 0
        else: patience += 1
        if (ep+1)%30==0 or ep==0:
            print(f"  Ep {ep+1:3d} | CE {ep_loss/batches:.3f} | val {va:.2f}% (best {best_val:.2f}%) | {time.time()-t0:.0f}s")
        if patience >= 40 and ep >= 50:
            print(f"  Early stop ep {ep+1}"); break

    model.load_state_dict(best_state)
    with torch.no_grad():
        for p in model.parameters(): p.data.copy_(quantize(p.data))

    model.eval()
    all_p, all_t, tot_spk, tot_pos = [], [], 0, 0
    with torch.no_grad():
        for f, t in te_loader:
            f, t = f.to(device), t.to(device)
            m, sp = model(f, steps)
            all_p.extend(m[-1].argmax(1).cpu().numpy())
            all_t.extend(t.cpu().numpy())
            tot_spk += sp.sum().item(); tot_pos += sp.numel()
    fire = tot_spk/tot_pos
    acc = np.mean(np.array(all_p)==np.array(all_t))*100

    # Energy: two small pathways + merger
    timing_macs = 5*16 + 16*16*steps*fire
    morph_macs = 9*24 + 24*24*steps*fire
    merge_macs = 40*16*steps*fire + 16*16*steps*fire + 16*5*steps*fire
    total_macs = timing_macs + morph_macs + merge_macs
    nJ = total_macs*2.0/1000 + 13.7

    print(f"\n{'='*60}")
    print(f"ARCH 7 RESULT: {acc:.2f}% inter-patient | {n_params:,} params | {nJ:.1f} nJ")
    print(f"  Fire rate: {fire:.3f}")
    print(f"{'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q']))
    cm = confusion_matrix(all_t, all_p)
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")

if __name__ == '__main__':
    run()
