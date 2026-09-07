"""
ARCH 4: SNN-Temporal — The ONLY architecture that can genuinely EXCEED RF.

KEY INSIGHT: RF treats every beat independently. This SNN processes a SEQUENCE
of 5 consecutive beats, one per timestep. The recurrence learns beat-to-beat
dynamics that RF is completely blind to:
  - Post-PVC compensatory pause patterns
  - Morphology evolution across consecutive beats
  - Rhythm context (was the last beat also abnormal?)

Input: 5 consecutive beats' features fed sequentially (not repeated).
Step 1: beat[t-4] features → hidden state
Step 2: beat[t-3] features → updated hidden state
Step 3: beat[t-2] features → ...
Step 4: beat[t-1] features → ...
Step 5: beat[t]   features → classification

This is TRUE temporal processing — the recurrence does real sequential work.
Energy: ~30 nJ total (5 timesteps instead of 20, but each step is meaningful)
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
import snntorch as snn
from sklearn.metrics import classification_report, confusion_matrix
from experiments.shared_data import load_data, make_loaders, NUM_FEATURES, NUM_CLASSES

torch.set_num_threads(2)
t0 = time.time()
SEQ_LEN = 5

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

class SNNTemporal(nn.Module):
    """Processes SEQUENCE of consecutive beats — true temporal SNN."""
    def __init__(self, n_in=14, h1=48, h2=24, n_out=5, beta=0.85):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)
        self.fc1 = nn.Linear(n_in, h1)
        self.drop1 = nn.Dropout(0.1)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=h1, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(0.1)
        self.rlif2 = snn.RLeaky(beta=beta, linear_features=h2, learn_beta=True, learn_threshold=True)
        self.fc3 = nn.Linear(h2, n_out)
        self.lif_out = snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True)
        self.h1, self.h2 = h1, h2

    def forward(self, x_seq):
        """x_seq: (batch, seq_len, n_features) — sequence of consecutive beats."""
        batch, seq_len, n_feat = x_seq.shape
        s1, m1 = self.rlif1.init_rleaky()
        s2, m2 = self.rlif2.init_rleaky()
        mo = self.lif_out.init_leaky()
        mem_rec, spk_rec = [], []

        for t in range(seq_len):
            xt = self.bn(x_seq[:, t, :])  # (batch, n_feat)
            fc1 = self.drop1(self.fc1(xt))
            s1, m1 = self.rlif1(fc1, s1, m1)
            spk_rec.append(s1)
            s2, m2 = self.rlif2(self.drop2(self.fc2(s1)), s2, m2)
            _, mo = self.lif_out(self.fc3(s2), mo)
            mem_rec.append(mo)

        return torch.stack(mem_rec), torch.stack(spk_rec)

def run():
    data = load_data(return_sequences=True, seq_len=SEQ_LEN)
    _, _, te_loader_static, _ = make_loaders(data, seq=False)
    tr_loader, val_loader, te_loader, cw = make_loaders(data, seq=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw = cw.to(device)

    torch.manual_seed(42); np.random.seed(42)
    model = SNNTemporal().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[ARCH 4: SNN-Temporal {SEQ_LEN}-beat sequence] Params: {n_params:,}")

    loss_fn = FocalLoss(weight=cw, gamma=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup, epochs = 5, 300
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
            mem, spk = model(feat)
            ce = sum(loss_fn(mem[s], tgt) for s in range(SEQ_LEN))
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
                m, _ = model(f)
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
            m, sp = model(f)
            all_p.extend(m[-1].argmax(1).cpu().numpy())
            all_t.extend(t.cpu().numpy())
            tot_spk += sp.sum().item(); tot_pos += sp.numel()
    fire = tot_spk/tot_pos
    acc = np.mean(np.array(all_p)==np.array(all_t))*100

    fc_macs = 14*48
    rec_macs = 48*48*SEQ_LEN*fire + 48*24*SEQ_LEN*fire + 24*5*SEQ_LEN*fire
    nJ = (fc_macs+rec_macs)*2.0/1000 + 13.7

    print(f"\n{'='*60}")
    print(f"ARCH 4 RESULT: {acc:.2f}% inter-patient | {n_params:,} params | {nJ:.1f} nJ")
    print(f"  Fire rate: {fire:.3f} | SEQ_LEN: {SEQ_LEN}")
    print(f"{'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q']))
    cm = confusion_matrix(all_t, all_p)
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")

if __name__ == '__main__':
    run()
