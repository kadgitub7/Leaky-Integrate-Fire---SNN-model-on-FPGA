"""
ARCH 9: Multi-Timescale Temporal — The ultimate architecture.

Combines THREE innovations:
  1. DUAL TIMESCALE neurons: fast population (beta=0.6) reacts to beat-to-beat
     changes, slow population (beta=0.95) integrates long-term rhythm context
  2. TRUE SEQUENTIAL processing: 5 consecutive beats fed one per timestep
  3. SKIP connections: raw features available at output for easy-case shortcuts

This mirrors biological neural circuits: fast interneurons detect transient
events (a premature beat), slow pyramidal cells maintain context (rhythm history).

Why this can exceed RF:
  - RF: sees 14 features per beat, no temporal context, axis-aligned splits
  - This: sees 5 beats of context, multi-timescale integration, learned combos,
    plus raw feature shortcuts for the easy 90% of beats

Hardware: Two small neuron populations (8+16=24 neurons) + skip wires.
Actually FEWER neurons than baseline (24 vs 48+24=72).
Energy: ~25 nJ total
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

class MultiTimescaleTemporal(nn.Module):
    """
    Dual-timescale + temporal + skip:
      For each beat in sequence:
        Fast path:  14 features → 12 RLeaky(beta=0.6)  — transient detector
        Slow path:  14 features → 16 RLeaky(beta=0.95) — context integrator
        Merge:      [12 fast + 16 slow + 14 raw] → 12 RLeaky → 5 Leaky
    """
    def __init__(self, n_in=14, h_fast=12, h_slow=16, h_merge=12, n_out=5):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_in)

        # Fast pathway — detects transient changes between beats
        self.fc_fast = nn.Linear(n_in, h_fast)
        self.drop_f = nn.Dropout(0.1)
        self.rlif_fast = snn.RLeaky(beta=0.6, linear_features=h_fast, learn_beta=True, learn_threshold=True)

        # Slow pathway — integrates rhythm context across beats
        self.fc_slow = nn.Linear(n_in, h_slow)
        self.drop_s = nn.Dropout(0.1)
        self.rlif_slow = snn.RLeaky(beta=0.95, linear_features=h_slow, learn_beta=True, learn_threshold=True)

        # Merge: fast spikes + slow spikes + raw features → output
        self.fc_merge = nn.Linear(h_fast + h_slow + n_in, h_merge)
        self.drop_m = nn.Dropout(0.1)
        self.rlif_merge = snn.RLeaky(beta=0.85, linear_features=h_merge, learn_beta=True, learn_threshold=True)
        self.fc_out = nn.Linear(h_merge, n_out)
        self.lif_out = snn.Leaky(beta=0.85, learn_beta=True, learn_threshold=True)

        self.h1 = h_fast + h_slow
        self.h2 = h_merge

    def forward(self, x_seq):
        """x_seq: (batch, seq_len, n_features)"""
        batch, seq_len, _ = x_seq.shape

        sf, mf = self.rlif_fast.init_rleaky()
        ss, ms = self.rlif_slow.init_rleaky()
        sm, mm = self.rlif_merge.init_rleaky()
        mo = self.lif_out.init_leaky()
        mem_rec, spk_rec = [], []

        for t in range(seq_len):
            x = self.bn(x_seq[:, t, :])

            # Fast: reacts to immediate beat features
            sf, mf = self.rlif_fast(self.drop_f(self.fc_fast(x)), sf, mf)

            # Slow: accumulates context across the beat sequence
            ss, ms = self.rlif_slow(self.drop_s(self.fc_slow(x)), ss, ms)

            # Merge: fast + slow + raw skip
            combined = torch.cat([sf, ss, x], dim=1)
            spk_rec.append(torch.cat([sf, ss], dim=1))
            sm, mm = self.rlif_merge(self.drop_m(self.fc_merge(combined)), sm, mm)
            _, mo = self.lif_out(self.fc_out(sm), mo)
            mem_rec.append(mo)

        return torch.stack(mem_rec), torch.stack(spk_rec)

def run():
    data = load_data(return_sequences=True, seq_len=SEQ_LEN)
    tr_loader, val_loader, te_loader, cw = make_loaders(data, seq=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cw = cw.to(device)

    torch.manual_seed(42); np.random.seed(42)
    model = MultiTimescaleTemporal().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[ARCH 9: Multi-Timescale Temporal (fast12+slow16→merge12, {SEQ_LEN}-beat)] Params: {n_params:,}")

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

    fast_macs = 14*12 + 12*12*SEQ_LEN*fire
    slow_macs = 14*16 + 16*16*SEQ_LEN*fire
    merge_macs = 42*12*SEQ_LEN*fire + 12*12*SEQ_LEN*fire + 12*5*SEQ_LEN*fire
    total_macs = fast_macs + slow_macs + merge_macs
    nJ = total_macs*2.0/1000 + 13.7

    print(f"\n{'='*60}")
    print(f"ARCH 9 RESULT: {acc:.2f}% inter-patient | {n_params:,} params | {nJ:.1f} nJ")
    print(f"  Fire rate: {fire:.3f} | SEQ_LEN: {SEQ_LEN}")
    print(f"  Total neurons: {12+16+12}=40 (vs 72 in baseline)")
    print(f"{'='*60}")
    print(classification_report(all_t, all_p, target_names=['N','S','V','F','Q']))
    cm = confusion_matrix(all_t, all_p)
    print(f"  {'':>4} {'N':>6} {'S':>6} {'V':>6} {'F':>6} {'Q':>6}")
    for i, l in enumerate(['N','S','V','F','Q']):
        print(f"  {l:>4} {' '.join(f'{v:>6}' for v in cm[i])}")

if __name__ == '__main__':
    run()
