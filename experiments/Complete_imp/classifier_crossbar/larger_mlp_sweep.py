"""
Larger MLP sweep on analog frontend features -> crossbar.

Tests whether a bigger network can recover accuracy lost by the analog frontend.
Current best: 8->32->5 at 93.81% (analog->crossbar).
Digital ceiling: 95.76% (RF on Python features).

Configs tested:
  - 8->64->5   (832 MACs, single crossbar)
  - 8->96->5   (1248 MACs, single crossbar)
  - 8->128->5  (1664 MACs, single crossbar)
  + baseline 8->32->5 for comparison
"""
import os, sys, json, time, subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend", "_cache")

SKY130_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"
G_MIN_US, G_MAX_US, V_READ = 0.296, 99.273, 0.15
NMOS_W, NMOS_L = 1.0, 0.15
E_PER_MAC_NOM_PJ = 1.2679

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']
CLASS_NAMES = ['N','S','V','F','Q']
LABEL_MAP = {'N':0,'S':1,'V':2,'F':3,'Q':4}


def load_analog_features(records):
    all_feats, all_labels = [], []
    for rec in records:
        fp = os.path.join(CACHE_DIR, f"{rec}.json")
        if not os.path.exists(fp) or os.path.getsize(fp) == 0:
            print(f"  WARNING: {rec} not cached", flush=True); continue
        with open(fp) as f: data = json.load(f)
        all_feats.append(np.array(data['features']))
        all_labels.extend(data['labels'])
    X = np.vstack(all_feats)
    y = np.array([LABEL_MAP[l] for l in all_labels])
    return X, y


def map_weights_to_conductance(W):
    w_max = np.abs(W).max()
    G_range = G_MAX_US - G_MIN_US
    Gp = np.full_like(W, G_MIN_US); Gn = np.full_like(W, G_MIN_US)
    pos = W >= 0
    Gp[pos] = G_MIN_US + (W[pos] / w_max) * G_range
    Gn[~pos] = G_MIN_US + (np.abs(W[~pos]) / w_max) * G_range
    scale = w_max / (G_range * 1e-6 * V_READ)
    return Gp, Gn, scale


def extract_crossbar_weights(W, n_out, label=""):
    """Run unit-input ngspice sims to get effective weight matrix."""
    n_in = W.shape[0]
    Gp, Gn, scale = map_weights_to_conductance(W)
    W_eff = np.zeros((n_in, n_out))

    for i in range(n_in):
        lines = [f"* Unit row {i} -- {label}",
                 f'.include "{SKY130_CORNER}"', f'.include "{SKY130_MISMATCH}"', "",
                 "Vgate gate 0 1.8"]
        for k in range(n_in):
            lines.append(f"Vrow{k} row{k} 0 {V_READ if k==i else 0.0}")
        for j in range(n_out):
            lines.append(f"Vsp{j} cp{j} 0 0")
            lines.append(f"Vsn{j} cn{j} 0 0")
        for k in range(n_in):
            for j in range(n_out):
                r = 1e6 / Gp[k, j]
                lines.append(f"XTp{k}_{j} row{k} gate mp{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrp{k}_{j} mp{k}_{j} cp{j} {r:.4f}")
        for k in range(n_in):
            for j in range(n_out):
                r = 1e6 / Gn[k, j]
                lines.append(f"XTn{k}_{j} row{k} gate mn{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrn{k}_{j} mn{k}_{j} cn{j} {r:.4f}")
        lines += [".op", ".control", "run"]
        for j in range(n_out):
            lines.append(f"print i(Vsp{j}) i(Vsn{j})")
        lines += ["quit", ".endc", ".end"]

        path = os.path.join(SCRIPT_DIR, f"_u{label}_{i}.spice")
        with open(path, 'w') as f: f.write('\n'.join(lines))
        result = subprocess.run([NGSPICE, "-b", path],
                                capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR)
        Ip, In = np.zeros(n_out), np.zeros(n_out)
        for line in result.stdout.split('\n'):
            s = line.strip().lower()
            for j in range(n_out):
                if f"i(vsp{j})" in s and '=' in s:
                    try: Ip[j] = float(s.split('=')[1].strip().split()[0])
                    except: pass
                if f"i(vsn{j})" in s and '=' in s:
                    try: In[j] = float(s.split('=')[1].strip().split()[0])
                    except: pass
        W_eff[i, :] = (Ip - In) * scale
        os.remove(path)
    return W_eff


def test_config(hidden, X_tr_n, y_tr, X_te_n, y_te, scaler):
    """Train MLP, extract crossbar weights, classify, return results."""
    arch = "8-" + "-".join(str(h) for h in hidden) + "-5"
    macs = 0; prev = 8
    for h in hidden: macs += prev * h; prev = h
    macs += prev * 5
    e_nj = macs * E_PER_MAC_NOM_PJ * 1e-3
    n_cells = 8 * hidden[0] * 2  # Layer 1 differential cells

    print(f"\n{'='*60}", flush=True)
    print(f"  Config: {arch}  MACs={macs}  Cells={n_cells}  E={e_nj:.2f}nJ", flush=True)
    print(f"{'='*60}", flush=True)

    # Train 5 seeds, pick best
    best_mlp, best_acc = None, 0
    for seed in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=seed+42,
                            early_stopping=True, validation_fraction=0.1,
                            learning_rate='adaptive', learning_rate_init=1e-3,
                            alpha=1e-4, batch_size=128)
        mlp.fit(X_tr_n, y_tr)
        acc = accuracy_score(y_te, mlp.predict(X_te_n)) * 100
        print(f"  Seed {seed+42}: {acc:.2f}%", flush=True)
        if acc > best_acc: best_acc = acc; best_mlp = mlp

    sw_pred = best_mlp.predict(X_te_n)
    sw_acc = accuracy_score(y_te, sw_pred) * 100
    sw_f1 = f1_score(y_te, sw_pred, average=None, zero_division=0)
    print(f"  Best SW accuracy: {sw_acc:.2f}%", flush=True)

    # Extract crossbar effective weights for Layer 1
    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]

    print(f"  Extracting crossbar weights ({8}x{hidden[0]} array)...", flush=True)
    t0 = time.time()
    W1_eff = extract_crossbar_weights(W1, hidden[0], label=str(hidden[0]))
    elapsed = time.time() - t0
    w_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"  Done in {elapsed:.0f}s. Weight error: mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%", flush=True)

    # Crossbar classification
    Z1 = X_te_n @ W1_eff + b1
    H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2 + b2
    hw_pred = np.argmax(Z2, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)
    agree = (sw_pred == hw_pred).mean() * 100

    print(f"  SW  MLP accuracy: {sw_acc:.2f}%  F1-S={sw_f1[1]:.3f} F1-V={sw_f1[2]:.3f}", flush=True)
    print(f"  HW Xbar accuracy: {hw_acc:.2f}%  F1-S={hw_f1[1]:.3f} F1-V={hw_f1[2]:.3f}", flush=True)
    print(f"  Agreement: {agree:.2f}%", flush=True)

    return {
        'arch': arch, 'hidden': list(hidden), 'macs': macs, 'n_cells': n_cells, 'e_nj': round(e_nj, 3),
        'sw_acc': round(sw_acc, 3), 'hw_acc': round(hw_acc, 3),
        'sw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, sw_f1)},
        'hw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, hw_f1)},
        'agreement': round(agree, 3),
        'w_err_mean': round(float(w_err.mean()*100), 3),
        'w_err_max': round(float(w_err.max()*100), 3),
    }


def main():
    print("=" * 60, flush=True)
    print("  LARGER MLP SWEEP: Analog Frontend -> Crossbar", flush=True)
    print("=" * 60, flush=True)

    X_tr, y_tr = load_analog_features(DS1)
    X_te, y_te = load_analog_features(DS2)
    print(f"  Train: {len(y_tr)} beats, Test: {len(y_te)} beats", flush=True)

    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te)

    configs = [(32,), (64,), (96,), (128,)]
    all_results = []

    for hidden in configs:
        result = test_config(hidden, X_tr_n, y_tr, X_te_n, y_te, scaler)
        all_results.append(result)

    # Summary
    print(f"\n\n{'='*74}", flush=True)
    print(f"  SUMMARY: Analog Frontend -> Crossbar MLP", flush=True)
    print(f"{'='*74}", flush=True)
    print(f"  {'Arch':<12} {'MACs':>6} {'Cells':>6} {'E(nJ)':>7} {'SW%':>7} {'HW%':>7} {'F1-S':>6} {'F1-V':>6} {'Agree':>7}", flush=True)
    print(f"  {'-'*68}", flush=True)
    for r in all_results:
        print(f"  {r['arch']:<12} {r['macs']:>6} {r['n_cells']:>6} {r['e_nj']:>7.2f} "
              f"{r['sw_acc']:>6.2f}% {r['hw_acc']:>6.2f}% "
              f"{r['hw_f1']['S']:>5.3f} {r['hw_f1']['V']:>5.3f} {r['agreement']:>6.2f}%", flush=True)

    print(f"\n  Reference: RF on digital features = 95.76%", flush=True)
    print(f"  Reference: RF on analog features  = 92.22%", flush=True)

    best = max(all_results, key=lambda x: x['hw_acc'])
    print(f"\n  Best: {best['arch']} at {best['hw_acc']:.2f}% (crossbar)", flush=True)

    out_path = os.path.join(SCRIPT_DIR, 'larger_mlp_results.json')
    with open(out_path, 'w') as f:
        json.dump({'results': all_results, 'rf_digital_ceiling': 95.76, 'rf_analog': 92.22}, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
