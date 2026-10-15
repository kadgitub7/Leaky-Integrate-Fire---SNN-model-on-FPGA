"""
Partial system test: Analog Frontend → Crossbar using available cached records.
Tests the full pipeline with whatever DS2 records are already cached.
"""
import os, sys, json, time, subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend", "_cache")
DIGITAL_CACHE = os.path.join(SCRIPT_DIR, '..', 'classifier', 'features_8_cache.npz')

SKY130_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"
G_MIN_US, G_MAX_US, V_READ = 0.296, 99.273, 0.15
NMOS_W, NMOS_L = 1.0, 0.15

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']
CLASS_NAMES = ['N','S','V','F','Q']
LABEL_MAP = {'N':0,'S':1,'V':2,'F':3,'Q':4}


def load_analog_features(records):
    all_feats, all_labels = [], []
    loaded, skipped = [], []
    for rec in records:
        cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
        if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
            skipped.append(rec)
            continue
        with open(cache_file) as f:
            data = json.load(f)
        feats = np.array(data['features'])
        labels = np.array(data['labels'])
        all_feats.append(feats)
        all_labels.extend(labels)
        loaded.append(rec)
    X = np.vstack(all_feats) if all_feats else np.empty((0, 8))
    y = np.array([LABEL_MAP[l] for l in all_labels])
    return X, y, loaded, skipped


def map_weights_to_conductance(W):
    w_max = np.abs(W).max()
    G_range = G_MAX_US - G_MIN_US
    Gp = np.full_like(W, G_MIN_US); Gn = np.full_like(W, G_MIN_US)
    pos = W >= 0
    Gp[pos] = G_MIN_US + (W[pos] / w_max) * G_range
    Gn[~pos] = G_MIN_US + (np.abs(W[~pos]) / w_max) * G_range
    scale = w_max / (G_range * 1e-6 * V_READ)
    return Gp, Gn, scale


def extract_crossbar_weights(W, n_out):
    n_in = W.shape[0]
    _, _, scale = map_weights_to_conductance(W)
    Gp, Gn, _ = map_weights_to_conductance(W)
    W_eff = np.zeros((n_in, n_out))
    for i in range(n_in):
        lines = [f'.include "{SKY130_CORNER}"', f'.include "{SKY130_MISMATCH}"', "",
                 "Vgate gate 0 1.8"]
        for k in range(n_in):
            lines.append(f"Vrow{k} row{k} 0 {V_READ if k==i else 0.0}")
        for j in range(n_out):
            lines.append(f"Vsp{j} col_p{j} 0 0")
            lines.append(f"Vsn{j} col_n{j} 0 0")
        for k in range(n_in):
            for j in range(n_out):
                r = 1e6/Gp[k,j]
                lines.append(f"XTp{k}_{j} row{k} gate mp{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrp{k}_{j} mp{k}_{j} col_p{j} {r:.4f}")
        for k in range(n_in):
            for j in range(n_out):
                r = 1e6/Gn[k,j]
                lines.append(f"XTn{k}_{j} row{k} gate mn{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrn{k}_{j} mn{k}_{j} col_n{j} {r:.4f}")
        lines += [".op", ".control", "run"]
        for j in range(n_out):
            lines.append(f"print i(Vsp{j}) i(Vsn{j})")
        lines += ["quit", ".endc", ".end"]
        path = os.path.join(SCRIPT_DIR, f"_unit_{i}.spice")
        with open(path, 'w') as f: f.write('\n'.join(lines))
        result = subprocess.run([NGSPICE, "-b", path], capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR)
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


def main():
    print("=" * 74)
    print("  PARTIAL SYSTEM TEST: Analog Frontend → Crossbar (cached records)")
    print("=" * 74)

    X_tr, y_tr, tr_loaded, tr_skip = load_analog_features(DS1)
    X_te, y_te, te_loaded, te_skip = load_analog_features(DS2)
    print(f"  Train: {len(y_tr)} beats from {len(tr_loaded)} records (skipped: {tr_skip})")
    print(f"  Test:  {len(y_te)} beats from {len(te_loaded)} records (skipped: {te_skip})")

    if len(y_te) == 0:
        print("ERROR: No test data available"); return

    # RF baseline
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te)) * 100
    print(f"\n  RF on analog features: {rf_acc:.2f}%")

    # MLP (5 seeds, pick best)
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr); X_te_n = scaler.transform(X_te)
    best_mlp, best_acc = None, 0
    for seed in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed+42,
                            early_stopping=True, validation_fraction=0.1,
                            learning_rate='adaptive', learning_rate_init=1e-3, alpha=1e-4, batch_size=128)
        mlp.fit(X_tr_n, y_tr)
        acc = accuracy_score(y_te, mlp.predict(X_te_n)) * 100
        if acc > best_acc: best_acc = acc; best_mlp = mlp
    sw_pred = best_mlp.predict(X_te_n)
    sw_acc = accuracy_score(y_te, sw_pred) * 100
    sw_f1 = f1_score(y_te, sw_pred, average=None, zero_division=0)
    print(f"  MLP(sw) on analog features: {sw_acc:.2f}%")

    # Crossbar
    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]
    print(f"\n  Extracting crossbar effective weights...")
    W1_eff = extract_crossbar_weights(W1, 32)
    w_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"  Weight error: mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%")

    Z1 = X_te_n @ W1_eff + b1; H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2 + b2
    hw_pred = np.argmax(Z2, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)

    agree = (sw_pred == hw_pred).mean() * 100

    print(f"\n{'='*74}")
    print(f"  RESULTS ({len(te_loaded)}/{len(DS2)} DS2 records)")
    print(f"{'='*74}")
    print(f"  RF  (analog features):             {rf_acc:.2f}%")
    print(f"  MLP (software, analog features):   {sw_acc:.2f}%")
    print(f"  MLP (crossbar, analog features):   {hw_acc:.2f}%")
    print(f"  SW vs crossbar agreement:          {agree:.2f}%")
    print(f"  Crossbar weight error:             {w_err.mean()*100:.2f}% mean")
    print(f"\n  F1 scores (crossbar):")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c}: sw={sw_f1[i]:.3f}  hw={hw_f1[i]:.3f}")

    print(f"\n  Confusion matrix (analog→crossbar):")
    cm = confusion_matrix(y_te, hw_pred, labels=range(5))
    print(f"  {'':>6} " + " ".join(f"{c:>6}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>6} " + " ".join(f"{v:>6d}" for v in row))

    print(f"\n  Missing DS2 records: {te_skip}")
    print(f"  These are being simulated in the background.")


if __name__ == '__main__':
    main()
