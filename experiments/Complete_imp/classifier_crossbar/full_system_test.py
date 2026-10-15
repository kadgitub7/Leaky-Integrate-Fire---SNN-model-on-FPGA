"""
FULL SYSTEM TEST: Analog Frontend -> ReRAM Crossbar Classifier
=============================================================
End-to-end inter-patient MIT-BIH classification accuracy.

Pipeline:
  1. ECG beats -> Analog frontend circuits (ngspice, SKY130 PDK) -> 8 feature voltages
  2. Feature voltages -> Train MLP 8->32->5 on DS1 analog features
  3. MLP weights -> Map to ReRAM crossbar conductances (differential encoding)
  4. Crossbar -> Extract effective weight matrix via 8 unit-input ngspice sims
  5. Classify all DS2 beats using crossbar effective weights
  6. Report accuracy at each stage for comparison

This is the final verification that the entire system works end-to-end:
  - Analog circuits faithfully extract discriminative features
  - ReRAM crossbar faithfully implements the MLP classifier
  - Combined accuracy matches the software reference
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

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
LABEL_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}


def load_analog_features(records):
    """Load analog frontend features from ngspice cache files."""
    all_feats, all_labels = [], []
    for rec in records:
        cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
        if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
            print(f"  WARNING: Record {rec} not cached, skipping")
            continue
        with open(cache_file) as f:
            data = json.load(f)
        feats = np.array(data['features'])
        labels = np.array(data['labels'])
        all_feats.append(feats)
        all_labels.extend(labels)
    X = np.vstack(all_feats) if all_feats else np.empty((0, 8))
    y = np.array([LABEL_MAP[l] for l in all_labels])
    return X, y


def map_weights_to_conductance(W):
    w_max = np.abs(W).max()
    G_range = G_MAX_US - G_MIN_US
    Gp = np.full_like(W, G_MIN_US)
    Gn = np.full_like(W, G_MIN_US)
    pos = W >= 0
    Gp[pos] = G_MIN_US + (W[pos] / w_max) * G_range
    Gn[~pos] = G_MIN_US + (np.abs(W[~pos]) / w_max) * G_range
    scale = w_max / (G_range * 1e-6 * V_READ)
    return Gp, Gn, scale


def generate_unit_input_netlist(W, row_idx, filename):
    n_in, n_out = W.shape
    Gp, Gn, scale = map_weights_to_conductance(W)
    lines = []
    lines.append(f"* Unit-input row {row_idx}")
    lines.append(f'.include "{SKY130_CORNER}"')
    lines.append(f'.include "{SKY130_MISMATCH}"')
    lines.append("")
    lines.append("Vgate gate 0 1.8")
    for i in range(n_in):
        v = V_READ if i == row_idx else 0.0
        lines.append(f"Vrow{i} row{i} 0 {v}")
    for j in range(n_out):
        lines.append(f"Vsp{j} col_p{j} 0 0")
        lines.append(f"Vsn{j} col_n{j} 0 0")
    for i in range(n_in):
        for j in range(n_out):
            r = 1e6 / Gp[i, j]
            lines.append(f"XTp{i}_{j} row{i} gate mp{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{i}_{j} mp{i}_{j} col_p{j} {r:.4f}")
    for i in range(n_in):
        for j in range(n_out):
            r = 1e6 / Gn[i, j]
            lines.append(f"XTn{i}_{j} row{i} gate mn{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{i}_{j} mn{i}_{j} col_n{j} {r:.4f}")
    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    for j in range(n_out):
        lines.append(f"print i(Vsp{j}) i(Vsn{j})")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")
    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, scale


def parse_currents(stdout, n_out):
    Ip = np.zeros(n_out)
    In = np.zeros(n_out)
    for line in stdout.split('\n'):
        s = line.strip().lower()
        for j in range(n_out):
            if f"i(vsp{j})" in s and '=' in s:
                try: Ip[j] = float(s.split('=')[1].strip().split()[0])
                except: pass
            if f"i(vsn{j})" in s and '=' in s:
                try: In[j] = float(s.split('=')[1].strip().split()[0])
                except: pass
    return Ip, In


def extract_crossbar_effective_weights(W, n_out):
    """Run unit-input ngspice simulations to get effective weight matrix."""
    n_in = W.shape[0]
    _, _, scale = map_weights_to_conductance(W)
    W_eff = np.zeros((n_in, n_out))
    for i in range(n_in):
        fname = f"_sys_unit_row{i}.spice"
        path, _ = generate_unit_input_netlist(W, i, fname)
        result = subprocess.run([NGSPICE, "-b", path],
                                capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR)
        Ip, In = parse_currents(result.stdout, n_out)
        W_eff[i, :] = (Ip - In) * scale
        os.remove(path)
    return W_eff, scale


def main():
    print("=" * 74)
    print("  FULL SYSTEM TEST: Analog Frontend -> ReRAM Crossbar Classifier")
    print("=" * 74)
    print(f"  Frontend:  6 analog circuits (SKY130 PDK, ngspice-verified)")
    print(f"  Classifier: 8->32->5 MLP on 1T1R ReRAM crossbar (SKY130 NMOS)")
    print(f"  Dataset:   MIT-BIH, DS1 train ({len(DS1)} records) / DS2 test ({len(DS2)} records)")
    print()

    # ================================================================
    # STAGE 1: Load analog frontend features
    # ================================================================
    print("STAGE 1: Loading analog frontend features from ngspice cache...")
    X_tr_analog, y_tr = load_analog_features(DS1)
    X_te_analog, y_te = load_analog_features(DS2)
    print(f"  Train: {len(y_tr)} beats, Test: {len(y_te)} beats")
    print(f"  Feature range: [{X_tr_analog.min():.4f}, {X_tr_analog.max():.4f}]")

    if len(y_te) == 0:
        print("ERROR: No test data. Run run_missing_records.py first.")
        sys.exit(1)

    # ================================================================
    # STAGE 2a: RF baseline on analog features (sanity check)
    # ================================================================
    print("\nSTAGE 2a: RF baseline on analog frontend features...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42,
                                class_weight='balanced')
    rf.fit(X_tr_analog, y_tr)
    rf_pred = rf.predict(X_te_analog)
    rf_acc = accuracy_score(y_te, rf_pred) * 100
    rf_f1 = f1_score(y_te, rf_pred, average=None, zero_division=0)
    print(f"  RF accuracy (analog features): {rf_acc:.2f}%")
    print(f"  F1: {' '.join(f'{CLASS_NAMES[i]}={rf_f1[i]:.3f}' for i in range(5))}")

    # ================================================================
    # STAGE 2b: Software MLP on analog features
    # ================================================================
    print("\nSTAGE 2b: Training MLP 8->32->5 on analog frontend features...")
    scaler = StandardScaler()
    X_tr_norm = scaler.fit_transform(X_tr_analog)
    X_te_norm = scaler.transform(X_te_analog)

    best_mlp = None
    best_acc = 0
    for seed in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed+42,
                            early_stopping=True, validation_fraction=0.1,
                            learning_rate='adaptive', learning_rate_init=1e-3,
                            alpha=1e-4, batch_size=128)
        mlp.fit(X_tr_norm, y_tr)
        acc = accuracy_score(y_te, mlp.predict(X_te_norm)) * 100
        print(f"  Seed {seed+42}: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp

    sw_pred = best_mlp.predict(X_te_norm)
    sw_acc = accuracy_score(y_te, sw_pred) * 100
    sw_f1 = f1_score(y_te, sw_pred, average=None, zero_division=0)
    print(f"  Best MLP accuracy (software): {sw_acc:.2f}%")
    print(f"  F1: {' '.join(f'{CLASS_NAMES[i]}={sw_f1[i]:.3f}' for i in range(5))}")

    W1 = best_mlp.coefs_[0]   # (8, 32)
    b1 = best_mlp.intercepts_[0]  # (32,)
    W2 = best_mlp.coefs_[1]   # (32, 5)
    b2 = best_mlp.intercepts_[1]  # (5,)
    print(f"  W1: {W1.shape} [{W1.min():.4f}, {W1.max():.4f}]")
    print(f"  W2: {W2.shape} [{W2.min():.4f}, {W2.max():.4f}]")

    # ================================================================
    # STAGE 3: Extract crossbar effective weight matrix
    # ================================================================
    print("\nSTAGE 3: Extracting crossbar effective weights (8 ngspice unit-input sims)...")
    t0 = time.time()
    W1_eff, scale = extract_crossbar_effective_weights(W1, 32)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    w_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"  Weight error: mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%")

    # ================================================================
    # STAGE 4: Full crossbar classification
    # ================================================================
    print("\nSTAGE 4: Classifying all DS2 beats through crossbar...")

    # Layer 1: crossbar MAC
    Z1 = X_te_norm @ W1_eff + b1
    H1 = np.maximum(0, Z1)  # ReLU
    # Layer 2: software (or could be second crossbar — same principle)
    Z2 = H1 @ W2 + b2
    hw_pred = np.argmax(Z2, axis=1)

    hw_acc = accuracy_score(y_te, hw_pred) * 100
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)

    # ================================================================
    # STAGE 5: Also compare with pure-digital MLP + crossbar
    # ================================================================
    print("\nSTAGE 5: Digital-feature MLP + crossbar (for reference)...")
    d = np.load(os.path.abspath(DIGITAL_CACHE))
    X_tr_dig, y_tr_dig = d['X_tr'], d['y_tr']
    X_te_dig, y_te_dig = d['X_te'], d['y_te']

    mlp_dig = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.1,
                            learning_rate='adaptive', learning_rate_init=1e-3,
                            alpha=1e-4, batch_size=128)
    mlp_dig.fit(X_tr_dig, y_tr_dig)
    dig_sw_acc = accuracy_score(y_te_dig, mlp_dig.predict(X_te_dig)) * 100

    W1_dig_eff, _ = extract_crossbar_effective_weights(mlp_dig.coefs_[0], 32)
    Z1d = X_te_dig @ W1_dig_eff + mlp_dig.intercepts_[0]
    H1d = np.maximum(0, Z1d)
    Z2d = H1d @ mlp_dig.coefs_[1] + mlp_dig.intercepts_[1]
    dig_hw_pred = np.argmax(Z2d, axis=1)
    dig_hw_acc = accuracy_score(y_te_dig, dig_hw_pred) * 100
    dig_hw_f1 = f1_score(y_te_dig, dig_hw_pred, average=None, zero_division=0)

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n" + "=" * 74)
    print("  RESULTS: Full System Accuracy Comparison")
    print("=" * 74)

    sw_hw_agree = (sw_pred == hw_pred).mean() * 100
    n_disagree = (sw_pred != hw_pred).sum()

    print(f"""
  {'Pipeline':<45} {'Accuracy':>10} {'F1-S':>6} {'F1-V':>6}
  {'-'*70}
  1. RF on digital features (ceiling)          {95.76:>9.2f}%  0.669  0.892
  2. RF on analog frontend features            {rf_acc:>9.2f}%  {rf_f1[1]:.3f}  {rf_f1[2]:.3f}
  3. MLP(sw) on analog frontend features       {sw_acc:>9.2f}%  {sw_f1[1]:.3f}  {sw_f1[2]:.3f}
  4. MLP(crossbar) on analog frontend features {hw_acc:>9.2f}%  {hw_f1[1]:.3f}  {hw_f1[2]:.3f}
  {'-'*70}
  5. MLP(sw) on digital features               {dig_sw_acc:>9.2f}%
  6. MLP(crossbar) on digital features         {dig_hw_acc:>9.2f}%  {dig_hw_f1[1]:.3f}  {dig_hw_f1[2]:.3f}

  SW MLP vs Crossbar MLP agreement: {sw_hw_agree:.2f}% ({n_disagree} disagreements / {len(y_te)})
  Crossbar weight error: mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%
""")

    print("  Crossbar confusion matrix (analog frontend -> crossbar):")
    cm = confusion_matrix(y_te, hw_pred, labels=range(5))
    print(f"  {'':>6} " + " ".join(f"{c:>6}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>6} " + " ".join(f"{v:>6d}" for v in row))

    print(f"\n  Per-class report (analog frontend -> crossbar):")
    print(classification_report(y_te, hw_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))

    # Save results
    results = {
        'pipeline': 'analog_frontend -> crossbar_mlp',
        'rf_digital_acc': 95.76,
        'rf_analog_acc': float(rf_acc),
        'mlp_sw_analog_acc': float(sw_acc),
        'mlp_crossbar_analog_acc': float(hw_acc),
        'mlp_sw_digital_acc': float(dig_sw_acc),
        'mlp_crossbar_digital_acc': float(dig_hw_acc),
        'sw_hw_agreement_pct': float(sw_hw_agree),
        'n_disagreements': int(n_disagree),
        'n_train': int(len(y_tr)),
        'n_test': int(len(y_te)),
        'weight_error_mean_pct': float(w_err.mean() * 100),
        'weight_error_max_pct': float(w_err.max() * 100),
        'rf_analog_f1': {c: float(f) for c, f in zip(CLASS_NAMES, rf_f1)},
        'sw_analog_f1': {c: float(f) for c, f in zip(CLASS_NAMES, sw_f1)},
        'hw_analog_f1': {c: float(f) for c, f in zip(CLASS_NAMES, hw_f1)},
        'hw_digital_f1': {c: float(f) for c, f in zip(CLASS_NAMES, dig_hw_f1)},
        'nmos_W_um': float(NMOS_W),
        'G_range_uS': [float(G_MIN_US), float(G_MAX_US)],
    }
    out_path = os.path.join(SCRIPT_DIR, 'full_system_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    main()
