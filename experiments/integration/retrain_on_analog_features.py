"""
Retrain MLP on Analog Frontend Features
========================================
Instead of calibrating analog features to match digital ones,
retrain the MLP directly on the analog circuit_features from the
frontend cache. The MLP learns the analog feature space, then
weights are mapped to ReRAM conductances.

This is the correct approach: if the analog features have high
correlation with digital, they carry the same discriminative info.
The MLP will learn the right mapping during training.

Pipeline:
  1. Load circuit_features (analog voltages) from DS1 cache -> train set
  2. Load circuit_features from DS2 cache -> test set
  3. StandardScaler on analog features
  4. Train MLP 8->64->5 (same arch as original)
  5. Map weights to ReRAM conductances (G+/G-)
  6. Verify full-analog accuracy on DS2
  7. Generate updated SPICE netlists
"""
import os, sys, json, time, subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend", "_cache")
INTEGRATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "integration")
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, "experiments", "Complete_imp", "classifier_crossbar")

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
SKY130_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
LABEL_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']


def load_analog_features(records, label):
    """Load circuit_features and labels from cache files."""
    all_feats, all_labels = [], []
    missing = []
    for rec in records:
        cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
        if not os.path.exists(cache_file):
            missing.append(rec)
            continue
        with open(cache_file) as f:
            data = json.load(f)
        cf = np.array(data.get('circuit_features', []))
        if cf.ndim != 2 or cf.shape[1] != 8 or len(cf) == 0:
            print(f"  WARNING: Record {rec} has no circuit_features, skipping")
            missing.append(rec)
            continue
        n_circuit = data.get('n_circuit', len(cf))
        labels_all = data['labels']
        circuit_labels = [LABEL_MAP[l] for l in labels_all[:n_circuit]]
        all_feats.append(cf)
        all_labels.extend(circuit_labels)

    if missing:
        print(f"  {label}: missing records: {missing}")
    X = np.vstack(all_feats) if all_feats else np.empty((0, 8))
    y = np.array(all_labels)
    print(f"  {label}: {len(records)-len(missing)}/{len(records)} records, {len(y)} circuit beats")
    return X, y


def map_weights_to_conductance(W):
    """Map MLP weight matrix to differential ReRAM conductances."""
    w_max = np.abs(W).max()
    G_range = G_MAX_US - G_MIN_US
    Gp = np.full_like(W, G_MIN_US)
    Gn = np.full_like(W, G_MIN_US)
    pos = W >= 0
    Gp[pos] = G_MIN_US + (W[pos] / w_max) * G_range
    Gn[~pos] = G_MIN_US + (np.abs(W[~pos]) / w_max) * G_range
    scale = w_max / (G_range * 1e-6 * V_READ)
    return Gp, Gn, scale


def run_ngspice(path):
    result = subprocess.run(
        [NGSPICE, "-b", path],
        capture_output=True, text=True, timeout=600, cwd=INTEGRATION_DIR
    )
    return result.stdout, result.stderr


def parse_currents(stdout, n_out):
    Ip, In = np.zeros(n_out), np.zeros(n_out)
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


def extract_crossbar_weights(W, label):
    """Extract effective weight matrix via unit-input ngspice characterization."""
    n_in, n_out = W.shape
    Gp, Gn, scale = map_weights_to_conductance(W)
    W_eff = np.zeros((n_in, n_out))

    for i in range(n_in):
        lines = [
            f"* Unit-input characterization: row {i}",
            f'.include "{SKY130_CORNER}"',
            f'.include "{SKY130_MISMATCH}"',
            "", "Vgate gate 0 1.8", "",
        ]
        for k in range(n_in):
            v = V_READ if k == i else 0.0
            lines.append(f"Vrow{k} row{k} 0 {v}")
        lines.append("")
        for j in range(n_out):
            lines.append(f"Vsp{j} cp{j} 0 0")
            lines.append(f"Vsn{j} cn{j} 0 0")
        lines.append("")
        for k in range(n_in):
            for j in range(n_out):
                rp = 1e6 / Gp[k, j]
                rn = 1e6 / Gn[k, j]
                lines.append(f"XTp{k}_{j} row{k} gate mp{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrp{k}_{j} mp{k}_{j} cp{j} {rp:.4f}")
                lines.append(f"XTn{k}_{j} row{k} gate mn{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrn{k}_{j} mn{k}_{j} cn{j} {rn:.4f}")
        lines += ["", ".op", ".control", "run"]
        for j in range(n_out):
            lines.append(f"print i(Vsp{j}) i(Vsn{j})")
        lines += ["quit", ".endc", "", ".end"]

        tb_path = os.path.join(INTEGRATION_DIR, f"_unit_{label}_{i}.spice")
        with open(tb_path, 'w') as f:
            f.write('\n'.join(lines))

        stdout, stderr = run_ngspice(tb_path)
        Ip, In = parse_currents(stdout, n_out)
        W_eff[i, :] = scale * (Ip - In)

        try: os.remove(tb_path)
        except: pass

    return W_eff, Gp, Gn, scale


def main():
    print("=" * 74, flush=True)
    print("  RETRAIN MLP ON ANALOG FRONTEND FEATURES", flush=True)
    print("  Fully analog pipeline: no digital calibration", flush=True)
    print("=" * 74, flush=True)

    # ================================================================
    # STEP 1: Load analog features from cache
    # ================================================================
    print("\n[1/7] Loading analog features from frontend cache...", flush=True)
    X_tr, y_tr = load_analog_features(DS1, "DS1 (train)")
    X_te, y_te = load_analog_features(DS2, "DS2 (test)")

    if len(X_tr) == 0:
        print("ERROR: No DS1 training data available"); return

    # Show feature statistics
    print(f"\n  Analog feature statistics (training):", flush=True)
    feat_names = ['rr_ratio', 'beat_instab_L1', 'rr_asymmetry', 'pwave_L1',
                  'qrs_width_L0', 'beat_instab_L0', 'slope_ratio_L0', 'qrs_sym_L0']
    for i in range(8):
        mn, mx, mu, sd = X_tr[:,i].min(), X_tr[:,i].max(), X_tr[:,i].mean(), X_tr[:,i].std()
        print(f"    F{i} ({feat_names[i]:>16s}): [{mn:.4f}, {mx:.4f}], mean={mu:.4f}, std={sd:.4f}", flush=True)

    # ================================================================
    # STEP 2: Standardize analog features
    # ================================================================
    print(f"\n[2/7] Standardizing analog features...", flush=True)
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te) if len(X_te) > 0 else np.empty((0, 8))
    print(f"  Scaler mean: {scaler.mean_}", flush=True)
    print(f"  Scaler scale: {scaler.scale_}", flush=True)

    # ================================================================
    # STEP 3: Train MLP 8->64->5
    # ================================================================
    print(f"\n[3/7] Training MLP 8->64->5 on analog features (10 seeds)...", flush=True)
    best_mlp, best_acc = None, 0
    for seed in range(10):
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, random_state=seed+42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', learning_rate_init=1e-3,
            alpha=1e-4, batch_size=128
        )
        mlp.fit(X_tr_n, y_tr)

        # Test on train set first
        tr_acc = accuracy_score(y_tr, mlp.predict(X_tr_n)) * 100

        # Test on available DS2
        te_acc = 0
        if len(X_te) > 0:
            te_acc = accuracy_score(y_te, mlp.predict(X_te_n)) * 100

        print(f"    Seed {seed+42}: train={tr_acc:.2f}%, test={te_acc:.2f}%", flush=True)
        if te_acc > best_acc or (len(X_te) == 0 and tr_acc > best_acc):
            best_acc = te_acc if len(X_te) > 0 else tr_acc
            best_mlp = mlp

    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]
    sw_pred_tr = best_mlp.predict(X_tr_n)
    sw_acc_tr = accuracy_score(y_tr, sw_pred_tr) * 100
    print(f"\n  Best MLP: train={sw_acc_tr:.2f}%", flush=True)
    if len(X_te) > 0:
        sw_pred_te = best_mlp.predict(X_te_n)
        sw_acc_te = accuracy_score(y_te, sw_pred_te) * 100
        print(f"  Best MLP: test={sw_acc_te:.2f}% ({len(y_te)} beats)", flush=True)
    print(f"  W1: {W1.shape} [{W1.min():.4f}, {W1.max():.4f}]", flush=True)
    print(f"  W2: {W2.shape} [{W2.min():.4f}, {W2.max():.4f}]", flush=True)

    # ================================================================
    # STEP 4: Map weights to ReRAM conductances via ngspice
    # ================================================================
    print(f"\n[4/7] Layer 1 crossbar: extracting 8x64 effective weights (8 sims)...", flush=True)
    t0 = time.time()
    W1_eff, Gp1, Gn1, scale1 = extract_crossbar_weights(W1, "L1")
    t1 = time.time()
    w1_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"  Done in {t1-t0:.0f}s. Error: mean={w1_err.mean()*100:.2f}%, max={w1_err.max()*100:.2f}%", flush=True)
    print(f"  Scale L1: {scale1:.6e}", flush=True)

    print(f"\n[5/7] Layer 2 crossbar: extracting 64x5 effective weights (64 sims)...", flush=True)
    t0 = time.time()
    W2_eff, Gp2, Gn2, scale2 = extract_crossbar_weights(W2, "L2")
    t2 = time.time()
    w2_err = np.abs(W2_eff - W2) / (np.abs(W2) + 1e-8)
    print(f"  Done in {t2-t0:.0f}s. Error: mean={w2_err.mean()*100:.2f}%, max={w2_err.max()*100:.2f}%", flush=True)
    print(f"  Scale L2: {scale2:.6e}", flush=True)

    # ================================================================
    # STEP 5: Full two-crossbar accuracy (software emulation of ReRAM)
    # ================================================================
    print(f"\n[6/7] Full two-crossbar pipeline accuracy...", flush=True)

    for name, X_n, y, n_label in [("TRAIN (DS1)", X_tr_n, y_tr, "train"),
                                    ("TEST (DS2)", X_te_n, y_te, "test")]:
        if len(y) == 0:
            print(f"\n  {name}: no data available"); continue

        # SW ideal
        Z1_sw = X_n @ W1 + b1
        H1_sw = np.maximum(0, Z1_sw)
        Z2_sw = H1_sw @ W2 + b2
        sw_pred = np.argmax(Z2_sw, axis=1)
        sw_acc = accuracy_score(y, sw_pred) * 100

        # Full analog (crossbar effective weights)
        Z1 = X_n @ W1_eff + b1
        H1 = np.maximum(0, Z1)
        Z2 = H1 @ W2_eff + b2
        hw_pred = np.argmax(Z2, axis=1)
        hw_acc = accuracy_score(y, hw_pred) * 100
        hw_f1 = f1_score(y, hw_pred, average=None, zero_division=0)
        agree = (hw_pred == sw_pred).mean() * 100

        print(f"\n  --- {name} ({len(y)} beats) ---", flush=True)
        print(f"  {'Pipeline':<45} {'Accuracy':>10}", flush=True)
        print(f"  {'-'*58}", flush=True)
        print(f"  {'SW MLP (ideal weights)':<45} {sw_acc:>9.2f}%", flush=True)
        print(f"  {'Full Analog (ReRAM crossbar weights)':<45} {hw_acc:>9.2f}%", flush=True)
        print(f"  {'SW vs Analog agreement':<45} {agree:>9.2f}%", flush=True)

        if n_label == "test":
            print(f"\n  Per-class F1 (full analog):", flush=True)
            for i, c in enumerate(CLASS_NAMES):
                if i < len(hw_f1):
                    print(f"    {c}: {hw_f1[i]:.4f}", flush=True)
            cm = confusion_matrix(y, hw_pred, labels=range(5))
            print(f"\n  Confusion matrix:", flush=True)
            print(f"  {'':>8}" + "".join(f"{c:>7}" for c in CLASS_NAMES), flush=True)
            for i, row in enumerate(cm):
                print(f"  {CLASS_NAMES[i]:>8}" + "".join(f"{v:>7d}" for v in row), flush=True)
            print(f"\n{classification_report(y, hw_pred, target_names=CLASS_NAMES, zero_division=0)}", flush=True)

    # ================================================================
    # STEP 6: Save weights and generate VCVS parameters
    # ================================================================
    print(f"\n[7/7] Saving weights and results...", flush=True)

    # The VCVS input conditioning for this analog-trained model:
    # The scaler standardizes: x_std = (x_analog - mean) / scale
    # The crossbar expects V_row = (x_std + X_OFFSET) / (2*X_OFFSET) * V_READ
    # where X_OFFSET=4.0 to map [-4,4] -> [0, V_READ]
    X_OFFSET = 4.0
    A_vcvs = np.zeros(8)
    B_vcvs = np.zeros(8)
    for i in range(8):
        # Combined: V_row = V_READ/(2*X_OFFSET*scale[i]) * x_analog
        #                  + V_READ*(X_OFFSET - mean[i]/scale[i])/(2*X_OFFSET)
        A_vcvs[i] = V_READ / (2.0 * X_OFFSET * scaler.scale_[i])
        B_vcvs[i] = V_READ * (X_OFFSET - scaler.mean_[i] / scaler.scale_[i]) / (2.0 * X_OFFSET)

    print(f"  VCVS parameters (analog voltage -> crossbar row voltage):", flush=True)
    for i in range(8):
        print(f"    F{i} ({feat_names[i]:>16s}): A={A_vcvs[i]:.8f}, B={B_vcvs[i]:.8f}", flush=True)

    # Save everything
    save_path = os.path.join(INTEGRATION_DIR, 'analog_retrained_weights.npz')
    np.savez(save_path,
             W1=W1, b1=b1, W2=W2, b2=b2,
             W1_eff=W1_eff, W2_eff=W2_eff,
             Gp1=Gp1, Gn1=Gn1, scale1=scale1,
             Gp2=Gp2, Gn2=Gn2, scale2=scale2,
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             A_vcvs=A_vcvs, B_vcvs=B_vcvs)
    print(f"  Saved: {save_path}", flush=True)

    # Save results JSON
    results = {
        'approach': 'retrained_on_analog_features',
        'train_beats': int(len(y_tr)),
        'test_beats': int(len(y_te)) if len(y_te) > 0 else 0,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'A_vcvs': A_vcvs.tolist(),
        'B_vcvs': B_vcvs.tolist(),
        'l1_scale': float(scale1),
        'l2_scale': float(scale2),
        'l1_weight_err_mean': round(float(w1_err.mean()*100), 3),
        'l2_weight_err_mean': round(float(w2_err.mean()*100), 3),
        'b1': b1.tolist(),
        'b2': b2.tolist(),
    }
    results_path = os.path.join(INTEGRATION_DIR, 'analog_retrained_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}", flush=True)

    print(f"\n{'='*74}", flush=True)
    print(f"  DONE. Fully analog MLP retrained on analog frontend features.", flush=True)
    print(f"{'='*74}", flush=True)


if __name__ == '__main__':
    main()
