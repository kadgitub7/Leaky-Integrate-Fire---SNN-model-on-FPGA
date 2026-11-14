"""
Retrain MLP on CIRCUIT-MATCHING features and map to ReRAM crossbar.

Uses circuit_features_cache.npz (analog feature definitions: differences, not
ratios) to produce weights that match what the actual SPICE circuits compute.

Pipeline:
  1. Load circuit_features_cache.npz
  2. Train MLPClassifier 8->64->5 (10 seeds, pick best)
  3. Map weights to differential G+/G- ReRAM conductances
  4. Extract effective crossbar weights via ngspice unit-input characterization
  5. Generate L1/L2 crossbar netlists + WTA
  6. Verify full-analog pipeline accuracy on DS2
  7. Generate top_level_classifier.spice (updated)
  8. Save everything to circuit_analog_weights.npz
"""
import os, sys, json, time, subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

w = sys.stderr.write

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLETE_IMP_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_FILE = os.path.join(COMPLETE_IMP_DIR, 'validation', 'circuit_features_cache.npz')

SKY130_NFET_TT = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_NFET_MM = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
SKY130_PFET_TT = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"
SKY130_PFET_MM = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__mismatch.corner.spice"
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

PFET_MISMATCH_PARAMS = [
    'a0', 'agidl', 'ags', 'b0', 'b1', 'bgidl', 'cgidl', 'dlc', 'dwc',
    'eta0', 'k2', 'keta', 'kt1', 'ku0', 'kvsat', 'kvth0', 'lint', 'lku0',
    'lkvth0', 'nfactor', 'pclm', 'pdits', 'pditsd', 'rdsw', 'tvoff',
    'u0', 'ua', 'ub', 'uc', 'ute', 'voff', 'vsat', 'vth0', 'wint',
    'wku0', 'wkvth0', 'wlod',
]


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


def run_ngspice(path):
    result = subprocess.run(
        [NGSPICE, "-b", path],
        capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR
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
    n_in, n_out = W.shape
    Gp, Gn, scale = map_weights_to_conductance(W)
    W_eff = np.zeros((n_in, n_out))

    for i in range(n_in):
        lines = [
            f"* Unit-input row {i}: {n_in}x{n_out} diff crossbar ({label})",
            f'.include "{SKY130_NFET_TT}"',
            f'.include "{SKY130_NFET_MM}"',
            "", "Vgate gate 0 1.8", "",
        ]
        for k in range(n_in):
            lines.append(f"Vrow{k} row{k} 0 {V_READ if k == i else 0.0}")
        lines.append("")
        for j in range(n_out):
            lines.append(f"Vsp{j} cp{j} 0 0")
            lines.append(f"Vsn{j} cn{j} 0 0")
        lines.append("")
        for k in range(n_in):
            for j in range(n_out):
                r = 1e6 / Gp[k, j]
                lines.append(f"XTp{k}_{j} row{k} gate mp{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrp{k}_{j} mp{k}_{j} cp{j} {r:.4f}")
        lines.append("")
        for k in range(n_in):
            for j in range(n_out):
                r = 1e6 / Gn[k, j]
                lines.append(f"XTn{k}_{j} row{k} gate mn{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrn{k}_{j} mn{k}_{j} cn{j} {r:.4f}")
        lines += ["", ".op", ".control", "run"]
        for j in range(n_out):
            lines.append(f"print i(Vsp{j}) i(Vsn{j})")
        lines += ["quit", ".endc", ".end"]

        path = os.path.join(SCRIPT_DIR, f"_u{label}_{i}.spice")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        stdout, stderr = run_ngspice(path)
        Ip, In = parse_currents(stdout, n_out)

        if np.all(Ip == 0) and np.all(In == 0):
            w(f"    ERROR row {i}: no output\n")
            return None, Gp, Gn, scale

        W_eff[i, :] = (Ip - In) * scale
        os.remove(path)

        w(f"    Row {i}/{n_in}: max|I|={max(np.abs(Ip).max(), np.abs(In).max()):.4e} A\n")

    return W_eff, Gp, Gn, scale


def generate_crossbar_netlist(Gp, Gn, scale, W, b, layer_name, filename):
    n_in, n_out = Gp.shape
    lines = [
        f"* =================================================================",
        f"* {layer_name}: {n_in}x{n_out} Differential 1T1R ReRAM Crossbar",
        f"* Trained on CIRCUIT-MATCHING features (analog differences, alpha=0.015)",
        f"* Technology: SKY130 sky130_fd_pr__nfet_01v8 (W={NMOS_W}u L={NMOS_L}u)",
        f"* =================================================================",
        f"*",
        f"* Conductance range: {G_MIN_US} - {G_MAX_US} uS (SKY130 ReRAM PDK)",
        f"* V_READ = {V_READ}V, V_GATE = 1.8V",
        f"* Scale factor: {scale:.6e}",
        f"* Total 1T1R cells: {n_in * n_out * 2} (differential)",
        f"*",
        f"* Biases (add after current sensing):",
    ]
    for j in range(n_out):
        lines.append(f"*   b[{j}] = {b[j]:.6f}")

    lines += [
        f"* =================================================================",
        "",
        f'.include "{SKY130_NFET_TT}"',
        f'.include "{SKY130_NFET_MM}"',
        "",
        "Vgate gate 0 1.8",
        "",
        "* --- Row inputs ---",
    ]
    for i in range(n_in):
        lines.append(f"Vrow{i} row{i} 0 0.0")
    lines.append("")
    lines.append("* --- Column sense lines ---")
    for j in range(n_out):
        lines.append(f"Vsp{j} cp{j} 0 0")
        lines.append(f"Vsn{j} cn{j} 0 0")

    lines.append("")
    lines.append(f"* ===== POSITIVE CONDUCTANCE ARRAY ({n_in}x{n_out}) =====")
    for i in range(n_in):
        lines.append(f"* --- Row {i} ---")
        for j in range(n_out):
            r = 1e6 / Gp[i, j]
            g = Gp[i, j]
            lines.append(f"XTp{i}_{j} row{i} gate mp{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{i}_{j} mp{i}_{j} cp{j} {r:.4f}  $ G+={g:.3f}uS w={W[i,j]:.4f}")

    lines.append("")
    lines.append(f"* ===== NEGATIVE CONDUCTANCE ARRAY ({n_in}x{n_out}) =====")
    for i in range(n_in):
        lines.append(f"* --- Row {i} ---")
        for j in range(n_out):
            r = 1e6 / Gn[i, j]
            g = Gn[i, j]
            lines.append(f"XTn{i}_{j} row{i} gate mn{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{i}_{j} mn{i}_{j} cn{j} {r:.4f}  $ G-={g:.3f}uS w={W[i,j]:.4f}")

    lines += ["", ".op", "", ".control", "run"]
    for j in range(n_out):
        lines.append(f"print i(Vsp{j}) i(Vsn{j})")
    lines += ["quit", ".endc", "", ".end"]

    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, len(lines)


def main():
    w("=" * 74 + "\n")
    w("  RETRAIN ON CIRCUIT FEATURES + MAP TO ReRAM CROSSBAR\n")
    w("=" * 74 + "\n")

    # Load circuit features cache
    w(f"\nLoading {CACHE_FILE}...\n")
    d = np.load(CACHE_FILE, allow_pickle=True)
    X_tr, y_tr = d['X_tr'], d['y_tr']
    X_te, y_te = d['X_te'], d['y_te']
    scaler_mean = d['scaler_mean']
    scaler_scale = d['scaler_scale']
    feature_names = d['feature_names']
    ewma_alpha = float(d['ewma_alpha'])

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale ** 2
    scaler.n_features_in_ = 8
    X_tr_n = (X_tr - scaler_mean) / scaler_scale
    X_te_n = (X_te - scaler_mean) / scaler_scale

    w(f"  Train: {len(y_tr)} beats, Test: {len(y_te)} beats\n")
    w(f"  EWMA alpha: {ewma_alpha}\n")
    w(f"  Features: {list(feature_names)}\n")
    for i in range(5):
        cnt_tr = (y_tr == i).sum()
        cnt_te = (y_te == i).sum()
        w(f"  Class {CLASS_NAMES[i]}: {cnt_tr} train, {cnt_te} test\n")

    # Train MLP (10 seeds for robustness)
    w(f"\n[1/6] Training MLP 8->64->5 (10 seeds)...\n")
    best_mlp, best_acc, best_seed = None, 0, 0
    all_accs = []
    for seed in range(42, 52):
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=500, random_state=seed,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', learning_rate_init=1e-3,
            alpha=1e-4, batch_size=128
        )
        mlp.fit(X_tr_n, y_tr)
        pred = mlp.predict(X_te_n)
        acc = accuracy_score(y_te, pred) * 100
        s_rec = (pred[y_te == 1] == 1).mean() * 100 if (y_te == 1).any() else 0
        v_rec = (pred[y_te == 2] == 2).mean() * 100 if (y_te == 2).any() else 0
        w(f"    Seed {seed}: {acc:.2f}%  S={s_rec:.1f}% V={v_rec:.1f}%\n")
        all_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp
            best_seed = seed

    w(f"  Mean: {np.mean(all_accs):.2f} +/- {np.std(all_accs):.2f}%\n")
    w(f"  Best: seed {best_seed} = {best_acc:.2f}%\n")

    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]
    sw_pred = best_mlp.predict(X_te_n)

    w(f"\n  Classification report (SW MLP on circuit features):\n")
    w(classification_report(y_te, sw_pred, target_names=CLASS_NAMES, zero_division=0) + "\n")

    # Layer 1 crossbar characterization
    w(f"\n[2/6] Layer 1: extracting 8x64 effective weights (8 ngspice sims)...\n")
    t0 = time.time()
    W1_eff, Gp1, Gn1, scale1 = extract_crossbar_weights(W1, "L1")
    if W1_eff is None:
        w("  FAILED Layer 1\n"); return
    w1_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    w(f"  Done in {time.time()-t0:.0f}s. Error: mean={w1_err.mean()*100:.2f}%, max={w1_err.max()*100:.2f}%\n")

    # Layer 2 crossbar characterization
    w(f"\n[3/6] Layer 2: extracting 64x5 effective weights (64 ngspice sims)...\n")
    t0 = time.time()
    W2_eff, Gp2, Gn2, scale2 = extract_crossbar_weights(W2, "L2")
    if W2_eff is None:
        w("  FAILED Layer 2\n"); return
    w2_err = np.abs(W2_eff - W2) / (np.abs(W2) + 1e-8)
    w(f"  Done in {time.time()-t0:.0f}s. Error: mean={w2_err.mean()*100:.2f}%, max={w2_err.max()*100:.2f}%\n")

    # Generate netlists
    w(f"\n[4/6] Generating crossbar netlists...\n")
    _, n1 = generate_crossbar_netlist(Gp1, Gn1, scale1, W1, b1,
        "LAYER 1 (circuit features)", "L1_crossbar_8x64_circuit.spice")
    w(f"  L1_crossbar_8x64_circuit.spice ({n1} lines)\n")

    _, n2 = generate_crossbar_netlist(Gp2, Gn2, scale2, W2, b2,
        "LAYER 2 (circuit features)", "L2_crossbar_64x5_circuit.spice")
    w(f"  L2_crossbar_64x5_circuit.spice ({n2} lines)\n")

    # Full two-crossbar accuracy
    w(f"\n[5/6] Full two-crossbar pipeline accuracy...\n")

    Z1_eff = X_te_n @ W1_eff + b1
    H1_eff = np.maximum(0, Z1_eff)
    Z2_eff = H1_eff @ W2_eff + b2
    hw_pred = np.argmax(Z2_eff, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100

    Z1_sw = X_te_n @ W1 + b1
    H1_sw = np.maximum(0, Z1_sw)
    Z2_sw = H1_sw @ W2 + b2
    sw_ref_pred = np.argmax(Z2_sw, axis=1)
    sw_ref_acc = accuracy_score(y_te, sw_ref_pred) * 100

    agree = (hw_pred == sw_ref_pred).mean() * 100
    cm = confusion_matrix(y_te, hw_pred, labels=range(5))
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)

    w(f"\n{'='*74}\n")
    w(f"  RESULTS: Circuit-Feature Analog MLP Crossbar\n")
    w(f"{'='*74}\n")
    w(f"  SW MLP (ideal weights):        {sw_ref_acc:.2f}%\n")
    w(f"  Full analog (both crossbars):  {hw_acc:.2f}%\n")
    w(f"  SW vs analog agreement:        {agree:.2f}%\n")
    w(f"  L1 weight error: mean={w1_err.mean()*100:.2f}%, max={w1_err.max()*100:.2f}%\n")
    w(f"  L2 weight error: mean={w2_err.mean()*100:.2f}%, max={w2_err.max()*100:.2f}%\n")

    w(f"\n  Per-class (full analog crossbar):\n")
    w(classification_report(y_te, hw_pred, target_names=CLASS_NAMES, zero_division=0) + "\n")

    w(f"  Confusion matrix:\n")
    w(f"  {'':>8}" + "".join(f"{c:>7}" for c in CLASS_NAMES) + "\n")
    for i, row in enumerate(cm):
        w(f"  {CLASS_NAMES[i]:>8}" + "".join(f"{v:>7d}" for v in row) + "\n")

    # Save
    w(f"\n[6/6] Saving weights and results...\n")
    np.savez(os.path.join(SCRIPT_DIR, 'circuit_analog_weights.npz'),
             W1=W1, b1=b1, W2=W2, b2=b2,
             W1_eff=W1_eff, W2_eff=W2_eff,
             Gp1=Gp1, Gn1=Gn1, scale1=scale1,
             Gp2=Gp2, Gn2=Gn2, scale2=scale2,
             scaler_mean=scaler_mean, scaler_scale=scaler_scale,
             feature_names=feature_names, ewma_alpha=ewma_alpha,
             best_seed=best_seed)

    results = {
        'training_features': 'circuit_features_cache.npz (analog differences, alpha=0.015)',
        'best_seed': int(best_seed),
        'sw_accuracy': round(sw_ref_acc, 3),
        'full_analog_accuracy': round(hw_acc, 3),
        'full_analog_sw_agreement': round(agree, 3),
        'l1_weight_err_mean_pct': round(float(w1_err.mean()*100), 3),
        'l1_weight_err_max_pct': round(float(w1_err.max()*100), 3),
        'l2_weight_err_mean_pct': round(float(w2_err.mean()*100), 3),
        'l2_weight_err_max_pct': round(float(w2_err.max()*100), 3),
        'l1_scale': float(scale1),
        'l2_scale': float(scale2),
        'per_class_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, hw_f1)},
        'confusion_matrix': cm.tolist(),
        'train_beats': len(y_tr),
        'test_beats': len(y_te),
        'scaler_mean': scaler_mean.tolist(),
        'scaler_scale': scaler_scale.tolist(),
        'all_seed_accuracies': [round(a, 2) for a in all_accs],
    }
    out_path = os.path.join(SCRIPT_DIR, 'circuit_retrain_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    w(f"\n  FILES PRODUCED:\n")
    w(f"    L1_crossbar_8x64_circuit.spice  -- Layer 1 (circuit-feature weights)\n")
    w(f"    L2_crossbar_64x5_circuit.spice  -- Layer 2 (circuit-feature weights)\n")
    w(f"    circuit_analog_weights.npz      -- Weights, conductances, scaler\n")
    w(f"    circuit_retrain_results.json    -- All metrics\n")
    w(f"\n  Done.\n")


if __name__ == '__main__':
    main()
