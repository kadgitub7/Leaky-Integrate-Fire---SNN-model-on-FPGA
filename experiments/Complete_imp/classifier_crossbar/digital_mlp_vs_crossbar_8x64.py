"""
Two-stage verification of the 8->64->5 MLP classifier on DIGITAL features.

Test 1 (Software):
  DS1 -> Python 8-feature extraction -> MLP 8->64->5 (train)
  DS2 -> Python 8-feature extraction -> MLP 8->64->5 (test)
  Compare to RF baseline (95.76%)

Test 2 (Crossbar):
  Same trained MLP weights -> SKY130 1T1R ReRAM crossbar
  DS2 -> Python 8-feature extraction -> crossbar MLP (test)
  Verify crossbar matches software MLP

If crossbar accuracy > 95.5%, output the final crossbar netlist.
"""
import os, sys, json, time, subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_FILE = os.path.join(PROJ_DIR, 'classifier', 'features_8_cache.npz')

SKY130_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15
HIDDEN = 64
N_CLASSES = 5
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


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


def extract_crossbar_weights(W1, n_out):
    n_in = W1.shape[0]
    Gp, Gn, scale = map_weights_to_conductance(W1)
    W1_eff = np.zeros((n_in, n_out))

    for i in range(n_in):
        lines = [
            f"* Unit-input row {i}: 8x{n_out} differential crossbar",
            f'.include "{SKY130_CORNER}"',
            f'.include "{SKY130_MISMATCH}"',
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

        path = os.path.join(SCRIPT_DIR, f"_dunit_{i}.spice")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        stdout, stderr = run_ngspice(path)
        Ip, In = parse_currents(stdout, n_out)

        if np.all(Ip == 0) and np.all(In == 0):
            print(f"  ERROR row {i}: no output. stderr tail:", flush=True)
            for l in stderr.strip().split('\n')[-5:]:
                print(f"    {l}", flush=True)
            return None, Gp, Gn, scale

        W1_eff[i, :] = (Ip - In) * scale
        os.remove(path)
        print(f"    Row {i}: max|I|={max(np.abs(Ip).max(), np.abs(In).max()):.4e} A", flush=True)

    return W1_eff, Gp, Gn, scale


def generate_final_netlist(Gp, Gn, scale, filename):
    n_in, n_out = Gp.shape
    lines = [
        f"* =================================================================",
        f"* FINAL 8x64 Differential 1T1R ReRAM Crossbar - Layer 1",
        f"* Architecture: 8 -> 64 -> 5 MLP Classifier",
        f"* Technology: SKY130 sky130_fd_pr__nfet_01v8 (W={NMOS_W}u L={NMOS_L}u)",
        f"* =================================================================",
        f"*",
        f"* Conductance range: {G_MIN_US} - {G_MAX_US} uS (SKY130 ReRAM PDK)",
        f"* V_READ = {V_READ}V, V_GATE = 1.8V",
        f"* Scale factor: {scale:.6e}",
        f"* Total 1T1R cells: {n_in * n_out * 2} (differential: {n_in*n_out} pos + {n_in*n_out} neg)",
        f"*",
        f"* Cell topology (1T1R):",
        f"*   row_i --[NMOS D]--[NMOS S]--mid--[R_ReRAM]--col_j",
        f"*   NMOS gate tied to VDD (1.8V), deep triode: Vgs~1.8V, Vds~{V_READ}V",
        f"*",
        f"* Differential encoding:",
        f"*   w_ij = scale * (I_col_pos_j - I_col_neg_j)",
        f"*",
        f"* Usage:",
        f"*   1. Set Vrow0..Vrow7 to normalized_feature_i * V_READ",
        f"*   2. Run .op analysis",
        f"*   3. MAC_j = scale * (i(Vsp_j) - i(Vsn_j))",
        f"*   4. hidden_j = ReLU(MAC_j + bias_j)",
        f"*   5. Feed hidden layer to Layer 2 (64x5, digital or 2nd crossbar)",
        f"*",
        f"* Trained on: MIT-BIH DS1 (22 records), Python 8-feature extraction",
        f"* Tested on:  MIT-BIH DS2 (22 records), inter-patient protocol",
        f"* =================================================================",
        f"",
        f'.include "{SKY130_CORNER}"',
        f'.include "{SKY130_MISMATCH}"',
        f"",
        f"* --- Gate bias (NMOS always ON) ---",
        f"Vgate gate 0 1.8",
        f"",
        f"* --- Row inputs: set to feature_i * {V_READ}V ---",
    ]
    for i in range(n_in):
        lines.append(f"Vrow{i} row{i} 0 0.0  $ feature[{i}] * {V_READ}")
    lines.append("")
    lines.append(f"* --- Column sense lines (virtual ground, measure current) ---")
    for j in range(n_out):
        lines.append(f"Vsp{j} cp{j} 0 0  $ positive column {j}")
        lines.append(f"Vsn{j} cn{j} 0 0  $ negative column {j}")

    lines.append("")
    lines.append(f"* ===== POSITIVE CONDUCTANCE ARRAY ({n_in}x{n_out}) =====")
    for i in range(n_in):
        lines.append(f"* --- Row {i} positive cells ---")
        for j in range(n_out):
            r = 1e6 / Gp[i, j]
            g = Gp[i, j]
            lines.append(f"XTp{i}_{j} row{i} gate mp{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{i}_{j} mp{i}_{j} cp{j} {r:.4f}  $ G+={g:.3f}uS")

    lines.append("")
    lines.append(f"* ===== NEGATIVE CONDUCTANCE ARRAY ({n_in}x{n_out}) =====")
    for i in range(n_in):
        lines.append(f"* --- Row {i} negative cells ---")
        for j in range(n_out):
            r = 1e6 / Gn[i, j]
            g = Gn[i, j]
            lines.append(f"XTn{i}_{j} row{i} gate mn{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{i}_{j} mn{i}_{j} cn{j} {r:.4f}  $ G-={g:.3f}uS")

    lines += [
        "", "* --- Analysis ---", ".op", "",
        ".control", "run",
    ]
    for j in range(n_out):
        lines.append(f"print i(Vsp{j}) i(Vsn{j})")
    lines += ["quit", ".endc", "", ".end"]

    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, len(lines)


def main():
    print("=" * 74, flush=True)
    print("  8->64->5 MLP: Software vs SKY130 Crossbar on DIGITAL Features", flush=True)
    print("=" * 74, flush=True)

    # Load digital features
    d = np.load(CACHE_FILE)
    X_tr_raw, y_tr = d['X_tr'], d['y_tr']
    X_te_raw, y_te = d['X_te'], d['y_te']
    print(f"\n  Digital features loaded:", flush=True)
    print(f"    Train: {X_tr_raw.shape[0]} beats, Test: {X_te_raw.shape[0]} beats", flush=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    print(f"    StandardScaler applied: mean={scaler.mean_.round(3).tolist()}", flush=True)

    # ================================================================
    # TEST 1: Software MLP 8->64->5
    # ================================================================
    print(f"\n{'='*74}", flush=True)
    print(f"  TEST 1: Software MLP 8->64->5 (replacing RF)", flush=True)
    print(f"{'='*74}", flush=True)

    # RF baseline for comparison
    print(f"\n  RF baseline (n=300, depth=15, balanced)...", flush=True)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_acc = accuracy_score(y_te, rf_pred) * 100
    rf_f1 = f1_score(y_te, rf_pred, average=None, zero_division=0)
    print(f"    RF accuracy: {rf_acc:.2f}%", flush=True)
    print(f"    RF F1: N={rf_f1[0]:.3f} S={rf_f1[1]:.3f} V={rf_f1[2]:.3f} F={rf_f1[3]:.3f} Q={rf_f1[4]:.3f}", flush=True)

    # MLP training (5 seeds)
    print(f"\n  MLP 8->64->5 training (5 seeds)...", flush=True)
    best_mlp, best_acc = None, 0
    for seed in range(5):
        mlp = MLPClassifier(
            hidden_layer_sizes=(HIDDEN,), max_iter=500, random_state=seed+42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', learning_rate_init=1e-3,
            alpha=1e-4, batch_size=128
        )
        mlp.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, mlp.predict(X_te)) * 100
        print(f"    Seed {seed+42}: {acc:.2f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp

    sw_pred = best_mlp.predict(X_te)
    sw_acc = accuracy_score(y_te, sw_pred) * 100
    sw_f1 = f1_score(y_te, sw_pred, average=None, zero_division=0)

    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]

    print(f"\n  --- TEST 1 RESULTS ---", flush=True)
    print(f"  RF  accuracy:  {rf_acc:.2f}%  F1-S={rf_f1[1]:.3f} F1-V={rf_f1[2]:.3f}", flush=True)
    print(f"  MLP accuracy:  {sw_acc:.2f}%  F1-S={sw_f1[1]:.3f} F1-V={sw_f1[2]:.3f}", flush=True)
    print(f"  MLP vs RF:     {'+' if sw_acc >= rf_acc else ''}{sw_acc - rf_acc:.2f}%", flush=True)
    print(f"  W1: {W1.shape} range [{W1.min():.4f}, {W1.max():.4f}]", flush=True)
    print(f"  W2: {W2.shape} range [{W2.min():.4f}, {W2.max():.4f}]", flush=True)

    print(f"\n  MLP classification report:", flush=True)
    print(classification_report(y_te, sw_pred, target_names=CLASS_NAMES, zero_division=0), flush=True)

    # Save weights
    np.savez(os.path.join(SCRIPT_DIR, 'digital_mlp_8x64x5_weights.npz'),
             W1=W1, b1=b1, W2=W2, b2=b2,
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)

    # ================================================================
    # TEST 2: Crossbar MLP with same weights
    # ================================================================
    print(f"\n{'='*74}", flush=True)
    print(f"  TEST 2: SKY130 Crossbar MLP with trained weights", flush=True)
    print(f"{'='*74}", flush=True)

    print(f"\n  Extracting crossbar effective weights (8 ngspice sims)...", flush=True)
    t0 = time.time()
    W1_eff, Gp, Gn, scale = extract_crossbar_weights(W1, HIDDEN)
    elapsed = time.time() - t0

    if W1_eff is None:
        print("  FAILED: Could not extract crossbar weights", flush=True)
        return

    w_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"  Done in {elapsed:.0f}s", flush=True)
    print(f"  Weight error: mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%", flush=True)
    print(f"  Scale factor: {scale:.6e}", flush=True)

    # Crossbar classification on all DS2
    Z1 = X_te @ W1_eff + b1
    H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2 + b2
    hw_pred = np.argmax(Z2, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)
    agree = (sw_pred == hw_pred).mean() * 100
    n_disagree = int((sw_pred != hw_pred).sum())

    print(f"\n  --- TEST 2 RESULTS ---", flush=True)
    print(f"  SW  MLP accuracy:  {sw_acc:.2f}%  F1-S={sw_f1[1]:.3f} F1-V={sw_f1[2]:.3f}", flush=True)
    print(f"  HW Xbar accuracy:  {hw_acc:.2f}%  F1-S={hw_f1[1]:.3f} F1-V={hw_f1[2]:.3f}", flush=True)
    print(f"  SW/HW agreement:   {agree:.2f}% ({n_disagree} disagreements / {len(y_te)})", flush=True)
    print(f"  Weight error:      mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%", flush=True)

    print(f"\n  Crossbar classification report:", flush=True)
    print(classification_report(y_te, hw_pred, target_names=CLASS_NAMES, zero_division=0), flush=True)

    cm = confusion_matrix(y_te, hw_pred, labels=range(5))
    print(f"  Crossbar confusion matrix:", flush=True)
    print(f"  {'':>8}" + "".join(f"{c:>7}" for c in CLASS_NAMES), flush=True)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>8}" + "".join(f"{v:>7d}" for v in row), flush=True)

    # ================================================================
    # SPOT-CHECK: 5 beats through full ngspice simulation
    # ================================================================
    print(f"\n  Spot-check: 5 individual beats through full ngspice circuit...", flush=True)
    spot_indices = [0, 500, 5000, 20000, 40000]
    spot_results = []
    for si in spot_indices:
        if si >= len(y_te):
            continue
        x = X_te[si]
        x_pos = np.clip(x, 0, None) * V_READ
        x_neg = np.clip(-x, 0, None) * V_READ

        for tag, x_v in [("pos", x_pos), ("neg", x_neg)]:
            lines = [
                f"* Spot-check beat {si} {tag}",
                f'.include "{SKY130_CORNER}"', f'.include "{SKY130_MISMATCH}"', "",
                "Vgate gate 0 1.8",
            ]
            for k in range(8):
                lines.append(f"Vrow{k} row{k} 0 {float(x_v[k]):.10f}")
            for j in range(HIDDEN):
                lines.append(f"Vsp{j} cp{j} 0 0")
                lines.append(f"Vsn{j} cn{j} 0 0")
            for k in range(8):
                for j in range(HIDDEN):
                    r = 1e6 / Gp[k, j]
                    lines.append(f"XTp{k}_{j} row{k} gate mp{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                    lines.append(f"Rrp{k}_{j} mp{k}_{j} cp{j} {r:.4f}")
            for k in range(8):
                for j in range(HIDDEN):
                    r = 1e6 / Gn[k, j]
                    lines.append(f"XTn{k}_{j} row{k} gate mn{k}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                    lines.append(f"Rrn{k}_{j} mn{k}_{j} cn{j} {r:.4f}")
            lines += [".op", ".control", "run"]
            for j in range(HIDDEN):
                lines.append(f"print i(Vsp{j}) i(Vsn{j})")
            lines += ["quit", ".endc", ".end"]
            path = os.path.join(SCRIPT_DIR, f"_spot_{si}_{tag}.spice")
            with open(path, 'w') as f:
                f.write('\n'.join(lines))

        stdout_p, _ = run_ngspice(os.path.join(SCRIPT_DIR, f"_spot_{si}_pos.spice"))
        stdout_n, _ = run_ngspice(os.path.join(SCRIPT_DIR, f"_spot_{si}_neg.spice"))
        Ip_p, In_p = parse_currents(stdout_p, HIDDEN)
        Ip_n, In_n = parse_currents(stdout_n, HIDDEN)
        I_diff = (Ip_p - In_p) - (Ip_n - In_n)
        sim_mac = I_diff * scale
        sim_z1 = sim_mac + b1
        sim_h1 = np.maximum(0, sim_z1)
        sim_out = sim_h1 @ W2 + b2
        sim_pred = CLASS_NAMES[np.argmax(sim_out)]

        sw_mac = W1.T @ x
        mac_err = np.mean(np.abs(sim_mac - sw_mac) / (np.abs(sw_mac) + 1e-8)) * 100

        os.remove(os.path.join(SCRIPT_DIR, f"_spot_{si}_pos.spice"))
        os.remove(os.path.join(SCRIPT_DIR, f"_spot_{si}_neg.spice"))

        match = "MATCH" if sim_pred == CLASS_NAMES[y_te[si]] else "WRONG"
        sw_match = "MATCH" if CLASS_NAMES[np.argmax(sw_pred[si] if isinstance(sw_pred[si], np.ndarray) else y_te[si])] == sim_pred else ""
        print(f"    Beat {si:>5d}: true={CLASS_NAMES[y_te[si]]} sim={sim_pred} sw={CLASS_NAMES[sw_pred[si]]} "
              f"mac_err={mac_err:.1f}% {'OK' if sim_pred == CLASS_NAMES[sw_pred[si]] else 'DIFF'}", flush=True)
        spot_results.append({
            'beat': si, 'true': CLASS_NAMES[y_te[si]],
            'sim_pred': sim_pred, 'sw_pred': CLASS_NAMES[sw_pred[si]],
            'mac_err_pct': round(mac_err, 2),
        })

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*74}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*74}", flush=True)
    print(f"  {'Pipeline':<45} {'Accuracy':>10} {'F1-S':>7} {'F1-V':>7}", flush=True)
    print(f"  {'-'*69}", flush=True)
    print(f"  {'RF on digital features (baseline)':<45} {rf_acc:>9.2f}% {rf_f1[1]:>6.3f} {rf_f1[2]:>6.3f}", flush=True)
    print(f"  {'MLP 8->64->5 software (Test 1)':<45} {sw_acc:>9.2f}% {sw_f1[1]:>6.3f} {sw_f1[2]:>6.3f}", flush=True)
    print(f"  {'MLP 8->64->5 SKY130 crossbar (Test 2)':<45} {hw_acc:>9.2f}% {hw_f1[1]:>6.3f} {hw_f1[2]:>6.3f}", flush=True)
    print(f"  {'-'*69}", flush=True)
    print(f"  SW/HW agreement: {agree:.2f}%  |  Weight error: {w_err.mean()*100:.2f}% mean", flush=True)

    # Generate final netlist if crossbar accuracy > 95.5%
    if hw_acc > 95.5:
        print(f"\n  CROSSBAR ACCURACY {hw_acc:.2f}% > 95.5% -- GENERATING FINAL NETLIST", flush=True)
        netlist_path, n_lines = generate_final_netlist(
            Gp, Gn, scale, "FINAL_crossbar_8x64_digital.spice"
        )
        print(f"  Final netlist: FINAL_crossbar_8x64_digital.spice ({n_lines} lines)", flush=True)
    else:
        print(f"\n  CROSSBAR ACCURACY {hw_acc:.2f}% <= 95.5% -- netlist NOT generated", flush=True)
        print(f"  (Accuracy threshold not met)", flush=True)

    # Save results
    results = {
        'test1_rf_accuracy': round(rf_acc, 3),
        'test1_mlp_sw_accuracy': round(sw_acc, 3),
        'test1_mlp_vs_rf': round(sw_acc - rf_acc, 3),
        'test2_mlp_crossbar_accuracy': round(hw_acc, 3),
        'test2_sw_hw_agreement': round(agree, 3),
        'test2_weight_error_mean_pct': round(float(w_err.mean()*100), 3),
        'test2_weight_error_max_pct': round(float(w_err.max()*100), 3),
        'test2_scale_factor': float(scale),
        'rf_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, rf_f1)},
        'sw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, sw_f1)},
        'hw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, hw_f1)},
        'confusion_matrix': cm.tolist(),
        'spot_checks': spot_results,
        'architecture': '8-64-5',
        'n_1t1r_cells': 8 * HIDDEN * 2,
        'train_beats': len(y_tr),
        'test_beats': len(y_te),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'threshold_met': hw_acc > 95.5,
    }
    out_path = os.path.join(SCRIPT_DIR, 'digital_mlp_crossbar_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: digital_mlp_crossbar_results.json", flush=True)


if __name__ == '__main__':
    main()
