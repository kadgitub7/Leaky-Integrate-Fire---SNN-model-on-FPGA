"""
Full inter-patient classification accuracy through the ngspice crossbar.

Strategy:
  1. Train MLP 8->32->5, extract weights
  2. Map weights to ReRAM conductances (differential encoding)
  3. Run 8 unit-input ngspice simulations to extract the EFFECTIVE weight matrix
     (accounts for NMOS R_on, parasitic effects)
  4. Classify all 48,181 test beats using the effective weight matrix
  5. Compare software MLP accuracy vs crossbar accuracy

The column nodes are at virtual ground (0V sources), so cells on different
rows don't interact — the unit-input characterization is exact for the
linear regime. NMOS in deep triode (Vds ~0.15V, Vgs ~1.8V) is nearly linear.
"""
import os, sys, json, subprocess, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_FILE = os.path.join(PROJ_DIR, 'classifier', 'features_8_cache.npz')

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15

SKY130_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


def train_mlp():
    from sklearn.neural_network import MLPClassifier
    d = np.load(CACHE_FILE)
    X_tr, y_tr, X_te, y_te = d['X_tr'], d['y_tr'], d['X_te'], d['y_te']
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1,
                        learning_rate='adaptive', learning_rate_init=1e-3,
                        alpha=1e-4, batch_size=128)
    mlp.fit(X_tr, y_tr)
    acc = (mlp.predict(X_te) == y_te).mean() * 100
    print(f"Software MLP accuracy: {acc:.2f}%")
    return mlp.coefs_[0], mlp.intercepts_[0], mlp.coefs_[1], mlp.intercepts_[1], X_te, y_te, mlp


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
    """Generate netlist with only row_idx active at V_READ, all others at 0V."""
    n_in, n_out = W.shape
    Gp, Gn, scale = map_weights_to_conductance(W)

    lines = []
    lines.append(f"* Unit-input characterization: row {row_idx} active")
    lines.append(f'.include "{SKY130_CORNER}"')
    lines.append(f'.include "{SKY130_MISMATCH}"')
    lines.append("")
    lines.append("Vgate gate 0 1.8")
    lines.append("")

    for i in range(n_in):
        v = V_READ if i == row_idx else 0.0
        lines.append(f"Vrow{i} row{i} 0 {v}")
    lines.append("")

    for j in range(n_out):
        lines.append(f"Vsp{j} col_p{j} 0 0")
        lines.append(f"Vsn{j} col_n{j} 0 0")
    lines.append("")

    for i in range(n_in):
        for j in range(n_out):
            r_ohm = 1e6 / Gp[i, j]
            mid = f"mp{i}_{j}"
            lines.append(f"XTp{i}_{j} row{i} gate {mid} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{i}_{j} {mid} col_p{j} {r_ohm:.4f}")
    lines.append("")

    for i in range(n_in):
        for j in range(n_out):
            r_ohm = 1e6 / Gn[i, j]
            mid = f"mn{i}_{j}"
            lines.append(f"XTn{i}_{j} row{i} gate {mid} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{i}_{j} {mid} col_n{j} {r_ohm:.4f}")
    lines.append("")

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


def run_ngspice(netlist_path):
    result = subprocess.run(
        [NGSPICE, "-b", netlist_path],
        capture_output=True, text=True, timeout=600,
        cwd=SCRIPT_DIR
    )
    return result.stdout, result.stderr


def parse_currents(stdout, n_out):
    Ip = np.zeros(n_out)
    In = np.zeros(n_out)
    for line in stdout.split('\n'):
        s = line.strip().lower()
        for j in range(n_out):
            if f"i(vsp{j})" in s and '=' in s:
                try:
                    Ip[j] = float(s.split('=')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if f"i(vsn{j})" in s and '=' in s:
                try:
                    In[j] = float(s.split('=')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
    return Ip, In


def extract_effective_weights(W1):
    """Run 8 unit-input sims to extract the effective weight matrix from ngspice."""
    n_in, n_out = W1.shape
    _, _, scale = map_weights_to_conductance(W1)

    W1_eff = np.zeros((n_in, n_out))

    print(f"\nExtracting effective weight matrix via {n_in} unit-input ngspice runs...")
    t0 = time.time()

    for i in range(n_in):
        fname = f"unit_row{i}.spice"
        path, _ = generate_unit_input_netlist(W1, i, fname)
        stdout, stderr = run_ngspice(path)
        Ip, In = parse_currents(stdout, n_out)

        if np.all(Ip == 0) and np.all(In == 0):
            print(f"  ERROR: No output for row {i}")
            print(f"  stderr: {stderr[-200:]}")
            return None, scale

        I_diff = Ip - In
        W1_eff[i, :] = I_diff * scale
        print(f"  Row {i}: max|I_diff|={np.max(np.abs(I_diff)):.3e} A, "
              f"max|w_eff|={np.max(np.abs(W1_eff[i,:])):.4f}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Clean up unit input netlists
    for i in range(n_in):
        f = os.path.join(SCRIPT_DIR, f"unit_row{i}.spice")
        if os.path.exists(f):
            os.remove(f)

    return W1_eff, scale


def classify_all(W1_eff, b1, W2, b2, X_te, y_te):
    """Run full MLP inference using effective (crossbar) weights for Layer 1."""
    # Layer 1: h = ReLU(W1_eff^T @ x + b1)  — crossbar MAC + software bias + ReLU
    # Layer 2: y = W2^T @ h + b2             — software (or second crossbar)
    Z1 = X_te @ W1_eff + b1  # (N, 32)
    H1 = np.maximum(0, Z1)   # ReLU
    Z2 = H1 @ W2 + b2        # (N, 5)
    preds = np.argmax(Z2, axis=1)
    return preds


def main():
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found"); sys.exit(1)

    print("=" * 70)
    print("CROSSBAR CLASSIFICATION ACCURACY — Full MIT-BIH Inter-Patient")
    print("=" * 70)

    W1, b1, W2, b2, X_te, y_te, mlp = train_mlp()

    # Software baseline
    sw_preds = mlp.predict(X_te)
    sw_acc = (sw_preds == y_te).mean() * 100
    from sklearn.metrics import f1_score, confusion_matrix
    sw_f1 = f1_score(y_te, sw_preds, average=None, zero_division=0)

    print(f"\n--- Software MLP (ideal) ---")
    print(f"Accuracy: {sw_acc:.2f}%")
    print(f"F1: {' '.join(f'{c}={f:.3f}' for c,f in zip(CLASS_NAMES, sw_f1))}")

    # Extract effective weight matrix from ngspice
    W1_eff, scale = extract_effective_weights(W1)
    if W1_eff is None:
        print("Failed to extract effective weights"); sys.exit(1)

    # Compare ideal vs effective weights
    w_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"\nWeight matrix error (ngspice vs ideal):")
    print(f"  Mean: {w_err.mean()*100:.2f}%")
    print(f"  Max:  {w_err.max()*100:.2f}%")
    print(f"  Median: {np.median(w_err)*100:.2f}%")

    # Classify all test beats using crossbar weights
    hw_preds = classify_all(W1_eff, b1, W2, b2, X_te, y_te)
    hw_acc = (hw_preds == y_te).mean() * 100
    hw_f1 = f1_score(y_te, hw_preds, average=None, zero_division=0)

    print(f"\n--- Crossbar MLP (ngspice effective weights) ---")
    print(f"Accuracy: {hw_acc:.2f}%")
    print(f"F1: {' '.join(f'{c}={f:.3f}' for c,f in zip(CLASS_NAMES, hw_f1))}")

    # Agreement between software and crossbar
    agree = (sw_preds == hw_preds).mean() * 100
    disagree_idx = np.where(sw_preds != hw_preds)[0]
    print(f"\nSW vs HW agreement: {agree:.2f}% ({len(disagree_idx)} disagreements out of {len(y_te)})")

    if len(disagree_idx) > 0 and len(disagree_idx) <= 20:
        print("\nDisagreements:")
        for idx in disagree_idx[:20]:
            print(f"  Beat {idx}: true={CLASS_NAMES[y_te[idx]]} "
                  f"sw={CLASS_NAMES[sw_preds[idx]]} hw={CLASS_NAMES[hw_preds[idx]]}")

    # Confusion matrix for crossbar
    cm = confusion_matrix(y_te, hw_preds, labels=range(5))
    print(f"\nCrossbar confusion matrix:")
    print(f"{'':>6} " + " ".join(f"{c:>6}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>6} " + " ".join(f"{v:>6d}" for v in row))

    # Also verify with a few explicit ngspice runs on individual beats
    print(f"\n--- Spot-check: 5 individual beats through full ngspice ---")
    from generate_crossbar import generate_netlist
    _, _, scale2 = map_weights_to_conductance(W1)

    spot_indices = [0, 100, 1000, 10000, 40000]
    for si in spot_indices:
        x = X_te[si]
        x_pos = np.clip(x, 0, None) * V_READ
        x_neg = np.clip(-x, 0, None) * V_READ

        path_p, _, _, _ = generate_netlist(W1, x_pos, f"spot_p_{si}.spice", f"spot+ {si}")
        path_n, _, _, _ = generate_netlist(W1, x_neg, f"spot_n_{si}.spice", f"spot- {si}")

        out_p, _ = run_ngspice(path_p)
        out_n, _ = run_ngspice(path_n)

        Ip_p, In_p = parse_currents(out_p, 32)
        Ip_n, In_n = parse_currents(out_n, 32)

        I_diff = (Ip_p - In_p) - (Ip_n - In_n)
        hw_mac = I_diff * scale2
        h1 = np.maximum(0, hw_mac + b1)
        out = W2.T @ h1 + b2
        pred = np.argmax(out)

        sw_mac = W1.T @ x
        h1_sw = np.maximum(0, sw_mac + b1)
        out_sw = W2.T @ h1_sw + b2
        pred_sw = np.argmax(out_sw)

        mac_err = np.mean(np.abs(hw_mac - sw_mac) / (np.abs(sw_mac) + 1e-8)) * 100
        print(f"  Beat {si:>5d}: true={CLASS_NAMES[y_te[si]]} "
              f"sw={CLASS_NAMES[pred_sw]} hw={CLASS_NAMES[pred]} "
              f"mac_err={mac_err:.1f}% {'MATCH' if pred==pred_sw else 'DIFF'}")

        os.remove(os.path.join(SCRIPT_DIR, f"spot_p_{si}.spice"))
        os.remove(os.path.join(SCRIPT_DIR, f"spot_n_{si}.spice"))

    # Save results
    results = {
        'software_accuracy_pct': float(sw_acc),
        'crossbar_accuracy_pct': float(hw_acc),
        'accuracy_drop_pct': float(sw_acc - hw_acc),
        'sw_hw_agreement_pct': float(agree),
        'n_disagreements': int(len(disagree_idx)),
        'n_test_beats': int(len(y_te)),
        'weight_error_mean_pct': float(w_err.mean() * 100),
        'weight_error_max_pct': float(w_err.max() * 100),
        'sw_f1': {c: float(f) for c, f in zip(CLASS_NAMES, sw_f1)},
        'hw_f1': {c: float(f) for c, f in zip(CLASS_NAMES, hw_f1)},
        'nmos_W_um': float(NMOS_W),
        'nmos_L_um': float(NMOS_L),
        'v_read_V': float(V_READ),
        'G_range_uS': [float(G_MIN_US), float(G_MAX_US)],
    }
    out_path = os.path.join(SCRIPT_DIR, 'crossbar_accuracy_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
