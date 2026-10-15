"""
Generate ngspice crossbar netlist for 8->32->5 MLP using SKY130 NMOS + ReRAM-as-resistor.

1. Train the 8->32->5 MLP on cached 8-feature data
2. Extract weights, map to differential ReRAM conductances (PDK range)
3. Generate ngspice netlist for Layer 1 (8x32) crossbar
4. Run ngspice, compare output column currents to expected software MAC

Cell topology (1T1R):
    row_i (input V) → NMOS(drain=row, gate=VDD, source=mid) → Resistor(mid, col) → col_j (0V)
    Vgs = VDD - V(mid) ≈ 1.8V, Vds = V(row) - V(mid) ≈ 0.15V → deep triode

Differential encoding:
    w_ij = scale * (I_col_j_pos - I_col_j_neg)
"""
import os, sys, json, subprocess
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
    print(f"MLP 8->32->5 accuracy: {acc:.2f}%")
    return mlp.coefs_[0], mlp.intercepts_[0], mlp.coefs_[1], mlp.intercepts_[1], X_te, y_te, mlp


def map_weights_to_conductance(W):
    """Differential encoding: w_ij → (G+_ij, G-_ij) in uS.

    Positive weight: G+ = G_MIN + |w_norm|*(G_MAX-G_MIN), G- = G_MIN
    Negative weight: G+ = G_MIN, G- = G_MIN + |w_norm|*(G_MAX-G_MIN)
    Scale factor: w_ij = scale * (G+_ij - G-_ij) * V_READ
    """
    w_max = np.abs(W).max()
    G_range = G_MAX_US - G_MIN_US
    Gp = np.full_like(W, G_MIN_US)
    Gn = np.full_like(W, G_MIN_US)
    pos = W >= 0
    Gp[pos] = G_MIN_US + (W[pos] / w_max) * G_range
    Gn[~pos] = G_MIN_US + (np.abs(W[~pos]) / w_max) * G_range
    scale = w_max / (G_range * 1e-6 * V_READ)
    return Gp, Gn, scale


def generate_netlist(W, input_voltages, filename, label="layer1"):
    """Generate ngspice .op netlist for NxM differential crossbar.

    Uses DC operating point analysis (fast).
    Column lines at 0V via voltage sources; current through them = MAC.
    """
    n_in, n_out = W.shape
    Gp, Gn, scale = map_weights_to_conductance(W)

    lines = []
    lines.append(f"* {n_in}x{n_out} 1T1R Differential Crossbar — {label}")
    lines.append(f"* SKY130 NFET + fixed resistors (ReRAM conductance model)")
    lines.append(f"* Scale factor: {scale:.6e}")
    lines.append("")
    lines.append(f'.include "{SKY130_CORNER}"')
    lines.append(f'.include "{SKY130_MISMATCH}"')
    lines.append("")

    lines.append("* NMOS gate (always on)")
    lines.append("Vgate gate 0 1.8")
    lines.append("")

    lines.append("* Row inputs (voltage-encoded features)")
    for i in range(n_in):
        lines.append(f"Vrow{i} row{i} 0 {float(input_voltages[i]):.10f}")
    lines.append("")

    lines.append("* Column sense (virtual ground)")
    for j in range(n_out):
        lines.append(f"Vsp{j} col_p{j} 0 0")
        lines.append(f"Vsn{j} col_n{j} 0 0")
    lines.append("")

    lines.append(f"* Positive array ({n_in*n_out} cells)")
    for i in range(n_in):
        for j in range(n_out):
            g_us = Gp[i, j]
            r_ohm = 1e6 / g_us
            mid = f"mp{i}_{j}"
            lines.append(f"XTp{i}_{j} row{i} gate {mid} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{i}_{j} {mid} col_p{j} {r_ohm:.4f}")
    lines.append("")

    lines.append(f"* Negative array ({n_in*n_out} cells)")
    for i in range(n_in):
        for j in range(n_out):
            g_us = Gn[i, j]
            r_ohm = 1e6 / g_us
            mid = f"mn{i}_{j}"
            lines.append(f"XTn{i}_{j} row{i} gate {mid} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{i}_{j} {mid} col_n{j} {r_ohm:.4f}")
    lines.append("")

    lines.append(".op")
    lines.append("")
    lines.append(".control")
    lines.append("run")
    for j in range(n_out):
        lines.append(f"print i(Vsp{j}) i(Vsn{j})")
    lines.append("quit")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {filename}: {n_in}x{n_out} diff crossbar, {n_in*n_out*2} cells")
    return path, Gp, Gn, scale


def run_ngspice(netlist_path):
    print(f"Running ngspice: {os.path.basename(netlist_path)} ...")
    result = subprocess.run(
        [NGSPICE, "-b", netlist_path],
        capture_output=True, text=True, timeout=600,
        cwd=SCRIPT_DIR
    )
    return result.stdout, result.stderr


def parse_currents(stdout, n_out):
    """Parse i(Vsp{j}) and i(Vsn{j}) from ngspice output."""
    Ip = np.zeros(n_out)
    In = np.zeros(n_out)
    for line in stdout.split('\n'):
        s = line.strip()
        for j in range(n_out):
            if f"i(vsp{j})" in s.lower() and '=' in s:
                try:
                    Ip[j] = float(s.split('=')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if f"i(vsn{j})" in s.lower() and '=' in s:
                try:
                    In[j] = float(s.split('=')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
    return Ip, In


def main():
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: Cache not found: {CACHE_FILE}")
        sys.exit(1)
    if not os.path.exists(SKY130_CORNER):
        print(f"ERROR: SKY130 PDK not found: {SKY130_CORNER}")
        sys.exit(1)

    print("=" * 70)
    print("8->32->5 MLP Crossbar — SKY130 ngspice Verification")
    print("=" * 70)

    W1, b1, W2, b2, X_te, y_te, mlp = train_mlp()
    print(f"W1: {W1.shape} [{W1.min():.4f}, {W1.max():.4f}]")
    print(f"W2: {W2.shape} [{W2.min():.4f}, {W2.max():.4f}]")

    np.savez(os.path.join(SCRIPT_DIR, 'mlp_weights.npz'),
             W1=W1, b1=b1, W2=W2, b2=b2)

    preds = mlp.predict(X_te)
    correct = np.where(preds == y_te)[0]
    test_idx = int(correct[0])
    x = X_te[test_idx]

    print(f"\nTest beat #{test_idx}, class={y_te[test_idx]}")
    for i, v in enumerate(x):
        print(f"  x[{i}] = {v:+.6f}")

    sw_mac = W1.T @ x
    print(f"\nSoftware W1^T @ x: [{sw_mac.min():.6f}, {sw_mac.max():.6f}]")

    # Signed inputs → two crossbar passes: x+ and x-
    x_pos = np.clip(x, 0, None) * V_READ
    x_neg = np.clip(-x, 0, None) * V_READ

    path_pos, Gp, Gn, scale = generate_netlist(W1, x_pos, "crossbar_L1_xpos.spice", "L1 x+")
    path_neg, _, _, _ = generate_netlist(W1, x_neg, "crossbar_L1_xneg.spice", "L1 x-")

    out_pos, err_pos = run_ngspice(path_pos)
    with open(os.path.join(SCRIPT_DIR, "ngspice_xpos.log"), 'w') as f:
        f.write(out_pos + "\n--- STDERR ---\n" + err_pos)

    out_neg, err_neg = run_ngspice(path_neg)
    with open(os.path.join(SCRIPT_DIR, "ngspice_xneg.log"), 'w') as f:
        f.write(out_neg + "\n--- STDERR ---\n" + err_neg)

    Ip_pos, In_pos = parse_currents(out_pos, 32)
    Ip_neg, In_neg = parse_currents(out_neg, 32)

    if np.all(Ip_pos == 0) and np.all(In_pos == 0):
        print("\nERROR: No measurements from x+ simulation.")
        print("Last 30 lines of stderr:")
        for line in err_pos.strip().split('\n')[-30:]:
            print(f"  {line}")
        return

    # MAC = scale * [(Ip+ - In+) - (Ip- - In-)]
    I_diff = (Ip_pos - In_pos) - (Ip_neg - In_neg)
    hw_mac = I_diff * scale

    print("\n" + "=" * 70)
    print("CROSSBAR MAC VERIFICATION")
    print("=" * 70)
    print(f"{'Neuron':>8} {'SW MAC':>12} {'HW MAC':>12} {'I_diff (A)':>14} {'Err%':>8}")
    print("-" * 60)

    errors = []
    for j in range(32):
        if abs(sw_mac[j]) < 1e-6:
            err = abs(hw_mac[j] - sw_mac[j]) * 100
        else:
            err = abs(hw_mac[j] - sw_mac[j]) / abs(sw_mac[j]) * 100
        errors.append(err)
        print(f"{j:>8d} {sw_mac[j]:>12.6f} {hw_mac[j]:>12.6f} {I_diff[j]:>14.6e} {err:>7.2f}%")

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"\nMean error: {mean_err:.2f}%  Max: {max_err:.2f}%")

    if mean_err < 5:
        print("PASS: Crossbar MAC matches software within 5%.")
    elif mean_err < 15:
        print("OK: Within 15%. NMOS R_on causes some deviation.")
    else:
        print("WARN: Large error. NMOS R_on significant vs ReRAM resistance.")
        Ron_est = V_READ / (np.mean(np.abs(Ip_pos[Ip_pos != 0])) + 1e-15) - 1e6/G_MAX_US
        print(f"Estimated NMOS R_on: ~{Ron_est:.0f} ohm")

    # Full MLP inference
    h1 = np.maximum(0, sw_mac + b1)
    sw_out = W2.T @ h1 + b2
    pred = np.argmax(sw_out)
    names = ['N', 'S', 'V', 'F', 'Q']
    print(f"\nFull MLP: predicted={names[pred]} true={names[y_te[test_idx]]}")
    print(f"Output scores: {dict(zip(names, [f'{v:.3f}' for v in sw_out]))}")

    results = {
        'mlp_accuracy_pct': float((preds == y_te).mean() * 100),
        'test_beat': int(test_idx),
        'true_class': int(y_te[test_idx]),
        'predicted_class': int(pred),
        'scale_factor': float(scale),
        'mean_mac_error_pct': float(mean_err),
        'max_mac_error_pct': float(max_err),
        'sw_mac': sw_mac.tolist(),
        'hw_mac': hw_mac.tolist(),
        'n_1t1r_cells': int(W1.shape[0] * W1.shape[1] * 2),
        'G_range_uS': [float(G_MIN_US), float(G_MAX_US)],
        'v_read_V': float(V_READ),
    }
    with open(os.path.join(SCRIPT_DIR, 'crossbar_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to crossbar_results.json")


if __name__ == '__main__':
    main()
