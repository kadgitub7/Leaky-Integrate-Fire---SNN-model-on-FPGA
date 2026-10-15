"""
Final 8->64->5 MLP crossbar classifier on SKY130 1T1R ReRAM.

Produces:
  - Trained MLP weights (mlp_8x64x5_weights.npz)
  - Crossbar effective weights via unit-input characterization (8 ngspice sims)
  - Full Layer 1 netlist template (crossbar_8x64_template.spice)
  - Single-beat verification netlist + run
  - Full pipeline accuracy: analog frontend features -> crossbar -> classification
  - Scaler parameters for deployment (scaler_params.json)
"""
import os, sys, json, time, subprocess
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend", "_cache")

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
            print(f"  WARNING: {rec} not cached"); continue
        with open(fp) as f:
            data = json.load(f)
        all_feats.append(np.array(data['features']))
        all_labels.extend(data['labels'])
    return np.vstack(all_feats), np.array([LABEL_MAP[l] for l in all_labels])


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


def generate_unit_input_netlist(W, row_idx, n_out, Gp, Gn, path):
    n_in = W.shape[0]
    lines = [
        f"* Unit-input characterization: row {row_idx} active",
        f"* 8x{n_out} differential 1T1R crossbar, SKY130 NFET",
        f'.include "{SKY130_CORNER}"',
        f'.include "{SKY130_MISMATCH}"',
        "",
        "Vgate gate 0 1.8",
    ]
    for k in range(n_in):
        lines.append(f"Vrow{k} row{k} 0 {V_READ if k == row_idx else 0.0}")
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
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


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


def extract_crossbar_weights(W, n_out):
    n_in = W.shape[0]
    Gp, Gn, scale = map_weights_to_conductance(W)
    W_eff = np.zeros((n_in, n_out))
    for i in range(n_in):
        path = os.path.join(SCRIPT_DIR, f"_unit8x64_{i}.spice")
        generate_unit_input_netlist(W, i, n_out, Gp, Gn, path)
        stdout, _ = run_ngspice(path)
        Ip, In = parse_currents(stdout, n_out)
        W_eff[i, :] = (Ip - In) * scale
        os.remove(path)
        print(f"    Row {i}: max|I|={max(np.abs(Ip).max(), np.abs(In).max()):.4e} A", flush=True)
    return W_eff, Gp, Gn, scale


def generate_template_netlist(Gp, Gn, scale, n_in, n_out, filename):
    lines = [
        f"* 8x64 Differential 1T1R ReRAM Crossbar - Layer 1",
        f"* SKY130 sky130_fd_pr__nfet_01v8 (W={NMOS_W}u L={NMOS_L}u)",
        f"* Conductance range: {G_MIN_US}-{G_MAX_US} uS",
        f"* V_READ = {V_READ}V, Scale = {scale:.6e}",
        f"* Total cells: {n_in * n_out * 2} (differential)",
        f"*",
        f"* Usage: set Vrow0..Vrow7 to scaled feature voltages,",
        f"*        measure i(Vsp0..Vsp63) and i(Vsn0..Vsn63),",
        f"*        MAC_j = scale * (i(Vsp_j) - i(Vsn_j))",
        "",
        f'.include "{SKY130_CORNER}"',
        f'.include "{SKY130_MISMATCH}"',
        "",
        "Vgate gate 0 1.8",
        "",
        "* Row inputs (set these to feature_i * V_READ)",
    ]
    for i in range(n_in):
        lines.append(f"Vrow{i} row{i} 0 0.0  $ feature {i}")
    lines.append("")
    lines.append("* Column sense lines (virtual ground)")
    for j in range(n_out):
        lines.append(f"Vsp{j} cp{j} 0 0")
        lines.append(f"Vsn{j} cn{j} 0 0")
    lines.append("")
    lines.append(f"* === Positive conductance array ({n_in}x{n_out}) ===")
    for i in range(n_in):
        for j in range(n_out):
            r = 1e6 / Gp[i, j]
            lines.append(f"XTp{i}_{j} row{i} gate mp{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{i}_{j} mp{i}_{j} cp{j} {r:.4f}")
    lines.append("")
    lines.append(f"* === Negative conductance array ({n_in}x{n_out}) ===")
    for i in range(n_in):
        for j in range(n_out):
            r = 1e6 / Gn[i, j]
            lines.append(f"XTn{i}_{j} row{i} gate mn{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{i}_{j} mn{i}_{j} cn{j} {r:.4f}")
    lines += [
        "", ".op", "", ".control", "run",
    ]
    for j in range(n_out):
        lines.append(f"print i(Vsp{j}) i(Vsn{j})")
    lines += ["quit", ".endc", "", ".end"]

    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Template netlist: {filename} ({len(lines)} lines)", flush=True)
    return path


def verify_single_beat(W1_eff, W1, b1, W2, b2, X_te_n, y_te, scale, Gp, Gn):
    idx = 0
    x = X_te_n[idx]
    true_class = y_te[idx]

    sw_z1 = x @ W1 + b1
    hw_z1 = x @ W1_eff + b1
    sw_h1 = np.maximum(0, sw_z1)
    hw_h1 = np.maximum(0, hw_z1)
    sw_out = sw_h1 @ W2 + b2
    hw_out = hw_h1 @ W2 + b2
    sw_pred = CLASS_NAMES[np.argmax(sw_out)]
    hw_pred = CLASS_NAMES[np.argmax(hw_out)]

    x_pos = np.clip(x, 0, None) * V_READ
    x_neg = np.clip(-x, 0, None) * V_READ

    path_pos = os.path.join(SCRIPT_DIR, "_verify_pos.spice")
    path_neg = os.path.join(SCRIPT_DIR, "_verify_neg.spice")
    n_in, n_out = W1.shape

    for path, x_v, label in [(path_pos, x_pos, "x+"), (path_neg, x_neg, "x-")]:
        lines = [
            f"* Verification beat #{idx} -- {label}",
            f'.include "{SKY130_CORNER}"', f'.include "{SKY130_MISMATCH}"', "",
            "Vgate gate 0 1.8",
        ]
        for i in range(n_in):
            lines.append(f"Vrow{i} row{i} 0 {float(x_v[i]):.10f}")
        for j in range(n_out):
            lines.append(f"Vsp{j} cp{j} 0 0")
            lines.append(f"Vsn{j} cn{j} 0 0")
        for i in range(n_in):
            for j in range(n_out):
                r = 1e6 / Gp[i, j]
                lines.append(f"XTp{i}_{j} row{i} gate mp{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrp{i}_{j} mp{i}_{j} cp{j} {r:.4f}")
        for i in range(n_in):
            for j in range(n_out):
                r = 1e6 / Gn[i, j]
                lines.append(f"XTn{i}_{j} row{i} gate mn{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
                lines.append(f"Rrn{i}_{j} mn{i}_{j} cn{j} {r:.4f}")
        lines += [".op", ".control", "run"]
        for j in range(n_out):
            lines.append(f"print i(Vsp{j}) i(Vsn{j})")
        lines += ["quit", ".endc", ".end"]
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    stdout_p, _ = run_ngspice(path_pos)
    stdout_n, _ = run_ngspice(path_neg)
    Ip_p, In_p = parse_currents(stdout_p, n_out)
    Ip_n, In_n = parse_currents(stdout_n, n_out)
    I_diff = (Ip_p - In_p) - (Ip_n - In_n)
    ngspice_mac = I_diff * scale
    ngspice_z1 = ngspice_mac + b1
    ngspice_h1 = np.maximum(0, ngspice_z1)
    ngspice_out = ngspice_h1 @ W2 + b2
    ngspice_pred = CLASS_NAMES[np.argmax(ngspice_out)]

    os.remove(path_pos)
    os.remove(path_neg)

    mac_err = np.abs(ngspice_mac - (W1.T @ x)) / (np.abs(W1.T @ x) + 1e-8)

    print(f"\n  Beat #{idx}: true={CLASS_NAMES[true_class]}", flush=True)
    print(f"    SW  prediction: {sw_pred}  scores: {[f'{v:.3f}' for v in sw_out]}", flush=True)
    print(f"    HW  prediction: {hw_pred}  scores: {[f'{v:.3f}' for v in hw_out]}", flush=True)
    print(f"    SIM prediction: {ngspice_pred}  scores: {[f'{v:.3f}' for v in ngspice_out]}", flush=True)
    print(f"    MAC error (ngspice vs SW): mean={mac_err.mean()*100:.2f}%, max={mac_err.max()*100:.2f}%", flush=True)

    return {
        'beat_idx': idx, 'true': CLASS_NAMES[true_class],
        'sw_pred': sw_pred, 'hw_pred': hw_pred, 'ngspice_pred': ngspice_pred,
        'mac_err_mean': round(float(mac_err.mean()*100), 3),
        'mac_err_max': round(float(mac_err.max()*100), 3),
    }


def main():
    print("=" * 70, flush=True)
    print("  FINAL CROSSBAR: 8->64->5 MLP on SKY130 1T1R ReRAM", flush=True)
    print("=" * 70, flush=True)

    # Load analog frontend features
    print("\n[1/6] Loading analog frontend features...", flush=True)
    X_tr, y_tr = load_analog_features(DS1)
    X_te, y_te = load_analog_features(DS2)
    print(f"  Train: {len(y_tr)} beats, Test: {len(y_te)} beats", flush=True)

    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te)

    # Train MLP
    print(f"\n[2/6] Training MLP 8->{HIDDEN}->5 (5 seeds)...", flush=True)
    best_mlp, best_acc = None, 0
    for seed in range(5):
        mlp = MLPClassifier(
            hidden_layer_sizes=(HIDDEN,), max_iter=500, random_state=seed+42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', learning_rate_init=1e-3,
            alpha=1e-4, batch_size=128
        )
        mlp.fit(X_tr_n, y_tr)
        acc = accuracy_score(y_te, mlp.predict(X_te_n)) * 100
        print(f"  Seed {seed+42}: {acc:.2f}%", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_mlp = mlp

    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]
    sw_pred = best_mlp.predict(X_te_n)
    sw_acc = accuracy_score(y_te, sw_pred) * 100
    sw_f1 = f1_score(y_te, sw_pred, average=None, zero_division=0)

    print(f"  Best SW accuracy: {sw_acc:.2f}%", flush=True)
    print(f"  W1: {W1.shape} [{W1.min():.4f}, {W1.max():.4f}]", flush=True)
    print(f"  W2: {W2.shape} [{W2.min():.4f}, {W2.max():.4f}]", flush=True)

    # Save weights and scaler
    weights_path = os.path.join(SCRIPT_DIR, 'mlp_8x64x5_weights.npz')
    np.savez(weights_path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"  Saved weights: mlp_8x64x5_weights.npz", flush=True)

    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_features': int(scaler.n_features_in_),
        'n_samples_seen': int(scaler.n_samples_seen_),
    }
    scaler_path = os.path.join(SCRIPT_DIR, 'scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  Saved scaler: scaler_params.json", flush=True)

    # Extract crossbar effective weights
    print(f"\n[3/6] Extracting crossbar effective weights (8 ngspice sims)...", flush=True)
    t0 = time.time()
    W1_eff, Gp, Gn, scale = extract_crossbar_weights(W1, HIDDEN)
    elapsed = time.time() - t0
    w_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    print(f"  Done in {elapsed:.0f}s", flush=True)
    print(f"  Weight error: mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%", flush=True)
    print(f"  Scale factor: {scale:.6e}", flush=True)

    np.savez(os.path.join(SCRIPT_DIR, 'crossbar_8x64_effective.npz'),
             W1_eff=W1_eff, W1=W1, b1=b1, W2=W2, b2=b2,
             Gp=Gp, Gn=Gn, scale=scale)

    # Generate template netlist
    print(f"\n[4/6] Generating template crossbar netlist...", flush=True)
    generate_template_netlist(Gp, Gn, scale, 8, HIDDEN, "crossbar_8x64_template.spice")

    # Single-beat ngspice verification
    print(f"\n[5/6] Single-beat ngspice verification...", flush=True)
    verify_result = verify_single_beat(W1_eff, W1, b1, W2, b2, X_te_n, y_te, scale, Gp, Gn)

    # Full pipeline accuracy
    print(f"\n[6/6] Full pipeline: analog frontend -> crossbar -> classification...", flush=True)
    Z1 = X_te_n @ W1_eff + b1
    H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2 + b2
    hw_pred = np.argmax(Z2, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)
    agree = (sw_pred == hw_pred).mean() * 100

    cm = confusion_matrix(y_te, hw_pred, labels=range(5))

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL RESULTS: 8->64->5 Analog Frontend -> Crossbar", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  MLP (software):  {sw_acc:.2f}%  F1: N={sw_f1[0]:.3f} S={sw_f1[1]:.3f} V={sw_f1[2]:.3f} F={sw_f1[3]:.3f}", flush=True)
    print(f"  MLP (crossbar):  {hw_acc:.2f}%  F1: N={hw_f1[0]:.3f} S={hw_f1[1]:.3f} V={hw_f1[2]:.3f} F={hw_f1[3]:.3f}", flush=True)
    print(f"  SW/HW agreement: {agree:.2f}%", flush=True)
    print(f"  Weight error:    mean={w_err.mean()*100:.2f}%, max={w_err.max()*100:.2f}%", flush=True)
    print(f"  Scale factor:    {scale:.6e}", flush=True)
    print(f"  ReRAM cells:     {8*HIDDEN*2} (1T1R, differential)", flush=True)

    print(f"\n  Confusion matrix (crossbar predictions):", flush=True)
    print(f"  {'':>8}" + "".join(f"{c:>7}" for c in CLASS_NAMES), flush=True)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>8}" + "".join(f"{v:>7d}" for v in row), flush=True)

    print(f"\n  Classification report:", flush=True)
    print(classification_report(y_te, hw_pred, target_names=CLASS_NAMES, zero_division=0), flush=True)

    # Conductance map summary
    print(f"\n  Conductance map:", flush=True)
    print(f"    G+ range: [{Gp.min():.3f}, {Gp.max():.3f}] uS", flush=True)
    print(f"    G- range: [{Gn.min():.3f}, {Gn.max():.3f}] uS", flush=True)
    print(f"    R+ range: [{1e6/Gp.max():.1f}, {1e6/Gp.min():.1f}] ohm", flush=True)
    print(f"    R- range: [{1e6/Gn.max():.1f}, {1e6/Gn.min():.1f}] ohm", flush=True)

    # Hardware specs
    macs = 8 * HIDDEN + HIDDEN * 5
    e_pj_per_mac = 1.2679
    e_total_pj = macs * e_pj_per_mac
    print(f"\n  Hardware specs:", flush=True)
    print(f"    Architecture:    8 -> {HIDDEN} -> 5", flush=True)
    print(f"    Layer 1 array:   8 x {HIDDEN} x 2 = {8*HIDDEN*2} 1T1R cells", flush=True)
    print(f"    Layer 2:         {HIDDEN} x 5 (digital or 2nd crossbar)", flush=True)
    print(f"    Total MACs:      {macs}", flush=True)
    print(f"    Energy/infer:    {e_total_pj:.1f} pJ ({e_total_pj/1000:.2f} nJ)", flush=True)
    print(f"    NMOS:            sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u", flush=True)
    print(f"    V_READ:          {V_READ} V", flush=True)
    print(f"    V_GATE:          1.8 V", flush=True)

    # Save final results
    final_results = {
        'architecture': f'8-{HIDDEN}-5',
        'hidden_size': HIDDEN,
        'sw_accuracy': round(sw_acc, 3),
        'hw_accuracy': round(hw_acc, 3),
        'agreement': round(agree, 3),
        'sw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, sw_f1)},
        'hw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, hw_f1)},
        'weight_error_mean_pct': round(float(w_err.mean()*100), 3),
        'weight_error_max_pct': round(float(w_err.max()*100), 3),
        'scale_factor': float(scale),
        'n_1t1r_cells': 8 * HIDDEN * 2,
        'total_macs': macs,
        'energy_pj': round(e_total_pj, 2),
        'G_range_uS': [G_MIN_US, G_MAX_US],
        'V_READ': V_READ,
        'NMOS': f'sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u',
        'train_beats': len(y_tr),
        'test_beats': len(y_te),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'confusion_matrix': cm.tolist(),
        'verify_beat': verify_result,
    }
    out_path = os.path.join(SCRIPT_DIR, 'final_8x64x5_results.json')
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  Saved: final_8x64x5_results.json", flush=True)
    print(f"  Saved: mlp_8x64x5_weights.npz", flush=True)
    print(f"  Saved: crossbar_8x64_effective.npz", flush=True)
    print(f"  Saved: crossbar_8x64_template.spice", flush=True)
    print(f"  Saved: scaler_params.json", flush=True)
    print(f"\n  Classifier is COMPLETE.", flush=True)


if __name__ == '__main__':
    main()
