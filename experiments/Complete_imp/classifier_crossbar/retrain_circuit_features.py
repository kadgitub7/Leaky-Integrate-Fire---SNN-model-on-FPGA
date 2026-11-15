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
V_COL_OFFSET = 0.52
COL_NFET_W = 50.0
COL_NFET_L = 1.0
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# SKY130 ReRAM PDK compact model parameters (from sky130_fd_pr_reram__reram_cell.va)
RERAM_I_K1 = 6.14e-5
RERAM_TOX = 5.0e-9
RERAM_TF_REF = 4.7249e-9
RERAM_V_REF = 0.430
RERAM_TF_MIN = 3.3e-9
RERAM_TF_MAX = 4.9e-9

PFET_MISMATCH_PARAMS = [
    'a0', 'agidl', 'ags', 'b0', 'b1', 'bgidl', 'cgidl', 'dlc', 'dwc',
    'eta0', 'k2', 'keta', 'kt1', 'ku0', 'kvsat', 'kvth0', 'lint', 'lku0',
    'lkvth0', 'nfactor', 'pclm', 'pdits', 'pditsd', 'rdsw', 'tvoff',
    'u0', 'ua', 'ub', 'uc', 'ute', 'voff', 'vsat', 'vth0', 'wint',
    'wku0', 'wkvth0', 'wlod',
]


def conductance_to_tfilament(G_uS):
    """Map target conductance (uS) to ReRAM filament thickness (m).

    Inverts the steady-state I-V: I = I_k1 * exp(-(Tox-Tf)/(Tox-Tf_ref)) * sinh(V/V_ref)
    at V=V_READ to find Tf0 that produces the target conductance.
    """
    G_S = np.asarray(G_uS, dtype=float) * 1e-6
    sinh_ratio = np.sinh(V_READ / RERAM_V_REF)
    exponent = np.log(G_S * V_READ / (RERAM_I_K1 * sinh_ratio))
    Tf0 = RERAM_TOX + (RERAM_TOX - RERAM_TF_REF) * exponent
    return np.clip(Tf0, RERAM_TF_MIN, RERAM_TF_MAX)


def reram_subckt_lines():
    """Return the SPICE subcircuit definition for a 1T1R ReRAM cell."""
    return [
        "* === 1T1R ReRAM cell (SKY130 PDK behavioral I-V) ===",
        ".subckt reram_1t1r row gate col gnd Tf0=3.3e-9",
        f".param Ik1={RERAM_I_K1}",
        f".param Tox={RERAM_TOX}",
        f".param Tfref={RERAM_TF_REF}",
        f".param Vref={RERAM_V_REF}",
        f"Xacc row gate mid gnd sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u",
        "Breram mid col I = Ik1*exp(-(Tox-Tf0)/(Tox-Tfref))*sinh(v(mid,col)/Vref)",
        ".ends reram_1t1r",
        "",
    ]


def col_readout_subckt_lines():
    """Return the SPICE subcircuit for differential column readout.
    Replaces Vsource(0V) + CCCS with real NFET/PFET current mirrors.
    G+ path: NFET diode + NFET mirror (sinks from nmac).
    G- path: NFET diode + NFET mirror + PFET conversion (sources into nmac).
    """
    return [
        "* === Differential column readout (replaces Vsource + CCCS) ===",
        f".subckt col_readout col_p col_n nmac vdd gnd",
        f"Xpd col_p col_p gnd gnd sky130_fd_pr__nfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u",
        f"Xpm nmac col_p gnd gnd sky130_fd_pr__nfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u",
        f"Xnd col_n col_n gnd gnd sky130_fd_pr__nfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u",
        f"Xnm mid_n col_n gnd gnd sky130_fd_pr__nfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u",
        f"Xpcd mid_n mid_n vdd vdd sky130_fd_pr__pfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u",
        f"Xpcm nmac mid_n vdd vdd sky130_fd_pr__pfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u",
        ".ends col_readout",
        "",
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
    Tf_p = conductance_to_tfilament(Gp)
    Tf_n = conductance_to_tfilament(Gn)
    W_eff = np.zeros((n_in, n_out))

    for i in range(n_in):
        lines = [
            f"* Unit-input row {i}: {n_in}x{n_out} diff 1T1R ReRAM crossbar ({label})",
            f"* Column readout: NFET diode W={COL_NFET_W}u L={COL_NFET_L}u, V_row offset={V_COL_OFFSET}V",
            f'.include "{SKY130_NFET_TT}"',
            f'.include "{SKY130_NFET_MM}"',
            "",
        ]
        lines.extend(reram_subckt_lines())
        lines.append("")
        for k in range(n_in):
            if k == i:
                lines.append(f"Vrow{k} row{k} 0 {V_READ + V_COL_OFFSET}")
                lines.append(f"Vgate{k} gate{k} 0 1.8")
            else:
                lines.append(f"Vrow{k} row{k} 0 0")
                lines.append(f"Vgate{k} gate{k} 0 0")
        lines.append("")
        for j in range(n_out):
            lines.append(f"Xcolpd{j} cp{j} cp{j} smp{j} 0 sky130_fd_pr__nfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u")
            lines.append(f"Vsp{j} smp{j} 0 0")
            lines.append(f"Xcolnd{j} cn{j} cn{j} smn{j} 0 sky130_fd_pr__nfet_01v8 W={COL_NFET_W}u L={COL_NFET_L}u")
            lines.append(f"Vsn{j} smn{j} 0 0")
        lines.append("")
        for k in range(n_in):
            for j in range(n_out):
                lines.append(f"Xrp{k}_{j} row{k} gate{k} cp{j} 0 reram_1t1r Tf0={Tf_p[k,j]:.6e}")
        lines.append("")
        for k in range(n_in):
            for j in range(n_out):
                lines.append(f"Xrn{k}_{j} row{k} gate{k} cn{j} 0 reram_1t1r Tf0={Tf_n[k,j]:.6e}")
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
    Tf_p = conductance_to_tfilament(Gp)
    Tf_n = conductance_to_tfilament(Gn)

    lines = [
        f"* =================================================================",
        f"* {layer_name}: {n_in}x{n_out} Differential 1T1R ReRAM Crossbar",
        f"* Technology: SKY130 sky130_fd_pr__nfet_01v8 + ReRAM PDK (behavioral sinh I-V)",
        f"* ReRAM model: sky130_fd_pr_reram__reram_cell (I_k1={RERAM_I_K1}, V_ref={RERAM_V_REF})",
        f"* =================================================================",
        f"*",
        f"* Conductance range: {G_MIN_US} - {G_MAX_US} uS (SKY130 ReRAM PDK)",
        f"* Tfilament range: {RERAM_TF_MIN*1e9:.1f} - {RERAM_TF_MAX*1e9:.1f} nm",
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
        f"* Column readout: NFET diode W={COL_NFET_W}u L={COL_NFET_L}u + col_readout subcircuit",
        f"* V_row offset: +{V_COL_OFFSET}V (compensates NFET diode column voltage)",
        "",
        f'.include "{SKY130_NFET_TT}"',
        f'.include "{SKY130_NFET_MM}"',
        f'.include "{SKY130_PFET_TT}"',
        f'.include "{SKY130_PFET_MM}"',
        "",
    ]
    for p in PFET_MISMATCH_PARAMS:
        lines.append(f".param sky130_fd_pr__pfet_01v8__{p}_diff = 0")
    lines.append("")
    lines.extend(reram_subckt_lines())
    lines.extend(col_readout_subckt_lines())
    lines += [
        "Vgate gate 0 1.8",
        "VDD vdd 0 1.8",
        "",
        "* --- Row inputs ---",
    ]
    for i in range(n_in):
        lines.append(f"Vrow{i} row{i} 0 0.0")
    lines.append("")
    lines.append(f"* --- Column readout (NFET diode + col_readout subcircuit) ---")
    for j in range(n_out):
        lines.append(f"Xcolrd{j} cp{j} cn{j} nmac{j} vdd 0 col_readout")

    lines.append("")
    lines.append(f"* ===== POSITIVE CONDUCTANCE ARRAY ({n_in}x{n_out}) =====")
    for i in range(n_in):
        lines.append(f"* --- Row {i} ---")
        for j in range(n_out):
            g = Gp[i, j]
            tf = Tf_p[i, j]
            lines.append(f"Xrp{i}_{j} row{i} gate cp{j} 0 reram_1t1r Tf0={tf:.6e}  $ G+={g:.3f}uS w={W[i,j]:.4f}")

    lines.append("")
    lines.append(f"* ===== NEGATIVE CONDUCTANCE ARRAY ({n_in}x{n_out}) =====")
    for i in range(n_in):
        lines.append(f"* --- Row {i} ---")
        for j in range(n_out):
            g = Gn[i, j]
            tf = Tf_n[i, j]
            lines.append(f"Xrn{i}_{j} row{i} gate cn{j} 0 reram_1t1r Tf0={tf:.6e}  $ G-={g:.3f}uS w={W[i,j]:.4f}")

    lines += ["", ".op", "", ".control", "run"]
    for j in range(n_out):
        lines.append(f"print i(Vsp{j}) i(Vsn{j})")
    lines += ["quit", ".endc", "", ".end"]

    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, len(lines)


def retrain_biases_for_hardware(X_tr_n, y_tr, W1_eff, b1, W2_eff, b2, n_classes=5, lr=0.01, epochs=200):
    """Retrain biases b1, b2 to compensate for crossbar weight distortion.

    Freezes W1_eff, W2_eff (from ngspice characterization) and optimizes
    b1, b2 using gradient descent on cross-entropy loss over training set.
    This recovers accuracy lost from systematic weight scaling errors.
    """
    from scipy.special import softmax as sp_softmax
    N = len(y_tr)
    b1_opt = b1.copy()
    b2_opt = b2.copy()
    Y_onehot = np.zeros((N, n_classes))
    Y_onehot[np.arange(N), y_tr] = 1.0

    best_acc = 0.0
    best_b1, best_b2 = b1_opt.copy(), b2_opt.copy()

    for epoch in range(epochs):
        Z1 = X_tr_n @ W1_eff + b1_opt
        H1 = np.maximum(0, Z1)
        Z2 = H1 @ W2_eff + b2_opt
        P = sp_softmax(Z2, axis=1)

        acc = (np.argmax(Z2, axis=1) == y_tr).mean()
        if acc > best_acc:
            best_acc = acc
            best_b1 = b1_opt.copy()
            best_b2 = b2_opt.copy()

        dZ2 = (P - Y_onehot) / N
        db2 = dZ2.sum(axis=0)
        dH1 = dZ2 @ W2_eff.T
        dZ1 = dH1 * (Z1 > 0).astype(float)
        db1 = dZ1.sum(axis=0)

        b2_opt -= lr * db2
        b1_opt -= lr * db1

    return best_b1, best_b2, best_acc


def calibrate_mirror_widths(target_currents_uA, mirror_type='nfet',
                            W_ref=2.0, L=0.5, I_ref=1.0):
    """Find calibrated W for each target current using ngspice sweep.

    Runs a single ngspice sim that sweeps mirror W from 0.5u to 60u,
    measures actual output current, then interpolates for each target.
    Returns array of calibrated W values.

    mirror_type: 'nfet' for NFET mirrors (negative bias channels),
                 'pfet' for PFET mirrors (positive bias channels).
    """
    from scipy.interpolate import interp1d

    W_sweep = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
               10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
    n_pts = len(W_sweep)

    lines = [
        f"* Mirror calibration sweep: {mirror_type.upper()} W_ref={W_ref}u L={L}u I_ref={I_ref}uA",
        f'.include "{SKY130_NFET_TT}"',
        f'.include "{SKY130_NFET_MM}"',
        f'.include "{SKY130_PFET_TT}"',
        f'.include "{SKY130_PFET_MM}"',
    ]
    for p in PFET_MISMATCH_PARAMS:
        lines.append(f".param sky130_fd_pr__pfet_01v8__{p}_diff = 0")
    lines.append("")
    lines.append("VDD vdd 0 1.8")
    lines.append("")

    if mirror_type == 'nfet':
        lines.append(f"Ibias vdd nbias {I_ref}u")
        lines.append(f"Xd nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W_ref}u L={L}u")
        for i, ww in enumerate(W_sweep):
            lines.append(f"Xm{i} sm{i} nbias 0 0 sky130_fd_pr__nfet_01v8 W={ww}u L={L}u")
            lines.append(f"Vs{i} vdd sm{i} 0")
    else:
        lines.append(f"Ibias pbias 0 {I_ref}u")
        lines.append(f"Xd pbias pbias vdd vdd sky130_fd_pr__pfet_01v8 W={W_ref}u L={L}u")
        for i, ww in enumerate(W_sweep):
            lines.append(f"Xm{i} sm{i} pbias vdd vdd sky130_fd_pr__pfet_01v8 W={ww}u L={L}u")
            lines.append(f"Vs{i} sm{i} 0 0")

    lines += ["", ".op", ".control", "run"]
    for i in range(n_pts):
        lines.append(f"print @vs{i}[i]")
    lines += ["quit", ".endc", ".end"]

    path = os.path.join(SCRIPT_DIR, f"_mirror_cal_{mirror_type}.spice")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    stdout, stderr = run_ngspice(path)
    os.remove(path)

    I_measured = []
    for line in stdout.split('\n'):
        s = line.strip().lower()
        for i in range(n_pts):
            tag = f"@vs{i}[i]"
            if tag in s and '=' in s:
                try:
                    val = float(s.split('=')[1].strip().split()[0])
                    I_measured.append((i, abs(val)))
                except:
                    pass

    I_by_idx = {}
    for idx, val in I_measured:
        I_by_idx[idx] = val

    W_valid = []
    I_valid = []
    for i, ww in enumerate(W_sweep):
        if i in I_by_idx and I_by_idx[i] > 0:
            W_valid.append(ww)
            I_valid.append(I_by_idx[i] * 1e6)

    if len(W_valid) < 3:
        return np.full(len(target_currents_uA), W_ref)

    interp_fn = interp1d(I_valid, W_valid, kind='linear',
                         fill_value='extrapolate', bounds_error=False)

    W_calibrated = interp_fn(np.abs(target_currents_uA))
    W_calibrated = np.clip(W_calibrated, 0.42, 100.0)
    return W_calibrated


def generate_top_level_classifier(W1, b1, W2, b2, Gp1, Gn1, scale1,
                                   Gp2, Gn2, scale2,
                                   scaler_mean, scaler_scale, feature_names,
                                   mirror_W_l1=None, mirror_W_l2=None,
                                   mirror_W_wta_n=None, mirror_W_wta_p=None,
                                   filename='top_level_classifier_circuit.spice'):
    """Generate complete top-level classifier netlist.

    Replaces ideal VCVS with B-sources, ideal Vsource+CCCS with col_readout
    subcircuit (real SKY130 transistor current mirrors). When mirror_W arrays
    are provided, bias sources use calibrated transistor mirrors instead of
    ideal current sources.

    mirror_W_l1: array of calibrated W values per L1 channel (64)
    mirror_W_l2: array of calibrated W values per L2 channel (5)
    mirror_W_wta_n: calibrated W for WTA NFET bias mirror
    mirror_W_wta_p: calibrated W for WTA PFET bias mirror
    """
    n_l1_in, n_l1_out = W1.shape
    n_l2_in, n_l2_out = W2.shape
    Tf_p1 = conductance_to_tfilament(Gp1)
    Tf_n1 = conductance_to_tfilament(Gn1)
    Tf_p2 = conductance_to_tfilament(Gp2)
    Tf_n2 = conductance_to_tfilament(Gn2)

    lines = []

    # ---- Header ----
    lines += [
        "* =================================================================",
        "* TOP-LEVEL: Complete Analog ECG Classifier",
        f"* Architecture: {n_l1_in} -> {n_l1_out} -> {n_l2_out} MLP on ReRAM Crossbar",
        "* Technology: SKY130 (NFET/PFET 01v8 + ReRAM)",
        "* Column readout: real NFET/PFET mirrors (col_readout subcircuit)",
        "* Input conditioning: B-source with V_COL_OFFSET",
        "* =================================================================",
        "*",
        f"* L1 scale: {scale1:.4e},  L2 scale: {scale2:.4e}",
        f"* V_READ: {V_READ}V,  V_COL_OFFSET: {V_COL_OFFSET}V,  VDD: 1.8V",
        f"* L1 cells: {n_l1_in*n_l1_out*2} 1T1R,  L2 cells: {n_l2_in*n_l2_out*2} 1T1R",
        "* =================================================================",
        "",
    ]

    # ---- Includes ----
    lines += [
        f'.include "{SKY130_NFET_TT}"',
        f'.include "{SKY130_NFET_MM}"',
        f'.include "{SKY130_PFET_TT}"',
        f'.include "{SKY130_PFET_MM}"',
    ]
    for p in PFET_MISMATCH_PARAMS:
        lines.append(f".param sky130_fd_pr__pfet_01v8__{p}_diff = 0")
    lines.append("")

    # ---- Subcircuits ----
    lines.extend(reram_subckt_lines())
    lines.extend(col_readout_subckt_lines())

    # ---- Supply ----
    lines += [
        "* ===== SUPPLY =====",
        "VDD vdd 0 1.8",
        "",
    ]

    # ---- Feature inputs ----
    lines.append("* ===== FEATURE INPUTS =====")
    for i in range(n_l1_in):
        lines.append(f"Vfeat{i} feat{i} 0 0.0")
    lines.append("")

    # ---- Input conditioning (B-sources with V_COL_OFFSET) ----
    lines.append(f"* ===== INPUT CONDITIONING ({n_l1_in} channels) =====")
    lines.append(f"* B-source: V_row = gain*V_feat + offset + V_COL_OFFSET")
    lines.append(f"* gain = V_READ/(8*scale), offset = V_READ*(4-mean/scale)/8")
    lines.append(f"* V_COL_OFFSET = {V_COL_OFFSET}V compensates NFET diode column voltage")
    lines.append("")
    for i in range(n_l1_in):
        mean_i = float(scaler_mean[i])
        scale_i = float(scaler_scale[i])
        gain = V_READ / (8.0 * scale_i)
        offset_base = V_READ * (4.0 - mean_i / scale_i) / 8.0
        offset_total = offset_base + V_COL_OFFSET
        fname = str(feature_names[i]) if feature_names is not None else f"feat{i}"
        lines.append(f"* --- Feature {i} ({fname}): mean={mean_i:.6f}, scale={scale_i:.6f} ---")
        lines.append(f"* gain={gain:.8e} V/V, offset_base={offset_base:.8e}, +V_COL_OFFSET={V_COL_OFFSET}")
        lines.append(f"Bcond{i} row{i} 0 V = v(feat{i})*{gain:.8e} + {offset_total:.8e}")
        lines.append("")

    # ---- L1 Crossbar ----
    lines.append(f"* ===== LAYER 1: {n_l1_in}x{n_l1_out} DIFFERENTIAL 1T1R ReRAM CROSSBAR =====")
    lines.append("Vgate_l1 gate_l1 0 1.8")
    lines.append("")
    lines.append("* --- Positive conductance array ---")
    for i in range(n_l1_in):
        for j in range(n_l1_out):
            lines.append(f"Xrpl1_{i}_{j} row{i} gate_l1 cpl1_{j} 0 reram_1t1r Tf0={Tf_p1[i,j]:.6e}  $ G+={Gp1[i,j]:.3f}uS")
    lines.append("")
    lines.append("* --- Negative conductance array ---")
    for i in range(n_l1_in):
        for j in range(n_l1_out):
            lines.append(f"Xrnl1_{i}_{j} row{i} gate_l1 cnl1_{j} 0 reram_1t1r Tf0={Tf_n1[i,j]:.6e}  $ G-={Gn1[i,j]:.3f}uS")
    lines.append("")

    # ---- L1 Column readout (replaces Vsource+CCCS) ----
    lines.append(f"* ===== L1 COLUMN READOUT ({n_l1_out} col_readout instances) =====")
    lines.append("* Replaces 128 Vsource(0V) + 128 CCCS with real transistor mirrors")
    for j in range(n_l1_out):
        lines.append(f"Xcolrd1_{j} cpl1_{j} cnl1_{j} nmac1_{j} vdd 0 col_readout")
    lines.append("")

    # ---- L1→L2 Interstage: bias + ReLU + output ----
    lines.append(f"* ===== L1->L2 INTERSTAGE: BIAS + ReLU + OUTPUT ({n_l1_out} channels) =====")
    lines.append("")
    for j in range(n_l1_out):
        I_bias = float(b1[j]) * scale1
        lines.append(f"* --- Ch {j}: b1={b1[j]:+.4f}, I_bias={I_bias/1e-6:+.3f}uA ---")
        if b1[j] < 0:
            lines.append(f"Ibr1_{j} vdd nbr1_{j} {abs(I_bias):.6e}")
            lines.append(f"Xbr1_{j} nbr1_{j} nbr1_{j} 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
            lines.append(f"Xbm1_{j} nmac1_{j} nbr1_{j} 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
        else:
            lines.append(f"Ibr1_{j} pbr1_{j} 0 {abs(I_bias):.6e}")
            lines.append(f"Xbr1_{j} pbr1_{j} pbr1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W=2.0u L=0.5u")
            lines.append(f"Xbm1_{j} nmac1_{j} pbr1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W=2.0u L=0.5u")
        lines.append(f"Xr1a_{j} nmac1_{j} nmac1_{j} 0 0 sky130_fd_pr__nfet_01v8 W=4.0u L=0.5u")
        lines.append(f"Xr1b_{j} nrelu1_{j} nmac1_{j} 0 0 sky130_fd_pr__nfet_01v8 W=4.0u L=0.5u")
        lines.append(f"Xopr_{j} nrelu1_{j} nrelu1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W=2.0u L=0.5u")
        lines.append(f"Xopm_{j} vout1_{j} nrelu1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W=2.0u L=0.5u")
        lines.append("")

    # ---- L2 Crossbar ----
    lines.append(f"* ===== LAYER 2: {n_l2_in}x{n_l2_out} DIFFERENTIAL 1T1R ReRAM CROSSBAR =====")
    lines.append("Vgate_l2 gate_l2 0 1.8")
    lines.append("")
    lines.append("* --- Positive conductance array ---")
    for i in range(n_l2_in):
        for j in range(n_l2_out):
            lines.append(f"Xrpl2_{i}_{j} vout1_{i} gate_l2 cpl2_{j} 0 reram_1t1r Tf0={Tf_p2[i,j]:.6e}  $ G+={Gp2[i,j]:.3f}uS")
    lines.append("")
    lines.append("* --- Negative conductance array ---")
    for i in range(n_l2_in):
        for j in range(n_l2_out):
            lines.append(f"Xrnl2_{i}_{j} vout1_{i} gate_l2 cnl2_{j} 0 reram_1t1r Tf0={Tf_n2[i,j]:.6e}  $ G-={Gn2[i,j]:.3f}uS")
    lines.append("")

    # ---- L2 Column readout (replaces Vsource+CCCS) ----
    lines.append(f"* ===== L2 COLUMN READOUT ({n_l2_out} col_readout instances) =====")
    for j in range(n_l2_out):
        lines.append(f"Xcolrd2_{j} cpl2_{j} cnl2_{j} nwta_{j} vdd 0 col_readout")
    lines.append("")

    # ---- L2 Bias injection ----
    lines.append(f"* ===== L2->WTA BIAS INJECTION ({n_l2_out} channels) =====")
    lines.append("")
    for j in range(n_l2_out):
        I_bias = float(b2[j]) * scale2
        lines.append(f"* --- Class {CLASS_NAMES[j]}: b2={b2[j]:+.4f}, I_bias={I_bias/1e-6:+.3f}uA ---")
        if b2[j] < 0:
            lines.append(f"Ibr2_{j} vdd nbr2_{j} {abs(I_bias):.6e}")
            lines.append(f"Xbr2_{j} nbr2_{j} nbr2_{j} 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
            lines.append(f"Xbm2_{j} nwta_{j} nbr2_{j} 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
        else:
            lines.append(f"Ibr2_{j} pbr2_{j} 0 {abs(I_bias):.6e}")
            lines.append(f"Xbr2_{j} pbr2_{j} pbr2_{j} vdd vdd sky130_fd_pr__pfet_01v8 W=2.0u L=0.5u")
            lines.append(f"Xbm2_{j} nwta_{j} pbr2_{j} vdd vdd sky130_fd_pr__pfet_01v8 W=2.0u L=0.5u")
        lines.append("")

    # ---- WTA ----
    lines += [
        "* ===== WINNER-TAKE-ALL: PFET-Input Max Detector =====",
        "* Winner = LOWEST output (argmin).",
        "* NFET bias",
        "Ibias_wta vdd nbias_wta 10u",
        "Xbias_wta nbias_wta nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u",
        "* PFET bias (tail current ~10uA)",
        "Ibias_wta_p pbias_wta 0 10u",
        "Xbias_wta_p pbias_wta pbias_wta vdd vdd sky130_fd_pr__pfet_01v8 W=4.0u L=0.5u",
        "",
        '.include "wta_5class_fixed.spice"',
        "XWTA nwta_0 nwta_1 nwta_2 nwta_3 nwta_4 out0 out1 out2 out3 out4 vdd 0 nbias_wta pbias_wta wta_5class",
        "",
    ]

    # ---- Analysis ----
    lines += [
        "* ===== ANALYSIS =====",
        ".op",
        "",
        ".control",
        "run",
    ]
    for j in range(n_l2_out):
        lines.append(f"print v(out{j})")
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

    # Train MLP — fixed seed 42 for deployment (no cherry-picking on test set)
    # 10-seed sweep reported for statistical context only
    DEPLOY_SEED = 42
    w(f"\n[1/7] Training MLP 8->64->5 (deploy seed={DEPLOY_SEED}, 10-seed stats)...\n")
    deploy_mlp = None
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
        if seed == DEPLOY_SEED:
            deploy_mlp = mlp

    w(f"  Mean: {np.mean(all_accs):.2f} +/- {np.std(all_accs):.2f}%\n")
    deploy_acc = all_accs[0]
    w(f"  Deploy seed {DEPLOY_SEED}: {deploy_acc:.2f}%\n")

    best_mlp = deploy_mlp
    best_seed = DEPLOY_SEED
    W1 = best_mlp.coefs_[0]; b1 = best_mlp.intercepts_[0]
    W2 = best_mlp.coefs_[1]; b2 = best_mlp.intercepts_[1]
    sw_pred = best_mlp.predict(X_te_n)

    w(f"\n  Classification report (SW MLP on circuit features):\n")
    w(classification_report(y_te, sw_pred, target_names=CLASS_NAMES, zero_division=0) + "\n")

    # Layer 1 crossbar characterization
    w(f"\n[2/7] Layer 1: extracting 8x64 effective weights (8 ngspice sims)...\n")
    t0 = time.time()
    W1_eff, Gp1, Gn1, scale1 = extract_crossbar_weights(W1, "L1")
    if W1_eff is None:
        w("  FAILED Layer 1\n"); return
    w1_err = np.abs(W1_eff - W1) / (np.abs(W1) + 1e-8)
    w(f"  Done in {time.time()-t0:.0f}s. Error: mean={w1_err.mean()*100:.2f}%, max={w1_err.max()*100:.2f}%\n")

    # Layer 2 crossbar characterization
    w(f"\n[3/7] Layer 2: extracting 64x5 effective weights (64 ngspice sims)...\n")
    t0 = time.time()
    W2_eff, Gp2, Gn2, scale2 = extract_crossbar_weights(W2, "L2")
    if W2_eff is None:
        w("  FAILED Layer 2\n"); return
    w2_err = np.abs(W2_eff - W2) / (np.abs(W2) + 1e-8)
    w(f"  Done in {time.time()-t0:.0f}s. Error: mean={w2_err.mean()*100:.2f}%, max={w2_err.max()*100:.2f}%\n")

    # Generate netlists
    w(f"\n[4/7] Generating crossbar netlists...\n")
    _, n1 = generate_crossbar_netlist(Gp1, Gn1, scale1, W1, b1,
        "LAYER 1 (circuit features)", "L1_crossbar_8x64_circuit.spice")
    w(f"  L1_crossbar_8x64_circuit.spice ({n1} lines)\n")

    _, n2 = generate_crossbar_netlist(Gp2, Gn2, scale2, W2, b2,
        "LAYER 2 (circuit features)", "L2_crossbar_64x5_circuit.spice")
    w(f"  L2_crossbar_64x5_circuit.spice ({n2} lines)\n")

    # Full two-crossbar accuracy
    w(f"\n[5/7] Full two-crossbar pipeline accuracy...\n")

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

    # Generate top-level classifier netlist
    n_l1_out = W1.shape[1]
    n_l2_out = W2.shape[1]
    w(f"\n[6/7] Generating top-level classifier netlist...\n")
    tl_path, tl_lines = generate_top_level_classifier(
        W1, b1, W2, b2, Gp1, Gn1, scale1, Gp2, Gn2, scale2,
        scaler_mean, scaler_scale, feature_names)
    w(f"  {os.path.basename(tl_path)} ({tl_lines} lines)\n")
    w(f"  Replaced: 8 VCVS -> B-sources, {n_l1_out+n_l2_out} Vsource+CCCS -> col_readout\n")
    w(f"  Remaining ideal: {n_l1_out+n_l2_out+3} bias current sources (pending mirror calibration)\n")

    # Save
    w(f"\n[7/7] Saving weights and results...\n")
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
    w(f"    L1_crossbar_8x64_circuit.spice          -- Layer 1 crossbar\n")
    w(f"    L2_crossbar_64x5_circuit.spice          -- Layer 2 crossbar\n")
    w(f"    top_level_classifier_circuit.spice       -- Complete top-level (B-sources + col_readout)\n")
    w(f"    circuit_analog_weights.npz              -- Weights, conductances, scaler\n")
    w(f"    circuit_retrain_results.json            -- All metrics\n")
    w(f"\n  Done.\n")


if __name__ == '__main__':
    main()
