"""
End-to-End Integration: Analog Frontend → ReRAM Crossbar Classifier
====================================================================
Two-pass approach for 96%+ DS2 inter-patient accuracy:

  Pass 1 (frontend): ECG PWL → ngspice .tran → 8 feature voltages per beat
  Pass 2 (classifier): Feature voltages → calibrated VCVS → crossbar → WTA → class

Strategy to preserve 96% accuracy:
  - Keep ReRAM weights UNCHANGED (trained on digital features, 96.04%)
  - Calibrate per-feature affine transform: V_row = A_i * V_analog_i + B_i
  - A_i and B_i are computed from linear regression between analog voltages
    and the standardized digital features they represent
  - This is equivalent to modifying only the VCVS input conditioning gains

Feature ordering (frontend → classifier):
  Frontend:   [F1:rr_ratio, F2:rr_asym, F3:bi_L0, F4:bi_L1, F5:pwave, F6:width, F7:slope, F8:sym]
  Classifier: [rr_ratio, bi_L1, rr_asym, pwave, width, bi_L0, slope, sym]
  Mapping:    classifier[0,1,2,3,4,5,6,7] = frontend[F1,F4,F2,F5,F6,F3,F7,F8]
"""
import os, sys, json, time, subprocess, re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.stats import spearmanr, pearsonr
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, "experiments", "Complete_imp", "classifier_crossbar")
INTEGRATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "integration")
CACHE_DIR = os.path.join(FRONTEND_DIR, "_cache")
DIGITAL_CACHE = os.path.join(PROJECT_ROOT, "experiments", "Complete_imp", "classifier", "features_8_cache.npz")

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"

# SKY130 PDK paths (classifier uses full PDK)
SKY130_NFET_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_NFET_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
SKY130_PFET_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"
SKY130_PFET_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__mismatch.corner.spice"
RERAM_MODEL = os.path.join(PROJECT_ROOT, "experiments", "Complete_imp",
                           "skywater-pdk-libs-sky130_fd_pr_reram").replace("\\", "/")

# Classifier constants (must match full_analog_mlp.py / generate_top_level.py)
G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15
X_OFFSET = 4.0

# De Chazal inter-patient protocol (48 MIT-BIH minus 4 paced: 102,104,107,217)
DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
LABEL_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

# Frontend-to-classifier feature reordering
# Frontend order:   [F1=rr_ratio, F2=rr_asym, F3=bi_L0, F4=bi_L1, F5=pwave, F6=width, F7=slope, F8=sym]
# Classifier order: [rr_ratio, bi_L1, rr_asym, pwave, width, bi_L0, slope, sym]
FRONTEND_TO_CLASSIFIER = [0, 3, 1, 4, 5, 2, 6, 7]  # classifier[i] = frontend[FRONTEND_TO_CLASSIFIER[i]]

FEATURE_NAMES_FRONTEND = ['rr_ratio', 'rr_asym', 'bi_L0', 'bi_L1', 'pwave', 'width', 'slope', 'symmetry']
FEATURE_NAMES_CLASSIFIER = ['rr_ratio', 'bi_L1', 'rr_asym', 'pwave', 'width', 'bi_L0', 'slope', 'symmetry']


# ==========================================================================
# PHASE 1: Load and validate analog feature cache
# ==========================================================================

def load_analog_cache(records):
    """Load analog frontend features from ngspice cache files.
    Returns features in FRONTEND order [F1..F8] and labels."""
    all_feats, all_labels, all_py = [], [], []
    for rec in records:
        cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
        if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
            print(f"  WARNING: Record {rec} not cached, skipping")
            continue
        with open(cache_file) as f:
            data = json.load(f)
        feats = np.array(data['features'])
        labels = [LABEL_MAP[l] for l in data['labels']]
        all_feats.append(feats)
        all_labels.extend(labels)
        if 'py_features' in data:
            all_py.append(np.array(data['py_features']))
    X = np.vstack(all_feats) if all_feats else np.empty((0, 8))
    y = np.array(all_labels)
    py = np.vstack(all_py) if all_py else None
    return X, y, py


def reorder_frontend_to_classifier(X_frontend):
    """Reorder features from frontend order to classifier order."""
    return X_frontend[:, FRONTEND_TO_CLASSIFIER]


# ==========================================================================
# PHASE 2: Calibrate per-feature affine transforms
# ==========================================================================

def calibrate_analog_to_digital(X_analog_frontend, X_digital_cache_classifier):
    """
    Find per-feature affine transform: feat_dig_i = alpha_i * V_analog_i + beta_i

    X_analog_frontend: (N, 8) in frontend order
    X_digital_cache_classifier: (N, 8) in classifier order

    Returns alpha (8,), beta (8,) in CLASSIFIER order.
    Also returns per-feature Pearson r and Spearman rho.
    """
    X_analog_classifier = reorder_frontend_to_classifier(X_analog_frontend)

    alpha = np.zeros(8)
    beta = np.zeros(8)
    pearson_r = np.zeros(8)
    spearman_rho = np.zeros(8)

    for i in range(8):
        v_analog = X_analog_classifier[:, i]
        f_digital = X_digital_cache_classifier[:, i]

        mask = np.isfinite(v_analog) & np.isfinite(f_digital)
        va = v_analog[mask]
        fd = f_digital[mask]

        A = np.vstack([va, np.ones(len(va))]).T
        result = np.linalg.lstsq(A, fd, rcond=None)
        alpha[i], beta[i] = result[0]

        pearson_r[i] = pearsonr(va, fd)[0]
        spearman_rho[i] = spearmanr(va, fd)[0]

    return alpha, beta, pearson_r, spearman_rho


def compute_vcvs_params(alpha, beta):
    """
    Compute VCVS parameters for each classifier input channel.

    The existing classifier expects:
      V_row_i = (feat_dig_i + X_OFFSET) * V_READ / (2 * X_OFFSET)

    With the calibrated transform feat_dig_i = alpha_i * V_analog_i + beta_i:
      V_row_i = (alpha_i * V_analog_i + beta_i + X_OFFSET) * V_READ / (2 * X_OFFSET)
              = [V_READ * alpha_i / (2 * X_OFFSET)] * V_analog_i
                + [V_READ * (beta_i + X_OFFSET) / (2 * X_OFFSET)]

    So: A_i = V_READ * alpha_i / (2 * X_OFFSET)
        B_i = V_READ * (beta_i + X_OFFSET) / (2 * X_OFFSET)

    Returns A (8,), B (8,) — the VCVS gain and offset per channel.
    """
    A = V_READ * alpha / (2 * X_OFFSET)
    B = V_READ * (beta + X_OFFSET) / (2 * X_OFFSET)
    return A, B


# ==========================================================================
# PHASE 3: Generate integrated classifier netlist
# ==========================================================================

def generate_integrated_classifier(A_vcvs, B_vcvs, output_path):
    """
    Generate a modified top_level_classifier.spice with VCVS conditioning
    calibrated for analog frontend voltage inputs.

    The netlist expects:
      feat0..feat7 = analog frontend voltage outputs (in CLASSIFIER order)

    The VCVS maps each to the correct crossbar row voltage:
      V_row_i = A_i * V_feat_i + B_i
    """
    w = np.load(os.path.join(CLASSIFIER_DIR, 'full_analog_weights.npz'))
    Gp1, Gn1 = w['Gp1'], w['Gn1']
    Gp2, Gn2 = w['Gp2'], w['Gn2']
    b1, b2 = w['b1'], w['b2']
    scale1, scale2 = float(w['scale1']), float(w['scale2'])

    PFET_DIFF_PARAMS = [
        'a0', 'agidl', 'ags', 'b0', 'b1', 'bgidl', 'cgidl', 'dlc', 'dwc',
        'eta0', 'k2', 'keta', 'kt1', 'ku0', 'kvsat', 'kvth0', 'lint', 'lku0',
        'lkvth0', 'nfactor', 'pclm', 'pdits', 'pditsd', 'rdsw', 'tvoff',
        'u0', 'ua', 'ub', 'voff', 'vsat', 'vth0', 'wint', 'wku0', 'wkvth0', 'wlod',
    ]

    lines = []
    lines.append("* =================================================================")
    lines.append("* INTEGRATED CLASSIFIER: Analog Frontend Voltages → Class Output")
    lines.append("* Architecture: 8 → 64 → 5 MLP on ReRAM Crossbar")
    lines.append("* VCVS calibrated for analog frontend voltage inputs")
    lines.append("* =================================================================")
    lines.append("*")
    lines.append("* Input:  feat0..feat7 (analog frontend voltages, CLASSIFIER order)")
    lines.append("*         feat0=rr_ratio, feat1=bi_L1, feat2=rr_asym, feat3=pwave,")
    lines.append("*         feat4=width, feat5=bi_L0, feat6=slope, feat7=symmetry")
    lines.append("* Output: out0..out4 (one-hot: N, S, V, F, Q)")
    lines.append("*")
    lines.append(f"* V_READ = {V_READ}V, G_range = [{G_MIN_US}, {G_MAX_US}] uS")
    lines.append(f"* L1 scale = {scale1:.4e}, L2 scale = {scale2:.4e}")
    lines.append("")

    # Model includes
    lines.append(f'.include "{SKY130_NFET_CORNER}"')
    lines.append(f'.include "{SKY130_NFET_MISMATCH}"')
    lines.append(f'.include "{SKY130_PFET_CORNER}"')
    lines.append(f'.include "{SKY130_PFET_MISMATCH}"')
    lines.append("")
    for p in PFET_DIFF_PARAMS:
        lines.append(f".param sky130_fd_pr__pfet_01v8__{p}_diff = 0")
    lines.append("")
    lines.append("Vdd vdd 0 1.8")
    lines.append("")

    # VCVS input conditioning — calibrated for analog frontend voltages
    lines.append("* ===== INPUT CONDITIONING (calibrated for analog frontend) =====")
    lines.append("* V_row_i = A_i * V_feat_i + B_i")
    lines.append("* Maps frontend voltage to crossbar row voltage preserving trained weights")
    lines.append("")
    for i in range(8):
        lines.append(f"* --- Classifier channel {i} ({FEATURE_NAMES_CLASSIFIER[i]}) ---")
        lines.append(f"* A={A_vcvs[i]:.8e}, B={B_vcvs[i]:.8e}")
        lines.append(f"Econd{i} row{i} 0 vol='v(feat{i})*{A_vcvs[i]:.8e} + {B_vcvs[i]:.8e}'")
        lines.append("")

    # L1 crossbar (8×64 differential)
    lines.append("* ===== L1: 8x64 DIFFERENTIAL 1T1R ReRAM CROSSBAR =====")
    lines.append(f"Vgate_l1 gate_l1 0 1.8")
    lines.append("")
    lines.append("* --- Positive conductance array ---")
    for i in range(8):
        for j in range(64):
            r = 1e6 / Gp1[i, j]
            lines.append(f"XTpl1_{i}_{j} row{i} gate_l1 mpl1_{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrpl1_{i}_{j} mpl1_{i}_{j} cpl1_{j} {r:.4f}")
    lines.append("")
    lines.append("* --- Negative conductance array ---")
    for i in range(8):
        for j in range(64):
            r = 1e6 / Gn1[i, j]
            lines.append(f"XTnl1_{i}_{j} row{i} gate_l1 mnl1_{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrnl1_{i}_{j} mnl1_{i}_{j} cnl1_{j} {r:.4f}")
    lines.append("")
    lines.append("* --- Column current sensing ---")
    for j in range(64):
        lines.append(f"Vspl1_{j} cpl1_{j} 0 0")
        lines.append(f"Vsnl1_{j} cnl1_{j} 0 0")
    lines.append("")

    # L1→L2 MAC connections (CCCS)
    lines.append("* ===== L1 CROSSBAR → INTERSTAGE CONNECTIONS =====")
    for j in range(64):
        lines.append(f"Fmac1p_{j} nmac1_{j} 0 Vspl1_{j} 1.0")
        lines.append(f"Fmac1n_{j} 0 nmac1_{j} Vsnl1_{j} 1.0")
    lines.append("")

    # Interstage: bias + ReLU + output mirrors (64 channels)
    RELU_W, RELU_L = 4.0, 0.5
    BIAS_W, BIAS_L = 2.0, 0.5
    lines.append("* ===== L1→L2 INTERSTAGE: BIAS + ReLU + OUTPUT (64 channels) =====")
    for j in range(64):
        I_bias = b1[j] / scale1
        lines.append(f"* --- Ch {j}: b1={b1[j]:+.4f}, I_bias={I_bias*1e6:+.3f}uA ---")
        if b1[j] >= 0:
            I_ref = abs(I_bias)
            lines.append(f"Ibr1_{j} pbr1_{j} 0 {I_ref:.6e}")
            lines.append(f"Xbr1_{j} pbr1_{j} pbr1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm1_{j} nmac1_{j} pbr1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        else:
            I_ref = abs(I_bias)
            lines.append(f"Ibr1_{j} vdd nbr1_{j} {I_ref:.6e}")
            lines.append(f"Xbr1_{j} nbr1_{j} nbr1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm1_{j} nmac1_{j} nbr1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        lines.append(f"Xr1a_{j} nmac1_{j} nmac1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u")
        lines.append(f"Xr1b_{j} nrelu1_{j} nmac1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u")
        lines.append(f"Xopr_{j} nrelu1_{j} nrelu1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        lines.append(f"Xopm_{j} vout1_{j} nrelu1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        lines.append("")

    # L2 crossbar (64×5 differential)
    lines.append("* ===== L2: 64x5 DIFFERENTIAL 1T1R ReRAM CROSSBAR =====")
    lines.append(f"Vgate_l2 gate_l2 0 1.8")
    lines.append("")
    lines.append("* --- Positive conductance array ---")
    for i in range(64):
        for j in range(5):
            r = 1e6 / Gp2[i, j]
            lines.append(f"XTpl2_{i}_{j} vout1_{i} gate_l2 mpl2_{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrpl2_{i}_{j} mpl2_{i}_{j} cpl2_{j} {r:.4f}")
    lines.append("")
    lines.append("* --- Negative conductance array ---")
    for i in range(64):
        for j in range(5):
            r = 1e6 / Gn2[i, j]
            lines.append(f"XTnl2_{i}_{j} vout1_{i} gate_l2 mnl2_{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrnl2_{i}_{j} mnl2_{i}_{j} cnl2_{j} {r:.4f}")
    lines.append("")
    lines.append("* --- Column current sensing ---")
    for j in range(5):
        lines.append(f"Vspl2_{j} cpl2_{j} 0 0")
        lines.append(f"Vsnl2_{j} cnl2_{j} 0 0")
    lines.append("")

    # L2→WTA MAC connections
    lines.append("* ===== L2 CROSSBAR → WTA CONNECTIONS =====")
    for k in range(5):
        lines.append(f"Fmac2p_{k} nwta_{k} 0 Vspl2_{k} 1.0")
        lines.append(f"Fmac2n_{k} 0 nwta_{k} Vsnl2_{k} 1.0")
    lines.append("")

    # L2 bias injection (5 channels)
    lines.append("* ===== L2→WTA BIAS INJECTION (5 channels) =====")
    for k in range(5):
        I_bias = b2[k] / scale2
        lines.append(f"* --- Class {CLASS_NAMES[k]}: b2={b2[k]:+.4f}, I_bias={I_bias*1e6:+.3f}uA ---")
        if b2[k] >= 0:
            I_ref = abs(I_bias)
            lines.append(f"Ibr2_{k} pbr2_{k} 0 {I_ref:.6e}")
            lines.append(f"Xbr2_{k} pbr2_{k} pbr2_{k} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm2_{k} nwta_{k} pbr2_{k} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        else:
            I_ref = abs(I_bias)
            lines.append(f"Ibr2_{k} vdd nbr2_{k} {I_ref:.6e}")
            lines.append(f"Xbr2_{k} nbr2_{k} nbr2_{k} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm2_{k} nwta_{k} nbr2_{k} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        lines.append("")

    # WTA (Lazzaro, 5-class)
    lines.append("* ===== WINNER-TAKE-ALL: 5-Input Lazzaro WTA =====")
    lines.append("Ibias_wta vdd nbias_wta 10u")
    lines.append("Xbias_wta nbias_wta nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
    lines.append("")
    for k in range(5):
        lines.append(f"* Cell {k} ({CLASS_NAMES[k]})")
        lines.append(f"Xwi{k} wcomm{k} nwta_{k} 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
        lines.append(f"Xwf{k} wcomm{k} wcomm{k} winhib 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
        lines.append(f"Xwl{k} out{k} wcomm{k} vdd vdd sky130_fd_pr__pfet_01v8 W=4.0u L=0.5u")
        lines.append(f"Xwd{k} out{k} nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=1.0u L=0.5u")
        lines.append("")
    lines.append("Xwinh winhib nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=10.0u L=0.5u")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Generated integrated classifier: {output_path} ({len(lines)} lines)")
    return output_path


# ==========================================================================
# PHASE 4: Per-beat classifier simulation
# ==========================================================================

def generate_beat_testbench(feat_voltages_classifier_order, beat_id, netlist_path):
    """
    Generate .op testbench for one beat through the integrated classifier.

    feat_voltages_classifier_order: (8,) array of analog frontend voltages
                                     already in classifier feature order.
    """
    lines = []
    lines.append(f"* Beat {beat_id} — integrated classification")
    lines.append(f'.include "{netlist_path}"')
    lines.append("")
    for i in range(8):
        lines.append(f"Vfeat{i} feat{i} 0 {feat_voltages_classifier_order[i]:.8f}")
    lines.append("")
    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    for k in range(5):
        lines.append(f"print v(out{k})")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    tb_path = os.path.join(INTEGRATION_DIR, f"_beat_{beat_id}.spice")
    with open(tb_path, 'w') as f:
        f.write('\n'.join(lines))
    return tb_path


def run_classifier_beat(tb_path, timeout=120):
    """Run ngspice .op simulation and return WTA output voltages."""
    result = subprocess.run(
        [NGSPICE, "-b", tb_path],
        capture_output=True, text=True, timeout=timeout, cwd=INTEGRATION_DIR
    )
    out_v = np.zeros(5)
    for line in result.stdout.split('\n'):
        s = line.strip().lower()
        for k in range(5):
            if f"v(out{k})" in s and '=' in s:
                try:
                    out_v[k] = float(s.split('=')[1].strip().split()[0])
                except:
                    pass
    return out_v


# ==========================================================================
# PHASE 5: Software-emulated classifier (fast validation)
# ==========================================================================

def classify_software_emulated(X_analog_frontend, alpha, beta):
    """
    Emulate the full analog classifier in software for fast validation.

    W1_eff and W2_eff are in the ORIGINAL feature space (not V_row space).
    They were extracted so that: Z1 = X_norm @ W1_eff + b1 ≈ Z1_software.

    Pipeline:
      1. Reorder features from frontend to classifier order
      2. Calibrate: feat_cal = alpha * V_analog + beta (maps to standardized space)
      3. Z1 = feat_cal @ W1_eff + b1
      4. H1 = ReLU(Z1)
      5. Z2 = H1 @ W2_eff + b2
      6. pred = argmax(Z2)
    """
    w = np.load(os.path.join(CLASSIFIER_DIR, 'full_analog_weights.npz'))
    W1_eff = w['W1_eff']
    W2_eff = w['W2_eff']
    b1, b2 = w['b1'], w['b2']

    X_cls = reorder_frontend_to_classifier(X_analog_frontend)
    X_cal = X_cls * alpha + beta

    Z1 = X_cal @ W1_eff + b1
    H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2_eff + b2

    predictions = np.argmax(Z2, axis=1)
    return predictions


# ==========================================================================
# MAIN: Run integration pipeline
# ==========================================================================

def main():
    print("=" * 74)
    print("  END-TO-END INTEGRATION: Analog Frontend → ReRAM Crossbar Classifier")
    print("=" * 74)
    print(f"  Strategy: Preserve trained weights (96.04%), calibrate VCVS only")
    print(f"  Protocol: De Chazal DS1 train / DS2 test (44 records)")
    print()

    # ── Step 1: Load digital feature cache (for calibration reference) ──
    print("[Step 1] Loading digital feature cache...")
    d = np.load(DIGITAL_CACHE)
    X_tr_dig, y_tr_dig = d['X_tr'], d['y_tr']
    X_te_dig, y_te_dig = d['X_te'], d['y_te']
    print(f"  Digital: train={X_tr_dig.shape}, test={X_te_dig.shape}")

    # ── Step 2: Load analog feature cache ──
    print("\n[Step 2] Loading analog frontend feature cache...")
    X_tr_ana, y_tr_ana, py_tr = load_analog_cache(DS1)
    X_te_ana, y_te_ana, py_te = load_analog_cache(DS2)
    print(f"  Analog: train={X_tr_ana.shape}, test={X_te_ana.shape}")

    if X_tr_ana.shape[0] == 0 or X_te_ana.shape[0] == 0:
        print("\n  ERROR: Analog feature cache is incomplete.")
        print("  Run full_accuracy_test.py first to populate the cache.")
        print("  Falling back to digital-only verification...")
        verify_digital_only()
        return

    # ── Step 3: Verify beat counts match ──
    print(f"\n[Step 3] Verifying data alignment...")
    print(f"  Analog train: {len(y_tr_ana)} beats, Digital train: {len(y_tr_dig)} beats")
    print(f"  Analog test:  {len(y_te_ana)} beats, Digital test:  {len(y_te_dig)} beats")
    if len(y_tr_ana) != len(y_tr_dig):
        print("  WARNING: Beat counts differ — using analog subset for calibration")

    # ── Step 4: Calibrate per-feature affine transforms ──
    print("\n[Step 4] Calibrating analog→digital feature transforms...")
    # Use training set for calibration
    n_cal = min(len(y_tr_ana), len(y_tr_dig))
    alpha, beta, r_pearson, rho_spearman = calibrate_analog_to_digital(
        X_tr_ana[:n_cal], X_tr_dig[:n_cal]
    )
    A_vcvs, B_vcvs = compute_vcvs_params(alpha, beta)

    print(f"\n  {'Feature':<12} {'alpha':>10} {'beta':>10} {'Pearson r':>10} {'Spearman ρ':>10} {'A_vcvs':>12} {'B_vcvs':>12}")
    print(f"  {'-'*76}")
    for i in range(8):
        print(f"  {FEATURE_NAMES_CLASSIFIER[i]:<12} {alpha[i]:>10.4f} {beta[i]:>10.4f} "
              f"{r_pearson[i]:>10.4f} {rho_spearman[i]:>10.4f} {A_vcvs[i]:>12.6e} {B_vcvs[i]:>12.6e}")

    # ── Step 5: Software-emulated end-to-end accuracy ──
    print("\n[Step 5] Software-emulated end-to-end accuracy (fast)...")
    pred_te = classify_software_emulated(X_te_ana, alpha, beta)
    acc = accuracy_score(y_te_ana, pred_te) * 100
    f1 = f1_score(y_te_ana, pred_te, average=None, zero_division=0)
    print(f"  Integrated accuracy: {acc:.2f}%")
    print(f"  F1: {' '.join(f'{CLASS_NAMES[i]}={f1[i]:.3f}' for i in range(5))}")

    # Compare to digital-only
    pred_dig = classify_software_emulated_digital(X_te_dig)
    acc_dig = accuracy_score(y_te_dig, pred_dig) * 100
    print(f"  Digital reference:   {acc_dig:.2f}%")
    print(f"  Δaccuracy: {acc - acc_dig:+.2f}%")

    # ── Step 6: Generate integrated classifier netlist ──
    print("\n[Step 6] Generating integrated classifier netlist...")
    netlist_path = os.path.join(INTEGRATION_DIR, "integrated_classifier.spice")
    generate_integrated_classifier(A_vcvs, B_vcvs, netlist_path)

    # ── Step 7: Save calibration results ──
    results = {
        'accuracy_integrated_pct': float(acc),
        'accuracy_digital_ref_pct': float(acc_dig),
        'delta_accuracy_pct': float(acc - acc_dig),
        'per_class_f1': {CLASS_NAMES[i]: float(f1[i]) for i in range(5)},
        'n_train': int(len(y_tr_ana)),
        'n_test': int(len(y_te_ana)),
        'calibration': {
            FEATURE_NAMES_CLASSIFIER[i]: {
                'alpha': float(alpha[i]),
                'beta': float(beta[i]),
                'pearson_r': float(r_pearson[i]),
                'spearman_rho': float(rho_spearman[i]),
                'A_vcvs': float(A_vcvs[i]),
                'B_vcvs': float(B_vcvs[i]),
            } for i in range(8)
        },
        'feature_reorder': FRONTEND_TO_CLASSIFIER,
    }
    results_path = os.path.join(INTEGRATION_DIR, "integration_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_path}")

    # ── Summary ──
    print("\n" + "=" * 74)
    print("  INTEGRATION SUMMARY")
    print("=" * 74)
    print(f"  Strategy:    Calibrated VCVS (preserve 96.04% trained weights)")
    print(f"  Integrated:  {acc:.2f}% ({len(y_te_ana)} DS2 beats)")
    print(f"  Digital ref: {acc_dig:.2f}% ({len(y_te_dig)} DS2 beats)")
    print(f"  Δ accuracy:  {acc - acc_dig:+.2f}%")
    target_met = "YES ✓" if acc >= 96.0 else "NO — needs investigation"
    print(f"  96% target:  {target_met}")
    print(f"  Netlist:     {netlist_path}")
    print(f"  Results:     {results_path}")


def classify_software_emulated_digital(X_digital):
    """Reference: classify using digital features directly (same path as full_analog_mlp.py).

    W1_eff is in original feature space: Z1 = X_norm @ W1_eff + b1
    No V_row transform needed — that's only for the physical crossbar."""
    w = np.load(os.path.join(CLASSIFIER_DIR, 'full_analog_weights.npz'))
    W1_eff, W2_eff = w['W1_eff'], w['W2_eff']
    b1, b2 = w['b1'], w['b2']

    Z1 = X_digital @ W1_eff + b1
    H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2_eff + b2
    return np.argmax(Z2, axis=1)


def verify_digital_only():
    """Fallback: verify digital-feature path while waiting for analog cache."""
    print("\n[Digital-only verification]")
    d = np.load(DIGITAL_CACHE)
    X_te, y_te = d['X_te'], d['y_te']
    pred = classify_software_emulated_digital(X_te)
    acc = accuracy_score(y_te, pred) * 100
    f1 = f1_score(y_te, pred, average=None, zero_division=0)
    print(f"  Digital software-emulated accuracy: {acc:.2f}%")
    print(f"  F1: {' '.join(f'{CLASS_NAMES[i]}={f1[i]:.3f}' for i in range(5))}")
    print(f"  (Reference: full_analog_results.json reported 96.04%)")

    cm = confusion_matrix(y_te, pred, labels=range(5))
    print(f"\n  Confusion matrix:")
    print(f"  {'':>6} " + " ".join(f"{c:>6}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>6} " + " ".join(f"{v:>6d}" for v in row))

    results = {
        'mode': 'digital_only_verification',
        'accuracy_pct': float(acc),
        'per_class_f1': {CLASS_NAMES[i]: float(f1[i]) for i in range(5)},
        'n_test': int(len(y_te)),
    }
    results_path = os.path.join(INTEGRATION_DIR, "digital_verification_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {results_path}")


if __name__ == '__main__':
    main()
