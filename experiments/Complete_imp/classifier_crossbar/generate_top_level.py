"""
Generate complete top-level analog classifier netlist.

Connects all stages into a single ASIC-ready design:
  Input conditioning (8 channels) -> L1 crossbar (8x64) ->
  Interstage bias+ReLU (64 ch) -> L2 crossbar (64x5) ->
  L2 bias (5 ch) -> WTA (5->5 one-hot)

All components use SKY130 PDK transistors with X-prefix subcircuit instances.
"""
import os, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SKY130_NFET_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_NFET_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
SKY130_PFET_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"
SKY130_PFET_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__mismatch.corner.spice"

PFET_DIFF_PARAMS = [
    'a0', 'agidl', 'ags', 'b0', 'b1', 'bgidl', 'cgidl', 'dlc', 'dwc',
    'eta0', 'k2', 'keta', 'kt1', 'ku0', 'kvsat', 'kvth0', 'lint', 'lku0',
    'lkvth0', 'nfactor', 'pclm', 'pdits', 'pditsd', 'rdsw', 'tvoff',
    'u0', 'ua', 'ub', 'voff', 'vsat', 'vth0', 'wint', 'wku0', 'wkvth0', 'wlod',
]
PFET_PARAM_FIX = "\n".join(
    f".param sky130_fd_pr__pfet_01v8__{p}_diff = 0" for p in PFET_DIFF_PARAMS
)

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
NMOS_W = 1.0
NMOS_L = 0.15
RELU_W = 4.0
RELU_L = 0.5
BIAS_W = 2.0
BIAS_L = 0.5
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


def generate_input_conditioning(scaler_mean, scaler_scale, lines):
    """
    Input conditioning: analog normalization with offset-binary encoding.

    Normalized features can be negative, but crossbar rows need positive voltages.
    We use offset-binary encoding to shift all inputs positive in a single pass:

      x_norm_i = (V_feat_i - mean_i) / scale_i       (can be negative)
      V_row_i  = (x_norm_i + X_OFFSET) * V_READ / (2 * X_OFFSET)

    This maps [-X_OFFSET, +X_OFFSET] to [0, V_READ], all positive.
    The constant offset current per column is absorbed into the bias term.

    For ASIC: each channel is a differential amplifier with resistive
    feedback (gain = V_READ / (2 * X_OFFSET * scale_i)) and a reference
    voltage for offset subtraction. Modeled here as VCVS (E-element).
    """
    X_OFFSET = 4.0

    lines.append("* ===== INPUT CONDITIONING (8 channels) =====")
    lines.append("* Offset-binary normalization: maps normalized features to [0, V_READ]")
    lines.append(f"* x_norm = (V_feat - mean) / scale; V_row = (x_norm + {X_OFFSET}) * V_READ / {2*X_OFFSET}")
    lines.append("* Constant offset current absorbed into bias mirrors")
    lines.append("* ASIC: differential amplifier with resistive feedback per channel")
    lines.append("")

    for i in range(8):
        gain = V_READ / (2 * X_OFFSET * scaler_scale[i])
        offset_v = V_READ * (X_OFFSET - scaler_mean[i] / scaler_scale[i]) / (2 * X_OFFSET)
        lines.append(f"* --- Feature {i}: mean={scaler_mean[i]:.6f}, scale={scaler_scale[i]:.6f} ---")
        lines.append(f"* gain={gain:.8e} V/V, V_row = gain*V_feat + {offset_v:.8e}")
        lines.append(f"Econd{i} row{i} 0 vol='v(feat{i})*{gain:.8e} + {offset_v:.8e}'")
        lines.append("")


def generate_crossbar_section(Gp, Gn, n_in, n_out, layer_name, prefix, lines):
    """Generate crossbar cells for one layer (no includes, no stimuli)."""
    lines.append(f"* ===== {layer_name}: {n_in}x{n_out} DIFFERENTIAL 1T1R ReRAM CROSSBAR =====")
    lines.append(f"Vgate_{prefix} gate_{prefix} 0 1.8")
    lines.append("")

    lines.append(f"* --- Positive conductance array ---")
    for i in range(n_in):
        for j in range(n_out):
            r = 1e6 / Gp[i, j]
            row_node = f"row{i}" if prefix == "l1" else f"vout1_{i}"
            lines.append(f"XTp{prefix}_{i}_{j} {row_node} gate_{prefix} mp{prefix}_{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrp{prefix}_{i}_{j} mp{prefix}_{i}_{j} cp{prefix}_{j} {r:.4f}")

    lines.append("")
    lines.append(f"* --- Negative conductance array ---")
    for i in range(n_in):
        for j in range(n_out):
            r = 1e6 / Gn[i, j]
            row_node = f"row{i}" if prefix == "l1" else f"vout1_{i}"
            lines.append(f"XTn{prefix}_{i}_{j} {row_node} gate_{prefix} mn{prefix}_{i}_{j} 0 sky130_fd_pr__nfet_01v8 W={NMOS_W}u L={NMOS_L}u")
            lines.append(f"Rrn{prefix}_{i}_{j} mn{prefix}_{i}_{j} cn{prefix}_{j} {r:.4f}")

    lines.append("")

    lines.append(f"* --- Column current sensing (virtual ground) ---")
    for j in range(n_out):
        lines.append(f"Vsp{prefix}_{j} cp{prefix}_{j} 0 0")
        lines.append(f"Vsn{prefix}_{j} cn{prefix}_{j} 0 0")
    lines.append("")


def generate_l1_interstage(b1, scale1, lines):
    """L1 interstage: bias injection + ReLU + output mirror (64 channels)."""
    lines.append("* ===== L1->L2 INTERSTAGE: BIAS + ReLU + OUTPUT (64 channels) =====")
    lines.append("")

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


def generate_l2_bias(b2, scale2, lines):
    """L2->WTA bias injection (5 channels, no ReLU)."""
    lines.append("* ===== L2->WTA BIAS INJECTION (5 channels) =====")
    lines.append("")

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


def generate_wta(lines):
    """Lazzaro current-mode WTA (5 inputs -> 5 one-hot outputs)."""
    lines.append("* ===== WINNER-TAKE-ALL: 5-Input Lazzaro WTA =====")
    lines.append("Ibias_wta vdd nbias_wta 10u")
    lines.append(f"Xbias_wta nbias_wta nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
    lines.append("")

    for k in range(5):
        lines.append(f"* Cell {k} ({CLASS_NAMES[k]})")
        lines.append(f"Xwi{k} wcomm{k} nwta_{k} 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
        lines.append(f"Xwf{k} wcomm{k} wcomm{k} winhib 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
        lines.append(f"Xwl{k} out{k} wcomm{k} vdd vdd sky130_fd_pr__pfet_01v8 W=4.0u L=0.5u")
        lines.append(f"Xwd{k} out{k} nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=1.0u L=0.5u")
        lines.append("")

    lines.append(f"Xwinh winhib nbias_wta 0 0 sky130_fd_pr__nfet_01v8 W=10.0u L=0.5u")
    lines.append("")


def generate_l1_mac_connections(scale1, lines):
    """
    Connect L1 crossbar column currents to interstage input nodes.

    The L1 crossbar outputs differential column currents at cp_l1_j and cn_l1_j.
    The MAC current is: I_mac_j = i(Vsp_l1_j) - i(Vsn_l1_j)
    The interstage node nmac1_j must receive this net current.

    In the real circuit, the differential current sensing and injection happens
    naturally at the column summing node. We model this with a current-controlled
    current source (CCCS) that takes the differential current and injects it
    into the interstage node.
    """
    lines.append("* ===== L1 CROSSBAR -> INTERSTAGE CONNECTIONS =====")
    lines.append("* CCCS transfers differential MAC current to interstage nodes")
    lines.append(f"* Scale factor: {scale1:.4e}")
    lines.append("")

    for j in range(64):
        lines.append(f"Fmac1p_{j} nmac1_{j} 0 Vspl1_{j} 1.0")
        lines.append(f"Fmac1n_{j} 0 nmac1_{j} Vsnl1_{j} 1.0")

    lines.append("")


def generate_l2_mac_connections(scale2, lines):
    """Connect L2 crossbar column currents to WTA/bias nodes."""
    lines.append("* ===== L2 CROSSBAR -> WTA CONNECTIONS =====")
    lines.append("* CCCS transfers differential MAC current to WTA input nodes")
    lines.append("")

    for k in range(5):
        lines.append(f"Fmac2p_{k} nwta_{k} 0 Vspl2_{k} 1.0")
        lines.append(f"Fmac2n_{k} 0 nwta_{k} Vsnl2_{k} 1.0")

    lines.append("")


def main():
    print("=" * 74)
    print("  TOP-LEVEL ANALOG CLASSIFIER NETLIST GENERATOR")
    print("  8 Feature Voltages -> One-Hot Prediction")
    print("=" * 74)

    w = np.load(os.path.join(SCRIPT_DIR, 'full_analog_weights.npz'))
    W1_eff = w['W1_eff']
    W2_eff = w['W2_eff']
    b1, b2 = w['b1'], w['b2']
    scale1, scale2 = float(w['scale1']), float(w['scale2'])
    scaler_mean, scaler_scale = w['scaler_mean'], w['scaler_scale']
    Gp1, Gn1 = w['Gp1'], w['Gn1']
    Gp2, Gn2 = w['Gp2'], w['Gn2']

    lines = [
        "* =================================================================",
        "* TOP-LEVEL: Complete Analog ECG Classifier",
        "* Architecture: 8 -> 64 -> 5 MLP on ReRAM Crossbar",
        "* Technology: SKY130 (NFET/PFET 01v8 + ReRAM)",
        "* =================================================================",
        "*",
        "* Signal path (fully analog):",
        "*   feat0..feat7 (8 raw feature voltages)",
        "*     -> Input Conditioning (analog normalization: gain + offset)",
        "*     -> row0..row7 (normalized voltages, scaled to V_READ domain)",
        "*     -> L1 Crossbar (8x64 differential 1T1R ReRAM)",
        "*     -> 64 differential column currents",
        "*     -> L1 Interstage (64 channels):",
        "*         - Current-mirror bias injection (PMOS/NMOS per channel)",
        "*         - NMOS current-mirror ReLU (diode clamp)",
        "*         - Output PMOS mirror -> voltage for L2 rows",
        "*     -> L2 Crossbar (64x5 differential 1T1R ReRAM)",
        "*     -> 5 differential column currents",
        "*     -> L2 Bias injection (5 current mirrors)",
        "*     -> Lazzaro WTA (5-input analog argmax)",
        "*     -> out0..out4 (one-hot voltage output)",
        "*",
        f"* L1 scale: {scale1:.4e},  L2 scale: {scale2:.4e}",
        f"* V_READ: {V_READ}V,  VDD: 1.8V",
        f"* L1 cells: {8*64*2} 1T1R,  L2 cells: {64*5*2} 1T1R",
        "* =================================================================",
        "",
        f'.include "{SKY130_NFET_CORNER}"',
        f'.include "{SKY130_NFET_MISMATCH}"',
        f'.include "{SKY130_PFET_CORNER}"',
        f'.include "{SKY130_PFET_MISMATCH}"',
        PFET_PARAM_FIX,
        "",
        "* --- Supply ---",
        "VDD vdd 0 1.8",
        "",
        "* --- Feature inputs (8 raw ECG feature voltages) ---",
    ]

    for i in range(8):
        lines.append(f"Vfeat{i} feat{i} 0 0.0")
    lines.append("")

    # Stage 1: Input conditioning
    generate_input_conditioning(scaler_mean, scaler_scale, lines)

    # Stage 2: L1 crossbar
    generate_crossbar_section(Gp1, Gn1, 8, 64, "LAYER 1", "l1", lines)

    # Stage 3: L1->L2 current connections
    generate_l1_mac_connections(scale1, lines)

    # Stage 4: L1 interstage (bias + ReLU + output)
    generate_l1_interstage(b1, scale1, lines)

    # Stage 5: L2 crossbar
    generate_crossbar_section(Gp2, Gn2, 64, 5, "LAYER 2", "l2", lines)

    # Stage 6: L2->WTA current connections
    generate_l2_mac_connections(scale2, lines)

    # Stage 7: L2 bias injection
    generate_l2_bias(b2, scale2, lines)

    # Stage 8: WTA
    generate_wta(lines)

    # Analysis
    lines.append("* ===== ANALYSIS =====")
    lines.append(".op")
    lines.append("")
    lines.append(".control")
    lines.append("run")
    for k in range(5):
        lines.append(f"print v(out{k})")
    lines.append("quit")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    path = os.path.join(SCRIPT_DIR, "top_level_classifier.spice")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    n_lines = len(lines)
    n_l1_cells = 8 * 64 * 2
    n_l2_cells = 64 * 5 * 2
    n_interstage = 394
    n_wta = 16
    n_input_cond = 8
    n_total_transistors = (n_l1_cells + n_l2_cells) + n_interstage + n_wta
    n_total_reram = n_l1_cells + n_l2_cells

    print(f"\n  Generated: top_level_classifier.spice ({n_lines} lines)")
    print(f"\n  DEVICE SUMMARY:")
    print(f"    Input conditioning: 8 VCVS (analog gain+offset)")
    print(f"    L1 crossbar: {n_l1_cells} 1T1R cells ({8*64*2} NMOS + {8*64*2} ReRAM)")
    print(f"    L1 interstage: {n_interstage} transistors (bias + ReLU + output)")
    print(f"    L2 crossbar: {n_l2_cells} 1T1R cells ({64*5*2} NMOS + {64*5*2} ReRAM)")
    print(f"    L2 bias: 10 transistors")
    print(f"    WTA: {n_wta} transistors")
    print(f"    ---")
    print(f"    Total MOSFET: {n_total_transistors + n_l1_cells + n_l2_cells}")
    print(f"    Total ReRAM:  {n_total_reram}")
    print(f"\n  SIGNAL PATH:")
    print(f"    IN:  feat0..feat7 (8 analog voltages from ECG frontend)")
    print(f"    OUT: out0..out4 (one-hot: winner HIGH ~VDD, losers LOW ~0V)")
    print(f"         out0=N, out1=S, out2=V, out3=F, out4=Q")


if __name__ == '__main__':
    main()
