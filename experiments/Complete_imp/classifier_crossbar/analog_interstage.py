"""
Analog Interstage Circuits: Bias Injection + ReLU
Design, test, and integrate into the MLP classifier pipeline.

Implements two analog circuits using SKY130 transistors:
  1. Current-mode bias injection (NMOS/PMOS current mirror)
  2. ReLU via diode-connected NMOS current mirror (diode clamp)

Tests both in isolation with ngspice DC sweeps, extracts transfer functions,
then re-runs the full DS2 inter-patient verification with the analog circuits.
"""
import os, sys, json, time, subprocess, re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_FILE = os.path.join(PROJ_DIR, 'classifier', 'features_8_cache.npz')

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
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"

G_MIN_US = 0.296
G_MAX_US = 99.273
V_READ = 0.15
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

RELU_W = 4.0
RELU_L = 0.5
BIAS_W = 2.0
BIAS_L = 0.5


def run_ngspice(path, timeout=120):
    result = subprocess.run(
        [NGSPICE, "-b", path],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR
    )
    return result.stdout, result.stderr


def parse_wrdata(filepath):
    """Parse ngspice wrdata output file. Format: sweep_val  vector1  [vector2 ...]"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue
            parts = line.split()
            try:
                vals = [float(p) for p in parts]
                if len(vals) >= 2:
                    data.append(vals)
            except ValueError:
                continue
    return np.array(data) if data else None


def parse_op_value(stdout, name):
    name_lower = name.lower()
    for line in stdout.split('\n'):
        s = line.strip().lower()
        if name_lower in s and '=' in s:
            try:
                return float(s.split('=')[1].strip().split()[0])
            except:
                pass
    return None


# =========================================================================
# PHASE 1: Signal Range Analysis
# =========================================================================
def analyze_signal_ranges(X_te_n, W1_eff, b1, W2_eff, b2, scale1, scale2):
    print("\n[PHASE 1] Signal Range Analysis", flush=True)
    print("=" * 60, flush=True)

    MAC1 = X_te_n @ W1_eff
    Z1 = MAC1 + b1
    H1 = np.maximum(0, Z1)
    MAC2 = H1 @ W2_eff
    Z2 = MAC2 + b2

    I_z1 = Z1 / scale1
    I_h1 = H1 / scale1

    print(f"\n  LAYER 1 (weight domain):", flush=True)
    print(f"    MAC range:      [{MAC1.min():.2f}, {MAC1.max():.2f}]", flush=True)
    print(f"    Z1 (pre-ReLU):  [{Z1.min():.2f}, {Z1.max():.2f}]", flush=True)
    print(f"    H1 (post-ReLU): [{H1.min():.2f}, {H1.max():.2f}]", flush=True)
    print(f"    Z1 negative frac: {(Z1 < 0).mean()*100:.1f}%", flush=True)

    pcts = [1, 5, 25, 50, 75, 95, 99, 99.9]
    z1_pcts = np.percentile(Z1.ravel(), pcts)
    h1_nz = H1[H1 > 0].ravel()
    h1_pcts = np.percentile(h1_nz, pcts) if len(h1_nz) > 0 else np.zeros(len(pcts))
    print(f"\n    Z1 percentiles:", flush=True)
    for p, v in zip(pcts, z1_pcts):
        print(f"      {p:>5.1f}%: {v:>+10.4f}  ({v/scale1*1e6:>+8.2f} uA)", flush=True)
    print(f"\n    H1 (non-zero) percentiles:", flush=True)
    for p, v in zip(pcts, h1_pcts):
        print(f"      {p:>5.1f}%: {v:>10.4f}  ({v/scale1*1e6:>8.2f} uA)", flush=True)

    print(f"\n  LAYER 1 (current domain):", flush=True)
    print(f"    I_z1:  [{I_z1.min()*1e6:.1f}, {I_z1.max()*1e6:.1f}] uA", flush=True)
    print(f"    I_h1:  [0, {I_h1.max()*1e6:.1f}] uA", flush=True)

    I_practical_max = np.percentile(h1_nz, 99.9) / scale1 if len(h1_nz) > 0 else 100e-6
    print(f"    Practical I_max (99.9 pct): {I_practical_max*1e6:.1f} uA", flush=True)

    print(f"\n  LAYER 2:", flush=True)
    print(f"    Z2 range: [{Z2.min():.2f}, {Z2.max():.2f}]", flush=True)

    return {
        'I_practical_max': float(I_practical_max),
        'I_h1_max': float(I_h1.max()),
        'Z1': Z1, 'H1': H1,
    }


# =========================================================================
# PHASE 2: ReLU Cell Characterization (DC sweep, positive currents only)
# =========================================================================
def characterize_relu(I_practical_uA):
    print("\n[PHASE 2] ReLU Cell: NMOS Current Mirror (Diode Clamp)", flush=True)
    print("=" * 60, flush=True)
    print(f"  NMOS: sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u", flush=True)

    I_max_uA = min(I_practical_uA, 200)
    I_step_uA = 0.5
    n_pts = int(I_max_uA / I_step_uA) + 1
    print(f"  DC sweep: 0.1 to {I_max_uA:.1f} uA, step {I_step_uA} uA ({n_pts} points)", flush=True)

    wrdata_file = os.path.join(SCRIPT_DIR, "_relu_sweep.dat")
    spice_file = os.path.join(SCRIPT_DIR, "_relu_sweep.spice")

    # Single DC sweep — positive currents only (negative = 0 by physics)
    netlist = f"""\
* ReLU NMOS current mirror - DC sweep characterization
.include "{SKY130_NFET_CORNER}"
.include "{SKY130_NFET_MISMATCH}"

* X1: diode-connected (the 'diode clamp')
X1 nref nref 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u
* X2: mirror (copies rectified current)
X2 nout nref 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u
* Fixed drain voltage for M2
Vfix nout 0 1.0
* Input current (positive = into nref = normal NMOS conduction)
Iin 0 nref 0

.dc Iin {0.1e-6:.8e} {I_max_uA*1e-6:.8e} {I_step_uA*1e-6:.8e}

.control
run
wrdata {wrdata_file.replace(os.sep, '/')} i(Vfix)
quit
.endc
.end
"""

    with open(spice_file, 'w') as f:
        f.write(netlist)

    print("  Running ngspice DC sweep...", flush=True)
    t0 = time.time()
    try:
        stdout, stderr = run_ngspice(spice_file, timeout=120)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s", flush=True)
    except subprocess.TimeoutExpired:
        print("  ERROR: DC sweep timed out!", flush=True)
        return np.array([0]), np.array([0])

    # Parse wrdata output
    if not os.path.exists(wrdata_file):
        print(f"  ERROR: wrdata file not created!", flush=True)
        print(f"  ngspice stdout (last 20 lines):", flush=True)
        for line in stdout.strip().split('\n')[-20:]:
            print(f"    {line}", flush=True)
        if stderr.strip():
            print(f"  ngspice stderr (last 10 lines):", flush=True)
            for line in stderr.strip().split('\n')[-10:]:
                print(f"    {line}", flush=True)
        return np.array([0]), np.array([0])

    data = parse_wrdata(wrdata_file)
    if data is None or len(data) == 0:
        print(f"  ERROR: No data parsed from wrdata file!", flush=True)
        # Try printing file contents for debugging
        with open(wrdata_file, 'r') as f:
            content = f.read(2000)
        print(f"  File contents (first 2000 chars):", flush=True)
        print(content, flush=True)
        return np.array([0]), np.array([0])

    # wrdata format: col0 = sweep value (Iin), col1 = i(Vfix)
    I_in_sweep = data[:, 0]   # sweep parameter = Iin current
    I_vfix = data[:, 1]       # i(Vfix)

    # i(Vfix) convention: M2 sinks current from nout, so external current
    # into Vfix+ terminal is negative. I_mirror = -i(Vfix)
    I_out_sweep = -I_vfix

    # Add the I_in=0 -> I_out=0 anchor point
    I_in_arr = np.concatenate([[0], I_in_sweep])
    I_out_arr = np.concatenate([[0], I_out_sweep])

    print(f"  Parsed {len(data)} sweep points", flush=True)

    # Analyze
    valid = I_in_arr > 1e-9
    if valid.sum() > 2:
        gains = I_out_arr[valid] / I_in_arr[valid]
        print(f"\n  Mirror gain: mean={gains.mean():.6f}, median={np.median(gains):.6f}", flush=True)
        print(f"  Gain range: [{gains.min():.6f}, {gains.max():.6f}]", flush=True)
        err = np.abs(gains - 1.0) * 100
        print(f"  Mean gain error: {err.mean():.3f}%", flush=True)
        print(f"  Max gain error:  {err.max():.3f}%", flush=True)

    # Sample points table
    print(f"\n  {'I_in(uA)':>10} {'I_out(uA)':>10} {'Gain':>8}", flush=True)
    sample_idx = np.linspace(0, len(I_in_arr)-1, min(15, len(I_in_arr)), dtype=int)
    for i in sample_idx:
        g = I_out_arr[i] / I_in_arr[i] if I_in_arr[i] > 1e-12 else 0
        print(f"  {I_in_arr[i]*1e6:>10.3f} {I_out_arr[i]*1e6:>10.3f} {g:>8.4f}", flush=True)

    # Cleanup
    for f_path in [spice_file, wrdata_file]:
        try: os.remove(f_path)
        except: pass

    return I_in_arr, I_out_arr


# =========================================================================
# PHASE 3: Bias Mirror Characterization
# =========================================================================
def characterize_bias_mirrors(b1, scale1, b2, scale2):
    print("\n[PHASE 3] Bias Injection: Current Mirror Characterization", flush=True)
    print("=" * 60, flush=True)

    I_bias1 = b1 / scale1
    I_bias2 = b2 / scale2
    print(f"  L1 bias currents: [{I_bias1.min()*1e6:.3f}, {I_bias1.max()*1e6:.3f}] uA", flush=True)
    print(f"  L2 bias currents: [{I_bias2.min()*1e6:.3f}, {I_bias2.max()*1e6:.3f}] uA", flush=True)

    # Test a few representative values with DC sweep
    test_vals = [
        ('L1_max_pos', b1.max(), scale1, 'pmos'),
        ('L1_max_neg', b1.min(), scale1, 'nmos'),
        ('L1_mid_pos', float(np.median(b1[b1 > 0])), scale1, 'pmos'),
        ('L1_mid_neg', float(np.median(b1[b1 < 0])), scale1, 'nmos'),
        ('L2_N',       float(b2[0]), scale2, 'pmos' if b2[0] >= 0 else 'nmos'),
        ('L2_Q',       float(b2[4]), scale2, 'pmos' if b2[4] >= 0 else 'nmos'),
    ]

    mirror_errors = []
    print(f"\n  {'Name':<12} {'b':>8} {'I_tgt(uA)':>10} {'I_act(uA)':>10} {'Err%':>6}", flush=True)
    print(f"  {'-'*50}", flush=True)

    for name, b_val, scale, mtype in test_vals:
        I_target = abs(b_val) / scale
        I_actual = test_mirror(I_target, mtype)

        if I_actual is not None:
            err = abs(I_actual - I_target) / (I_target + 1e-12) * 100
            mirror_errors.append(err)
            print(f"  {name:<12} {b_val:>+8.4f} {I_target*1e6:>9.3f} {I_actual*1e6:>9.3f} {err:>5.1f}%", flush=True)
        else:
            print(f"  {name:<12} {b_val:>+8.4f} {I_target*1e6:>9.3f}   FAILED", flush=True)

    avg_err = np.mean(mirror_errors) if mirror_errors else 0
    print(f"\n  Average mirror error: {avg_err:.2f}%", flush=True)
    return avg_err


def test_mirror(I_target, mirror_type):
    if mirror_type == 'nmos':
        lines = [
            "* NMOS current mirror test",
            f'.include "{SKY130_NFET_CORNER}"',
            f'.include "{SKY130_NFET_MISMATCH}"',
            "",
            "VDD vdd 0 1.8",
            f"Iref vdd nref {I_target:.8e}",
            f"Xref nref nref 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u",
            f"Xmirr nout nref 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u",
            "Vmeas vdd nout 0",
            "",
            ".op",
            ".control", "run", "print i(Vmeas)", "quit", ".endc", ".end",
        ]
    else:
        lines = [
            "* PMOS current mirror test",
            f'.include "{SKY130_NFET_CORNER}"',
            f'.include "{SKY130_NFET_MISMATCH}"',
            f'.include "{SKY130_PFET_CORNER}"',
            f'.include "{SKY130_PFET_MISMATCH}"',
            PFET_PARAM_FIX,
            "",
            "VDD vdd 0 1.8",
            f"Iref pref 0 {I_target:.8e}",
            f"Xref pref pref vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u",
            f"Xmirr pout pref vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u",
            "Vmeas pout 0 0.9",
            "",
            ".op",
            ".control", "run", "print i(Vmeas)", "quit", ".endc", ".end",
        ]

    path = os.path.join(SCRIPT_DIR, "_bias_pt.spice")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    try:
        stdout, _ = run_ngspice(path, timeout=60)
        I_meas = parse_op_value(stdout, 'i(vmeas)')
    except subprocess.TimeoutExpired:
        I_meas = None

    try: os.remove(path)
    except: pass

    return abs(I_meas) if I_meas is not None else None


# =========================================================================
# PHASE 4: Generate Standalone SPICE Netlists
# =========================================================================
def generate_relu_netlist():
    lines = [
        "* =================================================================",
        "* ReLU Cell: NMOS Current-Mirror Diode Clamp",
        "* Technology: SKY130 sky130_fd_pr__nfet_01v8",
        "* =================================================================",
        "* Current-domain ReLU: I_out = max(0, I_in)",
        "* M1 (diode-connected) conducts for positive current; M2 mirrors it.",
        "* Negative current turns M1 off -> I_out = 0.",
        f"* M1, M2: W={RELU_W}u L={RELU_L}u",
        "* =================================================================",
        "",
        f'.include "{SKY130_NFET_CORNER}"',
        f'.include "{SKY130_NFET_MISMATCH}"',
        "",
        f"X1 nref nref 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u",
        f"X2 nout nref 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u",
        "Vfix nout 0 1.0",
        "Iin 0 nref 5u",
        "",
        ".op",
        ".control", "run", "print i(Vfix)", "quit", ".endc", ".end",
    ]
    path = os.path.join(SCRIPT_DIR, "relu_cell.spice")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, len(lines)


def generate_bias_netlist():
    lines = [
        "* =================================================================",
        "* Bias Injection Cell: Current Mirror",
        "* Technology: SKY130 NFET/PFET",
        "* =================================================================",
        "* Positive bias: PMOS mirror sources current from VDD.",
        "* Negative bias: NMOS mirror sinks current to GND.",
        f"* Transistors: W={BIAS_W}u L={BIAS_L}u",
        "* =================================================================",
        "",
        f'.include "{SKY130_NFET_CORNER}"',
        f'.include "{SKY130_NFET_MISMATCH}"',
        f'.include "{SKY130_PFET_CORNER}"',
        f'.include "{SKY130_PFET_MISMATCH}"',
        PFET_PARAM_FIX,
        "",
        "VDD vdd 0 1.8",
        "",
        "* === POSITIVE BIAS: PMOS current mirror ===",
        "Iref_p pref_p 0 5u",
        f"Xref_p pref_p pref_p vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u",
        f"Xmirr_p pout pref_p vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u",
        "Vmeas_p pout 0 0.9",
        "",
        "* === NEGATIVE BIAS: NMOS current mirror ===",
        "Iref_n vdd nref_n 3u",
        f"Xref_n nref_n nref_n 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u",
        f"Xmirr_n nout nref_n 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u",
        "Vmeas_n vdd nout 0",
        "",
        ".op",
        ".control", "run", "print i(Vmeas_p) i(Vmeas_n)", "quit", ".endc", ".end",
    ]
    path = os.path.join(SCRIPT_DIR, "bias_inject_cell.spice")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, len(lines)


def generate_interstage_netlist(b1, scale1, b2, scale2):
    lines = [
        "* =================================================================",
        "* Complete Interstage: Bias Injection + ReLU (64 + 5 channels)",
        "* Architecture: 8 -> 64 -> 5 MLP Classifier",
        "* Technology: SKY130",
        "* =================================================================",
        f"* ReLU NMOS: W={RELU_W}u L={RELU_L}u",
        f"* Bias mirrors: W={BIAS_W}u L={BIAS_L}u",
        "* =================================================================",
        "",
        f'.include "{SKY130_NFET_CORNER}"',
        f'.include "{SKY130_NFET_MISMATCH}"',
        f'.include "{SKY130_PFET_CORNER}"',
        f'.include "{SKY130_PFET_MISMATCH}"',
        PFET_PARAM_FIX,
        "",
        "VDD vdd 0 1.8",
        "",
        "* ===== L1->L2 INTERSTAGE (64 channels) =====",
    ]

    n_pmos_l1, n_nmos_l1 = 0, 0

    for j in range(64):
        I_bias = b1[j] / scale1
        lines.append(f"* --- Ch {j}: b1={b1[j]:+.4f}, I_bias={I_bias*1e6:+.3f}uA ---")

        if b1[j] >= 0:
            I_ref = abs(I_bias)
            lines.append(f"Ibr1_{j} pbr1_{j} 0 {I_ref:.6e}")
            lines.append(f"Xbr1_{j} pbr1_{j} pbr1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm1_{j} nmac1_{j} pbr1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            n_pmos_l1 += 2
        else:
            I_ref = abs(I_bias)
            lines.append(f"Ibr1_{j} vdd nbr1_{j} {I_ref:.6e}")
            lines.append(f"Xbr1_{j} nbr1_{j} nbr1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm1_{j} nmac1_{j} nbr1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            n_nmos_l1 += 2

        lines.append(f"Xr1a_{j} nmac1_{j} nmac1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u")
        lines.append(f"Xr1b_{j} nrelu1_{j} nmac1_{j} 0 0 sky130_fd_pr__nfet_01v8 W={RELU_W}u L={RELU_L}u")
        lines.append(f"Xopr_{j} nrelu1_{j} nrelu1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        lines.append(f"Xopm_{j} vout1_{j} nrelu1_{j} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
        lines.append("")

    lines.append("")
    lines.append("* ===== L2->WTA INTERSTAGE (5 channels) =====")

    n_pmos_l2, n_nmos_l2 = 0, 0

    for k in range(5):
        I_bias = b2[k] / scale2
        lines.append(f"* --- Class {CLASS_NAMES[k]}: b2={b2[k]:+.4f}, I_bias={I_bias*1e6:+.3f}uA ---")

        if b2[k] >= 0:
            I_ref = abs(I_bias)
            lines.append(f"Ibr2_{k} pbr2_{k} 0 {I_ref:.6e}")
            lines.append(f"Xbr2_{k} pbr2_{k} pbr2_{k} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm2_{k} nwta_{k} pbr2_{k} vdd vdd sky130_fd_pr__pfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            n_pmos_l2 += 2
        else:
            I_ref = abs(I_bias)
            lines.append(f"Ibr2_{k} vdd nbr2_{k} {I_ref:.6e}")
            lines.append(f"Xbr2_{k} nbr2_{k} nbr2_{k} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            lines.append(f"Xbm2_{k} nwta_{k} nbr2_{k} 0 0 sky130_fd_pr__nfet_01v8 W={BIAS_W}u L={BIAS_L}u")
            n_nmos_l2 += 2
        lines.append("")

    total_relu_nmos = 128
    total_output_pmos = 128
    total = n_pmos_l1 + n_nmos_l1 + total_relu_nmos + total_output_pmos + n_pmos_l2 + n_nmos_l2
    lines.append(f"* TRANSISTOR COUNT: {total}")
    lines.append(f"*   L1 bias PMOS: {n_pmos_l1}, L1 bias NMOS: {n_nmos_l1}")
    lines.append(f"*   L1 ReLU NMOS: {total_relu_nmos}")
    lines.append(f"*   L1 output PMOS: {total_output_pmos}")
    lines.append(f"*   L2 bias PMOS: {n_pmos_l2}, L2 bias NMOS: {n_nmos_l2}")
    lines.append("")
    lines.append(".end")

    path = os.path.join(SCRIPT_DIR, "interstage_bias_relu.spice")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path, len(lines), total


# =========================================================================
# PHASE 5 & 6: Build Transfer Function and Verify
# =========================================================================
def build_relu_transfer_fn(I_in_arr, I_out_arr):
    sort_idx = np.argsort(I_in_arr)
    I_in_s = I_in_arr[sort_idx]
    I_out_s = I_out_arr[sort_idx]

    pos = I_in_s > 1e-9
    avg_gain = float(np.median(I_out_s[pos] / I_in_s[pos])) if pos.sum() > 2 else 1.0

    def analog_relu(I_in_values):
        result = np.zeros_like(I_in_values, dtype=float)
        positive = I_in_values > 0
        if positive.any():
            result[positive] = np.interp(
                I_in_values[positive], I_in_s[I_in_s >= 0], I_out_s[I_in_s >= 0],
                left=0.0, right=None
            )
            beyond = I_in_values > I_in_s[-1]
            if beyond.any():
                result[beyond] = I_in_values[beyond] * avg_gain
        return result

    return analog_relu, avg_gain


def run_verification(X_te_n, y_te, W1_eff, b1, W2_eff, b2,
                     scale1, scale2, analog_relu_fn, mirror_gain):
    print("\n[PHASE 6] Full DS2 Verification with Analog Circuits", flush=True)
    print("=" * 60, flush=True)

    # IDEAL REFERENCE
    Z1_sw = X_te_n @ W1_eff + b1
    H1_sw = np.maximum(0, Z1_sw)
    Z2_sw = H1_sw @ W2_eff + b2
    sw_pred = np.argmax(Z2_sw, axis=1)
    sw_acc = accuracy_score(y_te, sw_pred) * 100

    # ANALOG PIPELINE
    MAC1 = X_te_n @ W1_eff
    b1_analog = b1 * mirror_gain
    Z1_analog = MAC1 + b1_analog

    I_z1 = Z1_analog / scale1
    I_h1 = analog_relu_fn(I_z1.ravel()).reshape(I_z1.shape)
    H1_analog = I_h1 * scale1

    MAC2 = H1_analog @ W2_eff
    b2_analog = b2 * mirror_gain
    Z2_analog = MAC2 + b2_analog

    hw_pred = np.argmax(Z2_analog, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100
    hw_f1 = f1_score(y_te, hw_pred, average=None, zero_division=0)

    agree = (hw_pred == sw_pred).mean() * 100
    disagree = int((hw_pred != sw_pred).sum())
    cm = confusion_matrix(y_te, hw_pred, labels=range(5))

    relu_diff = np.abs(H1_analog - H1_sw)

    print(f"\n  {'Pipeline':<52} {'Accuracy':>10}", flush=True)
    print(f"  {'-'*64}", flush=True)
    print(f"  {'Crossbar + IDEAL bias + IDEAL ReLU (previous)':<52} {sw_acc:>9.2f}%", flush=True)
    print(f"  {'Crossbar + ANALOG bias + ANALOG ReLU':<52} {hw_acc:>9.2f}%", flush=True)
    print(f"  {'-'*64}", flush=True)
    print(f"  Agreement: {agree:.2f}% ({disagree} disagreements / {len(y_te)})", flush=True)

    print(f"\n  Hidden activation error (analog vs ideal):", flush=True)
    print(f"    Mean |diff|: {relu_diff.mean():.6f}", flush=True)
    print(f"    Max  |diff|: {relu_diff.max():.6f}", flush=True)

    print(f"\n  Confusion matrix:", flush=True)
    print(f"  {'':>8}" + "".join(f"{c:>7}" for c in CLASS_NAMES), flush=True)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>8}" + "".join(f"{v:>7d}" for v in row), flush=True)

    print(f"\n  Per-class F1:", flush=True)
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c}: {hw_f1[i]:.4f}", flush=True)

    print(f"\n{classification_report(y_te, hw_pred, target_names=CLASS_NAMES, zero_division=0)}", flush=True)

    return {
        'sw_accuracy': round(sw_acc, 3),
        'analog_accuracy': round(hw_acc, 3),
        'agreement': round(agree, 3),
        'disagreements': disagree,
        'hw_f1': {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, hw_f1)},
        'confusion_matrix': cm.tolist(),
        'relu_mean_err': float(relu_diff.mean()),
        'relu_max_err': float(relu_diff.max()),
    }


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 74, flush=True)
    print("  ANALOG INTERSTAGE: Bias Injection + ReLU", flush=True)
    print("  SKY130 current-mirror design, test, and verification", flush=True)
    print("=" * 74, flush=True)

    print("\n  Loading weights and features...", flush=True)
    w = np.load(os.path.join(SCRIPT_DIR, 'full_analog_weights.npz'))
    W1_eff, W2_eff = w['W1_eff'], w['W2_eff']
    b1, b2 = w['b1'], w['b2']
    scale1, scale2 = float(w['scale1']), float(w['scale2'])
    scaler_mean, scaler_scale = w['scaler_mean'], w['scaler_scale']

    d = np.load(CACHE_FILE)
    X_te, y_te = d['X_te'], d['y_te']
    X_te_n = (X_te - scaler_mean) / scaler_scale
    print(f"  Test: {len(y_te)} beats, Scale1={scale1:.0f}, Scale2={scale2:.0f}", flush=True)

    # PHASE 1
    ranges = analyze_signal_ranges(X_te_n, W1_eff, b1, W2_eff, b2, scale1, scale2)

    # PHASE 2: ReLU (single DC sweep)
    I_practical_uA = ranges['I_practical_max'] * 1e6
    I_in_arr, I_out_arr = characterize_relu(I_practical_uA)

    # PHASE 3: Bias mirrors (~6 individual .op sims)
    mirror_err = characterize_bias_mirrors(b1, scale1, b2, scale2)
    mirror_gain = 1.0 - mirror_err / 200

    # PHASE 4: Netlists
    print("\n[PHASE 4] Generating SPICE Netlists", flush=True)
    print("=" * 60, flush=True)
    p1, n1 = generate_relu_netlist()
    print(f"  relu_cell.spice ({n1} lines)", flush=True)
    p2, n2 = generate_bias_netlist()
    print(f"  bias_inject_cell.spice ({n2} lines)", flush=True)
    p3, n3, n_trans = generate_interstage_netlist(b1, scale1, b2, scale2)
    print(f"  interstage_bias_relu.spice ({n3} lines, {n_trans} transistors)", flush=True)

    # PHASE 5: Transfer function
    print("\n[PHASE 5] Building Analog Transfer Functions", flush=True)
    print("=" * 60, flush=True)

    if len(I_in_arr) > 1:
        analog_relu_fn, relu_gain = build_relu_transfer_fn(I_in_arr, I_out_arr)
        print(f"  ReLU mirror gain: {relu_gain:.6f}", flush=True)
        print(f"  Bias mirror gain: {mirror_gain:.6f}", flush=True)

        print(f"\n  {'I_in(uA)':>10} {'Ideal':>10} {'Analog':>10} {'Err':>8}", flush=True)
        for I_uA in [0, 0.1, 0.5, 1, 5, 10, 20, 50, 89]:
            I_A = I_uA * 1e-6
            ideal = max(0, I_A)
            analog = analog_relu_fn(np.array([I_A]))[0]
            if ideal > 1e-10:
                err_s = f"{abs(analog-ideal)/ideal*100:.2f}%"
            else:
                err_s = f"{analog*1e9:.1f}nA"
            print(f"  {I_uA:>10.1f} {ideal*1e6:>10.4f} {analog*1e6:>10.4f} {err_s:>8}", flush=True)
    else:
        print("  Using ideal transfer functions (ngspice data unavailable)", flush=True)
        analog_relu_fn = lambda x: np.maximum(0, x)
        relu_gain = 1.0
        mirror_gain = 1.0

    # PHASE 6: Verification
    results = run_verification(X_te_n, y_te, W1_eff, b1, W2_eff, b2,
                               scale1, scale2, analog_relu_fn, mirror_gain)

    results['relu_gain'] = relu_gain
    results['mirror_gain'] = mirror_gain
    results['mirror_error_pct'] = mirror_err
    results['relu_nmos_per_ch'] = 2
    results['relu_nmos_sizing'] = f"W={RELU_W}u L={RELU_L}u"
    results['bias_sizing'] = f"W={BIAS_W}u L={BIAS_L}u"

    with open(os.path.join(SCRIPT_DIR, 'analog_interstage_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*74}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*74}", flush=True)
    print(f"  Previous (ideal bias+ReLU):    {results['sw_accuracy']:.2f}%", flush=True)
    print(f"  Analog (mirror bias+ReLU):     {results['analog_accuracy']:.2f}%", flush=True)
    print(f"  Agreement:                     {results['agreement']:.2f}%", flush=True)
    acc_change = results['analog_accuracy'] - results['sw_accuracy']
    if abs(acc_change) < 0.01:
        print(f"  Accuracy change:               NONE", flush=True)
    else:
        print(f"  Accuracy change:               {acc_change:+.2f}%", flush=True)
    print(f"\n  Files produced:", flush=True)
    print(f"    relu_cell.spice", flush=True)
    print(f"    bias_inject_cell.spice", flush=True)
    print(f"    interstage_bias_relu.spice   ({n_trans} transistors)", flush=True)
    print(f"    analog_interstage_results.json", flush=True)


if __name__ == '__main__':
    main()
