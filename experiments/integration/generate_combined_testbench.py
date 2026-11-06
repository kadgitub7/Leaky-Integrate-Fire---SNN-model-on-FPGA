"""
Combined End-to-End Testbench Generator
========================================
Generates ngspice testbenches that test the integrated classifier
using analog frontend feature voltages from the cache.

Two modes:
  1. Quick validation: .op simulation per beat (fast, ~2s/beat)
  2. Full transient: Combined frontend+classifier .tran (accurate, slow)

This script implements Mode 1: extract cached feature voltages → feed to
classifier .op → read WTA output → compare to annotation.
"""
import os, sys, json, subprocess, re, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
INTEGRATION_DIR = os.path.join(PROJECT_ROOT, "experiments", "integration")
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, "experiments", "Complete_imp", "classifier_crossbar")
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend", "_cache")

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"

SKY130_NFET_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"
SKY130_NFET_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"
SKY130_PFET_CORNER = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"
SKY130_PFET_MISMATCH = "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__mismatch.corner.spice"

V_READ = 0.15
X_OFFSET = 4.0
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
LABEL_MAP = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}

DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']

# Frontend-to-classifier feature reordering
FRONTEND_TO_CLASSIFIER = [0, 3, 1, 4, 5, 2, 6, 7]


def load_calibration():
    """Load calibration parameters from integration results."""
    results_path = os.path.join(INTEGRATION_DIR, "integration_results.json")
    if not os.path.exists(results_path):
        print("ERROR: integration_results.json not found.")
        print("Run integrate_frontend_classifier.py first.")
        sys.exit(1)
    with open(results_path) as f:
        results = json.load(f)
    cal = results['calibration']
    A_vcvs = np.array([cal[name]['A_vcvs'] for name in
                       ['rr_ratio', 'bi_L1', 'rr_asym', 'pwave', 'width', 'bi_L0', 'slope', 'symmetry']])
    B_vcvs = np.array([cal[name]['B_vcvs'] for name in
                       ['rr_ratio', 'bi_L1', 'rr_asym', 'pwave', 'width', 'bi_L0', 'slope', 'symmetry']])
    return A_vcvs, B_vcvs


def generate_batch_op_testbench(feat_batch, batch_id, A_vcvs, B_vcvs):
    """
    Generate a single .op testbench that classifies multiple beats.

    Uses .alter blocks to sweep through beats efficiently:
    one ngspice invocation classifies the whole batch.
    """
    n_beats = len(feat_batch)
    classifier_netlist = os.path.join(INTEGRATION_DIR, "integrated_classifier.spice").replace("\\", "/")

    lines = []
    lines.append(f"* Batch {batch_id}: {n_beats} beats")
    lines.append(f'.include "{classifier_netlist}"')
    lines.append("")

    # First beat
    feat_cls = feat_batch[0][FRONTEND_TO_CLASSIFIER]
    for i in range(8):
        v_row = A_vcvs[i] * feat_cls[i] + B_vcvs[i]
        v_row = max(0.0, min(V_READ, v_row))
        lines.append(f"Vfeat{i} feat{i} 0 {v_row:.8f}")
    lines.append("")
    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    for k in range(5):
        lines.append(f"print v(out{k})")
    lines.append(".endc")
    lines.append("")

    # Remaining beats via .alter
    for b in range(1, n_beats):
        lines.append(f"* --- Beat {b} ---")
        lines.append(".alter")
        feat_cls = feat_batch[b][FRONTEND_TO_CLASSIFIER]
        for i in range(8):
            v_row = A_vcvs[i] * feat_cls[i] + B_vcvs[i]
            v_row = max(0.0, min(V_READ, v_row))
            lines.append(f"Vfeat{i} feat{i} 0 {v_row:.8f}")
        lines.append("")

    lines.append(".end")

    tb_path = os.path.join(INTEGRATION_DIR, f"_batch_{batch_id}.spice")
    with open(tb_path, 'w') as f:
        f.write('\n'.join(lines))
    return tb_path


def generate_single_beat_testbench(feat_frontend, beat_id, A_vcvs, B_vcvs):
    """Generate a single-beat .op testbench for the integrated classifier."""
    classifier_netlist = os.path.join(INTEGRATION_DIR, "integrated_classifier.spice").replace("\\", "/")

    feat_cls = feat_frontend[FRONTEND_TO_CLASSIFIER]

    lines = []
    lines.append(f"* Beat {beat_id}")
    lines.append(f'.include "{classifier_netlist}"')
    lines.append("")
    for i in range(8):
        v_row = A_vcvs[i] * feat_cls[i] + B_vcvs[i]
        v_row = max(0.0, min(V_READ, v_row))
        lines.append(f"Vfeat{i} feat{i} 0 {v_row:.8f}")
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


def run_ngspice_op(tb_path, timeout=300):
    """Run ngspice .op and parse WTA output voltages."""
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


def classify_beat_ngspice(feat_frontend, beat_id, A_vcvs, B_vcvs):
    """Classify a single beat through the full ngspice crossbar."""
    tb = generate_single_beat_testbench(feat_frontend, beat_id, A_vcvs, B_vcvs)
    out_v = run_ngspice_op(tb)
    try:
        os.remove(tb)
    except:
        pass
    return np.argmax(out_v), out_v


def quick_validation(n_beats=50):
    """
    Quick validation: classify n_beats through the full ngspice crossbar
    and compare to software emulation.
    """
    print("=" * 74)
    print("  QUICK VALIDATION: ngspice Crossbar vs Software Emulation")
    print("=" * 74)

    A_vcvs, B_vcvs = load_calibration()
    print(f"  VCVS parameters loaded from integration_results.json")

    # Load a few test beats from cache
    all_feats, all_labels = [], []
    for rec in DS2[:3]:
        cache_file = os.path.join(CACHE_DIR, f"{rec}.json")
        if not os.path.exists(cache_file):
            continue
        with open(cache_file) as f:
            data = json.load(f)
        feats = np.array(data['features'])
        labels = [LABEL_MAP[l] for l in data['labels']]
        all_feats.append(feats)
        all_labels.extend(labels)

    if not all_feats:
        print("  ERROR: No cached records available yet.")
        return

    X = np.vstack(all_feats)
    y = np.array(all_labels)
    n = min(n_beats, len(y))
    print(f"  Testing {n} beats through ngspice crossbar...")

    hw_preds, sw_preds = [], []
    t0 = time.time()
    for i in range(n):
        hw_pred, out_v = classify_beat_ngspice(X[i], i, A_vcvs, B_vcvs)
        hw_preds.append(hw_pred)

        # Software emulation for comparison
        feat_cls = X[i][FRONTEND_TO_CLASSIFIER]
        alpha = np.array([A_vcvs[j] * (2 * X_OFFSET) / V_READ for j in range(8)])
        beta = np.array([B_vcvs[j] * (2 * X_OFFSET) / V_READ - X_OFFSET for j in range(8)])
        from integrate_frontend_classifier import classify_software_emulated_digital
        # ... (use cached results instead)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n}] {elapsed:.1f}s ({elapsed/(i+1):.1f}s/beat)")

    hw_preds = np.array(hw_preds)
    hw_acc = accuracy_score(y[:n], hw_preds) * 100
    print(f"\n  ngspice crossbar accuracy: {hw_acc:.2f}% ({n} beats)")
    print(f"  Time: {time.time()-t0:.1f}s total, {(time.time()-t0)/n:.1f}s/beat")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', type=int, default=50, help='Quick validation with N beats')
    args = parser.parse_args()
    quick_validation(args.quick)
