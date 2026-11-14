"""
End-to-end verification: circuit-feature weights on the full analog crossbar pipeline.

Tests a representative set of beats through the complete two-crossbar analog pipeline
(using effective crossbar weights from ngspice characterization) and compares against
the software MLP prediction.

Also verifies the top_level_classifier_circuit.spice by running a few beats through
actual ngspice .op simulation.
"""
import os, sys, json, subprocess
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

w = sys.stderr.write

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLETE_IMP_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_FILE = os.path.join(COMPLETE_IMP_DIR, 'validation', 'circuit_features_cache.npz')
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'circuit_analog_weights.npz')
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
V_READ = 0.15
X_OFFSET = 4.0


def main():
    w("=" * 74 + "\n")
    w("  END-TO-END VERIFICATION: Circuit-Feature Analog Classifier\n")
    w("=" * 74 + "\n")

    # Load data
    d = np.load(CACHE_FILE, allow_pickle=True)
    X_te, y_te = d['X_te'], d['y_te']

    wt = np.load(WEIGHTS_FILE)
    W1 = wt['W1']; b1 = wt['b1']
    W2 = wt['W2']; b2 = wt['b2']
    W1_eff = wt['W1_eff']; W2_eff = wt['W2_eff']
    scale1 = float(wt['scale1']); scale2 = float(wt['scale2'])
    scaler_mean = wt['scaler_mean']
    scaler_scale = wt['scaler_scale']

    X_te_n = (X_te - scaler_mean) / scaler_scale

    w(f"\n  Test beats: {len(y_te)}\n")
    w(f"  Scaler mean: {scaler_mean}\n")
    w(f"  Scaler scale: {scaler_scale}\n")

    # Method A: Software MLP (ideal weights)
    Z1_sw = X_te_n @ W1 + b1
    H1_sw = np.maximum(0, Z1_sw)
    Z2_sw = H1_sw @ W2 + b2
    sw_pred = np.argmax(Z2_sw, axis=1)
    sw_acc = accuracy_score(y_te, sw_pred) * 100

    # Method B: Full analog crossbar (effective weights from ngspice)
    Z1_hw = X_te_n @ W1_eff + b1
    H1_hw = np.maximum(0, Z1_hw)
    Z2_hw = H1_hw @ W2_eff + b2
    hw_pred = np.argmax(Z2_hw, axis=1)
    hw_acc = accuracy_score(y_te, hw_pred) * 100

    agree = (sw_pred == hw_pred).mean() * 100

    w(f"\n  SW accuracy:        {sw_acc:.2f}%\n")
    w(f"  Full analog acc:    {hw_acc:.2f}%\n")
    w(f"  SW/analog agree:    {agree:.2f}%\n")

    cm = confusion_matrix(y_te, hw_pred, labels=range(5))
    w(f"\n  Confusion matrix (full analog):\n")
    w(f"  {'':>8}" + "".join(f"{c:>7}" for c in CLASS_NAMES) + "\n")
    for i, row in enumerate(cm):
        w(f"  {CLASS_NAMES[i]:>8}" + "".join(f"{v:>7d}" for v in row) + "\n")

    w(f"\n  Per-class report (full analog):\n")
    w(classification_report(y_te, hw_pred, target_names=CLASS_NAMES, zero_division=0) + "\n")

    # Per-class recall summary
    for i, name in enumerate(CLASS_NAMES):
        mask = y_te == i
        if mask.any():
            rec = (hw_pred[mask] == i).mean() * 100
            w(f"  {name} recall: {rec:.1f}% ({(hw_pred[mask]==i).sum()}/{mask.sum()})\n")

    # Spot-check: run 5 test beats through top_level_classifier_circuit.spice via ngspice
    w(f"\n  Spot-check: 5 beats through ngspice top-level netlist...\n")
    top_spice = os.path.join(SCRIPT_DIR, 'top_level_classifier_circuit.spice')
    if not os.path.exists(top_spice):
        w("  SKIP: top_level_classifier_circuit.spice not found\n")
    else:
        # Pick 5 representative beats (one per class if possible)
        test_indices = []
        for cls in range(5):
            idxs = np.where(y_te == cls)[0]
            if len(idxs) > 0:
                test_indices.append(idxs[len(idxs)//2])
        if len(test_indices) < 5:
            extra = np.random.choice(len(y_te), 5 - len(test_indices), replace=False)
            test_indices.extend(extra.tolist())

        ngspice_correct = 0
        for beat_idx in test_indices:
            raw_feat = X_te[beat_idx]
            true_cls = y_te[beat_idx]

            # Read the netlist, modify feat voltages, run .op
            with open(top_spice, 'r') as f:
                netlist = f.read()

            for i in range(8):
                old = f"Vfeat{i} feat{i} 0 0.0"
                new = f"Vfeat{i} feat{i} 0 {raw_feat[i]:.8f}"
                netlist = netlist.replace(old, new)

            tb_path = os.path.join(SCRIPT_DIR, '_verify_beat.spice')
            with open(tb_path, 'w') as f:
                f.write(netlist)

            result = subprocess.run(
                [NGSPICE, "-b", tb_path],
                capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR
            )

            # Parse WTA outputs
            vout = np.zeros(5)
            for line in result.stdout.split('\n'):
                s = line.strip().lower()
                for k in range(5):
                    if f"v(out{k})" in s and '=' in s:
                        try: vout[k] = float(s.split('=')[1].strip().split()[0])
                        except: pass

            ngspice_pred = np.argmax(vout)
            sw_beat_pred = sw_pred[beat_idx]
            match = "OK" if ngspice_pred == sw_beat_pred else "MISMATCH"
            if ngspice_pred == true_cls:
                ngspice_correct += 1

            w(f"    Beat {beat_idx}: true={CLASS_NAMES[true_cls]}, "
              f"ngspice={CLASS_NAMES[ngspice_pred]} (vout={vout.round(3)}), "
              f"sw={CLASS_NAMES[sw_beat_pred]} [{match}]\n")

            os.remove(tb_path)

        w(f"  Spot-check: {ngspice_correct}/5 correct vs true label\n")

    # Save verification results
    results = {
        'sw_accuracy_pct': round(sw_acc, 3),
        'full_analog_accuracy_pct': round(hw_acc, 3),
        'sw_analog_agreement_pct': round(agree, 3),
        'test_beats': len(y_te),
        'per_class_recall': {},
        'confusion_matrix': cm.tolist(),
    }
    for i, name in enumerate(CLASS_NAMES):
        mask = y_te == i
        if mask.any():
            results['per_class_recall'][name] = round((hw_pred[mask] == i).mean() * 100, 1)

    out_path = os.path.join(SCRIPT_DIR, 'circuit_verification_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    w(f"\n  Saved: {out_path}\n")
    w("  Done.\n")


if __name__ == '__main__':
    main()
