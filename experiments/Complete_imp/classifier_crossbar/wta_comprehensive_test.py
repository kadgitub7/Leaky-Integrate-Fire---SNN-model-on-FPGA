"""
Comprehensive WTA validation using real classifier output scores.

1. Loads the MLP weights and DS2 test data
2. Computes Z2 (L2 output scores) for all test beats using effective weights
3. Maps Z2 scores to WTA input voltage range (offset-binary encoding)
4. Generates ngspice batch testbench with multiple beats per run
5. Verifies argmin(WTA_output) == argmax(Z2) for every tested beat
6. Reports match rate, failure cases, and margin statistics
"""
import os, sys, json, subprocess, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLETE_IMP_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_FILE = os.path.join(COMPLETE_IMP_DIR, 'validation', 'circuit_features_cache.npz')
WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'circuit_analog_weights.npz')
NGSPICE = "C:/Spice64/bin/ngspice_con.exe"
WTA_SUBCKT = os.path.join(SCRIPT_DIR, 'wta_5class_fixed.spice')

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
V_WTA_MIN = 0.55
V_WTA_MAX = 1.05
SETTLE_US = 8
MEAS_OFFSET_US = 6


def compute_z2_scores():
    d = np.load(CACHE_FILE, allow_pickle=True)
    X_te, y_te = d['X_te'], d['y_te']

    wt = np.load(WEIGHTS_FILE)
    W1_eff = wt['W1_eff']; b1 = wt['b1']
    W2_eff = wt['W2_eff']; b2 = wt['b2']
    scaler_mean = wt['scaler_mean']
    scaler_scale = wt['scaler_scale']

    X_te_n = (X_te - scaler_mean) / scaler_scale
    Z1 = X_te_n @ W1_eff + b1
    H1 = np.maximum(0, Z1)
    Z2 = H1 @ W2_eff + b2

    return Z2, y_te


def z2_to_voltage(Z2):
    z_min = Z2.min()
    z_max = Z2.max()
    V = V_WTA_MIN + (Z2 - z_min) / (z_max - z_min) * (V_WTA_MAX - V_WTA_MIN)
    return V, z_min, z_max


def select_test_beats(Z2, y_te, n_total=100):
    n = len(y_te)
    indices = []

    margins_all = np.zeros(n)
    for i in range(n):
        s = np.sort(Z2[i])
        margins_all[i] = s[-1] - s[-2]

    tightest_50 = np.argsort(margins_all)[:50]
    indices.extend(tightest_50.tolist())

    for cls in range(5):
        cls_idx = np.where(y_te == cls)[0]
        if len(cls_idx) == 0:
            continue
        scores = Z2[cls_idx]
        sw_pred = np.argmax(scores, axis=1)
        correct = cls_idx[sw_pred == cls]
        wrong = cls_idx[sw_pred != cls]

        if len(correct) > 0:
            cls_margins = margins_all[correct]
            easy = correct[np.argsort(-cls_margins)[:5]]
            hard = correct[np.argsort(cls_margins)[:10]]
            indices.extend(easy.tolist())
            indices.extend(hard.tolist())

        if len(wrong) > 0:
            indices.extend(wrong[:5].tolist())

    indices = list(set(indices))

    remaining = n_total - len(indices)
    if remaining > 0:
        all_idx = set(range(n))
        pool = list(all_idx - set(indices))
        np.random.seed(42)
        extra = np.random.choice(pool, min(remaining, len(pool)), replace=False)
        indices.extend(extra.tolist())

    return sorted(indices[:n_total])


def generate_batch_testbench(V_beats, beat_indices, batch_id):
    n = len(beat_indices)
    total_time = n * SETTLE_US

    lines = []
    lines.append(f"* WTA Batch Test — batch {batch_id}, {n} beats")
    lines.append("")
    lines.append('.include "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice"')
    lines.append('.include "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice"')
    lines.append('.include "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice"')
    lines.append('.include "C:/src/skywater-pdk-libs-sky130_fd_pr/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__mismatch.corner.spice"')
    lines.append("")

    pfet_params = [
        "a0", "agidl", "ags", "b0", "b1", "bgidl", "cgidl", "dlc", "dwc",
        "eta0", "k2", "keta", "kt1", "ku0", "kvsat", "kvth0", "lint", "lku0",
        "lkvth0", "nfactor", "pclm", "pdits", "pditsd", "rdsw", "tvoff",
        "u0", "ua", "ub", "voff", "vsat", "vth0", "wint", "wku0", "wkvth0", "wlod"
    ]
    for p in pfet_params:
        lines.append(f".param sky130_fd_pr__pfet_01v8__{p}_diff = 0")
    lines.append("")

    lines.append("VDD vdd 0 1.8")
    lines.append("Ibias_n vdd nbias 10u")
    lines.append("Xbd nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W=2.0u L=0.5u")
    lines.append("Ibias_p pbias 0 10u")
    lines.append("Xpbd pbias pbias vdd vdd sky130_fd_pr__pfet_01v8 W=4.0u L=0.5u")
    lines.append("")
    lines.append('.include "wta_5class_fixed.spice"')
    lines.append("XWTA in0 in1 in2 in3 in4 out0 out1 out2 out3 out4 vdd 0 nbias pbias wta_5class")
    lines.append("")

    for ch in range(5):
        pwl_parts = ["0 0.0"]
        for i, bi in enumerate(beat_indices):
            t_start = i * SETTLE_US
            t_ramp = t_start + 0.5
            v = V_beats[bi, ch]
            if i == 0:
                pwl_parts.append(f"{t_ramp}u {v:.6f}")
            else:
                pwl_parts.append(f"{t_start}u {V_beats[beat_indices[i-1], ch]:.6f}")
                pwl_parts.append(f"{t_ramp}u {v:.6f}")
            if i == len(beat_indices) - 1:
                pwl_parts.append(f"{total_time}u {v:.6f}")

        lines.append(f"Vin{ch} in{ch} 0 PWL(" + " ".join(pwl_parts) + ")")

    lines.append("")
    lines.append(f".tran 100n {total_time}u")
    lines.append("")
    lines.append(".control")
    lines.append("run")
    lines.append("")

    for i, bi in enumerate(beat_indices):
        t_meas = i * SETTLE_US + MEAS_OFFSET_US
        for ch in range(5):
            lines.append(f"meas tran b{bi}_o{ch} FIND v(out{ch}) AT={t_meas}u")

    lines.append("")
    for i, bi in enumerate(beat_indices):
        lines.append(f"print b{bi}_o0 b{bi}_o1 b{bi}_o2 b{bi}_o3 b{bi}_o4")

    lines.append("")
    lines.append("quit")
    lines.append(".endc")
    lines.append(".end")

    return "\n".join(lines)


def parse_ngspice_output(stdout, beat_indices):
    results = {}
    for bi in beat_indices:
        results[bi] = np.zeros(5)

    for line in stdout.split('\n'):
        s = line.strip()
        for bi in beat_indices:
            for ch in range(5):
                tag = f"b{bi}_o{ch}"
                if s.startswith(tag) and '=' in s:
                    try:
                        val = float(s.split('=')[1].strip().split()[0])
                        results[bi][ch] = val
                    except:
                        pass

    return results


def main():
    print("=" * 70)
    print("  COMPREHENSIVE WTA VALIDATION WITH REAL CLASSIFIER SCORES")
    print("=" * 70)

    print("\n[1/5] Computing Z2 scores from effective weights...")
    Z2, y_te = compute_z2_scores()
    print(f"  Total test beats: {len(y_te)}")
    print(f"  Z2 shape: {Z2.shape}")
    print(f"  Z2 range: [{Z2.min():.4f}, {Z2.max():.4f}]")

    sw_pred = np.argmax(Z2, axis=1)
    margins = np.zeros(len(y_te))
    for i in range(len(y_te)):
        sorted_scores = np.sort(Z2[i])
        margins[i] = sorted_scores[-1] - sorted_scores[-2]
    print(f"  Score margins: min={margins.min():.4f}, median={np.median(margins):.4f}, max={margins.max():.4f}")

    close_beats = np.sum(margins < 0.1)
    print(f"  Beats with margin < 0.1: {close_beats} ({close_beats/len(y_te)*100:.1f}%)")

    print("\n[2/5] Mapping Z2 to WTA input voltages...")
    V_beats, z_min, z_max = z2_to_voltage(Z2)
    print(f"  Z2 range [{z_min:.4f}, {z_max:.4f}] -> voltage [{V_WTA_MIN:.3f}V, {V_WTA_MAX:.3f}V]")
    print(f"  Voltage margins: min={margins.min()/(z_max-z_min)*(V_WTA_MAX-V_WTA_MIN)*1000:.1f}mV")

    print("\n[3/5] Selecting test beats (500 diverse cases, 50 tightest-margin)...")
    test_indices = select_test_beats(Z2, y_te, n_total=500)
    print(f"  Selected {len(test_indices)} beats")

    class_dist = np.bincount(y_te[test_indices], minlength=5)
    print(f"  Class distribution: " + ", ".join(f"{CLASS_NAMES[i]}={class_dist[i]}" for i in range(5)))

    test_margins = margins[test_indices]
    print(f"  Test margins: min={test_margins.min():.4f}, median={np.median(test_margins):.4f}")

    print("\n[4/5] Running ngspice batch simulations...")
    BATCH_SIZE = 25
    all_results = {}
    n_batches = (len(test_indices) + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        batch_idx = test_indices[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        tb = generate_batch_testbench(V_beats, batch_idx, b)

        tb_path = os.path.join(SCRIPT_DIR, f'_wta_batch_{b}.spice')
        with open(tb_path, 'w') as f:
            f.write(tb)

        t0 = time.time()
        result = subprocess.run(
            [NGSPICE, "-b", tb_path],
            capture_output=True, text=True, timeout=600, cwd=SCRIPT_DIR
        )
        dt = time.time() - t0

        batch_results = parse_ngspice_output(result.stdout, batch_idx)
        all_results.update(batch_results)

        valid = sum(1 for v in batch_results.values() if np.any(v != 0))
        print(f"  Batch {b+1}/{n_batches}: {len(batch_idx)} beats, {valid} valid, {dt:.1f}s")

        os.remove(tb_path)

    print("\n[5/5] Analyzing results...")
    print("-" * 70)

    n_correct = 0
    n_wrong = 0
    n_invalid = 0
    failures = []
    wta_margins_mv = []

    for bi in test_indices:
        vout = all_results.get(bi, np.zeros(5))

        if np.all(vout == 0) or np.all(vout == vout[0]):
            n_invalid += 1
            continue

        wta_pred = np.argmin(vout)
        sw_expected = np.argmax(Z2[bi])

        winner_v = vout[wta_pred]
        runner_up_v = np.partition(vout, 1)[1]
        margin_mv = (runner_up_v - winner_v) * 1000
        wta_margins_mv.append(margin_mv)

        if wta_pred == sw_expected:
            n_correct += 1
        else:
            n_wrong += 1
            z2_margin = Z2[bi, sw_expected] - np.partition(Z2[bi], -2)[-2]
            failures.append({
                'beat': bi,
                'true_class': CLASS_NAMES[y_te[bi]],
                'sw_pred': CLASS_NAMES[sw_expected],
                'wta_pred': CLASS_NAMES[wta_pred],
                'z2_scores': Z2[bi].round(4).tolist(),
                'wta_voltages': vout.round(6).tolist(),
                'z2_margin': round(z2_margin, 4),
                'wta_margin_mv': round(margin_mv, 2),
            })

    total_valid = n_correct + n_wrong
    match_rate = n_correct / total_valid * 100 if total_valid > 0 else 0

    print(f"\n  RESULTS:")
    print(f"  Total tested:     {len(test_indices)}")
    print(f"  Valid results:    {total_valid}")
    print(f"  Invalid (no sim): {n_invalid}")
    print(f"  Correct (match):  {n_correct}")
    print(f"  Wrong (mismatch): {n_wrong}")
    print(f"  Match rate:       {match_rate:.1f}%")

    if wta_margins_mv:
        arr = np.array(wta_margins_mv)
        print(f"\n  WTA output margins:")
        print(f"    Min:    {arr.min():.2f} mV")
        print(f"    Median: {np.median(arr):.2f} mV")
        print(f"    Mean:   {arr.mean():.2f} mV")
        print(f"    Max:    {arr.max():.2f} mV")

    if failures:
        print(f"\n  FAILURE DETAILS ({len(failures)} cases):")
        for f in failures[:10]:
            print(f"    Beat {f['beat']}: true={f['true_class']}, "
                  f"sw={f['sw_pred']}, wta={f['wta_pred']}, "
                  f"z2_margin={f['z2_margin']:.4f}, "
                  f"wta_margin={f['wta_margin_mv']:.2f}mV")
            print(f"      Z2:  {f['z2_scores']}")
            print(f"      WTA: {f['wta_voltages']}")

    report = {
        'total_beats': len(test_indices),
        'valid': total_valid,
        'correct': n_correct,
        'wrong': n_wrong,
        'match_rate_pct': round(match_rate, 2),
        'wta_margin_min_mv': round(min(wta_margins_mv), 2) if wta_margins_mv else None,
        'wta_margin_median_mv': round(float(np.median(wta_margins_mv)), 2) if wta_margins_mv else None,
        'failures': failures[:20],
        'voltage_range': [V_WTA_MIN, V_WTA_MAX],
        'z2_range': [float(z_min), float(z_max)],
    }
    out_path = os.path.join(SCRIPT_DIR, 'wta_comprehensive_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
