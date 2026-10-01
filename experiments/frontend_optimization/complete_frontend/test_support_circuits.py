"""
Validation testbenches for all support circuits in the analog frontend.
Tests: bias_gen, ia, rpeak_detector, delay_timer
Each test creates a standalone SPICE netlist, runs ngspice, and verifies behavior.
"""
import subprocess, os, sys, re
import numpy as np

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
VDD = 0.9
VBIAS = 0.300

def run_ngspice(netlist_str, label="test"):
    tmp = os.path.join(FRONTEND_DIR, f"_tb_{label}.spice")
    with open(tmp, "w") as f:
        f.write(netlist_str)
    result = subprocess.run(
        [NGSPICE, "-b", tmp],
        capture_output=True, text=True, timeout=300,
        cwd=FRONTEND_DIR
    )
    return result, tmp

def load_wrdata(filepath):
    data = np.loadtxt(filepath)
    return data

# ============================================================
# TEST 1: Bias Generator
# ============================================================
def test_bias_gen():
    print("=" * 60)
    print("TEST 1: Bias Generator")
    print("=" * 60)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    data_file = os.path.join(FRONTEND_DIR, "_tb_bias_data.txt").replace("\\", "/")

    netlist = f"""\
Bias Generator Testbench
.lib "{lib_path}" tt
.option scale=1.0u method=gear gmin=1e-15

.include "bias_gen.spice"

Vdd vdd 0 {VDD}
Vss vss 0 0

Xbias vbias vref vdd vss bias_gen

.ic v(xbias.iref) = 0.3

.tran 1m 200m uic

.control
run
set wr_singlescale
wrdata {data_file} v(vbias) v(vref)
.endc
.end
"""
    result, tmp = run_ngspice(netlist, "bias_gen")

    if not os.path.exists(data_file):
        print(f"  FAIL: no output data")
        print(result.stdout[-500:] if result.stdout else "")
        print(result.stderr[-500:] if result.stderr else "")
        return False

    data = load_wrdata(data_file)
    time_col = data[:, 0]
    vbias_col = data[:, 1]
    vref_col = data[:, 2]

    final_vbias = np.mean(vbias_col[-10:])
    final_vref = np.mean(vref_col[-10:])

    print(f"  Vbias final: {final_vbias*1000:.1f} mV (target: ~300 mV)")
    print(f"  Vref final:  {final_vref*1000:.1f} mV (target: ~300 mV)")

    started = final_vbias > 0.05
    reasonable = 0.15 < final_vbias < 0.50
    ref_match = abs(final_vbias - final_vref) < 0.01

    if started and reasonable and ref_match:
        print("  PASS: Bias gen starts up and settles to reasonable voltage")
        return True
    else:
        if not started:
            print("  FAIL: Bias gen did not start up (Vbias near 0V)")
        if not reasonable:
            print(f"  FAIL: Vbias outside 150-500mV range: {final_vbias*1000:.1f}mV")
        if not ref_match:
            print(f"  FAIL: Vref != Vbias (diff = {abs(final_vbias-final_vref)*1000:.1f}mV)")
        return False


# ============================================================
# TEST 2: Instrumentation Amplifier
# ============================================================
def test_ia():
    print("\n" + "=" * 60)
    print("TEST 2: Instrumentation Amplifier (CCIA)")
    print("=" * 60)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    data_file = os.path.join(FRONTEND_DIR, "_tb_ia_data.txt").replace("\\", "/")

    freq = 10
    amp_mv = 1.0
    sim_time = 0.5

    netlist = f"""\
IA Testbench - 1mV differential sine at {freq}Hz
.lib "{lib_path}" tt
.option scale=1.0u method=gear gmin=1e-15

.include "ia.spice"

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 {VBIAS}
Vref vref 0 {VBIAS}

* Differential input: +/- 0.5mV around VREF
Vinp inp 0 SIN({VBIAS} {amp_mv/2}e-3 {freq})
Vinn inn 0 SIN({VBIAS} {-amp_mv/2}e-3 {freq})

Xia inp inn out vdd vss vbias vref ia

.tran 0.1m {sim_time} uic

.control
run
set wr_singlescale
wrdata {data_file} v(out) v(inp) v(inn)
.endc
.end
"""
    result, tmp = run_ngspice(netlist, "ia")

    if not os.path.exists(data_file):
        print(f"  FAIL: no output data")
        print(result.stdout[-500:] if result.stdout else "")
        return False

    data = load_wrdata(data_file)
    time_col = data[:, 0]
    vout = data[:, 1]
    vinp = data[:, 2]
    vinn = data[:, 3]

    settle_idx = len(time_col) // 4
    vout_ac = vout[settle_idx:]
    vinp_ac = vinp[settle_idx:]
    vinn_ac = vinn[settle_idx:]

    v_diff_pp = np.max(vinp_ac - vinn_ac) - np.min(vinp_ac - vinn_ac)
    vout_pp = np.max(vout_ac) - np.min(vout_ac)

    if v_diff_pp > 1e-6:
        gain = vout_pp / v_diff_pp
    else:
        gain = 0

    dc_out = np.mean(vout_ac)

    print(f"  Input diff pk-pk:  {v_diff_pp*1000:.3f} mV")
    print(f"  Output pk-pk:      {vout_pp*1000:.1f} mV")
    print(f"  Measured gain:     {gain:.1f} (target: ~100)")
    print(f"  Output DC:         {dc_out*1000:.1f} mV (target: ~{VBIAS*1000:.0f} mV)")

    gain_ok = 20 < gain < 500
    dc_ok = abs(dc_out - VBIAS) < 0.15

    if gain_ok and dc_ok:
        print("  PASS: IA amplifies correctly with DC near VREF")
        return True
    else:
        if not gain_ok:
            print(f"  FAIL: Gain {gain:.1f} outside 20-500 range")
        if not dc_ok:
            print(f"  FAIL: DC offset too far from VREF ({abs(dc_out-VBIAS)*1000:.1f}mV)")
        return False


# ============================================================
# TEST 3: R-Peak Detector
# ============================================================
def test_rpeak_detector():
    print("\n" + "=" * 60)
    print("TEST 3: R-Peak Detector")
    print("=" * 60)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    data_file = os.path.join(FRONTEND_DIR, "_tb_rpeak_data.txt").replace("\\", "/")

    threshold = 0.36
    r_peak_v = 0.42
    baseline_v = VBIAS

    pwl_points = []
    t = 0
    for i in range(3):
        pwl_points.append(f"{t:.4f} {baseline_v}")
        t += 0.3
        pwl_points.append(f"{t:.4f} {baseline_v}")
        t += 0.01
        pwl_points.append(f"{t:.4f} {r_peak_v}")
        t += 0.02
        pwl_points.append(f"{t:.4f} {r_peak_v}")
        t += 0.01
        pwl_points.append(f"{t:.4f} {baseline_v}")

    total_t = t + 0.2
    pwl_str = " ".join(pwl_points)

    netlist = f"""\
R-Peak Detector Testbench
.lib "{lib_path}" tt
.option scale=1.0u method=gear gmin=1e-15

.include "rpeak_detector.spice"

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 {VBIAS}
Vthresh threshold 0 {threshold}

Vecg ecg_in 0 PWL({pwl_str})

Xrpeak ecg_in threshold rpeak_out vdd vss vbias rpeak_detector

.tran 0.1m {total_t} uic

.control
run
set wr_singlescale
wrdata {data_file} v(rpeak_out) v(ecg_in)
.endc
.end
"""
    result, tmp = run_ngspice(netlist, "rpeak")

    if not os.path.exists(data_file):
        print(f"  FAIL: no output data")
        return False

    data = load_wrdata(data_file)
    time_col = data[:, 0]
    rpeak_out = data[:, 1]
    ecg_in = data[:, 2]

    above_thresh = ecg_in > threshold
    out_high = rpeak_out > 0.5 * VDD

    correct_when_above = np.sum(above_thresh & out_high)
    total_above = np.sum(above_thresh)
    correct_when_below = np.sum(~above_thresh & ~out_high)
    total_below = np.sum(~above_thresh)

    if total_above > 0:
        sensitivity = correct_when_above / total_above
    else:
        sensitivity = 0
    if total_below > 0:
        specificity = correct_when_below / total_below
    else:
        specificity = 0

    print(f"  Sensitivity (HIGH when ECG > thresh): {sensitivity*100:.1f}%")
    print(f"  Specificity (LOW when ECG < thresh):  {specificity*100:.1f}%")
    print(f"  Output range: {np.min(rpeak_out):.3f}V to {np.max(rpeak_out):.3f}V")

    passed = sensitivity > 0.7 and specificity > 0.7
    if passed:
        print("  PASS: R-peak detector correctly discriminates threshold")
    else:
        print("  FAIL: Poor discrimination")
    return passed


# ============================================================
# TEST 4: Delay Timer
# ============================================================
def test_delay_timer():
    print("\n" + "=" * 60)
    print("TEST 4: Delay Timer")
    print("=" * 60)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    data_file = os.path.join(FRONTEND_DIR, "_tb_delay_data.txt").replace("\\", "/")

    I_discharge = 300e-12
    C_timer = 100e-12
    vth_14_v = VDD - I_discharge * 0.014 / C_timer
    vth_28_v = VDD - I_discharge * 0.028 / C_timer
    vth_69_v = VDD - I_discharge * 0.069 / C_timer
    vth_300_v = max(0.01, VDD - I_discharge * 0.300 / C_timer)

    I_fast = 300e-12
    C_fast = 2.048e-12
    vth_10_v = max(0.01, VDD - I_fast * 0.010 / C_fast)

    print(f"  Threshold voltages:")
    print(f"    vth_14  = {vth_14_v:.4f}V (14ms delay)")
    print(f"    vth_28  = {vth_28_v:.4f}V (28ms width delay)")
    print(f"    vth_69  = {vth_69_v:.4f}V (69ms delay)")
    print(f"    vth_300 = {vth_300_v:.4f}V (300ms delay)")
    print(f"    vth_10  = {vth_10_v:.4f}V (10ms fast timer)")

    rpeak_pulse_t = 0.010
    pw = 0.0005

    netlist = f"""\
Delay Timer Testbench
.lib "{lib_path}" tt
.option scale=1.0u method=gear gmin=1e-15

.include "inverter.spice"
.include "delay_timer.spice"

Vdd vdd 0 {VDD}
Vss vss 0 0
Vbias vbias 0 {VBIAS}

* R-peak pulse at t=10ms
Vrpeak rpeak_pulse 0 PULSE(0 {VDD} {rpeak_pulse_t} 100n 100n {pw} 100)

* Threshold voltages
Vth14 vth_14 0 {vth_14_v}
Vth69 vth_69 0 {vth_69_v}
Vth83 vth_83 0 {vth_28_v}
Vth300 vth_300 0 {vth_300_v}
Vth10 vth_10 0 {vth_10_v}

Xdelay rpeak_pulse trig_s14 trig_s69 trig_wid trig_bi_copy trig_rr_copy
+ vdd vss vbias vth_14 vth_69 vth_83 vth_300 vth_10 delay_timer

* Initialize timer caps to VDD (as if already precharged)
.ic v(xdelay.timer_node) = {VDD}
.ic v(xdelay.timer2) = {VDD}

.tran 0.1m 400m uic

.control
run
set wr_singlescale
wrdata {data_file} v(trig_s14) v(trig_s69) v(trig_wid) v(trig_bi_copy) v(trig_rr_copy) v(xdelay.timer_node)
.endc
.end
"""
    result, tmp = run_ngspice(netlist, "delay_timer")

    if not os.path.exists(data_file):
        print(f"  FAIL: no output data")
        print(result.stdout[-500:] if result.stdout else "")
        return False

    data = load_wrdata(data_file)
    time_col = data[:, 0]
    trig_s14 = data[:, 1]
    trig_s69 = data[:, 2]
    trig_wid = data[:, 3]
    trig_bi  = data[:, 4]
    trig_rr  = data[:, 5]
    timer_v  = data[:, 6]

    # Debug: print timer voltage at key times
    for t_check in [0, 5, 10, 10.5, 15, 20, 25, 30, 50, 80, 100]:
        idx = np.argmin(np.abs(time_col - t_check * 1e-3))
        print(f"  Timer at t={t_check}ms: {timer_v[idx]*1000:.1f}mV, s14={trig_s14[idx]:.3f}V, s69={trig_s69[idx]:.3f}V")

    thresh_high = 0.5 * VDD
    expected_delays = {
        'trig_s14':     (14e-3, trig_s14),
        'trig_wid':     (28e-3, trig_wid),
        'trig_s69':     (69e-3, trig_s69),
        'trig_rr_copy': (10e-3, trig_rr),
    }

    all_pass = True
    for name, (expected_delay, signal) in expected_delays.items():
        crossings = np.where(
            (signal[:-1] < thresh_high) & (signal[1:] >= thresh_high)
        )[0]
        post_rpeak = [c for c in crossings if time_col[c] > rpeak_pulse_t]
        if len(post_rpeak) > 0:
            first_cross = post_rpeak[0]
            actual_delay = time_col[first_cross] - rpeak_pulse_t
            error_pct = abs(actual_delay - expected_delay) / expected_delay * 100
            ok = error_pct < 100
            status = "OK" if ok else "FAIL"
            print(f"  {name:15s}: expected {expected_delay*1000:.1f}ms, got {actual_delay*1000:.1f}ms (err: {error_pct:.0f}%) [{status}]")
            if not ok:
                all_pass = False
        else:
            print(f"  {name:15s}: NO rising edge after R-peak [FAIL]")
            all_pass = False

    if all_pass:
        print("  PASS: All delay timer triggers fire at correct times")
    else:
        print("  PARTIAL: Some triggers did not fire at expected times")
    return all_pass


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    results = {}

    results['bias_gen'] = test_bias_gen()
    results['ia'] = test_ia()
    results['rpeak_detector'] = test_rpeak_detector()
    results['delay_timer'] = test_delay_timer()

    print("\n" + "=" * 60)
    print("SUPPORT CIRCUIT VALIDATION SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} : {status}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")

    if n_pass == n_total:
        print("  ALL SUPPORT CIRCUITS VALIDATED")
    else:
        print("  SOME TESTS FAILED - review output above")
        sys.exit(1)
