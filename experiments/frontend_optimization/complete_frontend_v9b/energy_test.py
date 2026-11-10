"""
Energy measurement for the COMPLETE ASIC analog frontend.
Includes ALL circuits that would be fabricated:
  - Bias generator
  - 2× Instrumentation amplifiers (one per ECG lead)
  - R-peak detector (comparator)
  - Post-R delay timer (cap discharge + 5 comparators)
  - P-wave predictor (2 source followers + 2 comparators)
  - 8× Feature extraction circuits

Measures total VDD current over 5 heartbeats.
"""
import subprocess, os, re, sys, time
import numpy as np

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")
VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130
PW_S = 16; PW_MID = 43

RECORD = '100'
N_BEATS = 5


def load_beats():
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, RECORD))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, RECORD), 'atr')
    sig = record.p_signal
    aami_map = {}
    for s in 'NLRej': aami_map[s] = 'N'
    for s in 'AaJS': aami_map[s] = 'S'
    for s in 'VE': aami_map[s] = 'V'
    for s in 'F': aami_map[s] = 'F'
    for s in '/fQ': aami_map[s] = 'Q'
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]
    all_peaks = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(np.abs(sig[bs+QRS_S:bs+QRS_E, 0])))
    gain = 0.16 / max(all_peaks)

    beats = []
    for i in range(5, 5 + N_BEATS + 2):
        if i >= len(valid) - 1: break
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beats.append({'pre_rr': pre_rr, 'ecg_l0': sig[bs:be, 0].copy(),
                       'ecg_l1': sig[bs:be, min(1, sig.shape[1]-1)].copy()})
    return beats[:N_BEATS], gain


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join(f"{t:.7f} {v:.6f}" for t, v in c)


def make_trig(rpeak_times, offset_s, pulse_w, total_time):
    pts = [(0, 0)]
    for rpt in rpeak_times:
        t = rpt + offset_s
        if t > 0.01:
            pts += [(t - 0.0005, 0), (t, VDD), (t + pulse_w, VDD), (t + pulse_w + 0.0005, 0)]
    pts.append((total_time, 0))
    return make_pwl(pts)


def generate_testbench(beats, gain):
    rpeak_times = [2.0]
    for i in range(1, len(beats)):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 1.0
    dt = 1.0 / FS

    ecg_l0_pts, ecg_l1_pts = [(0, VBASE)], [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt
        t_before = beat_start - 0.005
        if t_before > 0.01:
            ecg_l0_pts.append((t_before, VBASE))
            ecg_l1_pts.append((t_before, VBASE))
        for j in range(len(beats[i]['ecg_l0'])):
            t = beat_start + j * dt
            if t > 0:
                ecg_l0_pts.append((t, max(0.05, min(0.85, VBASE + gain * beats[i]['ecg_l0'][j]))))
                ecg_l1_pts.append((t, max(0.05, min(0.85, VBASE + gain * beats[i]['ecg_l1'][j]))))
        t_after = beat_start + len(beats[i]['ecg_l0']) * dt + 0.005
        ecg_l0_pts.append((t_after, VBASE))
        ecg_l1_pts.append((t_after, VBASE))
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    PW = 0.0005
    PW_RR = 0.002
    PW_RPEAK_RR = 0.005

    trig_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)
    trig_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)
    trig_rr_cap = make_trig(rpeak_times, -0.005, PW_RR, total_time)
    trig_rpeak_sh = make_trig(rpeak_times, 0, PW, total_time)
    trig_rr_rpeak = make_trig(rpeak_times, 0, PW_RPEAK_RR, total_time)
    trig_s14 = make_trig(rpeak_times, 5 / FS, PW, total_time)
    trig_s69 = make_trig(rpeak_times, 25 / FS, PW, total_time)
    trig_wid = make_trig(rpeak_times, 30 / FS, PW, total_time)
    trig_rr_copy = make_trig(rpeak_times, 0.010, PW_RR, total_time)
    trig_bi_copy = make_trig(rpeak_times, 0.300, PW_RR, total_time)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    frontend_path = FRONTEND_DIR.replace("\\", "/")

    t_start = rpeak_times[1] - 0.01
    t_end = rpeak_times[-1] + 0.35

    sp = f""".title energy_measurement_complete_asic_frontend
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

* ============================================================
* POWER SUPPLY — single VDD with current measurement
* ALL circuits powered from this one supply
* ============================================================
Vmeas vdd_ext vdd 0
Vvdd vdd_ext 0 {VDD}
Vss vss 0 0

* ============================================================
* 1. BIAS GENERATOR
*    Use ideal source for correct biasing (ensures DC convergence)
*    AND instantiate bias_gen subcircuit for its current draw.
*    In real ASIC, bias_gen replaces the ideal source.
* ============================================================
.include "{frontend_path}/bias_gen.spice"
Vbias_ideal vbias 0 0.300
Vref_ideal vref 0 0.300

* Bias gen draws current from VDD (connected to its own internal nodes)
Xbias bg_vbias bg_vref vdd vss bias_gen
.ic V(bg_vbias)=0.3

* ============================================================
* 2. INSTRUMENTATION AMPLIFIERS (2×, one per lead)
*    In this testbench: ECG is pre-amplified, so IAs see the
*    same signal and add their quiescent current to VDD.
*    In real ASIC: IAs would amplify raw electrode signals.
* ============================================================
.include "{frontend_path}/ia.spice"

* IA for Lead 0: single-ended input (differential pair sees ecg vs vref)
Xia_l0 ecg_l0 vref ia_out_l0 vdd vss vbias vref ia

* IA for Lead 1: single-ended input
Xia_l1 ecg_l1 vref ia_out_l1 vdd vss vbias vref ia

* ============================================================
* 3. R-PEAK DETECTOR (comparator watching Lead 0)
* ============================================================
.include "{frontend_path}/rpeak_detector.spice"
Vthresh rpeak_thresh 0 0.36
Xrpeak ecg_l0 rpeak_thresh rpeak_det_out vdd vss vbias rpeak_detector

* ============================================================
* 4. POST-R DELAY TIMER (5 comparators + timer cap)
*    Powered and drawing quiescent current.
*    Triggers come from external PWL for simulation accuracy.
* ============================================================
.include "{frontend_path}/inverter.spice"
.include "{frontend_path}/delay_timer.spice"

* Threshold voltages for the delay comparators
Vth14 vth_14 0 0.858
Vth69 vth_69 0 0.693
Vth83 vth_83 0 0.651
Vth300 vth_300 0 0.010
Vth10 vth_10 0 0.870

* Delay timer driven by rpeak detection output
Xdelay rpeak_det_out dt_s14 dt_s69 dt_wid dt_bi_copy dt_rr_copy vdd vss vbias vth_14 vth_69 vth_83 vth_300 vth_10 delay_timer

* ============================================================
* 5. P-WAVE PREDICTOR (2 source followers + 2 comparators)
*    Connected to the RR timer node for adaptive prediction.
*    Draws quiescent current from VDD.
* ============================================================
.include "{frontend_path}/pwave_predictor.spice"

* Connect to a dummy timer/EWMA node (voltage divider)
* In real operation these connect to rr_timing internal nodes
Rpw_timer_dummy vdd pw_timer_dummy 1e12
Rpw_ewma_dummy vdd pw_ewma_dummy 1e12
Cpw_timer pw_timer_dummy vss 1p
Cpw_ewma pw_ewma_dummy vss 1p

Xpw_pred pw_timer_dummy pw_ewma_dummy pw_pred_base pw_pred_peak vdd vss vbias pwave_predictor

* ============================================================
* 6-13. FEATURE EXTRACTION CIRCUITS (8 total)
*        These are the validated circuits from individual testing.
*        All share VDD, VSS, VBIAS.
* ============================================================

* ECG inputs (pre-amplified for simulation accuracy)
Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

* Trigger signals (from external PWL for accurate timing)
Vpwb trig_pw_base 0 PWL({trig_pw_base})
Vpwp trig_pw_peak 0 PWL({trig_pw_peak})
Vrrc trig_rr_cap 0 PWL({trig_rr_cap})
Vrsh trig_rpeak_sh 0 PWL({trig_rpeak_sh})
Vrrp trig_rr_rpeak 0 PWL({trig_rr_rpeak})
Vs14 trig_s14 0 PWL({trig_s14})
Vs69 trig_s69 0 PWL({trig_s69})
Vwid trig_wid 0 PWL({trig_wid})
Vrco trig_rr_copy 0 PWL({trig_rr_copy})
Vbco trig_bi_copy 0 PWL({trig_bi_copy})

.include "{frontend_path}/rr_timing.spice"
.include "{frontend_path}/beat_instability.spice"
.include "{frontend_path}/pwave_diff.spice"
.include "{frontend_path}/qrs_width.spice"
.include "{frontend_path}/qrs_slope.spice"
.include "{frontend_path}/qrs_symmetry.spice"

* F1 & F2: RR Timing
Xrr ecg_l0 trig_rr_cap trig_rr_rpeak trig_rr_copy f1_rr_ratio f2_rr_asym vdd vss vbias rr_timing

* F3: Beat Instability L0
Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability

* F4: Beat Instability L1
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability

* F5: P-wave Diff (Lead 0)
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff

* F6: QRS Width (Lead 0)
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width

* F7: QRS Slope (Lead 0)
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope

* F8: QRS Symmetry (Lead 0)
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

* ============================================================
* SIMULATION
* ============================================================
* Initial conditions for convergence
.ic V(pw_timer_dummy)=0.5 V(pw_ewma_dummy)=0.3

.tran 0.5m {total_time:.4f} uic

* Average VDD current over steady-state beats
.meas tran avg_i_vdd AVG I(Vmeas) FROM={t_start:.4f} TO={t_end:.4f}
.meas tran max_i_vdd MAX I(Vmeas) FROM={t_start:.4f} TO={t_end:.4f}
.meas tran min_i_vdd MIN I(Vmeas) FROM={t_start:.4f} TO={t_end:.4f}

.end
"""
    tb_path = os.path.join(FRONTEND_DIR, "_energy_test.spice")
    with open(tb_path, 'w') as f:
        f.write(sp)
    return tb_path, t_start, t_end


def run_ngspice(tb):
    log = tb.replace('.spice', '.log')
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    if not os.path.exists(log): return None, log
    with open(log, 'r') as f:
        text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    return results, log


print("=" * 70)
print("COMPLETE ASIC FRONTEND — Energy Measurement")
print("=" * 70)
print()
print("Circuit inventory (all powered from VDD):")
print("  1. Bias generator         — self-biased current mirror")
print("  2. IA Lead 0              — OTA + cap feedback (gain=100)")
print("  3. IA Lead 1              — OTA + cap feedback (gain=100)")
print("  4. R-peak detector        — 2-stage comparator")
print("  5. Post-R delay timer     — cap discharge + 5 comparators")
print("  6. P-wave predictor       — 2 source followers + 2 comparators")
print("  7. RR Timing (F1,F2)      — cap timer + EWMA + 2 diff pairs")
print("  8. Beat Instability L0 (F3) — S&H + EWMA + diff pair")
print("  9. Beat Instability L1 (F4) — S&H + EWMA + diff pair")
print(" 10. P-wave Diff (F5)       — 2 S&H + diff pair")
print(" 11. QRS Width (F6)         — passive S&H")
print(" 12. QRS Slope (F7)         — 2 S&H + diff pair")
print(" 13. QRS Symmetry (F8)      — passive S&H")
print()

print("Loading ECG beats...")
beats, gain = load_beats()
print(f"Loaded {len(beats)} beats from record {RECORD}, gain={gain:.4f}")

print("Generating testbench with all ASIC circuits...")
tb, t_start, t_end = generate_testbench(beats, gain)

print("Running ngspice (complete ASIC frontend)...")
t0 = time.time()
results, logpath = run_ngspice(tb)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s\n")

if not results:
    print(f"ERROR: ngspice failed. Check log: {logpath}")
    # Try to print last 30 lines of log
    try:
        with open(logpath) as f:
            lines = f.readlines()
        print("Last 30 lines of log:")
        for l in lines[-30:]:
            print("  ", l.rstrip())
    except:
        pass
    sys.exit(1)

avg_i = results.get('avg_i_vdd', None)
max_i = results.get('max_i_vdd', None)
min_i = results.get('min_i_vdd', None)

if avg_i is not None:
    avg_power = abs(avg_i) * VDD
    meas_duration = t_end - t_start
    n_beats_measured = len(beats) - 1
    avg_rr = meas_duration / n_beats_measured if n_beats_measured > 0 else 0.8
    energy_per_beat = avg_power * avg_rr

    print("=" * 70)
    print("ENERGY MEASUREMENT — COMPLETE ASIC FRONTEND")
    print("=" * 70)
    print(f"  VDD                  = {VDD} V")
    print(f"  Technology           = SKY130 (130nm CMOS)")
    print(f"  Measurement window   = {t_start:.3f}s to {t_end:.3f}s ({meas_duration:.3f}s)")
    print(f"  Beats measured       = {n_beats_measured}")
    print(f"  Average RR interval  = {avg_rr*1000:.0f} ms")
    print()
    print(f"  Average VDD current  = {abs(avg_i)*1e12:.2f} pA")
    if abs(avg_i) >= 1e-9:
        print(f"                       = {abs(avg_i)*1e9:.3f} nA")
    if max_i:
        print(f"  Peak VDD current     = {abs(max_i)*1e9:.3f} nA")
    if min_i:
        print(f"  Min VDD current      = {abs(min_i)*1e12:.2f} pA")
    print()
    print(f"  Average power        = {avg_power*1e12:.2f} pW")
    print(f"                       = {avg_power*1e9:.4f} nW")
    if avg_power >= 1e-6:
        print(f"                       = {avg_power*1e6:.4f} µW")
    print(f"  Energy per heartbeat = {energy_per_beat*1e12:.2f} pJ")
    print()
    print("  Circuit-by-circuit static current budget (estimated):")
    print("  +----------------------------+-------------+-------+------------+")
    print("  | Block                      | Tail I (pA) | Count | Total (pA) |")
    print("  +----------------------------+-------------+-------+------------+")
    print("  | Bias generator             |     ~150    |   2   |     ~300   |")
    print("  | IA (OTA)                   |     ~150    |   2   |     ~300   |")
    print("  | R-peak detector (stg1)     |     ~150    |   1   |     ~150   |")
    print("  | Delay timer (discharge)    |     ~300    |   1   |     ~300   |")
    print("  | Delay timer (comparators)  |     ~150    |   5   |     ~750   |")
    print("  | P-wave pred (src follow)   |     ~150    |   2   |     ~300   |")
    print("  | P-wave pred (comparators)  |     ~150    |   2   |     ~300   |")
    print("  | RR timing (discharge)      |     ~148    |   1   |     ~148   |")
    print("  | RR timing (diff pairs)     |     ~150    |   2   |     ~300   |")
    print("  | Beat instability (x2)      |     ~150    |   2   |     ~300   |")
    print("  | P-wave diff (diff pair)    |     ~150    |   1   |     ~150   |")
    print("  | QRS slope (diff pair)      |     ~150    |   1   |     ~150   |")
    print("  | QRS width (passive)        |       0     |   1   |       0    |")
    print("  | QRS symmetry (passive)     |       0     |   1   |       0    |")
    print("  +----------------------------+-------------+-------+------------+")
    print("  | TOTAL ESTIMATED            |             |  24   |   ~3,448   |")
    print("  +----------------------------+-------------+-------+------------+")
    print()
    est_power = 3448e-12 * VDD
    print(f"  Estimated total power = {est_power*1e9:.2f} nW")
    print(f"  Simulated total power = {avg_power*1e9:.4f} nW")
    print(f"  Ratio (sim/est)       = {avg_power/est_power:.2f}×")
    print()
    print("  Note: Peak current includes dynamic switching during S&H")
    print("  transitions and timer precharge events.")
    print("=" * 70)
else:
    print("ERROR: could not parse VDD current measurement")
    print(f"Check log: {logpath}")
