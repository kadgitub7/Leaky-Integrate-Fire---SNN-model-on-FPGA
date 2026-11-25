"""
Diagnostic script: Trace every signal in the analog trigger chain on Record 100.
Identifies where the analog pipeline breaks down.

Instruments:
  1. rpeak_detector output (rpeak_level)
  2. rpeak_hold output (rpeak_level_held)
  3. true_peak_detector output (peak_detected)
  4. true_rpeak_pulse (after pulse_gen)
  5. delay_timer SF outputs at expected trigger times
  6. All feature outputs at multiple time offsets
  7. Comparison: annotation time vs actual analog trigger time
"""
import subprocess, os, re, sys, time
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend_v9b")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9
VBASE = 0.30
FS = 360
WIN_L, WIN_R = 90, 108
RPEAK_IDX = 20
QRS_S, QRS_E = 70, 130
EWMA_ALPHA = 0.015


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join(f"{t:.7f} {v:.6f}" for t, v in c)


def load_record(record_id):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal.copy()
    n_leads = sig.shape[1]
    aami_map = {}
    for s in 'NLRej': aami_map[s] = 'N'
    for s in 'AaJS': aami_map[s] = 'S'
    for s in 'VE': aami_map[s] = 'V'
    for s in 'F': aami_map[s] = 'F'
    for s in '/fQ': aami_map[s] = 'Q'
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    rpk_vals = [sig[valid[i][0], 0] for i in range(1, min(len(valid)-1, 30))
                if valid[i][0]-WIN_L >= 0 and valid[i][0]+WIN_R < len(sig)]
    if rpk_vals and np.median(rpk_vals) < 0:
        sig[:, 0] = -sig[:, 0]
        if n_leads > 1: sig[:, 1] = -sig[:, 1]

    all_peaks = []
    for i in range(1, len(valid)-1):
        idx = valid[i][0]
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(sig[bs+QRS_S:bs+QRS_E, 0]))
    gain = 0.16 / max(all_peaks) if max(all_peaks) > 0 else 0.05

    r_peak_scaled = [VBASE + gain * sig[valid[i][0], 0]
                     for i in range(1, min(len(valid)-1, 30))
                     if valid[i][0]-WIN_L >= 0 and valid[i][0]+WIN_R < len(sig)]
    max_r = max(r_peak_scaled) if r_peak_scaled else VBASE + 0.08
    rpeak_threshold = VBASE + 0.30 * (max_r - VBASE)

    beats = []
    ewma_rr = None
    for i in range(1, min(len(valid)-1, 10)):  # Only 8 beats for diagnostics
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        post_rr = (valid[i+1][0] - idx) / FS
        bs = idx - WIN_L
        be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]
        if ewma_rr is None:
            ewma_rr_before = pre_rr; ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr
        beats.append({
            'sample': idx, 'pre_rr': pre_rr, 'post_rr': post_rr,
            'ewma_rr': ewma_rr, 'ewma_rr_before': ewma_rr_before,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
        })

    return beats, gain, rpeak_threshold, sig, n_leads


def run_diagnostic(record_id='100'):
    beats, gain, rpeak_threshold, sig, n_leads = load_record(record_id)
    n = len(beats)
    print(f"Record {record_id}: {n} beats, gain={gain:.4f}, thresh={rpeak_threshold:.4f}")

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    # Build rpeak_times
    rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])

    # Continuous ECG
    dt_samp = 1.0 / FS
    total_time = rpeak_times[-1] + 1.0
    first_samp = beats[0]['sample'] - int(0.5 * FS)
    last_samp = beats[-1]['sample'] + int(1.2 * FS)
    first_samp = max(0, first_samp)
    last_samp = min(sig.shape[0]-1, last_samp)
    samp_at_first_rpk = beats[0]['sample']
    t_offset = rpeak_times[0] - samp_at_first_rpk / FS

    ecg_l0_pts, ecg_l1_pts = [], []
    for s in range(first_samp, last_samp+1):
        t = s / FS + t_offset
        if t < 0: continue
        if t > total_time: break
        v0 = max(0.05, min(0.85, VBASE + gain * sig[s, 0]))
        v1 = max(0.05, min(0.85, VBASE + gain * sig[s, min(1, n_leads-1)]))
        ecg_l0_pts.append((t, v0))
        ecg_l1_pts.append((t, v1))
    if ecg_l0_pts[0][0] > 0.001:
        ecg_l0_pts.insert(0, (0, VBASE))
        ecg_l1_pts.insert(0, (0, VBASE))
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    # Print expected ECG values at annotation R-peaks
    print(f"\n--- ECG at annotation R-peak times ---")
    for j, bt in enumerate(beats):
        t_rpk = rpeak_times[j]
        # Find ECG voltage at R-peak time
        ecg_at_rpk = None
        for i in range(len(ecg_l0_pts)-1):
            if ecg_l0_pts[i][0] <= t_rpk <= ecg_l0_pts[i+1][0]:
                frac = (t_rpk - ecg_l0_pts[i][0]) / (ecg_l0_pts[i+1][0] - ecg_l0_pts[i][0])
                ecg_at_rpk = ecg_l0_pts[i][1] + frac * (ecg_l0_pts[i+1][1] - ecg_l0_pts[i][1])
                break
        print(f"  Beat {j}: t={t_rpk:.4f}s, ECG@R={ecg_at_rpk:.4f}V, "
              f"thresh={rpeak_threshold:.4f}V, above={'YES' if ecg_at_rpk and ecg_at_rpk > rpeak_threshold else 'NO'}")

    # Build analog testbench with heavy instrumentation
    # Use simple constant pwave thresholds (approximate)
    ewma_rr_ms = beats[0]['pre_rr'] * 1000

    # We need calibration data - run inline calibration
    print(f"\nCalibrating timers...")
    cal_sp = f""".title timer_cal
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {VDD}
Vss vss 0 0
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/rr_timing.spice"
Xbias vbias vref vdd vss bias_gen
Vrpk rpeak_pulse 0 PWL(0 0 0.098 0 0.1 {VDD} 0.12 {VDD} 0.121 0 2.0 0)
Vt1 vth_14_fast 0 0.1
Vt2 vth_69 0 0.1
Vt3 vth_28 0 0.1
Vt4 vth_300 0 0.1
Vt5 vth_10 0 0.1
Vt6 vth_rpk_del 0 0.1
Vecg ecg_l0 0 0.5
Vcap cap_trig 0 0
Vcopy copy_trig 0 0
Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_rpk_del dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 vth_rpk_del delay_timer
Xrr ecg_l0 cap_trig rpeak_pulse copy_trig f1 f2 rr_timer rr_ewma vdd vss vbias rr_timing
.ic V(xbias.iref)=0.3
.tran 0.5m 2.0
"""
    for t in [1,2,3,5,10,15,20,50,100,200,300,400,500,600,700,800]:
        cal_sp += f".meas tran dt_sf_{t:03d} FIND V(xdelay.sf_timer) AT={0.121+t/1000:.4f}\n"
    for t in [1,2,3,5,10,20]:
        cal_sp += f".meas tran ft_sf_{t:03d} FIND V(xdelay.sf_timer2) AT={0.121+t/1000:.4f}\n"
    for t in [10,20,50,100,200,300,400,500,600,700,800,900,1000]:
        cal_sp += f".meas tran rr_sf_{t:04d} FIND V(rr_timer) AT={0.106+t/1000:.4f}\n"
    cal_sp += "\n.end\n"

    cal_tb = os.path.join(FRONTEND_DIR, "_diag_cal.spice")
    cal_log = cal_tb.replace('.spice', '.log')
    with open(cal_tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(cal_sp)
    subprocess.run([NGSPICE, '-b', '-o', cal_log, cal_tb],
                   capture_output=True, text=True, timeout=180, cwd=NGSPICE_DIR)
    with open(cal_log) as f:
        cal_text = f.read()
    cal_results = {}
    for line in cal_text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: cal_results[m.group(1).lower()] = float(m.group(2))
            except: pass
    try: os.remove(cal_tb); os.remove(cal_log)
    except: pass

    dt_times = [1,2,3,5,10,15,20,50,100,200,300,400,500,600,700,800]
    dt_sf = [cal_results.get(f'dt_sf_{t:03d}') for t in dt_times]
    ft_times = [1,2,3,5,10,20]
    ft_sf = [cal_results.get(f'ft_sf_{t:03d}') for t in ft_times]
    rr_times = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
    rr_sf = [cal_results.get(f'rr_sf_{t:04d}') for t in rr_times]

    dt_t = [t for t, v in zip(dt_times, dt_sf) if v is not None]
    dt_v = [v for v in dt_sf if v is not None]
    ft_t = [t for t, v in zip(ft_times, ft_sf) if v is not None]
    ft_v = [v for v in ft_sf if v is not None]
    rr_t = [t for t, v in zip(rr_times, rr_sf) if v is not None]
    rr_v = [v for v in rr_sf if v is not None]

    def interp_sf(times, voltages, target):
        for i in range(len(times)-1):
            if times[i] <= target <= times[i+1]:
                frac = (target - times[i]) / (times[i+1] - times[i])
                return voltages[i] + frac * (voltages[i+1] - voltages[i])
        if target < times[0]:
            rate = (voltages[0] - voltages[1]) / (times[1] - times[0])
            return voltages[0] + rate * (times[0] - target)
        rate = (voltages[-2] - voltages[-1]) / (times[-1] - times[-2])
        return voltages[-1] - rate * (target - times[-1])

    rpk_drop = 5.0
    pipe_main = 3.5
    pipe_fast = 1.0
    vth_14 = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_fast))
    vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_main))
    vth_69 = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_main))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_main))
    vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_fast))
    print(f"  vth_14={vth_14:.5f} vth_28={vth_28:.5f} vth_69={vth_69:.5f} vth_300={vth_300:.5f}")

    # Pwave thresholds (PWL - step per beat)
    pwl_base_pts = [(0, 0.3)]
    pwl_peak_pts = [(0, 0.3)]
    pwl_rrcap_pts = [(0, 0.3)]
    for j in range(n):
        ewma_rr_j = (beats[j].get('ewma_rr_before', beats[j]['ewma_rr'])) * 1000
        t_rpk = rpeak_times[j]
        pw_base_elapsed = max(1, 0.75 * ewma_rr_j - rpk_drop - pipe_main)
        pw_peak_elapsed = max(1, 0.80 * ewma_rr_j - rpk_drop - pipe_main)
        rr_cap_elapsed  = max(1, ewma_rr_j - 50 - rpk_drop - pipe_main)
        v_base = interp_sf(rr_t, rr_v, pw_base_elapsed)
        v_peak = interp_sf(rr_t, rr_v, pw_peak_elapsed)
        v_rrcap = interp_sf(rr_t, rr_v, rr_cap_elapsed)
        t_step = t_rpk + 0.001
        if t_step > pwl_base_pts[-1][0]:
            pwl_base_pts.append((t_step, v_base))
            pwl_peak_pts.append((t_step, v_peak))
            pwl_rrcap_pts.append((t_step, v_rrcap))
    pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
    pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
    pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))

    # Build the ANALOG testbench with DIAGNOSTIC measurements
    sp = f""".title diag_analog_{record_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0

Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

Vth_rpeak rpeak_threshold 0 {rpeak_threshold:.4f}
Vth14  vth_14_fast 0 {vth_14:.6f}
Vth69  vth_69      0 {vth_69:.6f}
Vth28  vth_28      0 {vth_28:.6f}
Vth300 vth_300     0 {vth_300:.6f}
Vth10  vth_10      0 {vth_10:.6f}
Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})
Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})
Vth_rrc vth_rr_cap  0 PWL({make_pwl(pwl_rrcap_pts)})

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/rpeak_hold.spice"
.include "{fp}/true_peak_detector.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/long_pulse_gen.spice"
.include "{fp}/edge_buffer.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

* 1. BIAS
Xbias vbias vref vdd vss bias_gen

* 2. R-PEAK CHAIN
Xrpeak ecg_l0 rpeak_threshold rpeak_level vdd vss vbias rpeak_detector
Xrpk_hold rpeak_level rpeak_level_held vdd vss vbias rpeak_hold
Xtpd ecg_l0 rpeak_level_held peak_detected vdd vss vbias true_peak_detector
Xpg_rpk peak_detected true_rpeak_pulse vdd vss vbias pulse_gen

* 3. DELAY TIMER
Vth_rpkdel vth_rpk_del 0 0.48
Xdt true_rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_rpk_del_unused dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 vth_rpk_del delay_timer

* 4. TRIGGER PULSES
Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias long_pulse_gen
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen
Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen

* 5. PWAVE COMPARATORS (on rr_timer)
XMcpwb1 cnpwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 pw_base_lev rr_timer ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 pw_base_lev cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

XMcpwp1 cnpwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 pw_peak_lev rr_timer ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 pw_peak_lev cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

XMcrrc1 cnrrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 rr_cap_lev rr_timer ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 rr_cap_lev cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen
Xeb_pwp pw_peak_lev pwp_sharp vdd vss edge_buffer
Xpg_pwp pwp_sharp   trig_pw_peak vdd vss vbias pulse_gen

* rr_cap: use comparator level directly
Xeb_rrc rr_cap_lev  trig_rr_cap vdd vss edge_buffer

* 6. FEATURES
Xrr ecg_l0 trig_rr_cap true_rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 true_rpeak_pulse trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 true_rpeak_pulse trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 true_rpeak_pulse f8_symmetry vdd vss qrs_symmetry

* === INITIAL CONDITIONS ===
.ic V(xbias.iref)=0.3
.ic V(xrr.ewma_node)=0.40
.ic V(xrr.esamp_node)=0.40
.ic V(xrr.prev_node)=0.40
.ic V(xbi0.prev_node)=0.30
.ic V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30
.ic V(xbi1.esamp_node)=0.30
.ic V(xtpd.peak_hold)=0.30

.tran 0.5m {total_time:.4f} 0 0.2m

"""

    # === DIAGNOSTIC MEASUREMENTS ===
    # For each beat: measure the trigger chain nodes at multiple times
    for j in range(n):
        t_rpk = rpeak_times[j]

        # 1. Trigger chain signals AT the annotation R-peak time
        sp += f"* --- Beat {j} (annotation t={t_rpk:.4f}s) ---\n"
        sp += f".meas tran ecg_b{j} FIND V(ecg_l0) AT={t_rpk:.7f}\n"
        sp += f".meas tran rpklev_b{j} FIND V(rpeak_level) AT={t_rpk:.7f}\n"
        sp += f".meas tran rpkheld_b{j} FIND V(rpeak_level_held) AT={t_rpk:.7f}\n"
        sp += f".meas tran peakdet_b{j} FIND V(peak_detected) AT={t_rpk:.7f}\n"
        sp += f".meas tran trpkpls_b{j} FIND V(true_rpeak_pulse) AT={t_rpk:.7f}\n"
        sp += f".meas tran peakhold_b{j} FIND V(xtpd.peak_hold) AT={t_rpk:.7f}\n"

        # 2. Trigger chain 5ms after annotation
        t5 = t_rpk + 0.005
        if t5 < total_time:
            sp += f".meas tran rpklev5_b{j} FIND V(rpeak_level) AT={t5:.7f}\n"
            sp += f".meas tran peakdet5_b{j} FIND V(peak_detected) AT={t5:.7f}\n"
            sp += f".meas tran trpkpls5_b{j} FIND V(true_rpeak_pulse) AT={t5:.7f}\n"

        # 3. Trigger chain 10ms after annotation
        t10 = t_rpk + 0.010
        if t10 < total_time:
            sp += f".meas tran rpklev10_b{j} FIND V(rpeak_level) AT={t10:.7f}\n"
            sp += f".meas tran peakdet10_b{j} FIND V(peak_detected) AT={t10:.7f}\n"
            sp += f".meas tran trpkpls10_b{j} FIND V(true_rpeak_pulse) AT={t10:.7f}\n"

        # 4. Actual trigger times: when does true_rpeak_pulse cross 0.45V?
        t_search_start = max(0.001, t_rpk - 0.010)
        t_search_end = min(total_time - 0.001, t_rpk + 0.050)
        sp += f".meas tran trpk_time_b{j} WHEN V(true_rpeak_pulse)=0.45 RISE=1 TD={t_search_start:.7f}\n"
        sp += f".meas tran rpkdet_time_b{j} WHEN V(rpeak_level)=0.45 RISE=1 TD={t_search_start:.7f}\n"
        sp += f".meas tran peakdet_time_b{j} WHEN V(peak_detected)=0.45 RISE=1 TD={t_search_start:.7f}\n"

        # 5. Delay timer outputs: when do they fire?
        sp += f".meas tran dt_s14_time_b{j} WHEN V(dt_s14_lev)=0.45 RISE=1 TD={t_search_start:.7f}\n"
        sp += f".meas tran dt_wid_time_b{j} WHEN V(dt_wid_lev)=0.45 RISE=1 TD={t_search_start:.7f}\n"
        sp += f".meas tran dt_s69_time_b{j} WHEN V(dt_s69_lev)=0.45 RISE=1 TD={t_search_start:.7f}\n"
        sp += f".meas tran dt_bi_time_b{j} WHEN V(dt_bi_lev)=0.45 RISE=1 TD={t_search_start:.7f}\n"

        # 6. Feature values at annotation + 100ms (current measurement time)
        t100 = t_rpk + 0.100
        if t100 < total_time:
            sp += f".meas tran f3_b{j}_t100 FIND V(f3_bi_l0) AT={t100:.7f}\n"
            sp += f".meas tran f6_b{j}_t100 FIND V(f6_width) AT={t100:.7f}\n"
            sp += f".meas tran f7_b{j}_t100 FIND V(f7_slope) AT={t100:.7f}\n"
            sp += f".meas tran f8_b{j}_t100 FIND V(f8_symmetry) AT={t100:.7f}\n"

        # 7. Feature values at annotation + 200ms (later measurement)
        t200 = t_rpk + 0.200
        if t200 < total_time:
            sp += f".meas tran f3_b{j}_t200 FIND V(f3_bi_l0) AT={t200:.7f}\n"
            sp += f".meas tran f6_b{j}_t200 FIND V(f6_width) AT={t200:.7f}\n"
            sp += f".meas tran f7_b{j}_t200 FIND V(f7_slope) AT={t200:.7f}\n"
            sp += f".meas tran f8_b{j}_t200 FIND V(f8_symmetry) AT={t200:.7f}\n"

        # 8. Feature values at annotation + 400ms (well after BI trigger at +300ms)
        t400 = t_rpk + 0.400
        if t400 < total_time:
            sp += f".meas tran f3_b{j}_t400 FIND V(f3_bi_l0) AT={t400:.7f}\n"
            sp += f".meas tran f6_b{j}_t400 FIND V(f6_width) AT={t400:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_diag_analog.spice")
    log = tb.replace('.spice', '.log')
    print(f"\nWriting testbench: {os.path.basename(tb)}")
    print(f"Sim time: {total_time:.1f}s, {n} beats")
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)

    print("Running ngspice (may take 2-5 min)...")
    t0 = time.time()
    try:
        subprocess.run([NGSPICE, '-b', '-o', log, tb],
                       capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    except subprocess.TimeoutExpired:
        print("TIMEOUT!")
        return
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    if not os.path.exists(log):
        print("ERROR: no log file produced")
        return

    with open(log) as f:
        text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass

    # Check for ngspice errors
    errors = [l for l in text.split('\n') if 'Error' in l or 'error' in l.lower()]
    if errors:
        print(f"\nngspice errors found:")
        for e in errors[:5]:
            print(f"  {e.strip()}")

    # === ANALYSIS ===
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC RESULTS — Record {record_id}")
    print(f"{'='*80}")

    print(f"\n--- 1. TRIGGER CHAIN TIMING ---")
    print(f"{'Beat':>4s} {'Annot_t':>8s} {'ECG@R':>7s} {'rpk_det':>8s} {'peak_det':>8s} {'trpk_pls':>8s} {'rpk_lat':>8s}")
    for j in range(n):
        t_rpk = rpeak_times[j]
        ecg = results.get(f'ecg_b{j}', float('nan'))
        rpk_det_t = results.get(f'rpkdet_time_b{j}')
        peak_det_t = results.get(f'peakdet_time_b{j}')
        trpk_t = results.get(f'trpk_time_b{j}')
        lat = f"{(trpk_t - t_rpk)*1000:.1f}ms" if trpk_t else "NEVER"
        rpk_str = f"{rpk_det_t:.4f}" if rpk_det_t else "NEVER"
        pk_str = f"{peak_det_t:.4f}" if peak_det_t else "NEVER"
        trpk_str = f"{trpk_t:.4f}" if trpk_t else "NEVER"
        print(f"  {j:>2d}  {t_rpk:8.4f}  {ecg:7.4f}  {rpk_str:>8s}  {pk_str:>8s}  {trpk_str:>8s}  {lat:>8s}")

    print(f"\n--- 2. TRIGGER CHAIN SIGNALS AT ANNOTATION TIME ---")
    print(f"{'Beat':>4s} {'rpk_lev':>8s} {'rpk_held':>8s} {'peak_det':>8s} {'trpk_pls':>8s} {'peak_hold':>9s}")
    for j in range(n):
        rpklev = results.get(f'rpklev_b{j}', float('nan'))
        rpkheld = results.get(f'rpkheld_b{j}', float('nan'))
        peakdet = results.get(f'peakdet_b{j}', float('nan'))
        trpkpls = results.get(f'trpkpls_b{j}', float('nan'))
        peakhold = results.get(f'peakhold_b{j}', float('nan'))
        print(f"  {j:>2d}  {rpklev:8.4f}  {rpkheld:8.4f}  {peakdet:8.4f}  {trpkpls:8.4f}  {peakhold:9.4f}")

    print(f"\n--- 3. DELAY TIMER OUTPUT TIMES ---")
    print(f"{'Beat':>4s} {'trpk_pls':>8s} {'dt_s14':>8s} {'dt_wid':>8s} {'dt_s69':>8s} {'dt_bi':>8s} {'s14_lat':>8s} {'wid_lat':>8s}")
    for j in range(n):
        t_rpk = rpeak_times[j]
        trpk_t = results.get(f'trpk_time_b{j}')
        s14_t = results.get(f'dt_s14_time_b{j}')
        wid_t = results.get(f'dt_wid_time_b{j}')
        s69_t = results.get(f'dt_s69_time_b{j}')
        bi_t = results.get(f'dt_bi_time_b{j}')

        trpk_s = f"{trpk_t:.4f}" if trpk_t else "NEVER"
        s14_s = f"{s14_t:.4f}" if s14_t else "NEVER"
        wid_s = f"{wid_t:.4f}" if wid_t else "NEVER"
        s69_s = f"{s69_t:.4f}" if s69_t else "NEVER"
        bi_s = f"{bi_t:.4f}" if bi_t else "NEVER"

        # Latency from annotation R-peak
        s14_lat = f"{(s14_t - t_rpk)*1000:.1f}ms" if s14_t else "---"
        wid_lat = f"{(wid_t - t_rpk)*1000:.1f}ms" if wid_t else "---"

        print(f"  {j:>2d}  {trpk_s:>8s}  {s14_s:>8s}  {wid_s:>8s}  {s69_s:>8s}  {bi_s:>8s}  {s14_lat:>8s}  {wid_lat:>8s}")

    print(f"\n--- 4. FEATURE VALUES AT DIFFERENT MEASUREMENT TIMES ---")
    print(f"{'Beat':>4s} {'F3@100ms':>9s} {'F3@200ms':>9s} {'F3@400ms':>9s} {'F6@100ms':>9s} {'F6@200ms':>9s} {'F7@100ms':>9s} {'F7@200ms':>9s} {'F8@100ms':>9s} {'F8@200ms':>9s}")
    for j in range(n):
        f3_100 = results.get(f'f3_b{j}_t100', float('nan'))
        f3_200 = results.get(f'f3_b{j}_t200', float('nan'))
        f3_400 = results.get(f'f3_b{j}_t400', float('nan'))
        f6_100 = results.get(f'f6_b{j}_t100', float('nan'))
        f6_200 = results.get(f'f6_b{j}_t200', float('nan'))
        f7_100 = results.get(f'f7_b{j}_t100', float('nan'))
        f7_200 = results.get(f'f7_b{j}_t200', float('nan'))
        f8_100 = results.get(f'f8_b{j}_t100', float('nan'))
        f8_200 = results.get(f'f8_b{j}_t200', float('nan'))
        print(f"  {j:>2d}  {f3_100:9.4f}  {f3_200:9.4f}  {f3_400:9.4f}  {f6_100:9.4f}  {f6_200:9.4f}  {f7_100:9.4f}  {f7_200:9.4f}  {f8_100:9.4f}  {f8_200:9.4f}")

    # Save log for reference
    diag_log = os.path.join(FRONTEND_DIR, f"_diag_{record_id}_results.txt")
    with open(diag_log, 'w') as f:
        f.write(f"Diagnostic Results for Record {record_id}\n")
        f.write(f"Beats: {n}, Gain: {gain:.4f}, Threshold: {rpeak_threshold:.4f}\n")
        f.write(f"All results:\n")
        for k in sorted(results.keys()):
            f.write(f"  {k} = {results[k]:.6f}\n")
    print(f"\nAll measurements saved to {os.path.basename(diag_log)}")

    # Cleanup
    try: os.remove(tb)
    except: pass


if __name__ == '__main__':
    record_id = sys.argv[1] if len(sys.argv) > 1 else '100'
    run_diagnostic(record_id)
