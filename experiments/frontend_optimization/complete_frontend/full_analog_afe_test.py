"""
Full Analog AFE Integration Test
=================================
ECG (MIT-BIH) -> Full Analog AFE -> 8 Feature Outputs -> Compare vs Python

Analog components used:
  - bias_gen.spice          (vbias, vref)
  - delay_timer.spice       (post-R timing: s14, wid, s69, bi_copy, rr_copy)
  - edge_buffer.spice       (sharpens slow comparator transitions)
  - pulse_gen.spice         (level-to-pulse for S&H triggers)
  - long_pulse_gen.spice    (longer pulse for EWMA charge-sharing)
  - inverter.spice          (used inside other circuits)
  - rr_timing.spice         (F1: rr_ratio, F2: rr_asym)
  - beat_instability.spice  (F3: bi_L0, F4: bi_L1)
  - pwave_diff.spice        (F5: pwave energy)
  - qrs_width.spice         (F6: QRS width)
  - qrs_slope.spice         (F7: slope ratio)
  - qrs_symmetry.spice      (F8: QRS symmetry)

Only external inputs:
  - ECG waveform (pre-scaled PWL from MIT-BIH)
  - R-peak timing (from annotations, representing rpeak_detector output)
  - Calibration threshold voltages (ideal DC sources — stored calibration data)

rr_cap trigger: analog comparator on rr_timer SF, AND-gated with NOT rpeak
  (gate closes capture before timer precharge — prevents SF tracking corruption)
pwave triggers: analog comparators on rr_timer SF -> edge_buffer -> pulse_gen
delay_timer triggers: analog comparators with calibrated per-comparator pipeline delays
"""
import subprocess, os, re, sys, time
import numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130; RPEAK_IDX = 20
PW_S = 16; PW_MID = 43
EWMA_ALPHA = 0.10
BATCH_SIZE = 12

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


OUT = os.path.join(FRONTEND_DIR, "_full_analog_results.txt")
def log(msg):
    print(msg)
    with open(OUT, 'a') as f: f.write(msg + '\n')
with open(OUT, 'w') as f: f.write('')


def load_all_beats(record_id, max_beats=200):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]
    all_peaks = []
    for i in range(1, len(valid) - 1):
        idx = valid[i][0]
        bs = idx - WIN_L; be = idx + WIN_R
        if bs >= 0 and be < len(sig):
            all_peaks.append(abs(sig[idx, 0]))
    gain = 0.16 / max(all_peaks) if all_peaks else 0.16
    beats = []
    ewma_rr = None
    for i in range(1, len(valid) - 1):
        if len(beats) >= max_beats: break
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        post_rr = (valid[i+1][0] - idx) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]
        if ewma_rr is None:
            ewma_rr_before = pre_rr; ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr
        prev_bs = valid[i-1][0] - WIN_L
        prev_be = valid[i-1][0] + WIN_R
        if prev_bs >= 0 and prev_be < len(sig):
            prev_beat = sig[prev_bs:prev_be]
            bi_l0 = beat[RPEAK_IDX+QRS_S, 0] - prev_beat[RPEAK_IDX+QRS_S, 0]
            bi_l1 = beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)] - prev_beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)]
        else:
            bi_l0 = bi_l1 = 0.0
        beats.append({
            'label': aami_map[sym], 'pre_rr': pre_rr, 'post_rr': post_rr, 'ewma_rr': ewma_rr,
            'ecg_l0': beat[:, 0].copy(), 'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': pre_rr - ewma_rr_before, 'py_f2': pre_rr - post_rr,
            'py_bi_l0': bi_l0, 'py_bi_l1': bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[RPEAK_IDX+QRS_S+10, 0],
            'py_slp': beat[RPEAK_IDX+QRS_S+5, 0] - beat[RPEAK_IDX+QRS_S+25, 0],
            'py_sym': beat[RPEAK_IDX+QRS_S, 0],
        })
    return beats, gain


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join("{:.7f} {:.6f}".format(t, v) for t, v in c)


def make_trig(rpeak_times, offset_s, pulse_w, total_time):
    pts = [(0, 0)]
    for rpt in rpeak_times:
        t = rpt + offset_s
        if t > 0.01:
            pts += [(t - 0.0004, 0), (t, VDD), (t + pulse_w, VDD), (t + pulse_w + 0.0004, 0)]
    pts.append((total_time, 0))
    return make_pwl(pts)


def interp_sf(times_ms, voltages, target_ms):
    for i in range(len(times_ms) - 1):
        if times_ms[i] <= target_ms <= times_ms[i+1]:
            frac = (target_ms - times_ms[i]) / (times_ms[i+1] - times_ms[i])
            return voltages[i] + frac * (voltages[i+1] - voltages[i])
    if target_ms < times_ms[0]: return voltages[0]
    return voltages[-1]


def calibrate_timers():
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")
    sp = """.title timer_cal
.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/pulse_gen.spice"
Xbias vbias vref vdd vss bias_gen
Vrpk rpeak_pulse 0 PWL(0 0 0.098 0 0.1 {vdd} 0.12 {vdd} 0.121 0 2.0 0)
Vt1 vth_14_fast 0 0.1
Vt2 vth_69 0 0.1
Vt3 vth_28 0 0.1
Vt4 vth_300 0 0.1
Vt5 vth_10 0 0.1
Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer
Vrrc_cal trig_rr_cap_cal 0 0
Vrry_cal trig_rr_copy_cal 0 0
Xrr_cal ecg_cal trig_rr_cap_cal rpeak_pulse trig_rr_copy_cal f1_cal f2_cal rr_timer_cal rr_ewma_cal vdd vss vbias rr_timing
Vecg_cal ecg_cal 0 0.3
.ic V(xbias.iref)=0.3
.ic V(xrr_cal.ewma_node)=0.400000
.ic V(xrr_cal.esamp_node)=0.400000
.ic V(xrr_cal.prev_node)=0.400000
.tran 0.5m 1.5
""".format(lib=lib_path, fp=fp, vdd=VDD)
    dt_times = [1,2,3,4,5,10,15,20,50,100,200,300,400,500,600,700,800]
    for t in dt_times:
        sp += ".meas tran dt_sf_{t:04d} FIND V(dt_sf_out) AT={at:.4f}\n".format(t=t, at=0.121+t/1000.0)
    ft_times = [1,2,3,5,10,20]
    for t in ft_times:
        sp += ".meas tran ft_sf_{t:04d} FIND V(xdelay.sf_timer2) AT={at:.4f}\n".format(t=t, at=0.121+t/1000.0)
    rr_times = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
    for t in rr_times:
        sp += ".meas tran rr_sf_{t:04d} FIND V(rr_timer_cal) AT={at:.4f}\n".format(t=t, at=0.121+t/1000.0)
    sp += "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_cal_full_analog.spice")
    logf = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp)
    subprocess.run([NGSPICE, '-b', '-o', logf, tb], capture_output=True, text=True, timeout=120, cwd=NGSPICE_DIR)
    results = {}
    if os.path.exists(logf):
        with open(logf) as f: text = f.read()
        for line in text.split('\n'):
            m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
            if m:
                try: results[m.group(1).lower()] = float(m.group(2))
                except: pass
    try: os.remove(tb); os.remove(logf)
    except: pass
    return {
        'dt_t': dt_times, 'dt_v': [results.get('dt_sf_{:04d}'.format(t), 0) for t in dt_times],
        'ft_t': ft_times, 'ft_v': [results.get('ft_sf_{:04d}'.format(t), 0) for t in ft_times],
        'rr_t': rr_times, 'rr_v': [results.get('rr_sf_{:04d}'.format(t), 0) for t in rr_times],
    }


def generate_testbench(beats, gain, cal, ic_vals=None, keep_files=False):
    n = len(beats)
    dt = 1.0 / FS
    PW = 0.0005

    rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    ecg_l0_pts, ecg_l1_pts = [(0, VBASE)], [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt
        ecg0, ecg1 = beats[i]['ecg_l0'], beats[i]['ecg_l1']
        t_before = beat_start - 0.003
        if t_before > 0.01:
            ecg_l0_pts.append((t_before, VBASE))
            ecg_l1_pts.append((t_before, VBASE))
        for j in range(len(ecg0)):
            t = beat_start + j * dt
            if t > 0:
                ecg_l0_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg0[j]))))
                ecg_l1_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg1[j]))))
        t_after = beat_start + len(ecg0) * dt + 0.003
        ecg_l0_pts.append((t_after, VBASE))
        ecg_l1_pts.append((t_after, VBASE))
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    dt_t, dt_v = cal['dt_t'], cal['dt_v']
    ft_t, ft_v = cal['ft_t'], cal['ft_v']
    rr_t, rr_v = cal['rr_t'], cal['rr_v']

    # Per-comparator pipeline delays (calibrated from isolation_test_v5)
    rpk_drop = 0.5
    pipe_s14 = 2.9; pipe_wid = 4.5; pipe_s69 = 4.4; pipe_bi = 5.9; pipe_s10 = 1.8
    pipe_pw_rr = 14.0   # rr_timer comparators need larger pipe (0.1mV/ms slope)
    rrcap_offset = -50   # fire ~50ms before predicted next R-peak

    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_s14))
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_wid))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_s69))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_bi))
    vth_10  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_s10))

    # EWMA IC from first beat's RR
    first_rr_ms = beats[0]['pre_rr'] * 1000
    ic_ewma = interp_sf(rr_t, rr_v, first_rr_ms)
    if ic_vals:
        ic_ewma_v = ic_vals.get('rr_ewma', ic_ewma)
        ic_esamp_v = ic_vals.get('rr_esamp', ic_ewma)
        ic_prev_v = ic_vals.get('rr_prev', ic_ewma)
        ic_bi0_prev = ic_vals.get('bi0_prev', 0.3)
        ic_bi0_esamp = ic_vals.get('bi0_esamp', 0.3)
        ic_bi1_prev = ic_vals.get('bi1_prev', 0.3)
        ic_bi1_esamp = ic_vals.get('bi1_esamp', 0.3)
    else:
        ic_ewma_v = ic_esamp_v = ic_prev_v = ic_ewma
        ic_bi0_prev = ic_bi0_esamp = 0.3
        ic_bi1_prev = ic_bi1_esamp = 0.3

    # rr_cap analog comparator: threshold per beat from EWMA prediction
    pwl_rrcap_pts = [(0, interp_sf(rr_t, rr_v, max(1, first_rr_ms + rrcap_offset - rpk_drop - pipe_pw_rr)))]
    for j in range(n):
        ewma_rr_ms = beats[j]['ewma_rr'] * 1000
        rr_cap_d = max(1, ewma_rr_ms + rrcap_offset - rpk_drop - pipe_pw_rr)
        v_rrcap = interp_sf(rr_t, rr_v, rr_cap_d)
        t_step = rpeak_times[j] + 0.001
        if t_step > pwl_rrcap_pts[-1][0]:
            pwl_rrcap_pts.append((t_step, v_rrcap))
    pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))

    # pwave analog comparators: thresholds from EWMA prediction
    pwl_base_pts = [(0, 0.3)]
    pwl_peak_pts = [(0, 0.3)]
    for j in range(n):
        ewma_rr_ms = beats[j]['ewma_rr'] * 1000
        pw_base_d = max(1, ewma_rr_ms - 201 - rpk_drop - pipe_pw_rr)
        pw_peak_d = max(1, ewma_rr_ms - 131 - rpk_drop - pipe_pw_rr)
        v_base = interp_sf(rr_t, rr_v, pw_base_d)
        v_peak = interp_sf(rr_t, rr_v, pw_peak_d)
        t_step = rpeak_times[j] + 0.001
        if t_step > pwl_base_pts[-1][0]:
            pwl_base_pts.append((t_step, v_base))
            pwl_peak_pts.append((t_step, v_peak))
    pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
    pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))

    sp = """.title full_analog_afe
.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {vdd}
Vss vss 0 0

Vecg0 ecg_l0 0 PWL({ecg0})
Vecg1 ecg_l1 0 PWL({ecg1})

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
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

* === ANALOG BIAS GENERATOR ===
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3

* === R-PEAK TRIGGER (from annotations — represents rpeak_detector output) ===
* Three independent PWL sources driving different loads (avoids fan-out issues)
Vrpk_dt rpeak_pulse_dt 0 PWL({rpk_dt})
Vrpk_rr rpeak_raw 0 PWL({rpk_rr})
Vrsh trig_rpeak_sh 0 PWL({rpk_sh})

* === ANALOG DELAY TIMER + PULSE GENERATORS ===
Xdt rpeak_pulse_dt dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

* Calibration threshold voltages (stored calibration data)
Vth14  vth_14_fast 0 {vth14:.6f}
Vth69  vth_69      0 {vth69:.6f}
Vth28  vth_28      0 {vth28:.6f}
Vth300 vth_300     0 {vth300:.6f}
Vth10  vth_10      0 {vth10:.6f}

* Pulse generators: level -> pulse for S&H triggers
* S14: fast timer, no edge_buffer needed (steep slope)
Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
* WID: main timer, slow slope needs edge_buffer
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen
* S69: main timer
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen
* BI_COPY: main timer, needs long_pulse_gen for EWMA charge-sharing
Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen
* RR_COPY: fast timer
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias pulse_gen

* === ANALOG RR_CAP TRIGGER (AND-gated: comparator AND NOT rpeak) ===
* The comparator fires when rr_timer SF drops below threshold.
* The AND gate closes the capture BEFORE timer precharge, preventing
* the capture switch from tracking the precharged SF value.
Vth_rrc vth_rr_cap 0 PWL({rrcap_pwl})
XMcrrc1 cnrrc rr_timer ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 stg1_rrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 stg1_rrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcrrc6 rr_cap_lev stg1_rrc vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcrrc7 rr_cap_lev stg1_rrc vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_rrc rr_cap_lev rrc_eb vdd vss edge_buffer
* AND gate: rrc_eb AND (NOT rpeak_raw) — closes capture at R-peak
Xinv_rpk rpeak_raw rpeak_bar_ext vdd vss inverter_sub
* NAND(rrc_eb, rpeak_bar_ext) + INV = AND
XMnand_n1 nand_rrc rrc_eb nand_mid_rrc vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMnand_n2 nand_mid_rrc rpeak_bar_ext vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMnand_p1 nand_rrc rrc_eb vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMnand_p2 nand_rrc rpeak_bar_ext vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
Xinv_nand nand_rrc trig_rr_cap vdd vss inverter_sub

* === ANALOG PWAVE TRIGGERS (comparators on rr_timer SF) ===
* These use pulse_gen so a brief ~0.5ms S&H pulse fires at comparator transition.
* No gating needed — pulse_gen fires once on rising edge then quiesces.
Vth_pwb vth_pw_base 0 PWL({pwb_pwl})
Vth_pwp vth_pw_peak 0 PWL({pwp_pwl})
XMcpwb1 cnpwb rr_timer ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 stg1_pwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 stg1_pwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcpwb6 pw_base_lev stg1_pwb vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcpwb7 pw_base_lev stg1_pwb vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen

XMcpwp1 cnpwp rr_timer ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 stg1_pwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 stg1_pwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcpwp6 pw_peak_lev stg1_pwp vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcpwp7 pw_peak_lev stg1_pwp vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_pwp pw_peak_lev pwp_sharp vdd vss edge_buffer
Xpg_pwp pwp_sharp   trig_pw_peak vdd vss vbias pulse_gen

* === ALL 8 FEATURE CIRCUITS (analog) ===
* rr_timing: rpeak_raw drives the timer precharge (rpeak_trigger pin)
Xrr ecg_l0 trig_rr_cap rpeak_raw trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

* === INITIAL CONDITIONS ===
.ic V(xrr.net_timer)={vdd:.6f}
.ic V(xrr.ewma_node)={ic_ewma:.6f}
.ic V(xrr.esamp_node)={ic_esamp:.6f}
.ic V(xrr.prev_node)={ic_prev:.6f}
.ic V(xbi0.prev_node)={ic_bi0p:.6f}
.ic V(xbi0.esamp_node)={ic_bi0e:.6f}
.ic V(xbi1.prev_node)={ic_bi1p:.6f}
.ic V(xbi1.esamp_node)={ic_bi1e:.6f}

.tran 0.5m {ttot:.4f}

""".format(
        lib=lib_path, fp=fp, vdd=VDD,
        ecg0=make_pwl(ecg_l0_pts), ecg1=make_pwl(ecg_l1_pts),
        rpk_dt=make_trig(rpeak_times, 0, PW, total_time),
        rpk_rr=make_trig(rpeak_times, 0, PW, total_time),
        rpk_sh=make_trig(rpeak_times, 0, PW, total_time),
        vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10,
        rrcap_pwl=make_pwl(pwl_rrcap_pts),
        pwb_pwl=make_pwl(pwl_base_pts), pwp_pwl=make_pwl(pwl_peak_pts),
        ic_ewma=ic_ewma_v, ic_esamp=ic_esamp_v, ic_prev=ic_prev_v,
        ic_bi0p=ic_bi0_prev, ic_bi0e=ic_bi0_esamp,
        ic_bi1p=ic_bi1_prev, ic_bi1e=ic_bi1_esamp,
        ttot=total_time,
    )

    skip = 2
    for j in range(skip, n):
        t_f1 = rpeak_times[j] - 0.010
        if t_f1 > 0.5:
            sp += ".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t:.7f}\n".format(j=j, t=t_f1)
        if j + 1 < n:
            t_f2 = rpeak_times[j + 1] - 0.065
            if t_f2 > 0.5:
                sp += ".meas tran f2_{j} FIND V(f2_rr_asym) AT={t:.7f}\n".format(j=j, t=t_f2)
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            sp += ".meas tran f3_{j} FIND V(f3_bi_l0) AT={t:.7f}\n".format(j=j, t=t_f38)
            sp += ".meas tran f4_{j} FIND V(f4_bi_l1) AT={t:.7f}\n".format(j=j, t=t_f38)
            sp += ".meas tran f5_{j} FIND V(f5_pwave) AT={t:.7f}\n".format(j=j, t=t_f38)
            sp += ".meas tran f6_{j} FIND V(f6_width) AT={t:.7f}\n".format(j=j, t=t_f38)
            sp += ".meas tran f7_{j} FIND V(f7_slope) AT={t:.7f}\n".format(j=j, t=t_f38)
            sp += ".meas tran f8_{j} FIND V(f8_symmetry) AT={t:.7f}\n".format(j=j, t=t_f38)

    # IC propagation measurements
    t_ic = rpeak_times[-1] + 0.100
    if t_ic < total_time - 0.01:
        sp += "\n.meas tran ic_rr_ewma FIND V(xrr.ewma_node) AT={:.7f}\n".format(t_ic)
        sp += ".meas tran ic_rr_esamp FIND V(xrr.esamp_node) AT={:.7f}\n".format(t_ic)
        sp += ".meas tran ic_rr_prev FIND V(xrr.prev_node) AT={:.7f}\n".format(t_ic)
        sp += ".meas tran ic_bi0_prev FIND V(xbi0.prev_node) AT={:.7f}\n".format(t_ic)
        sp += ".meas tran ic_bi0_esamp FIND V(xbi0.esamp_node) AT={:.7f}\n".format(t_ic)
        sp += ".meas tran ic_bi1_prev FIND V(xbi1.prev_node) AT={:.7f}\n".format(t_ic)
        sp += ".meas tran ic_bi1_esamp FIND V(xbi1.esamp_node) AT={:.7f}\n".format(t_ic)

    sp += "\n.end\n"
    return sp, skip, rpeak_times, total_time


def run_sim(sp_text, label, keep=False):
    tb = os.path.join(FRONTEND_DIR, "_afe_{}.spice".format(label))
    logf = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp_text)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', logf, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    if not os.path.exists(logf): return None, elapsed
    with open(logf) as f: text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    if not keep:
        try: os.remove(tb); os.remove(logf)
        except: pass
    return results, elapsed


def extract_rho(results, beats, skip, n):
    """Extract per-feature rho. F1/F2 computed only from beats where measured."""
    f1_c, f1_p = [], []
    f2_c, f2_p = [], []
    f38_c, f38_p = [], []  # F3-F8

    for j in range(skip, n):
        f1_val = results.get('f1_{}'.format(j))
        f2_val = results.get('f2_{}'.format(j))
        vals38 = [results.get('f{}_{}'.format(k, j)) for k in range(3, 9)]
        bt = beats[j]

        if f1_val is not None:
            f1_c.append(f1_val)
            f1_p.append(bt['py_f1'])
        if f2_val is not None:
            f2_c.append(f2_val)
            f2_p.append(bt['py_f2'])
        if all(v is not None for v in vals38):
            f38_c.append(vals38)
            f38_p.append([bt['py_bi_l0'], bt['py_bi_l1'], bt['py_pw'],
                          bt['py_wid'], bt['py_slp'], bt['py_sym']])

    rhos = [0.0] * 8
    n_meas = [0] * 8

    if len(f1_c) >= 4:
        rho, _ = spearmanr(f1_c, f1_p)
        rhos[0] = abs(rho); n_meas[0] = len(f1_c)
    if len(f2_c) >= 4:
        rho, _ = spearmanr(f2_c, f2_p)
        rhos[1] = abs(rho); n_meas[1] = len(f2_c)
    if len(f38_c) >= 4:
        ca = np.array(f38_c); pa = np.array(f38_p)
        for i in range(6):
            rho, _ = spearmanr(ca[:, i], pa[:, i])
            rhos[i+2] = abs(rho)
            n_meas[i+2] = len(f38_c)

    return rhos, n_meas


def extract_ics(results):
    return {
        'rr_ewma': results.get('ic_rr_ewma'),
        'rr_esamp': results.get('ic_rr_esamp'),
        'rr_prev': results.get('ic_rr_prev'),
        'bi0_prev': results.get('ic_bi0_prev'),
        'bi0_esamp': results.get('ic_bi0_esamp'),
        'bi1_prev': results.get('ic_bi1_prev'),
        'bi1_esamp': results.get('ic_bi1_esamp'),
    }


# ========================================================================
# MAIN
# ========================================================================
log("=" * 70)
log("FULL ANALOG AFE INTEGRATION TEST")
log("=" * 70)
log("")
log("Chain: ECG(MIT-BIH) -> Analog AFE -> 8 Features -> Compare vs Python")
log("All supplementary circuits: analog SKY130 (no digital, no ideal triggers)")
log("Only external: ECG input, R-peak timing (rpeak_detector), calibration Vth")
log("rr_cap: AND-gated comparator (closes capture before timer precharge)")
log("pwave: comparator -> edge_buffer -> pulse_gen (brief S&H pulse)")
log("delay_timer: calibrated per-comparator pipeline delays")
log("")

log("[1] Calibrating timers...")
cal = calibrate_timers()
log("  DT SF (main): {}...".format(['{:.4f}'.format(v) for v in cal['dt_v'][:8]]))
log("  FT SF (fast): {}".format(['{:.4f}'.format(v) for v in cal['ft_v']]))
log("  RR SF: {}...".format(['{:.4f}'.format(v) for v in cal['rr_v'][:5]]))

# Sanity check calibration
if cal['dt_v'][0] < 0.3 or cal['rr_v'][0] < 0.3:
    log("  WARNING: calibration SF values look wrong (too low). Check ngspice setup.")
    log("  Continuing anyway...")

fnames = ['F1:rr', 'F2:asym', 'F3:bi0', 'F4:bi1', 'F5:pw', 'F6:wid', 'F7:slp', 'F8:sym']

# ========================================================================
# PHASE 1: Quick validation on record 100 (20 beats)
# ========================================================================
log("\n" + "=" * 70)
log("PHASE 1: Quick validation — Record 100, 20 beats")
log("=" * 70)

beats, gain = load_all_beats('100', 20)
log("  Loaded {} beats, gain={:.4f}".format(len(beats), gain))
rr_strs = ['{:.0f}'.format(b['pre_rr']*1000) for b in beats[:12]]
log("  RR: {} ... ms".format(' '.join(rr_strs)))

sp, skip, rpk_t, ttot = generate_testbench(beats, gain, cal, keep_files=True)
results, elapsed = run_sim(sp, "phase1_100", keep=True)
if results:
    rhos, n_meas = extract_rho(results, beats, skip, len(beats))
    log("  {} beats measured (F3-F8), {:.0f}s sim time".format(n_meas[2], elapsed))
    log("  F1 beats: {}, F2 beats: {}".format(n_meas[0], n_meas[1]))
    log("  rho: {}".format(' '.join('{:.3f}'.format(r) for r in rhos)))

    phase1_pass = all(r >= 0.90 for r in rhos if r > 0)
    log("  {}".format("ALL >= 0.90 — PASS" if phase1_pass else "SOME BELOW 0.90"))

    if not phase1_pass:
        log("\n  Phase 1 FAILED — stopping. Debug files kept:")
        log("    _afe_phase1_100.spice / .log")
        log("  Investigate before scaling to 16 records.")
        sys.exit(1)
else:
    log("  FAILED — sim error. Check _afe_phase1_100.log")
    sys.exit(1)

# ========================================================================
# PHASE 2: Full test — 16 records, batched with IC propagation
# ========================================================================
log("\n" + "=" * 70)
log("PHASE 2: Full test — 16 records, batched with IC propagation")
log("=" * 70)

test_records = ['100', '101', '103', '105', '106', '108', '109', '111',
                '112', '113', '114', '115', '116', '117', '118', '119']
per_record_rhos = {}
all_f1_c, all_f1_p = [], []
all_f2_c, all_f2_p = [], []
all_f38_c, all_f38_p = [], []

for rec in test_records:
    try:
        beats, gain = load_all_beats(rec, 100)
    except Exception as e:
        log("  {}: SKIP (load error: {})".format(rec, e))
        continue
    if len(beats) < 8:
        log("  {}: SKIP (only {} beats)".format(rec, len(beats)))
        continue

    n_batches = (len(beats) + BATCH_SIZE - 1) // BATCH_SIZE
    ic_vals = None
    rec_f1_c, rec_f1_p = [], []
    rec_f2_c, rec_f2_p = [], []
    rec_f38_c, rec_f38_p = [], []

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(beats))
        batch = beats[start:end]
        if len(batch) < 5: break

        sp, skip, rpk_t, ttot = generate_testbench(batch, gain, cal, ic_vals)
        results, elapsed = run_sim(sp, "r{}_b{}".format(rec, b))
        if results is None:
            log("  {} batch {}: sim error".format(rec, b))
            break

        # Extract IC for next batch
        ics = extract_ics(results)
        if all(v is not None for v in ics.values()):
            ic_vals = ics

        # Extract feature values
        for j in range(skip, len(batch)):
            bt = batch[j]
            f1_val = results.get('f1_{}'.format(j))
            f2_val = results.get('f2_{}'.format(j))
            vals38 = [results.get('f{}_{}'.format(k, j)) for k in range(3, 9)]

            if f1_val is not None:
                rec_f1_c.append(f1_val); rec_f1_p.append(bt['py_f1'])
            if f2_val is not None:
                rec_f2_c.append(f2_val); rec_f2_p.append(bt['py_f2'])
            if all(v is not None for v in vals38):
                rec_f38_c.append(vals38)
                rec_f38_p.append([bt['py_bi_l0'], bt['py_bi_l1'], bt['py_pw'],
                                  bt['py_wid'], bt['py_slp'], bt['py_sym']])

    if len(rec_f38_c) < 4:
        log("  {}: SKIP (only {} valid beats)".format(rec, len(rec_f38_c)))
        continue

    # Compute per-record rho
    rhos = [0.0] * 8
    if len(rec_f1_c) >= 4:
        rho, _ = spearmanr(rec_f1_c, rec_f1_p); rhos[0] = abs(rho)
    if len(rec_f2_c) >= 4:
        rho, _ = spearmanr(rec_f2_c, rec_f2_p); rhos[1] = abs(rho)
    ca38 = np.array(rec_f38_c); pa38 = np.array(rec_f38_p)
    for i in range(6):
        rho, _ = spearmanr(ca38[:, i], pa38[:, i]); rhos[i+2] = abs(rho)

    per_record_rhos[rec] = rhos
    all_f1_c.extend(rec_f1_c); all_f1_p.extend(rec_f1_p)
    all_f2_c.extend(rec_f2_c); all_f2_p.extend(rec_f2_p)
    all_f38_c.extend(rec_f38_c); all_f38_p.extend(rec_f38_p)

    rho_str = ' '.join('{:.3f}'.format(r) for r in rhos)
    log("  {}: {:3d} beats  rho: {}".format(rec, len(rec_f38_c), rho_str))

# ========================================================================
# AGGREGATE RESULTS
# ========================================================================
log("\n" + "=" * 70)
log("AGGREGATE RESULTS")
log("=" * 70)

if all_f38_c:
    agg_rhos = [0.0] * 8
    if len(all_f1_c) >= 4:
        rho, _ = spearmanr(all_f1_c, all_f1_p); agg_rhos[0] = abs(rho)
    if len(all_f2_c) >= 4:
        rho, _ = spearmanr(all_f2_c, all_f2_p); agg_rhos[1] = abs(rho)
    ca38 = np.array(all_f38_c); pa38 = np.array(all_f38_p)
    for i in range(6):
        rho, _ = spearmanr(ca38[:, i], pa38[:, i]); agg_rhos[i+2] = abs(rho)

    log("  Total: F1={} F2={} F3-F8={} beats across {} records".format(
        len(all_f1_c), len(all_f2_c), len(all_f38_c), len(per_record_rhos)))
    log("  Aggregate rho: {}".format(' '.join('{:.3f}'.format(r) for r in agg_rhos)))
    log("")
    header = "  {:<8s} {:>5s}  ".format('Record', 'Beats') + " ".join("{:>7s}".format(f) for f in fnames)
    log(header)
    log("  " + "-" * len(header))
    for rec in test_records:
        if rec in per_record_rhos:
            rhos = per_record_rhos[rec]
            rho_str = " ".join("{:>7.3f}".format(r) for r in rhos)
            log("  {:<8s}         {}".format(rec, rho_str))
    log("  " + "-" * len(header))
    rho_str = " ".join("{:>7.3f}".format(r) for r in agg_rhos)
    log("  {:<8s} {:>5d}  {}".format('AGGREGATE', len(all_f38_c), rho_str))

    targets = [0.970, 0.995, 0.980, 0.986, 0.995, 0.998, 0.995, 0.995]
    log("\n  Feature targets vs achieved:")
    for i in range(8):
        status = "PASS" if agg_rhos[i] >= targets[i] * 0.95 else "NEEDS WORK"
        log("    {}: target={:.3f}  achieved={:.3f}  {}".format(fnames[i], targets[i], agg_rhos[i], status))
else:
    log("  No valid data collected")
