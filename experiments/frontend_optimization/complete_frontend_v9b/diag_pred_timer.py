"""
Diagnostic: test a dedicated prediction timer for pwave/rr_cap.

The rr_timer (320pF, ~32pA → 0.11 mV/ms) is too flat for reliable comparator
timing at 600-800ms. A prediction timer (100pF, ~32pA → 0.32 mV/ms, 3.2x steeper)
reduces jitter proportionally.

Configs:
  A: ideal rr_cap + ideal pwave + analog DT (baseline)
  D: ideal rr_cap + prediction-timer pwave + analog DT
  E: prediction-timer rr_cap (AND-gated) + prediction-timer pwave + analog DT
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

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


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
* PREDICTION TIMER: 100pF MIM, same discharge as rr_timer (L=8 W=0.42)
* Slope: ~32pA / 100pF = 0.32 mV/ms (3.2x steeper than rr_timer)
Xinv_rpk_cal rpeak_pulse rpk_bar_cal vdd vss inverter_sub
XMpt_chg pt_cap rpk_bar_cal vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMpt_dis pt_cap vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XCpt pt_cap vss sky130_fd_pr__cap_mim_m3_1 W=224 L=224
XMpt_sf vdd pt_cap pt_sf_out vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMpt_sfb pt_sf_out vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
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
    pt_times = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
    for t in pt_times:
        sp += ".meas tran pt_sf_{t:04d} FIND V(pt_sf_out) AT={at:.4f}\n".format(t=t, at=0.121+t/1000.0)
    sp += "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_cal_diag.spice")
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
        'pt_t': pt_times, 'pt_v': [results.get('pt_sf_{:04d}'.format(t), 0) for t in pt_times],
    }


def gen_tb(beats, gain, cal, mode):
    n = len(beats)
    dt_step = 1.0 / FS
    PW = 0.0005

    rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    ecg_l0_pts, ecg_l1_pts = [(0, VBASE)], [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt_step
        ecg0 = beats[i]['ecg_l0']; ecg1 = beats[i]['ecg_l1']
        t_before = beat_start - 0.003
        if t_before > 0.01:
            ecg_l0_pts.append((t_before, VBASE)); ecg_l1_pts.append((t_before, VBASE))
        for j in range(len(ecg0)):
            t = beat_start + j * dt_step
            if t > 0:
                ecg_l0_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg0[j]))))
                ecg_l1_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg1[j]))))
        t_after = beat_start + len(ecg0) * dt_step + 0.003
        ecg_l0_pts.append((t_after, VBASE)); ecg_l1_pts.append((t_after, VBASE))
    ecg_l0_pts.append((total_time, VBASE)); ecg_l1_pts.append((total_time, VBASE))

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")
    dt_t, dt_v = cal['dt_t'], cal['dt_v']
    ft_t, ft_v = cal['ft_t'], cal['ft_v']
    rr_t, rr_v = cal['rr_t'], cal['rr_v']
    pt_t, pt_v = cal['pt_t'], cal['pt_v']

    rpk_drop = 0.5
    pipe_s14 = 2.9; pipe_wid = 4.5; pipe_s69 = 4.4; pipe_bi = 5.9; pipe_s10 = 1.8

    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_s14))
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_wid))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_s69))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_bi))
    vth_10  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_s10))

    first_rr_ms = beats[0]['pre_rr'] * 1000
    ic_ewma = interp_sf(rr_t, rr_v, first_rr_ms)

    sp = """.title diag_{mode}
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
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3
""".format(mode=mode, lib=lib_path, vdd=VDD, fp=fp,
           ecg0=make_pwl(ecg_l0_pts), ecg1=make_pwl(ecg_l1_pts))

    # R-peak triggers (always ideal)
    sp += "Vrpk_dt rpeak_pulse_dt 0 PWL({rpk})\n".format(rpk=make_trig(rpeak_times, 0, PW, total_time))
    sp += "Vrpk_rr rpeak_raw 0 PWL({rpk})\n".format(rpk=make_trig(rpeak_times, 0, PW, total_time))
    sp += "Vrsh trig_rpeak_sh 0 PWL({rpk})\n".format(rpk=make_trig(rpeak_times, 0, PW, total_time))

    # Analog delay_timer (always analog)
    sp += """
Xdt rpeak_pulse_dt dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer
Vth14  vth_14_fast 0 {v14:.6f}
Vth69  vth_69      0 {v69:.6f}
Vth28  vth_28      0 {v28:.6f}
Vth300 vth_300     0 {v300:.6f}
Vth10  vth_10      0 {v10:.6f}
Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen
Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias pulse_gen
""".format(v14=vth_14, v69=vth_69, v28=vth_28, v300=vth_300, v10=vth_10)

    use_pred_timer = mode in ('D_pred_pwave', 'E_pred_both')
    use_analog_rrcap = mode in ('E_pred_both',)

    # Prediction timer pipeline — steeper slope means less pipeline needed
    # PT slope is ~0.32 mV/ms vs RR 0.11 mV/ms → pipeline ~4-5ms
    pipe_pw_pt = 5.0

    if use_pred_timer:
        # Inline prediction timer: 100pF MIM (W=224 L=224), same discharge NFET as rr_timer
        sp += """
* PREDICTION TIMER: 100pF MIM, NFET L=8 W=0.42 (~32pA), 0.32 mV/ms
Xinv_rpk_pt rpeak_raw rpk_bar_pt vdd vss inverter_sub
XMpt_chg pt_cap rpk_bar_pt vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMpt_dis pt_cap vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XCpt pt_cap vss sky130_fd_pr__cap_mim_m3_1 W=224 L=224
XMpt_sf vdd pt_cap pt_sf_out vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMpt_sfb pt_sf_out vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
"""
        # Pwave comparators on prediction timer
        pwl_base_pts = [(0, 0.3)]
        pwl_peak_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            pw_base_d = max(1, ewma_rr_ms - 206 - rpk_drop - pipe_pw_pt)
            pw_peak_d = max(1, ewma_rr_ms - 131 - rpk_drop - pipe_pw_pt)
            v_base = interp_sf(pt_t, pt_v, pw_base_d)
            v_peak = interp_sf(pt_t, pt_v, pw_peak_d)
            t_step = rpeak_times[j] + 0.001
            if t_step > pwl_base_pts[-1][0]:
                pwl_base_pts.append((t_step, v_base))
                pwl_peak_pts.append((t_step, v_peak))
        pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
        pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
        sp += """
* PWAVE COMPARATORS ON PREDICTION TIMER
Vth_pwb vth_pw_base 0 PWL({pwb})
Vth_pwp vth_pw_peak 0 PWL({pwp})
XMcpwb1 cnpwb pt_sf_out ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 stg1_pwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 stg1_pwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcpwb6 pw_base_lev stg1_pwb vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcpwb7 pw_base_lev stg1_pwb vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen
XMcpwp1 cnpwp pt_sf_out ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 stg1_pwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 stg1_pwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcpwp6 pw_peak_lev stg1_pwp vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcpwp7 pw_peak_lev stg1_pwp vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_pwp pw_peak_lev pwp_sharp vdd vss edge_buffer
Xpg_pwp pwp_sharp   trig_pw_peak vdd vss vbias pulse_gen
""".format(pwb=make_pwl(pwl_base_pts), pwp=make_pwl(pwl_peak_pts))
    else:
        # Ideal pwave triggers
        sp += "Vpwb trig_pw_base 0 PWL({pwb})\n".format(
            pwb=make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time))
        sp += "Vpwp trig_pw_peak 0 PWL({pwp})\n".format(
            pwp=make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time))

    # RR_CAP trigger
    if use_analog_rrcap:
        rrcap_offset = -50
        pwl_rrcap_pts = [(0, interp_sf(pt_t, pt_v, max(1, first_rr_ms + rrcap_offset - rpk_drop - pipe_pw_pt)))]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            rr_cap_d = max(1, ewma_rr_ms + rrcap_offset - rpk_drop - pipe_pw_pt)
            v_rrcap = interp_sf(pt_t, pt_v, rr_cap_d)
            t_step = rpeak_times[j] + 0.001
            if t_step > pwl_rrcap_pts[-1][0]:
                pwl_rrcap_pts.append((t_step, v_rrcap))
        pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))
        sp += """
* AND-GATED RR_CAP ON PREDICTION TIMER
Vth_rrc vth_rr_cap 0 PWL({rrcap})
XMcrrc1 cnrrc pt_sf_out ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 stg1_rrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 stg1_rrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcrrc6 rr_cap_lev stg1_rrc vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcrrc7 rr_cap_lev stg1_rrc vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_rrc rr_cap_lev rrc_eb vdd vss edge_buffer
Xinv_rpk rpeak_raw rpeak_bar_ext vdd vss inverter_sub
XMnand_n1 nand_rrc rrc_eb nand_mid_rrc vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMnand_n2 nand_mid_rrc rpeak_bar_ext vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMnand_p1 nand_rrc rrc_eb vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMnand_p2 nand_rrc rpeak_bar_ext vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
Xinv_nand nand_rrc trig_rr_cap vdd vss inverter_sub
""".format(rrcap=make_pwl(pwl_rrcap_pts))
    else:
        sp += "Vrrc trig_rr_cap 0 PWL({rrc})\n".format(
            rrc=make_trig(rpeak_times, -0.060, 0.055, total_time))

    # Feature circuits
    sp += """
Xrr ecg_l0 trig_rr_cap rpeak_raw trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.net_timer)={vdd:.6f}
.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xbi0.prev_node)=0.300000
.ic V(xbi0.esamp_node)=0.300000
.ic V(xbi1.prev_node)=0.300000
.ic V(xbi1.esamp_node)=0.300000

.tran 0.5m {ttot:.4f}
""".format(vdd=VDD, ic=ic_ewma, ttot=total_time)

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
            for k in range(3, 9):
                nm = ['','','','f3_bi_l0','f4_bi_l1','f5_pwave','f6_width','f7_slope','f8_symmetry'][k]
                sp += ".meas tran f{k}_{j} FIND V({nm}) AT={t:.7f}\n".format(
                    k=k, j=j, t=t_f38, nm=nm)

    sp += "\n.end\n"
    return sp, skip, rpeak_times, total_time


def run_sim(sp_text, label):
    tb = os.path.join(FRONTEND_DIR, "_diag_{}.spice".format(label))
    logf = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp_text)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', logf, tb], capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    if not os.path.exists(logf): return None, elapsed
    with open(logf) as f: text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    return results, elapsed


def compute_rhos(results, beats, skip, n):
    f1_c, f1_p, f2_c, f2_p = [], [], [], []
    f38_c, f38_p = [], []
    for j in range(skip, n):
        bt = beats[j]
        f1 = results.get('f1_{}'.format(j))
        f2 = results.get('f2_{}'.format(j))
        vals38 = [results.get('f{}_{}'.format(k, j)) for k in range(3, 9)]
        if f1 is not None: f1_c.append(f1); f1_p.append(bt['py_f1'])
        if f2 is not None: f2_c.append(f2); f2_p.append(bt['py_f2'])
        if all(v is not None for v in vals38):
            f38_c.append(vals38)
            f38_p.append([bt['py_bi_l0'], bt['py_bi_l1'], bt['py_pw'],
                          bt['py_wid'], bt['py_slp'], bt['py_sym']])
    rhos = [0.0]*8
    if len(f1_c) >= 4:
        rho, _ = spearmanr(f1_c, f1_p); rhos[0] = abs(rho)
    if len(f2_c) >= 4:
        rho, _ = spearmanr(f2_c, f2_p); rhos[1] = abs(rho)
    if len(f38_c) >= 4:
        ca, pa = np.array(f38_c), np.array(f38_p)
        for i in range(6):
            rho, _ = spearmanr(ca[:, i], pa[:, i]); rhos[i+2] = abs(rho)
    return rhos, len(f38_c)


print("Calibrating timers (including prediction timer)...")
cal = calibrate_timers()
print("  RR SF (0.11 mV/ms): {}".format(['{:.4f}'.format(v) for v in cal['rr_v']]))
print("  PT SF (0.32 mV/ms): {}".format(['{:.4f}'.format(v) for v in cal['pt_v']]))

# Compute slopes
rr_slope = (cal['rr_v'][4] - cal['rr_v'][8]) / (cal['rr_t'][8] - cal['rr_t'][4]) * 1000
pt_slope = (cal['pt_v'][4] - cal['pt_v'][8]) / (cal['pt_t'][8] - cal['pt_t'][4]) * 1000
print("  RR slope (200-600ms): {:.3f} mV/ms".format(rr_slope))
print("  PT slope (200-600ms): {:.3f} mV/ms".format(pt_slope))

beats, gain = load_all_beats('100', 20)
print("Loaded {} beats, gain={:.4f}".format(len(beats), gain))

fnames = ['F1:rr', 'F2:asym', 'F3:bi0', 'F4:bi1', 'F5:pw', 'F6:wid', 'F7:slp', 'F8:sym']

configs = [
    ('A_ideal', 'Ideal rr_cap + ideal pwave + analog DT'),
    ('D_pred_pwave', 'Ideal rr_cap + PRED TIMER pwave + analog DT'),
    ('E_pred_both', 'PRED TIMER rr_cap (AND-gated) + PRED TIMER pwave + analog DT'),
]

print("\n{:<20s} {}".format('Config', '  '.join('{:>7s}'.format(f) for f in fnames)))
print("-" * 90)
for mode, desc in configs:
    sp, skip, rpk_t, ttot = gen_tb(beats, gain, cal, mode)
    results, elapsed = run_sim(sp, mode)
    if results:
        rhos, n_meas = compute_rhos(results, beats, skip, len(beats))
        rho_str = '  '.join('{:>7.3f}'.format(r) for r in rhos)
        print("{:<20s} {}  ({:.0f}s, {} beats)".format(mode, rho_str, elapsed, n_meas))
    else:
        print("{:<20s} SIM ERROR ({:.0f}s)".format(mode, elapsed))
    print("  {}".format(desc))
