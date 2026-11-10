"""
Isolation test v2: Test fixes for each identified culprit.

Round 1 findings:
  E_RRCAP killed F2 (0.988->0.648): capture pulse too short (0.7ms vs 55ms ideal)
  F_PWAVE killed F5 (0.994->0.500): comparator resolves too slowly (pipe_pw=3.5ms too low)
  C_RPK_SH killed F4 (0.988->0.673): rpeak_pulse fires 2.7ms early, wrong ECG sample point

Fixes tested here:
  FIX_RRCAP:  Use long_pulse_gen (2ms) for rr_cap trigger
  FIX_PWAVE:  Increase pipe_pw from 3.5ms to 8/10/12ms to account for slow comparator
  FIX_RPK_SH: Use delay-compensated rpeak pulse for S&H (add 2.7ms RC delay)
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
N_BEATS = 12

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


def load_beats(record_id, n_beats):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]
    all_peaks = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(np.abs(sig[bs+QRS_S:bs+QRS_E, 0])))
    gain = 0.16 / max(all_peaks)
    r_peak_scaled = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        r_peak_scaled.append(VBASE + gain * sig[idx, 0])
    rpeak_threshold = VBASE + 0.5 * (max(r_peak_scaled) - VBASE)

    beats = []
    ewma_rr = None
    for i in range(1, len(valid) - 1):
        if len(beats) >= n_beats: break
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
            'label': aami_map[sym], 'pre_rr': pre_rr, 'post_rr': post_rr,
            'ewma_rr': ewma_rr,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': pre_rr - ewma_rr_before, 'py_f2': pre_rr - post_rr,
            'py_bi_l0': bi_l0, 'py_bi_l1': bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[RPEAK_IDX+QRS_S+10, 0],
            'py_slp': beat[RPEAK_IDX+QRS_S+5, 0] - beat[RPEAK_IDX+QRS_S+25, 0],
            'py_sym': beat[RPEAK_IDX+QRS_S, 0],
        })
    return beats, gain, rpeak_threshold


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


def calibrate_timer():
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")
    sp = f""".title timer_cal
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {VDD}
Vss vss 0 0
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/delay_timer.spice"
Xbias vbias vref vdd vss bias_gen
Vrpk rpeak_pulse 0 PWL(0 0 0.098 0 0.1 {VDD} 0.12 {VDD} 0.121 0 1.0 0)
Vt1 vth_14_fast 0 0.1
Vt2 vth_69 0 0.1
Vt3 vth_28 0 0.1
Vt4 vth_300 0 0.1
Vt5 vth_10 0 0.1
Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer
.ic V(xbias.iref)=0.3
.tran 0.1m 1.1
"""
    dt_times = [1,2,3,4,5,10,15,20,50,100,200,300,400,500,600,700,800]
    for t in dt_times:
        sp += f".meas tran dt_sf_{t:03d} FIND V(xdelay.sf_timer) AT={0.121+t/1000.0:.4f}\n"
    ft_times = [1,2,3,5,10,20]
    for t in ft_times:
        sp += f".meas tran ft_sf_{t:03d} FIND V(xdelay.sf_timer2) AT={0.121+t/1000.0:.4f}\n"
    sp += "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_cal_iso2.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp)
    subprocess.run([NGSPICE, '-b', '-o', log, tb], capture_output=True, text=True, timeout=120, cwd=NGSPICE_DIR)
    with open(log) as f: text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    try: os.remove(tb); os.remove(log)
    except: pass
    dt_sf = [results.get(f'dt_sf_{t:03d}') for t in dt_times]
    ft_sf = [results.get(f'ft_sf_{t:03d}') for t in ft_times]
    dt_t = [t for t, v in zip(dt_times, dt_sf) if v is not None]
    dt_v = [v for v in dt_sf if v is not None]
    ft_t = [t for t, v in zip(ft_times, ft_sf) if v is not None]
    ft_v = [v for v in ft_sf if v is not None]
    return dt_t, dt_v, ft_t, ft_v


def generate_testbench(beats, gain, rpeak_threshold, config, dt_t, dt_v, ft_t, ft_v):
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

    rpk_drop = 0.5
    pipe_fast = 1.0
    pipe_main = 3.5
    pipe_pw = config.get('pipe_pw', 3.5)

    vth_14 = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_fast))
    vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_main))
    vth_69 = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_main))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_main))
    vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_fast))

    use_analog_rrcap = config.get('analog_rrcap', False)
    use_analog_pwave = config.get('analog_pwave', False)
    use_analog_rpeak_sh = config.get('analog_rpeak_sh', False)
    rrcap_long = config.get('rrcap_long_pulse', False)

    sp = f""".title iso2_test
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0

Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

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
"""

    # Rpeak detector - include if needed
    if use_analog_rpeak_sh:
        sp += f'.include "{fp}/rpeak_detector.spice"\n'
        sp += f"Vth_rpeak rpeak_threshold 0 {rpeak_threshold:.4f}\n"
        sp += f"Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector\n"
        sp += f"Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen\n"

    # Delay timer always driven by ideal rpeak pulse (Phase 4 proven)
    rpeak_pulse_pwl = make_trig(rpeak_times, 0, PW, total_time)
    sp += f"Vrpk_dt rpeak_pulse_dt 0 PWL({rpeak_pulse_pwl})\n"
    sp += f"""
Vth14  vth_14_fast 0 {vth_14:.6f}
Vth69  vth_69      0 {vth_69:.6f}
Vth28  vth_28      0 {vth_28:.6f}
Vth300 vth_300     0 {vth_300:.6f}
Vth10  vth_10      0 {vth_10:.6f}

Xdt rpeak_pulse_dt dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias pulse_gen
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen
Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen
"""

    # Pwave thresholds
    if use_analog_pwave or use_analog_rrcap:
        pwl_base_pts = [(0, 0.3)]
        pwl_peak_pts = [(0, 0.3)]
        pwl_rrcap_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            pw_base_d = max(1, ewma_rr_ms - 201 - rpk_drop - pipe_pw)
            pw_peak_d = max(1, ewma_rr_ms - 131 - rpk_drop - pipe_pw)
            rr_cap_d  = max(1, ewma_rr_ms - 50  - rpk_drop - pipe_pw)
            v_base = interp_sf(dt_t, dt_v, pw_base_d)
            v_peak = interp_sf(dt_t, dt_v, pw_peak_d)
            v_rrcap = interp_sf(dt_t, dt_v, rr_cap_d)
            t_step = t_rpk + 0.001
            if t_step > pwl_base_pts[-1][0]:
                pwl_base_pts.append((t_step, v_base))
                pwl_peak_pts.append((t_step, v_peak))
                pwl_rrcap_pts.append((t_step, v_rrcap))
        pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
        pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
        pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))

    if use_analog_pwave:
        sp += f"""
Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})
Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})

XMcpwb1 cnpwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 pw_base_lev dt_sf_out ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 pw_base_lev cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

XMcpwp1 cnpwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 pw_peak_lev dt_sf_out ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 pw_peak_lev cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen
Xeb_pwp pw_peak_lev pwp_sharp vdd vss edge_buffer
Xpg_pwp pwp_sharp   trig_pw_peak vdd vss vbias pulse_gen
"""
    else:
        trig_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)
        trig_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)
        sp += f"Vpwb trig_pw_base 0 PWL({trig_pw_base})\n"
        sp += f"Vpwp trig_pw_peak 0 PWL({trig_pw_peak})\n"

    if use_analog_rrcap:
        rrcap_gen = "long_pulse_gen" if rrcap_long else "pulse_gen"
        sp += f"""
Vth_rrc vth_rr_cap 0 PWL({make_pwl(pwl_rrcap_pts)})
XMcrrc1 cnrrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 rr_cap_lev dt_sf_out ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 rr_cap_lev cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
Xeb_rrc rr_cap_lev  rrc_sharp vdd vss edge_buffer
Xpg_rrc rrc_sharp   trig_rr_cap vdd vss vbias {rrcap_gen}
"""
    else:
        trig_rr_cap = make_trig(rpeak_times, -0.060, 0.055, total_time)
        sp += f"Vrrc trig_rr_cap 0 PWL({trig_rr_cap})\n"

    # S&H trigger for F3/F4/F8
    if use_analog_rpeak_sh:
        rpeak_sh_net = "rpeak_pulse"
    else:
        trig_rpeak_sh = make_trig(rpeak_times, 0, PW, total_time)
        sp += f"Vrsh trig_rpeak_sh 0 PWL({trig_rpeak_sh})\n"
        rpeak_sh_net = "trig_rpeak_sh"

    # rr_timing precharge always ideal (D_RPK_RR had no impact)
    trig_rpeak_rr = make_trig(rpeak_times, 0, PW, total_time)
    sp += f"Vrpk_rr trig_rpeak_rr 0 PWL({trig_rpeak_rr})\n"

    sp += f"""
Xrr ecg_l0 trig_rr_cap trig_rpeak_rr trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 {rpeak_sh_net} trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 {rpeak_sh_net} trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 {rpeak_sh_net} f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.net_timer)={VDD:.6f}
.ic V(xrr.ewma_node)=0.400000
.ic V(xrr.esamp_node)=0.400000
.ic V(xrr.prev_node)=0.400000
.ic V(xbi0.prev_node)=0.300000
.ic V(xbi0.esamp_node)=0.300000
.ic V(xbi1.prev_node)=0.300000
.ic V(xbi1.esamp_node)=0.300000

.tran 0.5m {total_time:.4f}

"""
    skip = 2
    for j in range(skip, n):
        t_f1 = rpeak_times[j] - 0.010
        if t_f1 > 0.5:
            sp += f".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t_f1:.7f}\n"
        if j + 1 < n:
            t_f2 = rpeak_times[j + 1] - 0.065
            if t_f2 > 0.5:
                sp += f".meas tran f2_{j} FIND V(f2_rr_asym) AT={t_f2:.7f}\n"
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            sp += f".meas tran f3_{j} FIND V(f3_bi_l0) AT={t_f38:.7f}\n"
            sp += f".meas tran f4_{j} FIND V(f4_bi_l1) AT={t_f38:.7f}\n"
            sp += f".meas tran f5_{j} FIND V(f5_pwave) AT={t_f38:.7f}\n"
            sp += f".meas tran f6_{j} FIND V(f6_width) AT={t_f38:.7f}\n"
            sp += f".meas tran f7_{j} FIND V(f7_slope) AT={t_f38:.7f}\n"
            sp += f".meas tran f8_{j} FIND V(f8_symmetry) AT={t_f38:.7f}\n"
    sp += "\n.end\n"
    return sp, skip, rpeak_times, total_time


def run_sim(sp_text, label):
    tb = os.path.join(FRONTEND_DIR, f"_iso2_{label}.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp_text)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=1800, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    if not os.path.exists(log): return None, elapsed
    with open(log) as f: text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    try: os.remove(tb); os.remove(log)
    except: pass
    return results, elapsed


def extract_and_rho(results, beats, skip, n):
    circuit, python = [], []
    for j in range(skip, n):
        f1 = results.get(f'f1_{j}')
        f2 = results.get(f'f2_{j}')
        f3 = results.get(f'f3_{j}')
        f4 = results.get(f'f4_{j}')
        f5 = results.get(f'f5_{j}')
        f6 = results.get(f'f6_{j}')
        f7 = results.get(f'f7_{j}')
        f8 = results.get(f'f8_{j}')
        if all(v is not None for v in [f3, f4, f5, f6, f7, f8]):
            bt = beats[j]
            c = [f1 if f1 else 0, f2 if f2 else 0, f3, f4, f5, f6, f7, f8]
            p = [bt['py_f1'], bt['py_f2'], bt['py_bi_l0'], bt['py_bi_l1'],
                 bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']]
            circuit.append(c)
            python.append(p)
    circuit = np.array(circuit) if circuit else None
    python = np.array(python) if python else None
    if circuit is None or len(circuit) < 4:
        return [0.0] * 8, 0
    rhos = []
    for i in range(8):
        try:
            rho, _ = spearmanr(circuit[:, i], python[:, i])
            rhos.append(abs(rho))
        except:
            rhos.append(0.0)
    return rhos, len(circuit)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("ISOLATION TEST v2: Testing fixes for each culprit")
print("=" * 70)

beats, gain, rpeak_threshold = load_beats('100', N_BEATS)
print(f"Loaded {len(beats)} beats, gain={gain:.6f}, rpeak_th={rpeak_threshold:.4f}")

print("\nCalibrating timer...")
dt_t, dt_v, ft_t, ft_v = calibrate_timer()

configs = [
    # Baseline
    ("BASELINE", {}, "Phase 4 baseline"),

    # FIX 1: rr_cap with long_pulse_gen
    ("RRCAP_short", {'analog_rrcap': True}, "Analog rr_cap, standard pulse_gen (0.7ms)"),
    ("RRCAP_long",  {'analog_rrcap': True, 'rrcap_long_pulse': True}, "Analog rr_cap, long_pulse_gen (2ms)"),

    # FIX 2: pwave with different pipe_pw values
    ("PW_3.5ms", {'analog_pwave': True, 'pipe_pw': 3.5},  "Analog pwave, pipe_pw=3.5ms (original)"),
    ("PW_8ms",   {'analog_pwave': True, 'pipe_pw': 8.0},  "Analog pwave, pipe_pw=8ms"),
    ("PW_12ms",  {'analog_pwave': True, 'pipe_pw': 12.0}, "Analog pwave, pipe_pw=12ms"),
    ("PW_16ms",  {'analog_pwave': True, 'pipe_pw': 16.0}, "Analog pwave, pipe_pw=16ms"),

    # FIX 3: rpeak S&H (for reference, no easy fix)
    ("RPK_SH", {'analog_rpeak_sh': True}, "Analog rpeak_pulse for S&H (no fix, reference)"),
]

fnames = ['F1:rr', 'F2:asym', 'F3:bi0', 'F4:bi1', 'F5:pw', 'F6:wid', 'F7:slp', 'F8:sym']
all_results = {}

for i, (label, cfg, desc) in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] {label}: {desc}")
    sp, skip, rpk_times, total_time = generate_testbench(
        beats, gain, rpeak_threshold, cfg, dt_t, dt_v, ft_t, ft_v)
    results, elapsed = run_sim(sp, label)
    if results is None:
        print(f"  FAILED")
        all_results[label] = [0.0] * 8
        continue
    rhos, n_meas = extract_and_rho(results, beats, skip, len(beats))
    all_results[label] = rhos
    print(f"  {n_meas} beats, {elapsed:.0f}s")
    print(f"  rho: {' '.join(f'{r:.3f}' for r in rhos)}")

# Summary
print(f"\n{'='*70}")
print("FIX TEST RESULTS")
print(f"{'='*70}")
header = f"  {'Config':<15} " + " ".join(f"{fn:>7}" for fn in fnames)
print(header)
print("  " + "-" * (15 + 8 * len(fnames)))

baseline = all_results.get('BASELINE', [0]*8)
for label, _, desc in configs:
    rhos = all_results.get(label, [0]*8)
    parts = []
    for j, r in enumerate(rhos):
        delta = r - baseline[j]
        if label == 'BASELINE':
            parts.append(f"{r:7.3f}")
        elif abs(delta) > 0.05:
            if delta > 0:
                parts.append(f"{r:5.3f}++")
            else:
                parts.append(f"{r:5.3f}!!")
        elif abs(delta) > 0.02:
            parts.append(f"{r:5.3f} *")
        else:
            parts.append(f"{r:7.3f}")
    print(f"  {label:<15} " + " ".join(parts))

print(f"\n  Legend: ++ = improved >0.05, !! = degraded >0.05, * = changed >0.02")
print()
