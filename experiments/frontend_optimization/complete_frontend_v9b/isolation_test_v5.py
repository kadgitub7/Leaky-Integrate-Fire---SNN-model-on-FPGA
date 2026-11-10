"""
Isolation test v5: Root cause fix for F2 + F5 debug.

ROOT CAUSE (F2): rr_timing capture is a TRACK-AND-HOLD. The SF output impedance
is ~886MOhm at 44pA, so when capture caps (10pF total) connect through the switch,
the SF can't drive them in <1ms. Short pulses (0.5ms from pulse_gen) leave prev_node
at the old value. Windows >=20ms give the SF enough time to drive the caps.

FIX: Use comparator LEVEL output directly (bypass pulse_gen). The comparator stays
HIGH from threshold crossing (~R-80ms) until precharge (~R+0ms), creating a natural
50-80ms window — exactly what the track-and-hold needs.

Test configs:
  A. BASELINE         — all ideal triggers (55ms rr_cap window)
  B. ANALOG_LEV       — comparator LEVEL output (no pulse_gen), offset=-80
  C. ANALOG_LEV50     — comparator LEVEL output, offset=-50
  D. ANALOG_PULSE     — comparator + pulse_gen (v4 approach, should FAIL)
  E. PW_ANALOG        — pwave analog comparators (debug F5)
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


def calibrate_timers():
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")
    sp = f""".title timer_cal_v5
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {VDD}
Vss vss 0 0
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/pulse_gen.spice"
Xbias vbias vref vdd vss bias_gen
Vrpk rpeak_pulse 0 PWL(0 0 0.098 0 0.1 {VDD} 0.12 {VDD} 0.121 0 2.0 0)
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
.tran 0.1m 1.2
"""
    dt_times = [1,2,3,4,5,10,15,20,50,100,200,300,400,500,600,700,800]
    for t in dt_times:
        sp += f".meas tran dt_sf_{t:03d} FIND V(xdelay.sf_timer) AT={0.121+t/1000.0:.4f}\n"
    ft_times = [1,2,3,5,10,20]
    for t in ft_times:
        sp += f".meas tran ft_sf_{t:03d} FIND V(xdelay.sf_timer2) AT={0.121+t/1000.0:.4f}\n"
    rr_times = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
    for t in rr_times:
        sp += f".meas tran rr_sf_{t:04d} FIND V(rr_timer_cal) AT={0.121+t/1000.0:.4f}\n"
    sp += "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_cal_v5.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp)
    subprocess.run([NGSPICE, '-b', '-o', log, tb], capture_output=True, text=True, timeout=180, cwd=NGSPICE_DIR)
    with open(log) as f: text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    try: os.remove(tb); os.remove(log)
    except: pass
    return {
        'dt_t': dt_times, 'dt_v': [results.get(f'dt_sf_{t:03d}', 0) for t in dt_times],
        'ft_t': ft_times, 'ft_v': [results.get(f'ft_sf_{t:03d}', 0) for t in ft_times],
        'rr_t': rr_times, 'rr_v': [results.get(f'rr_sf_{t:04d}', 0) for t in rr_times],
    }


def generate_testbench(beats, gain, config, cal):
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
    rpk_drop = 0.5
    pipe_s14 = 2.9; pipe_wid = 4.5; pipe_s69 = 4.4; pipe_bi = 5.9; pipe_s10 = 1.8

    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_s14))
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_wid))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_s69))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_bi))
    vth_10  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_s10))

    first_rr_ms = beats[0]['pre_rr'] * 1000
    ic_ewma = interp_sf(rr_t, rr_v, first_rr_ms)

    rrcap_mode = config.get('rrcap_mode', 'ideal_55ms')
    rrcap_offset = config.get('rrcap_offset', -50)
    pwave_mode = config.get('pwave_mode', 'ideal')
    pipe_pw_rr = config.get('pipe_pw_rr', 14.0)
    dt_mode = config.get('dt_mode', 'analog')

    sp = f""".title iso_v5
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

    if dt_mode == 'ideal':
        sp += f"""
Vs14 trig_s14 0 PWL({make_trig(rpeak_times, 5/FS, PW, total_time)})
Vwid trig_wid 0 PWL({make_trig(rpeak_times, 10/FS, PW, total_time)})
Vs69 trig_s69 0 PWL({make_trig(rpeak_times, 25/FS, PW, total_time)})
Vbco trig_bi_copy 0 PWL({make_trig(rpeak_times, 0.300, 0.002, total_time)})
Vrry trig_rr_copy 0 PWL({make_trig(rpeak_times, 0.010, 0.002, total_time)})
"""
    else:
        sp += f"""
Vrpk_dt rpeak_pulse_dt 0 PWL({make_trig(rpeak_times, 0, PW, total_time)})
Xdt rpeak_pulse_dt dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

Vth14  vth_14_fast 0 {vth_14:.6f}
Vth69  vth_69      0 {vth_69:.6f}
Vth28  vth_28      0 {vth_28:.6f}
Vth300 vth_300     0 {vth_300:.6f}
Vth10  vth_10      0 {vth_10:.6f}

Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen
Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias pulse_gen
"""

    sp += f"""
Vrpk_rr trig_rpeak_rr 0 PWL({make_trig(rpeak_times, 0, PW, total_time)})
Vrsh trig_rpeak_sh 0 PWL({make_trig(rpeak_times, 0, PW, total_time)})
"""

    # === RR_CAP TRIGGER ===
    if rrcap_mode == 'ideal_55ms':
        sp += f"Vrrc trig_rr_cap 0 PWL({make_trig(rpeak_times, -0.060, 0.055, total_time)})\n"
    elif rrcap_mode == 'ideal_pulse':
        rrcap_width = config.get('rrcap_width', PW)
        sp += f"Vrrc trig_rr_cap 0 PWL({make_trig(rpeak_times, rrcap_offset/1000.0, rrcap_width, total_time)})\n"
    elif rrcap_mode == 'ideal_gated':
        # Window from R+offset to R-2ms (closes before precharge)
        offset_s = rrcap_offset / 1000.0
        width_s = -offset_s - 0.002
        sp += f"Vrrc trig_rr_cap 0 PWL({make_trig(rpeak_times, offset_s, width_s, total_time)})\n"
    elif rrcap_mode == 'ideal_gated_thru':
        # Window from R+offset to R+5ms (crosses precharge — should fail)
        offset_s = rrcap_offset / 1000.0
        width_s = -offset_s + 0.005
        sp += f"Vrrc trig_rr_cap 0 PWL({make_trig(rpeak_times, offset_s, width_s, total_time)})\n"
    elif rrcap_mode == 'ewma_pulse':
        rrcap_width = config.get('rrcap_width', 0.010)
        ewma_pulse_pts = [(0, 0)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            t_fire = t_rpk + (ewma_rr_ms + rrcap_offset) / 1000.0
            if t_fire > 0.01:
                ewma_pulse_pts += [
                    (t_fire - 0.0004, 0), (t_fire, VDD),
                    (t_fire + rrcap_width, VDD), (t_fire + rrcap_width + 0.0004, 0)
                ]
        ewma_pulse_pts.append((total_time, 0))
        sp += f"Vrrc trig_rr_cap 0 PWL({make_pwl(ewma_pulse_pts)})\n"
    elif rrcap_mode == 'analog_level':
        # Comparator LEVEL output directly drives capture (NO pulse_gen)
        pwl_rrcap_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            rr_cap_d = max(1, ewma_rr_ms + rrcap_offset - rpk_drop - pipe_pw_rr)
            v_rrcap = interp_sf(rr_t, rr_v, rr_cap_d)
            t_step = t_rpk + 0.001
            if t_step > pwl_rrcap_pts[-1][0]:
                pwl_rrcap_pts.append((t_step, v_rrcap))
        pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))
        sp += f"""
* TWO-STAGE RR_CAP COMPARATOR — LEVEL output (no pulse_gen)
Vth_rrc vth_rr_cap 0 PWL({make_pwl(pwl_rrcap_pts)})
XMcrrc1 cnrrc rr_timer ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 stg1_rrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 stg1_rrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcrrc6 rr_cap_lev stg1_rrc vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcrrc7 rr_cap_lev stg1_rrc vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_rrc rr_cap_lev  trig_rr_cap vdd vss edge_buffer
"""
    elif rrcap_mode == 'analog':
        # Comparator + pulse_gen (the v4 approach that fails)
        pwl_rrcap_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            rr_cap_d = max(1, ewma_rr_ms + rrcap_offset - rpk_drop - pipe_pw_rr)
            v_rrcap = interp_sf(rr_t, rr_v, rr_cap_d)
            t_step = t_rpk + 0.001
            if t_step > pwl_rrcap_pts[-1][0]:
                pwl_rrcap_pts.append((t_step, v_rrcap))
        pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))
        sp += f"""
* TWO-STAGE RR_CAP COMPARATOR + PULSE_GEN (v4 approach)
Vth_rrc vth_rr_cap 0 PWL({make_pwl(pwl_rrcap_pts)})
XMcrrc1 cnrrc rr_timer ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 stg1_rrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 stg1_rrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcrrc6 rr_cap_lev stg1_rrc vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcrrc7 rr_cap_lev stg1_rrc vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_rrc rr_cap_lev  rrc_sharp vdd vss edge_buffer
Xpg_rrc rrc_sharp   trig_rr_cap vdd vss vbias pulse_gen
"""
    elif rrcap_mode == 'analog_gated':
        # Comparator level output AND NOT rpeak — window closes at precharge
        pwl_rrcap_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            rr_cap_d = max(1, ewma_rr_ms + rrcap_offset - rpk_drop - pipe_pw_rr)
            v_rrcap = interp_sf(rr_t, rr_v, rr_cap_d)
            t_step = t_rpk + 0.001
            if t_step > pwl_rrcap_pts[-1][0]:
                pwl_rrcap_pts.append((t_step, v_rrcap))
        pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))
        sp += f"""
* TWO-STAGE RR_CAP COMPARATOR + AND GATE (level AND NOT rpeak)
Vth_rrc vth_rr_cap 0 PWL({make_pwl(pwl_rrcap_pts)})
XMcrrc1 cnrrc rr_timer ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 stg1_rrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 stg1_rrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcrrc6 rr_cap_lev stg1_rrc vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcrrc7 rr_cap_lev stg1_rrc vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_rrc rr_cap_lev  rrc_eb vdd vss edge_buffer
* Generate rpeak_bar_ext for AND gate
Xinv_rpk rpeak_raw rpeak_bar_ext vdd vss inverter_sub
* NAND(rrc_eb, rpeak_bar_ext) + INV = AND gate
XMnand_n1 nand_rrc rrc_eb nand_mid_rrc vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMnand_n2 nand_mid_rrc rpeak_bar_ext vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=2 nf=1
XMnand_p1 nand_rrc rrc_eb vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMnand_p2 nand_rrc rpeak_bar_ext vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
Xinv_nand nand_rrc trig_rr_cap vdd vss inverter_sub
"""

    # === PWAVE TRIGGERS ===
    if pwave_mode == 'ideal':
        sp += f"Vpwb trig_pw_base 0 PWL({make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)})\n"
        sp += f"Vpwp trig_pw_peak 0 PWL({make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)})\n"
    elif pwave_mode == 'analog':
        pwl_base_pts = [(0, 0.3)]
        pwl_peak_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            pw_base_d = max(1, ewma_rr_ms - 201 - rpk_drop - pipe_pw_rr)
            pw_peak_d = max(1, ewma_rr_ms - 131 - rpk_drop - pipe_pw_rr)
            v_base = interp_sf(rr_t, rr_v, pw_base_d)
            v_peak = interp_sf(rr_t, rr_v, pw_peak_d)
            t_step = t_rpk + 0.001
            if t_step > pwl_base_pts[-1][0]:
                pwl_base_pts.append((t_step, v_base))
                pwl_peak_pts.append((t_step, v_peak))
        pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
        pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
        sp += f"""
* TWO-STAGE PWAVE COMPARATORS on rr_timing SF
Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})
Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})
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
"""

    # === RR TIMING + FEATURE CIRCUITS ===
    sp += f"""
Xrr ecg_l0 trig_rr_cap trig_rpeak_rr trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.net_timer)={VDD:.6f}
.ic V(xrr.ewma_node)={ic_ewma:.6f}
.ic V(xrr.esamp_node)={ic_ewma:.6f}
.ic V(xrr.prev_node)={ic_ewma:.6f}
.ic V(xbi0.prev_node)=0.300000
.ic V(xbi0.esamp_node)=0.300000
.ic V(xbi1.prev_node)=0.300000
.ic V(xbi1.esamp_node)=0.300000

.tran 0.5m {total_time:.4f}

"""
    skip = 2
    debug_analog = rrcap_mode.startswith('analog')
    debug_dt = (dt_mode == 'analog')
    for j in range(skip, n):
        t_f1 = rpeak_times[j] - 0.010
        if t_f1 > 0.5:
            sp += f".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t_f1:.7f}\n"
        if j + 1 < n:
            t_f2 = rpeak_times[j + 1] - 0.065
            if t_f2 > 0.5:
                sp += f".meas tran f2_{j} FIND V(f2_rr_asym) AT={t_f2:.7f}\n"
                if debug_analog:
                    sp += f".meas tran rrc_at_f2_{j} FIND V(trig_rr_cap) AT={t_f2:.7f}\n"
                    sp += f".meas tran prev_at_f2_{j} FIND V(xrr.prev_node) AT={t_f2:.7f}\n"
                    sp += f".meas tran sf_at_f2_{j} FIND V(rr_timer) AT={t_f2:.7f}\n"
        if debug_dt:
            td = rpeak_times[j] + 0.005
            sp += f".meas tran wid_fire_{j} WHEN V(dt_wid_lev) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran s14_fire_{j} WHEN V(dt_s14_lev) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran s69_fire_{j} WHEN V(dt_s69_lev) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran bi_fire_{j} WHEN V(dt_bi_lev) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran rr_fire_{j} WHEN V(dt_rr_lev) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran twid_fire_{j} WHEN V(trig_wid) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran ts14_fire_{j} WHEN V(trig_s14) = 0.45 RISE=1 TD={td:.7f}\n"
            sp += f".meas tran ts69_fire_{j} WHEN V(trig_s69) = 0.45 RISE=1 TD={td:.7f}\n"
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
    tb = os.path.join(FRONTEND_DIR, f"_iso_v5_{label}.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp_text)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
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
        vals = [results.get(f'f{k}_{j}') for k in range(1, 9)]
        if all(v is not None for v in vals[2:]):
            bt = beats[j]
            c = [vals[0] or 0, vals[1] or 0] + vals[2:]
            p = [bt['py_f1'], bt['py_f2'], bt['py_bi_l0'], bt['py_bi_l1'],
                 bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']]
            circuit.append(c)
            python.append(p)
    if len(circuit) < 4: return [0]*8, 0
    ca, pa = np.array(circuit), np.array(python)
    rhos = []
    for i in range(8):
        try:
            rho, _ = spearmanr(ca[:, i], pa[:, i])
            rhos.append(abs(rho))
        except:
            rhos.append(0.0)
    return rhos, len(circuit)


# ========================================================================
OUT = os.path.join(FRONTEND_DIR, "_isolation_v5_results.txt")
def log(msg):
    print(msg)
    with open(OUT, 'a') as f: f.write(msg + '\n')
with open(OUT, 'w') as f: f.write('')

log("=" * 70)
log("ISOLATION TEST v5: Delay timer diagnostic (F6/F7)")
log("=" * 70)
log("")
log("DIAGNOSTIC: F6=0.123, F7=0.358 with analog delay_timer.")
log("  Comparing ideal DT triggers (v3 baseline) vs analog DT chain.")
log("  This isolates whether delay_timer->edge_buffer->pulse_gen is the issue.")
log("")

log("[1] Loading 12 beats from record 100...")
beats, gain = load_beats('100', N_BEATS)
log(f"  Loaded {len(beats)} beats")
log(f"  RR intervals: {' '.join(f'{b['pre_rr']*1000:.0f}' for b in beats)}ms")
log(f"  EWMA RR:      {' '.join(f'{b['ewma_rr']*1000:.0f}' for b in beats)}ms")

log("\n[2] Calibrating timers...")
cal = calibrate_timers()
log(f"  DT SF (main, 17 pts): {[f'{v:.6f}' for v in cal['dt_v']]}")
log(f"  DT times: {cal['dt_t']}")
log(f"  FT SF (fast, 6 pts): {[f'{v:.6f}' for v in cal['ft_v']]}")
log(f"  FT times: {cal['ft_t']}")
log(f"  RR SF: {cal['rr_v'][:5]}")

dt_t, dt_v = cal['dt_t'], cal['dt_v']
ft_t, ft_v = cal['ft_t'], cal['ft_v']
rpk_drop = 0.5
pipe_s14 = 2.9; pipe_wid = 4.5; pipe_s69 = 4.4; pipe_bi = 5.9; pipe_s10 = 1.8
vth_14_val  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_s14))
vth_28_val  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_wid))
vth_69_val  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_s69))
vth_300_val = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_bi))
vth_10_val  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_s10))
log(f"  Thresholds: vth_14={vth_14_val:.6f} vth_28={vth_28_val:.6f} vth_69={vth_69_val:.6f} vth_300={vth_300_val:.6f} vth_10={vth_10_val:.6f}")

# Step 1: Verify the timing theory with IDEAL triggers (no comparator issues)
configs = [
    ("A_IDEAL_DT", {
        'rrcap_mode': 'ideal_55ms',
        'pwave_mode': 'ideal',
        'dt_mode': 'ideal',
    }, "ALL ideal triggers (v3 baseline reference)"),

    ("B_ANALOG_DT", {
        'rrcap_mode': 'ideal_55ms',
        'pwave_mode': 'ideal',
        'dt_mode': 'analog',
    }, "Analog delay_timer + pulse_gen for F6/F7 triggers"),
]

fnames = ['F1:rr', 'F2:asym', 'F3:bi0', 'F4:bi1', 'F5:pw', 'F6:wid', 'F7:slp', 'F8:sym']
all_results = {}

for i, (label, cfg, desc) in enumerate(configs):
    log(f"\n[{i+3}] Config {label}: {desc}")
    sp, skip, rpk_times, total_time = generate_testbench(beats, gain, cfg, cal)
    results, elapsed = run_sim(sp, label)
    if results is None:
        log(f"  FAILED (sim error)")
        all_results[label] = [0]*8
        continue
    rhos, n_meas = extract_and_rho(results, beats, skip, len(beats))
    all_results[label] = rhos
    log(f"  {n_meas} beats, {elapsed:.0f}s")
    log(f"  rho: {' '.join(f'{r:.3f}' for r in rhos)}")
    if cfg.get('rrcap_mode', '').startswith('analog'):
        log(f"  DEBUG: trig_rr_cap / prev_node / sf_out at F2 times:")
        for j in range(skip, len(beats)):
            rrc = results.get(f'rrc_at_f2_{j}')
            prev = results.get(f'prev_at_f2_{j}')
            sf = results.get(f'sf_at_f2_{j}')
            if rrc is not None:
                rr_ms = beats[j]['pre_rr']*1000
                ewma_ms = beats[j]['ewma_rr']*1000
                log(f"    beat {j}: rrc={rrc:.4f}V prev={prev:.4f}V sf={sf:.4f}V RR={rr_ms:.0f}ms ewma={ewma_ms:.0f}ms {'OPEN!' if rrc > 0.4 else 'closed'}")
    if cfg.get('dt_mode', 'analog') == 'analog':
        log(f"  DT DEBUG: exact trigger fire times (WHEN V=0.45 RISE):")
        log(f"    Targets: s14=R+13.9ms  wid=R+27.8ms  s69=R+69.4ms  bi=R+300ms  rr=R+10ms")
        for j in range(skip, min(skip+4, len(beats))):
            rpk = rpk_times[j]
            wf = results.get(f'wid_fire_{j}')
            s14f = results.get(f's14_fire_{j}')
            s69f = results.get(f's69_fire_{j}')
            bif = results.get(f'bi_fire_{j}')
            rrf = results.get(f'rr_fire_{j}')
            twf = results.get(f'twid_fire_{j}')
            ts14f = results.get(f'ts14_fire_{j}')
            ts69f = results.get(f'ts69_fire_{j}')
            if wf is not None:
                log(f"    beat {j} (R={rpk*1000:.1f}ms):")
                log(f"      comparator levels:  s14={1000*(s14f-rpk):.1f}ms  wid={1000*(wf-rpk):.1f}ms  s69={1000*(s69f-rpk):.1f}ms  bi={1000*(bif-rpk):.1f}ms  rr={1000*(rrf-rpk):.1f}ms")
                if twf is not None:
                    log(f"      pulse_gen outputs:  s14={1000*(ts14f-rpk):.1f}ms  wid={1000*(twf-rpk):.1f}ms  s69={1000*(ts69f-rpk):.1f}ms")

# Summary
log("\n" + "=" * 70)
log("RESULTS SUMMARY")
log("=" * 70)
header = f"  {'Config':<16s} " + " ".join(f"{f:>7s}" for f in fnames)
log(header)
log("  " + "-" * (16 + 8 * len(fnames)))
baseline = all_results.get('A_BASELINE', [0]*8)
for label, _, desc in configs:
    rhos = all_results.get(label, [0]*8)
    parts = []
    for j, r in enumerate(rhos):
        if label != 'A_BASELINE' and j == 1:  # F2 column
            diff = r - baseline[j]
            if diff > 0.02: parts.append(f"\033[92m{r:.3f}\033[0m")
            elif diff < -0.05: parts.append(f"\033[91m{r:.3f}\033[0m")
            else: parts.append(f"{r:.3f}  ")
        else:
            parts.append(f"{r:.3f}  ")
    log(f"  {label:<16s} " + " ".join(f"{p:>7s}" for p in parts))
    log(f"    ({desc})")

log("\n  KEY RESULTS:")
log("  B: ideal 58ms window R-60 to R-2ms — expected to match baseline")
log("  C: same but crosses precharge — tests precharge effect")
log("  D: analog comparator (offset=-60) + AND gate — the fix")
log("  E: pwave analog debug")
