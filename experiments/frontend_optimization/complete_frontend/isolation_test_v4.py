"""
Isolation test v4: Move pwave/rr_cap comparators to rr_timing SF.

Root cause from v3: delay timer SF is 0.08-0.13V at 700-800ms delays.
NFET diff pair comparators can't operate below ~0.30V. This killed F2/F5.

Fix: use rr_timing's own SF (320pF, 0.14mV/ms) instead of delay timer SF.
  rr_timing SF at 800ms ≈ 0.39V — well within NFET comparator range.
  Two-stage comparators resolve in ~14ms (vs 3.5ms on delay timer).
  pipe_pw = 14ms compensates for the slower resolution.

Configs:
  A. BASELINE         - Phase 4 (all ideal triggers)
  H. RRTIMER_RRCAP    - rr_cap comparator on rr_timing SF
  I. RRTIMER_PWAVE    - pwave comparators on rr_timing SF
  J. RRTIMER_BOTH     - Both on rr_timing SF
  K. RRTIMER_FULL     - Both + analog rpeak_dt, rpeak_rr, long_rrcopy
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
            'py_f1': pre_rr - ewma_rr_before,
            'py_f2': pre_rr - post_rr,
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


def calibrate_timers():
    """Calibrate both delay timer SF and rr_timing SF curves."""
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title timer_cal_v4
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

* Delay timer: precharge at t=0.1s
Vrpk rpeak_pulse 0 PWL(0 0 0.098 0 0.1 {VDD} 0.12 {VDD} 0.121 0 2.0 0)
Vt1 vth_14_fast 0 0.1
Vt2 vth_69 0 0.1
Vt3 vth_28 0 0.1
Vt4 vth_300 0 0.1
Vt5 vth_10 0 0.1
Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

* RR timing: precharge at same rpeak, capture/copy triggers low
Vrrc_ideal trig_rr_cap_cal 0 0
Vrry_ideal trig_rr_copy_cal 0 0
Xrr_cal ecg_cal trig_rr_cap_cal rpeak_pulse trig_rr_copy_cal f1_cal f2_cal rr_timer_cal rr_ewma_cal vdd vss vbias rr_timing
Vecg_cal ecg_cal 0 0.3

.ic V(xbias.iref)=0.3
.ic V(xrr_cal.ewma_node)=0.400000
.ic V(xrr_cal.esamp_node)=0.400000
.ic V(xrr_cal.prev_node)=0.400000

.tran 0.1m 1.2
"""
    # Delay timer SF measurements
    dt_times = [1,2,3,4,5,10,15,20,50,100,200,300,400,500,600,700,800]
    for t in dt_times:
        sp += f".meas tran dt_sf_{t:03d} FIND V(xdelay.sf_timer) AT={0.121+t/1000.0:.4f}\n"
    ft_times = [1,2,3,5,10,20]
    for t in ft_times:
        sp += f".meas tran ft_sf_{t:03d} FIND V(xdelay.sf_timer2) AT={0.121+t/1000.0:.4f}\n"

    # RR timing SF measurements (same time offsets after rpeak)
    rr_times = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
    for t in rr_times:
        sp += f".meas tran rr_sf_{t:04d} FIND V(rr_timer_cal) AT={0.121+t/1000.0:.4f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_cal_v4.spice")
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

    dt_sf = [results.get(f'dt_sf_{t:03d}') for t in dt_times]
    ft_sf = [results.get(f'ft_sf_{t:03d}') for t in ft_times]
    rr_sf = [results.get(f'rr_sf_{t:04d}') for t in rr_times]

    return {
        'dt_t': dt_times, 'dt_v': [v for v in dt_sf if v is not None],
        'ft_t': ft_times, 'ft_v': [v for v in ft_sf if v is not None],
        'rr_t': rr_times, 'rr_v': [v for v in rr_sf if v is not None],
    }


def compute_thresholds(cal):
    rpk_drop = 0.5; pipe_main = 3.5; pipe_fast = 1.0
    dt_t, dt_v = cal['dt_t'], cal['dt_v']
    ft_t, ft_v = cal['ft_t'], cal['ft_v']
    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_fast))
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_main))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_main))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_main))
    vth_10  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_fast))
    return vth_14, vth_28, vth_69, vth_300, vth_10


def generate_testbench(beats, gain, rpeak_threshold, config, cal, thresholds):
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

    use_rpk_det = config.get('analog_rpeak_dt') or config.get('analog_rpeak_sh') or config.get('analog_rpeak_rr')
    use_rrtimer_pwave = config.get('rrtimer_pwave', False)
    use_rrtimer_rrcap = config.get('rrtimer_rrcap', False)
    rrcap_offset = config.get('rrcap_offset', -80)

    vth_14, vth_28, vth_69, vth_300, vth_10 = thresholds

    # For rr_timing SF comparators: slower slope needs larger pipe_pw
    pipe_pw_rr = 14.0  # ~14ms resolution for 0.14mV/ms slope

    sp = f""".title iso_v4
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

    if use_rpk_det:
        sp += f"""
.include "{fp}/rpeak_detector.spice"
Vth_rpeak rpeak_threshold 0 {rpeak_threshold:.4f}
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen
"""
    else:
        sp += f"""
Vrpk_raw rpeak_raw 0 PWL({make_trig(rpeak_times, 0, PW, total_time)})
Vrpk_pulse rpeak_pulse 0 PWL({make_trig(rpeak_times, 0, PW, total_time)})
"""

    if config.get('analog_rpeak_dt', False):
        sp += "Xdt rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer\n"
    else:
        rpeak_pulse_pwl = make_trig(rpeak_times, 0, PW, total_time)
        sp += f"Vrpk_dt rpeak_pulse_dt 0 PWL({rpeak_pulse_pwl})\n"
        sp += f"Xdt rpeak_pulse_dt dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer\n"

    sp += f"""
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
"""

    if config.get('long_rrcopy', False):
        sp += "Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias long_pulse_gen\n"
    else:
        sp += "Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias pulse_gen\n"

    # rr_timing must be instantiated BEFORE the comparators use rr_timer
    # But rr_timing needs trig_rr_cap as input — circular dependency!
    # Solution: instantiate rr_timing with the rr_cap net, which the comparator will drive.
    # For configs without analog rr_cap, use ideal trig_rr_cap.

    if config.get('analog_rpeak_rr', False):
        rpeak_rr_net = "rpeak_raw"
    else:
        trig_rpeak_rr = make_trig(rpeak_times, 0, PW, total_time)
        sp += f"Vrpk_rr trig_rpeak_rr 0 PWL({trig_rpeak_rr})\n"
        rpeak_rr_net = "trig_rpeak_rr"

    if config.get('analog_rpeak_sh', False):
        rpeak_sh_net = "rpeak_pulse"
    else:
        trig_rpeak_sh = make_trig(rpeak_times, 0, PW, total_time)
        sp += f"Vrsh trig_rpeak_sh 0 PWL({trig_rpeak_sh})\n"
        rpeak_sh_net = "trig_rpeak_sh"

    # For analog rr_cap: trig_rr_cap is driven by the comparator circuit below
    # For ideal rr_cap: trig_rr_cap is a PWL source
    if not use_rrtimer_rrcap:
        trig_rr_cap_ideal = make_trig(rpeak_times, -0.060, 0.055, total_time)
        sp += f"Vrrc trig_rr_cap 0 PWL({trig_rr_cap_ideal})\n"

    # Instantiate rr_timing (uses rr_timer net for SF output)
    sp += f"""
* === RR TIMING (Feature circuits F1/F2) ===
* rr_timer exposes the rr_timing SF for pwave prediction comparators
Xrr ecg_l0 trig_rr_cap {rpeak_rr_net} trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing

.ic V(xrr.net_timer)={VDD:.6f}
.ic V(xrr.ewma_node)=0.400000
.ic V(xrr.esamp_node)=0.400000
.ic V(xrr.prev_node)=0.400000
"""

    # Now add pwave/rr_cap comparators on rr_timer
    rr_t = cal['rr_t']
    rr_v = cal['rr_v']
    rpk_drop = 0.5

    if use_rrtimer_pwave or use_rrtimer_rrcap:
        pwl_base_pts = [(0, 0.3)]
        pwl_peak_pts = [(0, 0.3)]
        pwl_rrcap_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            t_rpk = rpeak_times[j]
            pw_base_d = max(1, ewma_rr_ms - 201 - rpk_drop - pipe_pw_rr)
            pw_peak_d = max(1, ewma_rr_ms - 131 - rpk_drop - pipe_pw_rr)
            rr_cap_d  = max(1, ewma_rr_ms + rrcap_offset - rpk_drop - pipe_pw_rr)
            v_base = interp_sf(rr_t, rr_v, pw_base_d)
            v_peak = interp_sf(rr_t, rr_v, pw_peak_d)
            v_rrcap = interp_sf(rr_t, rr_v, rr_cap_d)
            t_step = t_rpk + 0.001
            if t_step > pwl_base_pts[-1][0]:
                pwl_base_pts.append((t_step, v_base))
                pwl_peak_pts.append((t_step, v_peak))
                pwl_rrcap_pts.append((t_step, v_rrcap))
        pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
        pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
        pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))

    if use_rrtimer_pwave:
        sp += f"""
* TWO-STAGE PWAVE COMPARATORS on rr_timing SF (rr_timer)
* rr_timing SF at ~600-700ms: 0.40-0.41V — in NFET comparator range
Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})
Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})

* pw_base: two-stage (M1=signal on mirror side, M2=threshold on output side)
XMcpwb1 cnpwb rr_timer ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 stg1_pwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 stg1_pwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcpwb6 pw_base_lev stg1_pwb vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcpwb7 pw_base_lev stg1_pwb vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen

* pw_peak: two-stage
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
    else:
        trig_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)
        trig_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)
        sp += f"Vpwb trig_pw_base 0 PWL({trig_pw_base})\n"
        sp += f"Vpwp trig_pw_peak 0 PWL({trig_pw_peak})\n"

    if use_rrtimer_rrcap:
        sp += f"""
* TWO-STAGE RR_CAP COMPARATOR on rr_timing SF (rr_timer)
* Fires at next_R - {abs(rrcap_offset)}ms, before F2 measurement at next_R - 65ms
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

    # Rest of feature circuits
    sp += f"""
* === REMAINING FEATURE CIRCUITS ===
Xbi0 ecg_l0 {rpeak_sh_net} trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 {rpeak_sh_net} trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 {rpeak_sh_net} f8_symmetry vdd vss qrs_symmetry

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

    # Debug: measure rr_timer SF and comparator outputs at key times
    for j in range(skip, min(skip+3, n)):
        if j + 1 < n:
            t_cap = rpeak_times[j+1] - 0.080  # expected capture time
            t_f2m = rpeak_times[j+1] - 0.065  # F2 measurement time
            if t_cap > 0.5:
                sp += f".meas tran dbg_rrsf_cap_{j} FIND V(rr_timer) AT={t_cap:.7f}\n"
                sp += f".meas tran dbg_rrsf_f2_{j} FIND V(rr_timer) AT={t_f2m:.7f}\n"
                sp += f".meas tran dbg_prev_{j} FIND V(xrr.prev_node) AT={t_f2m:.7f}\n"

    sp += "\n.end\n"
    return sp, skip, rpeak_times, total_time


def run_sim(sp_text, label):
    tb = os.path.join(FRONTEND_DIR, f"_iso_v4_{label}.spice")
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


def extract_features(results, beats, skip, n):
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
    return np.array(circuit) if circuit else None, np.array(python) if python else None


def compute_rho(circuit, python):
    if circuit is None or len(circuit) < 4:
        return [0.0] * 8
    rhos = []
    for i in range(8):
        try:
            rho, _ = spearmanr(circuit[:, i], python[:, i])
            rhos.append(abs(rho))
        except:
            rhos.append(0.0)
    return rhos


# ═══════════════════════════════════════════════════════════════════
OUT = os.path.join(FRONTEND_DIR, "_isolation_v4_results.txt")

def log(msg):
    print(msg)
    with open(OUT, 'a') as f: f.write(msg + '\n')

with open(OUT, 'w') as f: f.write('')

log("=" * 70)
log("ISOLATION TEST v4: Comparators on rr_timing SF (0.39V at 800ms)")
log("=" * 70)

log("\n[1] Loading 12 beats from record 100...")
beats, gain, rpeak_threshold = load_beats('100', N_BEATS)
log(f"  Loaded {len(beats)} beats, gain={gain:.6f}, rpeak_th={rpeak_threshold:.4f}")

log("\n[2] Calibrating delay timer AND rr_timing SF curves...")
cal = calibrate_timers()
thresholds = compute_thresholds(cal)
log(f"  Delay timer thresholds: vth_14={thresholds[0]:.5f} vth_28={thresholds[1]:.5f} vth_69={thresholds[2]:.5f} vth_300={thresholds[3]:.5f}")

log(f"\n  RR timing SF curve (ms -> V):")
for t_ms, v in zip(cal['rr_t'], cal['rr_v']):
    log(f"    {t_ms:5d}ms: {v:.5f}V")

# Compute expected pwave thresholds for reference
pipe_pw_rr = 14.0; rpk_drop = 0.5
ewma_rr_ms = beats[2]['ewma_rr'] * 1000
pw_base_d = ewma_rr_ms - 201 - rpk_drop - pipe_pw_rr
pw_peak_d = ewma_rr_ms - 131 - rpk_drop - pipe_pw_rr
rr_cap_d  = ewma_rr_ms - 80  - rpk_drop - pipe_pw_rr
log(f"\n  Expected thresholds (ewma_rr={ewma_rr_ms:.1f}ms):")
log(f"    pw_base: SF({pw_base_d:.0f}ms) = {interp_sf(cal['rr_t'], cal['rr_v'], pw_base_d):.5f}V")
log(f"    pw_peak: SF({pw_peak_d:.0f}ms) = {interp_sf(cal['rr_t'], cal['rr_v'], pw_peak_d):.5f}V")
log(f"    rr_cap:  SF({rr_cap_d:.0f}ms) = {interp_sf(cal['rr_t'], cal['rr_v'], rr_cap_d):.5f}V")

configs = [
    ("A_BASELINE", {},
     "Phase 4 baseline (all ideal triggers)"),

    ("H_RRCAP", {'rrtimer_rrcap': True, 'rrcap_offset': -80},
     "rr_cap comparator on rr_timing SF"),

    ("I_PWAVE", {'rrtimer_pwave': True},
     "pwave comparators on rr_timing SF"),

    ("J_BOTH", {'rrtimer_rrcap': True, 'rrcap_offset': -80, 'rrtimer_pwave': True},
     "Both rr_cap + pwave on rr_timing SF"),

    ("K_FULL", {'rrtimer_rrcap': True, 'rrcap_offset': -80, 'rrtimer_pwave': True,
                'analog_rpeak_dt': True, 'analog_rpeak_rr': True, 'long_rrcopy': True},
     "Full analog (except rpeak_sh ideal)"),
]

fnames = ['F1:rr', 'F2:asym', 'F3:bi0', 'F4:bi1', 'F5:pw', 'F6:wid', 'F7:slp', 'F8:sym']
all_results = {}
all_raw = {}

for i, (label, cfg, desc) in enumerate(configs):
    log(f"\n[{i+3}] Config {label}: {desc}")
    sp, skip, rpk_times, total_time = generate_testbench(
        beats, gain, rpeak_threshold, cfg, cal, thresholds)
    results, elapsed = run_sim(sp, label)
    all_raw[label] = results
    if results is None:
        log(f"  FAILED (no results)")
        all_results[label] = [0.0] * 8
        continue
    circuit, python = extract_features(results, beats, skip, len(beats))
    rhos = compute_rho(circuit, python)
    all_results[label] = rhos
    n_meas = len(circuit) if circuit is not None else 0
    log(f"  {n_meas} beats measured, {elapsed:.0f}s")
    log(f"  rho: {' '.join(f'{r:.3f}' for r in rhos)}")

    # Print debug info for configs with analog rr_cap
    if cfg.get('rrtimer_rrcap') and results:
        for j in range(2, min(5, len(beats))):
            rrsf_cap = results.get(f'dbg_rrsf_cap_{j}')
            rrsf_f2 = results.get(f'dbg_rrsf_f2_{j}')
            prev = results.get(f'dbg_prev_{j}')
            if rrsf_cap is not None:
                log(f"    beat {j}: rr_sf@capture={rrsf_cap:.5f} rr_sf@F2={rrsf_f2:.5f} prev_node@F2={prev:.5f}")

log("\n" + "=" * 70)
log("RESULTS SUMMARY")
log("=" * 70)
header = f"  {'Config':<16s} " + " ".join(f"{f:>7s}" for f in fnames)
log(header)
log("  " + "-" * (16 + 8 * len(fnames)))
baseline = all_results.get('A_BASELINE', [0]*8)
for label, _, _ in configs:
    rhos = all_results.get(label, [0]*8)
    parts = []
    for j, r in enumerate(rhos):
        if label != 'A_BASELINE':
            diff = r - baseline[j]
            if diff > 0.02:
                parts.append(f"{r:.3f}++")
            elif diff < -0.05:
                parts.append(f"{r:.3f}!!")
            elif diff < -0.02:
                parts.append(f"{r:.3f} *")
            else:
                parts.append(f"{r:.3f}  ")
        else:
            parts.append(f"{r:.3f}  ")
    log(f"  {label:<16s} " + " ".join(f"{p:>7s}" for p in parts))

log(f"\n  Baseline: F2={baseline[1]:.3f} F5={baseline[4]:.3f}")
log(f"  Target:   F2>=0.95   F5>=0.95")
