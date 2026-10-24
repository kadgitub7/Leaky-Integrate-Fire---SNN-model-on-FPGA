"""
Step-by-step analog replacement diagnostic — COHERENT GROUPS.
Replace timing-coherent groups together (not individual triggers).

Config A: Baseline — all ideal triggers, analog bias_gen only
Config B: All rpeak-derived triggers analog:
          rpeak_detector + delay_timer (s14, s69, wid, bi_copy, rr_copy)
          Ideal: rr_cap, pwave
Config C: B + hold-and-copy rr_cap (rr_timing_v6)
          Ideal: pwave only
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
            ewma_rr_before = pre_rr
            ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr
        prev_bs = valid[i-1][0] - WIN_L
        prev_be = valid[i-1][0] + WIN_R
        if prev_bs >= 0 and prev_be < len(sig):
            prev_beat_data = sig[prev_bs:prev_be]
            py_bi_l0 = beat[WIN_L, 0] - prev_beat_data[WIN_L, 0]
            py_bi_l1 = beat[WIN_L, min(1,n_leads-1)] - prev_beat_data[WIN_L, min(1,n_leads-1)]
        else:
            py_bi_l0 = py_bi_l1 = 0.0
        beats.append({
            'pre_rr': pre_rr, 'post_rr': post_rr, 'ewma_rr': ewma_rr,
            'ecg_l0': beat[:, 0].copy(), 'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': pre_rr / ewma_rr_before if ewma_rr_before > 0 else 1.0,
            'py_f2': pre_rr / post_rr if post_rr > 0 else 1.0,
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[WIN_L + 10, 0],
            'py_slp': beat[WIN_L + 5, 0] - beat[WIN_L + 25, 0],
            'py_sym': beat[WIN_L, 0],
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

rr_t = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
rr_v = [0.4819, 0.4807, 0.4772, 0.4715, 0.4602, 0.4488, 0.4375, 0.4262, 0.4150, 0.4037, 0.3925, 0.3813, 0.3702]
dt_t = [1,2,3,4,5,10,15,20,50,100,200,300,400,500,600,700,800]
dt_v = [0.4823, 0.4817, 0.4812, 0.4807, 0.4802, 0.4776, 0.4749, 0.4723, 0.4620, 0.4466, 0.4107, 0.3750, 0.3396, 0.3044, 0.2696, 0.2350, 0.2008]
ft_t = [1,2,3,5,10,20]
ft_v = [0.4740, 0.4478, 0.4218, 0.3703, 0.2460, 0.0045]

# ---- Analysis functions ----
def get_feature_values(results, skip, n):
    feat_vals = {}
    for fk in range(1, 9):
        vals = []; idxs = []
        for j in range(skip, n):
            v = results.get('f{}_{}'.format(fk, j))
            if v is not None:
                vals.append(v); idxs.append(j)
        feat_vals[fk] = (idxs, vals)
    return feat_vals

def compute_rho_vs_python(feat_vals, beats):
    rhos = {}
    py_keys = {1:'py_f1', 2:'py_f2', 3:'py_bi_l0', 4:'py_bi_l1',
               5:'py_pw', 6:'py_wid', 7:'py_slp', 8:'py_sym'}
    for fk in range(1, 9):
        idxs, circ = feat_vals[fk]
        if len(circ) < 4: rhos[fk] = 0.0; continue
        pyth = [beats[j][py_keys[fk]] for j in idxs]
        rho, _ = spearmanr(circ, pyth)
        rhos[fk] = abs(rho)
    return rhos

def compute_cross_rho(feat_a, feat_b):
    rhos = {}
    for fk in range(1, 9):
        idxs_a, vals_a = feat_a[fk]
        idxs_b, vals_b = feat_b[fk]
        common = sorted(set(idxs_a) & set(idxs_b))
        if len(common) < 4: rhos[fk] = 0.0; continue
        a_map = dict(zip(idxs_a, vals_a))
        b_map = dict(zip(idxs_b, vals_b))
        rho, _ = spearmanr([a_map[j] for j in common], [b_map[j] for j in common])
        rhos[fk] = abs(rho)
    return rhos

def fmt_rhos(rhos):
    return " ".join("F{}={:.3f}".format(k, rhos.get(k,0)) for k in range(1,9))

# ---- Load data ----
beats, gain = load_all_beats('100', 30)
n = len(beats)
max_r_v = max(VBASE + gain * b['ecg_l0'][WIN_L] for b in beats)
rpeak_thresh = VBASE + 0.5 * (max_r_v - VBASE)
print("Loaded {} beats, gain={:.4f}, rpeak_thresh={:.4f}V".format(n, gain, rpeak_thresh))

rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
for i in range(1, n):
    rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
total_time = rpeak_times[-1] + 0.5

# ECG PWL
ecg_l0_pts = [(0, VBASE)]; ecg_l1_pts = [(0, VBASE)]
for i, rpt in enumerate(rpeak_times):
    bs_t = rpt - WIN_L / FS
    for lead, pts_list in [(0, ecg_l0_pts), (1, ecg_l1_pts)]:
        ecg = beats[i]['ecg_l0'] if lead == 0 else beats[i]['ecg_l1']
        t_before = bs_t - 0.003
        if t_before > 0.01: pts_list.append((t_before, VBASE))
        for j in range(len(ecg)):
            t = bs_t + j / FS
            if t > 0: pts_list.append((t, max(0.05, min(0.85, VBASE + gain * ecg[j]))))
        pts_list.append((bs_t + len(ecg)/FS + 0.003, VBASE))
ecg_l0_pts.append((total_time, VBASE)); ecg_l1_pts.append((total_time, VBASE))

lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
fp = FRONTEND_DIR.replace("\\", "/")

rpk_drop = 0.5
pipe_s14=2.9; pipe_wid=4.5; pipe_s69=4.4; pipe_bi=5.9; pipe_s10=1.8
vth_14 = interp_sf(ft_t, ft_v, max(0.3, 14-rpk_drop-pipe_s14))
vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28-rpk_drop-pipe_wid))
vth_69 = interp_sf(dt_t, dt_v, max(0.5, 69-rpk_drop-pipe_s69))
vth_300= interp_sf(dt_t, dt_v, max(0.5, 300-rpk_drop-pipe_bi))
vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10-rpk_drop-pipe_s10))
first_rr_ms = beats[0]['pre_rr'] * 1000
ic_ewma = interp_sf(rr_t, rr_v, first_rr_ms)
PW = 0.0005

# Ideal trigger PWLs
ideal_rpk = make_trig(rpeak_times, 0, PW, total_time)
ideal_rr_cap = make_trig(rpeak_times, -0.005, 0.055, total_time)
ideal_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S)/FS, PW, total_time)
ideal_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID)/FS, PW, total_time)
ideal_s14 = make_trig(rpeak_times, 5.0/FS, PW, total_time)
ideal_s69 = make_trig(rpeak_times, 25.0/FS, PW, total_time)
ideal_wid = make_trig(rpeak_times, 10.0/FS, PW, total_time)
ideal_bi_copy = make_trig(rpeak_times, 0.300, 0.002, total_time)
ideal_rr_copy = make_trig(rpeak_times, 0.010, 0.002, total_time)
# For Config C: capture at R+5ms for hold-copy rr_cap
ideal_cap_post = make_trig(rpeak_times, 0.005, 0.002, total_time)

SKIP = 5

def common_header():
    return """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vecg0 ecg_l0 0 PWL({ecg0})
Vecg1 ecg_l1 0 PWL({ecg1})
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/edge_buffer.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/long_pulse_gen.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3
""".format(lib=lib_path, vdd=VDD, fp=fp,
           ecg0=make_pwl(ecg_l0_pts), ecg1=make_pwl(ecg_l1_pts))

def meas_block(skip, n, rpeak_times, total_time):
    sp = ""
    for j in range(skip, n):
        t_f1 = rpeak_times[j] - 0.010
        if t_f1 > 0.5:
            sp += ".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t:.7f}\n".format(j=j, t=t_f1)
        if j+1 < n:
            t_f2 = rpeak_times[j+1] - 0.065
            if t_f2 > 0.5:
                sp += ".meas tran f2_{j} FIND V(f2_rr_asym) AT={t:.7f}\n".format(j=j, t=t_f2)
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            for fk, nm in [(3,'f3_bi_l0'),(4,'f4_bi_l1'),(5,'f5_pwave'),
                           (6,'f6_width'),(7,'f7_slope'),(8,'f8_symmetry')]:
                sp += ".meas tran f{k}_{j} FIND V({nm}) AT={t:.7f}\n".format(k=fk, j=j, nm=nm, t=t_f38)
    return sp

def run_sim(title, sp_body):
    sp = ".title {}\n".format(title) + sp_body + "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_diag_step.spice")
    logf = tb.replace('.spice', '.log')
    with open(tb, 'w') as f: f.write(sp)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', logf, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    results = {}
    if os.path.exists(logf):
        with open(logf) as f: text = f.read()
        for line in text.split('\n'):
            m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
            if m:
                try: results[m.group(1).lower()] = float(m.group(2))
                except: pass
    try: os.remove(tb)
    except: pass
    return results, elapsed


# ===== CONFIG A: All ideal triggers =====
print("\n" + "="*60)
print("CONFIG A: Baseline — all ideal triggers")
print("="*60)
sp_a = common_header()
sp_a += '.include "{fp}/rr_timing.spice"\n'.format(fp=fp)
sp_a += """
Vrpk rpk_ideal 0 PWL({rpk})
Vs14 trig_s14 0 PWL({s14})
Vs69 trig_s69 0 PWL({s69})
Vwid trig_wid 0 PWL({wid})
Vbi  trig_bi_raw 0 PWL({bi})
Vrrc trig_rr_copy 0 PWL({rrc})
Vcap trig_rr_cap 0 PWL({cap})
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})

Xrr ecg_l0 trig_rr_cap rpk_ideal trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 rpk_ideal trig_bi_raw f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 rpk_ideal trig_bi_raw f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 rpk_ideal f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xbi0.prev_node)=0.30
.ic V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30
.ic V(xbi1.esamp_node)=0.30

.tran 0.5m {ttot:.4f}
""".format(rpk=ideal_rpk, s14=ideal_s14, s69=ideal_s69, wid=ideal_wid,
           bi=ideal_bi_copy, rrc=ideal_rr_copy, cap=ideal_rr_cap,
           pwb=ideal_pw_base, pwp=ideal_pw_peak,
           ic=ic_ewma, ttot=total_time)
sp_a += meas_block(SKIP, n, rpeak_times, total_time)

res_a, t_a = run_sim("config_A_all_ideal", sp_a)
feat_a = get_feature_values(res_a, SKIP, n)
rho_a = compute_rho_vs_python(feat_a, beats)
print("  Sim time: {:.0f}s, {} measurements".format(t_a, len(res_a)))
print("  Abs rho: " + fmt_rhos(rho_a))


# ===== CONFIG B: All rpeak-derived triggers analog =====
print("\n" + "="*60)
print("CONFIG B: All rpeak-derived analog (rpeak + DT + s14/s69/wid/bi/rr_copy)")
print("="*60)
sp_b = common_header()
sp_b += '.include "{fp}/rr_timing.spice"\n'.format(fp=fp)
sp_b += '.include "{fp}/rpeak_detector.spice"\n'.format(fp=fp)
sp_b += '.include "{fp}/delay_timer.spice"\n'.format(fp=fp)
sp_b += """
* Analog rpeak detector
Vrpk_thresh rpeak_threshold 0 {thresh:.4f}
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

* Analog delay timer — driven by analog rpeak
Xdelay rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_copy_lev dt_sf_out vdd vss vbias {vth14:.6f} {vth69:.6f} {vth28:.6f} {vth300:.6f} {vth10:.6f} delay_timer
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp trig_wid vdd vss vbias pulse_gen
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp trig_s69 vdd vss vbias pulse_gen
Xeb_bi dt_bi_lev bi_sharp vdd vss edge_buffer
Xpg_bi bi_sharp trig_bi_raw vdd vss vbias long_pulse_gen
Xeb_rrc dt_rr_copy_lev rrc_sharp vdd vss edge_buffer
Xpg_rrc rrc_sharp trig_rr_copy vdd vss vbias long_pulse_gen
Xeb_s14 dt_s14_lev s14_sharp vdd vss edge_buffer
Xpg_s14 s14_sharp trig_s14 vdd vss vbias pulse_gen

* Still ideal: rr_cap, pwave
Vcap trig_rr_cap 0 PWL({cap})
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})

Xrr ecg_l0 trig_rr_cap rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 rpeak_pulse trig_bi_raw f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 rpeak_pulse trig_bi_raw f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 rpeak_pulse f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xbi0.prev_node)=0.30
.ic V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30
.ic V(xbi1.esamp_node)=0.30

.tran 0.5m {ttot:.4f}
""".format(thresh=rpeak_thresh, cap=ideal_rr_cap, pwb=ideal_pw_base, pwp=ideal_pw_peak,
           vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10,
           ic=ic_ewma, ttot=total_time)
sp_b += meas_block(SKIP, n, rpeak_times, total_time)

res_b, t_b = run_sim("config_B_analog_rpeak_dt", sp_b)
feat_b = get_feature_values(res_b, SKIP, n)
rho_b = compute_rho_vs_python(feat_b, beats)
xrho_b = compute_cross_rho(feat_a, feat_b)
print("  Sim time: {:.0f}s, {} measurements".format(t_b, len(res_b)))
print("  Abs rho: " + fmt_rhos(rho_b))
print("  vs A:    " + fmt_rhos(xrho_b))


# ===== CONFIG C: B + hold-and-copy rr_cap =====
print("\n" + "="*60)
print("CONFIG C: B + hold-copy rr_cap (rr_timing_v6)")
print("  capture=ideal R+5ms, copy=analog rr_copy R+10ms")
print("="*60)
sp_c = common_header()
sp_c += '.include "{fp}/rr_timing_v6.spice"\n'.format(fp=fp)
sp_c += '.include "{fp}/rpeak_detector.spice"\n'.format(fp=fp)
sp_c += '.include "{fp}/delay_timer.spice"\n'.format(fp=fp)
sp_c += """
* Analog rpeak detector
Vrpk_thresh rpeak_threshold 0 {thresh:.4f}
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

* Analog delay timer
Xdelay rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_copy_lev dt_sf_out vdd vss vbias {vth14:.6f} {vth69:.6f} {vth28:.6f} {vth300:.6f} {vth10:.6f} delay_timer
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp trig_wid vdd vss vbias pulse_gen
Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp trig_s69 vdd vss vbias pulse_gen
Xeb_bi dt_bi_lev bi_sharp vdd vss edge_buffer
Xpg_bi bi_sharp trig_bi_raw vdd vss vbias long_pulse_gen
Xeb_rrc dt_rr_copy_lev rrc_sharp vdd vss edge_buffer
Xpg_rrc rrc_sharp trig_rr_copy vdd vss vbias long_pulse_gen
Xeb_s14 dt_s14_lev s14_sharp vdd vss edge_buffer
Xpg_s14 s14_sharp trig_s14 vdd vss vbias pulse_gen

* Capture at R+5ms (ideal — in ASIC: fast-timer comparator)
Vcap_post trig_cap_post 0 PWL({cap_post})
* Pwave still ideal
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})

* rr_timing_v6: hold latches at rpeak, capture copies hold at R+5ms, EWMA at R+10ms
Xrr ecg_l0 trig_cap_post rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing_v6
Xbi0 ecg_l0 rpeak_pulse trig_bi_raw f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 rpeak_pulse trig_bi_raw f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 rpeak_pulse f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xrr.hold_node)={ic:.6f}
.ic V(xbi0.prev_node)=0.30
.ic V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30
.ic V(xbi1.esamp_node)=0.30

.tran 0.5m {ttot:.4f}
""".format(thresh=rpeak_thresh, cap_post=ideal_cap_post, pwb=ideal_pw_base, pwp=ideal_pw_peak,
           vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10,
           ic=ic_ewma, ttot=total_time)
sp_c += meas_block(SKIP, n, rpeak_times, total_time)

res_c, t_c = run_sim("config_C_hold_copy", sp_c)
feat_c = get_feature_values(res_c, SKIP, n)
rho_c = compute_rho_vs_python(feat_c, beats)
xrho_c = compute_cross_rho(feat_a, feat_c)
xrho_cb = compute_cross_rho(feat_b, feat_c)
print("  Sim time: {:.0f}s, {} measurements".format(t_c, len(res_c)))
print("  Abs rho: " + fmt_rhos(rho_c))
print("  vs A:    " + fmt_rhos(xrho_c))
print("  vs B:    " + fmt_rhos(xrho_cb))


# ===== SUMMARY =====
print("\n" + "="*78)
print("SUMMARY")
print("="*78)
print("\nAbsolute rho (vs Python):")
print("  {:>42s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
    "", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"))
for name, rhos in [("A: all ideal", rho_a),
                   ("B: analog rpeak+DT (ideal rr_cap/pw)", rho_b),
                   ("C: B + hold-copy rr_cap (ideal pw)", rho_c)]:
    print("  {:>42s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        name, *[rhos.get(k,0) for k in range(1,9)]))

print("\nCross-config rho (1.000 = identical to reference):")
for name, xrhos in [("B vs A (rpeak+DT effect)", xrho_b),
                    ("C vs A (rpeak+DT+rr_cap effect)", xrho_c),
                    ("C vs B (rr_cap-only effect)", xrho_cb)]:
    print("  {:>42s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        name, *[xrhos.get(k,0) for k in range(1,9)]))

# Debug: raw values
print("\nDebug: F1 raw values (A vs B vs C), first 8 beats:")
for j in range(SKIP, min(SKIP+8, n)):
    va = res_a.get('f1_{}'.format(j), float('nan'))
    vb = res_b.get('f1_{}'.format(j), float('nan'))
    vc = res_c.get('f1_{}'.format(j), float('nan'))
    print("  beat {:2d}: A={:.5f}  B={:.5f}  C={:.5f}  py={:.4f}".format(
        j, va, vb, vc, beats[j]['py_f1']))

print("\nDebug: F8 raw values (A vs B), first 8 beats:")
for j in range(SKIP, min(SKIP+8, n)):
    va = res_a.get('f8_{}'.format(j), float('nan'))
    vb = res_b.get('f8_{}'.format(j), float('nan'))
    print("  beat {:2d}: A={:.5f}  B={:.5f}  py={:.6f}".format(
        j, va, vb, beats[j]['py_sym']))

print("\nIdeal remaining: pwave triggers (F5)")
print("Ideal in C: capture_trig R+5ms (needs fast-timer comparator in ASIC)")
