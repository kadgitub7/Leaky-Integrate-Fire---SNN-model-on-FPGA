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
    # First pass: collect raw beat data (gain computed after from ONLY these beats)
    raw_beats = []
    ewma_rr = None
    for i in range(1, len(valid) - 1):
        if len(raw_beats) >= max_beats: break
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
        raw_beats.append({
            'pre_rr': pre_rr, 'post_rr': post_rr, 'ewma_rr': ewma_rr,
            'ewma_rr_before': ewma_rr_before,
            'ecg_l0': beat[:, 0].copy(), 'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
            'rpeak_raw': abs(beat[WIN_L, 0]),
        })
    # Compute gain from only the beats we loaded
    max_peak = max(b['rpeak_raw'] for b in raw_beats) if raw_beats else 1.0
    gain = 0.16 / max_peak
    # Second pass: compute Python references (don't depend on gain)
    beats = []
    for b in raw_beats:
        beats.append({
            'pre_rr': b['pre_rr'], 'post_rr': b['post_rr'], 'ewma_rr': b['ewma_rr'],
            'ecg_l0': b['ecg_l0'], 'ecg_l1': b['ecg_l1'],
            'py_f1': b['pre_rr'] / b['ewma_rr_before'] if b['ewma_rr_before'] > 0 else 1.0,
            'py_f2': b['pre_rr'] / b['post_rr'] if b['post_rr'] > 0 else 1.0,
            'py_bi_l0': b['py_bi_l0'], 'py_bi_l1': b['py_bi_l1'],
            'py_pw': b['ecg_l0'][PW_MID] - b['ecg_l0'][PW_S],
            'py_wid': b['ecg_l0'][WIN_L + 10],
            'py_slp': b['ecg_l0'][WIN_L + 5] - b['ecg_l0'][WIN_L + 25],
            'py_sym': b['ecg_l0'][WIN_L],
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
rpeak_thresh = VBASE + 0.30 * (max_r_v - VBASE)
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


# ===== CONFIG D: Analog DT only (ideal rpeak) — isolates delay_timer =====
print("\n" + "="*60)
print("CONFIG D: Analog delay_timer ONLY (ideal rpeak drives DT)")
print("  Tests F6(wid), F7(s14/s69), F3/F4(bi_copy), F1/F2(rr_copy)")
print("="*60)
sp_d = common_header()
sp_d += '.include "{fp}/rr_timing.spice"\n'.format(fp=fp)
sp_d += '.include "{fp}/delay_timer.spice"\n'.format(fp=fp)
sp_d += """
* Ideal rpeak pulse drives analog delay timer
Vrpk rpk_ideal 0 PWL({rpk})

* Threshold voltage sources for delay_timer
Vth14_d vth_14_d 0 {vth14:.6f}
Vth69_d vth_69_d 0 {vth69:.6f}
Vth28_d vth_28_d 0 {vth28:.6f}
Vth300_d vth_300_d 0 {vth300:.6f}
Vth10_d vth_10_d 0 {vth10:.6f}
Xdelay rpk_ideal dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_copy_lev dt_sf_out vdd vss vbias vth_14_d vth_69_d vth_28_d vth_300_d vth_10_d delay_timer
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

* All feature circuits use ideal rpeak, analog DT triggers
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
""".format(rpk=ideal_rpk, cap=ideal_rr_cap, pwb=ideal_pw_base, pwp=ideal_pw_peak,
           vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10,
           ic=ic_ewma, ttot=total_time)
sp_d += meas_block(SKIP, n, rpeak_times, total_time)

res_d, t_d = run_sim("config_D_analog_dt_only", sp_d)
feat_d = get_feature_values(res_d, SKIP, n)
rho_d = compute_rho_vs_python(feat_d, beats)
xrho_d = compute_cross_rho(feat_a, feat_d)
print("  Sim time: {:.0f}s, {} measurements".format(t_d, len(res_d)))
print("  Abs rho: " + fmt_rhos(rho_d))
print("  vs A:    " + fmt_rhos(xrho_d))


# ===== CONFIG E: Analog rpeak only (ideal DT) — isolates rpeak timing =====
print("\n" + "="*60)
print("CONFIG E: Analog rpeak ONLY (ideal DT triggers)")
print("  Tests rpeak timing effect on F3/F4(bi), F8(sym)")
print("="*60)
sp_e = common_header()
sp_e += '.include "{fp}/rr_timing.spice"\n'.format(fp=fp)
sp_e += '.include "{fp}/rpeak_detector.spice"\n'.format(fp=fp)
sp_e += """
* Analog rpeak detector
Vrpk_thresh rpeak_threshold 0 {thresh:.4f}
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

* All DT-derived triggers still ideal
Vs14 trig_s14 0 PWL({s14})
Vs69 trig_s69 0 PWL({s69})
Vwid trig_wid 0 PWL({wid})
Vbi  trig_bi_raw 0 PWL({bi})
Vrrc trig_rr_copy 0 PWL({rrc})
Vcap trig_rr_cap 0 PWL({cap})
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})

* rpeak_pulse used for: F3/F4 S&H capture, F8 S&H, rr_timing rpeak_trigger
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
""".format(thresh=rpeak_thresh, s14=ideal_s14, s69=ideal_s69, wid=ideal_wid,
           bi=ideal_bi_copy, rrc=ideal_rr_copy, cap=ideal_rr_cap,
           pwb=ideal_pw_base, pwp=ideal_pw_peak,
           ic=ic_ewma, ttot=total_time)
sp_e += meas_block(SKIP, n, rpeak_times, total_time)

res_e, t_e = run_sim("config_E_analog_rpeak_only", sp_e)
feat_e = get_feature_values(res_e, SKIP, n)
rho_e = compute_rho_vs_python(feat_e, beats)
xrho_e = compute_cross_rho(feat_a, feat_e)
print("  Sim time: {:.0f}s, {} measurements".format(t_e, len(res_e)))
print("  Abs rho: " + fmt_rhos(rho_e))
print("  vs A:    " + fmt_rhos(xrho_e))


# ===== CONFIG F: Analog rpeak + peak delay compensator =====
print("\n" + "="*60)
print("CONFIG F: Analog rpeak + peak delay timer (~6ms compensation)")
print("  rpeak_pulse -> rr_timing precharge")
print("  refined_rpeak -> F3/F4/F8 S&H captures")
print("="*60)
sp_f = common_header()
sp_f += '.include "{fp}/rr_timing.spice"\n'.format(fp=fp)
sp_f += '.include "{fp}/rpeak_detector.spice"\n'.format(fp=fp)
sp_f += """
* Analog rpeak detector
Vrpk_thresh rpeak_threshold 0 {thresh:.4f}
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

* Peak delay timer: 1pF cap discharges from VDD after rpeak
* Delay ~ C*dV/I = 1pF * 0.45V / 80pA ~ 5.6ms
* Inverter switches at VDD/2, fires ~6ms after rpeak_pulse
XCpk_timer pk_timer_node vss sky130_fd_pr__cap_mim_m3_1 W=23 L=23
Xinv_rpk rpeak_pulse rpk_pulse_bar vdd vss inverter_sub
XMpk_charge pk_timer_node rpk_pulse_bar vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMpk_drain pk_timer_node vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.84 nf=1
* Single inverter: timer HIGH -> LOW, timer falls past VDD/2 -> HIGH (triggers edge_buffer)
Xinv_pk1 pk_timer_node pk_timer_lev vdd vss inverter_sub
Xeb_rpk pk_timer_lev rpk_refined_sharp vdd vss edge_buffer
Xpg_rpk_ref rpk_refined_sharp refined_rpeak vdd vss vbias pulse_gen

* DT triggers still ideal
Vs14 trig_s14 0 PWL({s14})
Vs69 trig_s69 0 PWL({s69})
Vwid trig_wid 0 PWL({wid})
Vbi  trig_bi_raw 0 PWL({bi})
Vrrc trig_rr_copy 0 PWL({rrc})
Vcap trig_rr_cap 0 PWL({cap})
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})

* rpeak_pulse for rr_timing precharge; refined_rpeak for F3/F4/F8 S&H
Xrr ecg_l0 trig_rr_cap rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 refined_rpeak trig_bi_raw f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 refined_rpeak trig_bi_raw f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 refined_rpeak f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xbi0.prev_node)=0.30
.ic V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30
.ic V(xbi1.esamp_node)=0.30

.tran 0.5m {ttot:.4f}
""".format(thresh=rpeak_thresh, s14=ideal_s14, s69=ideal_s69, wid=ideal_wid,
           bi=ideal_bi_copy, rrc=ideal_rr_copy, cap=ideal_rr_cap,
           pwb=ideal_pw_base, pwp=ideal_pw_peak,
           ic=ic_ewma, ttot=total_time)
sp_f += meas_block(SKIP, n, rpeak_times, total_time)

res_f, t_f = run_sim("config_F_peak_delay", sp_f)
feat_f = get_feature_values(res_f, SKIP, n)
rho_f = compute_rho_vs_python(feat_f, beats)
xrho_f = compute_cross_rho(feat_a, feat_f)
print("  Sim time: {:.0f}s, {} measurements".format(t_f, len(res_f)))
print("  Abs rho: " + fmt_rhos(rho_f))
print("  vs A:    " + fmt_rhos(xrho_f))

# Debug: F8 with peak delay
print("\nDebug: F8(sym) raw — A vs E(no delay) vs F(peak delay), first 8 beats:")
for j in range(SKIP, min(SKIP+8, n)):
    va = res_a.get('f8_{}'.format(j), float('nan'))
    ve = res_e.get('f8_{}'.format(j), float('nan'))
    vf = res_f.get('f8_{}'.format(j), float('nan'))
    print("  beat {:2d}: A={:.5f}  E={:.5f}  F={:.5f}  py={:.6f}".format(
        j, va, ve, vf, beats[j]['py_sym']))


# ===== SUMMARY =====
print("\n" + "="*78)
print("SUMMARY — ISOLATION TESTS")
print("="*78)
print("\nAbsolute rho (vs Python):")
print("  {:>42s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
    "", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"))
for name, rhos in [("A: all ideal (baseline)", rho_a),
                   ("D: analog DT only (ideal rpeak)", rho_d),
                   ("E: analog rpeak only (ideal DT)", rho_e),
                   ("F: rpeak + peak delay (~6ms)", rho_f)]:
    print("  {:>42s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        name, *[rhos.get(k,0) for k in range(1,9)]))

print("\nCross-config rho vs A (1.000 = identical to baseline):")
print("  {:>42s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
    "", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"))
for name, xrhos in [("D vs A (DT-only effect)", xrho_d),
                    ("E vs A (rpeak-only effect)", xrho_e),
                    ("F vs A (rpeak+delay effect)", xrho_f)]:
    print("  {:>42s} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        name, *[xrhos.get(k,0) for k in range(1,9)]))

# Debug: F6/F7 raw (DT test)
print("\nDebug: F6(wid) raw values — A vs D, first 8 beats:")
for j in range(SKIP, min(SKIP+8, n)):
    va = res_a.get('f6_{}'.format(j), float('nan'))
    vd = res_d.get('f6_{}'.format(j), float('nan'))
    print("  beat {:2d}: A={:.5f}  D={:.5f}  py={:.6f}".format(
        j, va, vd, beats[j]['py_wid']))

print("\nDebug: F8(sym) raw values — A vs E, first 8 beats:")
for j in range(SKIP, min(SKIP+8, n)):
    va = res_a.get('f8_{}'.format(j), float('nan'))
    ve = res_e.get('f8_{}'.format(j), float('nan'))
    print("  beat {:2d}: A={:.5f}  E={:.5f}  py={:.6f}".format(
        j, va, ve, beats[j]['py_sym']))
