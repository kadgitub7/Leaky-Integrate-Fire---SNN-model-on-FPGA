"""
Diagnostic: measure ACTUAL pwave trigger fire times vs ideal.
Corrected measurement: search for trigger between rpeak[j-1] and rpeak[j].
Tests 3 prediction timer sizes + prints per-beat F5 values.
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
            ewma_rr_before = pre_rr; ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr
        beats.append({
            'pre_rr': pre_rr, 'post_rr': post_rr, 'ewma_rr': ewma_rr,
            'ecg_l0': beat[:, 0].copy(), 'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
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

# Calibration from previous run
rr_t = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
rr_v = [0.4819, 0.4807, 0.4772, 0.4715, 0.4602, 0.4488, 0.4375, 0.4262, 0.4150, 0.4037, 0.3925, 0.3813, 0.3702]
dt_t = [1,2,3,4,5,10,15,20,50,100,200,300,400,500,600,700,800]
dt_v = [0.4823, 0.4817, 0.4812, 0.4807, 0.4802, 0.4776, 0.4749, 0.4723, 0.4620, 0.4466, 0.4107, 0.3750, 0.3396, 0.3044, 0.2696, 0.2350, 0.2008]
ft_t = [1,2,3,5,10,20]
ft_v = [0.4740, 0.4478, 0.4218, 0.3703, 0.2460, 0.0045]

pt_cal = {
    '100pF': {'t': rr_t, 'v': [0.4793, 0.4756, 0.4647, 0.4466, 0.4107, 0.3750, 0.3396, 0.3044, 0.2696, 0.2350, 0.2008, 0.1668, 0.1333]},
    '150pF': {'t': rr_t, 'v': [0.4806, 0.4781, 0.4708, 0.4587, 0.4345, 0.4105, 0.3866, 0.3629, 0.3392, 0.3157, 0.2923, 0.2691, 0.2459]},
    '200pF': {'t': rr_t, 'v': [0.4812, 0.4793, 0.4738, 0.4647, 0.4465, 0.4284, 0.4104, 0.3924, 0.3745, 0.3566, 0.3389, 0.3212, 0.3036]},
}

beats, gain = load_all_beats('100', 20)
n = len(beats)
print("Loaded {} beats, gain={:.4f}".format(n, gain))

rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
for i in range(1, n):
    rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
total_time = rpeak_times[-1] + 0.5

ecg_l0_pts = [(0, VBASE)]
for i, rpt in enumerate(rpeak_times):
    beat_start = rpt - WIN_L / FS
    ecg0 = beats[i]['ecg_l0']
    t_before = beat_start - 0.003
    if t_before > 0.01:
        ecg_l0_pts.append((t_before, VBASE))
    for j in range(len(ecg0)):
        t = beat_start + j / FS
        if t > 0:
            ecg_l0_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg0[j]))))
    t_after = beat_start + len(ecg0) / FS + 0.003
    ecg_l0_pts.append((t_after, VBASE))
ecg_l0_pts.append((total_time, VBASE))

lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
fp = FRONTEND_DIR.replace("\\", "/")

# ===== IDEAL BASELINE =====
print("\n=== Config IDEAL: F5 per-beat values ===")
skip = 2
rpk_drop = 0.5

ideal_pw_base_pwl = make_trig(rpeak_times, -(WIN_L - PW_S)/FS, 0.0005, total_time)
ideal_pw_peak_pwl = make_trig(rpeak_times, -(WIN_L - PW_MID)/FS, 0.0005, total_time)

sp_ideal = """.title ideal_f5_baseline
.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vecg0 ecg_l0 0 PWL({ecg0})
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/pwave_diff.spice"
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
.tran 0.5m {ttot:.4f}
""".format(lib=lib_path, vdd=VDD, fp=fp, ecg0=make_pwl(ecg_l0_pts),
           pwb=ideal_pw_base_pwl, pwp=ideal_pw_peak_pwl, ttot=total_time)
for j in range(skip, n):
    t_f5 = rpeak_times[j] + 0.100
    if t_f5 < total_time - 0.01:
        sp_ideal += ".meas tran f5_{j} FIND V(f5_pwave) AT={t:.7f}\n".format(j=j, t=t_f5)
sp_ideal += "\n.end\n"

tb = os.path.join(FRONTEND_DIR, "_diag_ideal.spice")
logf = tb.replace('.spice', '.log')
with open(tb, 'w') as f: f.write(sp_ideal)
t0 = time.time()
subprocess.run([NGSPICE, '-b', '-o', logf, tb], capture_output=True, text=True, timeout=120, cwd=NGSPICE_DIR)
results_ideal = {}
if os.path.exists(logf):
    with open(logf) as f: text = f.read()
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results_ideal[m.group(1).lower()] = float(m.group(2))
            except: pass
try: os.remove(tb); os.remove(logf)
except: pass

f5_ideal = []
f5_py = []
for j in range(skip, n):
    v = results_ideal.get('f5_{}'.format(j))
    if v is not None:
        f5_ideal.append(v)
        f5_py.append(beats[j]['py_pw'])

if f5_ideal:
    rho, _ = spearmanr(f5_ideal, f5_py)
    print("  F5_rho(ideal) = {:.3f}".format(abs(rho)))
    print("  Beat    Ideal_F5     Py_F5")
    for i, j in enumerate(range(skip, min(skip+len(f5_ideal), n))):
        print("  {:>3d}    {:.5f}    {:.5f}".format(j, f5_ideal[i], f5_py[i]))

# ===== PREDICTION TIMER TESTS =====
print("\n=== Prediction Timer Pwave Tests ===")
for pt_name, pt_data, cap_w, cap_l in [
    ('PT-200pF', pt_cal['200pF'], 316, 316),
]:
    pt_t_arr = pt_data['t']
    pt_v_arr = pt_data['v']

    for pipe_pw in [3.0, 5.0, 8.0, 12.0, 16.0]:
        pwl_base_pts = [(0, 0.3)]
        pwl_peak_pts = [(0, 0.3)]
        for j in range(n):
            ewma_rr_ms = beats[j]['ewma_rr'] * 1000
            pw_base_d = max(1, ewma_rr_ms - 206 - rpk_drop - pipe_pw)
            pw_peak_d = max(1, ewma_rr_ms - 131 - rpk_drop - pipe_pw)
            v_base = interp_sf(pt_t_arr, pt_v_arr, pw_base_d)
            v_peak = interp_sf(pt_t_arr, pt_v_arr, pw_peak_d)
            t_step = rpeak_times[j] + 0.001
            if t_step > pwl_base_pts[-1][0]:
                pwl_base_pts.append((t_step, v_base))
                pwl_peak_pts.append((t_step, v_peak))
        pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
        pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))

        sp = """.title diag_{pn}_pipe{pw}
.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vecg0 ecg_l0 0 PWL({ecg0})
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/edge_buffer.spice"
.include "{fp}/pwave_diff.spice"
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3
Vrpk rpeak_raw 0 PWL({rpk})
Xinv_rpk rpeak_raw rpk_bar vdd vss inverter_sub
XMpt_chg pt_cap rpk_bar vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=2 nf=1
XMpt_dis pt_cap vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XCpt pt_cap vss sky130_fd_pr__cap_mim_m3_1 W={cw} L={cl}
XMpt_sf vdd pt_cap pt_sf vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMpt_sfb pt_sf vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
Vth_pwb vth_pw_base 0 PWL({pwb})
Vth_pwp vth_pw_peak 0 PWL({pwp})
XMcb1 cnb pt_sf ctb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcb2 s1b vth_pw_base ctb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcb3 cnb cnb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcb4 s1b cnb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcb5 ctb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcb6 lev_b s1b vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcb7 lev_b s1b vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xebb lev_b sharp_b vdd vss edge_buffer
Xpgb sharp_b trig_pw_base vdd vss vbias pulse_gen
XMcp1 cnp pt_sf ctp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcp2 s1p vth_pw_peak ctp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcp3 cnp cnp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcp4 s1p cnp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcp5 ctp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1
XMcp6 lev_p s1p vdd vdd sky130_fd_pr__pfet_01v8 L=0.5 W=8 nf=1
XMcp7 lev_p s1p vss vss sky130_fd_pr__nfet_01v8 L=0.5 W=4 nf=1
Xebp lev_p sharp_p vdd vss edge_buffer
Xpgp sharp_p trig_pw_peak vdd vss vbias pulse_gen
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
.tran 0.5m {ttot:.4f}
""".format(pn=pt_name, pw=pipe_pw, lib=lib_path, vdd=VDD, fp=fp,
           ecg0=make_pwl(ecg_l0_pts), rpk=make_trig(rpeak_times, 0, 0.0005, total_time),
           cw=cap_w, cl=cap_l,
           pwb=make_pwl(pwl_base_pts), pwp=make_pwl(pwl_peak_pts),
           ttot=total_time)

        for j in range(skip, n):
            t_f5 = rpeak_times[j] + 0.100
            if t_f5 < total_time - 0.01:
                sp += ".meas tran f5_{j} FIND V(f5_pwave) AT={t:.7f}\n".format(j=j, t=t_f5)
            if j > 0:
                t_search = rpeak_times[j-1] + 0.3
                sp += ".meas tran tb_{j} WHEN V(trig_pw_base)=0.45 RISE=1 TD={ts:.7f}\n".format(j=j, ts=t_search)
                sp += ".meas tran tp_{j} WHEN V(trig_pw_peak)=0.45 RISE=1 TD={ts:.7f}\n".format(j=j, ts=t_search)
        sp += "\n.end\n"

        tb = os.path.join(FRONTEND_DIR, "_diag_timing.spice")
        logf = tb.replace('.spice', '.log')
        with open(tb, 'w') as f: f.write(sp)
        t0 = time.time()
        subprocess.run([NGSPICE, '-b', '-o', logf, tb], capture_output=True, text=True, timeout=300, cwd=NGSPICE_DIR)
        elapsed = time.time() - t0
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

        f5_c, f5_p = [], []
        for j in range(skip, n):
            f5 = results.get('f5_{}'.format(j))
            if f5 is not None:
                f5_c.append(f5)
                f5_p.append(beats[j]['py_pw'])
        if len(f5_c) >= 4:
            rho, _ = spearmanr(f5_c, f5_p)
            f5_rho = abs(rho)
        else:
            f5_rho = 0.0

        base_err = []
        peak_err = []
        for j in range(skip, n):
            tb_fire = results.get('tb_{}'.format(j))
            tp_fire = results.get('tp_{}'.format(j))
            rpt = rpeak_times[j]
            if tb_fire is not None:
                base_err.append((tb_fire - (rpt - (WIN_L - PW_S)/FS)) * 1000)
            if tp_fire is not None:
                peak_err.append((tp_fire - (rpt - (WIN_L - PW_MID)/FS)) * 1000)

        print("\n  {} pipe={:.0f}ms: F5_rho={:.3f} ({:.0f}s)".format(pt_name, pipe_pw, f5_rho, elapsed))
        if base_err:
            print("    Base err: mean={:.1f}ms std={:.1f}ms range=[{:.1f},{:.1f}]".format(
                np.mean(base_err), np.std(base_err), min(base_err), max(base_err)))
        else:
            print("    Base: NO FIRES")
        if peak_err:
            print("    Peak err: mean={:.1f}ms std={:.1f}ms range=[{:.1f},{:.1f}]".format(
                np.mean(peak_err), np.std(peak_err), min(peak_err), max(peak_err)))
        else:
            print("    Peak: NO FIRES")

        # Print per-beat for pipe=5
        if pipe_pw == 5.0:
            print("\n    Beat  Ideal_F5  Analog_F5  Py_F5    Base_err  Peak_err")
            idx = 0
            for j in range(skip, n):
                fi = f5_ideal[idx] if idx < len(f5_ideal) else 0
                fa = results.get('f5_{}'.format(j), 0)
                py = beats[j]['py_pw']
                tb_fire = results.get('tb_{}'.format(j))
                tp_fire = results.get('tp_{}'.format(j))
                rpt = rpeak_times[j]
                be = "{:>+7.1f}".format((tb_fire - (rpt - (WIN_L-PW_S)/FS))*1000) if tb_fire else "  N/A  "
                pe = "{:>+7.1f}".format((tp_fire - (rpt - (WIN_L-PW_MID)/FS))*1000) if tp_fire else "  N/A  "
                print("    {:>3d}   {:.5f}   {:.5f}   {:.5f}  {}  {}".format(j, fi, fa, py, be, pe))
                idx += 1
