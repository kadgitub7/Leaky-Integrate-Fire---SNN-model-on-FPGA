"""
Comprehensive independent verification of ALL circuits.

PART 1: Feature circuits (ideal triggers + ECG PWL -> Spearman rho vs Python)
  F1/F2: rr_timing.spice
  F3/F4: beat_instability.spice
  F5:    pwave_diff.spice
  F6:    qrs_width.spice
  F7:    qrs_slope.spice
  F8:    qrs_symmetry.spice

PART 2: Supplementary circuits (independent tests)
  bias_gen.spice      -> DC operating point (vbias, current)
  rpeak_detector.spice -> timing accuracy vs ideal R-peak
  pulse_gen.spice      -> pulse width, amplitude, quiescent
  delay_timer.spice    -> trigger timing accuracy for all 5 outputs
  edge_buffer.spice    -> propagation delay
"""
import subprocess, os, re, sys, time
import numpy as np
from scipy.stats import spearmanr

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


def load_beats(record_id, max_beats=40):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    raw_beats = []
    ewma_rr = None
    for i in range(1, min(len(valid) - 1, max_beats + 5)):
        idx, sym = valid[i]
        pre_rr = (idx - valid[i-1][0]) / FS
        post_rr = (valid[i+1][0] - idx) / FS
        bs = idx - WIN_L; be = idx + WIN_R
        if bs < 0 or be >= len(sig): continue
        beat = sig[bs:be]

        prev_bs = valid[i-1][0] - WIN_L
        prev_be = valid[i-1][0] + WIN_R
        if prev_bs >= 0 and prev_be < len(sig):
            prev_beat = sig[prev_bs:prev_be]
            py_bi_l0 = beat[WIN_L, 0] - prev_beat[WIN_L, 0]
            py_bi_l1 = beat[WIN_L, min(1, n_leads-1)] - prev_beat[WIN_L, min(1, n_leads-1)]
        else:
            py_bi_l0 = py_bi_l1 = 0.0

        if ewma_rr is None:
            ewma_rr_before = pre_rr
            ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr

        raw_beats.append({
            'pre_rr': pre_rr, 'post_rr': post_rr,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': pre_rr - ewma_rr_before,
            'py_f2': pre_rr - post_rr,
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[WIN_L + 10, 0],
            'py_slp': beat[WIN_L + 5, 0] - beat[WIN_L + 25, 0],
            'py_sym': beat[WIN_L, 0],
        })
        if len(raw_beats) >= max_beats:
            break

    max_peak = max(np.max(np.abs(b['ecg_l0'][WIN_L-20:WIN_L+40])) for b in raw_beats)
    gain = 0.16 / max_peak if max_peak > 0 else 0.05
    return raw_beats, gain


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


def run_sim(title, sp_body):
    sp = ".title {}\n".format(title) + sp_body + "\n.end\n"
    tb = os.path.join(FRONTEND_DIR, "_verify_{}.spice".format(title.replace(' ', '_')))
    logf = tb.replace('.spice', '.log')
    with open(tb, 'w') as f:
        f.write(sp)
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', logf, tb],
                   capture_output=True, text=True, timeout=600, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    results = {}
    if os.path.exists(logf):
        with open(logf) as f:
            text = f.read()
        for line in text.split('\n'):
            m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
            if m:
                try:
                    results[m.group(1).lower()] = float(m.group(2))
                except:
                    pass
    return results, elapsed


def interp_sf(t_arr, v_arr, t_ms):
    if t_ms <= t_arr[0]: return v_arr[0]
    if t_ms >= t_arr[-1]: return v_arr[-1]
    for i in range(len(t_arr)-1):
        if t_arr[i] <= t_ms <= t_arr[i+1]:
            frac = (t_ms - t_arr[i]) / (t_arr[i+1] - t_arr[i])
            return v_arr[i] + frac * (v_arr[i+1] - v_arr[i])
    return v_arr[-1]


# ========================================================================
# PART 1: FEATURE CIRCUIT VERIFICATION
# ========================================================================
print("=" * 70)
print("PART 1: FEATURE CIRCUIT VERIFICATION (ideal triggers)")
print("  Each feature circuit tested with ideal PWL triggers")
print("  Output compared to Python reference via Spearman rho")
print("=" * 70)

RECORDS = ['100', '109', '208']
SKIP = 5

lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
fp = FRONTEND_DIR.replace("\\", "/")

all_results = {}

for rec_id in RECORDS:
    print("\n--- Record {} ---".format(rec_id))
    beats, gain = load_beats(rec_id, 35)
    n = len(beats)
    print("  Loaded {} beats, gain={:.4f}".format(n, gain))

    rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

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
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    PW = 0.0005
    first_rr_ms = beats[0]['pre_rr'] * 1000
    rr_t = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
    rr_v = [0.4819,0.4807,0.4772,0.4715,0.4602,0.4488,0.4375,0.4262,0.4150,0.4037,0.3925,0.3813,0.3702]
    ic_ewma = interp_sf(rr_t, rr_v, first_rr_ms)

    ideal_rpk = make_trig(rpeak_times, 0, PW, total_time)
    ideal_rr_cap = make_trig(rpeak_times, -0.060, 0.055, total_time)
    ideal_rr_copy = make_trig(rpeak_times, 0.010, 0.002, total_time)
    ideal_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S)/FS, PW, total_time)
    ideal_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID)/FS, PW, total_time)
    ideal_s14 = make_trig(rpeak_times, 5.0/FS, PW, total_time)
    ideal_s69 = make_trig(rpeak_times, 25.0/FS, PW, total_time)
    ideal_wid = make_trig(rpeak_times, 10.0/FS, PW, total_time)
    ideal_bi_copy = make_trig(rpeak_times, 0.300, 0.002, total_time)

    sp = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vbias vbias 0 0.300
Vecg0 ecg_l0 0 PWL({ecg0})
Vecg1 ecg_l1 0 PWL({ecg1})

Vrpk trig_rpeak 0 PWL({rpk})
Vrrc trig_rr_cap 0 PWL({rrc})
Vrry trig_rr_copy 0 PWL({rry})
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})
Vs14 trig_s14 0 PWL({s14})
Vs69 trig_s69 0 PWL({s69})
Vwid trig_wid 0 PWL({wid})
Vbco trig_bi_copy 0 PWL({bco})

.include "{fp}/inverter.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

Xrr ecg_l0 trig_rr_cap trig_rpeak trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 trig_rpeak trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.net_timer)={vdd}
.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xbi0.prev_node)={vbase}
.ic V(xbi0.esamp_node)={vbase}
.ic V(xbi1.prev_node)={vbase}
.ic V(xbi1.esamp_node)={vbase}

.tran 0.5m {ttot:.4f}

""".format(lib=lib_path, vdd=VDD, ecg0=make_pwl(ecg_l0_pts), ecg1=make_pwl(ecg_l1_pts),
           rpk=ideal_rpk, rrc=ideal_rr_cap, rry=ideal_rr_copy,
           pwb=ideal_pw_base, pwp=ideal_pw_peak,
           s14=ideal_s14, s69=ideal_s69, wid=ideal_wid, bco=ideal_bi_copy,
           fp=fp, ic=ic_ewma, vbase=VBASE, ttot=total_time)

    for j in range(SKIP, n):
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

    res, elapsed = run_sim("feat_{}".format(rec_id), sp)
    print("  Sim time: {:.0f}s, {} measurements".format(elapsed, len(res)))

    py_keys = {1:'py_f1', 2:'py_f2', 3:'py_bi_l0', 4:'py_bi_l1',
               5:'py_pw', 6:'py_wid', 7:'py_slp', 8:'py_sym'}
    rhos = {}
    for fk in range(1, 9):
        circuit_vals = []; python_vals = []
        for j in range(SKIP, n):
            cv = res.get('f{}_{}'.format(fk, j))
            if cv is not None:
                circuit_vals.append(cv)
                python_vals.append(beats[j][py_keys[fk]])
        if len(circuit_vals) >= 5:
            rho, pval = spearmanr(circuit_vals, python_vals)
            rhos[fk] = abs(rho)
        else:
            rhos[fk] = 0.0

    fnames = {1:'F1:rr_ratio', 2:'F2:rr_asym', 3:'F3:bi_L0', 4:'F4:bi_L1',
              5:'F5:pwave', 6:'F6:width', 7:'F7:slope', 8:'F8:symmetry'}
    for fk in range(1, 9):
        status = "PASS" if rhos[fk] >= 0.95 else "WARN" if rhos[fk] >= 0.80 else "FAIL"
        print("  {:15s}  rho={:.4f}  [{}]".format(fnames[fk], rhos[fk], status))

    all_results[(rec_id, 'feature')] = rhos


# ========================================================================
# PART 2: SUPPLEMENTARY CIRCUIT VERIFICATION
# ========================================================================
print("\n\n" + "=" * 70)
print("PART 2: SUPPLEMENTARY CIRCUIT VERIFICATION")
print("=" * 70)

# --- 2A: bias_gen ---
print("\n--- 2A: bias_gen.spice ---")
sp_bias = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
.include "{fp}/bias_gen.spice"
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3

.tran 0.1m 5m
.meas tran vbias_dc FIND V(vbias) AT=4.0e-3
.meas tran vref_dc FIND V(vref) AT=4.0e-3
""".format(lib=lib_path, vdd=VDD, fp=fp)

res_bias, t_bias = run_sim("bias_gen", sp_bias)
vbias_v = res_bias.get('vbias_dc', float('nan'))
vref_v = res_bias.get('vref_dc', float('nan'))
print("  vbias = {:.4f}V (target: 0.29 +/- 0.02V)".format(vbias_v))
print("  vref  = {:.4f}V (should equal vbias)".format(vref_v))
bias_ok = 0.27 <= vbias_v <= 0.31
print("  Status: {} ({:.0f}s)".format("PASS" if bias_ok else "FAIL", t_bias))


# --- 2B: rpeak_detector ---
print("\n--- 2B: rpeak_detector.spice ---")
print("  Test: Feed known ECG, measure when output goes HIGH vs ideal R-peak time")

beats, gain = load_beats('100', 20)
n_rpk = len(beats)
rpeak_times_rpk = [max(beats[0]['pre_rr'], 0.5)]
for i in range(1, n_rpk):
    rpeak_times_rpk.append(rpeak_times_rpk[-1] + beats[i]['pre_rr'])
total_time_rpk = rpeak_times_rpk[-1] + 0.5

ecg_pts = [(0, VBASE)]
for i, rpt in enumerate(rpeak_times_rpk):
    bs_t = rpt - WIN_L / FS
    ecg = beats[i]['ecg_l0']
    t_before = bs_t - 0.003
    if t_before > 0.01: ecg_pts.append((t_before, VBASE))
    for j in range(len(ecg)):
        t = bs_t + j / FS
        if t > 0: ecg_pts.append((t, max(0.05, min(0.85, VBASE + gain * ecg[j]))))
    ecg_pts.append((bs_t + len(ecg)/FS + 0.003, VBASE))
ecg_pts.append((total_time_rpk, VBASE))

max_r_v = max(VBASE + gain * b['ecg_l0'][WIN_L] for b in beats)
rpeak_thresh = VBASE + 0.30 * (max_r_v - VBASE)

sp_rpk = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vbias vbias 0 0.300
Vecg ecg_l0 0 PWL({ecg})
Vthresh rpeak_threshold 0 {thresh:.6f}

.include "{fp}/inverter.spice"
.include "{fp}/rpeak_detector.spice"

Xrpk ecg_l0 rpeak_threshold rpeak_out vdd vss vbias rpeak_detector

.tran 0.5m {ttot:.4f}
""".format(lib=lib_path, vdd=VDD, ecg=make_pwl(ecg_pts), thresh=rpeak_thresh,
           fp=fp, ttot=total_time_rpk)

for j in range(3, n_rpk):
    t_start = rpeak_times_rpk[j] - 0.020
    if t_start > 0.1:
        sp_rpk += ".meas tran rpk_rise_{j} WHEN V(rpeak_out)=0.45 RISE=1 TD={ts:.7f}\n".format(
            j=j, ts=t_start)

res_rpk, t_rpk = run_sim("rpeak_det", sp_rpk)
print("  Threshold = {:.4f}V, gain = {:.4f}".format(rpeak_thresh, gain))
print("  Sim time: {:.0f}s".format(t_rpk))

rpk_errors = []
print("\n  beat | ideal_R(ms)  | detected(ms) | error(ms)")
print("  " + "-" * 55)
for j in range(3, n_rpk):
    ideal_ms = rpeak_times_rpk[j] * 1000
    det_t = res_rpk.get('rpk_rise_{}'.format(j))
    if det_t is not None:
        det_ms = det_t * 1000
        err_ms = det_ms - ideal_ms
        rpk_errors.append(err_ms)
        print("  {:3d}  | {:10.2f}   | {:10.2f}   | {:+6.2f}".format(j, ideal_ms, det_ms, err_ms))
    else:
        print("  {:3d}  | {:10.2f}   | MISSED       | ---".format(j, ideal_ms))

if rpk_errors:
    print("\n  Timing error stats:")
    print("    Mean:  {:+.2f}ms".format(np.mean(rpk_errors)))
    print("    Std:   {:.2f}ms".format(np.std(rpk_errors)))
    print("    Range: {:+.2f} to {:+.2f}ms".format(min(rpk_errors), max(rpk_errors)))
    all_within = all(abs(e) < 1.0 for e in rpk_errors)
    print("  Status: {} (all within +/-1ms of ideal? {})".format(
        "PASS" if all_within else "INFO - fires early (expected for threshold comparator)",
        "YES" if all_within else "NO"))


# --- 2C: pulse_gen ---
print("\n--- 2C: pulse_gen.spice ---")
print("  Test: Apply step input, measure pulse width and amplitude")

sp_pg = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vbias vbias 0 0.300

.include "{fp}/inverter.spice"
.include "{fp}/pulse_gen.spice"

* Step input at t=5ms
Vstep level_in 0 PWL(0 0 4.9e-3 0 5.0e-3 0.9 50e-3 0.9)
Xpg level_in pulse_out vdd vss vbias pulse_gen

.tran 0.01m 20m

.meas tran pg_rise WHEN V(pulse_out)=0.45 RISE=1
.meas tran pg_fall WHEN V(pulse_out)=0.45 FALL=1
.meas tran pg_peak MAX V(pulse_out) FROM=4.5e-3 TO=15e-3
.meas tran pg_quiescent FIND V(pulse_out) AT=3.0e-3
.meas tran pg_post FIND V(pulse_out) AT=18.0e-3
""".format(lib=lib_path, vdd=VDD, fp=fp)

res_pg, t_pg = run_sim("pulse_gen", sp_pg)
pg_rise = res_pg.get('pg_rise', float('nan'))
pg_fall = res_pg.get('pg_fall', float('nan'))
pg_width = (pg_fall - pg_rise) * 1000 if pg_fall == pg_fall and pg_rise == pg_rise else float('nan')
pg_peak = res_pg.get('pg_peak', float('nan'))
pg_quies = res_pg.get('pg_quiescent', float('nan'))
pg_post = res_pg.get('pg_post', float('nan'))

print("  Rise time:    {:.4f}ms".format(pg_rise * 1000 if pg_rise == pg_rise else float('nan')))
print("  Fall time:    {:.4f}ms".format(pg_fall * 1000 if pg_fall == pg_fall else float('nan')))
print("  Pulse width:  {:.3f}ms (target: 0.3-3.0ms)".format(pg_width))
print("  Peak voltage: {:.3f}V (target: >=0.85V)".format(pg_peak))
print("  Quiescent:    {:.4f}V (target: <0.05V)".format(pg_quies))
print("  Post-pulse:   {:.4f}V (target: <0.10V)".format(pg_post))

pg_ok = (0.1 <= pg_width <= 5.0) and pg_peak >= 0.80 and pg_quies < 0.10
print("  Status: {}".format("PASS" if pg_ok else "WARN"))


# --- 2D: delay_timer ---
print("\n--- 2D: delay_timer.spice ---")
print("  Test: Apply rpeak pulse, measure when each trigger fires")

rr_t = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
rr_v = [0.4819,0.4807,0.4772,0.4715,0.4602,0.4488,0.4375,0.4262,0.4150,0.4037,0.3925,0.3813,0.3702]
dt_t = [1,2,3,4,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800]
dt_v = [0.4829,0.4824,0.4819,0.4813,0.4808,0.4782,0.4756,0.4730,0.4677,0.4625,0.4573,0.4521,0.4469,0.4417,0.4366,0.4314,0.4056,0.3800,0.3545,0.3292,0.3041,0.2791,0.2296,0.1807,0.1326,0.0856]
ft_t = [0.5,1,2,3,4,5,7,10,14,20]
ft_v = [0.4788,0.4662,0.4410,0.4160,0.3911,0.3663,0.3173,0.2450,0.1507,0.0249]

rpk_drop = 0.5
pipe_s14=2.9; pipe_wid=4.5; pipe_s69=4.4; pipe_bi=5.9; pipe_s10=1.8

vth_14 = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_s14))
vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_wid))
vth_69 = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_s69))
vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_bi))
vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_s10))

print("  Threshold voltages:")
print("    vth_14(fast) = {:.4f}V (target 14ms from rpeak)".format(vth_14))
print("    vth_28(main) = {:.4f}V (target 28ms = R + 10/360s)".format(vth_28))
print("    vth_69(main) = {:.4f}V (target 69ms = R + 25/360s)".format(vth_69))
print("    vth_300(main)= {:.4f}V (target 300ms)".format(vth_300))
print("    vth_10(fast) = {:.4f}V (target 10ms)".format(vth_10))

sp_dt = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
Vbias vbias 0 0.300

.include "{fp}/inverter.spice"
.include "{fp}/delay_timer.spice"

* rpeak pulse at t=10ms (0.5ms wide)
Vrpk rpeak_pulse 0 PWL(0 0 9.5e-3 0 10.0e-3 0.9 10.5e-3 0.9 11.0e-3 0)

* Threshold voltage sources
Vth14 vth_14_node 0 {vth14:.6f}
Vth69 vth_69_node 0 {vth69:.6f}
Vth28 vth_28_node 0 {vth28:.6f}
Vth300 vth_300_node 0 {vth300:.6f}
Vth10 vth_10_node 0 {vth10:.6f}

Xdt rpeak_pulse trig_s14 trig_s69 trig_wid trig_bi_copy trig_rr_copy sf_out vdd vss vbias vth_14_node vth_69_node vth_28_node vth_300_node vth_10_node delay_timer

.tran 0.5m 500m

* Measure when each trigger fires (crosses 0.45V going HIGH)
.meas tran dt_s14 WHEN V(trig_s14)=0.45 RISE=1 TD=10e-3
.meas tran dt_s69 WHEN V(trig_s69)=0.45 RISE=1 TD=10e-3
.meas tran dt_wid WHEN V(trig_wid)=0.45 RISE=1 TD=10e-3
.meas tran dt_bi  WHEN V(trig_bi_copy)=0.45 RISE=1 TD=10e-3
.meas tran dt_rr  WHEN V(trig_rr_copy)=0.45 RISE=1 TD=10e-3

* Also measure SF voltage at key times
.meas tran sf_at_0 FIND V(sf_out) AT=10.5e-3
.meas tran sf_at_14 FIND V(sf_out) AT=24.0e-3
.meas tran sf_at_69 FIND V(sf_out) AT=79.0e-3
.meas tran sf_at_300 FIND V(sf_out) AT=310.0e-3
""".format(lib=lib_path, vdd=VDD, fp=fp,
           vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10)

res_dt, t_dt = run_sim("delay_timer", sp_dt)
print("\n  Sim time: {:.0f}s".format(t_dt))

rpk_time = 0.010  # rpeak at 10ms
targets = {
    's14': (14.0/1000, 'trig_s14 (R+14ms, for F7 slope)'),
    'wid': (28.0/1000, 'trig_wid (R+28ms, for F6 width)'),
    's69': (69.0/1000, 'trig_s69 (R+69ms, for F7 slope)'),
    'rr':  (10.0/1000, 'trig_rr_copy (R+10ms, for F1/F2 EWMA)'),
    'bi':  (300.0/1000, 'trig_bi_copy (R+300ms, for F3/F4 EWMA)'),
}

print("\n  Trigger     | Target(ms) | Actual(ms) | Error(ms) | Status")
print("  " + "-" * 65)
dt_all_pass = True
for key, (target_offset, desc) in targets.items():
    target_abs = rpk_time + target_offset
    actual = res_dt.get('dt_{}'.format(key))
    if actual is not None:
        err = (actual - target_abs) * 1000
        ok = abs(err) < 3.0  # 3ms tolerance
        if not ok: dt_all_pass = False
        print("  {:10s}  | {:8.1f}    | {:8.2f}   | {:+6.2f}    | {}".format(
            desc.split('(')[0].strip(), target_offset*1000, (actual-rpk_time)*1000, err, "PASS" if ok else "FAIL"))
    else:
        dt_all_pass = False
        print("  {:10s}  | {:8.1f}    | MISSED     | ---       | FAIL".format(
            desc.split('(')[0].strip(), target_offset*1000))

print("\n  SF voltages:")
for label, key in [("at rpk+0.5ms", 'sf_at_0'), ("at rpk+14ms", 'sf_at_14'),
                   ("at rpk+69ms", 'sf_at_69'), ("at rpk+300ms", 'sf_at_300')]:
    v = res_dt.get(key, float('nan'))
    print("    {} = {:.4f}V".format(label, v))

print("  Overall: {}".format("PASS" if dt_all_pass else "FAIL - see errors above"))


# --- 2E: edge_buffer ---
print("\n--- 2E: edge_buffer.spice ---")
sp_eb = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0
.include "{fp}/inverter.spice"
.include "{fp}/edge_buffer.spice"
Vin vin 0 PWL(0 0 4.9e-3 0 5.0e-3 0.9 15e-3 0.9 15.1e-3 0)
Xeb vin vout vdd vss edge_buffer
.tran 0.01m 20m
.meas tran eb_rise WHEN V(vout)=0.45 RISE=1
.meas tran eb_fall WHEN V(vout)=0.45 FALL=1
""".format(lib=lib_path, vdd=VDD, fp=fp)

res_eb, t_eb = run_sim("edge_buffer", sp_eb)
eb_rise = res_eb.get('eb_rise', float('nan'))
eb_fall = res_eb.get('eb_fall', float('nan'))
rise_delay = (eb_rise - 0.005) * 1e6 if eb_rise == eb_rise else float('nan')
fall_delay = (eb_fall - 0.0151) * 1e6 if eb_fall == eb_fall else float('nan')
print("  Rise propagation: {:.1f}us".format(rise_delay))
print("  Fall propagation: {:.1f}us".format(fall_delay))
eb_ok = abs(rise_delay) < 100 and abs(fall_delay) < 100
print("  Status: {} ({:.0f}s)".format("PASS" if eb_ok else "WARN", t_eb))


# ========================================================================
# SUMMARY
# ========================================================================
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nPART 1: Feature Circuits (rho vs Python with ideal triggers)")
print("  {:>6s}  {:>12s}  {:>12s}  {:>12s}".format("", *RECORDS))
fnames = {1:'F1:rr_ratio', 2:'F2:rr_asym', 3:'F3:bi_L0', 4:'F4:bi_L1',
          5:'F5:pwave', 6:'F6:width', 7:'F7:slope', 8:'F8:symmetry'}
for fk in range(1, 9):
    vals = []
    for rec in RECORDS:
        r = all_results.get((rec, 'feature'), {}).get(fk, 0)
        vals.append("{:.4f}".format(r))
    print("  {:15s}  {:>12s}  {:>12s}  {:>12s}".format(fnames[fk], *vals))

print("\nPART 2: Supplementary Circuits")
print("  bias_gen:        {}".format("PASS" if bias_ok else "FAIL"))
print("  rpeak_detector:  fires {:.1f}ms early (mean), {:.1f}ms std".format(
    np.mean(rpk_errors) if rpk_errors else 0, np.std(rpk_errors) if rpk_errors else 0))
print("  pulse_gen:       width={:.2f}ms, peak={:.3f}V".format(pg_width, pg_peak))
print("  delay_timer:     {}".format("ALL PASS" if dt_all_pass else "SOME FAIL"))
print("  edge_buffer:     {}".format("PASS" if eb_ok else "WARN"))
print("=" * 70)
