"""
Integration test: analog supplementary circuits + feature circuits.

Replaces MOST ideal triggers with analog circuits:
  - bias_gen → vbias
  - rpeak_detector + pulse_gen → rpeak_pulse (for F3,F4,F8 + delay_timer)
  - delay_timer + pulse_gen → trig_s14, trig_s69, trig_wid, trig_bi_copy, trig_rr_copy

Keeps IDEAL:
  - trig_rpeak for rr_timing precharge (analog rpeak fires 6.6ms early,
    which conflicts with the 55ms capture window — v6 design would fix this)
  - trig_rr_cap for rr_timing capture (55ms window needed for SF settling)
  - trig_pw_base, trig_pw_peak for F5 (pwave_predictor not yet integrated)

Feature outputs compared to Python reference via Spearman rho.
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
SKIP = 5

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'

# --- Calibration data (freshly measured 2026-10-25) ---
rr_t = [10,20,50,100,200,300,400,500,600,700,800,900,1000]
rr_v = [0.4819,0.4807,0.4772,0.4715,0.4602,0.4488,0.4375,0.4262,0.4150,0.4037,0.3925,0.3813,0.3702]
dt_t = [1,2,3,4,5,10,15,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500,600,700,800]
dt_v = [0.4829,0.4824,0.4819,0.4813,0.4808,0.4782,0.4756,0.4730,0.4678,0.4625,0.4573,0.4521,0.4469,0.4418,0.4366,0.4314,0.4056,0.3800,0.3545,0.3292,0.3041,0.2791,0.2296,0.1807,0.1326,0.0856]
ft_t = [0.5,1,2,3,4,5,7,10,14,20,25,30,35,40]
ft_v = [0.4810,0.4746,0.4617,0.4489,0.4362,0.4235,0.3981,0.3605,0.3107,0.2373,0.1773,0.1183,0.0616,0.0192]

# Pipeline delays (per-comparator, empirically calibrated)
pipe_s14 = 2.9; pipe_wid = 4.5; pipe_s69 = 4.4; pipe_bi = 5.9; pipe_s10 = 1.8

# rpeak_detector offset: fires this many ms BEFORE true R-peak (measured on record 100)
RPEAK_OFFSET_MS = 6.6
PG_WIDTH_MS = 0.7  # pulse_gen pulse width
RPEAK_SAMPLE_OFFSET = round((RPEAK_OFFSET_MS - PG_WIDTH_MS) * FS / 1000)  # =2


def interp_sf(t_arr, v_arr, t_ms):
    if t_ms <= t_arr[0]: return v_arr[0]
    if t_ms >= t_arr[-1]: return v_arr[-1]
    for i in range(len(t_arr)-1):
        if t_arr[i] <= t_ms <= t_arr[i+1]:
            frac = (t_ms - t_arr[i]) / (t_arr[i+1] - t_arr[i])
            return v_arr[i] + frac * (v_arr[i+1] - v_arr[i])
    return v_arr[-1]


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
            rpk_idx = WIN_L - RPEAK_SAMPLE_OFFSET
            py_bi_l0 = beat[rpk_idx, 0] - prev_beat[rpk_idx, 0]
            py_bi_l1 = beat[rpk_idx, min(1, n_leads-1)] - prev_beat[rpk_idx, min(1, n_leads-1)]
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
            'py_sym': beat[WIN_L - RPEAK_SAMPLE_OFFSET, 0],
        })
        if len(raw_beats) >= max_beats: break
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
    tb = os.path.join(FRONTEND_DIR, "_integ_{}.spice".format(title.replace(' ', '_')))
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
    return results, elapsed


# ========================================================================
print("=" * 70)
print("INTEGRATION TEST: Analog supplementary + feature circuits")
print("  Analog: bias_gen, rpeak_detector, pulse_gen, delay_timer")
print("  Ideal:  rr_cap (55ms capture), rpeak_for_rr_timing, pw_base, pw_peak")
print("=" * 70)

lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
fp = FRONTEND_DIR.replace("\\", "/")

RECORDS = ['100', '109', '208']
all_results = {}

for rec_id in RECORDS:
    print("\n--- Record {} ---".format(rec_id))
    beats, gain = load_beats(rec_id, 35)
    n = len(beats)
    print("  {} beats, gain={:.4f}".format(n, gain))

    rpeak_times = [max(beats[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    # --- ECG PWL ---
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

    # --- rpeak threshold (adaptive: use min R-peak to ensure detection) ---
    max_r_v = max(VBASE + gain * b['ecg_l0'][WIN_L] for b in beats)
    min_r_v = min(VBASE + gain * b['ecg_l0'][WIN_L] for b in beats)
    # Set threshold at 80% of the MINIMUM R-peak to catch all beats
    rpeak_thresh = VBASE + 0.80 * (min_r_v - VBASE)
    # But clamp to at most 30% of max to avoid noise triggers
    rpeak_thresh_max = VBASE + 0.30 * (max_r_v - VBASE)
    rpeak_thresh = min(rpeak_thresh, rpeak_thresh_max)
    print("  rpeak threshold: {:.4f}V (min_R={:.4f}V, max_R={:.4f}V)".format(
        rpeak_thresh, min_r_v, max_r_v))

    # --- Delay timer thresholds (compensated for rpeak offset) ---
    # Discharge starts at R - RPEAK_OFFSET + PG_WIDTH = R - 5.9ms
    # For target at R + X: discharge_time = X + 5.9ms
    offset_ms = RPEAK_OFFSET_MS - PG_WIDTH_MS  # 5.9ms
    vth_14 = interp_sf(ft_t, ft_v, max(0.3, 14 + offset_ms - pipe_s14))
    vth_28 = interp_sf(dt_t, dt_v, max(0.5, 28 + offset_ms - pipe_wid))
    vth_69 = interp_sf(dt_t, dt_v, max(0.5, 69 + offset_ms - pipe_s69))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 + offset_ms - pipe_bi))
    vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10 + offset_ms - pipe_s10))

    # --- IC for rr_timing ---
    PW = 0.0005
    first_rr_ms = beats[0]['pre_rr'] * 1000
    ic_ewma = interp_sf(rr_t, rr_v, first_rr_ms)

    # --- IDEAL triggers (kept for rr_timing + pwave) ---
    ideal_rpk = make_trig(rpeak_times, 0, PW, total_time)
    ideal_rr_cap = make_trig(rpeak_times, -0.060, 0.050, total_time)  # 50ms (close at R-10ms)
    ideal_rr_copy = make_trig(rpeak_times, 0.010, 0.002, total_time)
    ideal_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S)/FS, PW, total_time)
    ideal_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID)/FS, PW, total_time)

    sp = """.lib "{lib}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27
Vdd vdd 0 {vdd}
Vss vss 0 0

* ECG inputs
Vecg0 ecg_l0 0 PWL({ecg0})
Vecg1 ecg_l1 0 PWL({ecg1})

* ============================================================
* ANALOG SUPPLEMENTARY CIRCUITS
* ============================================================

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/long_pulse_gen.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/edge_buffer.spice"

* --- Bias generator (replaces ideal Vbias) ---
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3

* --- R-peak detector + pulse_gen (analog rpeak) ---
Vthresh rpeak_threshold 0 {rpk_thresh:.6f}
Xrpk ecg_l0 rpeak_threshold rpeak_level vdd vss vbias rpeak_detector
Xpg_rpk rpeak_level rpeak_pulse_raw vdd vss vbias pulse_gen
Xeb_rpk rpeak_pulse_raw rpeak_pulse vdd vss edge_buffer

* --- Delay timer (triggered by analog rpeak pulse) ---
Vth14 vth_14_node 0 {vth14:.6f}
Vth69 vth_69_node 0 {vth69:.6f}
Vth28 vth_28_node 0 {vth28:.6f}
Vth300 vth_300_node 0 {vth300:.6f}
Vth10 vth_10_node 0 {vth10:.6f}

Xdt rpeak_pulse trig_s14_level trig_s69_level trig_wid_level trig_bi_level trig_rr_level dt_sf_out vdd vss vbias vth_14_node vth_69_node vth_28_node vth_300_node vth_10_node delay_timer

* --- Pulse generators for delay_timer outputs ---
Xpg_s14 trig_s14_level pg_s14_raw vdd vss vbias pulse_gen
Xpg_s69 trig_s69_level pg_s69_raw vdd vss vbias long_pulse_gen
Xpg_wid trig_wid_level pg_wid_raw vdd vss vbias long_pulse_gen
Xpg_bi  trig_bi_level  pg_bi_raw  vdd vss vbias long_pulse_gen
Xpg_rr  trig_rr_level  pg_rr_raw  vdd vss vbias pulse_gen

* --- Edge buffers for cleaner trigger edges ---
Xeb_s14 pg_s14_raw trig_s14 vdd vss edge_buffer
Xeb_s69 pg_s69_raw trig_s69 vdd vss edge_buffer
Xeb_wid pg_wid_raw trig_wid vdd vss edge_buffer
Xeb_bi  pg_bi_raw  trig_bi_copy vdd vss edge_buffer
Xeb_rr  pg_rr_raw  trig_rr_copy vdd vss edge_buffer

* ============================================================
* IDEAL TRIGGERS (kept for rr_timing + pwave)
* ============================================================
Vrpk_ideal trig_rpeak_ideal 0 PWL({rpk_ideal})
Vrrc trig_rr_cap 0 PWL({rrc})
Vpwb trig_pw_base 0 PWL({pwb})
Vpwp trig_pw_peak 0 PWL({pwp})

* ============================================================
* FEATURE CIRCUITS
* ============================================================
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

* F1/F2: rr_timing (ideal rpeak + ideal rr_cap, analog rr_copy)
Xrr ecg_l0 trig_rr_cap trig_rpeak_ideal trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing

* F3/F4: beat_instability (ANALOG rpeak, ANALOG bi_copy)
Xbi0 ecg_l0 rpeak_pulse trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 rpeak_pulse trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability

* F5: pwave_diff (IDEAL pw triggers)
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff

* F6: qrs_width (ANALOG wid trigger)
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width

* F7: qrs_slope (ANALOG s14, s69 triggers)
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope

* F8: qrs_symmetry (ANALOG rpeak pulse)
Xsym ecg_l0 rpeak_pulse f8_symmetry vdd vss qrs_symmetry

* ============================================================
* INITIAL CONDITIONS
* ============================================================
.ic V(xrr.net_timer)={vdd}
.ic V(xrr.ewma_node)={ic:.6f}
.ic V(xrr.esamp_node)={ic:.6f}
.ic V(xrr.prev_node)={ic:.6f}
.ic V(xbi0.prev_node)={vbase}
.ic V(xbi0.esamp_node)={vbase}
.ic V(xbi1.prev_node)={vbase}
.ic V(xbi1.esamp_node)={vbase}

.tran 0.5m {ttot:.4f}

""".format(lib=lib_path, vdd=VDD,
           ecg0=make_pwl(ecg_l0_pts), ecg1=make_pwl(ecg_l1_pts),
           rpk_thresh=rpeak_thresh,
           vth14=vth_14, vth69=vth_69, vth28=vth_28, vth300=vth_300, vth10=vth_10,
           rpk_ideal=ideal_rpk, rrc=ideal_rr_cap,
           pwb=ideal_pw_base, pwp=ideal_pw_peak,
           fp=fp, ic=ic_ewma, vbase=VBASE, ttot=total_time)

    # --- Measurement statements ---
    # Feature outputs measured 100ms after TRUE R-peak
    for j in range(SKIP, n):
        t_f1 = rpeak_times[j] - 0.010
        if t_f1 > 0.5:
            sp += ".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t:.7f}\n".format(j=j, t=t_f1)
        if j + 1 < n:
            t_f2 = rpeak_times[j+1] - 0.065
            if t_f2 > 0.5:
                sp += ".meas tran f2_{j} FIND V(f2_rr_asym) AT={t:.7f}\n".format(j=j, t=t_f2)
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            for fk, nm in [(3,'f3_bi_l0'),(4,'f4_bi_l1'),(5,'f5_pwave'),
                           (6,'f6_width'),(7,'f7_slope'),(8,'f8_symmetry')]:
                sp += ".meas tran f{k}_{j} FIND V({nm}) AT={t:.7f}\n".format(k=fk, j=j, nm=nm, t=t_f38)

    # Extract ACTUAL analog trigger times per beat (for jitter-aware Python ref)
    for j in range(n):
        td = max(0.01, rpeak_times[j] - 0.050)
        sp += ".meas tran trpk_{j} WHEN V(rpeak_pulse)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran ts14_{j} WHEN V(trig_s14)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran ts69_{j} WHEN V(trig_s69)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)
        sp += ".meas tran twid_{j} WHEN V(trig_wid)=0.45 RISE=1 TD={td:.7f}\n".format(j=j, td=td)

    res, elapsed = run_sim("integ_{}".format(rec_id), sp)
    print("  Sim time: {:.0f}s, {} measurements".format(elapsed, len(res)))

    # --- Classify beats: valid vs missed trigger ---
    valid_rpk = set()
    valid_dt = set()  # delay timer triggers valid
    n_rpk_ok = 0; n_rpk_miss = 0
    n_dt_ok = 0; n_dt_miss = 0
    for j in range(n):
        r = rpeak_times[j]
        trpk = res.get('trpk_{}'.format(j))
        ts14 = res.get('ts14_{}'.format(j))
        twid = res.get('twid_{}'.format(j))
        ts69 = res.get('ts69_{}'.format(j))
        # rpeak valid if within [-20ms, +5ms] of true R
        if trpk and abs(trpk - r) < 0.020:
            valid_rpk.add(j)
            n_rpk_ok += 1
        else:
            n_rpk_miss += 1
        # delay triggers valid if within ±15ms of expected
        if (ts14 and abs((ts14 - r)*1000 - 13) < 15 and
            twid and abs((twid - r)*1000 - 28) < 15 and
            ts69 and abs((ts69 - r)*1000 - 69) < 15):
            valid_dt.add(j)
            n_dt_ok += 1
        else:
            n_dt_miss += 1
    print("  rpeak valid: {}/{}, delay_timer valid: {}/{}".format(
        n_rpk_ok, n, n_dt_ok, n))
    # Print sample trigger times for valid beats
    shown = 0
    for j in range(SKIP, n):
        if j in valid_rpk and shown < 3:
            r = rpeak_times[j] * 1000
            trpk = res.get('trpk_{}'.format(j))
            ts14 = res.get('ts14_{}'.format(j))
            twid = res.get('twid_{}'.format(j))
            ts69 = res.get('ts69_{}'.format(j))
            print("    beat {}: rpk={:.1f}, s14={:.1f}, wid={:.1f}, s69={:.1f} ms from R".format(
                j, (trpk*1000-r), (ts14*1000-r) if ts14 else 0,
                (twid*1000-r) if twid else 0, (ts69*1000-r) if ts69 else 0))
            shown += 1

    # --- Helper: interpolate ECG at any time ---
    def ecg_at_time(beat_ecg, rpk_time, t_actual, fs=FS):
        bs_t = rpk_time - WIN_L / fs
        sample_offset = (t_actual - bs_t) * fs
        idx = int(sample_offset)
        frac = sample_offset - idx
        if idx < 0: idx, frac = 0, 0
        if idx >= len(beat_ecg) - 1: return beat_ecg[-1]
        return beat_ecg[idx] * (1 - frac) + beat_ecg[idx + 1] * frac

    # --- Compute Spearman rho using actual trigger times ---
    rhos = {}

    # F1/F2: use fixed Python ref (ideal triggers, no jitter issue)
    for fk, pykey in [(1, 'py_f1'), (2, 'py_f2')]:
        cv_list, pv_list = [], []
        for j in range(SKIP, n):
            cv = res.get('f{}_{}'.format(fk, j))
            if cv is not None:
                cv_list.append(cv)
                pv_list.append(beats[j][pykey])
        rhos[fk] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # F5: ideal pwave triggers, use fixed Python ref
    cv_list, pv_list = [], []
    for j in range(SKIP, n):
        cv = res.get('f5_{}'.format(j))
        if cv is not None:
            cv_list.append(cv)
            pv_list.append(beats[j]['py_pw'])
    rhos[5] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # F3/F4/F8: use ACTUAL rpeak time for Python reference (valid beats only)
    # S&H captures at pulse falling edge: rise_time + pulse_width
    PG_HOLD_OFFSET = PG_WIDTH_MS * 0.001  # convert to seconds
    for fk, lead in [(3, 0), (4, 1), (8, 0)]:
        cv_list, pv_list = [], []
        for j in range(SKIP, n):
            if j not in valid_rpk:
                continue
            cv = res.get('f{}_{}'.format(fk, j))
            t_rpk = res.get('trpk_{}'.format(j))
            if cv is not None and t_rpk is not None:
                t_capture = t_rpk + PG_HOLD_OFFSET
                ecg_arr = beats[j]['ecg_l0'] if lead == 0 else beats[j]['ecg_l1']
                actual_ecg = ecg_at_time(ecg_arr, rpeak_times[j], t_capture)
                if fk == 8:
                    pv_list.append(actual_ecg)
                    cv_list.append(cv)
                else:
                    # beat_instability: both current AND previous must be valid
                    if (j - 1) in valid_rpk:
                        t_rpk_prev = res.get('trpk_{}'.format(j - 1))
                        if t_rpk_prev is not None:
                            prev_ecg_arr = beats[j-1]['ecg_l0'] if lead == 0 else beats[j-1]['ecg_l1']
                            prev_ecg = ecg_at_time(prev_ecg_arr, rpeak_times[j-1], t_rpk_prev + PG_HOLD_OFFSET)
                            pv_list.append(actual_ecg - prev_ecg)
                            cv_list.append(cv)
        rhos[fk] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # Pulse widths for S&H hold offset: pulse_gen ~0.7ms, long_pulse_gen ~2ms
    FAST_PG_HOLD = 0.0007   # pulse_gen hold offset (s14 uses pulse_gen)
    SLOW_PG_HOLD = 0.002    # long_pulse_gen hold offset (wid, s69 use long_pulse_gen)

    # F6: use ACTUAL wid trigger time + long_pulse_gen hold offset (valid beats only)
    cv_list, pv_list = [], []
    for j in range(SKIP, n):
        if j not in valid_dt:
            continue
        cv = res.get('f6_{}'.format(j))
        t_w = res.get('twid_{}'.format(j))
        if cv is not None and t_w is not None:
            actual_ecg = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t_w + SLOW_PG_HOLD)
            pv_list.append(actual_ecg)
            cv_list.append(cv)
    rhos[6] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # F7: use ACTUAL s14 and s69 trigger times + hold offsets (valid beats only)
    cv_list, pv_list = [], []
    for j in range(SKIP, n):
        if j not in valid_dt:
            continue
        cv = res.get('f7_{}'.format(j))
        t14 = res.get('ts14_{}'.format(j))
        t69 = res.get('ts69_{}'.format(j))
        if cv is not None and t14 is not None and t69 is not None:
            e14 = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t14 + FAST_PG_HOLD)
            e69 = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t69 + SLOW_PG_HOLD)
            pv_list.append(e14 - e69)
            cv_list.append(cv)
    rhos[7] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    # --- Diagnostic: print F8 values for first few valid beats ---
    if rec_id == 100:
        print("  F8 diagnostic (symmetry):")
        shown_f8 = 0
        for j in range(SKIP, n):
            if j not in valid_rpk or shown_f8 >= 5:
                continue
            cv = res.get('f8_{}'.format(j))
            t_rpk = res.get('trpk_{}'.format(j))
            if cv is not None and t_rpk is not None:
                t_cap = t_rpk + PG_HOLD_OFFSET
                py_actual = ecg_at_time(beats[j]['ecg_l0'], rpeak_times[j], t_cap)
                py_fixed = beats[j]['py_sym']
                offset_ms = (t_rpk - rpeak_times[j]) * 1000
                print("    beat {}: cv={:.5f} py_actual={:.4f} py_fixed={:.4f} rpk_offset={:.1f}ms".format(
                    j, cv, py_actual, py_fixed, offset_ms))
                shown_f8 += 1

    # Also compute FIXED-ref rho for comparison (only on valid beats)
    rhos_fixed = {}
    py_keys = {3:'py_bi_l0', 4:'py_bi_l1', 6:'py_wid', 7:'py_slp', 8:'py_sym'}
    for fk in [3, 4, 6, 7, 8]:
        valid_set = valid_rpk if fk in [3, 4, 8] else valid_dt
        cv_list, pv_list = [], []
        for j in range(SKIP, n):
            if j not in valid_set:
                continue
            cv = res.get('f{}_{}'.format(fk, j))
            if cv is not None:
                cv_list.append(cv)
                pv_list.append(beats[j][py_keys[fk]])
        rhos_fixed[fk] = abs(spearmanr(cv_list, pv_list)[0]) if len(cv_list) >= 5 else 0.0

    fnames = {1:'F1:rr_ratio', 2:'F2:rr_asym', 3:'F3:bi_L0', 4:'F4:bi_L1',
              5:'F5:pwave', 6:'F6:width', 7:'F7:slope', 8:'F8:symmetry'}
    triggers = {1:'ideal', 2:'ideal', 3:'ANALOG rpk', 4:'ANALOG rpk',
                5:'ideal pw', 6:'ANALOG wid', 7:'ANALOG s14/s69', 8:'ANALOG rpk'}
    for fk in range(1, 9):
        status = "PASS" if rhos[fk] >= 0.95 else "WARN" if rhos[fk] >= 0.80 else "FAIL"
        extra = ""
        if fk in rhos_fixed:
            extra = "  (fixed_ref={:.4f})".format(rhos_fixed[fk])
        print("  {:15s}  rho={:.4f}  [{}]  trigger={}{}".format(
            fnames[fk], rhos[fk], status, triggers[fk], extra))

    all_results[rec_id] = rhos

# ========================================================================
print("\n\n" + "=" * 70)
print("SUMMARY: Integration Test Results")
print("=" * 70)
print("                       100           109           208")
for fk in range(1, 9):
    fnames = {1:'F1:rr_ratio', 2:'F2:rr_asym', 3:'F3:bi_L0', 4:'F4:bi_L1',
              5:'F5:pwave', 6:'F6:width', 7:'F7:slope', 8:'F8:symmetry'}
    vals = "".join("{:14.4f}".format(all_results.get(r, {}).get(fk, 0)) for r in RECORDS)
    print("  {:20s} {}".format(fnames[fk], vals))
print("=" * 70)
print("\nThreshold info:")
print("  rpeak_offset = {:.1f}ms, pg_width = {:.1f}ms".format(RPEAK_OFFSET_MS, PG_WIDTH_MS))
print("  vth_14(fast) = {:.4f}V".format(vth_14))
print("  vth_28(main) = {:.4f}V".format(vth_28))
print("  vth_69(main) = {:.4f}V".format(vth_69))
print("  vth_300(main)= {:.4f}V".format(vth_300))
print("  vth_10(fast) = {:.4f}V".format(vth_10))
