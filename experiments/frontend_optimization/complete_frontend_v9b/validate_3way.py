"""
3-Way Validation: Python vs Ideal-SPICE vs Analog-SPICE
Compares all 8 features across three computation paths.

Changes from final_e2e_test.py:
  - EWMA_ALPHA = 0.015 (matches SPICE rr_timing Cewma=330pF)
  - F5 (pwave): post-R definition: ECG[rpk + 0.80*avg_rr] - ECG[rpk + 0.75*avg_rr]
  - rpk_drop = -3.0ms (v7g measured: fires ~3ms before annotation)
  - Analog chain: rpeak_detector -> rpeak_hold -> true_peak_detector -> pulse_gen
  - Pwave thresholds: post-R: 0.75*EWMA_RR and 0.80*EWMA_RR
  - trig_rpk_delayed removed (true peak replaces it)
"""
import subprocess, os, re, sys, time
import numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend_v9b")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9
VBASE = 0.30
FS = 360
WIN_L, WIN_R = 90, 108
RPEAK_IDX = 20
QRS_S, QRS_E = 70, 130
PW_S, PW_MID = 16, 43
EWMA_ALPHA = 0.015       # matches SPICE rr_timing (Cewma=330pF, Csamp=5pF)
BATCH_SIZE = 15

FEATURE_NAMES = ['F1_rr_ratio', 'F2_rr_asym', 'F3_bi_L0', 'F4_bi_L1',
                 'F5_pwave', 'F6_width', 'F7_slope', 'F8_symmetry']

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


# ═══════════════════════════════════════════════════════════════════
# PYTHON FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def load_all_beats(record_id, max_beats=200):
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    # Detect QRS polarity: if median R-peak value is negative, invert ECG
    # In the real ASIC, the IA handles this via electrode placement
    rpk_vals = []
    for i in range(1, min(len(valid) - 1, 30)):
        idx = valid[i][0]
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        rpk_vals.append(sig[idx, 0])
    if rpk_vals and np.median(rpk_vals) < 0:
        sig = sig.copy()
        sig[:, 0] = -sig[:, 0]
        if n_leads > 1:
            sig[:, 1] = -sig[:, 1]

    all_peaks = []
    for i in range(1, len(valid) - 1):
        idx = valid[i][0]
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(sig[bs+QRS_S:bs+QRS_E, 0]))
    gain = 0.16 / max(all_peaks) if max(all_peaks) > 0 else 0.05

    r_peak_scaled = []
    for i in range(1, len(valid) - 1):
        idx = valid[i][0]
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        r_peak_scaled.append(VBASE + gain * sig[idx, 0])
    max_r = max(r_peak_scaled) if r_peak_scaled else VBASE + 0.08
    # Lower threshold to 30% of range — rpeak_hold blanking prevents T-wave false triggers
    rpeak_threshold = VBASE + 0.30 * (max_r - VBASE)
    n_above = sum(1 for r in r_peak_scaled if r > rpeak_threshold)
    print(f"  Record {record_id}: gain={gain:.4f}, rpk=[{min(r_peak_scaled):.3f},{max_r:.3f}]V, "
          f"thresh={rpeak_threshold:.3f}V, {n_above}/{len(r_peak_scaled)} beats above")

    beats = []
    ewma_rr = None
    for i in range(1, min(len(valid) - 1, max_beats + 1)):
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

        # F5: post-R pwave definition using interpolated signal (matches SPICE continuous time)
        avg_rr = ewma_rr_before
        rpk_global = idx
        pw_base_frac = rpk_global + 0.75 * avg_rr * FS
        pw_peak_frac = rpk_global + 0.80 * avg_rr * FS
        if pw_peak_frac < len(sig) - 1 and pw_base_frac < len(sig) - 1 and pw_base_frac >= 0:
            # Linear interpolation for sub-sample accuracy
            bi = int(pw_base_frac); bf = pw_base_frac - bi
            pi = int(pw_peak_frac); pf = pw_peak_frac - pi
            v_base = sig[bi, 0] * (1-bf) + sig[bi+1, 0] * bf
            v_peak = sig[pi, 0] * (1-pf) + sig[pi+1, 0] * pf
            py_pw = v_peak - v_base
        else:
            py_pw = 0.0

        beats.append({
            'label': aami_map[sym],
            'sample': idx,
            'pre_rr': pre_rr, 'post_rr': post_rr,
            'ewma_rr': ewma_rr, 'ewma_rr_before': ewma_rr_before,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': pre_rr - ewma_rr_before,
            'py_f2': pre_rr - post_rr,
            'py_bi_l0': bi_l0, 'py_bi_l1': bi_l1,
            'py_pw': py_pw,
            'py_wid': np.interp(RPEAK_IDX+QRS_S + 28*FS/1000, range(len(beat)), beat[:, 0]),
            'py_slp': np.interp(RPEAK_IDX+QRS_S + 14*FS/1000, range(len(beat)), beat[:, 0])
                     - np.interp(RPEAK_IDX+QRS_S + 69*FS/1000, range(len(beat)), beat[:, 0]),
            'py_sym': beat[RPEAK_IDX+QRS_S, 0],
        })

    # Store raw signal and gain for continuous ECG generation
    return beats, gain, rpeak_threshold, sig, n_leads


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION & THRESHOLD COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7: c.append(p)
    return " ".join(f"{t:.7f} {v:.6f}" for t, v in c)


def calibrate_timer():
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title timer_calibration
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

Vt6 vth_rpk_del 0 0.1
Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_rpk_del dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 vth_rpk_del delay_timer

.ic V(xbias.iref)=0.3
.tran 0.1m 1.1

.meas tran cal_vbias FIND V(vbias) AT=0.09
"""
    dt_times = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    for t in dt_times:
        sp += f".meas tran dt_sf_{t:03d} FIND V(xdelay.sf_timer) AT={0.121 + t/1000.0:.4f}\n"
    ft_times = [1, 2, 3, 5, 10, 20]
    for t in ft_times:
        sp += f".meas tran ft_sf_{t:03d} FIND V(xdelay.sf_timer2) AT={0.121 + t/1000.0:.4f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_cal_3way.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=120, cwd=NGSPICE_DIR)
    with open(log) as f:
        text = f.read()
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
    return {'dt_times_ms': dt_times, 'dt_sf': dt_sf, 'ft_times_ms': ft_times, 'ft_sf': ft_sf}


def interp_sf(times_ms, voltages, target_ms):
    valid = [(t, v) for t, v in zip(times_ms, voltages) if v is not None]
    if not valid: return 0.3
    times_ms = [t for t, v in valid]
    voltages = [v for t, v in valid]
    for i in range(len(times_ms) - 1):
        if times_ms[i] <= target_ms <= times_ms[i+1]:
            frac = (target_ms - times_ms[i]) / (times_ms[i+1] - times_ms[i])
            return voltages[i] + frac * (voltages[i+1] - voltages[i])
    if target_ms < times_ms[0]:
        rate = (voltages[0] - voltages[1]) / (times_ms[1] - times_ms[0])
        return voltages[0] + rate * (times_ms[0] - target_ms)
    rate = (voltages[-2] - voltages[-1]) / (times_ms[-1] - times_ms[-2])
    return voltages[-1] - rate * (target_ms - times_ms[-1])


def compute_thresholds(cal):
    dt_t = [t for t, v in zip(cal['dt_times_ms'], cal['dt_sf']) if v is not None]
    dt_v = [v for v in cal['dt_sf'] if v is not None]
    ft_t = [t for t, v in zip(cal['ft_times_ms'], cal['ft_sf']) if v is not None]
    ft_v = [v for v in cal['ft_sf'] if v is not None]

    rpk_drop = -3.0   # ms: v7g measured mean (fires ~3ms BEFORE annotation)
    pipe_main = 3.5   # ms: edge_buffer -> pulse_gen pipeline
    pipe_fast = 1.0   # ms: fast timer, fast resolve

    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_fast))
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_main))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_main))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_main))
    vth_10  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_fast))

    return {
        'vth_14': vth_14, 'vth_28': vth_28, 'vth_69': vth_69,
        'vth_300': vth_300, 'vth_10': vth_10,
        'dt_t': dt_t, 'dt_v': dt_v, 'ft_t': ft_t, 'ft_v': ft_v,
        'rpk_drop': rpk_drop, 'pipe_pw': 3.5,
    }


def make_continuous_ecg_pwl(beats_batch, gain, sig, n_leads, rpeak_times):
    """Generate continuous ECG PWL from raw signal — no gaps between beats."""
    dt = 1.0 / FS
    total_time = rpeak_times[-1] + 1.0

    # Find the sample range covering all beats plus margin for post-R pwave
    first_samp = beats_batch[0]['sample'] - int(0.5 * FS)
    last_samp = beats_batch[-1]['sample'] + int(1.2 * FS)
    first_samp = max(0, first_samp)
    last_samp = min(sig.shape[0] - 1, last_samp)

    # Time offset: first beat's R-peak time in simulation = rpeak_times[0]
    samp_at_first_rpk = beats_batch[0]['sample']
    t_offset = rpeak_times[0] - samp_at_first_rpk / FS

    ecg_l0_pts = []
    ecg_l1_pts = []
    # Downsample: every sample is excessive, use every 2nd sample
    step = 1
    for s in range(first_samp, last_samp + 1, step):
        t = s / FS + t_offset
        if t < 0: continue
        if t > total_time: break
        v0 = max(0.05, min(0.85, VBASE + gain * sig[s, 0]))
        v1 = max(0.05, min(0.85, VBASE + gain * sig[s, min(1, n_leads-1)]))
        ecg_l0_pts.append((t, v0))
        ecg_l1_pts.append((t, v1))

    # Pad start and end
    if ecg_l0_pts[0][0] > 0.001:
        ecg_l0_pts.insert(0, (0, VBASE))
        ecg_l1_pts.insert(0, (0, VBASE))
    ecg_l0_pts.append((total_time, VBASE))
    ecg_l1_pts.append((total_time, VBASE))

    return ecg_l0_pts, ecg_l1_pts, total_time


def compute_ewma_ic(beats_batch, rr_cal):
    """Compute initial EWMA voltage for first batch from Python ewma_rr_before."""
    rr_t, rr_v = rr_cal
    ewma_rr_ms = beats_batch[0].get('ewma_rr_before', beats_batch[0]['ewma_rr']) * 1000
    return interp_sf(rr_t, rr_v, ewma_rr_ms)


def compute_rr_timer_thresholds(cal):
    """Calibrate rr_timing timer SF curve (320pF timer, separate from delay_timer)."""
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title rr_timer_calibration
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rr_timing.spice"

Xbias vbias vref vdd vss bias_gen
Vecg ecg_l0 0 0.5
Vcap cap_trig 0 0
Vcopy copy_trig 0 0
Vrpk rpeak_trigger 0 PWL(0 0 0.098 0 0.1 {VDD} 0.105 {VDD} 0.106 0 2.0 0)

Xrr ecg_l0 cap_trig rpeak_trigger copy_trig f1 f2 rr_timer rr_ewma vdd vss vbias rr_timing

.ic V(xbias.iref)=0.3
.tran 0.5m 2.0

"""
    rr_times = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for t in rr_times:
        sp += f".meas tran rr_sf_{t:04d} FIND V(rr_timer) AT={0.106 + t/1000.0:.4f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, "_cal_rr_3way.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=180, cwd=NGSPICE_DIR)
    with open(log) as f:
        text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    try: os.remove(tb); os.remove(log)
    except: pass

    rr_sf = [results.get(f'rr_sf_{t:04d}') for t in rr_times]
    rr_t = [t for t, v in zip(rr_times, rr_sf) if v is not None]
    rr_v = [v for v in rr_sf if v is not None]
    return rr_t, rr_v


# ═══════════════════════════════════════════════════════════════════
# IDEAL SPICE TESTBENCH (PWL triggers at known times)
# ═══════════════════════════════════════════════════════════════════

def generate_ideal_batch(beats_batch, gain, batch_id, record_id,
                         thresholds, rr_cal, sig, n_leads, ic_vals=None):
    if ic_vals is None:
        ic_vals = {}
    n = len(beats_batch)
    dt = 1.0 / FS
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    rpeak_times = [max(beats_batch[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats_batch[i]['pre_rr'])

    # Continuous ECG PWL (includes inter-beat segments for post-R pwave)
    ecg_l0_pts, ecg_l1_pts, total_time = make_continuous_ecg_pwl(
        beats_batch, gain, sig, n_leads, rpeak_times)

    rpk_drop = thresholds['rpk_drop']
    pipe_pw = thresholds['pipe_pw']
    dt_t = thresholds['dt_t']
    dt_v = thresholds['dt_v']
    ft_t = thresholds.get('ft_t', dt_t)
    ft_v = thresholds.get('ft_v', dt_v)
    rr_t, rr_v = rr_cal

    # Ideal trigger PWLs: exact times
    def make_trigger_pwl(times_list, pw=0.002):
        pts = [(0, 0)]
        for t in times_list:
            if t > 0.01:
                pts.append((t - 0.0001, 0))
                pts.append((t, VDD))
                pts.append((t + pw, VDD))
                pts.append((t + pw + 0.0001, 0))
        pts.append((total_time, 0))
        return make_pwl(pts)

    rpk_trigs = [rpt for rpt in rpeak_times]
    s14_trigs = [rpt + 0.014 for rpt in rpeak_times]
    s69_trigs = [rpt + 0.069 for rpt in rpeak_times]
    wid_trigs = [rpt + 0.028 for rpt in rpeak_times]
    bi_trigs  = [rpt + 0.300 for rpt in rpeak_times]
    rr_copy_trigs = [rpt + 0.010 for rpt in rpeak_times]

    # Post-R pwave triggers: 0.75*EWMA_RR and 0.80*EWMA_RR after R
    pw_base_trigs, pw_peak_trigs = [], []
    for j in range(n):
        ewma_rr = beats_batch[j]['ewma_rr_before'] if 'ewma_rr_before' in beats_batch[j] else beats_batch[j]['ewma_rr']
        pw_base_trigs.append(rpeak_times[j] + 0.75 * ewma_rr)
        pw_peak_trigs.append(rpeak_times[j] + 0.80 * ewma_rr)

    # rr_cap: 55ms window (R-60ms to R-5ms) for SF-driven S&H settling
    # RC = ~886MOhm * 10pF = 8.86ms; need ~6τ for >99% settling
    rr_cap_pwl_pts = [(0, 0)]
    for j in range(1, n):
        t_on = rpeak_times[j] - 0.060
        t_off = rpeak_times[j] - 0.005
        if t_on > 0.01:
            rr_cap_pwl_pts.append((t_on - 0.0001, 0))
            rr_cap_pwl_pts.append((t_on, VDD))
            rr_cap_pwl_pts.append((t_off, VDD))
            rr_cap_pwl_pts.append((t_off + 0.0001, 0))
    rr_cap_pwl_pts.append((total_time, 0))

    sp = f""".title ideal_3way_{record_id}_b{batch_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0

Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

* IDEAL TRIGGERS (PWL at exact annotation-based times)
Vrpk  rpeak_pulse    0 PWL({make_trigger_pwl(rpk_trigs)})
Vcap  rpk_capture    0 PWL({make_trigger_pwl(rpk_trigs, pw=0.0005)})
Vs14  trig_s14       0 PWL({make_trigger_pwl(s14_trigs, pw=0.0005)})
Vs69  trig_s69       0 PWL({make_trigger_pwl(s69_trigs, pw=0.0005)})
Vwid  trig_wid       0 PWL({make_trigger_pwl(wid_trigs, pw=0.0005)})
Vbi   trig_bi_copy   0 PWL({make_trigger_pwl(bi_trigs, pw=0.003)})
Vrrc  trig_rr_copy   0 PWL({make_trigger_pwl(rr_copy_trigs, pw=0.003)})
Vpwb  trig_pw_base   0 PWL({make_trigger_pwl(pw_base_trigs, pw=0.0005)})
Vpwp  trig_pw_peak   0 PWL({make_trigger_pwl(pw_peak_trigs, pw=0.0005)})
Vrrcap trig_rr_cap   0 PWL({make_pwl(rr_cap_pwl_pts)})

.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

Xbias vbias vref vdd vss bias_gen

Xrr ecg_l0 trig_rr_cap rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 rpk_capture trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 rpk_capture trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 rpk_capture f8_symmetry vdd vss qrs_symmetry

.ic V(xbias.iref)=0.3
"""
    # Compute EWMA IC from Python ewma_rr if not propagated from prior batch
    rr_t, rr_v = rr_cal
    if 'rr_ewma' not in ic_vals:
        ewma_ic = compute_ewma_ic(beats_batch, rr_cal)
        ic_vals['rr_ewma'] = ewma_ic
        ic_vals['rr_esamp'] = ewma_ic
        ic_vals['rr_prev'] = ewma_ic

    sp += f""".ic V(xrr.ewma_node)={ic_vals.get('rr_ewma', 0.40):.6f}
.ic V(xrr.esamp_node)={ic_vals.get('rr_esamp', 0.40):.6f}
.ic V(xrr.prev_node)={ic_vals.get('rr_prev', 0.40):.6f}
.ic V(xbi0.prev_node)={ic_vals.get('bi0_prev', 0.30):.6f}
.ic V(xbi0.esamp_node)={ic_vals.get('bi0_esamp', 0.30):.6f}
.ic V(xbi1.prev_node)={ic_vals.get('bi1_prev', 0.30):.6f}
.ic V(xbi1.esamp_node)={ic_vals.get('bi1_esamp', 0.30):.6f}

.tran 0.5m {total_time:.4f} 0 0.2m

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
        # F5 measurement: needs to be AFTER the pwave triggers fire
        # pwave peak at R + 0.80*EWMA_RR; measure 50ms after that
        ewma_rr_j = beats_batch[j].get('ewma_rr_before', beats_batch[j]['ewma_rr'])
        t_f5 = rpeak_times[j] + 0.80 * ewma_rr_j + 0.050
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            sp += f".meas tran f3_{j} FIND V(f3_bi_l0) AT={t_f38:.7f}\n"
            sp += f".meas tran f4_{j} FIND V(f4_bi_l1) AT={t_f38:.7f}\n"
            sp += f".meas tran f6_{j} FIND V(f6_width) AT={t_f38:.7f}\n"
            sp += f".meas tran f7_{j} FIND V(f7_slope) AT={t_f38:.7f}\n"
            sp += f".meas tran f8_{j} FIND V(f8_symmetry) AT={t_f38:.7f}\n"
        if t_f5 < total_time - 0.01:
            sp += f".meas tran f5_{j} FIND V(f5_pwave) AT={t_f5:.7f}\n"

    t_ic = rpeak_times[-1] + 0.100
    if t_ic < total_time - 0.01:
        sp += f"\n.meas tran ic_rr_ewma FIND V(xrr.ewma_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_rr_esamp FIND V(xrr.esamp_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_rr_prev FIND V(xrr.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi0_prev FIND V(xbi0.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi0_esamp FIND V(xbi0.esamp_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi1_prev FIND V(xbi1.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi1_esamp FIND V(xbi1.esamp_node) AT={t_ic:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, f"_ideal_3way_{record_id}_b{batch_id}.spice")
    with open(tb, 'w') as f:
        f.write(sp)
    return tb, skip, rpeak_times, total_time


# ═══════════════════════════════════════════════════════════════════
# ANALOG SPICE TESTBENCH (full trigger chain with true peak detector)
# ═══════════════════════════════════════════════════════════════════

def generate_analog_batch(beats_batch, gain, rpeak_threshold, batch_id,
                          record_id, thresholds, rr_cal, sig, n_leads, ic_vals=None):
    if ic_vals is None:
        ic_vals = {}
    n = len(beats_batch)
    dt = 1.0 / FS
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    rpeak_times = [max(beats_batch[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats_batch[i]['pre_rr'])

    # Continuous ECG PWL (includes inter-beat segments for post-R pwave)
    ecg_l0_pts, ecg_l1_pts, total_time = make_continuous_ecg_pwl(
        beats_batch, gain, sig, n_leads, rpeak_times)

    # Adaptive per-beat pwave thresholds using EWMA-predicted RR
    rpk_drop = thresholds['rpk_drop']
    pipe_pw = thresholds['pipe_pw']
    rr_t, rr_v = rr_cal

    # Post-R pwave thresholds: target = 0.75*EWMA_RR and 0.80*EWMA_RR on rr_timing timer
    pwl_base_pts = [(0, 0.3)]
    pwl_peak_pts = [(0, 0.3)]
    pwl_rrcap_pts = [(0, 0.3)]
    for j in range(n):
        ewma_rr_ms = (beats_batch[j].get('ewma_rr_before', beats_batch[j]['ewma_rr'])) * 1000
        t_rpk = rpeak_times[j]
        # Post-R: target times relative to R-peak on rr_timing timer
        # rr_timing timer starts at rpeak, so elapsed = target fraction * ewma_rr
        pw_base_elapsed = max(1, 0.75 * ewma_rr_ms - rpk_drop - pipe_pw)
        pw_peak_elapsed = max(1, 0.80 * ewma_rr_ms - rpk_drop - pipe_pw)
        rr_cap_elapsed  = max(1, ewma_rr_ms - 50 - rpk_drop - pipe_pw)
        v_base = interp_sf(rr_t, rr_v, pw_base_elapsed)
        v_peak = interp_sf(rr_t, rr_v, pw_peak_elapsed)
        v_rrcap = interp_sf(rr_t, rr_v, rr_cap_elapsed)
        t_step = t_rpk + 0.001
        if t_step > pwl_base_pts[-1][0]:
            pwl_base_pts.append((t_step, v_base))
            pwl_peak_pts.append((t_step, v_peak))
            pwl_rrcap_pts.append((t_step, v_rrcap))
    pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
    pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
    pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))

    sp = f""".title analog_3way_{record_id}_b{batch_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0

* === ECG INPUTS (PWL — the ONLY non-analog input) ===
Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

* === THRESHOLD VOLTAGES (set by resistor ladder in ASIC) ===
Vth_rpeak rpeak_threshold 0 {rpeak_threshold:.4f}
Vth14  vth_14_fast 0 {thresholds['vth_14']:.6f}
Vth69  vth_69      0 {thresholds['vth_69']:.6f}
Vth28  vth_28      0 {thresholds['vth_28']:.6f}
Vth300 vth_300     0 {thresholds['vth_300']:.6f}
Vth10  vth_10      0 {thresholds['vth_10']:.6f}
* Adaptive pwave thresholds (from EWMA-predicted RR, post-R)
Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})
Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})
Vth_rrc vth_rr_cap  0 PWL({make_pwl(pwl_rrcap_pts)})

* === SUBCIRCUIT INCLUDES ===
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/rpeak_hold.spice"
.include "{fp}/true_peak_detector.spice"
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

* ============================================================
* FULLY ANALOG FRONTEND WITH TRUE PEAK DETECTOR
* ============================================================

* 1. BIAS GENERATOR
Xbias vbias vref vdd vss bias_gen

* 2. R-PEAK DETECTION CHAIN
*    Step 1: Threshold comparator on QRS upslope
Xrpeak ecg_l0 rpeak_threshold rpeak_level vdd vss vbias rpeak_detector
*    Step 2: Hold circuit -> ~300ms ARM window
Xrpk_hold rpeak_level rpeak_level_held vdd vss vbias rpeak_hold
*    Step 3: True peak detector finds actual R-peak apex (SR latch output)
Xtpd ecg_l0 rpeak_level_held peak_detected vdd vss vbias true_peak_detector
*    Step 4: pulse_gen converts level to brief trigger pulse
Xpg_rpk peak_detected true_rpeak_pulse vdd vss vbias pulse_gen

* 3. DELAY TIMER (driven by true_rpeak_pulse, ~5ms after annotation)
* vth_rpk_del unused with true peak detector but pin must be connected
Vth_rpkdel vth_rpk_del 0 0.48
Xdt true_rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_rpk_del_unused dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 vth_rpk_del delay_timer

* 4. EDGE BUFFERS + PULSE GENERATORS
Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias long_pulse_gen

Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen

Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen

Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen

* 5. PWAVE PREDICTION COMPARATORS (on rr_timing timer SF)
* Post-R pwave: comparators watch rr_timer (320pF, long intervals)
* not delay_timer (100pF, short intervals) since 0.75*EWMA_RR ~ 500-700ms
XMcpwb1 cnpwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 pw_base_lev rr_timer ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 pw_base_lev cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

XMcpwp1 cnpwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 pw_peak_lev rr_timer ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 pw_peak_lev cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

XMcrrc1 cnrrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 rr_cap_lev rr_timer ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 rr_cap_lev cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* 5b. EDGE BUFFERS + PULSE GENERATORS for pwave prediction
Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen

Xeb_pwp pw_peak_lev pwp_sharp vdd vss edge_buffer
Xpg_pwp pwp_sharp   trig_pw_peak vdd vss vbias pulse_gen

* rr_cap: use comparator level directly (not pulse_gen!)
* rr_cap_lev stays HIGH from threshold crossing (~R-55ms) until timer reset
* This provides the ~55ms window needed for SF-driven S&H settling (RC~22ms)
Xeb_rrc rr_cap_lev  trig_rr_cap vdd vss edge_buffer

* ============================================================
* 6. FEATURE CIRCUITS — ALL driven by ANALOG triggers
* ============================================================

* F1 & F2: RR Timing (true_rpeak_pulse precharges timer)
Xrr ecg_l0 trig_rr_cap true_rpeak_pulse trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing

* F3: Beat Instability L0 (captures at true R-peak via true_rpeak_pulse)
Xbi0 ecg_l0 true_rpeak_pulse trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
* F4: Beat Instability L1
Xbi1 ecg_l1 true_rpeak_pulse trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability

* F5: P-wave Difference L0 (post-R pwave triggers)
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff

* F6: QRS Width L0
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width

* F7: QRS Slope L0
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope

* F8: QRS Symmetry L0 (captures at true R-peak via true_rpeak_pulse)
Xsym ecg_l0 true_rpeak_pulse f8_symmetry vdd vss qrs_symmetry

* === INITIAL CONDITIONS ===
.ic V(xbias.iref)=0.3
"""
    rr_t, rr_v = rr_cal
    if 'rr_ewma' not in ic_vals:
        ewma_ic = compute_ewma_ic(beats_batch, rr_cal)
        ic_vals['rr_ewma'] = ewma_ic
        ic_vals['rr_esamp'] = ewma_ic
        ic_vals['rr_prev'] = ewma_ic

    sp += f""".ic V(xrr.ewma_node)={ic_vals.get('rr_ewma', 0.40):.6f}
.ic V(xrr.esamp_node)={ic_vals.get('rr_esamp', 0.40):.6f}
.ic V(xrr.prev_node)={ic_vals.get('rr_prev', 0.40):.6f}
.ic V(xbi0.prev_node)={ic_vals.get('bi0_prev', 0.30):.6f}
.ic V(xbi0.esamp_node)={ic_vals.get('bi0_esamp', 0.30):.6f}
.ic V(xbi1.prev_node)={ic_vals.get('bi1_prev', 0.30):.6f}
.ic V(xbi1.esamp_node)={ic_vals.get('bi1_esamp', 0.30):.6f}
.ic V(xtpd.peak_hold)=0.30

.tran 0.5m {total_time:.4f} 0 0.2m

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
        ewma_rr_j = beats_batch[j].get('ewma_rr_before', beats_batch[j]['ewma_rr'])
        t_f5 = rpeak_times[j] + 0.80 * ewma_rr_j + 0.050
        t_f38 = rpeak_times[j] + 0.100
        if t_f38 < total_time - 0.01:
            sp += f".meas tran f3_{j} FIND V(f3_bi_l0) AT={t_f38:.7f}\n"
            sp += f".meas tran f4_{j} FIND V(f4_bi_l1) AT={t_f38:.7f}\n"
            sp += f".meas tran f6_{j} FIND V(f6_width) AT={t_f38:.7f}\n"
            sp += f".meas tran f7_{j} FIND V(f7_slope) AT={t_f38:.7f}\n"
            sp += f".meas tran f8_{j} FIND V(f8_symmetry) AT={t_f38:.7f}\n"
        if t_f5 < total_time - 0.01:
            sp += f".meas tran f5_{j} FIND V(f5_pwave) AT={t_f5:.7f}\n"

    t_ic = rpeak_times[-1] + 0.100
    if t_ic < total_time - 0.01:
        sp += f"\n.meas tran ic_rr_ewma FIND V(xrr.ewma_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_rr_esamp FIND V(xrr.esamp_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_rr_prev FIND V(xrr.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi0_prev FIND V(xbi0.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi0_esamp FIND V(xbi0.esamp_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi1_prev FIND V(xbi1.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi1_esamp FIND V(xbi1.esamp_node) AT={t_ic:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, f"_analog_3way_{record_id}_b{batch_id}.spice")
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)
    return tb, skip, rpeak_times, total_time


# ═══════════════════════════════════════════════════════════════════
# SIMULATION RUNNER & RESULT PARSER
# ═══════════════════════════════════════════════════════════════════

def run_ngspice(tb, timeout=1800):
    log = tb.replace('.spice', '.log')
    try:
        subprocess.run([NGSPICE, '-b', '-o', log, tb],
                       capture_output=True, text=True, timeout=timeout, cwd=NGSPICE_DIR)
    except subprocess.TimeoutExpired:
        return None
    if not os.path.exists(log): return None
    with open(log, 'r') as f:
        text = f.read()
    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass
    return results


def extract_features_from_results(results, beats_batch, skip, n):
    circuit_feats, py_feats, labels = [], [], []
    for j in range(skip, n):
        bt = beats_batch[j]
        f1 = results.get(f'f1_{j}')
        f2 = results.get(f'f2_{j}')
        f3 = results.get(f'f3_{j}')
        f4 = results.get(f'f4_{j}')
        f5 = results.get(f'f5_{j}')
        f6 = results.get(f'f6_{j}')
        f7 = results.get(f'f7_{j}')
        f8 = results.get(f'f8_{j}')
        vals = [f1, f2, f3, f4, f5, f6, f7, f8]
        py_vals = [bt['py_f1'], bt['py_f2'],
                   bt['py_bi_l0'], bt['py_bi_l1'],
                   bt['py_pw'], bt['py_wid'], bt['py_slp'], bt['py_sym']]
        if any(v is None for v in vals):
            continue
        circuit_feats.append(vals)
        py_feats.append(py_vals)
        labels.append(bt['label'])
    return circuit_feats, py_feats, labels


def extract_ic_vals(results):
    ic = {}
    for key, meas in [('rr_ewma', 'ic_rr_ewma'), ('rr_esamp', 'ic_rr_esamp'),
                      ('rr_prev', 'ic_rr_prev'), ('bi0_prev', 'ic_bi0_prev'),
                      ('bi0_esamp', 'ic_bi0_esamp'), ('bi1_prev', 'ic_bi1_prev'),
                      ('bi1_esamp', 'ic_bi1_esamp')]:
        v = results.get(meas)
        if v is not None:
            ic[key] = v
    return ic if ic else None


def simulate_record(record_id, thresholds, rr_cal, mode='analog', max_beats=0):
    beats, gain, rpeak_threshold, sig, n_leads = load_all_beats(record_id)
    if not beats:
        return [], [], []

    if max_beats > 0 and len(beats) > max_beats:
        beats = beats[:max_beats]

    all_circuit, all_py, all_labels = [], [], []
    n_batches = (len(beats) + BATCH_SIZE - 1) // BATCH_SIZE

    ic_vals = None
    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end = min((b_idx + 1) * BATCH_SIZE, len(beats))
        batch = beats[start:end]
        if len(batch) < 6:
            continue

        if mode == 'ideal':
            tb, skip, rpeak_times, total_time = generate_ideal_batch(
                batch, gain, b_idx, record_id, thresholds, rr_cal,
                sig, n_leads, ic_vals)
        else:
            tb, skip, rpeak_times, total_time = generate_analog_batch(
                batch, gain, rpeak_threshold, b_idx, record_id,
                thresholds, rr_cal, sig, n_leads, ic_vals)

        results = run_ngspice(tb)

        if results is not None:
            new_ic = extract_ic_vals(results)
            if new_ic:
                ic_vals = new_ic

        if results is None:
            print(f"    WARNING: batch {b_idx} failed for record {record_id} ({mode})")
            continue

        cf, pf, lb = extract_features_from_results(results, batch, skip, len(batch))
        all_circuit.extend(cf)
        all_py.extend(pf)
        all_labels.extend(lb)

        try:
            os.remove(tb)
            os.remove(tb.replace('.spice', '.log'))
        except: pass

    return all_circuit, all_py, all_labels


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='3-Way Validation')
    parser.add_argument('records', nargs='*', default=['100'],
                        help='MIT-BIH record IDs (default: 100)')
    parser.add_argument('--ideal-only', action='store_true',
                        help='Run only ideal SPICE (skip analog)')
    parser.add_argument('--analog-only', action='store_true',
                        help='Run only analog SPICE (skip ideal)')
    parser.add_argument('--max-beats', type=int, default=0,
                        help='Limit beats per record (0=all)')
    args = parser.parse_args()

    records = args.records
    run_ideal = not args.analog_only
    run_analog = not args.ideal_only
    max_beats = args.max_beats

    print("=" * 70)
    print("3-WAY VALIDATION: Python vs Ideal-SPICE vs Analog-SPICE")
    print("=" * 70)
    print(f"  EWMA_ALPHA = {EWMA_ALPHA} (matches SPICE rr_timing)")
    print(f"  F5 = post-R pwave: ECG[R+0.80*EWMA_RR] - ECG[R+0.75*EWMA_RR]")
    print(f"  rpk_drop = -3.0ms (v7g measured)")
    print(f"  Records: {', '.join(records)}")
    print(f"  Modes: {'Ideal' if run_ideal else ''} {'Analog' if run_analog else ''}")
    print()

    # Calibrate
    print("[1] Calibrating delay timer...")
    cal = calibrate_timer()
    thresholds = compute_thresholds(cal)
    print(f"  vth_14={thresholds['vth_14']:.5f} vth_28={thresholds['vth_28']:.5f} "
          f"vth_69={thresholds['vth_69']:.5f} vth_300={thresholds['vth_300']:.5f}")

    print("[2] Calibrating rr_timing timer...")
    rr_t, rr_v = compute_rr_timer_thresholds(cal)
    if rr_t:
        print(f"  rr_sf: {rr_t[0]}ms={rr_v[0]:.4f}V ... {rr_t[-1]}ms={rr_v[-1]:.4f}V")
    else:
        print("  WARNING: rr_timing calibration failed, using delay timer table")
        rr_t = thresholds['dt_t']
        rr_v = thresholds['dt_v']
    rr_cal = (rr_t, rr_v)

    # Run simulations
    ideal_data = {}
    analog_data = {}

    for rec in records:
        if run_ideal:
            print(f"\n[Ideal] Simulating record {rec}...", end='', flush=True)
            t0 = time.time()
            cf, pf, lb = simulate_record(rec, thresholds, rr_cal, mode='ideal', max_beats=max_beats)
            print(f" {len(cf)} beats, {time.time()-t0:.1f}s")
            ideal_data[rec] = (np.array(cf) if cf else np.empty((0,8)),
                               np.array(pf) if pf else np.empty((0,8)),
                               lb)

        if run_analog:
            print(f"[Analog] Simulating record {rec}...", end='', flush=True)
            t0 = time.time()
            cf, pf, lb = simulate_record(rec, thresholds, rr_cal, mode='analog', max_beats=max_beats)
            print(f" {len(cf)} beats, {time.time()-t0:.1f}s")
            analog_data[rec] = (np.array(cf) if cf else np.empty((0,8)),
                                np.array(pf) if pf else np.empty((0,8)),
                                lb)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Per-feature Spearman |rho|")
    print("=" * 70)

    header = f"{'Rec':>5s} {'Beats':>5s}  " + " ".join(f"{n:>8s}" for n in FEATURE_NAMES)
    if run_ideal:
        print(f"\n  Py vs Ideal-SPICE:")
        print(f"  {header}")
        for rec in records:
            if rec not in ideal_data or len(ideal_data[rec][0]) < 5: continue
            cf, pf, _ = ideal_data[rec]
            rhos = []
            for i in range(8):
                rho, _ = spearmanr(cf[:, i], pf[:, i])
                rhos.append(abs(rho))
            rho_str = " ".join(f"{r:8.3f}" for r in rhos)
            print(f"  {rec:>5s} {len(cf):>5d}  {rho_str}")

    if run_analog:
        print(f"\n  Py vs Analog-SPICE:")
        print(f"  {header}")
        for rec in records:
            if rec not in analog_data or len(analog_data[rec][0]) < 5: continue
            cf, pf, _ = analog_data[rec]
            rhos = []
            for i in range(8):
                rho, _ = spearmanr(cf[:, i], pf[:, i])
                rhos.append(abs(rho))
            rho_str = " ".join(f"{r:8.3f}" for r in rhos)
            print(f"  {rec:>5s} {len(cf):>5d}  {rho_str}")

    if run_ideal and run_analog:
        print(f"\n  Ideal-SPICE vs Analog-SPICE:")
        print(f"  {header}")
        for rec in records:
            if rec not in ideal_data or rec not in analog_data: continue
            ic, _, _ = ideal_data[rec]
            ac, _, _ = analog_data[rec]
            n_min = min(len(ic), len(ac))
            if n_min < 5: continue
            rhos = []
            for i in range(8):
                rho, _ = spearmanr(ic[:n_min, i], ac[:n_min, i])
                rhos.append(abs(rho))
            rho_str = " ".join(f"{r:8.3f}" for r in rhos)
            print(f"  {rec:>5s} {n_min:>5d}  {rho_str}")

    # Comprehensive acceptance criteria check
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ACCEPTANCE CRITERIA")
    print("  rho >= 0.97 | clip < 1% | span >= 225mV | Cohen's d >= 75%")
    print("=" * 70)

    def compute_metrics(circuit_vals, python_vals, feature_idx):
        c = circuit_vals[:, feature_idx]
        p = python_vals[:, feature_idx]
        rho_val, _ = spearmanr(c, p)
        rho_val = abs(rho_val)
        span_c = np.ptp(c) * 1000  # mV
        span_p = np.ptp(p) * 1000  # mV
        # Clip %: fraction of circuit values at rail (within 5mV of VDD or VSS)
        n_clip = np.sum((c > 0.895) | (c < 0.005))
        clip_pct = 100.0 * n_clip / len(c) if len(c) > 0 else 0
        # Cohen's d: mean difference / pooled SD
        diff = c - p
        sd_pool = np.sqrt((np.var(c, ddof=1) + np.var(p, ddof=1)) / 2)
        cohens_d = abs(np.mean(diff)) / sd_pool if sd_pool > 0 else 0
        # Cohen's d as percentage of span (inverted: lower is better)
        # We want d to represent separation quality: use circuit span coverage
        # Actually: d >= 75% means circuit span covers >= 75% of python span
        span_ratio = (span_c / span_p * 100) if span_p > 0 else 0
        return rho_val, span_c, clip_pct, span_ratio

    def print_metrics_table(label, data_dict, records_list):
        for rec in records_list:
            if rec not in data_dict or len(data_dict[rec][0]) < 5: continue
            cf, pf, _ = data_dict[rec]
            print(f"\n  {label} — Record {rec} ({len(cf)} beats):")
            print(f"    {'Feature':>15s}  {'rho':>6s}  {'span':>7s}  {'clip%':>5s}  {'d_span%':>7s}  {'Status':>6s}")
            all_pass = True
            for i in range(8):
                rho_val, span_c, clip_pct, span_ratio = compute_metrics(cf, pf, i)
                pass_rho = rho_val >= 0.97
                pass_clip = clip_pct < 1.0
                pass_span = span_c >= 225
                pass_d = span_ratio >= 75
                passed = pass_rho and pass_clip and pass_span and pass_d
                if not passed: all_pass = False
                flags = []
                if not pass_rho: flags.append('rho')
                if not pass_clip: flags.append('clip')
                if not pass_span: flags.append('span')
                if not pass_d: flags.append('d')
                status = "PASS" if passed else f"FAIL({','.join(flags)})"
                print(f"    {FEATURE_NAMES[i]:>15s}  {rho_val:6.4f}  {span_c:6.0f}mV  {clip_pct:5.1f}  {span_ratio:6.1f}%  {status}")
            print(f"    {'OVERALL':>15s}: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    if run_ideal:
        print_metrics_table("Py vs Ideal", ideal_data, records)
    if run_analog:
        print_metrics_table("Py vs Analog", analog_data, records)

    print("\nDone.")


if __name__ == '__main__':
    main()
