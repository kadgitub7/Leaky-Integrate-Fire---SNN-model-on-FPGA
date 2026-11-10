"""
FINAL END-TO-END TEST: Fully Analog AFE -> RF Classifier
=========================================================
MIT-BIH ECG (PWL) -> Fully Analog AFE (SKY130) -> 8 Features -> RF Classifier -> Predictions

ALL triggers are analog. The ONLY external input is the raw ECG waveform.
Analog chain:
  ECG -> rpeak_detector -> pulse_gen -> delay_timer -> edge_buffer -> pulse_gen(x5)
      -> pwave prediction comparators -> edge_buffer -> pulse_gen(x3)
      -> feature circuits (8) -> voltages

Compares:
  1. RF predictions from Python-computed features (ground truth)
  2. RF predictions from analog circuit-extracted features (calibrated)
"""
import subprocess, os, re, sys, time, json
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings; warnings.filterwarnings('ignore')

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "Complete_imp", "frontend_optimization"))
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9; VBASE = 0.30; FS = 360
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130; RPEAK_IDX = 20
PW_S = 16; PW_MID = 43
BATCH_SIZE = 20
EWMA_ALPHA = 0.10

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']
DS2_TEST = ['100', '119', '208', '232', '205']

FEATURE_NAMES = ['rr_ratio', 'rr_asymmetry', 'beat_instability_L0',
                 'beat_instability_L1', 'pwave_energy_L0', 'qrs_width_L0',
                 'slope_ratio_L0', 'qrs_symmetry_L0']

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


# ═══════════════════════════════════════════════════════════════════
# PYTHON FEATURE EXTRACTION (matches circuit definitions exactly)
# ═══════════════════════════════════════════════════════════════════

def extract_python_features(rec_list):
    """Extract the 8 features that match our analog circuit outputs."""
    all_feats, all_labels = [], []
    for rec_id in rec_list:
        record = wfdb.rdrecord(os.path.join(MITDB_DIR, rec_id))
        ann = wfdb.rdann(os.path.join(MITDB_DIR, rec_id), 'atr')
        sig = record.p_signal
        n_leads = sig.shape[1]
        valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

        ewma_rr = None
        for i in range(1, len(valid) - 1):
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

            feat = [
                pre_rr - ewma_rr_before,                          # F1: rr_ratio (diff)
                pre_rr - post_rr,                                  # F2: rr_asymmetry (diff)
                bi_l0,                                             # F3: beat_instability L0
                bi_l1,                                             # F4: beat_instability L1
                beat[PW_MID, 0] - beat[PW_S, 0],                  # F5: pwave energy L0
                beat[RPEAK_IDX+QRS_S+10, 0],                      # F6: qrs_width L0
                beat[RPEAK_IDX+QRS_S+5, 0] - beat[RPEAK_IDX+QRS_S+25, 0],  # F7: slope_ratio L0
                beat[RPEAK_IDX+QRS_S, 0],                         # F8: qrs_symmetry L0
            ]
            all_feats.append(feat)
            all_labels.append(aami_map[sym])
    return np.array(all_feats), np.array(all_labels)


def extract_python_features_per_record(rec_id):
    """Extract per-beat features for a single record (for comparison with analog)."""
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, rec_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, rec_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]
    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol) if sym in aami_map]

    feats, labels = [], []
    ewma_rr = None
    for i in range(1, len(valid) - 1):
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

        feat = [
            pre_rr - ewma_rr_before,
            pre_rr - post_rr,
            bi_l0, bi_l1,
            beat[PW_MID, 0] - beat[PW_S, 0],
            beat[RPEAK_IDX+QRS_S+10, 0],
            beat[RPEAK_IDX+QRS_S+5, 0] - beat[RPEAK_IDX+QRS_S+25, 0],
            beat[RPEAK_IDX+QRS_S, 0],
        ]
        feats.append(feat)
        labels.append(aami_map[sym])
    return np.array(feats), np.array(labels)


# ═══════════════════════════════════════════════════════════════════
# ANALOG SIMULATION — FULLY ANALOG CHAIN
# ═══════════════════════════════════════════════════════════════════

def load_all_beats(record_id):
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
    gain = 0.16 / max(all_peaks) if all_peaks else 0.05

    r_peak_scaled = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        r_val = sig[idx, 0]
        r_peak_scaled.append(VBASE + gain * r_val)
    max_r = max(r_peak_scaled) if r_peak_scaled else VBASE + 0.08
    rpeak_threshold = VBASE + 0.5 * (max_r - VBASE)

    beats = []
    ewma_rr = None
    for i in range(1, len(valid) - 1):
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
            'label': aami_map[sym],
            'pre_rr': pre_rr, 'post_rr': post_rr, 'ewma_rr': ewma_rr,
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


def calibrate_timer():
    """Run calibration to get delay timer SF curve."""
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

Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

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

    tb = os.path.join(FRONTEND_DIR, "_cal_e2e.spice")
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

    rpk_drop = 0.5  # ms after R when rpeak_pulse drops
    pipe_main = 3.5  # ms: comparator -> edge_buffer -> pulse_gen
    pipe_fast = 1.0  # ms: fast timer has large differential -> fast resolve

    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_fast))
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_main))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_main))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_main))
    vth_10  = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_fast))

    return {
        'vth_14': vth_14, 'vth_28': vth_28, 'vth_69': vth_69,
        'vth_300': vth_300, 'vth_10': vth_10,
        'dt_t': dt_t, 'dt_v': dt_v,
        'rpk_drop': rpk_drop, 'pipe_pw': 3.5,
    }


def generate_fully_analog_batch(beats_batch, gain, rpeak_threshold, batch_id,
                                 record_id, thresholds, ic_vals=None):
    """Generate SPICE testbench with ALL analog triggers — no ideal sources."""
    if ic_vals is None:
        ic_vals = {}
    n = len(beats_batch)
    dt = 1.0 / FS
    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    rpeak_times = [max(beats_batch[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats_batch[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    # ECG PWL sources
    ecg_l0_pts, ecg_l1_pts = [(0, VBASE)], [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt
        ecg0 = beats_batch[i]['ecg_l0']
        ecg1 = beats_batch[i]['ecg_l1']
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

    # Adaptive per-beat pwave thresholds using EWMA-predicted RR
    dt_t = thresholds['dt_t']
    dt_v = thresholds['dt_v']
    rpk_drop = thresholds['rpk_drop']
    pipe_pw = thresholds['pipe_pw']

    pwl_base_pts = [(0, 0.3)]
    pwl_peak_pts = [(0, 0.3)]
    pwl_rrcap_pts = [(0, 0.3)]
    for j in range(n):
        ewma_rr_ms = beats_batch[j]['ewma_rr'] * 1000
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

    sp = f""".title e2e_analog_{record_id}_b{batch_id}
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
* Adaptive pwave thresholds (from EWMA-predicted RR)
Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})
Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})
Vth_rrc vth_rr_cap  0 PWL({make_pwl(pwl_rrcap_pts)})

* === SUBCIRCUIT INCLUDES ===
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
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
* FULLY ANALOG FRONTEND — NO IDEAL TRIGGER SOURCES
* ============================================================

* 1. BIAS GENERATOR
Xbias vbias vref vdd vss bias_gen

* 2. R-PEAK DETECTOR (analog comparator)
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector

* 3. PULSE GENERATOR: rpeak level -> short pulse (~0.7ms)
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

* 4. DELAY TIMER (driven by rpeak_pulse)
Xdt rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

* 5. EDGE BUFFERS + PULSE GENERATORS for delay timer outputs
* Fast timer outputs: direct to pulse_gen (fast edges, no buffer needed)
Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
* rr_copy needs long_pulse_gen (2ms) for EWMA charge sharing in rr_timing
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias long_pulse_gen

* Main timer outputs: edge_buffer -> pulse_gen (slow edges need sharpening)
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen

Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen

* bi_copy: long_pulse_gen (2ms) for EWMA charge sharing in beat_instability
Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen

* 6. PWAVE PREDICTION COMPARATORS (on delay timer SF)
* pw_base comparator: fires at R + (ewma_rr - 201)ms
XMcpwb1 cnpwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 pw_base_lev dt_sf_out ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 pw_base_lev cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* pw_peak comparator: fires at R + (ewma_rr - 131)ms
XMcpwp1 cnpwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 pw_peak_lev dt_sf_out ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 pw_peak_lev cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* rr_cap comparator: fires at R + (ewma_rr - 50)ms for rr_timing capture
XMcrrc1 cnrrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 rr_cap_lev dt_sf_out ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 rr_cap_lev cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* 6b. EDGE BUFFERS + PULSE GENERATORS for pwave prediction
Xeb_pwb pw_base_lev pwb_sharp vdd vss edge_buffer
Xpg_pwb pwb_sharp   trig_pw_base vdd vss vbias pulse_gen

Xeb_pwp pw_peak_lev pwp_sharp vdd vss edge_buffer
Xpg_pwp pwp_sharp   trig_pw_peak vdd vss vbias pulse_gen

Xeb_rrc rr_cap_lev  rrc_sharp vdd vss edge_buffer
Xpg_rrc rrc_sharp   trig_rr_cap vdd vss vbias pulse_gen

* ============================================================
* 7. FEATURE CIRCUITS — ALL driven by ANALOG triggers
* ============================================================

* F1 & F2: RR Timing
*   capture: trig_rr_cap (from pwave rr_cap comparator)
*   rpeak:   rpeak_raw (HIGH during QRS ~9ms, sufficient for 320pF precharge)
*   copy:    trig_rr_copy (long_pulse_gen, 2ms for EWMA charge sharing)
Xrr ecg_l0 trig_rr_cap rpeak_raw trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing

* F3: Beat Instability L0 (rpeak_pulse for S&H, trig_bi_copy for EWMA)
Xbi0 ecg_l0 rpeak_pulse trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
* F4: Beat Instability L1
Xbi1 ecg_l1 rpeak_pulse trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability

* F5: P-wave Difference L0 (analog pwave triggers)
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff

* F6: QRS Width L0 (analog wid trigger)
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width

* F7: QRS Slope L0 (both s14 and s69 from analog delay timer)
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope

* F8: QRS Symmetry L0 (rpeak_pulse for S&H capture)
Xsym ecg_l0 rpeak_pulse f8_symmetry vdd vss qrs_symmetry

* === INITIAL CONDITIONS ===
.ic V(xbias.iref)=0.3
.ic V(xrr.ewma_node)={ic_vals.get('rr_ewma', 0.40):.6f}
.ic V(xrr.esamp_node)={ic_vals.get('rr_esamp', 0.40):.6f}
.ic V(xrr.prev_node)={ic_vals.get('rr_prev', 0.40):.6f}
.ic V(xbi0.prev_node)={ic_vals.get('bi0_prev', 0.30):.6f}
.ic V(xbi0.esamp_node)={ic_vals.get('bi0_esamp', 0.30):.6f}
.ic V(xbi1.prev_node)={ic_vals.get('bi1_prev', 0.30):.6f}
.ic V(xbi1.esamp_node)={ic_vals.get('bi1_esamp', 0.30):.6f}

.tran 0.5m {total_time:.4f}

"""
    # Measurement commands
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

    # IC propagation
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

    tb = os.path.join(FRONTEND_DIR, f"_e2e_{record_id}_b{batch_id}.spice")
    with open(tb, 'w') as f:
        f.write(sp)
    return tb, skip, rpeak_times, total_time


def run_ngspice(tb):
    log = tb.replace('.spice', '.log')
    try:
        subprocess.run([NGSPICE, '-b', '-o', log, tb],
                       capture_output=True, text=True, timeout=1800, cwd=NGSPICE_DIR)
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


def simulate_record(record_id, thresholds):
    """Run fully analog simulation on one record, return per-beat circuit features."""
    beats, gain, rpeak_threshold = load_all_beats(record_id)
    if not beats:
        return [], [], []

    all_circuit, all_py, all_labels = [], [], []
    n_batches = (len(beats) + BATCH_SIZE - 1) // BATCH_SIZE

    ic_vals = None
    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end = min((b_idx + 1) * BATCH_SIZE, len(beats))
        batch = beats[start:end]
        if len(batch) < 6:
            continue

        tb, skip, rpeak_times, total_time = generate_fully_analog_batch(
            batch, gain, rpeak_threshold, b_idx, record_id, thresholds, ic_vals)
        results = run_ngspice(tb)

        if results is not None:
            next_ic = {}
            for key, meas_name in [('rr_ewma', 'ic_rr_ewma'), ('rr_esamp', 'ic_rr_esamp'),
                                   ('rr_prev', 'ic_rr_prev'), ('bi0_prev', 'ic_bi0_prev'),
                                   ('bi0_esamp', 'ic_bi0_esamp'), ('bi1_prev', 'ic_bi1_prev'),
                                   ('bi1_esamp', 'ic_bi1_esamp')]:
                v = results.get(meas_name)
                if v is not None:
                    next_ic[key] = v
            if next_ic:
                ic_vals = next_ic

        if results is None:
            print(f"    WARNING: batch {b_idx} failed for record {record_id}")
            continue

        for j in range(skip, len(batch)):
            orig_idx = start + j
            if orig_idx >= len(beats): break
            bt = beats[orig_idx]

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
            all_circuit.append(vals)
            all_py.append(py_vals)
            all_labels.append(bt['label'])

        try:
            os.remove(tb)
            os.remove(tb.replace('.spice', '.log'))
        except: pass

    return all_circuit, all_py, all_labels


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("FINAL END-TO-END TEST: Fully Analog AFE -> RF Classifier")
    print("=" * 70)
    print("  Signal chain: MIT-BIH ECG (PWL) -> Analog AFE (SKY130) -> RF Classifier")
    print("  ALL triggers are analog. No ideal sources.")
    print(f"  Test records: {', '.join(DS2_TEST)}")
    print()

    # ── Step 1: Train RF classifier on Python DS1 features ──
    print("[1/4] Extracting Python features for DS1 (training)...")
    t0 = time.time()
    X_train, y_train = extract_python_features(DS1)
    print(f"  DS1: {len(X_train)} beats, {time.time()-t0:.1f}s")
    print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42,
                                n_jobs=-1, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    print(f"  RF trained on {len(X_train)} DS1 beats (8 features)")

    # ── Step 2: Python baseline predictions on test records ──
    print(f"\n[2/4] Python baseline predictions on {len(DS2_TEST)} test records...")
    py_test_feats = {}
    py_test_labels = {}
    for rec in DS2_TEST:
        feats, labels = extract_python_features_per_record(rec)
        py_test_feats[rec] = feats
        py_test_labels[rec] = labels
        print(f"  Record {rec}: {len(feats)} beats")

    # ── Step 3: Run analog simulation ──
    print(f"\n[3/4] Running fully analog simulations...")
    print("  Calibrating timer...")
    cal = calibrate_timer()
    thresholds = compute_thresholds(cal)
    print(f"  Thresholds: vth_14={thresholds['vth_14']:.5f} vth_28={thresholds['vth_28']:.5f} "
          f"vth_69={thresholds['vth_69']:.5f} vth_300={thresholds['vth_300']:.5f}")

    analog_feats = {}
    analog_py_feats = {}
    analog_labels = {}
    for rec in DS2_TEST:
        t0 = time.time()
        print(f"  Simulating record {rec}...", end='', flush=True)
        circuit, py, labels = simulate_record(rec, thresholds)
        elapsed = time.time() - t0
        print(f" {len(circuit)} beats, {elapsed:.1f}s")
        analog_feats[rec] = np.array(circuit) if circuit else np.empty((0, 8))
        analog_py_feats[rec] = np.array(py) if py else np.empty((0, 8))
        analog_labels[rec] = np.array(labels) if labels else np.array([])

    # ── Step 4: Compare predictions ──
    print(f"\n[4/4] Comparing RF predictions: Python vs Analog")
    print("=" * 70)

    # First, show per-feature Spearman rho (analog vs Python)
    print("\n  Per-record Spearman |rho| (analog circuit vs Python):")
    print(f"  {'Rec':>5s} {'Beats':>6s}  {'F1':>6s} {'F2':>6s} {'F3':>6s} {'F4':>6s} {'F5':>6s} {'F6':>6s} {'F7':>6s} {'F8':>6s}")
    for rec in DS2_TEST:
        if len(analog_feats[rec]) < 10:
            continue
        rhos = []
        for i in range(8):
            rho, _ = spearmanr(analog_feats[rec][:, i], analog_py_feats[rec][:, i])
            rhos.append(abs(rho))
        rho_str = " ".join(f"{r:6.3f}" for r in rhos)
        print(f"  {rec:>5s} {len(analog_feats[rec]):>6d}  {rho_str}")

    # Pool all analog data for calibration
    all_analog = np.concatenate([analog_feats[r] for r in DS2_TEST if len(analog_feats[r]) > 0])
    all_py_paired = np.concatenate([analog_py_feats[r] for r in DS2_TEST if len(analog_py_feats[r]) > 0])
    all_analog_labels = np.concatenate([analog_labels[r] for r in DS2_TEST if len(analog_labels[r]) > 0])

    if len(all_analog) < 50:
        print("\nERROR: Not enough analog data for comparison")
        sys.exit(1)

    # Calibrate analog -> Python scale using linear regression per feature
    from sklearn.linear_model import LinearRegression
    calibrated_analog = np.zeros_like(all_analog)
    print(f"\n  Per-feature linear calibration (analog voltage -> Python scale):")
    print(f"  {'Feature':>20s} {'slope':>10s} {'intercept':>10s} {'R²':>8s}")
    for i in range(8):
        lr = LinearRegression()
        lr.fit(all_analog[:, i:i+1], all_py_paired[:, i])
        calibrated_analog[:, i] = lr.predict(all_analog[:, i:i+1])
        r2 = lr.score(all_analog[:, i:i+1], all_py_paired[:, i])
        print(f"  {FEATURE_NAMES[i]:>20s} {lr.coef_[0]:10.4f} {lr.intercept_:10.4f} {r2:8.4f}")

    # Python predictions on test beats
    py_feats_test = scaler.transform(all_py_paired)
    py_preds = rf.predict(py_feats_test)

    # Analog predictions (after calibration)
    analog_feats_test = scaler.transform(calibrated_analog)
    analog_preds = rf.predict(analog_feats_test)

    # Compare
    agreement = np.mean(py_preds == analog_preds) * 100
    py_accuracy = accuracy_score(all_analog_labels, py_preds) * 100
    analog_accuracy = accuracy_score(all_analog_labels, analog_preds) * 100

    print(f"\n{'=' * 70}")
    print(f"RESULTS ({len(all_analog)} beats across {len(DS2_TEST)} records)")
    print(f"{'=' * 70}")
    print(f"  Python RF accuracy:     {py_accuracy:.2f}%")
    print(f"  Analog RF accuracy:     {analog_accuracy:.2f}%")
    print(f"  Prediction agreement:   {agreement:.2f}% (same class for same beat)")
    print(f"  Accuracy difference:    {abs(py_accuracy - analog_accuracy):.2f}%")

    print(f"\n  Python confusion matrix:")
    classes = ['N', 'S', 'V', 'F', 'Q']
    cm_py = confusion_matrix(all_analog_labels, py_preds, labels=classes)
    print(f"  {'':>8s} " + " ".join(f"{c:>6s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:>8s} " + " ".join(f"{cm_py[i,j]:6d}" for j in range(len(classes))))

    print(f"\n  Analog confusion matrix:")
    cm_an = confusion_matrix(all_analog_labels, analog_preds, labels=classes)
    print(f"  {'':>8s} " + " ".join(f"{c:>6s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:>8s} " + " ".join(f"{cm_an[i,j]:6d}" for j in range(len(classes))))

    # Per-class comparison
    print(f"\n  Per-class prediction agreement:")
    for c in classes:
        mask = all_analog_labels == c
        if np.sum(mask) > 0:
            agree = np.mean(py_preds[mask] == analog_preds[mask]) * 100
            print(f"    {c}: {agree:.1f}% ({np.sum(mask)} beats)")

    print(f"\n  Disagreements by class pair:")
    disagree_mask = py_preds != analog_preds
    if np.sum(disagree_mask) > 0:
        from collections import Counter
        pairs = Counter(zip(py_preds[disagree_mask], analog_preds[disagree_mask]))
        for (py_c, an_c), count in pairs.most_common(10):
            print(f"    Python={py_c} -> Analog={an_c}: {count} beats")

    print(f"\n{'=' * 70}")
    if agreement > 95:
        print("PASS — Analog AFE produces predictions matching Python simulation")
    elif agreement > 90:
        print("GOOD — Minor disagreements, analog AFE functionally equivalent")
    elif agreement > 80:
        print("ACCEPTABLE — Some prediction drift, investigate per-feature calibration")
    else:
        print("NEEDS WORK — Significant prediction disagreement")
    print(f"{'=' * 70}")
