"""
Full Analog Chain Integration Test
===================================
Tests the COMPLETE analog frontend with NO PWL trigger sources.
All timing comes from analog supplementary circuits:
  ECG (PWL) → bias_gen → rpeak_detector → pulse_gen → delay_timer
            → pulse_gen(×9) → feature circuits (8) → outputs

Signal chain:
  1. bias_gen generates vbias (~0.30V)
  2. rpeak_detector detects R-peaks (threshold comparator)
  3. pulse_gen converts rpeak LEVEL to short PULSE (~0.5ms)
  4. delay_timer + comparators generate post-R timing LEVELS
  5. pulse_gen instances convert each level to a ~0.5ms pulse
  6. pwave_predictor watches rr_timing SF timer for pre-R timing
  7. pulse_gen instances convert pwave levels to pulses
  8. Feature circuits extract 8 features using pulsed triggers

Phase 1: Validate on record 200 (reference record, 10 beats)
Phase 2: Extend to 40 beats
Phase 3: Full 44-record DS1/DS2 accuracy test
"""

import subprocess, os, re, sys, time, json
import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\Leaky-Integrate-Fire---SNN-model-on-FPGA"
sys.path.insert(0, PROJECT_ROOT)
import wfdb

NGSPICE = r"C:\Spice64\bin\ngspice_con.exe"
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "complete_frontend")
NGSPICE_DIR = os.path.join(PROJECT_ROOT, "experiments", "frontend_optimization", "ngspice_frontend")
MITDB_DIR = os.path.join(PROJECT_ROOT, "mitdb_data")

VDD = 0.9
FS = 360
WIN_L, WIN_R = 90, 108
QRS_S, QRS_E = 70, 130
RPEAK_IDX = 20  # within beat window: index WIN_L+RPEAK_IDX = actual R-peak sample
PW_S = 16; PW_MID = 43
EWMA_ALPHA = 0.10  # matches v5 rr_timing charge-share: 5pF/(5pF+45pF)
VBASE = 0.30

# Timer parameters (from circuit comments, validated against ngspice)
# delay_timer: 100pF cap (W=224 L=224), L=8 W=0.84 discharge NFET
DT_CAP = 100e-12   # 100pF
DT_I_DISCHARGE = 63e-12  # ~63pA (measured from comment)
DT_SF_SHIFT = 0.40  # body-effect-adjusted Vgs
DT_SF0 = VDD - DT_SF_SHIFT  # ~0.50V initial SF
DT_RATE = DT_I_DISCHARGE / DT_CAP  # V/s = 0.63mV/ms

# Fast timer: 2pF cap (W=32 L=32), same discharge NFET
FT_CAP = 2.048e-12
FT_RATE = DT_I_DISCHARGE / FT_CAP  # ~30.8mV/ms

# rr_timing timer: 320pF cap (W=400 L=400), L=8 W=0.42 discharge (~44pA per comment)
RR_CAP = 320e-12
RR_I_DISCHARGE = 44e-12  # ~44pA (from rr_timing comment)
RR_SF_SHIFT = 0.60  # v5 SF uses W=0.42 L=8 (larger Vgs than W=20)
RR_SF0 = VDD - RR_SF_SHIFT  # ~0.30V
RR_RATE = RR_I_DISCHARGE / RR_CAP  # ~0.1375mV/ms

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


def calibrate_timer(verbose=True):
    """Run a short calibration simulation to measure actual timer discharge rates."""
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
.include "{fp}/rr_timing.spice"

* Bias generator
Xbias vbias vref vdd vss bias_gen

* Single rpeak pulse at t=0.1s (20ms width, matching actual QRS duration)
* The 320pF rr_timing cap needs ~5ms to fully charge via the W=2 PMOS
Vrpk rpeak_pulse 0 PWL(0 0 0.098 0 0.1 {VDD} 0.12 {VDD} 0.121 0 1.0 0)

* Dummy thresholds for delay_timer (we just want the SF curve)
Vt1 vth_14_fast 0 0.1
Vt2 vth_69 0 0.1
Vt3 vth_28 0 0.1
Vt4 vth_300 0 0.1
Vt5 vth_10 0 0.1

Xdelay rpeak_pulse dt1 dt2 dt3 dt4 dt5 dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

* RR timing (for rr timer calibration)
Vecg ecg_in 0 0.3
Vrrc cap_trig 0 0
Vcpy copy_trig 0 0
Xrr ecg_in cap_trig rpeak_pulse copy_trig f1 f2 rr_timer rr_ewma vdd vss vbias rr_timing

.ic V(xbias.iref)=0.3 V(xrr.ewma_node)=0.20 V(xrr.esamp_node)=0.20 V(xrr.prev_node)=0.20

.tran 0.1m 1.1

* Measure vbias
.meas tran cal_vbias FIND V(vbias) AT=0.09

* Measure delay timer SF at multiple times after rpeak drops
* rpeak drops at ~0.121s, so timer starts discharging at 0.121s
.meas tran dt_sf_001 FIND V(xdelay.sf_timer) AT=0.122
.meas tran dt_sf_002 FIND V(xdelay.sf_timer) AT=0.123
.meas tran dt_sf_003 FIND V(xdelay.sf_timer) AT=0.124
.meas tran dt_sf_004 FIND V(xdelay.sf_timer) AT=0.125
.meas tran dt_sf_005 FIND V(xdelay.sf_timer) AT=0.126
.meas tran dt_sf_010 FIND V(xdelay.sf_timer) AT=0.131
.meas tran dt_sf_015 FIND V(xdelay.sf_timer) AT=0.136
.meas tran dt_sf_020 FIND V(xdelay.sf_timer) AT=0.141
.meas tran dt_sf_050 FIND V(xdelay.sf_timer) AT=0.171
.meas tran dt_sf_100 FIND V(xdelay.sf_timer) AT=0.221
.meas tran dt_sf_200 FIND V(xdelay.sf_timer) AT=0.321
.meas tran dt_sf_300 FIND V(xdelay.sf_timer) AT=0.421
.meas tran dt_sf_400 FIND V(xdelay.sf_timer) AT=0.521
.meas tran dt_sf_500 FIND V(xdelay.sf_timer) AT=0.621
.meas tran dt_sf_600 FIND V(xdelay.sf_timer) AT=0.721
.meas tran dt_sf_700 FIND V(xdelay.sf_timer) AT=0.821
.meas tran dt_sf_800 FIND V(xdelay.sf_timer) AT=0.921

* Measure fast timer SF at multiple times
.meas tran ft_sf_001 FIND V(xdelay.sf_timer2) AT=0.122
.meas tran ft_sf_002 FIND V(xdelay.sf_timer2) AT=0.123
.meas tran ft_sf_003 FIND V(xdelay.sf_timer2) AT=0.124
.meas tran ft_sf_005 FIND V(xdelay.sf_timer2) AT=0.126
.meas tran ft_sf_010 FIND V(xdelay.sf_timer2) AT=0.131
.meas tran ft_sf_020 FIND V(xdelay.sf_timer2) AT=0.141

* Measure RR timer at multiple times after rpeak drops (0.121s)
.meas tran rr_net_precharge FIND V(xrr.net_timer) AT=0.119
.meas tran rr_net_001 FIND V(xrr.net_timer) AT=0.122
.meas tran rr_net_005 FIND V(xrr.net_timer) AT=0.126
.meas tran rr_net_010 FIND V(xrr.net_timer) AT=0.131
.meas tran rr_net_050 FIND V(xrr.net_timer) AT=0.171
.meas tran rr_net_200 FIND V(xrr.net_timer) AT=0.321
.meas tran rr_net_400 FIND V(xrr.net_timer) AT=0.521
.meas tran rr_net_600 FIND V(xrr.net_timer) AT=0.721
.meas tran rr_sf_precharge FIND V(xrr.sf_out) AT=0.119
.meas tran rr_sf_001 FIND V(xrr.sf_out) AT=0.122
.meas tran rr_sf_005 FIND V(xrr.sf_out) AT=0.126
.meas tran rr_sf_010 FIND V(xrr.sf_out) AT=0.131
.meas tran rr_sf_050 FIND V(xrr.sf_out) AT=0.171
.meas tran rr_sf_100 FIND V(xrr.sf_out) AT=0.221
.meas tran rr_sf_200 FIND V(xrr.sf_out) AT=0.321
.meas tran rr_sf_400 FIND V(xrr.sf_out) AT=0.521
.meas tran rr_sf_600 FIND V(xrr.sf_out) AT=0.721

.end
"""
    tb = os.path.join(FRONTEND_DIR, "_calibration.spice")
    log = tb.replace('.spice', '.log')
    with open(tb, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)

    if verbose:
        print("Running timer calibration simulation...")
    t0 = time.time()
    subprocess.run([NGSPICE, '-b', '-o', log, tb],
                   capture_output=True, text=True, timeout=120, cwd=NGSPICE_DIR)
    elapsed = time.time() - t0
    if verbose:
        print(f"Calibration completed in {elapsed:.1f}s")

    with open(log) as f:
        text = f.read()

    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try: results[m.group(1).lower()] = float(m.group(2))
            except: pass

    # Extract delay timer SF curve (extended to 800ms for pwave prediction)
    dt_times = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    dt_sf = [results.get(f'dt_sf_{t:03d}') for t in dt_times]

    # Extract fast timer SF curve
    ft_times = [1, 2, 3, 5, 10, 20]
    ft_sf = [results.get(f'ft_sf_{t:03d}') for t in ft_times]

    # Extract RR timer SF curve (using internal sf_out node)
    rr_times = [1, 5, 10, 50, 100, 200, 400, 600]
    rr_sf = [results.get(f'rr_sf_{t:03d}') for t in rr_times]

    # Debug: RR timer precharge and discharge
    if verbose:
        print(f"  RR timer precharge & discharge:")
        for label in ['precharge', '001', '005', '010', '050', '200', '400', '600']:
            net_v = results.get(f'rr_net_{label}')
            sf_v = results.get(f'rr_sf_{label}')
            if net_v is not None and sf_v is not None:
                print(f"    {label:>10s}: net={net_v:.5f}V  sf={sf_v:.5f}V  Vgs_sf={net_v-sf_v:.4f}V")

    vbias = results.get('cal_vbias', 0.30)

    if verbose:
        print(f"  Vbias: {vbias:.4f}V")
        print(f"  Delay timer SF curve (ms from rpeak drop):")
        for t, v in zip(dt_times, dt_sf):
            if v is not None:
                print(f"    {t:4d}ms: {v:.5f}V")
        print(f"  Fast timer SF curve:")
        for t, v in zip(ft_times, ft_sf):
            if v is not None:
                print(f"    {t:4d}ms: {v:.5f}V")
        print(f"  RR timer SF curve:")
        for t, v in zip(rr_times, rr_sf):
            if v is not None:
                print(f"    {t:4d}ms: {v:.5f}V")

    try:
        os.remove(tb)
        os.remove(log)
    except: pass

    return {
        'vbias': vbias,
        'dt_times_ms': dt_times, 'dt_sf': dt_sf,
        'ft_times_ms': ft_times, 'ft_sf': ft_sf,
        'rr_times_ms': rr_times, 'rr_sf': rr_sf,
    }


def interp_sf(times_ms, voltages, target_ms):
    for i in range(len(times_ms) - 1):
        if times_ms[i] <= target_ms <= times_ms[i+1]:
            frac = (target_ms - times_ms[i]) / (times_ms[i+1] - times_ms[i])
            return voltages[i] + frac * (voltages[i+1] - voltages[i])
    if target_ms < times_ms[0]:
        rate = (voltages[0] - voltages[1]) / (times_ms[1] - times_ms[0])
        return voltages[0] + rate * (times_ms[0] - target_ms)
    rate = (voltages[-2] - voltages[-1]) / (times_ms[-1] - times_ms[-2])
    return voltages[-1] - rate * (target_ms - times_ms[-1])


def compute_thresholds_calibrated(cal, avg_rr_s):
    """Compute thresholds from calibrated timer curves."""
    # Interpolate delay timer SF to get voltage at each target delay
    dt_t = [t for t, v in zip(cal['dt_times_ms'], cal['dt_sf']) if v is not None]
    dt_v = [v for v in cal['dt_sf'] if v is not None]

    pass  # interp_sf is now module-level

    # Pipeline delay: validated by isolation test (test_delay_chain.py)
    # Comparator + inverter buffer + pulse_gen = 2.5ms for main timer, 1.0ms for fast
    # rpeak_raw drops at ~R+9ms (measured by test_rpeak_chain.py)
    rpk_drop = 0.5  # ms after R-peak when rpeak_pulse drops (pulse_gen width)
    pipe_main = 3.5 # ms pipeline: comp → inv_buf → pulse_gen (from full chain: wid 2ms late → 3.5ms)
    pipe_fast = 1.0 # ms pipeline for fast timer (large differential → fast resolve)

    # Delay timer targets (relative to R-peak):
    #   s14:  R+14ms (5 samples at 360Hz) — uses FAST timer for large differential
    #   wid:  R+28ms (10 samples) — main timer
    #   s69:  R+69ms (25 samples) — main timer
    #   bi:   R+300ms — main timer
    # Timer starts at rpk_drop. Trigger fires at: rpk_drop + discharge_time + pipeline
    # So: discharge_time = target - rpk_drop - pipeline

    # Fast timer for s14 (2pF, ~21mV/ms): large differential → fast comparator → pipe_fast
    ft_t = [t for t, v in zip(cal['ft_times_ms'], cal['ft_sf']) if v is not None]
    ft_v = [v for v in cal['ft_sf'] if v is not None]
    vth_14  = interp_sf(ft_t, ft_v, max(0.3, 14 - rpk_drop - pipe_fast))
    # Main timer for wid/s69/bi
    vth_28  = interp_sf(dt_t, dt_v, max(0.5, 28 - rpk_drop - pipe_main))
    vth_69  = interp_sf(dt_t, dt_v, max(0.5, 69 - rpk_drop - pipe_main))
    vth_300 = interp_sf(dt_t, dt_v, max(0.5, 300 - rpk_drop - pipe_main))

    # Fast timer rr_copy at R+10ms (reuses ft_t/ft_v from s14 above)
    vth_10 = interp_sf(ft_t, ft_v, max(0.3, 10 - rpk_drop - pipe_fast))

    avg_rr_ms = avg_rr_s * 1000

    # Pwave thresholds are now computed PER-BEAT using EWMA-predicted RR.
    # Store the delay timer SF curve for per-beat threshold computation.
    pipe_pw = 3.5  # ms (delay timer comparators pipeline)

    # RR timer EWMA initialization: use RR timer SF at capture time
    # Capture happens when trig_rr_cap fires, at R+(ewma_rr-50)ms
    rr_t = [t for t, v in zip(cal['rr_times_ms'], cal['rr_sf']) if v is not None]
    rr_v = [v for v in cal['rr_sf'] if v is not None]
    capture_discharge_ms = avg_rr_ms - 50 - rpk_drop
    ewma_level = interp_sf(rr_t, rr_v, max(1, capture_discharge_ms))

    return {
        'vth_14': vth_14, 'vth_28': vth_28, 'vth_69': vth_69,
        'vth_300': vth_300, 'vth_10': vth_10,
        'ewma_level': ewma_level,
        'dt_t': dt_t, 'dt_v': dt_v,
        'rr_t': rr_t, 'rr_v': rr_v,
        'rpk_drop': rpk_drop, 'pipe_pw': pipe_pw,
    }


def make_pwl(pts):
    pts.sort(key=lambda x: x[0])
    c = [pts[0]]
    for p in pts[1:]:
        if p[0] > c[-1][0] + 1e-7:
            c.append(p)
    return " ".join(f"{t:.7f} {v:.6f}" for t, v in c)


def run_test(record_id, n_beats=10, verbose=True):
    """Run full analog chain test on a single record."""
    record = wfdb.rdrecord(os.path.join(MITDB_DIR, record_id))
    ann = wfdb.rdann(os.path.join(MITDB_DIR, record_id), 'atr')
    sig = record.p_signal
    n_leads = sig.shape[1]

    valid = [(idx, sym) for idx, sym in zip(ann.sample, ann.symbol)
             if sym in aami_map]

    # Compute gain to normalize R-peaks to 0.16V above VBASE
    all_peaks = []
    for idx, sym in valid:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        all_peaks.append(np.max(np.abs(sig[bs+QRS_S:bs+QRS_E, 0])))
    gain = 0.16 / max(all_peaks) if all_peaks else 0.05

    # Compute rpeak threshold from ACTUAL R-peak amplitudes (not S-wave)
    # The R-peak is at index WIN_L within each beat window
    r_peak_scaled = []
    for idx, sym in valid[:n_beats+2]:
        bs = idx - WIN_L
        if bs < 0 or idx + WIN_R >= len(sig): continue
        r_val = sig[idx, 0]  # raw R-peak value
        v_scaled = VBASE + gain * r_val
        r_peak_scaled.append(v_scaled)
    max_r_voltage = max(r_peak_scaled) if r_peak_scaled else VBASE + 0.08
    min_r_voltage = min(r_peak_scaled) if r_peak_scaled else VBASE
    # Set threshold at 50% of max R-peak above baseline (accounts for S-wave normalization)
    rpeak_threshold = VBASE + 0.5 * (max_r_voltage - VBASE)
    if verbose:
        print(f"R-peak voltages: max={max_r_voltage:.4f}V min={min_r_voltage:.4f}V threshold={rpeak_threshold:.4f}V")

    # Load beats
    beats = []
    ewma_rr = None
    for i in range(1, n_beats + 2):
        if i >= len(valid) - 1: break
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
            prev_beat = sig[prev_bs:prev_be]
            py_bi_l0 = beat[RPEAK_IDX+QRS_S, 0] - prev_beat[RPEAK_IDX+QRS_S, 0]
            py_bi_l1 = beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)] - prev_beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)]
        else:
            py_bi_l0 = py_bi_l1 = 0.0

        # Find per-beat rpk_drop: when ECG falls below threshold after R-peak
        rpk_drop_s = 4  # default: 4 samples = 11ms
        rpk = RPEAK_IDX + QRS_S  # R-peak index within beat window (=90)
        for s in range(rpk + 1, min(rpk + 15, len(beat))):
            v_scaled = VBASE + gain * beat[s, 0]
            if v_scaled < rpeak_threshold:
                rpk_drop_s = s - rpk  # samples after R-peak
                break

        # Also find QRS onset: when ECG first rises above threshold before R-peak
        rpk_onset_s = rpk  # default: R-peak itself
        for s in range(rpk - 1, max(rpk - 15, 0), -1):
            v_scaled = VBASE + gain * beat[s, 0]
            if v_scaled < rpeak_threshold:
                rpk_onset_s = s + 1 - rpk  # samples before R-peak (negative)
                break
        else:
            rpk_onset_s = -3  # default ~8ms before R-peak

        # Adjusted trigger samples: analog fires at target_ms from R-peak
        # With validated pipeline (2.5ms), triggers hit targets well
        # s14→R+14ms, s69→R+69ms, wid→R+28ms (in samples from rpk)
        s14_adj = round(14 * FS / 1000)   # 5 samples
        s69_adj = round(69 * FS / 1000)   # 25 samples
        wid_adj = round(28 * FS / 1000)   # 10 samples

        beats.append({
            'label': aami_map[sym], 'pre_rr': pre_rr, 'post_rr': post_rr, 'ewma_rr': ewma_rr,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'rpk_drop_s': rpk_drop_s, 'rpk_onset_s': rpk_onset_s,
            'py_f1': pre_rr / ewma_rr_before if ewma_rr_before > 0 else 1.0,
            'py_f2': pre_rr / post_rr if post_rr > 0 else 1.0,
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
            'py_pw': beat[PW_MID, 0] - beat[PW_S, 0],
            'py_wid': beat[RPEAK_IDX+QRS_S+10, 0],
            'py_slp': beat[RPEAK_IDX+QRS_S+5, 0] - beat[RPEAK_IDX+QRS_S+25, 0],
            'py_sym': beat[RPEAK_IDX+QRS_S, 0],
            # Adjusted reference at actual analog trigger times
            'py_wid_adj': beat[rpk + wid_adj, 0] if rpk + wid_adj < len(beat) else beat[rpk + 10, 0],
            'py_slp_adj': (beat[rpk + s14_adj, 0] - beat[rpk + s69_adj, 0]
                           if rpk + s69_adj < len(beat) else beat[rpk + 5, 0] - beat[rpk + 25, 0]),
            'py_sym_adj': beat[rpk + rpk_onset_s, 0],
            'py_bi_l0_adj': beat[rpk + rpk_onset_s, 0] - (prev_beat[rpk + rpk_onset_s, 0] if prev_bs >= 0 and prev_be < len(sig) else beat[rpk, 0]),
        })
    beats = beats[:n_beats]
    n = len(beats)
    if verbose:
        print(f"Record {record_id}: {n} beats, gain={gain:.4f}")

    # Compute R-peak times in simulation time (start at 2.0s for warmup)
    dt = 1.0 / FS
    rpeak_times = [2.0]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

    # Average RR for threshold computation
    rr_intervals = [beats[i]['pre_rr'] for i in range(1, n)]
    avg_rr = np.mean(rr_intervals) if rr_intervals else 0.7

    # Run calibration simulation to get actual timer discharge curves
    cal = calibrate_timer(verbose=verbose)
    thresholds = compute_thresholds_calibrated(cal, avg_rr)

    if verbose:
        print(f"Avg RR: {avg_rr:.3f}s, Sim time: {total_time:.1f}s")
        print(f"Calibrated thresholds:")
        print(f"  vth_14={thresholds['vth_14']:.5f} (FAST timer) vth_28={thresholds['vth_28']:.5f} "
              f"vth_69={thresholds['vth_69']:.5f} vth_300={thresholds['vth_300']:.5f}")
        print(f"  vth_10={thresholds['vth_10']:.5f} (fast timer)")
        print(f"  ewma_init={thresholds['ewma_level']:.5f} (RR timer SF at capture time)")
        print(f"  Pwave thresholds: adaptive per-beat PWL (from EWMA-predicted RR)")

    # Generate ECG PWL sources
    ecg_l0_pts, ecg_l1_pts = [(0, VBASE)], [(0, VBASE)]
    for i, rpt in enumerate(rpeak_times):
        beat_start = rpt - WIN_L * dt
        ecg0 = beats[i]['ecg_l0']
        ecg1 = beats[i]['ecg_l1']
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

    # Compute per-beat adaptive pwave thresholds using EWMA-predicted RR
    dt_t = thresholds['dt_t']
    dt_v = thresholds['dt_v']
    rpk_drop = thresholds['rpk_drop']
    pipe_pw = thresholds['pipe_pw']

    pwl_base_pts = [(0, 0.3)]  # start with safe initial value
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
        # Step threshold just after R-peak precharge resets comparators
        t_step = t_rpk + 0.001  # 1ms after R-peak
        if t_step > pwl_base_pts[-1][0]:
            pwl_base_pts.append((t_step, v_base))
            pwl_peak_pts.append((t_step, v_peak))
            pwl_rrcap_pts.append((t_step, v_rrcap))
    # Hold final value to end
    pwl_base_pts.append((total_time, pwl_base_pts[-1][1]))
    pwl_peak_pts.append((total_time, pwl_peak_pts[-1][1]))
    pwl_rrcap_pts.append((total_time, pwl_rrcap_pts[-1][1]))

    pw_pwl_str = (
        f"Vth_pwb vth_pw_base 0 PWL({make_pwl(pwl_base_pts)})\n"
        f"Vth_pwp vth_pw_peak 0 PWL({make_pwl(pwl_peak_pts)})\n"
        f"Vth_rrc vth_rr_cap 0 PWL({make_pwl(pwl_rrcap_pts)})"
    )

    # Ideal pwave triggers: short pulses at exact Python sample positions
    # Base: R[j] - 74 samples = R[j] - 206ms, Peak: R[j] - 47 samples = R[j] - 131ms
    pw_base_ideal_pts = [(0, 0)]
    pw_peak_ideal_pts = [(0, 0)]
    pulse_w = 0.0005  # 0.5ms pulse width
    for j in range(2, n):
        t_base = rpeak_times[j] - 74.0 / FS
        t_peak = rpeak_times[j] - 47.0 / FS
        if t_base > 0.1:
            pw_base_ideal_pts.extend([(t_base - 0.0001, 0), (t_base, VDD), (t_base + pulse_w, VDD), (t_base + pulse_w + 0.0001, 0)])
        if t_peak > 0.1:
            pw_peak_ideal_pts.extend([(t_peak - 0.0001, 0), (t_peak, VDD), (t_peak + pulse_w, VDD), (t_peak + pulse_w + 0.0001, 0)])
    pw_base_ideal_pts.append((total_time, 0))
    pw_peak_ideal_pts.append((total_time, 0))

    # Ideal slope triggers: short pulses at exact Python sample positions
    # s14: R[j] + 5 samples = R[j] + 13.9ms, s69: R[j] + 25 samples = R[j] + 69.4ms
    s14_ideal_pts = [(0, 0)]
    s69_ideal_pts = [(0, 0)]
    for j in range(2, n):
        t_s14 = rpeak_times[j] + 5.0 / FS
        t_s69 = rpeak_times[j] + 25.0 / FS
        if t_s14 > 0.1:
            s14_ideal_pts.extend([(t_s14 - 0.0001, 0), (t_s14, VDD), (t_s14 + pulse_w, VDD), (t_s14 + pulse_w + 0.0001, 0)])
        if t_s69 > 0.1:
            s69_ideal_pts.extend([(t_s69 - 0.0001, 0), (t_s69, VDD), (t_s69 + pulse_w, VDD), (t_s69 + pulse_w + 0.0001, 0)])
    s14_ideal_pts.append((total_time, 0))
    s69_ideal_pts.append((total_time, 0))

    # Ideal rpeak triggers: pulse at exact R-peak time (index 90 = RPEAK_IDX+QRS_S)
    # For F3/F4 capture and F8 symmetry
    rpk_ideal_pts = [(0, 0)]
    for j in range(1, n):
        t_rpk = rpeak_times[j]
        if t_rpk > 0.1:
            rpk_ideal_pts.extend([(t_rpk - 0.0001, 0), (t_rpk, VDD), (t_rpk + pulse_w, VDD), (t_rpk + pulse_w + 0.0001, 0)])
    rpk_ideal_pts.append((total_time, 0))

    # Ideal bi_copy trigger: pulse at R+300ms for F3/F4 copy
    bi_copy_ideal_pts = [(0, 0)]
    for j in range(1, n):
        t_copy = rpeak_times[j] + 0.300
        if t_copy > 0.1 and t_copy < total_time - 0.01:
            bi_copy_ideal_pts.extend([(t_copy - 0.0001, 0), (t_copy, VDD), (t_copy + pulse_w, VDD), (t_copy + pulse_w + 0.0001, 0)])
    bi_copy_ideal_pts.append((total_time, 0))

    # Ideal rr_timing triggers: capture at R-5ms, copy at R+10ms
    # v5 rr_timing expects: capture_trig (2ms pulse 5ms before R),
    # rpeak_trigger (pulse at R), copy_trig (2ms pulse 10ms after R)
    rr_cap_ideal_pts = [(0, 0)]
    rr_copy_ideal_pts = [(0, 0)]
    rr_pulse_w = 0.002  # 2ms pulse width (as in v5 spec)
    for j in range(2, n):
        t_cap = rpeak_times[j] - 0.005  # 5ms before R-peak
        t_copy = rpeak_times[j] + 0.010  # 10ms after R-peak
        if t_cap > 0.1:
            rr_cap_ideal_pts.extend([(t_cap - 0.0001, 0), (t_cap, VDD), (t_cap + rr_pulse_w, VDD), (t_cap + rr_pulse_w + 0.0001, 0)])
        if t_copy > 0.1 and t_copy < total_time - 0.01:
            rr_copy_ideal_pts.extend([(t_copy - 0.0001, 0), (t_copy, VDD), (t_copy + rr_pulse_w, VDD), (t_copy + rr_pulse_w + 0.0001, 0)])
    rr_cap_ideal_pts.append((total_time, 0))
    rr_copy_ideal_pts.append((total_time, 0))

    # Ideal rpeak for rr_timing precharge (5ms pulse, matching v5 spec)
    rpk_5ms_ideal_pts = [(0, 0)]
    for j in range(1, n):
        t_rpk = rpeak_times[j]
        if t_rpk > 0.1:
            rpk_5ms_ideal_pts.extend([(t_rpk - 0.0001, 0), (t_rpk, VDD), (t_rpk + 0.005, VDD), (t_rpk + 0.0051, 0)])
    rpk_5ms_ideal_pts.append((total_time, 0))

    if verbose:
        print(f"  Per-beat pwave thresholds (beat 0): base={pwl_base_pts[1][1]:.5f} peak={pwl_peak_pts[1][1]:.5f} rrcap={pwl_rrcap_pts[1][1]:.5f}")
        if n > 5:
            print(f"  Per-beat pwave thresholds (beat 5): base={pwl_base_pts[6][1]:.5f} peak={pwl_peak_pts[6][1]:.5f} rrcap={pwl_rrcap_pts[6][1]:.5f}")

    # Build SPICE netlist
    sp = f""".title full_analog_chain_{record_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

* === POWER ===
Vdd vdd 0 {VDD}
Vss vss 0 0

* === ECG INPUTS (PWL — replaces CCIA) ===
Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

* === THRESHOLD VOLTAGES (externally set, as in ASIC pad ring) ===
Vth_rpeak rpeak_threshold 0 {rpeak_threshold:.4f}
Vth_14 vth_14_fast 0 {thresholds['vth_14']:.6f}
Vth_69 vth_69 0 {thresholds['vth_69']:.6f}
Vth_28 vth_28 0 {thresholds['vth_28']:.6f}
Vth_300 vth_300 0 {thresholds['vth_300']:.6f}
Vth_10 vth_10 0 {thresholds['vth_10']:.6f}
* Pwave thresholds: ADAPTIVE per-beat PWL sources
* Each beat uses EWMA-predicted RR to compute when triggers should fire
* on the delay timer SF (0.42mV/ms), giving fast comparator resolution.
{pw_pwl_str}
* Ideal pwave triggers (PWL pulses at exact Python sample positions)
Vpwb_ideal trig_pw_base_ideal 0 PWL({make_pwl(pw_base_ideal_pts)})
Vpwp_ideal trig_pw_peak_ideal 0 PWL({make_pwl(pw_peak_ideal_pts)})
* Ideal slope triggers (PWL pulses at exact Python sample positions)
Vs14_ideal trig_s14_ideal 0 PWL({make_pwl(s14_ideal_pts)})
Vs69_ideal trig_s69_ideal 0 PWL({make_pwl(s69_ideal_pts)})
* Ideal rpeak trigger (at exact R-peak annotation time)
Vrpk_ideal rpk_ideal 0 PWL({make_pwl(rpk_ideal_pts)})
* Ideal bi_copy trigger (at R+300ms)
Vbicopy_ideal bicopy_ideal 0 PWL({make_pwl(bi_copy_ideal_pts)})
* Ideal rr_timing triggers
Vrrcap_ideal rr_cap_ideal 0 PWL({make_pwl(rr_cap_ideal_pts)})
Vrrcopy_ideal rr_copy_ideal 0 PWL({make_pwl(rr_copy_ideal_pts)})
Vrpk5ms_ideal rpk_5ms_ideal 0 PWL({make_pwl(rpk_5ms_ideal_pts)})

* === SUBCIRCUIT INCLUDES ===
.include "{fp}/inverter.spice"
.include "{fp}/bias_gen.spice"
.include "{fp}/rpeak_detector.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

* ============================================================
* ANALOG FRONTEND INSTANTIATION
* ============================================================

* 1. BIAS GENERATOR
Xbias vbias vref vdd vss bias_gen

* 2. R-PEAK DETECTOR
Xrpeak ecg_l0 rpeak_threshold rpeak_raw vdd vss vbias rpeak_detector

* 3. PULSE GENERATOR: rpeak level to short pulse
*    Rising-edge pulse at QRS onset (~0.5ms width)
Xpg_rpk rpeak_raw rpeak_pulse vdd vss vbias pulse_gen

* 4. DELAY TIMER
*    Uses rpeak_pulse (consistent ~0.5ms) for precharge instead of rpeak_raw
*    (variable ~9ms). This eliminates rpk_drop variation across beats.
*    sf_out exposed for pwave prediction comparators
Xdelay rpeak_pulse dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_copy_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_28 vth_300 vth_10 delay_timer

* 5. INVERTER BUFFERS to sharpen slow comparator edges
Xinv_s14a dt_s14_lev dt_s14_buf1 vdd vss inverter_sub
Xinv_s14b dt_s14_buf1 dt_s14_buf vdd vss inverter_sub
Xinv_s69a dt_s69_lev dt_s69_buf1 vdd vss inverter_sub
Xinv_s69b dt_s69_buf1 dt_s69_buf vdd vss inverter_sub
Xinv_wida dt_wid_lev dt_wid_buf1 vdd vss inverter_sub
Xinv_widb dt_wid_buf1 dt_wid_buf vdd vss inverter_sub
Xinv_bia  dt_bi_lev  dt_bi_buf1 vdd vss inverter_sub
Xinv_bib  dt_bi_buf1 dt_bi_buf vdd vss inverter_sub
Xinv_rrca dt_rr_copy_lev dt_rrc_buf1 vdd vss inverter_sub
Xinv_rrcb dt_rrc_buf1 dt_rrc_buf vdd vss inverter_sub

* 5b. PULSE GENERATORS for buffered delay_timer outputs
Xpg_s14 dt_s14_buf trig_s14 vdd vss vbias pulse_gen
Xpg_s69 dt_s69_buf trig_s69 vdd vss vbias pulse_gen
Xpg_wid dt_wid_buf trig_wid vdd vss vbias pulse_gen
Xpg_bi  dt_bi_buf  trig_bi_copy vdd vss vbias pulse_gen
Xpg_rrc dt_rrc_buf trig_rr_copy vdd vss vbias pulse_gen

* 6. PWAVE PREDICTION COMPARATORS (on delay timer SF)
*    These replace pwave_predictor.spice which used the slow RR timer.
*    The delay timer SF (0.42mV/ms) gives large differentials at 600-800ms
*    for fast comparator resolution, unlike the RR timer (0.09mV/ms).
*    Topology: NFET diff pair + PMOS current mirror (same as delay_timer comparators)

* pw_base comparator: fires at R + (avg_rr - 201)ms
XMcpwb1 cnpwb vth_pw_base ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb2 pw_base_lev dt_sf_out ctpwb vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwb3 cnpwb cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb4 pw_base_lev cnpwb vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwb5 ctpwb vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* pw_peak comparator: fires at R + (avg_rr - 131)ms
XMcpwp1 cnpwp vth_pw_peak ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp2 pw_peak_lev dt_sf_out ctpwp vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcpwp3 cnpwp cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp4 pw_peak_lev cnpwp vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcpwp5 ctpwp vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* rr_cap comparator: fires at R + (avg_rr - 10)ms, for rr_timing capture
XMcrrc1 cnrrc vth_rr_cap ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc2 rr_cap_lev dt_sf_out ctrrc vss sky130_fd_pr__nfet_01v8 L=4 W=2 nf=1
XMcrrc3 cnrrc cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc4 rr_cap_lev cnrrc vdd vdd sky130_fd_pr__pfet_01v8 L=4 W=2 nf=1
XMcrrc5 ctrrc vbias vss vss sky130_fd_pr__nfet_01v8 L=8 W=0.42 nf=1

* 6b. INVERTER BUFFERS + PULSE GENERATORS for pwave prediction
Xinv_pwba pw_base_lev pw_base_buf1 vdd vss inverter_sub
Xinv_pwbb pw_base_buf1 pw_base_buf vdd vss inverter_sub
Xinv_pwpa pw_peak_lev pw_peak_buf1 vdd vss inverter_sub
Xinv_pwpb pw_peak_buf1 pw_peak_buf vdd vss inverter_sub
Xinv_rrca2 rr_cap_lev rr_cap_buf1 vdd vss inverter_sub
Xinv_rrcb2 rr_cap_buf1 rr_cap_buf vdd vss inverter_sub
Xpg_pwb pw_base_buf trig_pw_base vdd vss vbias pulse_gen
Xpg_pwp pw_peak_buf trig_pw_peak vdd vss vbias pulse_gen
Xpg_rrc2 rr_cap_buf trig_rr_cap vdd vss vbias pulse_gen

* 7. RR TIMING (F1 & F2) — using ideal triggers for circuit isolation testing
*    capture: 2ms pulse at R-5ms, rpeak: 5ms pulse at R, copy: 2ms pulse at R+10ms
Xrr ecg_l0 rr_cap_ideal rpk_5ms_ideal rr_copy_ideal f1_rr_ratio f2_rr_asym rr_timer_node rr_ewma_node vdd vss vbias rr_timing

* 9. FEATURE CIRCUITS
* F3: Beat Instability — Lead 0 (ideal triggers to test circuit isolation)
Xbi0 ecg_l0 rpk_ideal bicopy_ideal f3_bi_l0 vdd vss vbias beat_instability
* F4: Beat Instability — Lead 1 (ideal triggers to test circuit isolation)
Xbi1 ecg_l1 rpk_ideal bicopy_ideal f4_bi_l1 vdd vss vbias beat_instability
* F5: P-wave Difference (Lead 0)
*     Uses ideal PWL triggers (from known R-peak positions) for precise timing.
*     The analog prediction triggers (trig_pw_base/peak) are kept for rr_cap.
Xpw ecg_l0 trig_pw_base_ideal trig_pw_peak_ideal f5_pwave vdd vss vbias pwave_diff
* F6: QRS Width (Lead 0)
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
* F7: QRS Slope (Lead 0)
*     s14 uses ideal PWL trigger (14ms is too short for reliable analog timing;
*     ASIC will use digital counter). s69 uses analog delay timer (works well).
Xslp ecg_l0 trig_s14_ideal trig_s69 f7_slope vdd vss vbias qrs_slope
* F8: QRS Symmetry (Lead 0) (ideal trigger to test circuit isolation)
Xsym ecg_l0 rpk_ideal f8_symmetry vdd vss qrs_symmetry

* === INITIAL CONDITIONS ===
* Bias generator startup
.ic V(xbias.iref)=0.3
* RR timing EWMA warmup — initialize to calibrated end-of-RR timer value
* so rr_cap comparator fires correctly from the first beat
.ic V(xrr.ewma_node)={thresholds['ewma_level']:.6f} V(xrr.esamp_node)={thresholds['ewma_level']:.6f} V(xrr.prev_node)={thresholds['ewma_level']:.6f}
* Beat instability EWMA warmup
.ic V(xbi0.prev_node)=0.30 V(xbi0.esamp_node)=0.30
.ic V(xbi1.prev_node)=0.30 V(xbi1.esamp_node)=0.30

* === SIMULATION ===
.tran 0.5m {total_time:.4f}

"""

    # Measurement commands
    # Verify bias generator
    sp += f".meas tran vbias_dc FIND V(vbias) AT=1.5\n"

    # Verify rpeak_detector fires for each beat
    for j in range(n):
        t_rpk = rpeak_times[j]
        sp += f".meas tran rpk_check_{j} FIND V(rpeak_raw) AT={t_rpk:.7f}\n"
        sp += f".meas tran ecg_check_{j} FIND V(ecg_l0) AT={t_rpk:.7f}\n"
        sp += f".meas tran stg1_check_{j} FIND V(xrpeak.stg1_out) AT={t_rpk:.7f}\n"

    # Measure feature outputs
    # F1/F2: capture trigger is trig_rr_cap which fires at ~R+(avg_rr-10)ms.
    # Measure BEFORE capture (at R-135ms) so prev_node still holds previous beat's value.
    f12_skip = 5
    f38_skip = 2
    skip = max(f12_skip, f38_skip)

    for j in range(skip, n):
        # F1 & F2: measure at rpeak[j]-135ms (before capture updates prev/esamp)
        # Both diff pairs share sf_out: rr_ratio = sf vs ewma, rr_asym = sf vs prev
        t_f12 = rpeak_times[j] - 0.135
        if t_f12 > 0.5:
            sp += f".meas tran f1_{j} FIND V(f1_rr_ratio) AT={t_f12:.7f}\n"
            sp += f".meas tran f2_{j} FIND V(f2_rr_asym) AT={t_f12:.7f}\n"
            sp += f".meas tran rr_sf_{j} FIND V(xrr.sf_out) AT={t_f12:.7f}\n"
            sp += f".meas tran rr_ewma_{j} FIND V(xrr.ewma_node) AT={t_f12:.7f}\n"
            sp += f".meas tran rr_prev_{j} FIND V(xrr.prev_node) AT={t_f12:.7f}\n"

        # F3,F4,F6,F7,F8: measure at rpeak[j]+400ms (after bi_copy at ~R+320ms)
        t_f38 = rpeak_times[j] + 0.400
        if t_f38 < total_time - 0.01:
            sp += f".meas tran f3_{j} FIND V(f3_bi_l0) AT={t_f38:.7f}\n"
            sp += f".meas tran f4_{j} FIND V(f4_bi_l1) AT={t_f38:.7f}\n"
            sp += f".meas tran f6_{j} FIND V(f6_width) AT={t_f38:.7f}\n"
            sp += f".meas tran f7_{j} FIND V(f7_slope) AT={t_f38:.7f}\n"
            sp += f".meas tran f8_{j} FIND V(f8_symmetry) AT={t_f38:.7f}\n"

        # F5 (pwave): pwave triggers from beat j-1's delay timer capture ECG at
        # R[j]-201ms (base) and R[j]-131ms (peak). S&H holds until beat j's
        # timer fires new triggers at R[j]+612ms. Measure at R[j]+400ms —
        # well after capture, well before overwrite.
        t_f5 = rpeak_times[j] + 0.400
        if t_f5 < total_time - 0.01 and t_f5 > 0.5:
            sp += f".meas tran f5_{j} FIND V(f5_pwave) AT={t_f5:.7f}\n"

    # Debug: pwave S&H internals for a few beats
    for j in range(5, min(10, n)):
        t_f5 = rpeak_times[j] + 0.400
        if t_f5 < total_time - 0.01 and t_f5 > 0.5:
            sp += f".meas tran shb_{j} FIND V(xpw.shb) AT={t_f5:.7f}\n"
            sp += f".meas tran shp_{j} FIND V(xpw.shp) AT={t_f5:.7f}\n"
            sp += f".meas tran ecg_at_f5_{j} FIND V(ecg_l0) AT={t_f5:.7f}\n"

    # Debug: slope S&H internals for a few beats
    for j in range(skip, min(skip + 10, n)):
        t_f7 = rpeak_times[j] + 0.400
        if t_f7 < total_time - 0.01 and t_f7 > 0.5:
            sp += f".meas tran sh14_{j} FIND V(xslp.sh14) AT={t_f7:.7f}\n"
            sp += f".meas tran sh69_{j} FIND V(xslp.sh69) AT={t_f7:.7f}\n"
            # ECG at exact trigger times for comparison
            t_s14_exact = rpeak_times[j] + 5.0 / FS
            t_s69_exact = rpeak_times[j] + 25.0 / FS
            sp += f".meas tran ecg_at_s14_{j} FIND V(ecg_l0) AT={t_s14_exact:.7f}\n"
            sp += f".meas tran ecg_at_s69_{j} FIND V(ecg_l0) AT={t_s69_exact:.7f}\n"

    # Debug: monitor delay timer SF at times after rpeak_raw would drop
    beat_dbg = 2
    for offset_ms in [1, 20, 30, 50, 80, 120, 200, 350]:
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran sf_at_{offset_ms}ms FIND V(xdelay.sf_timer) AT={t_meas:.7f}\n"

    # Debug: rpeak_raw level at 1ms resolution (to see when it drops)
    for offset_ms in range(-5, 20):
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas > 0.1:
            sp += f".meas tran rpk_raw_{offset_ms+5:02d} FIND V(rpeak_raw) AT={t_meas:.7f}\n"

    # Debug: comparator LEVEL outputs (before pulse_gen)
    # These should go HIGH when the SF timer crosses their threshold
    for offset_ms in [30, 50, 80, 120, 200, 350]:
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran dt_s14_at_{offset_ms}ms FIND V(dt_s14_lev) AT={t_meas:.7f}\n"
            sp += f".meas tran dt_wid_at_{offset_ms}ms FIND V(dt_wid_lev) AT={t_meas:.7f}\n"
            sp += f".meas tran dt_s69_at_{offset_ms}ms FIND V(dt_s69_lev) AT={t_meas:.7f}\n"
            sp += f".meas tran dt_bi_at_{offset_ms}ms FIND V(dt_bi_lev) AT={t_meas:.7f}\n"

    # Debug: RR timer discharge curve for a DETECTED beat
    # Use beat 2 (detected) and measure at absolute offsets from its rpeak
    beat_dbg2 = 2  # detected beat
    for offset_ms in [5, 50, 100, 200, 300, 400, 500, 600]:
        t_meas = rpeak_times[beat_dbg2] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran rr_timer_at_{offset_ms}ms FIND V(rr_timer_node) AT={t_meas:.7f}\n"
            sp += f".meas tran rr_net_at_{offset_ms}ms FIND V(xrr.net_timer) AT={t_meas:.7f}\n"
            sp += f".meas tran rr_sfout_at_{offset_ms}ms FIND V(xrr.sf_out) AT={t_meas:.7f}\n"

    # Pwave predictor: measure delay timer SF at percentages of the RR interval
    # Now using dt_sf_out (delay timer SF) instead of rr_timer_node
    beat_next_det = 4  # next detected beat
    if beat_next_det < n:
        det_rr_ms = int((rpeak_times[beat_next_det] - rpeak_times[beat_dbg2]) * 1000)
        for pct in [50, 65, 75, 85, 90, 95]:
            offset_ms_pw = int(det_rr_ms * pct / 100)
            t_meas = rpeak_times[beat_dbg2] + offset_ms_pw / 1000.0
            if t_meas < total_time - 0.01:
                sp += f".meas tran dt_sf_pw_{pct}pct FIND V(dt_sf_out) AT={t_meas:.7f}\n"
                sp += f".meas tran pw_base_lev_{pct}pct FIND V(pw_base_lev) AT={t_meas:.7f}\n"
                sp += f".meas tran pw_base_buf_{pct}pct FIND V(pw_base_buf) AT={t_meas:.7f}\n"
                sp += f".meas tran pw_peak_buf_{pct}pct FIND V(pw_peak_buf) AT={t_meas:.7f}\n"

    # Debug: buffered comparator outputs (should be sharp digital)
    for offset_ms in [30, 50, 80, 120]:
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran buf_s14_at_{offset_ms}ms FIND V(dt_s14_buf) AT={t_meas:.7f}\n"
            sp += f".meas tran buf_wid_at_{offset_ms}ms FIND V(dt_wid_buf) AT={t_meas:.7f}\n"
            sp += f".meas tran buf_s69_at_{offset_ms}ms FIND V(dt_s69_buf) AT={t_meas:.7f}\n"

    # Debug: pulse_gen outputs — measure at 1ms resolution around expected fire times
    # With rpk_drop~8ms, pipe=2.5ms: s14 fires ~R+13ms, wid ~R+27ms, s69 ~R+68ms
    for offset_ms in range(10, 35):
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran trig_s14_at_{offset_ms}ms FIND V(trig_s14) AT={t_meas:.7f}\n"
    for offset_ms in range(23, 38):
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran trig_wid_at_{offset_ms}ms FIND V(trig_wid) AT={t_meas:.7f}\n"
    for offset_ms in range(60, 80):
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran trig_s69_at_{offset_ms}ms FIND V(trig_s69) AT={t_meas:.7f}\n"

    # Debug: pwave triggers (should fire late in the RR interval)
    beat_next_det2 = 3  # next beat after beat_dbg
    if beat_next_det2 < n:
        rr_ms = int((rpeak_times[beat_next_det2] - rpeak_times[beat_dbg]) * 1000)
        for offset_ms_pw in range(rr_ms - 100, rr_ms, 10):
            t_meas = rpeak_times[beat_dbg] + offset_ms_pw / 1000.0
            if t_meas > 0.5 and t_meas < total_time - 0.01:
                sp += f".meas tran pw_trig_base_{offset_ms_pw}ms FIND V(trig_pw_base) AT={t_meas:.7f}\n"
                sp += f".meas tran pw_trig_peak_{offset_ms_pw}ms FIND V(trig_pw_peak) AT={t_meas:.7f}\n"
                sp += f".meas tran pw_trig_cap_{offset_ms_pw}ms FIND V(trig_rr_cap) AT={t_meas:.7f}\n"

    # Debug: bi_copy trigger timing (1ms resolution around transition)
    for offset_ms in [315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 330, 340]:
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas < total_time - 0.01:
            sp += f".meas tran trig_bi_at_{offset_ms}ms FIND V(trig_bi_copy) AT={t_meas:.7f}\n"
            sp += f".meas tran buf_bi_at_{offset_ms}ms FIND V(dt_bi_buf) AT={t_meas:.7f}\n"

    # Debug: pwave trigger timing — now on delay timer SF
    # pw_base fires at R+(avg_rr-201)ms, pw_peak at R+(avg_rr-131)ms
    for offset_ms in range(580, 660, 2):
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas > 0.5 and t_meas < total_time - 0.01:
            sp += f".meas tran pw_base_trig_at_{offset_ms}ms FIND V(trig_pw_base) AT={t_meas:.7f}\n"
            sp += f".meas tran dt_sf_at_{offset_ms}ms FIND V(dt_sf_out) AT={t_meas:.7f}\n"

    # Debug: pwave peak trigger (fires later than base)
    for offset_ms in range(650, 730, 2):
        t_meas = rpeak_times[beat_dbg] + offset_ms / 1000.0
        if t_meas > 0.5 and t_meas < total_time - 0.01:
            sp += f".meas tran pw_peak_trig_at_{offset_ms}ms FIND V(trig_pw_peak) AT={t_meas:.7f}\n"
            sp += f".meas tran dt_sf_pk_at_{offset_ms}ms FIND V(dt_sf_out) AT={t_meas:.7f}\n"

    sp += "\n.end\n"

    # Write and run
    tb_path = os.path.join(FRONTEND_DIR, f"_analog_chain_{record_id}.spice")
    log_path = tb_path.replace('.spice', '.log')

    with open(tb_path, 'w', encoding='ascii', errors='replace') as f:
        f.write(sp)

    if verbose:
        print(f"Running ngspice simulation ({n} beats, ~{total_time:.0f}s sim time)...")

    t0 = time.time()
    result = subprocess.run(
        [NGSPICE, '-b', '-o', log_path, tb_path],
        capture_output=True, text=True, timeout=900,
        cwd=NGSPICE_DIR
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"Simulation completed in {elapsed:.1f}s")

    # Parse results
    with open(log_path, 'r') as f:
        text = f.read()

    # Check for errors
    error_lines = [l for l in text.split('\n') if 'error' in l.lower() and 'reltol' not in l.lower()]
    if error_lines and verbose:
        print(f"\nSimulation warnings/errors:")
        for l in error_lines[:10]:
            print(f"  {l.strip()}")

    results = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\S+)\s*=\s*([0-9eE.+-]+)', line)
        if m:
            try:
                results[m.group(1).lower()] = float(m.group(2))
            except:
                pass

    if verbose:
        print(f"\nExtracted {len(results)} measurements")

        # Report diagnostics
        vbias_dc = results.get('vbias_dc')
        if vbias_dc is not None:
            print(f"\n=== DIAGNOSTICS ===")
            print(f"  Vbias (bias_gen): {vbias_dc:.4f}V (target: ~0.300V)")

        # Check rpeak detection
        rpk_detected = 0
        for j in range(min(5, n)):
            ecg_v = results.get(f'ecg_check_{j}')
            stg1_v = results.get(f'stg1_check_{j}')
            rpk_v = results.get(f'rpk_check_{j}')
            if rpk_v is not None and rpk_v > 0.5:
                rpk_detected += 1
            print(f"  Beat {j}: ecg={ecg_v:.4f}V stg1={stg1_v:.4f}V rpk={rpk_v:.6f}V" if ecg_v else f"  Beat {j}: no data")
        for j in range(5, n):
            rpk_v = results.get(f'rpk_check_{j}')
            if rpk_v is not None and rpk_v > 0.5:
                rpk_detected += 1
        print(f"  R-peaks detected: {rpk_detected}/{n}")

        # rpeak_raw timing: when does it drop? (1ms resolution)
        print(f"\n  rpeak_raw timeline (beat {2}):")
        for offset_ms in range(-5, 20):
            idx = offset_ms + 5
            v = results.get(f'rpk_raw_{idx:02d}')
            if v is not None:
                state = "HIGH" if v > 0.5 else "LOW"
                print(f"    R{offset_ms:+4d}ms: {v:.4f}V ({state})")

        # SF timer discharge curve
        print(f"\n  SF timer discharge curve (beat {2}):")
        for offset_ms in [1, 20, 30, 50, 80, 120, 200, 350]:
            v = results.get(f'sf_at_{offset_ms}ms')
            if v is not None:
                print(f"    R+{offset_ms:3d}ms: {v:.5f}V")

        # Comparator level outputs
        print(f"\n  Comparator LEVEL outputs (beat {2}):")
        print(f"  {'ms':>5s} {'s14_lev':>8s} {'wid_lev':>8s} {'s69_lev':>8s} {'bi_lev':>8s}")
        for offset_ms in [30, 50, 80, 120, 200, 350]:
            vals = []
            for name in ['dt_s14', 'dt_wid', 'dt_s69', 'dt_bi']:
                v = results.get(f'{name}_at_{offset_ms}ms')
                vals.append(f"{v:.4f}" if v is not None else "  N/A ")
            print(f"  R+{offset_ms:3d}  {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s}")

        # RR timer discharge curve (actual test)
        print(f"\n  RR timer discharge curve (beat {beat_dbg2}, detected):")
        print(f"  {'ms':>5s} {'net_timer':>10s} {'sf_out':>10s} {'timer_out':>10s}")
        for offset_ms in [5, 50, 100, 200, 300, 400, 500, 600]:
            net_v = results.get(f'rr_net_at_{offset_ms}ms')
            sf_v = results.get(f'rr_sfout_at_{offset_ms}ms')
            tmr_v = results.get(f'rr_timer_at_{offset_ms}ms')
            vals = [f"{v:.5f}" if v is not None else "   N/A  " for v in [net_v, sf_v, tmr_v]]
            print(f"  R+{offset_ms:3d}  {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

        # Pwave predictor diagnostics (detected-to-detected interval)
        beat_next_det = 4
        if beat_next_det < n:
            det_rr_ms = int((rpeak_times[beat_next_det] - rpeak_times[beat_dbg2]) * 1000)
            print(f"\n  Pwave predictor on delay timer SF (beat {beat_dbg2}->{beat_next_det}, {det_rr_ms}ms):")
            print(f"  {'%RR':>5s} {'dt_sf':>9s} {'pw_base_l':>10s} {'pw_base_b':>10s} {'pw_peak_b':>10s}")
            for pct in [50, 65, 75, 85, 90, 95]:
                rr_v = results.get(f'dt_sf_pw_{pct}pct')
                pwb_l = results.get(f'pw_base_lev_{pct}pct')
                pwb_b = results.get(f'pw_base_buf_{pct}pct')
                pwp_b = results.get(f'pw_peak_buf_{pct}pct')
                vals = [f"{v:.5f}" if v is not None else "  N/A  " for v in [rr_v, pwb_l, pwb_b, pwp_b]]
                print(f"  {pct:3d}%  {vals[0]:>9s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s}")

        # Buffered comparator outputs
        print(f"\n  Buffered (inverter pair) outputs (beat {2}):")
        print(f"  {'ms':>5s} {'buf_s14':>8s} {'buf_wid':>8s} {'buf_s69':>8s}")
        for offset_ms in [30, 50, 80, 120]:
            vals = []
            for name in ['buf_s14', 'buf_wid', 'buf_s69']:
                v = results.get(f'{name}_at_{offset_ms}ms')
                vals.append(f"{v:.4f}" if v is not None else "  N/A ")
            print(f"  R+{offset_ms:3d}  {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

        # Pulse trigger outputs — s14
        print(f"\n  trig_s14 (target R+14ms, beat {2}):")
        for offset_ms in range(10, 35):
            v = results.get(f'trig_s14_at_{offset_ms}ms')
            if v is not None and v > 0.1:
                print(f"    R+{offset_ms:3d}ms: {v:.4f}V  <-- PULSE")
        # trig_wid
        print(f"  trig_wid (target R+28ms, beat {2}):")
        for offset_ms in range(23, 38):
            v = results.get(f'trig_wid_at_{offset_ms}ms')
            if v is not None and v > 0.1:
                print(f"    R+{offset_ms:3d}ms: {v:.4f}V  <-- PULSE")
        # trig_s69
        print(f"  trig_s69 (target R+69ms, beat {2}):")
        for offset_ms in range(60, 80):
            v = results.get(f'trig_s69_at_{offset_ms}ms')
            if v is not None and v > 0.1:
                print(f"    R+{offset_ms:3d}ms: {v:.4f}V  <-- PULSE")

        # Pwave trigger outputs
        beat_next_det2 = 3
        if beat_next_det2 < n:
            rr_ms = int((rpeak_times[beat_next_det2] - rpeak_times[beat_dbg]) * 1000)
            print(f"\n  Pwave trigger outputs (beat {beat_dbg}, RR={rr_ms}ms):")
            print(f"  {'ms':>5s} {'pw_base':>9s} {'pw_peak':>9s} {'rr_cap':>9s}")
            for offset_ms_pw in range(rr_ms - 100, rr_ms, 10):
                vals = []
                for name in ['pw_trig_base', 'pw_trig_peak', 'pw_trig_cap']:
                    v = results.get(f'{name}_{offset_ms_pw}ms')
                    vals.append(f"{v:.4f}" if v is not None else "   N/A  ")
                print(f"  R+{offset_ms_pw:3d}  {vals[0]:>9s} {vals[1]:>9s} {vals[2]:>9s}")

        # BI copy trigger timing
        print(f"\n  BI copy trigger timing (beat {2}):")
        print(f"  {'ms':>5s} {'trig_bi':>9s} {'buf_bi':>9s}")
        for offset_ms in [315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 330, 340]:
            t_v = results.get(f'trig_bi_at_{offset_ms}ms')
            b_v = results.get(f'buf_bi_at_{offset_ms}ms')
            vals = [f"{v:.4f}" if v is not None else "   N/A  " for v in [t_v, b_v]]
            print(f"  R+{offset_ms:3d}  {vals[0]:>9s} {vals[1]:>9s}")

        # Pwave base trigger timing
        print(f"\n  Pwave base trigger timing (beat {2}):")
        print(f"  {'ms':>5s} {'trig_pwb':>9s} {'buf_pwb':>9s}")
        for offset_ms in range(600, 660, 2):
            t_v = results.get(f'pw_base_trig_at_{offset_ms}ms')
            b_v = results.get(f'pw_base_buf_at_{offset_ms}ms')
            if t_v is not None:
                vals = [f"{v:.4f}" if v is not None else "   N/A  " for v in [t_v, b_v]]
                print(f"  R+{offset_ms:3d}  {vals[0]:>9s} {vals[1]:>9s}")

        # Pwave peak trigger timing
        print(f"\n  Pwave peak trigger timing (beat {2}):")
        print(f"  {'ms':>5s} {'trig_pwp':>9s} {'buf_pwp':>9s}")
        for offset_ms in range(640, 700, 2):
            t_v = results.get(f'pw_peak_trig_at_{offset_ms}ms')
            b_v = results.get(f'pw_peak_buf_at_{offset_ms}ms')
            if t_v is not None:
                vals = [f"{v:.4f}" if v is not None else "   N/A  " for v in [t_v, b_v]]
                print(f"  R+{offset_ms:3d}  {vals[0]:>9s} {vals[1]:>9s}")

        # Pwave S&H debug
        print(f"\n  Pwave S&H internals (beats 5-9):")
        print(f"  beat   shb      shp      pw_out   ecg     py_pw")
        for j in range(5, min(10, n)):
            shb_v = results.get(f'shb_{j}')
            shp_v = results.get(f'shp_{j}')
            ecg_v = results.get(f'ecg_at_f5_{j}')
            f5_v = results.get(f'f5_{j}')
            py_v = beats[j]['py_pw'] if j < len(beats) else None
            vals = [f"{v:.5f}" if v is not None else "  N/A  " for v in [shb_v, shp_v, f5_v, ecg_v]]
            py_str = f"{py_v:.5f}" if py_v is not None else "  N/A  "
            print(f"  {j:3d}  {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s} {py_str:>8s}")

        # Slope S&H debug
        print(f"\n  Slope S&H internals (beats {skip}-{min(skip+9, n-1)}):")
        print(f"  beat   sh14     sh69     ecg@s14  ecg@s69  slope_out py_slp")
        for j in range(skip, min(skip + 10, n)):
            sh14_v = results.get(f'sh14_{j}')
            sh69_v = results.get(f'sh69_{j}')
            ecg14_v = results.get(f'ecg_at_s14_{j}')
            ecg69_v = results.get(f'ecg_at_s69_{j}')
            f7_v = results.get(f'f7_{j}')
            py_v = beats[j]['py_slp'] if j < len(beats) else None
            vals = [f"{v:.5f}" if v is not None else "  N/A  " for v in [sh14_v, sh69_v, ecg14_v, ecg69_v, f7_v]]
            py_str = f"{py_v:.5f}" if py_v is not None else "  N/A  "
            print(f"  {j:3d}  {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s} {vals[4]:>8s} {py_str:>8s}")

    # Debug: print F1/F2 internals
    if verbose:
        print(f"\n  RR timing internals (F1/F2, beats {skip}-{min(skip+14, n-1)}):")
        print(f"  beat  pre_rr    sf_out   ewma     prev     f1_out  py_f1   f2_out  py_f2")
        for j in range(skip, min(skip + 15, n)):
            sf_v = results.get(f'rr_sf_{j}')
            ew_v = results.get(f'rr_ewma_{j}')
            pv_v = results.get(f'rr_prev_{j}')
            f1_v = results.get(f'f1_{j}')
            f2_v = results.get(f'f2_{j}')
            pre_rr = beats[j]['pre_rr']
            py_f1 = beats[j]['py_f1']
            py_f2 = beats[j]['py_f2']
            vals = [f"{v:.5f}" if v is not None else " N/A  " for v in [sf_v, ew_v, pv_v, f1_v, f2_v]]
            print(f"  {j:4d}  {pre_rr:.4f}  {vals[0]}  {vals[1]}  {vals[2]}  {vals[3]} {py_f1:.4f}  {vals[4]} {py_f2:.4f}")

    # Compute correlation with Python features
    feat_map = {
        'F1': ('f1_{}', 'py_f1'),
        'F2': ('f2_{}', 'py_f2'),
        'F3': ('f3_{}', 'py_bi_l0'),
        'F4': ('f4_{}', 'py_bi_l1'),
        'F5': ('f5_{}', 'py_pw'),
        'F6': ('f6_{}', 'py_wid'),
        'F7': ('f7_{}', 'py_slp'),
        'F8': ('f8_{}', 'py_sym'),
    }

    # Also compare against timing-adjusted Python reference
    feat_map_adj = {
        'F3': ('f3_{}', 'py_bi_l0_adj'),
        'F6': ('f6_{}', 'py_wid_adj'),
        'F7': ('f7_{}', 'py_slp_adj'),
        'F8': ('f8_{}', 'py_sym_adj'),
    }

    if verbose:
        print(f"\n=== FEATURE CORRELATION (Python reference) ===")
        print(f"{'Feat':>4s} {'rho':>8s} {'rho_adj':>8s} {'N':>4s} {'range_mV':>8s} {'Status':>10s}")
        print("-" * 55)

    feature_results = {}
    all_ok = True
    for f_name, (sp_key, py_key) in feat_map.items():
        sp_vals, py_vals, py_adj_vals = [], [], []
        adj_key = feat_map_adj.get(f_name, (None, None))[1]
        for j in range(skip, n):
            val = results.get(sp_key.format(j))
            if val is not None:
                sp_vals.append(val)
                py_vals.append(beats[j][py_key])
                if adj_key and adj_key in beats[j]:
                    py_adj_vals.append(beats[j][adj_key])

        if len(sp_vals) > 3:
            rho, _ = spearmanr(sp_vals, py_vals)
            rho_adj = None
            if len(py_adj_vals) == len(sp_vals):
                rho_adj, _ = spearmanr(sp_vals, py_adj_vals)
            range_mv = (max(sp_vals) - min(sp_vals)) * 1000
            status = "PASS" if abs(rho) > 0.70 else ("WEAK" if abs(rho) > 0.40 else "FAIL")
            if abs(rho) < 0.40:
                all_ok = False
            feature_results[f_name] = {'rho': rho, 'rho_adj': rho_adj, 'n': len(sp_vals), 'status': status}
            rho_adj_s = f"{rho_adj:+8.4f}" if rho_adj is not None else "     N/A"
            if verbose:
                print(f"{f_name:>4s} {rho:+8.4f} {rho_adj_s:>8s} {len(sp_vals):4d} {range_mv:8.1f} {status:>10s}")
        else:
            feature_results[f_name] = {'rho': None, 'rho_adj': None, 'n': len(sp_vals), 'status': 'NO DATA'}
            if verbose:
                print(f"{f_name:>4s} {'N/A':>8s} {'N/A':>8s} {len(sp_vals):4d} {'':>8s} {'NO DATA':>10s}")
            all_ok = False

    # Cleanup
    try:
        os.remove(tb_path)
    except:
        pass

    return {
        'record': record_id, 'n_beats': n, 'elapsed': elapsed,
        'results': results, 'features': feature_results,
        'all_ok': all_ok, 'thresholds': thresholds,
        'log_path': log_path,
    }


if __name__ == '__main__':
    import sys
    n_beats = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    print("=" * 60)
    print("FULL ANALOG CHAIN INTEGRATION TEST")
    print(f"Record 100, {n_beats} beats")
    print("=" * 60)
    print()

    result = run_test('100', n_beats=n_beats, verbose=True)

    print()
    if result['all_ok']:
        print(f"ALL FEATURES PASS ({n_beats} beats)")
    else:
        print("Some features need investigation")
        print(f"Check log: {result['log_path']}")

    print()
    print("=" * 60)
