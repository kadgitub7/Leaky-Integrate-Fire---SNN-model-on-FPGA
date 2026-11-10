"""
Phase 4: Full integration test — analog supplementary circuits replace ideal PWL triggers.

Replaces:
  - Vbias → bias_gen.spice (analog)
  - Vs14, Vs69, Vwid, Vrry → delay_timer + edge_buffer + pulse_gen (analog, 0.5ms)
  - Vbco → delay_timer + edge_buffer + long_pulse_gen (analog, 2ms for EWMA copy)

Keeps ideal:
  - trig_rpeak, trig_rpeak_sh (rpeak detector has known -2.7ms offset)
  - trig_rr_cap (55ms capture window; pwave_predictor architecture differs)
  - trig_pw_base, trig_pw_peak (will test analog pwave in Round 2)
"""
import subprocess, os, re, sys, time, json
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
BATCH_SIZE = 20
EWMA_ALPHA = 0.10

TEST_RECORDS = ['100', '119', '208', '232', '205']

aami_map = {}
for s in 'NLRej': aami_map[s] = 'N'
for s in 'AaJS': aami_map[s] = 'S'
for s in 'VE': aami_map[s] = 'V'
for s in 'F': aami_map[s] = 'F'
for s in '/fQ': aami_map[s] = 'Q'


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
            ewma_rr_before = pre_rr
            ewma_rr = pre_rr
        else:
            ewma_rr_before = ewma_rr
            ewma_rr = EWMA_ALPHA * pre_rr + (1 - EWMA_ALPHA) * ewma_rr

        py_f1 = pre_rr - ewma_rr_before
        py_f2 = pre_rr - post_rr

        prev_bs = valid[i-1][0] - WIN_L
        prev_be = valid[i-1][0] + WIN_R
        if prev_bs >= 0 and prev_be < len(sig):
            prev_beat = sig[prev_bs:prev_be]
            py_bi_l0 = beat[RPEAK_IDX+QRS_S, 0] - prev_beat[RPEAK_IDX+QRS_S, 0]
            py_bi_l1 = beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)] - prev_beat[RPEAK_IDX+QRS_S, min(1,n_leads-1)]
        else:
            py_bi_l0 = py_bi_l1 = 0.0

        beats.append({
            'label': aami_map[sym],
            'pre_rr': pre_rr, 'post_rr': post_rr,
            'ecg_l0': beat[:, 0].copy(),
            'ecg_l1': beat[:, min(1, n_leads-1)].copy(),
            'py_f1': py_f1, 'py_f2': py_f2,
            'py_bi_l0': py_bi_l0, 'py_bi_l1': py_bi_l1,
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


def generate_batch_testbench(beats_batch, gain, batch_id, record_id, ic_vals=None):
    if ic_vals is None:
        ic_vals = {}
    n = len(beats_batch)
    dt = 1.0 / FS
    PW = 0.0005

    rpeak_times = [max(beats_batch[0]['pre_rr'], 0.5)]
    for i in range(1, n):
        rpeak_times.append(rpeak_times[-1] + beats_batch[i]['pre_rr'])
    total_time = rpeak_times[-1] + 0.5

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

    # Ideal triggers we still keep
    trig_rpeak = make_trig(rpeak_times, 0, PW, total_time)
    trig_rr_cap = make_trig(rpeak_times, -0.060, 0.055, total_time)
    trig_rpeak_sh = make_trig(rpeak_times, 0, PW, total_time)
    trig_pw_base = make_trig(rpeak_times, -(WIN_L - PW_S) / FS, PW, total_time)
    trig_pw_peak = make_trig(rpeak_times, -(WIN_L - PW_MID) / FS, PW, total_time)

    # Rpeak pulse for delay_timer (same timing as trig_rpeak, 0.5ms pulse with 0.4ms ramps)
    rpeak_pulse_pwl = make_trig(rpeak_times, 0, PW, total_time)

    lib_path = os.path.join(NGSPICE_DIR, "sky130_minimal.lib.spice").replace("\\", "/")
    fp = FRONTEND_DIR.replace("\\", "/")

    sp = f""".title phase4_{record_id}_b{batch_id}
.lib "{lib_path}" tt
.option scale=1.0u method=gear reltol=0.01 gmin=1e-15
.temp 27

Vdd vdd 0 {VDD}
Vss vss 0 0

Vecg0 ecg_l0 0 PWL({make_pwl(ecg_l0_pts)})
Vecg1 ecg_l1 0 PWL({make_pwl(ecg_l1_pts)})

* === IDEAL TRIGGERS (kept for now) ===
Vrpk trig_rpeak 0 PWL({trig_rpeak})
Vrrc trig_rr_cap 0 PWL({trig_rr_cap})
Vrsh trig_rpeak_sh 0 PWL({trig_rpeak_sh})
Vpwb trig_pw_base 0 PWL({trig_pw_base})
Vpwp trig_pw_peak 0 PWL({trig_pw_peak})

* === ANALOG BIAS GENERATOR (replaces Vbias) ===
.include "{fp}/bias_gen.spice"
Xbias vbias vref vdd vss bias_gen
.ic V(xbias.iref)=0.3

* === ANALOG DELAY TIMER (replaces Vs14, Vs69, Vwid, Vbco, Vrry) ===
.include "{fp}/inverter.spice"
.include "{fp}/delay_timer.spice"
.include "{fp}/pulse_gen.spice"
.include "{fp}/long_pulse_gen.spice"
.include "{fp}/edge_buffer.spice"

* Rpeak pulse to delay_timer (same PWL as trig_rpeak)
Vrpk_dt rpeak_pulse_dt 0 PWL({rpeak_pulse_pwl})

* Threshold voltages (calibrated for real bias_gen vbias=0.30031V)
Vth10  vth_10      0 0.2892
Vth14  vth_14_fast 0 0.2117
Vth83  vth_83      0 0.47093
Vth69  vth_69      0 0.44934
Vth300 vth_300     0 0.3303

Xdt rpeak_pulse_dt dt_s14_lev dt_s69_lev dt_wid_lev dt_bi_lev dt_rr_lev dt_sf_out vdd vss vbias vth_14_fast vth_69 vth_83 vth_300 vth_10 delay_timer

* Fast timer outputs: direct to pulse_gen (fast edges)
Xpg_s14 dt_s14_lev trig_s14 vdd vss vbias pulse_gen
Xpg_rr  dt_rr_lev  trig_rr_copy vdd vss vbias pulse_gen

* Main timer outputs: edge_buffer + pulse_gen (slow edges need sharpening)
Xeb_wid dt_wid_lev wid_sharp vdd vss edge_buffer
Xpg_wid wid_sharp  trig_wid  vdd vss vbias pulse_gen

Xeb_s69 dt_s69_lev s69_sharp vdd vss edge_buffer
Xpg_s69 s69_sharp  trig_s69  vdd vss vbias pulse_gen

Xeb_bi  dt_bi_lev  bi_sharp  vdd vss edge_buffer
Xpg_bi  bi_sharp   trig_bi_copy vdd vss vbias long_pulse_gen

* === FEATURE CIRCUITS (unchanged from v3 baseline) ===
.include "{fp}/rr_timing.spice"
.include "{fp}/beat_instability.spice"
.include "{fp}/pwave_diff.spice"
.include "{fp}/qrs_width.spice"
.include "{fp}/qrs_slope.spice"
.include "{fp}/qrs_symmetry.spice"

Xrr ecg_l0 trig_rr_cap trig_rpeak trig_rr_copy f1_rr_ratio f2_rr_asym rr_timer rr_ewma vdd vss vbias rr_timing
Xbi0 ecg_l0 trig_rpeak_sh trig_bi_copy f3_bi_l0 vdd vss vbias beat_instability
Xbi1 ecg_l1 trig_rpeak_sh trig_bi_copy f4_bi_l1 vdd vss vbias beat_instability
Xpw ecg_l0 trig_pw_base trig_pw_peak f5_pwave vdd vss vbias pwave_diff
Xwid ecg_l0 trig_wid f6_width vdd vss qrs_width
Xslp ecg_l0 trig_s14 trig_s69 f7_slope vdd vss vbias qrs_slope
Xsym ecg_l0 trig_rpeak_sh f8_symmetry vdd vss qrs_symmetry

.ic V(xrr.net_timer)={ic_vals.get('rr_timer', VDD):.6f}
.ic V(xrr.ewma_node)={ic_vals.get('rr_ewma', 0.40):.6f}
.ic V(xrr.esamp_node)={ic_vals.get('rr_esamp', 0.40):.6f}
.ic V(xrr.prev_node)={ic_vals.get('rr_prev', 0.40):.6f}
.ic V(xbi0.prev_node)={ic_vals.get('bi0_prev', 0.30):.6f}
.ic V(xbi0.esamp_node)={ic_vals.get('bi0_esamp', 0.30):.6f}
.ic V(xbi1.prev_node)={ic_vals.get('bi1_prev', 0.30):.6f}
.ic V(xbi1.esamp_node)={ic_vals.get('bi1_esamp', 0.30):.6f}

.tran 0.5m {total_time:.4f}

"""
    f12_skip = 2
    f38_skip = 2
    skip = max(f12_skip, f38_skip)

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

    t_ic = rpeak_times[-1] + 0.100
    if t_ic < total_time - 0.01:
        sp += f"\n* IC propagation measurements\n"
        sp += f".meas tran ic_rr_ewma FIND V(xrr.ewma_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_rr_esamp FIND V(xrr.esamp_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_rr_prev FIND V(xrr.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi0_prev FIND V(xbi0.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi0_esamp FIND V(xbi0.esamp_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi1_prev FIND V(xbi1.prev_node) AT={t_ic:.7f}\n"
        sp += f".meas tran ic_bi1_esamp FIND V(xbi1.esamp_node) AT={t_ic:.7f}\n"

    sp += "\n.end\n"

    tb = os.path.join(FRONTEND_DIR, f"_p4_{record_id}_b{batch_id}.spice")
    with open(tb, 'w') as f:
        f.write(sp)
    return tb, skip


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


def process_record(record_id):
    beats, gain = load_all_beats(record_id)
    if not beats:
        return [], []

    all_circuit, all_py = [], []
    n_batches = (len(beats) + BATCH_SIZE - 1) // BATCH_SIZE

    ic_vals = None
    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end = min((b_idx + 1) * BATCH_SIZE, len(beats))
        batch = beats[start:end]
        if len(batch) < 6:
            continue

        tb, skip = generate_batch_testbench(batch, gain, b_idx, record_id, ic_vals)
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
            next_ic['rr_timer'] = VDD
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

        try:
            os.remove(tb)
            os.remove(tb.replace('.spice', '.log'))
        except: pass

    return all_circuit, all_py


# === MAIN ===
print("=" * 70)
print("PHASE 4: ANALOG SUPPLEMENTARY CIRCUITS INTEGRATION TEST")
print("=" * 70)
print("  Analog: bias_gen, delay_timer, edge_buffer, pulse_gen, long_pulse_gen (bi_copy)")
print("  Ideal:  rpeak, rpeak_sh, rr_cap, pw_base, pw_peak")
print(f"  Records: {', '.join(TEST_RECORDS)}")
print()

feature_names = ['F1:rr_ratio', 'F2:rr_asym', 'F3:bi_L0', 'F4:bi_L1',
                 'F5:pwave', 'F6:width', 'F7:slope', 'F8:symmetry']

all_circuit_global = []
all_py_global = []

for rec in TEST_RECORDS:
    t0 = time.time()
    print(f"  Processing record {rec}...", end='', flush=True)
    circuit_feats, py_feats = process_record(rec)
    elapsed = time.time() - t0
    n = len(circuit_feats)
    print(f" {n} beats, {elapsed:.1f}s")

    if n < 10:
        print(f"    WARNING: only {n} matched beats, skipping rho")
        continue

    circuit_arr = np.array(circuit_feats)
    py_arr = np.array(py_feats)
    all_circuit_global.extend(circuit_feats)
    all_py_global.extend(py_feats)

    print(f"    {'Feature':<15s} {'|rho|':>8s} {'Status':>8s}")
    for i, fname in enumerate(feature_names):
        ci = circuit_arr[:, i]
        py = py_arr[:, i]
        mask = np.isfinite(ci) & np.isfinite(py)
        if np.sum(mask) < 5:
            print(f"    {fname:<15s} {'SKIP':>8s}")
            continue
        rho, _ = spearmanr(ci[mask], py[mask])
        status = "PASS" if abs(rho) > 0.98 else "OK" if abs(rho) > 0.95 else "WARN" if abs(rho) > 0.90 else "FAIL"
        print(f"    {fname:<15s} {abs(rho):8.4f} {status:>8s}")

print()
print("=" * 70)
print("AGGREGATE RESULTS (all records combined)")
print("=" * 70)

if len(all_circuit_global) > 50:
    circuit_all = np.array(all_circuit_global)
    py_all = np.array(all_py_global)
    print(f"  Total matched beats: {len(circuit_all)}")
    print()
    print(f"  {'Feature':<15s} {'|rho|':>8s} {'v3 thresh':>10s} {'Status':>8s}")
    print(f"  {'-'*45}")

    thresholds = [0.970, 0.995, 0.980, 0.986, 0.995, 0.998, 0.995, 0.995]
    for i, (fname, thresh) in enumerate(zip(feature_names, thresholds)):
        ci = circuit_all[:, i]
        py = py_all[:, i]
        mask = np.isfinite(ci) & np.isfinite(py)
        if np.sum(mask) < 5:
            print(f"  {fname:<15s} {'SKIP':>8s}")
            continue
        rho, pval = spearmanr(ci[mask], py[mask])
        status = "PASS" if abs(rho) >= thresh else "CLOSE" if abs(rho) >= thresh - 0.01 else "FAIL"
        print(f"  {fname:<15s} {abs(rho):8.4f} {thresh:10.3f} {status:>8s}")
else:
    print("  Not enough data for aggregate analysis")

print("=" * 70)
